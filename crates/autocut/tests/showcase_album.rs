//! A Google Photos album as the seed of a showcase, offline: `album` on the showcase jobs picks
//! the album (a `FixedAlbum` standing in for Google's picker and the screen agent), imports it
//! without duplicating what the project already holds, and analyses only the album's clips —
//! never the demo project's six; a prompt names the subject (by the built-in rules, or the
//! classifier) before anything is picked.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use filmcraft_autocut::album::{AlbumPick, FixedAlbum};
use filmcraft_autocut::host::{Models, ShowcaseHost};
use filmcraft_autocut::llm::FixedModel;
use filmcraft_autocut::showcase;
use filmcraft_engine::Session;
use filmcraft_engine::project::{ItemKind, Label, MediaClip, MediaRef, Project, SequenceSettings};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaSource};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use serde_json::{Value, json};

fn tmp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("filmcraft-showcase-album-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A short 160x90 ProRes movie of a demo scene (our own encoder), with a Google Photos picker
/// sidecar dating it.
fn movie(path: &Path, scene: DemoScene, created: &str) -> String {
    let (w, h, rate) = (160, 90, FrameRate::FPS_24);
    let dur = rate.tick_of(12);
    let src = GeneratorSource::new(Generator::Demo(scene), w, h, rate, dur);
    let clip = MediaClip {
        media: MediaRef::Generator(Generator::Demo(scene)),
        info: src.info().clone(),
        interpret: Default::default(),
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    let mut p = Project::new("fixture");
    let item = p.add_item("scene", Label::Iris, ItemKind::Media(clip), None);
    let seq = p.new_sequence("s", SequenceSettings { width: w, height: h, frame_rate: rate, ..Default::default() }, 1, 0, None);
    let mut v = p.make_track_item(item, filmcraft_engine::project::TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
    for e in &mut v.effects {
        filmcraft_engine::project::resolve_auto_points(e, (w, h), (w, h));
    }
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    let settings = filmcraft_engine::export::ExportSettings {
        format: filmcraft_engine::export::Format::ProRes,
        path: path.to_string_lossy().into_owned(),
        include_audio: false,
        ..Default::default()
    };
    let shared: filmcraft_media::SharedSource = Arc::new(src);
    let provider = move |id| (id == item).then(|| shared.clone());
    filmcraft_engine::export::export(&Arc::new(p), seq, &settings, &provider, &Default::default()).unwrap();
    let path = path.to_string_lossy().into_owned();
    std::fs::write(format!("{path}.json"), json!({"created": created}).to_string()).unwrap();
    path
}

/// Two album clips: a.mov filmed in 2019, b.mov in 2023.
fn album_clips(dir: &Path) -> Vec<String> {
    vec![movie(&dir.join("a.mov"), DemoScene::Aurora, "2019-05-01T10:00:00Z"), movie(&dir.join("b.mov"), DemoScene::CityNight, "2023-03-04T10:00:00Z")]
}

/// The demo project (six clips no album holds) with a host picking `album`.
fn session(album: Arc<FixedAlbum>, models: Models) -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(models).with_album_source(album)));
    s
}

fn media_count(s: &Session) -> usize {
    s.project.items.values().filter(|i| i.as_media().is_some()).count()
}

fn clip_items(report: &Value) -> Vec<u64> {
    let mut v: Vec<u64> = report["clips"].as_array().map(|a| a.iter().filter_map(|c| c["item"].as_u64()).collect()).unwrap_or_default();
    v.sort_unstable();
    v
}

#[test]
fn only_the_album_is_analysed_and_nothing_is_imported_twice() {
    let dir = tmp_dir("only");
    let paths = album_clips(&dir);
    let album = Arc::new(FixedAlbum::new(paths.clone()));
    let mut s = session(album.clone(), Models::scripted());
    // one of the album's clips is in the project already
    let had = s.execute("file.import", json!({"paths": [paths[0]]})).unwrap()["items"][0].as_u64().unwrap();
    let before = media_count(&s);

    let r = s.execute("autocut.analyze", json!({"subject": "robot", "album": "Robot showcase", "wait": true})).unwrap();
    let summary = &r["showcase"]["summary"];
    assert_eq!((summary["clips"].as_u64(), summary["showing"].as_u64()), (Some(2), Some(2)), "the album's two clips, not the demo's six: {summary}");
    let seed = &summary["album"];
    assert_eq!(
        (seed["title"].as_str(), seed["picked"].as_u64(), seed["imported"].as_u64(), seed["reused"].as_u64()),
        (Some("Robot showcase"), Some(2), Some(1), Some(1)),
        "{seed}"
    );
    let items: Vec<u64> = seed["items"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    assert_eq!(items.first(), Some(&had), "the clip already in the project is reused: {seed}");
    assert_eq!(media_count(&s), before + 1, "only the new clip was imported");
    assert_eq!(album.asked(), vec![AlbumPick { title: "Robot showcase".into(), videos_only: false, since: None }], "photos are kept");
    let report = s.execute("autocut.showcaseInspect", json!({"report": true})).unwrap()["report"].clone();
    let mut want = items.clone();
    want.sort_unstable();
    assert_eq!(clip_items(&report), want, "{report}");
    assert_eq!(report["album"], "Robot showcase");
    // oldest first, dated by the picker's sidecars
    assert_eq!(report["clips"][0]["item"].as_u64(), Some(had));

    // the whole showcase from the album again: nothing is imported twice
    let r = s.execute("autocut.makeShowcase", json!({"subject": "robot", "album": "Robot showcase", "title": "Robots", "voice": "", "wait": true})).unwrap();
    let out = &r["showcase"];
    assert_eq!((out["summary"]["album"]["imported"].as_u64(), out["summary"]["album"]["reused"].as_u64()), (Some(0), Some(2)), "{out}");
    assert_eq!(media_count(&s), before + 1);
    assert_eq!(out["plan"]["entries"].as_array().map(Vec::len), Some(2), "{out}");
    assert_eq!(out["plan"]["title"], "Robots");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_prompt_names_the_subject_before_anything_is_picked() {
    let dir = tmp_dir("prompt");
    let paths = album_clips(&dir);
    // the built-in rules read this one: no model is asked (the writer has no answers)
    let album = Arc::new(FixedAlbum::new(paths.clone()));
    let quiet = Models { analysis: Models::scripted().analysis, script: Arc::new(FixedModel::new(vec![])) };
    let mut s = session(album.clone(), quiet);
    let r = s.execute("autocut.analyze", json!({"prompt": "a montage of all my robot dogs", "album": "Dogs", "wait": true})).unwrap();
    assert_eq!(r["showcase"]["summary"]["subject"], "robot dog", "{r}");
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["report"]["subject"], "robot dog");

    // a date in the prompt: the picker skips older items, and the analysis keeps the clips filmed since
    let r = s.execute("autocut.analyze", json!({"prompt": "make a video of every robot since 2020", "album": "Dogs", "wait": true})).unwrap();
    assert_eq!(album.asked().last().and_then(|p| p.since.clone()).as_deref(), Some("2020-01-01"));
    assert_eq!(r["showcase"]["summary"]["clips"], 1, "only b.mov was filmed since 2020: {r}");

    // one the rules cannot read goes to the classifier, which may name a title too
    let classifier = FixedModel::new(vec![json!({"intent": "showcase", "confidence": 0.9, "params": {"subject": "drone", "title": "Drones of the Gulf"}})]);
    let album = Arc::new(FixedAlbum::new(paths.clone()));
    let mut s = session(album.clone(), Models { analysis: Models::scripted().analysis, script: Arc::new(classifier) });
    let r = s.execute("autocut.analyze", json!({"prompt": "show off what I flew on my trips", "album": "Trips", "wait": true})).unwrap();
    let summary = &r["showcase"]["summary"];
    assert_eq!(
        (summary["subject"].as_str(), summary["title"].as_str(), summary["clips"].as_u64()),
        (Some("drone"), Some("Drones of the Gulf"), Some(2)),
        "{summary}"
    );

    // a prompt nobody can read fails the job before the album is picked
    let album = Arc::new(FixedAlbum::new(paths));
    let unclear = FixedModel::new(vec![json!({"intent": "custom", "confidence": 0.2, "params": {}})]);
    let mut s = session(album.clone(), Models { analysis: Models::scripted().analysis, script: Arc::new(unclear) });
    let e = s.execute("autocut.analyze", json!({"prompt": "do something nice", "album": "Trips", "wait": true})).unwrap_err().to_string();
    assert!(e.contains("could not tell what the showcase is of"), "{e}");
    assert!(album.asked().is_empty(), "nothing was picked");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A photo in the album (drawn here, dated by its picker sidecar): a robot box on a dark ground.
fn photo(path: &Path, created: &str) -> String {
    let img = image::RgbaImage::from_fn(300, 400, |x, y| {
        if (100..200).contains(&x) && (80..300).contains(&y) { image::Rgba([40, 200, 80, 255]) } else { image::Rgba([20, 30, 60, 255]) }
    });
    img.save(path).unwrap();
    let path = path.to_string_lossy().into_owned();
    std::fs::write(format!("{path}.json"), json!({"created": created}).to_string()).unwrap();
    path
}

#[test]
fn the_albums_photos_are_kept_analysed_once_and_planned_as_photos() {
    let dir = tmp_dir("photos");
    let mut paths = album_clips(&dir);
    paths.push(photo(&dir.join("IMG_0001.png"), "2021-02-03T10:00:00Z"));
    let album = Arc::new(FixedAlbum::new(paths.clone()));
    let mut s = session(album.clone(), Models::scripted());
    let r = s.execute("autocut.makeShowcase", json!({"subject": "robot", "album": "Robot showcase", "title": "Robots", "voice": "", "wait": true})).unwrap();
    let out = &r["showcase"];
    assert!(!album.asked()[0].videos_only, "photos are kept");
    assert_eq!(out["summary"]["clips"].as_u64(), Some(3), "two clips and a photo: {out}");
    let report = s.execute("autocut.showcaseInspect", json!({"report": true})).unwrap()["report"].clone();
    let photos: Vec<&Value> = report["clips"].as_array().unwrap().iter().filter(|c| c["photo"] == json!(true)).collect();
    assert_eq!(photos.len(), 1, "{report}");
    let p = photos[0];
    assert_eq!(p["frames"].as_array().map(Vec::len), Some(1), "a photo is looked at once: {p}");
    assert!(p["subject"].as_array().is_some_and(|b| b.len() == 4), "its robot's box is kept: {p}");
    assert_eq!(p["created"], "2021-02-03T10:00:00Z", "dated by its sidecar");
    // the robot only photographed is an entry of its photo, in time order between the clips
    let entries = out["plan"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 3, "{out}");
    let shot = &entries[1]["shots"][0];
    assert_eq!((shot["photo"].as_bool(), shot["clip_seconds"].as_f64()), (Some(true), Some(showcase::PHOTO_REACH)), "{shot}");
    // and it builds: the photo held still-but-moving, silent, with the clips around it
    let built = s.execute("autocut.showcaseBuild", json!({"wait": true})).unwrap();
    let layout = built["showcase"]["built"]["layout"].as_array().cloned().unwrap_or_default();
    assert_eq!(layout.len(), 3, "{built}");
    let shots: Vec<&Value> = layout.iter().flat_map(|e| e["shots"].as_array().into_iter().flatten()).collect();
    assert!(shots.iter().any(|s| s["photo"] == json!(true) && s.get("sound").is_none() && s["move"].is_string()), "{shots:?}");
    // videos only: the photo is left out of the pick and the analysis
    let album = Arc::new(FixedAlbum::new(paths));
    let mut s = session(album.clone(), Models::scripted());
    let r = s.execute("autocut.analyze", json!({"subject": "robot", "album": "Robot showcase", "photos": false, "wait": true})).unwrap();
    assert!(album.asked()[0].videos_only, "videos only");
    assert_eq!(r["showcase"]["summary"]["clips"].as_u64(), Some(2), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_album_or_a_host_without_albums_says_why() {
    let mut s = session(Arc::new(FixedAlbum::new(vec![])), Models::scripted());
    let before = media_count(&s);
    let e = s.execute("autocut.analyze", json!({"subject": "robot", "album": "Nothing here", "wait": true})).unwrap_err().to_string();
    assert!(e.contains("has no video"), "{e}");
    assert_eq!(media_count(&s), before);
    // a host that cannot pick albums: refused before any job starts
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(Models::scripted())));
    let e = s.execute("autocut.makeShowcase", json!({"subject": "robot", "album": "Robots"})).unwrap_err().to_string();
    assert!(e.contains("album") && e.contains("not available"), "{e}");
    assert!(s.jobs.is_empty());
}
