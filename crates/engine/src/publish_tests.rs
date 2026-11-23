//! Tests of [`crate::publish`]: the pre-flight of real exports (a good Reel, a landscape clip too
//! short for a Reel and too wide for a Short, sound without a picture, a damaged file), the fixed
//! copies (fast start without re-encoding, a re-encode at a frame rate Instagram takes), the
//! destinations' connect states, `publish.run` dispatching to the publishers after the pre-flight,
//! and hostile files and parameters — errors, never panics.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use super::*;
use crate::instagram::{InstagramPublisher, ReelRequest};
use crate::publish::{self as pb, Destination, Status};
use crate::youtube::{ConnectOptions, UploadRequest, YouTubePublisher};
use serde_json::{Value, json};

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("filmcraft-publish-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The demo project exported to `name` (H.264 + AAC MP4 unless `extra` says otherwise).
fn export(name: &str, extra: Value) -> String {
    let path = dir().join(name).to_string_lossy().into_owned();
    if std::path::Path::new(&path).exists() {
        return path;
    }
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let mut p = json!({"path": path, "format": "h264", "bitrateKbps": 1500, "range": "custom", "startSeconds": 0.0, "endSeconds": 4.0, "wait": true});
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        p[k] = v;
    }
    let r = s.execute("file.exportMedia", p).unwrap();
    r["path"].as_str().unwrap().to_string()
}

/// A legacy MP4 with metadata at the end, made with the normal muxer finalizer.
/// Keep exercising publish.fixCopy now that native exports already use fast start.
fn late_metadata_fixture(path: String) -> String {
    use filmcraft_isobmff::{Brand, Mp4Writer, TrackConfig, WriteSample, WriterOptions};
    let bytes = std::fs::read(&path).unwrap();
    let movie = filmcraft_isobmff::open(bytes.as_slice()).unwrap();
    let mut options = WriterOptions::new(Brand::Mp4);
    options.movie_timescale = movie.timescale;
    let mut writer = Mp4Writer::new(std::io::Cursor::new(Vec::new()), options).unwrap();
    for (index, track) in movie.tracks.iter().enumerate() {
        assert_eq!(track.entries.len(), 1, "these fixtures have one sample description");
        let mut config = TrackConfig::new(track.entries[0].clone(), track.timescale);
        config.edits = track.edits.clone();
        config.language = track.language.clone();
        let target = writer.add_track(config).unwrap();
        for (sample_index, sample) in track.samples.iter().enumerate() {
            let data = movie.read_sample(bytes.as_slice(), index, sample_index).unwrap();
            writer
                .write_sample(
                    target,
                    WriteSample {
                        data: &data,
                        duration: sample.duration,
                        composition_offset: i32::try_from(sample.pts - sample.dts).unwrap(),
                        is_sync: sample.is_sync,
                    },
                )
                .unwrap();
        }
    }
    std::fs::write(&path, writer.finish().unwrap().into_inner()).unwrap();
    path
}

/// The test files, made once per run.
struct Files {
    reel: String,
    wide: String,
    silent: String,
    fast: String,
    wav: String,
}

fn files() -> &'static Files {
    static F: OnceLock<Files> = OnceLock::new();
    F.get_or_init(|| Files {
        reel: late_metadata_fixture(export("reel.mp4", json!({"width": 360, "height": 640}))),
        wide: export("wide.mp4", json!({"width": 640, "height": 360, "endSeconds": 2.0})),
        silent: late_metadata_fixture(export("silent.mp4", json!({"width": 360, "height": 640, "audio": false}))),
        fast: export("fast.mp4", json!({"width": 180, "height": 320, "fps": 72.0, "audio": false})),
        wav: export("sound.wav", json!({"format": "wav", "endSeconds": 1.0})),
    })
}

fn statuses(r: &Value) -> Vec<(String, String)> {
    r["checks"].as_array().unwrap().iter().map(|c| (c["id"].as_str().unwrap().to_string(), c["status"].as_str().unwrap().to_string())).collect()
}

fn status_of(r: &Value, id: &str) -> String {
    statuses(r).into_iter().find(|(i, _)| i == id).map(|(_, s)| s).unwrap_or_else(|| panic!("no check {id}: {r}"))
}

#[test]
fn a_good_reel_passes_and_says_what_it_measured() {
    let f = files();
    let mut s = Session::default();
    let r = s.execute("publish.preflight", json!({"destination": "instagramReel", "path": f.reel})).unwrap();
    assert_eq!(r["errors"], 0, "{r:#}");
    assert_eq!(r["ok"], true);
    for id in
        ["container", "videoCodec", "chroma", "frameRate", "width", "aspect", "bitrate", "duration", "fileSize", "audioCodec", "audioRate", "audioChannels"]
    {
        assert_eq!(status_of(&r, id), "pass", "{id}: {r:#}");
    }
    // The explicit legacy fixture keeps metadata last, so a fast-start fix remains available.
    assert_eq!(status_of(&r, "moovFirst"), "warn");
    assert!(r["checks"].as_array().unwrap().iter().any(|c| c["id"] == "loudness" && c["value"].as_str().unwrap().contains("LUFS")), "{r:#}");
    assert!(r["fixable"].is_string());
    assert_eq!(r["media"]["video"]["width"], 360);
    // a Short and a YouTube video take it too
    for d in ["youtubeShort", "youtube"] {
        let r = s.execute("publish.preflight", json!({"destination": d, "path": f.reel, "loudness": false})).unwrap();
        assert_eq!(r["errors"], 0, "{d}: {r:#}");
    }
    // the Media File takes anything
    let r = s.execute("publish.preflight", json!({"destination": "mediaFile", "path": f.wav})).unwrap();
    assert_eq!(r["ok"], true);
}

#[test]
fn failures_name_their_fix() {
    let f = files();
    let mut s = Session::default();
    let r = s.execute("publish.preflight", json!({"destination": "instagramReel", "path": f.wide})).unwrap();
    assert_eq!(status_of(&r, "duration"), "fail", "2 s is under a Reel's 3 s");
    assert_eq!(status_of(&r, "aspect"), "warn", "16:9 is taken but not recommended");
    assert!(r["unfixable"].as_array().unwrap().contains(&json!("duration")));
    let dur = r["checks"].as_array().unwrap().iter().find(|c| c["id"] == "duration").unwrap();
    assert!(dur["fix"].as_str().unwrap().contains("at least 3 s"), "{dur}");
    let r = s.execute("publish.preflight", json!({"destination": "youtubeShort", "path": f.wide, "loudness": false})).unwrap();
    assert_eq!(status_of(&r, "aspect"), "fail", "a landscape video is not a Short");
    let r = s.execute("publish.preflight", json!({"destination": "youtube", "path": f.wide, "loudness": false})).unwrap();
    assert_eq!(r["errors"], 0, "YouTube takes it: {r:#}");
    // sound without a picture
    let r = s.execute("publish.preflight", json!({"destination": "instagramReel", "path": f.wav})).unwrap();
    assert_eq!((status_of(&r, "container").as_str(), status_of(&r, "video").as_str()), ("fail", "fail"));
    // too fast for Instagram: a copy at 60 fps fixes it
    let r = s.execute("publish.preflight", json!({"destination": "instagramReel", "path": f.fast})).unwrap();
    assert_eq!(status_of(&r, "frameRate"), "fail");
    assert_eq!(r["fixable"], "reexport");
}

#[test]
fn fixed_copies_fast_start_and_re_encode() {
    let f = files();
    let mut s = Session::default();
    // only the moov is out of place: moved, nothing re-encoded
    let v = s.execute("publish.fixCopy", json!({"destination": "instagramReel", "path": f.silent, "wait": true})).unwrap();
    assert_eq!(v["fix"], "faststart", "{v}");
    let out = v["publish"]["path"].as_str().unwrap().to_string();
    assert!(out.ends_with("silent (Instagram Reel).mp4"), "{out}");
    assert_eq!(v["publish"]["preflight"]["ok"], true, "{v:#}");
    assert_eq!(v["publish"]["preflight"]["warnings"], 0, "{v:#}");
    // the same pictures, from the shifted chunk offsets
    let a = s.media.open_file(&f.silent, &*s.services).unwrap();
    let b = s.media.open_file(&out, &*s.services).unwrap();
    for secs in [0.0, 1.5, 3.5] {
        let req = filmcraft_media::FrameRequest::full(filmcraft_time::Tick::from_seconds_f64(secs));
        let (fa, fb) = (a.video_frame(req).unwrap(), b.video_frame(req).unwrap());
        assert!(fa.width == fb.width && fa.height == fb.height && fa.to_rgba8() == fb.to_rgba8(), "frame at {secs} s differs");
    }
    assert_eq!(std::fs::metadata(&out).unwrap().len(), std::fs::metadata(&f.silent).unwrap().len(), "the same bytes, reordered");
    // too fast: re-encoded at 60 fps (and fast-started)
    let v = s.execute("publish.fixCopy", json!({"destination": "instagramReel", "path": f.fast, "wait": true})).unwrap();
    assert_eq!(v["fix"], "reexport");
    let pre = &v["publish"]["preflight"];
    assert_eq!(pre["errors"], 0, "{pre:#}");
    assert!((pre["media"]["video"]["fps"].as_f64().unwrap() - 60.0).abs() < 0.1, "{pre:#}");
    assert_eq!(status_of(pre, "moovFirst"), "pass");
    // nothing to fix, or nothing a copy can fix
    let e = s.execute("publish.fixCopy", json!({"destination": "youtube", "path": f.silent})).unwrap_err().to_string();
    assert!(e.contains("nothing to fix"), "{e}");
    let e = s.execute("publish.fixCopy", json!({"destination": "youtubeShort", "path": f.wide})).unwrap_err().to_string();
    assert!(e.contains("cannot fix"), "{e}");
    let e = s.execute("publish.fixCopy", json!({"destination": "instagramReel", "path": f.fast, "out": f.fast})).unwrap_err().to_string();
    assert!(e.contains("another file"), "{e}");
}

#[derive(Default)]
struct FakeYt {
    uploads: Mutex<Vec<UploadRequest>>,
}

impl YouTubePublisher for FakeYt {
    fn connect_ready(&self) -> std::result::Result<(), String> {
        Ok(())
    }
    fn ready(&self) -> std::result::Result<(), String> {
        Ok(())
    }
    fn channels(&self) -> std::result::Result<Value, String> {
        Ok(json!({"channels": []}))
    }
    fn connect(&self, _: ConnectOptions, _: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn set_default(&self, _: &str) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn disconnect(&self, _: &str) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn check(&self, _: &UploadRequest) -> std::result::Result<Value, String> {
        Ok(json!({"channel": {"title": "H4hacks"}}))
    }
    fn upload(&self, req: &UploadRequest, p: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.uploads.lock().unwrap().push(req.clone());
        p.done.store(p.total.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(json!({"videoId": "abc", "url": "https://www.youtube.com/watch?v=abc"}))
    }
}

#[derive(Default)]
struct FakeIg {
    not_ready: Option<String>,
    reels: Mutex<Vec<ReelRequest>>,
}

impl InstagramPublisher for FakeIg {
    fn connect_ready(&self) -> std::result::Result<(), String> {
        Err("not set up: run `filmcraft-cli instagram setup`".into())
    }
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }
    fn account(&self) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn connect(&self, _: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn disconnect(&self) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn limits(&self) -> std::result::Result<Value, String> {
        Ok(json!({}))
    }
    fn check(&self, _: &ReelRequest) -> std::result::Result<Value, String> {
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn publish(&self, req: &ReelRequest, _: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.reels.lock().unwrap().push(req.clone());
        Ok(json!({"published": true, "mediaId": "18000000000001", "permalink": "https://www.instagram.com/reel/Cx/"}))
    }
}

#[test]
fn destinations_list_their_connect_state() {
    let mut s = Session::default();
    let v = s.execute("publish.destinations", json!({})).unwrap();
    let rows = v["destinations"].as_array().unwrap();
    assert_eq!(rows.iter().map(|r| r["id"].as_str().unwrap()).collect::<Vec<_>>(), ["mediaFile", "youtube", "youtubeShort", "instagramReel"]);
    assert_eq!(rows[0]["connect"]["needed"], false);
    assert_eq!(rows[3]["connect"]["available"], false, "no publisher here");
    s.youtube.publisher = Some(Arc::new(FakeYt::default()));
    s.instagram.publisher =
        Some(Arc::new(FakeIg { not_ready: Some("Instagram is not connected: run `filmcraft-cli instagram auth`".into()), ..Default::default() }));
    let v = s.execute("publish.destinations", json!({})).unwrap();
    let rows = v["destinations"].as_array().unwrap();
    assert_eq!((rows[1]["connect"]["ready"].as_bool(), rows[2]["connect"]["connect"].as_str()), (Some(true), Some("youtube.connect")));
    assert_eq!(rows[3]["connect"]["ready"], false);
    assert!(rows[3]["connect"]["why"].as_str().unwrap().contains("instagram auth"));
    assert!(rows[3]["connect"]["connectWhy"].as_str().unwrap().contains("instagram setup"));
}

#[test]
fn publish_run_checks_first_then_starts_the_destinations_job() {
    let f = files();
    let mut s = Session::default();
    let yt = Arc::new(FakeYt::default());
    let ig = Arc::new(FakeIg::default());
    s.youtube.publisher = Some(yt.clone());
    s.instagram.publisher = Some(ig.clone());
    let v =
        s.execute("publish.run", json!({"destination": "instagramReel", "path": f.reel, "caption": "Robots #ai", "coverSeconds": 1.0, "wait": true})).unwrap();
    assert_eq!((v["destination"].as_str(), v["via"].as_str()), (Some("instagramReel"), Some("instagram.publish")));
    assert_eq!(v["instagram"]["permalink"], "https://www.instagram.com/reel/Cx/", "{v}");
    assert_eq!(v["preflight"]["ok"], true);
    let r = ig.reels.lock().unwrap()[0].clone();
    assert_eq!((r.caption.as_deref(), r.cover_ms, r.path.as_deref()), (Some("Robots #ai"), Some(1000), Some(f.reel.as_str())));
    // YouTube: the title defaults to the file's name
    let v = s.execute("publish.run", json!({"destination": "youtube", "path": f.wide, "privacy": "unlisted", "wait": true})).unwrap();
    assert_eq!(v["youtube"]["videoId"], "abc");
    let u = yt.uploads.lock().unwrap()[0].clone();
    assert_eq!((u.meta.title.as_str(), u.meta.privacy.as_str()), ("wide", "unlisted"));
    // the pre-flight stops what the platform would refuse, before any job
    let e = s.execute("publish.run", json!({"destination": "youtubeShort", "path": f.wide})).unwrap_err().to_string();
    assert!(e.contains("Aspect ratio") && e.contains("pre-flight"), "{e}");
    let e = s.execute("publish.run", json!({"destination": "instagramReel", "path": f.wide})).unwrap_err().to_string();
    assert!(e.contains("Duration"), "{e}");
    assert_eq!(ig.reels.lock().unwrap().len(), 1);
    // a public URL needs no local file
    s.execute("publish.run", json!({"destination": "instagramReel", "videoUrl": "https://cdn.example.com/a.mp4", "wait": true})).unwrap();
    assert_eq!(ig.reels.lock().unwrap().len(), 2);
}

#[test]
fn hostile_parameters_and_files_are_errors() {
    let f = files();
    let mut s = Session::default();
    let junk = dir().join("junk.mp4");
    std::fs::write(&junk, b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00junk junk junk").unwrap();
    let empty = dir().join("empty.mp4");
    std::fs::write(&empty, b"").unwrap();
    for (cmd, p, want) in [
        ("publish.preflight", json!({"path": f.reel}), "need `destination`"),
        ("publish.preflight", json!({"destination": "tiktok", "path": f.reel}), "not a destination"),
        ("publish.preflight", json!({"destination": 5, "path": f.reel}), "must be a string"),
        ("publish.preflight", json!({"destination": "youtube"}), "need `path`"),
        ("publish.preflight", json!({"destination": "youtube", "path": "a\u{0}b"}), "usable path"),
        ("publish.preflight", json!({"destination": "youtube", "path": "/no/such.mp4"}), "/no/such.mp4"),
        ("publish.preflight", json!({"destination": "youtube", "path": empty.to_string_lossy()}), "is empty"),
        ("publish.preflight", json!({"destination": "youtube", "path": junk.to_string_lossy()}), "not a video"),
        ("publish.run", json!({"destination": "mediaFile", "path": f.reel}), "the export itself"),
        ("publish.run", json!({"destination": "youtube"}), "need `path`"),
        ("publish.run", json!([1]), "need `destination`"),
        ("publish.fixCopy", json!({"destination": "mediaFile", "path": f.reel}), "takes any export"),
    ] {
        let e = s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{cmd} {p}: {e}");
    }
    // without publishers the run says the platform is not here
    let e = s.execute("publish.run", json!({"destination": "youtube", "path": f.reel})).unwrap_err().to_string();
    assert!(e.contains("YouTube is not available"), "{e}");
    for (alias, d) in
        [("Instagram", Destination::InstagramReel), ("shorts", Destination::YoutubeShort), ("file", Destination::MediaFile), ("YouTube", Destination::Youtube)]
    {
        assert_eq!(Destination::parse(alias).unwrap(), d);
    }
}

#[test]
fn damaged_files_never_panic_the_box_walk_or_the_fast_start() {
    let f = files();
    let good = std::fs::read(&f.silent).unwrap();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let out = dir().join("fuzz-out.mp4");
    for round in 0..300 {
        let mut b = good.clone();
        match round % 3 {
            0 => b.truncate((next() as usize) % good.len()),
            1 => {
                for _ in 0..8 {
                    let i = (next() as usize) % good.len().min(4096);
                    b[i] ^= 1 << (next() % 8);
                }
            }
            _ => {
                // a box size that lies
                let i = (next() as usize) % good.len().min(2048) / 4 * 4;
                if i + 4 <= b.len() {
                    b[i..i + 4].copy_from_slice(&(next() as u32).to_be_bytes());
                }
            }
        }
        let size = b.len() as u64;
        let read = |off: u64, len: usize| -> std::io::Result<Vec<u8>> {
            let a = (off as usize).min(b.len());
            Ok(b[a..(a + len).min(b.len())].to_vec())
        };
        let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pb::walk_boxes(&read, size)));
        assert!(walked.is_ok(), "round {round}: the walk panicked");
        let path = dir().join("fuzz.mp4");
        std::fs::write(&path, &b).unwrap();
        let fixed = std::panic::catch_unwind(|| pb::faststart(&path, &out));
        assert!(fixed.is_ok(), "round {round}: the fast start panicked");
    }
    let read = |_: u64, _: usize| -> std::io::Result<Vec<u8>> { Ok(Vec::new()) };
    assert!(pb::walk_boxes(&read, 0).is_none());
    assert!(pb::walk_boxes(&read, 1 << 40).is_none());
}

#[test]
fn checks_of_hand_made_facts() {
    let base = pb::Probe {
        path: "/x/a.mp4".into(),
        bytes: 10 << 20,
        extension: "mp4".into(),
        container: "MP4".into(),
        seconds: 30.0,
        video: Some(pb::VideoFacts {
            codec: "H.264 (High)".into(),
            width: 1080,
            height: 1920,
            fps: 30.0,
            kbps: Some(8000),
            pixel_format: "YUV 4:2:0 8-bit".into(),
        }),
        audio: Some(pb::AudioFacts { codec: "AAC".into(), rate: 48_000, channels: 2 }),
        boxes: Some(pb::Boxes { has_moov: true, moov_first: true, ..Default::default() }),
        loudness: Some(pb::LoudnessFacts { integrated_lufs: -14.2, true_peak_dbtp: -1.5, seconds: 30.0 }),
        loudness_note: None,
    };
    let all_pass = pb::checks(Destination::InstagramReel, &base);
    assert!(all_pass.iter().all(|c| c.status == Status::Pass), "{all_pass:#?}");
    let mut p = base.clone();
    p.video = Some(pb::VideoFacts {
        codec: "Apple ProRes 422".into(),
        width: 3840,
        height: 2160,
        fps: 120.0,
        kbps: Some(80_000),
        pixel_format: "YUV 4:2:2 10-bit".into(),
    });
    p.audio = Some(pb::AudioFacts { codec: "PCM 24-bit".into(), rate: 96_000, channels: 6 });
    p.loudness = Some(pb::LoudnessFacts { integrated_lufs: -6.0, true_peak_dbtp: 0.5, seconds: 30.0 });
    p.seconds = 1200.0;
    p.bytes = 2 << 30;
    let r = pb::report(Destination::InstagramReel, &p);
    for id in ["videoCodec", "chroma", "frameRate", "width", "bitrate", "duration", "fileSize", "audioCodec", "audioRate", "audioChannels"] {
        assert_eq!(status_of(&r, id), "fail", "{id}");
    }
    for id in ["aspect", "loudness", "truePeak"] {
        assert_eq!(status_of(&r, id), "warn", "{id}");
    }
    assert_eq!(r["fixable"], "reexport");
    assert!(r["unfixable"].as_array().unwrap().contains(&json!("duration")));
    // odd numbers from a hostile file stay harmless
    let mut odd = base.clone();
    odd.video = Some(pb::VideoFacts { codec: String::new(), width: 0, height: 0, fps: f64::NAN, kbps: None, pixel_format: String::new() });
    odd.seconds = 0.0;
    let _ = pb::report(Destination::YoutubeShort, &odd);
    let (settings, said) = pb::fix_settings(Destination::InstagramReel, &p, "/x/out.mp4");
    assert_eq!(settings.frame_size, Some((1920, 1080)), "{said:?}");
    assert_eq!(settings.frame_rate.map(|r| r.as_f64().round()), Some(60.0));
    assert!(settings.bitrate_kbps <= 25_000 && settings.effects.loudness.enabled && settings.audio.sample_rate == Some(48_000));
    assert_eq!(pb::aspect_name(1080, 1920), "9:16");
    assert_eq!(pb::aspect_name(0, 5), "?");
}

#[test]
fn current_native_exports_pass_fast_start_without_an_unnecessary_copy() {
    let path = export("current-native.mp4", json!({"width": 360, "height": 640, "audio": false}));
    let mut session = Session::default();
    let result = session.execute("publish.preflight", json!({"destination": "instagramReel", "path": path, "loudness": false})).unwrap();
    assert_eq!(status_of(&result, "moovFirst"), "pass", "{result}");
    assert!(result["fixable"].is_null(), "no container repair is needed: {result}");
    let error = session.execute("publish.fixCopy", json!({"destination": "instagramReel", "path": path})).unwrap_err();
    assert!(error.to_string().contains("nothing to fix"), "{error}");
}

/// Publish copy is a leg of the pre-flight (a warning, never an error: D7) and of the fixed copy
/// (only the safe fixes, answered at once); `publish.run` lints what it publishes for the record
/// and publishes it as given.
#[test]
fn publish_copy_is_a_leg_of_the_preflight_and_of_the_fixed_copy() {
    let f = files();
    let mut s = Session::default();
    let base = s.execute("publish.preflight", json!({"destination": "youtube", "path": f.reel, "loudness": false})).unwrap();
    assert!(statuses(&base).iter().all(|(id, _)| id != "copy"), "no copy, no copy check");
    let stock = "Hey guys! We rebuilt the extruder. Smash that like button!";
    let r = s
        .execute(
            "publish.preflight",
            json!({"destination": "youtube", "path": f.reel, "loudness": false, "title": "Printer build!!!", "description": stock, "tags": ["printer"]}),
        )
        .unwrap();
    assert_eq!(status_of(&r, "copy"), "warn", "{r:#}");
    assert_eq!((&r["ok"], &r["errors"]), (&base["ok"], &base["errors"]), "the copy never fails the file");
    assert_eq!(r["warnings"].as_u64(), base["warnings"].as_u64().map(|w| w + 1));
    assert!(r["copy"]["findings"].as_array().unwrap().iter().any(|x| x["phrase"] == "smash that like button"), "{r:#}");
    let clean = s
        .execute(
            "publish.preflight",
            json!({"destination": "youtube", "path": f.reel, "loudness": false, "title": "Rebuilding the extruder", "description": "We rebuilt the extruder twice before it fed cleanly."}),
        )
        .unwrap();
    assert_eq!(status_of(&clean, "copy"), "pass", "{clean:#}");
    // a file that needs nothing: the copy's safe fixes, and no job
    let v = s.execute("publish.fixCopy", json!({"destination": "youtube", "path": f.silent, "title": "Printer build!!!", "description": stock})).unwrap();
    assert!(v["job"].is_null(), "{v}");
    assert_eq!(v["copy"]["fixed"], json!({"title": "Printer build!", "description": "We rebuilt the extruder."}), "{v:#}");
    assert_eq!(v["copy"]["changed"], true);
    assert!(v["copy"]["lint"]["findings"].as_array().unwrap().is_empty(), "{v:#}");
    // a file that needs a fix: the job runs and the copy is answered with it
    let out = dir().join("copy-leg (Instagram Reel).mp4");
    let v =
        s.execute("publish.fixCopy", json!({"destination": "instagramReel", "path": f.silent, "out": out, "wait": true, "caption": "Robots!! #ai"})).unwrap();
    assert_eq!(v["fix"], "faststart", "{v}");
    assert_eq!(v["publish"]["preflight"]["ok"], true, "{v:#}");
    assert_eq!(v["copy"]["fixed"]["caption"], "Robots! #ai");
    // bad copy is refused before anything is read
    for (p, want) in [
        (json!({"destination": "youtube", "path": f.reel, "tags": "a,b"}), "`tags`"),
        (json!({"destination": "youtube", "path": f.reel, "title": 5}), "`title`"),
        (json!({"destination": "youtube", "path": f.silent, "description": "x", "facts": "1"}), "`facts`"),
    ] {
        for cmd in ["publish.preflight", "publish.fixCopy"] {
            let e = s.execute(cmd, p.clone()).unwrap_err().to_string();
            assert!(e.contains(want), "{cmd} {p}: {e}");
        }
    }
    // publish.run lints the caption for the record and publishes it unchanged
    let ig = Arc::new(FakeIg::default());
    s.instagram.publisher = Some(ig.clone());
    let v = s.execute("publish.run", json!({"destination": "instagramReel", "path": f.reel, "caption": "Hey guys!! Robots #ai", "wait": true})).unwrap();
    assert!(v["preflight"]["copy"]["findings"].as_array().unwrap().iter().any(|x| x["phrase"] == "hey guys"), "{v:#}");
    assert_eq!(ig.reels.lock().unwrap()[0].caption.as_deref(), Some("Hey guys!! Robots #ai"));
}
