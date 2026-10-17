//! `template.*`: list, inspect, validate, apply (one undo step, imported media, beats of a new
//! track, the look, the export preset, variants), save and markSlot; hostile parameters and
//! packages.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;
use filmcraft_project::{ClipId, ItemId, TrackItem};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

const S: i64 = TICKS_PER_SECOND;

fn session(dir: &Path) -> Session {
    Session { prefs_path: Some(dir.join("prefs.json")), ..Default::default() }
}

fn png(dir: &Path, name: &str, w: u32, h: u32, shade: u8) -> String {
    let p = dir.join(name);
    image::RgbImage::from_fn(w, h, |x, y| image::Rgb([shade, (x % 256) as u8, (y % 256) as u8])).save(&p).unwrap();
    p.to_string_lossy().into_owned()
}

/// A generated music bed at `bpm`, `secs` long, written as a WAV (the "user's" song).
fn song(s: &mut Session, dir: &Path, bpm: f64, secs: f64) -> String {
    let r =
        s.execute("audio.generate", json!({"kind": "music", "bpm": bpm, "duration": secs, "seed": 5, "place": false, "dir": dir.to_string_lossy()})).unwrap();
    r["path"].as_str().unwrap().to_string()
}

fn clip(s: &Session, seq: u64, id: u64) -> TrackItem {
    s.project.sequence(ItemId(seq)).unwrap().find_item(ClipId(id)).unwrap().1.clone()
}

fn slot_clips(r: &Value, slot: &str) -> Vec<u64> {
    r["slots"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| x["slot"] == slot)
        .flat_map(|x| x["clips"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()))
        .collect()
}

#[test]
fn list_and_inspect_the_starters() {
    let dir = tmp_dir("vt-list");
    let mut s = session(&dir);
    let r = s.execute("template.list", json!({})).unwrap();
    let ids: Vec<&str> = r["templates"].as_array().unwrap().iter().map(|t| t["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["builtin:photo-beat-slideshow", "builtin:channel-intro", "builtin:top-3-listicle"]);
    assert!(r["userFolder"].as_str().unwrap().ends_with("Video Templates"));
    let shorts = s.execute("template.list", json!({"query": "countdown"})).unwrap();
    assert_eq!(shorts["templates"].as_array().unwrap().len(), 1);
    let i = s.execute("template.inspect", json!({"template": "Photo Beat Slideshow"})).unwrap();
    assert_eq!(i["id"], "builtin:photo-beat-slideshow");
    assert_eq!(i["seconds"], 16.0);
    let photos = i["slots"].as_array().unwrap().iter().find(|x| x["id"] == "photos").unwrap();
    assert_eq!(photos["placements"].as_array().unwrap().len(), 8);
    assert_eq!(photos["placements"][1]["start"], 2 * S);
    assert_eq!(i["beatGrid"]["bpm"], 120.0);
    assert_eq!(i["checks"].as_array().unwrap().len(), 0, "the look and export presets exist: {}", i["checks"]);
    assert_eq!(i["variants"][1]["exportPreset"], "Social Vertical 1080×1920");
    assert!(s.execute("template.inspect", json!({"template": "no such thing"})).is_err());
    assert!(s.execute("template.inspect", json!({})).is_err());
}

#[test]
fn apply_builds_a_beat_cut_sequence_in_one_undo_step() {
    let dir = tmp_dir("vt-apply");
    let mut s = session(&dir);
    let photos: Vec<String> = (0..3).map(|i| png(&dir, &format!("p{i}.png"), 400, 300, 40 * i as u8)).collect();
    let music = song(&mut s, &dir, 100.0, 40.0);
    // the song is a file the project does not hold yet; the generated item above is removed
    s.execute("file.newProject", json!({"name": "trip"})).unwrap();
    let undo_before = s.history.undo.len();
    let items_before = s.project.items.len();
    let r = s
        .execute(
            "template.apply",
            json!({"template": "builtin:photo-beat-slideshow", "slots": {
                "photos": [photos[0], photos[1], photos[2], photos[0]],
                "music": music,
                "title": "Lisbon 2026",
            }}),
        )
        .unwrap();
    // one undo step, whatever it took (imports, the new sequence, the look)
    assert_eq!(s.history.undo.len(), undo_before + 1, "{:?}", s.history.undo.iter().map(|u| &u.0).collect::<Vec<_>>());
    assert_eq!(s.history.undo.last().unwrap().0, "Apply Template Photo Beat Slideshow");
    let seq = r["sequence"].as_u64().unwrap();
    assert_eq!(s.state.active_sequence, Some(ItemId(seq)));
    // four photos named, three files: imported once each
    assert_eq!(r["imported"].as_array().unwrap().len(), 4, "{}", r["imported"]);
    let photo_items: std::collections::BTreeSet<u64> =
        r["slots"].as_array().unwrap().iter().filter(|x| x["slot"] == "photos").map(|x| x["item"].as_u64().unwrap()).collect();
    assert_eq!(photo_items.len(), 3);
    // the cuts follow the song's beats
    assert_eq!(r["retimed"], true, "{r}");
    let song_item = r["music"]["item"].as_u64().unwrap();
    let len = s.project.item(ItemId(song_item)).unwrap().duration();
    let (_, beats) = crate::remix::beat_ticks(&s, ItemId(song_item), Tick::ZERO, len.min(Tick(48 * S))).unwrap();
    let rate = s.project.sequence(ItemId(seq)).unwrap().settings.frame_rate;
    let starts: Vec<Tick> = slot_clips(&r, "photos").iter().map(|c| clip(&s, seq, *c).start).collect();
    for (k, t) in starts.iter().enumerate().skip(1) {
        assert_eq!(*t, rate.snap_nearest(beats[4 * k]), "photo {k} starts on the song's beat {}", 4 * k);
    }
    // the look is on the photos, the export preset is Quick Export's now
    let first = clip(&s, seq, slot_clips(&r, "photos")[0]);
    assert!(first.effect("lumetri").is_some());
    assert_eq!(r["look"]["lumetri"], "Golden Dusk");
    assert_eq!(s.export_queue.quick_preset.as_deref(), Some("YouTube 1080p Full HD"));
    s.project.sequence(ItemId(seq)).unwrap().check().unwrap();
    // undo takes all of it away
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.project.items.len(), items_before);
    assert!(s.project.sequence(ItemId(seq)).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_unfilled_music_slot_gets_a_generated_bed_and_variants_resize() {
    let dir = tmp_dir("vt-default-music");
    let mut s = session(&dir);
    let logo = png(&dir, "logo.png", 300, 300, 200);
    let r = s
        .execute(
            "template.apply",
            json!({"template": "builtin:channel-intro", "variant": "9:16", "slots": {"logo": logo, "name": "Robot Lab"}, "name": "Intro Vertical"}),
        )
        .unwrap();
    assert_eq!(r["music"]["generated"], true, "{r}");
    assert_eq!(r["name"], "Intro Vertical");
    let seq = s.project.sequence(ItemId(r["sequence"].as_u64().unwrap())).unwrap();
    assert_eq!((seq.settings.width, seq.settings.height), (1080, 1920));
    assert_eq!(seq.duration(), Tick(5 * S));
    // the music clip covers the intro exactly
    let m = slot_clips(&r, "music");
    let total: Tick = m.iter().map(|c| clip(&s, r["sequence"].as_u64().unwrap(), *c).duration).fold(Tick::ZERO, |a, b| a + b);
    assert_eq!(total, Tick(5 * S));
    assert_eq!(s.export_queue.quick_preset.as_deref(), Some("Social Vertical 1080×1920"));
    // without generation the placeholder stays
    let r = s.execute("template.apply", json!({"template": "builtin:channel-intro", "slots": {"logo": logo}, "generateMusic": false})).unwrap();
    assert_eq!(r["music"], Value::Null);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn short_music_is_remixed_or_looped_to_the_show() {
    let dir = tmp_dir("vt-remix");
    let mut s = session(&dir);
    let music = song(&mut s, &dir, 120.0, 6.0);
    let clip_file = png(&dir, "c.png", 320, 180, 9);
    let r = s.execute("template.apply", json!({"template": "builtin:top-3-listicle", "slots": {"items": [clip_file], "music": music}})).unwrap();
    let seq = r["sequence"].as_u64().unwrap();
    let q = s.project.sequence(ItemId(seq)).unwrap();
    let music_total: Tick = q.audio_tracks[0].items.iter().map(|c| c.duration).fold(Tick::ZERO, |a, b| a + b);
    let fps = q.settings.frame_rate.frame_duration();
    assert!((music_total - q.duration()).abs() <= fps, "music {music_total:?} vs show {:?}: {r}", q.duration());
    assert_eq!(r["remixed"].as_array().unwrap().len(), 1, "{r}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn validate_names_the_field_and_save_round_trips() {
    let dir = tmp_dir("vt-save");
    let mut s = session(&dir);
    let photo = png(&dir, "me.png", 640, 360, 90);
    let r = s.execute("template.apply", json!({"template": "builtin:photo-beat-slideshow", "slots": {"photos": [photo], "title": "Mine"}})).unwrap();
    let seq = r["sequence"].as_u64().unwrap();
    // a clip of the user's own media that is not a slot is left out of the saved template
    let own = s.execute("file.import", json!({"paths": [photo]})).unwrap()["items"][0].as_u64().unwrap();
    s.execute("sequence.addTracks", json!({"video": 1, "audio": 0})).unwrap();
    let vtracks = s.active_sequence().unwrap().video_tracks.len();
    s.execute("timeline.place", json!({"item": own, "time": 0, "track": format!("V{vtracks}"), "audio": false})).unwrap();
    let out = dir.join("Mine.fcvideo-template");
    let saved = s.execute("template.save", json!({"path": out.to_string_lossy(), "sequence": seq, "name": "My Slideshow", "author": "me"})).unwrap();
    assert_eq!(saved["template"]["id"], "user:my-slideshow");
    assert!(saved["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("your own media")), "{saved}");
    let kinds: Vec<&str> = saved["template"]["slots"].as_array().unwrap().iter().map(|x| x["kind"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"photo") && kinds.contains(&"music") && kinds.contains(&"text"), "{kinds:?}");
    // no personal file paths in the package
    let seqfile = std::fs::read_to_string(out.join("sequence.json")).unwrap();
    assert!(!seqfile.contains("me.png"), "the user's photo must not travel with the template");
    let v = s.execute("template.validate", json!({"path": out.to_string_lossy()})).unwrap();
    assert_eq!(v["valid"], true, "{v}");
    // refuses to overwrite
    assert!(s.execute("template.save", json!({"path": out.to_string_lossy(), "sequence": seq})).is_err());
    // and the saved template applies again
    let again = s.execute("template.apply", json!({"template": out.to_string_lossy(), "slots": {"photos": [photo]}})).unwrap();
    assert!(again["sequence"].as_u64().is_some());
    // as a zip too, into the user folder, where template.list finds it
    let user = dir.join("Video Templates");
    std::fs::create_dir_all(&user).unwrap();
    let zip = user.join("mine.zip");
    s.execute("template.save", json!({"path": zip.to_string_lossy(), "sequence": seq, "name": "Zipped"})).unwrap();
    let l = s.execute("template.list", json!({"source": "user"})).unwrap();
    assert!(l["templates"].as_array().unwrap().iter().any(|t| t["name"] == "Zipped"), "{l}");
    // a broken manifest: invalid, with the field named; the library lists it under `invalid`
    let broken = user.join("broken.fcvideo-template");
    std::fs::create_dir_all(&broken).unwrap();
    for (rel, bytes) in std::fs::read_dir(&out).unwrap().flatten().map(|e| (e.file_name(), std::fs::read(e.path()).unwrap())) {
        std::fs::write(broken.join(rel), bytes).unwrap();
    }
    let mut m: Value = serde_json::from_slice(&std::fs::read(broken.join("template.json")).unwrap()).unwrap();
    m["slots"][0]["kind"] = json!("hologram");
    std::fs::write(broken.join("template.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    let v = s.execute("template.validate", json!({"path": broken.to_string_lossy()})).unwrap();
    assert_eq!(v["valid"], false);
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e["field"] == "slots[0].kind"), "{v}");
    let l = s.execute("template.list", json!({})).unwrap();
    assert!(l["invalid"].as_array().unwrap().iter().any(|x| x["path"].as_str().unwrap().contains("broken")), "{l}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn mark_slot_flags_clips_for_save() {
    let dir = tmp_dir("vt-mark");
    let mut s = session(&dir);
    s.execute("file.openDemoProject", json!({})).unwrap();
    let q = s.active_sequence().unwrap().clone();
    let clips: Vec<u64> = q.video_tracks[0].items.iter().take(2).map(|c| c.id.0).collect();
    let r = s.execute("template.markSlot", json!({"clips": clips, "kind": "video", "slot": "shots"})).unwrap();
    assert_eq!(r["clips"], 2);
    let audio = q.audio_tracks.iter().flat_map(|t| t.items.iter()).next().unwrap().id.0;
    assert!(s.execute("template.markSlot", json!({"clips": [audio], "kind": "photo"})).is_err(), "a picture slot on an audio track");
    assert!(s.execute("template.markSlot", json!({"clips": clips, "kind": "sticker"})).is_err());
    assert!(s.execute("template.markSlot", json!({"clips": clips, "kind": "video", "slot": "Bad Id!"})).is_err());
    assert!(s.execute("template.markSlot", json!({"clips": [999_999_999], "kind": "video"})).is_err());
    let out = dir.join("demo.fcvideo-template");
    let saved = s.execute("template.save", json!({"path": out.to_string_lossy(), "name": "Demo Cut"})).unwrap();
    let shots = saved["template"]["slots"].as_array().unwrap().iter().find(|x| x["id"] == "shots").unwrap().clone();
    assert_eq!(shots["placements"], 2);
    // clearing the flag
    s.execute("template.markSlot", json!({"clips": clips, "clear": true})).unwrap();
    assert!(s.active_sequence().unwrap().video_tracks[0].items[0].markers.iter().all(|m| !m.name.starts_with("slot:")));
    std::fs::remove_dir_all(&dir).ok();
}

fn write_zip(path: &PathBuf, files: &[(String, Vec<u8>)]) {
    std::fs::write(path, filmcraft_templates::zip::write(files).unwrap()).unwrap();
}

#[test]
fn hostile_parameters_and_packages_are_refused() {
    let dir = tmp_dir("vt-hostile");
    let mut s = session(&dir);
    let t = "builtin:photo-beat-slideshow";
    let bad: Vec<Value> = vec![
        json!({}),
        json!({"template": 7}),
        json!({"template": t, "slots": []}),
        json!({"template": t, "slots": {"nope": "x"}}),
        json!({"template": t, "slots": {"photos": [true]}}),
        json!({"template": t, "slots": {"photos": [{"path": "/x.png", "item": 3}]}}),
        json!({"template": t, "slots": {"photos": [{"path": "/x.png", "inSeconds": -4}]}}),
        json!({"template": t, "slots": {"photos": [{"path": "/x.png", "inSeconds": f64::MAX}]}}),
        json!({"template": t, "slots": {"photos": [{"path": "/x.png", "focus": [1, 2, 3]}]}}),
        json!({"template": t, "slots": {"photos": [{"item": 1, "focusByVariant": [0.5]}]}}),
        json!({"template": t, "slots": {"photos": [-1]}}),
        json!({"template": t, "slots": {"photos": [18446744073709551615u64]}}),
        json!({"template": t, "slots": {"photos": ["/definitely/not/here.png"]}}),
        json!({"template": t, "slots": {"photos": vec!["/x.png"; 2000]}}),
        json!({"template": t, "slots": {"title": {"nope": 1}}}),
        json!({"template": t, "slots": {"title": "x".repeat(5000)}}),
        json!({"template": t, "variant": "3:2"}),
        json!({"template": "/no/such/template.fcvideo-template"}),
        json!({"template": "../../../etc/passwd.zip"}),
    ];
    for p in bad {
        let before = s.project.items.len();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.apply", p.clone())));
        assert!(r.is_ok(), "panicked on {p}");
        assert!(r.unwrap().is_err(), "accepted {p}");
        assert_eq!(s.project.items.len(), before, "a refused apply leaves the project alone: {p}");
    }
    // packages: traversal, not a zip, a zip bomb-ish entry, junk
    let good = filmcraft_templates::builtin::builtin_templates().into_iter().next().unwrap();
    let mut files = good.files();
    files.push(("assets/x.bin".into(), vec![0; 10]));
    let p1 = dir.join("unlisted.zip");
    write_zip(&p1, &files);
    let p2 = dir.join("junk.zip");
    std::fs::write(&p2, b"PK\x03\x04 this is not really a zip").unwrap();
    let p3 = dir.join("text.zip");
    std::fs::write(&p3, b"hello").unwrap();
    for p in [&p1, &p2, &p3] {
        let v = s.execute("template.validate", json!({"path": p.to_string_lossy()})).unwrap();
        assert_eq!(v["valid"], false, "{}: {v}", p.display());
        assert!(!v["errors"].as_array().unwrap().is_empty());
        assert!(s.execute("template.apply", json!({"template": p.to_string_lossy()})).is_err());
    }
    for p in [json!({}), json!({"path": 3}), json!({"path": "/nope"})] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.validate", p.clone())));
        assert!(r.is_ok());
    }
    for p in [json!({}), json!({"path": ""}), json!({"path": "/tmp/x.zip", "sequence": 999_999})] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.save", p.clone())));
        assert!(r.is_ok_and(|r| r.is_err()), "{p}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
