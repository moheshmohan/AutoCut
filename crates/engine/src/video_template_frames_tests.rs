//! Video template thumbnails, previews and slot media, and every built-in starter applied through
//! the engine with generated media: one undo step, its Elements sounds synthesized, captions
//! transcribed, a frame rendered, the project saved and opened again; hostile parameters refused.

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::video_template_frames::{THUMB_SIDE, candidates, is_google_photos_pick, thumbnail};
use filmcraft_project::{ItemId, MediaRef, Transcript, Word};
use filmcraft_templates::builtin::{SOUND_SECONDS, shared};
use filmcraft_templates::manifest::SlotKind;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

const S: i64 = TICKS_PER_SECOND;

fn session(dir: &Path) -> Session {
    Session { prefs_path: Some(dir.join("prefs.json")), ..Default::default() }
}

/// The demo project (generated media), generated files going to `dir`.
fn demo(dir: &Path) -> Session {
    let mut s = session(dir);
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.execute("file.projectSettings.scratchDisks", json!({"captured": dir.to_string_lossy()})).unwrap();
    s
}

/// A picture with something in it: several distinct shades (a flat card with text has a few dozen at
/// its anti-aliased edges; an empty frame has one).
fn varied(rgba: &[u8]) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    for px in rgba.as_chunks::<4>().0.iter() {
        seen.insert((px[0] / 8, px[1] / 8, px[2] / 8));
        if seen.len() > 12 {
            return true;
        }
    }
    false
}

fn png(dir: &Path, name: &str, w: u32, h: u32) -> String {
    let p = dir.join(name);
    image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 7 % 256) as u8, (y * 5 % 256) as u8, 120])).save(&p).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn every_starter_has_a_thumbnail_in_every_layout() {
    let dir = tmp_dir("vtf-thumbs");
    let s = session(&dir);
    for pkg in shared() {
        let m = &pkg.manifest;
        for v in &m.variants {
            let f = thumbnail(&s, &m.id, Some(&v.id), None, 160).unwrap_or_else(|e| panic!("{} {}: {e}", m.id, v.id));
            assert_eq!(f.width.max(f.height), 160, "{} {}", m.id, v.id);
            let want = f64::from(v.width) / f64::from(v.height);
            assert!((f64::from(f.width) / f64::from(f.height) - want).abs() < 0.03, "{} {}: {}x{}", m.id, v.id, f.width, f.height);
            assert_eq!(f.rgba.len(), (f.width * f.height * 4) as usize);
            assert!(varied(&f.rgba), "{} {}: the thumbnail shows something", m.id, v.id);
            assert_eq!(f.at, m.poster_time().min(m.duration), "{} at its poster frame", m.id);
        }
    }
    // drawn once
    let a = thumbnail(&s, "builtin:quote-card", None, None, 160).unwrap();
    let b = thumbnail(&s, "Quote Card", None, None, 160).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_thumbnail_and_preview_commands() {
    let dir = tmp_dir("vtf-cmds");
    let mut s = demo(&dir);
    let r = s.execute("template.thumbnail", json!({"template": "builtin:robot-showcase", "variant": "9:16"})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(180), Some(u64::from(THUMB_SIDE))));
    let png = filmcraft_project::gtemplate::base64_decode(r["pngBase64"].as_str().unwrap()).unwrap();
    assert!(png.starts_with(b"\x89PNG"));
    let out = dir.join("t.png");
    let r = s.execute("template.thumbnail", json!({"template": "builtin:robot-showcase", "seconds": 5.0, "path": out.to_string_lossy()})).unwrap();
    assert_eq!(r["seconds"], 5.0);
    assert!(std::fs::read(&out).unwrap().starts_with(b"\x89PNG"));
    // a preview with the project's own media changes nothing in the project
    let clips: Vec<u64> = candidates(&s, Some(SlotKind::Video)).iter().map(|c| c.item.0).take(5).collect();
    assert!(clips.len() >= 3);
    let (items, undo, next) = (s.project.items.len(), s.history.undo.len(), s.project.next_id);
    let r = s
        .execute(
            "template.preview",
            json!({"template": "builtin:top-5-listicle", "variant": "1:1", "slots": {"items": clips, "title": "My Picks"}, "frames": 3, "maxSide": 200}),
        )
        .unwrap();
    let frames = r["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!((frames[0]["width"].as_u64(), frames[0]["height"].as_u64()), (Some(200), Some(200)));
    assert!(frames.windows(2).all(|w| w[0]["time"].as_i64() < w[1]["time"].as_i64()));
    assert_eq!((s.project.items.len(), s.history.undo.len(), s.project.next_id), (items, undo, next), "the project did not change");
    // to files
    let pdir = dir.join("preview");
    let r = s.execute("template.preview", json!({"template": "builtin:quote-card", "frames": 2, "dir": pdir.to_string_lossy()})).unwrap();
    assert_eq!(r["frames"].as_array().unwrap().len(), 2);
    assert!(pdir.join("preview-02.png").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn template_media_lists_what_can_fill_a_slot() {
    let dir = tmp_dir("vtf-media");
    let mut s = demo(&dir);
    let video = candidates(&s, Some(SlotKind::Video));
    let music = candidates(&s, Some(SlotKind::Music));
    assert!(!video.is_empty() && !music.is_empty());
    assert!(video.iter().all(|c| c.width > 0 && c.kind != "audio"));
    assert!(music.iter().all(|c| c.has_audio));
    assert!(candidates(&s, Some(SlotKind::Text)).is_empty());
    // a Google Photos pick: its file is in the picker's download cache
    let cache = dir.join("google-photos").join("cache").join("AF1Qip123");
    std::fs::create_dir_all(&cache).unwrap();
    let pick = png(&cache, "IMG_0001.png", 64, 48);
    s.execute("file.import", json!({"paths": [pick]})).unwrap();
    let r = s.execute("template.media", json!({"kind": "photo", "googlePhotos": true})).unwrap();
    let picks = r["media"].as_array().unwrap();
    assert_eq!(picks.len(), 1, "{r}");
    assert_eq!((picks[0]["name"].as_str(), picks[0]["kind"].as_str(), picks[0]["googlePhotos"].as_bool()), (Some("IMG_0001.png"), Some("photo"), Some(true)));
    assert!(is_google_photos_pick("C:\\Users\\me\\AppData\\FilmCraft\\google-photos\\cache\\x\\a.mp4"));
    assert!(!is_google_photos_pick("/Users/me/Movies/google-photos.mp4"));
    assert!(s.execute("template.media", json!({"kind": "sticker"})).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

/// `template.apply` slots for `id` from the demo project's media.
fn demo_slots(s: &Session, id: &str, photo: &str) -> Value {
    let m = &filmcraft_templates::builtin::find(id).unwrap().manifest;
    let video: Vec<u64> = candidates(s, Some(SlotKind::Video)).iter().filter(|c| c.kind == "video").map(|c| c.item.0).collect();
    let mut slots = serde_json::Map::new();
    for sl in &m.slots {
        let n = sl.count as usize;
        let v = match sl.kind {
            SlotKind::Text => json!(
                (0..n)
                    .map(|i| format!("{} {}", sl.name, i + 1).chars().take(sl.constraints.max_chars.unwrap_or(30) as usize).collect::<String>())
                    .collect::<Vec<_>>()
            ),
            SlotKind::Video => json!((0..n).map(|i| video[i % video.len()]).collect::<Vec<_>>()),
            SlotKind::Photo | SlotKind::Logo => json!((0..n).map(|_| photo.to_string()).collect::<Vec<_>>()),
            SlotKind::Music | SlotKind::Captions => continue,
        };
        slots.insert(sl.id.clone(), v);
    }
    Value::Object(slots)
}

#[test]
fn every_starter_applies_with_generated_media_and_renders() {
    let dir = tmp_dir("vtf-apply-all");
    let mut s = demo(&dir);
    let mut t = Transcript { language: "en".into(), ..Default::default() };
    for (k, w) in ["Hello", "and", "welcome", "to", "the", "show", "today"].iter().enumerate() {
        t.words.push(Word::new(*w, Tick::from_seconds_f64(0.3 + 0.35 * k as f64), Tick::from_seconds_f64(0.6 + 0.35 * k as f64)));
    }
    s.transcriber = Some(Arc::new(filmcraft_speech::FixedTranscriber { transcript: t, id: "fixed".into() }));
    let photo = png(&dir, "photo.png", 1200, 900);
    let mut made = Vec::new();
    for pkg in shared() {
        let m = &pkg.manifest;
        let undo = s.history.undo.len();
        let slots = demo_slots(&s, &m.id, &photo);
        let r = s.execute("template.apply", json!({"template": m.id, "slots": slots, "generateMusic": false})).unwrap_or_else(|e| panic!("{}: {e}", m.id));
        assert_eq!(s.history.undo.len(), undo + 1, "{}: one undo step", m.id);
        assert_eq!(r["duration"].as_i64(), Some(m.duration.0), "{}: exactly its length: {r}", m.id);
        let seq = ItemId(r["sequence"].as_u64().unwrap());
        // the Elements sounds are the synthesized ones now
        let stand_ins = pkg.fragment.items.values().filter(|i| i.metadata.contains_key(filmcraft_templates::ELEMENT_METADATA)).count();
        assert_eq!(r["sounds"].as_array().map_or(0, Vec::len), stand_ins, "{}: {r}", m.id);
        let q = s.project.sequence(seq).unwrap();
        for c in q.audio_tracks.iter().flat_map(|t| t.items.iter()) {
            let it = s.project.item(c.item).unwrap();
            if it.metadata.contains_key(filmcraft_templates::ELEMENT_METADATA) {
                assert!(matches!(it.as_media().map(|x| &x.media), Some(MediaRef::File { .. })), "{}: {} is a WAV", m.id, it.name);
            }
        }
        // captions of what the speakers say
        if m.captions_slot().is_some() {
            assert!(r["captions"]["captions"].as_u64().unwrap_or(0) > 0, "{}: {r}", m.id);
            assert!(!q.caption_tracks[0].captions.is_empty());
        }
        // a frame of it
        let img = s.try_render_program_at(0.2, m.poster_time()).unwrap_or_else(|e| panic!("{}: {e}", m.id));
        assert!(varied(&img.over_black_rgba8()), "{}: the frame shows something", m.id);
        made.push((seq, m.duration));
    }
    // saved and opened again, every sequence the same length
    let path = dir.join("starters.fcproj");
    s.execute("file.saveAs", json!({"path": path.to_string_lossy()})).unwrap();
    let mut back = session(&dir);
    back.execute("file.open", json!({"path": path.to_string_lossy()})).unwrap();
    for (seq, dur) in made {
        assert_eq!(back.project.sequence(seq).map(|q| q.duration()), Some(dur));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn generated_music_keeps_the_length_and_cuts_on_frames() {
    let dir = tmp_dir("vtf-music");
    let mut s = demo(&dir);
    let photo = png(&dir, "p.png", 800, 600);
    let slots = demo_slots(&s, "builtin:event-recap", &photo);
    let r = s.execute("template.apply", json!({"template": "builtin:event-recap", "slots": slots, "variant": "9:16"})).unwrap();
    assert_eq!(r["music"]["generated"], true, "{r}");
    assert_eq!(r["music"]["genre"], "electronic");
    assert_eq!(r["duration"].as_i64(), Some(16 * S), "{r}");
    let q = s.project.sequence(ItemId(r["sequence"].as_u64().unwrap())).unwrap();
    let frame = q.settings.frame_rate;
    for c in q.video_tracks[0].items.iter() {
        assert_eq!(frame.snap_nearest(c.start), c.start);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn without_speech_to_text_the_captions_wait_with_a_warning() {
    let dir = tmp_dir("vtf-nocaptions");
    let mut s = demo(&dir);
    let video = candidates(&s, Some(SlotKind::Video)).into_iter().find(|c| c.has_audio).unwrap().item.0;
    let r = s.execute("template.apply", json!({"template": "builtin:talking-head-captions", "slots": {"speaker": video}, "generateMusic": false})).unwrap();
    let q = s.project.sequence(ItemId(r["sequence"].as_u64().unwrap())).unwrap();
    if crate::transcript::speech_available() {
        return;
    }
    assert!(q.caption_tracks[0].captions.is_empty(), "the preview's captions are not the speaker's words");
    assert!(r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap_or("").starts_with("auto-captions")), "{r}");
    // cues given directly are used as they are
    let r = s
        .execute(
            "template.apply",
            json!({"template": "builtin:talking-head-captions", "slots": {"speaker": video, "captions": [{"startSeconds": 1.0, "endSeconds": 2.5, "text": "Hi there"}]}, "generateMusic": false}),
        )
        .unwrap();
    let q = s.project.sequence(ItemId(r["sequence"].as_u64().unwrap())).unwrap();
    assert_eq!(q.caption_tracks[0].captions.len(), 1);
    assert_eq!(q.caption_tracks[0].captions[0].text, "Hi there");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_starters_name_only_what_the_engine_has() {
    // the stand-ins' lengths are the Elements sounds' own
    for (id, secs) in SOUND_SECONDS {
        let snd = crate::elements::sound(id).unwrap_or_else(|| panic!("{id}"));
        assert_eq!(snd.seconds(), secs, "{id}");
    }
    assert_eq!(SOUND_SECONDS.len(), crate::elements::SOUNDS.len());
    for m in filmcraft_templates::manifest::MOODS {
        assert!(filmcraft_audio_dsp::music::Mood::from_id(m).is_some(), "{m}");
    }
    for g in filmcraft_templates::manifest::GENRES {
        assert!(filmcraft_audio_dsp::music::Genre::ALL.iter().any(|x| x.id() == g), "{g}");
    }
    // looks, export presets, sounds, moods: nothing a starter names is missing
    let dir = tmp_dir("vtf-checks");
    let mut s = session(&dir);
    for pkg in shared() {
        let i = s.execute("template.inspect", json!({"template": pkg.manifest.id})).unwrap();
        assert_eq!(i["checks"].as_array().unwrap().len(), 0, "{}: {}", pkg.manifest.id, i["checks"]);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn hostile_parameters_are_refused() {
    let dir = tmp_dir("vtf-hostile");
    let mut s = demo(&dir);
    let cases = [
        ("template.thumbnail", json!({"template": "builtin:countdown", "maxSide": 0})),
        ("template.thumbnail", json!({"template": "builtin:countdown", "maxSide": 1_000_000})),
        ("template.thumbnail", json!({"template": "builtin:countdown", "maxSide": "big"})),
        ("template.thumbnail", json!({"template": "builtin:countdown", "seconds": -1.0})),
        ("template.thumbnail", json!({"template": "builtin:countdown", "seconds": "noon"})),
        ("template.thumbnail", json!({"template": "builtin:countdown", "variant": "3:2"})),
        ("template.thumbnail", json!({"template": ""})),
        ("template.thumbnail", json!({"template": "../../../etc/passwd"})),
        ("template.preview", json!({"template": "builtin:countdown", "frames": 0})),
        ("template.preview", json!({"template": "builtin:countdown", "frames": 99})),
        ("template.preview", json!({"template": "builtin:countdown", "variant": "7:3"})),
        ("template.preview", json!({"template": "builtin:photo-collage", "slots": {"photos": "/etc/passwd"}})),
        ("template.preview", json!({"template": "builtin:photo-collage", "slots": {"photos": [987_654_321]}})),
        ("template.preview", json!({"template": "builtin:photo-collage", "slots": "all"})),
        ("template.preview", json!({"template": "builtin:talking-head-captions", "slots": {"captions": [{"start": -5, "end": 2, "text": "x"}]}})),
        (
            "template.apply",
            json!({"template": "builtin:talking-head-captions", "slots": {"captions": [{"startSeconds": 1e300, "endSeconds": 2, "text": "x"}]}}),
        ),
        ("template.apply", json!({"template": "builtin:talking-head-captions", "slots": {"captions": [{"start": 0, "end": 10, "text": "x".repeat(5000)}]}})),
        ("template.apply", json!({"template": "builtin:talking-head-captions", "slots": {"captions": "words"}})),
        ("template.apply", json!({"template": "builtin:talking-head-captions", "slots": {"captions": [{"start": 5, "end": 1, "text": "back"}]}})),
        ("template.markSlot", json!({"kind": "captions", "clips": [1]})),
        ("template.list", json!({"kind": 5})),
    ];
    for (cmd, p) in cases {
        let items = s.project.items.len();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(cmd, p.clone())));
        let r = r.unwrap_or_else(|_| panic!("{cmd} {p} panicked"));
        if cmd != "template.list" {
            assert!(r.is_err(), "{cmd} {p}: {r:?}");
        }
        assert_eq!(s.project.items.len(), items, "{cmd} {p}: nothing added");
    }
    // a template file whose stand-in names no sound still applies, silently, with a warning
    let mut pkg = filmcraft_templates::builtin::find("builtin:countdown").unwrap().clone();
    if let Some(it) = pkg.fragment.items.values_mut().find(|i| i.metadata.contains_key(filmcraft_templates::ELEMENT_METADATA)) {
        it.metadata.insert(filmcraft_templates::ELEMENT_METADATA.into(), "sound.not-a-sound".into());
    }
    pkg.manifest.id = "user:odd-countdown".into();
    let zip = pkg.to_zip().unwrap();
    let path = dir.join("odd.zip");
    std::fs::write(&path, zip).unwrap();
    let v = s.execute("template.validate", json!({"path": path.to_string_lossy()})).unwrap();
    assert_eq!(v["valid"], false, "{v}");
    let r = s.execute("template.apply", json!({"template": path.to_string_lossy(), "generateMusic": false})).unwrap();
    assert!(r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap_or("").contains("sound.not-a-sound")), "{r}");
    std::fs::remove_dir_all(&dir).ok();
}
