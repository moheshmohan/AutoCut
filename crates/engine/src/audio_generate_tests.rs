//! Tests of [`crate::audio_generate`]: generated cues are real stereo WAVs, placed without
//! overwriting, one undo step, reproducible by seed, and hostile parameters are refused.

use super::*;
use filmcraft_project::ClipId;
use serde_json::json;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-gen-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d.to_string_lossy().into_owned()
}

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

/// (channels, rate, frames) of a float WAV written by `write_wav_f32_stereo`.
fn wav_shape(path: &str) -> (u16, u32, usize) {
    let b = std::fs::read(path).unwrap();
    let ch = u16::from_le_bytes([b[22], b[23]]);
    let rate = u32::from_le_bytes([b[24], b[25], b[26], b[27]]);
    let data = u32::from_le_bytes([b[40], b[41], b[42], b[43]]) as usize;
    (ch, rate, data / (4 * ch as usize))
}

#[test]
fn whoosh_is_written_imported_and_placed_as_one_undo_step() {
    let mut s = demo();
    let dir = tmp("whoosh");
    let undo0 = s.history.undo.len();
    let clips0: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    let r = s.execute("audio.generate", json!({"kind": "whoosh", "duration": 1.5, "seconds": 3.0, "dir": dir})).unwrap();
    let path = r["path"].as_str().unwrap();
    assert!(path.ends_with("Whoosh 1.wav"), "{path}");
    assert_eq!(wav_shape(path), (2, 48_000, 72_000));
    let (_, it) = s.active_sequence().unwrap().find_item(ClipId(r["clip"].as_u64().unwrap())).unwrap();
    assert_eq!(it.start, Tick::from_seconds_f64(3.0));
    assert_eq!(it.duration, Tick::from_units(72_000, 48_000));
    assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
    // nothing was overwritten: one more clip than before
    let clips1: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    assert_eq!(clips1, clips0 + 1);
    s.undo().unwrap();
    let clips2: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    assert_eq!(clips2, clips0);
}

#[test]
fn a_second_cue_of_a_kind_gets_a_new_name_and_seed_and_its_own_place() {
    let mut s = demo();
    let dir = tmp("second");
    let a = s.execute("audio.generate", json!({"kind": "impact", "duration": 1.0, "seconds": 1.0, "dir": dir})).unwrap();
    let b = s.execute("audio.generate", json!({"kind": "impact", "duration": 1.0, "seconds": 1.0, "dir": dir})).unwrap();
    assert_ne!(a["seed"], b["seed"]);
    assert!(b["path"].as_str().unwrap().ends_with("Impact 2.wav"));
    // the same moment: the second goes on another (free) track
    assert_ne!(a["track"], b["track"]);
    // an explicit seed reproduces the audio exactly
    let c = s.execute("audio.generate", json!({"kind": "impact", "duration": 1.0, "seed": a["seed"], "place": false, "dir": dir})).unwrap();
    assert_eq!(std::fs::read(a["path"].as_str().unwrap()).unwrap(), std::fs::read(c["path"].as_str().unwrap()).unwrap());
    assert!(c["clip"].is_null(), "place: false only imports");
}

#[test]
fn music_bed_takes_mood_key_and_tempo() {
    let mut s = demo();
    let dir = tmp("music");
    let r = s
        .execute("audio.generate", json!({"kind": "music", "duration": 6.0, "mood": "uplifting", "key": "C", "bpm": 120, "seconds": 0.0, "dir": dir}))
        .unwrap();
    assert_eq!(r["kind"], "music");
    assert!((r["seconds"].as_f64().unwrap() - 6.0).abs() < 1e-6);
    let kinds = s.execute("audio.generateKinds", json!({})).unwrap();
    assert_eq!(kinds["kinds"].as_array().unwrap().len(), 7);
}

#[test]
fn hostile_parameters_are_refused_before_anything_is_written() {
    let mut s = demo();
    let dir = tmp("hostile");
    let undo0 = s.history.undo.len();
    for p in [
        json!({"kind": "explosion", "dir": dir}),
        json!({"kind": "whoosh", "duration": -1.0, "dir": dir}),
        json!({"kind": "whoosh", "duration": 0.0, "dir": dir}),
        json!({"kind": "music", "key": "H", "dir": dir}),
        json!({"kind": "music", "mood": "jazzy", "dir": dir}),
        json!({"kind": "whoosh", "track": "A99", "dir": dir}),
    ] {
        assert!(s.execute("audio.generate", p.clone()).is_err(), "{p}");
    }
    assert_eq!(s.history.undo.len(), undo0);
    assert!(std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) == 0, "no file was written");
    // a huge duration is clamped, not allocated
    let r = s.execute("audio.generate", json!({"kind": "whoosh", "duration": 1e9, "place": false, "dir": dir})).unwrap();
    assert!(r["seconds"].as_f64().unwrap() <= filmcraft_audio_dsp::synth::MAX_SFX_SECONDS + 1e-9);
}

#[test]
fn a_busy_track_is_refused() {
    let mut s = demo();
    let dir = tmp("busy");
    // A1 holds the demo's clips from 0 s
    let e = s.execute("audio.generate", json!({"kind": "whoosh", "duration": 1.0, "seconds": 0.5, "track": "A1", "dir": dir})).unwrap_err();
    assert!(e.to_string().contains("has clips there"), "{e}");
}

#[test]
fn generated_files_go_in_a_folder_beside_the_project_not_loose_in_it() {
    let mut s = demo();
    let dir = tmp("beside");
    std::fs::create_dir_all(&dir).unwrap();
    let project = std::path::Path::new(&dir).join("Show.fcproj").to_string_lossy().into_owned();
    s.path = Some(project);
    let r = s.execute("audio.generate", json!({"kind": "impact", "duration": 0.5, "seconds": 1.0})).unwrap();
    let path = std::path::PathBuf::from(r["path"].as_str().unwrap());
    assert_eq!(path.parent().unwrap(), std::path::Path::new(&dir).join("Generated Audio"));
    assert!(path.exists());
    // an explicit `dir` is used as it is
    let other = tmp("explicit");
    let r = s.execute("audio.generate", json!({"kind": "impact", "duration": 0.5, "seconds": 3.0, "dir": other})).unwrap();
    assert_eq!(std::path::Path::new(r["path"].as_str().unwrap()).parent().unwrap(), std::path::Path::new(&other));
}
