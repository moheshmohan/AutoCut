//! Tests of [`crate::music`] (`music.suggest`, `music.taxonomy`) and of `audio.generate`'s
//! music: exact lengths, the beat map and taxonomy in the result and the item's metadata,
//! stems placed on their own tracks as one undo step, hostile parameters refused.

use super::*;
use filmcraft_project::ClipId;
use filmcraft_time::TICKS_PER_SECOND;
use serde_json::json;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-music-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d.to_string_lossy().into_owned()
}

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

fn ticks(v: &serde_json::Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

#[test]
fn suggest_ranks_styles_for_a_brief_and_its_params_generate_that_track() {
    let mut s = demo();
    let r = s.execute("music.suggest", json!({"mood": "relaxing", "useCase": "tutorial", "duration": 30})).unwrap();
    let sug = r["suggestions"].as_array().unwrap();
    assert_eq!(sug.len(), 3);
    assert_eq!((sug[0]["style"]["mood"].as_str(), sug[0]["style"]["genre"].as_str()), (Some("calm"), Some("ambient")), "{}", sug[0]);
    assert!(sug[0]["why"].as_str().unwrap().contains("tutorial"));
    // how a 30 s track would be arranged: sections, the final hit on a downbeat, the ring-out
    let t = &sug[0]["track"];
    assert_eq!(t["length"].as_i64(), Some(30 * TICKS_PER_SECOND));
    assert_eq!(t["ending"].as_i64().unwrap() + t["tail"].as_i64().unwrap(), 30 * TICKS_PER_SECOND);
    assert!(t["sections"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(r["vocabulary"]["useCases"].as_array().unwrap().iter().any(|u| u == "vlog"));
    // the params make exactly that track
    let mut params = sug[0]["params"].clone();
    params["place"] = json!(false);
    params["dir"] = json!(tmp("suggest"));
    let g = s.execute("audio.generate", params).unwrap();
    assert_eq!((g["mood"].as_str(), g["genre"].as_str()), (Some("calm"), Some("ambient")));
    assert_eq!(g["ending"], t["ending"]);
    assert_eq!(g["sections"], t["sections"]);
    assert_eq!(g["seconds"].as_f64(), Some(30.0));
    // more, fewer
    let r = s.execute("music.suggest", json!({"energy": 0.95, "bpm": 128, "count": 10})).unwrap();
    assert_eq!(r["suggestions"].as_array().unwrap().len(), 10);
    assert_eq!(r["suggestions"][0]["style"]["mood"], "energetic");
    let r = s.execute("music.suggest", json!({"energy": [0.2, 0.9, 0.3], "count": 1, "durationTicks": TICKS_PER_SECOND * 12})).unwrap();
    assert_eq!(r["suggestions"][0]["params"]["energy"], json!([0.2, 0.9, 0.3]));
    assert_eq!(r["seconds"], 12.0);
}

#[test]
fn suggest_refuses_hostile_parameters() {
    let mut s = demo();
    let undo0 = s.history.undo.len();
    for p in [
        json!({"mood": "jazzy"}),
        json!({"mood": 7}),
        json!({"genre": "polka"}),
        json!({"genre": ["pop"]}),
        json!({"useCase": "party"}),
        json!({"energy": 1.5}),
        json!({"energy": -0.1}),
        json!({"energy": "loud"}),
        json!({"energy": []}),
        json!({"energy": vec![0.5; 65]}),
        json!({"energy": [0.5, "x"]}),
        json!({"energy": [0.5, 2.0]}),
        json!({"energy": {"a": 1}}),
        json!({"bpm": 10}),
        json!({"bpm": 1e300}),
        json!({"bpm": "fast"}),
        json!({"duration": 0}),
        json!({"duration": -3}),
        json!({"duration": 1e300}),
        json!({"duration": 901}),
        json!({"duration": "long"}),
        json!({"duration": 0.01}),
        json!({"durationTicks": 0}),
        json!({"durationTicks": -5}),
        json!({"durationTicks": 1.5}),
        json!({"durationTicks": i64::MAX}),
        json!({"count": 0}),
        json!({"count": 11}),
        json!({"count": 2.5}),
        json!({"count": -1}),
        json!({"seed": -1}),
        json!({"seed": "x"}),
    ] {
        let e = s.execute("music.suggest", p.clone());
        assert!(e.is_err(), "{p} → {e:?}");
    }
    assert_eq!(s.history.undo.len(), undo0, "a query changes nothing");
    let t = s.execute("music.taxonomy", json!({})).unwrap();
    assert_eq!(t["moods"].as_array().unwrap().len(), 9);
    assert_eq!(t["genres"].as_array().unwrap().len(), 6);
    assert_eq!(t["stems"], json!(["drums", "bass", "melody", "pad"]));
    assert!(t["styles"].as_array().unwrap().len() >= 9);
    assert!(t["moods"][1]["synonyms"].as_array().unwrap().iter().any(|w| w == "suspense"));
}

#[test]
fn music_is_exactly_as_long_as_asked_with_its_beat_map_and_taxonomy() {
    let mut s = demo();
    let dir = tmp("exact");
    // an awkward length: 7 s and a fraction of a sample
    let length = TICKS_PER_SECOND * 7 + 12_345_678;
    let r =
        s.execute("audio.generate", json!({"kind": "music", "durationTicks": length, "mood": "uplifting", "bpm": 120, "seconds": 2.0, "dir": dir})).unwrap();
    let samples = (i128::from(length) * 48_000 * 2 + i128::from(TICKS_PER_SECOND)) / (2 * i128::from(TICKS_PER_SECOND));
    assert_eq!(r["samples"].as_i64(), Some(samples as i64));
    let (_, it) = s.active_sequence().unwrap().find_item(ClipId(r["clip"].as_u64().unwrap())).unwrap();
    assert_eq!(it.duration, Tick::from_units(samples as i64, 48_000));
    assert_eq!(it.start, Tick::from_seconds_f64(2.0));
    // the beat map: every beat at 120 BPM from 0, the final hit on a downbeat
    let beats = ticks(&r["beats"]);
    let downbeats = ticks(&r["downbeats"]);
    assert_eq!(beats[0], 0);
    assert!(beats.windows(2).all(|w| w[1] - w[0] == TICKS_PER_SECOND / 2), "120 BPM: half a second a beat");
    assert_eq!(downbeats.last().copied(), r["ending"].as_i64());
    assert_eq!(beats.last().copied(), r["ending"].as_i64());
    assert!(r["tail"].as_i64().unwrap() >= TICKS_PER_SECOND * 6 / 5);
    assert_eq!(r["ending"].as_i64().unwrap() + r["tail"].as_i64().unwrap(), length);
    assert_eq!(r["taxonomy"]["mood"], "uplifting");
    assert_eq!(r["taxonomy"]["genre"], "pop", "the mood's usual genre");
    assert_eq!(r["key"], "A major");
    assert_eq!(r["chords"].as_array().unwrap().len(), downbeats.len());
    // the item carries it too, and so does the file
    let item = s.project.item(ItemId(r["item"].as_u64().unwrap())).unwrap();
    assert!(item.metadata.get("Description").is_some_and(|d| d.contains("Uplifting") && d.contains("120 BPM")), "{:?}", item.metadata);
    assert_eq!(item.metadata.get("Music Genre").map(String::as_str), Some("Pop"));
    let map: serde_json::Value = serde_json::from_str(item.metadata.get("Beat Map").unwrap()).unwrap();
    assert_eq!(map["milliBpm"], 120_000);
    assert_eq!(map["ending"], r["ending"]);
    let bytes = std::fs::read(r["path"].as_str().unwrap()).unwrap();
    let data = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    assert_eq!(data, samples as usize * 8);
    let tail = &bytes[44 + data..];
    assert!(tail.starts_with(b"LIST") && tail.windows(4).any(|w| w == b"IGNR"), "INFO tags after the samples");
    let riff = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    assert_eq!(riff + 8, bytes.len());
}

#[test]
fn stems_go_on_their_own_tracks_as_one_undo_step() {
    let mut s = demo();
    let dir = tmp("stems");
    let undo0 = s.history.undo.len();
    let clips0: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    let r = s.execute("audio.generate", json!({"kind": "music", "duration": 6.0, "mood": "energetic", "stems": true, "seconds": 30.0, "dir": dir})).unwrap();
    let stems = r["stems"].as_array().unwrap();
    assert_eq!(stems.iter().map(|x| x["stem"].as_str().unwrap()).collect::<Vec<_>>(), ["drums", "bass", "melody", "pad"]);
    let tracks: std::collections::HashSet<String> = stems.iter().map(|x| x["track"].as_str().unwrap().to_string()).collect();
    assert_eq!(tracks.len(), 4, "each on its own track: {stems:?}");
    assert!(r["clip"].is_null(), "the stems are placed instead of the mix");
    for x in stems {
        assert!(std::path::Path::new(x["path"].as_str().unwrap()).exists());
        let item = s.project.item(ItemId(x["item"].as_u64().unwrap())).unwrap();
        assert_eq!(item.metadata.get(STEM_KEY_TEST), x["stem"].as_str().map(str::to_string).as_ref());
    }
    assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
    let clips1: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    assert_eq!(clips1, clips0 + 4);
    assert_eq!(s.state.selection.len(), 4);
    // stems do not use up seeds: the next cue is Music 2 with seed 2
    let n = s.execute("audio.generate", json!({"kind": "music", "duration": 2.0, "place": false, "dir": dir})).unwrap();
    assert_eq!(n["seed"], 2);
    assert!(n["path"].as_str().unwrap().ends_with("Music 2.wav"));
    s.undo().unwrap();
    s.undo().unwrap();
    let clips2: usize = s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum();
    assert_eq!(clips2, clips0);
}

const STEM_KEY_TEST: &str = crate::audio_generate::STEM_KEY;

#[test]
fn music_options_are_validated_before_anything_is_written() {
    let mut s = demo();
    let dir = tmp("hostile");
    let undo0 = s.history.undo.len();
    for p in [
        json!({"kind": "music", "genre": "polka", "dir": dir}),
        json!({"kind": "music", "energy": 3, "dir": dir}),
        json!({"kind": "music", "energy": [], "dir": dir}),
        json!({"kind": "music", "energy": "high", "dir": dir}),
        json!({"kind": "music", "stems": "yes", "dir": dir}),
        json!({"kind": "music", "stems": true, "duration": 301, "dir": dir}),
        json!({"kind": "whoosh", "stems": true, "dir": dir}),
        json!({"kind": "music", "levels": {"vocals": -3}, "dir": dir}),
        json!({"kind": "music", "levels": {"drums": "loud"}, "dir": dir}),
        json!({"kind": "music", "levels": {"drums": 40}, "dir": dir}),
        json!({"kind": "music", "levels": [1, 2], "dir": dir}),
        json!({"kind": "music", "mute": ["drums", "bass", "melody", "pad"], "dir": dir}),
        json!({"kind": "music", "mute": "drums", "dir": dir}),
        json!({"kind": "music", "mute": [3], "dir": dir}),
        json!({"kind": "music", "durationTicks": -1, "dir": dir}),
        json!({"kind": "music", "durationTicks": "x", "dir": dir}),
        json!({"kind": "music", "duration": 0.01, "dir": dir}),
        json!({"kind": "music", "duration": 2.0, "track": "A99", "dir": dir}),
        json!({"kind": "music", "duration": 2.0, "seconds": 0.5, "track": "A1", "dir": dir}),
    ] {
        assert!(s.execute("audio.generate", p.clone()).is_err(), "{p}");
    }
    assert_eq!(s.history.undo.len(), undo0);
    assert!(std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) == 0, "no file was written");
    // a muted stem and a level are fine; an old-style call (intensity, no energy) still works
    let r =
        s.execute("audio.generate", json!({"kind": "music", "duration": 3.0, "mute": ["drums"], "levels": {"pad": -6}, "place": false, "dir": dir})).unwrap();
    assert_eq!(r["seconds"], 3.0);
    let r = s.execute("audio.generate", json!({"kind": "music", "duration": 3.0, "intensity": 0.2, "mood": "suspense", "place": false, "dir": dir})).unwrap();
    assert_eq!((r["mood"].as_str(), r["energy"].as_f64()), (Some("tense"), Some(0.2)));
    // a huge length is clamped to the longest, as before (planned, not rendered here)
    let r = s.execute("music.suggest", json!({"duration": 900, "count": 1})).unwrap();
    assert_eq!(r["seconds"], 900.0);
}
