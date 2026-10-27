//! Tests of [`crate::transition_tools`]: snapping transitions to beats (explicit grids, odd tempos,
//! the music's own beats, hostile lists), one undo step each; sounds coupled to transitions, placed
//! to the sample and gone with the transition.

use super::*;
use filmcraft_project::ClipId;
use filmcraft_time::TICKS_PER_SECOND;
use serde_json::json;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-trtools-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d.to_string_lossy().into_owned()
}

/// V1's transitions as (id, start, duration), by start.
fn v1(s: &Session) -> Vec<(u64, Tick, Tick)> {
    let mut v: Vec<_> = s.active_sequence().unwrap().video_tracks[0].transitions.iter().map(|x| (x.id.0, x.start, x.duration)).collect();
    v.sort_by_key(|x| x.1);
    v
}

fn frame(s: &Session) -> Tick {
    s.active_sequence().unwrap().settings.frame_rate.frame_duration()
}

/// Every snapped transition: its midpoint within half a frame of its beat, its length the musical
/// fraction it reports (whole frames), still over its cut.
fn check_snapped(s: &Session, r: &Value, period: Tick) {
    let fr = frame(s);
    let seq = s.active_sequence().unwrap();
    for t in r["snapped"].as_array().unwrap() {
        let id = filmcraft_project::TransitionId(t["transition"].as_u64().unwrap());
        let (track, tr) = seq.all_tracks().find_map(|k| k.transitions.iter().find(|x| x.id == id).map(|x| (k, x))).unwrap();
        assert_eq!((tr.start.0, tr.duration.0), (t["start"].as_i64().unwrap(), t["duration"].as_i64().unwrap()));
        let (_, cut, _) = filmcraft_edit::transitions::bounds(track, tr).unwrap();
        if tr.from.is_some() && tr.to.is_some() {
            let beat = Tick(t["beat"].as_i64().unwrap());
            let mid = tr.start + tr.duration.mul_ratio(1, 2);
            assert!((mid - beat).abs() <= fr.mul_ratio(1, 2), "{t}: midpoint off its beat");
            assert_eq!(t["offBeat"].as_i64().unwrap(), (mid - beat).0);
            assert!(tr.start <= cut && cut <= tr.end(), "{t}: over its cut");
        } else {
            assert!(t["beat"].is_null(), "{t}: a fade keeps its edge");
        }
        let (n, d) = match t["beats"].as_str().unwrap() {
            "1/4" => (1, 4),
            "1/2" => (1, 2),
            "1" => (1, 1),
            "2" => (2, 1),
            "4" => (4, 1),
            other => panic!("{other}"),
        };
        assert!((tr.duration - period.mul_ratio(n, d)).abs() <= fr.mul_ratio(1, 2) + Tick(1) || tr.duration == fr, "{t}: musical length");
    }
}

#[test]
fn snapping_to_a_beat_grid_is_one_undo_step() {
    let mut s = demo();
    let before = v1(&s);
    assert_eq!(before.len(), 4);
    let undo0 = s.history.undo.len();
    // 100 BPM, the first beat 70 ms in (off the frame grid), listed out of order with repeats
    let mut beats: Vec<f64> = (0..200).map(|k| 0.07 + k as f64 * 0.6).collect();
    beats.reverse();
    beats.extend([0.67, 0.67, 1.27]);
    let r = s.execute("transitions.snapToBeats", json!({"beats": beats})).unwrap();
    assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
    assert_eq!(r["period"].as_i64(), Some(Tick::from_seconds_f64(0.6).0));
    assert!((r["bpm"].as_f64().unwrap() - 100.0).abs() < 1e-6);
    assert_eq!(r["snapped"].as_array().unwrap().len() + r["skipped"].as_array().unwrap().len(), 4);
    assert!(r["snapped"].as_array().unwrap().len() >= 3, "{r}");
    check_snapped(&s, &r, Tick::from_seconds_f64(0.6));
    assert_ne!(v1(&s), before);
    // snapping again changes nothing (but is still allowed)
    let again = v1(&s);
    s.execute("transitions.snapToBeats", json!({"beats": beats})).unwrap();
    assert_eq!(v1(&s), again);
    s.undo().unwrap();
    s.undo().unwrap();
    assert_eq!(v1(&s), before, "undo restores every transition");
}

#[test]
fn odd_tempos_tracks_and_subsets() {
    for bpm in [77.7, 203.0, 41.3, 133.33] {
        let mut s = demo();
        let period = Tick::from_seconds_f64(60.0 / bpm);
        let beats: Vec<f64> = (0..400).map(|k| 0.013 + k as f64 * 60.0 / bpm).collect();
        let r = s.execute("transitions.snapToBeats", json!({"beats": beats, "bpm": bpm, "track": "V1"})).unwrap();
        assert_eq!(r["period"].as_i64(), Some(period.0), "{bpm}");
        check_snapped(&s, &r, period);
    }
    // only the transitions asked for move
    let mut s = demo();
    let before = v1(&s);
    let one = before[2].0;
    let r = s.execute("transitions.snapToBeats", json!({"beats": [0.5, 1.0, 1.5], "bpm": 120, "transitions": [one]})).unwrap();
    let after = v1(&s);
    for (b, a) in before.iter().zip(&after) {
        if b.0 != one {
            assert_eq!(b, a);
        }
    }
    // (that beat is far from its cut: skipped, nothing moved)
    assert_eq!(r["skipped"][0]["transition"].as_u64(), Some(one), "{r}");
    // a locked track is reported, not edited
    let mut s = demo();
    s.execute("timeline.setTrack", json!({"track": "V1", "locked": true})).unwrap();
    let before = v1(&s);
    let beats: Vec<f64> = (0..200).map(|k| k as f64 * 0.5).collect();
    let r = s.execute("transitions.snapToBeats", json!({"beats": beats})).unwrap();
    assert!(r["snapped"].as_array().unwrap().is_empty());
    assert_eq!(r["skipped"].as_array().unwrap().len(), 4);
    assert_eq!(v1(&s), before);
}

#[test]
fn hostile_beat_lists_and_params_are_refused_without_a_change() {
    let mut s = demo();
    let before = v1(&s);
    let undo0 = s.history.undo.len();
    let many: Vec<f64> = (0..crate::transition_tools::MAX_BEATS + 1).map(|k| k as f64 * 0.01).collect();
    for p in [
        json!({"beats": "lots"}),
        json!({"beats": [1, "x"]}),
        json!({"beats": [1, null]}),
        json!({"beats": [-1.0, 2.0]}),
        json!({"beats": [1e9, 2.0]}),
        json!({"beats": []}),
        json!({"beats": [5.0]}),
        json!({"beats": [5.0, 5.0, 5.0]}),
        json!({"beats": many}),
        json!({"beats": [1.0, 2.0], "bpm": 0}),
        json!({"beats": [1.0, 2.0], "bpm": "fast"}),
        json!({"beats": [1.0, 2.0], "bpm": 1e9}),
        json!({"beats": [1.0, 2.0], "bpm": -120}),
        json!({"beats": [1.0, 2.0], "track": "V99"}),
        json!({"beats": [1.0, 2.0], "track": "X1"}),
        json!({"beats": [1.0, 2.0], "transitions": [123_456]}),
        json!({"beats": [1.0, 2.0], "transitions": "all"}),
        json!({"beats": [1.0, 2.0], "sequence": 987_654}),
        json!({"music": 999_999}),
        json!({"music": "song"}),
        json!({"music": u64::MAX}),
    ] {
        assert!(s.execute("transitions.snapToBeats", p.clone()).is_err(), "{p}");
        assert_eq!(v1(&s), before, "{p}");
        assert_eq!(s.history.undo.len(), undo0, "{p}");
    }
    // a video clip is not music
    let video = s.active_sequence().unwrap().video_tracks[0].items[0].id.0;
    assert!(s.execute("transitions.snapToBeats", json!({"music": video})).is_err());
    // one beat with a tempo is a grid of one: transitions out of its reach are skipped
    let r = s.execute("transitions.snapToBeats", json!({"beats": [0.0], "bpm": 90})).unwrap();
    assert!(r["skipped"].as_array().unwrap().len() >= 3, "{r}");
}

#[test]
fn snapping_to_the_music_uses_its_beats() {
    let mut s = demo();
    let dir = tmp("music");
    let m = s.execute("audio.generate", json!({"kind": "music", "bpm": 120, "duration": 30, "seed": 4, "seconds": 0.0, "dir": dir})).unwrap();
    let clip = m["clip"].as_u64().unwrap();
    let item = m["item"].as_u64().unwrap();
    let beats = s.execute("media.beats", json!({"item": item})).unwrap();
    let beat_ticks: Vec<Tick> = beats["beats"].as_array().unwrap().iter().map(|b| Tick::from_seconds_f64(b.as_f64().unwrap())).collect();
    assert!(beat_ticks.len() > 20, "{beats}");
    let r = s.execute("transitions.snapToBeats", json!({"music": clip})).unwrap();
    assert_eq!(r["source"]["clip"].as_u64(), Some(clip), "{r}");
    assert!(!r["snapped"].as_array().unwrap().is_empty(), "{r}");
    // each snapped transition sits on one of the music's beats (those `media.beats` reports)
    for t in r["snapped"].as_array().unwrap() {
        if let Some(b) = t["beat"].as_i64() {
            assert!(beat_ticks.iter().any(|x| (x.0 - b).abs() <= TICKS_PER_SECOND / 1_000_000), "{t}: not a beat of the music");
        }
    }
    let period = Tick(r["period"].as_i64().unwrap());
    check_snapped(&s, &r, period);
    // without `music` the sequence's longest audio clip is the music (the 30 s bed here)
    s.undo().unwrap();
    let r2 = s.execute("transitions.snapToBeats", json!({})).unwrap();
    assert_eq!(r2["source"]["clip"].as_u64(), Some(clip));
    // a reversed music clip is refused
    s.execute("clip.setReverse", json!({"clips": [clip], "reverse": true})).ok();
    if s.active_sequence().unwrap().find_item(ClipId(clip)).is_some_and(|(_, i)| i.reverse) {
        assert!(s.execute("transitions.snapToBeats", json!({"music": clip})).is_err());
    }
}

#[test]
fn transition_sounds_are_placed_to_the_sample_and_go_with_their_transition() {
    let mut s = demo();
    let seq = s.active_sequence().unwrap();
    let sr = i64::from(seq.settings.sample_rate);
    let tr = seq.video_tracks[0].transitions.iter().find(|x| x.from.is_some() && x.to.is_some()).unwrap().clone();
    let mid = tr.start + tr.duration.mul_ratio(1, 2);
    let mid_sample = mid.to_rational_round(1, sr);
    let undo0 = s.history.undo.len();
    for (kind, len_s) in [("riser", 2.0), ("impact", 2.5), ("shutter", 0.25), ("pop", 0.25), ("thump", 0.8), ("whoosh", 0.0)] {
        let r = s.execute("transitions.addSound", json!({"transitions": [tr.id.0], "kind": kind, "gain": -3})).unwrap();
        let snd = &r["sounds"][0];
        let (first, n) = (snd["startSample"].as_i64().unwrap(), snd["samples"].as_i64().unwrap());
        match kind {
            "riser" => assert_eq!((first + n, n), (mid_sample, (len_s * sr as f64) as i64)),
            "impact" | "pop" | "thump" => assert_eq!((first, n), (mid_sample, (len_s * sr as f64) as i64)),
            // its first click, 2 ms in, on the swap
            "shutter" => assert!((first + sr / 500 - mid_sample).abs() <= 1 && n == (len_s * sr as f64) as i64, "{snd}"),
            _ => assert!((first + n / 2 - mid_sample).abs() <= 1, "{snd}"),
        }
        assert_eq!(snd["gain"].as_f64(), Some(-3.0));
        let stored = s.active_sequence().unwrap().video_tracks[0].transitions.iter().find(|x| x.id == tr.id).unwrap().sound.clone().unwrap();
        assert_eq!(stored.kind.id(), kind);
        assert_eq!(stored.seed, tr.id.0, "a stable seed of its own");
    }
    assert_eq!(s.history.undo.len(), undo0 + 6, "one undo step each");
    // the sequence reports it
    let q = s.execute("sequence.inspect", json!({})).unwrap();
    let js = q["video"][0]["transitions"].as_array().unwrap().iter().find(|t| t["id"] == tr.id.0).unwrap().clone();
    assert_eq!(js["sound"]["kind"], "whoosh", "{js}");
    // a sound set length, then removed by command
    let r = s.execute("transitions.addSound", json!({"transitions": [tr.id.0], "kind": "impact", "seconds": 0.75, "seed": 42, "intensity": 0.2})).unwrap();
    assert_eq!(r["sounds"][0]["samples"].as_i64(), Some(36_000 * sr / 48_000));
    s.execute("transitions.removeSound", json!({"transitions": [tr.id.0]})).unwrap();
    assert!(s.active_sequence().unwrap().video_tracks[0].transitions.iter().all(|x| x.sound.is_none()));
    assert!(s.execute("transitions.removeSound", json!({"transitions": [tr.id.0]})).is_err(), "nothing left to remove");
    // ...and removed with its transition
    s.execute("transitions.addSound", json!({"transitions": [tr.id.0]})).unwrap();
    s.execute("sequence.removeTransition", json!({"transitions": [tr.id.0]})).unwrap();
    assert!(s.active_sequence().unwrap().all_tracks().flat_map(|t| t.transitions.iter()).all(|x| x.sound.is_none()));
    s.undo().unwrap();
    assert!(s.active_sequence().unwrap().video_tracks[0].transitions.iter().any(|x| x.id == tr.id && x.sound.is_some()), "undo brings both back");
    // it follows its transition when the transition is snapped to the beat
    let beats: Vec<f64> = (0..200).map(|k| 0.03 + k as f64 * 0.5).collect();
    s.execute("transitions.snapToBeats", json!({"beats": beats, "transitions": [tr.id.0]})).unwrap();
    let moved = s.active_sequence().unwrap().video_tracks[0].transitions.iter().find(|x| x.id == tr.id).unwrap().clone();
    let r = s.execute("transitions.addSound", json!({"transitions": [tr.id.0], "kind": "impact"})).unwrap();
    assert_eq!(r["sounds"][0]["startSample"].as_i64(), Some((moved.start + moved.duration.mul_ratio(1, 2)).to_rational_round(1, sr)));
}

#[test]
fn hostile_sound_params_are_refused() {
    let mut s = demo();
    let seq = s.active_sequence().unwrap();
    let video = seq.video_tracks[0].transitions[0].id.0;
    let audio = seq.audio_tracks[0].transitions[0].id.0;
    let undo0 = s.history.undo.len();
    for p in [
        json!({}),
        json!({"transitions": []}),
        json!({"transitions": [999_999]}),
        json!({"transitions": "x"}),
        json!({"transitions": [audio]}),
        json!({"transitions": [video], "kind": "drone"}),
        json!({"transitions": [video], "kind": 7}),
        json!({"transitions": [video], "gain": 1e300}),
        json!({"transitions": [video], "gain": "loud"}),
        json!({"transitions": [video], "intensity": 2}),
        json!({"transitions": [video], "seconds": 0}),
        json!({"transitions": [video], "seconds": -1}),
        json!({"transitions": [video], "seconds": 1e9}),
        json!({"transitions": [video], "seconds": "long"}),
        json!({"transitions": [video], "seed": -4}),
    ] {
        assert!(s.execute("transitions.addSound", p.clone()).is_err(), "{p}");
    }
    assert_eq!(s.history.undo.len(), undo0);
    // locked tracks are refused
    s.execute("timeline.setTrack", json!({"track": "V1", "locked": true})).unwrap();
    assert!(s.execute("transitions.addSound", json!({"transitions": [video]})).is_err());
    // the selection is the default
    s.execute("timeline.setTrack", json!({"track": "V1", "locked": false})).unwrap();
    s.execute("timeline.select", json!({"clips": [], "transitions": [video]})).unwrap();
    let r = s.execute("transitions.addSound", json!({"kind": "Riser"})).unwrap();
    assert_eq!(r["sounds"][0]["transition"].as_u64(), Some(video));
}
