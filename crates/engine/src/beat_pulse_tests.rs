//! `sequence.beatPulse`: keyframes on the beats, each kind of pulse, every Nth beat, ranges, undo
//! and refusals.

use filmcraft_project::{ClipId, ParamValue, TrackItem};

use filmcraft_time::Tick;
use serde_json::json;

use crate::Session;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.state.selection.clear();
    s
}

fn clip(s: &Session, id: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(id).map(|(_, i)| i.clone()).unwrap()
}

/// The first V1 clip and a 120 BPM grid of beats inside it.
fn first(s: &Session) -> (TrackItem, Vec<f64>) {
    let it = s.active_sequence().unwrap().video_tracks[0].items[0].clone();
    let (a, b) = (it.start.seconds(), it.end().seconds());
    let beats: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).filter(|t| *t >= a + 0.25 && *t < b - 0.25).collect();
    (it, beats)
}

#[test]
fn zoom_pulses_peak_on_each_beat() {
    let mut s = demo();
    let (it, beats) = first(&s);
    assert!(beats.len() >= 3, "{beats:?}");
    let r = s.execute("sequence.beatPulse", json!({"clips": [it.id.0], "effect": "zoomPulse", "strength": 0.5, "beats": beats})).unwrap();
    assert_eq!(r["clips"][0]["pulses"], json!(beats.len()));
    let after = clip(&s, it.id);
    let scale = after.effect("motion").unwrap().param("scale").unwrap().clone();
    // a keyframe exactly where each beat plays, at +10 %, resting between beats
    for b in &beats {
        let mt = after.effect_time_at(Tick::from_seconds_f64(*b));
        let k = scale.keyframes.iter().find(|k| k.time == mt).unwrap_or_else(|| panic!("no keyframe at the beat {b}"));
        assert!((k.value.as_f64().unwrap() - 110.0).abs() < 1e-9, "{:?}", k.value);
    }
    let between = after.effect_time_at(Tick::from_seconds_f64(beats[0] + 0.4));
    assert!((scale.f64_at(between) - 100.0).abs() < 1e-9, "settled by the next beat");
    assert_eq!(scale.keyframes.len(), beats.len() * 3);
    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(clip(&s, it.id), it);
}

#[test]
fn every_kind_and_every_nth_beat() {
    for (kind, effect, param) in [("flash", "brightness_contrast", "brightness"), ("glitch", "cr_glitch", "amount"), ("shake", "motion", "position")] {
        let mut s = demo();
        let (it, beats) = first(&s);
        s.execute("sequence.beatPulse", json!({"clips": [it.id.0], "effect": kind, "strength": 1.0, "everyNthBeat": 2, "beats": beats})).unwrap();
        let after = clip(&s, it.id);
        let e = after.effect(effect).unwrap_or_else(|| panic!("{kind}: {effect} added"));
        let p = e.param(param).unwrap();
        let pulsed: Vec<f64> = beats.iter().step_by(2).copied().collect();
        for b in &pulsed {
            let mt = after.effect_time_at(Tick::from_seconds_f64(*b));
            assert!(p.keyframes.iter().any(|k| k.time == mt), "{kind}: keyframe on {b}");
        }
        for b in beats.iter().skip(1).step_by(2) {
            let mt = after.effect_time_at(Tick::from_seconds_f64(*b));
            assert!(!p.keyframes.iter().any(|k| k.time == mt), "{kind}: no pulse on the off beat {b}");
        }
        match kind {
            "flash" => assert!(p.f64_at(after.effect_time_at(Tick::from_seconds_f64(pulsed[0]))) >= 69.0),
            "glitch" => {
                assert_eq!(p.f64_at(after.effect_time_at(Tick::from_seconds_f64(pulsed[0]))), 100.0);
                assert_eq!(e.param("rhythm").map(|r| r.value.clone()), Some(ParamValue::Choice(0)));
            }
            _ => {
                let at = after.effect_time_at(Tick::from_seconds_f64(pulsed[0]));
                let v = p.vec2_at(at);
                let st = &s.active_sequence().unwrap().settings;
                let rest = it.effect("motion").and_then(|m| m.param("position")).map(|q| q.vec2_at(at)).unwrap();
                let (rx, ry) =
                    (if rest.x.is_nan() { f64::from(st.width) / 2.0 } else { rest.x }, if rest.y.is_nan() { f64::from(st.height) / 2.0 } else { rest.y });
                assert!(((v.x - rx).powi(2) + (v.y - ry).powi(2)).sqrt() > 10.0, "shaken off its place: {v:?}");
            }
        }
    }
}

#[test]
fn ranges_and_bad_params() {
    let mut s = demo();
    let (it, beats) = first(&s);
    // a range: clips overlapping it, beats inside it only
    let (a, b) = (beats[0] - 0.1, beats[1] + 0.1);
    let r = s.execute("sequence.beatPulse", json!({"range": {"start": a, "end": b}, "effect": "zoomPulse", "beats": beats})).unwrap();
    let pulses: u64 = r["clips"].as_array().unwrap().iter().filter_map(|c| c["pulses"].as_u64()).sum();
    assert_eq!(pulses, 2, "{r}");
    let _ = it;
    for p in [
        json!({"effect": "wobble", "beats": beats}),
        json!({"effect": "flash", "strength": 0, "beats": beats}),
        json!({"effect": "flash", "strength": 1.5, "beats": beats}),
        json!({"effect": "flash", "everyNthBeat": 0, "beats": beats}),
        json!({"effect": "flash", "decay": -1, "beats": beats}),
        json!({"effect": "flash", "range": {"start": 5, "end": 2}, "beats": beats}),
        json!({"effect": "flash", "clips": [987_654_321u64], "beats": beats}),
        json!({"effect": "flash", "range": {"start": 1000, "end": 1001}, "beats": beats}),
        json!({"effect": "flash", "beats": [f64::MAX]}),
    ] {
        assert!(s.execute("sequence.beatPulse", p.clone()).is_err(), "{p}");
    }
}
