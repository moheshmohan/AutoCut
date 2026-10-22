//! `clip.speedRamp` on the demo project: the curve, the new length, linked sound, undo, and
//! refusals.

use filmcraft_project::{ClipId, TimeInterpolation, TrackItem};
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

/// The last clip of V1 (nothing after it to run into).
fn last_v1(s: &Session) -> TrackItem {
    s.active_sequence().unwrap().video_tracks[0].items.last().cloned().unwrap()
}

#[test]
fn a_ramp_keeps_the_media_and_changes_the_length_by_the_mean_speed() {
    let mut s = demo();
    let before = last_v1(&s);
    let r = s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero"})).unwrap();
    let after = clip(&s, before.id);
    let mean = r["clips"][0]["meanSpeed"].as_f64().unwrap();
    // the same media, In to Out (to a frame of rounding)
    assert_eq!(after.source_in, before.source_in);
    let frame = s.active_sequence().unwrap().settings.frame_rate.frame_duration();
    assert!(((after.source_out() - before.source_out()).0.abs() as f64) < frame.0 as f64 * mean.max(1.0) + 2.0);
    // the length: the media over the mean speed, on whole frames
    let want = before.duration.0 as f64 / mean;
    assert!((after.duration.0 as f64 - want).abs() <= frame.0 as f64, "{:?} vs {want}", after.duration);
    assert!(after.has_time_remap(), "a keyframed curve");
    assert_eq!(after.time_interpolation, TimeInterpolation::OpticalFlow, "slow parts interpolate");
    // the picture follows the curve: fast at the start, slow in the middle
    let step = |a: f64, b: f64| {
        let t = |f: f64| after.start + Tick((after.duration.0 as f64 * f) as i64);
        (after.source_time_at(t(b)) - after.source_time_at(t(a))).0 as f64 / (t(b) - t(a)).0 as f64
    };
    assert!(step(0.05, 0.15) > 2.0 && step(0.48, 0.52) < 0.4, "{} {}", step(0.05, 0.15), step(0.48, 0.52));
    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    let undone = clip(&s, before.id);
    assert_eq!(undone, before);
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(clip(&s, before.id), after);
    // none takes it off and keeps the length
    s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "none"})).unwrap();
    let off = clip(&s, before.id);
    assert!(!off.has_time_remap() && off.duration == after.duration);
}

#[test]
fn intensity_zero_is_no_ramp_and_linked_sound_ramps_too() {
    let mut s = demo();
    s.execute("sequence.linkedSelection", json!({"on": true})).unwrap();
    let q = s.active_sequence().unwrap();
    // a video clip with a linked sound clip
    let v = q
        .video_tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .find(|i| i.link.is_some() && q.audio_tracks.iter().flat_map(|t| t.items.iter()).any(|a| a.link == i.link));
    let v = v.cloned().expect("the demo has a clip with linked sound");
    let r = s.execute("clip.speedRamp", json!({"clips": [v.id.0], "preset": "Flash In", "intensity": 0.0, "maintainPitch": true})).unwrap();
    let after = clip(&s, v.id);
    assert!((r["clips"][0]["meanSpeed"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert!((after.duration.0 - v.duration.0).abs() <= 1, "no ramp, no length change");
    let q = s.active_sequence().unwrap();
    let a = q.audio_tracks.iter().flat_map(|t| t.items.iter()).find(|a| a.link == v.link).unwrap();
    assert!(a.has_time_remap() && a.maintain_pitch, "the sound carries the same curve and keeps its pitch");
    assert_eq!(a.duration, after.duration);
    assert_ne!(a.time_interpolation, TimeInterpolation::OpticalFlow);
}

#[test]
fn every_preset_applies_and_bad_params_are_refused() {
    for preset in filmcraft_project::time_remap::RAMP_PRESETS {
        let mut s = demo();
        let c = last_v1(&s);
        s.execute("clip.speedRamp", json!({"clips": [c.id.0], "preset": preset.id, "intensity": 1.5})).unwrap_or_else(|e| panic!("{}: {e}", preset.id));
        let after = clip(&s, c.id);
        assert!(after.has_time_remap(), "{}", preset.id);
        // monotonic media over the clip
        let mut last = Tick(i64::MIN);
        for i in 0..=400 {
            let m = after.source_time_at(after.start + Tick(after.duration.0 * i / 400));
            assert!(m >= last, "{}", preset.id);
            last = m;
        }
    }
    let mut s = demo();
    let c = last_v1(&s);
    for p in [
        json!({"clips": [c.id.0]}),
        json!({"clips": [c.id.0], "preset": "warp9"}),
        json!({"clips": [c.id.0], "preset": "hero", "intensity": -1}),
        json!({"clips": [c.id.0], "preset": "hero", "intensity": 2.5}),
        json!({"clips": [c.id.0], "preset": "hero", "intensity": "lots"}),
        json!({"clips": [c.id.0], "preset": "hero", "maintainPitch": 3}),
        json!({"clips": [987_654_321u64], "preset": "hero"}),
    ] {
        assert!(s.execute("clip.speedRamp", p.clone()).is_err(), "{p}");
    }
    // a reversed clip is refused, nothing changes
    s.execute("clip.speedDuration", json!({"clips": [c.id.0], "speed": 100, "reverse": true})).unwrap();
    let rev = clip(&s, c.id);
    assert!(s.execute("clip.speedRamp", json!({"clips": [c.id.0], "preset": "hero"})).is_err());
    assert_eq!(clip(&s, c.id), rev);
}
