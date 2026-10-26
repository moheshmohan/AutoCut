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
    // none takes it off and puts back the speed from before it: the clip's length again
    let r = s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "none"})).unwrap();
    assert_eq!(r["clips"][0]["removed"], true, "{r}");
    let off = clip(&s, before.id);
    assert!(!off.has_time_remap() && off.speed == before.speed, "{off:?}");
    assert!((off.duration.0 - before.duration.0).abs() <= frame.0, "{:?} vs {:?}", off.duration, before.duration);
    assert!(off.effect("time_remap").is_none_or(|e| e.param(crate::speed_ramp::RAMP_BASE).is_none()), "nothing remembered");
    // none on a clip without a ramp changes nothing
    let r = s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "none"})).unwrap();
    assert_eq!(r["clips"][0]["removed"], false, "{r}");
    assert_eq!(clip(&s, before.id), off);
}

/// Live (the Akku Tesla hero shot): `ripple: false` with a ramp averaging 2× halved the clip and
/// left a gap, and `none` kept the 204 %. `fit: "length"` keeps the clip's place and length.
#[test]
fn a_ramp_that_keeps_the_length_plays_more_media_or_eases_to_the_source() {
    let mut s = demo();
    let before = last_v1(&s);
    let end = crate::media_duration(&s.project, &s.media, before.item).expect("demo media has a length");
    let frame = s.active_sequence().unwrap().settings.frame_rate.frame_duration();
    // room after In for the whole ramp: the clip plays on into the source's handle
    let r = s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero", "fit": "length", "ripple": false})).unwrap();
    assert_eq!(r["fit"], "length");
    let after = clip(&s, before.id);
    let mean = r["clips"][0]["meanSpeed"].as_f64().unwrap();
    assert!(mean > 1.1, "hero is faster on average: {mean}");
    assert_eq!((after.start, after.duration), (before.start, before.duration), "its place and length");
    // more of the source than before, never past its end (the demo's handle is short: eased)
    assert!(after.source_out() > before.source_out() && after.source_out() <= end + frame, "{:?} {:?} {end:?}", after.source_out(), before.source_out());
    match r["clips"][0].get("eased").and_then(|e| e.as_f64()) {
        None => assert!((after.speed - before.speed * mean).abs() < 1e-6, "{} vs {}", after.speed, before.speed * mean),
        Some(k) => assert!((0.0..1.0).contains(&k) && (end - after.source_out()).0.abs() <= frame.0, "eased to the source's end: {r}"),
    }
    assert!(after.has_time_remap());
    // a ramp on the ramped clip replaces it: the same curve, not one on top of the other
    s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero", "fit": "length"})).unwrap();
    let again = clip(&s, before.id);
    assert_eq!((again.start, again.duration, again.source_in), (after.start, after.duration, after.source_in));
    assert!((again.speed - after.speed).abs() < 1e-9, "{} vs {}", again.speed, after.speed);
    let keys = |it: &TrackItem| filmcraft_project::time_remap::curve_of(it).map(|p| p.keyframes.len());
    assert_eq!(keys(&again), keys(&after));
    // and none with fit: length puts the clip back as it was
    s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "none", "fit": "length"})).unwrap();
    let off = clip(&s, before.id);
    assert!(!off.has_time_remap() && off.speed == before.speed && off.duration == before.duration && off.source_in == before.source_in, "{off:?}");

    // no room after Out (its Out at the media's end): the curve is eased until it fits
    let seq_id = s.state.active_sequence.unwrap();
    let tight = {
        let q = std::sync::Arc::make_mut(&mut s.project).sequence_mut(seq_id).unwrap();
        let (_, it) = q.find_item_mut(before.id).unwrap();
        it.source_in = end - Tick((it.duration.0 as f64 * it.speed.abs() * 1.2) as i64);
        it.clone()
    };
    let r = s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero", "fit": "length"})).unwrap();
    let eased = r["clips"][0]["eased"].as_f64().expect("eased");
    assert!((0.0..1.0).contains(&eased), "{r}");
    let after = clip(&s, before.id);
    assert_eq!((after.start, after.duration), (tight.start, tight.duration));
    assert!(after.source_out() <= end + frame, "{:?} past {end:?}", after.source_out());
    assert!(after.speed > tight.speed * 1.1, "still a ramp faster on average: {}", after.speed);
    // media over the clip still runs forward
    let mut last = Tick(i64::MIN);
    for i in 0..=200 {
        let m = after.source_time_at(after.start + Tick(after.duration.0 * i / 200));
        assert!(m >= last);
        last = m;
    }
    // a bad fit is refused
    assert!(s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero", "fit": "stretch"})).is_err());
    assert!(s.execute("clip.speedRamp", json!({"clips": [before.id.0], "preset": "hero", "fit": 1})).is_err());
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
