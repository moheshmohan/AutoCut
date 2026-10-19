//! New Adjustment Clip (`timeline.newAdjustmentClip`): placement, one undo step, hostile
//! parameters, and the picture: an effect on the adjustment clip renders exactly like the same
//! effect on every clip below it, for the adjustment clip's duration only.

use filmcraft_project::ItemKind;
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::Session;

/// A 24 fps sequence with two color mattes back to back on V1 (0–2 s, 2–4 s).
fn session(video_tracks: u64) -> (Session, u64, u64) {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "adj", "fps": 24, "width": 64, "height": 36, "video": video_tracks, "audio": 0})).unwrap();
    let mut clips = Vec::new();
    for (color, at) in [("#c03020", 0.0), ("#2050c0", 2.0)] {
        let item = s.execute("file.newColorMatte", json!({"color": color, "seconds": 2})).unwrap()["item"].as_u64().unwrap();
        let r = s.execute("timeline.place", json!({"item": item, "track": "V1", "seconds": at})).unwrap();
        clips.push(r["clips"][0].as_u64().unwrap());
    }
    s.execute("edit.deselectAll", json!({})).unwrap();
    (s, clips[0], clips[1])
}

fn frame(s: &Session, seconds: f64) -> Vec<f32> {
    s.render_program_at(1.0, Tick::from_seconds_f64(seconds)).unwrap().px
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn grade(s: &mut Session, clip: u64) {
    s.execute("effects.apply", json!({"clips": [clip], "effect": "lumetri"})).unwrap();
    s.execute("effects.setParam", json!({"clip": clip, "effect": "lumetri", "param": "exposure", "value": 1.5})).unwrap();
    s.execute("effects.setParam", json!({"clip": clip, "effect": "lumetri", "param": "saturation", "value": 40.0})).unwrap();
}

#[test]
fn an_adjustment_clip_renders_like_the_effect_on_each_clip_below() {
    let (mut s, a, b) = session(2);
    let plain = [frame(&s, 1.0), frame(&s, 3.0), frame(&s, 3.5)];
    let r = s.execute("timeline.newAdjustmentClip", json!({"startSeconds": 0.0, "seconds": 3.25, "effects": ["lumetri"]})).unwrap();
    let adj = r["clip"].as_u64().unwrap();
    assert_eq!(r["trackName"], "V2", "{r}");
    assert_eq!(r["duration"].as_i64(), Some(Tick::from_seconds_f64(3.25).0));
    assert!(r["effects"].as_array().unwrap().iter().any(|e| e == "lumetri"), "{r}");
    let item = s.project.item(filmcraft_project::ItemId(r["item"].as_u64().unwrap())).unwrap();
    assert!(matches!(item.kind, ItemKind::AdjustmentLayer { .. }));
    for p in ["exposure", "saturation"] {
        let v = if p == "exposure" { 1.5 } else { 40.0 };
        s.execute("effects.setParam", json!({"clip": adj, "effect": "lumetri", "param": p, "value": v})).unwrap();
    }
    let adjusted = [frame(&s, 1.0), frame(&s, 3.0), frame(&s, 3.5)];
    assert!(max_diff(&adjusted[0], &plain[0]) > 0.05, "the grade changes the picture");
    // the same grade on each clip, without the adjustment clip
    s.execute("clip.enable", json!({"clips": [adj]})).unwrap();
    grade(&mut s, a);
    grade(&mut s, b);
    for (i, t) in [1.0, 3.0].into_iter().enumerate() {
        let per_clip = frame(&s, t);
        let d = max_diff(&adjusted[i], &per_clip);
        assert!(d < 1e-4, "at {t} s the adjustment clip and the per-clip effect differ by {d}");
    }
    // past the adjustment clip's end the picture is untouched
    assert!(max_diff(&adjusted[2], &plain[2]) < 1e-6, "after its end nothing changes");
}

#[test]
fn placement_selection_new_tracks_and_one_undo_step() {
    let (mut s, a, b) = session(1);
    let items = s.project.items.len();
    // over the selected clips; no track above them, so a new one
    s.execute("timeline.select", json!({"clips": [a, b]})).unwrap();
    let r = s.execute("timeline.newAdjustmentClip", json!({})).unwrap();
    assert_eq!(r["trackName"], "V2");
    assert_eq!(r["start"].as_i64(), Some(0));
    assert_eq!(r["duration"].as_i64(), Some(Tick::from_seconds_f64(4.0).0));
    assert_eq!(s.active_sequence().unwrap().video_tracks.len(), 2);
    assert_eq!(s.state.selection, vec![filmcraft_project::ClipId(r["clip"].as_u64().unwrap())], "the new clip is selected");
    // one undo step takes the clip, the item and the new track away
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.project.items.len(), items);
    assert_eq!(s.active_sequence().unwrap().video_tracks.len(), 1);
    // with nothing selected and no In/Out: 5 s from the playhead, on the first free track above
    s.execute("sequence.addTracks", json!({"video": 2})).unwrap();
    s.execute("edit.deselectAll", json!({})).unwrap();
    s.execute("playhead.set", json!({"seconds": 1.0})).unwrap();
    let r = s.execute("timeline.newAdjustmentClip", json!({"name": "Look"})).unwrap();
    assert_eq!((r["trackName"].as_str(), r["seconds"].as_f64()), (Some("V2"), Some(5.0)), "{r}");
    assert_eq!(s.project.item(filmcraft_project::ItemId(r["item"].as_u64().unwrap())).unwrap().name, "Look");
    // a second one over the same span stacks above it
    let r2 = s.execute("timeline.newAdjustmentClip", json!({})).unwrap();
    assert_eq!(r2["trackName"], "V3");
    // an explicit free track
    let r3 = s.execute("timeline.newAdjustmentClip", json!({"track": "V3", "startSeconds": 7.0, "seconds": 1.0})).unwrap();
    assert_eq!(r3["trackName"], "V3");
}

#[test]
fn hostile_parameters_are_refused_and_leave_nothing_behind() {
    let (mut s, _, _) = session(2);
    let before = (s.project.items.len(), s.history.undo.len());
    for p in [
        json!({"seconds": -1.0}),
        json!({"seconds": 1e300}),
        json!({"duration": i64::MAX, "startTime": i64::MAX}),
        json!({"startSeconds": -5.0}),
        json!({"seconds": 0.0}),
        json!({"effects": "not_an_effect"}),
        json!({"effects": ["lumetri", "no_such_effect"]}),
        json!({"effects": [1, 2]}),
        json!({"effects": vec!["lumetri"; 100]}),
        json!({"track": "V1", "startSeconds": 0.0, "seconds": 1.0}),
        json!({"track": "V9"}),
        json!({"track": "A1"}),
        json!({"track": {"x": 1}}),
    ] {
        let e = s.execute("timeline.newAdjustmentClip", p.clone());
        assert!(e.is_err(), "{p} should be refused: {e:?}");
        assert_eq!((s.project.items.len(), s.history.undo.len()), before, "{p} left something behind");
    }
    s.execute("timeline.setTrack", json!({"track": "V2", "locked": true})).unwrap();
    let r: Value = s.execute("timeline.newAdjustmentClip", json!({"startSeconds": 0.0, "seconds": 1.0})).unwrap();
    assert_eq!(r["trackName"], "V3", "a locked track is passed over");
}
