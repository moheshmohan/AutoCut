//! Cut-outs with a stand-in cutter: the outline written as a switched-off mask, and each effect's
//! copies, titles and plate above the clip.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filmcraft_media::DemoScene;
use filmcraft_project::{ClipId, MaskMode, TrackItem};
use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::media_test_util::{make_movie, session_with, tmp_dir};

/// A person 40 × 80 pixels walking right 12 pixels a frame it is shown (picture pixels).
#[derive(Default)]
struct Walker {
    calls: AtomicUsize,
}

impl PersonCutter for Walker {
    fn cut(&self, _: &[u8], w: usize, h: usize, _: Pick, _: Option<[f32; 4]>, points: usize) -> std::result::Result<Option<Cut>, String> {
        let k = self.calls.fetch_add(1, Ordering::Relaxed) as f32;
        let (x, y) = (10.0 + 12.0 * k, h as f32 * 0.2);
        if x + 40.0 > w as f32 {
            return Ok(None);
        }
        let outline: Vec<(f32, f32)> = (0..points)
            .map(|i| {
                let a = i as f32 / points as f32 * std::f32::consts::TAU;
                (x + 20.0 + 20.0 * a.sin(), y + 40.0 - 40.0 * a.cos())
            })
            .collect();
        Ok(Some(Cut { bbox: [x, y, 40.0, 80.0], outline }))
    }
}

/// A session with a 2 s clip (320 × 180) on V1 and the walker as the cutter; the clip.
fn session() -> (Session, ClipId) {
    let dir = tmp_dir("cutout");
    let movie = dir.join("walk.mp4");
    make_movie(&movie, DemoScene::Aurora, 320, 180, 48);
    let (mut s, _, _) = session_with(&[&movie]);
    s.cutouts.cutter = Some(Arc::new(Walker::default()));
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    (s, clip)
}

fn clip(s: &Session, c: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
}

fn cut(s: &mut Session, c: ClipId) -> Value {
    s.execute("clip.cutOut", json!({"clip": c.0, "wait": true})).unwrap()
}

#[test]
fn the_person_is_outlined_as_a_switched_off_mask() {
    let (mut s, c) = session();
    let before = s.history.undo.len();
    let r = cut(&mut s, c);
    assert!(r["outlines"].as_u64().unwrap() >= 8, "{r}");
    let it = clip(&s, c);
    let m = person_mask(&it).expect("the Person mask");
    assert_eq!(m.mode, MaskMode::None, "the clip still shows whole");
    assert!(!it.has_opacity_masks());
    // every 0.2 s of media, 96 points, moving right
    let keys = &m.path.keyframes;
    assert!(keys.windows(2).all(|w| (w[1].time - w[0].time).seconds() > 0.19 && (w[1].time - w[0].time).seconds() < 0.21));
    let xs: Vec<f64> = keys.iter().map(|k| m.path_at(k.time).vertices.iter().map(|v| v.p.x).sum::<f64>() / 96.0).collect();
    assert!(xs.windows(2).all(|w| w[1] > w[0]), "{xs:?}");
    assert_eq!(m.path_at(keys[0].time).vertices.len(), 96);
    // one undo step takes it away
    assert_eq!(s.history.undo.len(), before + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(person_mask(&clip(&s, c)).is_none());
}

#[test]
fn the_effects_need_the_person_first_and_refuse_bad_parameters() {
    let (mut s, c) = session();
    for cmd in ["clip.cloneTrail", "clip.shadowClone", "clip.personGlow", "clip.timeFreeze"] {
        let e = s.execute(cmd, json!({"clip": c.0})).unwrap_err().to_string();
        assert!(e.contains("clip.cutOut"), "{cmd}: {e}");
    }
    cut(&mut s, c);
    for (cmd, p) in [
        ("clip.cloneTrail", json!({"clip": c.0, "copies": 0})),
        ("clip.cloneTrail", json!({"clip": c.0, "copies": 99})),
        ("clip.shadowClone", json!({"clip": c.0, "delay": -1})),
        ("clip.shadowClone", json!({"clip": c.0, "opacity": 400})),
        ("clip.personGlow", json!({"clip": c.0, "color": "red"})),
        ("clip.textBehind", json!({"clip": c.0})),
        ("clip.timeFreeze", json!({"clip": c.0, "at": 99})),
        ("clip.cutOut", json!({"clip": c.0, "every": 0})),
        ("clip.cutOut", json!({"clip": c.0, "points": 3})),
        ("clip.cutOut", json!({"clip": c.0, "person": [2, 0.5]})),
        ("clip.cutOut", json!({"clip": 424_242})),
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} {p}");
    }
}

/// The video tracks above V1 holding a clip, and their clips.
fn above(s: &Session) -> Vec<Vec<TrackItem>> {
    s.active_sequence().unwrap().video_tracks.iter().skip(1).map(|t| t.items.clone()).filter(|v| !v.is_empty()).collect()
}

#[test]
fn a_clone_trail_leaves_frozen_copies_where_the_person_was() {
    let (mut s, c) = session();
    cut(&mut s, c);
    let n0 = s.history.undo.len();
    let r = s.execute("clip.cloneTrail", json!({"clip": c.0, "copies": 3})).unwrap();
    assert_eq!(r["copies"].as_array().unwrap().len(), 3, "{r}");
    let base = clip(&s, c);
    let copies: Vec<TrackItem> = above(&s).into_iter().flatten().collect();
    assert_eq!(copies.len(), 3);
    for k in &copies {
        assert!(k.frame_hold.is_some() && k.hold_filters, "frozen, its mask with it");
        assert_eq!(k.end(), base.end(), "to the clip's end");
        let m = &k.effect("opacity").unwrap().masks;
        assert_eq!((m.len(), m[0].mode), (1, MaskMode::Add));
        assert!(k.link.is_none());
    }
    // later copies start later and are further on
    let starts: Vec<_> = copies.iter().map(|k| k.start).collect();
    assert!(starts.windows(2).all(|w| w[0] < w[1]), "{starts:?}");
    assert_eq!(s.history.undo.len(), n0 + 1);
    // the frame renders with them
    s.execute("playhead.set", json!({"seconds": 1.9})).unwrap();
    assert!(s.render_program(0.5).is_some());
}

#[test]
fn shadow_clones_follow_the_person_and_the_person_is_in_front() {
    let (mut s, c) = session();
    cut(&mut s, c);
    s.execute("clip.shadowClone", json!({"clip": c.0, "copies": 2, "delay": 0.25})).unwrap();
    let base = clip(&s, c);
    let layers = above(&s);
    assert_eq!(layers.len(), 3, "two shadows and the person: {layers:?}");
    let first = &layers[0][0];
    let top = &layers[2][0];
    assert_eq!(first.start - base.start, filmcraft_time::Tick::from_seconds_f64(0.5), "the furthest behind lowest");
    assert_eq!((top.start, top.source_in), (base.start, base.source_in), "the person, in step, on top");
    let op = |k: &TrackItem| match k.effect("opacity").unwrap().param("opacity").unwrap().value {
        filmcraft_project::ParamValue::Float(v) => v,
        _ => 0.0,
    };
    assert!(op(first) < op(&layers[1][0]) && op(&layers[1][0]) < 100.0 && op(top) == 100.0);
    // same media later: the mask's keyframes are the clip's own (media time)
    assert_eq!(first.effect("opacity").unwrap().masks[0].path.keyframes.len(), person_mask(&base).unwrap().path.keyframes.len());
}

#[test]
fn a_glow_and_a_title_go_round_and_behind_the_person() {
    let (mut s, c) = session();
    cut(&mut s, c);
    s.execute("clip.personGlow", json!({"clip": c.0, "color": "#33ccff", "width": 10})).unwrap();
    let layers = above(&s);
    let glow = &layers[0][0];
    assert!(glow.effects.iter().any(|e| e.effect == "tint"));
    let blend = glow.effect("opacity").unwrap().param("blend").unwrap().value.clone();
    assert_eq!(blend, filmcraft_project::ParamValue::Choice(8), "Screen");
    s.execute("edit.undo", json!({})).unwrap();
    assert!(above(&s).is_empty());
    let n0 = s.history.undo.len();
    let r = s.execute("clip.textBehind", json!({"clip": c.0, "text": "ISTANBUL"})).unwrap();
    assert_eq!(s.history.undo.len(), n0 + 1, "one undo step");
    let seq = s.active_sequence().unwrap();
    let title = r["title"].as_u64().map(ClipId).unwrap();
    let person = r["copies"][0].as_u64().map(ClipId).unwrap();
    let track_of = |id: ClipId| seq.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == id)).unwrap();
    assert!(track_of(title) >= 1 && track_of(person) > track_of(title), "the title behind the person, over the clip");
}

#[test]
fn time_freezes_with_a_clean_plate_over_the_moving_person() {
    let (mut s, c) = session();
    cut(&mut s, c);
    let dir = tmp_dir("cutout-plate");
    let r = s.execute("clip.timeFreeze", json!({"clip": c.0, "at": 0.5, "plateFolder": dir.to_string_lossy()})).unwrap();
    let plate = r["plate"].as_str().unwrap();
    assert!(std::path::Path::new(plate).exists());
    let layers = above(&s);
    assert_eq!(layers.len(), 2, "{r}");
    let (p, frozen) = (&layers[0][0], &layers[1][0]);
    assert_ne!(p.item, clip(&s, c).item, "the plate is its own still");
    assert!(frozen.frame_hold.is_some() && frozen.hold_filters);
    assert_eq!(p.start, frozen.start);
    // the plate's mask follows the moving person from the still's own time
    let k0 = p.effect("opacity").unwrap().masks[0].path.keyframes.first().unwrap().time;
    assert!(k0.0 <= 0, "{k0:?}");
    assert!(r["plateCovered"].as_f64().unwrap() > 0.5, "{r}");
    s.execute("playhead.set", json!({"seconds": 1.5})).unwrap();
    assert!(s.render_program(0.5).is_some());
}

#[test]
fn people_are_chosen_and_outlines_turned_to_match() {
    let boxes = [[10.0, 10.0, 20.0, 40.0], [100.0, 20.0, 60.0, 120.0], [300.0, 50.0, 30.0, 60.0]];
    assert_eq!(choose(&boxes, 400, 200, Pick::Largest, None), Some(1));
    assert_eq!(choose(&boxes, 400, 200, Pick::Centre, None), Some(1));
    assert_eq!(choose(&boxes, 400, 200, Pick::At(0.79, 0.4), None), Some(2));
    assert_eq!(choose(&boxes, 400, 200, Pick::At(0.5, 0.95), None), None);
    // after the first frame: whoever is where they were
    assert_eq!(choose(&boxes, 400, 200, Pick::Largest, Some([12.0, 12.0, 20.0, 40.0])), Some(0));
    assert_eq!(choose(&boxes, 400, 200, Pick::Largest, Some([45.0, 10.0, 20.0, 40.0])), Some(0), "the nearest");
    assert_eq!(choose(&boxes, 400, 200, Pick::Largest, Some([380.0, 180.0, 5.0, 5.0])), None, "too far: lost");
    assert_eq!(choose(&[[f32::NAN, 0.0, 1.0, 1.0]], 10, 10, Pick::Largest, None), None);
    let prev = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let next = vec![(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)];
    assert_eq!(aligned(&prev, next), prev);
}

/// A person who stays where they are (the camera held by them).
struct Stayer;

impl PersonCutter for Stayer {
    fn cut(&self, _: &[u8], w: usize, h: usize, _: Pick, _: Option<[f32; 4]>, points: usize) -> std::result::Result<Option<Cut>, String> {
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, h as f32 * 0.3);
        let outline = (0..points).map(|i| i as f32 / points as f32 * std::f32::consts::TAU).map(|a| (cx + r * a.sin(), cy - r * a.cos())).collect();
        Ok(Some(Cut { bbox: [cx - r, cy - r, 2.0 * r, 2.0 * r], outline }))
    }
}

#[test]
fn a_person_who_never_moves_cannot_be_frozen() {
    let (mut s, c) = session();
    s.cutouts.cutter = Some(Arc::new(Stayer));
    cut(&mut s, c);
    let dir = tmp_dir("cutout-plate-still");
    let e = s.execute("clip.timeFreeze", json!({"clip": c.0, "plateFolder": dir.to_string_lossy()})).unwrap_err().to_string();
    assert!(e.contains("still camera"), "{e}");
    assert!(above(&s).is_empty(), "nothing was added");
    // the glow and the person in front need no moving person
    s.execute("clip.personInFront", json!({"clip": c.0})).unwrap();
    assert_eq!(above(&s).len(), 1);
}
