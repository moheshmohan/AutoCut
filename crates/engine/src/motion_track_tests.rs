//! Motion tracks on synthetic clips (generated here, no media committed): a textured square on
//! noise that moves, grows and turns, as an image sequence.

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{ClipId, ItemId, ParamValue, SequenceSettings, TrackKind};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;

const W: u32 = 320;
const H: u32 = 180;
const FRAMES: u32 = 48;
/// Half the square's side at scale 1, pixels.
const HALF: f64 = 18.0;

fn hash(x: i64, y: i64, salt: u64) -> f64 {
    let mut v = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    (v >> 40) as f64 / (1u64 << 24) as f64
}

/// The square in frame `k`: centre (pixels), scale, rotation (degrees).
fn truth(k: u32) -> (Vec2, f64, f64) {
    let u = f64::from(k);
    (Vec2::new(70.0 + 3.6 * u, 90.0 + 12.0 * (u * 0.1).sin()), 1.0 + 0.005 * u, 0.6 * u)
}

/// The square's box in frame `k` as `[x, y, w, h]` shares of the picture (its unrotated size).
fn truth_box(k: u32) -> [f64; 4] {
    let (c, s, _) = truth(k);
    let half = HALF * s;
    [(c.x - half) / f64::from(W), (c.y - half) / f64::from(H), 2.0 * half / f64::from(W), 2.0 * half / f64::from(H)]
}

fn frame(k: u32) -> image::RgbaImage {
    let (c, s, a) = truth(k);
    let inv = Affine::translate(c.x, c.y).then_apply(&Affine::rotate_deg(a)).then_apply(&Affine::scale(s, s)).inverse().unwrap();
    image::RgbaImage::from_fn(W, H, |x, y| {
        let mut acc = [0.0f64; 3];
        for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
            let p = Vec2::new(f64::from(x) + ox, f64::from(y) + oy);
            let o = inv.apply(p);
            let c = if o.x.abs() < HALF && o.y.abs() < HALF {
                let edge = o.x.abs() > HALF - 2.5 || o.y.abs() > HALF - 2.5;
                let b = hash((o.x / 4.0).floor() as i64, (o.y / 4.0).floor() as i64, 0xA5);
                let v = if edge { 0.05 } else { 0.2 + 0.75 * b };
                [v, 0.6 * v + 0.1, 0.2 * v]
            } else {
                let n = 0.3 + 0.35 * hash((p.x / 2.0).floor() as i64, (p.y / 2.0).floor() as i64, 0x51);
                [0.4 * n, 0.6 * n, n]
            };
            for (a, v) in acc.iter_mut().zip(c) {
                *a += v / 4.0;
            }
        }
        image::Rgba([(acc[0] * 255.0) as u8, (acc[1] * 255.0) as u8, (acc[2] * 255.0) as u8, 255])
    })
}

/// A session with the moving square as an image-sequence clip on V1 of a 320×180 sequence.
fn session() -> (Session, ItemId, ClipId, FrameRate) {
    let dir = tmp_dir("motion-track");
    for k in 0..FRAMES {
        frame(k).save(dir.join(format!("shot{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let first = dir.join("shot0000.png");
    let r = s.execute("file.importImageSequence", json!({"path": first.to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}")));
    let rate = s.project.item(item).unwrap().as_media().unwrap().info.video.clone().unwrap().frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Robots", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 3, 1, None);
    let d = p.item(item).unwrap().duration();
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
    }
    let clip = v.id;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    (s, seq, clip, rate)
}

fn secs(rate: FrameRate, k: i64) -> f64 {
    rate.tick_of(k).seconds()
}

fn create(s: &mut Session, clip: ClipId, at: f64, b: [f64; 4], method: &str) -> u64 {
    let r = s.execute("track.create", json!({"clip": clip.0, "box": b, "at": at, "method": method, "wait": true})).unwrap();
    r["track"]["id"].as_u64().unwrap_or_else(|| panic!("{r}"))
}

/// Mean and worst centre error (pixels) of a track's samples against the truth, and the worst
/// rotation and scale errors.
fn accuracy(s: &mut Session, id: u64, rate: FrameRate, seed: u32) -> (f64, f64, f64, f64) {
    let v = s.execute("track.inspect", json!({"track": id})).unwrap();
    let samples = v["samples"].as_array().unwrap();
    assert_eq!(samples.len(), FRAMES as usize, "every frame tracked: {}", samples.len());
    let (s0, a0) = (truth(seed).1, truth(seed).2);
    let (mut sum, mut worst, mut rot, mut scale) = (0.0, 0.0f64, 0.0f64, 0.0f64);
    for x in samples {
        let k = (x["seconds"].as_f64().unwrap() * rate.as_f64()).round() as u32;
        let b: Vec<f64> = x["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let c = Vec2::new((b[0] + b[2] / 2.0) * f64::from(W), (b[1] + b[3] / 2.0) * f64::from(H));
        let (tc, ts, ta) = truth(k);
        let e = (c - tc).length();
        sum += e;
        worst = worst.max(e);
        assert_eq!(x["lost"], json!(false), "lost at frame {k}");
        rot = rot.max((x["rotation"].as_f64().unwrap() - (ta - a0)).abs());
        scale = scale.max((x["scale"].as_f64().unwrap() / (ts / s0) - 1.0).abs());
    }
    (sum / samples.len() as f64, worst, rot, scale)
}

#[test]
fn tracks_follow_the_object_both_ways_and_undo() {
    let (mut s, _, clip, rate) = session();
    let undo = s.history.undo.len();
    let id = create(&mut s, clip, secs(rate, 20), truth_box(20), "points");
    assert_eq!(s.history.undo.len(), undo + 1);
    assert_eq!(s.history.undo.last().unwrap().0, "Track Object");
    let (mean, worst, rot, scale) = accuracy(&mut s, id, rate, 20);
    eprintln!("points: mean {mean:.2} px, worst {worst:.2} px, rotation within {rot:.2}°, scale within {:.1}%", scale * 100.0);
    assert!(mean < 1.0 && worst < 2.0, "points: mean {mean} worst {worst}");
    assert!(rot < 2.0 && scale < 0.04, "points: rotation {rot} scale {scale}");
    let id2 = create(&mut s, clip, secs(rate, 0), truth_box(0), "subject");
    let v = s.execute("track.inspect", json!({"track": id2})).unwrap();
    let mut worst = 0.0f64;
    let mut sum = 0.0;
    for x in v["samples"].as_array().unwrap() {
        let k = (x["seconds"].as_f64().unwrap() * rate.as_f64()).round() as u32;
        let b: Vec<f64> = x["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let e = (Vec2::new((b[0] + b[2] / 2.0) * f64::from(W), (b[1] + b[3] / 2.0) * f64::from(H)) - truth(k).0).length();
        worst = worst.max(e);
        sum += e;
    }
    eprintln!("subject: mean {:.2} px, worst {worst:.2} px", sum / f64::from(FRAMES));
    assert!(worst < 3.5, "subject: worst {worst}");
    let list = s.execute("track.list", json!({})).unwrap();
    assert_eq!(list["tracks"].as_array().unwrap().len(), 2);
    assert_eq!(list["tracks"][0]["seen"], json!(1.0));
    // the sample's frame position is where the object shows in the sequence (no Motion: the same)
    let x = &v["samples"][10];
    let f: Vec<f64> = x["frame"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((Vec2::new(f[0], f[1]) - truth(10).0).length() < 3.0, "{x}");
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.motion_tracks.is_empty(), "undone");
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(s.project.motion_tracks.len(), 1);
    s.execute("track.delete", json!({"track": id})).unwrap();
    assert!(s.project.motion_tracks.is_empty());
}

/// The tracked clip at 200 % with its head trimmed and a Motion transform; a text clip attached
/// to the track sits on the object, frame by frame, in sequence time.
#[test]
fn attached_text_lands_on_the_object_with_speed_trim_and_motion() {
    let (mut s, seq, clip, rate) = session();
    let fd = rate.frame_duration();
    s.edit("test: speed", |pr, _| {
        let q = pr.sequence_mut(seq).unwrap();
        let (_, it) = q.find_item_mut(clip).unwrap();
        it.speed = 2.0;
        it.source_in = Tick(fd.0 * 4);
        it.duration = Tick(fd.0 * 21);
        let m = it.effect_mut("motion").unwrap();
        m.param_mut("scale").unwrap().value = ParamValue::Float(150.0);
        m.param_mut("position").unwrap().value = ParamValue::Vec2(Vec2::new(150.0, 70.0));
        Ok(())
    })
    .unwrap();
    // drawn at sequence frame 5 = media frame 4 + 2·5 = 14
    let id = create(&mut s, clip, secs(rate, 5), truth_box(14), "points");
    let txt = s.execute("graphics.newText", json!({"text": "R-01", "time": 0, "seconds": 2.0, "size": 18})).unwrap();
    let label = ClipId(txt["clip"].as_u64().unwrap());
    let r = s.execute("track.attach", json!({"track": id, "target": {"clip": label.0}, "offset": "center"})).unwrap();
    assert!(r["keyframes"].as_u64().unwrap() >= 2, "{r}");
    let q = s.active_sequence().unwrap().clone();
    let (_, c) = q.find_item(clip).unwrap();
    let (_, l) = q.find_item(label).unwrap();
    let mut worst = 0.0f64;
    for k in 0..21 {
        let t = Tick(fd.0 * k);
        let m = filmcraft_render::motion_matrix(&q, c, (W, H), None, c.effect_time_at(t));
        let media_frame = (4 + 2 * k) as u32;
        let want = m.apply(truth(media_frame).0);
        let got = l.effect("motion").unwrap().vec2_at("position", l.effect_time_at(t));
        worst = worst.max((got - want).length());
    }
    eprintln!("attached text at 200 %, trimmed, scaled 150 %: worst {worst:.2} px");
    assert!(worst < 3.0, "worst {worst}");
    // an offset kept: the text stays where it was relative to the object
    let undo = s.history.undo.len();
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.history.undo.len(), undo - 1);
    let before = s.active_sequence().unwrap().find_item(label).unwrap().1.effect("motion").unwrap().vec2_at("position", Tick::ZERO);
    s.set_playhead(Tick::ZERO);
    s.execute("track.attach", json!({"track": id, "target": {"clip": label.0}, "offset": "keep"})).unwrap();
    let after = s.active_sequence().unwrap().find_item(label).unwrap().1.effect("motion").unwrap().vec2_at("position", Tick::ZERO);
    assert!((after - before).length() < 0.6, "kept: {before:?} → {after:?}");
    let ins = s.execute("track.inspect", json!({"track": id, "samples": false})).unwrap();
    assert_eq!(ins["attachments"].as_array().unwrap().len(), 1, "{ins}");
}

#[test]
fn refine_reruns_from_the_corrected_frame_and_rebakes() {
    let (mut s, _, clip, rate) = session();
    // seeded a few pixels off: the track follows the wrong spot
    let mut off = truth_box(5);
    off[0] += 6.0 / f64::from(W);
    let id = create(&mut s, clip, secs(rate, 5), off, "points");
    let txt = s.execute("graphics.newText", json!({"text": "A", "time": 0, "seconds": 2.0})).unwrap();
    let label = ClipId(txt["clip"].as_u64().unwrap());
    s.execute("track.attach", json!({"track": id, "target": {"clip": label.0}, "offset": "center"})).unwrap();
    let centre = |s: &mut Session, k: u32| -> f64 {
        let v = s.execute("track.inspect", json!({"track": id})).unwrap();
        let x = v["samples"].as_array().unwrap().iter().find(|x| ((x["seconds"].as_f64().unwrap() * rate.as_f64()).round() as u32) == k).unwrap().clone();
        let b: Vec<f64> = x["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        (b[0] + b[2] / 2.0) * f64::from(W) - truth(k).0.x
    };
    let pos_x = |s: &Session, k: i64| s.active_sequence().unwrap().find_item(label).unwrap().1.effect("motion").unwrap().vec2_at("position", rate.tick_of(k)).x;
    let early = centre(&mut s, 10);
    assert!(early > 4.0, "off by the seed's error: {early}");
    let late_label = pos_x(&s, 40) - truth(40).0.x;
    assert!(late_label > 3.0, "{late_label}");
    // corrected at frame 30: from there on it follows the object; before it, as it was
    let r = s.execute("track.refine", json!({"track": id, "at": secs(rate, 30), "box": truth_box(30), "wait": true})).unwrap();
    assert_eq!(s.history.undo.last().unwrap().0, "Refine Track", "{r}");
    assert!((centre(&mut s, 10) - early).abs() < 1e-9, "the samples before the correction stay");
    for k in [30, 36, 47] {
        let e = centre(&mut s, k);
        assert!(e.abs() < 2.0, "frame {k} after refining: {e}");
    }
    // the attachment was re-written from the new samples
    let after = pos_x(&s, 40) - truth(40).0.x;
    assert!(after.abs() < 2.0, "rebaked: {after}");
    // undo brings both back
    s.execute("edit.undo", json!({})).unwrap();
    assert!((pos_x(&s, 40) - truth(40).0.x - late_label).abs() < 1e-6);
}

#[test]
fn convenience_commands_follow_the_object() {
    let (mut s, seq, clip, rate) = session();
    // blur: a masked Gaussian Blur on the clip whose mask is on the object
    let r = s.execute("track.blurObject", json!({"clip": clip.0, "box": truth_box(0), "at": 0.0, "method": "points", "wait": true})).unwrap();
    assert_eq!(s.history.undo.last().unwrap().0, "Blur Object", "{r}");
    let q = s.project.sequence(seq).unwrap().clone();
    let (_, it) = q.find_item(clip).unwrap();
    let blur = it.effects.iter().find(|e| e.effect == "gaussian_blur").expect("a blur");
    let m = &blur.masks[0];
    assert!(m.path.keyframes.len() >= 2);
    for k in [0u32, 20, 40] {
        let c = m.path_at(it.effect_time_at(rate.tick_of(i64::from(k)))).centroid();
        assert!((c - truth(k).0).length() < 2.5, "blur mask at {k}: {c:?} vs {:?}", truth(k).0);
    }
    // mosaic, spotlight
    s.execute("track.blurObject", json!({"clip": clip.0, "box": truth_box(0), "at": 0.0, "style": "mosaic", "shape": "rectangle", "wait": true})).unwrap();
    s.execute("track.spotlight", json!({"clip": clip.0, "box": truth_box(0), "at": 0.0, "dim": 70, "wait": true})).unwrap();
    let q = s.project.sequence(seq).unwrap().clone();
    let (_, it) = q.find_item(clip).unwrap();
    assert!(it.effects.iter().any(|e| e.effect == "mosaic" && e.masks.len() == 1));
    let lv = it.effects.iter().find(|e| e.effect == "levels").expect("spotlight");
    assert!(lv.masks[0].inverted && lv.f64_at("out_white", Tick::ZERO) < 100.0);
    // a label on V2 whose Motion keeps its centre on the object
    let r = s.execute("track.label", json!({"clip": clip.0, "box": truth_box(0), "at": 0.0, "text": "Spot · 32.5 kg", "wait": true})).unwrap();
    assert_eq!(s.history.undo.last().unwrap().0, "Label Object", "{r}");
    let lid = ClipId(r["track"]["made"]["clip"].as_u64().unwrap_or_else(|| panic!("{r}")));
    let q = s.project.sequence(seq).unwrap().clone();
    let (tid, l) = q.find_item(lid).unwrap();
    assert_ne!(tid, q.video_tracks[0].id, "above the tracked clip");
    assert_eq!(l.graphic_layers().count(), 3, "leader, dot and text");
    for k in [0i64, 24, 47] {
        let p = l.effect("motion").unwrap().vec2_at("position", l.effect_time_at(rate.tick_of(k)));
        assert!((p - truth(k as u32).0).length() < 3.0, "label at {k}: {p:?}");
    }
    // the tracks know what follows them
    let list = s.execute("track.list", json!({})).unwrap();
    assert!(list["tracks"].as_array().unwrap().iter().all(|t| t["attachments"].as_array().unwrap().len() == 1), "{list}");
}

#[test]
fn hostile_parameters_are_errors() {
    let (mut s, _, clip, rate) = session();
    let c = clip.0;
    let bad: Vec<Value> = vec![
        json!({"clip": c}),
        json!({"clip": c, "box": [2.0, 2.0, 0.1, 0.1]}),
        json!({"clip": c, "box": [0.1, 0.1, 0.0, 0.1]}),
        json!({"clip": c, "box": [0.1, 0.1, -0.2, 0.1]}),
        json!({"clip": c, "box": [0.1, "x", 0.1, 0.1]}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1]}),
        json!({"clip": c, "box": {"x": 0.1, "y": null, "w": 0.1, "h": 0.1}}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "at": -5.0}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "at": 1e300}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "direction": "sideways"}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "method": 7}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "space": "moon"}),
        json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "name": "a\nb"}),
        json!({"clip": 999_999, "box": [0.1, 0.1, 0.1, 0.1]}),
        json!({"clip": -1, "box": [0.1, 0.1, 0.1, 0.1]}),
        json!([1, 2]),
    ];
    for p in bad {
        assert!(s.execute("track.create", p.clone()).is_err(), "accepted {p}");
    }
    // a clip with no picture: a graphic
    let txt = s.execute("graphics.newText", json!({"text": "x", "time": 0})).unwrap();
    assert!(s.execute("track.create", json!({"clip": txt["clip"], "box": [0.1, 0.1, 0.2, 0.2], "wait": true})).is_err());
    for (cmd, p) in [
        ("track.inspect", json!({"track": 12345})),
        ("track.delete", json!({})),
        ("track.refine", json!({"track": 12345, "box": [0.1, 0.1, 0.1, 0.1]})),
        ("track.attach", json!({"track": 12345, "target": {"clip": c}})),
        ("track.rebake", json!({"track": "x"})),
        ("track.label", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "text": ""})),
        ("track.label", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "text": "x".repeat(5000)})),
        ("track.label", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "text": "a", "accent": "#zz0000"})),
        ("track.blurObject", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "amount": 500})),
        ("track.blurObject", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "style": "fire"})),
        ("track.spotlight", json!({"clip": c, "box": [0.1, 0.1, 0.1, 0.1], "dim": f64::MAX})),
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} accepted {p}");
    }
    // a real track and hostile attachments
    let id = create(&mut s, clip, secs(rate, 0), truth_box(0), "subject");
    for target in [
        json!({"clip": 999_999}),
        json!({"clip": c, "effect": "nope", "param": "center"}),
        json!({"clip": c, "effect": "motion", "param": "scale"}),
        json!({"clip": c, "mask": 9}),
        json!({"clip": c, "layer": 3}),
        json!("x"),
    ] {
        assert!(s.execute("track.attach", json!({"track": id, "target": target})).is_err(), "attached to {target}");
    }
    for p in [
        json!({"track": id, "target": {"clip": c}, "follow": "teleport"}),
        json!({"track": id, "target": {"clip": c}, "offset": [1e300, 0]}),
        json!({"track": id, "target": {"clip": c}, "offset": "left"}),
    ] {
        assert!(s.execute("track.attach", p.clone()).is_err(), "{p}");
    }
    assert!(s.execute("track.detach", json!({"track": id, "attachment": 77})).is_err());
    // a box partly outside the picture is cut to it
    let r = s.execute("track.create", json!({"clip": c, "box": [-0.05, 0.4, 0.2, 0.2], "at": 0.0, "direction": "forward", "wait": true})).unwrap();
    assert!(r["track"]["seed"]["box"][0].as_f64().unwrap() >= 0.0, "{r}");
}

/// A clip whose frames do not decode (its files gone) tracks to a track of lost samples, not a
/// crash.
#[test]
fn a_clip_without_frames_is_lost_not_a_crash() {
    let dir = tmp_dir("motion-track-gone");
    for k in 0..6 {
        frame(k).save(dir.join(format!("shot{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let r = s.execute("file.importImageSequence", json!({"path": dir.join("shot0000.png").to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap());
    let rate = s.project.item(item).unwrap().as_media().unwrap().info.video.clone().unwrap().frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Gone", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let d = p.item(item).unwrap().duration();
    let v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    let clip = v.id;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    std::fs::remove_dir_all(&dir).unwrap();
    // the frames may still come from the media cache, or be lost: either way an answer
    match s.execute("track.create", json!({"clip": clip.0, "box": [0.2, 0.3, 0.2, 0.3], "at": 0.0, "wait": true})) {
        Ok(r) => assert!(r["track"]["frames"].as_u64().unwrap() >= 1, "{r}"),
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}

#[test]
fn tracks_survive_save_and_open() {
    let (mut s, _, clip, rate) = session();
    let id = create(&mut s, clip, secs(rate, 0), truth_box(0), "points");
    let txt = s.execute("graphics.newText", json!({"text": "Tag", "time": 0, "seconds": 2.0})).unwrap();
    s.execute("track.attach", json!({"track": id, "target": {"clip": txt["clip"], "layer": 0}, "follow": "positionScaleRotation"})).unwrap();
    let dir = tmp_dir("motion-track-save");
    let path = dir.join("tracks.fcproj").to_string_lossy().to_string();
    s.execute("file.saveAs", json!({"path": path})).unwrap();
    let mut t = Session::default();
    t.execute("file.open", json!({"path": path})).unwrap();
    assert_eq!(t.project.motion_tracks, s.project.motion_tracks);
    let a = &t.project.motion_tracks.values().next().unwrap().attachments[0];
    assert_eq!(a.target.kind_name(), "graphicLayer");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn camera_track_keeps_the_box_framed() {
    let (mut s, seq, clip, rate) = session();
    let r = s
        .execute("clip.cameraTrack", json!({"clip": clip.0, "target": "box", "box": truth_box(0), "at": 0.0, "zoom": "medium", "smooth": 0.3, "wait": true}))
        .unwrap();
    assert_eq!(s.history.undo.last().unwrap().0, "Camera Track", "{r}");
    let q = s.project.sequence(seq).unwrap().clone();
    let (_, it) = q.find_item(clip).unwrap();
    let m = it.effect("motion").unwrap();
    assert!(m.param("position").unwrap().keyframes.len() >= 2, "the crop follows");
    for k in [5i64, 24, 44] {
        let mt = it.effect_time_at(rate.tick_of(k));
        let sc = m.f64_at("scale", mt);
        assert!((sc - 130.0).abs() < 8.0, "a medium push-in: {sc}");
        let mm = filmcraft_render::motion_matrix(&q, it, (W, H), None, mt);
        let x = mm.apply(truth(k as u32).0).x;
        assert!((x - f64::from(W) / 2.0).abs() < 70.0, "frame {k}: the object at x {x}");
    }
    for bad in [
        json!({"clip": clip.0, "target": "moon"}),
        json!({"clip": clip.0, "target": "box"}),
        json!({"clip": clip.0, "target": "subject", "zoom": "huge"}),
        json!({"clip": clip.0, "smooth": "very"}),
    ] {
        assert!(s.execute("clip.cameraTrack", bad.clone()).is_err(), "{bad}");
    }
}
