//! Synthetic moving objects (generated here, no media): a textured square on noise that moves,
//! grows and turns, sometimes behind a static occluder; the tracker's box is compared with the
//! truth frame by frame.

use super::*;

const W: usize = 320;
const H: usize = 240;
/// Half the object's side at scale 1, pixels.
const HALF: f64 = 24.0;

fn hash(x: i64, y: i64, salt: u64) -> f64 {
    // splitmix64 of the cell and the salt
    let mut v = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    (v >> 40) as f64 / (1u64 << 24) as f64
}

/// The object's texture at object coordinates (pixels from its centre at scale 1): blocks and
/// waves, with a dark frame line so it stands out of the noise.
fn object_texture(u: f64, v: f64) -> f64 {
    let blocks = hash((u / 5.0).floor() as i64, (v / 5.0).floor() as i64, 0xA5A5);
    let waves = 0.5 + 0.5 * ((u * 0.35).sin() * (v * 0.29).cos());
    let edge = if u.abs() > HALF - 3.0 || v.abs() > HALF - 3.0 { 0.0 } else { 1.0 };
    (0.15 + 0.55 * blocks + 0.3 * waves) * edge
}

/// The static background: fine noise.
fn background(x: f64, y: f64) -> f64 {
    0.35 + 0.3 * hash((x / 2.0).floor() as i64, (y / 2.0).floor() as i64, 0x1234)
}

/// The occluder's texture (a different noise).
fn occluder(x: f64, y: f64) -> f64 {
    0.5 + 0.4 * hash((x / 3.0).floor() as i64, (y / 3.0).floor() as i64, 0x7777)
}

#[derive(Clone, Copy)]
struct Truth {
    c: Vec2,
    scale: f64,
    angle: f64,
}

/// One RGBA frame with the object at `pose` and, when given, an occluder rectangle on top.
fn render(pose: Truth, occ: Option<[f64; 4]>) -> Vec<u8> {
    let inv =
        Affine::translate(pose.c.x, pose.c.y).then_apply(&Affine::rotate_deg(pose.angle)).then_apply(&Affine::scale(pose.scale, pose.scale)).inverse().unwrap();
    let mut out = vec![0u8; W * H * 4];
    for y in 0..H {
        for x in 0..W {
            let mut acc = 0.0;
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let p = Vec2::new(x as f64 + ox - 0.5, y as f64 + oy - 0.5);
                let o = inv.apply(p);
                let hidden = occ.is_some_and(|r| p.x >= r[0] && p.x < r[2] && p.y >= r[1] && p.y < r[3]);
                acc += if hidden {
                    occluder(p.x, p.y)
                } else if o.x.abs() < HALF && o.y.abs() < HALF {
                    object_texture(o.x, o.y)
                } else {
                    background(p.x, p.y)
                };
            }
            let v = (acc / 4.0 * 255.0).round().clamp(0.0, 255.0) as u8;
            let i = (y * W + x) * 4;
            out[i..i + 4].copy_from_slice(&[v, v.saturating_add(8), v.saturating_sub(8), 255]);
        }
    }
    out
}

fn seed_of(p: Truth) -> SubjectBox {
    let s = 2.0 * HALF * p.scale;
    SubjectBox::new(p.c.x / W as f64, p.c.y / H as f64, s / W as f64, s / H as f64).unwrap()
}

fn centre_error(o: &ObjectObservation, p: Truth) -> f64 {
    (Vec2::new(o.bbox.cx * W as f64, o.bbox.cy * H as f64) - p.c).length()
}

/// Track `frames` poses (in order) from the first; the observations.
fn run(method: TrackingMethod, poses: &[Truth], occ: Option<[f64; 4]>) -> Vec<ObjectObservation> {
    let mut tr = ObjectTracker::new(method, seed_of(poses[0]));
    poses.iter().enumerate().map(|(k, p)| tr.push(&render(*p, occ), W, H, k as f64 / 30.0)).collect()
}

fn moving(n: usize, rot: f64, grow: f64) -> Vec<Truth> {
    (0..n)
        .map(|k| {
            let u = k as f64;
            Truth { c: Vec2::new(80.0 + 2.6 * u, 110.0 + 18.0 * (u * 0.08).sin()), scale: 1.0 + grow * u, angle: rot * u }
        })
        .collect()
}

#[test]
fn points_follow_a_moving_growing_turning_square() {
    let poses = moving(50, 0.9, 0.006);
    let obs = run(TrackingMethod::Points, &poses, None);
    let mut worst = 0.0f64;
    let mut sum = 0.0;
    for (o, p) in obs.iter().zip(&poses) {
        assert!(!o.lost, "lost at t={}", o.t);
        let e = centre_error(o, *p);
        worst = worst.max(e);
        sum += e;
        assert!((o.bbox.rotation - p.angle).abs() < 3.0, "rotation {} vs {} at t={}", o.bbox.rotation, p.angle, o.t);
        assert!((o.bbox.scale / p.scale - 1.0).abs() < 0.05, "scale {} vs {} at t={}", o.bbox.scale, p.scale, o.t);
    }
    let mean = sum / obs.len() as f64;
    eprintln!("points: mean centre error {mean:.2} px, worst {worst:.2} px");
    assert!(worst < 3.0 && mean < 1.5, "mean {mean} worst {worst}");
}

#[test]
fn subject_follows_a_moving_growing_square() {
    let poses = moving(50, 0.0, 0.006);
    let obs = run(TrackingMethod::Subject, &poses, None);
    let mut worst = 0.0f64;
    let mut sum = 0.0;
    for (o, p) in obs.iter().zip(&poses) {
        let e = centre_error(o, *p);
        worst = worst.max(e);
        sum += e;
        assert!(!o.lost, "lost at t={}", o.t);
    }
    let mean = sum / obs.len() as f64;
    let last = obs.last().unwrap();
    eprintln!("subject: mean centre error {mean:.2} px, worst {worst:.2} px, last scale {:.3}", last.bbox.scale);
    assert!(worst < 4.0 && mean < 2.0, "mean {mean} worst {worst}");
    assert!((last.bbox.scale / poses.last().unwrap().scale - 1.0).abs() < 0.1);
}

#[test]
fn backward_tracking_from_the_last_frame() {
    let mut poses = moving(40, 0.6, 0.004);
    poses.reverse();
    let obs = run(TrackingMethod::Points, &poses, None);
    let worst = obs.iter().zip(&poses).map(|(o, p)| centre_error(o, *p)).fold(0.0, f64::max);
    // rotation and scale are relative to the seed (the last frame)
    let first = poses[0];
    for (o, p) in obs.iter().zip(&poses) {
        assert!((o.bbox.rotation - (p.angle - first.angle)).abs() < 3.0);
    }
    assert!(worst < 3.0, "worst {worst}");
}

/// The object passes behind a static occluder, wholly hidden for a few frames: the frames under
/// it are lost, and it is found again soon after it comes out.
fn occlusion(method: TrackingMethod) {
    let occ = [135.0, 50.0, 225.0, 190.0];
    let poses: Vec<Truth> = (0..56)
        .map(|k| Truth {
            c: Vec2::new(70.0 + 4.0 * k as f64, 120.0 + 0.4 * k as f64),
            scale: 1.0,
            angle: if method == TrackingMethod::Points { 0.3 * k as f64 } else { 0.0 },
        })
        .collect();
    let obs = run(method, &poses, Some(occ));
    // wholly hidden: the object's box inside the occluder
    let hidden = |p: &Truth| p.c.x - HALF * 1.3 >= occ[0] && p.c.x + HALF * 1.3 <= occ[2];
    let clear = |p: &Truth| p.c.x + HALF * 1.5 < occ[0] || p.c.x - HALF * 1.5 > occ[2];
    assert!(poses.iter().any(hidden), "the test hides the object");
    for (o, p) in obs.iter().zip(&poses) {
        if hidden(p) {
            assert!(o.lost || o.confidence < 0.5, "{method:?}: seen while hidden at t={} ({o:?})", o.t);
        }
    }
    // out again: re-acquired within a few frames, on the object
    let out = poses.iter().position(|p| p.c.x - HALF * 1.5 > occ[2]).unwrap();
    let back = (out..poses.len()).find(|&k| !obs[k].lost).unwrap_or_else(|| panic!("{method:?}: never re-acquired"));
    assert!(back <= out + 4, "{method:?}: re-acquired {} frames after coming out", back - out);
    let mut worst = 0.0f64;
    for k in back..poses.len() {
        worst = worst.max(centre_error(&obs[k], poses[k]));
        assert!(!obs[k].lost, "{method:?}: lost again at frame {k}");
    }
    let before = obs.iter().zip(&poses).filter(|(_, p)| clear(p) && p.c.x < occ[0]).map(|(o, p)| centre_error(o, *p)).fold(0.0, f64::max);
    eprintln!("{method:?}: before the occluder worst {before:.2} px; re-acquired {} frame(s) after coming out, then worst {worst:.2} px", back - out);
    assert!(before < 3.0 && worst < 4.0, "{method:?}: before {before} after {worst}");
}

#[test]
fn points_reacquire_after_an_occlusion() {
    occlusion(TrackingMethod::Points);
}

#[test]
fn subject_reacquires_after_an_occlusion() {
    occlusion(TrackingMethod::Subject);
}

#[test]
fn hostile_frames_never_panic() {
    for method in TrackingMethod::ALL {
        let seed = SubjectBox::new(0.5, 0.5, 0.2, 0.2).unwrap();
        let mut tr = ObjectTracker::new(method, seed);
        let frame = render(Truth { c: Vec2::new(160.0, 120.0), scale: 1.0, angle: 0.0 }, None);
        for (buf, w, h) in [(&frame[..], 0usize, 0usize), (&frame[..10], W, H), (&frame[..], usize::MAX, 2), (&frame[..], 8, 8), (&frame[..], W, H)] {
            let o = tr.push(buf, w, h, f64::NAN);
            assert!(o.bbox.cx.is_finite() && o.bbox.cy.is_finite() && o.confidence.is_finite(), "{method:?}: {o:?}");
        }
        // a frame of another size keeps the normalised pose
        let small: Vec<u8> = frame.chunks(4).step_by(2).flatten().copied().collect();
        let o = tr.push(&small, W / 2, H, 1.0);
        assert!(o.bbox.cx.is_finite());
    }
}
