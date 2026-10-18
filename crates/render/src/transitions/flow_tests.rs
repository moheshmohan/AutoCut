//! Roadmap C3: the flow engine and the Content-Aware Wipe on synthetic content (moving texture,
//! salient objects), the content field and its cache. The generic contract, golden, parameter
//! and fuzz tests in `tests.rs` and `shader/tests.rs` cover them too; GPU parity is in
//! `filmcraft-gpu`.

use std::sync::Arc;

use filmcraft_project::{EffectInstance, ParamValue};

use super::content::{self, Field};
use super::*;

const W: usize = 128;
const H: usize = 72;

fn instance(id: &str) -> EffectInstance {
    filmcraft_project::find_effect(id).unwrap().instance()
}

fn set(e: &mut EffectInstance, id: &str, v: f64) {
    e.param_mut(id).unwrap().value = ParamValue::Float(v);
}

/// A smooth texture with its content offset by (dx, dy).
fn texture(dx: f32, dy: f32) -> Image {
    paint(W, H, |x, y| {
        let (u, v) = (x - dx, y - dy);
        let l = 0.35 + 0.15 * (u * 0.21).sin() * (v * 0.17).cos() + 0.1 * ((u + v) * 0.11).sin() + 0.05 * (u * 0.53 - v * 0.31).cos();
        [l, l * 0.9, l * 0.8, 1.0]
    })
}

/// Mean absolute RGB difference over pixels at least `m` from every edge.
fn interior(a: &Image, b: &Image, m: usize) -> f32 {
    let (mut s, mut n) = (0.0f32, 0usize);
    for y in m..a.h - m {
        for x in m..a.w - m {
            let (p, q) = (a.get(x, y), b.get(x, y));
            s += (0..3).map(|k| (p[k] - q[k]).abs()).sum::<f32>();
            n += 3;
        }
    }
    s / n as f32
}

/// The largest jump between neighbouring pixels (RGB), away from the edges.
fn max_step(img: &Image, m: usize) -> f32 {
    let mut worst = 0.0f32;
    for y in m..img.h - m {
        for x in m..img.w - m {
            let c = img.get(x, y);
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                let d = img.get(nx, ny);
                worst = worst.max((0..3).map(|k| (c[k] - d[k]).abs()).fold(0.0, f32::max));
            }
        }
    }
    worst
}

#[test]
fn flow_morph_moves_content_instead_of_double_exposing() {
    // the content moves 10 px right and 4 px up between the clips: half way, the morph shows it
    // once, half way along; a cross dissolve shows two copies
    let (a, b) = (texture(0.0, 0.0), texture(10.0, -4.0));
    let mut e = instance("flow_morph");
    set(&mut e, "smoothness", 0.0);
    for (p, at) in [(0.5f32, (5.0f32, -2.0f32)), (0.3, (3.0, -1.2)), (0.7, (7.0, -2.8))] {
        let want = texture(at.0, at.1);
        let morph = apply(&e, &a, &b, p);
        let dissolve = a.clone().lerp(&b, p);
        let (em, ed) = (interior(&morph, &want, 14), interior(&dissolve, &want, 14));
        assert!(em * 3.0 < ed, "p={p}: morph {em} vs dissolve {ed}");
        assert!(em < 0.02, "p={p}: {em}");
    }
    // no strength: a dissolve across the middle (no warp)
    let mut flat = instance("flow_morph");
    set(&mut flat, "strength", 0.0);
    set(&mut flat, "width", 100.0);
    let m = apply(&flat, &a, &b, 0.5);
    let want = a.clone().lerp(&b, 0.5);
    assert!(interior(&m, &want, 0) < 1e-5);
}

#[test]
fn flow_transitions_never_tear() {
    // moving texture, and two unrelated shots (the flow is then guesswork): every flow style
    // stays as smooth as its inputs, no seams or rips
    let shots = [(texture(0.0, 0.0), texture(9.0, 5.0)), (texture(0.0, 0.0), paint(W, H, |x, y| [0.2 + 0.6 * (x / W as f32), 0.3, 0.6 * (y / H as f32), 1.0]))];
    for (a, b) in &shots {
        let limit = max_step(a, 2).max(max_step(b, 2));
        for p in filmcraft_project::flow_transitions::FLOW_PRESETS.iter().filter(|p| matches!(p.id, "flow_morph" | "flow_doppler_warp" | "flow_hype_ramp")) {
            let e = instance(p.id);
            for t in [0.2f32, 0.4, 0.5, 0.6, 0.8] {
                let m = apply(&e, a, b, t);
                // (Hype Ramp's brightness pulse scales every step with the picture)
                let d = (t - 0.5) / 0.12;
                let gain = if p.id == "flow_hype_ramp" { 1.0 + 0.8 * (-d * d).exp() } else { 1.0 };
                let step = max_step(&m, 2) / gain;
                let mut at = (0, 0, 0.0f32);
                for y in 2..H - 2 {
                    for x in 2..W - 2 {
                        let c = m.get(x, y);
                        for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                            let d = m.get(nx, ny);
                            let s = (0..3).map(|k| (c[k] - d[k]).abs()).fold(0.0, f32::max);
                            if s > at.2 {
                                at = (x, y, s);
                            }
                        }
                    }
                }
                assert!(step <= limit * 1.6 + 0.01, "{} t={t}: step {step} vs inputs {limit} at {at:?} shot {}", p.id, a.get(0, 0)[2]);
            }
        }
    }
}

#[test]
fn the_flow_is_computed_once_and_reused() {
    let (a, b) = (texture(0.0, 0.0), texture(6.0, 2.0));
    let e = instance("flow_morph");
    let f1 = content::for_frames(&e, &a, &b).unwrap();
    let f2 = content::for_frames(&e, &a, &b).unwrap();
    assert!(Arc::ptr_eq(&f1, &f2), "the second request is a cache hit");
    // the other flow styles with the same smoothness share the field too
    let doppler = instance("flow_doppler_warp");
    assert!(Arc::ptr_eq(&f1, &content::for_frames(&doppler, &a, &b).unwrap()));
    // another smoothness is another field
    let mut smooth = instance("flow_morph");
    set(&mut smooth, "smoothness", 90.0);
    assert!(!Arc::ptr_eq(&f1, &content::for_frames(&smooth, &a, &b).unwrap()));
    // the field holds the motion both ways (frame pixels): about (6, 2) and back
    let (rx, ry) = content::field_ratio(W, H);
    let mid = f1.sample(64.0, 36.0, (rx, ry));
    assert!((mid[0] - 6.0).abs() < 0.6 && (mid[1] - 2.0).abs() < 0.6, "{mid:?}");
    assert!((mid[2] + 6.0).abs() < 0.6 && (mid[3] + 2.0).abs() < 0.6, "{mid:?}");
    // a given field is used as is: rendering with it equals rendering without (it is the same)
    let with = apply_with(&e, &a, &b, 0.4, 1.0, Some(&f1));
    assert_eq!(with, apply(&e, &a, &b, 0.4));
    // a field from other anchors changes the warp (the timeline's anchors are not the frames
    // at hand), and one of the wrong size is ignored
    let other = content::analyse(&e, &texture(0.0, 0.0), &texture(-8.0, 0.0)).unwrap();
    assert_ne!(apply_with(&e, &a, &b, 0.4, 1.0, Some(&other)), with);
    let wrong = Field { w: 3, h: 2, px: vec![[50.0; 4]; 6], key: 7 };
    assert_eq!(apply_with(&e, &a, &b, 0.4, 1.0, Some(&wrong)), with);
}

#[test]
fn flow_styles_do_what_their_names_say() {
    let (a, b) = (texture(0.0, 0.0), texture(8.0, 0.0));
    let lum = |m: &Image| m.px.chunks(4).map(|c| c[0] + c[1] + c[2]).sum::<f32>() / (m.w * m.h) as f32;
    let morph = apply(&instance("flow_morph"), &a, &b, 0.5);
    // Hype Ramp: brighter at the swap (the pulse), and nearly still early on (the ramp)
    let hype = apply(&instance("flow_hype_ramp"), &a, &b, 0.5);
    assert!(lum(&hype) > lum(&morph) * 1.3, "{} vs {}", lum(&hype), lum(&morph));
    assert!(interior(&apply(&instance("flow_hype_ramp"), &a, &b, 0.15), &a, 0) < 0.01);
    // Doppler Warp: streaks blur the picture at the swap
    let doppler = apply(&instance("flow_doppler_warp"), &a, &b, 0.5);
    assert!(max_step(&doppler, 2) < max_step(&morph, 2));
    // Glitch Flow: some blocks differ from the morph at the cut, none near the ends
    let glitch = apply(&instance("flow_glitch"), &a, &b, 0.5);
    assert!(interior(&glitch, &morph, 0) > 1e-3);
    assert_eq!(apply(&instance("flow_glitch"), &a, &b, 0.05), apply(&instance("flow_morph"), &a, &b, 0.05));
    // Split Bands: each pixel is one clip or the other (no dissolve); bands alternate
    let solid_a = Image::filled(W, H, [1.0, 0.0, 0.0, 1.0]);
    let solid_b = Image::filled(W, H, [0.0, 0.0, 1.0, 1.0]);
    let mut bands = instance("flow_split_bands");
    set(&mut bands, "stagger", 0.0);
    set(&mut bands, "motion_blur", 0.0);
    let m = apply(&bands, &solid_a, &solid_b, 0.5);
    assert!(m.px.chunks(4).all(|c| (c[0] > 0.99 && c[2] < 0.01) || (c[2] > 0.99 && c[0] < 0.01)));
    // band 0 (top) slides right: B enters on its left; band 1 slides left: B on its right
    assert!(m.get(2, 2)[2] > 0.99 && m.get(W - 3, 2)[0] > 0.99, "band 0");
    assert!(m.get(2, H / 6 + 2)[0] > 0.99 && m.get(W - 3, H / 6 + 2)[2] > 0.99, "band 1");
    // Datamosh: blocks refresh to the incoming clip over time
    let (ra, rb) = (Image::filled(W, H, [1.0, 0.0, 0.0, 1.0]), Image::filled(W, H, [0.0, 0.0, 1.0, 1.0]));
    let share = |t: f32| apply(&instance("flow_datamosh"), &ra, &rb, t).px.chunks(4).filter(|c| c[2] > 0.99).count();
    let (early, mid, late) = (share(0.2), share(0.55), share(0.95));
    assert!(early == 0 && mid > 0 && late == W * H && early < mid, "{early} {mid} {late}");
}

// ---------------------------------------------------------------------------------------------
// Content-Aware Wipe

/// A picture with a salient textured object at columns `x0..x1`, rows 18..54, on flat `ground`;
/// `tint` is added to the object (so the two clips' objects differ).
fn subject(x0: usize, x1: usize, ground: [f32; 4], tint: f32) -> Image {
    paint(W, H, |x, y| {
        if (x0 as f32..x1 as f32).contains(&x) && (18.0..54.0).contains(&y) {
            let l = 0.5 + 0.4 * ((x * 1.3).sin() * (y * 0.9).cos());
            [l, l * 0.8, 0.3 + tint, 1.0]
        } else {
            ground
        }
    })
}

/// Whether pixel (x, y) of `m` is nearer the incoming picture `b` than the outgoing `a`.
fn shows_b(m: &Image, a: &Image, b: &Image, x: usize, y: usize) -> bool {
    let (c, pa, pb) = (m.get(x, y), a.get(x, y), b.get(x, y));
    let d = |p: [f32; 4]| (0..3).map(|k| (c[k] - p[k]).abs()).sum::<f32>();
    d(pb) < d(pa)
}

/// For each row, how many leading pixels show `b`, or None when the row is not one run of `b`
/// followed by `a`.
fn revealed(m: &Image, a: &Image, b: &Image) -> Vec<Option<usize>> {
    (0..m.h)
        .map(|y| {
            let row: Vec<bool> = (0..m.w).map(|x| shows_b(m, a, b, x, y)).collect();
            let n = row.iter().take_while(|v| **v).count();
            row[n..].iter().all(|v| !v).then_some(n)
        })
        .collect()
}

#[test]
fn the_content_aware_wipe_is_connected_and_monotone() {
    let (a, b) = (subject(48, 80, [0.9, 0.1, 0.1, 1.0], 0.0), subject(48, 80, [0.1, 0.1, 0.9, 1.0], 0.5));
    let mut e = instance("content_aware_wipe");
    set(&mut e, "softness", 0.0);
    let mut last: Vec<usize> = vec![0; H];
    for i in 1..40 {
        let p = i as f32 / 40.0;
        let m = apply(&e, &a, &b, p);
        let rows = revealed(&m, &a, &b);
        for (y, r) in rows.iter().enumerate() {
            // each row is revealed from the leading (left) edge as one run...
            let n = r.unwrap_or_else(|| panic!("p={p}: row {y} is not a single revealed run"));
            // ...that never shrinks
            assert!(n >= last[y], "p={p}: row {y} went back from {} to {n}", last[y]);
            last[y] = n;
        }
        // the edge is one connected line: neighbouring rows' edges are close (a field cell is
        // one pixel here, plus the bilinear rounding)
        for y in 1..H {
            let (u, v) = (rows[y].unwrap_or(0) as i64, rows[y - 1].unwrap_or(0) as i64);
            assert!((u - v).abs() <= 2, "p={p}: the edge jumps between rows {} and {y}: {v} → {u}", y - 1);
        }
    }
    // complete at the end, untouched at the start (but for the uv round trip of the sampling)
    assert!(interior(&apply(&e, &a, &b, 0.999_9), &b, 0) < 1e-5);
    assert!(interior(&apply(&e, &a, &b, 0.000_1), &a, 0) < 1e-5);
}

#[test]
fn the_content_aware_wipe_goes_round_what_stands_out() {
    // a salient object in the middle: the protected wipe cuts through it for a much shorter
    // stretch of the transition than the straight wipe (protection 0)
    let (a, b) = (subject(52, 76, [0.8, 0.8, 0.8, 1.0], 0.0), subject(52, 76, [0.3, 0.3, 0.9, 1.0], 0.6));
    let sliced_steps = |protect: f64| {
        let mut e = instance("content_aware_wipe");
        set(&mut e, "softness", 0.0);
        set(&mut e, "protect", protect);
        set(&mut e, "freedom", 100.0);
        (1..100)
            .filter(|i| {
                let m = apply(&e, &a, &b, *i as f32 / 100.0);
                let obj: Vec<bool> = (20..52).flat_map(|y| (54..74).map(move |x| (x, y))).map(|(x, y)| shows_b(&m, &a, &b, x, y)).collect();
                obj.iter().any(|v| *v) && !obj.iter().all(|v| *v)
            })
            .count()
    };
    let (straight, aware) = (sliced_steps(0.0), sliced_steps(100.0));
    assert!(aware * 2 < straight, "object sliced for {aware} of 99 steps (straight wipe: {straight})");
    // protection 0 is a straight wipe from the west
    let mut plain = instance("content_aware_wipe");
    set(&mut plain, "protect", 0.0);
    set(&mut plain, "softness", 0.0);
    let m = apply(&plain, &a, &b, 0.5);
    let edges: Vec<usize> = revealed(&m, &a, &b).into_iter().map(|r| r.unwrap_or(usize::MAX)).collect();
    assert!(edges.iter().all(|&n| (n as i64 - 64).abs() <= 1), "{edges:?}");
}

#[test]
fn the_content_aware_wipe_follows_its_direction() {
    let a = Image::filled(W, H, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(W, H, [0.0, 0.0, 1.0, 1.0]);
    let is_b = |c: [f32; 4]| c[2] > 0.99;
    for (dir, first, last) in
        [(0u32, (W / 2, 1), (W / 2, H - 2)), (1, (W - 2, H / 2), (1, H / 2)), (2, (W / 2, H - 2), (W / 2, 1)), (3, (1, H / 2), (W - 2, H / 2))]
    {
        let mut e = instance("content_aware_wipe");
        e.param_mut("direction").unwrap().value = ParamValue::Choice(dir);
        let m = apply(&e, &a, &b, 0.3);
        assert!(is_b(m.get(first.0, first.1)) && !is_b(m.get(last.0, last.1)), "direction {dir}");
    }
}
