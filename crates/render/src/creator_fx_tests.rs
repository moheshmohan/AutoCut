//! The Creator effects' CPU reference: deterministic, identity at an amount of zero, hostile
//! parameters clamped (never a panic, never a non-finite pixel), and each one visibly changes a
//! picture at its defaults. GPU parity is tested in `filmcraft-gpu` (`fx_tests.rs`).

use super::*;
use filmcraft_project::effect::CREATOR_EFFECTS;
use filmcraft_project::{ParamValue, find_effect};
use filmcraft_time::Tick;

fn cx(t: Tick, seconds: f64) -> FxCtx<'static> {
    FxCtx { t, px_scale: 1.0, seconds, timecode: "", clip_name: "", project: None, env: None, frame_rate: 30.0, working: filmcraft_color::WorkingSpace::Rec709 }
}

/// A colourful picture with a gradient, edges, partial alpha and super-whites.
fn picture(w: usize, h: usize) -> Image {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let c = filmcraft_color::hsl_to_rgb(fx, 0.8, 0.15 + 0.7 * fy).map(srgb_to_linear);
            let c = if (x / 7 + y / 5) % 4 == 0 { [1.4, 1.2, 0.9] } else { c };
            let a = if y % 9 == 0 { 0.5 } else { 1.0 };
            img.px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[c[0] * a, c[1] * a, c[2] * a, a]);
        }
    }
    img
}

fn run(id: &str, params: &[(&str, ParamValue)], t: Tick, seconds: f64) -> Image {
    let mut e = find_effect(id).unwrap_or_else(|| panic!("{id}")).instance();
    for (k, v) in params {
        e.params.get_mut(*k).unwrap_or_else(|| panic!("{id}.{k}")).value = v.clone();
    }
    let mut img = picture(48, 32);
    let op = eval(&e, &cx(t, seconds), img.w, img.h).unwrap_or_else(|| panic!("{id} evaluates"));
    assert!(op.gpu_ok(), "{id}: evaluated op is finite");
    op.apply(&mut img);
    img
}

/// The parameter that turns each effect off at zero.
fn amount_key(id: &str) -> &'static str {
    match id {
        "cr_motion_blur" => "length",
        "cr_light_sweep" => "intensity",
        _ => "amount",
    }
}

#[test]
fn every_creator_effect_changes_the_picture_and_is_deterministic() {
    let base = picture(48, 32);
    for id in CREATOR_EFFECTS {
        // (Glitch's default Bursts rhythm rests on some frames: test it constant)
        let p: &[(&str, ParamValue)] = if *id == "cr_glitch" { &[("rhythm", ParamValue::Choice(0))] } else { &[] };
        let a = run(id, p, Tick::from_seconds_f64(0.4), 0.4);
        let b = run(id, p, Tick::from_seconds_f64(0.4), 0.4);
        assert_eq!(a.px, b.px, "{id}: same frame, same pixels");
        assert!(a.px.iter().all(|v| v.is_finite()), "{id}: finite");
        let diff: f32 = a.px.iter().zip(&base.px).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.px.len() as f32;
        assert!(diff > 1e-3, "{id}: changes the picture at its defaults ({diff})");
    }
}

#[test]
fn zero_amount_is_the_identity() {
    let base = picture(48, 32);
    for id in CREATOR_EFFECTS {
        let out = run(id, &[(amount_key(id), ParamValue::Float(0.0))], Tick::from_seconds_f64(1.0), 1.0);
        assert_eq!(out.px, base.px, "{id}: amount 0 leaves the picture untouched");
    }
}

#[test]
fn hostile_parameters_are_clamped_and_never_panic() {
    let hostile = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, 1e300, -5.0, 0.0];
    for id in CREATOR_EFFECTS {
        let d = find_effect(id).unwrap();
        for v in hostile {
            // every number at once, then each alone
            let all: Vec<(&str, ParamValue)> = d
                .params
                .iter()
                .filter(|p| matches!(p.kind, filmcraft_project::ParamKind::Float { .. } | filmcraft_project::ParamKind::Angle))
                .map(|p| (p.id, ParamValue::Float(v)))
                .collect();
            for t in [Tick(i64::MAX / 2), Tick(-5), Tick::ZERO] {
                let out = std::panic::catch_unwind(|| run(id, &all, t, v)).unwrap_or_else(|_| panic!("{id} panicked at {v}"));
                assert!(out.px.iter().all(|x| x.is_finite()), "{id} at {v}: finite pixels");
            }
            for (k, val) in &all {
                let out = run(id, &[(k, val.clone())], Tick::ZERO, 0.0);
                assert!(out.px.iter().all(|x| x.is_finite()), "{id}.{k} at {v}: finite pixels");
            }
        }
        // hostile points and colours
        for p in d.params.iter().filter(|p| matches!(p.kind, filmcraft_project::ParamKind::Point)) {
            let out = run(id, &[(p.id, ParamValue::Vec2(filmcraft_geom::Vec2::new(f64::NAN, 1e300)))], Tick::ZERO, 0.0);
            assert!(out.px.iter().all(|x| x.is_finite()), "{id}.{}", p.id);
        }
        for p in d.params.iter().filter(|p| matches!(p.kind, filmcraft_project::ParamKind::Color)) {
            let out = run(id, &[(p.id, ParamValue::Color([f32::NAN, -3.0, f32::INFINITY, 1.0]))], Tick::ZERO, 0.0);
            assert!(out.px.iter().all(|x| x.is_finite()), "{id}.{}", p.id);
        }
    }
    // a 1×1 and an empty image
    for id in CREATOR_EFFECTS {
        let e = find_effect(id).unwrap().instance();
        for (w, h) in [(1, 1), (0, 0), (1, 7)] {
            let mut img = Image::filled(w, h, [0.5, 0.5, 0.5, 1.0]);
            if let Some(op) = eval(&e, &cx(Tick::ZERO, 0.0), w, h) {
                op.apply(&mut img);
            }
            assert!(img.px.iter().all(|x| x.is_finite()), "{id} {w}x{h}");
        }
    }
}

#[test]
fn animated_noise_moves_with_the_frame_and_stays_with_the_seed() {
    let f0 = run("cr_film_grain", &[], Tick::from_seconds_f64(1.0), 1.0);
    let f1 = run("cr_film_grain", &[], Tick::from_seconds_f64(1.0 + 1.0 / 30.0), 1.0 + 1.0 / 30.0);
    assert_ne!(f0.px, f1.px, "grain changes per frame");
    let still0 = run("cr_film_grain", &[("animate", ParamValue::Bool(false))], Tick::from_seconds_f64(1.0), 1.0);
    let still1 = run("cr_film_grain", &[("animate", ParamValue::Bool(false))], Tick::from_seconds_f64(2.0), 2.0);
    assert_eq!(still0.px, still1.px, "static grain");
    let other = run("cr_film_grain", &[("seed", ParamValue::Float(7.0))], Tick::from_seconds_f64(1.0), 1.0);
    assert_ne!(f0.px, other.px, "the seed changes the grain");
    // glitch: the same step of the rate gives the same frame
    let g0 = run("cr_glitch", &[("rhythm", ParamValue::Choice(0))], Tick::from_seconds_f64(0.51), 0.51);
    let g1 = run("cr_glitch", &[("rhythm", ParamValue::Choice(0))], Tick::from_seconds_f64(0.52), 0.52);
    assert_eq!(g0.px, g1.px, "12 Hz: 0.51 s and 0.52 s are the same step");
    // bursts: some steps rest (identity), others glitch
    let base = picture(48, 32);
    let steps: Vec<bool> = (0..24).map(|i| run("cr_glitch", &[], Tick::ZERO, i as f64 / 12.0).px != base.px).collect();
    assert!(steps.iter().any(|g| *g) && steps.iter().any(|g| !*g), "{steps:?}");
}

#[test]
fn hash_is_the_documented_pcg() {
    // fixed values guard the WGSL twin (`fx.wgsl` `pcg`) against drift
    assert_eq!(pcg(0), 129_708_002);
    assert_eq!(pcg(1), 2_831_084_092);
    assert!((0..1000).map(|i| h01(i, 3, 9)).all(|v| (0.0..1.0).contains(&v)));
}
