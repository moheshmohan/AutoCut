//! GPU-vs-CPU parity of the Creator effects (`filmcraft_render::creator_fx`, the reference).
//!
//! The same criteria as the standard effects (`fx_tests.rs`): every value within
//! [`super::fx_tests::EXACT`] (relative) of the CPU on the working image read back as f32, except
//! effects with hard edges (glitch blocks, halftone dots, kaleidoscope seams, the tech HUD's
//! lines), where at most [`super::fx_tests::FLIPS`] of the values may land on the other side of
//! an edge; and the compositor's 8-bit criterion (p99 ≤ 6, mean < 1.5) for composited layers.

use super::blend_tests::{device, interior, stats8, yuv_frame};
use super::fx_tests::{EXACT, FLIPS, ch, col, compare, effect, fl, picture, pt};
use super::*;
use filmcraft_geom::Affine;
use filmcraft_project::ParamValue;
use filmcraft_render::effects::FxCtx;
use filmcraft_render::gpufx::FxOp;
use filmcraft_render::plan::{LayerFx, execute_cpu};
use filmcraft_time::Tick;

fn ctx(seconds: f64, px_scale: f32) -> FxCtx<'static> {
    FxCtx {
        t: Tick::from_seconds_f64(seconds),
        px_scale,
        seconds,
        timecode: "",
        clip_name: "",
        project: None,
        env: None,
        frame_rate: 30.0,
        working: filmcraft_color::WorkingSpace::Rec709,
    }
}

/// (effect, parameters, clip seconds, whether hard edges may flip pixels).
#[allow(clippy::type_complexity)]
fn cases() -> Vec<(&'static str, Vec<(&'static str, ParamValue)>, f64, bool)> {
    let on = ParamValue::Bool(true);
    vec![
        ("cr_film_grain", vec![], 0.5, false),
        ("cr_film_grain", vec![("amount", fl(100.0)), ("size", fl(0.5)), ("color", on.clone()), ("response", fl(0.0))], 2.0, false),
        ("cr_film_grain", vec![("size", fl(6.0)), ("animate", ParamValue::Bool(false)), ("seed", fl(42.0))], 0.0, false),
        ("cr_vhs", vec![], 0.7, false),
        ("cr_vhs", vec![("amount", fl(100.0)), ("bleed", fl(20.0)), ("tracking", fl(100.0)), ("noise", fl(80.0)), ("wobble", fl(10.0))], 3.3, false),
        ("cr_vhs", vec![("saturation", fl(0.0)), ("seed", fl(9.0))], 12.0, false),
        ("cr_kaleidoscope", vec![], 0.0, true),
        (
            "cr_kaleidoscope",
            vec![("segments", fl(3.0)), ("rotation", fl(33.0)), ("center", pt(20.0, 15.0)), ("zoom", fl(150.0)), ("amount", fl(60.0))],
            0.0,
            true,
        ),
        ("cr_kaleidoscope", vec![("segments", fl(17.0)), ("zoom", fl(40.0))], 0.0, true),
        ("cr_glitch", vec![("rhythm", ch(0))], 0.25, true),
        ("cr_glitch", vec![("amount", fl(100.0)), ("block", fl(6.0)), ("split", fl(20.0)), ("jitter", fl(100.0)), ("rhythm", ch(0))], 1.0, true),
        ("cr_glitch", vec![("seed", fl(3.0))], 0.75, true),
        ("cr_zoom_blur", vec![], 0.0, false),
        ("cr_zoom_blur", vec![("amount", fl(80.0)), ("center", pt(10.0, 30.0))], 0.0, false),
        ("cr_zoom_blur", vec![("kind", ch(1)), ("amount", fl(50.0))], 0.0, false),
        ("cr_motion_blur", vec![("length", fl(12.0))], 0.0, false),
        ("cr_motion_blur", vec![("length", fl(30.0)), ("direction", fl(30.0)), ("shutter", ch(1)), ("falloff", fl(70.0))], 0.0, false),
        ("cr_motion_blur", vec![("length", fl(1000.0)), ("falloff", fl(100.0))], 0.0, false),
        ("cr_tech_hud", vec![], 0.4, true),
        (
            "cr_tech_hud",
            vec![("grid_size", fl(12.0)), ("grid", fl(100.0)), ("scanlines", fl(100.0)), ("spacing", fl(2.0)), ("color", col(1.0, 0.3, 0.1))],
            1.1,
            true,
        ),
        ("cr_chromatic_aberration", vec![], 0.0, false),
        ("cr_chromatic_aberration", vec![("amount", fl(30.0)), ("center", pt(5.0, 5.0))], 0.0, false),
        ("cr_chromatic_aberration", vec![("kind", ch(1)), ("amount", fl(4.5)), ("angle", fl(120.0))], 0.0, false),
        ("cr_light_sweep", vec![], 0.0, false),
        ("cr_light_sweep", vec![("intensity", fl(300.0)), ("width", fl(40.0)), ("softness", fl(0.0)), ("angle", fl(-60.0)), ("position", fl(20.0))], 0.0, true),
        ("cr_light_sweep", vec![("cycle", fl(1.5)), ("color", col(0.4, 0.8, 1.0))], 2.2, false),
        ("cr_duotone", vec![], 0.0, false),
        ("cr_duotone", vec![("contrast", fl(80.0)), ("amount", fl(50.0)), ("shadows", col(0.0, 0.2, 0.1))], 0.0, false),
        ("cr_duotone", vec![("contrast", fl(-100.0))], 0.0, false),
        ("cr_halftone", vec![], 0.0, true),
        ("cr_halftone", vec![("mode", ch(1)), ("size", fl(6.0)), ("angle", fl(15.0))], 0.0, true),
        ("cr_halftone", vec![("size", fl(3.0)), ("amount", fl(40.0)), ("ink", col(0.6, 0.0, 0.2))], 0.0, true),
    ]
}

/// Every Creator case on the GPU against its CPU reference, on the working image read back as f32.
#[test]
fn creator_effects_match_cpu() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let (w, h) = (67u32, 41u32);
    let px = picture(w, h);
    let frame = VideoFrame::rgba_f32(w, h, px.clone());
    let mut failures = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (id, params, secs, edges) in cases() {
        seen.insert(id);
        let e = effect(id, &params);
        let op = FxOp::eval(&e, &ctx(secs, 1.0), w as usize, h as usize).unwrap_or_else(|| panic!("{id} has no GPU op"));
        assert!(op.gpu_ok(), "{id} {params:?}: not GPU-capable");
        let mut cpu = filmcraft_render::Image { w: w as usize, h: h as usize, px: px.clone() };
        op.apply(&mut cpu);
        assert_ne!(cpu.px, px, "{id} {params:?}: the case changes the picture");
        let fx = LayerFx { size: (w, h), decimation: 1, ops: vec![op] };
        let (_, _, gpu) = c.effect_image(&frame, &fx).expect("effect image");
        let (worst, flips) = compare(&cpu.px, &gpu);
        // values off by more than the exact tolerance: a pixel on the other side of an edge
        // (a dot, a block or a seam a rounding away) moves its neighbourhood a little too
        let off = cpu.px.iter().zip(&gpu).filter(|(a, b)| (*a - *b).abs() / (1.0 + a.abs()) > EXACT * 2.0).count() as f64 / gpu.len() as f64;
        eprintln!("{id} {params:?}: max rel diff {worst:.2e}, {:.3}% flipped, {:.3}% off", flips * 100.0, off * 100.0);
        let ok = if edges { off <= FLIPS } else { worst <= EXACT * 2.0 && flips == 0.0 };
        if !ok {
            failures.push(format!("{id} {params:?}: {worst:.2e}, flips {flips:.4}, off {off:.4}"));
        }
    }
    assert_eq!(seen.len(), filmcraft_project::effect::CREATOR_EFFECTS.len(), "a case per Creator effect");
    assert!(failures.is_empty(), "GPU Creator effects differ from the CPU:\n{}", failures.join("\n"));
}

/// Zero amounts run no pass at all (the source comes back as it went in).
#[test]
fn zero_amount_creator_effects_run_no_pass() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let (w, h) = (33u32, 17u32);
    let px = picture(w, h);
    let frame = VideoFrame::rgba_f32(w, h, px.clone());
    for id in filmcraft_project::effect::CREATOR_EFFECTS {
        let key = match *id {
            "cr_motion_blur" => "length",
            "cr_light_sweep" => "intensity",
            _ => "amount",
        };
        let op = FxOp::eval(&effect(id, &[(key, fl(0.0))]), &ctx(0.0, 1.0), w as usize, h as usize).expect("op");
        let (_, _, gpu) = c.effect_image(&frame, &LayerFx { size: (w, h), decimation: 1, ops: vec![op] }).expect("effect image");
        assert_eq!(gpu, px, "{id}: identity");
    }
}

/// A clip with a Creator chain composited by the GPU plan (YUV source decoded at half size, a
/// blend mode) against the CPU plan, at the compositor's 8-bit criterion.
#[test]
fn creator_chains_composite_like_the_cpu() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let (w, h) = (320usize, 180usize);
    let chains = [
        vec![effect("cr_vhs", &[]), effect("cr_film_grain", &[("amount", fl(60.0))]), effect("cr_tech_hud", &[])],
        vec![effect("cr_glitch", &[("rhythm", ch(0))]), effect("cr_chromatic_aberration", &[("amount", fl(14.0))]), effect("cr_light_sweep", &[])],
        vec![effect("cr_kaleidoscope", &[]), effect("cr_duotone", &[]), effect("cr_zoom_blur", &[]), effect("cr_motion_blur", &[]), effect("cr_halftone", &[])],
    ];
    for (i, chain) in chains.iter().enumerate() {
        let ops: Vec<FxOp> = chain.iter().filter_map(|e| FxOp::eval(e, &ctx(0.6, 0.5), 320, 180)).collect();
        assert_eq!(ops.len(), chain.len());
        assert!(ops.iter().all(FxOp::gpu_ok));
        let plan = FramePlan::Layers {
            width: w,
            height: h,
            layers: vec![PlanLayer {
                frame: yuv_frame(640, 360),
                matrix: Affine::IDENTITY,
                opacity: 1.0,
                blend: Blend::Normal,
                fx: Some(std::sync::Arc::new(LayerFx { size: (320, 180), decimation: 2, ops })),
                input: None,
            }],
        };
        let cpu = execute_cpu(&plan).over_black_rgba8();
        c.composite(&plan);
        let (_, _, gpu) = c.read_output().expect("readback");
        let keep = interior(&plan);
        let (p99, mean, max) = stats8(&cpu, &gpu, &keep);
        eprintln!("chain {i}: 8-bit p99 {p99} mean {mean:.3} max {max}");
        assert!(p99 <= 6 && mean < 1.5, "chain {i}: p99 {p99}, mean {mean}");
    }
}
