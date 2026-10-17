//! Shader-transition tests that need no GPU: the registry, the catalogue link, the host's time and
//! parameter handling, and hostile inputs. (Every transition also goes through the generic
//! contract, golden and parameter tests in `transitions/tests.rs`; GPU-vs-CPU parity is in
//! `filmcraft-gpu`.)

use filmcraft_project::{EffectInstance, ParamKind, ParamValue, effect_defs, gltransition};

use super::*;
use crate::transitions::apply;

fn frame(w: usize, h: usize, seed: f32) -> Image {
    paint(w, h, |x, y| {
        let u = x / w as f32;
        let v = y / h as f32;
        [0.1 + 0.8 * u * seed.fract(), 0.2 + 0.6 * v, 0.5 + 0.4 * ((x * 0.37 + seed).sin()), 1.0]
    })
}

fn instance(id: &str) -> EffectInstance {
    filmcraft_project::find_effect(id).unwrap().instance()
}

fn set(e: &mut EffectInstance, id: &str, v: ParamValue) {
    e.param_mut(id).unwrap().value = v;
}

#[test]
fn every_shader_validates() {
    let reg = registry();
    assert!(reg.errors().is_empty(), "{:#?}", reg.errors());
    for d in shaders().iter().chain(engines()) {
        let src = reg.source(d.id).unwrap();
        assert!(src.contains("fn transition("), "{}", d.id);
        assert_eq!(src.contains("getMap("), d.reads_map() || src.contains("fn getMap("), "{}", d.id);
    }
    // every preset is drawn by its engine; ports by themselves
    let ids: Vec<&str> = transition_ids().collect();
    assert_eq!(ids.len(), shaders().len() + engines::LUMA_PRESETS.len() + engines::BLEND_PRESETS.len());
    for id in &ids {
        let d = find(id).unwrap();
        assert!(is_shader(id) && filmcraft_project::find_effect(id).is_some(), "{id}");
        if id.starts_with("luma_") {
            assert_eq!(d.id, "engine_luma_wipe");
            assert!(d.reads_map());
        } else if id.starts_with("blend_") {
            assert_eq!(d.id, "engine_blend_dissolve");
        } else {
            assert_eq!(d.id, *id);
        }
    }
    for d in engines() {
        assert!(d.body.contains("original work, MIT OR Apache-2.0") && d.license == "MIT OR Apache-2.0", "{}", d.id);
    }
    assert_eq!(reg.source("cross_dissolve"), Err(ShaderError::Unknown("cross_dissolve".into())));
}

#[test]
fn invalid_wgsl_is_an_error_not_a_panic() {
    for bad in ["", "fn transition(", "fn transition(uv: vec2<f32>) -> vec4<f32> { return 1; }", "\u{0}\u{ffff}@@@", &"(".repeat(10_000)] {
        let src = format!("{PRELUDE}\n{bad}");
        match validate("gl_test", &src) {
            Err(ShaderError::Invalid { id, message }) => {
                assert_eq!(id, "gl_test");
                assert!(!message.is_empty());
            }
            other => panic!("{bad:.40}: {other:?}"),
        }
    }
    // the prelude alone lacks `transition`
    assert!(validate("gl_test", PRELUDE).is_err());
}

#[test]
fn catalogue_and_registry_match() {
    let gl: Vec<&str> = effect_defs().iter().filter(|d| d.category == gltransition::GL_TRANSITIONS).map(|d| d.id).collect();
    let ids: Vec<&str> = shaders().iter().map(|d| d.id).collect();
    assert_eq!(gl, ids, "catalogue order (by name) and registry order (by id) agree");
    for d in shaders() {
        assert!(is_shader(d.id));
        let body = d.body;
        assert!(body.contains(&format!("Author: {}", d.author)), "{}: upstream Author header", d.id);
        assert!(body.contains("License: MIT") && d.license == "MIT", "{}: MIT only", d.id);
        assert!(body.contains(&format!("transitions/{}.glsl", d.upstream)), "{}: upstream link", d.id);
        // every port is attributed next to its file
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/transitions/shader");
        let side = std::fs::read_to_string(dir.join(format!("{}.wgsl.attribution", d.upstream))).unwrap_or_else(|e| panic!("{}: {e}", d.id));
        assert!(side.contains("license:      MIT") && side.contains("LICENSE-gl-transitions.txt"), "{}", d.id);
    }
    assert!(!is_shader("push"));
}

#[test]
fn uniforms_pack_the_host_values() {
    let e = instance("gl_starwipe");
    let u = uniforms(&e, 320, 180, 0.25, 1.0).unwrap();
    assert_eq!(&u[..4], &[320.0, 180.0, 0.25, 320.0 / 180.0]);
    // border thickness, rotation, centre (uv, frame centre by default), white border
    assert_eq!(&u[4..12], &[0.01, 0.75, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]);
    // then 1 / width and 1 / height: a pixel's uv is its centre times these, on CPU and GPU
    assert_eq!(&u[4 + PARAMS..], &[1.0 / 320.0, 1.0 / 180.0, 0.0, 0.0]);
    assert!(uniforms(&instance("cross_dissolve"), 320, 180, 0.5, 1.0).is_none());
    // a centre in sequence pixels maps to uv with y up; ½-resolution frames scale it
    let mut e = instance("gl_starwipe");
    set(&mut e, "center", ParamValue::Vec2(filmcraft_geom::Vec2::new(80.0, 45.0)));
    let u = uniforms(&e, 160, 90, 0.5, 0.5).unwrap();
    assert_eq!((u[6], u[7]), (0.25, 0.75));
    // integer parameters are whole numbers
    let mut e = instance("gl_pixelize");
    set(&mut e, "squares_min_x", ParamValue::Float(7.6));
    set(&mut e, "steps", ParamValue::Float(-3.0));
    let u = uniforms(&e, 64, 64, 0.5, 1.0).unwrap();
    assert_eq!((u[4], u[5], u[6]), (8.0, 20.0, 0.0));
    // degenerate sizes still give a finite ratio
    let u = uniforms(&instance("gl_dreamyzoom"), 0, 0, 0.5, f32::NAN).unwrap();
    assert!(u.iter().all(|v| v.is_finite()));
}

#[test]
fn easing_is_applied_by_the_host() {
    let (a, b) = (frame(64, 36, 0.3), frame(64, 36, 0.8));
    for id in ["gl_crosswarp", "gl_angular", "gl_ripple"] {
        let linear = instance(id);
        let mut cubic = instance(id);
        set(&mut cubic, "easing", ParamValue::Choice(7)); // Cubic In: 0.5 → 0.125
        assert_eq!(eased_progress(&cubic, 0.5), 0.125);
        assert_eq!(apply(&cubic, &a, &b, 0.5), apply(&linear, &a, &b, 0.125), "{id}");
        // a custom Bézier is data: (0.25, 0.25, 0.75, 0.75) is the identity
        let mut bez = instance(id);
        set(&mut bez, "easing", ParamValue::Choice(gltransition::EASING_CUSTOM_BEZIER));
        set(&mut bez, "bezier_x1", ParamValue::Float(0.25));
        set(&mut bez, "bezier_y1", ParamValue::Float(0.25));
        set(&mut bez, "bezier_x2", ParamValue::Float(0.75));
        set(&mut bez, "bezier_y2", ParamValue::Float(0.75));
        let p = eased_progress(&bez, 0.4);
        assert!((p - 0.4).abs() < 1e-5, "{p}");
        // an easing that stays at 0 shows the outgoing frame exactly (and 1 the incoming)
        let mut back = instance(id);
        set(&mut back, "easing", ParamValue::Choice(22)); // Back In dips below 0 early on
        assert_eq!(apply(&back, &a, &b, 0.1), a, "{id}");
        set(&mut back, "easing", ParamValue::Choice(23)); // Back Out overshoots late
        assert_eq!(apply(&back, &a, &b, 0.9), b, "{id}");
    }
}

#[test]
fn endpoints_are_exact_at_every_size() {
    for (w, h) in [(1, 1), (1, 9), (9, 1), (3, 2), (17, 5), (64, 36)] {
        let (a, b) = (frame(w, h, 0.2), frame(w, h, 0.9));
        for id in transition_ids() {
            let e = instance(id);
            assert_eq!(apply(&e, &a, &b, 0.0), a, "{id} {w}x{h}");
            assert_eq!(apply(&e, &a, &b, 1.0), b, "{id} {w}x{h}");
        }
    }
}

/// Deterministic pseudo-random sequence for fuzzing.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

#[test]
fn fuzzed_parameters_and_odd_frames_never_panic() {
    let hostile = [f64::NAN, f64::INFINITY, -f64::INFINITY, f64::MAX, f64::MIN, -1e30, 1e30, -1.0, 0.0, 1e-30, 0.5, 7.0, 360.0, -720.0];
    let sizes = [(1usize, 1usize), (2, 1), (1, 2), (7, 3), (3, 7), (31, 2), (2, 31), (40, 23)];
    let progresses = [f32::NAN, -1.0, 0.0, 1e-7, 0.013, 0.25, 0.5, 0.5001, 0.75, 0.999_99, 1.0, 3.0, f32::INFINITY];
    let mut state = 0x5eed_u64;
    for id in transition_ids() {
        let def = filmcraft_project::find_effect(id).unwrap();
        for round in 0..24 {
            let mut e = def.instance();
            for p in &def.params {
                let pick = hostile[(lcg(&mut state) as usize) % hostile.len()];
                let v = match &p.kind {
                    ParamKind::Float { .. } | ParamKind::Angle => ParamValue::Float(pick),
                    ParamKind::Choice(opts) => ParamValue::Choice(match round % 3 {
                        0 => lcg(&mut state) as u32 % (opts.len() as u32 + 4),
                        1 => u32::MAX,
                        _ => gltransition::EASING_CUSTOM_BEZIER,
                    }),
                    ParamKind::Bool => ParamValue::Bool(round % 2 == 0),
                    ParamKind::Color => ParamValue::Color([pick as f32, -3.0, f32::NAN, f32::INFINITY]),
                    ParamKind::Point => ParamValue::Vec2(filmcraft_geom::Vec2::new(pick, -pick)),
                    _ => continue,
                };
                set(&mut e, p.id, v);
            }
            // occasionally a value of the wrong type
            if round == 5
                && let Some(first) = def.params.first()
            {
                set(&mut e, first.id, ParamValue::Bool(true));
            }
            let (w, h) = sizes[round % sizes.len()];
            let (a, b) = (frame(w, h, 0.4), frame(w, h, 0.6));
            for &p in &progresses {
                let m = apply(&e, &a, &b, p);
                assert_eq!((m.w, m.h), (w, h), "{id} {w}x{h}");
                assert!(m.px.iter().all(|v| v.is_finite()), "{id} p={p} {w}x{h} round {round}: non-finite");
                let u = uniforms(&e, w as u32, h as u32, p, 1.0).unwrap();
                assert!(u.iter().all(|v| v.is_finite()), "{id} uniforms {u:?}");
                assert!((0.0..=1.0).contains(&u[2]), "{id} eased progress {}", u[2]);
            }
        }
    }
}

#[test]
fn shaders_do_what_their_names_say() {
    let (w, h) = (64usize, 36usize);
    let a = Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]);
    let is_b = |c: [f32; 4]| c[2] > 0.99 && c[0] < 0.01;
    let is_a = |c: [f32; 4]| c[0] > 0.99 && c[2] < 0.01;
    // circleopen: the centre opens first, the corners last
    let m = apply(&instance("gl_circleopen"), &a, &b, 0.5);
    assert!(is_b(m.get(w / 2, h / 2)) && is_a(m.get(0, 0)));
    // ...and closes instead when not opening
    let mut closing = instance("gl_circleopen");
    set(&mut closing, "opening", ParamValue::Bool(false));
    let m = apply(&closing, &a, &b, 0.5);
    assert!(is_a(m.get(w / 2, h / 2)) && is_b(m.get(0, 0)));
    // angular: the wipe sweeps from 90° (straight up from the centre's left, GL coordinates)
    let m = apply(&instance("gl_angular"), &a, &b, 0.5);
    let bs = m.px.chunks(4).filter(|c| is_b([c[0], c[1], c[2], c[3]])).count();
    assert!((bs as f32 / (w * h) as f32 - 0.5).abs() < 0.05, "half swept: {bs}");
    // StarWipe: the star grows from the centre, with a white border between
    let m = apply(&instance("gl_starwipe"), &a, &b, 0.15);
    assert!(is_b(m.get(w / 2, h / 2)) && is_a(m.get(0, 0)));
    assert!(m.px.chunks(4).any(|c| c[0] > 0.99 && c[1] > 0.99 && c[2] > 0.99), "border");
    // DreamyZoom flashes to white at the midpoint without making alpha exceed 1
    let m = apply(&instance("gl_dreamyzoom"), &a, &b, 0.5);
    assert!(m.px.chunks(4).all(|c| c[1] >= 0.99 && c[3] <= 1.0));
    // Mosaic ends on the incoming tile; pixelize is blocky at its strongest
    let m = apply(&instance("gl_mosaic"), &a, &b, 0.999);
    assert!(m.px.chunks(4).all(|c| c[2] > 0.99));
    let ramp = paint(w, h, |x, _| [x / w as f32, 0.0, 0.0, 1.0]);
    let m = apply(&instance("gl_pixelize"), &ramp, &ramp, 0.5);
    let distinct = {
        let mut v: Vec<u32> = (0..w).map(|x| m.get(x, 0)[0].to_bits()).collect();
        v.dedup();
        v.len()
    };
    assert!(distinct <= 21, "20 squares across: {distinct} distinct columns");
}

#[test]
fn batch_two_shaders_do_what_their_names_say() {
    let (w, h) = (64usize, 36usize);
    let a = Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]);
    let is_b = |c: [f32; 4]| c[2] > 0.99 && c[0] < 0.01;
    let is_a = |c: [f32; 4]| c[0] > 0.99 && c[2] < 0.01;
    let share_b = |m: &Image| m.px.chunks(4).filter(|c| is_b([c[0], c[1], c[2], c[3]])).count() as f32 / (w * h) as f32;
    // TVStatic: opaque grey static in the middle, new on every frame
    let s1 = apply(&instance("gl_tvstatic"), &a, &b, 0.5);
    let s2 = apply(&instance("gl_tvstatic"), &a, &b, 0.51);
    assert!(s1.px.chunks(4).all(|c| c[0] == c[1] && c[1] == c[2] && c[3] == 1.0));
    assert_ne!(s1, s2);
    // polar_function: the flower opens from the centre
    let m = apply(&instance("gl_polar_function"), &a, &b, 0.3);
    assert!(is_b(m.get(w / 2, h / 2)) && is_a(m.get(0, 0)));
    // wind: left to right, or right to left reversed
    let m = apply(&instance("gl_wind"), &a, &b, 0.5);
    assert!(is_b(m.get(0, h / 2)) && is_a(m.get(w - 1, h / 2)));
    let mut reversed = instance("gl_wind");
    set(&mut reversed, "reversed", ParamValue::Bool(true));
    let m = apply(&reversed, &a, &b, 0.5);
    assert!(is_a(m.get(0, h / 2)) && is_b(m.get(w - 1, h / 2)));
    // chessboard: half the squares have turned at the midpoint
    let m = apply(&instance("gl_chessboard"), &a, &b, 0.5);
    assert!((share_b(&m) - 0.5).abs() < 0.05, "{}", share_b(&m));
    // squeeze: the outgoing image is squeezed into the middle
    let m = apply(&instance("gl_squeeze"), &a, &b, 0.5);
    assert!(is_a(m.get(w / 2, h / 2)) && is_b(m.get(w / 2, 0)) && is_b(m.get(w / 2, h - 1)));
    // burn and the floors of doorway / swap / cube never make alpha exceed 1
    for id in ["gl_burn", "gl_doorway", "gl_swap", "gl_cube", "gl_glitchdisplace"] {
        for p in [0.2, 0.5, 0.8] {
            let m = apply(&instance(id), &a, &b, p);
            assert!(m.px.chunks(4).all(|c| c[3] <= 1.0 && c[3] >= 0.0), "{id} p={p}");
        }
    }
    // PolkaDotsCurtain's dots grow from the bottom left corner by default
    let m = apply(&instance("gl_polkadotscurtain"), &a, &b, 0.05);
    assert!(is_b(m.get(0, h - 1)));
    assert!(is_a(m.get(w - 1, 0)));
}

// ---------------------------------------------------------------------------------------------
// C2: the engines and the map-reading ports

fn share(m: &Image, pred: impl Fn([f32; 4]) -> bool) -> f32 {
    m.px.chunks(4).filter(|c| pred([c[0], c[1], c[2], c[3]])).count() as f32 / (m.w * m.h) as f32
}

#[test]
fn luma_wipes_follow_their_maps() {
    let (w, h) = (64usize, 36usize);
    let a = Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]);
    let is_b = |c: [f32; 4]| c[2] > 0.99 && c[0] < 0.01;
    let is_a = |c: [f32; 4]| c[0] > 0.99 && c[2] < 0.01;
    // Luma Linear (90°, left to right): left half in, right half out at the midpoint
    let m = apply(&instance("luma_linear"), &a, &b, 0.5);
    assert!(is_b(m.get(2, 10)) && is_a(m.get(w - 3, 10)));
    assert!((share(&m, is_b) - 0.5).abs() < 0.08, "{}", share(&m, is_b));
    // the incoming share grows with the progress
    let shares: Vec<f32> = [0.1, 0.3, 0.5, 0.7, 0.9].iter().map(|&p| share(&apply(&instance("luma_clouds"), &a, &b, p), is_b)).collect();
    assert!(shares.windows(2).all(|s| s[0] <= s[1]), "{shares:?}");
    // Invert reverses the order: the radial iris opens at the corners instead
    let m = apply(&instance("luma_radial"), &a, &b, 0.3);
    assert!(is_b(m.get(w / 2, h / 2)) && is_a(m.get(0, 0)));
    let m = apply(&instance("luma_radial_in"), &a, &b, 0.3);
    assert!(is_a(m.get(w / 2, h / 2)) && is_b(m.get(0, 0)));
    // softness 0 is a hard edge: only the two pictures, nothing between
    let mut hard = instance("luma_diagonal");
    set(&mut hard, "softness", ParamValue::Float(0.0));
    let m = apply(&hard, &a, &b, 0.4);
    assert!(m.px.chunks(4).all(|c| is_a([c[0], c[1], c[2], c[3]]) || is_b([c[0], c[1], c[2], c[3]])));
    // ...and softness makes a band of mixed pixels
    let mut soft = instance("luma_diagonal");
    set(&mut soft, "softness", ParamValue::Float(60.0));
    let m = apply(&soft, &a, &b, 0.4);
    assert!(share(&m, |c| !is_a(c) && !is_b(c)) > 0.2);
    // feathering a hard-edged map softens the edge too
    let mut feathered = instance("luma_checkerboard");
    set(&mut feathered, "softness", ParamValue::Float(0.0));
    let plain = apply(&feathered, &a, &b, 0.5);
    set(&mut feathered, "feather", ParamValue::Float(30.0));
    assert_ne!(apply(&feathered, &a, &b, 0.5), plain);
    // the wipe is complete exactly at the end, whatever the softness
    for softness in [0.0, 50.0, 100.0] {
        let mut e = instance("luma_star");
        set(&mut e, "softness", ParamValue::Float(softness));
        let m = apply(&e, &a, &b, 0.999_99);
        assert!(share(&m, is_b) > 0.99, "softness {softness}");
        let m = apply(&e, &a, &b, 0.000_01);
        assert!(share(&m, is_a) > 0.99, "softness {softness}");
    }
    // a centre in sequence pixels, scaled with the frame
    let mut e = instance("luma_radial");
    set(&mut e, "center", ParamValue::Vec2(filmcraft_geom::Vec2::new(0.0, 0.0)));
    let m = apply(&e, &a, &b, 0.2);
    assert!(is_b(m.get(0, 0)) && is_a(m.get(w - 1, h - 1)));
}

#[test]
fn every_luma_preset_reveals_and_completes() {
    let (w, h) = (48usize, 27usize);
    let a = Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]);
    for p in engines::LUMA_PRESETS {
        let e = instance(p.id);
        let blue = |q: f32| apply(&e, &a, &b, q).px.chunks(4).map(|c| c[2]).sum::<f32>() / (w * h) as f32;
        let (early, mid, late) = (blue(0.1), blue(0.5), blue(0.9));
        assert!(early < mid && mid < late, "{}: {early} {mid} {late}", p.id);
        assert!((0.15..=0.85).contains(&mid), "{}: half way {mid}", p.id);
    }
}

#[test]
fn blend_dissolves_pass_through_their_blend_mode() {
    let (w, h) = (16usize, 9usize);
    let a = Image::filled(w, h, [0.6, 0.1, 0.05, 1.0]);
    let b = Image::filled(w, h, [0.05, 0.2, 0.5, 1.0]);
    // at the midpoint with full strength the picture is the blend itself
    for (id, mode) in [("blend_screen", crate::Blend::Screen), ("blend_multiply", crate::Blend::Multiply), ("blend_difference", crate::Blend::Difference)] {
        let m = apply(&instance(id), &a, &b, 0.5);
        let mut want = a.clone();
        crate::blend::composite(&mut want, &b, 1.0, mode);
        for (g, x) in m.px.iter().zip(&want.px) {
            assert!((g - x).abs() < 1e-5, "{id}: {g} vs {x}");
        }
    }
    // strength 0 is a plain cross dissolve
    let mut e = instance("blend_overlay");
    set(&mut e, "strength", ParamValue::Float(0.0));
    let m = apply(&e, &a, &b, 0.3);
    let want = a.clone().lerp(&b, 0.3);
    assert!(m.px.iter().zip(&want.px).all(|(g, x)| (g - x).abs() < 1e-6));
    // Screen lightens and Multiply darkens through the middle
    let lum = |m: &Image| m.px.chunks(4).map(|c| c[0] + c[1] + c[2]).sum::<f32>();
    let fade = a.clone().lerp(&b, 0.5);
    assert!(lum(&apply(&instance("blend_screen"), &a, &b, 0.5)) > lum(&fade));
    assert!(lum(&apply(&instance("blend_multiply"), &a, &b, 0.5)) < lum(&fade));
    // the mode can be changed from a preset: Screen Dissolve set to Multiply is Multiply Dissolve
    let mut e = instance("blend_screen");
    set(&mut e, "mode", ParamValue::Choice(3));
    assert_eq!(apply(&e, &a, &b, 0.4), apply(&instance("blend_multiply"), &a, &b, 0.4));
    // Dither Dissolve: each pixel is one picture or the other, the incoming share follows p
    let (w, h) = (128usize, 72usize);
    let (a, b) = (Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]), Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]));
    for p in [0.2f32, 0.5, 0.8] {
        let m = apply(&instance("blend_dissolve"), &a, &b, p);
        let blue = share(&m, |c| c[2] > 0.99);
        assert!(m.px.chunks(4).all(|c| c[0] > 0.99 || c[2] > 0.99));
        assert!((blue - p).abs() < 0.03, "p={p}: {blue}");
    }
    // hostile modes stay in range
    let mut e = instance("blend_hue");
    set(&mut e, "mode", ParamValue::Choice(u32::MAX));
    assert!(apply(&e, &a, &b, 0.5).px.iter().all(|v| v.is_finite()));
}

#[test]
fn displacement_and_luma_read_generated_maps() {
    let (w, h) = (64usize, 36usize);
    let a = Image::filled(w, h, [1.0, 0.0, 0.0, 1.0]);
    let b = Image::filled(w, h, [0.0, 0.0, 1.0, 1.0]);
    let is_b = |c: [f32; 4]| c[2] > 0.99 && c[0] < 0.01;
    // GL luma: a hard wipe, B where the map is below the progress
    let e = instance("gl_luma");
    let lmap = map(&e, w as u32, h as u32, 1.0).unwrap();
    let m = apply(&e, &a, &b, 0.4);
    for y in 0..h {
        for x in 0..w {
            let v = lmap.at(x as i64, y as i64);
            assert_eq!(is_b(m.get(x, y)), v < 0.4, "({x}, {y}): map {v}");
        }
    }
    // each map choice is a different map (at a scale that makes several cells and slats)
    let mut maps_seen = std::collections::HashSet::new();
    for c in 0..gltransition::MAP_OPTS.len() as u32 {
        let mut e = instance("gl_luma");
        set(&mut e, "map", ParamValue::Choice(c));
        set(&mut e, "map_scale", ParamValue::Float(16.0));
        maps_seen.insert(map(&e, w as u32, h as u32, 1.0).unwrap().px.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(maps_seen.len(), gltransition::MAP_OPTS.len());
    // displacement: with no strength it is a cross dissolve; with some, the pictures move
    let (fa, fb) = (frame(w, h, 0.3), frame(w, h, 0.7));
    let mut flat = instance("gl_displacement");
    set(&mut flat, "strength", ParamValue::Float(0.0));
    let m = apply(&flat, &fa, &fb, 0.5);
    let want = fa.clone().lerp(&fb, 0.5);
    assert!(m.px.iter().zip(&want.px).all(|(g, x)| (g - x).abs() < 1e-5));
    assert_ne!(apply(&instance("gl_displacement"), &fa, &fb, 0.5), m);
    // only shaders that read a map get one
    assert!(map(&instance("gl_crosswarp"), 64, 36, 1.0).is_none());
    assert!(map(&instance("blend_screen"), 64, 36, 1.0).is_none());
    assert!(map(&instance("luma_radial"), 64, 36, 1.0).is_some());
}
