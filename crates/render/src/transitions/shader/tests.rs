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
    for d in shaders() {
        let src = reg.source(d.id).unwrap();
        assert!(src.contains("fn transition("), "{}", d.id);
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
        for d in shaders() {
            let e = instance(d.id);
            assert_eq!(apply(&e, &a, &b, 0.0), a, "{} {w}x{h}", d.id);
            assert_eq!(apply(&e, &a, &b, 1.0), b, "{} {w}x{h}", d.id);
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
    for d in shaders() {
        let def = filmcraft_project::find_effect(d.id).unwrap();
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
                assert_eq!((m.w, m.h), (w, h), "{} {w}x{h}", d.id);
                assert!(m.px.iter().all(|v| v.is_finite()), "{} p={p} {w}x{h} round {round}: non-finite", d.id);
                let u = uniforms(&e, w as u32, h as u32, p, 1.0).unwrap();
                assert!(u.iter().all(|v| v.is_finite()), "{} uniforms {u:?}", d.id);
                assert!((0.0..=1.0).contains(&u[2]), "{} eased progress {}", d.id, u[2]);
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
