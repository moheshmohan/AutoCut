use super::*;

/// A 1920×1080 picture into a 1080×1920 frame.
const G: Geometry = Geometry { picture: (1920.0, 1080.0), frame: (1080.0, 1920.0) };

fn sample(t: f64, cx: f64, cy: f64) -> Sample {
    Sample { t, bbox: SubjectBox::new(cx, cy, 0.12, 0.4), confidence: 0.9, cut: false }
}

fn times(s: &[Sample]) -> Vec<f64> {
    s.iter().map(|x| x.t).collect()
}

/// The picture covers the frame at placement `(position, scale)`.
fn covers(g: &Geometry, (p, s): ([f64; 2], f64)) -> bool {
    let (hw, hh) = (g.picture.0 * s / 2.0, g.picture.1 * s / 2.0);
    p[0] - hw <= 1e-6 && p[0] + hw >= g.frame.0 - 1e-6 && p[1] - hh <= 1e-6 && p[1] + hh >= g.frame.1 - 1e-6
}

/// The subject's centre `c` (0..1 of the picture) in the frame at a placement.
fn on_screen(g: &Geometry, (p, s): ([f64; 2], f64), c: [f64; 2]) -> [f64; 2] {
    [p[0] + (c[0] - 0.5) * g.picture.0 * s, p[1] + (c[1] - 0.5) * g.picture.1 * s]
}

#[test]
fn parses_aspects_and_sizes_frames() {
    assert_eq!(parse_aspect("9:16"), Some((9, 16)));
    assert_eq!(parse_aspect(" 4x5 "), Some((4, 5)));
    assert_eq!(parse_aspect("16/9"), Some((16, 9)));
    for bad in ["", "9", "0:16", "9:0", "1:9", "-9:16", "9:16:1", "a:b", "1000:1"] {
        assert_eq!(parse_aspect(bad), None, "{bad}");
    }
    assert_eq!(frame_for((1920, 1080), (9, 16)), Some((1080, 1920)));
    assert_eq!(frame_for((1920, 1080), (1, 1)), Some((1080, 1080)));
    assert_eq!(frame_for((1920, 1080), (4, 5)), Some((1080, 1350)));
    assert_eq!(frame_for((1080, 1920), (16, 9)), Some((1920, 1080)));
    // hostile sizes stay sane
    let (w, h) = frame_for((0, u32::MAX), (9, 16)).unwrap();
    assert!((16..=8192).contains(&w) && (16..=8192).contains(&h) && w % 2 == 0 && h % 2 == 0);
}

#[test]
fn the_spring_settles_without_overshoot_and_without_lag() {
    let t: Vec<f64> = (0..300).map(|k| k as f64 / 30.0).collect();
    // a step: monotonic, inside the input range
    let step: Vec<f64> = t.iter().map(|x| if *x < 5.0 { 0.0 } else { 1.0 }).collect();
    let y = follow(&step, &t, Follow::Smooth.omega());
    assert!(y.iter().all(|v| (-1e-9..=1.0 + 1e-9).contains(v)), "no overshoot");
    assert!(y.windows(2).all(|w| w[1] >= w[0] - 1e-12), "monotonic");
    // zero phase: the step is half done at the step
    let mid = y[150];
    assert!((mid - 0.5).abs() < 0.05, "half way at the step: {mid}");
    // a ramp comes out where it went in (no lag)
    let ramp: Vec<f64> = t.iter().map(|x| x * 0.1).collect();
    let r = follow(&ramp, &t, Follow::Smooth.omega());
    // ... the ends included: the springs settle in the padding
    for k in 0..300 {
        assert!((r[k] - ramp[k]).abs() < 1e-3, "lag at {k}: {} vs {}", r[k], ramp[k]);
    }
}

#[test]
fn a_moving_subject_gives_a_smooth_path_that_stays_in_the_picture() {
    // walking from the left third to the right edge and back, with detector jitter
    let n = 300;
    let samples: Vec<Sample> = (0..n)
        .map(|k| {
            let t = k as f64 / 30.0;
            let jitter = ((k * 7919) % 13) as f64 / 13.0 - 0.5;
            sample(t, 0.3 + 0.65 * (t / 10.0 * std::f64::consts::PI).sin() + 0.01 * jitter, 0.5 + 0.005 * jitter)
        })
        .collect();
    for follow_mode in [Follow::Smooth, Follow::Responsive] {
        let plan = plan(&samples, G, follow_mode, Zoom::Track, Fallback::Letterbox);
        assert!(!plan.letterbox && plan.animate_position, "{follow_mode:?}");
        assert!(plan.keys.len() < n / 4, "{follow_mode:?}: {} keys for {n} frames", plan.keys.len());
        let t = times(&samples);
        let placed: Vec<([f64; 2], f64)> = (0..n).map(|i| placement_at(&plan, &t, i).unwrap()).collect();
        for (i, p) in placed.iter().enumerate() {
            assert!(covers(&G, *p), "{follow_mode:?}: frame {i} shows past the picture: {p:?}");
        }
        // smooth: no frame-to-frame acceleration bigger than a few pixels (the raw jitter alone is
        // ~19 px of picture at this scale)
        let accel = placed.windows(3).map(|w| (w[2].0[0] - 2.0 * w[1].0[0] + w[0].0[0]).abs()).fold(0.0, f64::max);
        assert!(accel < 3.0, "{follow_mode:?}: acceleration {accel:.2} px/frame²");
        // the subject stays in the frame, near the middle while it can
        for (i, p) in placed.iter().enumerate().skip(30).take(200) {
            let c = samples[i].bbox.unwrap();
            let x = on_screen(&G, *p, [c.cx, c.cy])[0];
            assert!(x > 0.0 && x < G.frame.0, "{follow_mode:?}: frame {i}: subject at {x:.0} px");
        }
    }
}

#[test]
fn a_still_subject_is_one_static_placement() {
    let samples: Vec<Sample> = (0..90).map(|k| sample(k as f64 / 30.0, 0.62, 0.5)).collect();
    let plan = plan(&samples, G, Follow::Smooth, Zoom::Track, Fallback::Letterbox);
    assert_eq!(plan.keys.len(), 1, "{:?}", plan.keys);
    assert!(!plan.animate_position && !plan.animate_scale);
    // centred on the subject
    let x = on_screen(&G, (plan.keys[0].position, plan.keys[0].scale), [0.62, 0.5])[0];
    assert!((x - G.frame.0 / 2.0).abs() < 2.0, "{x}");
}

#[test]
fn a_cut_jumps_instead_of_panning() {
    let mut samples: Vec<Sample> = (0..60).map(|k| sample(k as f64 / 30.0, 0.25, 0.5)).collect();
    for (k, s) in samples.iter_mut().enumerate().skip(30) {
        *s = sample(k as f64 / 30.0, 0.8, 0.5);
    }
    samples[30].cut = true;
    let plan = plan(&samples, G, Follow::Smooth, Zoom::Track, Fallback::Letterbox);
    assert_eq!(plan.shots, 2);
    let t = times(&samples);
    let before = placement_at(&plan, &t, 29).unwrap();
    let after = placement_at(&plan, &t, 30).unwrap();
    // each shot is centred on its subject from its first frame to its last
    for (i, c) in [(0, 0.25), (29, 0.25), (30, 0.8), (59, 0.8)] {
        let x = on_screen(&G, placement_at(&plan, &t, i).unwrap(), [c, 0.5])[0];
        assert!((x - G.frame.0 / 2.0).abs() < 3.0, "frame {i}: subject at {x:.1}");
    }
    assert!((before.0[0] - after.0[0]).abs() > 500.0, "the crop jumps at the cut");
    assert!(plan.keys.iter().any(|k| k.hold), "the first shot's last key holds");
}

#[test]
fn a_clip_without_a_subject_is_letterboxed_or_centred() {
    let samples: Vec<Sample> = (0..60).map(|k| Sample { t: k as f64 / 30.0, bbox: None, confidence: 0.0, cut: false }).collect();
    let p = plan(&samples, G, Follow::Smooth, Zoom::Track, Fallback::Letterbox);
    assert!(p.letterbox);
    assert_eq!(p.keys.len(), 1);
    assert!((p.keys[0].scale - G.fit_scale()).abs() < 1e-9);
    assert_eq!(p.keys[0].position, [540.0, 960.0]);
    let c = plan(&samples, G, Follow::Smooth, Zoom::Track, Fallback::Center);
    assert!(c.letterbox && (c.keys[0].scale - G.fill_scale()).abs() < 1e-9);
    // seen in a few frames only, unsure in the rest: still the fallback
    let mut few = samples.clone();
    for s in few.iter_mut().take(10) {
        *s = sample(s.t, 0.4, 0.5);
    }
    assert!(plan(&few, G, Follow::Smooth, Zoom::Track, Fallback::Letterbox).letterbox);
}

#[test]
fn gaps_are_bridged() {
    // seen, lost for a second, seen further right: the crop eases across, it does not snap back
    let samples: Vec<Sample> = (0..90)
        .map(|k| {
            let t = k as f64 / 30.0;
            if (30..60).contains(&k) {
                Sample { t, bbox: SubjectBox::new(0.1, 0.1, 0.1, 0.1), confidence: 0.05, cut: false }
            } else {
                sample(t, if k < 30 { 0.3 } else { 0.7 }, 0.5)
            }
        })
        .collect();
    let p = plan(&samples, G, Follow::Responsive, Zoom::Track, Fallback::Letterbox);
    let t = times(&samples);
    let xs: Vec<f64> = (0..90).map(|i| placement_at(&p, &t, i).unwrap().0[0]).collect();
    // monotonic across the gap (the picture moves left as the crop moves right)
    assert!(xs.windows(2).all(|w| w[1] <= w[0] + 0.5), "{xs:?}");
}

#[test]
fn a_small_subject_is_pushed_in_on_within_limits() {
    let samples: Vec<Sample> =
        (0..90).map(|k| Sample { t: k as f64 / 30.0, bbox: SubjectBox::new(0.5, 0.5, 0.04, 0.08), confidence: 0.9, cut: false }).collect();
    for f in [Follow::Smooth, Follow::Responsive] {
        let p = plan(&samples, G, f, Zoom::Track, Fallback::Letterbox);
        let s = p.keys.iter().map(|k| k.scale).fold(0.0, f64::max);
        assert!(s > G.fill_scale() * 1.05 && s <= G.fill_scale() * f.max_zoom() + 1e-9, "{f:?}: {s}");
    }
}

#[test]
fn kept_scales_only_move_the_position() {
    // a push-in from fill to 6 % larger, the subject drifting right
    let n = 120;
    let s0 = G.fill_scale();
    let scales: Vec<f64> = (0..n).map(|k| s0 * (1.0 + 0.06 * k as f64 / n as f64)).collect();
    let samples: Vec<Sample> = (0..n).map(|k| sample(k as f64 / 30.0, 0.4 + 0.5 * k as f64 / n as f64, 0.5)).collect();
    let p = plan(&samples, G, Follow::Smooth, Zoom::Keep(&scales), Fallback::Letterbox);
    assert!(p.animate_position && !p.letterbox);
    let t = times(&samples);
    for i in 0..n {
        let (pos, _) = placement_at(&p, &t, i).unwrap();
        // inside the picture at any scale the push-in passes through
        for s in [s0, scales[i], scales[n - 1]] {
            assert!(covers(&G, (pos, s)), "frame {i}: {pos:?} at {s}");
        }
    }
}

#[test]
fn hostile_samples_never_panic() {
    let nan = Sample { t: f64::NAN, bbox: SubjectBox::new(0.5, 0.5, 0.1, 0.1), confidence: f32::NAN, cut: true };
    let inf = Sample { t: f64::INFINITY, bbox: SubjectBox::new(0.5, 0.5, 0.1, 0.1), confidence: 1.0, cut: false };
    let cases: Vec<Vec<Sample>> = vec![
        vec![],
        vec![nan],
        vec![inf, nan, inf],
        vec![sample(0.0, 0.5, 0.5); 3],
        (0..50).map(|k| Sample { cut: k % 2 == 0, ..sample(1.0, 0.5, 0.5) }).collect(),
    ];
    let geometries = [
        G,
        Geometry { picture: (0.0, 0.0), frame: (1080.0, 1920.0) },
        Geometry { picture: (f64::NAN, 1.0), frame: (1.0, f64::INFINITY) },
        Geometry { picture: (1e9, 1.0), frame: (1.0, 1e9) },
    ];
    for s in &cases {
        for g in geometries {
            for z in [Zoom::Track, Zoom::Keep(&[f64::NAN, -1.0, 0.0])] {
                let p = plan(s, g, Follow::Responsive, z, Fallback::Center);
                assert!(!p.keys.is_empty());
                for i in 0..s.len() + 2 {
                    let _ = placement_at(&p, &times(s), i);
                }
            }
        }
    }
    assert!(simplify(&[], &[], 1.0).is_empty());
    assert!(fill(&[None, None], &[0.0, 1.0]).is_none());
    assert_eq!(window(&[], &[]), Vec::<f64>::new());
}
