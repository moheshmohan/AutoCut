use super::*;
use filmcraft_geom::Vec2;

const W: usize = 384;
const H: usize = 216;

fn hash(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Bilinear value noise.
fn noise(u: f64, v: f64, seed: u64) -> f32 {
    let (x0, y0) = (u.floor(), v.floor());
    let (fx, fy) = ((u - x0) as f32, (v - y0) as f32);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let a = hash(x0, y0, seed);
    let b = hash(x0 + 1, y0, seed);
    let c = hash(x0, y0 + 1, seed);
    let d = hash(x0 + 1, y0 + 1, seed);
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

fn texture(u: f64, v: f64, seed: u64) -> f32 {
    0.5 * noise(u / 9.0, v / 9.0, seed) + 0.3 * noise(u / 4.0, v / 4.0, seed + 1) + 0.2 * noise(u / 1.7, v / 1.7, seed + 2)
}

#[derive(Clone, Copy)]
struct Scene {
    bg: u64,
    /// A green background instead of the blue one (another shot).
    green: bool,
    /// Camera pan: the background's offset.
    pan: (f64, f64),
    /// Subject centre and radius, pixels.
    subject: Option<(f64, f64, f64)>,
    /// An occluding bar `[x0, y0, x1, y1]`.
    occluder: Option<[f64; 4]>,
}

impl Scene {
    fn at(subject: (f64, f64, f64)) -> Self {
        Scene { bg: 7, green: false, pan: (0.0, 0.0), subject: Some(subject), occluder: None }
    }
}

/// An RGBA8 frame: a cool textured background, a warm textured disc on it (its texture moves and
/// scales with it), and a grey striped occluder over both.
fn render(s: &Scene, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let b = texture(px + s.pan.0 + 1000.0, py + s.pan.1 + 500.0, s.bg);
            let mut c = if s.green { [0.12 + 0.2 * b, 0.4 + 0.35 * b, 0.15 + 0.2 * b] } else { [0.25 + 0.3 * b, 0.35 + 0.35 * b, 0.4 + 0.4 * b] };
            if let Some((cx, cy, r)) = s.subject {
                let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                let a = (r - d + 0.5).clamp(0.0, 1.0) as f32;
                if a > 0.0 {
                    let k = 40.0 / r.max(1.0);
                    let t = texture((px - cx) * k + 300.0, (py - cy) * k + 200.0, 99);
                    let sc = [0.55 + 0.45 * t, 0.25 + 0.4 * t, 0.15 + 0.2 * t];
                    for (o, v) in c.iter_mut().zip(sc) {
                        *o = *o * (1.0 - a) + v * a;
                    }
                }
            }
            if let Some(o) = s.occluder
                && px >= o[0]
                && px < o[2]
                && py >= o[1]
                && py < o[3]
            {
                let stripe = if ((px / 6.0) as i64) % 2 == 0 { 0.35 } else { 0.55 };
                c = [stripe, stripe, stripe];
            }
            let i = (y * w + x) * 4;
            for (k, v) in c.iter().enumerate() {
                out[i + k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            out[i + 3] = 255;
        }
    }
    out
}

fn truth_box(s: (f64, f64, f64)) -> SubjectBox {
    SubjectBox::new(s.0 / W as f64, s.1 / H as f64, 2.0 * s.2 / W as f64, 2.0 * s.2 / H as f64).unwrap()
}

/// Centre error in pixels of the analysis frame.
fn err(b: &SubjectBox, s: (f64, f64, f64)) -> f64 {
    ((b.cx * W as f64 - s.0).powi(2) + (b.cy * H as f64 - s.1).powi(2)).sqrt()
}

/// Run `n` frames of `scene(k)`, seeding with the true box at frame 0; returns the observations.
fn run(n: usize, scene: impl Fn(usize) -> Scene) -> Vec<Observation> {
    let mut tr = SubjectTracker::new();
    (0..n)
        .map(|k| {
            let s = scene(k);
            let hint = if k == 0 { s.subject.map(truth_box) } else { None };
            tr.push(&render(&s, W, H), W, H, k as f64 / 30.0, hint)
        })
        .collect()
}

#[test]
fn follows_a_moving_subject_within_three_pixels() {
    let path = |k: usize| {
        let t = k as f64;
        (70.0 + 2.6 * t, 108.0 + 30.0 * (t / 14.0).sin(), 34.0)
    };
    let obs = run(90, |k| Scene::at(path(k)));
    let worst = obs.iter().enumerate().map(|(k, o)| err(&o.bbox.unwrap(), path(k))).fold(0.0, f64::max);
    assert!(worst < 3.0, "worst centre error {worst:.2} px");
    assert!(obs.iter().skip(1).all(|o| o.state == TrackState::Tracked), "{:?}", obs.iter().map(|o| o.state).collect::<Vec<_>>());
    assert!(obs.iter().all(|o| o.confidence > 0.5 && !o.cut));
}

#[test]
fn follows_a_subject_that_grows_and_pans_with_the_camera() {
    // the camera pans (the background moves) while the subject comes closer
    let path = |k: usize| {
        let t = k as f64;
        (120.0 + 1.5 * t, 100.0 + 0.4 * t, 26.0 + 0.45 * t)
    };
    let obs = run(60, |k| Scene { pan: (k as f64 * 1.8, 0.0), ..Scene::at(path(k)) });
    let mut worst = 0.0f64;
    let mut worst_size = 0.0f64;
    for (k, o) in obs.iter().enumerate() {
        let b = o.bbox.unwrap();
        worst = worst.max(err(&b, path(k)));
        let size = b.w * W as f64 / 2.0;
        worst_size = worst_size.max((size / path(k).2 - 1.0).abs());
    }
    assert!(worst < 3.0, "worst centre error {worst:.2} px");
    assert!(worst_size < 0.15, "worst size error {:.0}%", worst_size * 100.0);
}

#[test]
fn reacquires_the_subject_after_an_occluder_passes() {
    let path = |k: usize| (90.0 + 2.0 * k as f64, 110.0, 30.0);
    // a bar sweeps right to left across the subject around frames 30..45
    let bar = |k: usize| {
        let x = 330.0 - 9.0 * (k as f64 - 22.0);
        (k >= 22).then_some([x, 0.0, x + 70.0, H as f64])
    };
    let obs = run(80, |k| Scene { occluder: bar(k), ..Scene::at(path(k)) });
    let states: Vec<TrackState> = obs.iter().map(|o| o.state).collect();
    assert!(states.iter().any(|s| matches!(s, TrackState::Coasting | TrackState::Redetected)), "the occlusion was noticed: {states:?}");
    for (k, o) in obs.iter().enumerate() {
        let e = err(&o.bbox.unwrap(), path(k));
        // held near where it was while hidden, and back on it once the bar has gone (by frame 60)
        let budget = if (25..60).contains(&k) { 40.0 } else { 4.0 };
        assert!(e < budget, "frame {k} ({:?}): centre error {e:.1} px", o.state);
    }
}

#[test]
fn finds_the_same_subject_again_after_a_cut() {
    let a = |k: usize| (100.0 + 1.5 * k as f64, 100.0, 30.0);
    let b = |k: usize| (280.0 - 1.0 * (k as f64 - 40.0), 140.0, 30.0);
    let obs = run(70, |k| if k < 40 { Scene::at(a(k)) } else { Scene { bg: 31, green: true, ..Scene::at(b(k)) } });
    assert!(obs[40].cut, "the cut is seen");
    assert_eq!(obs.iter().filter(|o| o.cut).count(), 1, "one cut only");
    assert_eq!(obs[40].state, TrackState::Redetected);
    for (k, o) in obs.iter().enumerate() {
        let e = err(&o.bbox.unwrap(), if k < 40 { a(k) } else { b(k) });
        assert!(e < 4.0, "frame {k} ({:?}): centre error {e:.1} px", o.state);
    }
}

#[test]
fn a_hint_after_a_cut_to_a_new_subject_reseeds() {
    let mut tr = SubjectTracker::new();
    let first = Scene::at((100.0, 100.0, 30.0));
    tr.push(&render(&first, W, H), W, H, 0.0, Some(truth_box((100.0, 100.0, 30.0))));
    tr.push(&render(&first, W, H), W, H, 0.033, None);
    // a different shot: no disc, a bright square where the hint says the subject is
    let mut img = render(&Scene { bg: 55, green: true, subject: None, ..first }, W, H);
    for y in 150..190 {
        for x in 40..80 {
            let i = (y * W + x) * 4;
            img[i..i + 3].copy_from_slice(&[240, 230, 40]);
        }
    }
    let hint = SubjectBox::new(60.0 / W as f64, 170.0 / H as f64, 40.0 / W as f64, 40.0 / H as f64);
    let o = tr.push(&img, W, H, 0.066, hint);
    assert!(o.cut);
    let b = o.bbox.unwrap();
    assert!((b.cx * W as f64 - 60.0).abs() < 3.0 && (b.cy * H as f64 - 170.0).abs() < 3.0, "{b:?} ({:?})", o.state);
}

#[test]
fn seeds_on_the_salient_region_without_a_hint() {
    // a dull, even background and a bright, colourful disc off-centre
    let mut img = vec![0u8; W * H * 4];
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) * 4;
            let g = 90 + (hash(x as i64, y as i64, 3) * 12.0) as u8;
            let inside = ((x as f64 - 250.0).powi(2) + (y as f64 - 90.0).powi(2)).sqrt() < 28.0;
            let c = if inside { [230, 60, 40] } else { [g, g, g + 5] };
            img[i..i + 3].copy_from_slice(&c);
            img[i + 3] = 255;
        }
    }
    let b = salient_box(&img, W, H).expect("something stands out");
    assert!((b.cx * W as f64 - 250.0).abs() < 25.0 && (b.cy * H as f64 - 90.0).abs() < 20.0, "{b:?}");
    let mut tr = SubjectTracker::new();
    let o = tr.push(&img, W, H, 0.0, None);
    assert_eq!(o.state, TrackState::Seeded);
    assert!(o.confidence <= 0.5, "a saliency seed is only a guess");
}

#[test]
fn a_flat_frame_has_no_subject() {
    let img = vec![128u8; W * H * 4];
    assert!(salient_box(&img, W, H).is_none());
    let mut tr = SubjectTracker::new();
    let o = tr.push(&img, W, H, 0.0, None);
    assert_eq!((o.bbox, o.state), (None, TrackState::None));
}

#[test]
fn hostile_inputs_never_panic() {
    let img = render(&Scene::at((100.0, 100.0, 30.0)), W, H);
    let mut tr = SubjectTracker::new();
    tr.push(&img, W, H, 0.0, None);
    // short buffers, zero and odd sizes, huge sizes, NaN times and boxes
    for (buf, w, h) in [(&img[..10], W, H), (&img[..], 0, 0), (&img[..], 1, H), (&img[..], usize::MAX, 2), (&img[..], 8, 8), (&img[..], W * 2, H)] {
        let o = tr.push(buf, w, h, f64::NAN, None);
        assert!(o.confidence.is_finite());
    }
    assert!(SubjectBox::new(f64::NAN, 0.5, 0.1, 0.1).is_none());
    assert!(SubjectBox::new(0.5, 0.5, -1.0, 0.1).is_none());
    assert!(SubjectBox::from_area([0.5, 0.5], f64::INFINITY, 1.0).is_none());
    let b = SubjectBox::new(7.0, -3.0, 50.0, 1e-9).unwrap();
    assert!((0.0..=1.0).contains(&b.cx) && (0.0..=1.0).contains(&b.cy) && b.w <= 1.0 && b.h >= MIN_BOX);
    // a box at the very edge of the frame, and a whole-frame box
    for hint in [SubjectBox::new(0.0, 0.0, 0.02, 0.02), SubjectBox::new(1.0, 1.0, 1.0, 1.0), SubjectBox::new(0.5, 0.5, 1.0, 1.0)] {
        let mut tr = SubjectTracker::new();
        for k in 0..4 {
            let o = tr.push(&img, W, H, k as f64, if k == 0 { hint } else { None });
            assert!(o.bbox.is_some_and(|b| b.cx.is_finite() && b.w > 0.0));
        }
    }
    // frames of different sizes in a row: a cut, not a crash
    let small = render(&Scene::at((50.0, 50.0, 20.0)), 160, 90);
    let o = tr.push(&small, 160, 90, 10.0, None);
    assert!(o.cut);
    assert!(salient_box(&img[..100], W, H).is_none());
    assert!(downscale_rgba(&img[..100], W, H, 64).is_none());
    assert!(downscale_rgba(&img, usize::MAX, usize::MAX, 64).is_none());
}

#[test]
fn downscales_by_area_average() {
    let w = 1920;
    let h = 1080;
    let mut img = vec![0u8; w * h * 4];
    for (i, p) in img.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let v = if (i % w) < w / 2 { 0 } else { 200 };
        p.copy_from_slice(&[v, v, v, 255]);
    }
    let (out, nw, nh) = downscale_rgba(&img, w, h, 384).unwrap();
    assert_eq!((nw, nh), (384, 216));
    assert_eq!(out.len(), nw * nh * 4);
    assert_eq!(out[0], 0);
    assert_eq!(out[(nw - 1) * 4], 200);
    // small frames come back as they are
    let (same, sw, sh) = downscale_rgba(&out, nw, nh, 400).unwrap();
    assert_eq!((same.len(), sw, sh), (out.len(), nw, nh));
}

/// Tracking speed on 1080p frames (downscaled to the analysis size, as the engine does): the
/// frames are made once and reused, so this measures the tracker, not the test's renderer.
#[test]
fn tracks_a_1080p_minute_quickly() {
    let (w, h) = (1920usize, 1080usize);
    let frames: Vec<Vec<u8>> = (0..8)
        .map(|k| {
            let s = Scene::at((500.0 + 30.0 * k as f64, 520.0, 160.0));
            // a 1080p frame from the 384-wide render, nearest neighbour (cheap to make)
            let small = render(&Scene { subject: s.subject.map(|(x, y, r)| (x / 5.0, y / 5.0, r / 5.0)), ..s }, W, H);
            let mut big = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = ((x * W / w).min(W - 1), (y * H / h).min(H - 1));
                    let (i, j) = ((y * w + x) * 4, (sy * W + sx) * 4);
                    big[i..i + 4].copy_from_slice(&small[j..j + 4]);
                }
            }
            big
        })
        .collect();
    let n = if cfg!(debug_assertions) { 16 } else { 150 };
    let mut tr = SubjectTracker::new();
    let (mut down, mut track) = (0.0f64, 0.0f64);
    for k in 0..n {
        let f = &frames[(k / 4) % frames.len()];
        let t0 = std::time::Instant::now();
        let (small, sw, sh) = downscale_rgba(f, w, h, 384).unwrap();
        let t1 = std::time::Instant::now();
        let hint = (k == 0).then(|| SubjectBox::new(500.0 / w as f64, 520.0 / h as f64, 320.0 / w as f64, 320.0 / h as f64)).flatten();
        tr.push(&small, sw, sh, k as f64 / 30.0, hint);
        down += (t1 - t0).as_secs_f64();
        track += t1.elapsed().as_secs_f64();
    }
    let per_frame = (down + track) / n as f64;
    let minute = per_frame * 1800.0;
    eprintln!(
        "subject tracking: {:.1} ms a 1080p frame ({:.1} ms downscale + {:.1} ms track) → {minute:.1} s for a minute at 30 fps",
        per_frame * 1000.0,
        down * 1000.0 / n as f64,
        track * 1000.0 / n as f64
    );
    if !cfg!(debug_assertions) {
        assert!(minute < 120.0, "a 1080p minute took {minute:.0} s to track");
    }
}

/// The tracker's own pyramid and corner finder give what the mask tracker's do (they only skip
/// work): the same pyramid levels, the same strongest corners.
#[test]
fn fast_pyramid_and_corners_match_the_mask_tracker() {
    let img = render(&Scene::at((200.0, 108.0, 32.0)), W, H);
    let fast = pyramid(Gray::from_rgba8(W, H, &img));
    let slow = Prepared::new(Gray::from_rgba8(W, H, &img));
    assert_eq!(fast.levels.len(), slow.levels.len());
    for (a, b) in fast.levels.iter().zip(&slow.levels) {
        assert_eq!((a.w, a.h), (b.w, b.h));
        assert!(a.px.iter().zip(&b.px).all(|(u, v)| (u - v).abs() < 1e-5));
    }
    let p = truth_box((200.0, 108.0, 32.0)).px(W, H);
    let poly = [Vec2::new(p[0], p[1]), Vec2::new(p[2], p[1]), Vec2::new(p[2], p[3]), Vec2::new(p[0], p[3])];
    let ours = corners(fast.levels.first().unwrap(), p, MAX_POINTS, 4.0);
    let theirs = crate::track::features(slow.levels.first().unwrap(), &poly, MAX_POINTS, 4.0);
    assert!(ours.len() >= 20, "{}", ours.len());
    let same = theirs.iter().take(10).filter(|q| ours.contains(q)).count();
    assert!(same >= 8, "{same} of the 10 strongest corners agree");
}
