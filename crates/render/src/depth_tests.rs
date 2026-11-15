//! The depth effects' CPU reference on drawn pictures with a stand-in depth model (red is near):
//! near things move and far ones stay, the still depth stays, Zoom to Fill keeps the edges out,
//! a sharp thing in front is not smeared by the blurred ground, hostile values never panic.
//! GPU parity is tested in `filmcraft-gpu` (`depthfx.rs`).

use super::*;
use filmcraft_project::ParamValue;
use filmcraft_project::find_effect;
use filmcraft_time::Tick;

fn cx() -> FxCtx<'static> {
    FxCtx {
        t: Tick(0),
        px_scale: 1.0,
        mask_scale: 1.0,
        seconds: 0.0,
        timecode: "",
        clip_name: "",
        project: None,
        env: None,
        frame_rate: 30.0,
        working: filmcraft_color::WorkingSpace::Rec709,
    }
}

/// The stand-in: a picture's red is its nearness.
struct RedIsNear;

impl DepthModel for RedIsNear {
    fn depth(&self, _: u64, rgba: &[u8], w: usize, h: usize) -> Option<DepthField> {
        DepthField::new(w, h, rgba.as_chunks::<4>().0.iter().map(|p| f32::from(p[0]) / 255.0).collect())
    }
}

fn model() {
    set_model(Arc::new(RedIsNear));
}

const W: usize = 120;
const H: usize = 80;

/// A far ground of green stripes (no red) and a near red square at x 40..60, y 30..50.
fn picture() -> Image {
    let mut img = Image::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let c = if (40..60).contains(&x) && (30..50).contains(&y) { [1.0, 0.0, 0.0] } else { [0.0, 0.2 + 0.1 * (x % 6) as f32, 0.3] };
            img.px[(y * W + x) * 4..(y * W + x) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 1.0]);
        }
    }
    img
}

fn effect(id: &str, params: &[(&str, ParamValue)]) -> EffectInstance {
    let mut e = find_effect(id).unwrap_or_else(|| panic!("{id}")).instance();
    for (k, v) in params {
        e.params.get_mut(*k).unwrap_or_else(|| panic!("{id}.{k}")).value = v.clone();
    }
    e
}

fn run(id: &str, params: &[(&str, ParamValue)]) -> Image {
    model();
    let mut img = picture();
    assert!(apply(&mut img, &effect(id, params), &cx()));
    assert!(img.px.iter().all(|v| v.is_finite()));
    img
}

fn red(img: &Image, x: usize, y: usize) -> f32 {
    img.get(x, y)[0]
}

#[test]
fn the_near_square_moves_and_the_far_ground_stays() {
    use ParamValue::{Bool, Float};
    // the nearest moves 10 % of the width (12 px) right; the farthest stays
    let img = run(PHOTO_3D, &[("travel_x", Float(10.0)), ("pivot", Float(0.0)), ("fill", Bool(false))]);
    let src = picture();
    assert_eq!(img.get(10, 10), src.get(10, 10), "the ground stays");
    assert_eq!(img.get(100, 70), src.get(100, 70));
    for x in 53..72 {
        assert!(red(&img, x, 40) > 0.99, "the square moved to {x}");
    }
    // where it was, the ground shows again (stretched from beside it)
    for x in 40..48 {
        assert!(red(&img, x, 40) < 0.6, "uncovered at {x}: {}", red(&img, x, 40));
    }
    // through the dispatcher too
    model();
    let mut again = picture();
    crate::effects::apply(&mut again, &effect(PHOTO_3D, &[("travel_x", Float(10.0)), ("fill", Bool(false))]), &cx());
    assert_eq!(again.px, img.px);
}

#[test]
fn the_still_depth_stays_and_the_rest_goes_the_other_way() {
    use ParamValue::{Bool, Float};
    // the nearest held: the ground moves 12 px left
    let img = run(PHOTO_3D, &[("travel_x", Float(10.0)), ("pivot", Float(100.0)), ("fill", Bool(false))]);
    let src = picture();
    for x in 41..59 {
        assert!(red(&img, x, 40) > 0.99, "the square stays at {x}");
    }
    for x in [2, 10, 80, 100] {
        assert_eq!(img.get(x, 10), src.get(x + 12, 10), "the ground moved left at {x}");
    }
}

#[test]
fn pushing_in_grows_the_near_square() {
    use ParamValue::{Bool, Float};
    let img = run(PHOTO_3D, &[("push", Float(50.0)), ("pivot", Float(0.0)), ("fill", Bool(false))]);
    // the nearest seen 1 / (1 − push) times as large: the square's left edge (40, held to 39)
    // from the middle (59.5) twice as far, about 19.5
    assert!(red(&img, 22, 40) > 0.99 && red(&picture(), 22, 40) == 0.0);
    assert!(red(&img, 15, 40) < 0.01);
    assert_eq!(img.get(2, 2), picture().get(2, 2), "the farthest stays");
}

#[test]
fn zoom_to_fill_keeps_the_edges_out_of_view() {
    use ParamValue::Float;
    let z = |params: &[(&str, ParamValue)]| Parallax::eval(&effect(PHOTO_3D, params), &cx(), W, H).zoom;
    // across: half the travel either way at the middle depth: 1 / (1 − 0.5 × 0.05 × 2)
    assert!((z(&[("travel_x", Float(5.0)), ("pivot", Float(50.0))]) - 1.0 / 0.95).abs() < 1e-5);
    // pushing in from the farthest only grows the picture: no zoom; pushing out does
    assert!((z(&[("push", Float(20.0)), ("pivot", Float(0.0))]) - 1.0).abs() < 1e-6);
    assert!((z(&[("push", Float(-20.0)), ("pivot", Float(0.0))]) - 1.2).abs() < 1e-5);
    // the keyframes' largest counts, whatever the value now
    let mut e = effect(PHOTO_3D, &[("pivot", Float(50.0))]);
    let p = e.params.get_mut("travel_x").unwrap();
    for (t, v) in [(0, -4.0), (254_016_000_000, 6.0)] {
        p.keyframes.push(filmcraft_project::Keyframe::new(Tick(t), Float(v)));
    }
    assert!((Parallax::eval(&e, &cx(), W, H).zoom - 1.0 / 0.94).abs() < 1e-5);
    // every pixel then looks inside the picture
    model();
    let held = DepthField::new(W, H, (0..W * H).map(|i| if (i % W) < 60 { 1.0 } else { 0.0 }).collect()).unwrap();
    for (tx, ty, push, pivot) in [(0.06, 0.0, 0.0, 0.5), (0.0, -0.05, 0.1, 0.0), (0.04, 0.03, -0.15, 0.3)] {
        let e = effect(
            PHOTO_3D,
            &[("travel_x", Float(tx * 100.0)), ("travel_y", Float(ty * 100.0)), ("push", Float(push * 100.0)), ("pivot", Float(pivot * 100.0))],
        );
        let p = Parallax::eval(&e, &cx(), W, H);
        for y in 0..H {
            for x in 0..W {
                let (sx, sy) = source_of(x as f32, y as f32, W, H, &held, &p);
                assert!((-0.6..=W as f32 - 0.4).contains(&sx) && (-0.6..=H as f32 - 0.4).contains(&sy), "{tx} {ty} {push}: ({x}, {y}) looks at ({sx}, {sy})");
            }
        }
    }
}

#[test]
fn a_sharp_thing_in_front_stays_sharp_over_the_blurred_ground() {
    use ParamValue::{Bool, Float};
    // the near square in focus, the ground 6 px blurred (5 % of 120)
    let img = run(DEPTH_BLUR, &[("focus", Float(100.0)), ("range", Float(10.0)), ("blur", Float(5.0))]);
    let src = picture();
    for x in 40..60 {
        assert_eq!(img.get(x, 40), src.get(x, 40), "the square is untouched at {x}");
    }
    // the stripes smoothed: their spread falls
    let spread = |img: &Image| {
        let v: Vec<f32> = (70..110).map(|x| img.get(x, 12)[1]).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|g| (g - m) * (g - m)).sum::<f32>() / v.len() as f32).sqrt()
    };
    assert!(spread(&img) < spread(&src) * 0.3, "{} against {}", spread(&img), spread(&src));
    // and no red of the sharp square comes onto the ground round it
    for (x, y) in [(38, 40), (36, 40), (61, 35), (50, 28), (50, 52)] {
        assert!(red(&img, x, y) < 1e-6, "red at ({x}, {y}): {}", red(&img, x, y));
    }
    // the ground in focus and the square blurred: its blur reaches onto the ground
    let front = run(DEPTH_BLUR, &[("focus", Float(0.0)), ("blur", Float(5.0))]);
    assert!(red(&front, 37, 40) > 0.05 && red(&front, 10, 10) == 0.0);
    assert_eq!(front.get(10, 10), src.get(10, 10), "the sharp ground stays");
    // with the foreground left sharp, nothing changes
    let kept = run(DEPTH_BLUR, &[("focus", Float(0.0)), ("blur", Float(5.0)), ("near", Bool(false))]);
    assert_eq!(kept.px, src.px);
}

#[test]
fn hostile_values_never_panic() {
    use ParamValue::Float;
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e30, -1e30] {
        for (id, k) in [
            (PHOTO_3D, "travel_x"),
            (PHOTO_3D, "travel_y"),
            (PHOTO_3D, "push"),
            (PHOTO_3D, "pivot"),
            (DEPTH_BLUR, "blur"),
            (DEPTH_BLUR, "focus"),
            (DEPTH_BLUR, "range"),
        ] {
            let img = run(id, &[(k, Float(v))]);
            assert_eq!((img.w, img.h), (W, H));
        }
    }
    assert!(DepthField::new(2, 2, vec![0.0; 3]).is_none() && DepthField::new(0, 0, vec![]).is_none());
    let f = DepthField::new(2, 1, vec![f32::NAN, 2.0]).unwrap();
    assert_eq!(f.near, vec![0.0, 1.0]);
    assert!(taps(f32::NAN).is_empty() && taps(0.2).is_empty());
    let many = taps(1e9);
    assert!(many.len() <= MAX_TAPS && many.len() > 200);
    // a disc of 256 pixels or fewer: every pixel in it, each its own area
    let small = taps(6.0);
    assert!(
        small.iter().all(|t| t.area == 1.0 && t.dist <= 6.0)
            && small.len() == (-6i32..=6).flat_map(|y| (-6i32..=6).map(move |x| (x, y))).filter(|&(x, y)| (x, y) != (0, 0) && x * x + y * y <= 36).count()
    );
    // a tiny picture
    model();
    let mut one = Image::filled(1, 1, [1.0, 0.0, 0.0, 1.0]);
    assert!(apply(&mut one, &effect(PHOTO_3D, &[("travel_x", Float(10.0))]), &cx()));
    let mut blank = Image::new(3, 2);
    assert!(apply(&mut blank, &effect(DEPTH_BLUR, &[("blur", Float(10.0))]), &cx()));
    assert!(blank.px.iter().all(|v| v.is_finite()));
}

#[test]
fn outlines_are_held_and_maps_read_between_samples() {
    let f = DepthField::new(5, 1, vec![0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert_eq!(f.held(1).near, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    assert_eq!(f.held(0), f);
    assert!((f.at(0.375, 0.0) - 0.5).abs() < 1e-6);
    assert_eq!(f.at(f32::NAN, f32::INFINITY), 1.0);
    // the same pixels, the same key; a pixel changed, another
    let a = vec![1u8, 2, 3, 4];
    assert_eq!(picture_key(&a, 1, 1), picture_key(&a, 1, 1));
    assert_ne!(picture_key(&a, 1, 1), picture_key(&[1, 2, 3, 5], 1, 1));
    assert_ne!(picture_key(&a, 1, 1), picture_key(&a, 2, 1));
}

/// How long each takes on the CPU at 1080p (printed).
#[test]
fn speed_at_1080p() {
    use ParamValue::Float;
    model();
    let (w, h) = (1920, 1080);
    let img = Image::filled(w, h, [0.5, 0.2, 0.1, 1.0]);
    let held = DepthField::new(64, 36, (0..64 * 36).map(|i| (i % 64) as f32 / 63.0).collect()).unwrap();
    let p = Parallax::eval(&effect(PHOTO_3D, &[("travel_x", Float(3.0)), ("pivot", Float(50.0))]), &cx(), w, h);
    let t = std::time::Instant::now();
    let _ = parallax_cpu(&img, &held, &p);
    let par = t.elapsed();
    let op = BlurOp::eval(&effect(DEPTH_BLUR, &[("focus", Float(100.0)), ("blur", Float(1.5))]), &cx(), w);
    let nr = near_and_radius(&held, &op, w, h);
    let t = std::time::Instant::now();
    let _ = blur_cpu(&img, &nr, &taps(op.radius));
    eprintln!(
        "CPU at 1080p: 3D photo {:.0} ms, depth blur {:.0} ms ({} taps)",
        par.as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64() * 1e3,
        taps(op.radius).len()
    );
}
