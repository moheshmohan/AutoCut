//! Cheap picture measures of a shot: sharpness, exposure, the camera's motion and shake, and the
//! overall score that ranks shots.
//!
//! - **Sharpness**: the variance of the Laplacian of the luma (on a frame scaled to
//!   [`MEASURE_SIDE`] px), the usual blur measure — a sharp picture has strong second
//!   derivatives at its edges, a blurred one has none — mapped to 0..1 on a log scale.
//! - **Exposure**: the mean luma and the shares of crushed blacks and blown whites.
//! - **Motion**: the global shift between consecutive analysis frames (exhaustive block matching
//!   of the whole small luma, mean removed so a change of light is not motion). Its steady part is
//!   the camera's move (a pan or tilt); its jitter about a moving average is shake.

use filmcraft_project::scenes::{CameraMove, ShotQuality};

/// Long side of the picture sharpness and exposure are measured on.
pub const MEASURE_SIDE: usize = 512;
/// Long side of the frames motion is measured on.
pub const MOTION_SIDE: usize = 96;

/// A luma picture, values 0..255.
#[derive(Clone, Debug, PartialEq)]
pub struct Luma {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f32>,
}

/// The luma (BT.709 weights) of an RGBA8 picture, area-averaged down so its long side is at most
/// `max_side`. `None` for an empty or short buffer.
pub fn luma(rgba: &[u8], w: usize, h: usize, max_side: usize) -> Option<Luma> {
    let need = w.checked_mul(h)?.checked_mul(4)?;
    if w == 0 || h == 0 || rgba.len() < need || max_side == 0 {
        return None;
    }
    let long = w.max(h);
    let (nw, nh) = if long <= max_side { (w, h) } else { ((w * max_side / long).max(1), (h * max_side / long).max(1)) };
    let mut v = vec![0f32; nw * nh];
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h / nh).max(y * h / nh + 1).min(h));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w / nw).max(x * w / nw + 1).min(w));
            let (mut acc, mut n) = (0f32, 0u32);
            for sy in y0..y1 {
                let row = sy * w;
                for sx in x0..x1 {
                    let i = (row + sx) * 4;
                    if let Some(p) = rgba.get(i..i + 3) {
                        acc += 0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]);
                        n += 1;
                    }
                }
            }
            if let Some(d) = v.get_mut(y * nw + x) {
                *d = acc / n.max(1) as f32;
            }
        }
    }
    Some(Luma { w: nw, h: nh, v })
}

/// The variance of the 4-neighbour Laplacian over the interior (0 for pictures under 3×3).
pub fn laplacian_variance(l: &Luma) -> f32 {
    if l.w < 3 || l.h < 3 || l.w.checked_mul(l.h).is_none_or(|area| l.v.len() < area) {
        return 0.0;
    }
    let (mut s, mut s2, mut n) = (0f64, 0f64, 0f64);
    // each interior pixel with its neighbours: the row above, its own three, the row below
    let rows = || l.v.chunks_exact(l.w).take(l.h);
    for ((up, row), down) in rows().zip(rows().skip(1)).zip(rows().skip(2)) {
        for ((three, u), d) in row.windows(3).zip(up.iter().skip(1)).zip(down.iter().skip(1)) {
            let &[left, c, right] = three else { continue };
            let lap = f64::from(left + right + u + d - 4.0 * c);
            s += lap;
            s2 += lap * lap;
            n += 1.0;
        }
    }
    if n == 0.0 {
        return 0.0;
    }
    let mean = s / n;
    ((s2 / n - mean * mean).max(0.0)) as f32
}

/// Sharpness 0..1 from the variance of the Laplacian: 20 or less is soft, 1000 or more crisp
/// (on a [`MEASURE_SIDE`] frame), log scale between.
pub fn sharpness_score(var: f32) -> f32 {
    if !var.is_finite() || var <= 0.0 {
        return 0.0;
    }
    ((var.log10() - 20f32.log10()) / (1000f32.log10() - 20f32.log10())).clamp(0.0, 1.0)
}

/// Exposure measures: (mean luma 0..1, share crushed, share blown, score 0..1).
pub fn exposure(l: &Luma) -> (f32, f32, f32, f32) {
    if l.v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = l.v.len() as f32;
    let mean = l.v.iter().sum::<f32>() / n / 255.0;
    let dark = l.v.iter().filter(|v| **v <= 0.02 * 255.0).count() as f32 / n;
    let blown = l.v.iter().filter(|v| **v >= 0.98 * 255.0).count() as f32 / n;
    let off = ((mean - 0.45).abs() / 0.45).min(1.0);
    let score = (1.0 - off.powf(1.5)) - 1.5 * (dark - 0.05).max(0.0) - 2.0 * (blown - 0.02).max(0.0);
    let ok = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    (ok(mean), ok(dark), ok(blown), ok(score))
}

/// The shift `(dx, dy)` (pixels) that best maps `a` onto `b`: what is at `(x, y)` in `a` is at
/// `(x + dx, y + dy)` in `b`. Integer search within ±`range`, then a parabolic sub-pixel step.
/// `None` when the pictures differ in size or are too small.
pub fn global_shift(a: &Luma, b: &Luma, range: usize) -> Option<(f32, f32)> {
    if a.w != b.w || a.h != b.h || a.w < 2 * range + 4 || a.h < 2 * range + 4 || a.v.len() < a.w * a.h || b.v.len() < b.w * b.h {
        return None;
    }
    let r = range as isize;
    let mean = |l: &Luma| l.v.iter().sum::<f32>() / l.v.len().max(1) as f32;
    let (ma, mb) = (mean(a), mean(b));
    let side = 2 * range + 1;
    let mut cost = vec![f32::INFINITY; side * side];
    let (w, h) = (a.w as isize, a.h as isize);
    for dy in -r..=r {
        for dx in -r..=r {
            // the overlap: x in a such that x + dx is in b
            let (x0, x1) = ((-dx).max(0), (w - dx).min(w));
            let (y0, y1) = ((-dy).max(0), (h - dy).min(h));
            let (mut sad, mut n) = (0f32, 0f32);
            let len = usize::try_from(x1 - x0).unwrap_or(0);
            for y in y0..y1 {
                // the overlapping parts of row y of a and row y + dy of b
                let (Ok(ra), Ok(rb)) = (usize::try_from(y * w + x0), usize::try_from((y + dy) * w + x0 + dx)) else { continue };
                let (Some(row_a), Some(row_b)) = (a.v.get(ra..ra + len), b.v.get(rb..rb + len)) else { continue };
                for (va, vb) in row_a.iter().zip(row_b) {
                    sad += ((va - ma) - (vb - mb)).abs();
                    n += 1.0;
                }
            }
            let at = usize::try_from((dy + r) * side as isize + dx + r).ok();
            if n > 0.0
                && let Some(c) = at.and_then(|i| cost.get_mut(i))
            {
                *c = sad / n;
            }
        }
    }
    let best = cost.iter().enumerate().filter(|(_, c)| c.is_finite()).min_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i)?;
    let (bx, by) = (best % side, best / side);
    // sub-pixel: a parabola through the best cost and its neighbours on each axis
    let refine = |l: Option<f32>, c: f32, r: Option<f32>| -> f32 {
        match (l, r) {
            (Some(l), Some(r)) if l.is_finite() && r.is_finite() => {
                let d = l - 2.0 * c + r;
                if d > 1e-6 { (0.5 * (l - r) / d).clamp(-0.5, 0.5) } else { 0.0 }
            }
            _ => 0.0,
        }
    };
    let c = *cost.get(best)?;
    let fx = refine(if bx > 0 { cost.get(best - 1).copied() } else { None }, c, if bx + 1 < side { cost.get(best + 1).copied() } else { None });
    let fy = refine(if by > 0 { cost.get(best - side).copied() } else { None }, c, cost.get(best + side).copied().filter(|_| by + 1 < side));
    Some((bx as f32 - r as f32 + fx, by as f32 - r as f32 + fy))
}

/// The camera's move in a shot from the frame-to-frame shifts of its analysis frames (pixels at
/// `width`, frames `fps` a second): (the move, mean motion in frame widths a second, shake 0..1).
pub fn camera_motion(shifts: &[(f32, f32)], fps: f64, width: usize) -> (CameraMove, f32, f32) {
    let s: Vec<(f32, f32)> = shifts.iter().copied().filter(|(x, y)| x.is_finite() && y.is_finite()).collect();
    if s.is_empty() || width == 0 || !fps.is_finite() || fps <= 0.0 {
        return (CameraMove::Static, 0.0, 0.0);
    }
    let n = s.len() as f32;
    let w = width as f32;
    let fps = fps as f32;
    let (mx, my) = (s.iter().map(|v| v.0).sum::<f32>() / n, s.iter().map(|v| v.1).sum::<f32>() / n);
    let speed = s.iter().map(|(x, y)| (x * x + y * y).sqrt()).sum::<f32>() / n * fps / w;
    // jitter about a 5-frame moving average
    let mut jitter = 0f32;
    for (i, (x, y)) in s.iter().enumerate() {
        let near = s.get(i.saturating_sub(2)..(i + 3).min(s.len())).unwrap_or_default();
        let k = near.len().max(1) as f32;
        let (ax, ay) = (near.iter().map(|v| v.0).sum::<f32>() / k, near.iter().map(|v| v.1).sum::<f32>() / k);
        jitter += ((x - ax).powi(2) + (y - ay).powi(2)).sqrt();
    }
    jitter /= n;
    // a jitter of 1.2 % of the frame width each frame is as shaky as it gets
    let shake = (jitter / (0.012 * w)).clamp(0.0, 1.0);
    let steady = (mx * mx + my * my).sqrt() * fps / w;
    let mv = if steady < 0.03 {
        if shake >= 0.4 { CameraMove::Handheld } else { CameraMove::Static }
    } else if mx.abs() > 2.0 * my.abs() {
        // the picture moving left is the camera turning right
        if mx < 0.0 { CameraMove::PanRight } else { CameraMove::PanLeft }
    } else if my.abs() > 2.0 * mx.abs() {
        // the picture moving down is the camera tilting up
        if my > 0.0 { CameraMove::TiltUp } else { CameraMove::TiltDown }
    } else {
        CameraMove::Moving
    };
    let speed = if speed.is_finite() { speed.clamp(0.0, 100.0) } else { 0.0 };
    (mv, speed, shake)
}

/// The overall score, 0..1: sharpness, exposure and steadiness, and the aesthetic score (about
/// 1..10, mapped from 3.5..6.5) when there is one.
pub fn overall(q: &ShotQuality) -> f32 {
    let base = 0.4 * q.sharpness + 0.3 * q.exposure + 0.3 * (1.0 - q.shake);
    let s = match q.aesthetic {
        Some(a) if a.is_finite() => 0.75 * base + 0.25 * ((a - 3.5) / 3.0).clamp(0.0, 1.0),
        _ => base,
    };
    if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 }
}

/// Sharpness and exposure of one keyframe (RGBA8), into `q` (averaged over `n` keyframes so far).
pub fn measure_keyframe(q: &mut ShotQuality, rgba: &[u8], w: usize, h: usize, n: usize) {
    let Some(l) = luma(rgba, w, h, MEASURE_SIDE) else { return };
    let var = laplacian_variance(&l);
    let (mean, dark, blown, score) = exposure(&l);
    let k = n as f32;
    let avg = |old: f32, new: f32| (old * k + new) / (k + 1.0);
    q.laplacian = avg(q.laplacian, var);
    q.sharpness = avg(q.sharpness, sharpness_score(var));
    q.brightness = avg(q.brightness, mean);
    q.dark = avg(q.dark, dark);
    q.blown = avg(q.blown, blown);
    q.exposure = avg(q.exposure, score);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured test card: checks, stripes and noise, `w`×`h`, shifted by `(sx, sy)` pixels.
    fn card(w: usize, h: usize, sx: f32, sy: f32, gain: f32) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - sx, y as f32 - sy);
                let checks = if ((fx / 7.0).floor() as i32 + (fy / 7.0).floor() as i32) % 2 == 0 { 200.0 } else { 40.0 };
                let wave = 30.0 * (fx * 0.31).sin() * (fy * 0.17).cos();
                let l = ((checks + wave) * gain).clamp(0.0, 255.0) as u8;
                let i = (y * w + x) * 4;
                v[i..i + 4].copy_from_slice(&[l, l, l, 255]);
            }
        }
        v
    }

    fn blur(rgba: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
        let mut out = rgba.to_vec();
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut n) = (0u32, 0u32);
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                        acc += u32::from(rgba[(yy * w + xx) * 4]);
                        n += 1;
                    }
                }
                let v = (acc / n) as u8;
                let i = (y * w + x) * 4;
                out[i..i + 3].copy_from_slice(&[v, v, v]);
            }
        }
        out
    }

    #[test]
    fn blur_lowers_sharpness() {
        let (w, h) = (320, 180);
        let sharp = card(w, h, 0.0, 0.0, 1.0);
        let soft = blur(&sharp, w, h, 6);
        let s = laplacian_variance(&luma(&sharp, w, h, 512).unwrap());
        let b = laplacian_variance(&luma(&soft, w, h, 512).unwrap());
        assert!(s > 10.0 * b, "sharp {s} vs blurred {b}");
        assert!(sharpness_score(s) > 0.8, "{}", sharpness_score(s));
        assert!(sharpness_score(b) < 0.5, "{}", sharpness_score(b));
        // a flat picture has no sharpness at all
        let flat = vec![128u8; w * h * 4];
        assert_eq!(laplacian_variance(&luma(&flat, w, h, 512).unwrap()), 0.0);
        assert_eq!(sharpness_score(0.0), 0.0);
        assert_eq!(sharpness_score(f32::NAN), 0.0);
    }

    #[test]
    fn exposure_sees_dark_and_blown_pictures() {
        let (w, h) = (160, 90);
        let good = luma(&card(w, h, 0.0, 0.0, 1.0), w, h, 512).unwrap();
        let dark = luma(&card(w, h, 0.0, 0.0, 0.05), w, h, 512).unwrap();
        let blown = luma(&card(w, h, 0.0, 0.0, 4.0), w, h, 512).unwrap();
        let (gm, _, _, gs) = exposure(&good);
        let (dm, dd, _, ds) = exposure(&dark);
        let (bm, _, bb, bs) = exposure(&blown);
        assert!(gs > 0.7 && (0.3..0.7).contains(&gm), "good {gm} {gs}");
        assert!(ds < 0.3 && dm < 0.1 && dd > 0.3, "dark {dm} {dd} {ds}");
        assert!(bs < 0.3 && bm > 0.8 && bb > 0.4, "blown {bm} {bb} {bs}");
        assert_eq!(exposure(&Luma { w: 0, h: 0, v: vec![] }), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn shifts_are_measured() {
        let (w, h) = (96, 54);
        let a = luma(&card(w, h, 0.0, 0.0, 1.0), w, h, MOTION_SIDE).unwrap();
        for (sx, sy) in [(2.0f32, 0.0f32), (-3.0, 1.0), (0.0, -2.0), (1.5, 0.5)] {
            // the light changes too: the mean is removed
            let b = luma(&card(w, h, sx, sy, 1.1), w, h, MOTION_SIDE).unwrap();
            let (dx, dy) = global_shift(&a, &b, 4).unwrap();
            assert!((dx - sx).abs() < 0.6 && (dy - sy).abs() < 0.6, "({sx}, {sy}) measured ({dx}, {dy})");
        }
        assert!(global_shift(&a, &Luma { w: 10, h: 10, v: vec![0.0; 100] }, 4).is_none());
    }

    #[test]
    fn camera_moves_from_shifts() {
        let fps = 30.0;
        // steady
        assert_eq!(camera_motion(&vec![(0.0, 0.0); 30], fps, 96).0, CameraMove::Static);
        // the picture slides left: the camera pans right
        let (m, speed, shake) = camera_motion(&vec![(-1.0, 0.0); 30], fps, 96);
        assert_eq!(m, CameraMove::PanRight);
        assert!((speed - 30.0 / 96.0).abs() < 1e-3 && shake < 0.05, "{speed} {shake}");
        assert_eq!(camera_motion(&vec![(1.0, 0.1); 30], fps, 96).0, CameraMove::PanLeft);
        assert_eq!(camera_motion(&vec![(0.0, 1.0); 30], fps, 96).0, CameraMove::TiltUp);
        assert_eq!(camera_motion(&vec![(0.0, -1.0); 30], fps, 96).0, CameraMove::TiltDown);
        assert_eq!(camera_motion(&vec![(1.0, 1.0); 30], fps, 96).0, CameraMove::Moving);
        // jitter with no direction: hand-held and shaky
        let jitter: Vec<(f32, f32)> = (0..60).map(|i| if i % 2 == 0 { (1.5, -1.0) } else { (-1.5, 1.0) }).collect();
        let (m, _, shake) = camera_motion(&jitter, fps, 96);
        assert_eq!(m, CameraMove::Handheld);
        assert!(shake > 0.8, "{shake}");
        // hostile input
        assert_eq!(camera_motion(&[(f32::NAN, 1.0)], fps, 96), (CameraMove::Static, 0.0, 0.0));
        assert_eq!(camera_motion(&[(1.0, 1.0)], 0.0, 96), (CameraMove::Static, 0.0, 0.0));
        assert_eq!(camera_motion(&[], fps, 96), (CameraMove::Static, 0.0, 0.0));
    }

    #[test]
    fn overall_score_weighs_everything() {
        let good = ShotQuality { sharpness: 1.0, exposure: 1.0, shake: 0.0, ..Default::default() };
        assert!((overall(&good) - 1.0).abs() < 1e-6);
        let bad = ShotQuality { sharpness: 0.1, exposure: 0.2, shake: 0.9, ..Default::default() };
        assert!(overall(&bad) < 0.3);
        let pretty = ShotQuality { aesthetic: Some(6.5), ..good.clone() };
        let plain = ShotQuality { aesthetic: Some(3.0), ..good };
        assert!(overall(&pretty) > overall(&plain));
        assert_eq!(overall(&ShotQuality { sharpness: f32::NAN, ..Default::default() }), 0.0);
    }

    #[test]
    fn keyframes_average_in() {
        let (w, h) = (200, 100);
        let mut q = ShotQuality::default();
        measure_keyframe(&mut q, &card(w, h, 0.0, 0.0, 1.0), w, h, 0);
        let first = q.sharpness;
        measure_keyframe(&mut q, &vec![128u8; w * h * 4], w, h, 1);
        assert!((q.sharpness - first / 2.0).abs() < 1e-5);
        // a short buffer changes nothing
        let before = q.clone();
        measure_keyframe(&mut q, &[1, 2, 3], w, h, 2);
        assert_eq!(q, before);
        assert!(luma(&[0u8; 16], usize::MAX, 2, 64).is_none());
    }
}
