//! Mask coverage (CPU reference; `filmcraft_gpu::GpuMask` runs the same math in WGSL).
//!
//! Each [`Mask`] is flattened to a polygon in working-image pixels (chord error ≤ 0.05 px). For
//! a pixel centre `p` the signed distance `sd` to the polygon (positive inside, nonzero winding)
//! gives the coverage
//!
//! ```text
//! s  = sd + expansion
//! w  = max(feather, 1)
//! u  = clamp(s / w + 0.5, 0, 1)
//! c  = lerp(u, smoothstep(u), min(feather, 1))
//! m  = opacity · (inverted ? 1 − c : c)
//! ```
//!
//! so a hard mask (feather 0) is the analytic box-filtered coverage of the edge (a one-pixel
//! linear ramp across the boundary: exact for straight edges, antialiased everywhere), and a
//! feathered mask falls off smoothly over `feather` pixels centred on the (expanded) edge — a true
//! distance falloff, so corners round off and the width is the same along the whole contour.
//! Masks combine top to bottom with their [`MaskMode`]. All of this runs in linear light on the
//! premultiplied working image: a masked effect is `lerp(original, effected, coverage)` per
//! channel, an opacity mask scales the layer's premultiplied pixels.

use filmcraft_project::{EffectInstance, Mask, MaskMode};
use filmcraft_time::Tick;
use rayon::prelude::*;

use crate::image::Image;

/// Flattening tolerance in working pixels.
pub const FLATTEN_TOLERANCE: f64 = 0.05;

/// One mask ready for rasterizing, in working-image pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatMask {
    /// Closed polygon (last vertex connects to the first).
    pub pts: Vec<[f32; 2]>,
    pub feather: f32,
    pub expansion: f32,
    /// 0..1.
    pub opacity: f32,
    pub inverted: bool,
    pub mode: MaskMode,
}

/// Flatten the active masks (mode ≠ None) at clip time `t`. `px_scale` = working pixels per clip pixel.
pub fn prepare(masks: &[Mask], t: Tick, px_scale: f32) -> Vec<FlatMask> {
    let s = px_scale as f64;
    masks
        .iter()
        .filter(|m| m.mode != MaskMode::None)
        .map(|m| {
            let path = m.path_at(t);
            let pts = path.flatten(FLATTEN_TOLERANCE / s.max(1e-6)).into_iter().map(|p| [(p.x * s) as f32, (p.y * s) as f32]).collect();
            FlatMask {
                pts,
                feather: (m.feather.f64_at(t).max(0.0) * s) as f32,
                expansion: (m.expansion.f64_at(t) * s) as f32,
                opacity: (m.opacity.f64_at(t) / 100.0).clamp(0.0, 1.0) as f32,
                inverted: m.inverted,
                mode: m.mode,
            }
        })
        .collect()
}

/// The falloff of a signed (expanded) distance for a feather width.
#[inline]
pub fn falloff(s: f32, feather: f32) -> f32 {
    let w = feather.max(1.0);
    let u = (s / w + 0.5).clamp(0.0, 1.0);
    let smooth = u * u * (3.0 - 2.0 * u);
    u + (smooth - u) * feather.min(1.0)
}

/// Distance band beyond which coverage is saturated (0 or 1).
fn band(m: &FlatMask) -> f32 {
    m.expansion.abs() + m.feather.max(1.0) * 0.5 + 1.0
}

#[inline]
fn seg_dist2(px: f32, py: f32, a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (qx, qy) = (px - a[0], py - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { ((qx * dx + qy * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let (ex, ey) = (qx - dx * t, qy - dy * t);
    ex * ex + ey * ey
}

/// One mask's coverage of a row ([`mask_row`]): `outside` everywhere except the pixels of `span`,
/// whose values were written to the row buffer.
struct RowCoverage {
    outside: f32,
    span: std::ops::Range<usize>,
}

/// Coverage of one mask for row `y` (per-row algorithm: winding from the row's sorted edge
/// crossings, distance only against edges within the band). Only the pixels an edge's band reaches
/// are worked out one by one, into `out` (length = width); every other pixel of the row has the
/// same value, the one a pixel with no edge within the band has (it is outside: left of every
/// crossing, or right of all of them, where a closed polygon's winding is back to zero). A face on
/// a redacted clip covers a few thousand of its two million pixels, and the pixel loop used to
/// visit every one for every mask: two fifths of a redacted 4K export's CPU. Each value is
/// computed exactly as before, so the coverage is the same to the bit; a row whose crossings fall
/// outside the edges' band (a non-finite path or band) takes the whole-row loop as before.
fn mask_row(m: &FlatMask, y: usize, out: &mut [f32]) -> RowCoverage {
    let w = out.len();
    let n = m.pts.len();
    let py = y as f32 + 0.5;
    let b = band(m);
    let value = |sd: f32| {
        let c = falloff(sd + m.expansion, m.feather);
        m.opacity * if m.inverted { 1.0 - c } else { c }
    };
    if n < 3 {
        return RowCoverage { outside: value(-b), span: 0..0 };
    }
    // crossings of the scanline with the polygon edges (x, direction)
    let mut cross: Vec<(f32, i32)> = Vec::new();
    // edges within the band of this row: (a, b, min x, max x)
    let mut near: Vec<([f32; 2], [f32; 2], f32, f32)> = Vec::new();
    for i in 0..n {
        let (Some(&a), Some(&c)) = (m.pts.get(i), m.pts.get((i + 1) % n)) else { continue };
        if (a[1] <= py) != (c[1] <= py) {
            let t = (py - a[1]) / (c[1] - a[1]);
            cross.push((a[0] + (c[0] - a[0]) * t, if c[1] > a[1] { 1 } else { -1 }));
        }
        if py >= a[1].min(c[1]) - b && py <= a[1].max(c[1]) + b {
            near.push((a, c, a[0].min(c[0]) - b, a[0].max(c[0]) + b));
        }
    }
    cross.sort_by(|p, q| p.0.total_cmp(&q.0));
    let b2 = b * b;
    let far = b2.sqrt();
    let outside = value(-far);
    // where an edge's band reaches along the row (pixels beyond it are `outside`)
    let (lo, hi) = near.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), e| (lo.min(e.2), hi.max(e.3)));
    let bounded = b.is_finite() && cross.iter().all(|c| c.0 >= lo && c.0 <= hi);
    let span = if !bounded {
        0..w
    } else if near.is_empty() {
        0..0
    } else {
        let mut s = ((lo - 0.5).floor().max(0.0) as usize).min(w);
        while s > 0 && (s - 1) as f32 + 0.5 >= lo {
            s -= 1;
        }
        while s < w && (s as f32 + 0.5) < lo {
            s += 1;
        }
        let mut e = ((hi + 0.5).floor().max(0.0) as usize).max(s).min(w);
        while e < w && e as f32 + 0.5 <= hi {
            e += 1;
        }
        while e > s && (e - 1) as f32 + 0.5 > hi {
            e -= 1;
        }
        s..e
    };
    let mut ci = 0;
    let mut wind = 0;
    for (x, o) in out.iter_mut().enumerate().take(span.end).skip(span.start) {
        let px = x as f32 + 0.5;
        while let Some(c) = cross.get(ci).filter(|c| c.0 < px) {
            wind += c.1;
            ci += 1;
        }
        let inside = wind != 0;
        let mut d2 = b2;
        for (a, c, x0, x1) in &near {
            if px >= *x0 && px <= *x1 {
                d2 = d2.min(seg_dist2(px, py, *a, *c));
            }
        }
        let d = d2.sqrt();
        *o = value(if inside { d } else { -d });
    }
    RowCoverage { outside, span }
}

/// Whether mixing coverage `v` into the accumulated coverage with `mode` (and clamping) leaves it
/// as it is, to the bit: subtracting nothing or intersecting with everything always does; adding
/// nothing or taking the difference with nothing does for coverage that is never −0.0 or NaN
/// (`plain`: no mask's value can be one, see [`coverage`]).
fn leaves_unchanged(mode: MaskMode, v: f32, plain: bool) -> bool {
    match mode {
        MaskMode::Subtract => v.to_bits() == 0,
        MaskMode::Intersect => v.to_bits() == 1.0f32.to_bits(),
        MaskMode::Add | MaskMode::Difference => plain && v.to_bits() == 0,
        _ => false,
    }
}

/// Combined coverage of `masks` over a `w`×`h` image, or `None` when there is no active mask
/// (the effect applies everywhere).
pub fn coverage(masks: &[FlatMask], w: usize, h: usize) -> Option<Vec<f32>> {
    let first = masks.iter().find(|m| m.mode != MaskMode::None)?;
    let start = first.mode.start();
    let mut acc = vec![start; w * h];
    if w == 0 {
        return Some(acc);
    }
    // A mask's values are opacity × a falloff in 0..1 (never −0.0 or NaN for a finite band): with
    // every opacity a positive finite number, no coverage is ever −0.0 or NaN.
    let plain = masks.iter().all(|m| m.opacity.is_finite() && m.opacity.is_sign_positive() && m.expansion.is_finite() && m.feather.is_finite());
    acc.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0.0f32; w],
        |tmp, (y, row)| {
            for m in masks.iter().filter(|m| m.mode != MaskMode::None) {
                let RowCoverage { outside, span } = mask_row(m, y, tmp);
                let mix = |a: &mut f32, v: f32| *a = m.mode.combine(*a, v).clamp(0.0, 1.0);
                let (Some(within), Some(cov)) = (row.get_mut(span.clone()), tmp.get(span.clone())) else { continue };
                within.iter_mut().zip(cov).for_each(|(a, v)| mix(a, *v));
                // the rest of the row: one value, which mostly changes nothing (a face's mask
                // seen from the rest of the frame)
                if !leaves_unchanged(m.mode, outside, plain) {
                    let (before, after) = row.split_at_mut(span.start);
                    before.iter_mut().chain(after.iter_mut().skip(span.len())).for_each(|a| mix(a, outside));
                }
            }
        },
    );
    Some(acc)
}

/// Coverage of an effect's masks on a working image (`px_scale` working px per clip px).
pub fn effect_coverage(masks: &[Mask], t: Tick, px_scale: f32, w: usize, h: usize) -> Option<Vec<f32>> {
    if masks.is_empty() {
        return None;
    }
    coverage(&prepare(masks, t, px_scale), w, h)
}

/// `img = lerp(original, img, cov)` on premultiplied pixels.
pub fn mix(img: &mut Image, original: &Image, cov: &[f32]) {
    img.px.par_chunks_mut(4).zip(original.px.par_chunks(4)).zip(cov.par_iter()).for_each(|((p, o), &c)| {
        for k in 0..4 {
            p[k] = o[k] + (p[k] - o[k]) * c;
        }
    });
}

/// Scale premultiplied pixels by coverage (opacity masks).
pub fn scale_by(img: &mut Image, cov: &[f32]) {
    img.px.par_chunks_mut(4).zip(cov.par_iter()).for_each(|(p, &c)| {
        for v in p {
            *v *= c;
        }
    });
}

/// Apply an effect limited to its masks (Premiere: a masked effect only applies inside the mask).
pub fn apply_effect(img: &mut Image, e: &EffectInstance, cx: &crate::effects::FxCtx) {
    if !e.enabled || e.masks.is_empty() {
        crate::effects::apply(img, e, cx);
        return;
    }
    let Some(cov) = effect_coverage(&e.masks, cx.t, cx.px_scale, img.w, img.h) else {
        crate::effects::apply(img, e, cx);
        return;
    };
    let original = img.clone();
    crate::effects::apply(img, e, cx);
    mix(img, &original, &cov);
}

/// Apply the masks of an item's Opacity effect to its layer (before Motion), if any.
pub fn apply_opacity_masks(img: &mut Image, item: &filmcraft_project::TrackItem, t: Tick, px_scale: f32) {
    let Some(op) = item.effect("opacity").filter(|e| e.enabled && !e.masks.is_empty()) else { return };
    if let Some(cov) = effect_coverage(&op.masks, t, px_scale, img.w, img.h) {
        scale_by(img, &cov);
    }
}

/// Brute-force coverage of one pixel (every edge; the definition the row algorithm and the WGSL
/// shader implement). For tests and documentation.
pub fn coverage_at(masks: &[FlatMask], x: f32, y: f32) -> Option<f32> {
    let first = masks.iter().find(|m| m.mode != MaskMode::None)?;
    let mut acc = first.mode.start();
    for m in masks.iter().filter(|m| m.mode != MaskMode::None) {
        let n = m.pts.len();
        let mut wind = 0;
        let mut d2 = f32::INFINITY;
        for i in 0..n {
            let a = m.pts[i];
            let c = m.pts[(i + 1) % n];
            if (a[1] <= y) != (c[1] <= y) {
                let t = (y - a[1]) / (c[1] - a[1]);
                if a[0] + (c[0] - a[0]) * t < x {
                    wind += if c[1] > a[1] { 1 } else { -1 };
                }
            }
            d2 = d2.min(seg_dist2(x, y, a, c));
        }
        let d = if n < 3 { band(m) } else { d2.sqrt().min(band(m)) };
        let sd = if wind != 0 && n >= 3 { d } else { -d };
        let c = falloff(sd + m.expansion, m.feather);
        let v = m.opacity * if m.inverted { 1.0 - c } else { c };
        acc = m.mode.combine(acc, v).clamp(0.0, 1.0);
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_geom::Vec2;
    use filmcraft_project::MaskPath;

    fn mask(path: MaskPath, feather: f64) -> Mask {
        let mut m = Mask::new("m", path);
        m.feather.value = filmcraft_project::ParamValue::Float(feather);
        m
    }

    #[test]
    fn hard_rect_is_exact_box_coverage() {
        // rect edges on pixel boundaries → exactly 1 inside, 0 outside; half-pixel edge → 0.5
        let m = mask(MaskPath::rect(2.0, 2.0, 6.5, 6.0), 0.0);
        let c = effect_coverage(&[m], Tick(0), 1.0, 10, 10).unwrap();
        assert_eq!(c[3 * 10 + 3], 1.0);
        assert_eq!(c[3 * 10 + 1], 0.0);
        assert!((c[3 * 10 + 6] - 0.5).abs() < 1e-6, "{}", c[3 * 10 + 6]);
        assert_eq!(c[8 * 10 + 3], 0.0);
    }

    #[test]
    fn ellipse_area_matches_analytic() {
        let (rx, ry) = (37.3, 21.7);
        let m = mask(MaskPath::ellipse(Vec2::new(50.2, 40.1), Vec2::new(rx, ry)), 0.0);
        let c = effect_coverage(&[m], Tick(0), 1.0, 100, 80).unwrap();
        let area: f64 = c.iter().map(|v| *v as f64).sum();
        let exact = std::f64::consts::PI * rx * ry;
        assert!((area - exact).abs() / exact < 0.002, "{area} vs {exact}");
    }

    #[test]
    fn feather_is_a_symmetric_distance_falloff() {
        let m = mask(MaskPath::rect(20.0, -100.0, 200.0, 200.0), 10.0);
        let c = effect_coverage(&[m], Tick(0), 1.0, 60, 4).unwrap();
        // edge at x = 20: 0.5 there, saturated 5 px away, symmetric
        let at = |x: usize| c[x];
        assert!((at(19) + at(20) - 1.0).abs() < 1e-5, "{} {}", at(19), at(20));
        assert!(at(13) < 1e-6 && at(26) > 1.0 - 1e-6, "{} {}", at(13), at(26));
        for x in 14..26 {
            assert!(at(x + 1) >= at(x), "monotonic");
        }
    }

    #[test]
    fn expansion_inverted_opacity_modes() {
        let mut m = mask(MaskPath::rect(10.0, 10.0, 20.0, 20.0), 0.0);
        m.expansion.value = filmcraft_project::ParamValue::Float(5.0);
        let c = effect_coverage(std::slice::from_ref(&m), Tick(0), 1.0, 30, 30).unwrap();
        assert_eq!(c[15 * 30 + 6], 1.0, "expanded by 5");
        assert_eq!(c[15 * 30 + 4], 0.0);
        m.inverted = true;
        m.opacity.value = filmcraft_project::ParamValue::Float(50.0);
        let c = effect_coverage(&[m.clone()], Tick(0), 1.0, 30, 30).unwrap();
        assert_eq!(c[15 * 30 + 15], 0.0);
        assert_eq!(c[2], 0.5);
        // subtract a hole from a big rect
        let mut big = mask(MaskPath::rect(0.0, 0.0, 30.0, 30.0), 0.0);
        big.mode = MaskMode::Add;
        let mut hole = mask(MaskPath::rect(10.0, 10.0, 20.0, 20.0), 0.0);
        hole.mode = MaskMode::Subtract;
        let c = effect_coverage(&[big, hole], Tick(0), 1.0, 30, 30).unwrap();
        assert_eq!(c[15 * 30 + 15], 0.0);
        assert_eq!(c[5 * 30 + 5], 1.0);
    }

    /// [`coverage`] as it was: every pixel of every row through the per-pixel loop.
    fn coverage_every_pixel(masks: &[FlatMask], w: usize, h: usize) -> Option<Vec<f32>> {
        let first = masks.iter().find(|m| m.mode != MaskMode::None)?;
        let mut acc = vec![first.mode.start(); w * h];
        for y in 0..h {
            for m in masks.iter().filter(|m| m.mode != MaskMode::None) {
                let n = m.pts.len();
                let py = y as f32 + 0.5;
                let b = band(m);
                let value = |sd: f32| {
                    let c = falloff(sd + m.expansion, m.feather);
                    m.opacity * if m.inverted { 1.0 - c } else { c }
                };
                let mut tmp = vec![0.0f32; w];
                if n < 3 {
                    tmp.fill(value(-b));
                } else {
                    let mut cross: Vec<(f32, i32)> = Vec::new();
                    let mut near: Vec<([f32; 2], [f32; 2], f32, f32)> = Vec::new();
                    for i in 0..n {
                        let a = m.pts[i];
                        let c = m.pts[(i + 1) % n];
                        if (a[1] <= py) != (c[1] <= py) {
                            let t = (py - a[1]) / (c[1] - a[1]);
                            cross.push((a[0] + (c[0] - a[0]) * t, if c[1] > a[1] { 1 } else { -1 }));
                        }
                        if py >= a[1].min(c[1]) - b && py <= a[1].max(c[1]) + b {
                            near.push((a, c, a[0].min(c[0]) - b, a[0].max(c[0]) + b));
                        }
                    }
                    cross.sort_by(|p, q| p.0.total_cmp(&q.0));
                    let (mut ci, mut wind, b2) = (0, 0, b * b);
                    for (x, o) in tmp.iter_mut().enumerate() {
                        let px = x as f32 + 0.5;
                        while ci < cross.len() && cross[ci].0 < px {
                            wind += cross[ci].1;
                            ci += 1;
                        }
                        let mut d2 = b2;
                        for (a, c, x0, x1) in &near {
                            if px >= *x0 && px <= *x1 {
                                d2 = d2.min(seg_dist2(px, py, *a, *c));
                            }
                        }
                        let d = d2.sqrt();
                        *o = value(if wind != 0 { d } else { -d });
                    }
                }
                for (a, v) in acc[y * w..(y + 1) * w].iter_mut().zip(&tmp) {
                    *a = m.mode.combine(*a, *v).clamp(0.0, 1.0);
                }
            }
        }
        Some(acc)
    }

    /// Working out only the pixels near a mask's edges gives every pixel the coverage the
    /// whole-row loop gave, to the bit: ellipses, polygons and rectangles, inside, across and
    /// outside the frame, hard and feathered, expanded and contracted, inverted, every mode, any
    /// opacity, several masks at once, degenerate paths, tiny and odd widths.
    #[test]
    fn coverage_near_the_edges_matches_every_pixel_to_the_bit() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move |lo: f64, hi: f64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            lo + (hi - lo) * ((seed >> 11) as f64 / (1u64 << 53) as f64)
        };
        let modes = [MaskMode::Add, MaskMode::Subtract, MaskMode::Intersect, MaskMode::Lighten, MaskMode::Darken, MaskMode::Difference, MaskMode::None];
        for case in 0..400 {
            let (w, h) = ([1usize, 2, 3, 17, 40, 64, 97][case % 7], [1usize, 5, 31, 50][case % 4]);
            let count = 1 + case % 4;
            let mut masks = Vec::new();
            for k in 0..count {
                let (cx, cy) = (rnd(-20.0, w as f64 + 20.0), rnd(-20.0, h as f64 + 20.0));
                let path = match (case + k) % 5 {
                    0 | 1 => MaskPath::ellipse(Vec2::new(cx, cy), Vec2::new(rnd(0.3, 30.0), rnd(0.3, 30.0))),
                    2 => MaskPath::rect(cx, cy, cx + rnd(0.0, 25.0), cy + rnd(0.0, 25.0)),
                    3 => MaskPath::polygon(&[
                        Vec2::new(cx, cy),
                        Vec2::new(cx + rnd(-30.0, 30.0), cy + rnd(-30.0, 30.0)),
                        Vec2::new(cx + rnd(-30.0, 30.0), cy + rnd(-30.0, 30.0)),
                        Vec2::new(cx + rnd(-30.0, 30.0), cy + rnd(-30.0, 30.0)),
                    ]),
                    _ => MaskPath::polygon(&[Vec2::new(cx, cy), Vec2::new(cx + rnd(-30.0, 30.0), cy)]),
                };
                let mut m = mask(path, if case % 3 == 0 { 0.0 } else { rnd(0.0, 20.0) });
                m.expansion.value = filmcraft_project::ParamValue::Float(if case % 5 == 0 { 0.0 } else { rnd(-6.0, 6.0) });
                m.opacity.value = filmcraft_project::ParamValue::Float([100.0, 0.0, 37.5, rnd(0.0, 100.0)][(case + k) % 4]);
                m.inverted = (case + k) % 6 == 1;
                m.mode = modes[(case * 3 + k) % modes.len()];
                masks.push(m);
            }
            let flat = prepare(&masks, Tick(0), [1.0, 0.8, 1.7][case % 3]);
            let (old, new) = (coverage_every_pixel(&flat, w, h), coverage(&flat, w, h));
            match (old, new) {
                (Some(o), Some(n)) => {
                    for (i, (a, b)) in o.iter().zip(&n).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(), "case {case}: pixel {i} of {w}x{h}: {a} vs {b}");
                    }
                }
                (o, n) => assert_eq!(o.is_some(), n.is_some(), "case {case}"),
            }
        }
    }

    /// Hostile masks (non-finite paths, expansions, opacities) give what the whole-row loop gave
    /// and never panic.
    #[test]
    fn hostile_masks_cover_as_before() {
        let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 1e30];
        for (i, v) in bad.iter().enumerate() {
            for field in 0..4 {
                let mut m = FlatMask {
                    pts: vec![[3.0, 2.0], [20.0, 4.0], [11.0, 18.0], [2.0, 9.0]],
                    feather: 4.0,
                    expansion: 1.0,
                    opacity: 1.0,
                    inverted: i % 2 == 0,
                    mode: MaskMode::Add,
                };
                match field {
                    0 => m.pts[1] = [*v, 4.0],
                    1 => m.pts[2] = [11.0, *v],
                    2 => m.expansion = *v,
                    _ => m.opacity = *v,
                }
                let other = FlatMask {
                    pts: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                    feather: 0.0,
                    expansion: 0.0,
                    opacity: 0.5,
                    inverted: false,
                    mode: MaskMode::Difference,
                };
                let masks = [m, other];
                let (o, n) = (coverage_every_pixel(&masks, 24, 20).unwrap(), coverage(&masks, 24, 20).unwrap());
                for (a, b) in o.iter().zip(&n) {
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) || (*a == 0.0 && *b == 0.0), "{v} in field {field}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn row_algorithm_matches_brute_force() {
        let mut a = mask(MaskPath::ellipse(Vec2::new(30.0, 25.0), Vec2::new(20.0, 12.0)), 6.0);
        a.expansion.value = filmcraft_project::ParamValue::Float(-2.0);
        let mut b = mask(MaskPath::polygon(&[Vec2::new(5.0, 5.0), Vec2::new(50.0, 10.0), Vec2::new(20.0, 45.0)]), 0.0);
        b.mode = MaskMode::Difference;
        b.inverted = true;
        let flat = prepare(&[a, b], Tick(0), 0.8);
        let (w, h) = (48, 40);
        let c = coverage(&flat, w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let r = coverage_at(&flat, x as f32 + 0.5, y as f32 + 0.5).unwrap();
                assert!((r - c[y * w + x]).abs() < 1e-5, "({x},{y}) {r} vs {}", c[y * w + x]);
            }
        }
    }
}
