//! Optical-flow stitching along the seam of a dual-fisheye picture (the CPU reference; the GPU
//! compositor warps with the same shifts in WGSL, `composite.wgsl`'s `vr_fisheye`).
//!
//! Where both lenses of a dual-fisheye camera see — a band a few degrees either side of the great
//! circle half way between their axes — they see from two points a few centimetres apart. Far
//! things agree; a hand or a face within a metre or so is seen a few degrees apart across the
//! seam (parallax: the lenses' distance apart over the thing's distance), and a cross-fade shows
//! it twice.
//!
//! [`SeamFlow::measure`] finds, all round the seam, how far lens 1's picture of the band is
//! shifted from lens 0's. Both pictures of the band are unrolled into strips — [`SEAM_COLUMNS`]
//! round the seam by [`SEAM_ROWS`] across it — and for each column the shift (across the seam,
//! where parallax moves things, and a column either way along it, where calibration leaves a
//! little) whose window of the two strips correlates best is taken, refined to a fraction of a
//! row, trusted by how well and how unambiguously it matched and how much detail there was, and
//! smoothed round the seam by that trust: where there is nothing to go by the shift fades to none.
//!
//! [`SeamFlow::warped`] then moves each lens's sample part of the way towards the other's — lens
//! 0's by (1 − w) of the shift, lens 1's by w, where w is lens 0's weight in the cross-fade — so a
//! near thing is seen once, at a place between the two, and far things stay where they are. The
//! cross-fade itself is widened to [`SEAM_BLEND_DEGREES`] either side of the seam when the shifts
//! are known ([`stitched`]). Measuring takes a few milliseconds a frame (512 × 32 samples a lens
//! and a search of 93 shifts a column, on every core).

use std::f64::consts::TAU;
use std::sync::Arc;

use filmcraft_frame::{PixelData, VideoFrame};
use rayon::prelude::*;

use crate::vr::{LensPx, SourceModel, V3, dot};

/// Columns of the seam strips, all round it (0.7° each).
pub const SEAM_COLUMNS: usize = 512;
/// Rows of the strips, across the band.
pub const SEAM_ROWS: usize = 32;
/// Frames either side of the one stitched whose seams are measured with it: the shifts then
/// hold steady from frame to frame (and stay the same whatever order frames are drawn in).
pub const SEAM_NEIGHBOURS: usize = 3;
/// Half the height of the band measured, degrees (an Insta360's lenses each see 7.5° past the
/// seam).
pub const SEAM_BAND_DEGREES: f64 = 6.0;
/// Half the width of the cross-fade with the shifts known, degrees either side of the seam (the
/// shifted samples stay inside the band).
pub const SEAM_BLEND_DEGREES: f64 = 4.5;
/// The largest shift across the seam looked for, rows (5.6°: a hand about 30 cm away).
const MAX_SHIFT_ROWS: i32 = 15;
/// Shifts along the seam looked for, columns either way.
const MAX_SLIDE_COLS: i32 = 1;
/// Half the width of the matching window, columns (9 columns, 6.3°).
const WINDOW: i32 = 7;
/// Fewest pairs of samples a window must compare.
const MIN_PAIRS: usize = 48;
/// σ of the smoothing round the seam, columns.
const SMOOTH_COLS: f64 = 5.0;
/// Contrast (standard deviation of luma, 0..1) below which a window has nothing to match.
const FLAT: f64 = 0.012;
/// Correlation a match must reach to be trusted at all.
const MIN_NCC: f64 = 0.7;
/// How much better than any other peak (a shift off the best one's hill) a match must be to be
/// fully trusted.
const CLEAR_BY: f64 = 0.08;
/// Trust round a column (smoothed) at which its shift is taken whole; below, it fades to none.
const FULL_TRUST: f64 = 0.45;

/// The shifts between a dual-fisheye picture's lenses all round their seam.
#[derive(Clone, Debug, PartialEq)]
pub struct SeamFlow {
    /// The seam's frame (world directions): `n` across it towards lens 0, `e1` and `e2` round it.
    pub n: V3,
    pub e1: V3,
    pub e2: V3,
    /// Half the height of the band, radians: outside it nothing is moved.
    pub band: f64,
    /// Per column (column `c` at φ = 2π·c / columns, from `e1` towards `e2`): lens 1 sees what
    /// lens 0 sees at (φ, ψ) at (φ + `[1]`, ψ + `[0]`), radians.
    pub shift: Vec<[f32; 2]>,
}

/// Lens `l`'s axis, world.
fn axis(l: &LensPx) -> V3 {
    [l.rotation[0][2], l.rotation[1][2], l.rotation[2][2]]
}

fn normalize(v: V3) -> Option<V3> {
    let n = dot(v, v).sqrt();
    (n > 1e-9 && n.is_finite()).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

impl SeamFlow {
    /// The seam between lenses `l0` and `l1` with no shifts; `None` when they look the same way.
    pub fn unmoved(l0: &LensPx, l1: &LensPx) -> Option<SeamFlow> {
        let (a0, a1) = (axis(l0), axis(l1));
        let n = normalize([a0[0] - a1[0], a0[1] - a1[1], a0[2] - a1[2]])?;
        // round the seam from whichever world axis is furthest from `n`
        let pick = if n[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let e1 = normalize(cross(pick, n))?;
        let e2 = cross(n, e1);
        Some(SeamFlow { n, e1, e2, band: SEAM_BAND_DEGREES.to_radians(), shift: vec![[0.0, 0.0]; SEAM_COLUMNS] })
    }

    /// The direction at `phi` round the seam and `psi` across it (towards lens 0).
    #[inline]
    pub fn direction(&self, phi: f64, psi: f64) -> V3 {
        let (sp, cp) = psi.sin_cos();
        let (sf, cf) = phi.sin_cos();
        let mut d = [0.0; 3];
        for (k, v) in d.iter_mut().enumerate() {
            *v = cp * (cf * self.e1[k] + sf * self.e2[k]) + sp * self.n[k];
        }
        d
    }

    /// The shift at `phi` (radians, any turn), linear between columns, round the seam.
    #[inline]
    pub fn shift_at(&self, phi: f64) -> [f64; 2] {
        let c = self.shift.len();
        if c == 0 {
            return [0.0, 0.0];
        }
        let x = (phi / TAU * c as f64).rem_euclid(c as f64);
        let i0 = (x.floor() as usize).min(c - 1);
        let i1 = (i0 + 1) % c;
        let f = x - x.floor();
        let (a, b) = (self.shift[i0], self.shift[i1]);
        [f64::from(a[0]) + (f64::from(b[0]) - f64::from(a[0])) * f, f64::from(a[1]) + (f64::from(b[1]) - f64::from(a[1])) * f]
    }

    /// Where lens 0 and lens 1 are sampled for direction `d` in the cross-fade, `w0` lens 0's
    /// weight: each moved part of the way towards the other along the shift. `None` outside the
    /// band; a lens whose moved sample falls outside its picture keeps `None` (sample it unmoved).
    #[inline]
    pub fn warped(&self, d: V3, w0: f64, l0: &LensPx, l1: &LensPx) -> Option<(Option<(f64, f64)>, Option<(f64, f64)>)> {
        let psi = dot(d, self.n).clamp(-1.0, 1.0).asin();
        if psi.abs() >= self.band {
            return None;
        }
        let phi = dot(d, self.e2).atan2(dot(d, self.e1));
        let [across, along] = self.shift_at(phi);
        let k0 = 1.0 - w0;
        let a = l0.locate(self.direction(phi - k0 * along, psi - k0 * across)).1;
        let b = l1.locate(self.direction(phi + w0 * along, psi + w0 * across)).1;
        Some((a, b))
    }

    /// The shifts of the seam between the first two of `lenses`, from `luma` (0..1 at a picture
    /// position, pixel centres at +0.5; `None` outside the picture). `None` without two lenses
    /// looking different ways.
    pub fn measure(lenses: &[LensPx], luma: &(dyn Fn(f64, f64) -> Option<f32> + Sync)) -> Option<SeamFlow> {
        Self::measure_frames(lenses, &[luma])
    }

    /// [`SeamFlow::measure`] over several frames at once (the frame stitched and its neighbours,
    /// [`seam_frames`]): each column's windows of every frame are matched together, so the
    /// shifts of one frame and the next share most of what they were measured on and hold steady.
    pub fn measure_frames(lenses: &[LensPx], lumas: &[&(dyn Fn(f64, f64) -> Option<f32> + Sync)]) -> Option<SeamFlow> {
        let (l0, l1) = (lenses.first()?, lenses.get(1)?);
        let mut flow = SeamFlow::unmoved(l0, l1)?;
        let band = flow.band;
        // the two strips (NaN where the lens does not see)
        let row_psi = |r: usize| -band + 2.0 * band * (r as f64 + 0.5) / SEAM_ROWS as f64;
        let strip = |l: &LensPx, luma: &(dyn Fn(f64, f64) -> Option<f32> + Sync)| -> Vec<f32> {
            (0..SEAM_COLUMNS)
                .into_par_iter()
                .flat_map_iter(|c| {
                    let phi = TAU * c as f64 / SEAM_COLUMNS as f64;
                    let f = &flow;
                    (0..SEAM_ROWS)
                        .map(move |r| l.locate(f.direction(phi, row_psi(r))).1.and_then(|(x, y)| luma(x, y)).filter(|v| v.is_finite()).unwrap_or(f32::NAN))
                })
                .collect()
        };
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = lumas.iter().take(1 + 2 * SEAM_NEIGHBOURS).map(|luma| (strip(l0, *luma), strip(l1, *luma))).collect();
        if pairs.is_empty() {
            return None;
        }
        let grid = scores(&pairs);
        let found: Vec<(f64, f64, f64)> = grid.chunks(SHIFTS).map(best_shift).collect();
        let row = 2.0 * band / SEAM_ROWS as f64;
        let col = TAU / SEAM_COLUMNS as f64;
        // smoothed round the seam by trust; little trust round a column fades its shift to none
        let half = (3.0 * SMOOTH_COLS).ceil() as i32;
        for (c, out) in flow.shift.iter_mut().enumerate() {
            let (mut w, mut s, mut t) = (0.0, 0.0, 0.0);
            for k in -half..=half {
                let j = (c as i32 + k).rem_euclid(SEAM_COLUMNS as i32) as usize;
                let Some(&(across, along, trust)) = found.get(j) else { continue };
                let g = (-(f64::from(k) * f64::from(k)) / (2.0 * SMOOTH_COLS * SMOOTH_COLS)).exp() * trust;
                w += g;
                s += g * across;
                t += g * along;
            }
            let total = (0..=2 * half).map(|k| (-(f64::from(k - half).powi(2)) / (2.0 * SMOOTH_COLS * SMOOTH_COLS)).exp()).sum::<f64>().max(1e-9);
            let keep = (w / total / FULL_TRUST).clamp(0.0, 1.0);
            if w > 1e-9 {
                *out = [(s / w * row * keep) as f32, (t / w * col * keep) as f32];
            }
        }
        Some(flow)
    }

    /// The mean disagreement of the two lenses across the band (luma, 0..1) with the pictures
    /// moved by the shifts — or unmoved (`moved` false), as a cross-fade shows them: how much a
    /// near thing is seen twice.
    pub fn ghosting(&self, lenses: &[LensPx], luma: &dyn Fn(f64, f64) -> Option<f32>, moved: bool) -> Option<f64> {
        let (l0, l1) = (lenses.first()?, lenses.get(1)?);
        let (mut sum, mut n) = (0.0, 0usize);
        let blend = SEAM_BLEND_DEGREES.to_radians();
        for c in 0..SEAM_COLUMNS * 2 {
            let phi = TAU * c as f64 / (SEAM_COLUMNS * 2) as f64;
            for r in 0..SEAM_ROWS {
                let psi = -blend + 2.0 * blend * (r as f64 + 0.5) / SEAM_ROWS as f64;
                let d = self.direction(phi, psi);
                let w0 = smoothstep(-blend, blend, psi);
                let (a, b) = if moved {
                    match self.warped(d, w0, l0, l1) {
                        Some((Some(a), Some(b))) => (a, b),
                        _ => continue,
                    }
                } else {
                    match (l0.locate(d).1, l1.locate(d).1) {
                        (Some(a), Some(b)) => (a, b),
                        _ => continue,
                    }
                };
                if let (Some(x), Some(y)) = (luma(a.0, a.1), luma(b.0, b.1)) {
                    sum += f64::from((x - y).abs());
                    n += 1;
                }
            }
        }
        (n > 0).then(|| sum / n as f64)
    }
}

impl SeamFlow {
    /// The band unrolled round the seam, `width` × `rows` a strip (ψ from +band, lens 0's side, at
    /// the top): lens 0's picture, lens 1's, the plain cross-fade ([`crate::vr::BLEND_DEGREES`])
    /// and the stitch, one above the other with a dark line between them; grey, 0..1.
    pub fn band_picture(&self, lenses: &[LensPx], luma: &dyn Fn(f64, f64) -> Option<f32>, width: usize, rows: usize) -> Option<(Vec<f32>, usize, usize)> {
        let (l0, l1) = (lenses.first()?, lenses.get(1)?);
        let (width, rows) = (width.clamp(16, 4096), rows.clamp(4, 512));
        let h = 4 * rows + 3;
        let mut px = vec![0.05f32; width * h];
        let plain = crate::vr::BLEND_DEGREES.to_radians();
        let wide = 2.0 * SEAM_BLEND_DEGREES.to_radians();
        let at = |p: Option<(f64, f64)>| p.and_then(|(x, y)| luma(x, y));
        for r in 0..rows {
            let psi = self.band - 2.0 * self.band * (r as f64 + 0.5) / rows as f64;
            for c in 0..width {
                let d = self.direction(TAU * c as f64 / width as f64, psi);
                let ((t0, a), (t1, b)) = (l0.locate(d), l1.locate(d));
                let mix = |w0: f64, x: Option<f32>, y: Option<f32>| match (x, y) {
                    (Some(x), Some(y)) => Some(x * w0 as f32 + y * (1.0 - w0) as f32),
                    (x, y) => x.or(y),
                };
                let w_plain = smoothstep(-plain, plain, t1 - t0);
                let w_wide = smoothstep(-wide, wide, t1 - t0);
                let (ma, mb) = self.warped(d, w_wide, l0, l1).map_or((a, b), |(ma, mb)| (ma.or(a), mb.or(b)));
                let vals = [at(a), at(b), mix(w_plain, at(a), at(b)), mix(w_wide, at(ma), at(mb))];
                for (k, v) in vals.into_iter().enumerate() {
                    if let (Some(v), Some(slot)) = (v, px.get_mut((k * (rows + 1) + r) * width + c)) {
                        *slot = v;
                    }
                }
            }
        }
        Some((px, width, h))
    }
}

#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Shifts searched: (rows across, columns along), in the order of [`scores`]'s grid.
const SHIFTS: usize = ((2 * MAX_SHIFT_ROWS + 1) * (2 * MAX_SLIDE_COLS + 1)) as usize;

fn shift_index(sr: i32, sc: i32) -> Option<usize> {
    (sr.abs() <= MAX_SHIFT_ROWS && sc.abs() <= MAX_SLIDE_COLS).then(|| ((sc + MAX_SLIDE_COLS) * (2 * MAX_SHIFT_ROWS + 1) + sr + MAX_SHIFT_ROWS) as usize)
}

/// Every column's correlation (and the lesser contrast) at every shift of the strips `b` against
/// the strips `a` of the frames `pairs`, over the window of columns round it: `[c * SHIFTS + s]`,
/// `None` where too few samples pair up. Each shift's sums are made once a column and slid round
/// the seam.
fn scores(pairs: &[(Vec<f32>, Vec<f32>)]) -> Vec<Option<(f64, f64)>> {
    let cols = SEAM_COLUMNS as i32;
    let rows = SEAM_ROWS as i32;
    let at = |s: &[f32], col: i32, row: i32| -> f32 { s.get((col.rem_euclid(cols) * rows + row) as usize).copied().unwrap_or(f32::NAN) };
    let per_shift: Vec<Vec<Option<(f64, f64)>>> = (0..SHIFTS)
        .into_par_iter()
        .map(|s| {
            let s = s as i32;
            let (sc, sr) = (s / (2 * MAX_SHIFT_ROWS + 1) - MAX_SLIDE_COLS, s % (2 * MAX_SHIFT_ROWS + 1) - MAX_SHIFT_ROWS);
            // each column's sums: n, Σx, Σy, Σx², Σy², Σxy
            let col_sums: Vec<[f64; 6]> = (0..cols)
                .map(|c| {
                    let mut t = [0.0f64; 6];
                    for (a, b) in pairs {
                        for r in 0.max(-sr)..rows.min(rows - sr) {
                            let (x, y) = (at(a, c, r), at(b, c + sc, r + sr));
                            if x.is_nan() || y.is_nan() {
                                continue;
                            }
                            let (x, y) = (f64::from(x), f64::from(y));
                            t[0] += 1.0;
                            t[1] += x;
                            t[2] += y;
                            t[3] += x * x;
                            t[4] += y * y;
                            t[5] += x * y;
                        }
                    }
                    t
                })
                .collect();
            let get = |c: i32| col_sums.get(c.rem_euclid(cols) as usize).copied().unwrap_or([0.0; 6]);
            let mut w = [0.0f64; 6];
            for k in -WINDOW..=WINDOW {
                let v = get(k);
                for (a, b) in w.iter_mut().zip(v) {
                    *a += b;
                }
            }
            let mut out = Vec::with_capacity(SEAM_COLUMNS);
            for c in 0..cols {
                out.push(correlation(&w));
                // slide the window one column round
                let (add, drop) = (get(c + WINDOW + 1), get(c - WINDOW));
                for ((a, x), y) in w.iter_mut().zip(add).zip(drop) {
                    *a += x - y;
                }
            }
            out
        })
        .collect();
    let mut grid = vec![None; SEAM_COLUMNS * SHIFTS];
    for (s, col) in per_shift.iter().enumerate() {
        for (c, v) in col.iter().enumerate() {
            if let Some(slot) = grid.get_mut(c * SHIFTS + s) {
                *slot = *v;
            }
        }
    }
    grid
}

/// The correlation and the lesser contrast of a window's sums (n, Σx, Σy, Σx², Σy², Σxy).
fn correlation(t: &[f64; 6]) -> Option<(f64, f64)> {
    let n = t[0].round();
    if n < MIN_PAIRS as f64 {
        return None;
    }
    let (mx, my) = (t[1] / n, t[2] / n);
    let (vx, vy) = ((t[3] / n - mx * mx).max(0.0), (t[4] / n - my * my).max(0.0));
    let cov = t[5] / n - mx * my;
    let contrast = vx.sqrt().min(vy.sqrt());
    if contrast < FLAT {
        return Some((0.0, contrast));
    }
    Some((cov / (vx * vy).sqrt().max(1e-12), contrast))
}

/// The best shift in one column's scores (`[s]` of [`scores`]): (rows across, columns along,
/// trust 0..1).
fn best_shift(column: &[Option<(f64, f64)>]) -> (f64, f64, f64) {
    let mut best: Option<(i32, i32, f64, f64)> = None;
    for sc in -MAX_SLIDE_COLS..=MAX_SLIDE_COLS {
        for sr in -MAX_SHIFT_ROWS..=MAX_SHIFT_ROWS {
            let Some(Some((ncc, contrast))) = shift_index(sr, sc).and_then(|i| column.get(i)).copied() else { continue };
            if best.is_none_or(|b| ncc > b.2) {
                best = Some((sr, sc, ncc, contrast));
            }
        }
    }
    let Some((sr, sc, ncc, contrast)) = best else { return (0.0, 0.0, 0.0) };
    let g = |r: i32, k: i32| -> f64 { shift_index(r, k).and_then(|i| column.get(i)).copied().flatten().map_or(f64::NEG_INFINITY, |v| v.0) };
    // the best's hill (across the seam, falling away either side of it), and the best other
    // peak: a clear winner, or a repeating pattern
    let (mut lo_r, mut hi_r) = (sr, sr);
    while lo_r > -MAX_SHIFT_ROWS && g(lo_r - 1, sc) <= g(lo_r, sc) {
        lo_r -= 1;
    }
    while hi_r < MAX_SHIFT_ROWS && g(hi_r + 1, sc) <= g(hi_r, sc) {
        hi_r += 1;
    }
    let mut rival = f64::NEG_INFINITY;
    for k in -MAX_SLIDE_COLS..=MAX_SLIDE_COLS {
        for r in -MAX_SHIFT_ROWS..=MAX_SHIFT_ROWS {
            // (an interior peak: at the end of the search, a rising edge is not one)
            let (v, before, after) = (g(r, k), g(r - 1, k), g(r + 1, k));
            let peak = before.is_finite() && after.is_finite() && v >= before && v >= after;
            if peak && (r < lo_r || r > hi_r) {
                rival = rival.max(v);
            }
        }
    }
    // a fraction of a row: the parabola through the best and its neighbours
    let (lo, hi) = (g(sr - 1, sc), g(sr + 1, sc));
    let mut frac = 0.0;
    if lo.is_finite() && hi.is_finite() {
        let den = lo - 2.0 * ncc + hi;
        if den < -1e-9 {
            frac = (0.5 * (lo - hi) / den).clamp(-0.5, 0.5);
        }
    }
    let good = ((ncc - MIN_NCC) / (1.0 - MIN_NCC)).clamp(0.0, 1.0);
    let clear = if rival.is_finite() { ((ncc - rival) / CLEAR_BY).clamp(0.0, 1.0) } else { 1.0 };
    let detail = (contrast / (4.0 * FLAT)).clamp(0.0, 1.0);
    (f64::from(sr) + frac, f64::from(sc), good * clear * detail)
}

/// `model` (of `frame`'s size) with the shifts of its seam measured on `frame` ([`seam_of`]) and
/// its cross-fade widened to [`SEAM_BLEND_DEGREES`] either side ([`with_seam`]).
pub fn stitched(model: SourceModel, frame: &VideoFrame) -> SourceModel {
    let seam = seam_of(&model, &[frame]);
    with_seam(model, seam)
}

/// The shifts of the seam of a dual-fisheye `model` of the frames' size, measured on `frames` (the
/// frame stitched first, then its neighbours: [`seam_frames`]); `None` for anything else.
pub fn seam_of(model: &SourceModel, frames: &[&VideoFrame]) -> Option<Arc<SeamFlow>> {
    let SourceModel::DualFisheye { lenses, .. } = model else { return None };
    let first = frames.first()?;
    if lenses.len() < 2 {
        return None;
    }
    // (only frames of the first's size: the lenses' pixels are where the model says)
    let lumas: Vec<_> = frames.iter().filter(|f| f.width == first.width && f.height == first.height).map(|f| frame_luma(f)).collect();
    let refs: Vec<&(dyn Fn(f64, f64) -> Option<f32> + Sync)> = lumas.iter().map(|l| l as &(dyn Fn(f64, f64) -> Option<f32> + Sync)).collect();
    SeamFlow::measure_frames(lenses, &refs).map(Arc::new)
}

/// The frames a seam is measured on at media time `time`: `frame` (the one decoded there, at
/// `scale`) and [`SEAM_NEIGHBOURS`] either side of it, as far as `src` has them (decoders keep the
/// frames round the one they decoded, so these cost little).
pub fn seam_frames(src: &filmcraft_media::SharedSource, time: filmcraft_time::Tick, scale: f32, frame: Arc<VideoFrame>) -> Vec<Arc<VideoFrame>> {
    let fd = src.info().frame_rate().frame_duration();
    let end = src.info().duration;
    let mut out = vec![frame];
    for k in 1..=SEAM_NEIGHBOURS as i64 {
        for t in [time - filmcraft_time::Tick(fd.0.saturating_mul(k)), time + filmcraft_time::Tick(fd.0.saturating_mul(k))] {
            if t >= filmcraft_time::Tick::ZERO
                && t < end
                && let Ok(f) = src.video_frame(filmcraft_media::FrameRequest { time: t, scale })
            {
                out.push(f);
            }
        }
    }
    out
}

/// A dual-fisheye `model` (of any size: the shifts are directions) moving its lenses' samples by
/// `seam`, its cross-fade widened to [`SEAM_BLEND_DEGREES`] either side; with `None` as it is.
pub fn with_seam(model: SourceModel, seam: Option<Arc<SeamFlow>>) -> SourceModel {
    match (model, seam) {
        (SourceModel::DualFisheye { lenses, .. }, Some(f)) => SourceModel::DualFisheye { lenses, blend: 2.0 * SEAM_BLEND_DEGREES.to_radians(), seam: Some(f) },
        (m, _) => m,
    }
}

/// Bilinear luma (0..1) of a frame at a picture position (pixel centres at +0.5): Y' of Y'CbCr,
/// Rec. 709 weights of RGB.
pub fn frame_luma(frame: &VideoFrame) -> impl Fn(f64, f64) -> Option<f32> + Sync + '_ {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let texel = move |x: usize, y: usize| -> Option<f32> {
        let i = y.checked_mul(w)?.checked_add(x)?;
        match &frame.data {
            PixelData::Yuv8 { planes, .. } => planes[0].get(i).map(|v| f32::from(*v) / 255.0),
            PixelData::Yuv16 { planes, bits, .. } => {
                let max = ((1u32 << (*bits).clamp(1, 16)) - 1) as f32;
                planes[0].get(i).map(|v| f32::from(*v) / max)
            }
            PixelData::Rgba8(px) => {
                let p = px.get(i * 4..i * 4 + 3)?;
                Some((0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0)
            }
            PixelData::RgbaF32(px) => {
                let p = px.get(i * 4..i * 4 + 3)?;
                Some(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            }
        }
    };
    move |x: f64, y: f64| -> Option<f32> {
        if w == 0 || h == 0 || !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (fx, fy) = ((x - 0.5).clamp(0.0, (w - 1) as f64), (y - 0.5).clamp(0.0, (h - 1) as f64));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        let top = texel(x0, y0)? * (1.0 - tx) + texel(x1, y0)? * tx;
        let bottom = texel(x0, y1)? * (1.0 - tx) + texel(x1, y1)? * tx;
        Some(top * (1.0 - ty) + bottom * ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vr::mvec;
    use filmcraft_frame::Chroma;
    use filmcraft_media::Spherical;
    use filmcraft_media::spherical::FisheyeLens;

    /// Two equidistant fisheyes side by side, the left one looking back.
    fn dual() -> Spherical {
        let lens = |k: usize| FisheyeLens {
            rect: [k as f64 * 0.5, 0.0, 0.5, 1.0],
            cx: 0.5,
            cy: 0.5,
            radius: 0.5,
            half_fov: 97.5,
            poly: Vec::new(),
            yaw: if k == 0 { 180.0 } else { 0.0 },
            pitch: 0.0,
            roll: 0.0,
        };
        Spherical::dual_fisheye(vec![lens(0), lens(1)], "test")
    }

    /// A random value (0..1) for a grid point.
    fn hash(i: i64, j: i64) -> f64 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 31;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Detail everywhere that never repeats: smooth value noise on a 0.8° grid.
    fn texture(phi: f64, psi: f64) -> f64 {
        let cell = 0.8f64.to_radians();
        let (x, y) = (phi / cell, psi / cell);
        let (i, j) = (x.floor() as i64, y.floor() as i64);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let s = |t: f64| t * t * (3.0 - 2.0 * t);
        let (sx, sy) = (s(fx), s(fy));
        let top = hash(i, j) * (1.0 - sx) + hash(i + 1, j) * sx;
        let bottom = hash(i, j + 1) * (1.0 - sx) + hash(i + 1, j + 1) * sx;
        0.2 + 0.6 * (top * (1.0 - sy) + bottom * sy)
    }

    /// The dual-fisheye picture (`w`×`h`) of a world of [`texture`], where between `near.0` and
    /// `near.1` round the seam something near is seen `delta` radians apart by the two lenses
    /// (lens 0 sees it `delta / 2` towards itself's far side, lens 1 the other way).
    fn picture(w: u32, h: u32, near: (f64, f64), delta: f64) -> VideoFrame {
        let s = dual();
        let SourceModel::DualFisheye { lenses, .. } = SourceModel::new(&s, (w as usize, h as usize)) else { panic!() };
        let seam = SeamFlow::unmoved(&lenses[0], &lenses[1]).unwrap();
        let mut y = vec![0u8; (w * h) as usize];
        for py in 0..h {
            for px in 0..w {
                let k = if f64::from(px) < f64::from(w) / 2.0 { 0 } else { 1 };
                let l = &lenses[k];
                let (dx, dy) = (f64::from(px) + 0.5 - l.cx, l.cy - (f64::from(py) + 0.5));
                let theta = (dx * dx + dy * dy).sqrt() / l.radius * l.half_fov;
                if theta > l.half_fov {
                    continue;
                }
                let a = dy.atan2(dx);
                let d = mvec(&l.rotation, [theta.sin() * a.cos(), theta.sin() * a.sin(), theta.cos()]);
                let psi = dot(d, seam.n).clamp(-1.0, 1.0).asin();
                let phi = dot(d, seam.e2).atan2(dot(d, seam.e1)).rem_euclid(TAU);
                let off = if phi > near.0 && phi < near.1 { if k == 0 { delta / 2.0 } else { -delta / 2.0 } } else { 0.0 };
                y[(py * w + px) as usize] = (texture(phi, psi + off) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let c = vec![128u8; (w.div_ceil(2) * h.div_ceil(2)) as usize];
        VideoFrame {
            width: w,
            height: h,
            data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(c.clone()), Arc::new(c)], chroma: Chroma::C420, alpha: None },
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            pts: Default::default(),
        }
    }

    fn lenses_of(frame: &VideoFrame) -> Vec<LensPx> {
        let SourceModel::DualFisheye { lenses, .. } = SourceModel::new(&dual(), (frame.width as usize, frame.height as usize)) else { panic!() };
        lenses
    }

    fn mean_across(f: &SeamFlow, from: f64, to: f64) -> f64 {
        let cols: Vec<f64> = (0..SEAM_COLUMNS).map(|c| TAU * c as f64 / SEAM_COLUMNS as f64).filter(|p| *p > from && *p < to).collect();
        cols.iter().map(|p| f.shift_at(*p)[0]).sum::<f64>() / cols.len().max(1) as f64
    }

    #[test]
    fn a_near_thing_on_the_seam_is_measured_and_seen_once() {
        let delta = 3.0f64.to_radians();
        let frame = picture(1024, 512, (1.0, 1.9), delta);
        let lenses = lenses_of(&frame);
        let luma = frame_luma(&frame);
        let f = SeamFlow::measure(&lenses, &luma).unwrap();
        let near = mean_across(&f, 1.2, 1.7);
        assert!((near - delta).abs() < 0.4f64.to_radians(), "near: {:.2}° (want 3°)", near.to_degrees());
        let far = mean_across(&f, 3.0, 5.5);
        assert!(far.abs() < 0.25f64.to_radians(), "far: {:.2}°", far.to_degrees());
        // seen once: the lenses' samples agree where the cross-fade mixes them
        let before = f.ghosting(&lenses, &luma, false).unwrap();
        let after = f.ghosting(&lenses, &luma, true).unwrap();
        eprintln!("near {:.2}°, far {:.2}°, ghosting {before:.4} → {after:.4}", near.to_degrees(), far.to_degrees());
        assert!(after < before * 0.6, "ghosting {before:.4} → {after:.4}");
        // and the near thing's own band agrees as well as the far scene's does
        let only = SeamFlow {
            shift: f.shift.iter().enumerate().map(|(c, s)| if (TAU * c as f64 / SEAM_COLUMNS as f64 - 1.45).abs() < 0.25 { *s } else { [0.0, 0.0] }).collect(),
            ..f.clone()
        };
        assert!(only.ghosting(&lenses, &luma, true).unwrap() < before);
    }

    #[test]
    fn a_seam_without_parallax_or_detail_is_left_alone() {
        // no near thing: the far scene agrees already
        let frame = picture(1024, 512, (0.0, 0.0), 0.0);
        let lenses = lenses_of(&frame);
        let f = SeamFlow::measure(&lenses, &frame_luma(&frame)).unwrap();
        let worst = f.shift.iter().map(|s| f64::from(s[0].abs()).max(f64::from(s[1].abs()))).fold(0.0, f64::max);
        assert!(worst < 0.3f64.to_radians(), "{:.2}°", worst.to_degrees());
        // a plain grey picture: nothing to match, nothing moved
        let mut grey = picture(256, 128, (0.0, 0.0), 0.0);
        grey.data = PixelData::Yuv8 {
            planes: [Arc::new(vec![120u8; 256 * 128]), Arc::new(vec![128u8; 128 * 64]), Arc::new(vec![128u8; 128 * 64])],
            chroma: Chroma::C420,
            alpha: None,
        };
        let g = SeamFlow::measure(&lenses_of(&grey), &frame_luma(&grey)).unwrap();
        assert!(g.shift.iter().all(|s| s[0] == 0.0 && s[1] == 0.0));
        // the model keeps its lenses; the cross-fade widens with the shifts known
        let m = stitched(SourceModel::new(&dual(), (256, 128)), &grey);
        assert!(matches!(&m, SourceModel::DualFisheye { seam: Some(_), blend, .. } if (*blend - 9.0f64.to_radians()).abs() < 1e-9));
    }

    #[test]
    fn hostile_pictures_never_panic() {
        let lenses = lenses_of(&picture(64, 32, (0.0, 0.0), 0.0));
        // lenses looking the same way have no seam
        assert!(SeamFlow::unmoved(&lenses[1], &lenses[1]).is_none());
        // pictures of nothing, of one pixel, of NaN and of the wrong size
        for (w, h, data) in [
            (0u32, 0u32, PixelData::RgbaF32(Arc::new(vec![]))),
            (1, 1, PixelData::Rgba8(Arc::new(vec![1, 2, 3, 4]))),
            (8, 4, PixelData::RgbaF32(Arc::new(vec![f32::NAN; 8 * 4 * 4]))),
            (64, 32, PixelData::Yuv16 { planes: [Arc::new(vec![900; 10]), Arc::new(vec![]), Arc::new(vec![])], chroma: Chroma::C420, bits: 99, alpha: None }),
        ] {
            let frame = VideoFrame { width: w, height: h, data, color: filmcraft_color::ColorInfo::REC709, par: (1, 1), pts: Default::default() };
            let m = stitched(SourceModel::new(&dual(), (w.max(1) as usize, h.max(1) as usize)), &frame);
            if let SourceModel::DualFisheye { seam: Some(f), .. } = &m {
                assert!(f.shift.iter().flatten().all(|x| x.is_finite()));
            }
            let _ = m.locate([0.0, 0.0, 1.0]);
            let _ = m.locate([1.0, 0.0, 0.0]);
        }
        // an equirectangular picture has no seam
        let e = SourceModel::new(&Spherical::equirect(), (64, 32));
        assert_eq!(stitched(e.clone(), &picture(64, 32, (0.0, 0.0), 0.0)), e);
    }
}
