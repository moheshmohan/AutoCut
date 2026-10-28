//! Temporal noise reduction: each frame averaged with its neighbours, each neighbour first moved
//! to line up with it and then taken only where it agrees — so a night shot's grain, different in
//! every frame, averages away while what moves does not smear. No model: motion and agreement.
//!
//! 1. **Motion** ([`motion`]): both frames' luma box-averaged to an eighth of their size; for every
//!    8 × 8 block of that (64 × 64 pixels of the frame) the shift within ±3 (±24 pixels) whose
//!    sum of absolute differences is least, refined to a fraction by the parabola through its
//!    neighbours, then each component the median of the 3 × 3 blocks round it.
//! 2. **Merge** ([`merge_plane`]): for each pixel of each plane, the neighbours' pixels where the
//!    motion (bilinear between block centres) takes it, each weighted by `exp(-(d/σ)²)` for its
//!    difference `d` from the pixel itself, and averaged with it; `σ` is the frame's own noise
//!    ([`noise_sigma`], the median difference of neighbouring pixels) times 1.5–4.5 by strength.
//!
//! The motion is measured here on the CPU (a few milliseconds even at 5.7K); the merge, the heavy
//! part, runs on the GPU when the host installs one ([`set_gpu`], `filmcraft_gpu::denoise`), this
//! being the reference and the fallback; the two are tested against each other.

use std::sync::{Arc, Mutex, OnceLock};

use filmcraft_frame::{AudioBuffer, Chroma, PixelData, VideoFrame};
use filmcraft_media::{FrameRequest, MediaInfo, MediaSource, SharedSource, Spherical};
use filmcraft_time::Tick;
use rayon::prelude::*;

/// The size the motion is measured at: an eighth of the frame.
pub const MOTION_SCALE: usize = 8;
/// A block of the motion, in pixels of the eighth-size frame.
pub const BLOCK: usize = 8;
/// The farthest a block is looked for, eighth-size pixels either way.
pub const SEARCH: i32 = 3;
/// Most frames either side merged.
pub const MAX_RADIUS: usize = 3;
/// Samples this near the picture's black or white count as clipped in [`noise_sigma`].
const CLIP: f32 = 1.5 / 255.0;

/// How hard, and over how many frames either side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseOptions {
    /// 0..1.
    pub strength: f32,
    pub radius: usize,
}

impl DenoiseOptions {
    pub fn sanitized(self) -> Self {
        DenoiseOptions { strength: if self.strength.is_finite() { self.strength.clamp(0.0, 1.0) } else { 0.5 }, radius: self.radius.clamp(1, MAX_RADIUS) }
    }
    /// The weight's width for a frame of noise `noise` (0..1): 1.5–4.5 times the noise by
    /// strength, and never under 1–4 levels of 255 (a camera that cleaned its frames up itself
    /// leaves little grain but still a shimmer from frame to frame in the dark).
    pub fn sigma(&self, noise: f32) -> f32 {
        (noise * (1.5 + 3.0 * self.strength)).max((1.0 + 3.0 * self.strength) / 255.0)
    }
}

/// One plane of a frame, its samples 0..`max`.
#[derive(Clone, Copy, Debug)]
pub enum Plane<'a> {
    U8 { px: &'a [u8], w: usize, h: usize },
    U16 { px: &'a [u16], w: usize, h: usize, max: f32 },
}

impl Plane<'_> {
    pub fn size(&self) -> (usize, usize) {
        match self {
            Plane::U8 { w, h, .. } | Plane::U16 { w, h, .. } => (*w, *h),
        }
    }
    /// The sample at (x, y), 0..1 (clamped to the plane).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        let (w, h) = self.size();
        let i = y.min(h.saturating_sub(1)) * w + x.min(w.saturating_sub(1));
        match self {
            Plane::U8 { px, .. } => px.get(i).map_or(0.0, |v| f32::from(*v) / 255.0),
            Plane::U16 { px, max, .. } => px.get(i).map_or(0.0, |v| f32::from(*v) / max),
        }
    }
    /// Bilinear at (x, y), pixel centres at whole numbers.
    #[inline]
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let (w, h) = self.size();
        let (fx, fy) = (x.clamp(0.0, (w.max(1) - 1) as f32), y.clamp(0.0, (h.max(1) - 1) as f32));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let top = self.at(x0, y0) * (1.0 - tx) + self.at(x0 + 1, y0) * tx;
        let bottom = self.at(x0, y0 + 1) * (1.0 - tx) + self.at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// The planes of a Y'CbCr frame (`None` for anything else) and its chroma subsampling.
pub fn planes(f: &VideoFrame) -> Option<([Plane<'_>; 3], Chroma)> {
    let (w, h) = (f.width as usize, f.height as usize);
    match &f.data {
        PixelData::Yuv8 { planes, chroma, .. } => {
            let (sx, sy) = chroma.shifts();
            let (cw, ch) = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
            let ok = planes[0].len() >= w * h && planes[1].len() >= cw * ch && planes[2].len() >= cw * ch;
            ok.then(|| ([Plane::U8 { px: &planes[0], w, h }, Plane::U8 { px: &planes[1], w: cw, h: ch }, Plane::U8 { px: &planes[2], w: cw, h: ch }], *chroma))
        }
        PixelData::Yuv16 { planes, chroma, bits, .. } => {
            let (sx, sy) = chroma.shifts();
            let (cw, ch) = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
            let max = ((1u32 << (*bits).clamp(1, 16)) - 1) as f32;
            let ok = planes[0].len() >= w * h && planes[1].len() >= cw * ch && planes[2].len() >= cw * ch;
            ok.then(|| {
                (
                    [
                        Plane::U16 { px: &planes[0], w, h, max },
                        Plane::U16 { px: &planes[1], w: cw, h: ch, max },
                        Plane::U16 { px: &planes[2], w: cw, h: ch, max },
                    ],
                    *chroma,
                )
            })
        }
        _ => None,
    }
}

/// The motion of a neighbour against the frame: per block of [`MOTION_SCALE`]·[`BLOCK`] luma
/// pixels, where the frame's pixel is in the neighbour (luma pixels, `[dx, dy]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Motion {
    pub bw: usize,
    pub bh: usize,
    /// The block's side in luma pixels.
    pub block: usize,
    pub vec: Vec<[f32; 2]>,
}

impl Motion {
    /// The shift at luma (x, y): bilinear between block centres.
    #[inline]
    pub fn at(&self, x: f32, y: f32) -> [f32; 2] {
        if self.bw == 0 || self.bh == 0 {
            return [0.0, 0.0];
        }
        let b = self.block as f32;
        let (fx, fy) = ((x / b - 0.5).clamp(0.0, (self.bw - 1) as f32), (y / b - 0.5).clamp(0.0, (self.bh - 1) as f32));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.bw - 1), (y0 + 1).min(self.bh - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let v = |i: usize, j: usize| self.vec.get(j * self.bw + i).copied().unwrap_or([0.0, 0.0]);
        let mut out = [0.0; 2];
        for (k, o) in out.iter_mut().enumerate() {
            let top = v(x0, y0)[k] * (1.0 - tx) + v(x1, y0)[k] * tx;
            let bottom = v(x0, y1)[k] * (1.0 - tx) + v(x1, y1)[k] * tx;
            *o = top * (1.0 - ty) + bottom * ty;
        }
        out
    }
}

/// A plane box-averaged by `k` (the edge's partial boxes averaged over what they hold).
fn reduce(p: &Plane<'_>, k: usize) -> (Vec<f32>, usize, usize) {
    let (w, h) = p.size();
    let (rw, rh) = (w.div_ceil(k).max(1), h.div_ceil(k).max(1));
    let out: Vec<f32> = (0..rh)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..rw).map(move |x| {
                let (mut s, mut n) = (0f32, 0f32);
                for yy in y * k..((y + 1) * k).min(h) {
                    for xx in x * k..((x + 1) * k).min(w) {
                        s += p.at(xx, yy);
                        n += 1.0;
                    }
                }
                s / n.max(1.0)
            })
        })
        .collect();
    (out, rw, rh)
}

/// [`Motion`] of luma plane `nb` against luma plane `center` (the same size).
pub fn motion(center: &Plane<'_>, nb: &Plane<'_>) -> Motion {
    let (a, w, h) = reduce(center, MOTION_SCALE);
    let (b, _, _) = reduce(nb, MOTION_SCALE);
    let (bw, bh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
    let at = |v: &[f32], x: i64, y: i64| -> f32 { v[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize] };
    let sad = |bx: usize, by: usize, dx: i32, dy: i32| -> f32 {
        let mut s = 0f32;
        for y in by * BLOCK..((by + 1) * BLOCK).min(h) {
            for x in bx * BLOCK..((bx + 1) * BLOCK).min(w) {
                s += (at(&a, x as i64, y as i64) - at(&b, x as i64 + i64::from(dx), y as i64 + i64::from(dy))).abs();
            }
        }
        s
    };
    let raw: Vec<[f32; 2]> = (0..bw * bh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % bw, i / bw);
            let mut best = (0, 0, sad(bx, by, 0, 0));
            for dy in -SEARCH..=SEARCH {
                for dx in -SEARCH..=SEARCH {
                    let s = sad(bx, by, dx, dy);
                    // (ties stay still)
                    if s < best.2 - 1e-6 {
                        best = (dx, dy, s);
                    }
                }
            }
            let (dx, dy, s0) = best;
            let frac = |lo: f32, hi: f32| -> f32 {
                let den = lo - 2.0 * s0 + hi;
                if den > 1e-9 { (0.5 * (lo - hi) / den).clamp(-0.5, 0.5) } else { 0.0 }
            };
            let fx = if dx.abs() < SEARCH { frac(sad(bx, by, dx - 1, dy), sad(bx, by, dx + 1, dy)) } else { 0.0 };
            let fy = if dy.abs() < SEARCH { frac(sad(bx, by, dx, dy - 1), sad(bx, by, dx, dy + 1)) } else { 0.0 };
            [(dx as f32 + fx) * MOTION_SCALE as f32, (dy as f32 + fy) * MOTION_SCALE as f32]
        })
        .collect();
    // each component the median of the blocks round it
    let vec = (0..bw * bh)
        .map(|i| {
            let (bx, by) = ((i % bw) as i64, (i / bw) as i64);
            let mut out = [0.0; 2];
            for (k, o) in out.iter_mut().enumerate() {
                let mut v: Vec<f32> = (-1..=1)
                    .flat_map(|dy| (-1..=1).map(move |dx| (bx + dx, by + dy)))
                    .filter(|(x, y)| *x >= 0 && *y >= 0 && (*x as usize) < bw && (*y as usize) < bh)
                    .filter_map(|(x, y)| raw.get(y as usize * bw + x as usize).map(|r| r[k]))
                    .collect();
                v.sort_by(f32::total_cmp);
                *o = v.get(v.len() / 2).copied().unwrap_or(0.0);
            }
            out
        })
        .collect();
    Motion { bw, bh, block: MOTION_SCALE * BLOCK, vec }
}

/// The frame's own noise (0..1, a standard deviation): on a sparse grid of luma pixels, the
/// response of Immerkær's mask (the difference of two Laplacians, blind to edges and ramps, 6σ
/// for white noise), its median over 6 × 0.6745. Pixels clipped to the picture's black or white
/// are left out (no noise there: a fisheye's black surround, a blown sky).
pub fn noise_sigma(y: &Plane<'_>) -> f32 {
    grain(y).0
}

/// The frame's noise ([`noise_sigma`]) and light (the mean luma, 0..1), both without the clipped
/// pixels.
pub fn grain(y: &Plane<'_>) -> (f32, f32) {
    let (w, h) = y.size();
    if w < 3 || h < 3 {
        return (0.0, 0.0);
    }
    let inner = (w - 2) * (h - 2);
    let step = (inner / 200_000).max(1);
    // (the centre, the mask's response, the 3 × 3's least and most)
    let taps: Vec<(f32, f32, f32, f32)> = (0..inner)
        .step_by(step)
        .map(|i| {
            let (x, yy) = (i % (w - 2) + 1, i / (w - 2) + 1);
            let mut r = 0.0f32;
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for (dy, row) in [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]].iter().enumerate() {
                for (dx, k) in row.iter().enumerate() {
                    let v = y.at(x + dx - 1, yy + dy - 1);
                    r += k * v;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            (y.at(x, yy), r, lo, hi)
        })
        .collect();
    // black and white as this picture has them (limited range: 16 and 235)
    let (black, white) = taps.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(b, w), t| (b.min(t.2), w.max(t.3)));
    let clipped = |t: &(f32, f32, f32, f32)| (black < 0.1 && t.3 - black <= CLIP) || (white > 0.9 && white - t.2 <= CLIP);
    let kept: Vec<&(f32, f32, f32, f32)> = taps.iter().filter(|t| !clipped(t)).collect();
    if kept.is_empty() {
        return (0.0, 0.0);
    }
    let light = kept.iter().map(|t| t.0).sum::<f32>() / kept.len() as f32;
    let mut r: Vec<f32> = kept.iter().map(|t| t.1.abs()).collect();
    let mid = r.len() / 2;
    r.select_nth_unstable_by(mid, f32::total_cmp);
    (r[mid] / (6.0 * 0.6745), light)
}

/// A plane merged with its neighbours' (each with its motion; `scale` the plane's size over the
/// luma's), samples 0..1.
pub fn merge_plane(center: &Plane<'_>, nbs: &[(Plane<'_>, &Motion)], scale: [f32; 2], sigma: f32) -> Vec<f32> {
    let (w, h) = center.size();
    let inv = 1.0 / (sigma * sigma).max(1e-12);
    (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..w).map(move |x| {
                let c = center.at(x, y);
                let (mut sum, mut wsum) = (c, 1.0f32);
                for (p, m) in nbs {
                    // the pixel's luma position, the motion there, back in this plane's pixels
                    let v = m.at((x as f32 + 0.5) / scale[0], (y as f32 + 0.5) / scale[1]);
                    let s = p.bilinear(x as f32 + v[0] * scale[0], y as f32 + v[1] * scale[1]);
                    let d = s - c;
                    let wt = (-(d * d) * inv).exp();
                    sum += wt * s;
                    wsum += wt;
                }
                sum / wsum
            })
        })
        .collect()
}

/// Merges planes on the GPU (installed by the host): the merged plane's samples 0..1, or `None`
/// to merge on the CPU.
pub trait PlaneMerger: Send + Sync {
    fn merge(&self, center: &Plane<'_>, nbs: &[(Plane<'_>, &Motion)], scale: [f32; 2], sigma: f32) -> Option<Vec<f32>>;
}

static GPU: OnceLock<Arc<dyn PlaneMerger>> = OnceLock::new();

/// Install the GPU's merger (once; later calls are ignored).
pub fn set_gpu(m: Arc<dyn PlaneMerger>) {
    let _ = GPU.set(m);
}

/// Whether merging runs on the GPU.
pub fn on_gpu() -> bool {
    GPU.get().is_some()
}

/// `center` denoised with `neighbours` (frames of its size and format: the others are left out);
/// `None` for frames it cannot work on (not Y'CbCr) or without neighbours.
pub fn denoise(center: &VideoFrame, neighbours: &[Arc<VideoFrame>], o: DenoiseOptions) -> Option<VideoFrame> {
    let o = o.sanitized();
    let (cp, chroma) = planes(center)?;
    let nbs: Vec<[Plane<'_>; 3]> = neighbours
        .iter()
        .filter(|n| n.width == center.width && n.height == center.height)
        .filter_map(|n| planes(n).filter(|(_, c)| *c == chroma).map(|(p, _)| p))
        .collect();
    if nbs.is_empty() {
        return None;
    }
    let motions: Vec<Motion> = nbs.iter().map(|n| motion(&cp[0], &n[0])).collect();
    let sigma = o.sigma(noise_sigma(&cp[0]));
    let (lw, lh) = cp[0].size();
    let merged: Vec<Vec<f32>> = (0..3)
        .map(|k| {
            let (pw, ph) = cp[k].size();
            let scale = [pw as f32 / lw.max(1) as f32, ph as f32 / lh.max(1) as f32];
            // (chroma noise reads a little wider)
            let s = if k == 0 { sigma } else { sigma * 1.5 };
            let with: Vec<(Plane<'_>, &Motion)> = nbs.iter().zip(&motions).map(|(n, m)| (n[k], m)).collect();
            GPU.get().and_then(|g| g.merge(&cp[k], &with, scale, s)).filter(|v| v.len() == pw * ph).unwrap_or_else(|| merge_plane(&cp[k], &with, scale, s))
        })
        .collect();
    let data = match &center.data {
        PixelData::Yuv8 { alpha, .. } => {
            let q = |v: &Vec<f32>| Arc::new(v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8).collect::<Vec<u8>>());
            PixelData::Yuv8 { planes: [q(&merged[0]), q(&merged[1]), q(&merged[2])], chroma, alpha: alpha.clone() }
        }
        PixelData::Yuv16 { bits, alpha, .. } => {
            let max = ((1u32 << (*bits).clamp(1, 16)) - 1) as f32;
            let q = |v: &Vec<f32>| Arc::new(v.iter().map(|x| (x.clamp(0.0, 1.0) * max).round() as u16).collect::<Vec<u16>>());
            PixelData::Yuv16 { planes: [q(&merged[0]), q(&merged[1]), q(&merged[2])], chroma, bits: *bits, alpha: alpha.clone() }
        }
        _ => return None,
    };
    Some(VideoFrame { data, ..center.clone() })
}

/// A source whose frames are denoised with their neighbours ([`denoise`]); sound, timing and the
/// sphere pass through. The last few frames made are kept (scrubbing asks again).
pub struct DenoisedSource {
    inner: SharedSource,
    options: DenoiseOptions,
    recent: Mutex<Vec<((Tick, u32), Arc<VideoFrame>)>>,
}

/// Frames a [`DenoisedSource`] keeps.
const RECENT: usize = 4;

impl DenoisedSource {
    pub fn new(inner: SharedSource, options: DenoiseOptions) -> Self {
        DenoisedSource { inner, options: options.sanitized(), recent: Mutex::new(Vec::new()) }
    }
}

impl MediaSource for DenoisedSource {
    fn info(&self) -> &MediaInfo {
        self.inner.info()
    }

    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let key = (req.time, req.scale.to_bits());
        if let Some(f) = self.recent.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, f)| f.clone()) {
            return Ok(f);
        }
        let center = self.inner.video_frame(req)?;
        let fd = self.inner.info().frame_rate().frame_duration();
        let end = self.inner.info().duration;
        let mut nbs = Vec::new();
        for k in 1..=self.options.radius as i64 {
            for t in [req.time - Tick(fd.0.saturating_mul(k)), req.time + Tick(fd.0.saturating_mul(k))] {
                if t >= Tick::ZERO
                    && t < end
                    && let Ok(f) = self.inner.video_frame(FrameRequest { time: t, ..req })
                {
                    nbs.push(f);
                }
            }
        }
        let out = denoise(&center, &nbs, self.options).map(Arc::new).unwrap_or(center);
        let mut recent = self.recent.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        recent.push((key, out.clone()));
        if recent.len() > RECENT {
            recent.remove(0);
        }
        Ok(out)
    }

    fn plain_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        self.inner.plain_frame(req)
    }

    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }

    fn audio_stream(&self, stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio_stream(stream, start, frames, sample_rate)
    }

    fn spherical(&self) -> Option<Spherical> {
        self.inner.spherical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noisy frame of a pattern moved `shift` pixels right: luma 8-bit, 4:2:0, the noise fixed
    /// by `seed`.
    pub(crate) fn frame(w: u32, h: u32, shift: f32, seed: u64, noise: f32) -> VideoFrame {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let pattern = |x: f32, y: f32| 0.5 + 0.25 * ((x - shift) * 0.05).sin() * (y * 0.07).cos() + 0.15 * (((x - shift) * 0.013 + y * 0.011).sin());
        let y: Vec<u8> = (0..w * h).map(|i| ((pattern((i % w) as f32, (i / w) as f32) + noise * next() * 3.46) * 255.0).clamp(0.0, 255.0) as u8).collect();
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let c: Vec<u8> = (0..cw * ch).map(|_| ((0.5 + noise * next() * 3.46) * 255.0).clamp(0.0, 255.0) as u8).collect();
        VideoFrame {
            width: w,
            height: h,
            data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(c.clone()), Arc::new(c)], chroma: Chroma::C420, alpha: None },
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            pts: Tick::ZERO,
        }
    }

    /// The luma's difference from the clean pattern (moved `shift`), RMS, 0..1.
    pub(crate) fn error(f: &VideoFrame, shift: f32) -> f32 {
        let clean = frame(f.width, f.height, shift, 1, 0.0);
        let (a, _) = planes(f).unwrap();
        let (b, _) = planes(&clean).unwrap();
        let (w, h) = a[0].size();
        let mut s = 0f32;
        for y in 0..h {
            for x in 0..w {
                s += (a[0].at(x, y) - b[0].at(x, y)).powi(2);
            }
        }
        (s / (w * h) as f32).sqrt()
    }

    #[test]
    fn grain_averages_away_and_motion_is_followed() {
        let (w, h) = (320, 192);
        let noise = 0.03;
        // the pattern moving 4 px a frame to the right
        let center = frame(w, h, 0.0, 11, noise);
        let nbs: Vec<Arc<VideoFrame>> =
            [(-8.0, 12), (-4.0, 13), (4.0, 14), (8.0, 15)].iter().map(|(s, seed)| Arc::new(frame(w, h, *s, *seed, noise))).collect();
        let before = error(&center, 0.0);
        let out = denoise(&center, &nbs, DenoiseOptions { strength: 0.6, radius: 2 }).unwrap();
        let after = error(&out, 0.0);
        assert!(after < before * 0.65, "noise {before:.4} → {after:.4}");
        // the motion was found: about −4 px (the neighbour a frame later shows it 4 px right)
        let (cp, _) = planes(&center).unwrap();
        let (np, _) = planes(&nbs[2]).unwrap();
        let m = motion(&cp[0], &np[0]);
        let mid = m.at(160.0, 96.0);
        assert!((mid[0] - 4.0).abs() < 1.5 && mid[1].abs() < 1.5, "{mid:?}");
        // without motion compensation the moving pattern would smear: the merge without the
        // neighbours' motion is worse
        let zero = Motion { bw: m.bw, bh: m.bh, block: m.block, vec: vec![[0.0, 0.0]; m.vec.len()] };
        let smeared = merge_plane(&cp[0], &[(np[0], &zero)], [1.0, 1.0], 0.2);
        let followed = merge_plane(&cp[0], &[(np[0], &m)], [1.0, 1.0], 0.2);
        let clean = frame(w, h, 0.0, 1, 0.0);
        let (cl, _) = planes(&clean).unwrap();
        let err = |v: &[f32]| (v.iter().enumerate().map(|(i, s)| (s - cl[0].at(i % w as usize, i / w as usize)).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        assert!(err(&followed) < err(&smeared), "{} vs {}", err(&followed), err(&smeared));
        // the noise estimate is near the truth (σ of a uniform ±1.73σ is σ)
        let est = noise_sigma(&cp[0]);
        assert!((est - noise).abs() < noise * 0.5, "{est}");
    }

    #[test]
    fn hostile_frames_are_left_as_they_are() {
        let f = frame(64, 32, 0.0, 3, 0.02);
        assert!(denoise(&f, &[], DenoiseOptions { strength: 0.5, radius: 1 }).is_none());
        let other = Arc::new(frame(32, 32, 0.0, 4, 0.02));
        assert!(denoise(&f, &[other], DenoiseOptions { strength: f32::NAN, radius: 99 }).is_none(), "another size");
        let rgba = VideoFrame::rgba8(4, 4, vec![0; 64]);
        assert!(denoise(&rgba, &[Arc::new(rgba.clone())], DenoiseOptions { strength: 0.5, radius: 1 }).is_none());
        let short = VideoFrame {
            data: PixelData::Yuv8 { planes: [Arc::new(vec![0; 3]), Arc::new(vec![]), Arc::new(vec![])], chroma: Chroma::C420, alpha: None },
            ..f.clone()
        };
        assert!(planes(&short).is_none());
        assert_eq!(noise_sigma(&Plane::U8 { px: &[], w: 0, h: 0 }), 0.0);
        // a fisheye's black surround (limited range: 16) is not counted
        let (w, h) = (200usize, 100usize);
        let lit = frame(w as u32, h as u32, 0.0, 9, 0.02);
        let (lp, _) = planes(&lit).unwrap();
        let alone = grain(&lp[0]);
        let ringed: Vec<u8> = (0..w * h).map(|i| if i % w < 60 { 16 } else { (lp[0].at(i % w, i / w) * 255.0).round() as u8 }).collect();
        let with = grain(&Plane::U8 { px: &ringed, w, h });
        assert!((with.0 - alone.0).abs() < alone.0 * 0.25, "{alone:?} {with:?}");
        assert!((with.1 - alone.1).abs() < 0.05, "{alone:?} {with:?}");
        let m = Motion { bw: 0, bh: 0, block: 64, vec: vec![] };
        assert_eq!(m.at(5.0, 5.0), [0.0, 0.0]);
    }
}
