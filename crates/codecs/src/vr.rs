//! 360° video in MP4 files: Insta360 recordings (one video track a lens, decoded side by side
//! into one dual-fisheye frame) and the spherical metadata of stitched equirectangular video
//! (`sv3d` / `st3d`, or the V1 `uuid` box; `filmcraft_isobmff::spherical`).

use filmcraft_frame::{PixelData, VideoFrame};
use filmcraft_isobmff::insta360::{Info, Motion, parse_offset};
use filmcraft_isobmff::spherical::{Projection, StereoMode};
use filmcraft_isobmff::{Mp4File, Track, TrackKind};
use filmcraft_media::Spherical;
use filmcraft_media::spherical::{FisheyeLens, Horizon, StereoLayout};

use crate::{CodecError, Result};

/// The video track of an Insta360 recording's second lens, beside the first lens's track
/// `first`: another playable video track of the same codec, picture size, timescale and number of
/// pictures (give or take one).
pub(crate) fn second_lens(file: &Mp4File, first: usize) -> Option<usize> {
    let a = file.tracks.get(first)?;
    let size = |t: &Track| t.video().map(|v| (v.width, v.height));
    let codec = |t: &Track| t.codec().map(|c| c.name());
    file.tracks.iter().enumerate().position(|(j, t)| {
        j != first
            && t.kind == TrackKind::Video
            && !t.entries.is_empty()
            && !t.samples.is_empty()
            && t.timescale == a.timescale
            && size(t) == size(a)
            && codec(t) == codec(a)
            && t.samples.len().abs_diff(a.samples.len()) <= 1
    })
}

/// Two pictures of the same size and pixel format side by side: `a` on the left.
pub(crate) fn side_by_side(a: &VideoFrame, b: &VideoFrame) -> Result<VideoFrame> {
    if (a.width, a.height) != (b.width, b.height) {
        return Err(CodecError::Decode(format!("the two lenses decoded to different sizes ({}x{} and {}x{})", a.width, a.height, b.width, b.height)));
    }
    let (w, h) = (a.width as usize, a.height as usize);
    let data = match (&a.data, &b.data) {
        (PixelData::Yuv8 { planes: pa, chroma: ca, alpha: None }, PixelData::Yuv8 { planes: pb, chroma: cb, alpha: None }) if ca == cb => {
            let (sx, sy) = ca.shifts();
            if w % (1 << sx) != 0 {
                return Err(CodecError::Decode("odd lens picture width".into()));
            }
            let (cw, ch) = (w >> sx, h.div_ceil(1 << sy));
            PixelData::Yuv8 {
                planes: [
                    std::sync::Arc::new(join_rows(&pa[0], &pb[0], w, h)?),
                    std::sync::Arc::new(join_rows(&pa[1], &pb[1], cw, ch)?),
                    std::sync::Arc::new(join_rows(&pa[2], &pb[2], cw, ch)?),
                ],
                chroma: *ca,
                alpha: None,
            }
        }
        (PixelData::Yuv16 { planes: pa, chroma: ca, bits: ba, alpha: None }, PixelData::Yuv16 { planes: pb, chroma: cb, bits: bb, alpha: None })
            if ca == cb && ba == bb =>
        {
            let (sx, sy) = ca.shifts();
            if w % (1 << sx) != 0 {
                return Err(CodecError::Decode("odd lens picture width".into()));
            }
            let (cw, ch) = (w >> sx, h.div_ceil(1 << sy));
            PixelData::Yuv16 {
                planes: [
                    std::sync::Arc::new(join_rows(&pa[0], &pb[0], w, h)?),
                    std::sync::Arc::new(join_rows(&pa[1], &pb[1], cw, ch)?),
                    std::sync::Arc::new(join_rows(&pa[2], &pb[2], cw, ch)?),
                ],
                chroma: *ca,
                bits: *ba,
                alpha: None,
            }
        }
        (PixelData::Rgba8(pa), PixelData::Rgba8(pb)) => PixelData::Rgba8(std::sync::Arc::new(join_rows(pa, pb, w * 4, h)?)),
        _ => return Err(CodecError::Decode("the two lenses decoded to different pixel formats".into())),
    };
    let width = a.width.checked_mul(2).ok_or_else(|| CodecError::Decode("dual-lens picture too wide".into()))?;
    Ok(VideoFrame { width, height: a.height, data, color: a.color, par: a.par, pts: a.pts })
}

/// Rows of `a` and `b` (each `w` × `h`) joined: every output row is a row of `a` then `b`'s.
fn join_rows<T: Copy>(a: &[T], b: &[T], w: usize, h: usize) -> Result<Vec<T>> {
    let n = w.checked_mul(h).ok_or_else(|| CodecError::Decode("lens picture too large".into()))?;
    let (Some(a), Some(b)) = (a.get(..n), b.get(..n)) else {
        return Err(CodecError::Decode("short lens picture plane".into()));
    };
    let mut out = Vec::with_capacity(n * 2);
    if w == 0 {
        return Ok(out);
    }
    for (ra, rb) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        out.extend_from_slice(ra);
        out.extend_from_slice(rb);
    }
    Ok(out)
}

/// Half the field of view of an Insta360 X-series lens, degrees: where the image circle's edge
/// is from the optical axis, read as an equidistant lens (the image-circle radius from the
/// calibration ↔ this angle). Measured on X5 footage, where the two lenses' pictures agree best
/// along the seam at 97.5° (the alignment in `filmcraft_render::vr::align_lenses` refines it per
/// clip).
pub const INSTA360_HALF_FOV: f64 = 97.5;

/// The dual-fisheye description of an Insta360 recording whose decoded picture is `size`: two
/// lenses side by side, in the camera's own order — a recording's two tracks are decoded so
/// ([`crate::mp4`]), its `.lrv` proxy holds them so in one picture. The image circles come from the
/// trailer's `offset` calibration, which measures the lenses side by side in a canvas of squares
/// (the circles of the decoded pictures were measured to match it, unrotated); without a usable one
/// the circles fill their halves.
pub(crate) fn insta360_spherical(info: &Info, size: (u32, u32)) -> Option<Spherical> {
    if size.0 == 0 || size.1 == 0 || size.0 != size.1.saturating_mul(2) {
        return None;
    }
    let calibrated = parse_offset(&info.offset).filter(|o| o.lenses.len() == 2 && o.canvas.1 > 0.0 && (o.canvas.0 - 2.0 * o.canvas.1).abs() < 1.0);
    let lens = |k: usize| -> FisheyeLens {
        let rect = [k as f64 * 0.5, 0.0, 0.5, 1.0];
        let (cx, cy, radius, roll) = match calibrated.as_ref().and_then(|o| Some((o.lenses.get(k)?, o.canvas.1))) {
            Some((l, side)) => ((l.cx - k as f64 * side) / side, l.cy / side, l.radius / side, 0.0),
            None => (0.5, 0.5, 0.5, 0.0),
        };
        // the left lens looks back (at the person holding the camera), the right one forward
        let yaw = if k == 0 { 180.0 } else { 0.0 };
        FisheyeLens { rect, cx, cy, radius, half_fov: INSTA360_HALF_FOV, poly: Vec::new(), yaw, pitch: 0.0, roll }
    };
    Some(Spherical::dual_fisheye(vec![lens(0), lens(1)], &info.camera))
}

/// Accelerometer counts a g of an Insta360 motion sensor (measured: 1026 at rest).
const ACCEL_PER_G: f64 = 1024.0;
/// Gyroscope counts a degree a second (measured against the accelerometer on X5 footage).
const GYRO_PER_DEGREE: f64 = 16.4;
/// Motion samples averaged into one before fusing (1 kHz → 200 a second).
pub const MOTION_MEAN: usize = 5;
/// Seconds between the samples of a horizon.
pub const HORIZON_STEP: f64 = 0.1;
/// How long the heading is smoothed over, seconds (a Gaussian's σ): turns slower than this are
/// the operator's and kept; the sway of a walk (a step or two a second) is taken out.
const STEADY_SECONDS: f64 = 0.5;
/// Largest turn the steadying makes, degrees: past it a quick turn of the camera is let through.
pub const MAX_STEADY: f64 = 20.0;
/// How long the accelerometer takes to pull the gyroscope's estimate of down back to what it
/// measures, seconds: the gyroscope follows quick turns, the accelerometer stops its drift.
const ACCEL_TRUST_SECONDS: f64 = 1.0;
/// Seconds of accelerometer readings that start each pass's estimate of down.
const SEED_SECONDS: f64 = 0.5;

type V3 = [f64; 3];

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(v: V3) -> Option<V3> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (n > 1e-9 && n.is_finite()).then(|| v.map(|c| c / n))
}

/// How much an accelerometer reading says about gravity: all of it at 1 g, nothing when the
/// camera is being pushed about (a quarter of a g off).
fn accel_weight(a: V3) -> f64 {
    let g = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() / ACCEL_PER_G;
    let off = (g - 1.0).abs();
    if !off.is_finite() {
        return 0.0;
    }
    (1.0 - (off - 0.1).max(0.0) / 0.15).clamp(0.0, 1.0)
}

fn accel(m: &Motion) -> V3 {
    m.accel.map(f64::from)
}

/// Down as the accelerometer measures it, averaged over the readings `idx` gives (weighted by
/// how still the camera was).
fn seed(motion: &[Motion], idx: impl Iterator<Item = usize>) -> Option<V3> {
    let mut acc = [0.0; 3];
    for i in idx {
        let Some(m) = motion.get(i) else { continue };
        let a = accel(m);
        let w = accel_weight(a);
        for c in 0..3 {
            acc[c] += a[c] * w;
        }
    }
    unit(acc)
}

/// One pass of the filter over `order` (sample indices, forwards or backwards): down carried by
/// the gyroscope's turning, pulled towards the accelerometer's reading.
fn pass(motion: &[Motion], order: impl Iterator<Item = usize>, start: V3, out: &mut [V3]) {
    let k = (1.0 / GYRO_PER_DEGREE).to_radians();
    let mut g = start;
    let mut prev: Option<(u64, V3)> = None;
    for i in order {
        let Some(m) = motion.get(i) else { continue };
        let w = m.gyro.map(|c| f64::from(c) * k);
        if let Some((us, w_prev)) = prev {
            // a direction fixed in the world turns against the camera: dg/dt = g × ω (backwards in
            // time with the later sample's turning)
            let (dt, w_used) = if m.us >= us { (m.us.saturating_sub(us) as f64 / 1e6, w) } else { (-(us.saturating_sub(m.us) as f64) / 1e6, w_prev) };
            let dt = dt.clamp(-0.1, 0.1);
            let d = cross(g, w_used);
            g = [g[0] + d[0] * dt, g[1] + d[1] * dt, g[2] + d[2] * dt];
            if let Some(a) = unit(accel(m)) {
                let pull = (dt.abs() / ACCEL_TRUST_SECONDS * accel_weight(accel(m))).clamp(0.0, 1.0);
                g = [g[0] + (a[0] - g[0]) * pull, g[1] + (a[1] - g[1]) * pull, g[2] + (a[2] - g[2]) * pull];
            }
            g = unit(g).unwrap_or(start);
        }
        if let Some(o) = out.get_mut(i) {
            *o = g;
        }
        prev = Some((m.us, w));
    }
}

/// Where the straight line fitted to `v` is at its first sample (`v` in reverse: at its last).
fn line_at_start(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = v.clone().count() as f64;
    let (sx, sy) = v.clone().enumerate().fold((0.0, 0.0), |(sx, sy), (i, y)| (sx + i as f64, sy + y));
    let (mx, my) = (sx / n.max(1.0), sy / n.max(1.0));
    let (sxy, sxx) = v.enumerate().fold((0.0, 0.0), |(sxy, sxx), (i, y)| (sxy + (i as f64 - mx) * (y - my), sxx + (i as f64 - mx).powi(2)));
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let at = my - slope * mx;
    if at.is_finite() { at } else { 0.0 }
}

/// The Gaussian smoothing of `v` with σ `sigma` samples. Past its ends it is mirrored about the
/// line fitted to the last four σ of them, so a steady turn stays a turn up to the end and a sway
/// there is still taken out.
fn smoothed(v: &[f64], sigma: f64) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return v.to_vec();
    }
    let fit = ((4.0 * sigma).ceil() as usize).saturating_add(1).max(2).min(n);
    let a = line_at_start(v.iter().take(fit).copied());
    let z = line_at_start(v.iter().rev().take(fit).copied());
    let r = (3.0 * sigma).ceil().min(1000.0) as usize;
    let kernel: Vec<f64> = (0..=2 * r).map(|k| (-0.5 * ((k as f64 - r as f64) / sigma).powi(2)).exp()).collect();
    let total: f64 = kernel.iter().sum();
    let last = n - 1;
    let at = |j: isize| -> f64 {
        if j < 0 {
            2.0 * a - v.get(j.unsigned_abs().min(last)).copied().unwrap_or(a)
        } else if j as usize > last {
            2.0 * z - v.get((2 * last).saturating_sub(j as usize)).copied().unwrap_or(z)
        } else {
            v.get(j as usize).copied().unwrap_or(z)
        }
    };
    (0..n).map(|i| kernel.iter().enumerate().map(|(k, w)| w * at(i as isize + k as isize - r as isize)).sum::<f64>() / total).collect()
}

/// Which way was up through an Insta360 recording, and how to steady its heading, from its motion
/// sensor (`motion`, in the camera's order, as [`filmcraft_isobmff::insta360::read_motion`] gives
/// it) and the time its first picture's exposure started and lasted (`first`, microseconds and
/// seconds; the first sample's time without one): a sample every [`HORIZON_STEP`] seconds of the
/// recording.
///
/// The gyroscope and accelerometer are fused (a complementary filter) forwards and backwards in
/// time and the two averaged, so a turn is followed without lag; each sample is the average over
/// its step. The sensor's axes were matched to the camera's directions on X5 footage by levelling
/// its panoramas (a lounge, a hotel lobby, an escalator): the camera's +X, +Y and +Z are the
/// sensor's −third, first and −second axes, along which the accelerometer measures down.
///
/// The heading is the gyroscope's turning about that down, added up; smoothed over
/// [`STEADY_SECONDS`], what is left is the swing to take out (at most [`MAX_STEADY`]). Its sign
/// was measured, not derived: on X5 footage walked with, turning the picture back the way the
/// derivation says doubled the sway (a 24 px swing in a 135 px wide view became 41), the other
/// way it shrank to 16 px, and the pictures' own frame-to-frame turning (`clip.stabilize360`'s
/// measurement) agreed with it (0.86 correlated, 0.89 of its size). The time between the sensor
/// and the pictures differs from recording to recording by up to five frames, so the samples
/// stay a tenth of a second apart: the sway (a step or two a second) is followed, the jolts
/// between are left to `clip.stabilize360`.
pub fn insta360_horizon(motion: &[Motion], first: Option<(u64, f64)>) -> Option<Horizon> {
    let (t0, t_last) = (motion.first()?.us, motion.last()?.us);
    let n = motion.len();
    let within = |from: u64, secs: f64| motion.iter().take_while(move |m| (m.us.saturating_sub(from) as f64) < secs * 1e6).count();
    let head = within(t0, SEED_SECONDS).max(1);
    let tail = motion.iter().rev().take_while(|m| (t_last.saturating_sub(m.us) as f64) < SEED_SECONDS * 1e6).count().max(1);
    let (start_f, start_b) = (seed(motion, 0..head)?, seed(motion, n.saturating_sub(tail)..n)?);
    let mut fwd = vec![start_f; n];
    let mut bwd = vec![start_b; n];
    pass(motion, 0..n, start_f, &mut fwd);
    pass(motion, (0..n).rev(), start_b, &mut bwd);
    let down: Vec<V3> = fwd.iter().zip(&bwd).map(|(a, b)| unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]]).unwrap_or(*a)).collect();
    // the heading, degrees to the right: the gyroscope's turning about the accelerometer's axis
    // (the sign measured on footage)
    let mut heading = Vec::with_capacity(n);
    let mut psi = 0.0f64;
    let mut prev_us: Option<u64> = None;
    for (m, d) in motion.iter().zip(&down) {
        if let Some(us) = prev_us {
            let dt = (m.us.saturating_sub(us) as f64 / 1e6).clamp(0.0, 0.1);
            let w = m.gyro.map(|c| f64::from(c) / GYRO_PER_DEGREE);
            let rate = w[0] * d[0] + w[1] * d[1] + w[2] * d[2];
            if rate.is_finite() {
                psi += rate.clamp(-3600.0, 3600.0) * dt;
            }
        }
        heading.push(psi);
        prev_us = Some(m.us);
    }
    // the middle of the first picture's exposure is the recording's time zero
    let zero = match first {
        Some((us, exposure)) => us.saturating_add((exposure.clamp(0.0, 1.0) * 5e5) as u64),
        None => t0,
    };
    let span = t_last.saturating_sub(zero) as f64 / 1e6;
    let step = HORIZON_STEP;
    let count = ((span / step).floor() as usize).saturating_add(1).min(filmcraft_media::spherical::MAX_HORIZON);
    let mut up = Vec::with_capacity(count);
    let mut turned = Vec::with_capacity(count);
    let mut lo = 0usize;
    for j in 0..count {
        let mid = zero as f64 + j as f64 * step * 1e6;
        let (a, b) = (mid - step * 5e5, mid + step * 5e5);
        while lo < n && motion.get(lo).is_some_and(|m| (m.us as f64) < a) {
            lo += 1;
        }
        let mut sum = [0.0; 3];
        let (mut psi_sum, mut psi_n) = (0.0, 0usize);
        let mut hi = lo;
        while motion.get(hi).is_some_and(|m| (m.us as f64) < b) {
            if let Some(d) = down.get(hi) {
                for c in 0..3 {
                    sum[c] += d[c];
                }
            }
            if let Some(p) = heading.get(hi) {
                psi_sum += p;
                psi_n += 1;
            }
            hi += 1;
        }
        // (no sample in the step: the nearest one)
        let near = lo.min(n.saturating_sub(1));
        let d = unit(sum).or_else(|| down.get(near).copied()).unwrap_or([0.0, -1.0, 0.0]);
        // sensor → camera directions, and up is the other way from down
        let u = [d[2], -d[0], d[1]];
        up.push(u.map(|c| ((c * 1000.0).round() / 1000.0) as f32));
        turned.push(if psi_n > 0 { psi_sum / psi_n as f64 } else { heading.get(near).copied().unwrap_or(0.0) });
    }
    // the swing about the smoothed heading, turned back
    let smooth = smoothed(&turned, STEADY_SECONDS / step);
    let heading = turned.iter().zip(&smooth).map(|(t, s)| (((t - s).clamp(-MAX_STEADY, MAX_STEADY) * 100.0).round() / 100.0) as f32).collect();
    (!up.is_empty()).then_some(Horizon { level: true, steady: true, step, up, heading })
}

/// The spherical metadata of a stitched video track: V2 (`sv3d`, `st3d`) first, then V1. Only
/// equirectangular projections are described; a cube map or mesh is not (yet) understood, and
/// is treated as a flat picture.
pub(crate) fn track_spherical(t: &Track) -> Option<Spherical> {
    if let Some(v) = t.video()
        && let Some(sv) = &v.spherical
    {
        let Some(Projection::Equirect { top, bottom, left, right }) = sv.projection else { return None };
        let mut s = Spherical::equirect();
        s.layout = match v.stereo {
            Some(StereoMode::TopBottom) => StereoLayout::OverUnder,
            Some(StereoMode::LeftRight) => StereoLayout::SideBySide,
            _ => StereoLayout::Mono,
        };
        // bounds are 0.32 fixed-point shares of the sphere cut off each side
        let share = |v: u32| f64::from(v) / 4_294_967_296.0;
        s.captured = [360.0 * (1.0 - share(left) - share(right)).clamp(0.0, 1.0), 180.0 * (1.0 - share(top) - share(bottom)).clamp(0.0, 1.0)];
        s.pose = sv.pose;
        s.camera = sv.source.clone();
        return Some(s.sanitized());
    }
    let v1 = t.spherical_v1.as_ref().filter(|v| v.spherical)?;
    if !(v1.projection.is_empty() || v1.projection == "equirectangular") {
        return None;
    }
    let mut s = Spherical::equirect();
    s.layout = match v1.stereo.as_str() {
        "top-bottom" => StereoLayout::OverUnder,
        "left-right" => StereoLayout::SideBySide,
        _ => StereoLayout::Mono,
    };
    if let Some([fw, fh, cw, ch, _, _]) = v1.cropped
        && fw > 0
        && fh > 0
    {
        s.captured = [360.0 * cw as f64 / fw as f64, 180.0 * ch as f64 / fh as f64];
    }
    s.pose = v1.initial_view;
    s.camera = v1.stitching_software.clone();
    Some(s.sanitized())
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_frame::Chroma;
    use std::sync::Arc;

    fn yuv(w: u32, h: u32, y: u8) -> VideoFrame {
        let (cw, ch) = (w as usize / 2, (h as usize).div_ceil(2));
        VideoFrame {
            width: w,
            height: h,
            data: PixelData::Yuv8 {
                planes: [Arc::new(vec![y; (w * h) as usize]), Arc::new(vec![1; cw * ch]), Arc::new(vec![2; cw * ch])],
                chroma: Chroma::C420,
                alpha: None,
            },
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            pts: filmcraft_time::Tick::ZERO,
        }
    }

    #[test]
    fn lenses_join_side_by_side() {
        let f = side_by_side(&yuv(4, 3, 10), &yuv(4, 3, 200)).unwrap();
        assert_eq!((f.width, f.height), (8, 3));
        let PixelData::Yuv8 { planes, .. } = &f.data else { panic!() };
        assert_eq!(&planes[0][..8], &[10, 10, 10, 10, 200, 200, 200, 200]);
        assert_eq!(planes[1].len(), 4 * 2);
    }

    #[test]
    fn mismatched_lenses_are_an_error() {
        assert!(side_by_side(&yuv(4, 2, 0), &yuv(6, 2, 0)).is_err());
        assert!(side_by_side(&yuv(3, 2, 0), &yuv(3, 2, 0)).is_err());
        let mut short = yuv(4, 2, 0);
        short.data = PixelData::Yuv8 { planes: [Arc::new(vec![0; 3]), Arc::new(vec![0; 2]), Arc::new(vec![0; 2])], chroma: Chroma::C420, alpha: None };
        assert!(side_by_side(&short, &yuv(4, 2, 0)).is_err());
    }

    /// The sensor's reading of down for a camera whose up (its own directions) is `u`.
    fn sensor_down(u: V3) -> V3 {
        [-u[1], u[2], u[0]]
    }

    /// `v` turned about unit `axis` by `angle` radians.
    fn turned(v: V3, axis: V3, angle: f64) -> V3 {
        let (s, c) = angle.sin_cos();
        let k = cross(axis, v);
        let d = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
        std::array::from_fn(|i| v[i] * c + k[i] * s + axis[i] * d * (1.0 - c))
    }

    /// A recording's motion at 1 kHz for `secs` from `t0` µs: down starts at `down0` (sensor axes)
    /// and the camera turns at `omega` degrees a second (sensor axes) between `from` and `to`
    /// seconds; `push` adds a linear acceleration (g, sensor axes) over a time span.
    fn recording(t0: u64, secs: f64, down0: V3, omega: V3, from: f64, to: f64, push: Option<(f64, f64, V3)>) -> Vec<Motion> {
        let w = omega.map(f64::to_radians);
        let speed = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        let axis = unit(w).unwrap_or([0.0, 0.0, 1.0]);
        (0..(secs * 1000.0) as u64)
            .map(|k| {
                let t = k as f64 / 1000.0;
                let turning = t > from && t <= to;
                // a direction fixed in the world turns against the camera
                let d = turned(down0, axis, -speed * (t.clamp(from, to) - from));
                let mut a = d.map(|c| c * ACCEL_PER_G);
                if let Some((p0, p1, g)) = push
                    && t >= p0
                    && t < p1
                {
                    a = std::array::from_fn(|i| a[i] + g[i] * ACCEL_PER_G);
                }
                let gyro = if turning { omega.map(|c| (c * GYRO_PER_DEGREE) as f32) } else { [0.0; 3] };
                Motion { us: t0.saturating_add(k * 1000), accel: a.map(|c| c.round() as f32), gyro }
            })
            .collect()
    }

    fn averaged(m: &[Motion]) -> Vec<Motion> {
        m.chunks(MOTION_MEAN)
            .map(|g| {
                let n = g.len() as f32;
                Motion {
                    us: g[0].us + (g[g.len() - 1].us - g[0].us) / 2,
                    accel: std::array::from_fn(|c| g.iter().map(|s| s.accel[c]).sum::<f32>() / n),
                    gyro: std::array::from_fn(|c| g.iter().map(|s| s.gyro[c]).sum::<f32>() / n),
                }
            })
            .collect()
    }

    fn degrees(a: [f32; 3], b: V3) -> f64 {
        let a = unit(a.map(f64::from)).unwrap();
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos().to_degrees()
    }

    #[test]
    fn a_still_tilted_camera_has_a_steady_horizon() {
        let up = [30f64.to_radians().sin(), 30f64.to_radians().cos(), 0.0];
        let m = averaged(&recording(5_000_000, 3.0, sensor_down(up), [0.0; 3], 0.0, 0.0, None));
        let h = insta360_horizon(&m, None).unwrap();
        assert_eq!((h.step, h.level), (HORIZON_STEP, true));
        assert!((h.up.len() as i64 - 30).abs() <= 1, "{}", h.up.len());
        assert!(h.up.iter().all(|u| degrees(*u, up) < 0.2), "{:?}", h.up.first());
        assert!((h.max_tilt() - 30.0).abs() < 0.2);
        assert!(insta360_horizon(&[], None).is_none());
    }

    #[test]
    fn a_turning_camera_is_followed_without_lag_and_a_push_is_ignored() {
        // upright, then rolled 90° about the lens's axis in one second (the camera's +Z is the
        // sensor's −second axis), held; a hard shove at 2.4 s
        let down0 = sensor_down([0.0, 1.0, 0.0]);
        let m = averaged(&recording(0, 4.0, down0, [0.0, 90.0, 0.0], 1.0, 2.0, Some((2.4, 2.7, [0.0, 0.0, 0.6]))));
        let h = insta360_horizon(&m, None).unwrap();
        let truth = |t: f64| {
            let d = turned(down0, [0.0, 1.0, 0.0], -(t.clamp(1.0, 2.0) - 1.0) * std::f64::consts::FRAC_PI_2);
            [d[2], -d[0], d[1]]
        };
        for (j, u) in h.up.iter().enumerate() {
            let t = j as f64 * HORIZON_STEP;
            // (each sample is the average over its step: a turn of 9° a step is a little behind
            // or ahead at its corners)
            assert!(degrees(*u, truth(t)) < 2.5, "{t}s: {u:?} vs {:?}", truth(t));
        }
        assert!(degrees(h.up[15], truth(1.5)) < 1.0);
        assert!((h.max_tilt() - 90.0).abs() < 1.5);
    }

    #[test]
    fn the_horizon_starts_at_the_first_picture() {
        // the motion starts a second before the first picture; the camera rolls 90° in the half
        // second after it
        let m = averaged(&recording(1_000_000, 3.0, sensor_down([0.0, 1.0, 0.0]), [0.0, 90.0 / 0.5, 0.0], 1.0, 1.5, None));
        let tilt = |u: [f32; 3]| degrees(u, [0.0, 1.0, 0.0]);
        let h = insta360_horizon(&m, Some((1_995_000, 0.01))).unwrap();
        assert!((h.up.len() as i64 - 20).abs() <= 1, "{}", h.up.len());
        // (the first sample averages the turn's first 50 ms: 9°, half of the time)
        assert!(tilt(h.up[0]) < 3.0 && (tilt(h.up[6]) - 90.0).abs() < 2.0, "{} {}", tilt(h.up[0]), tilt(h.up[6]));
        // without exposure times: from the first sample
        let h = insta360_horizon(&m, None).unwrap();
        assert!((h.up.len() as i64 - 30).abs() <= 1);
        assert!(tilt(h.up[9]) < 2.0 && (tilt(h.up[16]) - 90.0).abs() < 2.0);
    }

    #[test]
    fn a_walk_is_steadied_and_its_turn_kept() {
        // upright, swaying 6° either way once a second, and turning 90° between 4 and 6 s
        let sway = |t: f64| 6.0 * (std::f64::consts::TAU * t).sin();
        let turn = |t: f64| {
            let x = ((t - 4.0) / 2.0).clamp(0.0, 1.0);
            90.0 * x * x * (3.0 - 2.0 * x)
        };
        let heading = |t: f64| sway(t) + turn(t);
        let down = sensor_down([0.0, 1.0, 0.0]);
        let raw: Vec<Motion> = (0..10_000u64)
            .map(|k| {
                let t = k as f64 / 1000.0;
                // (about the vertical: the sensor's first axis, as footage measured it)
                let rate = (heading(t + 0.0005) - heading(t - 0.0005)) / 0.001;
                Motion { us: k * 1000, accel: down.map(|c| (c * ACCEL_PER_G) as f32), gyro: [(-rate * GYRO_PER_DEGREE) as f32, 0.0, 0.0] }
            })
            .collect();
        let h = insta360_horizon(&averaged(&raw), None).unwrap();
        assert!(h.steady && h.heading.len() == h.up.len() && (h.up.len() as i64 - 100).abs() <= 1, "{}", h.up.len());
        for (j, d) in h.heading.iter().enumerate() {
            let t = j as f64 * h.step;
            let d = f64::from(*d);
            assert!(d.abs() <= MAX_STEADY);
            // the steadied heading (the camera's, less the turn back) does not sway ...
            let steadied = heading(t) - d;
            if (1.5..3.0).contains(&t) {
                assert!((d - sway(t)).abs() < 0.6 && steadied.abs() < 0.6, "{t}s: turned back {d}, the sway was {}", sway(t));
            }
            // ... and still makes the turn
            if (7.5..8.5).contains(&t) {
                assert!((d - sway(t)).abs() < 0.6 && (steadied - 90.0).abs() < 0.6, "{t}s: {steadied}");
            }
            // (the first and last second and a half, smoothed with what their ends mirror, sway
            // less than half as much)
            if t < 1.5 || t >= 8.5 {
                assert!((steadied - turn(t)).abs() < 2.5, "{t}s: {steadied}");
            }
        }
        // a steady camera is not turned at all
        let still = recording(0, 3.0, down, [0.0; 3], 0.0, 0.0, None);
        assert!(insta360_horizon(&averaged(&still), None).unwrap().heading.iter().all(|d| *d == 0.0));
        // the smoothing: nothing to smooth, no width, a steady turn kept up to its ends
        assert_eq!(smoothed(&[], 3.0), Vec::<f64>::new());
        assert_eq!(smoothed(&[1.0, 2.0, 3.0], f64::NAN), vec![1.0, 2.0, 3.0]);
        let ramp: Vec<f64> = (0..50).map(f64::from).collect();
        assert!(smoothed(&ramp, 5.0).iter().zip(&ramp).all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn hostile_motion_never_panics() {
        let mut m = recording(u64::MAX - 5_000, 1.0, [f64::NAN, 0.0, 0.0], [1e30, 0.0, f64::NAN], 0.0, 1.0, None);
        m.push(Motion { us: 0, accel: [0.0; 3], gyro: [f32::INFINITY; 3] });
        m.push(Motion { us: u64::MAX, accel: [f32::NAN; 3], gyro: [f32::NAN; 3] });
        let _ = insta360_horizon(&m, Some((u64::MAX, f64::NAN)));
        let _ = insta360_horizon(&m, Some((0, 1e300)));
        let one = [Motion { us: 7, accel: [0.0, -1024.0, 0.0], gyro: [0.0; 3] }];
        assert_eq!(insta360_horizon(&one, None).map(|h| h.up.len()), Some(1));
    }

    #[test]
    fn insta360_metadata_describes_two_lenses() {
        let info = Info {
            camera: "Insta360 X5".into(),
            offset: "2_2657.856_2681.350_2673.940_0.249_-0.624_89.769_2655.921_8091.830_2669.660_-0.436_-0.465_90.327_10752_5376_1137".into(),
            ..Default::default()
        };
        let s = insta360_spherical(&info, (5760, 2880)).unwrap();
        let filmcraft_media::spherical::Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        assert_eq!(lenses.len(), 2);
        assert!((lenses[1].cx - (8091.83 - 5376.0) / 5376.0).abs() < 1e-9);
        assert_eq!(s.camera, "Insta360 X5");
        // not a 2:1 picture: not described
        assert!(insta360_spherical(&info, (2880, 2880)).is_none());
        // no calibration: circles fill their halves
        let s = insta360_spherical(&Info::default(), (1664, 832)).unwrap();
        let filmcraft_media::spherical::Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        assert_eq!((lenses[0].cx, lenses[0].radius), (0.5, 0.5));
    }
}
