//! How a 360° camera turned, from its pictures (stabilisation), and smooth paths for the virtual
//! camera.
//!
//! A 360° camera sees every direction, so between two frames the scene's far-away directions all
//! turn by the camera's own rotation (its small movement makes near things shift too: those are
//! outliers). The method (textbook, implemented from the papers):
//!
//! 1. **Views** — six cube faces of each frame ([`CUBE`], 100° across so they overlap), rendered
//!    small through [`crate::vr`] in the camera's own frame.
//! 2. **Points** — Shi–Tomasi corners in each face of frame A, followed into the same face of
//!    frame B with pyramidal Lucas–Kanade forward and back ([`crate::track`]); each kept pair is
//!    a pair of directions (the face's ray through each point).
//! 3. **Rotation** — RANSAC (Fischler & Bolles 1981) over pairs of point pairs, each solved by
//!    Horn's closed form (Horn 1987, the quaternion of the largest eigenvalue of his 4 × 4
//!    matrix, found here by Jacobi rotations), inliers within [`INLIER_DEGREES`], then refit on
//!    all inliers.
//! 4. **Path** — the camera's orientation chained frame to frame, then smoothed ([`smooth`]: a
//!    Gaussian over unit quaternions kept in one hemisphere, renormalised), and the correction
//!    that holds the view to the smooth path ([`corrections`]).

use crate::image::Image;
use crate::vr::{M3, SourceModel, V3, View, dot, mmul, mvec, transpose};

/// Inlier threshold of the rotation fit, degrees.
pub const INLIER_DEGREES: f64 = 0.35;
/// RANSAC samples per frame pair.
pub const RANSAC_ITERS: usize = 200;
/// Fewest inliers a rotation is accepted on.
pub const MIN_INLIERS: usize = 12;

/// A unit quaternion (w, x, y, z).
pub type Quat = [f64; 4];

pub fn quat_from_matrix(m: &M3) -> Quat {
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).max(0.0).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).max(0.0).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).max(0.0).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    qnormalize(q)
}

pub fn matrix_from_quat(q: Quat) -> M3 {
    let [w, x, y, z] = qnormalize(q);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn qnormalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 && n.is_finite() { [q[0] / n, q[1] / n, q[2] / n, q[3] / n] } else { [1.0, 0.0, 0.0, 0.0] }
}

/// The eigenvector of the largest eigenvalue of a symmetric 4 × 4 matrix (cyclic Jacobi).
fn top_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let off: f64 = (0..4).flat_map(|i| (0..4).filter(move |j| *j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let best = (0..4).max_by(|i, j| a[*i][*i].total_cmp(&a[*j][*j])).unwrap_or(0);
    [v[0][best], v[1][best], v[2][best], v[3][best]]
}

/// The rotation R that best turns every `a` into its `b` (weighted least squares on unit
/// vectors; Horn's closed form). `None` without at least two pairs.
pub fn fit_rotation(pairs: &[(V3, V3)]) -> Option<M3> {
    if pairs.len() < 2 {
        return None;
    }
    // H = Σ a bᵀ
    let mut h = [[0.0f64; 3]; 3];
    for (a, b) in pairs {
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += a[i] * b[j];
            }
        }
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2]);
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    if n.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    Some(matrix_from_quat(top_eigenvector(n)))
}

/// The angle between two directions, radians (atan2 of the cross and dot products: accurate for
/// the tiny angles inliers make, where acos of the dot product is not).
fn angle(a: V3, b: V3) -> f64 {
    let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    dot(c, c).sqrt().atan2(dot(a, b))
}

/// A small deterministic generator (the fit is the same every run).
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n.max(1) as u64) as usize
    }
}

/// The rotation most direction pairs agree on, with how many agree ([`INLIER_DEGREES`]).
pub fn robust_rotation(pairs: &[(V3, V3)]) -> Option<(M3, usize)> {
    if pairs.len() < MIN_INLIERS {
        return None;
    }
    let thr = INLIER_DEGREES.to_radians();
    let count = |r: &M3| pairs.iter().filter(|(a, b)| angle(mvec(r, *a), *b) < thr).count();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut best: Option<(M3, usize)> = None;
    for _ in 0..RANSAC_ITERS {
        let (i, j) = (rng.below(pairs.len()), rng.below(pairs.len()));
        let (Some(p), Some(q)) = (pairs.get(i), pairs.get(j)) else { continue };
        // two directions that are not (nearly) parallel pin a rotation down
        if i == j || angle(p.0, q.0) < 0.05 {
            continue;
        }
        let Some(r) = fit_rotation(&[*p, *q]) else { continue };
        let n = count(&r);
        if best.as_ref().is_none_or(|b| n > b.1) {
            best = Some((r, n));
        }
    }
    let (r, _) = best?;
    let inliers: Vec<(V3, V3)> = pairs.iter().copied().filter(|(a, b)| angle(mvec(&r, *a), *b) < thr).collect();
    if inliers.len() < MIN_INLIERS {
        return None;
    }
    let refit = fit_rotation(&inliers)?;
    let n = count(&refit);
    Some((refit, n))
}

/// Gaussian smoothing (σ = `sigma` seconds) of orientations sampled at `times`: unit quaternions
/// averaged (all turned into the hemisphere of the one being smoothed) and renormalised.
pub fn smooth(orient: &[M3], times: &[f64], sigma: f64) -> Vec<M3> {
    let n = orient.len().min(times.len());
    if n == 0 || sigma.is_nan() || sigma <= 0.0 || !sigma.is_finite() {
        return orient.to_vec();
    }
    let q: Vec<Quat> = orient.iter().take(n).map(quat_from_matrix).collect();
    let reach = 3.0 * sigma;
    (0..n)
        .map(|i| {
            let (Some(qi), Some(ti)) = (q.get(i), times.get(i)) else { return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
            let mut acc = [0.0; 4];
            // the neighbours within three sigma (times are sorted)
            let lo = times.partition_point(|t| *t < ti - reach);
            let hi = times.partition_point(|t| *t <= ti + reach).min(n);
            for k in lo..hi {
                let (Some(qk), Some(tk)) = (q.get(k), times.get(k)) else { continue };
                let w = (-(tk - ti).powi(2) / (2.0 * sigma * sigma)).exp();
                let s = if qk[0] * qi[0] + qk[1] * qi[1] + qk[2] * qi[2] + qk[3] * qi[3] < 0.0 { -w } else { w };
                for c in 0..4 {
                    acc[c] += s * qk[c];
                }
            }
            matrix_from_quat(acc)
        })
        .collect()
}

/// The turn that holds a camera of orientation `orient[k]` to `target[k]`: a view's world rays
/// are turned by it before they are looked up in the frame (Cₖ = Rₖᵀ · Tₖ).
pub fn corrections(orient: &[M3], target: &[M3]) -> Vec<M3> {
    orient.iter().zip(target).map(|(r, t)| mmul(&transpose(r), t)).collect()
}

/// The six cube faces, 100° across (overlapping), front, right, back, left, up, down.
pub fn cube_views(side: u32) -> [View; 6] {
    let v = |yaw: f64, pitch: f64| View::flat(yaw, pitch, 0.0, 100.0, 0.0, side, side);
    [v(0.0, 0.0), v(90.0, 0.0), v(180.0, 0.0), v(-90.0, 0.0), v(0.0, 89.9), v(0.0, -89.9)]
}

/// A face of a 360° picture as grey levels (display-referred luma), `side` × `side`.
pub fn face_gray(src: &Image, model: &SourceModel, view: &View, side: usize) -> crate::track::Gray {
    let img = crate::vr::reframe(src, model, view, side, side, 1);
    let px = img
        .px
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let a = p[3].max(1e-6);
            let l = 0.2126 * p[0] / a + 0.7152 * p[1] / a + 0.0722 * p[2] / a;
            filmcraft_color::linear_to_srgb(l.clamp(0.0, 1.0)) * p[3].min(1.0)
        })
        .collect();
    crate::track::Gray::new(side, side, px)
}

/// One frame ready for rotation tracking: its cube faces' pyramids.
pub struct Faces {
    side: usize,
    views: [View; 6],
    faces: Vec<crate::track::Prepared>,
}

impl Faces {
    pub fn new(src: &Image, model: &SourceModel, side: usize) -> Self {
        let views = cube_views(side as u32);
        let faces = views.iter().map(|v| crate::track::Prepared::new(face_gray(src, model, v, side))).collect();
        Faces { side, views, faces }
    }
}

/// Most corners a face contributes.
const FACE_FEATURES: usize = 60;

/// The rotation ΔR with b ≈ ΔR · a for the directions the camera saw in frame `a` and `b`
/// (camera coordinates), with its inlier count; `None` when too few points agree.
pub fn rotation_between(a: &Faces, b: &Faces) -> Option<(M3, usize)> {
    if a.side != b.side {
        return None;
    }
    let s = a.side as f64;
    // the inner part of each face (its edges are other faces' middles)
    let m = s * 0.08;
    let region =
        [filmcraft_geom::Vec2::new(m, m), filmcraft_geom::Vec2::new(s - m, m), filmcraft_geom::Vec2::new(s - m, s - m), filmcraft_geom::Vec2::new(m, s - m)];
    let mut pairs: Vec<(V3, V3)> = Vec::new();
    for (k, (fa, fb)) in a.faces.iter().zip(&b.faces).enumerate() {
        let (Some(va), Some(vb)) = (a.views.get(k), b.views.get(k)) else { continue };
        for p in crate::track::features(fa.base(), &region, FACE_FEATURES, 6.0) {
            let Some(q) = crate::track::lk(&fa.levels, &fb.levels, p) else { continue };
            let Some(back) = crate::track::lk(&fb.levels, &fa.levels, q) else { continue };
            if ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2)).sqrt() > 0.5 {
                continue;
            }
            if !(0.0..s as f32).contains(&q[0]) || !(0.0..s as f32).contains(&q[1]) {
                continue;
            }
            let da = va.ray((f64::from(p[0]) + 0.5) / s, (f64::from(p[1]) + 0.5) / s);
            let db = vb.ray((f64::from(q[0]) + 0.5) / s, (f64::from(q[1]) + 0.5) / s);
            pairs.push((da, db));
        }
    }
    robust_rotation(&pairs)
}

/// Field of view of the tracking camera, degrees across.
pub const TRACK_FOV: f64 = 110.0;
/// The tracking camera turns to the subject once it is this far from the middle (a share of half
/// the frame).
const RECENTRE: f64 = 0.35;

/// A wide camera kept on the subject while the tracker follows it (`filmcraft_render::subject`
/// works on flat pictures: this gives it one wherever the subject goes around the sphere).
pub struct TrackingCamera {
    pub view: View,
    pub size: (u32, u32),
}

impl TrackingCamera {
    pub fn new(yaw: f64, pitch: f64, size: (u32, u32)) -> Self {
        let pitch = pitch.clamp(-75.0, 75.0);
        TrackingCamera { view: View::flat(yaw, pitch, 0.0, TRACK_FOV, 0.0, size.0, size.1), size }
    }

    /// The direction the middle of box (`cx`, `cy` shares of the frame) is in.
    pub fn direction(&self, cx: f64, cy: f64) -> V3 {
        self.view.ray(cx, cy)
    }

    /// Whether a box centred at (`cx`, `cy`) has drifted far enough that the camera should turn.
    pub fn should_recentre(&self, cx: f64, cy: f64) -> bool {
        (cx - 0.5).abs() > RECENTRE / 2.0 || (cy - 0.5).abs() > RECENTRE / 2.0
    }

    /// The camera turned onto direction `d`; a box of `w` × `h` (shares) seen at `d` before is the
    /// same size in the new picture (same field of view), centred.
    pub fn recentred(&self, d: V3) -> Self {
        let (lon, lat) = crate::vr::lonlat(d);
        TrackingCamera::new(lon.to_degrees(), lat.to_degrees(), self.size)
    }
}

/// Unwrapped angles: each within 180° of the one before (a turn through ±180° stays continuous).
pub fn unwrap_degrees(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    let mut prev: Option<f64> = None;
    for &x in v {
        let y = match prev {
            Some(p) if x.is_finite() => {
                let mut y = x;
                while y - p > 180.0 {
                    y -= 360.0;
                }
                while p - y > 180.0 {
                    y += 360.0;
                }
                y
            }
            _ => x,
        };
        prev = Some(y);
        out.push(y);
    }
    out
}

/// Critically damped smoothing (ω in rad/s) run forwards then backwards (zero phase), for samples
/// at `times`; values not finite are filled from their neighbours first.
pub fn smooth_path(times: &[f64], values: &[f64], omega: f64) -> Vec<f64> {
    let n = times.len().min(values.len());
    if n == 0 {
        return Vec::new();
    }
    // gaps: linear between the known samples, held at the ends
    let mut v: Vec<f64> = values.iter().take(n).copied().collect();
    let known: Vec<usize> = (0..n).filter(|&i| v[i].is_finite()).collect();
    if known.is_empty() {
        return vec![0.0; n];
    }
    for i in 0..n {
        if v[i].is_finite() {
            continue;
        }
        let after = known.iter().copied().find(|&k| k > i);
        let before = known.iter().copied().rev().find(|&k| k < i);
        v[i] = match (before, after) {
            (Some(a), Some(b)) => {
                let t = (times[i] - times[a]) / (times[b] - times[a]).max(1e-9);
                v[a] + (v[b] - v[a]) * t
            }
            (Some(a), None) => v[a],
            (None, Some(b)) => v[b],
            (None, None) => 0.0,
        };
    }
    if omega.is_nan() || omega <= 0.0 || !omega.is_finite() {
        return v;
    }
    let pass = |xs: &[f64], ts: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(xs.len());
        let (mut x, mut dx) = (xs[0], 0.0f64);
        for i in 0..xs.len() {
            let dt = if i == 0 { 0.0 } else { (ts[i] - ts[i - 1]).abs().min(1.0) };
            // the exact step of x'' = ω²(target − x) − 2ωx' over dt, target held
            let e = (-omega * dt).exp();
            let y0 = x - xs[i];
            let c = dx + omega * y0;
            let y = (y0 + c * dt) * e;
            dx = (c - omega * (y0 + c * dt)) * e;
            x = xs[i] + y;
            out.push(x);
        }
        out
    };
    let f = pass(&v, &times[..n]);
    let rt: Vec<f64> = times[..n].iter().rev().copied().collect();
    let mut b = pass(&f.iter().rev().copied().collect::<Vec<_>>(), &rt);
    b.reverse();
    b
}

/// The fewest of `points` (time, value…) that redraw them within `tolerance` with straight
/// lines between (Ramer–Douglas–Peucker); the first and last always kept. Indices, sorted.
pub fn reduce(times: &[f64], series: &[&[f64]], tolerance: f64) -> Vec<usize> {
    let n = times.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ta, tb) = (times[a], times[b]);
        let mut worst = (0.0f64, a);
        for i in a + 1..b {
            let t = if tb > ta { (times[i] - ta) / (tb - ta) } else { 0.0 };
            let err = series.iter().map(|s| (s[i] - (s[a] + (s[b] - s[a]) * t)).abs()).fold(0.0, f64::max);
            if err > worst.0 {
                worst = (err, i);
            }
        }
        if worst.0 > tolerance {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vr::camera_rotation;

    fn close(a: &M3, b: &M3, eps: f64) -> bool {
        (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() < eps))
    }

    #[test]
    fn quaternions_round_trip() {
        for (y, p, r) in [(0.0, 0.0, 0.0), (30.0, -20.0, 10.0), (179.0, 80.0, -170.0), (-90.0, 45.0, 90.0)] {
            let m = camera_rotation(y, p, r);
            assert!(close(&matrix_from_quat(quat_from_matrix(&m)), &m, 1e-9), "{y} {p} {r}");
        }
    }

    #[test]
    fn rotations_are_recovered_from_noisy_pairs_with_outliers() {
        let truth = camera_rotation(3.0, -1.5, 0.7);
        let mut pairs = Vec::new();
        let mut rng = Rng(42);
        for k in 0..300 {
            let lon = (k as f64 * 0.37).sin() * 3.0;
            let lat = (k as f64 * 0.11).cos() * 1.2;
            let a = crate::vr::dir(lon, lat);
            let mut b = mvec(&truth, a);
            if k % 5 == 0 {
                // a near thing moving on its own
                b = crate::vr::dir(lon + 0.2 + rng.below(100) as f64 / 500.0, lat);
            } else {
                b[0] += (rng.below(100) as f64 - 50.0) * 1e-6;
            }
            pairs.push((a, b));
        }
        let (r, n) = robust_rotation(&pairs).unwrap();
        assert_eq!(n, 240);
        assert!(close(&r, &truth, 1e-4));
        assert!(robust_rotation(&pairs[..5]).is_none());
        assert!(fit_rotation(&[]).is_none());
    }

    #[test]
    fn smoothing_keeps_a_steady_turn_and_removes_shake() {
        let times: Vec<f64> = (0..300).map(|k| k as f64 / 30.0).collect();
        let orient: Vec<M3> = times.iter().map(|t| camera_rotation(10.0 * t + 2.0 * (t * 40.0).sin(), 1.5 * (t * 33.0).cos(), 0.0)).collect();
        let s = smooth(&orient, &times, 0.5);
        // half way: the turn kept (≈ 50°), the shake gone
        let (y, p, _) = crate::vr::rotation_angles(&s[150]);
        assert!((y - 50.0).abs() < 1.0 && p.abs() < 0.3, "{y} {p}");
        // corrections turn each camera onto the smooth path
        let c = corrections(&orient, &s);
        for k in [0, 77, 299] {
            assert!(close(&mmul(&orient[k], &c[k]), &s[k], 1e-9));
        }
        assert_eq!(smooth(&[], &[], 1.0).len(), 0);
        assert_eq!(smooth(&orient, &times, f64::NAN).len(), 300);
    }

    /// A sphere with texture everywhere (blobs and fine detail), equirectangular.
    fn textured(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (lon, lat) = ((x as f64 + 0.5) / w as f64 * std::f64::consts::TAU, (y as f64 + 0.5) / h as f64 * std::f64::consts::PI);
                let d = crate::vr::dir(lon, lat - std::f64::consts::FRAC_PI_2);
                let v = 0.5
                    + 0.2 * (d[0] * 13.0).sin() * (d[1] * 11.0).cos()
                    + 0.15 * (d[2] * 29.0 + d[0] * 7.0).sin()
                    + 0.1 * ((d[1] * 41.0).sin() * (d[2] * 37.0).cos());
                let i = (y * w + x) * 4;
                img.px[i..i + 4].copy_from_slice(&[v as f32, v as f32, v as f32, 1.0]);
            }
        }
        img
    }

    #[test]
    fn the_camera_turn_between_two_frames_is_measured() {
        let a = textured(1440, 720);
        let model = SourceModel::new(&filmcraft_media::Spherical::equirect(), (1440, 720));
        // frame b: the same sphere seen by a camera turned by `turn` (b's rays are turn·a's)
        let turn = camera_rotation(2.0, 1.0, -0.8);
        let view = View { rotation: turn, output: crate::vr::Output::Sphere };
        let b = crate::vr::reframe(&a, &model, &view, 1440, 720, 1);
        let fa = Faces::new(&a, &model, 160);
        let fb = Faces::new(&b, &model, 160);
        let (dr, inliers) = rotation_between(&fa, &fb).expect("a rotation");
        assert!(inliers > 50, "{inliers}");
        // b ≈ ΔR · a: a direction seen at d in a is seen at turnᵀ·d in b
        let want = transpose(&turn);
        assert!(close(&dr, &want, 0.004), "{dr:?} vs {want:?}");
    }

    #[test]
    fn paths_unwrap_smooth_and_reduce() {
        assert_eq!(unwrap_degrees(&[170.0, -170.0, -150.0, 179.0]), vec![170.0, 190.0, 210.0, 179.0]);
        assert_eq!(unwrap_degrees(&[-179.0, 179.0]), vec![-179.0, -181.0]);
        let times: Vec<f64> = (0..200).map(|k| k as f64 / 20.0).collect();
        // a step at 5 s: smoothed both ways, it starts before and ends after, symmetric
        let step: Vec<f64> = times.iter().map(|t| if *t < 5.0 { 0.0 } else { 10.0 }).collect();
        let s = smooth_path(&times, &step, 2.0);
        assert!(s[80] > 0.0 && s[80] < 5.0 && s[120] > 5.0 && s[120] < 10.0, "{} {}", s[80], s[120]);
        assert!((s[99] + s[100] - 10.0).abs() < 0.5);
        // gaps are bridged
        let mut gaps = step.clone();
        for v in &mut gaps[90..110] {
            *v = f64::NAN;
        }
        assert!(smooth_path(&times, &gaps, 2.0).iter().all(|v| v.is_finite()));
        // a straight ramp reduces to its ends
        let ramp: Vec<f64> = times.iter().map(|t| t * 3.0).collect();
        assert_eq!(reduce(&times, &[&ramp], 0.01), vec![0, 199]);
        let keep = reduce(&times, &[&s], 0.05);
        assert!(keep.len() > 2 && keep.len() < 60, "{}", keep.len());
        assert!(smooth_path(&[], &[], 1.0).is_empty());
    }

    #[test]
    fn the_tracking_camera_recentres_on_its_subject() {
        let cam = TrackingCamera::new(30.0, 10.0, (384, 216));
        assert!(!cam.should_recentre(0.5, 0.5));
        assert!(cam.should_recentre(0.9, 0.5));
        let d = cam.direction(0.9, 0.5);
        let next = cam.recentred(d);
        let mid = next.direction(0.5, 0.5);
        assert!((dot(mid, d) - 1.0).abs() < 1e-9);
    }
}
