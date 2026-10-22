//! Auto Reframe paths: turn a clip's subject observations ([`crate::subject`]) into Motion
//! Position / Scale keyframes that keep the subject in a frame of another shape.
//!
//! Per shot (a cut starts a new one; the crop jumps with the picture, it never pans across a cut):
//!
//! 1. **Gaps** — frames where the subject is not seen confidently take the value interpolated
//!    between the frames around them (held at the ends of the shot).
//! 2. **Zoom** — with [`Zoom::Track`], a small subject is pushed in on (up to the follow mode's
//!    limit) so it fills a fair share of the frame, never so far that it no longer fits across.
//! 3. **Camera window** — the crop only moves once the subject leaves a small window around
//!    where it points, and then by the excess (a steady subject keeps a steady pan, no pulsing).
//! 4. **Critically damped follow** — the target is followed by a critically damped spring
//!    (x'' = ω²(target − x) − 2ωx', stepped with its exact solution for a target held over the
//!    step), run forwards and then backwards over the result so the path has no lag (a zero-phase
//!    filter), the shot's ends extended by point reflection so a subject already moving there is
//!    not lagged either. The spring's impulse response is non-negative, so the smoothed crop stays
//!    within the targets (kept inside the picture) but for the ends, where it is clamped.
//! 5. **Keyframes** — the dense path is reduced with Ramer–Douglas–Peucker in (time, position,
//!    scale) to keyframes that reproduce it within [`TOLERANCE_PX`] sequence pixels with linear
//!    interpolation; the last keyframe of a shot holds until the next shot's first.
//!
//! A clip where the subject is seen in less than [`MIN_FOUND`] of its frames gets the
//! [`Fallback`]: letterboxed (fitted whole), a centred crop, or — [`Fallback::Blur`] — fitted
//! whole over a blurred, dimmed, enlarged copy of itself (the engine adds the copy, under the
//! [`Plan::fill`] stretches).
//!
//! With [`Fallback::Blur`] it also works per stretch of a followed clip: where the subject is lost
//! for [`MIN_LOST_SECONDS`] or longer (or a whole shot has nobody in it) the crop eases out to the
//! whole picture over [`EASE_SECONDS`] (a smoothstep on the position and on the logarithm of the
//! scale), stays there, and eases back in to the tracked crop as the subject comes back; a cut
//! jumps as before. Shorter losses are bridged.

use serde::{Deserialize, Serialize};

use crate::subject::SubjectBox;

/// Below this confidence an observation does not count as seeing the subject.
pub const MIN_CONFIDENCE: f32 = 0.25;
/// Share of a clip's frames that must show the subject for it to be followed.
pub const MIN_FOUND: f64 = 0.3;
/// How closely the keyframes reproduce the smoothed path, sequence pixels.
pub const TOLERANCE_PX: f64 = 1.0;
/// Most samples planned for one clip (a long clip is analysed at a lower rate).
pub const MAX_SAMPLES: usize = 400_000;
/// With [`Fallback::Blur`], a followed clip's subject lost this long (seconds) gets the blurred
/// fill for the stretch; a shorter loss is bridged.
pub const MIN_LOST_SECONDS: f64 = 1.0;
/// How long the crop takes to ease out to the whole picture, and back in, seconds (each way; a
/// short loss eases over half its length).
pub const EASE_SECONDS: f64 = 0.5;

/// How the crop follows the subject.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Follow {
    /// A calm operator: a camera window, slow springs, little zoom.
    #[default]
    Smooth,
    /// Keeps the subject centred closely; faster springs, more zoom.
    Responsive,
}

impl Follow {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "smooth" | "slower" | "slow" => Some(Follow::Smooth),
            "responsive" | "faster" | "fast" => Some(Follow::Responsive),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Follow::Smooth => "smooth",
            Follow::Responsive => "responsive",
        }
    }
    /// The numbers behind the mode.
    pub fn tuning(self) -> Tuning {
        Tuning { omega: self.omega(), zoom_omega: self.zoom_omega(), window: self.window(), max_zoom: self.max_zoom() }
    }
    /// The position spring's natural frequency, rad/s.
    fn omega(self) -> f64 {
        match self {
            Follow::Smooth => 2.6,
            Follow::Responsive => 6.5,
        }
    }
    /// The zoom spring's.
    fn zoom_omega(self) -> f64 {
        match self {
            Follow::Smooth => 1.0,
            Follow::Responsive => 2.5,
        }
    }
    /// The camera window's half width, as a share of the visible picture.
    fn window(self) -> f64 {
        match self {
            Follow::Smooth => 0.06,
            Follow::Responsive => 0.015,
        }
    }
    /// The furthest push-in on a small subject.
    fn max_zoom(self) -> f64 {
        match self {
            Follow::Smooth => 1.2,
            Follow::Responsive => 1.35,
        }
    }
}

/// How a crop follows, in numbers: what a [`Follow`] mode picks ([`Follow::tuning`]), or a camera
/// track's own ([`Tuning::smoothness`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// The position spring's natural frequency, rad/s (lower is calmer).
    pub omega: f64,
    /// The zoom spring's.
    pub zoom_omega: f64,
    /// The camera window's half width, as a share of the visible picture.
    pub window: f64,
    /// The furthest push-in on a small subject ([`Zoom::Track`]).
    pub max_zoom: f64,
}

impl Tuning {
    /// A camera track's: `smooth` 0 follows closely (a fast spring, a small window), 1 floats
    /// (a slow spring, a wide window); 0.5 is about [`Follow::Smooth`]. Out-of-range or NaN
    /// values are clamped to 0..1 (NaN: 0.5).
    pub fn smoothness(smooth: f64) -> Tuning {
        let u = if smooth.is_finite() { smooth.clamp(0.0, 1.0) } else { 0.5 };
        // geometric between the ends: 9 rad/s … 1.2 rad/s
        let omega = 9.0 * (1.2f64 / 9.0).powf(u);
        Tuning { omega, zoom_omega: (omega * 0.4).max(0.5), window: 0.01 + 0.09 * u, max_zoom: 1.35 }
    }
    fn sane(self) -> Tuning {
        let pos = |v: f64, d: f64| if v.is_finite() && v > 0.0 { v } else { d };
        Tuning {
            omega: pos(self.omega, 2.6).clamp(0.05, 100.0),
            zoom_omega: pos(self.zoom_omega, 1.0).clamp(0.05, 100.0),
            window: if self.window.is_finite() { self.window.clamp(0.0, 0.4) } else { 0.06 },
            max_zoom: pos(self.max_zoom, 1.2).clamp(1.0, 8.0),
        }
    }
}

/// What a clip gets when its subject is not found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Fallback {
    /// Fitted whole, with bars.
    #[default]
    Letterbox,
    /// Filling the frame, centred.
    Center,
    /// Fitted whole over a blurred, dimmed, enlarged copy of itself; per stretch in a followed
    /// clip (see the module docs).
    Blur,
}

impl Fallback {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "letterbox" | "fit" | "bars" => Some(Fallback::Letterbox),
            "center" | "centre" | "crop" => Some(Fallback::Center),
            "blur" | "blurred" | "blur-fill" | "blurfill" | "blurred-fill" => Some(Fallback::Blur),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Fallback::Letterbox => "letterbox",
            Fallback::Center => "center",
            Fallback::Blur => "blur",
        }
    }
}

/// `"9:16"` (also `9x16`, `9/16`) as `(9, 16)`: both sides 1..=100, the shape between 1:4 and 4:1.
pub fn parse_aspect(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once([':', 'x', 'X', '/'])?;
    let (a, b): (u32, u32) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    let ok = (1..=100).contains(&a) && (1..=100).contains(&b) && a <= b.saturating_mul(4) && b <= a.saturating_mul(4);
    ok.then_some((a, b))
}

/// The frame of a reframed copy of a `source` sequence: the source's short side stays (a
/// 1920×1080 sequence becomes 1080×1920 at 9:16, 1080×1080 at 1:1, 1080×1350 at 4:5), even
/// sizes, at most 8192 a side.
pub fn frame_for(source: (u32, u32), aspect: (u32, u32)) -> Option<(u32, u32)> {
    let short = source.0.min(source.1).clamp(16, 8192) as f64;
    let (a, b) = (aspect.0.max(1) as f64, aspect.1.max(1) as f64);
    let (w, h) = if a <= b { (short, short * b / a) } else { (short * a / b, short) };
    let even = |v: f64| -> u32 { ((v / 2.0).round() * 2.0).clamp(16.0, 8192.0) as u32 };
    Some((even(w), even(h)))
}

/// A clip's picture and the frame it is reframed into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// The picture's display size in sequence pixels at 100 % scale.
    pub picture: (f64, f64),
    /// The frame, sequence pixels.
    pub frame: (f64, f64),
}

impl Geometry {
    fn valid(&self) -> bool {
        let ok = |v: f64| v.is_finite() && v >= 1.0;
        ok(self.picture.0) && ok(self.picture.1) && ok(self.frame.0) && ok(self.frame.1)
    }
    /// The scale (a share: 1 = 100 %) at which the picture covers the frame.
    pub fn fill_scale(&self) -> f64 {
        (self.frame.0 / self.picture.0.max(1.0)).max(self.frame.1 / self.picture.1.max(1.0))
    }
    /// The scale at which the whole picture fits in the frame.
    pub fn fit_scale(&self) -> f64 {
        (self.frame.0 / self.picture.0.max(1.0)).min(self.frame.1 / self.picture.1.max(1.0))
    }
    /// Whether the picture (its anchor at its centre) covers the frame at Motion `position` and
    /// scale `s` (half a pixel of slack).
    pub fn covers(&self, position: [f64; 2], s: f64) -> bool {
        let (hw, hh) = (self.picture.0 * s / 2.0, self.picture.1 * s / 2.0);
        let [x, y] = position;
        x - hw <= 0.5 && x + hw >= self.frame.0 - 0.5 && y - hh <= 0.5 && y + hh >= self.frame.1 - 0.5
    }
    /// The share of the picture the frame shows at scale `s`, per axis (at most 1).
    fn visible(&self, s: f64) -> [f64; 2] {
        let s = s.max(1e-6);
        [(self.frame.0 / (self.picture.0 * s)).min(1.0), (self.frame.1 / (self.picture.1 * s)).min(1.0)]
    }
    /// The Motion position that puts the picture's point `c` (0..1) mid-frame at scale `s`.
    fn position(&self, c: [f64; 2], s: f64) -> [f64; 2] {
        [self.frame.0 / 2.0 + (0.5 - c[0]) * self.picture.0 * s, self.frame.1 / 2.0 + (0.5 - c[1]) * self.picture.1 * s]
    }
}

/// One analysed frame of a clip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Seconds on the timeline (the path is smoothed in the time the viewer sees).
    pub t: f64,
    pub bbox: Option<SubjectBox>,
    pub confidence: f32,
    /// This frame starts a new shot.
    pub cut: bool,
}

/// What the crop's scale does.
#[derive(Clone, Copy, Debug)]
pub enum Zoom<'a> {
    /// Fill the frame, pushing in on a small subject.
    Track,
    /// The clip keeps its own scale (a share, one per sample: e.g. a push-in already keyframed);
    /// only the position follows.
    Keep(&'a [f64]),
    /// A camera track's push-in: this many times the fill scale (1..4), less where the subject
    /// would no longer fit across the frame, never under the fill scale.
    Push(f64),
}

/// One keyframe of the plan, at sample `index`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Key {
    pub index: usize,
    /// Motion ▸ Position, sequence pixels (the anchor at the picture's centre).
    pub position: [f64; 2],
    /// Motion ▸ Scale as a share (1 = 100 %).
    pub scale: f64,
    /// Holds until the next keyframe (the last one of a shot).
    pub hold: bool,
}

/// A clip's reframing.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Plan {
    /// At least one; a single key is a static placement.
    pub keys: Vec<Key>,
    pub animate_position: bool,
    pub animate_scale: bool,
    /// The subject was not found: the [`Fallback`] placement.
    pub letterbox: bool,
    /// Share of the samples that saw the subject.
    pub found: f64,
    /// Shots in the clip.
    pub shots: usize,
    /// With [`Fallback::Blur`]: the stretches (first and last sample, inclusive; a sample either
    /// side of the eased change included) where the picture may not cover the frame, which the
    /// blurred copy goes under. Empty otherwise.
    pub fill: Vec<(usize, usize)>,
}

impl Plan {
    fn fixed(g: &Geometry, scale: f64, letterbox: bool, found: f64, fill: Vec<(usize, usize)>) -> Self {
        let position = [g.frame.0 / 2.0, g.frame.1 / 2.0];
        Plan { keys: vec![Key { index: 0, position, scale, hold: false }], animate_position: false, animate_scale: false, letterbox, found, shots: 1, fill }
    }
}

/// Smoothstep: 0 at 0, 1 at 1, flat at both ends.
fn ease(u: f64) -> f64 {
    let u = if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.0 };
    u * u * (3.0 - 2.0 * u)
}

/// How far (0: the tracked crop, 1: the whole picture) each sample of a shot is towards the
/// blurred fill: the shot's runs of samples where the subject is not `seen` that last
/// [`MIN_LOST_SECONDS`] or more (until it is seen again), or the whole shot, eased in and out over
/// [`EASE_SECONDS`] — not at the shot's own ends, where the cut is the change.
fn lost_weights(seen: &[bool], times: &[f64]) -> Vec<f64> {
    let m = seen.len().min(times.len());
    let mut w = vec![0.0; m];
    let time = |k: usize| times.get(k).copied().filter(|t| t.is_finite()).unwrap_or(k as f64 / 30.0);
    let lost = |k: usize| !seen.get(k).copied().unwrap_or(true);
    let mut k = 0;
    while k < m {
        if !lost(k) {
            k += 1;
            continue;
        }
        let a = k;
        while k < m && lost(k) {
            k += 1;
        }
        let b = k.saturating_sub(1);
        let (at_start, at_end) = (a == 0, k >= m);
        // lost until it is seen again (the next sample), or to the shot's end
        let until = if at_end { time(b) } else { time(k) };
        let long = (until - time(a)).abs() >= MIN_LOST_SECONDS;
        if !(long || (at_start && at_end)) {
            continue;
        }
        let ramp = ((time(b) - time(a)).abs() / 2.0).min(EASE_SECONDS);
        for (j, v) in w.iter_mut().enumerate().take(b + 1).skip(a) {
            let into = if at_start || ramp <= 0.0 { 1.0 } else { ease((time(j) - time(a)) / ramp) };
            let out = if at_end || ramp <= 0.0 { 1.0 } else { ease((time(b) - time(j)) / ramp) };
            *v = into.min(out);
        }
    }
    w
}

/// Add the runs of `weights` above zero (sample `offset + k`), a sample either side included, to
/// `out`, merged with what is there.
fn spans(weights: &[f64], offset: usize, out: &mut Vec<(usize, usize)>) {
    let n = weights.len();
    let on = |k: usize| weights.get(k).copied().unwrap_or(0.0) > 0.0;
    let mut k = 0;
    while k < n {
        if !on(k) {
            k += 1;
            continue;
        }
        let a = k;
        while k < n && on(k) {
            k += 1;
        }
        let (a, b) = (offset.saturating_add(a.saturating_sub(1)), offset.saturating_add(k.min(n.saturating_sub(1))));
        match out.last_mut() {
            Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
}

/// The placement `u` (0..1) of the way from `a` to `b`: the position in a straight line, the
/// scale evenly on a logarithmic scale (a zoom that looks even).
fn towards(a: ([f64; 2], f64), b: ([f64; 2], f64), u: f64) -> ([f64; 2], f64) {
    let u = if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.0 };
    let lerp = |x: f64, y: f64| x + (y - x) * u;
    let positive = |s: f64| s.is_finite() && s > 0.0;
    let scale = if positive(a.1) && positive(b.1) { (a.1.ln() + (b.1.ln() - a.1.ln()) * u).exp() } else { lerp(a.1, b.1) };
    ([lerp(a.0[0], b.0[0]), lerp(a.0[1], b.0[1])], scale)
}

/// Step a critically damped spring at `x` (velocity `v`) towards `target` over `dt` seconds,
/// exactly (the closed-form solution for a target held over the step).
fn spring(x: &mut f64, v: &mut f64, target: f64, omega: f64, dt: f64) {
    let e = (-omega * dt).exp();
    let change = *x - target;
    let temp = (*v + omega * change) * dt;
    *v = (*v - omega * temp) * e;
    *x = target + (change + temp) * e;
}

/// Zero-phase critically damped smoothing of `targets` at times `times`. Both ends are extended
/// by point reflection (an odd extension, as zero-phase filters are padded) for as long as the
/// spring takes to settle, so a subject moving at the start or the end of a shot is not lagged
/// there: the springs settle in the padding.
fn follow(targets: &[f64], times: &[f64], omega: f64) -> Vec<f64> {
    let n = targets.len().min(times.len());
    let (Some(&x0), Some(&t0)) = (targets.first(), times.first()) else { return Vec::new() };
    let (Some(&xn), Some(&tn)) = (targets.get(n - 1), times.get(n - 1)) else { return Vec::new() };
    let typical = if n > 1 && (tn - t0).is_finite() { ((tn - t0).abs() / (n - 1) as f64).clamp(1e-3, 1.0) } else { 1.0 / 30.0 };
    // (1 + ωT)·e^(−ωT) of a transient is left after T: 0.3 % at ωT = 8
    let settle = 8.0 / omega.max(1e-3);
    let pad = ((settle / typical).ceil() as usize).min(n.saturating_sub(1)).min(8192);
    let mut xs = Vec::with_capacity(n + 2 * pad);
    let mut ts = Vec::with_capacity(n + 2 * pad);
    for k in (1..=pad).rev() {
        xs.push(2.0 * x0 - targets.get(k).copied().unwrap_or(x0));
        ts.push(2.0 * t0 - times.get(k).copied().unwrap_or(t0));
    }
    xs.extend_from_slice(targets.get(..n).unwrap_or_default());
    ts.extend_from_slice(times.get(..n).unwrap_or_default());
    for k in 1..=pad {
        xs.push(2.0 * xn - targets.get(n - 1 - k).copied().unwrap_or(xn));
        ts.push(2.0 * tn - times.get(n - 1 - k).copied().unwrap_or(tn));
    }
    let smooth = follow_raw(&xs, &ts, omega);
    smooth.get(pad..pad + n).map(<[f64]>::to_vec).unwrap_or_else(|| targets.get(..n).unwrap_or_default().to_vec())
}

/// Forward then backward critically damped springs over `targets`.
fn follow_raw(targets: &[f64], times: &[f64], omega: f64) -> Vec<f64> {
    let n = targets.len();
    let Some(&first) = targets.first() else { return Vec::new() };
    let dt = |a: usize, b: usize| -> f64 {
        let d = (times.get(b).copied().unwrap_or(0.0) - times.get(a).copied().unwrap_or(0.0)).abs();
        if d.is_finite() { d.clamp(1e-4, 1.0) } else { 1.0 / 30.0 }
    };
    let mut out = Vec::with_capacity(n);
    let (mut x, mut v) = (first, 0.0);
    out.push(x);
    for k in 1..n {
        let target = targets.get(k).copied().unwrap_or(x);
        spring(&mut x, &mut v, target, omega, dt(k - 1, k));
        out.push(x);
    }
    let (mut x, mut v) = (out.last().copied().unwrap_or(first), 0.0);
    for k in (0..n.saturating_sub(1)).rev() {
        let target = out.get(k).copied().unwrap_or(x);
        spring(&mut x, &mut v, target, omega, dt(k, k + 1));
        if let Some(o) = out.get_mut(k) {
            *o = x;
        }
    }
    out
}

/// A camera window of half width `zone[k]`: the aim moves only by how far the target leaves it.
fn window(targets: &[f64], zone: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(targets.len());
    let Some(&first) = targets.first() else { return out };
    let mut aim = first;
    for (k, t) in targets.iter().enumerate() {
        let z = zone.get(k).copied().unwrap_or(0.0).max(0.0);
        if *t > aim + z {
            aim = *t - z;
        } else if *t < aim - z {
            aim = *t + z;
        }
        out.push(aim);
    }
    out
}

/// Fill the gaps (`None`) of `v` by linear interpolation in time, holding at the ends; `None` when
/// nothing is known.
fn fill(v: &[Option<f64>], times: &[f64]) -> Option<Vec<f64>> {
    let known: Vec<usize> = (0..v.len()).filter(|&i| v.get(i).copied().flatten().is_some()).collect();
    let (&first, &last) = (known.first()?, known.last()?);
    let val = |i: usize| v.get(i).copied().flatten().unwrap_or(0.0);
    let time = |i: usize| times.get(i).copied().unwrap_or(i as f64);
    let mut out = Vec::with_capacity(v.len());
    let mut next = 0usize;
    for i in 0..v.len() {
        if i <= first {
            out.push(val(first));
            continue;
        }
        if i >= last {
            out.push(val(last));
            continue;
        }
        if v.get(i).copied().flatten().is_some() {
            out.push(val(i));
            continue;
        }
        while known.get(next + 1).is_some_and(|&k| k < i) {
            next += 1;
        }
        let (a, b) = (known.get(next).copied().unwrap_or(first), known.get(next + 1).copied().unwrap_or(last));
        let span = time(b) - time(a);
        let u = if span.abs() > 1e-12 { ((time(i) - time(a)) / span).clamp(0.0, 1.0) } else { 0.0 };
        out.push(val(a) + (val(b) - val(a)) * u);
    }
    Some(out)
}

/// Ramer–Douglas–Peucker over points `(t, values)`: the indices to keep so that linear
/// interpolation between them stays within `tol` (Euclidean over the values) of every point.
fn simplify(times: &[f64], values: &[[f64; 3]], tol: f64) -> Vec<usize> {
    let n = values.len().min(times.len());
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    if let Some(a) = keep.first_mut() {
        *a = true;
    }
    if let Some(z) = keep.get_mut(n - 1) {
        *z = true;
    }
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ta, tb) = (times.get(a).copied().unwrap_or(0.0), times.get(b).copied().unwrap_or(0.0));
        let (va, vb) = (values.get(a).copied().unwrap_or_default(), values.get(b).copied().unwrap_or_default());
        let mut worst = (0.0f64, a);
        for i in a + 1..b {
            let u = if (tb - ta).abs() > 1e-12 { (times.get(i).copied().unwrap_or(ta) - ta) / (tb - ta) } else { 0.5 };
            let vi = values.get(i).copied().unwrap_or_default();
            let e = vi.iter().zip(va.iter().zip(&vb)).map(|(v, (x, y))| (v - (x + (y - x) * u)).powi(2)).sum::<f64>().sqrt();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        if worst.0 > tol {
            if let Some(k) = keep.get_mut(worst.1) {
                *k = true;
            }
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..n).filter(|&i| keep.get(i).copied().unwrap_or(false)).collect()
}

/// Plan a clip's reframing from its samples (see the module docs).
pub fn plan(samples: &[Sample], g: Geometry, follow_mode: Follow, zoom: Zoom, fallback: Fallback) -> Plan {
    plan_tuned(samples, g, follow_mode.tuning(), zoom, fallback)
}

/// [`plan`] with the follow's numbers given (a camera track's [`Tuning::smoothness`]).
pub fn plan_tuned(samples: &[Sample], g: Geometry, tuning: Tuning, zoom: Zoom, fallback: Fallback) -> Plan {
    let tuning = tuning.sane();
    let n = samples.len().min(MAX_SAMPLES);
    let samples = samples.get(..n).unwrap_or_default();
    let keep = match zoom {
        Zoom::Keep(s) => Some(s),
        Zoom::Track | Zoom::Push(_) => None,
    };
    let push = match zoom {
        Zoom::Push(f) if f.is_finite() => Some(f.clamp(1.0, 4.0)),
        Zoom::Push(_) => Some(1.0),
        _ => None,
    };
    let s0 = g.fill_scale();
    // the scale a sample sits at before any push-in
    let base = |i: usize| -> f64 {
        match keep {
            Some(s) => s.get(i).copied().filter(|v| v.is_finite() && *v > 0.0).unwrap_or(s0),
            None => s0,
        }
    };
    // the whole picture, over the blurred copy for the whole clip
    let blur_all = |n: usize| if n > 0 { vec![(0, n - 1)] } else { Vec::new() };
    if !g.valid() || n == 0 {
        return match (fallback, keep) {
            (Fallback::Letterbox, _) => Plan::fixed(&g, g.fit_scale(), true, 0.0, Vec::new()),
            (Fallback::Blur, None) if g.valid() => Plan::fixed(&g, g.fit_scale(), true, 0.0, blur_all(n)),
            _ => Plan::fixed(&g, base(0), true, 0.0, Vec::new()),
        };
    }
    let seen = |s: &Sample| s.bbox.is_some() && s.confidence >= MIN_CONFIDENCE && s.confidence.is_finite();
    let found = samples.iter().filter(|s| seen(s)).count() as f64 / n as f64;
    if found < MIN_FOUND {
        return match (fallback, keep) {
            (Fallback::Letterbox, None) => Plan::fixed(&g, g.fit_scale(), true, found, Vec::new()),
            (Fallback::Blur, None) => Plan::fixed(&g, g.fit_scale(), true, found, blur_all(n)),
            _ => Plan::fixed(&g, base(0), true, found, Vec::new()),
        };
    }
    // with the blurred fill, lost stretches ease out to the whole picture (not with the clip's
    // own scale: only its position follows then)
    let blur = fallback == Fallback::Blur && keep.is_none();
    let whole = ([g.frame.0 / 2.0, g.frame.1 / 2.0], g.fit_scale());
    let mut fill_spans: Vec<(usize, usize)> = Vec::new();
    let times: Vec<f64> = samples.iter().enumerate().map(|(i, s)| if s.t.is_finite() { s.t } else { i as f64 / 30.0 }).collect();
    // shots: [start, end) ranges
    let mut shots: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (i, s) in samples.iter().enumerate().skip(1) {
        if s.cut {
            shots.push((start, i));
            start = i;
        }
    }
    shots.push((start, n));
    let mut path: Vec<([f64; 2], f64)> = Vec::with_capacity(n);
    for &(a, b) in &shots {
        let shot = samples.get(a..b).unwrap_or_default();
        let t = times.get(a..b).unwrap_or_default();
        let pick = |f: &dyn Fn(&SubjectBox) -> f64| -> Vec<Option<f64>> { shot.iter().map(|s| if seen(s) { s.bbox.as_ref().map(f) } else { None }).collect() };
        let cx = fill(&pick(&|b| b.cx), t);
        let cy = fill(&pick(&|b| b.cy), t);
        let bw = fill(&pick(&|b| b.w), t);
        let bh = fill(&pick(&|b| b.h), t);
        let m = shot.len();
        // the scale at each sample
        let scales: Vec<f64> = match (keep, &bw, &bh) {
            (Some(_), _, _) => (a..b).map(base).collect(),
            (None, Some(bw), Some(bh)) => {
                let want: Vec<f64> = (0..m)
                    .map(|k| {
                        let (w, h) = (bw.get(k).copied().unwrap_or(1.0), bh.get(k).copied().unwrap_or(1.0));
                        // the subject's height as a share of the frame at fill
                        let share = h * g.picture.1 * s0 / g.frame.1;
                        let fits = 0.9 * g.frame.0 / (w * g.picture.0 * s0).max(1e-6);
                        match push {
                            Some(f) => f.min(fits).max(1.0).ln(),
                            None => (0.45 / share.max(1e-6)).min(fits).clamp(1.0, tuning.max_zoom).ln(),
                        }
                    })
                    .collect();
                let zone = vec![0.08; m];
                let z = follow(&window(&want, &zone), t, tuning.zoom_omega);
                let lo = z.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // a barely changing push-in is no push-in
                if hi - lo < 0.03 { vec![s0 * lo.exp().max(1.0); m] } else { z.iter().map(|v| s0 * v.exp().max(1.0)).collect() }
            }
            _ => vec![s0; m],
        };
        // with the clip's own scale, keep the crop inside at the smallest scale of the shot
        let floor = scales.iter().copied().fold(f64::INFINITY, f64::min);
        let vis_at = |k: usize| -> [f64; 2] { g.visible(if keep.is_some() { floor } else { scales.get(k).copied().unwrap_or(s0) }) };
        let clamp = |axis: usize, k: usize, x: f64| -> f64 {
            let v = vis_at(k)[axis];
            let (lo, hi) = (v / 2.0, 1.0 - v / 2.0);
            if !x.is_finite() || lo >= hi { 0.5 } else { x.clamp(lo, hi) }
        };
        let centres: Vec<[f64; 2]> = match (&cx, &cy) {
            (Some(cx), Some(cy)) => {
                let tx: Vec<f64> = (0..m).map(|k| clamp(0, k, cx.get(k).copied().unwrap_or(0.5))).collect();
                let ty: Vec<f64> = (0..m).map(|k| clamp(1, k, cy.get(k).copied().unwrap_or(0.5))).collect();
                let zx: Vec<f64> = (0..m).map(|k| tuning.window * vis_at(k)[0]).collect();
                let zy: Vec<f64> = (0..m).map(|k| tuning.window * vis_at(k)[1]).collect();
                let sx = follow(&window(&tx, &zx), t, tuning.omega);
                let sy = follow(&window(&ty, &zy), t, tuning.omega);
                (0..m).map(|k| [clamp(0, k, sx.get(k).copied().unwrap_or(0.5)), clamp(1, k, sy.get(k).copied().unwrap_or(0.5))]).collect()
            }
            // nobody in this shot: a centred crop
            _ => vec![[0.5, 0.5]; m],
        };
        let weights = if blur { lost_weights(&shot.iter().map(seen).collect::<Vec<_>>(), t) } else { Vec::new() };
        for k in 0..m {
            let s = scales.get(k).copied().unwrap_or(s0);
            // with the clip's own scale the position is figured at the shot's smallest: the
            // picture then covers the frame at any scale it animates to between keyframes
            let at = if keep.is_some() && floor.is_finite() { floor } else { s };
            let tracked = (g.position(centres.get(k).copied().unwrap_or([0.5, 0.5]), at), s);
            let u = weights.get(k).copied().unwrap_or(0.0);
            path.push(if u > 0.0 { towards(tracked, whole, u) } else { tracked });
        }
        spans(&weights, a, &mut fill_spans);
    }
    // keyframes: per shot, the fewest that reproduce the path
    let unit = g.picture.0.max(g.picture.1) / 2.0;
    let mut keys: Vec<Key> = Vec::new();
    for (si, &(a, b)) in shots.iter().enumerate() {
        let vals: Vec<[f64; 3]> = path.get(a..b).unwrap_or_default().iter().map(|(p, s)| [p[0], p[1], s * unit]).collect();
        let kept = simplify(times.get(a..b).unwrap_or_default(), &vals, TOLERANCE_PX);
        let last_shot = si + 1 == shots.len();
        let count = kept.len();
        for (j, k) in kept.into_iter().enumerate() {
            let Some((p, s)) = path.get(a + k).copied() else { continue };
            keys.push(Key { index: a + k, position: p, scale: s, hold: !last_shot && j + 1 == count });
        }
    }
    let first = keys.first().copied();
    let animate_position =
        first.is_some_and(|f| keys.iter().any(|k| (k.position[0] - f.position[0]).abs() > 0.5 || (k.position[1] - f.position[1]).abs() > 0.5));
    let animate_scale = first.is_some_and(|f| keys.iter().any(|k| (k.scale - f.scale).abs() > 5e-4));
    if !animate_position && !animate_scale {
        keys.truncate(1);
    }
    Plan { keys, animate_position, animate_scale, letterbox: false, found, shots: shots.len(), fill: fill_spans }
}

/// The plan's placement at sample `i` (linear between keyframes, holding after a shot's last),
/// for checks and previews.
pub fn placement_at(plan: &Plan, times: &[f64], i: usize) -> Option<([f64; 2], f64)> {
    let keys = &plan.keys;
    let first = keys.first()?;
    if keys.len() == 1 || i <= first.index {
        return Some((first.position, first.scale));
    }
    let j = keys.iter().rposition(|k| k.index <= i)?;
    let a = keys.get(j)?;
    let Some(b) = keys.get(j + 1) else { return Some((a.position, a.scale)) };
    if a.hold || a.index == i {
        return Some((a.position, a.scale));
    }
    let (ta, tb, ti) = (times.get(a.index)?, times.get(b.index)?, times.get(i)?);
    let u = if (tb - ta).abs() > 1e-12 { ((ti - ta) / (tb - ta)).clamp(0.0, 1.0) } else { 0.0 };
    let lerp = |x: f64, y: f64| x + (y - x) * u;
    Some(([lerp(a.position[0], b.position[0]), lerp(a.position[1], b.position[1])], lerp(a.scale, b.scale)))
}

#[cfg(test)]
#[path = "reframe_tests.rs"]
mod tests;
