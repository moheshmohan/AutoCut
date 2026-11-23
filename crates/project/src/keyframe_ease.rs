//! Keyframe eases beyond Premiere's temporal set (PX5.1): a damped spring, `steps(n)`, a seeded
//! wiggle and custom curves, and the perceptual option that interpolates in log space (an even
//! zoom from 100 % to 400 %).
//!
//! A keyframe's [`crate::Interpolation`] picks the mode of the segment leaving it; the mode's
//! numbers live in the keyframe's [`SegmentEase`] (absent on most keyframes and in every older
//! project, so those load and save unchanged). Each mode maps the segment's progress `u` (0 at
//! the keyframe, 1 at the next one) to an eased progress that is exactly 0 at `u = 0` and exactly
//! 1 at `u = 1`, so a segment always starts and lands on its keyframes:
//!
//! - **Spring**: a damped harmonic oscillator released from rest (see [`Spring::progress`]).
//! - **Steps**: `n` equal jumps, the last one landing on the next keyframe (CSS `jump-end`).
//! - **Custom**: a curve through the given `(time, progress)` points, monotone cubic between them.
//! - **Wiggle**: linear, plus a smooth seeded wobble that fades in and out at the segment's ends.
//!
//! Everything here is plain `f64` math on hostile numbers: values read from a project file or a
//! command are clamped by the `sanitized` functions, non-finite ones take the defaults, every loop
//! is bounded and no step divides by a value that can be zero.

use serde::{Deserialize, Serialize};

/// Most steps of a `steps(n)` segment.
pub const MAX_STEPS: u32 = 1000;
/// Steps of a segment set to Steps without a count.
pub const DEFAULT_STEPS: u32 = 5;
/// Most points of a custom curve (the implied ends not counted); later points are ignored.
pub const MAX_CURVE_POINTS: usize = 64;
/// A custom curve's progress is kept within this range (an overshoot of up to one whole change).
pub const CURVE_RANGE: (f64, f64) = (-1.0, 2.0);
/// The curve of a segment set to Custom without points: a gentle S (slow out, slow in).
pub const DEFAULT_CURVE: [[f64; 2]; 2] = [[0.3, 0.1], [0.7, 0.9]];
/// Largest wiggle amplitude, in the parameter's own units.
pub const MAX_WIGGLE_AMPLITUDE: f64 = 1.0e6;
/// Fastest wiggle, wobbles a second.
pub const MAX_WIGGLE_FREQUENCY: f64 = 120.0;
/// Share of a wiggle segment at each end over which the wobble fades in and out.
const WIGGLE_FADE: f64 = 0.1;

/// Safe ranges of the spring's numbers.
pub const SPRING_DAMPING_RATIO: (f64, f64) = (0.05, 10.0);
pub const SPRING_STIFFNESS: (f64, f64) = (0.1, 10_000.0);
pub const SPRING_MASS: (f64, f64) = (0.01, 100.0);
pub const SPRING_MAX_OVERSHOOT: (f64, f64) = (0.0, 1.0);
/// The spring has settled once what is left of its motion is under this share of the change.
const SETTLE: f64 = 1.0e-3;
/// Damping ratios this close to 1 use the critically damped solution.
const CRITICAL_BAND: f64 = 1.0e-3;

/// A spring: a mass on a damped spring pulled from one keyframe's value to the next.
///
/// The damping ratio ζ says how the motion ends: 1 is critically damped (the quickest arrival
/// without passing the target), above 1 arrives more slowly, still without passing it, and below 1
/// passes the target and rings back. The default 0.9 passes it by 0.15 % of the change, which no
/// one sees. **0.5–0.7 is for a declared playful style only** (an overshoot of 16 % to 5 %); keep
/// the default, or 1, for everything else.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Spring {
    /// ζ = c / (2 √(k m)), [`SPRING_DAMPING_RATIO`].
    pub damping_ratio: f64,
    /// k, [`SPRING_STIFFNESS`]: with the mass, how fast the spring moves (ω = √(k / m)).
    pub stiffness: f64,
    /// m, [`SPRING_MASS`].
    pub mass: f64,
    /// The most the value may pass the next keyframe, as a share of the segment's change (0..1);
    /// a spring that would go further is held there.
    pub max_overshoot: f64,
}

impl Default for Spring {
    fn default() -> Self {
        Spring::DEFAULT
    }
}

impl Spring {
    /// ζ 0.9 (no visible overshoot), k 150, m 1: settles in about 0.7 s.
    pub const DEFAULT: Spring = Spring { damping_ratio: 0.9, stiffness: 150.0, mass: 1.0, max_overshoot: 0.25 };

    /// The spring with hostile numbers made safe: each within its range, a non-finite one the
    /// default's.
    pub fn sanitized(self) -> Spring {
        let d = Spring::DEFAULT;
        Spring {
            damping_ratio: within(self.damping_ratio, SPRING_DAMPING_RATIO, d.damping_ratio),
            stiffness: within(self.stiffness, SPRING_STIFFNESS, d.stiffness),
            mass: within(self.mass, SPRING_MASS, d.mass),
            max_overshoot: within(self.max_overshoot, SPRING_MAX_OVERSHOOT, d.max_overshoot),
        }
    }

    /// The natural angular frequency ω = √(k / m), rad/s (the mass is at least 0.01).
    fn omega(self) -> f64 {
        (self.stiffness / self.mass).sqrt()
    }

    /// What is left of the motion `tau` seconds after release, as a share of the change: the
    /// solution of x'' + 2ζω x' + ω² (x − 1) = 0 with x(0) = 0, x'(0) = 0 is x = 1 − d(τ), where
    ///
    /// - ζ < 1 (under-damped), ω_d = ω √(1 − ζ²): d = e^(−ζωτ) (cos ω_d τ + (ζω / ω_d) sin ω_d τ)
    /// - ζ = 1 (critical): d = e^(−ωτ) (1 + ωτ)
    /// - ζ > 1 (over-damped), roots r₁,₂ = −ζω ± ω √(ζ² − 1): d = (r₂ e^(r₁τ) − r₁ e^(r₂τ)) / (r₂ − r₁)
    ///
    /// Spring must be sanitized (ω > 0).
    fn left(self, w: f64, tau: f64) -> f64 {
        let z = self.damping_ratio;
        if (z - 1.0).abs() < CRITICAL_BAND {
            (-w * tau).exp() * (1.0 + w * tau)
        } else if z < 1.0 {
            let wd = w * (1.0 - z * z).sqrt();
            (-z * w * tau).exp() * ((wd * tau).cos() + (z * w / wd) * (wd * tau).sin())
        } else {
            let s = w * (z * z - 1.0).sqrt();
            let (r1, r2) = (-z * w + s, -z * w - s);
            (r2 * (r1 * tau).exp() - r1 * (r2 * tau).exp()) / (r2 - r1)
        }
    }

    /// A bound on |d(τ)| that only falls: the under-damped envelope e^(−ζωτ) / √(1 − ζ²), or d
    /// itself, which falls monotonically from 1 when ζ ≥ 1.
    fn envelope(self, w: f64, tau: f64) -> f64 {
        let z = self.damping_ratio;
        if z < 1.0 - CRITICAL_BAND { (-z * w * tau).exp() / (1.0 - z * z).sqrt() } else { self.left(w, tau) }
    }

    /// Seconds until the spring has settled (what is left under 0.1 % of the change), on its own
    /// clock. Bounded: a doubling search for an upper end, then bisection.
    pub fn settle_seconds(self) -> f64 {
        let s = self.sanitized();
        let w = s.omega();
        let mut hi = 1.0 / w;
        for _ in 0..64 {
            if s.envelope(w, hi) <= SETTLE {
                break;
            }
            hi *= 2.0;
        }
        let mut lo = 0.0;
        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            if s.envelope(w, mid) <= SETTLE {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Eased progress at `u` (0..1) of a segment `seconds` long: 0 at the start, exactly 1 at the
    /// end, past 1 by at most [`Spring::max_overshoot`] on the way.
    ///
    /// The spring runs on its own clock, set by stiffness and mass. In a segment at least as long
    /// as the spring takes to settle it moves at its natural speed and then rests on the target; in
    /// a shorter one its clock runs faster, by the ratio of the two, so that it settles exactly at
    /// the segment's end. Normalisation: whatever is left of the motion at the end (under 0.1 % of
    /// the change, see [`Spring::settle_seconds`]) is added back in proportion to `u`, so the value
    /// reaches the next keyframe exactly at the segment's end and the curve stays continuous.
    pub fn progress(self, u: f64, seconds: f64) -> f64 {
        if u.is_nan() || u <= 0.0 {
            return 0.0;
        }
        if u >= 1.0 {
            return 1.0;
        }
        let s = self.sanitized();
        let w = s.omega();
        let settle = s.settle_seconds();
        // the spring's clock at the segment's end: its natural time, or the settle time if longer
        let end = if seconds.is_finite() && seconds > 0.0 { seconds.max(settle) } else { settle };
        let rest = s.left(w, end);
        let p = 1.0 - s.left(w, u * end) + rest * u;
        if p.is_finite() { p.min(1.0 + s.max_overshoot) } else { u }
    }
}

/// Eased progress of a `steps(n)` segment at `u`: `n` equal jumps, the last one at the end
/// (`floor(u n) / n` before it). `n` is clamped to 1..[`MAX_STEPS`].
pub fn steps_progress(n: u32, u: f64) -> f64 {
    if u.is_nan() || u <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return 1.0;
    }
    let n = f64::from(n.clamp(1, MAX_STEPS));
    (u * n).floor() / n
}

/// Eased progress of a custom curve at `u`. The curve runs through (0, 0), the `points` and
/// (1, 1); between them it is a monotone cubic (Fritsch and Carlson's method: tangents from the
/// neighbouring slopes, zero at a local peak, limited so the curve never swings past its points).
/// Points are `(time, progress)` with time strictly increasing inside (0, 1); a point that is not
/// (non-finite, out of order, outside) is skipped, progress is clamped to [`CURVE_RANGE`] and at
/// most [`MAX_CURVE_POINTS`] are read. No points: [`DEFAULT_CURVE`]. Allocation-free.
pub fn curve_progress(points: &[[f64; 2]], u: f64) -> f64 {
    if u.is_nan() || u <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return 1.0;
    }
    const N: usize = MAX_CURVE_POINTS + 2;
    let mut xs = [0.0f64; N];
    let mut ys = [0.0f64; N];
    let mut n = 1usize;
    let src: &[[f64; 2]] = if points.is_empty() { &DEFAULT_CURVE } else { points };
    for &[x, y] in src.iter().take(MAX_CURVE_POINTS) {
        let prev = xs.get(n - 1).copied().unwrap_or(0.0);
        if !(x.is_finite() && y.is_finite()) || x <= prev || x >= 1.0 {
            continue;
        }
        if let (Some(sx), Some(sy)) = (xs.get_mut(n), ys.get_mut(n)) {
            *sx = x;
            *sy = y.clamp(CURVE_RANGE.0, CURVE_RANGE.1);
            n += 1;
        }
    }
    if let (Some(sx), Some(sy)) = (xs.get_mut(n), ys.get_mut(n)) {
        *sx = 1.0;
        *sy = 1.0;
        n += 1;
    }
    let (xs, ys) = (&xs[..n], &ys[..n]);
    // secant slopes of each interval (every interval is at least one ulp wide: x increases)
    let mut d = [0.0f64; N];
    for (k, slot) in d.iter_mut().enumerate().take(n - 1) {
        let (x0, x1, y0, y1) = (xs[k], xs[k + 1], ys[k], ys[k + 1]);
        let h = x1 - x0;
        *slot = if h > 0.0 { (y1 - y0) / h } else { 0.0 };
    }
    // tangents: the mean of the neighbouring slopes, zero where they differ in sign
    let mut m = [0.0f64; N];
    for (k, slot) in m.iter_mut().enumerate().take(n) {
        *slot = if k == 0 {
            d[0]
        } else if k == n - 1 {
            d[n - 2]
        } else if d[k - 1] * d[k] <= 0.0 {
            0.0
        } else {
            (d[k - 1] + d[k]) / 2.0
        };
    }
    // limit them so the curve stays monotone where its points are
    for (k, &dk) in d.iter().enumerate().take(n - 1) {
        if dk == 0.0 {
            m[k] = 0.0;
            m[k + 1] = 0.0;
            continue;
        }
        let (a, b) = (m[k] / dk, m[k + 1] / dk);
        let r = a * a + b * b;
        if !r.is_finite() {
            // slopes too steep to compare (points a hair apart): flat tangents
            m[k] = 0.0;
            m[k + 1] = 0.0;
        } else if r > 9.0 {
            let t = 3.0 / r.sqrt();
            m[k] = t * a.max(0.0) * dk;
            m[k + 1] = t * b.max(0.0) * dk;
        } else {
            m[k] = a.max(0.0) * dk;
            m[k + 1] = b.max(0.0) * dk;
        }
    }
    // the interval holding u, and cubic Hermite on it
    let k = xs.partition_point(|&x| x <= u).saturating_sub(1).min(n - 2);
    let (x0, x1, y0, y1) = (xs[k], xs[k + 1], ys[k], ys[k + 1]);
    let h = x1 - x0;
    if h <= 0.0 {
        return y0;
    }
    let t = (u - x0) / h;
    let (t2, t3) = (t * t, t * t * t);
    let v = (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * m[k] + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * h * m[k + 1];
    if v.is_finite() { v.clamp(CURVE_RANGE.0, CURVE_RANGE.1) } else { u }
}

/// A seeded wiggle: a smooth wobble added to a linear segment, the same every time for one seed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Wiggle {
    /// How far the value wobbles each way, in the parameter's own units (pixels, percent,
    /// degrees), 0..[`MAX_WIGGLE_AMPLITUDE`].
    pub amplitude: f64,
    /// Wobbles a second, 0..[`MAX_WIGGLE_FREQUENCY`].
    pub frequency: f64,
    pub seed: u64,
}

impl Default for Wiggle {
    fn default() -> Self {
        Wiggle::DEFAULT
    }
}

impl Wiggle {
    pub const DEFAULT: Wiggle = Wiggle { amplitude: 10.0, frequency: 2.0, seed: 1 };

    /// The wiggle with hostile numbers made safe.
    pub fn sanitized(self) -> Wiggle {
        let d = Wiggle::DEFAULT;
        Wiggle {
            amplitude: within(self.amplitude, (0.0, MAX_WIGGLE_AMPLITUDE), d.amplitude),
            frequency: within(self.frequency, (0.0, MAX_WIGGLE_FREQUENCY), d.frequency),
            seed: self.seed,
        }
    }

    /// The wobble of value component `component` at clip time `seconds`, `u` (0..1) through its
    /// segment: smooth value noise at `frequency` (a hash of the seed, the component and the
    /// whole wobble number on each side, blended with a smoothstep), times the amplitude, faded in
    /// over the first and out over the last tenth of the segment so the segment still starts and
    /// ends on its keyframes. No state: the same seed, component and time give the same offset.
    pub fn offset(self, component: usize, seconds: f64, u: f64) -> f64 {
        let w = self.sanitized();
        if w.amplitude == 0.0 || !seconds.is_finite() || u.is_nan() {
            return 0.0;
        }
        let fade = |s: f64| {
            let s = (s / WIGGLE_FADE).clamp(0.0, 1.0);
            s * s * (3.0 - 2.0 * s)
        };
        let env = fade(u) * fade(1.0 - u);
        if env == 0.0 {
            return 0.0;
        }
        // the noise coordinate is kept well inside i64 (a cast saturates anyway)
        let x = (seconds * w.frequency).clamp(-1.0e15, 1.0e15);
        let i = x.floor();
        let f = x - i;
        let i = i as i64;
        let (v0, v1) = (noise(w.seed, component, i), noise(w.seed, component, i.wrapping_add(1)));
        let s = f * f * (3.0 - 2.0 * f);
        w.amplitude * env * (v0 + (v1 - v0) * s)
    }
}

/// A value in −1..1 for lattice point `i` of component `component` under `seed`: the 64-bit
/// finalizer of the SplitMix generator (public domain) over the three mixed together.
fn noise(seed: u64, component: usize, i: i64) -> f64 {
    let mut z = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (component as u64).wrapping_add(1).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// `x` to `y` at eased progress `e` in log space, for values of one sign: the same ratio each
/// step (100 → 200 → 400 % evenly). None when the values are zero, of different signs or not
/// finite; the caller then interpolates linearly.
pub fn log_lerp(x: f64, y: f64, e: f64) -> Option<f64> {
    if !(x.is_finite() && y.is_finite() && e.is_finite()) {
        return None;
    }
    let sign = if x > 0.0 && y > 0.0 {
        1.0
    } else if x < 0.0 && y < 0.0 {
        -1.0
    } else {
        return None;
    };
    let (lx, ly) = ((x * sign).ln(), (y * sign).ln());
    let v = sign * (lx + (ly - lx) * e).exp();
    v.is_finite().then_some(v)
}

/// The numbers of a keyframe's segment beyond its interpolation mode: the spring, the step count,
/// the wiggle and the custom curve (each read only by its own mode, and kept when the mode
/// changes so switching back restores them) and the perceptual option.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SegmentEase {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spring: Option<Spring>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wiggle: Option<Wiggle>,
    /// Custom curve points `(time, progress)`, see [`curve_progress`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub curve: Vec<[f64; 2]>,
    /// Interpolate numbers and points in log space (perceptual scale): values of one sign only,
    /// others (and colours and mask paths) linearly.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub perceptual: bool,
}

/// A segment with nothing beyond its interpolation mode.
pub static PLAIN: SegmentEase = SegmentEase { spring: None, steps: None, wiggle: None, curve: Vec::new(), perceptual: false };

impl SegmentEase {
    pub fn is_plain(&self) -> bool {
        *self == PLAIN
    }
    /// The step count (clamped), or [`DEFAULT_STEPS`].
    pub fn step_count(&self) -> u32 {
        self.steps.unwrap_or(DEFAULT_STEPS).clamp(1, MAX_STEPS)
    }
    /// Feed every number to `eat` (cache keys of evaluated curves).
    pub fn fingerprint(&self, eat: &mut impl FnMut(u64)) {
        if let Some(s) = self.spring {
            for v in [s.damping_ratio, s.stiffness, s.mass, s.max_overshoot] {
                eat(v.to_bits());
            }
        }
        eat(u64::from(self.steps.unwrap_or(0)));
        if let Some(w) = self.wiggle {
            eat(w.amplitude.to_bits());
            eat(w.frequency.to_bits());
            eat(w.seed);
        }
        for p in self.curve.iter().take(MAX_CURVE_POINTS) {
            eat(p[0].to_bits());
            eat(p[1].to_bits());
        }
        eat(u64::from(self.perceptual));
    }
}

/// `v` within `(lo, hi)`, or `or` when it is not finite.
fn within(v: f64, (lo, hi): (f64, f64), or: f64) -> f64 {
    if v.is_finite() { v.clamp(lo, hi) } else { or }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spring(damping_ratio: f64) -> Spring {
        Spring { damping_ratio, ..Spring::DEFAULT }
    }

    fn samples(f: impl Fn(f64) -> f64) -> Vec<f64> {
        (0..=400).map(|i| f(i as f64 / 400.0)).collect()
    }

    #[test]
    fn a_spring_reaches_the_target_exactly_at_the_segment_end() {
        for z in [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 4.0, 10.0] {
            for seconds in [0.05, 0.3, 1.0, 4.0, 60.0] {
                let s = spring(z);
                assert_eq!(s.progress(0.0, seconds), 0.0);
                assert_eq!(s.progress(1.0, seconds), 1.0);
                // continuous into the end: the last sample is a hair away from 1
                let near = s.progress(1.0 - 1e-9, seconds);
                assert!((near - 1.0).abs() < 1e-3, "ζ {z}, {seconds} s: {near}");
                // and it has settled well before the end of a long segment
                if seconds >= 4.0 {
                    assert!((s.progress(0.9, seconds) - 1.0).abs() < 2e-3, "ζ {z}");
                }
            }
        }
        // a short segment runs the spring's clock faster: settled at the end, not cut short
        let s = Spring::DEFAULT;
        assert!(s.settle_seconds() > 0.3 && s.settle_seconds() < 1.5, "{}", s.settle_seconds());
        let fast = s.progress(0.95, 0.1);
        assert!(fast > 0.99, "{fast}");
    }

    #[test]
    fn a_spring_at_damping_one_or_more_never_overshoots_and_only_rises() {
        for z in [1.0, 1.0005, 1.2, 2.0, 10.0] {
            for seconds in [0.2, 1.0, 10.0] {
                let v = samples(|u| spring(z).progress(u, seconds));
                assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)), "ζ {z}");
                assert!(v.windows(2).all(|w| w[1] >= w[0] - 1e-12), "ζ {z}: monotone");
            }
        }
        // the default (0.9) passes the target by no more than 0.2 % of the change
        let peak = samples(|u| Spring::DEFAULT.progress(u, 2.0)).into_iter().fold(0.0, f64::max);
        assert!(peak < 1.002, "{peak}");
    }

    #[test]
    fn overshoot_is_bounded_by_the_clamp() {
        // 0.5 overshoots about 16 %: under the default clamp of 25 %
        let free = samples(|u| spring(0.5).progress(u, 2.0)).into_iter().fold(0.0, f64::max);
        assert!(free > 1.12 && free < 1.2, "{free}");
        for clamp in [0.0, 0.05, 0.1] {
            let s = Spring { damping_ratio: 0.3, max_overshoot: clamp, ..Spring::DEFAULT };
            let peak = samples(|u| s.progress(u, 2.0)).into_iter().fold(0.0, f64::max);
            assert!(peak <= 1.0 + clamp + 1e-12 && peak > 1.0 + clamp - 1e-3, "clamp {clamp}: {peak}");
        }
    }

    #[test]
    fn hostile_spring_numbers_are_made_safe() {
        let wild = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0, 1e300, -1e300, 1e-300];
        for &z in &wild {
            for &k in &wild {
                for &m in &wild {
                    for &o in &wild {
                        let s = Spring { damping_ratio: z, stiffness: k, mass: m, max_overshoot: o };
                        let c = s.sanitized();
                        assert!(c.damping_ratio >= 0.05 && c.stiffness >= 0.1 && c.mass >= 0.01 && (0.0..=1.0).contains(&c.max_overshoot));
                        assert!(c.settle_seconds().is_finite() && c.settle_seconds() > 0.0);
                        for &u in &[f64::NAN, -1.0, 0.5, 0.999, 7.0] {
                            for &secs in &[f64::NAN, 0.0, 1e-9, 1e12] {
                                let p = s.progress(u, secs);
                                assert!(p.is_finite() && (-1.0..=2.0).contains(&p), "{s:?} u {u} {secs}: {p}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn steps_jump_evenly_and_land_on_the_end() {
        assert_eq!(steps_progress(4, 0.0), 0.0);
        assert_eq!(steps_progress(4, 0.24), 0.0);
        assert_eq!(steps_progress(4, 0.25), 0.25);
        assert_eq!(steps_progress(4, 0.74), 0.5);
        assert_eq!(steps_progress(4, 0.99), 0.75);
        assert_eq!(steps_progress(4, 1.0), 1.0);
        // n is bounded: 0 is one step, huge is MAX_STEPS
        assert_eq!(steps_progress(0, 0.5), 0.0);
        let huge = steps_progress(u32::MAX, 0.5004);
        assert!((huge - 0.5).abs() < 1e-12, "{huge}");
        for u in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(steps_progress(3, u).is_finite());
        }
        let distinct: std::collections::BTreeSet<u64> = samples(|u| steps_progress(5, u)).iter().map(|v| v.to_bits()).collect();
        assert_eq!(distinct.len(), 6, "five plateaus and the end");
    }

    #[test]
    fn custom_curves_pass_their_points_and_stay_monotone() {
        let pts = [[0.2, 0.5], [0.5, 0.8], [0.8, 0.95]];
        for p in pts {
            assert!((curve_progress(&pts, p[0]) - p[1]).abs() < 1e-12, "{p:?}");
        }
        assert_eq!(curve_progress(&pts, 0.0), 0.0);
        assert_eq!(curve_progress(&pts, 1.0), 1.0);
        let v = samples(|u| curve_progress(&pts, u));
        assert!(v.windows(2).all(|w| w[1] >= w[0] - 1e-12), "monotone points give a monotone curve");
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)), "and no swing past them");
        // an overshooting curve keeps its peak and is clamped to the range
        let back = [[0.6, 1.1]];
        assert!((curve_progress(&back, 0.6) - 1.1).abs() < 1e-12);
        assert!(curve_progress(&[[0.5, 50.0]], 0.5) <= CURVE_RANGE.1);
        // no points: the default S
        assert!(curve_progress(&[], 0.1) < 0.1 && curve_progress(&[], 0.9) > 0.9);
    }

    #[test]
    fn hostile_curves_are_read_safely() {
        let bad = vec![[f64::NAN, 0.5], [0.5, f64::INFINITY], [-0.2, 0.1], [0.4, 0.3], [0.3, 0.9], [0.4, 0.2], [1.0, 0.5], [7.0, 1.0]];
        let v = samples(|u| curve_progress(&bad, u));
        assert!(v.iter().all(|x| x.is_finite()));
        // only (0.4, 0.3) is usable: the curve goes through it
        assert!((curve_progress(&bad, 0.4) - 0.3).abs() < 1e-12);
        // far too many points: the first MAX_CURVE_POINTS are read, nothing allocates
        let many: Vec<[f64; 2]> = (1..100_000).map(|i| [i as f64 / 100_000.0, i as f64 / 100_000.0]).collect();
        assert!((curve_progress(&many, 0.5) - 0.5).abs() < 1e-9);
        // points too close to divide by safely still give finite values
        let tight = [[1e-300, 0.5], [2e-300, -0.5], [0.5, 2.0]];
        assert!(samples(|u| curve_progress(&tight, u)).iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_wiggle_is_deterministic_bounded_and_fades_at_the_ends() {
        let w = Wiggle { amplitude: 8.0, frequency: 3.0, seed: 42 };
        for c in 0..3 {
            for i in 0..200 {
                let t = i as f64 * 0.013;
                let u = (i as f64 / 199.0).clamp(0.0, 1.0);
                let a = w.offset(c, t, u);
                assert_eq!(a, w.offset(c, t, u), "the same every time");
                assert!(a.abs() <= 8.0 + 1e-9, "{a}");
            }
            assert_eq!(w.offset(c, 1.0, 0.0), 0.0);
            assert_eq!(w.offset(c, 1.0, 1.0), 0.0);
        }
        // another seed or another component gives another wobble; it actually moves
        let trace = |w: Wiggle, c: usize| (0..100).map(|i| w.offset(c, i as f64 * 0.05, 0.5)).collect::<Vec<_>>();
        assert_ne!(trace(w, 0), trace(Wiggle { seed: 43, ..w }, 0));
        assert_ne!(trace(w, 0), trace(w, 1));
        let span = trace(w, 0).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(span > 2.0, "{span}");
        // smooth: neighbouring samples 1 ms apart are close
        assert!((w.offset(0, 1.0, 0.5) - w.offset(0, 1.001, 0.5)).abs() < 0.2);
        // hostile numbers
        for (a, f, t, u) in [(f64::NAN, 1.0, 1.0, 0.5), (1e300, 1e300, 1e300, 0.5), (5.0, -3.0, f64::INFINITY, 0.5), (5.0, 2.0, -1e20, f64::NAN)] {
            let o = Wiggle { amplitude: a, frequency: f, seed: u64::MAX }.offset(usize::MAX, t, u);
            assert!(o.is_finite() && o.abs() <= MAX_WIGGLE_AMPLITUDE, "{o}");
        }
    }

    #[test]
    fn log_space_steps_by_equal_ratios_and_refuses_zero_and_mixed_signs() {
        // 100 → 400 %: halfway is 200 %, a quarter is 141 %
        assert!((log_lerp(100.0, 400.0, 0.5).unwrap() - 200.0).abs() < 1e-9);
        assert!((log_lerp(100.0, 400.0, 0.25).unwrap() - 100.0 * 2f64.sqrt()).abs() < 1e-9);
        let steps: Vec<f64> = (0..=4).map(|i| log_lerp(100.0, 1600.0, i as f64 / 4.0).unwrap()).collect();
        for w in steps.windows(2) {
            assert!((w[1] / w[0] - 2.0).abs() < 1e-9, "{steps:?}");
        }
        // both negative: magnitudes in log space, the sign kept
        assert!((log_lerp(-100.0, -400.0, 0.5).unwrap() + 200.0).abs() < 1e-9);
        for (x, y) in [(0.0, 100.0), (100.0, 0.0), (-5.0, 5.0), (0.0, 0.0), (f64::NAN, 1.0), (1.0, f64::INFINITY)] {
            assert_eq!(log_lerp(x, y, 0.5), None, "{x} → {y}");
        }
        assert!(log_lerp(1e-300, 1e300, 0.5).is_some_and(f64::is_finite));
    }

    #[test]
    fn segment_ease_round_trips_and_a_plain_one_writes_nothing() {
        assert!(PLAIN.is_plain());
        assert_eq!(serde_json::to_string(&SegmentEase::default()).unwrap(), "{}");
        let full = SegmentEase {
            spring: Some(Spring { damping_ratio: 0.6, stiffness: 300.0, mass: 2.0, max_overshoot: 0.1 }),
            steps: Some(7),
            wiggle: Some(Wiggle { amplitude: 3.0, frequency: 5.0, seed: u64::MAX }),
            curve: vec![[0.25, 0.5], [0.75, 1.05]],
            perceptual: true,
        };
        let j = serde_json::to_string(&full).unwrap();
        assert_eq!(serde_json::from_str::<SegmentEase>(&j).unwrap(), full);
        // partial objects take the defaults
        let s: Spring = serde_json::from_str(r#"{"damping_ratio": 1.0}"#).unwrap();
        assert_eq!(s, Spring { damping_ratio: 1.0, ..Spring::DEFAULT });
        let w: Wiggle = serde_json::from_str("{}").unwrap();
        assert_eq!(w, Wiggle::DEFAULT);
    }
}
