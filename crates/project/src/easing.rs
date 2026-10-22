//! Easing curves: Robert Penner's easing families (sine, quad, cubic, quart, quint, expo, circ,
//! back, elastic, bounce; in / out / in-out), written here from their mathematical definitions,
//! and a CSS-style `cubic-bezier(x1, y1, x2, y2)` given as data. The choice indices are those of
//! [`crate::gltransition::EASING_OPTS`].
//!
//! Transitions ease their progress with [`ease`], owned by the host: evaluated on the CPU in `f64`
//! from the progress the timeline computes at the exact frame `Tick`, so the shader (GPU) and its
//! reference (CPU) receive the same value and preview and export agree. Curves that overshoot
//! (back, elastic, a Bézier with y outside 0..1) are clamped to 0..1 there: the shaders are only
//! defined there. Text animators ([`crate::text_anim`]) use [`ease_overshoot`], which keeps the
//! overshoot (a pop that grows past its size and settles back).
//!
//! Every curve maps 0 → 0 and 1 → 1 exactly.

use std::f64::consts::PI;

/// A CSS-style cubic Bézier timing curve through (0, 0), (x1, y1), (x2, y2), (1, 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bezier {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl Bezier {
    /// CSS `ease`.
    pub const EASE: Bezier = Bezier { x1: 0.25, y1: 0.1, x2: 0.25, y2: 1.0 };

    /// The curve with hostile values made safe: x in 0..1 (so time stays monotonic), y finite
    /// and within -1..2; a non-finite coordinate takes [`Bezier::EASE`]'s.
    pub fn sanitized(self) -> Bezier {
        let x = |v: f64, d: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { d };
        let y = |v: f64, d: f64| if v.is_finite() { v.clamp(-1.0, 2.0) } else { d };
        let e = Bezier::EASE;
        Bezier { x1: x(self.x1, e.x1), y1: y(self.y1, e.y1), x2: x(self.x2, e.x2), y2: y(self.y2, e.y2) }
    }
}

/// Number of easing curves ([`crate::gltransition::EASING_OPTS`]).
pub const COUNT: u32 = 32;

/// Ease `t` with curve `kind` (an index into `EASING_OPTS`; unknown indices are linear). The
/// result is in 0..1, 0 for t ≤ 0 (or NaN) and 1 for t ≥ 1.
pub fn ease(kind: u32, bezier: Bezier, t: f64) -> f64 {
    ease_overshoot(kind, bezier, t).clamp(0.0, 1.0)
}

/// [`ease`] keeping the overshoot: back, elastic and a Bézier with y outside 0..1 go past 0 and 1
/// between the ends (back by about 10 %; the result is kept within -2..3). 0 for t ≤ 0 (or NaN),
/// 1 for t ≥ 1, always finite.
pub fn ease_overshoot(kind: u32, bezier: Bezier, t: f64) -> f64 {
    if t.is_nan() || t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let v = match kind {
        0 => t,
        // Back and Elastic in-out use their own overshoot / period, as in the standard set.
        24 => {
            let c2 = 1.70158 * 1.525;
            if t < 0.5 {
                (2.0 * t).powi(2) * ((c2 + 1.0) * 2.0 * t - c2) / 2.0
            } else {
                ((2.0 * t - 2.0).powi(2) * ((c2 + 1.0) * (2.0 * t - 2.0) + c2) + 2.0) / 2.0
            }
        }
        27 => {
            let c5 = 2.0 * PI / 4.5;
            if t < 0.5 {
                -((2.0f64).powf(20.0 * t - 10.0) * ((20.0 * t - 11.125) * c5).sin()) / 2.0
            } else {
                (2.0f64).powf(-20.0 * t + 10.0) * ((20.0 * t - 11.125) * c5).sin() / 2.0 + 1.0
            }
        }
        1..=30 => {
            let family = (kind - 1) / 3;
            let f = |t: f64| ease_in(family, t);
            match (kind - 1) % 3 {
                0 => f(t),
                1 => 1.0 - f(1.0 - t),
                _ => {
                    if t < 0.5 {
                        f(2.0 * t) / 2.0
                    } else {
                        1.0 - f(2.0 - 2.0 * t) / 2.0
                    }
                }
            }
        }
        31 => {
            let b = bezier.sanitized();
            crate::keyframe::cubic_bezier_y_for_x(b.x1, b.y1, b.x2, b.y2, t)
        }
        _ => t,
    };
    if v.is_finite() { v.clamp(-2.0, 3.0) } else { t }
}

/// The "in" curve of a Penner family (0 sine … 9 bounce); "out" and "in-out" are derived from it
/// by symmetry, which gives the standard definitions.
fn ease_in(family: u32, t: f64) -> f64 {
    match family {
        0 => 1.0 - (t * PI / 2.0).cos(),
        1 => t * t,
        2 => t * t * t,
        3 => t * t * t * t,
        4 => t * t * t * t * t,
        5 => {
            if t <= 0.0 {
                0.0
            } else {
                (2.0f64).powf(10.0 * t - 10.0)
            }
        }
        6 => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        7 => {
            // overshoot 1.70158 (10 % back)
            let c1 = 1.70158;
            (c1 + 1.0) * t * t * t - c1 * t * t
        }
        8 => {
            if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                let c4 = 2.0 * PI / 3.0;
                -(2.0f64).powf(10.0 * t - 10.0) * ((10.0 * t - 10.75) * c4).sin()
            }
        }
        _ => 1.0 - bounce_out(1.0 - t),
    }
}

fn bounce_out(t: f64) -> f64 {
    let (n1, d1) = (7.5625, 2.75);
    if t < 1.0 / d1 {
        n1 * t * t
    } else if t < 2.0 / d1 {
        let t = t - 1.5 / d1;
        n1 * t * t + 0.75
    } else if t < 2.5 / d1 {
        let t = t - 2.25 / d1;
        n1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / d1;
        n1 * t * t + 0.984375
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overshoot_is_kept_for_text_animators() {
        // back out goes ~10 % past 1 before settling; back in dips below 0
        let peak = (1..100).map(|i| ease_overshoot(23, Bezier::EASE, i as f64 / 100.0)).fold(0.0, f64::max);
        assert!(peak > 1.05 && peak < 1.15, "{peak}");
        assert!(ease_overshoot(22, Bezier::EASE, 0.2) < 0.0);
        for kind in 0..COUNT + 3 {
            assert_eq!(ease_overshoot(kind, Bezier::EASE, 0.0), 0.0);
            assert_eq!(ease_overshoot(kind, Bezier::EASE, 1.0), 1.0);
            for i in 0..=100 {
                let v = ease_overshoot(kind, Bezier { x1: f64::NAN, y1: 9.0, x2: -3.0, y2: f64::INFINITY }, i as f64 / 100.0);
                assert!(v.is_finite() && (-2.0..=3.0).contains(&v), "{kind}: {v}");
            }
        }
    }

    #[test]
    fn every_curve_pins_its_ends_and_stays_in_range() {
        for kind in 0..COUNT + 3 {
            assert_eq!(ease(kind, Bezier::EASE, 0.0), 0.0, "{kind}");
            assert_eq!(ease(kind, Bezier::EASE, 1.0), 1.0, "{kind}");
            assert_eq!(ease(kind, Bezier::EASE, -3.0), 0.0, "{kind}");
            assert_eq!(ease(kind, Bezier::EASE, f64::NAN), 0.0, "{kind}");
            assert_eq!(ease(kind, Bezier::EASE, f64::INFINITY), 1.0, "{kind}");
            for i in 1..200 {
                let t = i as f64 / 200.0;
                let v = ease(kind, Bezier::EASE, t);
                assert!((0.0..=1.0).contains(&v), "{kind} at {t}: {v}");
            }
            // continuous at both ends (within the clamp)
            assert!(ease(kind, Bezier::EASE, 1e-9) < 1e-3, "{kind}");
            assert!(ease(kind, Bezier::EASE, 1.0 - 1e-9) > 1.0 - 1e-3, "{kind}");
        }
    }

    #[test]
    fn standard_values() {
        let e = |k, t| ease(k, Bezier::EASE, t);
        assert_eq!(e(0, 0.3), 0.3);
        assert!((e(1, 0.5) - (1.0 - (PI / 4.0).cos())).abs() < 1e-12, "sine in");
        assert!((e(2, 0.5) - (PI / 4.0).sin()).abs() < 1e-12, "sine out");
        assert!((e(4, 0.5) - 0.25).abs() < 1e-12, "quad in");
        assert!((e(5, 0.5) - 0.75).abs() < 1e-12, "quad out");
        assert!((e(9, 0.25) - 4.0 * 0.25f64.powi(3)).abs() < 1e-12, "cubic in-out, first half");
        assert!((e(9, 0.75) - (1.0 - 0.5f64.powi(3) / 2.0)).abs() < 1e-12, "cubic in-out, second half");
        for k in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30] {
            assert!((e(k, 0.5) - 0.5).abs() < 1e-9, "in-out {k} is symmetric about the midpoint");
        }
        assert!((e(16, 0.5) - 1.0 / 32.0).abs() < 1e-12, "expo in");
        assert!((e(29, 0.5) - 0.765625).abs() < 1e-12, "bounce out");
        // back in dips below zero (clamped), back out overshoots (clamped)
        assert_eq!(e(22, 0.2), 0.0);
        assert_eq!(e(23, 0.8), 1.0);
        // custom Bézier: linear control points give the identity, CSS ease is fast in the middle
        let lin = Bezier { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
        assert!((ease(31, lin, 0.3) - 0.3).abs() < 1e-6);
        assert!(ease(31, Bezier::EASE, 0.5) > 0.75);
    }

    #[test]
    fn hostile_bezier_never_panics() {
        let hostile = [f64::NAN, f64::INFINITY, -f64::INFINITY, -1e300, 1e300, 0.0, 1.0, -0.5, 1.5];
        for &x1 in &hostile {
            for &y1 in &hostile {
                for &x2 in &hostile {
                    for &y2 in &hostile {
                        let b = Bezier { x1, y1, x2, y2 };
                        for t in [0.0, 1e-7, 0.25, 0.5, 0.9999, 1.0] {
                            let v = ease(31, b, t);
                            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{b:?} at {t}: {v}");
                        }
                    }
                }
            }
        }
    }
}
