//! Keyframed parameters.
//!
//! A parameter is a static value or a list of keyframes (clip-relative ticks). Interpolation per
//! keyframe governs the segment leaving it: Linear, Hold, Bezier (with influence/speed handles),
//! Auto Bezier, Continuous Bezier, Ease In/Out (Bezier presets) — Premiere's temporal set — and
//! beyond it Spring, Steps, Wiggle and Custom curves, whose numbers and the perceptual (log-space)
//! option live in the keyframe's [`SegmentEase`] ([`crate::keyframe_ease`]).
//!
//! Every evaluation goes through [`Param::value_at`] (and its allocation-free scalar twin
//! [`Param::scalar_at`], used per audio sample), so preview, export, audio and the graphs all see
//! the same curve. The new modes shape only the segment leaving their keyframe (the next
//! keyframe's Ease In does not bend them); a Wiggle segment is Linear plus its wobble. Perceptual
//! and Wiggle apply to numbers and points; colours and mask paths take the mode's progress
//! linearly (a Wiggle colour or path is plain Linear).

use filmcraft_geom::Vec2;
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::{Deserialize, Serialize};

use crate::keyframe_ease::{PLAIN, curve_progress, log_lerp, steps_progress};
pub use crate::keyframe_ease::{SegmentEase, Spring, Wiggle};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f64),
    /// A point. A coordinate may be NaN: point parameters use NaN for "auto" (the frame or source
    /// centre) until [`crate::resolve_auto_points`] fills it in. JSON has no NaN and writes it as
    /// `null`, so `null` reads back as NaN here, and only here: a `null` coordinate anywhere else
    /// in a project (a mask vertex, a layer rectangle) is still a damaged file.
    Vec2(#[serde(deserialize_with = "point_or_auto")] Vec2),
    Color([f32; 4]),
    Bool(bool),
    Choice(u32),
    Text(String),
    /// Curve control points (x, y) in 0..1, sorted by x (Lumetri curves).
    Curve(Vec<[f32; 2]>),
    /// A mask path (Bézier vertices in clip pixels); interpolates vertex-wise.
    Path(crate::mask::MaskPath),
}

/// Read a point parameter's value, taking a `null` coordinate as NaN ("auto").
fn point_or_auto<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec2, D::Error> {
    fn nan_if_null<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
    #[derive(Deserialize)]
    struct Point {
        #[serde(deserialize_with = "nan_if_null")]
        x: f64,
        #[serde(deserialize_with = "nan_if_null")]
        y: f64,
    }
    let p = Point::deserialize(d)?;
    Ok(Vec2::new(p.x, p.y))
}

impl ParamValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            ParamValue::Choice(c) => Some(*c as f64),
            ParamValue::Bool(b) => Some(*b as u8 as f64),
            _ => None,
        }
    }
    pub fn as_vec2(&self) -> Option<Vec2> {
        match self {
            ParamValue::Vec2(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_color(&self) -> Option<[f32; 4]> {
        match self {
            ParamValue::Color(c) => Some(*c),
            _ => None,
        }
    }
    pub fn as_curve(&self) -> Option<&[[f32; 2]]> {
        match self {
            ParamValue::Curve(c) => Some(c),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// Whether values of this kind can be interpolated (others hold).
    pub fn interpolates(&self) -> bool {
        matches!(self, ParamValue::Float(_) | ParamValue::Vec2(_) | ParamValue::Color(_) | ParamValue::Path(_))
    }
    pub fn as_path(&self) -> Option<&crate::mask::MaskPath> {
        match self {
            ParamValue::Path(p) => Some(p),
            _ => None,
        }
    }
    fn components(&self) -> Vec<f64> {
        match self {
            ParamValue::Float(v) => vec![*v],
            ParamValue::Vec2(v) => vec![v.x, v.y],
            ParamValue::Color(c) => c.iter().map(|&x| x as f64).collect(),
            ParamValue::Path(p) => p.components(),
            _ => vec![],
        }
    }
    fn from_components(template: &ParamValue, c: &[f64]) -> ParamValue {
        match template {
            ParamValue::Float(_) => ParamValue::Float(c[0]),
            ParamValue::Vec2(_) => ParamValue::Vec2(Vec2::new(c[0], c[1])),
            ParamValue::Color(_) => ParamValue::Color([c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]),
            ParamValue::Path(p) => ParamValue::Path(p.with_components(c)),
            other => other.clone(),
        }
    }
}

/// How the segment leaving a keyframe moves. New variants go at the end (the order is a cache
/// key, `as u64`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    #[default]
    Linear,
    Hold,
    Bezier,
    AutoBezier,
    ContinuousBezier,
    EaseIn,
    EaseOut,
    /// A damped spring ([`Spring`], from [`SegmentEase::spring`] or the default).
    Spring,
    /// `steps(n)` ([`SegmentEase::steps`], default [`crate::keyframe_ease::DEFAULT_STEPS`]).
    Steps,
    /// Linear plus a seeded wobble ([`Wiggle`]).
    Wiggle,
    /// A custom curve ([`SegmentEase::curve`]).
    Custom,
}

impl Interpolation {
    pub const ALL: [Interpolation; 11] = [
        Interpolation::Linear,
        Interpolation::Bezier,
        Interpolation::AutoBezier,
        Interpolation::ContinuousBezier,
        Interpolation::Hold,
        Interpolation::EaseIn,
        Interpolation::EaseOut,
        Interpolation::Spring,
        Interpolation::Steps,
        Interpolation::Wiggle,
        Interpolation::Custom,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Interpolation::Linear => "Linear",
            Interpolation::Hold => "Hold",
            Interpolation::Bezier => "Bezier",
            Interpolation::AutoBezier => "Auto Bezier",
            Interpolation::ContinuousBezier => "Continuous Bezier",
            Interpolation::EaseIn => "Ease In",
            Interpolation::EaseOut => "Ease Out",
            Interpolation::Spring => "Spring",
            Interpolation::Steps => "Steps",
            Interpolation::Wiggle => "Wiggle",
            Interpolation::Custom => "Custom Curve",
        }
    }
    /// The command name (`effects.setInterpolation`, the Effect Controls menu's automation ids).
    pub fn name(self) -> &'static str {
        match self {
            Interpolation::Linear => "linear",
            Interpolation::Hold => "hold",
            Interpolation::Bezier => "bezier",
            Interpolation::AutoBezier => "autoBezier",
            Interpolation::ContinuousBezier => "continuousBezier",
            Interpolation::EaseIn => "easeIn",
            Interpolation::EaseOut => "easeOut",
            Interpolation::Spring => "spring",
            Interpolation::Steps => "steps",
            Interpolation::Wiggle => "wiggle",
            Interpolation::Custom => "custom",
        }
    }
    /// The mode a name says, any case (`customCurve` and `curve` name Custom).
    pub fn from_name(s: &str) -> Option<Interpolation> {
        let k = s.trim().to_ascii_lowercase();
        let k = if matches!(k.as_str(), "customcurve" | "curve") { "custom" } else { k.as_str() };
        Interpolation::ALL.into_iter().find(|i| i.name().eq_ignore_ascii_case(k))
    }
    /// Whether the segment leaving a keyframe of this mode starts with a Bézier ease (the
    /// keyframe's outgoing influence handle applies).
    pub fn eases_out(self) -> bool {
        matches!(self, Interpolation::Bezier | Interpolation::AutoBezier | Interpolation::ContinuousBezier | Interpolation::EaseOut)
    }
    /// Whether the segment arriving at a keyframe of this mode ends with a Bézier ease (the
    /// keyframe's incoming influence handle applies, unless the segment's own mode ignores it).
    pub fn eases_in(self) -> bool {
        matches!(self, Interpolation::Bezier | Interpolation::AutoBezier | Interpolation::ContinuousBezier | Interpolation::EaseIn)
    }
    /// Whether a segment of this mode has a shape of its own that the next keyframe's Ease In
    /// does not bend (Spring, Steps and Custom; Wiggle is Linear underneath).
    pub fn own_shape(self) -> bool {
        matches!(self, Interpolation::Spring | Interpolation::Steps | Interpolation::Custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time relative to the start of the track item (clip time).
    pub time: Tick,
    pub value: ParamValue,
    /// Interpolation of the segment leaving this keyframe.
    pub interp: Interpolation,
    /// Temporal ease: influence (0..1) of the outgoing/incoming handles.
    #[serde(default = "default_influence")]
    pub out_influence: f64,
    #[serde(default = "default_influence")]
    pub in_influence: f64,
    /// The leaving segment's spring, steps, wiggle and custom curve numbers and its perceptual
    /// option (PX5.1). None on most keyframes, and on every keyframe of an older project.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ease: Option<Box<SegmentEase>>,
}

fn default_influence() -> f64 {
    1.0 / 3.0
}

impl Keyframe {
    pub fn new(time: Tick, value: ParamValue) -> Self {
        Self { time, value, interp: Interpolation::Linear, out_influence: default_influence(), in_influence: default_influence(), ease: None }
    }
    /// The leaving segment's numbers (all defaults when it has none).
    pub fn segment(&self) -> &SegmentEase {
        self.ease.as_deref().unwrap_or(&PLAIN)
    }
    /// The leaving segment's numbers, to edit; call [`Keyframe::tidy_ease`] after.
    pub fn segment_mut(&mut self) -> &mut SegmentEase {
        self.ease.get_or_insert_with(Default::default)
    }
    /// Drop the segment numbers when nothing is set (so the file stays as before).
    pub fn tidy_ease(&mut self) {
        if self.ease.as_deref().is_some_and(SegmentEase::is_plain) {
            self.ease = None;
        }
    }
}

/// A (possibly animated) parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub value: ParamValue,
    /// Sorted by time. Empty = static.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keyframes: Vec<Keyframe>,
}

impl Param {
    pub fn new(value: ParamValue) -> Self {
        Self { value, keyframes: Vec::new() }
    }
    pub fn is_animated(&self) -> bool {
        !self.keyframes.is_empty()
    }
    /// Apply `f` to the static value and every keyframe value.
    pub fn map_values(&mut self, f: impl Fn(ParamValue) -> ParamValue) {
        self.value = f(std::mem::replace(&mut self.value, ParamValue::Float(0.0)));
        for k in &mut self.keyframes {
            k.value = f(std::mem::replace(&mut k.value, ParamValue::Float(0.0)));
        }
    }

    /// Set a value at `t`: updates the static value, or adds/replaces a keyframe when animated.
    pub fn set_at(&mut self, t: Tick, value: ParamValue) {
        if self.keyframes.is_empty() {
            self.value = value;
            return;
        }
        match self.keyframes.binary_search_by_key(&t, |k| k.time) {
            Ok(i) => self.keyframes[i].value = value,
            Err(i) => {
                let prev = i.checked_sub(1).and_then(|j| self.keyframes.get(j));
                let mut k = Keyframe::new(t, value);
                k.interp = prev.map(|k| k.interp).unwrap_or_default();
                k.ease = prev.and_then(|k| k.ease.clone());
                self.keyframes.insert(i, k);
            }
        }
    }

    /// Toggle animation (the stopwatch): on → one keyframe at `t` with the current value; off → keep value at `t`.
    pub fn toggle_animation(&mut self, t: Tick) {
        if self.keyframes.is_empty() {
            self.keyframes.push(Keyframe::new(t, self.value.clone()));
        } else {
            self.value = self.value_at(t);
            self.keyframes.clear();
        }
    }

    pub fn add_keyframe(&mut self, t: Tick) {
        let v = self.value_at(t);
        if self.keyframes.is_empty() {
            self.value = v.clone();
        }
        self.set_at_force(t, v);
    }

    /// Add or replace a keyframe at `t` (the parameter becomes animated).
    pub fn put_keyframe(&mut self, t: Tick, value: ParamValue) {
        self.set_at_force(t, value);
    }

    fn set_at_force(&mut self, t: Tick, value: ParamValue) {
        match self.keyframes.binary_search_by_key(&t, |k| k.time) {
            Ok(i) => self.keyframes[i].value = value,
            Err(i) => self.keyframes.insert(i, Keyframe::new(t, value)),
        }
    }

    pub fn remove_keyframe_at(&mut self, t: Tick) -> bool {
        if let Ok(i) = self.keyframes.binary_search_by_key(&t, |k| k.time) {
            let removed = self.keyframes.remove(i);
            if self.keyframes.is_empty() {
                self.value = removed.value;
            }
            true
        } else {
            false
        }
    }

    /// The keyframes around clip time `t` and how far `t` is between them, or the value that
    /// holds there (before the first keyframe, after the last, on one, or a static parameter).
    /// Hostile keyframe lists (unsorted, repeated times) give a held value, never a panic.
    fn locate(&self, t: Tick) -> At<'_> {
        let k = &self.keyframes;
        let (Some(first), Some(last)) = (k.first(), k.last()) else { return At::Held(&self.value) };
        if t <= first.time {
            return At::Held(&first.value);
        }
        if t >= last.time {
            return At::Held(&last.value);
        }
        let i = match k.binary_search_by_key(&t, |kf| kf.time) {
            Ok(i) => return At::Held(k.get(i).map_or(&last.value, |x| &x.value)),
            Err(i) => i.saturating_sub(1),
        };
        let (Some(a), Some(b)) = (k.get(i), k.get(i + 1)) else { return At::Held(&last.value) };
        let span = b.time.0.saturating_sub(a.time.0);
        if span <= 0 {
            return At::Held(&a.value);
        }
        let u = (t.0.saturating_sub(a.time.0) as f64 / span as f64).clamp(0.0, 1.0);
        At::Between { a, b, u, seconds: span as f64 / TICKS_PER_SECOND as f64, at: t.0 as f64 / TICKS_PER_SECOND as f64 }
    }

    /// Evaluate at clip time `t`.
    pub fn value_at(&self, t: Tick) -> ParamValue {
        let (a, b, seg) = match self.locate(t) {
            At::Held(v) => return v.clone(),
            At::Between { a, b, u, seconds, at } => (a, b, Segment::new(a, b, u, seconds, at)),
        };
        if a.interp == Interpolation::Hold || !a.value.interpolates() {
            return a.value.clone();
        }
        let ca = a.value.components();
        let cb = b.value.components();
        if ca.len() != cb.len() {
            // e.g. mask paths with different vertex counts: hold
            return a.value.clone();
        }
        let numeric = matches!(a.value, ParamValue::Float(_) | ParamValue::Vec2(_));
        let c: Vec<f64> = ca.iter().zip(&cb).enumerate().map(|(i, (x, y))| seg.mix(i, *x, *y, numeric)).collect();
        ParamValue::from_components(&a.value, &c)
    }

    pub fn f64_at(&self, t: Tick) -> f64 {
        self.value_at(t).as_f64().unwrap_or(0.0)
    }

    /// Allocation-free [`Param::f64_at`] for scalar parameters (Float / Choice / Bool), for
    /// per-sample evaluation in the audio mixer. Non-scalar values evaluate to 0. Every mode
    /// evaluates as in [`Param::value_at`].
    pub fn scalar_at(&self, t: Tick) -> f64 {
        let s = |v: &ParamValue| v.as_f64().unwrap_or(0.0);
        let (a, b, seg) = match self.locate(t) {
            At::Held(v) => return s(v),
            At::Between { a, b, u, seconds, at } => (a, b, Segment::new(a, b, u, seconds, at)),
        };
        let (va, vb) = (s(&a.value), s(&b.value));
        if a.interp == Interpolation::Hold || !matches!(a.value, ParamValue::Float(_)) {
            return va;
        }
        seg.mix(0, va, vb, true)
    }

    pub fn vec2_at(&self, t: Tick) -> Vec2 {
        self.value_at(t).as_vec2().unwrap_or_default()
    }

    /// Previous/next keyframe times for navigation.
    pub fn prev_keyframe(&self, t: Tick) -> Option<Tick> {
        self.keyframes.iter().rev().map(|k| k.time).find(|&k| k < t)
    }
    pub fn next_keyframe(&self, t: Tick) -> Option<Tick> {
        self.keyframes.iter().map(|k| k.time).find(|&k| k > t)
    }
}

/// Where a clip time falls in a parameter's keyframes ([`Param::locate`]).
enum At<'a> {
    /// A value that holds there.
    Held(&'a ParamValue),
    /// Between keyframes `a` and `b`, `u` (0..1) of the way, the segment `seconds` long, at clip
    /// time `at` seconds.
    Between { a: &'a Keyframe, b: &'a Keyframe, u: f64, seconds: f64, at: f64 },
}

/// One evaluation of the segment leaving keyframe `a`: its eased progress and what each value
/// component adds to it.
struct Segment<'a> {
    a: &'a Keyframe,
    /// Eased progress (0 at `a`, 1 at the next keyframe; beyond for an overshoot).
    e: f64,
    u: f64,
    at: f64,
}

impl<'a> Segment<'a> {
    fn new(a: &'a Keyframe, b: &Keyframe, u: f64, seconds: f64, at: f64) -> Self {
        let e = match a.interp {
            Interpolation::Spring => a.segment().spring.unwrap_or_default().progress(u, seconds),
            Interpolation::Steps => steps_progress(a.segment().step_count(), u),
            Interpolation::Custom => curve_progress(&a.segment().curve, u),
            // Wiggle's base is Linear (an Ease In on `b` bends it as it bends Linear)
            _ => ease(a, b, u),
        };
        Segment { a, e: if e.is_finite() { e } else { u }, u, at }
    }

    /// Component `i` between `x` and `y`: in log space for a perceptual segment of numbers or
    /// points (linear when the two differ in sign or one is zero), plus a Wiggle's wobble.
    fn mix(&self, i: usize, x: f64, y: f64, numeric: bool) -> f64 {
        let ease = self.a.segment();
        let linear = x + (y - x) * self.e;
        let mut v = if numeric && ease.perceptual { log_lerp(x, y, self.e).unwrap_or(linear) } else { linear };
        if numeric && self.a.interp == Interpolation::Wiggle {
            v += ease.wiggle.unwrap_or_default().offset(i, self.at, self.u);
        }
        if v.is_finite() { v } else { linear }
    }
}

/// Temporal easing for u∈[0,1] between two keyframes (1D cubic Bezier in time/value space).
fn ease(a: &Keyframe, b: &Keyframe, u: f64) -> f64 {
    let out_ease = a.interp.eases_out();
    let in_ease = b.interp.eases_in() || matches!(a.interp, Interpolation::Bezier | Interpolation::ContinuousBezier);
    if !out_ease && !in_ease {
        return u;
    }
    // Control points in normalized (time, value) space; flat handles = ease (velocity 0).
    let x1 = if out_ease { a.out_influence.clamp(0.01, 1.0) } else { 1.0 / 3.0 };
    let y1 = if out_ease { 0.0 } else { 1.0 / 3.0 };
    let x2 = if in_ease { 1.0 - b.in_influence.clamp(0.01, 1.0) } else { 2.0 / 3.0 };
    let y2 = if in_ease { 1.0 } else { 2.0 / 3.0 };
    cubic_bezier_y_for_x(x1, y1, x2, y2, u)
}

/// Solve a CSS-style cubic-bezier(x1,y1,x2,y2) for y at x (Newton + bisection).
pub fn cubic_bezier_y_for_x(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    let bx = |s: f64| 3.0 * (1.0 - s) * (1.0 - s) * s * x1 + 3.0 * (1.0 - s) * s * s * x2 + s * s * s;
    let by = |s: f64| 3.0 * (1.0 - s) * (1.0 - s) * s * y1 + 3.0 * (1.0 - s) * s * s * y2 + s * s * s;
    let dbx = |s: f64| 3.0 * (1.0 - s) * (1.0 - s) * x1 + 6.0 * (1.0 - s) * s * (x2 - x1) + 3.0 * s * s * (1.0 - x2);
    let mut s = x;
    for _ in 0..8 {
        let d = dbx(s);
        if d.abs() < 1e-9 {
            break;
        }
        let ns = s - (bx(s) - x) / d;
        if !(0.0..=1.0).contains(&ns) {
            break;
        }
        s = ns;
    }
    if (bx(s) - x).abs() > 1e-7 {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..60 {
            s = (lo + hi) / 2.0;
            if bx(s) < x { lo = s } else { hi = s }
        }
    }
    by(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_hold() {
        let mut p = Param::new(ParamValue::Float(0.0));
        p.toggle_animation(Tick(0));
        p.set_at(Tick(100), ParamValue::Float(10.0));
        assert_eq!(p.f64_at(Tick(50)), 5.0);
        assert_eq!(p.f64_at(Tick(-5)), 0.0);
        assert_eq!(p.f64_at(Tick(500)), 10.0);
        p.keyframes[0].interp = Interpolation::Hold;
        assert_eq!(p.f64_at(Tick(99)), 0.0);
    }

    #[test]
    fn set_at_before_first_keyframe_uses_default_interp() {
        let mut p = Param::new(ParamValue::Float(0.0));
        p.toggle_animation(Tick(100));
        p.keyframes[0].interp = Interpolation::Hold;
        p.set_at(Tick(0), ParamValue::Float(5.0));
        assert_eq!(p.keyframes[0].interp, Interpolation::Linear);
        assert_eq!(p.keyframes[1].interp, Interpolation::Hold);
        p.set_at(Tick(200), ParamValue::Float(7.0));
        assert_eq!(p.keyframes[2].interp, Interpolation::Hold);
    }

    #[test]
    fn bezier_eases() {
        let mut p = Param::new(ParamValue::Vec2(Vec2::new(0.0, 0.0)));
        p.toggle_animation(Tick(0));
        p.set_at(Tick(1000), ParamValue::Vec2(Vec2::new(100.0, 50.0)));
        p.keyframes[0].interp = Interpolation::Bezier;
        p.keyframes[1].interp = Interpolation::Bezier;
        let early = p.vec2_at(Tick(100)).x;
        let mid = p.vec2_at(Tick(500)).x;
        assert!(early < 10.0, "ease-in start should be slow: {early}");
        assert!((mid - 50.0).abs() < 1.0, "symmetric ease midpoint: {mid}");
        // monotonic
        let mut last = -1.0;
        for i in 0..=100 {
            let v = p.vec2_at(Tick(i * 10)).x;
            assert!(v >= last - 1e-9);
            last = v;
        }
    }

    #[test]
    fn toggle_off_keeps_value() {
        let mut p = Param::new(ParamValue::Float(1.0));
        p.toggle_animation(Tick(0));
        p.set_at(Tick(10), ParamValue::Float(3.0));
        p.toggle_animation(Tick(5));
        assert!(!p.is_animated());
        assert_eq!(p.value, ParamValue::Float(2.0));
    }

    /// A parameter animated from `a` at 0 to `b` at one second, the first segment in `mode`.
    fn segment(a: ParamValue, b: ParamValue, mode: Interpolation, ease: SegmentEase) -> Param {
        let mut p = Param::new(a.clone());
        p.put_keyframe(Tick::ZERO, a);
        p.put_keyframe(Tick(TICKS_PER_SECOND), b);
        p.keyframes[0].interp = mode;
        p.keyframes[0].ease = Some(Box::new(ease));
        p.keyframes[0].tidy_ease();
        p
    }

    fn at(p: &Param, u: f64) -> f64 {
        p.f64_at(Tick((u * TICKS_PER_SECOND as f64).round() as i64))
    }

    #[test]
    fn every_mode_lands_on_both_keyframes_and_scalar_evaluation_agrees() {
        let modes = [
            (Interpolation::Spring, SegmentEase { spring: Some(Spring { damping_ratio: 0.5, ..Spring::DEFAULT }), ..Default::default() }),
            (Interpolation::Spring, SegmentEase::default()),
            (Interpolation::Steps, SegmentEase { steps: Some(4), ..Default::default() }),
            (Interpolation::Wiggle, SegmentEase { wiggle: Some(Wiggle { amplitude: 5.0, frequency: 3.0, seed: 9 }), ..Default::default() }),
            (Interpolation::Custom, SegmentEase { curve: vec![[0.5, 0.9]], ..Default::default() }),
            (Interpolation::Custom, SegmentEase::default()),
            (Interpolation::Linear, SegmentEase { perceptual: true, ..Default::default() }),
            (Interpolation::Spring, SegmentEase { perceptual: true, ..Default::default() }),
        ];
        for (mode, ease) in modes {
            let p = segment(ParamValue::Float(100.0), ParamValue::Float(400.0), mode, ease.clone());
            assert_eq!(p.f64_at(Tick::ZERO), 100.0, "{mode:?}");
            assert_eq!(p.f64_at(Tick(TICKS_PER_SECOND)), 400.0, "{mode:?}");
            for i in 0..=100 {
                let t = Tick(TICKS_PER_SECOND / 100 * i);
                let (v, s) = (p.f64_at(t), p.scalar_at(t));
                assert!(v.is_finite() && (v - s).abs() < 1e-9, "{mode:?} {ease:?} at {i}: {v} vs {s}");
            }
            // continuous into the last keyframe (Steps jumps onto it by design)
            let near = at(&p, 0.999_999);
            assert!(mode == Interpolation::Steps || (near - 400.0).abs() < 1.0, "{mode:?}: {near}");
        }
        // the modes differ from Linear where they should
        let lin = segment(ParamValue::Float(0.0), ParamValue::Float(1.0), Interpolation::Linear, SegmentEase::default());
        let steps = segment(ParamValue::Float(0.0), ParamValue::Float(1.0), Interpolation::Steps, SegmentEase { steps: Some(4), ..Default::default() });
        assert_eq!(at(&steps, 0.3), 0.25);
        assert!((at(&lin, 0.3) - 0.3).abs() < 1e-9);
        let spring = segment(ParamValue::Float(0.0), ParamValue::Float(1.0), Interpolation::Spring, SegmentEase::default());
        assert!(at(&spring, 0.3) > 0.8, "a spring is quick off the mark: {}", at(&spring, 0.3));
    }

    #[test]
    fn perceptual_scale_zooms_by_even_ratios_and_points_too() {
        let p = segment(ParamValue::Float(100.0), ParamValue::Float(400.0), Interpolation::Linear, SegmentEase { perceptual: true, ..Default::default() });
        assert!((at(&p, 0.5) - 200.0).abs() < 1e-6, "{}", at(&p, 0.5));
        let q: Vec<f64> = (0..=4).map(|i| at(&p, i as f64 / 4.0)).collect();
        for w in q.windows(2) {
            assert!((w[1] / w[0] - 2f64.sqrt()).abs() < 1e-6, "{q:?}");
        }
        // a point: each coordinate in log space
        let v = segment(
            ParamValue::Vec2(Vec2::new(100.0, 50.0)),
            ParamValue::Vec2(Vec2::new(400.0, 200.0)),
            Interpolation::Linear,
            SegmentEase { perceptual: true, ..Default::default() },
        );
        let mid = v.vec2_at(Tick(TICKS_PER_SECOND / 2));
        assert!((mid.x - 200.0).abs() < 1e-6 && (mid.y - 100.0).abs() < 1e-6, "{mid:?}");
        // zero and negative ends: linear, never NaN
        for (a, b) in [(0.0, 100.0), (100.0, 0.0), (-50.0, 50.0), (0.0, 0.0), (-100.0, -400.0)] {
            let p = segment(ParamValue::Float(a), ParamValue::Float(b), Interpolation::Linear, SegmentEase { perceptual: true, ..Default::default() });
            for i in 0..=20 {
                let x = at(&p, i as f64 / 20.0);
                assert!(x.is_finite(), "{a} → {b}");
            }
            let mid = at(&p, 0.5);
            let expect = if a < 0.0 && b < 0.0 { -200.0 } else { (a + b) / 2.0 };
            assert!((mid - expect).abs() < 1e-6, "{a} → {b}: {mid}");
        }
        // colours ignore it (linear)
        let c = segment(
            ParamValue::Color([0.2, 0.2, 0.2, 1.0]),
            ParamValue::Color([0.8, 0.8, 0.8, 1.0]),
            Interpolation::Linear,
            SegmentEase { perceptual: true, ..Default::default() },
        );
        let mid = c.value_at(Tick(TICKS_PER_SECOND / 2)).as_color().unwrap();
        assert!((mid[0] - 0.5).abs() < 1e-6, "{mid:?}");
    }

    #[test]
    fn a_wiggle_segment_is_linear_plus_a_repeatable_wobble() {
        let ease = SegmentEase { wiggle: Some(Wiggle { amplitude: 20.0, frequency: 4.0, seed: 3 }), ..Default::default() };
        let p = segment(ParamValue::Vec2(Vec2::new(0.0, 0.0)), ParamValue::Vec2(Vec2::new(0.0, 0.0)), Interpolation::Wiggle, ease);
        let trace: Vec<Vec2> = (0..=50).map(|i| p.vec2_at(Tick(TICKS_PER_SECOND / 50 * i))).collect();
        assert_eq!(trace, (0..=50).map(|i| p.vec2_at(Tick(TICKS_PER_SECOND / 50 * i))).collect::<Vec<_>>(), "deterministic");
        assert!(trace.iter().all(|v| v.x.abs() <= 20.0 && v.y.abs() <= 20.0));
        assert!(trace.iter().any(|v| v.x.abs() > 1.0), "it moves");
        assert_ne!(trace.iter().map(|v| v.x).collect::<Vec<_>>(), trace.iter().map(|v| v.y).collect::<Vec<_>>(), "x and y wobble apart");
        assert_eq!(p.vec2_at(Tick::ZERO), Vec2::new(0.0, 0.0));
    }

    #[test]
    fn hostile_keyframe_lists_evaluate_without_panicking() {
        // repeated and extreme times, then the same out of order
        for times in [[i64::MIN, 0, 0, 0, i64::MAX], [0, i64::MIN + 1, i64::MAX - 1, 5, -7]] {
            let mut p = Param::new(ParamValue::Float(1.0));
            for (t, v) in times.into_iter().zip([5.0, -5.0, 1.0, 2.0, f64::MAX]) {
                p.keyframes.push(Keyframe::new(Tick(t), ParamValue::Float(v)));
            }
            for (i, k) in p.keyframes.iter_mut().enumerate() {
                k.interp = Interpolation::ALL[i % Interpolation::ALL.len()];
                k.ease = Some(Box::new(SegmentEase {
                    spring: Some(Spring { damping_ratio: f64::NAN, stiffness: f64::INFINITY, mass: 0.0, max_overshoot: -1.0 }),
                    steps: Some(u32::MAX),
                    wiggle: Some(Wiggle { amplitude: f64::INFINITY, frequency: f64::NAN, seed: 7 }),
                    curve: vec![[f64::NAN, 1.0], [0.5, f64::INFINITY]],
                    perceptual: true,
                }));
            }
            for t in [i64::MIN, -8, -7, -1, 0, 1, 3, 1 << 40, i64::MAX - 1, i64::MAX] {
                let v = p.value_at(Tick(t));
                let s = p.scalar_at(Tick(t));
                assert!(v.as_f64().is_some_and(f64::is_finite) && s.is_finite(), "{t}: {v:?} {s}");
            }
        }
    }

    #[test]
    fn keyframes_without_the_new_fields_load_and_save_unchanged() {
        // a keyframe as every build before PX5.1 wrote it
        let old = r#"{"time":0,"value":{"Float":100.0},"interp":"EaseOut","out_influence":0.5,"in_influence":0.25}"#;
        let k: Keyframe = serde_json::from_str(old).unwrap();
        assert_eq!((k.interp, k.out_influence, k.in_influence, k.ease.is_none()), (Interpolation::EaseOut, 0.5, 0.25, true));
        assert_eq!(serde_json::to_string(&k).unwrap(), old);
        // the new modes and their numbers round-trip
        let mut n = k.clone();
        n.interp = Interpolation::Spring;
        n.segment_mut().spring = Some(Spring { damping_ratio: 1.0, ..Spring::DEFAULT });
        n.segment_mut().perceptual = true;
        let j = serde_json::to_string(&n).unwrap();
        assert!(j.contains(r#""interp":"Spring""#) && j.contains(r#""perceptual":true"#), "{j}");
        assert_eq!(serde_json::from_str::<Keyframe>(&j).unwrap(), n);
        for mode in Interpolation::ALL {
            assert_eq!(Interpolation::from_name(mode.name()), Some(mode));
            let j = serde_json::to_string(&mode).unwrap();
            assert_eq!(serde_json::from_str::<Interpolation>(&j).unwrap(), mode);
        }
        assert_eq!(Interpolation::from_name("CustomCurve"), Some(Interpolation::Custom));
        assert_eq!(Interpolation::from_name("bouncy"), None);
        // an emptied segment is written as before
        n.ease = Some(Box::default());
        n.tidy_ease();
        assert!(n.ease.is_none());
    }

    #[test]
    fn a_new_keyframe_inside_a_spring_segment_keeps_its_spring() {
        let mut p = segment(
            ParamValue::Float(0.0),
            ParamValue::Float(10.0),
            Interpolation::Spring,
            SegmentEase { spring: Some(Spring { damping_ratio: 2.0, ..Spring::DEFAULT }), ..Default::default() },
        );
        p.set_at(Tick(TICKS_PER_SECOND / 2), ParamValue::Float(4.0));
        assert_eq!(p.keyframes[1].interp, Interpolation::Spring);
        assert_eq!(p.keyframes[1].segment().spring.map(|s| s.damping_ratio), Some(2.0));
    }
}
