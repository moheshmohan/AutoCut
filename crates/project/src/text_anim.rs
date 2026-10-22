//! Text animators: a text layer's letters, words or lines animating in, out and in a loop, each
//! unit a little after the one before it (the stagger).
//!
//! A [`TextAnimation`] lives on the layer ([`crate::LayerExtra::anim`], so it is saved with the
//! project and inside `.fcgt` graphics templates) and has up to three [`Animator`]s:
//!
//! - **in** ([`AnimKind::IN`]): fade, typewriter, pop (an overshoot), slide from the left, right,
//!   top or bottom, blur in, scale up, drop (falls and bounces) and glitch in. Unit `i` starts
//!   `i × stagger` seconds after the clip starts and takes `duration` seconds.
//! - **out**: the same kinds, played backwards (the mirror of the in animation): the units leave
//!   in the same order, the last one finishing as the clip ends.
//! - **loop** ([`AnimKind::LOOP`]): wave, bounce, shake, pulse and flicker, with a period of
//!   `duration` seconds and a phase lag of `stagger` seconds between units.
//!
//! Each animator has an `amount` (percent of the kind's natural strength: 100 is the designed
//! look), a stagger, an easing (an index into [`crate::gltransition::EASING_OPTS`], Penner's
//! curves and a custom Bézier, evaluated with [`crate::easing::ease_overshoot`] so a pop overshoots
//! and settles) and a duration. [`unit_state`] evaluates unit `i` of `n` at clip time `u` of a
//! clip `d` seconds long into a [`UnitState`] (an offset and a blur in ems of the font size, a
//! scale and a rotation about the unit's centre, an opacity); the renderer applies it to the
//! unit's glyphs (`filmcraft_render::graphic_clip`). Everything is plain `f64` math on hostile
//! values: non-finite or out-of-range numbers are made safe by [`Animator::sanitized`].

use serde::{Deserialize, Serialize};

use crate::easing::{Bezier, ease_overshoot};

/// What an animator does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimKind {
    /// Opacity from 0.
    #[default]
    Fade,
    /// Each unit appears at once, in turn.
    Typewriter,
    /// Grows from nothing past its size and settles back (an overshoot).
    Pop,
    /// Slides in from the left (out: to the left).
    SlideLeft,
    /// From the right.
    SlideRight,
    /// From above.
    SlideDown,
    /// From below.
    SlideUp,
    /// Sharpens from a blur while fading in.
    BlurIn,
    /// Grows from half its size while fading in.
    ScaleUp,
    /// Falls from above and bounces.
    Drop,
    /// Jitters and flickers into place.
    Glitch,
    /// Units rise and fall in a travelling wave.
    Wave,
    /// Units hop.
    Bounce,
    /// Units jitter.
    Shake,
    /// Units swell and shrink.
    Pulse,
    /// Units flicker like a failing sign.
    Flicker,
}

impl AnimKind {
    /// The kinds for the in and out animators.
    pub const IN: [AnimKind; 11] = [
        AnimKind::Fade,
        AnimKind::Typewriter,
        AnimKind::Pop,
        AnimKind::SlideLeft,
        AnimKind::SlideRight,
        AnimKind::SlideDown,
        AnimKind::SlideUp,
        AnimKind::BlurIn,
        AnimKind::ScaleUp,
        AnimKind::Drop,
        AnimKind::Glitch,
    ];
    /// The kinds for the loop animator.
    pub const LOOP: [AnimKind; 5] = [AnimKind::Wave, AnimKind::Bounce, AnimKind::Shake, AnimKind::Pulse, AnimKind::Flicker];

    pub fn is_loop(self) -> bool {
        Self::LOOP.contains(&self)
    }
    /// The name parameters take (`fade`, `slideLeft`, …).
    pub fn name(self) -> &'static str {
        match self {
            AnimKind::Fade => "fade",
            AnimKind::Typewriter => "typewriter",
            AnimKind::Pop => "pop",
            AnimKind::SlideLeft => "slideLeft",
            AnimKind::SlideRight => "slideRight",
            AnimKind::SlideDown => "slideDown",
            AnimKind::SlideUp => "slideUp",
            AnimKind::BlurIn => "blurIn",
            AnimKind::ScaleUp => "scaleUp",
            AnimKind::Drop => "drop",
            AnimKind::Glitch => "glitch",
            AnimKind::Wave => "wave",
            AnimKind::Bounce => "bounce",
            AnimKind::Shake => "shake",
            AnimKind::Pulse => "pulse",
            AnimKind::Flicker => "flicker",
        }
    }
    /// The label shown in the interface.
    pub fn label(self) -> &'static str {
        match self {
            AnimKind::Fade => "Fade",
            AnimKind::Typewriter => "Typewriter",
            AnimKind::Pop => "Pop",
            AnimKind::SlideLeft => "Slide from Left",
            AnimKind::SlideRight => "Slide from Right",
            AnimKind::SlideDown => "Slide from Top",
            AnimKind::SlideUp => "Slide from Bottom",
            AnimKind::BlurIn => "Blur",
            AnimKind::ScaleUp => "Scale Up",
            AnimKind::Drop => "Drop",
            AnimKind::Glitch => "Glitch",
            AnimKind::Wave => "Wave",
            AnimKind::Bounce => "Bounce",
            AnimKind::Shake => "Shake",
            AnimKind::Pulse => "Pulse",
            AnimKind::Flicker => "Flicker",
        }
    }
    /// A kind by name or label; case, spaces, `-` and `_` do not matter (`slide-left`, `Slide
    /// from Left`, `blur in`, `glitch in`).
    pub fn from_name(s: &str) -> Option<AnimKind> {
        let k: String = s.chars().filter(|c| !matches!(c, ' ' | '-' | '_')).flat_map(char::to_lowercase).collect();
        let k = k.strip_suffix("in").filter(|r| matches!(*r, "glitch" | "fade" | "pop" | "drop")).unwrap_or(&k);
        let all = Self::IN.iter().chain(Self::LOOP.iter());
        all.copied().find(|a| {
            let n = a.name().to_ascii_lowercase();
            let l: String = a.label().chars().filter(|c| *c != ' ').flat_map(char::to_lowercase).collect();
            n == k || l == k
        })
    }
}

/// What one stagger step animates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimUnit {
    /// Each letter (a shaped cluster: a Malayalam conjunct with its vowel sign is one letter).
    #[default]
    Letter,
    Word,
    Line,
}

impl AnimUnit {
    pub const ALL: [AnimUnit; 3] = [AnimUnit::Letter, AnimUnit::Word, AnimUnit::Line];
    pub fn name(self) -> &'static str {
        match self {
            AnimUnit::Letter => "letter",
            AnimUnit::Word => "word",
            AnimUnit::Line => "line",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            AnimUnit::Letter => "By Letter",
            AnimUnit::Word => "By Word",
            AnimUnit::Line => "By Line",
        }
    }
    pub fn from_name(s: &str) -> Option<AnimUnit> {
        let k = s.trim().to_ascii_lowercase();
        let k = k.trim_start_matches("by ").trim_end_matches('s');
        Self::ALL.into_iter().find(|u| u.name() == k || ((k == "character" || k == "char") && *u == AnimUnit::Letter))
    }
}

/// One animator (see the module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Animator {
    pub kind: AnimKind,
    pub unit: AnimUnit,
    /// Strength, percent of the kind's natural amount.
    pub amount: f64,
    /// Seconds between one unit and the next.
    pub stagger: f64,
    /// Easing curve: an index into [`crate::gltransition::EASING_OPTS`].
    pub easing: u32,
    /// The Custom Bézier's control points `[x1, y1, x2, y2]` (easing 31).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bezier: Option<[f64; 4]>,
    /// Seconds one unit takes (a loop's period).
    pub duration: f64,
}

/// Easing indices of [`crate::gltransition::EASING_OPTS`] the presets use.
pub mod curve {
    pub const LINEAR: u32 = 0;
    pub const SINE_IN_OUT: u32 = 3;
    pub const CUBIC_OUT: u32 = 8;
    pub const QUART_OUT: u32 = 11;
    pub const EXPO_OUT: u32 = 17;
    pub const BACK_OUT: u32 = 23;
    pub const BOUNCE_OUT: u32 = 29;
}

impl Default for Animator {
    fn default() -> Self {
        Animator::preset(AnimKind::Fade)
    }
}

/// Longest stagger and duration, seconds.
pub const MAX_SECONDS: f64 = 60.0;
/// Largest amount, percent.
pub const MAX_AMOUNT: f64 = 1000.0;
/// Shortest loop period, seconds.
pub const MIN_PERIOD: f64 = 0.05;

impl Animator {
    /// The designed look of a kind: its unit, stagger, easing and duration.
    pub fn preset(kind: AnimKind) -> Animator {
        use curve::*;
        let (unit, stagger, easing, duration) = match kind {
            AnimKind::Fade => (AnimUnit::Letter, 0.03, CUBIC_OUT, 0.5),
            AnimKind::Typewriter => (AnimUnit::Letter, 0.06, LINEAR, 0.0),
            AnimKind::Pop => (AnimUnit::Letter, 0.04, BACK_OUT, 0.45),
            AnimKind::SlideLeft | AnimKind::SlideRight | AnimKind::SlideDown | AnimKind::SlideUp => (AnimUnit::Word, 0.08, QUART_OUT, 0.6),
            AnimKind::BlurIn => (AnimUnit::Letter, 0.03, CUBIC_OUT, 0.6),
            AnimKind::ScaleUp => (AnimUnit::Word, 0.08, EXPO_OUT, 0.5),
            AnimKind::Drop => (AnimUnit::Letter, 0.05, BOUNCE_OUT, 0.7),
            AnimKind::Glitch => (AnimUnit::Letter, 0.02, LINEAR, 0.5),
            AnimKind::Wave => (AnimUnit::Letter, 0.08, SINE_IN_OUT, 1.2),
            AnimKind::Bounce => (AnimUnit::Letter, 0.1, SINE_IN_OUT, 0.7),
            AnimKind::Shake => (AnimUnit::Letter, 0.0, LINEAR, 0.3),
            AnimKind::Pulse => (AnimUnit::Word, 0.15, SINE_IN_OUT, 1.0),
            AnimKind::Flicker => (AnimUnit::Letter, 0.0, LINEAR, 1.0),
        };
        Animator { kind, unit, amount: 100.0, stagger, easing, bezier: None, duration }
    }

    /// The animator with hostile numbers made safe: amount 0..1000 %, stagger and duration
    /// 0..60 s (a loop's period at least 0.05 s), a known easing; non-finite values take the
    /// kind's preset's.
    pub fn sanitized(&self) -> Animator {
        let p = Animator::preset(self.kind);
        let num = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.clamp(lo, hi) } else { or };
        let min_d = if self.kind.is_loop() { MIN_PERIOD } else { 0.0 };
        Animator {
            kind: self.kind,
            unit: self.unit,
            amount: num(self.amount, 0.0, MAX_AMOUNT, p.amount),
            stagger: num(self.stagger, 0.0, MAX_SECONDS, p.stagger),
            easing: if self.easing < crate::easing::COUNT { self.easing } else { p.easing },
            bezier: self.bezier,
            duration: num(self.duration, min_d, MAX_SECONDS, p.duration.max(min_d)),
        }
    }

    fn eased(&self, t: f64) -> f64 {
        let b = self.bezier.map_or(Bezier::EASE, |[x1, y1, x2, y2]| Bezier { x1, y1, x2, y2 });
        ease_overshoot(self.easing, b, t)
    }

    /// Progress (0..1) of unit `i` whose animation starts at `start` seconds, at time `u`.
    fn progress(&self, start: f64, u: f64) -> f64 {
        if self.duration <= 1e-9 {
            return if u >= start { 1.0 } else { 0.0 };
        }
        ((u - start) / self.duration).clamp(0.0, 1.0)
    }

    /// Seconds from the first unit's start to the last unit's end, for `n` units.
    pub fn span(&self, n: usize) -> f64 {
        self.stagger * n.saturating_sub(1) as f64 + self.duration
    }
}

/// A text layer's animators (see the module docs).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextAnimation {
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    pub enter: Option<Animator>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<Animator>,
    #[serde(rename = "loop", skip_serializing_if = "Option::is_none")]
    pub repeat: Option<Animator>,
}

impl TextAnimation {
    pub fn is_empty(&self) -> bool {
        self.enter.is_none() && self.out.is_none() && self.repeat.is_none()
    }
    /// The animators with hostile numbers made safe (an in or out animator of a loop kind, or a
    /// loop animator of an in kind, is dropped).
    pub fn sanitized(&self) -> TextAnimation {
        TextAnimation {
            enter: self.enter.as_ref().filter(|a| !a.kind.is_loop()).map(Animator::sanitized),
            out: self.out.as_ref().filter(|a| !a.kind.is_loop()).map(Animator::sanitized),
            repeat: self.repeat.as_ref().filter(|a| a.kind.is_loop()).map(Animator::sanitized),
        }
    }
    /// The unit the animators work on (the in animator's, else the out's, else the loop's).
    pub fn unit(&self) -> AnimUnit {
        self.enter.as_ref().or(self.out.as_ref()).or(self.repeat.as_ref()).map_or(AnimUnit::Letter, |a| a.unit)
    }
    /// Whether every unit of `n` is at rest at clip time `u` (seconds) of a clip `d` seconds long
    /// (`f64::INFINITY` when unknown): no loop, the in animation finished and the out animation
    /// not started. Rendering at rest is the same as rendering without animation.
    pub fn at_rest(&self, n: usize, u: f64, d: f64) -> bool {
        let a = self.sanitized();
        if a.repeat.is_some() {
            return false;
        }
        let entered = a.enter.as_ref().is_none_or(|e| u >= e.span(n));
        let not_leaving = a.out.as_ref().is_none_or(|o| !d.is_finite() || u <= d - o.span(n));
        entered && not_leaving
    }
}

/// How one unit is drawn at one moment: offsets and blur in ems of the unit's font size, a scale
/// and rotation (degrees, clockwise) about the unit's centre, and an opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitState {
    pub dx: f64,
    pub dy: f64,
    pub scale: f64,
    pub rotation: f64,
    pub opacity: f64,
    pub blur: f64,
}

impl UnitState {
    pub const REST: UnitState = UnitState { dx: 0.0, dy: 0.0, scale: 1.0, rotation: 0.0, opacity: 1.0, blur: 0.0 };

    pub fn is_rest(&self) -> bool {
        *self == UnitState::REST
    }
    fn then(self, o: UnitState) -> UnitState {
        UnitState {
            dx: self.dx + o.dx,
            dy: self.dy + o.dy,
            scale: self.scale * o.scale,
            rotation: self.rotation + o.rotation,
            opacity: self.opacity * o.opacity,
            blur: self.blur + o.blur,
        }
    }
    /// Every number finite: scale 0..20, opacity 0..1, offsets ±1000 em, blur 0..10 em.
    fn safe(self) -> UnitState {
        let f = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.clamp(lo, hi) } else { or };
        UnitState {
            dx: f(self.dx, -1000.0, 1000.0, 0.0),
            dy: f(self.dy, -1000.0, 1000.0, 0.0),
            scale: f(self.scale, 0.0, 20.0, 1.0),
            rotation: f(self.rotation, -3600.0, 3600.0, 0.0),
            opacity: f(self.opacity, 0.0, 1.0, 1.0),
            blur: f(self.blur, 0.0, 10.0, 0.0),
        }
    }
}

/// A repeatable pseudo-random number in 0..1 for unit `i` at step `k` (glitch, shake, flicker).
pub fn noise(i: usize, k: i64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Distance a slide travels, ems (at amount 100).
pub const SLIDE_EM: f64 = 1.2;
/// Height a drop falls from, ems.
pub const DROP_EM: f64 = 1.6;

/// The state of an in-kind unit `k` away from rest (1 = where it starts, 0 = at rest; an eased
/// value, so it can go below 0 for an overshoot). `i` and `u` seed the glitch.
fn away(kind: AnimKind, k: f64, amount: f64, i: usize, u: f64) -> UnitState {
    let a = amount / 100.0;
    let shown = (1.0 - k).clamp(0.0, 1.0);
    let mut s = UnitState::REST;
    match kind {
        AnimKind::Fade => s.opacity = 1.0 - k.clamp(0.0, 1.0) * a.min(1.0),
        AnimKind::Typewriter => s.opacity = if k >= 0.5 { 0.0 } else { 1.0 },
        AnimKind::Pop => {
            s.scale = (1.0 - k * a).max(0.0);
            s.opacity = (shown * 3.0).min(1.0);
        }
        AnimKind::SlideLeft => {
            s.dx = -k * SLIDE_EM * a;
            s.opacity = shown;
        }
        AnimKind::SlideRight => {
            s.dx = k * SLIDE_EM * a;
            s.opacity = shown;
        }
        AnimKind::SlideDown => {
            s.dy = -k * SLIDE_EM * a;
            s.opacity = shown;
        }
        AnimKind::SlideUp => {
            s.dy = k * SLIDE_EM * a;
            s.opacity = shown;
        }
        AnimKind::BlurIn => {
            s.blur = k.max(0.0) * 0.25 * a;
            s.opacity = shown;
        }
        AnimKind::ScaleUp => {
            s.scale = (1.0 - k * 0.5 * a).max(0.0);
            s.opacity = shown;
        }
        AnimKind::Drop => {
            s.dy = -k * DROP_EM * a;
            s.opacity = (shown * 4.0).min(1.0);
        }
        AnimKind::Glitch if k > 1e-6 => {
            let step = (u * 24.0).floor() as i64;
            s.dx = (noise(i, step) - 0.5) * 0.5 * a * k;
            s.dy = (noise(i.wrapping_add(7919), step) - 0.5) * 0.15 * a * k;
            // more and more often shown as it settles
            s.opacity = if noise(i.wrapping_add(104_729), step) < k.clamp(0.0, 1.0) * 0.8 { 0.0 } else { 1.0 };
        }
        _ => {}
    }
    s
}

/// A loop kind's state of unit `i` at time `u`.
fn looping(a: &Animator, i: usize, u: f64) -> UnitState {
    let amt = a.amount / 100.0;
    let period = a.duration.max(MIN_PERIOD);
    let phase = (u - i as f64 * a.stagger) / period;
    let wave = a.eased(phase.rem_euclid(1.0));
    let angle = std::f64::consts::TAU * wave;
    let mut s = UnitState::REST;
    match a.kind {
        AnimKind::Wave => s.dy = -0.15 * amt * angle.sin(),
        AnimKind::Bounce => s.dy = -0.25 * amt * (std::f64::consts::PI * wave).sin().abs(),
        AnimKind::Shake => {
            let step = (phase * 4.0).floor() as i64;
            s.dx = (noise(i, step) - 0.5) * 0.08 * amt;
            s.dy = (noise(i.wrapping_add(31), step) - 0.5) * 0.08 * amt;
            s.rotation = (noise(i.wrapping_add(97), step) - 0.5) * 6.0 * amt;
        }
        AnimKind::Pulse => s.scale = 1.0 + 0.1 * amt * (0.5 - 0.5 * angle.cos()),
        AnimKind::Flicker => {
            let step = (phase * 12.0).floor() as i64;
            if noise(i, step) < 0.15 * amt.min(4.0) {
                s.opacity = 0.2;
                s.dx = (noise(i.wrapping_add(13), step) - 0.5) * 0.1 * amt;
            }
        }
        _ => {}
    }
    s
}

/// The state of unit `i` of `n` at clip time `u` (seconds) of a clip `d` seconds long
/// (`f64::INFINITY` when the clip's length is unknown: no out animation).
pub fn unit_state(anim: &TextAnimation, i: usize, n: usize, u: f64, d: f64) -> UnitState {
    let a = anim.sanitized();
    let n = n.max(1);
    let i = i.min(n - 1);
    let u = if u.is_finite() { u } else { 0.0 };
    let mut s = UnitState::REST;
    if let Some(e) = &a.enter {
        let p = e.progress(i as f64 * e.stagger, u);
        let k = if e.kind == AnimKind::Typewriter { if p > 0.0 || u >= i as f64 * e.stagger { 0.0 } else { 1.0 } } else { 1.0 - e.eased(p) };
        s = s.then(away(e.kind, k, e.amount, i, u));
    }
    if let Some(o) = &a.out
        && d.is_finite()
    {
        let start = d - o.span(n) + i as f64 * o.stagger;
        let p = o.progress(start, u);
        // the mirror of the in animation: played backwards
        let k = if o.kind == AnimKind::Typewriter { if u >= start + o.duration.max(1e-9) { 1.0 } else { 0.0 } } else { 1.0 - o.eased(1.0 - p) };
        s = s.then(away(o.kind, k, o.amount, i, u));
    }
    if let Some(l) = &a.repeat {
        s = s.then(looping(l, i, u));
    }
    s.safe()
}

/// Word index of every byte of `text` (whitespace bytes take the word before them; leading
/// whitespace the first word), and the number of words (at least 1).
pub fn word_units(text: &str) -> (Vec<usize>, usize) {
    let mut out = Vec::with_capacity(text.len());
    let mut word = 0usize;
    let mut seen = false;
    let mut in_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            in_space = true;
        } else {
            if in_space && seen {
                word += 1;
            }
            in_space = false;
            seen = true;
        }
        out.extend(std::iter::repeat_n(word, c.len_utf8()));
    }
    (out, word + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn stagger_and_easing_at_sample_times() {
        // fade in by letter, 0.1 s apart, 0.5 s each, linear
        let fade = Animator { kind: AnimKind::Fade, unit: AnimUnit::Letter, amount: 100.0, stagger: 0.1, easing: curve::LINEAR, bezier: None, duration: 0.5 };
        let anim = TextAnimation { enter: Some(fade.clone()), ..Default::default() };
        let op = |i: usize, u: f64| unit_state(&anim, i, 5, u, f64::INFINITY).opacity;
        assert!(close(op(0, 0.0), 0.0));
        assert!(close(op(0, 0.25), 0.5));
        assert!(close(op(0, 0.5), 1.0));
        // letter 3 starts 0.3 s later
        assert!(close(op(3, 0.3), 0.0));
        assert!(close(op(3, 0.55), 0.5));
        assert!(close(op(3, 0.8), 1.0));
        // cubic out is ahead of linear in the middle: 1 - (1 - 0.5)^3 = 0.875
        let cubic = TextAnimation { enter: Some(Animator { easing: curve::CUBIC_OUT, ..fade.clone() }), ..Default::default() };
        assert!(close(unit_state(&cubic, 0, 5, 0.25, f64::INFINITY).opacity, 0.875));
        // at rest after the last letter (0.4 + 0.5 s)
        assert!(!anim.at_rest(5, 0.85, 10.0));
        assert!(anim.at_rest(5, 0.9, 10.0));
        assert!(unit_state(&anim, 4, 5, 0.9, 10.0).is_rest());
    }

    #[test]
    fn pop_overshoots_and_settles() {
        let anim = TextAnimation { enter: Some(Animator::preset(AnimKind::Pop)), ..Default::default() };
        let d = Animator::preset(AnimKind::Pop).duration;
        let scales: Vec<f64> = (0..=40).map(|k| unit_state(&anim, 0, 1, d * k as f64 / 40.0, f64::INFINITY).scale).collect();
        assert!(close(scales[0], 0.0), "starts from nothing");
        let peak = scales.iter().copied().fold(0.0, f64::max);
        assert!(peak > 1.05 && peak < 1.2, "overshoots: {peak}");
        assert!(close(*scales.last().unwrap(), 1.0), "settles");
    }

    #[test]
    fn out_mirrors_in_and_ends_with_the_clip() {
        let slide =
            Animator { kind: AnimKind::SlideLeft, unit: AnimUnit::Word, amount: 100.0, stagger: 0.2, easing: curve::CUBIC_OUT, bezier: None, duration: 0.4 };
        let anim = TextAnimation { enter: Some(slide.clone()), out: Some(slide.clone()), repeat: None };
        let d = 5.0;
        // the in animation at time t of unit i equals the out animation mirrored in time
        for (i, t) in [(0usize, 0.1), (1, 0.35), (2, 0.5)] {
            let a = unit_state(&TextAnimation { enter: Some(slide.clone()), ..Default::default() }, i, 3, t, d);
            let span = slide.span(3);
            let start_in = i as f64 * 0.2;
            let start_out = d - span + i as f64 * 0.2;
            let mirrored = start_out + slide.duration - (t - start_in);
            let b = unit_state(&TextAnimation { out: Some(slide.clone()), ..Default::default() }, i, 3, mirrored, d);
            assert!(close(a.dx, b.dx) && close(a.opacity, b.opacity), "unit {i}: {a:?} vs {b:?}");
        }
        // gone at the end, at rest in the middle
        let end = unit_state(&anim, 2, 3, d, d);
        assert!(close(end.opacity, 0.0) && close(end.dx, -SLIDE_EM), "{end:?}");
        assert!(unit_state(&anim, 1, 3, 2.5, d).is_rest());
        assert!(anim.at_rest(3, 2.5, d) && !anim.at_rest(3, d - 0.1, d));
    }

    #[test]
    fn typewriter_shows_one_unit_after_another() {
        let anim = TextAnimation { enter: Some(Animator::preset(AnimKind::Typewriter)), ..Default::default() };
        let shown = |u: f64| (0..10).filter(|&i| unit_state(&anim, i, 10, u, f64::INFINITY).opacity > 0.5).count();
        assert_eq!(shown(0.0), 1);
        assert_eq!(shown(0.13), 3);
        assert_eq!(shown(1.0), 10);
    }

    #[test]
    fn loops_move_and_repeat() {
        let wave = Animator::preset(AnimKind::Wave);
        let anim = TextAnimation { repeat: Some(wave.clone()), ..Default::default() };
        let dy = |i: usize, u: f64| unit_state(&anim, i, 6, u, 10.0).dy;
        // periodic
        assert!(close(dy(0, 0.3), dy(0, 0.3 + wave.duration)));
        // the phase travels: unit 1 is where unit 0 was one stagger earlier
        assert!(close(dy(1, 0.5 + wave.stagger), dy(0, 0.5)));
        assert!((0..20).any(|k| dy(0, k as f64 * 0.05).abs() > 0.05), "it moves");
        assert!(!anim.at_rest(6, 3.0, 10.0));
        let pulse = TextAnimation { repeat: Some(Animator::preset(AnimKind::Pulse)), ..Default::default() };
        let s: Vec<f64> = (0..20).map(|k| unit_state(&pulse, 0, 1, k as f64 * 0.05, 10.0).scale).collect();
        assert!(s.iter().all(|v| (1.0..=1.1 + 1e-9).contains(v)) && s.iter().any(|v| *v > 1.05), "{s:?}");
        for kind in AnimKind::LOOP {
            let a = TextAnimation { repeat: Some(Animator::preset(kind)), ..Default::default() };
            for k in 0..50 {
                let st = unit_state(&a, k % 4, 4, k as f64 * 0.037, 10.0);
                assert!(st.opacity > 0.0 && st.scale > 0.5, "{kind:?}: {st:?}");
            }
        }
    }

    #[test]
    fn hostile_numbers_are_made_safe() {
        let wild = Animator {
            kind: AnimKind::Pop,
            unit: AnimUnit::Word,
            amount: f64::NAN,
            stagger: -5.0,
            easing: 999,
            bezier: Some([f64::NAN, 1e300, -1e300, f64::INFINITY]),
            duration: f64::INFINITY,
        };
        let s = wild.sanitized();
        assert!(s.amount == 100.0 && s.stagger == 0.0 && s.easing == curve::BACK_OUT && s.duration.is_finite());
        let anim = TextAnimation {
            enter: Some(wild.clone()),
            out: Some(wild.clone()),
            repeat: Some(Animator { kind: AnimKind::Shake, duration: 0.0, amount: 1e9, ..wild }),
        };
        for u in [f64::NAN, -1.0, 0.0, 0.3, 1e300, f64::INFINITY] {
            for d in [f64::NAN, 0.0, 1.0, f64::INFINITY, -5.0] {
                for (i, n) in [(0, 0), (5, 2), (usize::MAX, usize::MAX)] {
                    let st = unit_state(&anim, i, n, u, d);
                    for v in [st.dx, st.dy, st.scale, st.rotation, st.opacity, st.blur] {
                        assert!(v.is_finite(), "{u} {d} {i} {n}: {st:?}");
                    }
                }
            }
        }
        // an in animator of a loop kind is dropped
        let bad = TextAnimation { enter: Some(Animator::preset(AnimKind::Wave)), ..Default::default() };
        assert!(bad.sanitized().is_empty());
    }

    #[test]
    fn names_round_trip_and_serde() {
        for k in AnimKind::IN.iter().chain(AnimKind::LOOP.iter()) {
            assert_eq!(AnimKind::from_name(k.name()), Some(*k));
            assert_eq!(AnimKind::from_name(k.label()), Some(*k), "{}", k.label());
        }
        assert_eq!(AnimKind::from_name("glitch-in"), Some(AnimKind::Glitch));
        assert_eq!(AnimKind::from_name("slide_left"), Some(AnimKind::SlideLeft));
        assert_eq!(AnimKind::from_name("nope"), None);
        assert_eq!(AnimUnit::from_name("words"), Some(AnimUnit::Word));
        assert_eq!(AnimUnit::from_name("By Line"), Some(AnimUnit::Line));
        let a = TextAnimation { enter: Some(Animator::preset(AnimKind::Pop)), out: None, repeat: Some(Animator::preset(AnimKind::Wave)) };
        let j = serde_json::to_value(&a).unwrap();
        assert_eq!(j["in"]["kind"], "pop");
        assert_eq!(j["loop"]["kind"], "wave");
        assert!(j.get("out").is_none());
        let back: TextAnimation = serde_json::from_value(j).unwrap();
        assert_eq!(back, a);
        // partial objects fill in defaults
        let p: TextAnimation = serde_json::from_str(r#"{"in":{"kind":"drop"}}"#).unwrap();
        assert_eq!(p.enter.unwrap().kind, AnimKind::Drop);
    }

    #[test]
    fn words_of_text() {
        let (w, n) = word_units("  ab  cd e");
        assert_eq!(n, 3);
        assert_eq!(w, vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 2]);
        let (m, n) = word_units("മലയാളം വാക്ക്");
        assert_eq!(n, 2);
        assert_eq!(m.len(), "മലയാളം വാക്ക്".len());
        assert_eq!(*m.last().unwrap(), 1);
        assert_eq!(word_units(""), (vec![], 1));
    }
}
