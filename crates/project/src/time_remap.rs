//! Time Remapping: a clip whose `time_remap` Speed is keyframed plays its media at that speed,
//! frame by frame, and the speed-ramp presets (`clip.speedRamp`) that write such curves.
//!
//! **The curve.** Speed keyframes are keyed like every other effect parameter, in media ticks at
//! the clip's average speed: timeline offset `rel` from the clip's start reads the curve at
//! `source_in + rel × speed` (so Effect Controls draws them where they play). The media shown at
//! `rel` is the integral of the curve from the clip's start, scaled so the whole clip plays
//! exactly its media span (`source_in` … `source_out`, `duration × speed`): editing a speed
//! keyframe re-times the clip inside its span instead of running past the media. Speeds are
//! clamped to 0 … 10 000 % (0 freezes; never backwards), so the mapping is monotonic. A curve
//! that integrates to nothing (all zero), a reversed clip or a clip of no length plays at its
//! constant speed as before.
//!
//! **Cost.** The integral is a table of [`RemapTable::cells`] trapezoids (240 a second, 64 …
//! 8192 a clip), built once per curve and cached (a few hundred curves at most), so a frame's
//! media time is a table lookup and a linear interpolation.
//!
//! **Presets** ([`RAMP_PRESETS`]): Montage, Hero, Bullet, Jump Cut, Flash In, Flash Out, Smooth In
//! and Smooth Out, each a speed shape over the clip's new length (multiples of the clip's
//! current speed), eased with Bézier keyframes (held for Jump Cut). `intensity` scales each
//! speed's distance from 1× in log space (`s^intensity`: 0 = no ramp, 1 = the preset, 2 =
//! exaggerated). [`ramp_keyframes`] writes a preset onto a clip's span; the clip's new length is
//! its media span over the curve's mean speed ([`ramp_mean`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::keyframe::{Interpolation, Keyframe, Param, ParamValue};
use crate::{EffectInstance, TrackItem};

/// Highest speed a curve may ask for, percent.
pub const MAX_SPEED_PCT: f64 = 10_000.0;
/// Trapezoids a second of clip.
const CELLS_PER_SECOND: f64 = 240.0;
const MIN_CELLS: usize = 64;
const MAX_CELLS: usize = 8192;
/// Cached tables (cleared when full).
const CACHE_LIMIT: usize = 512;

/// The integral of a clip's speed curve: the fraction of its media span shown at each timeline
/// offset.
#[derive(Debug, Clone, PartialEq)]
pub struct RemapTable {
    /// Normalised running integral at each cell edge (0 … 1, `cells + 1` values).
    cum: Vec<f64>,
    /// Cell width, timeline ticks.
    cell: f64,
    /// Clip duration, timeline ticks.
    duration: f64,
}

impl RemapTable {
    /// Integrate `curve` (percent) over a clip of `duration` at average `speed` from `source_in`.
    pub fn build(curve: &Param, source_in: Tick, speed: f64, duration: Tick) -> Option<RemapTable> {
        let dur = duration.0 as f64;
        let speed = speed.abs();
        if !(dur > 0.0 && dur.is_finite() && speed.is_finite() && speed > 0.0) {
            return None;
        }
        let secs = dur / TICKS_PER_SECOND as f64;
        let cells = ((secs * CELLS_PER_SECOND).ceil().max(0.0) as usize).clamp(MIN_CELLS, MAX_CELLS);
        let cell = dur / cells as f64;
        let s = |tau: f64| speed_factor(curve, source_in, speed, tau);
        let mut cum = Vec::with_capacity(cells + 1);
        cum.push(0.0);
        let (mut acc, mut prev) = (0.0f64, s(0.0));
        for k in 1..=cells {
            let cur = s(k as f64 * cell);
            acc += (prev + cur) * 0.5 * cell;
            cum.push(acc);
            prev = cur;
        }
        if !(acc > 0.0 && acc.is_finite()) {
            return None;
        }
        for v in &mut cum {
            *v /= acc;
        }
        Some(RemapTable { cum, cell, duration: dur })
    }

    /// Trapezoids in the table.
    pub fn cells(&self) -> usize {
        self.cum.len().saturating_sub(1)
    }

    /// The fraction (0 … 1) of the media span shown `rel` timeline ticks into the clip.
    pub fn fraction(&self, rel: f64) -> f64 {
        let r = if rel.is_finite() { rel.clamp(0.0, self.duration) } else { 0.0 };
        let x = r / self.cell;
        let k = (x.floor().max(0.0) as usize).min(self.cells().saturating_sub(1));
        let f = (x - k as f64).clamp(0.0, 1.0);
        let a = self.cum.get(k).copied().unwrap_or(0.0);
        let b = self.cum.get(k + 1).copied().unwrap_or(a);
        a + (b - a) * f
    }
}

/// The curve's speed factor (1 = 100 %) `tau` timeline ticks into the clip.
fn speed_factor(curve: &Param, source_in: Tick, speed: f64, tau: f64) -> f64 {
    let key = source_in.0 as f64 + tau * speed;
    let key = if key.is_finite() { Tick(key.clamp(i64::MIN as f64, i64::MAX as f64) as i64) } else { source_in };
    let v = curve.scalar_at(key);
    if v.is_finite() { v.clamp(0.0, MAX_SPEED_PCT) / 100.0 } else { 0.0 }
}

/// The keyframed Speed of an item's enabled Time Remapping (two keyframes or more).
pub fn curve_of(item: &TrackItem) -> Option<&Param> {
    let e = item.effects.iter().find(|e| e.effect == "time_remap" && e.enabled)?;
    let p = e.params.get("speed")?;
    (p.keyframes.len() >= 2).then_some(p)
}

fn cache() -> &'static Mutex<HashMap<u64, Arc<RemapTable>>> {
    static C: OnceLock<Mutex<HashMap<u64, Arc<RemapTable>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key_of(curve: &Param, source_in: Tick, speed: f64, duration: Tick) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |v: u64| h = (h ^ v).wrapping_mul(0x0100_0000_01b3);
    eat(source_in.0 as u64);
    eat(duration.0 as u64);
    eat(speed.to_bits());
    for k in &curve.keyframes {
        eat(k.time.0 as u64);
        eat(k.value.as_f64().unwrap_or(f64::NAN).to_bits());
        eat(k.interp as u64);
        eat(k.in_influence.to_bits());
        eat(k.out_influence.to_bits());
    }
    h
}

/// The remap table of `item` (None: it plays at its constant speed).
pub fn table_for(item: &TrackItem) -> Option<Arc<RemapTable>> {
    if item.reverse {
        return None;
    }
    let curve = curve_of(item)?;
    let key = key_of(curve, item.source_in, item.speed, item.duration);
    if let Some(t) = cache().lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
        return Some(t.clone());
    }
    let t = Arc::new(RemapTable::build(curve, item.source_in, item.speed, item.duration)?);
    let mut c = cache().lock().unwrap_or_else(PoisonError::into_inner);
    if c.len() >= CACHE_LIMIT {
        c.clear();
    }
    c.insert(key, t.clone());
    Some(t)
}

/// Media offset (ticks after `source_in`, fractional) `rel` timeline ticks into a remapped clip.
pub fn media_offset(item: &TrackItem, rel: f64) -> Option<f64> {
    let t = table_for(item)?;
    let span = item.duration.0 as f64 * item.speed.abs();
    span.is_finite().then(|| span * t.fraction(rel))
}

// ---------------------------------------------------------------------------------------------
// Speed-ramp presets

/// A speed-ramp preset: speed multiples at fractions of the clip's (new) length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RampPreset {
    pub id: &'static str,
    pub name: &'static str,
    /// (fraction of the clip 0 … 1, speed multiple of the clip's speed), in order.
    pub points: &'static [(f64, f64)],
    /// Hold each speed to the next point (Jump Cut) instead of easing (Bézier).
    pub hold: bool,
}

/// The presets, in menu order. Ids are stored nowhere (the curve is), so they may grow.
pub const RAMP_PRESETS: &[RampPreset] = &[
    RampPreset { id: "montage", name: "Montage", points: &[(0.0, 1.0), (0.22, 3.0), (0.45, 0.35), (0.7, 3.0), (1.0, 1.0)], hold: false },
    RampPreset { id: "hero", name: "Hero", points: &[(0.0, 2.5), (0.3, 2.5), (0.5, 0.2), (0.7, 2.5), (1.0, 2.5)], hold: false },
    RampPreset { id: "bullet", name: "Bullet", points: &[(0.0, 1.5), (0.25, 1.5), (0.4, 0.12), (0.8, 0.12), (1.0, 3.0)], hold: false },
    RampPreset { id: "jumpCut", name: "Jump Cut", points: &[(0.0, 1.0), (0.3, 4.0), (0.45, 1.0), (0.75, 4.0), (0.88, 1.0), (1.0, 1.0)], hold: true },
    RampPreset { id: "flashIn", name: "Flash In", points: &[(0.0, 5.0), (0.35, 1.0), (1.0, 1.0)], hold: false },
    RampPreset { id: "flashOut", name: "Flash Out", points: &[(0.0, 1.0), (0.65, 1.0), (1.0, 5.0)], hold: false },
    RampPreset { id: "smoothIn", name: "Smooth In", points: &[(0.0, 0.25), (0.6, 1.0), (1.0, 1.0)], hold: false },
    RampPreset { id: "smoothOut", name: "Smooth Out", points: &[(0.0, 1.0), (0.4, 1.0), (1.0, 0.25)], hold: false },
];

/// The preset with id or name `name` (case-insensitive).
pub fn ramp_preset(name: &str) -> Option<&'static RampPreset> {
    let n = name.trim();
    RAMP_PRESETS.iter().find(|p| p.id.eq_ignore_ascii_case(n) || p.name.eq_ignore_ascii_case(n))
}

/// Highest intensity.
pub const MAX_INTENSITY: f64 = 2.0;

/// The Speed keyframes (percent) of `preset` at `intensity` over a clip whose media span starts at
/// `source_in` and lasts `span` media ticks, at `base_pct` (the clip's speed before the ramp).
pub fn ramp_keyframes(preset: &RampPreset, intensity: f64, source_in: Tick, span: Tick, base_pct: f64) -> Vec<Keyframe> {
    let k = if intensity.is_finite() { intensity.clamp(0.0, MAX_INTENSITY) } else { 1.0 };
    let base = if base_pct.is_finite() && base_pct > 0.0 { base_pct } else { 100.0 };
    let mut out: Vec<Keyframe> = Vec::with_capacity(preset.points.len());
    for &(u, s) in preset.points {
        let t = source_in + Tick((span.0 as f64 * u.clamp(0.0, 1.0)).round() as i64);
        let v = (base * s.max(1e-3).powf(k)).clamp(0.0, MAX_SPEED_PCT);
        // (points that land on one tick, in a very short clip: the later one wins)
        if out.last().is_some_and(|l| l.time >= t) {
            out.pop();
        }
        let mut kf = Keyframe::new(t, ParamValue::Float(v));
        kf.interp = if preset.hold { Interpolation::Hold } else { Interpolation::Bezier };
        out.push(kf);
    }
    out
}

/// Mean speed factor of a curve (multiples of `base_pct`) over a clip's span: the clip's length
/// divides by it. Integrated the way playback does ([`RemapTable`]).
pub fn ramp_mean(keys: &[Keyframe], source_in: Tick, span: Tick, base_pct: f64) -> Option<f64> {
    if keys.len() < 2 || span.0 <= 0 {
        return None;
    }
    let curve = Param { value: ParamValue::Float(base_pct), keyframes: keys.to_vec() };
    // a unit-speed clip as long as the span: offsets are media ticks
    let n = 4096usize;
    let (mut acc, mut prev) = (0.0f64, speed_factor(&curve, source_in, 1.0, 0.0));
    let cell = span.0 as f64 / n as f64;
    for i in 1..=n {
        let cur = speed_factor(&curve, source_in, 1.0, i as f64 * cell);
        acc += (prev + cur) * 0.5;
        prev = cur;
    }
    let mean = acc / n as f64 / (base_pct / 100.0).max(1e-9);
    (mean.is_finite() && mean > 0.0).then_some(mean)
}

/// A Time Remapping effect instance whose Speed is `keys` (the static value is the clip's speed).
pub fn remap_effect(keys: Vec<Keyframe>, base_pct: f64) -> Option<EffectInstance> {
    let mut e = crate::find_effect("time_remap")?.instance();
    let p = e.params.get_mut("speed")?;
    p.value = ParamValue::Float(base_pct);
    p.keyframes = keys;
    Some(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClipId, ItemId};

    fn clip(dur_s: f64, speed: f64) -> TrackItem {
        TrackItem {
            id: ClipId(1),
            item: ItemId(1),
            name: "c".into(),
            label: crate::Label::Iris,
            start: Tick::from_seconds_f64(2.0),
            duration: Tick::from_seconds_f64(dur_s),
            source_in: Tick::from_seconds_f64(1.0),
            speed,
            reverse: false,
            enabled: true,
            link: None,
            group: None,
            effects: crate::effect::intrinsic_video(),
            markers: vec![],
            gain_db: 0.0,
            frame_hold: None,
            scale_to_frame: false,
            essential: None,
            multicam: None,
            time_interpolation: Default::default(),
            maintain_pitch: false,
            hold_filters: false,
            field_options: None,
            source_channels: Vec::new(),
            audio_stream: 0,
            graphic: None,
        }
    }

    fn ramped(preset: &str, intensity: f64, media_s: f64) -> TrackItem {
        let p = ramp_preset(preset).unwrap();
        let span = Tick::from_seconds_f64(media_s);
        let src_in = Tick::from_seconds_f64(1.0);
        let keys = ramp_keyframes(p, intensity, src_in, span, 100.0);
        let mean = ramp_mean(&keys, src_in, span, 100.0).unwrap();
        let mut it = clip(media_s / mean, mean);
        // the clip plays exactly its media span
        it.duration = Tick((span.0 as f64 / mean).round() as i64);
        it.speed = span.0 as f64 / it.duration.0 as f64;
        if let Some(e) = it.effects.iter_mut().find(|e| e.effect == "time_remap") {
            *e = remap_effect(keys, 100.0).unwrap();
        }
        it
    }

    #[test]
    fn every_preset_integrates_to_its_media_span_and_is_monotonic() {
        for p in RAMP_PRESETS {
            for intensity in [0.0, 0.5, 1.0, 2.0] {
                let media = 3.0;
                let it = ramped(p.id, intensity, media);
                // the output length is the span over the mean speed
                let keys = ramp_keyframes(p, intensity, it.source_in, it.source_out() - it.source_in, 100.0);
                let mean = ramp_mean(&keys, it.source_in, it.source_out() - it.source_in, 100.0).unwrap();
                assert!((it.duration.seconds() - media / mean).abs() < 1e-6, "{}: {} vs {}", p.id, it.duration.seconds(), media / mean);
                // first and last frame: the span's ends
                assert_eq!(it.source_time_at(it.start), it.source_in, "{}", p.id);
                let end = it.source_time_at(it.end());
                assert!((end - it.source_out()).0.abs() <= 1, "{} {intensity}: ends at {:?}, out {:?}", p.id, end, it.source_out());
                // monotonic, frame by frame and between frames
                let mut last = Tick(i64::MIN);
                let n = 2000;
                for i in 0..=n {
                    let t = it.start + Tick(it.duration.0 * i / n);
                    let m = it.source_time_at(t);
                    assert!(m >= last, "{} {intensity}: not monotonic at {i}", p.id);
                    last = m;
                }
                if intensity == 0.0 {
                    // no ramp: the constant-speed mapping
                    let mid = it.start + Tick(it.duration.0 / 2);
                    let lin = it.source_in + Tick((it.duration.0 as f64 / 2.0 * it.speed).round() as i64);
                    assert!((it.source_time_at(mid) - lin).0.abs() < TICKS_PER_SECOND / 1000, "{}", p.id);
                }
            }
        }
    }

    #[test]
    fn the_curve_sets_the_local_speed() {
        // Hero: 2.5× at the start, 0.2× in the middle
        let it = ramped("hero", 1.0, 4.0);
        let rate = |a: f64, b: f64| {
            let ta = it.start + Tick::from_seconds_f64(it.duration.seconds() * a);
            let tb = it.start + Tick::from_seconds_f64(it.duration.seconds() * b);
            (it.source_time_at(tb) - it.source_time_at(ta)).seconds() / (tb - ta).seconds()
        };
        let fast = rate(0.05, 0.15);
        let slow = rate(0.48, 0.52);
        assert!((fast - 2.5).abs() < 0.05, "fast {fast}");
        assert!(slow < 0.3, "slow {slow}");
    }

    #[test]
    fn hostile_curves_never_panic_and_fall_back() {
        let mut it = clip(2.0, 1.0);
        let bad = [f64::NAN, f64::INFINITY, -500.0, 0.0, 1e300];
        for v in bad {
            let keys = vec![Keyframe::new(it.source_in, ParamValue::Float(v)), Keyframe::new(it.source_in + Tick::from_seconds_f64(1.0), ParamValue::Float(v))];
            if let Some(e) = it.effects.iter_mut().find(|e| e.effect == "time_remap") {
                *e = remap_effect(keys, 100.0).unwrap();
            }
            for f in [0.0, 0.3, 1.0] {
                let m = it.source_time_at(it.start + Tick::from_seconds_f64(2.0 * f));
                assert!(m >= it.source_in && m <= it.source_out(), "{v}: {m:?}");
            }
        }
        // no length, zero or NaN speed: constant mapping, no panic
        for (d, s) in [(0.0, 1.0), (1.0, 0.0), (1.0, f64::NAN)] {
            let mut c = it.clone();
            c.duration = Tick::from_seconds_f64(d);
            c.speed = s;
            let _ = c.source_time_at(c.start + Tick(1000));
        }
        assert!(RemapTable::build(&Param::new(ParamValue::Float(100.0)), Tick::ZERO, 1.0, Tick(-5)).is_none());
        assert!(ramp_preset("Jump Cut").is_some() && ramp_preset("JUMPCUT").is_some() && ramp_preset("nope").is_none());
        assert!(ramp_keyframes(&RAMP_PRESETS[0], f64::NAN, Tick::ZERO, Tick(3), f64::NAN).iter().all(|k| k.value.as_f64().is_some_and(f64::is_finite)));
    }
}
