//! Beat effects: `sequence.beatPulse` keyframes a short pulse on clips at the music's beats.
//!
//! | param | meaning |
//! |---|---|
//! | `clips` | the clips (default: those overlapping `range`, else the selection) |
//! | `range` | `{"start": s, "end": s}` (sequence seconds): only beats inside it |
//! | `effect` | `zoomPulse` (Motion ▸ Scale), `flash` (Brightness & Contrast ▸ Brightness), `shake` (Motion ▸ Position), `glitch` (the Creator Glitch's Amount) |
//! | `strength` | 0 … 1 (default 0.5): +20 % scale, +70 brightness, 4 % of the frame, Glitch 100 at 1 |
//! | `everyNthBeat` | 1 … 64 (default 1): every beat, every other one… counted from the first beat |
//! | `decay` | seconds the pulse takes to settle (default 0.18; shortened when beats are closer) |
//! | `beats` / `bpm` / `music` | the beat grid, as `transitions.snapToBeats` takes it |
//!
//! Each pulse is three keyframes (Bézier): the resting value a moment (two frames at most, a
//! fifth of the gap) before the beat, the peak on the beat, the resting value `decay` later; a
//! shake swings once each way. Keyframes are in each clip's media time where the beat plays, so
//! a pulse stays on its beat through speed changes and ramps. Flash and glitch add their effect to
//! a clip without one (Glitch at a constant rhythm, its Amount 0 between beats). One undo step.

use serde_json::{Value, json};

use filmcraft_project::{EffectInstance, Interpolation, Keyframe, ParamValue, TrackItem, TrackKind};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, f64_p, has_seq, str_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.beatPulse";
/// Pulses per call, at most.
const MAX_PULSES: usize = 20_000;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Beat Pulse",
        menu: &[],
        shortcut: None,
        params: r#"{"clips":[id]?,"range":{"start":s,"end":s}?,"effect":"zoomPulse|flash|shake|glitch","strength":0..1=0.5,"everyNthBeat":1..64=1,"decay":s=0.18,"beats":[seconds]?,"bpm":20..400?,"music":clip id?}"#,
        enabled: has_seq,
        run: beat_pulse,
        journal: true,
    }]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pulse {
    Zoom,
    Flash,
    Shake,
    Glitch,
}

impl Pulse {
    fn from_id(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "zoompulse" | "zoom" => Pulse::Zoom,
            "flash" => Pulse::Flash,
            "shake" => Pulse::Shake,
            "glitch" => Pulse::Glitch,
            _ => return None,
        })
    }
    fn id(self) -> &'static str {
        match self {
            Pulse::Zoom => "zoomPulse",
            Pulse::Flash => "flash",
            Pulse::Shake => "shake",
            Pulse::Glitch => "glitch",
        }
    }
}

/// A keyframe with Bézier easing.
fn key(t: Tick, v: ParamValue) -> Keyframe {
    let mut k = Keyframe::new(t, v);
    k.interp = Interpolation::Bezier;
    k
}

/// The effect a pulse animates on `it` (added when missing): its index.
fn target(it: &mut TrackItem, kind: Pulse) -> Option<usize> {
    let id = match kind {
        Pulse::Zoom | Pulse::Shake => "motion",
        Pulse::Flash => "brightness_contrast",
        Pulse::Glitch => "cr_glitch",
    };
    if let Some(i) = it.effects.iter().position(|e| e.effect == id) {
        return Some(i);
    }
    let mut e: EffectInstance = filmcraft_project::find_effect(id)?.instance();
    match kind {
        Pulse::Flash => {
            e.params.get_mut("brightness")?.value = ParamValue::Float(0.0);
        }
        Pulse::Glitch => {
            e.params.get_mut("amount")?.value = ParamValue::Float(0.0);
            e.params.get_mut("rhythm")?.value = ParamValue::Choice(0);
            e.params.get_mut("rate")?.value = ParamValue::Float(24.0);
        }
        _ => {}
    }
    it.effects.push(e);
    Some(it.effects.len() - 1)
}

fn beat_pulse(s: &mut Session, p: &Value) -> Result<Value> {
    let kind = str_p(p, "effect").and_then(Pulse::from_id).ok_or_else(|| bad(CMD, "`effect` is zoomPulse, flash, shake or glitch"))?;
    let strength = match p.get("strength") {
        None | Some(Value::Null) => 0.5,
        Some(_) => {
            f64_p(p, "strength").filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0).ok_or_else(|| bad(CMD, "`strength` is more than 0 and at most 1"))?
        }
    };
    let every = match p.get("everyNthBeat") {
        None | Some(Value::Null) => 1,
        Some(v) => v.as_u64().filter(|n| (1..=64).contains(n)).ok_or_else(|| bad(CMD, "`everyNthBeat` is a whole number from 1 to 64"))? as usize,
    };
    let decay = match p.get("decay") {
        None | Some(Value::Null) => 0.18,
        Some(_) => f64_p(p, "decay").filter(|v| v.is_finite() && (0.02..=2.0).contains(v)).ok_or_else(|| bad(CMD, "`decay` is seconds from 0.02 to 2"))?,
    };
    let range = match p.get("range") {
        None | Some(Value::Null) => None,
        Some(r) => {
            let a = r.get("start").and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= 0.0);
            let b = r.get("end").and_then(Value::as_f64).filter(|v| v.is_finite());
            match (a, b) {
                (Some(a), Some(b)) if b > a => Some((Tick::from_seconds_f64(a), Tick::from_seconds_f64(b))),
                _ => return Err(bad(CMD, "`range` is {\"start\": seconds, \"end\": seconds} with end after start")),
            }
        }
    };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (w, h) = (f64::from(seq.settings.width), f64::from(seq.settings.height));
    let frame = seq.settings.frame_rate.sane().frame_duration();
    // the clips: named, else those in the range, else the selection (video clips, unlocked)
    let named = p.get("clips").is_some_and(|v| !v.is_null());
    let ids: Vec<filmcraft_project::ClipId> = if named || range.is_none() {
        crate::commands::clips_p(s, p)
    } else {
        let (a, b) = range.unwrap_or_default();
        seq.video_tracks.iter().filter(|t| t.enabled).flat_map(|t| t.items.iter()).filter(|i| i.start < b && i.end() > a).map(|i| i.id).collect()
    };
    let clips: Vec<filmcraft_project::ClipId> =
        ids.into_iter().filter(|c| seq.find_item(*c).and_then(|(t, _)| seq.track(t)).is_some_and(|t| t.kind == TrackKind::Video && !t.locked)).collect();
    if clips.is_empty() {
        return Err(bad(CMD, "no unlocked video clip given, selected or in the range"));
    }
    let (beats, _period, source) = crate::transition_tools::beat_grid(s, p, seq, CMD)?;
    let beats: Vec<Tick> =
        beats.into_iter().enumerate().filter(|(i, _)| i % every == 0).map(|(_, b)| b).filter(|b| range.is_none_or(|(a, e)| *b >= a && *b < e)).collect();
    if beats.is_empty() {
        return Err(bad(CMD, "no beat falls in the range"));
    }
    let decay_t = Tick::from_seconds_f64(decay);
    let report = s.edit_sequence(&format!("Beat Pulse ({})", kind.id()), |q, _, _| {
        let mut report = Vec::new();
        let mut total = 0usize;
        for c in &clips {
            let Some((_, it)) = q.find_item_mut(*c) else { continue };
            let inside: Vec<Tick> = beats.iter().copied().filter(|b| *b >= it.start && *b < it.end()).collect();
            if inside.is_empty() {
                report.push(json!({"clip": c.0, "pulses": 0}));
                continue;
            }
            total += inside.len();
            if total > MAX_PULSES {
                return Err(bad(CMD, format!("more than {MAX_PULSES} pulses: pulse fewer beats (`everyNthBeat`, `range`)")));
            }
            let item = it.clone();
            let Some(ei) = target(it, kind) else { return Err(bad(CMD, "the effect to pulse is missing")) };
            let Some(e) = it.effects.get_mut(ei) else { continue };
            e.enabled = true;
            let pid = match kind {
                Pulse::Zoom => "scale",
                Pulse::Shake => "position",
                Pulse::Flash => "brightness",
                Pulse::Glitch => "amount",
            };
            if !e.params.contains_key(pid)
                && let Some(d) = e.def().and_then(|d| d.param(pid))
            {
                e.params.insert(pid.to_string(), filmcraft_project::Param::new(d.default.clone()));
            }
            let Some(prm) = e.params.get_mut(pid) else { continue };
            // resting values, sampled before any pulse is written
            let media = |t: Tick| item.effect_time_at(t.clamp(item.start, (item.end() - Tick(1)).max(item.start)));
            let rest: Vec<ParamValue> = inside.iter().map(|b| prm.value_at(media(*b))).collect();
            let mut keys = Vec::new();
            for (i, b) in inside.iter().enumerate() {
                let two_frames = Tick(frame.0.saturating_mul(2));
                let prev_gap = i.checked_sub(1).and_then(|j| inside.get(j)).map_or(two_frames, |pb| *b - *pb);
                let next_gap = inside.get(i + 1).map_or(decay_t, |n| *n - *b);
                let attack = two_frames.min(prev_gap.mul_ratio(1, 5)).max(Tick(1));
                let settle = decay_t.min(next_gap.mul_ratio(4, 5)).max(frame);
                let base = rest.get(i).cloned().unwrap_or(ParamValue::Float(0.0));
                let peak = match (kind, &base) {
                    (Pulse::Zoom, ParamValue::Float(v)) => ParamValue::Float(v * (1.0 + 0.2 * strength)),
                    (Pulse::Flash, ParamValue::Float(v)) => ParamValue::Float((v + 70.0 * strength).min(100.0)),
                    (Pulse::Glitch, ParamValue::Float(_)) => ParamValue::Float(100.0 * strength),
                    (Pulse::Shake, ParamValue::Vec2(v)) => {
                        let c = filmcraft_geom::Vec2::new(if v.x.is_nan() { w / 2.0 } else { v.x }, if v.y.is_nan() { h / 2.0 } else { v.y });
                        // a hashed direction per beat: deterministic
                        let a = (b.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) as f64 / u64::MAX as f64 * std::f64::consts::TAU;
                        let m = 0.04 * w * strength;
                        keys.push(key(media(*b - attack), ParamValue::Vec2(c)));
                        keys.push(key(media(*b), ParamValue::Vec2(filmcraft_geom::Vec2::new(c.x + m * a.cos(), c.y + m * a.sin()))));
                        keys.push(key(
                            media(*b + settle.mul_ratio(1, 2)),
                            ParamValue::Vec2(filmcraft_geom::Vec2::new(c.x - 0.6 * m * a.cos(), c.y - 0.6 * m * a.sin())),
                        ));
                        keys.push(key(media(*b + settle), ParamValue::Vec2(c)));
                        continue;
                    }
                    _ => continue,
                };
                let rest_v = if kind == Pulse::Glitch { ParamValue::Float(0.0) } else { base };
                keys.push(key(media(*b - attack), rest_v.clone()));
                keys.push(key(media(*b), peak));
                keys.push(key(media(*b + settle), rest_v));
            }
            for k in keys {
                prm.put_keyframe(k.time, k.value.clone());
                if let Ok(i) = prm.keyframes.binary_search_by_key(&k.time, |x| x.time)
                    && let Some(x) = prm.keyframes.get_mut(i)
                {
                    x.interp = Interpolation::Bezier;
                }
            }
            report.push(json!({"clip": c.0, "pulses": inside.len(), "effect": e.effect, "param": pid, "keyframes": prm.keyframes.len()}));
        }
        Ok(report)
    })?;
    Ok(json!({"effect": kind.id(), "beats": beats.iter().map(|b| b.seconds()).collect::<Vec<_>>(), "source": source, "clips": report}))
}

#[cfg(test)]
#[path = "beat_pulse_tests.rs"]
mod tests;
