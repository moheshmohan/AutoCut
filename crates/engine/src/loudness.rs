//! One-command loudness normalisation of a sequence's mix to a destination's target
//! (`audio.normalizeLoudness`): social video −14 LUFS, voice-forward (podcasts, explainers)
//! −16 LUFS, broadcast (EBU R128) −23 LUFS, or any integrated loudness, with a true-peak ceiling.
//!
//! The Mix fader sets the level and a Hard Limiter (true peak, look-ahead) after the fader holds
//! the ceiling, so playback, previews and every export hear the result; the mix is measured
//! through the mixer graph (ITU-R BS.1770-4, `filmcraft_audio_dsp::LoudnessMeter`), the fader is
//! moved by the difference, and the limited mix measured again (limiting lowers the loudness),
//! a bounded number of times. One undo step. The publish pre-flight's fixed copies are normalised
//! by the same routine ([`normalize_sequence`]).
//!
//! | command | does |
//! |---|---|
//! | `audio.normalizeLoudness {target?, truePeak?, sequence?}` | `target`: `social` (default), `voice`, `broadcast` or a number (LUFS, −70…0); `truePeak`: the ceiling in dBTP (−1 by default, −30…0) |

use serde_json::{Value, json};

use filmcraft_audio_dsp::LoudnessMeter;
use filmcraft_audio_dsp::loudness::LoudnessTarget;
use filmcraft_project::{ItemId, Project, find_effect};
use filmcraft_render::SourceProvider;

use crate::commands::{CommandSpec, bad, has_seq, u64_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "audio.normalizeLoudness";
/// Measuring passes at most (the first, then refinements through the limiter).
const PASSES: usize = 5;
/// Longest mix measured (seconds): four hours.
pub const MAX_SECONDS: f64 = 4.0 * 3600.0;
/// Within this of the target (LU) the fader stops moving.
const CLOSE_LU: f64 = 0.1;
/// The limiter sits this far under the ceiling (dB) …
const LIMITER_MARGIN_DB: f64 = 0.3;
/// … and goes lower at most this many times while the measured true peak is over it.
const CEILING_ROUNDS: usize = 3;

/// What a normalisation measured and did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalized {
    pub target_lufs: f64,
    pub ceiling_dbtp: f64,
    /// The mix before.
    pub measured_lufs: f64,
    pub measured_true_peak_dbtp: f64,
    /// The mix after.
    pub lufs: f64,
    pub true_peak_dbtp: f64,
    /// How far the Mix fader moved.
    pub gain_db: f64,
    pub master_volume_db: f64,
    /// Master insert slot of the limiter.
    pub limiter_slot: usize,
}

/// Integrated loudness and true peak of `seq`'s mix over its whole length.
fn measure(project: &Project, seq: ItemId, sources: &dyn SourceProvider, cancel: &dyn Fn() -> bool) -> std::result::Result<(f64, f64), String> {
    let q = project.sequence(seq).ok_or("no such sequence")?;
    let sr = q.settings.sample_rate.clamp(8_000, 384_000);
    let seconds = q.duration().seconds();
    if seconds > MAX_SECONDS {
        return Err(format!("the sequence is longer than {} hours: normalise it on export instead", MAX_SECONDS / 3600.0));
    }
    let total = q.duration().to_units_floor(i64::from(sr)).max(0);
    let layout = filmcraft_render::audio::master_layout(q);
    let mut meter = LoudnessMeter::new(f64::from(sr), layout.channels());
    // this pass's own stream: its stateful DSP (the limiter) starts fresh and continues only itself
    let stream = filmcraft_render::MixStream::new();
    let mut pos = 0i64;
    while pos < total {
        if cancel() {
            return Err("cancelled".into());
        }
        let n = (total - pos).min(i64::from(sr)) as usize;
        let b = filmcraft_render::audio::mix_sequence_layout(project, q, pos, n, sources, layout, filmcraft_audio_dsp::channels::Mixdown::FrontRear, &stream);
        let refs: Vec<&[f32]> = b.channels.iter().map(Vec::as_slice).collect();
        meter.process(&refs);
        pos += n as i64;
    }
    Ok((meter.integrated(), meter.true_peak_dbtp()))
}

/// The master limiter: an enabled post-fader Hard Limiter (added after the other inserts when
/// there is none) set to `ceiling`. Returns its slot.
fn set_limiter(project: &mut Project, seq: ItemId, ceiling: f64) -> std::result::Result<usize, String> {
    let q = project.sequence_mut(seq).ok_or("no such sequence")?;
    let found = q.master_effects.iter().take(filmcraft_project::mixer::MAX_INSERTS).position(|e| e.effect == "hard_limiter" && e.enabled && e.post_fader);
    let slot = match found {
        Some(i) => i,
        None => {
            if q.master_effects.len() >= filmcraft_project::mixer::MAX_INSERTS {
                return Err(format!("the Mix has no free insert for the limiter (at most {})", filmcraft_project::mixer::MAX_INSERTS));
            }
            let mut e = find_effect("hard_limiter").ok_or("the Hard Limiter effect is missing")?.instance();
            e.post_fader = true;
            q.master_effects.push(e);
            q.master_effects.len() - 1
        }
    };
    let e = q.master_effects.get_mut(slot).ok_or("internal: limiter slot")?;
    for (id, v) in [("max", ceiling), ("boost", 0.0)] {
        if let Some(p) = e.param_mut(id) {
            p.keyframes.clear();
            p.value = filmcraft_project::ParamValue::Float(v);
        }
    }
    Ok(slot)
}

/// Normalise `seq` of `project` to `target` LUFS with true peaks at most `ceiling` dBTP (see the
/// module docs). `project` changes only when it succeeds.
pub fn normalize_sequence(
    project: &mut Project,
    seq: ItemId,
    target: f64,
    ceiling: f64,
    sources: &dyn SourceProvider,
    cancel: &dyn Fn() -> bool,
) -> std::result::Result<Normalized, String> {
    if !(target.is_finite() && (-70.0..=0.0).contains(&target)) {
        return Err("the target must be between -70 and 0 LUFS".into());
    }
    if !(ceiling.is_finite() && (-30.0..=0.0).contains(&ceiling)) {
        return Err("the true-peak ceiling must be between -30 and 0 dBTP".into());
    }
    let (before, before_tp) = measure(project, seq, sources, cancel)?;
    if !before.is_finite() || before <= -70.0 {
        return Err("the mix is silent: nothing to normalize".into());
    }
    let mut p = project.clone();
    let start = p.sequence(seq).map(|q| q.master_volume_db).unwrap_or(0.0);
    // the limiter's own true-peak estimate runs a little under BS.1770's 4× oversampled reading:
    // it sits a margin below the ceiling, lowered again while the measured peak is still over
    let mut limit = (ceiling - LIMITER_MARGIN_DB).max(-30.0);
    let mut slot = set_limiter(&mut p, seq, limit)?;
    let mut now = before;
    let mut tp = before_tp;
    let mut fader = start;
    for _round in 0..CEILING_ROUNDS {
        let mut best: Option<(f64, f64)> = None; // (|error|, fader)
        let mut last: Option<f64> = None;
        for pass in 0..PASSES {
            let error = target - now;
            let q = p.sequence_mut(seq).ok_or("no such sequence")?;
            // the first move is the whole difference; refinements are bounded (an impossible
            // target must not drive the limiter without end)
            let step = if pass == 0 { error.clamp(-60.0, 60.0) } else { error.clamp(-6.0, 6.0) };
            q.master_volume_db = (q.master_volume_db + step).clamp(-96.0, 48.0);
            let f = q.master_volume_db;
            let (lufs, _) = measure(&p, seq, sources, cancel)?;
            if !lufs.is_finite() {
                break;
            }
            now = lufs;
            if best.is_none_or(|b| (target - lufs).abs() < b.0) {
                best = Some(((target - lufs).abs(), f));
            }
            if (target - lufs).abs() <= CLOSE_LU || last.is_some_and(|l: f64| (lufs - l).abs() < 0.02) {
                break;
            }
            last = Some(lufs);
        }
        fader = best.ok_or("the mix could not be measured")?.1;
        if let Some(q) = p.sequence_mut(seq) {
            q.master_volume_db = fader;
        }
        let (lufs, peak) = measure(&p, seq, sources, cancel)?;
        now = lufs;
        tp = peak;
        if !tp.is_finite() || tp <= ceiling || limit <= -30.0 {
            break;
        }
        limit = (limit - (tp - ceiling) - 0.1).max(-30.0);
        slot = set_limiter(&mut p, seq, limit)?;
        let (lufs, _) = measure(&p, seq, sources, cancel)?;
        now = lufs;
    }
    *project = p;
    Ok(Normalized {
        target_lufs: target,
        ceiling_dbtp: ceiling,
        measured_lufs: before,
        measured_true_peak_dbtp: before_tp,
        lufs: now,
        true_peak_dbtp: tp,
        gain_db: fader - start,
        master_volume_db: fader,
        limiter_slot: slot,
    })
}

/// `target`: a destination name or a number of LUFS.
fn target_p(p: &Value) -> Result<LoudnessTarget> {
    match p.get("target") {
        None | Some(Value::Null) => Ok(LoudnessTarget::Social),
        Some(Value::String(name)) => {
            LoudnessTarget::parse(name).ok_or_else(|| bad(CMD, format!("`target` {name:?}: social, voice, broadcast or a number of LUFS")))
        }
        Some(Value::Number(n)) => match n.as_f64() {
            Some(v) if v.is_finite() && (-70.0..=0.0).contains(&v) => Ok(LoudnessTarget::Lufs(v)),
            _ => Err(bad(CMD, "`target`: a loudness between -70 and 0 LUFS")),
        },
        Some(other) => Err(bad(CMD, format!("`target`: social, voice, broadcast or a number of LUFS, not {other}"))),
    }
}

fn normalize(s: &mut Session, p: &Value) -> Result<Value> {
    let target = target_p(p)?;
    let ceiling = match p.get("truePeak").filter(|v| !v.is_null()) {
        None => target.true_peak_dbtp(),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && (-30.0..=0.0).contains(&x) => x,
            _ => return Err(bad(CMD, "`truePeak`: a ceiling between -30 and 0 dBTP")),
        },
    };
    let seq = match u64_p(p, "sequence") {
        Some(id) => {
            let id = ItemId(id);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
    let lufs = target.lufs();
    let r = s.edit("Normalize Loudness", |pr, _| normalize_sequence(pr, seq, lufs, ceiling, &provider, &|| false).map_err(EngineError::Other))?;
    Ok(json!({
        "sequence": seq.0,
        "target": target.name(),
        "targetLufs": r.target_lufs,
        "truePeakCeilingDbtp": r.ceiling_dbtp,
        "measuredLufs": r.measured_lufs,
        "measuredTruePeakDbtp": r.measured_true_peak_dbtp,
        "loudnessLufs": r.lufs,
        "truePeakDbtp": r.true_peak_dbtp,
        "gainDb": r.gain_db,
        "masterVolumeDb": r.master_volume_db,
        "limiterSlot": r.limiter_slot,
    }))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Normalize Loudness",
        menu: &[],
        shortcut: None,
        params: r#"{"target":"social|voice|broadcast"|lufs=social,"truePeak":dBTP=-1,"sequence":id?}"#,
        enabled: has_seq,
        run: normalize,
        journal: true,
    }]
}
