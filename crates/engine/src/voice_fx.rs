//! Voice isolation and the one-knob voice boost, on clips and on files.
//!
//! | command | does |
//! |---|---|
//! | `audio.isolateVoice` | put a Voice Isolation effect (intensity `amount`, 80 % by default) on audio clips (`clips`, default the selection; a video clip's linked audio), first in their chain, or set the one they have; one undo step |
//! | `audio.enhanceVoice` | the same with Enhance Voice (`amount`, 70 %), then match each clip's loudness, measured through its effects, to `lufs` (−16 LUFS; `"mix"`: the Dialogue target in Preferences ▸ Audio; `"none"`: leave the level) with the clip's gain; one undo step |
//! | `audio.enhanceFile` | process a media file's sound straight to a file: `path` → `out` (`.wav` 24-bit or `.m4a` AAC), `mode` `enhance` (default) or `isolate`, `amount`, `lufs` as above; reports loudness, noise floor and the 1–4 kHz share before and after |
//! | `audio.voiceModelStatus` | which voice model the effects run (DeepFilterNet 3, or the DSP fallback), whether it is downloaded, its size, source and licence |
//! | `audio.downloadVoiceModel` | download DeepFilterNet 3 (8 MB, SHA-256 checked; MIT / Apache-2.0) as a background job, after the user agreed; the effects use it from then on |
//!
//! The model is pure Rust (`filmcraft-voice`); hosts call [`install_loader`] at start-up so the
//! effects find it in `<data dir>/models`. Without it every command still works, on the DSP
//! fallback, and says so (`neural: false`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use filmcraft_audio_dsp::effects::voice::{VoiceModel, voice_model};
use filmcraft_audio_dsp::{AudioEffect, LoudnessMeter};
use filmcraft_project::{ClipId, EffectInstance, ParamValue, TrackItem, TrackKind};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, audio_effect_targets, bad, clips_p, has_seq, str_p};
use crate::{EngineError, Result, Session};

const ISOLATE: &str = "audio.isolateVoice";
const ENHANCE: &str = "audio.enhanceVoice";
const FILE: &str = "audio.enhanceFile";
const STATUS: &str = "audio.voiceModelStatus";
const DOWNLOAD: &str = "audio.downloadVoiceModel";
/// Loudness each enhanced clip is matched to by default (LUFS).
pub const DEFAULT_LUFS: f64 = -16.0;
/// Longest file `audio.enhanceFile` processes (seconds), and the most samples (all channels) it
/// holds: twenty minutes of stereo (the sound is processed in memory, ≈1 GB at most).
pub const MAX_FILE_SECONDS: f64 = 30.0 * 60.0;
const MAX_FILE_SAMPLES: usize = 2 * 48_000 * 60 * 20;
/// The rate files are processed at (the model's).
const FILE_RATE: u32 = 48_000;
const MAX_FILE_CHANNELS: usize = 8;

/// Make the audio effects load the voice model from `<data dir>/models` the first time one
/// needs it (once per process; later calls do nothing).
pub fn install_loader() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(dir) = crate::transcript::models_dir() {
            let installed = filmcraft_voice::catalog::installed(&dir);
            filmcraft_voice::install_loader(dir);
            // load it now, off the audio thread, so the first effect does not wait for it
            #[cfg(not(target_arch = "wasm32"))]
            if installed {
                let _ = std::thread::Builder::new().name("filmcraft-voice-load".into()).spawn(|| {
                    let _ = std::panic::catch_unwind(voice_model);
                });
            }
            #[cfg(target_arch = "wasm32")]
            let _ = installed;
        }
    });
}

/// The model the effects run now (`None`: the DSP fallback).
pub fn model() -> Option<Arc<dyn VoiceModel>> {
    install_loader();
    voice_model()
}

fn model_json() -> Value {
    let m = model();
    let (name, neural) = m.as_ref().map_or(("DSP noise gate (voice model not installed)".to_string(), false), |m| (m.name().to_string(), m.neural()));
    let dir = crate::transcript::models_dir();
    let installed = dir.as_deref().is_some_and(filmcraft_voice::catalog::installed);
    let mut v = json!({"model": name, "neural": neural, "installed": installed});
    if !neural {
        v["hint"] = json!(format!(
            "the voice model is not {}: run {DOWNLOAD} ({:.1} MB; {}) for the neural one",
            if installed { "loadable" } else { "downloaded" },
            filmcraft_voice::catalog::total_bytes() as f64 / 1e6,
            filmcraft_voice::catalog::LICENSE
        ));
    }
    v
}

fn amount_p(cmd: &str, p: &Value, default: f64) -> Result<f64> {
    match p.get("amount") {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_f64().filter(|a| a.is_finite() && (0.0..=100.0).contains(a)).ok_or_else(|| bad(cmd, "`amount` is a number from 0 to 100")),
    }
}

/// `lufs`: a number (−60…0), `"mix"` (the Dialogue target), `"none"` / `false` (no matching).
fn lufs_p(s: &Session, cmd: &str, p: &Value) -> Result<Option<f64>> {
    match p.get("lufs") {
        None | Some(Value::Null) => Ok(Some(DEFAULT_LUFS)),
        Some(Value::Bool(false)) => Ok(None),
        Some(Value::String(t)) if t.eq_ignore_ascii_case("none") || t.eq_ignore_ascii_case("off") => Ok(None),
        Some(Value::String(t)) if t.eq_ignore_ascii_case("mix") => Ok(Some(s.prefs.audio.loudness_target(filmcraft_project::AudioType::Dialogue))),
        Some(v) => v
            .as_f64()
            .filter(|l| l.is_finite() && (-60.0..=0.0).contains(l))
            .map(Some)
            .ok_or_else(|| bad(cmd, "`lufs` is a loudness from −60 to 0, \"mix\" or \"none\"")),
    }
}

/// The audio clips a command acts on.
fn targets(s: &Session, cmd: &str, p: &Value) -> Result<Vec<ClipId>> {
    let clips = audio_effect_targets(s, &clips_p(s, p));
    if clips.is_empty() {
        return Err(bad(cmd, "select audio clips (or video clips with linked audio)"));
    }
    Ok(clips)
}

/// Put effect `id` with `param = value` on `it`, before its other standard effects, or set the one
/// it has (non-essential); the value holds for the whole clip.
fn upsert(it: &mut TrackItem, id: &str, param: &str, value: f64) -> bool {
    let idx = match it.effects.iter().position(|e| e.effect == id && !e.essential) {
        Some(i) => i,
        None => {
            let Some(def) = filmcraft_project::find_effect(id) else { return false };
            let pos = it.effects.iter().position(|e| !e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
            it.effects.insert(pos, def.instance());
            pos
        }
    };
    let Some(e) = it.effects.get_mut(idx) else { return false };
    e.enabled = true;
    set_static(e, param, value);
    true
}

fn set_static(e: &mut EffectInstance, param: &str, value: f64) {
    if let Some(p) = e.param_mut(param) {
        p.keyframes.clear();
        p.value = ParamValue::Float(value);
    }
}

fn isolate_voice(s: &mut Session, p: &Value) -> Result<Value> {
    let amount = amount_p(ISOLATE, p, 80.0)?;
    let clips = targets(s, ISOLATE, p)?;
    let n = s.edit_sequence("Isolate Voice", |q, _, _| {
        let mut n = 0;
        for t in q.all_tracks_mut().filter(|t| t.kind == TrackKind::Audio) {
            for it in t.items.iter_mut().filter(|i| clips.contains(&i.id)) {
                n += usize::from(upsert(it, "voice_isolation", "intensity", amount));
            }
        }
        Ok(n)
    })?;
    let mut out = model_json();
    out["clips"] = json!(n);
    out["amount"] = json!(amount);
    Ok(out)
}

fn enhance_voice(s: &mut Session, p: &Value) -> Result<Value> {
    let amount = amount_p(ENHANCE, p, 70.0)?;
    let lufs = lufs_p(s, ENHANCE, p)?;
    let clips = targets(s, ENHANCE, p)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let sr = seq.settings.sample_rate.max(1);
    // measure each clip as it will sound (a copy with the effect), then write everything in one step
    let mut gains: Vec<(ClipId, f64)> = Vec::new();
    let mut report = Vec::new();
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    for it in seq.all_tracks().filter(|t| t.kind == TrackKind::Audio).flat_map(|t| t.items.iter()).filter(|i| clips.contains(&i.id)) {
        let mut copy = it.clone();
        upsert(&mut copy, "enhance_voice", "amount", amount);
        let Some(target) = lufs else {
            report.push(json!({"clip": it.id.0}));
            continue;
        };
        match crate::essential_sound::clip_loudness(&copy, sr, &provider) {
            Some(l) => {
                let delta = filmcraft_audio_dsp::normalize_gain_db(l, target);
                let gain = (it.gain_db + delta).clamp(-96.0, 96.0);
                gains.push((it.id, gain));
                report.push(json!({"clip": it.id.0, "measuredLufs": l, "targetLufs": target, "gainDb": gain}));
            }
            None => report.push(json!({"clip": it.id.0, "skipped": "silent"})),
        }
    }
    let n = s.edit_sequence("Enhance Voice", |q, _, _| {
        let mut n = 0;
        for t in q.all_tracks_mut().filter(|t| t.kind == TrackKind::Audio) {
            for it in t.items.iter_mut().filter(|i| clips.contains(&i.id)) {
                n += usize::from(upsert(it, "enhance_voice", "amount", amount));
                if let Some((_, g)) = gains.iter().find(|(c, _)| *c == it.id) {
                    it.gain_db = *g;
                }
            }
        }
        Ok(n)
    })?;
    let mut out = model_json();
    out["clips"] = json!(n);
    out["amount"] = json!(amount);
    out["loudness"] = Value::Array(report);
    Ok(out)
}

// ------------------------------------------------------------------------------------- files

/// Loudness, noise floor and speech-band share of planar audio at `sr`.
pub fn measure(ch: &[Vec<f32>], sr: u32) -> Value {
    let n = ch.iter().map(Vec::len).min().unwrap_or(0);
    let mut meter = LoudnessMeter::new(f64::from(sr), ch.len().clamp(1, 2));
    let refs: Vec<&[f32]> = ch.iter().take(2).map(|c| &c[..n]).collect();
    if !refs.is_empty() {
        meter.process(&refs);
    }
    let lufs = meter.integrated();
    let mono: Vec<f32> = (0..n).map(|i| ch.iter().map(|c| c[i]).sum::<f32>() / ch.len().max(1) as f32).collect();
    json!({
        "lufs": if lufs.is_finite() { json!((lufs * 10.0).round() / 10.0) } else { Value::Null },
        "truePeakDbtp": (meter.true_peak_dbtp() * 10.0).round() / 10.0,
        "noiseFloorDbfs": (noise_floor_db(&mono, sr) * 10.0).round() / 10.0,
        "speechBandShare": (band_share(&mono, sr, 1000.0, 4000.0) * 1000.0).round() / 1000.0,
    })
}

/// The 10th percentile of 50 ms RMS levels (dBFS): the level between words.
pub fn noise_floor_db(x: &[f32], sr: u32) -> f64 {
    let w = (sr as usize / 20).max(1);
    let mut lv: Vec<f64> =
        x.chunks(w).filter(|c| c.len() == w).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / w as f64 + 1e-20).log10()).collect();
    lv.sort_by(f64::total_cmp);
    lv.get(lv.len() / 10).copied().unwrap_or(-200.0)
}

/// Share of the energy between `lo` and `hi` Hz (Hann-windowed 4096-point frames, at most 2000
/// of them, spread over the sound).
pub fn band_share(x: &[f32], sr: u32, lo: f32, hi: f32) -> f64 {
    const N: usize = 4096;
    let fft = filmcraft_audio_dsp::fft::Fft::new(N);
    let frames = x.len() / N;
    let step = frames.div_ceil(2000).max(1);
    let (mut band, mut total) = (0.0f64, 0.0f64);
    let (mut re, mut im) = (vec![0.0f32; N], vec![0.0f32; N]);
    for f in (0..frames).step_by(step) {
        let Some(frame) = x.get(f * N..(f + 1) * N) else { break };
        for (i, (r, v)) in re.iter_mut().zip(frame).enumerate() {
            *r = v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos());
        }
        im.fill(0.0);
        fft.forward(&mut re, &mut im);
        for k in 1..N / 2 {
            let pw = f64::from(re[k] * re[k] + im[k] * im[k]);
            let hz = k as f32 * sr as f32 / N as f32;
            total += pw;
            if (lo..hi).contains(&hz) {
                band += pw;
            }
        }
    }
    if total > 0.0 { band / total } else { 0.0 }
}

#[cfg(not(target_arch = "wasm32"))]
fn open_media(path: &str) -> Result<filmcraft_media::SharedSource> {
    filmcraft_media::open_path(std::path::Path::new(path), &filmcraft_codecs::openers()).map_err(|e| bad(FILE, format!("{path}: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn open_media(_: &str) -> Result<filmcraft_media::SharedSource> {
    Err(bad(FILE, "files on disk are not processed in the browser; put the clip on the timeline and use audio.enhanceVoice"))
}

/// Process `path`'s sound into `out` (see the module docs).
fn enhance_file(s: &mut Session, p: &Value) -> Result<Value> {
    let path = str_p(p, "path").ok_or_else(|| bad(FILE, "need `path` (a media file with sound)"))?;
    let out = str_p(p, "out").ok_or_else(|| bad(FILE, "need `out` (a .wav or .m4a file to write)"))?;
    let ext = std::path::Path::new(out).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    if ext != "wav" && ext != "m4a" {
        return Err(bad(FILE, "`out` must end in .wav or .m4a"));
    }
    let isolate = match str_p(p, "mode").unwrap_or("enhance") {
        "enhance" => false,
        "isolate" => true,
        other => return Err(bad(FILE, format!("`mode` is enhance or isolate, not {other:?}"))),
    };
    let amount = amount_p(FILE, p, if isolate { 80.0 } else { 70.0 })?;
    let lufs = lufs_p(s, FILE, p)?;
    let bitrate = match p.get("bitrate") {
        None | Some(Value::Null) => 160,
        Some(v) => v.as_u64().filter(|b| (32..=512).contains(b)).ok_or_else(|| bad(FILE, "`bitrate` is 32–512 kbps"))? as u32,
    };
    let src = open_media(path)?;
    let info = src.info().clone();
    let a = info.audio().ok_or_else(|| bad(FILE, format!("{path} has no sound")))?;
    let secs = info.duration.0 as f64 / filmcraft_time::TICKS_PER_SECOND as f64;
    if !secs.is_finite() || secs <= 0.0 {
        return Err(bad(FILE, format!("{path}: no duration")));
    }
    if secs > MAX_FILE_SECONDS {
        return Err(bad(
            FILE,
            format!("{path} is {:.0} minutes long; files of at most {:.0} minutes are processed (split it)", secs / 60.0, MAX_FILE_SECONDS / 60.0),
        ));
    }
    let channels = (a.channels as usize).clamp(1, MAX_FILE_CHANNELS);
    let frames = (secs * f64::from(FILE_RATE)).ceil() as usize;
    if frames.saturating_mul(channels) > MAX_FILE_SAMPLES {
        return Err(bad(FILE, format!("{path}: {:.0} minutes of {channels} channels is more than this processes at once (split it)", secs / 60.0)));
    }
    // decode in 10 s pieces
    let mut input: Vec<Vec<f32>> = vec![Vec::with_capacity(frames); channels];
    let piece = FILE_RATE as usize * 10;
    let mut at = 0usize;
    while at < frames {
        let len = piece.min(frames - at);
        let buf = src.audio(at as i64, len, FILE_RATE).map_err(|e| bad(FILE, format!("{path}: {e}")))?;
        for (c, dst) in input.iter_mut().enumerate() {
            let chan = buf.channels.get(c).or_else(|| buf.channels.first());
            if let Some(v) = chan {
                dst.extend(v.iter().take(len).map(|x| if x.is_finite() { *x } else { 0.0 }));
            }
            dst.resize(at + len, 0.0);
        }
        at += len;
    }
    let before = measure(&input, FILE_RATE);
    let m = model();
    let (name, neural) = m.as_ref().map_or(("DSP noise gate (voice model not installed)".to_string(), false), |m| (m.name().to_string(), m.neural()));
    let mut fx: Box<dyn AudioEffect> = if isolate {
        let mut f = filmcraft_audio_dsp::effects::VoiceIsolation::with_model(FILE_RATE as f32, channels, m);
        f.set_param("intensity", amount as f32);
        Box::new(f)
    } else {
        let mut f = filmcraft_audio_dsp::effects::EnhanceVoice::with_model(FILE_RATE as f32, channels, m);
        f.set_param("amount", amount as f32);
        Box::new(f)
    };
    fx.reset();
    let t0 = web_time::Instant::now();
    let mut audio = input;
    filmcraft_audio_dsp::effects::voice::process_offline(fx.as_mut(), &mut audio);
    let proc_secs = t0.elapsed().as_secs_f64();
    let mut gain_db = 0.0;
    if let Some(target) = lufs {
        // the level, then a true-peak limiter at −1 dBFS for the peaks it pushes over, which takes
        // a little loudness back: a few rounds until it measures within 0.2 LU
        for _ in 0..4 {
            let mut meter = LoudnessMeter::new(f64::from(FILE_RATE), channels.min(2));
            let refs: Vec<&[f32]> = audio.iter().take(2).map(Vec::as_slice).collect();
            meter.process(&refs);
            let l = meter.integrated();
            if !l.is_finite() {
                break;
            }
            let delta = filmcraft_audio_dsp::normalize_gain_db(l, target).clamp(-40.0 - gain_db, 40.0 - gain_db);
            if delta.abs() < 0.2 {
                break;
            }
            gain_db += delta;
            let g = filmcraft_audio_dsp::db_to_gain(delta as f32);
            audio.iter_mut().for_each(|c| c.iter_mut().for_each(|v| *v *= g));
            let mut lim = filmcraft_audio_dsp::effects::Limiter::new(FILE_RATE as f32, channels);
            lim.set_param("ceiling", -1.0);
            filmcraft_audio_dsp::effects::voice::process_offline(&mut lim, &mut audio);
        }
    }
    let after = measure(&audio, FILE_RATE);
    let bytes = if ext == "wav" {
        let n = audio.iter().map(Vec::len).min().unwrap_or(0);
        let mut inter = Vec::with_capacity(n * channels);
        for i in 0..n {
            for c in &audio {
                inter.push(c[i]);
            }
        }
        let data = filmcraft_export::write_wav(&inter, channels as u16, FILE_RATE, filmcraft_export::SampleFormat::Int24).map_err(|e| bad(FILE, e))?;
        filmcraft_format::atomic_write(std::path::Path::new(out), &data).map_err(|e| bad(FILE, format!("{out}: {e}")))?;
        data.len() as u64
    } else {
        filmcraft_export::encode_m4a(out, &audio, FILE_RATE, bitrate).map_err(|e| bad(FILE, format!("{out}: {e}")))?
    };
    Ok(json!({
        "path": path,
        "out": out,
        "bytes": bytes,
        "mode": if isolate { "isolate" } else { "enhance" },
        "amount": amount,
        "model": name,
        "neural": neural,
        "seconds": (secs * 100.0).round() / 100.0,
        "realTimeFactor": (proc_secs / secs * 10_000.0).round() / 10_000.0,
        "gainDb": (gain_db * 10.0).round() / 10.0,
        "targetLufs": lufs,
        "before": before,
        "after": after,
    }))
}

// ------------------------------------------------------------------------------------- the model

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    let mut v = model_json();
    v["available"] = json!(cfg!(feature = "voice-model"));
    v["missingBytes"] = json!(dir.as_deref().map_or(filmcraft_voice::catalog::total_bytes(), filmcraft_voice::catalog::missing_bytes));
    v["package"] = json!(filmcraft_voice::catalog::PACKAGE_NAME);
    v["license"] = json!(filmcraft_voice::catalog::LICENSE);
    v["source"] = json!(filmcraft_voice::catalog::SOURCE);
    v["files"] = json!(filmcraft_voice::catalog::FILES);
    Ok(v)
}

/// Download the model in the background; the effects use it once it is in.
fn download(s: &mut Session, p: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(DOWNLOAD, "no data directory for the model"))?;
    if filmcraft_voice::catalog::installed(&dir) {
        let loaded = filmcraft_voice::install_now(&dir).map_err(|e| bad(DOWNLOAD, e.to_string()));
        return loaded.map(|()| json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "voice-model") {
        return Err(bad(DOWNLOAD, "model downloads are not available in this build (built without the `voice-model` feature)"));
    }
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_voice::catalog::PACKAGE_NAME));
    job.progress.total.store(filmcraft_voice::catalog::missing_bytes(&dir), Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-voice-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_package(&dir, &prog).and_then(|()| filmcraft_voice::install_now(&dir).map_err(|e| e.to_string()));
        crate::narration::finish(&prog, &res, r.map(|()| "Voice isolation model installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| j.progress.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone())
    {
        return Err(EngineError::Other(format!("{DOWNLOAD}: {e}")));
    }
    Ok(json!({"job": id, "installed": false}))
}

#[cfg(feature = "voice-model")]
fn download_package(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_voice::catalog::FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_voice::catalog::package_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "voice-model"))]
fn download_package(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("model downloads are not available in this build".into())
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: ISOLATE,
            label: "Isolate Voice",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"clip":id?,"amount":0..100=80}"#,
            enabled: has_seq,
            run: isolate_voice,
            journal: true,
        },
        CommandSpec {
            id: ENHANCE,
            label: "Enhance Voice",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"clip":id?,"amount":0..100=70,"lufs":-60..0|"mix"|"none"=-16}"#,
            enabled: has_seq,
            run: enhance_voice,
            journal: true,
        },
        CommandSpec {
            id: FILE,
            label: "Enhance Voice in a File",
            menu: &[],
            shortcut: None,
            params: r#"{"path":str,"out":"….wav|….m4a","mode":"enhance"|"isolate"="enhance","amount":0..100?,"lufs":-60..0|"mix"|"none"=-16,"bitrate":32..512=160}"#,
            enabled: always,
            run: enhance_file,
            journal: false,
        },
        CommandSpec {
            id: STATUS,
            label: "Voice Isolation Model Status",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: status,
            journal: false,
        },
        CommandSpec {
            id: DOWNLOAD,
            label: "Download Voice Isolation Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool?}"#,
            enabled: always,
            run: download,
            journal: false,
        },
    ]
}

#[cfg(test)]
#[path = "voice_fx_tests.rs"]
mod tests;
