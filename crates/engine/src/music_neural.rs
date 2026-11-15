//! Neural music: text to music with **ACE-Step 1.5** (ACE Studio and StepFun, MIT;
//! `filmcraft-acestep`), an optional download next to the procedural music of `audio.generate`.
//!
//! | command | does |
//! |---|---|
//! | `music.generateNeural` | generate `seconds` (5–240) of instrumental music from a description (`prompt`: genre, mood, instruments, tempo words), optionally with `bpm`, `key`, `timeSignature`; write `Neural Music <n>.wav` (48 kHz stereo float, peak −1 dBFS), import it and place it at `atTime`/`atSeconds` (default: the playhead) on `track` or the first free audio track (else a new one) — one undo step. A background job (`jobs.list`; `wait: true` blocks until placed). The model must be downloaded first |
//! | `music.neuralModels` | the neural music models: installed, size, missing bytes, licence, source |
//! | `music.downloadNeuralModel` | download a model as a background job after the user agreed (`consent: true`), every file SHA-256 checked |
//!
//! **Reproducible.** The same description, settings and `seed` give the same music on a device;
//! without a seed each new clip takes the next one. The model, seed and settings are kept in the
//! item's metadata (Description, Log Note).
//!
//! **When it lands.** The job writes nothing to the project: its WAV is imported and placed when
//! the session next looks at its jobs (each UI frame through [`Session::poll_persistence`],
//! `jobs.list`, or any `music.*` command), so a busy track is checked at that moment.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_acestep::{Request, catalog, request};
use filmcraft_project::{ClipId, ItemId, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, time_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "music.generateNeural";
/// Base name of the files and items.
const BASE: &str = "Neural Music";
/// Peak of the finished clip, dBFS (the official pipeline's normalisation).
#[cfg(feature = "neural-music")]
const PEAK_DB: f32 = -1.0;

type Outcome = Arc<Mutex<Option<std::result::Result<Vec<u8>, String>>>>;

/// A generation whose WAV is imported and placed when its job finishes.
pub struct PendingMusic {
    job: u64,
    out: Outcome,
    path: String,
    place: bool,
    at: Tick,
    track: Option<TrackId>,
    req: Request,
    model: &'static str,
}

/// Generations in flight, and what finished ones made (job → item, clip, track), reported in
/// `jobs.list` as `music`.
#[derive(Default)]
pub struct NeuralMusic {
    pending: Vec<PendingMusic>,
    outcomes: std::collections::HashMap<u64, Value>,
}

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            CMD,
            "Generate Neural Music",
            r#"{"prompt":str,"seconds":5..240,"seed":u64?,"model":"acestep-v15-turbo"?,"bpm":30..300?,"key":"A minor|F#|Eb major…"?,"timeSignature":2|3|4|6?,"shift":1|2|3=3,"atTime":ticks?|"atSeconds":f64?,"track":"A2"|id?,"place":bool=true,"dir":str?,"wait":bool?}"#,
            generate,
            true,
        ),
        spec("music.neuralModels", "Neural Music Models", "{}", models, false),
        spec(
            "music.downloadNeuralModel",
            "Download Neural Music Model",
            r#"{"model":"acestep-v15-turbo"?,"consent":bool,"wait":bool?}"#,
            download_model,
            false,
        ),
    ]
}

/// Whether this build runs the networks (feature `neural-music`).
pub fn available() -> bool {
    cfg!(feature = "neural-music")
}

/// The models as JSON: id, name, description, installed, size, missing bytes, licence, source.
pub fn models_json() -> Value {
    let dir = crate::transcript::models_dir();
    json!(
        catalog::MODELS
            .iter()
            .map(|m| json!({
                "id": m.id,
                "name": m.name,
                "description": m.description,
                "installed": dir.as_deref().is_some_and(|d| catalog::installed(d, m)),
                "size": catalog::size(m),
                "missingBytes": dir.as_deref().map_or(catalog::size(m), |d| catalog::missing_bytes(d, m)),
                "license": m.license,
                "source": m.source,
                "steps": m.steps,
                "available": available(),
            }))
            .collect::<Vec<_>>()
    )
}

fn models(s: &mut Session, _p: &Value) -> Result<Value> {
    poll(s);
    Ok(
        json!({"models": models_json(), "default": catalog::DEFAULT_MODEL, "available": available(), "maxSeconds": request::MAX_SECONDS, "minSeconds": request::MIN_SECONDS}),
    )
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn model_p(cmd: &str, p: &Value) -> Result<&'static catalog::Package> {
    match p.get("model") {
        None | Some(Value::Null) => catalog::model(catalog::DEFAULT_MODEL).ok_or_else(|| bad(cmd, "no default model")),
        Some(Value::String(m)) => {
            catalog::model(m).ok_or_else(|| bad(cmd, format!("unknown model `{m}` ({})", catalog::MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", "))))
        }
        Some(v) => Err(bad(cmd, format!("`model` must be a model id, got {v}"))),
    }
}

/// An optional whole number in `range`, refused when present but anything else.
fn whole(p: &Value, key: &str, range: std::ops::RangeInclusive<u64>, what: &str) -> Result<Option<u64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let n = v.as_u64().or_else(|| v.as_f64().filter(|f| f.is_finite() && f.fract() == 0.0 && *f >= 0.0 && *f <= u64::MAX as f64).map(|f| f as u64));
            n.filter(|n| range.contains(n)).map(Some).ok_or_else(|| bad(CMD, format!("`{key}` must be {what}")))
        }
    }
}

/// The request `p` asks for, checked (nothing is generated or written for a bad one).
fn request_p(s: &Session, p: &Value) -> Result<Request> {
    let prompt = match p.get("prompt") {
        Some(Value::String(t)) => t.clone(),
        Some(_) => return Err(bad(CMD, "`prompt` must be text describing the music")),
        None => return Err(bad(CMD, "`prompt` is required: describe the music (genre, mood, instruments, tempo)")),
    };
    let seconds = p
        .get("seconds")
        .ok_or_else(|| bad(CMD, format!("`seconds` is required ({}–{} s)", request::MIN_SECONDS, request::MAX_SECONDS)))?
        .as_f64()
        .ok_or_else(|| bad(CMD, "`seconds` must be a number"))?;
    let seed = match p.get("seed") {
        None | Some(Value::Null) => 1 + s.project.items.values().filter(|i| i.name.starts_with(&format!("{BASE} "))).count() as u64,
        Some(v) => v.as_u64().ok_or_else(|| bad(CMD, "`seed` must be a whole number from 0"))?,
    };
    let mut r = Request::new(&prompt, seconds, seed);
    r.bpm = whole(p, "bpm", u64::from(request::BPM_RANGE.0)..=u64::from(request::BPM_RANGE.1), "a tempo of 30–300 BPM")?.map(|b| b as u32);
    r.time_signature = whole(p, "timeSignature", 2..=6, "2, 3, 4 or 6 beats per bar")?.map(|t| t as u32);
    r.shift = whole(p, "shift", 1..=3, "1, 2 or 3")?.map_or(3, |x| x as u32);
    r.key = match p.get("key") {
        None | Some(Value::Null) => None,
        Some(Value::String(k)) => Some(request::key_scale(k).map_err(|e| bad(CMD, e.to_string()))?),
        Some(_) => return Err(bad(CMD, "`key` must be text such as \"A minor\"")),
    };
    r.check().map_err(|e| bad(CMD, e.to_string()))?;
    Ok(r)
}

fn track_is_free(s: &Session, track: TrackId, range: TimeRange) -> bool {
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().find(|t| t.id == track))
        .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    poll(s);
    let req = request_p(s, p)?;
    let pkg = model_p(CMD, p)?;
    let place = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "at").unwrap_or(s.playhead());
    if at < Tick::ZERO {
        return Err(bad(CMD, "the start time cannot be negative"));
    }
    let track = match (place, p.get("track")) {
        (true, Some(v)) => {
            let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
            let id = crate::mixer::strip_ref(seq, v)
                .filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id))
                .ok_or_else(|| bad(CMD, format!("no audio track {v}")))?;
            let dur = Tick::from_units(req.samples() as i64, i64::from(request::SAMPLE_RATE));
            if !track_is_free(s, id, TimeRange::new(at, dur)) {
                return Err(bad(CMD, format!("audio track {v} is locked or has clips there")));
            }
            Some(id)
        }
        _ => None,
    };
    if !available() {
        return Err(bad(CMD, "neural music is not available in this build (built without `neural-music`)"));
    }
    // one at a time: each holds a multi-gigabyte model
    if let Some(q) = s.neural_music.pending.first() {
        return Err(bad(CMD, format!("music is already being generated (job {}); wait for it or cancel it (jobs.cancel)", q.job)));
    }
    let models = crate::transcript::models_dir().ok_or_else(|| bad(CMD, "no data directory for models"))?;
    if !catalog::installed(&models, pkg) {
        return Err(bad(
            CMD,
            format!(
                "the music model {} is not downloaded yet ({} still to download from {}, licence: {}); once the user agrees, run music.downloadNeuralModel {{\"model\": \"{}\", \"consent\": true}}",
                pkg.name,
                gb(catalog::missing_bytes(&models, pkg)),
                pkg.source,
                pkg.license,
                pkg.id
            ),
        ));
    }
    let dir = crate::voiceover::generated_dir(s, p, "Generated Audio")?;
    let path = crate::voiceover::unique_wav_path(s, &dir, BASE)?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let short: String = request::clean_prompt(&req.prompt).chars().take(40).collect();
    let job = crate::narration::new_job(s, format!("Generating music: {short}"));
    job.progress.total.store(1000, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: Outcome = Default::default();
    let job_id = job.id;
    s.neural_music.pending.push(PendingMusic { job: job_id, out: out.clone(), path: path.clone(), place, at, track, req: req.clone(), model: pkg.id });
    let package = catalog::package_dir(&models, pkg);
    let r2 = req.clone();
    let spawned = crate::narration::spawn(s, job, wait, "filmcraft-neural-music", move || {
        let t0 = web_time::Instant::now();
        let r = render(&package, &r2, &prog);
        let status = match &r {
            Ok(_) => Ok(format!("{:.0} s of music generated", r2.seconds)),
            Err(e) => Err(e.clone()),
        };
        *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
        crate::narration::finish(&prog, &res, status, t0);
    });
    if let Err(e) = spawned {
        s.neural_music.pending.retain(|q| q.job != job_id);
        return Err(e);
    }
    let mut v = json!({"job": job_id, "path": path, "model": pkg.id, "seed": req.seed, "seconds": req.seconds});
    if wait {
        poll(s);
        let o = s.neural_music.outcomes.get(&job_id).cloned().unwrap_or(Value::Null);
        if let Some(e) = o.get("error").and_then(Value::as_str) {
            return Err(bad(CMD, e.to_string()));
        }
        if let (Some(dst), Some(src)) = (v.as_object_mut(), o.as_object()) {
            for (k, x) in src {
                dst.insert(k.clone(), x.clone());
            }
        }
    }
    Ok(v)
}

/// What generation job `job` made, when it is one (`music` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.neural_music.outcomes.get(&job).cloned()
}

/// Import and place the music of finished jobs. Called each UI frame (through
/// [`Session::poll_persistence`]), by `jobs.list` and by the `music.*` commands.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.neural_music.pending.len() {
        let done = s.neural_music.pending.get(i).is_some_and(|q| q.out.lock().unwrap_or_else(PoisonError::into_inner).is_some());
        if !done {
            i += 1;
            continue;
        }
        let q = s.neural_music.pending.remove(i);
        let r = q.out.lock().unwrap_or_else(PoisonError::into_inner).take();
        let outcome = match r {
            Some(Ok(bytes)) => match land(s, &q, bytes) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(CMD, format!("Neural music: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            None => continue,
        };
        s.neural_music.outcomes.insert(q.job, outcome);
        // bounded: the oldest outcomes go
        if s.neural_music.outcomes.len() > 256
            && let Some(old) = s.neural_music.outcomes.keys().min().copied()
        {
            s.neural_music.outcomes.remove(&old);
        }
    }
}

/// Write, import and place one finished generation (one undo step).
fn land(s: &mut Session, q: &PendingMusic, bytes: Vec<u8>) -> Result<Value> {
    let samples = bytes.len().saturating_sub(44) / 8;
    let dur = Tick::from_units(samples as i64, i64::from(request::SAMPLE_RATE));
    let range = TimeRange::new(q.at, dur);
    let place = q.place && s.state.active_sequence.is_some();
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        if !cfg!(target_arch = "wasm32")
            && let Some(dir) = std::path::Path::new(&q.path).parent()
        {
            std::fs::create_dir_all(dir).map_err(|e| EngineError::Other(format!("{}: {e}", dir.display())))?;
        }
        s.services.write_file(&q.path, &bytes).map_err(|e| EngineError::Other(format!("{}: {e}", q.path)))?;
        let item: ItemId = crate::commands::import_bytes(s, &q.path, bytes.into(), None)?;
        let description = request::clean_prompt(&q.req.prompt);
        let note = format!(
            "{} · seed {} · {:.1} s · shift {}{}{}{}",
            q.model,
            q.req.seed,
            q.req.seconds,
            q.req.shift,
            q.req.bpm.map(|b| format!(" · {b} BPM")).unwrap_or_default(),
            q.req.key.as_deref().map(|k| format!(" · {k}")).unwrap_or_default(),
            q.req.time_signature.map(|t| format!(" · {t}/4")).unwrap_or_default(),
        );
        s.edit("Generate Neural Music", |pr, _| {
            if let Some(it) = pr.item_mut(item) {
                it.metadata.insert("Description".into(), description);
                it.metadata.insert("Log Note".into(), note);
            }
            Ok(())
        })?;
        let mut out = json!({
            "item": item.0, "path": q.path, "seconds": samples as f64 / f64::from(request::SAMPLE_RATE), "seed": q.req.seed, "model": q.model,
            "sampleRate": request::SAMPLE_RATE, "clip": Value::Null, "track": Value::Null,
        });
        if place {
            let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
            // the chosen track, if it is still free; else the first free one (the music is never
            // dropped because the timeline changed while it was generated)
            let chosen = match q.track {
                Some(t) if track_is_free(s, t, range) => Some(t),
                _ => s.active_sequence().and_then(|seq| seq.audio_tracks.iter().map(|t| t.id).find(|id| track_is_free(s, *id, range))),
            };
            let track = match chosen {
                Some(t) => t,
                None => {
                    let before: Vec<TrackId> = s.active_sequence().map(|q| q.audio_tracks.iter().map(|t| t.id).collect()).unwrap_or_default();
                    s.execute("sequence.addTracks", json!({"video": 0, "audio": 1, "audioAfter": "last"}))?;
                    s.active_sequence()
                        .and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| !before.contains(id)))
                        .ok_or_else(|| EngineError::Other("could not add an audio track".into()))?
                }
            };
            let seq_rate = s.project.sequence(seq_id).map(|q| q.settings.frame_rate).unwrap_or_default();
            let at = q.at;
            let clip: ClipId = s.edit("Generate Neural Music", |pr, st| {
                let mut it = pr
                    .make_track_item(item, TrackKind::Audio, at, TimeRange::new(Tick::ZERO, dur), seq_rate)
                    .ok_or(EngineError::Other("the generated music is not importable".into()))?;
                it.duration = dur;
                let id = it.id;
                let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
                let t = seq.audio_tracks.iter_mut().find(|t| t.id == track).ok_or_else(|| bad(CMD, "the audio track was deleted"))?;
                let pos = t.items.partition_point(|x| x.start <= at);
                t.items.insert(pos, it);
                seq.check().map_err(EngineError::Other)?;
                st.selection = vec![id];
                Ok(id)
            })?;
            let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
            out["clip"] = json!(clip.0);
            out["track"] = json!(crate::mixer::strip_label(seq, track));
            out["start"] = json!(at.0);
            out["duration"] = json!(dur.0);
        }
        Ok(out)
    })();
    crate::clip_ops::collapse_history(s, n0, "Generate Neural Music");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

/// Interleaved stereo 32-bit float WAV.
#[cfg(any(test, feature = "neural-music"))]
fn wav(left: &[f32], right: &[f32], rate: u32) -> Vec<u8> {
    let frames = left.len().min(right.len());
    let data_len = frames.saturating_mul(8);
    let mut v = Vec::with_capacity(44usize.saturating_add(data_len));
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(u32::try_from(36usize.saturating_add(data_len)).unwrap_or(u32::MAX)).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&rate.saturating_mul(8).to_le_bytes());
    v.extend_from_slice(&8u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(u32::try_from(data_len).unwrap_or(u32::MAX)).to_le_bytes());
    for (l, r) in left.iter().zip(right.iter()) {
        v.extend_from_slice(&l.to_le_bytes());
        v.extend_from_slice(&r.to_le_bytes());
    }
    v
}

/// Run the model on the job's thread: the finished clip as WAV bytes.
#[cfg(feature = "neural-music")]
fn render(package: &std::path::Path, req: &Request, prog: &filmcraft_export::Progress) -> std::result::Result<Vec<u8>, String> {
    use std::sync::atomic::Ordering;
    let m = filmcraft_acestep::model::AceStep::load(package).map_err(|e| e.to_string())?;
    let mut music = m
        .generate(req, &mut |p| {
            prog.done.store((p.fraction() * 1000.0) as u64, Ordering::Relaxed);
            let line = match p.stage {
                "text" => "Reading the description".to_string(),
                "load" => "Loading the music model".to_string(),
                "diffusion" => format!("Generating (step {} of {})", p.done + 1, p.total),
                _ => format!("Decoding audio ({} of {})", p.done, p.total.max(1)),
            };
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = line;
            !prog.cancel.load(Ordering::Relaxed)
        })
        .map_err(|e| e.to_string())?;
    music.normalize_peak(PEAK_DB);
    Ok(wav(&music.left, &music.right, music.sample_rate))
}

#[cfg(not(feature = "neural-music"))]
fn render(_: &std::path::Path, _: &Request, _: &filmcraft_export::Progress) -> std::result::Result<Vec<u8>, String> {
    Err("neural music is not available in this build".into())
}

fn download_model(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "music.downloadNeuralModel";
    poll(s);
    let pkg = model_p(cmd, p)?;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    let missing = catalog::missing_bytes(&dir, pkg);
    if missing == 0 {
        return Ok(json!({"job": null, "installed": true, "model": pkg.id}));
    }
    if !bool_p(p, "consent").unwrap_or(false) {
        return Err(bad(
            cmd,
            format!(
                "downloading {} ({} from {}, licence: {}) needs the user's agreement: pass `consent: true` once they agreed",
                pkg.name,
                gb(missing),
                pkg.source,
                pkg.license
            ),
        ));
    }
    if !available() {
        return Err(bad(cmd, "neural music is not available in this build (built without `neural-music`)"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {}", pkg.name));
    job.progress.total.store(missing, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let package = catalog::package_dir(&dir, pkg);
    let done = format!("{} installed", pkg.name);
    let files = pkg.files;
    let id = crate::narration::spawn(s, job, wait, "filmcraft-music-download", move || {
        let t0 = web_time::Instant::now();
        let r = fetch(&package, files, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| done), t0);
    })?;
    Ok(json!({"job": id, "installed": false, "model": pkg.id, "bytes": missing}))
}

#[cfg(feature = "neural-music")]
fn fetch(package: &std::path::Path, files: &'static [catalog::ModelFile], prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> =
        files.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(package, &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "neural-music"))]
fn fetch(_: &std::path::Path, _: &'static [catalog::ModelFile], _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("neural music is not available in this build".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    #[test]
    fn hostile_parameters_are_refused_before_any_work() {
        let mut s = demo();
        let jobs = s.jobs.len();
        for p in [
            json!({}),
            json!({"seconds": 20}),
            json!({"prompt": 5, "seconds": 20}),
            json!({"prompt": "", "seconds": 20}),
            json!({"prompt": "   \n", "seconds": 20}),
            json!({"prompt": "<|endoftext|>", "seconds": 20}),
            json!({"prompt": "x".repeat(600), "seconds": 20}),
            json!({"prompt": "lo-fi"}),
            json!({"prompt": "lo-fi", "seconds": "20"}),
            json!({"prompt": "lo-fi", "seconds": -1}),
            json!({"prompt": "lo-fi", "seconds": 0}),
            json!({"prompt": "lo-fi", "seconds": 1e12}),
            json!({"prompt": "lo-fi", "seconds": 20, "seed": -3}),
            json!({"prompt": "lo-fi", "seconds": 20, "seed": "abc"}),
            json!({"prompt": "lo-fi", "seconds": 20, "seed": 1.5}),
            json!({"prompt": "lo-fi", "seconds": 20, "bpm": 9000}),
            json!({"prompt": "lo-fi", "seconds": 20, "bpm": 12.5}),
            json!({"prompt": "lo-fi", "seconds": 20, "bpm": "fast"}),
            json!({"prompt": "lo-fi", "seconds": 20, "key": "H minor"}),
            json!({"prompt": "lo-fi", "seconds": 20, "key": 7}),
            json!({"prompt": "lo-fi", "seconds": 20, "timeSignature": 5}),
            json!({"prompt": "lo-fi", "seconds": 20, "shift": 4}),
            json!({"prompt": "lo-fi", "seconds": 20, "model": "musicgen"}),
            json!({"prompt": "lo-fi", "seconds": 20, "model": 3}),
            json!({"prompt": "lo-fi", "seconds": 20, "atTime": -5}),
            json!({"prompt": "lo-fi", "seconds": 20, "track": "A99"}),
            json!({"prompt": "lo-fi", "seconds": 20, "track": {"x": 1}}),
        ] {
            let e = s.execute(CMD, p.clone()).unwrap_err();
            assert!(!e.to_string().is_empty(), "{p}");
        }
        assert_eq!(s.jobs.len(), jobs, "no job was started");
        assert!(s.neural_music.pending.is_empty());
    }

    #[test]
    fn a_good_request_explains_what_is_missing() {
        // the model is not downloaded in tests (or the build lacks the feature): the error says
        // what to do, with the size, source and licence
        let installed = crate::transcript::models_dir().is_some_and(|d| catalog::MODELS.iter().any(|m| catalog::installed(&d, m)));
        if installed && available() {
            // a developer machine with the model in its data directory would generate for real
            return;
        }
        let mut s = demo();
        let m = s.execute(CMD, json!({"prompt": "calm piano", "seconds": 20, "seed": 3})).unwrap_err().to_string();
        assert!(m.contains("neural-music") || (m.contains("music.downloadNeuralModel") && m.contains("MIT") && m.contains("GB")), "{m}");
        assert!(s.neural_music.pending.is_empty());
    }

    #[test]
    fn the_models_are_listed_with_licence_size_and_source() {
        let mut s = demo();
        let r = s.execute("music.neuralModels", json!({})).unwrap();
        assert_eq!(r["default"], catalog::DEFAULT_MODEL);
        let ms = r["models"].as_array().unwrap();
        assert_eq!(ms.len(), catalog::MODELS.len());
        for m in ms {
            assert!(m["license"].as_str().is_some_and(|l| l.contains("MIT") && !l.contains("NC")), "{m}");
            assert!(m["size"].as_u64().is_some_and(|b| b > 1_000_000_000), "{m}");
            assert!(m["source"].as_str().is_some_and(|u| u.starts_with("https://huggingface.co/ACE-Step/")), "{m}");
            assert!(m["installed"].is_boolean() && m["missingBytes"].is_u64());
        }
        assert_eq!(r["available"], available());
    }

    #[test]
    fn downloads_need_consent_and_a_known_model() {
        let mut s = demo();
        assert!(s.execute("music.downloadNeuralModel", json!({"model": "nope", "consent": true})).is_err());
        let installed = crate::transcript::models_dir().is_some_and(|d| catalog::MODELS.iter().all(|m| catalog::installed(&d, m)));
        if !installed {
            let e = s.execute("music.downloadNeuralModel", json!({})).unwrap_err().to_string();
            assert!(e.contains("consent") && e.contains("GB") && e.contains("huggingface.co"), "{e}");
            let e = s.execute("music.downloadNeuralModel", json!({"consent": false})).unwrap_err().to_string();
            assert!(e.contains("consent"), "{e}");
        }
    }

    #[test]
    fn a_finished_generation_lands_on_a_free_track_as_one_undo_step() {
        let mut s = demo();
        let dir = std::env::temp_dir().join(format!("filmcraft-neural-music-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let req = Request::new("calm piano", 6.0, 9);
        // a fake finished job: half a second of a sine as the model's output
        let n = 24_000;
        let l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let bytes = wav(&l, &l, request::SAMPLE_RATE);
        let out: Outcome = Arc::new(Mutex::new(Some(Ok(bytes))));
        let path = dir.join("Neural Music 1.wav").to_string_lossy().into_owned();
        let undo0 = s.history.undo.len();
        let items0 = s.project.items.len();
        s.neural_music.pending.push(PendingMusic {
            job: 77,
            out,
            path: path.clone(),
            place: true,
            at: Tick::ZERO,
            track: None,
            req,
            model: "acestep-v15-turbo",
        });
        // jobs.list applies it
        s.execute("jobs.list", json!({})).unwrap();
        assert!(s.neural_music.pending.is_empty());
        let o = outcome_json(&s, 77).unwrap();
        assert!(o.get("error").is_none(), "{o}");
        let item = ItemId(o["item"].as_u64().unwrap());
        assert_eq!(s.project.items.len(), items0 + 1);
        let it = s.project.item(item).unwrap();
        assert_eq!(it.metadata.get("Description").map(String::as_str), Some("calm piano"));
        assert!(it.metadata.get("Log Note").is_some_and(|n| n.contains("seed 9") && n.contains("acestep-v15-turbo")));
        assert!(o["clip"].as_u64().is_some(), "{o}");
        assert_eq!(o["duration"].as_i64(), Some(Tick::from_units(n as i64, 48_000).0));
        assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
        assert!(std::path::Path::new(&path).is_file());
        // a failed job only reports its error
        let out: Outcome = Arc::new(Mutex::new(Some(Err("model failed".into()))));
        s.neural_music.pending.push(PendingMusic {
            job: 78,
            out,
            path: path.replace(" 1.wav", " 2.wav"),
            place: true,
            at: Tick::ZERO,
            track: None,
            req: Request::new("x", 6.0, 1),
            model: "acestep-v15-turbo",
        });
        poll(&mut s);
        assert_eq!(outcome_json(&s, 78).unwrap()["error"], "model failed");
        assert_eq!(s.history.undo.len(), undo0 + 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End to end with the real model (gated): a 10 s clip generated, written, imported and
    /// placed, blocking (`wait`) and as a background job landed by `jobs.list`.
    ///
    /// `FILMCRAFT_ACESTEP_E2E=1 FILMCRAFT_DATA_DIR=<dir whose models/acestep-v15-turbo is the
    /// package> cargo test --release -p filmcraft-engine --features neural-music live_neural_music
    /// -- --nocapture`
    #[cfg(feature = "neural-music")]
    #[test]
    fn live_neural_music_lands_on_the_timeline() {
        if std::env::var_os("FILMCRAFT_ACESTEP_E2E").is_none() {
            eprintln!("skipped (FILMCRAFT_ACESTEP_E2E unset)");
            return;
        }
        let mut s = demo();
        let dir = std::env::temp_dir().join(format!("filmcraft-neural-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let d = dir.to_string_lossy().into_owned();
        let t0 = web_time::Instant::now();
        let r = s
            .execute(CMD, json!({"prompt": "Warm acoustic folk with fingerpicked guitar and soft shaker, gentle and hopeful", "seconds": 10, "seed": 5, "bpm": 96, "wait": true, "dir": d}))
            .unwrap();
        eprintln!("blocking: {r} in {:.1?}", t0.elapsed());
        let clip = ClipId(r["clip"].as_u64().unwrap());
        let seq = s.active_sequence().unwrap();
        let (_, it) = seq.find_item(clip).unwrap();
        assert_eq!(it.duration, Tick::from_units(480_000, 48_000));
        let bytes = std::fs::read(r["path"].as_str().unwrap()).unwrap();
        assert_eq!(bytes.len(), 44 + 480_000 * 8);
        let peak = bytes[44..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b).abs()).fold(0f32, f32::max);
        assert!((20.0 * peak.log10() + 1.0).abs() < 0.01, "peak {peak}");
        // in the background: the job runs, jobs.list lands it
        let r = s.execute(CMD, json!({"prompt": "Driving synthwave with retro drum machine and pulsing bass", "seconds": 8, "seed": 6, "dir": d})).unwrap();
        let job = r["job"].as_u64().unwrap();
        let landed = loop {
            let jobs = s.execute("jobs.list", json!({})).unwrap();
            let j = jobs.as_array().unwrap().iter().find(|j| j["id"].as_u64() == Some(job)).cloned().unwrap();
            if j["finished"] == true {
                break j;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        };
        eprintln!("background: {landed}");
        assert!(landed["music"]["clip"].as_u64().is_some(), "{landed}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_generation_at_a_time() {
        let mut s = demo();
        // a generation still running (no outcome yet)
        s.neural_music.pending.push(PendingMusic {
            job: 5,
            out: Default::default(),
            path: "x.wav".into(),
            place: false,
            at: Tick::ZERO,
            track: None,
            req: Request::new("x", 6.0, 1),
            model: "acestep-v15-turbo",
        });
        let e = s.execute(CMD, json!({"prompt": "calm piano", "seconds": 20})).unwrap_err().to_string();
        if available() {
            assert!(e.contains("already being generated") && e.contains("job 5"), "{e}");
        } else {
            assert!(e.contains("not available"), "{e}");
        }
        poll(&mut s);
        assert_eq!(s.neural_music.pending.len(), 1, "an unfinished job stays pending");
    }

    #[test]
    fn the_wav_is_stereo_float_at_48k() {
        let b = wav(&[0.5, -0.5], &[0.25, 1.0], 48_000);
        assert_eq!(&b[..4], b"RIFF");
        assert_eq!(b.len(), 44 + 16);
        assert_eq!(u16::from_le_bytes([b[22], b[23]]), 2);
        assert_eq!(u32::from_le_bytes([b[24], b[25], b[26], b[27]]), 48_000);
        assert_eq!(f32::from_le_bytes([b[44], b[45], b[46], b[47]]), 0.5);
    }
}
