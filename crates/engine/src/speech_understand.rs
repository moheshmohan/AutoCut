//! Speech understanding of raw clips, and the edits built on it.
//!
//! | command | does |
//! |---|---|
//! | `media.understandSpeech` | a background job: per media item, where speech is (Silero VAD, else energy), the words (Whisper), who speaks (ECAPA embeddings, else MFCC), pauses, fillers, hesitations and retakes with the take to keep, and a talking-head hint (a face on screen, when the face detector is downloaded); stored with the project (transcript and analysis, one undo step) |
//! | `media.inspectSpeech` | the stored analyses |
//! | `transcript.removeRetakes` | ripple-delete from the active sequence every take a retake drops (one undo step) |
//! | `sequence.roughCut` | a new sequence of the items' kept speech: pauses, fillers and dropped takes cut out, every cut between words, the pieces back to back (one undo step) |
//! | `sequence.speakerTrack` | colour each clip of the active sequence by the speaker who talks most in it |
//!
//! **Models** (all downloaded on request, `transcript.downloadModel`): the recogniser is `model`,
//! else for `language: "ml"` the Malayalam model and for `"en"` Whisper large-v3-turbo when they
//! are downloaded, else the preferences' model; Silero VAD (`silero-vad`) and the speaker model
//! (`ecapa-voxceleb`) are used when downloaded, the energy detector and MFCC clustering
//! otherwise (the analysis says which: `vad`, `diarization`). The job holds the GPU lock while it
//! hears (`filmcraft-cli locks`).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_edit::rough_cut;
use filmcraft_edit::transcript as tx;
use filmcraft_project::speech::{ShotHint, SpeechAnalysis};
use filmcraft_project::{ItemId, ItemKind, Label, Transcript};
use filmcraft_speech::understand;
use filmcraft_time::{TICKS_PER_SECOND, Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

/// Longest clip understood, seconds (three hours: the audio is held in memory, 16 kHz).
pub const MAX_SECONDS: f64 = 3.0 * 3600.0;
/// Finished jobs whose outcome is kept for `jobs.list`.
const MAX_OUTCOMES: usize = 16;

type Results = Arc<Mutex<Option<std::result::Result<Vec<Made>, String>>>>;

/// What one item's understanding made.
struct Made {
    item: ItemId,
    transcript: Transcript,
    analysis: SpeechAnalysis,
}

struct Pending {
    job: u64,
    results: Results,
}

/// The session's speech understanding jobs and what the finished ones found.
#[derive(Default)]
pub struct Understandings {
    pending: Vec<Pending>,
    outcomes: HashMap<u64, Value>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn ids_p(p: &Value, k: &str) -> Option<Vec<ItemId>> {
    p.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(ItemId).collect())
}

/// The media items a command works on: `items` / `item`, else the Project panel selection, else
/// the media of the active sequence's clips (subclips resolve to their media, no duplicates).
fn targets(s: &Session, p: &Value) -> Vec<ItemId> {
    let mut raw = ids_p(p, "items").or_else(|| u64_p(p, "item").map(|i| vec![ItemId(i)])).unwrap_or_default();
    if raw.is_empty() {
        raw = s.state.project_selection.clone();
    }
    if raw.is_empty()
        && let Some(q) = s.active_sequence()
    {
        raw = q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|it| it.enabled).map(|it| it.item).collect();
    }
    let mut out = Vec::new();
    for i in raw {
        if let Some((m, _, _)) = s.project.resolve_media(i)
            && !out.contains(&m)
        {
            out.push(m);
        }
    }
    out
}

/// A finite number of seconds in `lo..=hi`, or the default.
fn secs_p(cmd: &str, p: &Value, k: &str, lo: f64, hi: f64, default: f64) -> Result<f64> {
    match p.get(k).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => {
            v.as_f64().filter(|f| f.is_finite() && (lo..=hi).contains(f)).ok_or_else(|| bad(cmd, format!("`{k}` is a number of seconds from {lo} to {hi}")))
        }
    }
}

fn flag_p(cmd: &str, p: &Value, k: &str, default: bool) -> Result<bool> {
    match p.get(k).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, format!("`{k}` must be true or false"))),
    }
}

/// The recogniser for `language` (see the module docs).
fn recogniser(s: &Session, p: &Value, language: Option<&str>) -> Result<Arc<dyn filmcraft_speech::Transcriber>> {
    if s.transcriber.is_some() || str_p(p, "model").is_some() {
        return crate::transcript::transcriber(s, p);
    }
    let preferred = match language {
        Some("ml") => Some(filmcraft_speech::models::MALAYALAM_MODEL),
        Some("en") => Some(filmcraft_speech::models::TURBO_MODEL),
        _ => None,
    };
    if let (Some(id), Some(dir)) = (preferred, crate::transcript::models_dir())
        && filmcraft_speech::models::find(id).is_some_and(|m| filmcraft_speech::models::installed(&dir, m))
    {
        return crate::transcript::transcriber(s, &json!({"model": id}));
    }
    crate::transcript::transcriber(s, p)
}

/// One item to understand.
struct Work {
    item: ItemId,
    name: String,
    src: filmcraft_media::SharedSource,
    samples: usize,
    /// The video's size, when it has a picture.
    video: Option<(u32, u32)>,
}

struct Settings {
    language: Option<String>,
    diarize: bool,
    max_speakers: usize,
    faces: bool,
    understand: understand::Params,
}

/// `media.understandSpeech`.
fn understand_speech(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.understandSpeech";
    let items = targets(s, p);
    if items.is_empty() {
        return Err(bad(cmd, "nothing to understand (pass `items`, select clips, or open a sequence with audio)"));
    }
    let language = match p.get("language").filter(|v| !v.is_null()) {
        None => None,
        Some(Value::String(l)) if l.is_empty() || l == "auto" => None,
        Some(Value::String(l)) if l.len() <= 8 && l.chars().all(|c| c.is_ascii_alphabetic() || c == '-') => {
            Some(l.split('-').next().unwrap_or(l).to_ascii_lowercase())
        }
        Some(_) => return Err(bad(cmd, "`language` is an ISO 639-1 code (\"en\", \"ml\") or \"auto\"")),
    };
    let fillers: Vec<String> = match p.get("fillers").filter(|v| !v.is_null()) {
        None => Vec::new(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).take(200).collect(),
        Some(_) => return Err(bad(cmd, "`fillers` is a list of words and phrases")),
    };
    let settings = Settings {
        language: language.clone(),
        diarize: flag_p(cmd, p, "diarize", true)?,
        max_speakers: match p.get("maxSpeakers").filter(|v| !v.is_null()) {
            None => 6,
            Some(v) => v.as_u64().filter(|n| (1..=32).contains(n)).ok_or_else(|| bad(cmd, "`maxSpeakers` is a whole number from 1 to 32"))? as usize,
        },
        faces: flag_p(cmd, p, "faces", true)?,
        understand: understand::Params { min_pause: secs_p(cmd, p, "minPause", 0.1, 10.0, 0.5)?, fillers, ..Default::default() },
    };
    let t = recogniser(s, p, language.as_deref())?;
    let sr = filmcraft_speech::SAMPLE_RATE;
    let mut work = Vec::new();
    let mut skipped = Vec::new();
    for item in items {
        let Some(pi) = s.project.item(item) else { continue };
        let ItemKind::Media(m) = &pi.kind else {
            skipped.push(json!({"item": item.0, "why": "not a media clip"}));
            continue;
        };
        let secs = m.duration().0 as f64 / TICKS_PER_SECOND as f64;
        if !secs.is_finite() || secs > MAX_SECONDS {
            return Err(bad(cmd, format!("{} is {secs:.0} s long; at most {MAX_SECONDS:.0} s are understood at once", pi.name)));
        }
        let Some(src) = s.source(item) else {
            skipped.push(json!({"item": item.0, "why": "offline"}));
            continue;
        };
        if !src.info().has_audio() {
            skipped.push(json!({"item": item.0, "why": "no audio"}));
            continue;
        }
        let samples = m.duration().to_units_floor(i64::from(sr)).max(0) as usize;
        let video = src.info().video.as_ref().map(|v| (v.width, v.height));
        work.push(Work { item, name: pi.name.clone(), src, samples, video });
    }
    if work.is_empty() {
        return Err(bad(cmd, "none of the clips has audio"));
    }
    let models = crate::transcript::models_dir();
    let detect = if settings.faces && work.iter().any(|w| w.video.is_some()) {
        crate::redact::load_detectors(&crate::redact::Options { faces: true, plates: false, ..Default::default() }).ok()
    } else {
        None
    };
    let n = work.len();
    let job = crate::narration::new_job(s, format!("Understanding speech ({n} clip{})", if n == 1 { "" } else { "s" }));
    job.progress.total.store(n as u64, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let host_transcriber = s.transcriber.is_some();
    let run = move || {
        let t0 = web_time::Instant::now();
        // the GPU is shared with other sessions' heavy models (not for a host's own recogniser)
        let _gpu = if host_transcriber || cfg!(test) { None } else { gpu_lock() };
        let r = (|| -> std::result::Result<Vec<Made>, String> {
            let mut made = Vec::with_capacity(work.len());
            for (i, w) in work.iter().enumerate() {
                if prog.cancel.load(Ordering::Relaxed) {
                    return Err("stopped".into());
                }
                *lock(&prog.status) = format!("{}: reading the audio ({} of {})", w.name, i + 1, work.len());
                made.push(understand_one(w, t.as_ref(), models.as_deref(), detect.as_ref(), &settings, &prog, i, work.len())?);
                prog.done.store(i as u64 + 1, Ordering::Relaxed);
            }
            Ok(made)
        })();
        let secs = t0.elapsed().as_secs_f64();
        *lock(&prog.status) = match &r {
            Ok(m) => {
                let retakes: usize = m.iter().map(|x| x.analysis.retakes.len()).sum();
                let fillers: usize = m.iter().map(|x| x.analysis.fillers.len()).sum();
                format!("Understood {} clip(s) in {secs:.1}s: {retakes} retake(s), {fillers} filler(s)", m.len())
            }
            Err(e) => e.clone(),
        };
        let report = r
            .as_ref()
            .map(|_| filmcraft_export::Report { path: String::new(), frames: 0, seconds: secs, bytes: 0, render_fps: 0.0, ..Default::default() })
            .map_err(Clone::clone);
        if let Err(e) = &report {
            *lock(&prog.error) = Some(e.clone());
        }
        // the results before `finished`: whoever sees the job finished can apply them
        *lock(&out) = Some(r);
        *lock(&res) = Some(report);
        prog.finished.store(true, Ordering::Relaxed);
    };
    s.speech_jobs.pending.push(Pending { job: id, results });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-understand-speech", run)?;
    let mut answer = json!({"job": id, "items": n, "skipped": skipped});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.speech_jobs.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["speech"] = o.clone();
        }
    }
    Ok(answer)
}

/// The GPU lock while speech models run (None: no place for locks; a busy GPU waits).
fn gpu_lock() -> Option<filmcraft_locks::Guard> {
    filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, "understanding speech", std::time::Duration::from_secs(30 * 60), &mut |_| {}).ok()
}

#[allow(clippy::too_many_arguments)]
fn understand_one(
    w: &Work,
    t: &dyn filmcraft_speech::Transcriber,
    models: Option<&std::path::Path>,
    detect: Option<&crate::redact::DetectFn>,
    st: &Settings,
    prog: &filmcraft_export::Progress,
    i: usize,
    n: usize,
) -> std::result::Result<Made, String> {
    let sr = filmcraft_speech::SAMPLE_RATE;
    let buf = w.src.audio(0, w.samples, sr).map_err(|e| format!("{}: can't read the audio: {e}", w.name))?;
    let audio = filmcraft_speech::downmix(&buf.channels);
    // where speech is
    let silero = models.and_then(filmcraft_speech::vad::load_silero);
    let (segs, how) = filmcraft_speech::vad::speech_segments(&audio, silero.as_deref(), &filmcraft_speech::silero::SegmentParams::default());
    let vad = match how {
        filmcraft_speech::vad::VadMethod::Silero => "silero",
        filmcraft_speech::vad::VadMethod::Energy => "energy",
    };
    // the words
    *lock(&prog.status) = format!("{}: hearing the words ({} of {n})", w.name, i + 1);
    let opts = filmcraft_speech::Options { language: st.language.clone(), diarize: false, max_speakers: st.max_speakers };
    let mut stop = |_: f32, _: &str| !prog.cancel.load(Ordering::Relaxed);
    // speech only, in chunks a decoder holds: Malayalam takes about two tokens a character, so
    // its chunks are 12 s; other languages' 28 s
    let max = match st.language.as_deref() {
        Some("en") => 28.0,
        Some("ml") | None => tts_heard_seconds(),
        Some(_) => 20.0,
    };
    let chunks =
        filmcraft_speech::vad::chunks(&audio, &segs, (max * f64::from(filmcraft_speech::SAMPLE_RATE)) as usize, filmcraft_speech::SAMPLE_RATE as usize / 5);
    let mut tr = t.transcribe_chunks(&audio, &chunks, &opts, &mut stop).map_err(|e| format!("{}: {e}", w.name))?;
    tr.normalize();
    // who speaks
    let diarization = if st.diarize && !tr.words.is_empty() {
        *lock(&prog.status) = format!("{}: telling the speakers apart ({} of {n})", w.name, i + 1);
        filmcraft_speech::diarize::diarize_auto(&audio, &mut tr, models, Some(&segs), st.max_speakers).1.id()
    } else {
        "off"
    };
    tr.normalize();
    let mut a = understand::analyse(&tr, &audio, &segs, vad, diarization, &st.understand);
    // a talking head: a face in most of three frames taken while someone speaks
    if let (Some(det), Some((vw, vh))) = (detect, w.video) {
        let times: Vec<Tick> = pick_times(&a);
        let scale = (640.0 / f64::from(vw.max(vh).max(1))).min(1.0) as f32;
        let mut seen = 0;
        let mut looked = 0;
        for at in times {
            let Ok(f) = w.src.video_frame(filmcraft_media::FrameRequest { time: at, scale }) else { continue };
            looked += 1;
            let rgba = f.to_rgba8();
            let faces = det(&rgba, f.width as usize, f.height as usize, 640);
            if faces.iter().any(|d| d.score >= 0.6 && d.bbox[2] * d.bbox[3] >= 0.004) {
                seen += 1;
            }
        }
        if looked > 0 {
            a.face = Some(seen * 2 > looked);
        }
    }
    let speaking = a.speech_share() >= 0.15 || tr.words.len() >= 5;
    a.shot = match (speaking, a.face) {
        (false, _) => ShotHint::BRoll,
        (true, Some(true)) => ShotHint::TalkingHead,
        (true, Some(false)) => ShotHint::Voice,
        (true, None) => ShotHint::Unknown,
    };
    a.check(tr.words.len())?;
    Ok(Made { item: w.item, transcript: tr, analysis: a })
}

/// The longest stretch a Malayalam chunk is heard in, seconds (what the narration check hears at
/// once: the decoder holds about 12 s of Malayalam).
fn tts_heard_seconds() -> f64 {
    filmcraft_tts::heard::HEARD_SECONDS
}

/// Three moments of speech: a quarter, half and three quarters of the way through the segments.
fn pick_times(a: &SpeechAnalysis) -> Vec<Tick> {
    if a.segments.is_empty() {
        return [0.25, 0.5, 0.75].iter().map(|f| Tick((a.duration.0 as f64 * f) as i64)).collect();
    }
    let n = a.segments.len();
    [n / 4, n / 2, (3 * n) / 4].iter().filter_map(|&k| a.segments.get(k.min(n - 1))).map(|s| Tick(s.start.0 + (s.end.0 - s.start.0) / 2)).collect()
}

/// Apply finished jobs (one undo step each) and keep what they found for `jobs.list`. Called each
/// UI frame (through [`Session::poll_persistence`]), by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.speech_jobs.pending.len() {
        let Some(pj) = s.speech_jobs.pending.get(i) else { break };
        let ready = lock(&pj.results).is_some();
        let job = s.jobs.iter().find(|j| j.id == pj.job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if !ready && !finished {
            i += 1;
            continue;
        }
        let pj = s.speech_jobs.pending.remove(i);
        let results = lock(&pj.results).take();
        let outcome = match results {
            Some(Ok(made)) => {
                let items: Vec<Value> = made.iter().map(|m| analysis_json(m.item, &m.transcript, &m.analysis)).collect();
                let r = s.edit("Understand Speech", move |pr, _| {
                    for m in made {
                        // only items still in the project
                        if pr.items.contains_key(&m.item) {
                            pr.transcripts.insert(m.item, Arc::new(m.transcript));
                            pr.speech.insert(m.item, Arc::new(m.analysis));
                        }
                    }
                    Ok(())
                });
                match r {
                    Ok(()) => json!({"items": items}),
                    Err(e) => json!({"error": e.to_string()}),
                }
            }
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "stopped: nothing was changed"}),
        };
        s.speech_jobs.outcomes.insert(pj.job, outcome);
        if s.speech_jobs.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.speech_jobs.outcomes.keys().min().copied()
        {
            s.speech_jobs.outcomes.remove(&old);
        }
    }
}

/// What a finished job found (`speech` in its `jobs.list` entry).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.speech_jobs.outcomes.get(&job).cloned()
}

fn secs(t: Tick) -> f64 {
    (t.0 as f64 / TICKS_PER_SECOND as f64 * 1000.0).round() / 1000.0
}

/// One item's analysis as JSON: times in seconds (`start`, `end`) and ticks (`startTicks`…).
pub fn analysis_json(item: ItemId, t: &Transcript, a: &SpeechAnalysis) -> Value {
    let span = |x: Tick, y: Tick| json!({"start": secs(x), "end": secs(y), "startTicks": x.0, "endTicks": y.0});
    let speaker = |k: Option<u32>| k.and_then(|k| t.speakers.get(k as usize)).map(|s| s.name.clone());
    let segments: Vec<Value> = a
        .segments
        .iter()
        .enumerate()
        .map(|(k, sg)| {
            let mut v = span(sg.start, sg.end);
            let words: Vec<Value> = t
                .words
                .get(sg.from..=sg.to.min(t.words.len().saturating_sub(1)))
                .unwrap_or(&[])
                .iter()
                .map(|w| json!({"text": w.text, "start": secs(w.start), "end": secs(w.end), "confidence": (w.confidence * 1000.0).round() / 1000.0}))
                .collect();
            v["index"] = json!(k);
            v["speaker"] = json!(speaker(sg.speaker));
            v["text"] = json!(sg.text);
            v["from"] = json!(sg.from);
            v["to"] = json!(sg.to);
            v["confidence"] = json!((sg.confidence * 1000.0).round() / 1000.0);
            v["fluency"] = json!((sg.fluency * 1000.0).round() / 1000.0);
            v["fillers"] = json!(sg.fillers);
            v["restarts"] = json!(sg.restarts);
            if sg.clipped {
                v["clipped"] = json!(true);
            }
            v["words"] = Value::Array(words);
            v
        })
        .collect();
    let dropped = a.dropped_takes();
    json!({
        "item": item.0,
        "language": a.language,
        "source": a.source,
        "vad": a.vad,
        "diarization": a.diarization,
        "duration": secs(a.duration),
        "speechShare": (a.speech_share() * 1000.0).round() / 1000.0,
        "shot": a.shot,
        "face": a.face,
        "speakers": t.speakers.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "speech": a.speech.iter().map(|s| span(s.start, s.end)).collect::<Vec<_>>(),
        "segments": segments,
        "pauses": a.pauses.iter().map(|s| span(s.start, s.end)).collect::<Vec<_>>(),
        "fillers": a.fillers.iter().map(|f| { let mut v = span(f.start, f.end); v["text"] = json!(f.text); v["from"] = json!(f.from); v["to"] = json!(f.to); v["hesitation"] = json!(f.is_hesitation()); v }).collect::<Vec<_>>(),
        "retakes": a.retakes.iter().map(|r| json!({
            "takes": r.takes, "keep": r.keep, "similarity": (r.similarity * 1000.0).round() / 1000.0, "reason": r.reason,
            "texts": r.takes.iter().filter_map(|k| a.segments.get(*k)).map(|s| s.text.clone()).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "droppedTakes": dropped,
    })
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let items: Vec<ItemId> = match ids_p(p, "items").or_else(|| u64_p(p, "item").map(|i| vec![ItemId(i)])) {
        Some(v) => v.into_iter().filter_map(|i| s.project.resolve_media(i).map(|(m, _, _)| m)).collect(),
        None => s.project.speech.keys().copied().collect(),
    };
    let empty = Transcript::default();
    let out: Vec<Value> = items
        .iter()
        .filter_map(|i| {
            let a = s.project.speech.get(i)?;
            let t = s.project.transcripts.get(i).map(|t| t.as_ref()).unwrap_or(&empty);
            Some(analysis_json(*i, t, a))
        })
        .collect();
    Ok(json!({"items": out}))
}

fn has_speech(s: &Session) -> std::result::Result<(), String> {
    if s.project.speech.is_empty() { Err("no clip has been understood yet (media.understandSpeech)".into()) } else { Ok(()) }
}

fn has_speech_seq(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    has_speech(s)
}

/// `transcript.removeRetakes`: the takes retakes drop, cut out of the active sequence.
fn remove_retakes(s: &mut Session, _p: &Value) -> Result<Value> {
    let words = crate::transcript::sequence_words(s);
    // sequence word index ranges of the dropped takes
    let mut hits: Vec<std::ops::Range<usize>> = Vec::new();
    let mut takes = 0;
    for (item, a) in &s.project.speech {
        for k in a.dropped_takes() {
            let Some(sg) = a.segments.get(k) else { continue };
            let idx: Vec<usize> = words.iter().enumerate().filter(|(_, w)| w.item == *item && w.index >= sg.from && w.index <= sg.to).map(|(i, _)| i).collect();
            if idx.is_empty() {
                continue;
            }
            takes += 1;
            // consecutive runs (a take cut by an edit reads as several)
            let mut a0 = idx[0];
            for pair in idx.windows(2) {
                if pair[1] != pair[0] + 1 {
                    hits.push(a0..pair[0] + 1);
                    a0 = pair[1];
                }
            }
            if let Some(last) = idx.last() {
                hits.push(a0..last + 1);
            }
        }
    }
    hits.sort_by_key(|r| r.start);
    let ranges = tx::filler_ranges(&words, &hits, s.sequence_rate());
    let n = ranges.len();
    if n == 0 {
        return Ok(json!({"removed": 0, "takes": 0, "ticks": 0, "seconds": 0.0}));
    }
    let total = s.edit_sequence("Remove Retakes", |q, ctx, _| Ok(tx::ripple_delete_ranges(q, ranges, ctx)))?;
    Ok(json!({"removed": n, "takes": takes, "ticks": total.0, "seconds": secs(total)}))
}

/// `sequence.roughCut`.
fn rough_cut(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "sequence.roughCut";
    let items = targets(s, p);
    if items.is_empty() {
        return Err(bad(cmd, "nothing to cut (pass `items`, or select understood clips)"));
    }
    let missing: Vec<u64> = items.iter().filter(|i| !s.project.speech.contains_key(i) || !s.project.transcripts.contains_key(i)).map(|i| i.0).collect();
    if !missing.is_empty() {
        return Err(bad(cmd, format!("understand the clips first (media.understandSpeech): {missing:?} have no speech analysis")));
    }
    let o = rough_cut::Options {
        remove_pauses: flag_p(cmd, p, "removePauses", true)?,
        remove_fillers: flag_p(cmd, p, "removeFillers", true)?,
        remove_retakes: flag_p(cmd, p, "removeRetakes", true)?,
        remove_hesitations: flag_p(cmd, p, "removeHesitations", false)?,
        min_pause: Tick::from_seconds_f64(secs_p(cmd, p, "minPause", 0.1, 30.0, 0.6)?),
        keep: Tick::from_seconds_f64(secs_p(cmd, p, "keepPause", 0.0, 2.0, 0.12)?),
    };
    let first = items[0];
    let settings = s.project.resolve_media(first).map(|(_, m, _)| crate::commands::default_seq_settings_for(m)).unwrap_or_default();
    let rate = settings.frame_rate;
    // the ranges of every item, in order
    let mut plan: Vec<(ItemId, rough_cut::Kept)> = Vec::new();
    let mut removed = rough_cut::Removed::default();
    let mut source = Tick::ZERO;
    for &item in &items {
        let (Some(t), Some(a)) = (s.project.transcripts.get(&item), s.project.speech.get(&item)) else { continue };
        let end = match s.project.item(item).map(|i| &i.kind) {
            Some(ItemKind::Media(m)) => m.duration(),
            _ => continue,
        };
        source += end;
        let (kept, r) = rough_cut::keep_ranges(t, Some(a), &o, rate, end);
        removed.pauses += r.pauses;
        removed.fillers += r.fillers;
        removed.retakes += r.retakes;
        removed.words += r.words;
        plan.extend(kept.into_iter().map(|k| (item, k)));
    }
    if plan.is_empty() {
        return Err(bad(cmd, "no speech to keep in these clips"));
    }
    let name = match str_p(p, "name").map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => n.chars().take(200).collect::<String>(),
        None => format!("{} (rough cut)", s.project.item(first).map(|i| i.name.clone()).unwrap_or_default()),
    };
    let n_audio = 1;
    let n0 = s.history.undo.len();
    let bin = s.project.root.parent_of(first).filter(|b| *b != s.project.root.id);
    let seq = s.edit("Rough Cut", |pr, st| {
        let id = pr.new_sequence(&name, settings, 1, n_audio, bin);
        st.active_sequence = Some(id);
        if !st.open_sequences.contains(&id) {
            st.open_sequences.push(id);
        }
        Ok(id)
    })?;
    let mut at = Tick::ZERO;
    let mut clips = Vec::new();
    let mut ranges = Vec::new();
    for (item, k) in &plan {
        let tg = s.targeting();
        let ids = crate::commands::place_item(s, *item, k.range, at, tg.video_dest, tg.audio_dest, false, "Rough Cut", None)?;
        at = s.active_sequence().map(|q| q.duration()).unwrap_or(at);
        clips.extend(ids.iter().map(|c| c.0));
        ranges.push(
            json!({"item": item.0, "start": secs(k.range.start), "end": secs(k.range.end()), "from": k.words.start, "to": k.words.end.saturating_sub(1)}),
        );
    }
    crate::clip_ops::collapse_history(s, n0, "Rough Cut");
    s.state.selection.clear();
    s.events.push(crate::Event::OpenSequence(seq));
    let duration = s.active_sequence().map(|q| q.duration()).unwrap_or_default();
    Ok(json!({
        "sequence": seq.0,
        "clips": clips,
        "ranges": ranges,
        "duration": secs(duration),
        "source": secs(source),
        "removed": {"pauses": removed.pauses, "fillers": removed.fillers, "retakes": removed.retakes, "words": removed.words, "seconds": secs((source - duration).max(Tick::ZERO))},
    }))
}

/// `sequence.speakerTrack`: every clip of the active sequence whose media has speaker labels is
/// coloured by the speaker who talks most in it (speaker 1 Iris, 2 Mango, 3 Forest, …).
fn speaker_track(s: &mut Session, _p: &Value) -> Result<Value> {
    const COLOURS: [Label; 6] = [Label::Iris, Label::Mango, Label::Forest, Label::Rose, Label::Teal, Label::Yellow];
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let mut set: Vec<(filmcraft_project::ClipId, Label, String)> = Vec::new();
    for tr in seq.video_tracks.iter().chain(seq.audio_tracks.iter()) {
        for it in &tr.items {
            let Some(t) = s.project.transcripts.get(&it.item) else { continue };
            if t.speakers.len() < 2 {
                continue;
            }
            let media = TimeRange::from_bounds(it.source_in, it.source_in + Tick((it.duration.0 as f64 * it.speed.max(0.0)) as i64));
            let mut talk = vec![0i64; t.speakers.len()];
            for wi in t.words_in(media) {
                if let Some(w) = t.words.get(wi)
                    && let Some(k) = w.speaker.and_then(|k| talk.get_mut(k as usize))
                {
                    *k += (w.end.min(media.end()) - w.start.max(media.start)).0.max(0);
                }
            }
            let Some((k, _)) = talk.iter().enumerate().filter(|(_, v)| **v > 0).max_by_key(|(_, v)| **v) else { continue };
            let name = t.speakers.get(k).map(|s| s.name.clone()).unwrap_or_default();
            set.push((it.id, COLOURS[k % COLOURS.len()], name));
        }
    }
    if set.is_empty() {
        return Ok(json!({"clips": []}));
    }
    let report: Vec<Value> = set.iter().map(|(c, l, n)| json!({"clip": c.0, "speaker": n, "label": format!("{l:?}")})).collect();
    s.edit_sequence("Label Clips by Speaker", move |q, _, _| {
        for (c, l, _) in &set {
            if let Some((_, it)) = q.find_item_mut(*c) {
                it.label = *l;
            }
        }
        Ok(())
    })?;
    Ok(json!({"clips": report}))
}

fn spec(
    id: &'static str,
    label: &'static str,
    menu: &'static [&'static str],
    params: &'static str,
    enabled: fn(&Session) -> std::result::Result<(), String>,
    run: fn(&mut Session, &Value) -> Result<Value>,
    journal: bool,
) -> CommandSpec {
    CommandSpec { id, label, menu, shortcut: None, params, enabled, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "media.understandSpeech",
            "Understand Speech…",
            &["Clip", "Speech"],
            r#"{"items":[id]?,"language":"en|ml|auto"?,"model":str?,"diarize":bool=true,"maxSpeakers":1..32=6,"faces":bool=true,"minPause":0.1..10=0.5,"fillers":[str]?,"wait":bool=false}"#,
            crate::transcript::can_transcribe,
            understand_speech,
            false,
        ),
        spec("media.inspectSpeech", "Inspect Speech Analysis", &[], r#"{"items":[id]?}"#, always, inspect, false),
        spec("transcript.removeRetakes", "Remove Retakes", &["Sequence", "Transcript"], "{}", has_speech_seq, remove_retakes, true),
        spec(
            "sequence.roughCut",
            "Rough Cut from Speech…",
            &["Sequence"],
            r#"{"items":[id]?,"removePauses":bool=true,"removeFillers":bool=true,"removeRetakes":bool=true,"removeHesitations":bool=false,"minPause":0.1..30=0.6,"keepPause":0..2=0.12,"name":str?}"#,
            has_speech,
            rough_cut,
            true,
        ),
        spec("sequence.speakerTrack", "Label Clips by Speaker", &["Sequence"], "{}", has_seq, speaker_track, true),
    ]
}

#[cfg(test)]
#[path = "speech_understand_tests.rs"]
mod tests;
