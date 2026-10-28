//! The voice lab's work (feature `neural-voices`): decoding the clips, the models, the files.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};

use filmcraft_tts::voice_lab::{self as vl, Span, Target, audition as au, signal};
use filmcraft_tts::{Params, Voice, clone};

use super::{AuditionParams, ChainStep, FindParams};
use crate::Services;
use crate::media_pool::MediaPool;

type R<T> = std::result::Result<T, String>;

/// The rate clips are analysed and auditioned at (IndicF5's; VoxCPM2 encodes at 16 kHz).
const RATE: u32 = 24_000;
/// The rate a promoted reference is read from its recording at (full band).
const FULL_RATE: u32 = 44_100;
/// The rate mix chains run at (a sequence's).
const MIX_RATE: u32 = 48_000;
/// Silence between joined references, and between the readings of a test set, seconds.
const JOIN_GAP: f64 = 0.3;
const LINES_GAP: f64 = 0.7;
/// The noise reduction of a cleaned reference whose signal to noise is under 30 dB.
const CLEAN_DENOISE_DB: f32 = 6.0;
/// Longest wait for the GPU, which another session may be speaking on.
const GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
/// Largest report read back.
const MAX_REPORT_BYTES: u64 = 64 << 20;
/// Files a folder walk looks at, and how deep.
const MAX_WALK: usize = 100_000;
const MAX_DEPTH: usize = 5;
/// Media file extensions the finder looks at.
const MEDIA_EXT: &[&str] = &["mov", "mp4", "m4v", "mkv", "webm", "avi", "mts", "m2ts", "mxf", "wav", "m4a", "mp3", "flac", "ogg", "opus", "aif", "aiff"];
/// Folders that hold FilmCraft's own sound, never the speaker's: narrations and their cache (the
/// cloned voice itself), generated effects, previews, the voices.
const NOT_SOURCES: &[&str] = &["Narrations", "Generated Audio", "FilmCraft Previews", "narration-cache", "voices", "voice-lab"];

fn status(prog: &filmcraft_export::Progress, line: &str) {
    *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = line.to_string();
}

fn cancelled(prog: &filmcraft_export::Progress) -> bool {
    prog.cancel.load(Ordering::Relaxed)
}

fn models() -> R<PathBuf> {
    crate::transcript::models_dir().ok_or_else(|| "no data directory for the models".to_string())
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The GPU, ours while the guard lives (a session speaking on it is waited for); without a place
/// for locks, unlocked.
fn gpu(what: &str) -> R<Option<filmcraft_locks::Guard>> {
    match filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, what, GPU_WAIT, &mut |_| {}) {
        Ok(g) => Ok(Some(g)),
        Err(e @ filmcraft_locks::LockError::Busy(..)) => Err(format!("{e}; try again when it is free (`filmcraft-cli locks`)")),
        Err(filmcraft_locks::LockError::Io(_)) => Ok(None),
    }
}

/// The lab's ears: OpenVoice's tone colour and Whisper (base: English and language id).
struct Ears {
    conv: Arc<filmcraft_tts::openvoice::Converter>,
    base: Option<Arc<dyn filmcraft_speech::Transcriber>>,
    models: PathBuf,
    ml: Option<Arc<dyn filmcraft_speech::Transcriber>>,
}

impl Ears {
    fn new(models: &Path, whisper: bool) -> R<Ears> {
        if !filmcraft_tts::catalog::clone_installed(models) {
            return Err("the voice lab compares voices with OpenVoice's tone colour: download it first (tts.downloadCloneModel)".into());
        }
        let conv = Arc::new(filmcraft_tts::openvoice::Converter::load(&filmcraft_tts::catalog::clone_package_dir(models)).map_err(|e| e.to_string())?);
        let base = if whisper {
            Some(
                filmcraft_speech::load_gpu_first(models, "whisper-base")
                    .map_err(|e| format!("the speech model whisper-base: {e} (transcript.downloadModel)"))?,
            )
        } else {
            None
        };
        Ok(Ears { conv, base, models: models.to_path_buf(), ml: None })
    }

    /// The tone colour of speech `x` (mono at `rate`).
    fn colour_of(&self, x: &[f32], rate: u32) -> R<Vec<f32>> {
        let y = filmcraft_audio_dsp::resample::resample_sinc(x, rate, filmcraft_tts::openvoice::RATE);
        self.conv.tone_colour(&y).map_err(|e| e.to_string())
    }

    /// The tone colour of the speech in `x` (its pauses dropped).
    fn speech_colour(&self, x: &[f32], rate: u32) -> R<Vec<f32>> {
        let speech = clone::speech_only(x, rate);
        self.colour_of(if speech.len() as f64 >= 1.5 * f64::from(rate) { &speech } else { x }, rate)
    }

    fn hear_en(&self, x: &[f32], rate: u32) -> R<String> {
        let t = self.base.as_ref().ok_or("no English speech model")?;
        let x16 = filmcraft_audio_dsp::resample::resample_sinc(x, rate, filmcraft_speech::SAMPLE_RATE);
        let mut each = |piece: &[f32]| t.hear(piece, "en").map_err(|e| filmcraft_tts::TtsError::Model(e.to_string()));
        filmcraft_tts::heard::hear_in_pieces(&x16, filmcraft_speech::SAMPLE_RATE, 25.0, &mut each).map_err(|e| e.to_string())
    }

    /// What the Malayalam speech model (on the GPU: the caller holds the lock) hears in `x`.
    fn hear_ml(&mut self, x: &[f32], rate: u32) -> R<String> {
        if self.ml.is_none() {
            let id = filmcraft_speech::models::MALAYALAM_MODEL;
            self.ml = Some(
                filmcraft_speech::load_on(&self.models, id, true).map_err(|e| format!("the Malayalam speech model {id}: {e} (transcript.downloadModel)"))?,
            );
        }
        let t = self.ml.clone().ok_or("no Malayalam speech model")?;
        let mut each = |piece: &[f32]| {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(piece, rate, filmcraft_speech::SAMPLE_RATE);
            t.hear(&x16, "ml").map_err(|e| filmcraft_tts::TtsError::Model(e.to_string()))
        };
        filmcraft_tts::heard::hear_in_pieces(x, rate, filmcraft_tts::heard::HEARD_SECONDS, &mut each).map_err(|e| e.to_string())
    }
}

impl vl::Models for Ears {
    fn colour(&mut self, x: &[f32], rate: u32) -> Result<Vec<f32>, filmcraft_tts::TtsError> {
        self.colour_of(x, rate).map_err(filmcraft_tts::TtsError::Model)
    }

    fn transcribe(&mut self, x: &[f32], rate: u32) -> Result<vl::Heard, filmcraft_tts::TtsError> {
        let t = self.base.as_ref().ok_or_else(|| filmcraft_tts::TtsError::Model("no speech model".into()))?;
        let x16 = filmcraft_audio_dsp::resample::resample_sinc(x, rate, filmcraft_speech::SAMPLE_RATE);
        let opts = filmcraft_speech::Options { language: None, diarize: false, ..Default::default() };
        let tr = t.transcribe(&x16, &opts, &mut |_, _| true).map_err(|e| filmcraft_tts::TtsError::Model(e.to_string()))?;
        Ok(vl::Heard {
            language: tr.language.clone(),
            words: tr.words.iter().map(|w| vl::Word { text: w.text.clone(), start: w.start.seconds(), end: w.end.seconds() }).collect(),
        })
    }
}

fn open(media: &MediaPool, services: &dyn Services, path: &str) -> R<filmcraft_media::SharedSource> {
    let src = media.open_file(path, services).map_err(|e| e.to_string())?;
    if !src.info().has_audio() {
        return Err("no sound".into());
    }
    Ok(src)
}

/// Mono audio of `src` from `start` for `len` seconds at `rate`, read 30 s at a time.
fn decode(src: &filmcraft_media::SharedSource, start: f64, len: f64, rate: u32) -> R<Vec<f32>> {
    if !(start.is_finite() && len.is_finite() && start >= 0.0 && len > 0.0) {
        return Err("bad times".into());
    }
    let total = (len.min(super::CAP_CLIP_SECONDS) * f64::from(rate)) as usize;
    let first = (start * f64::from(rate)) as i64;
    let chunk = 30 * rate as usize;
    let mut out: Vec<f32> = Vec::with_capacity(total.min(64 << 20));
    while out.len() < total {
        let n = chunk.min(total - out.len());
        let at = first.saturating_add(out.len() as i64);
        let buf = src.audio(at, n, rate).map_err(|e| e.to_string())?;
        let mono = filmcraft_speech::downmix(&buf.channels);
        if mono.is_empty() {
            break;
        }
        let got = mono.len().min(n);
        out.extend(mono.iter().take(got).map(|v| if v.is_finite() { *v } else { 0.0 }));
        if got < n {
            break;
        }
    }
    Ok(out)
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

/// A name as renders are compared: letters and digits only, lower case (Google Photos keeps
/// "I've" as "I_ve" in the file name and "I've" in its title).
fn render_key(name: &str) -> String {
    let stem = name.strip_suffix(".qa.json").unwrap_or(name);
    let stem = Path::new(stem).file_stem().map_or_else(|| stem.to_string(), |s| s.to_string_lossy().into_owned());
    stem.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// AutoCut's renders known in `dir`: the files its QA report (`<name>.qa.json`) sits beside.
fn renders_in(dir: &Path, out: &mut std::collections::HashSet<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.filter_map(|e| e.ok()).take(MAX_WALK) {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.ends_with(".qa.json") {
            out.insert(render_key(&n));
        }
    }
}

/// The title Google Photos gave a cached file (its `<file>.json` beside it), if any.
fn photos_title(path: &Path) -> Option<String> {
    let side = PathBuf::from(format!("{}.json", path.to_string_lossy()));
    if std::fs::metadata(&side).ok()?.len() > 64 * 1024 {
        return None;
    }
    let v: Value = serde_json::from_slice(&std::fs::read(side).ok()?).ok()?;
    v["title"].as_str().map(str::to_string)
}

/// `path` is FilmCraft's own sound (a narration, a generated effect, a voice) or one of AutoCut's
/// renders (a `.qa.json` beside it, or a copy of one: the same name as a render in `renders`,
/// such as a review build uploaded to Google Photos and cached back): never the speaker's
/// recording — a render's narration is the cloned voice itself.
fn not_a_source(path: &Path, renders: &std::collections::HashSet<String>) -> Option<&'static str> {
    if path.components().any(|c| NOT_SOURCES.iter().any(|n| c.as_os_str() == *n)) {
        return Some("FilmCraft's own sound (narrations, generated audio, voices)");
    }
    let stem = path.file_stem()?.to_string_lossy();
    if path.with_file_name(format!("{stem}.qa.json")).is_file() {
        return Some("an AutoCut render (it holds the cloned narration)");
    }
    let name = path.file_name()?.to_string_lossy();
    if renders.contains(&render_key(&name)) || photos_title(path).is_some_and(|t| renders.contains(&render_key(&t))) {
        return Some("a copy of an AutoCut render (it holds the cloned narration)");
    }
    None
}

fn is_media(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| MEDIA_EXT.contains(&e.to_ascii_lowercase().as_str()))
}

/// The media files of a project file: every `{"File": {"path": …}}` in it.
fn project_media(path: &Path, out: &mut Vec<PathBuf>) -> R<()> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() > 512 << 20 {
        return Err("too large for a project".into());
    }
    let v: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut stack = vec![(&v, 0usize)];
    let mut seen = 0usize;
    while let Some((v, depth)) = stack.pop() {
        seen += 1;
        if seen > 50_000_000 || depth > 256 {
            break;
        }
        match v {
            Value::Object(o) => {
                if let Some(p) = o.get("File").and_then(|f| f.get("path")).and_then(Value::as_str) {
                    out.push(PathBuf::from(p));
                }
                stack.extend(o.values().map(|x| (x, depth + 1)));
            }
            Value::Array(a) => stack.extend(a.iter().map(|x| (x, depth + 1))),
            _ => {}
        }
    }
    Ok(())
}

/// The files to look in: files as given, folders walked, projects' media; FilmCraft's own sound
/// and AutoCut's renders left out (noted in `skipped`).
fn gather(paths: &[String], max_files: usize, skipped: &mut Vec<Value>) -> Vec<String> {
    let mut found: Vec<PathBuf> = Vec::new();
    // the renders AutoCut made where the projects and folders given are (and in its default
    // folder): their copies elsewhere are skipped too
    let mut renders = std::collections::HashSet::new();
    if let Some(home) = std::env::var_os("HOME") {
        renders_in(&PathBuf::from(home).join("Movies").join("AutoCut"), &mut renders);
    }
    for p in paths {
        let path = PathBuf::from(p);
        let dir = if path.is_dir() { Some(path.as_path()) } else { path.parent() };
        if let Some(d) = dir {
            renders_in(d, &mut renders);
        }
    }
    for p in paths {
        let path = PathBuf::from(p);
        if path.is_dir() {
            let mut stack = vec![(path, 0usize)];
            let mut walked = 0usize;
            while let Some((d, depth)) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else { continue };
                for e in rd.filter_map(|e| e.ok()) {
                    walked += 1;
                    if walked > MAX_WALK {
                        break;
                    }
                    let q = e.path();
                    if e.file_name().to_string_lossy().starts_with('.') {
                        continue;
                    }
                    if q.is_dir() {
                        if depth < MAX_DEPTH {
                            stack.push((q, depth + 1));
                        }
                    } else if is_media(&q) {
                        found.push(q);
                    }
                }
            }
        } else if path.extension().is_some_and(|e| e == "fcproj") {
            if let Err(e) = project_media(&path, &mut found) {
                skipped.push(json!({"path": p, "reason": format!("project: {e}")}));
            }
        } else {
            found.push(path);
        }
    }
    found.sort();
    found.dedup();
    let mut out = Vec::new();
    for f in found {
        if let Some(why) = not_a_source(&f, &renders) {
            skipped.push(json!({"path": f.to_string_lossy(), "reason": why}));
            continue;
        }
        if out.len() >= max_files {
            skipped.push(json!({"path": f.to_string_lossy(), "reason": format!("over {max_files} files")}));
            continue;
        }
        out.push(f.to_string_lossy().into_owned());
    }
    out
}

/// `IMG_3337.MOV [56.1–80.8 s]` → the file name and where the stretch starts.
fn source_range(src: &str) -> Option<(String, f64)> {
    let open = src.rfind('[')?;
    let name = src.get(..open)?.trim().to_string();
    let inner = src.get(open + 1..)?;
    let a: String = inner.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let start = a.parse::<f64>().ok().filter(|v| v.is_finite())?;
    (!name.is_empty()).then_some((name, start))
}

/// Where the voice's current references are in their recordings (file name, span), for leaving
/// them out of the held-out speech; and a description.
fn current_ranges(models: &Path, p: &clone::Profile) -> (Vec<(String, Span)>, Value) {
    let mut ranges = Vec::new();
    let mut desc = json!({});
    if let (Some(n), Some((name, at))) = (&p.natural, source_range(&p.source)) {
        let sp = Span::new(at + n.from, at + n.to);
        ranges.push((name.clone(), sp));
        desc["voxcpm"] = json!({"source": p.source, "clip": name, "start": sp.start, "end": sp.end});
    }
    if let Ok(Some(r)) = clone::cross_reference(models, &p.id)
        && let Some((name, at)) = source_range(&r.source)
    {
        let sp = Span::new(at + r.from, at + r.to);
        ranges.push((name.clone(), sp));
        desc["indicf5"] = json!({"source": r.source, "clip": name, "start": sp.start, "end": sp.end, "transcript": r.transcript});
    }
    // the stretches are approximate (pauses were shortened): leave a margin
    (ranges.into_iter().map(|(n, s)| (n, s.widened(3.0))).collect(), desc)
}

/// The voice's reference audio, at [`RATE`]: its sample (the full-band copy when kept) and its
/// Malayalam clip.
fn reference_audio(models: &Path, p: &clone::Profile) -> R<Vec<f32>> {
    let voices = clone::voices_dir(models);
    let full = clone::full_sample_path(&voices, p).map_err(|e| e.to_string())?;
    let sample = if full.is_file() { full } else { clone::sample_path(&voices, p).ok_or("the voice has no sample (a timbre-only clone)")? };
    let (x, rate) = filmcraft_tts::wav::read(&sample).map_err(|e| e.to_string())?;
    let mut out = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, RATE);
    if let Ok(Some(r)) = clone::cross_reference(models, &p.id)
        && let Ok(c) = clone::cross_audio(models, &r)
    {
        out.extend(filmcraft_audio_dsp::resample::resample_sinc(&c, clone::CROSS_REFERENCE_RATE, RATE));
    }
    Ok(out)
}

fn out_dir(given: Option<&str>, models: &Path, p: &clone::Profile, what: &str) -> R<PathBuf> {
    let d = match given {
        Some(o) => PathBuf::from(o),
        None => models
            .parent()
            .unwrap_or(models)
            .join("voice-lab")
            .join(clone::slug(p.id.trim_start_matches(filmcraft_tts::CLONE_PREFIX)))
            .join(format!("{what}-{}", now())),
    };
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    Ok(d)
}

fn write_json(path: &Path, v: &Value) -> R<()> {
    let bytes = serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_json(path: &str) -> R<Value> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
    if meta.len() > MAX_REPORT_BYTES {
        return Err(format!("{path}: too large for a report"));
    }
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))
}

fn natural_profile(models: &Path, voice: &str) -> R<clone::Profile> {
    let p = clone::load_profile(&clone::voices_dir(models), voice).map_err(|e| e.to_string())?;
    if p.natural.is_none() {
        return Err(format!("{voice} is a timbre-only clone: the voice lab works on natural clones (VoxCPM)"));
    }
    Ok(p)
}

/// `voice.findReferences`.
pub(super) fn find(fp: &FindParams, media: &MediaPool, services: &dyn Services, prog: &filmcraft_export::Progress) -> R<Value> {
    let models = models()?;
    let profile = natural_profile(&models, &fp.voice)?;
    let mut ears = Ears::new(&models, true)?;
    status(prog, "measuring the voice's reference");
    let speaker = ears.speech_colour(&reference_audio(&models, &profile)?, RATE)?;
    let (current, current_desc) = current_ranges(&models, &profile);
    let out = out_dir(fp.out.as_deref(), &models, &profile, "refs")?;
    let mut skipped: Vec<Value> = Vec::new();
    let files = gather(&fp.paths, fp.max_files, &mut skipped);
    prog.total.store(files.len().max(1) as u64, Ordering::Relaxed);
    let settings = vl::FindSettings { min_similarity: fp.min_similarity, per_clip: 3 };
    let budget = fp.max_minutes * 60.0;
    let (mut scanned, mut scans) = (0.0f64, Vec::new());
    let t0 = web_time::Instant::now();
    for (i, f) in files.iter().enumerate() {
        if cancelled(prog) {
            return Err("cancelled".into());
        }
        prog.done.store(i as u64, Ordering::Relaxed);
        let src = match open(media, services, f) {
            Ok(s) => s,
            Err(e) => {
                skipped.push(json!({"path": f, "reason": e}));
                continue;
            }
        };
        let secs = src.info().duration.seconds();
        if !secs.is_finite() || secs < 3.0 {
            skipped.push(json!({"path": f, "reason": "shorter than 3 s"}));
            continue;
        }
        if secs > fp.max_clip_seconds {
            skipped.push(json!({"path": f, "reason": format!("{secs:.0} s, longer than {:.0} s (maxClipSeconds)", fp.max_clip_seconds)}));
            continue;
        }
        if scanned + secs > budget {
            skipped.push(json!({"path": f, "reason": format!("over the {:.0} minutes looked at (maxMinutes)", fp.max_minutes)}));
            continue;
        }
        status(prog, &format!("{}/{}: {} ({secs:.0} s)", i + 1, files.len(), file_name(f)));
        let x = match decode(&src, 0.0, secs, RATE) {
            Ok(x) => x,
            Err(e) => {
                skipped.push(json!({"path": f, "reason": format!("decoding: {e}")}));
                continue;
            }
        };
        scanned += secs;
        match vl::scan(f, &x, RATE, &speaker, &settings, &mut ears, &mut |l| status(prog, l)) {
            Ok(sc) => scans.push(sc),
            Err(e) => skipped.push(json!({"path": f, "reason": format!("scan: {e}")})),
        }
    }
    let all: Vec<vl::Candidate> = scans.iter().flat_map(|s| s.candidates.iter().cloned()).collect();
    let ranked = vl::rank(&all, fp.top, fp.per_clip);
    // the candidates' words (Malayalam by its own speech model, under the GPU lock) and audio
    let cands_dir = out.join("candidates");
    std::fs::create_dir_all(&cands_dir).map_err(|e| format!("{}: {e}", cands_dir.display()))?;
    let mut cands: Vec<Value> = Vec::new();
    let needs_ml = ranked.iter().any(|c| c.target == Target::Indicf5 && c.language == "ml");
    let _gpu = if needs_ml { gpu("voice lab: hearing the Malayalam references")? } else { None };
    for c in &ranked {
        if cancelled(prog) {
            return Err("cancelled".into());
        }
        status(prog, &format!("hearing {} ({} {:.1}–{:.1} s)", c.id, file_name(&c.clip), c.span.start, c.span.end));
        let audio = open(media, services, &c.clip).and_then(|s| decode(&s, c.span.start, c.span.seconds(), RATE))?;
        let wav = cands_dir.join(format!("{}.wav", c.id));
        filmcraft_tts::wav::write(&wav, &audio, RATE).map_err(|e| e.to_string())?;
        // the words in the language spoken: IndicF5 reads English words of an English clip
        // through the English front end
        let (transcript, note) = match c.target {
            Target::Indicf5 if c.language == "ml" => match ears.hear_ml(&audio, RATE) {
                Ok(t) if filmcraft_tts::malayalam::detect(&t) => (t, Value::Null),
                Ok(t) => (t, json!("the Malayalam speech model heard little Malayalam: check the words (or it is another language)")),
                Err(e) => (String::new(), json!(e)),
            },
            Target::Indicf5 | Target::Voxcpm => match ears.hear_en(&audio, RATE) {
                Ok(t) => (t, Value::Null),
                Err(e) => (String::new(), json!(e)),
            },
        };
        let mut v = serde_json::to_value(c).map_err(|e| e.to_string())?;
        v["wav"] = json!(wav);
        v["transcript"] = json!(transcript);
        v["clipName"] = json!(file_name(&c.clip));
        v["overlapsCurrent"] = json!(current.iter().any(|(n, s)| *n == file_name(&c.clip) && s.overlaps(&c.span)));
        if !note.is_null() {
            v["note"] = note;
        }
        cands.push(v);
    }
    drop(_gpu);
    // held-out genuine speech: by the speaker, never a candidate or a current reference
    let by_name: Vec<vl::ClipScan> = scans.iter().map(|s| vl::ClipScan { clip: file_name(&s.clip), ..s.clone() }).collect();
    let mut exclude: Vec<(String, Span)> = current.clone();
    exclude.extend(ranked.iter().map(|c| (file_name(&c.clip), c.span)));
    let held = vl::held_out(&by_name, &exclude, fp.min_similarity, fp.held_out_seconds);
    let path_of = |name: &str| scans.iter().find(|s| file_name(&s.clip) == name).map(|s| s.clip.clone()).unwrap_or_default();
    let held_json: Vec<Value> = held.iter().map(|(n, sp)| json!({"clip": path_of(n), "clipName": n, "start": sp.start, "end": sp.end})).collect();
    // every clip with speech, how close its speech came to the speaker (the speaker's first)
    let mut clips: Vec<Value> = scans
        .iter()
        .filter(|s| !s.chunks.is_empty())
        .map(|s| {
            let mut sims: Vec<f64> = s.chunks.iter().map(|c| c.similarity).collect();
            sims.sort_by(f64::total_cmp);
            json!({
                "clip": s.clip, "clipName": file_name(&s.clip), "seconds": s.seconds, "speakerSeconds": s.speaker_seconds, "language": s.language,
                "chunks": s.chunks.len(), "bestSimilarity": sims.last(), "medianSimilarity": sims.get(sims.len() / 2),
            })
        })
        .collect();
    clips.sort_by(|a, b| b["speakerSeconds"].as_f64().unwrap_or(0.0).total_cmp(&a["speakerSeconds"].as_f64().unwrap_or(0.0)));
    let others = scans.iter().filter(|s| s.speaker_seconds == 0.0 && !s.chunks.is_empty()).count();
    let with_speaker = scans.iter().filter(|s| s.speaker_seconds > 0.0).count();
    let speaker_seconds: f64 = scans.iter().map(|s| s.speaker_seconds).sum();
    let report_path = out.join("refs.json");
    let summary = format!(
        "{} candidates from {} clips with the speaker ({:.0} s of their speech) in {} clips ({:.0} min) looked at; {} held out",
        cands.len(),
        with_speaker,
        speaker_seconds,
        scans.len(),
        scanned / 60.0,
        held_json.len()
    );
    // the similarity of every chunk, for setting the threshold
    let mut sims: Vec<f64> = scans.iter().flat_map(|s| s.chunks.iter().map(|c| c.similarity)).collect();
    sims.sort_by(f64::total_cmp);
    let pct = |p: f64| sims.get(((sims.len().max(1) - 1) as f64 * p) as usize).copied();
    let report = json!({
        "voice": fp.voice,
        "summary": summary,
        "report": report_path,
        "out": out,
        "current": current_desc,
        "settings": {"minSimilarity": fp.min_similarity, "maxClipSeconds": fp.max_clip_seconds, "maxMinutes": fp.max_minutes, "top": fp.top, "perClip": fp.per_clip, "rate": RATE},
        "scanned": {
            "files": files.len(), "clips": scans.len(), "minutes": scanned / 60.0, "seconds": t0.elapsed().as_secs_f64(),
            "clipsWithTheSpeaker": with_speaker, "clipsWithOthersOnly": others, "speakerSeconds": speaker_seconds,
            "chunkSimilarity": {"chunks": sims.len(), "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9)},
            "skipped": skipped.iter().take(500).collect::<Vec<_>>(), "skippedCount": skipped.len(),
        },
        "clips": clips,
        "candidates": cands,
        "heldOut": held_json,
    });
    write_json(&report_path, &report)?;
    Ok(report)
}

/// A reference to audition: where its audio comes from and what it is.
#[derive(Clone, Debug)]
struct Entry {
    id: String,
    kind: &'static str,
    target: Target,
    /// The recording stretches it is made of (none: the voice's current reference).
    parts: Vec<(String, Span)>,
    denoise_db: Option<f32>,
    transcript: String,
    speaker_similarity: f64,
    source: String,
    /// The language spoken in it (`en`, `ml`, …; empty for the current reference).
    language: String,
}

fn entry_json(e: &Entry) -> Value {
    json!({
        "parts": e.parts.iter().map(|(c, s)| json!({"clip": c, "start": s.start, "end": s.end})).collect::<Vec<_>>(),
        "cleaned": e.denoise_db.map(|d| json!({"highPassHz": 70, "denoiseDb": d})),
        "transcript": e.transcript,
        "speakerSimilarity": e.speaker_similarity,
    })
}

fn entry_from(v: &Value, target: Target) -> Option<Entry> {
    let id = v["id"].as_str()?.to_string();
    let clip = v["clip"].as_str()?.to_string();
    let (a, b) = (v["span"]["start"].as_f64()?, v["span"]["end"].as_f64()?);
    Some(Entry {
        source: format!("{} [{a:.1}–{b:.1} s]", file_name(&clip)),
        id,
        kind: "candidate",
        target,
        parts: vec![(clip, Span::new(a, b))],
        denoise_db: None,
        transcript: v["transcript"].as_str().unwrap_or("").to_string(),
        speaker_similarity: v["similarity"].as_f64().unwrap_or(0.0),
        language: v["language"].as_str().unwrap_or("").to_string(),
    })
}

/// The audio of `e`'s parts at `rate`: joined, cleaned as it says.
fn entry_audio(e: &Entry, rate: u32, media: &MediaPool, services: &dyn Services) -> R<Vec<f32>> {
    let mut parts = Vec::new();
    for (clip, sp) in &e.parts {
        parts.push(open(media, services, clip).and_then(|s| decode(&s, sp.start, sp.seconds(), rate))?);
    }
    let refs: Vec<&[f32]> = parts.iter().map(Vec::as_slice).collect();
    let x = if refs.len() == 1 { refs.first().map(|r| r.to_vec()).unwrap_or_default() } else { signal::join(&refs, rate, JOIN_GAP) };
    Ok(match e.denoise_db {
        Some(d) => signal::clean(&x, rate, d),
        None => x,
    })
}

/// File-safe form of an entry id.
fn safe(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect()
}

/// Variants of the best candidates: two joined (VoxCPM up to 30 s, IndicF5 up to 15 s; from two
/// recordings when there are) and the best lightly cleaned.
fn variants(cands: &[Entry], quality: &dyn Fn(&str) -> f64) -> (Vec<Entry>, Vec<String>) {
    let mut out = Vec::new();
    let mut notes = Vec::new();
    let Some(best) = cands.first() else { return (out, notes) };
    let target = best.target;
    let max = if target == Target::Voxcpm { 30.0 } else { 15.0 };
    let secs = |e: &Entry| e.parts.iter().map(|(_, s)| s.seconds()).sum::<f64>();
    let pair = cands
        .iter()
        .enumerate()
        .flat_map(|(i, a)| cands.iter().skip(i + 1).map(move |b| (a, b)))
        .filter(|(a, b)| a.language == b.language && secs(a) + secs(b) + JOIN_GAP <= max)
        .min_by_key(|(a, b)| usize::from(a.parts.first().map(|p| &p.0) == b.parts.first().map(|p| &p.0)));
    match pair {
        Some((a, b)) => out.push(Entry {
            id: format!("{}+{}", a.id, b.id),
            kind: "multi",
            target,
            parts: a.parts.iter().chain(&b.parts).cloned().collect(),
            denoise_db: None,
            transcript: format!("{} {}", a.transcript, b.transcript).trim().to_string(),
            speaker_similarity: a.speaker_similarity.min(b.speaker_similarity),
            source: format!("{} + {}", a.source, b.source),
            language: a.language.clone(),
        }),
        None => notes.push(format!("{}: no two candidates in one language fit {max:.0} s together; no joined reference", target.id())),
    }
    let d = if quality(&best.id) < 30.0 { CLEAN_DENOISE_DB } else { 0.0 };
    out.push(Entry {
        id: format!("{} cleaned", best.id),
        kind: "cleaned",
        denoise_db: Some(d),
        source: format!("{} (high-pass 70 Hz, noise reduction {d} dB, trimmed)", best.source),
        ..best.clone()
    });
    (out, notes)
}

/// `voice.audition`.
pub(super) fn audition(ap: &AuditionParams, media: &MediaPool, services: &dyn Services, prog: &filmcraft_export::Progress) -> R<Value> {
    let models = models()?;
    let profile = natural_profile(&models, &ap.voice)?;
    let report = read_json(&ap.report)?;
    if report["voice"].as_str() != Some(ap.voice.as_str()) {
        return Err(format!("{} is a report for {}, not {}", ap.report, report["voice"], ap.voice));
    }
    let out = match &ap.out {
        Some(o) => PathBuf::from(o),
        None => Path::new(&ap.report).parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".")),
    };
    std::fs::create_dir_all(out.join("audition")).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut ears = Ears::new(&models, true)?;
    // the held-out genuine speech
    status(prog, "reading the held-out speech");
    let mut pool_audio: Vec<f32> = Vec::new();
    let mut from = Vec::new();
    for h in report["heldOut"].as_array().into_iter().flatten().take(200) {
        let (Some(clip), Some(a), Some(b)) = (h["clip"].as_str(), h["start"].as_f64(), h["end"].as_f64()) else { continue };
        if let Ok(x) = open(media, services, clip).and_then(|s| decode(&s, a, b - a, RATE)) {
            pool_audio.extend(clone::speech_only(&x, RATE));
            from.push(format!("{} [{a:.1}–{b:.1} s]", file_name(clip)));
        }
    }
    if (pool_audio.len() as f64) < 10.0 * f64::from(RATE) {
        return Err("the report has under 10 s of held-out speech of the speaker: find references in more of their clips".into());
    }
    let pool = au::Pool {
        colour: ears.colour_of(&pool_audio, RATE)?,
        pitch_hz: clone::median_pitch(&pool_audio, RATE),
        seconds: pool_audio.len() as f64 / f64::from(RATE),
        from,
    };
    let n = profile.natural.clone().ok_or("not a natural clone")?;
    let gender = clone::gender(&profile);
    let quality_of = |id: &str| {
        report["candidates"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|c| c["id"].as_str() == Some(id))
            .and_then(|c| c["quality"]["snrDb"].as_f64())
            .unwrap_or(40.0)
    };
    let mut scored: Vec<(au::Scored, Value)> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for engine in &ap.engines {
        let target = Target::from_id(engine).ok_or("unknown engine")?;
        let cands: Vec<Entry> = report["candidates"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|c| c["target"].as_str() == Some(target.id()))
            .filter_map(|c| entry_from(c, target))
            .collect();
        let mut entries: Vec<Entry> = vec![Entry {
            id: "current".into(),
            kind: "current",
            target,
            parts: Vec::new(),
            denoise_db: None,
            transcript: String::new(),
            speaker_similarity: 1.0,
            language: String::new(),
            source: match target {
                Target::Voxcpm => profile.source.clone(),
                Target::Indicf5 => clone::cross_reference(&models, &ap.voice).ok().flatten().map(|r| r.source).unwrap_or_default(),
            },
        }];
        let top: Vec<Entry> = cands.iter().take(ap.top).cloned().collect();
        entries.extend(top.iter().cloned());
        if ap.variants {
            let (v, nt) = variants(&cands, &quality_of);
            entries.extend(v);
            notes.extend(nt);
        }
        for e in &entries {
            if cancelled(prog) {
                return Err("cancelled".into());
            }
            status(prog, &format!("{}: {} ({})", target.id(), e.id, e.source));
            let r = audition_one(e, target, &n, gender, &models, &ap.voice, &profile, media, services, &mut ears, &pool, &out, ap.m4a, ap.takes, prog);
            match r {
                Ok((s, files)) => {
                    let mut extra = entry_json(e);
                    extra["files"] = files;
                    scored.push((s, extra));
                }
                Err(err) => {
                    let s = au::Scored {
                        id: e.id.clone(),
                        target: Some(target),
                        kind: e.kind.into(),
                        source: e.source.clone(),
                        score: f64::NEG_INFINITY,
                        error: Some(err.clone()),
                        ..Default::default()
                    };
                    notes.push(format!("{}: {err}", e.id));
                    scored.push((s, entry_json(e)));
                }
            }
        }
    }
    let mut ranked: Vec<au::Scored> = scored.iter().map(|(s, _)| s.clone()).collect();
    au::rank(&mut ranked);
    let entries: Vec<Value> = ranked
        .iter()
        .map(|s| {
            let mut v = serde_json::to_value(s).unwrap_or(Value::Null);
            if let Some((_, extra)) = scored.iter().find(|(x, _)| x.id == s.id && x.target == s.target)
                && let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object())
            {
                for (k, x) in e {
                    o.insert(k.clone(), x.clone());
                }
            }
            v
        })
        .collect();
    let best: Vec<Value> = Target::ALL
        .iter()
        .filter_map(|t| {
            let cur = ranked.iter().find(|s| s.target == Some(*t) && s.kind == "current");
            ranked.iter().find(|s| s.target == Some(*t)).map(|b| json!({"engine": t.id(), "best": b.id, "score": b.score, "current": cur.map(|c| c.score), "beatsCurrent": cur.is_some_and(|c| b.id != c.id)}))
        })
        .collect();
    let summary = best
        .iter()
        .map(|b| {
            format!(
                "{}: best {} ({})",
                b["engine"].as_str().unwrap_or(""),
                b["best"].as_str().unwrap_or(""),
                if b["beatsCurrent"] == true { "beats the current reference" } else { "the current reference is best" }
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let path = out.join("audition.json");
    let v = json!({
        "voice": ap.voice,
        "summary": summary,
        "report": path,
        "out": out,
        "refs": ap.report,
        "pool": pool,
        "lines": {"voxcpm": au::ENGLISH, "indicf5": au::MALAYALAM},
        "takes": ap.takes,
        "best": best,
        "entries": entries,
        "notes": notes,
        "scoring": "score = similarity to held-out genuine speech (OpenVoice tone colour) − 0.5 × CER − 0.5 × mean sentence-start CER − 0.02 × signal findings − 0.005 × |pitch difference in semitones|",
    });
    write_json(&path, &v)?;
    std::fs::write(out.join("README.txt"), readme(&v)).map_err(|e| e.to_string())?;
    Ok(v)
}

/// Audition one reference: speak the test lines, hear and score them, write the files.
#[allow(clippy::too_many_arguments)]
fn audition_one(
    e: &Entry,
    target: Target,
    n: &clone::Natural,
    gender: filmcraft_tts::Gender,
    models: &Path,
    voice_id: &str,
    profile: &clone::Profile,
    media: &MediaPool,
    services: &dyn Services,
    ears: &mut Ears,
    pool: &au::Pool,
    out: &Path,
    m4a: bool,
    readings: u32,
    prog: &filmcraft_export::Progress,
) -> R<(au::Scored, Value)> {
    // the reference, and the voice speaking from it
    let (reference, voice): (Vec<f32>, Box<dyn Voice>) = if e.kind == "current" {
        match target {
            Target::Voxcpm => {
                let voices = clone::voices_dir(models);
                let path = clone::sample_path(&voices, profile).ok_or("no voice sample")?;
                let (x, rate) = filmcraft_tts::wav::read(&path).map_err(|e| e.to_string())?;
                (filmcraft_audio_dsp::resample::resample_sinc(&x, rate, RATE), filmcraft_tts::voice_in(voice_id, Some(models)).map_err(|e| e.to_string())?)
            }
            Target::Indicf5 => {
                let r = clone::cross_reference(models, voice_id).map_err(|e| e.to_string())?.ok_or("the voice has no Malayalam reference")?;
                let x = clone::cross_audio(models, &r).map_err(|e| e.to_string())?;
                (
                    filmcraft_audio_dsp::resample::resample_sinc(&x, clone::CROSS_REFERENCE_RATE, RATE),
                    filmcraft_tts::voice_for(voice_id, filmcraft_tts::malayalam::TAG, Some(models)).map_err(|e| e.to_string())?,
                )
            }
        }
    } else {
        let x = entry_audio(e, RATE, media, services)?;
        match target {
            Target::Voxcpm => {
                let sample = clone::sample_from(&x, RATE).map_err(|e| e.to_string())?;
                let v = clone::audition_voice(models, &n.model, &sample.audio, RATE, n.cfg, n.timesteps).map_err(|e| e.to_string())?;
                (sample.audio, v)
            }
            Target::Indicf5 => {
                if !filmcraft_tts::malayalam::detect(&e.transcript) && !e.transcript.chars().any(char::is_alphanumeric) {
                    return Err("no words for the clip (IndicF5 needs them)".into());
                }
                let v = clone::audition_malayalam(models, &x, RATE, &e.transcript, gender).map_err(|e| e.to_string())?;
                (x, v)
            }
        }
    };
    let dir = out.join("audition").join(target.id()).join(safe(&e.id));
    std::fs::create_dir_all(&dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    let ref_wav = out.join("audition").join(target.id()).join(format!("{}.reference.wav", safe(&e.id)));
    filmcraft_tts::wav::write(&ref_wav, &reference, RATE).map_err(|e| e.to_string())?;
    let reference_similarity = vl::cosine(&ears.speech_colour(&reference, RATE)?, &pool.colour);
    // speak every line first (the voice takes the GPU per line), then hear them
    let lines = au::lines(target);
    let mut spoken: Vec<(usize, u32, filmcraft_tts::Audio)> = Vec::new();
    for take in 0..readings.max(1) {
        for (k, line) in lines.iter().enumerate() {
            if cancelled(prog) {
                return Err("cancelled".into());
            }
            status(prog, &format!("{}: {} line {} take {}", target.id(), e.id, k + 1, take + 1));
            let params = Params { take, ..Params::default() };
            spoken.push((k, take, voice.synthesize(line, &params).map_err(|e| e.to_string())?));
        }
    }
    drop(voice);
    let _gpu = if target == Target::Indicf5 { gpu("voice lab: hearing the Malayalam auditions")? } else { None };
    let mut takes = Vec::new();
    let mut joined: Vec<f32> = Vec::new();
    let mut joined_rate = 0u32;
    for (k, take, a) in &spoken {
        let line = lines.get(*k).copied().unwrap_or("");
        let heard = match target {
            Target::Voxcpm => ears.hear_en(&a.samples, a.sample_rate)?,
            Target::Indicf5 => ears.hear_ml(&a.samples, a.sample_rate)?,
        };
        let colour = ears.speech_colour(&a.samples, a.sample_rate)?;
        let mut t = au::take(k + 1, line, &heard, &a.samples, a.sample_rate, &colour, pool);
        t.take = *take;
        let wav = dir.join(format!("line{}-take{}.wav", k + 1, take + 1));
        filmcraft_tts::wav::write(&wav, &a.samples, a.sample_rate).map_err(|e| e.to_string())?;
        t.file = wav.to_string_lossy().into_owned();
        takes.push(t);
        // the set to listen to: every line's first reading
        if *take > 0 {
            continue;
        }
        if joined_rate == 0 {
            joined_rate = a.sample_rate;
        }
        let x = filmcraft_audio_dsp::resample::resample_sinc(&a.samples, a.sample_rate, joined_rate);
        if !joined.is_empty() {
            joined.resize(joined.len() + (LINES_GAP * f64::from(joined_rate)) as usize, 0.0);
        }
        joined.extend_from_slice(&x);
    }
    drop(_gpu);
    let all_wav = out.join("audition").join(target.id()).join(format!("{}.wav", safe(&e.id)));
    filmcraft_tts::wav::write(&all_wav, &joined, joined_rate.max(1)).map_err(|e| e.to_string())?;
    let mut files = json!({"reference": ref_wav, "readings": all_wav, "lines": dir});
    if m4a {
        let path = all_wav.with_extension("m4a");
        let p = path.to_string_lossy().into_owned();
        match filmcraft_export::encode_m4a(&p, &[joined], joined_rate.max(1), 96) {
            Ok(_) => files["readingsM4a"] = json!(p),
            Err(err) => files["m4aError"] = json!(err.to_string()),
        }
    }
    let base = au::Scored {
        id: e.id.clone(),
        target: Some(target),
        kind: e.kind.into(),
        source: e.source.clone(),
        reference_seconds: reference.len() as f64 / f64::from(RATE),
        reference_similarity,
        ..Default::default()
    };
    Ok((au::summarise(base, takes, pool), files))
}

/// `README.txt` of an audition: which file is which, best first.
fn readme(v: &Value) -> String {
    let mut s = String::new();
    s.push_str(&format!("Voice auditions for {}\n\n", v["voice"].as_str().unwrap_or("")));
    s.push_str(&format!(
        "Each reference read the same test lines (English for VoxCPM, Malayalam for IndicF5), {} reading(s) of each; the numbers are means over them,\n",
        v["takes"]
    ));
    s.push_str("and the files to listen to (readings) are every line's first reading.\n");
    s.push_str("Similarity is to held-out genuine speech of the speaker (OpenVoice tone colour, 1 = same);\n");
    s.push_str("CER is what the speech model heard against the line; start CER the first syllables of each sentence.\n\n");
    // the short list: per engine the current reference and the best three others, small files
    s.push_str("Listen to (M4A; the same readings as WAV beside them):\n");
    for t in Target::ALL {
        let rows: Vec<&Value> =
            v["entries"].as_array().into_iter().flatten().filter(|e| e["target"].as_str() == Some(t.id()) && e["error"].is_null()).collect();
        let current = rows.iter().find(|e| e["kind"] == "current");
        let best = rows.iter().filter(|e| e["kind"] != "current").take(3);
        for e in current.into_iter().chain(best) {
            let file = e["files"]["readingsM4a"].as_str().or_else(|| e["files"]["readings"].as_str()).unwrap_or("");
            let name = Path::new(file).strip_prefix(v["out"].as_str().unwrap_or("")).map_or_else(|_| file.to_string(), |p| p.to_string_lossy().into_owned());
            s.push_str(&format!("  {:8} #{} {:24} {}  ({})\n", t.id(), e["rank"], e["id"].as_str().unwrap_or(""), name, e["source"].as_str().unwrap_or("")));
        }
    }
    s.push('\n');
    for t in Target::ALL {
        let rows: Vec<&Value> = v["entries"].as_array().into_iter().flatten().filter(|e| e["target"].as_str() == Some(t.id())).collect();
        if rows.is_empty() {
            continue;
        }
        s.push_str(&format!("== {} ==\n", if t == Target::Voxcpm { "VoxCPM2 (English)" } else { "IndicF5 (Malayalam)" }));
        for e in rows {
            let num = |k: &str| e[k].as_f64().map_or("-".to_string(), |x| format!("{x:.3}"));
            s.push_str(&format!(
                "#{} {} [{}] {}\n   similarity {}  CER {}  start CER {}  findings {}  score {}\n",
                e["rank"],
                e["id"].as_str().unwrap_or(""),
                e["kind"].as_str().unwrap_or(""),
                e["source"].as_str().unwrap_or(""),
                num("similarity"),
                num("cer"),
                num("startCer"),
                e["artefacts"],
                num("score"),
            ));
            if let Some(err) = e["error"].as_str() {
                s.push_str(&format!("   not auditioned: {err}\n"));
            }
            for k in ["readingsM4a", "readings", "reference"] {
                if let Some(f) = e["files"][k].as_str() {
                    s.push_str(&format!("   {k}: {f}\n"));
                }
            }
        }
        s.push('\n');
    }
    s.push_str("Keep a reference with `filmcraft-cli voice promote --voice <voice> --report <audition.json> <id>`;\n");
    s.push_str("the old one is kept (`voice versions`, `voice rollback`).\n");
    s
}

/// `voice.promote`.
pub(super) fn promote(voice: &str, report: &str, candidate: &str, words: Option<&str>, min: f64, media: &MediaPool, services: &dyn Services) -> R<Value> {
    let models = models()?;
    let _ = natural_profile(&models, voice)?;
    let r = read_json(report)?;
    if r["voice"].as_str() != Some(voice) {
        return Err(format!("{report} is a report for {}, not {voice}", r["voice"]));
    }
    let list = r["entries"].as_array().or_else(|| r["candidates"].as_array()).ok_or("the report has no candidates")?;
    let v = list.iter().find(|e| e["id"].as_str() == Some(candidate)).ok_or_else(|| format!("no candidate `{candidate}` in {report}"))?;
    if v["kind"].as_str() == Some("current") {
        return Err(format!("`{candidate}` is the voice's reference already"));
    }
    if v["error"].is_string() {
        return Err(format!("`{candidate}` could not be auditioned: {}", v["error"]));
    }
    let target = v["target"].as_str().and_then(Target::from_id).ok_or("the candidate has no engine")?;
    // the parts: an audition entry lists them; a found candidate is one stretch
    let parts: Vec<(String, Span)> = match v["parts"].as_array() {
        Some(a) => a.iter().filter_map(|p| Some((p["clip"].as_str()?.to_string(), Span::new(p["start"].as_f64()?, p["end"].as_f64()?)))).collect(),
        None => entry_from(v, target).map(|e| e.parts).unwrap_or_default(),
    };
    if parts.is_empty() {
        return Err("the candidate names no recording".into());
    }
    let similarity = v["speakerSimilarity"].as_f64().or_else(|| v["similarity"].as_f64()).unwrap_or(0.0);
    if similarity < min {
        return Err(format!(
            "`{candidate}` is not clearly the voice's speaker (similarity {similarity:.3}, at least {min} wanted): the consent given covers only their own voice"
        ));
    }
    let transcript = words.map(str::to_string).unwrap_or_else(|| v["transcript"].as_str().unwrap_or("").to_string());
    let denoise = v["cleaned"]["denoiseDb"].as_f64().map(|d| d as f32);
    let e = Entry {
        id: candidate.into(),
        kind: "candidate",
        target,
        parts: parts.clone(),
        denoise_db: denoise,
        transcript: transcript.clone(),
        speaker_similarity: similarity,
        source: String::new(),
        language: String::new(),
    };
    let audio = entry_audio(&e, FULL_RATE, media, services)?;
    let source = parts.iter().map(|(c, s)| format!("{} [{:.1}–{:.1} s]", file_name(c), s.start, s.end)).collect::<Vec<_>>().join(" + ");
    let (first, last) = (parts.first().map_or(0.0, |p| p.1.start), parts.last().map_or(0.0, |p| p.1.end));
    let score = v["score"].as_f64().map(|s| format!(", audition score {s:.3}")).unwrap_or_default();
    let note = format!("{candidate} of {report}: speaker similarity {similarity:.3}{score}");
    let to = if parts.len() == 1 { last } else { first + audio.len() as f64 / f64::from(FULL_RATE) };
    let nr = clone::NewReference {
        part: if target == Target::Voxcpm { clone::Part::Sample } else { clone::Part::Malayalam },
        audio: &audio,
        rate: FULL_RATE,
        source: &source,
        from: first,
        to,
        transcript: Some(transcript.as_str()).filter(|t| !t.trim().is_empty()),
        note: &note,
    };
    let kept = clone::promote(&models, voice, &nr).map_err(|e| e.to_string())?;
    let p = clone::load_profile(&clone::voices_dir(&models), voice).map_err(|e| e.to_string())?;
    Ok(
        json!({"voice": voice, "promoted": candidate, "engine": target.id(), "source": source, "revision": p.revision, "kept": kept, "note": "the earlier reference is kept: voice.rollback {\"version\": …} brings it back"}),
    )
}

/// Run `chain` over mono `x` at `rate` as the mixer runs track inserts (2 channels, the project
/// effects' own DSP and settings), latency compensated.
fn run_chain(chain: &[ChainStep], x: &[f32], rate: u32) -> R<Vec<f32>> {
    let mut l = x.to_vec();
    for st in chain {
        let def = filmcraft_project::find_effect(&st.effect).ok_or_else(|| format!("no effect `{}`", st.effect))?;
        let mut inst = def.instance();
        for (k, v) in &st.params {
            let par = inst.param_mut(k).ok_or_else(|| format!("{}: no parameter `{k}`", st.effect))?;
            par.value = match (&par.value, v) {
                (filmcraft_project::ParamValue::Bool(_), Value::Bool(b)) => filmcraft_project::ParamValue::Bool(*b),
                (filmcraft_project::ParamValue::Choice(_), v) => filmcraft_project::ParamValue::Choice(v.as_u64().unwrap_or(0).min(u64::from(u32::MAX)) as u32),
                (_, v) => {
                    filmcraft_project::ParamValue::Float(v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| format!("{}: `{k}` needs a number", st.effect))?)
                }
            };
        }
        let mut fx = filmcraft_render::audio_fx::configured(&inst, filmcraft_time::Tick::ZERO, rate)
            .ok_or_else(|| format!("`{}` has no sound processing", st.effect))?;
        let lat = fx.latency();
        let n = l.len();
        l.resize(n + lat, 0.0);
        let mut r = l.clone();
        for (a, b) in l.chunks_mut(1024).zip(r.chunks_mut(1024)) {
            fx.process(&mut [a, b]);
        }
        l.drain(..lat.min(l.len()));
        l.truncate(n);
    }
    Ok(l.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect())
}

fn read_wav(path: &str, rate: u32) -> R<Vec<f32>> {
    let (x, r) = filmcraft_tts::wav::read(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    Ok(filmcraft_audio_dsp::resample::resample_sinc(&x, r, rate))
}

/// `voice.mixCheck`.
pub(super) fn mix_check(
    takes: &[String],
    genuine: &[String],
    report: Option<&str>,
    chains: &[(String, Vec<ChainStep>)],
    lufs: f64,
    media: &MediaPool,
    services: &dyn Services,
) -> R<Value> {
    let models = models()?;
    let ears = Ears::new(&models, false)?;
    let mut voice: Vec<Vec<f32>> = takes.iter().map(|t| read_wav(t, MIX_RATE)).collect::<R<_>>()?;
    let mut real: Vec<Vec<f32>> = genuine.iter().map(|g| read_wav(g, MIX_RATE)).collect::<R<_>>()?;
    if let Some(rp) = report {
        let r = read_json(rp)?;
        for h in r["heldOut"].as_array().into_iter().flatten().take(100) {
            let (Some(clip), Some(a), Some(b)) = (h["clip"].as_str(), h["start"].as_f64(), h["end"].as_f64()) else { continue };
            if let Ok(x) = open(media, services, clip).and_then(|s| decode(&s, a, b - a, MIX_RATE)) {
                real.push(x);
            }
        }
    }
    // the genuine speech in two halves: one is what the voice is compared with, the other is
    // compared with it too (genuine speech is never compared with itself)
    if real.len() == 1 {
        let x = real.remove(0);
        let mid = x.len() / 2;
        real = vec![x.get(..mid).unwrap_or(&[]).to_vec(), x.get(mid..).unwrap_or(&[]).to_vec()];
    }
    let pool: Vec<Vec<f32>> = real.iter().step_by(2).cloned().collect();
    let test: Vec<Vec<f32>> = real.iter().skip(1).step_by(2).cloned().collect();
    let secs = |v: &[Vec<f32>]| v.iter().map(|x| x.len()).sum::<usize>() as f64 / f64::from(MIX_RATE);
    if secs(&pool) < 5.0 || secs(&test) < 5.0 {
        return Err("too little genuine speech (at least 10 s, in two or more clips) to compare with".into());
    }
    let level = |x: &mut Vec<f32>| signal::normalise(x, MIX_RATE, lufs);
    voice.iter_mut().for_each(level);
    let mut pool_l: Vec<Vec<f32>> = pool;
    pool_l.iter_mut().for_each(level);
    let mut test_l: Vec<Vec<f32>> = test;
    test_l.iter_mut().for_each(level);
    let concat = |v: &[Vec<f32>]| v.iter().flat_map(|x| x.iter().copied()).collect::<Vec<f32>>();
    let pool_raw = ears.speech_colour(&concat(&pool_l), MIX_RATE)?;
    let colours = |v: &[Vec<f32>]| v.iter().map(|x| ears.speech_colour(x, MIX_RATE)).collect::<R<Vec<_>>>();
    let mean_sim = |cs: &[Vec<f32>], to: &[f32]| cs.iter().map(|c| vl::cosine(c, to)).sum::<f64>() / cs.len().max(1) as f64;
    let balance = |v: &[Vec<f32>]| signal::balance(&concat(v), MIX_RATE);
    let voice_raw = colours(&voice)?;
    let test_raw = colours(&test_l)?;
    let mut rows = Vec::new();
    for (name, chain) in chains {
        let mixed = |v: &[Vec<f32>]| -> R<Vec<Vec<f32>>> {
            v.iter()
                .map(|x| {
                    let mut y = run_chain(chain, x, MIX_RATE)?;
                    // the fader makes up the level the chain took
                    signal::normalise(&mut y, MIX_RATE, lufs);
                    Ok(y)
                })
                .collect()
        };
        let (vm, tm, pm) = (mixed(&voice)?, mixed(&test_l)?, mixed(&pool_l)?);
        let pool_mixed = ears.speech_colour(&concat(&pm), MIX_RATE)?;
        let (vc, tc) = (colours(&vm)?, colours(&tm)?);
        let peak = vm.iter().flat_map(|x| x.iter()).fold(0f32, |m, v| m.max(v.abs()));
        rows.push(json!({
            "chain": name,
            "voice": {
                "before": mean_sim(&voice_raw, &pool_raw),
                "after": mean_sim(&vc, &pool_raw),
                "afterAgainstMixedGenuine": mean_sim(&vc, &pool_mixed),
                "balanceBefore": balance(&voice),
                "balanceAfter": balance(&vm),
                "peakDb": 20.0 * f64::from(peak.max(1e-6)).log10(),
            },
            "genuine": {
                "before": mean_sim(&test_raw, &pool_raw),
                "after": mean_sim(&tc, &pool_raw),
                "balanceBefore": balance(&test_l),
                "balanceAfter": balance(&tm),
            },
        }));
    }
    Ok(json!({
        "summary": format!("{} chains over {} readings and {:.0} s of genuine speech", rows.len(), voice.len(), secs(&pool_l) + secs(&test_l)),
        "takes": takes,
        "genuineSeconds": {"compareWith": secs(&pool_l), "test": secs(&test_l)},
        "lufs": lufs,
        "chains": rows,
        "measure": "similarity: OpenVoice tone colour cosine to the genuine speech (raw), every signal at the same loudness; balance: energy below 120 Hz (sub), 1–4 kHz (presence), and the 1–4 kHz octaves over the 125–250 Hz ones (tiltDb)",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_name_their_recording_and_start() {
        assert_eq!(source_range("IMG_3337.MOV [56.1–80.8 s]"), Some(("IMG_3337.MOV".into(), 56.1)));
        assert_eq!(source_range("IMG_3337.MOV [0.0–120.0 s]"), Some(("IMG_3337.MOV".into(), 0.0)));
        assert_eq!(source_range("a b.mov [3-5 s]"), Some(("a b.mov".into(), 3.0)));
        for bad in ["", "[1–2 s]", "x.mov", "x.mov [", "x.mov [abc]", "x.mov [nan–1 s]"] {
            assert_eq!(source_range(bad), None, "{bad}");
        }
    }

    #[test]
    fn filmcraft_sound_and_renders_are_not_sources() {
        let d = std::env::temp_dir().join(format!("filmcraft-voice-lab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("Narrations")).unwrap();
        std::fs::write(d.join("show.mp4"), b"x").unwrap();
        std::fs::write(d.join("show.qa.json"), b"{}").unwrap();
        std::fs::write(d.join("IMG_1.MOV"), b"x").unwrap();
        std::fs::write(d.join("notes.txt"), b"x").unwrap();
        std::fs::write(d.join("Narrations").join("line.wav"), b"x").unwrap();
        // a review build uploaded to Google Photos and cached back under another folder
        std::fs::create_dir_all(d.join("cache").join("AG_1")).unwrap();
        std::fs::write(d.join("cache").join("AG_1").join("show v2.mp4"), b"x").unwrap();
        std::fs::write(d.join("cache").join("AG_1").join("show v2.mp4.json"), br#"{"title": "show v2.mp4"}"#).unwrap();
        std::fs::write(d.join("cache").join("AG_1").join("Every Robot I_ve Met.mp4"), b"x").unwrap();
        std::fs::write(d.join("cache").join("AG_1").join("Every Robot I_ve Met.mp4.json"), br#"{"title": "Every Robot I've Met.mp4"}"#).unwrap();
        std::fs::write(d.join("show v2.qa.json"), b"{}").unwrap();
        std::fs::write(d.join("Every Robot I've Met.qa.json"), b"{}").unwrap();
        let proj = d.join("p.fcproj");
        let media = d.join("IMG_2.MOV");
        std::fs::write(&media, b"x").unwrap();
        std::fs::write(&proj, serde_json::to_vec(&json!({"items": [{"kind": {"Media": {"media": {"File": {"path": media}}}}}]})).unwrap()).unwrap();
        let mut skipped = Vec::new();
        let got = gather(&[d.to_string_lossy().into_owned(), proj.to_string_lossy().into_owned(), "/no/such/file.mov".into()], 100, &mut skipped);
        let names: Vec<String> = got.iter().map(|p| file_name(p)).collect();
        assert_eq!(names, ["file.mov", "IMG_1.MOV", "IMG_2.MOV"], "{got:?}");
        assert_eq!(skipped.len(), 4, "{skipped:?}");
        assert!(skipped.iter().filter(|s| s["reason"].as_str().unwrap().contains("copy of an AutoCut render")).count() == 2, "{skipped:?}");
        assert_eq!(render_key("Every Robot I've Met - Pro.qa.json"), render_key("Every Robot I_ve Met - Pro.mp4"));
        let mut skipped = Vec::new();
        assert_eq!(gather(&[d.to_string_lossy().into_owned()], 1, &mut skipped).len(), 1);
        std::fs::write(&proj, b"not json").unwrap();
        let mut skipped = Vec::new();
        let _ = gather(&[proj.to_string_lossy().into_owned()], 10, &mut skipped);
        assert_eq!(skipped.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_mix_chain_runs_as_the_mixer_does() {
        let rate = MIX_RATE;
        let x: Vec<f32> = (0..rate).map(|i| 0.2 * (std::f32::consts::TAU * 60.0 * i as f32 / rate as f32).sin()).collect();
        let hp =
            vec![ChainStep { effect: "parametric_eq".into(), params: json!({"hp_on": true, "hp_freq": 200.0, "hp_slope": 1}).as_object().unwrap().clone() }];
        let y = run_chain(&hp, &x, rate).unwrap();
        assert_eq!(y.len(), x.len());
        let rms = |v: &[f32]| (v.iter().skip(rate as usize / 2).map(|s| s * s).sum::<f32>() / (v.len() / 2) as f32).sqrt();
        assert!(rms(&y) < 0.2 * rms(&x), "60 Hz under a 200 Hz high-pass: {} vs {}", rms(&y), rms(&x));
        let bad = vec![ChainStep { effect: "parametric_eq".into(), params: json!({"nope": 1}).as_object().unwrap().clone() }];
        assert!(run_chain(&bad, &x, rate).is_err());
        let nan = vec![ChainStep { effect: "parametric_eq".into(), params: json!({"hp_freq": "x"}).as_object().unwrap().clone() }];
        assert!(run_chain(&nan, &x, rate).is_err());
        assert!(run_chain(&[], &[], rate).unwrap().is_empty());
    }

    #[test]
    fn the_readme_lists_the_current_reference_and_the_best_first() {
        let e = |id: &str, kind: &str, t: &str, rank: u32| {
            json!({"id": id, "kind": kind, "target": t, "rank": rank, "source": "IMG_1.MOV [0.0–10.0 s]", "similarity": 0.9, "cer": 0.05, "startCer": 0.02, "artefacts": 0, "score": 0.8,
                   "files": {"readingsM4a": format!("/out/audition/{t}/{id}.m4a"), "readings": format!("/out/audition/{t}/{id}.wav")}})
        };
        let v = json!({"voice": "clone:me", "out": "/out", "takes": 2, "entries": [
            e("voxcpm-2", "candidate", "voxcpm", 1), e("current", "current", "voxcpm", 2), e("voxcpm-1", "candidate", "voxcpm", 3),
            {"id": "voxcpm-3", "kind": "candidate", "target": "voxcpm", "rank": 4, "error": "no words"},
            e("current", "current", "indicf5", 1),
        ]});
        let r = readme(&v);
        let listen: Vec<&str> = r.lines().skip_while(|l| !l.starts_with("Listen to")).skip(1).take_while(|l| !l.is_empty()).collect();
        assert_eq!(listen.len(), 4, "{r}");
        assert!(listen[0].contains("current") && listen[0].contains("audition/voxcpm/current.m4a"), "{r}");
        assert!(listen[1].contains("voxcpm-2") && listen[2].contains("voxcpm-1") && listen[3].contains("indicf5"), "{r}");
        assert!(r.contains("not auditioned: no words"), "{r}");
        assert!(r.contains("2 reading(s)"), "{r}");
    }

    #[test]
    fn variants_join_two_recordings_and_clean_the_best() {
        let e = |id: &str, clip: &str, len: f64| Entry {
            id: id.into(),
            kind: "candidate",
            target: Target::Voxcpm,
            parts: vec![(clip.into(), Span::new(0.0, len))],
            denoise_db: None,
            transcript: id.into(),
            speaker_similarity: 0.9,
            source: id.into(),
            language: "en".into(),
        };
        let cands = vec![e("a", "x", 14.0), e("b", "x", 12.0), e("c", "y", 13.0)];
        let (v, notes) = variants(&cands, &|_| 20.0);
        assert!(notes.is_empty());
        assert_eq!(v.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["a+c", "a cleaned"], "two recordings preferred");
        assert_eq!(v[1].denoise_db, Some(CLEAN_DENOISE_DB));
        let (v, notes) = variants(&[e("a", "x", 25.0), e("b", "y", 20.0)], &|_| 40.0);
        assert_eq!((v.len(), notes.len()), (1, 1));
        assert_eq!(v[0].denoise_db, Some(0.0));
        assert!(variants(&[], &|_| 0.0).0.is_empty());
        assert_eq!(safe("voxcpm-1+voxcpm-2 cleaned"), "voxcpm-1_voxcpm-2_cleaned");
    }
}

/// The style finder and the prosody measure.
#[path = "voice_lab_styles.rs"]
pub(super) mod styles;
