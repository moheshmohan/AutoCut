//! The engine's showcase jobs (`autocut.analyze`, `autocut.showcasePlan`,
//! `autocut.makeShowcase`, `autocut.showcaseBuild`; see `filmcraft_engine::autocut`): this
//! crate's analysis, grouping, voice-over and build, run for the app's Showcase panel, the CLI
//! and MCP agents alike. Hosts install [`engine_host`] as the session's `autocut.host`; tests
//! install [`ShowcaseHost::with_models`] with scripted models.
//!
//! A job runs on its own thread with a small async runtime, and every engine command goes over
//! the session's command port ([`PortExecutor`]), so the UI stays responsive and every edit is an
//! ordinary command. Two kinds of command would hold up whoever runs the port, so they don't run
//! on it as asked: a long command with `wait: true` (an export, an upload) is started without
//! waiting and its job polled, and a narration line (`tts.create`) is synthesized first as a
//! background job (`tts.render`), so placing it takes the cached audio.
//!
//! With `album`, an analysis takes its clips from a Google Photos album ([`crate::album`]): the
//! host's [`AlbumSource`] picks all of it (after the subject is known, so a prompt that cannot
//! be read fails before anything is picked), the files are imported over the port (reusing
//! what the project already holds), and only those clips are analysed.
//!
//! A reviewed plan ([`PlanDoc`]) is the CLI's `plan.json` (`entries` and `script`) plus the
//! review: per entry `include` and `mergePrevious` ([`ReviewEntry`]), and the title, voice and
//! mood. [`resolve`] turns the review into what the build shows.

use std::collections::HashSet;
use std::future::Future;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};
use std::task::Poll;
use std::time::Duration;

use filmcraft_engine::autocut::{AutoCutHost, CommandPort, Task, TaskKind};
use filmcraft_engine::export::Progress;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::album::{self, AlbumPick, AlbumSource};
use crate::analysis_cache::AnalysisCache;
use crate::analyze::{self, ClipFacts};
use crate::config::Config;
use crate::llm::{ChatModel, JsonFuture};
use crate::ollama::Ollama;
use crate::showcase::{self, Entry, Script, Shot};
use crate::{AutoCutError, ExecFuture, Executor, RenderFuture};

/// Shots an entry keeps when others are merged into it (as many as a grouped entry shows).
const MERGED_SHOTS: usize = 3;
/// Engine commands that block until their job is done when called with `wait: true` (as
/// `filmcraft_automation::long_job::LONG_COMMANDS`, plus `tts.render`).
const LONG_COMMANDS: &[&str] = &[
    "file.exportMedia",
    "tts.cloneVoice",
    "tts.downloadVoices",
    "tts.downloadCloneModel",
    "photos.upload",
    "tts.render",
    "sequence.autoReframe",
    "clip.autoReframe",
    "youtube.upload",
    "media.understandSpeech",
    "media.analyzeScenes",
    "media.checkText",
    "scenes.downloadModel",
    "clip.autoRedact",
    "sequence.autoRedact",
    "redact.downloadModels",
];
/// How often a long command's job is looked at.
const POLL: Duration = Duration::from_millis(if cfg!(test) { 20 } else { 100 });

/// One entry of a reviewed plan: a [`showcase::Entry`] and the review's choices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewEntry {
    #[serde(flatten)]
    pub entry: Entry,
    /// Shown in the video.
    #[serde(default = "yes")]
    pub include: bool,
    /// Its clips show the robot of the included entry before it: merged into that one.
    #[serde(default, rename = "mergePrevious")]
    pub merge_previous: bool,
}

fn yes() -> bool {
    true
}

impl ReviewEntry {
    pub fn new(entry: Entry) -> Self {
        Self { entry, include: true, merge_previous: false }
    }
}

/// A reviewed showcase plan (`autocut.showcasePlan` writes one, `autocut.showcaseBuild` builds
/// one); every field but `entries` may be missing, as in the CLI's `plan.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanDoc {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    /// Narration voice (`clone:<name>` for a cloned one); none or empty: no narration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mood: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_coverage: Option<f64>,
    pub entries: Vec<ReviewEntry>,
    /// The voice-over; `lines` follows `entries` one to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<Script>,
    /// The look (`style`, `timeline`, `events`, `transitions`, `loudness`, `mix`, `embedCaptions`,
    /// `soundfont`: [`crate::look`]), kept as written and checked when the plan is built.
    #[serde(flatten)]
    pub look: serde_json::Map<String, Value>,
}

/// `Some(trimmed)` unless empty.
fn non_empty(s: &Option<String>) -> Option<String> {
    s.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// The title a showcase of `subject` gets when none is given: "Every Robot I've Filmed".
pub fn default_title(subject: &str) -> String {
    let words: Vec<String> = subject
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
        })
        .collect();
    format!("Every {} I've Filmed", words.join(" "))
}

/// A line for an entry the script has none for.
fn plain_line(e: &Entry) -> String {
    showcase::template_line(e)
}

/// About how many seconds a shot shows the subject: its length times the share of its sampled
/// moments that show it (all of it without samples).
fn in_view(s: &Shot) -> f64 {
    let (seen, unseen) = (s.seen.len() as f64, s.unseen.len() as f64);
    let share = if seen + unseen > 0.0 { seen / (seen + unseen) } else { 1.0 };
    let v = s.clip_seconds.max(0.0) * share;
    if v.is_finite() { v } else { 0.0 }
}

/// The (up to) `n` shots longest in view, in their order.
fn keep_longest(shots: Vec<Shot>, n: usize) -> Vec<Shot> {
    if shots.len() <= n {
        return shots;
    }
    let mut ranked: Vec<(usize, f64)> = shots.iter().enumerate().map(|(i, s)| (i, in_view(s))).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let keep: HashSet<usize> = ranked.into_iter().take(n).map(|(i, _)| i).collect();
    shots.into_iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, s)| s).collect()
}

/// What a review shows: the included entries, each one marked `mergePrevious` folded into the
/// included entry before it (its shots added, the [`MERGED_SHOTS`] longest in view kept), numbered
/// from 1 — and for each, the index of the review entry it comes from, whose script line it
/// speaks.
pub fn resolve(review: &[ReviewEntry]) -> (Vec<Entry>, Vec<usize>) {
    let mut out: Vec<(usize, Entry, bool)> = Vec::new();
    for (i, r) in review.iter().enumerate() {
        if !r.include {
            continue;
        }
        match out.last_mut() {
            Some((_, prev, merged)) if r.merge_previous => {
                prev.shots.extend(r.entry.shots.iter().cloned());
                *merged = true;
            }
            _ => out.push((i, r.entry.clone(), false)),
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(k, (i, mut e, merged))| {
            e.number = k + 1;
            if merged {
                e.shots = keep_longest(std::mem::take(&mut e.shots), MERGED_SHOTS);
            }
            (e, i)
        })
        .unzip()
}

impl PlanDoc {
    /// What to build: the options, the entries shown ([`resolve`]) and the words they speak (a
    /// plain line where the script has none, a plain intro or outro where it is empty).
    pub fn resolve(&self) -> Result<(showcase::Options, Vec<Entry>, Script), AutoCutError> {
        let d = showcase::Options::default();
        let subject = non_empty(&self.subject).unwrap_or(d.subject.clone());
        let mut opts = showcase::Options {
            title: non_empty(&self.title).unwrap_or_else(|| default_title(&subject)),
            subtitle: non_empty(&self.subtitle),
            voice: non_empty(&self.voice),
            mood: non_empty(&self.mood).unwrap_or(d.mood.clone()),
            subject,
            ..d
        };
        if let Some(b) = self.bpm.filter(|b| b.is_finite()) {
            opts.bpm = b.clamp(60.0, 180.0);
        }
        if let (Some(w), Some(h)) = (self.width, self.height) {
            // even sizes the encoders take, at most 8K
            opts.width = (w.clamp(64, 7680) / 2) * 2;
            opts.height = (h.clamp(64, 7680) / 2) * 2;
        }
        if let Some(f) = self.fps.filter(|f| f.is_finite()) {
            opts.fps = f.clamp(1.0, 120.0);
        }
        opts.look = crate::look::Look::from_plan(&Value::Object(self.look.clone()))?;
        // the narration's language: a plan's top-level `"language"` (`ml`: Malayalam)
        opts.language = self.look.get("language").and_then(Value::as_str).map(str::trim).filter(|l| !l.is_empty()).map(str::to_string);
        // how a Malayalam narration says numbers: the plan's top-level `"numbers"`
        opts.numbers = match self.look.get("numbers") {
            None | Some(Value::Null) => None,
            Some(Value::String(n)) => Some(showcase::number_style(n)?),
            Some(_) => return Err(AutoCutError::Invalid("numbers: english or malayalam".into())),
        };
        opts.redact = crate::privacy::Redact::from_plan(self.look.get("redact"))?;
        // what the counter counts, apart from the subject QA looks for (`"counter"`,
        // `"subjectCheck"`)
        let (counter, check) = showcase::counting_from_plan(&Value::Object(self.look.clone()))?;
        opts.counter = counter;
        opts.subject_check = check.unwrap_or(true);
        // talking clips: Enhance Voice on the shots' own sound (`"enhanceVoice": true`)
        opts.enhance_voice = match self.look.get("enhanceVoice") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(AutoCutError::Invalid("enhanceVoice: true or false".into())),
        };
        // spec chips that follow the robot (`"followLabels": true`; off by default)
        opts.follow_labels = match self.look.get("followLabels") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(AutoCutError::Invalid("followLabels: true or false".into())),
        };
        let (entries, heads) = resolve(&self.entries);
        if entries.is_empty() {
            return Err(AutoCutError::Invalid("no entry is included: include at least one".into()));
        }
        let script = self.script.clone().unwrap_or_default();
        let lines = entries
            .iter()
            .zip(&heads)
            .map(|(e, &h)| script.lines.get(h).map(|l| l.trim()).filter(|l| !l.is_empty()).map_or_else(|| plain_line(e), str::to_string))
            .collect();
        let intro = if script.intro.trim().is_empty() { format!("This is every {} I have filmed, in order.", opts.counted()) } else { script.intro.clone() };
        let outro = if script.outro.trim().is_empty() {
            format!("That's {} {}s so far. Thanks for watching.", entries.len(), opts.counted())
        } else {
            script.outro.clone()
        };
        // spoken versions follow their entry when others are left out or merged: `say` is keyed
        // by entry number, and the entries are numbered afresh
        let mut say: std::collections::BTreeMap<String, String> =
            script.say.iter().filter(|(k, _)| k.as_str() == "intro" || k.as_str() == "outro").map(|(k, v)| (k.clone(), v.clone())).collect();
        for (k, &h) in heads.iter().enumerate() {
            if let Some(v) = script.say.get(&(h + 1).to_string()) {
                say.insert((k + 1).to_string(), v.clone());
            }
        }
        // deliveries follow their entry the same way
        let mut delivery: std::collections::BTreeMap<String, String> =
            script.delivery.iter().filter(|(k, _)| k.as_str() == "intro" || k.as_str() == "outro").map(|(k, v)| (k.clone(), v.clone())).collect();
        for (k, &h) in heads.iter().enumerate() {
            if let Some(v) = script.delivery.get(&(h + 1).to_string()) {
                delivery.insert((k + 1).to_string(), v.clone());
            }
        }
        // and so do the spec chips
        let mut chips: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
        for (k, &h) in heads.iter().enumerate() {
            if let Some(v) = script.chips.get(&(h + 1).to_string()) {
                chips.insert((k + 1).to_string(), v.clone());
            }
        }
        Ok((opts, entries, Script { intro, lines, outro, templated: script.templated, pronounce: script.pronounce.clone(), say, delivery, chips }))
    }
}

/// [`Executor`] over a job's command port: commands run on the session that started the job.
pub struct PortExecutor {
    pub port: CommandPort,
    pub progress: Arc<Progress>,
}

impl Executor for PortExecutor {
    fn execute(&self, id: &str, params: Value) -> ExecFuture {
        let port = self.port.clone();
        let id = id.to_string();
        Box::pin(async move { tokio::task::spawn_blocking(move || call(&port, &id, params)).await.map_err(|e| format!("internal error: {e}"))? })
    }

    fn render(&self, seconds: f64, max_side: u32) -> RenderFuture {
        let port = self.port.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let (w, h, rgba) = port.render(seconds, max_side)?;
                crate::png(w, h, rgba, max_side)
            })
            .await
            .ok()
            .flatten()
        })
    }
}

/// Run `id` over the port, so that nothing long runs on the port's owner (see the module docs).
fn call(port: &CommandPort, id: &str, params: Value) -> Result<Value, String> {
    if LONG_COMMANDS.contains(&id) && params.get("wait").and_then(Value::as_bool) == Some(true) {
        return long(port, id, params);
    }
    if id == "tts.create" {
        prerender(port, &params);
    }
    port.call(id, params)
}

/// A long command started without waiting, its job polled until it finishes (cancelled with
/// this job): the answer the engine gives with `wait: true`.
fn long(port: &CommandPort, id: &str, mut params: Value) -> Result<Value, String> {
    if let Some(o) = params.as_object_mut() {
        o.insert("wait".into(), json!(false));
    }
    let mut out = port.call(id, params)?;
    let Some(job) = out.get("job").and_then(Value::as_u64) else { return Ok(out) };
    loop {
        std::thread::sleep(POLL);
        let jobs = match port.call("jobs.list", json!({})) {
            Ok(j) => j,
            Err(e) => {
                if port.cancelled() {
                    port.cancel_job(job);
                }
                return Err(e);
            }
        };
        let j = jobs.as_array().and_then(|a| a.iter().find(|j| j["id"].as_u64() == Some(job))).cloned().ok_or_else(|| format!("{id}: job {job} is gone"))?;
        if j["finished"].as_bool() == Some(true) {
            if let Some(e) = j["result"]["error"].as_str() {
                return Err(if id == "file.exportMedia" { format!("export failed: {e}") } else { format!("{id} failed: {e}") });
            }
            out["result"] = j["result"].clone();
            for k in ["offlineMedia", "upload", "reframe", "youtube", "speech", "scenes", "redact"] {
                if let Some(v) = j.get(k) {
                    out[k] = v.clone();
                }
            }
            return Ok(out);
        }
    }
}

/// Synthesize a narration line in a background job (`tts.render`) before `tts.create` places
/// it, so placing it takes the cached audio. A failure is left for `tts.create` to report.
fn prerender(port: &CommandPort, params: &Value) {
    let mut p = json!({"wait": true});
    for k in ["text", "voice", "pitch", "pace"] {
        if let Some(v) = params.get(k) {
            p[k] = v.clone();
        }
    }
    let _ = long(port, "tts.render", p);
}

/// A model whose requests end as soon as the job is cancelled.
struct Cancellable {
    inner: Arc<dyn ChatModel>,
    progress: Arc<Progress>,
}

impl ChatModel for Cancellable {
    fn id(&self) -> String {
        self.inner.id()
    }

    fn complete_json(&self, system: &str, user: &str, images: &[Vec<u8>]) -> JsonFuture {
        let p = self.progress.clone();
        if p.cancel.load(Ordering::Relaxed) {
            return Box::pin(async { Err(AutoCutError::Llm("cancelled".into())) });
        }
        let mut answer = self.inner.complete_json(system, user, images);
        let mut watch = Box::pin(async move {
            while !p.cancel.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        Box::pin(std::future::poll_fn(move |cx| {
            if let Poll::Ready(r) = answer.as_mut().poll(cx) {
                return Poll::Ready(r);
            }
            match watch.as_mut().poll(cx) {
                Poll::Ready(()) => Poll::Ready(Err(AutoCutError::Llm("cancelled".into()))),
                Poll::Pending => Poll::Pending,
            }
        }))
    }
}

/// The models a showcase job uses.
#[derive(Clone)]
pub struct Models {
    /// Locates, reads and describes the subject in clips (and checks it in the built show).
    pub analysis: Arc<dyn ChatModel>,
    /// Writes the voice-over, and reads prompts the built-in rules do not.
    pub script: Arc<dyn ChatModel>,
}

/// Labels [`Models::scripted`] gives clips, in turn.
pub const SCRIPTED_LABELS: [&str; 6] = ["red toy robot", "white humanoid robot", "yellow robot dog", "floor-cleaning robot", "delivery robot", "robot arm"];

impl Models {
    /// Stand-ins for Ollama, as `llm::FixedModel` is for the classifier (tests, demos without
    /// models): the subject is in every frame, nothing is legible, the clips are different robots
    /// labelled from [`SCRIPTED_LABELS`] in turn, and the writer answers every request with
    /// plain sentences.
    pub fn scripted() -> Self {
        let described = std::sync::atomic::AtomicUsize::new(0);
        let analysis = crate::llm::FnModel::new("scripted-vision", move |system, _, _| {
            Ok(if system == analyze::GROUND_SYSTEM {
                json!({"present": true, "bbox_2d": [120, 80, 520, 420]})
            } else if system.contains("describe") {
                let n = described.fetch_add(1, Ordering::Relaxed);
                let label = SCRIPTED_LABELS.get(n % SCRIPTED_LABELS.len()).copied().unwrap_or("robot");
                json!({"label": label, "kind": "toy", "description": format!("a {label} rolling around")})
            } else if system.contains("compare") {
                json!({"same": false, "confidence": 0.9})
            } else {
                json!({"on_subject": [], "nearby": []})
            })
        });
        let script = crate::llm::FnModel::new("scripted-writer", |_, user, _| {
            Ok(if user.contains("\"intro\"") {
                json!({"intro": "This is every robot I have met, in order.", "outro": "That was all of them, thanks for watching."})
            } else {
                // "Write \"lines\": exactly N entries"
                let n = user.split("exactly ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse::<usize>().ok()).unwrap_or(0).min(100);
                json!({"lines": (1..=n).map(|i| format!("Here is robot number {i} of this batch.")).collect::<Vec<_>>()})
            })
        });
        Self { analysis: Arc::new(analysis), script: Arc::new(script) }
    }
}

/// Where a job's models come from.
enum Source {
    /// Ollama, as configured (`AUTOCUT_*`); availability is checked when a job starts.
    Config(Config),
    Fixed(Models),
}

/// Runs the engine's showcase jobs with this crate ([`AutoCutHost`]).
pub struct ShowcaseHost {
    source: Source,
    /// Picks Google Photos albums for jobs given `album` (none: such jobs are refused).
    albums: Option<Arc<dyn AlbumSource>>,
    /// Where clip analyses are kept, so a clip is analysed once (none: not kept). A job's
    /// `fresh: true` analyses again.
    analysis_cache: Option<std::path::PathBuf>,
}

/// The host the app, the CLI and the MCP server install: Ollama models from the environment.
pub fn engine_host() -> Arc<dyn AutoCutHost> {
    Arc::new(ShowcaseHost::from_config(Config::from_env()))
}

/// [`engine_host`] whose jobs can also take their clips from a Google Photos `album`, picked by
/// `albums` (`filmcraft_automation::photos::album_source()` in the app, the CLI and the MCP server).
pub fn engine_host_with_albums(albums: Arc<dyn AlbumSource>) -> Arc<dyn AutoCutHost> {
    Arc::new(ShowcaseHost::from_config(Config::from_env()).with_album_source(albums))
}

/// What a job's models are, once looked up.
struct Resolved {
    analysis: Option<Arc<dyn ChatModel>>,
    script: Arc<dyn ChatModel>,
}

impl ShowcaseHost {
    pub fn from_config(cfg: Config) -> Self {
        let analysis_cache = filmcraft_engine::autosave::default_data_dir().map(|d| d.join("autocut").join("analysis-cache"));
        Self { source: Source::Config(cfg), albums: None, analysis_cache }
    }

    /// A host with these models (tests, scripting).
    pub fn with_models(models: Models) -> Self {
        Self { source: Source::Fixed(models), albums: None, analysis_cache: None }
    }

    /// This host, keeping clip analyses in `dir` (none: not kept).
    pub fn with_analysis_cache(mut self, dir: Option<std::path::PathBuf>) -> Self {
        self.analysis_cache = dir;
        self
    }

    /// The analysis cache for a job: the host's, bypassed by `fresh: true`.
    fn analysis_cache(&self, p: &Value) -> AnalysisCache {
        match &self.analysis_cache {
            Some(dir) => AnalysisCache::at(dir.clone()).fresh(p["fresh"].as_bool().unwrap_or(false)),
            None => AnalysisCache::off(),
        }
    }

    /// This host, picking Google Photos albums with `albums` for jobs given `album`.
    pub fn with_album_source(mut self, albums: Arc<dyn AlbumSource>) -> Self {
        self.albums = Some(albums);
        self
    }

    /// The models job `kind` needs: the vision model must be there to analyse (Ollama is asked
    /// which it has); the script model falls back to the classifier when it is not installed.
    async fn models(&self, kind: TaskKind, progress: &Arc<Progress>) -> Result<Resolved, AutoCutError> {
        let cfg = match &self.source {
            Source::Fixed(m) => return Ok(Resolved { analysis: Some(m.analysis.clone()), script: m.script.clone() }),
            Source::Config(cfg) => cfg.clone(),
        };
        let installed = |model: String| {
            let o = Ollama::new(&cfg.ollama_host, &model, cfg.http_timeout);
            async move {
                let probe = o.clone();
                let ok = tokio::task::spawn_blocking(move || probe.available().unwrap_or(false)).await.unwrap_or(false);
                (o, ok)
            }
        };
        let (vision, vision_ok) = installed(cfg.analysis_model.clone()).await;
        if matches!(kind, TaskKind::Analyze | TaskKind::Make) && !vision_ok {
            return Err(AutoCutError::Llm(format!(
                "the vision model `{}` is not in Ollama at {} — start Ollama (`ollama serve`) and `ollama pull {}`",
                cfg.analysis_model, cfg.ollama_host, cfg.analysis_model
            )));
        }
        let mut script: Arc<dyn ChatModel> = Arc::new(Ollama::new(&cfg.ollama_host, &cfg.llm_model, cfg.http_timeout));
        if matches!(kind, TaskKind::Plan | TaskKind::Make) {
            let (writer, ok) = installed(cfg.script_model.clone()).await;
            if ok {
                script = Arc::new(writer);
            } else {
                note(progress, &format!("{} is not installed; writing with {}", cfg.script_model, cfg.llm_model));
            }
        }
        let analysis: Option<Arc<dyn ChatModel>> = if vision_ok { Some(Arc::new(vision)) } else { None };
        Ok(Resolved { analysis, script })
    }

    /// The model that rates the shorts' openings: the script model, else the classifier, when
    /// Ollama has one; none (the shorts are scored without it) otherwise.
    async fn rater(&self, progress: &Arc<Progress>) -> Option<Arc<dyn ChatModel>> {
        let cfg = match &self.source {
            Source::Fixed(m) => return Some(m.script.clone()),
            Source::Config(cfg) => cfg.clone(),
        };
        for name in [cfg.script_model.clone(), cfg.llm_model.clone()] {
            let o = Ollama::new(&cfg.ollama_host, &name, cfg.http_timeout);
            let probe = o.clone();
            if tokio::task::spawn_blocking(move || probe.available().unwrap_or(false)).await.unwrap_or(false) {
                return Some(Arc::new(o));
            }
        }
        note(progress, &format!("no local model at {}: the shorts are scored without hook ratings", cfg.ollama_host));
        None
    }

    /// The model that writes a quick short's text: the script model, else the classifier, when
    /// Ollama has one; none otherwise (the text is written from the shots' descriptions).
    async fn writer(&self) -> Option<Arc<dyn ChatModel>> {
        let cfg = match &self.source {
            Source::Fixed(m) => return Some(m.script.clone()),
            Source::Config(cfg) => cfg.clone(),
        };
        for name in [cfg.script_model.clone(), cfg.llm_model.clone()] {
            let o = Ollama::new(&cfg.ollama_host, &name, cfg.http_timeout);
            let probe = o.clone();
            if tokio::task::spawn_blocking(move || probe.available().unwrap_or(false)).await.unwrap_or(false) {
                return Some(Arc::new(o));
            }
        }
        None
    }

    /// The model that directs an edit: the script model, else the classifier, with room for the
    /// shots' facts (a long library's prompt runs past Ollama's default context, and a prompt cut
    /// short gets a garbled plan back).
    async fn director(&self) -> Option<Arc<dyn ChatModel>> {
        let cfg = match &self.source {
            Source::Fixed(m) => return Some(m.script.clone()),
            Source::Config(cfg) => cfg.clone(),
        };
        for name in [cfg.script_model.clone(), cfg.llm_model.clone()] {
            let o = Ollama::new(&cfg.ollama_host, &name, cfg.http_timeout).with_context(crate::director::CONTEXT_TOKENS);
            let probe = o.clone();
            if tokio::task::spawn_blocking(move || probe.available().unwrap_or(false)).await.unwrap_or(false) {
                return Some(Arc::new(o));
            }
        }
        None
    }

    /// The model that writes Malayalam: `model` when given, else the script model, with room for
    /// Malayalam's many tokens. No fallback to the small classifier: its Malayalam is unusable.
    async fn translator(&self, p: &Value) -> Result<Arc<dyn ChatModel>, AutoCutError> {
        let cfg = match &self.source {
            Source::Fixed(m) => return Ok(m.script.clone()),
            Source::Config(cfg) => cfg.clone(),
        };
        let name = p["model"].as_str().map_or_else(|| cfg.script_model.clone(), str::to_string);
        let o = Ollama::new(&cfg.ollama_host, &name, cfg.http_timeout).with_context(crate::ml_translate::CONTEXT_TOKENS);
        let probe = o.clone();
        let ok = tokio::task::spawn_blocking(move || probe.available().unwrap_or(false)).await.unwrap_or(false);
        if !ok {
            return Err(AutoCutError::Llm(format!(
                "the model `{name}` is not in Ollama at {} — start Ollama (`ollama serve`) and `ollama pull {name}`",
                cfg.ollama_host
            )));
        }
        Ok(Arc::new(o))
    }

    /// `autocut.malayalamScript`: the plan's English voice-over in natural spoken Malayalam.
    async fn malayalam_script(&self, task: &Task, ex: &PortExecutor, progress: &Arc<Progress>) -> Result<Value, AutoCutError> {
        let p = &task.params;
        // the taste profile's wording and narration rules ("our son", "tell it with feeling")
        let taste = crate::taste::resolve(ex).await;
        let plan = p.get("plan").cloned().or_else(|| task.plan.clone()).ok_or_else(|| AutoCutError::Invalid("no plan to translate".into()))?;
        let model = Cancellable { inner: self.translator(p).await?, progress: progress.clone() };
        let examples_path = p["examples"]
            .as_str()
            .map(|e| std::path::PathBuf::from(filmcraft_engine::export_tools::expand_home(e)))
            .or_else(crate::ml_translate::examples_path);
        let examples = match examples_path.as_deref().map(crate::ml_translate::load_examples) {
            Some(Ok(e)) => e,
            Some(Err(e)) => {
                note(progress, &format!("the approved examples were not read ({e}); writing with the built-in ones"));
                Vec::new()
            }
            None => Vec::new(),
        };
        let defaults = crate::ml_translate::Settings::default();
        let settings = crate::ml_translate::Settings {
            max_words: p["maxWords"].as_u64().map_or(defaults.max_words, |n| n as usize),
            revise: p["revise"].as_bool().unwrap_or(true),
            batch: defaults.batch,
            examples,
            rules: taste.writing_rules(),
        };
        let t = crate::ml_translate::translate(&model, &plan, &settings, &mut |l| note(progress, l)).await?;
        let script = t.script();
        if let Some(out) = p["out"].as_str() {
            let path = filmcraft_engine::export_tools::expand_home(out);
            let text = serde_json::to_string_pretty(&script).unwrap_or_default();
            std::fs::write(&path, format!("{text}\n")).map_err(|e| AutoCutError::Other(format!("{path}: {e}")))?;
        }
        let summary = t.summary();
        let status = format!(
            "Malayalam: {} of {} parts written, {} clean",
            summary["translated"].as_u64().unwrap_or(0),
            summary["parts"].as_u64().unwrap_or(0),
            summary["clean"].as_u64().unwrap_or(0)
        );
        Ok(json!({"malayalam": script, "lines": t.lines, "notes": t.notes(), "summary": summary, "out": p["out"], "status": status}))
    }

    async fn job(&self, task: Task, ex: &PortExecutor, progress: &Arc<Progress>) -> Result<Value, AutoCutError> {
        if task.kind == TaskKind::Shorts {
            let rater = self.rater(progress).await.map(|m| Arc::new(Cancellable { inner: m, progress: progress.clone() }) as Arc<dyn ChatModel>);
            let mut lines = |l: &str| note(progress, l);
            return crate::shorts::make(ex, &task.params, rater, &mut lines).await;
        }
        if task.kind == TaskKind::MalayalamScript {
            return self.malayalam_script(&task, ex, progress).await;
        }
        if task.kind == TaskKind::RoughCut {
            let mut lines = |l: &str| note(progress, l);
            return crate::rough_cut::make(ex, &task.params, &mut lines).await;
        }
        if task.kind == TaskKind::Quick {
            let writer = self.writer().await.map(|m| Cancellable { inner: m, progress: progress.clone() });
            let mut lines = |l: &str| note(progress, l);
            return crate::quick::make(ex, &task.params, writer.as_ref().map(|w| w as &dyn ChatModel), &mut lines).await;
        }
        if task.kind == TaskKind::Direct {
            let mut p = task.params.clone();
            // a Google Photos album seeds the clips: picked and imported first
            if let Some(title) = p["album"].as_str().map(str::to_string) {
                let req = Request { subject: String::new(), since: None, title: None, photos: true };
                let seed = self.album_seed(&title, &req, ex, progress).await?;
                let mut items: Vec<Value> = p["items"].as_array().cloned().unwrap_or_default();
                items.extend(seed.imported.items.iter().map(|i| json!(i)));
                p["items"] = Value::Array(items);
            }
            // the director: the script model (it plans in JSON well), else the small one
            let director = self.director().await.map(|m| Cancellable { inner: m, progress: progress.clone() });
            let mut lines = |l: &str| note(progress, l);
            return crate::director::make(ex, &p, director.as_ref().map(|w| w as &dyn ChatModel), &mut lines).await;
        }
        let models = self.models(task.kind, progress).await?;
        let guard = |m: &Arc<dyn ChatModel>| Cancellable { inner: m.clone(), progress: progress.clone() };
        let script_model = guard(&models.script);
        let p = &task.params;
        match task.kind {
            TaskKind::Analyze | TaskKind::Make => {
                let vision = models.analysis.as_ref().map(guard).ok_or_else(|| AutoCutError::Llm("no vision model".into()))?;
                let request = request_of(p, &script_model).await?;
                // an album seeds the clips: picked and imported once the subject is known
                let mut p = p.clone();
                let seed = match p["album"].as_str() {
                    Some(title) => Some(self.album_seed(title, &request, ex, progress).await?),
                    None => None,
                };
                if let Some(seed) = &seed {
                    p["items"] = json!(seed.imported.items);
                }
                let p = &p;
                let (mut report, mut summary) = analyze_clips(&vision, ex, &request, p, &self.analysis_cache(p), progress).await?;
                if let Some(seed) = &seed {
                    report["album"] = json!(seed.title);
                    summary["album"] = seed.to_json();
                }
                if let Some(title) = &request.title {
                    summary["title"] = json!(title);
                }
                if task.kind == TaskKind::Analyze {
                    let status = format!("Found {} clip(s) showing a {}", summary["showing"], request.subject);
                    return Ok(json!({"report": report, "summary": summary, "status": status}));
                }
                let mut plan_params = p.clone();
                if plan_params.get("title").is_none() {
                    plan_params["title"] = json!(request.title.clone().unwrap_or_else(|| default_title(&request.subject)));
                }
                let taste = crate::taste::resolve(ex).await;
                let plan = write_plan(&script_model, &plan_params, Some(&report), None, &taste, progress).await?;
                let status = format!("{} entries planned from {} clip(s) showing a {}", plan.entries.len(), summary["showing"], request.subject);
                Ok(json!({"report": report, "summary": summary, "plan": plan, "status": status}))
            }
            TaskKind::Plan => {
                let taste = crate::taste::resolve(ex).await;
                let plan = write_plan(&script_model, p, task.report.as_ref(), task.plan.as_ref(), &taste, progress).await?;
                let status = format!("Script written for {} entries", plan.entries.iter().filter(|e| e.include && !e.merge_previous).count());
                Ok(json!({"plan": plan, "status": status}))
            }
            TaskKind::Build => {
                let qa_model = models.analysis.as_ref().map(guard);
                build_plan(ex, p, task.plan.as_ref(), qa_model.as_ref().map(|m| m as &dyn ChatModel), progress).await
            }
            TaskKind::Trailer => {
                // the vision model, when Ollama has it, grounds QA's subject check
                let qa_model = models.analysis.as_ref().map(guard);
                let mut lines = |l: &str| note(progress, l);
                crate::trailer::make(ex, p, task.plan.as_ref(), qa_model.as_ref().map(|m| m as &dyn ChatModel), &mut lines).await
            }
            // (handled above, without the showcase models)
            TaskKind::Shorts | TaskKind::MalayalamScript | TaskKind::RoughCut | TaskKind::Quick | TaskKind::Direct => {
                Err(AutoCutError::Other("not a showcase step".into()))
            }
        }
    }
}

impl ShowcaseHost {
    /// Pick album `title` with the host's [`AlbumSource`] (its videos and photos — videos only
    /// with `photos: false` — none created before the request's `since`) and import it over the
    /// port, reusing what the project already holds.
    async fn album_seed(&self, title: &str, req: &Request, ex: &PortExecutor, progress: &Arc<Progress>) -> Result<Seed, AutoCutError> {
        let source = self.albums.clone().ok_or_else(|| AutoCutError::Other(filmcraft_engine::autocut::NO_ALBUMS.into()))?;
        let pick = AlbumPick { title: title.to_string(), videos_only: !req.photos, since: req.since.as_ref().map(|(_, t)| album::day_of(*t)) };
        note(progress, &format!("picking the Google Photos album \"{title}\""));
        let prog = progress.clone();
        let fetched = tokio::task::spawn_blocking(move || source.pick(&pick, &mut |l| note(&prog, l), &|| prog.cancel.load(Ordering::Relaxed)))
            .await
            .map_err(|e| AutoCutError::Other(format!("internal error: {e}")))?
            .map_err(AutoCutError::Other)?;
        if progress.cancel.load(Ordering::Relaxed) {
            return Err(AutoCutError::Other("cancelled".into()));
        }
        if fetched.paths.is_empty() {
            let why = if !fetched.failed.is_empty() {
                format!(": none of its {} item(s) could be downloaded ({})", fetched.failed.len(), fetched.failed.join("; "))
            } else if !fetched.skipped.is_empty() {
                let left = if req.photos { "created before the date asked for" } else { "photos, or created before the date asked for" };
                format!(" ({} item(s) left out: {left})", fetched.skipped.len())
            } else {
                String::new()
            };
            let what = if req.photos { "no video or photo" } else { "no video" };
            return Err(AutoCutError::Other(format!("the album \"{title}\" has {what} to make a showcase from{why}")));
        }
        let (photos, videos) = album::count_kinds(&fetched.paths);
        note(progress, &format!("importing {videos} clip(s) and {photos} photo(s) from \"{title}\""));
        let imported = album::import(ex, &fetched.paths).await?;
        if imported.items.is_empty() {
            return Err(AutoCutError::Other(format!("nothing from the album \"{title}\" could be imported: {}", imported.errors.join("; "))));
        }
        Ok(Seed { title: title.to_string(), fetched, imported })
    }
}

/// An album picked and imported for an analysis.
struct Seed {
    title: String,
    fetched: album::AlbumFetched,
    imported: album::Imported,
}

impl Seed {
    /// What the job's summary says of it.
    fn to_json(&self) -> Value {
        let (f, i) = (&self.fetched, &self.imported);
        json!({
            "title": self.title,
            "picked": f.paths.len(),
            "downloaded": f.downloaded,
            "cached": f.cached,
            "items": i.items,
            "imported": i.items.len().saturating_sub(i.reused.len()),
            "reused": i.reused.len(),
            "failed": f.failed,
            "skipped": f.skipped.len(),
            "importErrors": i.errors,
            "note": f.note,
        })
    }
}

impl AutoCutHost for ShowcaseHost {
    fn albums_ready(&self) -> Result<(), String> {
        self.albums.as_ref().ok_or_else(|| filmcraft_engine::autocut::NO_ALBUMS.to_string())?.ready()
    }

    fn run(&self, task: Task, port: &CommandPort, progress: &Arc<Progress>) -> Result<Value, String> {
        let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().map_err(|e| format!("could not start the job: {e}"))?;
        let ex = PortExecutor { port: port.clone(), progress: progress.clone() };
        let r = rt.block_on(self.job(task, &ex, progress)).map_err(|e| e.to_string());
        // a model request a cancel left running finishes on its own; don't wait for it
        rt.shutdown_background();
        r
    }
}

/// A progress line: the job's status, and its counts when the line starts with `[k/n]`.
fn note(p: &Progress, line: &str) {
    let line = line.trim();
    if let Some((k, n)) = counts(line) {
        p.total.store(n, Ordering::Relaxed);
        p.done.store(k.min(n), Ordering::Relaxed);
    }
    *p.status.lock().unwrap_or_else(PoisonError::into_inner) = line.chars().take(200).collect();
}

/// `(3, 12)` of a line starting `[3/12]`.
fn counts(line: &str) -> Option<(u64, u64)> {
    let (inside, _) = line.strip_prefix('[')?.split_once(']')?;
    let (k, n) = inside.split_once('/')?;
    Some((k.trim().parse().ok()?, n.trim().parse().ok()?))
}

/// What to look for: the subject, a date to look from, and a title, from the parameters or the
/// prompt ("make a video of every robot since 2020").
struct Request {
    subject: String,
    since: Option<(Value, i64)>,
    title: Option<String>,
    /// Photos are kept (picked, imported, analysed) as well as videos; `photos: false` leaves
    /// them out.
    photos: bool,
}

async fn request_of(p: &Value, model: &dyn ChatModel) -> Result<Request, AutoCutError> {
    let mut since = p.get("since").cloned().zip(p["sinceUnix"].as_i64());
    let mut title = p["title"].as_str().map(str::to_string);
    let subject = match p["subject"].as_str() {
        Some(s) => s.to_string(),
        None => {
            let prompt = p["prompt"].as_str().unwrap_or_default();
            let c = match crate::intent::showcase_request(prompt) {
                Some(c) => c,
                None => crate::intent::classify(model, prompt).await?,
            };
            let subject = c.params["subject"].as_str().map(str::trim).filter(|s| c.intent == "showcase" && !s.is_empty() && s.chars().count() <= 80);
            let subject = subject
                .ok_or_else(|| AutoCutError::Invalid(format!("could not tell what the showcase is of from \"{prompt}\": name a subject, e.g. \"robot\"")))?;
            if since.is_none()
                && let Some(s) = c.params.get("since").filter(|s| !s.is_null())
            {
                since = filmcraft_engine::autocut::parse_since(s).ok().map(|t| (s.clone(), t));
            }
            if title.is_none() {
                title = c.params["title"].as_str().map(str::to_string);
            }
            subject.to_string()
        }
    };
    Ok(Request { subject, since, title, photos: p["photos"].as_bool().unwrap_or(true) })
}

/// The clips of `items` (all video items without; the photos too with `photos`) filmed on or
/// after `since`, and how many had no date (left out).
async fn filmed_since(ex: &dyn Executor, items: Option<&[u64]>, since: i64, photos: bool) -> Result<(Vec<u64>, usize), AutoCutError> {
    let clips = media_of(ex, items, photos).await?;
    let dates = ex.execute("media.captureDate", json!({"items": clips})).await.map_err(AutoCutError::Executor)?;
    let unix_of = |item: u64| dates["items"].as_array().and_then(|a| a.iter().find(|d| d["item"].as_u64() == Some(item))).and_then(|d| d["unix"].as_i64());
    let (mut kept, mut undated) = (Vec::new(), 0);
    for i in clips {
        match unix_of(i) {
            Some(t) if t >= since => kept.push(i),
            Some(_) => {}
            None => undated += 1,
        }
    }
    Ok((kept, undated))
}

/// The video items of `items` (all of the project's without), and its photos with `photos`.
async fn media_of(ex: &dyn Executor, items: Option<&[u64]>, photos: bool) -> Result<Vec<u64>, AutoCutError> {
    let project = ex.execute("project.inspect", json!({})).await.map_err(AutoCutError::Executor)?;
    let want: Option<HashSet<u64>> = items.map(|i| i.iter().copied().collect());
    Ok(analyze::media_items(&project).into_iter().filter(|m| photos || !m.3).map(|m| m.0).filter(|i| want.as_ref().is_none_or(|w| w.contains(i))).collect())
}

/// Analyse the clips for the request's subject: the report (as `analyze --out` writes it) and a
/// summary.
async fn analyze_clips(
    model: &dyn ChatModel,
    ex: &PortExecutor,
    req: &Request,
    p: &Value,
    cache: &AnalysisCache,
    progress: &Arc<Progress>,
) -> Result<(Value, Value), AutoCutError> {
    let mut items: Option<Vec<u64>> = p["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect());
    let mut undated = 0;
    if let Some((text, t)) = &req.since {
        note(progress, &format!("looking for clips filmed since {}", text.as_str().map_or_else(|| text.to_string(), str::to_string)));
        let (kept, no_date) = filmed_since(ex, items.as_deref(), *t, req.photos).await?;
        if kept.is_empty() {
            return Err(AutoCutError::Other(format!(
                "no video clip was filmed since {}{}",
                text.as_str().map_or_else(|| text.to_string(), str::to_string),
                if no_date > 0 { format!(" ({no_date} have no capture date)") } else { String::new() }
            )));
        }
        (items, undated) = (Some(kept), no_date);
    } else if !req.photos {
        // videos only: the photos among the items (or the project's) are left out
        items = Some(media_of(ex, items.as_deref(), false).await?);
    }
    note(progress, &format!("looking for a {} in {}", req.subject, if items.is_some() { "the clips" } else { "every clip" }));
    let frames = p["frames"].as_u64().map_or(6, |n| n.clamp(1, analyze::MAX_FRAMES as u64) as usize);
    let compare = p["compare"].as_u64().map_or(crate::identity::MAX_COMPARISONS, |n| n as usize);
    let screen = p["screen"].as_u64().map_or(0, |n| n.min(analyze::MAX_FRAMES as u64) as usize);
    let facts: Vec<ClipFacts> =
        analyze::run(model, ex, &req.subject, items.as_deref(), frames, screen, None, compare, cache, &mut |l| note(progress, l)).await?;
    let showing = facts.iter().filter(|f| f.present).count();
    let mut report = json!({"subject": req.subject, "model": model.id(), "clips": facts});
    // clips analysed before (the analysis cache) were not looked at again
    let mut summary = json!({"subject": req.subject, "clips": facts.len(), "showing": showing, "cached": cache.hits()});
    if let Some((text, _)) = &req.since {
        report["since"] = text.clone();
        summary["since"] = text.clone();
        summary["undatedLeftOut"] = json!(undated);
    }
    Ok((report, summary))
}

/// Group a report into entries (or take reviewed ones) and write the voice-over: a [`PlanDoc`]
/// whose entries keep the review's choices and whose lines follow them one to one. Options not
/// given come from the plan kept before, then the defaults.
async fn write_plan(
    model: &dyn ChatModel,
    p: &Value,
    report: Option<&Value>,
    kept: Option<&Value>,
    taste: &filmcraft_engine::project::taste::Taste,
    progress: &Arc<Progress>,
) -> Result<PlanDoc, AutoCutError> {
    let kept: PlanDoc = kept.and_then(|k| serde_json::from_value(k.clone()).ok()).unwrap_or_default();
    let text = |k: &str| p[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let subject = text("subject")
        .or_else(|| report.and_then(|r| r["subject"].as_str()).map(str::to_string))
        .or_else(|| non_empty(&kept.subject))
        .unwrap_or_else(|| showcase::Options::default().subject);
    let d = showcase::Options::default();
    let opts = showcase::Options {
        title: text("title").or_else(|| non_empty(&kept.title)).unwrap_or_else(|| default_title(&subject)),
        subtitle: p.get("subtitle").map_or_else(|| non_empty(&kept.subtitle), |_| text("subtitle")),
        voice: p.get("voice").map_or_else(|| non_empty(&kept.voice), |_| text("voice")),
        mood: text("mood").or_else(|| non_empty(&kept.mood)).unwrap_or(d.mood.clone()),
        min_coverage: p["minCoverage"].as_f64().or(kept.min_coverage).unwrap_or(d.min_coverage).clamp(0.0, 1.0),
        subject: subject.clone(),
        taste: taste.clone(),
        ..d
    };
    let review: Vec<ReviewEntry> = match p.get("entries").and_then(Value::as_array) {
        Some(list) => list
            .iter()
            .enumerate()
            .map(|(i, e)| serde_json::from_value(e.clone()).map_err(|err| AutoCutError::Invalid(format!("entry {}: {err}", i + 1))))
            .collect::<Result<_, _>>()?,
        None => {
            let clips =
                report.and_then(|r| r.get("clips")).cloned().ok_or_else(|| AutoCutError::Other("no analysis to plan from: find the clips first".into()))?;
            let mut facts: Vec<ClipFacts> = serde_json::from_value(clips).map_err(|e| AutoCutError::Other(format!("not an analysis report ({e})")))?;
            for f in &mut facts {
                f.label = analyze::sanitize_label_with(&f.label, &f.kind, &subject, &f.text);
            }
            showcase::group(&facts, &subject, opts.min_coverage).into_iter().map(ReviewEntry::new).collect()
        }
    };
    let (shown, heads) = resolve(&review);
    if shown.is_empty() {
        return Err(AutoCutError::Other(if review.is_empty() {
            format!("no clip shows a {subject} clearly enough to count")
        } else {
            "no entry is included: include at least one".into()
        }));
    }
    note(progress, &format!("writing the voice-over for {} entries", shown.len()));
    let script = showcase::write_script(model, &opts, &shown, &mut |l| note(progress, l)).await;
    let mut lines = vec![String::new(); review.len()];
    for (k, &h) in heads.iter().enumerate() {
        if let (Some(slot), Some(line)) = (lines.get_mut(h), script.lines.get(k)) {
            line.clone_into(slot);
        }
    }
    Ok(PlanDoc {
        subject: Some(subject),
        title: Some(opts.title),
        subtitle: opts.subtitle,
        voice: opts.voice,
        mood: Some(opts.mood),
        min_coverage: Some(opts.min_coverage),
        bpm: kept.bpm,
        width: kept.width,
        height: kept.height,
        fps: kept.fps,
        entries: review,
        script: Some(Script { lines, ..script }),
        look: kept.look,
    })
}

/// Build a plan into a new sequence; with `export`, export it, its chapters and QA report beside
/// it, and with `upload`, start its Google Photos upload (a job of its own).
async fn build_plan(
    ex: &PortExecutor,
    p: &Value,
    kept: Option<&Value>,
    qa_model: Option<&dyn ChatModel>,
    progress: &Arc<Progress>,
) -> Result<Value, AutoCutError> {
    let plan = p.get("plan").or(kept).ok_or_else(|| AutoCutError::Invalid("nothing to build: no plan".into()))?;
    let plan: PlanDoc = serde_json::from_value(plan.clone()).map_err(|e| AutoCutError::Invalid(format!("not a showcase plan ({e})")))?;
    let (mut opts, entries, script) = plan.resolve()?;
    opts.export = p["export"].as_str().map(str::to_string);
    // a Malayalam script is checked before it is voiced (docs/malayalam-style.md)
    let script_check = opts.malayalam().then(|| crate::ml_translate::check_script(&script, &entries));
    if let Some(c) = &script_check {
        note(progress, &format!("Malayalam script: {} warning(s), {} hint(s) (script.lintMalayalam lists them)", c["warnings"], c["hints"]));
    }
    let built = showcase::build(ex, &opts, &entries, &script, &mut |l| note(progress, l)).await?;
    let mut out = serde_json::to_value(&built).map_err(|e| AutoCutError::Other(e.to_string()))?;
    if let Some(c) = script_check {
        out["scriptCheck"] = c;
    }
    let chapters = showcase::youtube_chapters(&built.chapters, built.seconds);
    out["youtubeChapters"] = json!(chapters);
    // where the export went (the engine expands `~` and adds the extension)
    let written = built.export.as_ref().and_then(|r| r["path"].as_str()).map(str::to_string).or_else(|| opts.export.clone());
    if let Some(path) = &written
        && !chapters.is_empty()
    {
        let txt = Path::new(path).with_extension("chapters.txt");
        match std::fs::write(&txt, format!("{chapters}\n")) {
            Ok(()) => out["chaptersFile"] = json!(txt.to_string_lossy()),
            Err(e) => note(progress, &format!("chapters: {}: {e}", txt.display())),
        }
    }
    let qa_on = p["qa"].as_bool().unwrap_or(written.is_some());
    // a person's face shown unblurred stops the upload (`QaReport::publish_blocker`)
    let mut blocked = None;
    if qa_on {
        let q = crate::qa::Options::for_showcase(&opts, &built);
        let mut report = crate::qa::check(ex, &q, qa_model, &mut |l| note(progress, l)).await;
        if let Some(path) = &written
            && let Err(e) = crate::qa::write_next_to(&mut report, Path::new(path))
        {
            note(progress, &format!("QA: {e}"));
        }
        out["qaSummary"] = json!(report.summary());
        out["qa"] = serde_json::to_value(&report).unwrap_or_default();
        blocked = report.publish_blocker();
        if let Some(b) = &blocked {
            note(progress, &format!("PRIVACY: {b}"));
            out["privacyBlocked"] = json!(b);
        }
    }
    if p["upload"].as_bool() == Some(true)
        && written.is_some()
        && let Some(b) = &blocked
    {
        out["uploadError"] = json!(format!("not uploaded: {b}"));
    } else if p["upload"].as_bool() == Some(true)
        && let Some(path) = &written
    {
        let mut up = json!({"path": path, "description": opts.title});
        if let Some(a) = p["uploadAlbum"].as_str() {
            up["album"] = json!(a);
        }
        match ex.execute("photos.upload", up).await {
            Ok(r) => out["uploadJob"] = r["job"].clone(),
            Err(e) => out["uploadError"] = json!(e),
        }
    }
    let secs = built.seconds.max(0.0).round() as u64;
    let mut status = format!("Built {} entries ({}:{:02})", built.entries, secs / 60, secs % 60);
    if let Some(path) = &written {
        status.push_str(&format!(", exported to {path}"));
    }
    if let Some(b) = &blocked {
        status.push_str(&format!(". PRIVACY: {b}"));
    }
    Ok(json!({"built": out, "status": status}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(item: u64, seconds: f64, seen: usize, unseen: usize) -> Shot {
        Shot {
            item,
            name: format!("{item}.mp4"),
            clip_seconds: seconds,
            best: seconds / 2.0,
            seen: (0..seen).map(|i| i as f64).collect(),
            unseen: (0..unseen).map(|i| i as f64 + 0.5).collect(),
            ..Shot::default()
        }
    }

    fn entry(n: usize, label: &str, shots: Vec<Shot>) -> Entry {
        Entry { number: n, label: label.into(), kind: "toy".into(), description: format!("a {label}"), unix: Some(1_665_741_668), shots, ..Entry::default() }
    }

    #[test]
    fn a_review_drops_and_merges_entries() {
        let review = vec![
            ReviewEntry::new(entry(1, "red robot", vec![shot(1, 10.0, 4, 0), shot(2, 4.0, 1, 3)])),
            ReviewEntry { merge_previous: true, ..ReviewEntry::new(entry(2, "red toy robot", vec![shot(3, 20.0, 4, 0), shot(4, 30.0, 1, 0)])) },
            ReviewEntry { include: false, ..ReviewEntry::new(entry(3, "a person", vec![shot(5, 9.0, 1, 0)])) },
            ReviewEntry::new(entry(4, "Spot", vec![shot(6, 8.0, 2, 0)])),
            // merged into an excluded entry's place: the included one before it
            ReviewEntry { merge_previous: true, ..ReviewEntry::new(entry(5, "Spot again", vec![shot(7, 5.0, 1, 0)])) },
        ];
        let (shown, heads) = resolve(&review);
        assert_eq!(heads, vec![0, 3]);
        assert_eq!(shown.iter().map(|e| (e.number, e.label.as_str())).collect::<Vec<_>>(), vec![(1, "red robot"), (2, "Spot")]);
        // the three longest in view of the four, in order
        assert_eq!(shown[0].shots.iter().map(|s| s.item).collect::<Vec<_>>(), vec![1, 3, 4]);
        assert_eq!(shown[1].shots.iter().map(|s| s.item).collect::<Vec<_>>(), vec![6, 7]);
        // a first entry marked merge has nothing before it: it stands alone
        let lone = vec![ReviewEntry { merge_previous: true, ..ReviewEntry::new(entry(1, "x", vec![shot(1, 1.0, 1, 0)])) }];
        assert_eq!(resolve(&lone).1, vec![0]);
        assert!(resolve(&[]).0.is_empty());
    }

    #[test]
    fn spoken_versions_follow_their_entries_through_the_review() {
        let plan: PlanDoc = serde_json::from_value(json!({
            "entries": [
                serde_json::to_value(entry(1, "Zaku II model kit", vec![shot(1, 10.0, 1, 0)])).unwrap(),
                serde_json::to_value(entry(2, "a person", vec![shot(2, 10.0, 1, 0)])).unwrap(),
                serde_json::to_value(entry(3, "DP World Promobot", vec![shot(3, 10.0, 1, 0)])).unwrap(),
            ],
            "script": {"intro": "Hi.", "lines": ["I built a Zaku II.", "Someone.", "A Promobot."], "outro": "Bye.",
                       "pronounce": {"Zaku II": "Zaku Two"}, "say": {"intro": "Hello!", "1": "I built it.", "3": "A Promobot from D P World."},
                       "delivery": {"intro": "excited", "2": "calm", "3": "amazed"}},
        }))
        .unwrap();
        let mut v = serde_json::to_value(&plan).unwrap();
        v["entries"][1]["include"] = json!(false);
        let plan: PlanDoc = serde_json::from_value(v).unwrap();
        let (_, entries, script) = plan.resolve().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(script.pronounce.get("Zaku II").map(String::as_str), Some("Zaku Two"));
        // the third entry is now the second: its spoken version moves with it
        let say: Vec<(&str, &str)> = script.say.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(say, vec![("1", "I built it."), ("2", "A Promobot from D P World."), ("intro", "Hello!")]);
        // and so does its delivery; the one of the entry left out goes with it
        let delivery: Vec<(&str, &str)> = script.delivery.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(delivery, vec![("2", "amazed"), ("intro", "excited")]);
    }

    #[test]
    fn a_plan_reads_the_cli_file_and_the_review() {
        // the CLI's plan.json: entries and script, no review fields
        let cli = json!({
            "entries": [serde_json::to_value(entry(1, "red robot", vec![shot(1, 10.0, 1, 0)])).unwrap(), serde_json::to_value(entry(2, "Spot", vec![shot(2, 10.0, 1, 0)])).unwrap()],
            "script": {"intro": "Hello.", "lines": ["One.", "  "], "outro": ""},
        });
        let plan: PlanDoc = serde_json::from_value(cli).unwrap();
        assert!(plan.entries.iter().all(|e| e.include && !e.merge_previous));
        let (opts, entries, script) = plan.resolve().unwrap();
        assert_eq!((opts.title.as_str(), opts.subject.as_str(), opts.voice.as_deref()), ("Every Robot I've Filmed", "robot", None));
        assert_eq!(entries.len(), 2);
        // an empty line and outro get plain ones
        assert_eq!(script.lines, vec!["One.".to_string(), "In October 2022, I met the Spot.".to_string()]);
        assert!(script.outro.contains("2 robots"), "{}", script.outro);
        assert!(!opts.malayalam());
        // the narration's language: the plan's top-level `language`, kept when written back
        let mut ml = serde_json::to_value(&plan).unwrap();
        ml["language"] = json!("ml");
        let ml: PlanDoc = serde_json::from_value(ml).unwrap();
        assert!(ml.resolve().unwrap().0.malayalam());
        assert_eq!(serde_json::to_value(&ml).unwrap()["language"], "ml");
        assert_eq!(ml.resolve().unwrap().0.numbers, None);
        // and how it says numbers: the plan's `numbers`, checked
        let mut words = serde_json::to_value(&ml).unwrap();
        words["numbers"] = json!("Malayalam");
        let words: PlanDoc = serde_json::from_value(words).unwrap();
        assert_eq!(words.resolve().unwrap().0.numbers.as_deref(), Some("malayalam"));
        for bad in [json!("roman"), json!(3)] {
            let mut v = serde_json::to_value(&ml).unwrap();
            v["numbers"] = bad;
            assert!(serde_json::from_value::<PlanDoc>(v).unwrap().resolve().is_err());
        }
        // what the counter counts, apart from what QA looks for: a show of versions of one toy
        // car counts "VERSION #n" and looks for the car (or not at all)
        assert_eq!((opts.counted(), opts.counter.as_deref(), opts.subject_check), ("robot", None, true));
        let mut versions = serde_json::to_value(&plan).unwrap();
        versions["subject"] = json!("toy car");
        versions["counter"] = json!(" version ");
        versions["subjectCheck"] = json!(false);
        let versions: PlanDoc = serde_json::from_value(versions).unwrap();
        let (o, entries, script) = versions.resolve().unwrap();
        assert_eq!((o.subject.as_str(), o.counted(), o.subject_check), ("toy car", "version", false));
        assert!(script.outro.contains(&format!("{} versions", entries.len())), "{}", script.outro);
        assert_eq!(serde_json::to_value(&versions).unwrap()["counter"], " version ", "kept as written");
        for (k, bad) in [("counter", json!(3)), ("counter", json!("VERSION #")), ("counter", json!("  ")), ("subjectCheck", json!("no"))] {
            let mut v = serde_json::to_value(&plan).unwrap();
            v[k] = bad;
            assert!(serde_json::from_value::<PlanDoc>(v).unwrap().resolve().is_err(), "{k}");
        }
        // the review's fields round-trip
        let mut v = serde_json::to_value(&plan).unwrap();
        v["entries"][0]["include"] = json!(false);
        v["voice"] = json!(" clone:me ");
        v["width"] = json!(1_000_000);
        v["height"] = json!(3);
        v["fps"] = json!(1e9);
        let plan: PlanDoc = serde_json::from_value(v).unwrap();
        let (opts, entries, script) = plan.resolve().unwrap();
        assert_eq!((entries.len(), entries[0].label.as_str(), script.lines[0].as_str()), (1, "Spot", "In October 2022, I met the Spot."));
        assert_eq!((opts.voice.as_deref(), opts.width, opts.height, opts.fps), (Some("clone:me"), 7680, 64, 120.0), "hostile sizes are clamped");
        // nothing included: an error, not an empty show
        let none = PlanDoc { entries: plan.entries.iter().map(|e| ReviewEntry { include: false, ..e.clone() }).collect(), ..PlanDoc::default() };
        assert!(none.resolve().is_err());
        // a hostile entry is refused by serde, not trusted
        assert!(serde_json::from_value::<PlanDoc>(json!({"entries": [{"label": "x"}]})).is_err());
        assert!(
            serde_json::from_value::<PlanDoc>(json!({"entries": [{"number": -1, "label": "x", "kind": "", "description": "", "unix": null, "shots": []}]}))
                .is_err()
        );
    }

    #[test]
    fn progress_lines_carry_counts() {
        assert_eq!(counts("[3/12] c3.mp4: red robot"), Some((3, 12)));
        assert_eq!(counts("[x/12] nope"), None);
        assert_eq!(counts("writing"), None);
        let p = Progress::default();
        note(&p, "[20/12] more than all");
        assert_eq!((p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed)), (12, 12));
        note(&p, &"long ".repeat(500));
        assert_eq!(p.status.lock().unwrap().chars().count(), 200);
    }

    #[test]
    fn default_titles() {
        assert_eq!(default_title("robot"), "Every Robot I've Filmed");
        assert_eq!(default_title("vintage car"), "Every Vintage Car I've Filmed");
    }
}
