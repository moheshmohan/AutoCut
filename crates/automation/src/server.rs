//! The MCP server. Tool names are snake_case.

use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine as _;
use filmcraft_autocut::Executor as _;
use filmcraft_engine::Session;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock as Content};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AutomationError, BridgeClient, png_rgba};

pub enum Backend {
    Headless(Arc<Mutex<Session>>),
    Bridge(Arc<BridgeClient>),
}

/// A panic payload as text.
fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// A failed blocking task: a panic is reported as an internal error and the session stays usable.
fn join_error(e: tokio::task::JoinError) -> AutomationError {
    match e.try_into_panic() {
        Ok(p) => AutomationError::Other(format!("internal error: {}", panic_message(p))),
        Err(e) => AutomationError::Other(e.to_string()),
    }
}

#[derive(Clone)]
pub struct FilmcraftMcp {
    backend: Arc<Backend>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListParams {
    /// Only commands whose id or label contains this text (case-insensitive).
    #[serde(default)]
    pub filter: Option<String>,
    /// Only commands that can run right now.
    #[serde(default)]
    pub enabled_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunParams {
    /// Command id, e.g. `sequence.addEdit`, `timeline.trim`, `effects.apply`, `file.import`.
    pub id: String,
    /// Parameters object (see `params` in `command_list`).
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchStep {
    /// Command id.
    pub id: String,
    /// Parameters object.
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchParams {
    /// Commands to run in order.
    pub steps: Vec<BatchStep>,
    /// Stop at the first failing step (default true).
    #[serde(default)]
    pub stop_on_error: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct RenderParams {
    /// Timeline time in seconds (default: the playhead).
    #[serde(default)]
    pub seconds: Option<f64>,
    /// Longest side of the PNG (default 960).
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShotParams {
    /// Panel to crop to (e.g. `Timeline`, `Program`); omit for the whole window.
    #[serde(default)]
    pub panel: Option<String>,
    /// Longest side of the PNG (default 1600).
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClickParams {
    /// Element id from `ui_elements` (e.g. `timeline.track.V1.lock`, `tools.Razor`, `project.item.12`).
    #[serde(default)]
    pub id: Option<String>,
    /// Or screen coordinates in points.
    #[serde(default)]
    pub x: Option<f64>,
    #[serde(default)]
    pub y: Option<f64>,
    /// left | right | middle
    #[serde(default)]
    pub button: Option<String>,
    /// 2 for double-click.
    #[serde(default)]
    pub count: Option<u32>,
    /// {"shift":bool,"alt":bool,"command":bool,"ctrl":bool}
    #[serde(default)]
    pub modifiers: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DragParams {
    /// Start: {"id":..} or {"x":..,"y":..} (optionally "fx"/"fy" fractions inside the element).
    pub from: Value,
    /// End point, same forms.
    pub to: Value,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub modifiers: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KeyParams {
    /// Key with optional modifiers, e.g. `Space`, `Cmd+K`, `Shift+Delete`, `I`, `L`.
    pub key: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TypeParams {
    pub text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MediaImportParams {
    /// Absolute paths to media files (MP4/MOV, WAV/MP3/FLAC/AIFF/Ogg, PNG/JPEG/…).
    pub paths: Vec<String>,
    /// Bin ID to import into (optional).
    #[serde(default)]
    pub bin: Option<u64>,
    /// Import numbered stills as an image sequence (optional, default false).
    #[serde(default)]
    pub image_sequence: Option<bool>,
}

/// `file.import` params for a `media_import` call.
fn media_import_params(p: MediaImportParams) -> Value {
    let mut params = json!({"paths": p.paths});
    if let Some(bin) = p.bin {
        params["bin"] = json!(bin);
    }
    if let Some(seq) = p.image_sequence {
        params["imageSequence"] = json!(seq);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ElementsParams {
    /// Only element ids starting with this prefix (e.g. `timeline.clip.`).
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ControlParams {
    /// Any control-channel method (e.g. `ui.set`, `ui.scroll`, `ui.timeline.hit`, `ui.playback`).
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PhotosPickParams {
    /// At most this many items selectable in the picker (1–2000; default 2000).
    #[serde(default)]
    pub max_items: Option<u32>,
    /// Seconds to wait for the user to press Done (default 300, at most 3600). 0 opens the
    /// picker and returns at once with a pending session to resume.
    #[serde(default)]
    pub wait_seconds: Option<u64>,
    /// Resume this pending session (from an earlier `status: pending` result) instead of
    /// opening a new picker.
    #[serde(default)]
    pub session: Option<String>,
    /// When set, a local vision agent drives the picker itself: it types this query into the
    /// picker's search field, selects matching video tiles (up to max_items or 6), and
    /// presses Done — no human needed. Requires the Ollama grounding model
    /// (AUTOCUT_SCREEN_MODEL, default qwen2.5vl:7b — `ollama pull qwen2.5vl:7b`) and Screen
    /// Recording + Accessibility for the process running this server. The agent is
    /// best-effort: its report comes back as `autopick` in the result, and the pick itself
    /// is still authoritative.
    #[serde(default)]
    pub query: Option<String>,
    /// More searches (with or without `query`): the agent types them one after another into
    /// ONE picking session in ONE browser tab — the selection accumulates — and presses Done
    /// once; many run in as few sessions as the item cap (max_items, up to 2000) allows, each
    /// taking up to 20 searches, and searches that did not fit come back as `left` for another
    /// call. A search with no results is skipped. Date searches ("videos October 2022") are
    /// selected by the picker's date groups. Use with `all` for whole results.
    #[serde(default)]
    pub queries: Option<Vec<String>>,
    /// With `query`/`queries`: select every search result (the date groups of a date search,
    /// else the first tile and a Shift+click on the last) instead of up to max_items tiles.
    #[serde(default)]
    pub all: Option<bool>,
    /// Skip picked items created before this date (YYYY-MM-DD); they are not downloaded.
    #[serde(default)]
    pub since: Option<String>,
    /// Instead of `query`: pick every item of the user's Google Photos album with this exact
    /// title (the album is the seed of the video — the user adds what they want to it in
    /// Google Photos). The vision agent opens the album in the picker (its Albums view, or by
    /// searching the title) and selects all of it; when no album matches it selects nothing
    /// and reports why in `autopick`. Same prerequisites as `query`.
    #[serde(default)]
    pub album: Option<String>,
    /// With `album`: download only its videos (photos are skipped).
    #[serde(default)]
    pub videos_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PhotosAddParams {
    /// Media item ids of already-downloaded items (from googlephotos_pick or googlephotos_cached).
    pub ids: Vec<String>,
}

/// The most ids one googlephotos_add call takes.
const MAX_ADD_IDS: usize = 2000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PhotosUploadParams {
    /// Absolute path of the file to upload, typically a finished export (MP4/MOV/M4V/MKV/AVI…
    /// videos up to 10 GB, JPEG/PNG/HEIC/WebP/GIF/TIFF… photos up to 200 MB).
    pub path: String,
    /// Shown under the item in Google Photos (at most 1000 characters).
    #[serde(default)]
    pub description: Option<String>,
    /// Also put the item into this album (a title, or an album id): the album FilmCraft made
    /// with exactly that title, created when there is none. The user can then add more to it
    /// by hand in Google Photos. Needs `filmcraft-cli photos auth --albums` once.
    #[serde(default)]
    pub album: Option<String>,
}

/// `photos.upload` params for a `googlephotos_upload` call (a long job: progress and cancel).
fn photos_upload_params(p: PhotosUploadParams) -> Value {
    let mut params = json!({"path": p.path, "wait": true});
    if let Some(d) = p.description {
        params["description"] = json!(d);
    }
    if let Some(a) = p.album {
        params["album"] = json!(a);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseAnalyzeParams {
    /// What to look for in every clip, singular: "robot", "vintage car". Or give a `prompt`.
    #[serde(default)]
    pub subject: Option<String>,
    /// Instead of `subject`: the video to make, in words ("a video of every robot since 2020"):
    /// the subject (and a date and title, when it names them) is read from it — by built-in
    /// rules, else the local classifier.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Only these media items (ids from project_inspect); default: every video item.
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// The clips come from this Google Photos album (its exact title) instead of the project:
    /// the album the user filled in Google Photos is picked whole (the local screen agent opens
    /// it in Google's picker and selects all of it — Google lets apps read only albums they
    /// made, so the picker is the way in), its videos downloaded and imported (files the
    /// project already holds are reused, not imported twice), and only those are analysed. Not
    /// with `items`. Needs `filmcraft-cli photos auth` once, the grounding model
    /// (`ollama pull qwen2.5vl:7b`) and Screen Recording + Accessibility for this process.
    #[serde(default)]
    pub album: Option<String>,
    /// Only clips filmed on or after this date: "2020", "2020-06" or "2020-06-15" (clips without a
    /// capture date are left out).
    #[serde(default)]
    pub since: Option<String>,
    /// Frames sampled per clip, 1–8 (default 6).
    #[serde(default)]
    pub frames: Option<u32>,
    /// Look at this many spread-out frames of a clip first and skip the rest when none shows the
    /// subject — much faster over many clips that mostly do not (default 0: off).
    #[serde(default)]
    pub screen: Option<u32>,
    /// Most side-by-side comparisons for telling the same robot across clips (default 150; 0:
    /// by name only).
    #[serde(default)]
    pub compare: Option<u32>,
    /// Analyse every clip again. By default a clip analysed before — the same file (its Google
    /// Photos id, or its size and content), subject, model and frames/screen — comes from the
    /// analysis cache in the data directory (`cached` in the summary counts them).
    #[serde(default)]
    pub fresh: Option<bool>,
    /// Also group the clips into entries and write the voice-over (`autocut.makeShowcase`),
    /// default false: the result then holds the plan to review.
    #[serde(default)]
    pub plan: Option<bool>,
    /// With `plan`: the video's title (default "Every <Subject> I've Filmed").
    #[serde(default)]
    pub title: Option<String>,
    /// With `plan`: the narration voice (`clone:<name>` for a cloned one; "" none).
    #[serde(default)]
    pub voice: Option<String>,
}

/// `autocut.analyze` (or, with `plan`, `autocut.makeShowcase`) and its params, as a long job.
fn showcase_analyze_params(p: ShowcaseAnalyzeParams) -> (&'static str, Value) {
    let make = p.plan.unwrap_or(false);
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    let text = |v: Option<String>| v.filter(|t| !t.trim().is_empty()).map(Value::String);
    put("subject", text(p.subject));
    put("prompt", text(p.prompt));
    put("album", text(p.album));
    put("items", p.items.map(|i| json!(i)));
    put("since", p.since.map(Value::String));
    put("frames", p.frames.map(|n| json!(n)));
    put("screen", p.screen.map(|n| json!(n)));
    put("compare", p.compare.map(|n| json!(n)));
    put("fresh", p.fresh.filter(|f| *f).map(Value::Bool));
    if make {
        put("title", p.title.map(Value::String));
        put("voice", p.voice.map(Value::String));
    }
    (if make { "autocut.makeShowcase" } else { "autocut.analyze" }, params)
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcasePlanParams {
    /// The video's title (default: the plan's, else "Every <Subject> I've Filmed").
    #[serde(default)]
    pub title: Option<String>,
    /// Under the title on the opening card (default: the years the entries span).
    #[serde(default)]
    pub subtitle: Option<String>,
    /// Narration voice: a built-in id, or `clone:<name>` for a cloned voice; "" no narration.
    #[serde(default)]
    pub voice: Option<String>,
    /// The music bed's mood: epic, tense, uplifting or calm.
    #[serde(default)]
    pub mood: Option<String>,
    /// Clips showing the subject in less of their frames are left out (0–1, default 0.3).
    #[serde(default)]
    pub min_coverage: Option<f64>,
    /// Reviewed entries (a plan's `entries`, each with `include` and `mergePrevious`): the voice-over
    /// is written for exactly these. Default: the kept analysis, grouped afresh.
    #[serde(default)]
    pub entries: Option<Vec<Value>>,
}

fn showcase_plan_params(p: ShowcasePlanParams) -> Value {
    let mut params = json!({"wait": true});
    for (k, v) in [("title", p.title), ("subtitle", p.subtitle), ("voice", p.voice), ("mood", p.mood)] {
        if let Some(v) = v {
            params[k] = json!(v);
        }
    }
    if let Some(c) = p.min_coverage {
        params["minCoverage"] = json!(c);
    }
    if let Some(e) = p.entries {
        params["entries"] = json!(e);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseBuildParams {
    /// The plan to build (as showcase_plan returns it, edited); default: the kept plan.
    #[serde(default)]
    pub plan: Option<Value>,
    /// Export the built showcase here (YouTube 1080p; chapters and a QA report beside it).
    #[serde(default)]
    pub export: Option<String>,
    /// Then upload the export to the user's Google Photos library (its own job; needs `export`
    /// and `filmcraft-cli photos auth --upload` once).
    #[serde(default)]
    pub upload: Option<bool>,
    /// With `upload`: also put the export into this album (FilmCraft's album of that title,
    /// made when missing; needs `filmcraft-cli photos auth --albums` once).
    #[serde(default)]
    pub upload_album: Option<String>,
    /// Check the built show (gaps, black frames, the mix, the subject in every shot); default:
    /// with an export.
    #[serde(default)]
    pub qa: Option<bool>,
}

fn showcase_build_params(p: ShowcaseBuildParams) -> Value {
    let mut params = json!({"wait": true, "upload": p.upload.unwrap_or(false)});
    if let Some(plan) = p.plan {
        params["plan"] = plan;
    }
    if let Some(e) = p.export {
        params["export"] = json!(e);
    }
    if let Some(a) = p.upload_album.filter(|a| !a.trim().is_empty()) {
        params["uploadAlbum"] = json!(a);
    }
    if let Some(q) = p.qa {
        params["qa"] = json!(q);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseInspectParams {
    /// The whole analysis report (every clip's facts) instead of its summary.
    #[serde(default)]
    pub report: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct TemplateListParams {
    /// Only templates whose name, description, id or tags contain this text.
    #[serde(default)]
    pub query: Option<String>,
    /// One template in full (id, name or path of a .fcvideo-template folder or .zip): its slots with
    /// their placements, variants, beat grid and manifest.
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TemplateApplyParams {
    /// Template id (`builtin:photo-beat-slideshow`, `builtin:channel-intro`, `builtin:top-3-listicle`),
    /// name, or path of a .fcvideo-template folder or .zip.
    pub template: String,
    /// Aspect variant: `16:9`, `9:16`… (default: the template's own).
    #[serde(default)]
    pub variant: Option<String>,
    /// What fills each slot, by slot id: a file path, a project item id, a text (text slots), or
    /// {"path"|"item"|"text", "inSeconds"?, "focus"?: [x, y], "focusByVariant"?: {"9:16": [x, y]}},
    /// or a list of them for slots that take several. Unfilled slots keep their placeholders.
    #[serde(default)]
    pub slots: Option<Value>,
    /// Name of the new sequence.
    #[serde(default)]
    pub name: Option<String>,
}

/// Signed-in Google Photos, or the tool-shaped setup error.
fn gphotos() -> Result<filmcraft_gphotos::GooglePhotos, AutomationError> {
    filmcraft_gphotos::GooglePhotos::configured().map_err(|e| AutomationError::Other(e.to_string()))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct AutocutParams {
    /// What to do, in plain words, e.g. "remove the silences and add captions" or "cut 10 to 25 seconds".
    pub prompt: String,
    /// Caption sampled frames with the local vision model and include them in the returned
    /// document when the prompt is handed to you (default false; slower).
    #[serde(default)]
    pub frames: Option<bool>,
    /// Plan only; nothing executes (autocut_edit only, default false).
    #[serde(default)]
    pub dry_run: Option<bool>,
}

/// Shared body of the AutoCut tools: run the pipeline and return the report as JSON.
async fn autocut_report(ex: &AutocutExecutor, prompt: &str, opts: filmcraft_autocut::runner::Options) -> Result<Value, AutomationError> {
    let cfg = filmcraft_autocut::config::Config::from_env();
    let runners = filmcraft_autocut::runner::Runners::from_config(&cfg);
    filmcraft_autocut::runner::run_with(&runners, prompt, ex, &cfg, opts)
        .await
        .map_err(|e| AutomationError::Other(e.to_string()))
        .and_then(|r| serde_json::to_value(&r).map_err(|e| AutomationError::Other(e.to_string())))
}

/// The most searches one `googlephotos_pick` call takes.
const MAX_PICK_QUERIES: usize = 1000;

/// The files of a pick to import, what to report of each item (no short-lived download link),
/// and how many came from the cache.
fn picked_items(picked: &filmcraft_gphotos::Picked) -> (Vec<String>, Vec<Value>, usize) {
    let paths = picked.fetched.iter().map(|f| f.path.to_string_lossy().into_owned()).collect();
    let items = picked
        .fetched
        .iter()
        .map(|f| {
            let mut v = f.item.summary();
            if let Some(o) = v.as_object_mut() {
                o.insert("path".into(), json!(f.path.to_string_lossy()));
            }
            v
        })
        .collect();
    (paths, items, picked.fetched.iter().filter(|f| f.was_cached).count())
}

fn ok_json(v: &Value) -> CallToolResult {
    CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(v).unwrap_or_default())])
}
fn fail(e: impl std::fmt::Display) -> CallToolResult {
    CallToolResult::error(vec![Content::text(e.to_string())])
}
fn png_result(png: &[u8], note: &str) -> CallToolResult {
    CallToolResult::success(vec![Content::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"), Content::text(note.to_string())])
}
fn wrap(r: Result<Value, AutomationError>) -> Result<CallToolResult, McpError> {
    Ok(match r {
        Ok(v) => ok_json(&v),
        Err(e) => fail(e),
    })
}

/// `Executor` over either MCP backend: the same dispatch `FilmcraftMcp::run` uses, as a
/// standalone handle AutoCut can plan against.
#[derive(Clone)]
pub struct AutocutExecutor(Arc<Backend>);

impl AutocutExecutor {
    /// A bridge-mode executor over a (cloned) [`BridgeClient`], for callers that already have
    /// one (the CLI's `--bridge` backend).
    pub fn bridge(client: BridgeClient) -> Self {
        Self(Arc::new(Backend::Bridge(Arc::new(client))))
    }
}

impl filmcraft_autocut::Executor for AutocutExecutor {
    fn execute(&self, id: &str, params: Value) -> filmcraft_autocut::ExecFuture {
        match &*self.0 {
            Backend::Headless(s) if id.starts_with("autocut.") && crate::long_job::is_long(id, &params) => {
                Box::pin(long_headless(s.clone(), id.to_string(), params))
            }
            Backend::Headless(s) => {
                let s = s.clone();
                let id = id.to_string();
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        // A command that panicked earlier poisoned the lock; the session is still usable.
                        let mut g = s.lock().unwrap_or_else(PoisonError::into_inner);
                        g.execute(&id, params).map_err(|e| e.to_string())
                    })
                    .await
                    .map_err(|e| format!("internal error: {e}"))?
                })
            }
            Backend::Bridge(b) => {
                let b = b.clone();
                let id = id.to_string();
                Box::pin(async move { b.execute(&id, params).await.map_err(|e| e.to_string()) })
            }
        }
    }

    fn render(&self, seconds: f64, max_side: u32) -> filmcraft_autocut::RenderFuture {
        match &*self.0 {
            Backend::Headless(s) => {
                let s = s.clone();
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        let g = s.lock().unwrap_or_else(PoisonError::into_inner);
                        let (w, h) = g.active_sequence().map(|q| (q.settings.width, q.settings.height))?;
                        let scale = (max_side as f32 / w.max(h).max(1) as f32).min(1.0);
                        let img = g.try_render_program_at(scale, filmcraft_time::Tick::from_seconds_f64(seconds)).ok()?;
                        render_png(img.w as u32, img.h as u32, img.over_black_rgba8(), max_side)
                    })
                    .await
                    .ok()
                    .flatten()
                })
            }
            // the live app renders through its UI; engine.execute can't reach the compositor,
            // so bridge mode plans without eyes for now
            Backend::Bridge(_) => Box::pin(async move { None }),
        }
    }
}

/// Same encoding as `filmcraft_autocut`'s `SessionExecutor::render`: RGBA8 → downscaled PNG.
fn render_png(w: u32, h: u32, rgba: Vec<u8>, max_side: u32) -> Option<Vec<u8>> {
    let mut img = image::RgbaImage::from_raw(w, h, rgba)?;
    if max_side > 0 && w.max(h) > max_side {
        let s = max_side as f32 / w.max(h) as f32;
        img = image::imageops::resize(&img, ((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1), image::imageops::FilterType::Triangle);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

/// Run what background jobs (AutoCut showcases) send a headless session over its command port,
/// as the app's UI does once per frame: a thread woken by the port that holds the session lock
/// only for a burst of requests, so the server answers other calls while a job runs. It ends
/// with the session (whose waker holds `woken`'s sender).
fn pump_port(session: std::sync::Weak<Mutex<Session>>, woken: std::sync::mpsc::Receiver<()>) {
    use std::sync::mpsc::RecvTimeoutError;
    let pump = move || loop {
        match woken.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(()) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        let Some(s) = session.upgrade() else { return };
        s.lock().unwrap_or_else(PoisonError::into_inner).pump_port(std::time::Duration::from_millis(50));
    };
    // a failed spawn leaves `wait: true` calls working (they run the port themselves)
    let _ = std::thread::Builder::new().name("filmcraft-mcp-port".into()).spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(pump));
    });
}

/// A long showcase command (`wait: true`) on the headless session, started without waiting and
/// its job polled, as the bridge does: the session lock is free meanwhile, so the port's pump
/// thread can run the job's commands. The answer is the engine's `wait: true` answer.
async fn long_headless(s: Arc<Mutex<Session>>, id: String, mut params: Value) -> Result<Value, String> {
    let exec = |id: String, params: Value| {
        let s = s.clone();
        async move {
            tokio::task::spawn_blocking(move || s.lock().unwrap_or_else(PoisonError::into_inner).execute(&id, params).map_err(|e| e.to_string()))
                .await
                .map_err(|e| format!("internal error: {e}"))?
        }
    };
    if let Some(p) = params.as_object_mut() {
        p.insert("wait".into(), json!(false));
    }
    let mut start = exec(id.clone(), params).await?;
    let Some(job) = start.get("job").and_then(Value::as_u64) else { return Ok(start) };
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let jobs = exec("jobs.list".into(), json!({})).await?;
        let Some(j) = jobs.as_array().and_then(|a| a.iter().find(|j| j["id"].as_u64() == Some(job))).cloned() else {
            return Err(format!("`{id}`: job {job} is gone"));
        };
        if j["finished"].as_bool() == Some(true) {
            if let Some(e) = j["result"]["error"].as_str() {
                return Err(format!("`{id}` failed: {e}"));
            }
            if let Some(o) = start.as_object_mut() {
                o.insert("result".into(), j["result"].clone());
                for k in crate::long_job::JOB_EXTRAS {
                    if let Some(v) = j.get(*k) {
                        o.insert((*k).into(), v.clone());
                    }
                }
            }
            return Ok(start);
        }
    }
}

impl FilmcraftMcp {
    pub fn headless(mut session: Session) -> Self {
        let (wake, woken) = std::sync::mpsc::channel::<()>();
        session.set_port_waker(Arc::new(move || {
            let _ = wake.send(());
        }));
        let shared = Arc::new(Mutex::new(session));
        pump_port(Arc::downgrade(&shared), woken);
        Self { backend: Arc::new(Backend::Headless(shared)), tool_router: Self::tool_router() }
    }
    pub fn bridge(addr: &str) -> Result<Self, AutomationError> {
        Ok(Self { backend: Arc::new(Backend::Bridge(Arc::new(BridgeClient::new(addr)?))), tool_router: Self::tool_router() })
    }

    /// An [`AutocutExecutor`] handle onto this server's backend.
    pub fn executor(&self) -> AutocutExecutor {
        AutocutExecutor(self.backend.clone())
    }

    pub async fn serve_stdio(self) -> Result<(), AutomationError> {
        self.serve_io(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serve MCP over any line stream. A line that isn't JSON gets a JSON-RPC parse error
    /// (`-32700`, id null) instead of being dropped silently; the rest go to the rmcp service. One
    /// writer task owns `output`, so replies never interleave.
    pub async fn serve_io(
        self,
        input: impl tokio::io::AsyncRead + Unpin + Send + 'static,
        output: impl tokio::io::AsyncWrite + Unpin + Send + 'static,
    ) -> Result<(), AutomationError> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let (mut to_service, service_in) = tokio::io::duplex(1 << 20);
        let (service_out, from_service) = tokio::io::duplex(1 << 20);
        let (err_tx, mut err_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let reader = tokio::spawn(async move {
            let mut lines = BufReader::new(input).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(_) => {
                        if to_service.write_all(format!("{line}\n").as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let reply = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                        let _ = err_tx.send(reply.to_string());
                    }
                }
            }
            // EOF: closing the pipe ends the service
            let _ = to_service.shutdown().await;
        });
        let writer = tokio::spawn(async move {
            let mut output = output;
            let mut replies = BufReader::new(from_service).lines();
            loop {
                let line = tokio::select! {
                    l = replies.next_line() => match l { Ok(Some(l)) => l, _ => break },
                    Some(e) = err_rx.recv() => e,
                };
                if output.write_all(format!("{line}\n").as_bytes()).await.is_err() || output.flush().await.is_err() {
                    break;
                }
            }
        });
        let running = self.serve((service_in, service_out)).await.map_err(|e| AutomationError::Other(format!("MCP init: {e}")))?;
        let r = running.waiting().await.map_err(|e| AutomationError::Other(e.to_string()));
        reader.abort();
        let _ = writer.await;
        r.map(|_| ())
    }

    /// Run an engine command on whichever backend.
    pub async fn run(&self, id: &str, params: Value) -> Result<Value, AutomationError> {
        self.executor().execute(id, params).await.map_err(AutomationError::Other)
    }

    /// The project tree plus the active sequence (null when there is none).
    async fn document(&self) -> Result<Value, AutomationError> {
        let project = self.run("project.inspect", json!({})).await?;
        let sequence = self.run("sequence.inspect", json!({})).await.ok();
        Ok(json!({"project": project, "sequence": sequence}))
    }

    fn bridge_client(&self) -> Option<Arc<BridgeClient>> {
        match &*self.backend {
            Backend::Bridge(b) => Some(b.clone()),
            Backend::Headless(_) => None,
        }
    }

    async fn ui(&self, method: &str, params: Value) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => wrap(b.call(method, params).await),
            None => Ok(fail(
                "this tool drives the live app: run `filmcraft-cli mcp --bridge 127.0.0.1:<port>` with the app started as `filmcraft --control <port>`",
            )),
        }
    }
}

#[tool_router]
impl FilmcraftMcp {
    #[tool(
        title = "List commands",
        description = "List every command (id, label, menu path, shortcut, parameter doc, enabled now). Menus, shortcuts, panels and timeline gestures all map to these ids.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn command_list(&self, Parameters(p): Parameters<ListParams>) -> Result<CallToolResult, McpError> {
        let r = self.run("command.list", json!({})).await.map(|v| {
            let f = p.filter.unwrap_or_default().to_ascii_lowercase();
            let only = p.enabled_only.unwrap_or(false);
            Value::Array(
                v.as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|c| {
                        let id = c["id"].as_str().unwrap_or("").to_ascii_lowercase();
                        let label = c["label"].as_str().unwrap_or("").to_ascii_lowercase();
                        (f.is_empty() || id.contains(&f) || label.contains(&f)) && (!only || c["enabled"].as_bool() == Some(true))
                    })
                    .collect(),
            )
        });
        wrap(r)
    }

    #[tool(
        title = "Run a command",
        description = "Run a command by id with JSON params, e.g. {\"id\":\"timeline.razor\",\"params\":{\"seconds\":3.5}} or {\"id\":\"effects.apply\",\"params\":{\"effect\":\"Gaussian Blur\"}}. Undoable edits land in History.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn command_run(&self, Parameters(p): Parameters<RunParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run(&p.id, p.params.unwrap_or(json!({}))).await)
    }

    #[tool(
        title = "Run several commands",
        description = "Run several commands in order: {steps: [{id, params?}], stop_on_error?: true}. Returns {completed, failed, results: [{ok, result | error}]}; each edit is its own undo step.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn command_batch(&self, Parameters(p): Parameters<BatchParams>) -> Result<CallToolResult, McpError> {
        let stop = p.stop_on_error.unwrap_or(true);
        let (mut completed, mut failed, mut results) = (0u32, 0u32, Vec::new());
        for st in p.steps {
            match self.run(&st.id, st.params.unwrap_or(json!({}))).await {
                Ok(v) => {
                    completed += 1;
                    results.push(json!({"ok": true, "result": v}));
                }
                Err(e) => {
                    failed += 1;
                    results.push(json!({"ok": false, "error": e.to_string()}));
                    if stop {
                        break;
                    }
                }
            }
        }
        let v = json!({"completed": completed, "failed": failed, "results": results});
        Ok(if failed > 0 { CallToolResult::error(vec![Content::text(v.to_string())]) } else { ok_json(&v) })
    }

    #[tool(
        title = "Inspect the document",
        description = "The project as JSON: the tree (as project_inspect) plus the active sequence (as sequence_inspect; null when there is none).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn doc_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.document().await)
    }

    #[tool(
        title = "Render a preview",
        description = "The active sequence's frame at `seconds` (default: the playhead) as PNG, same as render_frame (bridge mode: Program monitor screenshot).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn render_preview(&self, p: Parameters<RenderParams>) -> Result<CallToolResult, McpError> {
        self.render_frame(p).await
    }

    #[tool(
        title = "Inspect the project tree",
        description = "Project tree (bins and items with ids, types, durations), active sequence, source clip.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn project_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.run("project.inspect", json!({})).await)
    }

    #[tool(
        title = "Inspect the active sequence",
        description = "The active sequence as JSON: settings, tracks, clips (ids, start/duration in ticks and frames, effects), transitions, markers, in/out, playhead, selection. 254016000000 ticks = 1 second.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn sequence_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.run("sequence.inspect", json!({})).await)
    }

    #[tool(
        title = "Import media",
        description = "Import media files by absolute path (MP4/MOV, WAV/MP3/FLAC/AIFF/Ogg, PNG/JPEG/…).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn media_import(&self, Parameters(p): Parameters<MediaImportParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("file.import", media_import_params(p)).await)
    }

    #[tool(
        title = "AutoCut: plan an edit",
        description = "Turn a plain-language prompt (\"remove the silences\", \"cut 0:10 to 0:25\") into a validated batch of engine commands (dry run: nothing executes). Local models classify and plan routine edits. When the report has needsSupervisor, the prompt is beyond the routine catalog and YOU are the supervisor: plan the edit yourself with command_list / command_run / command_batch (render_preview to check) using the returned document; pass frames: true to also get vision captions of sampled frames.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = true)
    )]
    async fn autocut_plan(&self, Parameters(p): Parameters<AutocutParams>) -> Result<CallToolResult, McpError> {
        wrap(
            autocut_report(
                &self.executor(),
                &p.prompt,
                filmcraft_autocut::runner::Options { mode: filmcraft_autocut::runner::Mode::PlanOnly, frames: p.frames.unwrap_or(false) },
            )
            .await,
        )
    }

    #[tool(
        title = "AutoCut: edit",
        description = "Turn a plain-language prompt into engine commands and run them (each an ordinary undoable edit; `edit.undo` reverts). For prompts beyond the routine catalog the report comes back with needsSupervisor and nothing runs: plan it yourself with command_list / command_run / command_batch as in autocut_plan. Configure models with AUTOCUT_* environment variables.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = true)
    )]
    async fn autocut_edit(&self, Parameters(p): Parameters<AutocutParams>) -> Result<CallToolResult, McpError> {
        let mode = if p.dry_run.unwrap_or(false) { filmcraft_autocut::runner::Mode::PlanOnly } else { filmcraft_autocut::runner::Mode::Execute };
        wrap(autocut_report(&self.executor(), &p.prompt, filmcraft_autocut::runner::Options { mode, frames: p.frames.unwrap_or(false) }).await)
    }

    #[tool(
        title = "Google Photos: pick and import",
        description = "Open the Google Photos picker in the user's browser, wait for a selection and press Done, download the picks (cached under the app's data dir) and import them into the open project with file.import (a file the project already holds is reused, not imported twice). Google offers apps no server-side search of the library (retired 2025-03-31): the choice happens in the picker. Either the user chooses (tell them what to pick), or — pass `query` — a local vision agent (Qwen2.5-VL via Ollama) drives the picker itself: it types the query into the picker's own search, selects matching video tiles and presses Done; needs `ollama pull qwen2.5vl:7b` plus Screen Recording and Accessibility for this process, and its report comes back as `autopick`. Pass `queries` (and `all`) to run many searches — a list, or a sweep of a whole history such as one \"videos <Month> <Year>\" per month — in ONE picking session and ONE browser tab (the selection accumulates across searches; date searches are selected by the picker's date groups; a search with no results is skipped; Done once), in as few sessions as the item cap allows; searches that did not fit come back as `left`. The agent closes the picker tab when it finishes, and a session it gave up on is deleted. Pass `album` instead to use a whole Google Photos album as the seed (the user curates it in Google Photos): the agent opens the album with that exact title in the picker and selects all of it (`videos_only` keeps only its videos); no matching album selects nothing. Items a user adds to an album are reachable only this way (the Library API shows apps only what they uploaded). If the wait runs out the result is `status: pending` with a `session` and `pickerUri`; call again with that `session` to keep waiting (with short client tool timeouts, start with wait_seconds 0, then resume). Videos download whole, so they can take a while. One-time setup: `filmcraft-cli photos auth` (see crates/gphotos/README.md).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn googlephotos_pick(&self, Parameters(p): Parameters<PhotosPickParams>) -> Result<CallToolResult, McpError> {
        let searching = p.query.as_deref().is_some_and(|q| !q.trim().is_empty()) || p.queries.as_ref().is_some_and(|q| q.iter().any(|q| !q.trim().is_empty()));
        if p.album.as_deref().is_some_and(|a| !a.trim().is_empty()) && searching {
            return Ok(fail("pass `query`/`queries` (searches) or `album` (a whole album), not both"));
        }
        let asked = p.query.iter().count().saturating_add(p.queries.as_ref().map_or(0, Vec::len));
        if asked > MAX_PICK_QUERIES {
            return Ok(fail(format!("at most {MAX_PICK_QUERIES} searches per call (got {asked}); the rest can go in the next call")));
        }
        let gp = match gphotos() {
            Ok(gp) => gp,
            Err(e) => return Ok(fail(e)),
        };
        let since = p.since.filter(|d| !d.trim().is_empty());
        if let Some(d) = since.as_deref().filter(|d| !filmcraft_gphotos::is_day(d)) {
            return Ok(fail(format!("`since` wants a date like 2020-01-01, not `{d}`")));
        }
        let queries = filmcraft_screenagent::agent::clean_queries(p.query.into_iter().chain(p.queries.unwrap_or_default()).collect());
        let album = p.album.map(|a| a.trim().to_string()).filter(|a| !a.is_empty());
        if let Some(a) = album.as_deref()
            && let Err(e) = filmcraft_gphotos::albums::clean_title(a)
        {
            return Ok(fail(format!("`album`: {e}")));
        }
        let videos_only = p.videos_only.unwrap_or(false);
        if videos_only && album.is_none() {
            return Ok(fail("`videos_only` goes with `album` (`query` always picks videos)"));
        }
        let select_all = p.all.unwrap_or(false);
        let max_items = |default_tiles: u32| p.max_items.unwrap_or(if select_all { filmcraft_gphotos::api::MAX_PICK } else { default_tiles });
        if queries.len() > 1 {
            if p.session.as_deref().is_some_and(|s| !s.trim().is_empty()) {
                return Ok(fail("`session` resumes one pick; it does not go with many `queries`"));
            }
            let req = crate::photos::Searches { queries, select_all, max_items: max_items(6), per_session: crate::photos::PER_SESSION };
            let opts = filmcraft_gphotos::PickOptions {
                max_items: None,
                wait: std::time::Duration::from_secs(p.wait_seconds.unwrap_or(900)),
                resume: None,
                open_browser: true,
                since,
                videos_only: false,
            };
            let swept = match tokio::task::spawn_blocking(move || crate::photos::pick_searches(&gp, &opts, &req, &mut |_| {}, &|| false)).await {
                Ok(s) => s,
                Err(e) => return Ok(fail(format!("internal error: {e}"))),
            };
            let (paths, items, cached) = picked_items(&swept.picked);
            let imported = if paths.is_empty() {
                Err(crate::AutomationError::Other("nothing was picked".into()))
            } else {
                self.run("file.import", json!({"paths": paths, "reuse": true})).await
            };
            let v = json!({
                "status": "done", "items": items, "downloaded": swept.picked.fetched.len().saturating_sub(cached), "cached": cached,
                "failed": swept.picked.failed, "skipped": swept.picked.skipped, "imported": imported.as_ref().ok(),
                "importError": imported.as_ref().err().map(|e| e.to_string()),
                "sessions": swept.sessions, "left": swept.left, "tooBig": swept.too_big, "stopped": swept.stopped,
                "next": if swept.left.is_empty() { Value::Null } else if swept.too_big.is_empty() {
                    json!("call googlephotos_pick again with `queries` = `left` (and the same `all`) for the searches no session got to")
                } else {
                    json!("call googlephotos_pick again with `queries` = `left` (and the same `all`); split each of `tooBig` first (by day, or narrower searches): its results alone are more than a session holds")
                },
            });
            let ok = imported.is_ok() && swept.picked.failed.is_empty() && swept.stopped.is_none();
            return Ok(if ok { ok_json(&v) } else { CallToolResult::error(vec![Content::text(v.to_string())]) });
        }
        let opts = filmcraft_gphotos::PickOptions {
            max_items: p.max_items,
            wait: std::time::Duration::from_secs(p.wait_seconds.unwrap_or(if !queries.is_empty() || album.is_some() { 900 } else { 300 })),
            resume: p.session.filter(|s| !s.trim().is_empty()),
            open_browser: true,
            since,
            videos_only,
        };
        let request = match (queries.first(), album) {
            (Some(_), _) => Some(filmcraft_screenagent::agent::Request { queries, max_items: max_items(6), select_all, album: None, videos_only: true }),
            (None, Some(title)) => Some(filmcraft_screenagent::agent::Request {
                max_items: p.max_items.unwrap_or(filmcraft_gphotos::api::MAX_PICK),
                ..crate::photos::album_request(&title, videos_only)
            }),
            (None, None) => None,
        };
        // blocking HTTP, sleeps and the agent: keep them off the async threads. The wait ends
        // early when the agent finishes without pressing Done
        let driven = match tokio::task::spawn_blocking(move || crate::photos::pick_driven(&gp, &opts, request, &mut |_| {}, &|| false)).await {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => return Ok(fail(e)),
            Err(e) => return Ok(fail(format!("internal error: {e}"))),
        };
        let autopick: Value = match &driven.agent {
            Some(Ok(s)) => serde_json::to_value(s).unwrap_or(Value::Null),
            Some(Err(e)) => json!({"error": e}),
            None => Value::Null,
        };
        if driven.nothing_to_pick() {
            return Ok(fail(json!({
                "status": "done", "error": "the agent found nothing to pick (no matching album or search result), so nothing was selected",
                "autopick": autopick,
            })));
        }
        let picked = match driven.outcome {
            filmcraft_gphotos::PickOutcome::Done(picked) => picked,
            filmcraft_gphotos::PickOutcome::Pending(_) if driven.abandoned => {
                return Ok(fail(json!({
                    "status": "done", "error": "the agent did not finish the pick in time; its picking session was deleted and its tab closed",
                    "autopick": autopick,
                })));
            }
            filmcraft_gphotos::PickOutcome::Pending(s) => {
                return Ok(ok_json(&json!({
                    "status": "pending", "session": s.id, "pickerUri": s.autoclose_uri(), "expires": s.expire_time,
                    "next": "no selection is confirmed yet; give the user pickerUri if the browser did not open, then call googlephotos_pick again with this session",
                })));
            }
        };
        if picked.fetched.is_empty() {
            return Ok(fail(json!({"status": "done", "error": "nothing was picked or nothing could be downloaded", "failed": picked.failed})));
        }
        let (paths, items, cached) = picked_items(&picked);
        let imported = self.run("file.import", json!({"paths": paths, "reuse": true})).await;
        let ok = imported.is_ok() && picked.failed.is_empty();
        let v = json!({
            "status": "done", "items": items, "downloaded": picked.fetched.len().saturating_sub(cached), "cached": cached,
            "failed": picked.failed, "skipped": picked.skipped, "imported": imported.as_ref().ok(), "importError": imported.as_ref().err().map(|e| e.to_string()),
            "autopick": autopick,
        });
        Ok(if ok { ok_json(&v) } else { CallToolResult::error(vec![Content::text(v.to_string())]) })
    }

    #[tool(
        title = "Google Photos: list downloaded items",
        description = "The Google Photos originals already downloaded by earlier picks: {id, filename, bytes, path}. Pass ids to googlephotos_add to import them again without the picker.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn googlephotos_cached(&self) -> Result<CallToolResult, McpError> {
        let list = tokio::task::spawn_blocking(filmcraft_gphotos::cache::list).await.unwrap_or_default();
        Ok(ok_json(&Value::Array(list)))
    }

    #[tool(
        title = "Google Photos: import downloaded items",
        description = "Import already-downloaded Google Photos items (ids from googlephotos_pick or googlephotos_cached) into the open project with file.import (files the project already holds are reused, not imported twice). Items never picked are not reachable: Google shares only what the user picks, so use googlephotos_pick for new ones.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn googlephotos_add(&self, Parameters(p): Parameters<PhotosAddParams>) -> Result<CallToolResult, McpError> {
        if p.ids.is_empty() {
            return Ok(fail("no media item ids given"));
        }
        if p.ids.len() > MAX_ADD_IDS {
            return Ok(fail(format!("at most {MAX_ADD_IDS} ids per call")));
        }
        let (paths, missing) = match tokio::task::spawn_blocking(move || {
            let mut paths = Vec::new();
            let mut missing = Vec::new();
            for id in p.ids {
                match filmcraft_gphotos::cache::find(&id) {
                    Some(path) => paths.push(path.to_string_lossy().into_owned()),
                    None => missing.push(id),
                }
            }
            (paths, missing)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(fail(format!("internal error: {e}"))),
        };
        if !missing.is_empty() {
            return Ok(fail(json!({"error": "not downloaded yet; pick these with googlephotos_pick", "missing": missing})));
        }
        wrap(self.run("file.import", json!({"paths": paths, "reuse": true})).await.map(|imported| json!({"imported": imported, "paths": paths})))
    }

    #[tool(
        title = "Google Photos: upload a file",
        description = "Upload a local file — typically a finished export (`file.exportMedia`) — to the user's Google Photos library, resumably in 8 MiB chunks, and return the new media item (mediaItemId, productUrl). With `album` (a title or id) the item also goes into that album: the one FilmCraft made with exactly that title, created when missing, so the user can add more to it by hand for the next iteration (read those back with googlephotos_pick `album`). Runs as the `photos.upload` engine job: progress is reported as MCP progress notifications (bytes) and cancelling the request stops the upload (the local file is never touched). Videos up to 10 GB, photos up to 200 MB. Needs the user's one-time consent to add to their library (`filmcraft-cli photos auth --upload`; `--albums` for albums) and the Photos Library API enabled in their Google Cloud project; without them the error says what to run. Albums are also engine commands: `photos.createAlbum`, `photos.albums`, `photos.addToAlbum` (command_run).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn googlephotos_upload(
        &self,
        Parameters(p): Parameters<PhotosUploadParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if p.path.trim().is_empty() {
            return Ok(fail("`path` is empty: pass the file to upload"));
        }
        Ok(self.run_long("photos.upload", photos_upload_params(p), &context).await)
    }

    #[tool(
        title = "Showcase: find every clip of a subject",
        description = "Step 1 of a showcase (\"every robot I've filmed, in order\": a counted, narrated video). A local vision model (Ollama, AUTOCUT_ANALYSIS_MODEL) looks at every video clip (or `items`, filmed `since` a date) for `subject` (or what a `prompt` asks for), names and describes what it finds, reads capture dates and tells which clips show the same one. With `album`, the clips are a Google Photos album the user filled: the whole album is picked through Google's picker by the local screen agent, imported (no duplicates) and only its clips are analysed — the one-call way to turn \"the clips in my album X\" into a video (then plan: true, review, showcase_build). Runs as the `autocut.analyze` engine job (progress notifications, cancellable); the report is kept in the session (showcase_inspect). With plan: true it goes on to step 2 (`autocut.makeShowcase`) and returns the plan. In the app the Showcase panel shows the job and the plan.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_analyze(
        &self,
        Parameters(p): Parameters<ShowcaseAnalyzeParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let given = |v: &Option<String>| v.as_deref().is_some_and(|t| !t.trim().is_empty());
        if !given(&p.subject) && !given(&p.prompt) {
            return Ok(fail("say what to look for: a `subject` (e.g. \"robot\") or a `prompt` (\"a video of every robot since 2020\")"));
        }
        let (id, params) = showcase_analyze_params(p);
        Ok(self.run_long(id, params, &context).await)
    }

    #[tool(
        title = "Showcase: plan the entries and the voice-over",
        description = "Step 2 of a showcase: group the analysed clips into entries (one per robot, oldest first, up to three shots each) and have the local script model write the voice-over — or, given reviewed `entries` (drop one with include: false, fold one into the one before with mergePrevious: true, fix a label or description), write it for exactly those. Runs as the `autocut.showcasePlan` engine job; returns the plan ({title, voice, mood, entries, script: {intro, lines, outro}}, lines one per entry). Review it with the user before building: wrong subjects, labels, pronunciations.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_plan(
        &self,
        Parameters(p): Parameters<ShowcasePlanParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.showcasePlan", showcase_plan_params(p), &context).await)
    }

    #[tool(
        title = "Showcase: build (and export)",
        description = "Step 3 of a showcase: build the plan (the kept one, or `plan` as edited) into a new 16:9 sequence — cold open, each entry with its narration, shots where the subject is in view, a lower third, transitions, the year cards, an on-screen counter, music ducked under the voice — all as ordinary undoable engine commands. With `export`, export it (YouTube 1080p) with chapters and a QA report beside it; with `upload`, then send the export to Google Photos (into `upload_album` too). Runs as the `autocut.showcaseBuild` engine job (progress, cancellable).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_build(
        &self,
        Parameters(p): Parameters<ShowcaseBuildParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.showcaseBuild", showcase_build_params(p), &context).await)
    }

    #[tool(
        title = "Showcase: inspect",
        description = "What the session keeps of the showcase: the analysis report (a summary, or every clip with report: true), the plan, the last build, and the latest showcase job (running or done).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn showcase_inspect(&self, Parameters(p): Parameters<ShowcaseInspectParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("autocut.showcaseInspect", json!({"report": p.report.unwrap_or(false)})).await)
    }

    #[tool(
        title = "Video templates: list",
        description = "The video templates (whole-sequence starters with media slots: photo-beat slideshow, channel intro, top-3 countdown, and the user's own from <data dir>/Video Templates): id, name, length, variants (16:9, 9:16), slots (kind, count, required). Pass `template` for one in full, with every placement's time and the beat grid. Then template_apply.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn template_list(&self, Parameters(p): Parameters<TemplateListParams>) -> Result<CallToolResult, McpError> {
        match p.template {
            Some(t) => wrap(self.run("template.inspect", json!({"template": t})).await),
            None => wrap(self.run("template.list", p.query.map(|q| json!({"query": q})).unwrap_or(json!({}))).await),
        }
    }

    #[tool(
        title = "Video templates: apply",
        description = "Make a new sequence from a video template (`template.apply`): media conforms to each slot (trimmed, looped, stretched or held; music Remixed), effects, transitions and titles stay with the slots, cuts land on the music's beats (re-measured when you give your own track; an unfilled music slot gets a generated bed), the variant sets the frame (16:9 or 9:16) and the template's look, caption style and export preset come along. Files are imported (once each). One undo step; returns the sequence, what filled each placement and how, and warnings.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn template_apply(&self, Parameters(p): Parameters<TemplateApplyParams>) -> Result<CallToolResult, McpError> {
        let mut params = json!({"template": p.template});
        if let Some(v) = p.variant {
            params["variant"] = json!(v);
        }
        if let Some(s) = p.slots {
            params["slots"] = s;
        }
        if let Some(n) = p.name {
            params["name"] = json!(n);
        }
        wrap(self.run("template.apply", params).await)
    }

    #[tool(
        title = "Render a frame",
        description = "Render the active sequence's frame (at `seconds` or the playhead) and return it as PNG.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn render_frame(&self, Parameters(p): Parameters<RenderParams>) -> Result<CallToolResult, McpError> {
        let max = p.max_side.unwrap_or(960);
        match &*self.backend {
            Backend::Headless(s) => {
                let s = s.clone();
                let r = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, AutomationError> {
                    let g = s.lock().unwrap_or_else(PoisonError::into_inner);
                    // read-only: render at `seconds` without moving the playhead (#114)
                    let t = p.seconds.map_or_else(|| g.playhead(), filmcraft_time::Tick::from_seconds_f64);
                    let (w, h) =
                        g.active_sequence().map(|q| (q.settings.width, q.settings.height)).ok_or_else(|| AutomationError::Other("no sequence".into()))?;
                    let scale = render_scale(max, w, h);
                    let img = g.try_render_program_at(scale, t).map_err(|e| AutomationError::Other(e.to_string()))?;
                    png_rgba(img.w as u32, img.h as u32, img.over_black_rgba8(), max)
                })
                .await
                .unwrap_or_else(|e| Err(join_error(e)));
                Ok(match r {
                    Ok(png) => png_result(&png, "rendered frame"),
                    Err(e) => fail(e),
                })
            }
            Backend::Bridge(b) => {
                let Some(sec) = p.seconds else {
                    return self.screenshot(&b.clone(), Some("Program".into()), max).await;
                };
                // The Program monitor shows the playhead's frame, so the live playhead has to move
                // for the screenshot; put it (and the selection, which may follow it) back after,
                // so the tool stays read-only (#114).
                let before = match self.run("sequence.inspect", json!({})).await {
                    Ok(v) => v,
                    Err(e) => return Ok(fail(e)),
                };
                if let Err(e) = self.run("playhead.set", json!({"seconds": sec})).await {
                    return Ok(fail(e));
                }
                let shot = self.screenshot(&b.clone(), Some("Program".into()), max).await;
                if let Some(t) = before.get("playhead").and_then(Value::as_i64) {
                    let _ = self.run("playhead.set", json!({"time": t})).await;
                }
                // only when it changed: selecting also leaves trim mode
                let now = self.run("sequence.inspect", json!({})).await.ok();
                if let Some(sel) = before.get("selection").filter(|v| v.is_array())
                    && now.as_ref().and_then(|n| n.get("selection")) != Some(sel)
                {
                    let _ = self.run("timeline.select", json!({"clips": sel})).await;
                }
                shot
            }
        }
    }

    #[tool(
        title = "Live app: UI state",
        description = "Live app: UI state (tool, workspace, panels, timeline zoom, playback, selection, fps).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_inspect(&self) -> Result<CallToolResult, McpError> {
        self.ui("ui.inspect", json!({})).await
    }

    #[tool(
        title = "Live app: on-screen elements",
        description = "Live app: every on-screen interactive element with id, label and rect (points). Use ids with ui_click/ui_drag.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_elements(&self, Parameters(p): Parameters<ElementsParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.elements", json!({"prefix": p.prefix.unwrap_or_default()})).await
    }

    #[tool(
        title = "Live app: click",
        description = "Live app: click an element by id or at x,y (points). Supports right/middle button, double-click (count=2) and modifiers.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_click(&self, Parameters(p): Parameters<ClickParams>) -> Result<CallToolResult, McpError> {
        let mut v = json!({"button": p.button, "count": p.count, "modifiers": p.modifiers});
        if let Some(id) = p.id {
            v["id"] = json!(id);
        } else {
            v["x"] = json!(p.x);
            v["y"] = json!(p.y);
        }
        self.ui("ui.click", v).await
    }

    #[tool(
        title = "Live app: drag",
        description = "Live app: press-drag-release from one point/element to another (move clips, trim edges, scrub, resize panels, drop project items onto tracks).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_drag(&self, Parameters(p): Parameters<DragParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.drag", json!({"from": p.from, "to": p.to, "steps": p.steps, "modifiers": p.modifiers})).await
    }

    #[tool(
        title = "Live app: press a key",
        description = "Live app: press a key or shortcut (e.g. `Space`, `L`, `Cmd+K`, `Shift+Delete`, `Cmd+Z`).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_key(&self, Parameters(p): Parameters<KeyParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.key", json!({"key": p.key})).await
    }

    #[tool(
        title = "Live app: type text",
        description = "Live app: type text into the focused field.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_type(&self, Parameters(p): Parameters<TypeParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.type", json!({"text": p.text})).await
    }

    #[tool(
        title = "Live app: screenshot",
        description = "Live app: screenshot of the window or one panel (PNG).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_screenshot(&self, Parameters(p): Parameters<ShotParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => self.screenshot(&b, p.panel, p.max_side.unwrap_or(1600)).await,
            None => Ok(fail("ui_screenshot needs bridge mode")),
        }
    }

    #[tool(
        title = "Live app: control-channel call",
        description = "Live app: call any control-channel method directly (ui.set, ui.scroll, ui.timeline.hit, ui.timeline.locate, ui.playback, ui.panel.show, ui.menu.list, …).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_control(&self, Parameters(p): Parameters<ControlParams>) -> Result<CallToolResult, McpError> {
        self.ui(&p.method, p.params.unwrap_or(json!({}))).await
    }
}

impl FilmcraftMcp {
    async fn screenshot(&self, b: &BridgeClient, panel: Option<String>, max: u32) -> Result<CallToolResult, McpError> {
        let path = std::env::temp_dir().join(format!("filmcraft-mcp-{}.png", std::process::id()));
        if let Err(e) = b.call("ui.screenshot", json!({"path": path.to_string_lossy(), "panel": panel})).await {
            return Ok(fail(e));
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => return Ok(fail(format!("screenshot: {e}"))),
        };
        let _ = std::fs::remove_file(&path);
        let bytes = match image::load_from_memory(&bytes) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                png_rgba(w, h, rgba.into_raw(), max).unwrap_or(bytes)
            }
            Err(_) => bytes,
        };
        Ok(png_result(&bytes, "screenshot of the live FilmCraft window"))
    }
}

impl FilmcraftMcp {
    /// The first argument `tool` doesn't take, as a message naming the ones it does.
    fn unknown_arg(&self, tool: &str, args: Option<&rmcp::model::JsonObject>) -> Option<String> {
        let def = self.tool_router.get(tool)?;
        let props = def.input_schema.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
        let bad = args?.keys().find(|k| !props.contains_key(*k))?;
        let accepted: Vec<&str> = props.keys().map(String::as_str).collect();
        let accepted = if accepted.is_empty() { "no arguments".to_string() } else { accepted.join(", ") };
        Some(format!("unknown argument \"{bad}\" for {tool}; expected: {accepted}"))
    }
}

const INSTRUCTIONS: &str = "FilmCraft video editor (Premiere Pro-class). Every edit is an engine command: `command_list` to discover ids/params, `command_run` to execute (undoable; `command_batch` runs several). `doc_inspect` (or `project_inspect`/`sequence_inspect`) returns ids you can pass to commands; `render_preview` shows the result. In bridge mode the `ui_*` tools drive the live app: `ui_elements` lists clickable ids, `ui_click`/`ui_drag`/`ui_key` operate it, `ui_screenshot` shows it. Time is in ticks: 254016000000 per second (commands also accept `seconds`, `frame` or `timecode`). For plain-language requests, `autocut_plan` turns a prompt into a validated command batch (dry run) and `autocut_edit` runs it; local models (Ollama) handle routine edits, and a needsSupervisor result means you plan that one yourself with the command tools. Cloud media: `googlephotos_pick` opens the Google Photos picker — the user chooses items, or pass `query` and a local vision agent (Qwen2.5-VL via Ollama, plus Screen Recording/Accessibility grants) drives the picker itself: it types the query into the picker's search, selects matching videos and presses Done (pass `queries` with `all` for many searches — a list, or one per month of a whole history — in one session and one tab, `left` returning what did not fit), or pass `album` and it picks that whole album (the user's curated seed); `googlephotos_cached`/`googlephotos_add` re-import earlier picks (after the one-time `filmcraft-cli photos auth`). `googlephotos_upload` sends a finished export to the user's Google Photos library (a long job with progress; needs `filmcraft-cli photos auth --upload` once); with `album` it also goes into that album (FilmCraft's album of that title, created when missing; needs `photos auth --albums`), and `photos.createAlbum` / `photos.albums` / `photos.addToAlbum` (command_run) manage the albums FilmCraft made. Showcases (\"make a video of every robot since 2020\"): `showcase_analyze` finds every clip of a subject or of what a `prompt` asks for (plan: true also plans; with `album`, the clips are that whole Google Photos album, picked and imported first), `showcase_plan` groups them into entries and writes the voice-over, the user reviews the plan with you, `showcase_build` builds (and exports) it; `showcase_inspect` shows what is kept.";

/// Resources: the project (as `doc_inspect`) and the command catalog (as `command_list`).
const DOCUMENT_URI: &str = "filmcraft://document";
const COMMANDS_URI: &str = "filmcraft://commands";

/// MCP 2026-07-28 clients (current Claude Code) require `ttlMs` and `cacheScope` on list and read
/// results; rmcp fills `resultType` and leaves the cache hints to the server, as the generated
/// `tools/list` does. The resource list never changes; the document does, so reads aren't cached.
fn cache_hints(context: &rmcp::service::RequestContext<rmcp::RoleServer>, ttl_ms: u64) -> Option<(u64, rmcp::model::CacheScope)> {
    context.protocol_version().is_some_and(|v| v >= rmcp::model::ProtocolVersion::V_2026_07_28).then_some((ttl_ms, rmcp::model::CacheScope::Private))
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for FilmcraftMcp {
    fn get_info(&self) -> rmcp::model::ServerConfig {
        rmcp::model::ServerConfig::new(rmcp::model::ServerCapabilities::builder().enable_tools().enable_resources().build())
            .with_server_info(rmcp::model::Implementation::new("filmcraft", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS.to_string())
    }

    /// Strict arguments and long exports in front of the tool router.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        if let Some(m) = self.unknown_arg(&request.name, request.arguments.as_ref()) {
            return Err(McpError::invalid_params(m, None));
        }
        // A blocking export reports progress and can be cancelled (docs/agents.md § Long exports).
        // Bridge mode too: the app runs it as a job, so it stays responsive (#91, #92).
        if request.name == "command_run"
            && let Some(a) = &request.arguments
            && let Some(id) = a.get("id").and_then(Value::as_str)
        {
            let params = a.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
            if crate::long_job::is_long(id, &params) {
                return Ok(self.run_long(id, params, &context).await.into());
            }
        }
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    async fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListResourcesResult, McpError> {
        use rmcp::model::Resource;
        let mut r = rmcp::model::ListResourcesResult::with_all_items(vec![
            Resource::new(DOCUMENT_URI, "document")
                .with_title("Project")
                .with_description("The project tree and the active sequence (same as doc_inspect).")
                .with_mime_type("application/json"),
            Resource::new(COMMANDS_URI, "commands")
                .with_title("Command catalog")
                .with_description("Every command with id, label, menu path, shortcut, params and enabled state (same as command_list).")
                .with_mime_type("application/json"),
        ]);
        if let Some((ttl, scope)) = cache_hints(&context, 600_000) {
            r = r.with_ttl_ms(ttl).with_cache_scope(scope);
        }
        Ok(r)
    }

    async fn read_resource(
        &self,
        request: rmcp::model::ReadResourceRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ReadResourceResponse, McpError> {
        let v = match request.uri.as_str() {
            DOCUMENT_URI => self.document().await,
            COMMANDS_URI => self.run("command.list", json!({})).await,
            other => return Err(McpError::resource_not_found(format!("resource not found: {other}"), None)),
        }
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let text = serde_json::to_string_pretty(&v).map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let mut r = rmcp::model::ReadResourceResult::new(vec![rmcp::model::ResourceContents::text(text, request.uri).with_mime_type("application/json")]);
        if let Some((ttl, scope)) = cache_hints(&context, 0) {
            r = r.with_ttl_ms(ttl).with_cache_scope(scope);
        }
        Ok(r.into())
    }
}

/// Render scale that fits the longest side of a `w` x `h` sequence into `max_side` (never upscales; 0 = no limit, as in `png_rgba`).
fn render_scale(max_side: u32, w: u32, h: u32) -> f32 {
    if max_side == 0 {
        return 1.0;
    }
    (max_side as f32 / w.max(h).max(1) as f32).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_scale_zero_means_no_limit() {
        assert_eq!(render_scale(0, 1920, 1080), 1.0);
        assert_eq!(render_scale(960, 1920, 1080), 0.5);
        assert_eq!(render_scale(960, 640, 360), 1.0);
        assert_eq!(render_scale(960, 0, 0), 1.0);
    }

    #[tokio::test]
    async fn headless_commands_and_render() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let m = FilmcraftMcp::headless(s);
        let v = m.run("sequence.addEditAllTracks", json!({"seconds": 2.0})).await.unwrap();
        assert!(v["cuts"].as_u64().unwrap() >= 2);
        let r = m.render_frame(Parameters(RenderParams { seconds: Some(1.0), max_side: Some(320) })).await.unwrap();
        assert_ne!(r.is_error, Some(true));
    }

    /// `template_list` / `template_apply`: the starters are listed, one is inspected, and applying
    /// one makes the new sequence the active one.
    #[tokio::test]
    async fn template_tools_list_and_apply() {
        let m = FilmcraftMcp::headless(Session::default());
        let l = m.template_list(Parameters(TemplateListParams::default())).await.unwrap();
        assert_ne!(l.is_error, Some(true));
        let one = m.template_list(Parameters(TemplateListParams { query: None, template: Some("builtin:top-3-listicle".into()) })).await.unwrap();
        assert_ne!(one.is_error, Some(true));
        let p = TemplateApplyParams {
            template: "builtin:top-3-listicle".into(),
            variant: Some("9:16".into()),
            slots: Some(json!({"title": "Best Bots", "labels": ["A", "B", "C"]})),
            name: Some("Countdown".into()),
        };
        let r = m.template_apply(Parameters(p)).await.unwrap();
        assert_ne!(r.is_error, Some(true), "{r:?}");
        let q = m.run("sequence.inspect", json!({})).await.unwrap();
        assert_eq!(q["settings"]["width"], 1080, "{q}");
        let bad = m.template_apply(Parameters(TemplateApplyParams { template: "nope".into(), variant: None, slots: None, name: None })).await.unwrap();
        assert_eq!(bad.is_error, Some(true));
    }

    /// `render_frame` / `render_preview` are annotated read-only (#114): rendering at `seconds`
    /// leaves the playhead, and the selection that follows it, where they were, and still renders
    /// the frame at `seconds`.
    #[tokio::test]
    async fn rendering_at_a_time_does_not_move_the_playhead() {
        let m = FilmcraftMcp::headless(demo());
        m.run("sequence.selectionFollowsPlayhead", json!({"on": true})).await.unwrap();
        m.run("playhead.set", json!({"seconds": 1.0})).await.unwrap();
        let before = m.run("sequence.inspect", json!({})).await.unwrap();
        let png = |r: CallToolResult| serde_json::to_string(&r.content).unwrap();
        let at_playhead = png(m.render_frame(Parameters(RenderParams { seconds: None, max_side: Some(64) })).await.unwrap());
        for seconds in [5.0, 8.0] {
            let r = m.render_frame(Parameters(RenderParams { seconds: Some(seconds), max_side: Some(64) })).await.unwrap();
            assert_ne!(png(r), at_playhead, "renders the frame at {seconds} s");
            m.render_preview(Parameters(RenderParams { seconds: Some(seconds), max_side: Some(64) })).await.unwrap();
        }
        let after = m.run("sequence.inspect", json!({})).await.unwrap();
        assert_eq!(after["playhead"], before["playhead"]);
        assert_eq!(after["selection"], before["selection"]);
    }

    /// One request/response exchange over `serve_io`, as an MCP client sees it.
    struct Client {
        tx: tokio::io::DuplexStream,
        rx: tokio::io::Lines<tokio::io::BufReader<tokio::io::DuplexStream>>,
    }

    impl Client {
        fn start(s: Session) -> Self {
            use tokio::io::AsyncBufReadExt;
            let (tx, server_in) = tokio::io::duplex(1 << 22);
            let (server_out, rx) = tokio::io::duplex(1 << 22);
            tokio::spawn(FilmcraftMcp::headless(s).serve_io(server_in, server_out));
            Self { tx, rx: tokio::io::BufReader::new(rx).lines() }
        }
        async fn send(&mut self, line: &str) {
            use tokio::io::AsyncWriteExt;
            self.tx.write_all(format!("{line}\n").as_bytes()).await.unwrap();
        }
        async fn next(&mut self) -> Value {
            serde_json::from_str(&self.rx.next_line().await.unwrap().unwrap()).unwrap()
        }
        async fn ask(&mut self, msg: Value) -> Value {
            self.send(&msg.to_string()).await;
            self.next().await
        }
        async fn init(&mut self) -> Value {
            let r = self.ask(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})).await;
            self.send(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string()).await;
            r
        }
        async fn call(&mut self, id: u64, name: &str, arguments: Value) -> Value {
            self.ask(json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":arguments}})).await
        }
    }

    /// `googlephotos_pick` exposes the optional autopick `query`; its schema stays offline-checkable.
    #[test]
    fn googlephotos_pick_schema_has_query() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("googlephotos_pick") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        let Some(props) = schema["properties"].as_object() else { panic!("schema: {schema}") };
        assert!(props.contains_key("query"), "properties: {props:?}");
        assert!(props.contains_key("all") && props.contains_key("since"), "properties: {props:?}");
        assert!(props.contains_key("session"));
        assert!(props.contains_key("max_items"));
        assert!(props.contains_key("album") && props.contains_key("videos_only"), "properties: {props:?}");
        assert_eq!(schema["properties"]["queries"]["type"], json!(["array", "null"]), "{schema}");
        assert!(schema["required"].as_array().is_none_or(|r| r.is_empty()), "{schema}");
    }

    /// `query` and `album` are two different picks: both at once is refused before anything
    /// (even the sign-in) is looked at.
    #[tokio::test]
    async fn googlephotos_pick_takes_a_query_or_an_album() {
        let mut c = Client::start(Session::default());
        c.init().await;
        let r = c.call(1, "googlephotos_pick", json!({"query": "robots", "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not both"), "{r}");
        // many searches and an album: the same refusal
        let r = c.call(2, "googlephotos_pick", json!({"queries": ["videos October 2022", "videos November 2022"], "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not both"), "{r}");
        // too many searches for one call: refused before the sign-in is looked at
        let many: Vec<String> = (0..=MAX_PICK_QUERIES).map(|i| format!("q{i}")).collect();
        let r = c.call(3, "googlephotos_pick", json!({"queries": many})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("at most"), "{r}");
    }

    /// `googlephotos_upload` takes a required `path` and an optional `description` and `album`,
    /// and runs `photos.upload` as a long job (progress, cancel).
    #[test]
    fn googlephotos_upload_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("googlephotos_upload") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        let Some(props) = schema["properties"].as_object() else { panic!("schema: {schema}") };
        assert_eq!(props.keys().map(String::as_str).collect::<Vec<_>>(), ["album", "description", "path"], "{schema}");
        assert_eq!(schema["properties"]["path"]["type"], "string", "{schema}");
        assert_eq!(schema["required"], json!(["path"]), "{schema}");
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4", "description": "Robots"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "description": "Robots", "wait": true}));
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4", "album": "Robot showcase"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "album": "Robot showcase", "wait": true}));
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "wait": true}));
        assert!(serde_json::from_value::<PhotosUploadParams>(json!({"description": "no path"})).is_err());
        assert!(crate::long_job::is_long("photos.upload", &json!({"wait": true})));
    }

    /// Over MCP, an upload is the engine job: the media item comes back in the answer, and a
    /// host without an uploader (or without consent) gets the reason, not a crash.
    #[tokio::test]
    async fn googlephotos_upload_runs_the_engine_job() {
        struct Fake;
        impl filmcraft_engine::photos::PhotosUploader for Fake {
            fn ready(&self) -> Result<(), String> {
                Ok(())
            }
            fn upload(&self, _: &str, d: Option<&str>, p: &filmcraft_engine::export::Progress) -> Result<Value, String> {
                p.done.store(p.total.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                Ok(json!({"mediaItemId": "M1", "description": d}))
            }
        }
        let file = std::env::temp_dir().join(format!("fc-mcp-upload-{}.mp4", std::process::id()));
        std::fs::write(&file, b"0123456789").unwrap();
        let path = file.to_string_lossy().into_owned();

        let mut c = Client::start(Session::default());
        c.init().await;
        let r = c.call(1, "googlephotos_upload", json!({"path": path})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not available"), "{r}");
        let r = c.call(2, "googlephotos_upload", json!({"path": path, "folder": "x"})).await;
        assert!(r["error"]["message"].as_str().unwrap().contains("unknown argument \"folder\""), "{r}");

        let mut s = Session::default();
        s.photos = Some(std::sync::Arc::new(Fake));
        let mut c = Client::start(s);
        c.init().await;
        let r = c.call(3, "googlephotos_upload", json!({"path": path, "description": "Robots"})).await;
        let text = r["result"]["content"][0]["text"].as_str().unwrap_or_default();
        let v: Value = serde_json::from_str(text).unwrap_or_else(|_| panic!("{r}"));
        assert_eq!(v["upload"], json!({"mediaItemId": "M1", "description": "Robots"}), "{v}");
        assert_eq!((v["result"]["bytes"].as_u64(), v["path"].as_str()), (Some(10), Some(path.as_str())), "{v}");
        let r = c.call(4, "googlephotos_upload", json!({"path": "/no/such/export.mp4"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        // an album needs an uploader that does albums: the reason, not a crash
        let r = c.call(5, "googlephotos_upload", json!({"path": path, "album": "Robots"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("albums are not available"), "{r}");
        assert!(file.exists(), "the uploaded file is left alone");
        std::fs::remove_file(&file).ok();
    }

    /// An upload reports byte progress and stops on `notifications/cancelled`; unlike an
    /// export's partial output, the file being uploaded is the user's and stays.
    #[tokio::test(flavor = "multi_thread")]
    async fn googlephotos_upload_progress_and_cancel() {
        struct Slow;
        impl filmcraft_engine::photos::PhotosUploader for Slow {
            fn ready(&self) -> Result<(), String> {
                Ok(())
            }
            fn upload(&self, _: &str, _: Option<&str>, p: &filmcraft_engine::export::Progress) -> Result<Value, String> {
                use std::sync::atomic::Ordering;
                for i in 1..=4000u64 {
                    if p.cancel.load(Ordering::Relaxed) {
                        return Err("cancelled".into());
                    }
                    p.done.store(i.min(p.total.load(Ordering::Relaxed)), Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err("never cancelled".into())
            }
        }
        let file = std::env::temp_dir().join(format!("fc-mcp-upload-cancel-{}.mp4", std::process::id()));
        std::fs::write(&file, vec![1u8; 100_000]).unwrap();
        let mut s = Session::default();
        s.photos = Some(std::sync::Arc::new(Slow));
        let mut c = Client::start(s);
        c.init().await;
        let call = json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"googlephotos_upload","arguments":{"path":file.to_string_lossy()},"_meta":{"progressToken":"up"}}});
        c.send(&call.to_string()).await;
        loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "finished before the cancel: {m}");
            if m["method"] == "notifications/progress" {
                assert_eq!((m["params"]["progressToken"].as_str(), m["params"]["total"].as_f64()), (Some("up"), Some(100_000.0)), "{m}");
                break;
            }
        }
        c.send(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":4}}"#).await;
        let t0 = std::time::Instant::now();
        let last = loop {
            assert!(t0.elapsed().as_secs() < 30, "the upload never stopped");
            c.send(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"jobs.list"}}}"#).await;
            let m = loop {
                let m = c.next().await;
                assert_ne!(m["id"], 4, "no response for a cancelled request: {m}");
                if m["id"] == 5 {
                    break m;
                }
            };
            let jobs: Value = serde_json::from_str(m["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
            let last = jobs.as_array().and_then(|a| a.last()).cloned().unwrap_or_default();
            if last["finished"] == true {
                break last;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        };
        assert_eq!(last["result"]["error"], "cancelled", "{last}");
        // give a (wrong) partial-file cleanup its chance, then check the file is still there
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(file.exists(), "cancelling an upload must not delete the user's file");
        std::fs::remove_file(&file).ok();
    }

    /// The schema of a tool, offline.
    fn schema(tool: &str) -> Value {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(t) = m.tool_router.get(tool) else { panic!("{tool} missing") };
        Value::Object((*t.input_schema).clone())
    }

    fn props(schema: &Value) -> Vec<String> {
        let mut p: Vec<String> = schema["properties"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        p.sort();
        p
    }

    /// The showcase tools' arguments and the engine jobs they run (as long jobs: progress, cancel).
    #[test]
    fn showcase_tool_schemas_and_params() {
        let s = schema("showcase_analyze");
        assert_eq!(props(&s), ["album", "compare", "frames", "fresh", "items", "plan", "prompt", "screen", "since", "subject", "title", "voice"], "{s}");
        assert!(s["required"].as_array().is_none_or(|r| r.is_empty()), "a subject or a prompt: {s}");
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "since": "2020", "items": [5, 6]})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.analyze", json!({"subject": "robot", "since": "2020", "items": [5, 6], "wait": true})));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "plan": true, "title": "Robots", "voice": ""})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.makeShowcase", json!({"subject": "robot", "title": "Robots", "voice": "", "wait": true})));
        // title and voice only matter with a plan
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "title": "Robots"})).unwrap();
        assert!(showcase_analyze_params(p).1.get("title").is_none());
        // a Google Photos album as the seed, with the video asked for in words
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"prompt": "every robot since 2020", "album": "Robot showcase", "subject": " "})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.analyze", json!({"prompt": "every robot since 2020", "album": "Robot showcase", "wait": true})));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "fresh": true})).unwrap();
        assert_eq!(showcase_analyze_params(p).1, json!({"subject": "robot", "fresh": true, "wait": true}));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "fresh": false})).unwrap();
        assert!(showcase_analyze_params(p).1.get("fresh").is_none());

        let s = schema("showcase_plan");
        assert_eq!(props(&s), ["entries", "min_coverage", "mood", "subtitle", "title", "voice"], "{s}");
        assert!(s["required"].as_array().is_none_or(|r| r.is_empty()), "{s}");
        let p: ShowcasePlanParams = serde_json::from_value(json!({"title": "T", "min_coverage": 0.5, "entries": [{"label": "x"}]})).unwrap();
        assert_eq!(showcase_plan_params(p), json!({"title": "T", "minCoverage": 0.5, "entries": [{"label": "x"}], "wait": true}));

        let s = schema("showcase_build");
        assert_eq!(props(&s), ["export", "plan", "qa", "upload", "upload_album"], "{s}");
        let p: ShowcaseBuildParams = serde_json::from_value(json!({"export": "/x/robots.mp4", "upload": true})).unwrap();
        assert_eq!(showcase_build_params(p), json!({"export": "/x/robots.mp4", "upload": true, "wait": true}));
        let p: ShowcaseBuildParams = serde_json::from_value(json!({"export": "/x/robots.mp4", "upload": true, "upload_album": "Showcases"})).unwrap();
        assert_eq!(showcase_build_params(p), json!({"export": "/x/robots.mp4", "upload": true, "uploadAlbum": "Showcases", "wait": true}));
        assert_eq!(showcase_build_params(ShowcaseBuildParams::default()), json!({"upload": false, "wait": true}));

        assert_eq!(props(&schema("showcase_inspect")), ["report"]);
        for id in ["autocut.analyze", "autocut.showcasePlan", "autocut.makeShowcase", "autocut.showcaseBuild"] {
            assert!(crate::long_job::is_long(id, &json!({"wait": true})), "{id}");
        }
        assert!(crate::long_job::JOB_EXTRAS.contains(&"showcase"));
    }

    /// A demo session that runs showcase jobs with scripted models.
    fn showcase_demo() -> Session {
        let mut s = demo();
        let host = filmcraft_autocut::host::ShowcaseHost::with_models(filmcraft_autocut::host::Models::scripted());
        s.autocut.host = Some(std::sync::Arc::new(host));
        s
    }

    fn tool_json(r: &Value) -> Value {
        let text = r["result"]["content"][0]["text"].as_str().unwrap_or_default();
        serde_json::from_str(text).unwrap_or_else(|_| panic!("{r}"))
    }

    /// Analyse, plan, review and build over MCP: the jobs run in the background (the port's pump
    /// thread runs their commands) while the server reports progress and keeps answering.
    #[tokio::test(flavor = "multi_thread")]
    async fn showcase_tools_analyze_plan_and_build() {
        let mut c = Client::start(showcase_demo());
        c.init().await;
        let call = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"showcase_analyze","arguments":{"subject":"robot","plan":true,"title":"Robots","voice":""},"_meta":{"progressToken":"sc"}}});
        c.send(&call.to_string()).await;
        let (mut notes, mut pinged, mut pong) = (0, false, false);
        let r = loop {
            let m = c.next().await;
            if m["method"] == "notifications/progress" {
                notes += 1;
                if !pinged {
                    pinged = true;
                    c.send(&json!({"jsonrpc": "2.0", "id": 99, "method": "ping"}).to_string()).await;
                }
            } else if m["id"] == 99 {
                pong = true;
            } else if m["id"] == 1 {
                break m;
            }
        };
        assert_eq!(r["result"]["isError"], false, "{r}");
        let v = tool_json(&r);
        let plan = v["showcase"]["plan"].clone();
        assert_eq!(plan["entries"].as_array().map(Vec::len), Some(6), "{v}");
        assert!(notes >= 1, "progress was reported");
        if pinged && !pong {
            pong = c.next().await["id"] == 99;
        }
        assert!(!pinged || pong, "the server answered while the job ran");

        let r = c.call(2, "showcase_inspect", json!({})).await;
        let v = tool_json(&r);
        assert_eq!((v["report"]["showing"].as_u64(), v["planRevision"].as_u64()), (Some(6), Some(1)), "{v}");

        // review: keep two robots, write the words for them
        let mut entries = plan["entries"].as_array().cloned().unwrap();
        for e in entries.iter_mut().skip(2) {
            e["include"] = json!(false);
        }
        let r = c.call(3, "showcase_plan", json!({"entries": entries})).await;
        let v = tool_json(&r);
        let mut plan = v["showcase"]["plan"].clone();
        assert_eq!(plan["title"], "Robots", "{v}");
        plan["width"] = json!(320);
        plan["height"] = json!(180);
        let r = c.call(4, "showcase_build", json!({"plan": plan})).await;
        let v = tool_json(&r);
        assert_eq!(v["showcase"]["built"]["entries"], 2, "{v}");
        let r = c.call(5, "showcase_analyze", json!({"subject": " "})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        // an album needs a host that picks albums: this one says so before any job starts
        let r = c.call(7, "showcase_analyze", json!({"prompt": "a video of every robot", "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("album") && r.to_string().contains("not available"), "{r}");
        let r = c.call(6, "showcase_build", json!({"plan": {"entries": []}})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("no entry is included"), "{r}");
    }

    /// The AutoCut prompt "make a video of every robot" starts the showcase job over MCP too,
    /// and its plan comes back in the report.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_showcase_prompt_over_mcp() {
        let m = FilmcraftMcp::headless(showcase_demo());
        let r = m.autocut_edit(Parameters(AutocutParams { prompt: "make a video of every robot".into(), frames: None, dry_run: None })).await.unwrap();
        assert_ne!(r.is_error, Some(true), "{r:?}");
        let text = serde_json::to_value(&r.content).unwrap()[0]["text"].as_str().unwrap_or_default().to_string();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["intent"], "showcase", "{v}");
        assert_eq!(v["results"][0]["result"]["showcase"]["plan"]["entries"].as_array().map(Vec::len), Some(6), "{v}");
    }

    /// #499: `media_import` takes `paths` (a required array), and forwards `bin` / `image_sequence` to `file.import`.
    #[test]
    fn media_import_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let def = m.tool_router.get("media_import").unwrap();
        let schema = Value::Object((*def.input_schema).clone());
        assert_eq!(schema["properties"]["paths"]["type"], "array", "{schema}");
        assert!(schema["required"].as_array().unwrap().iter().any(|r| r == "paths"), "{schema}");
        let p: MediaImportParams = serde_json::from_value(json!({"paths": ["/a.png"], "bin": 7, "image_sequence": true})).unwrap();
        assert_eq!(media_import_params(p), json!({"paths": ["/a.png"], "bin": 7, "imageSequence": true}));
        let p: MediaImportParams = serde_json::from_value(json!({"paths": ["/a.wav"]})).unwrap();
        assert_eq!(media_import_params(p), json!({"paths": ["/a.wav"]}));
        assert!(serde_json::from_value::<MediaImportParams>(json!({"text": "/a.wav"})).is_err());
    }

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    /// Every listed tool has a title and the four hints, and the core tools are listed.
    #[tokio::test(flavor = "multi_thread")]
    async fn tools_are_titled_and_annotated() {
        let mut c = Client::start(demo());
        let init = c.init().await;
        assert_eq!(init["result"]["serverInfo"]["name"], "filmcraft");
        let tools = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})).await["result"]["tools"].as_array().cloned().unwrap();
        for t in &tools {
            assert!(t["title"].is_string(), "{} has no title", t["name"]);
            for h in ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] {
                assert!(t["annotations"][h].is_boolean(), "{} lacks {h}", t["name"]);
            }
        }
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        for want in [
            "command_list",
            "command_run",
            "command_batch",
            "doc_inspect",
            "render_preview",
            "project_inspect",
            "render_frame",
            "ui_click",
            "autocut_plan",
            "autocut_edit",
            "showcase_analyze",
            "showcase_plan",
            "showcase_build",
            "showcase_inspect",
            "template_list",
            "template_apply",
        ] {
            assert!(names.contains(&want), "{want} not listed");
        }
        let ro = |n: &str| tools.iter().find(|t| t["name"] == n).map(|t| t["annotations"]["readOnlyHint"].clone());
        assert_eq!((ro("command_list"), ro("command_run")), (Some(json!(true)), Some(json!(false))));
    }

    /// Strict arguments, malformed JSON, batch, doc_inspect and render_preview over the raw line
    /// protocol.
    #[tokio::test(flavor = "multi_thread")]
    async fn strict_arguments_and_parse_errors() {
        let mut c = Client::start(demo());
        c.init().await;
        let r = c.call(1, "command_list", json!({"filter": "razor", "enabled_only": true})).await;
        assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("timeline.razor"), "{r}");
        // an unknown argument names itself and the accepted ones
        let r = c.call(2, "command_list", json!({"filtr": "x"})).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        let m = r["error"]["message"].as_str().unwrap();
        assert!(m.contains("\"filtr\"") && m.contains("filter") && m.contains("enabled_only"), "{m}");
        let r = c.call(3, "doc_inspect", json!({"verbose": true})).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        // command params stay free-form
        let r = c.call(4, "command_run", json!({"id": "playhead.set", "params": {"seconds": 0.5}})).await;
        assert_eq!(r["result"]["isError"], false, "{r}");
        // malformed JSON: a parse error, and the server keeps serving
        c.send("{\"jsonrpc\": \"2.0\", \"id\": 5, BROKEN").await;
        let r = c.next().await;
        assert_eq!((r["error"]["code"].as_i64(), r["id"].is_null()), (Some(-32700), true), "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":6,"method":"ping"})).await;
        assert!(r["result"].is_object(), "{r}");
        // batch: stops at the first failure and reports both
        let r =
            c.call(7, "command_batch", json!({"steps": [{"id": "playhead.set", "params": {"seconds": 0.5}}, {"id": "no.such"}, {"id": "playhead.set"}]})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        let v: Value = serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!((v["completed"].as_u64(), v["failed"].as_u64(), v["results"].as_array().map(Vec::len)), (Some(1), Some(1), Some(2)), "{v}");
        let r = c.call(8, "doc_inspect", json!({})).await;
        let v: Value = serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert!(v["project"].is_object() && v["sequence"].is_object(), "{v}");
        let r = c.call(9, "render_preview", json!({"max_side": 64})).await;
        assert_eq!(r["result"]["content"][0]["type"], "image", "{r}");
    }

    /// `filmcraft://document` and `filmcraft://commands` are listed and readable as JSON.
    #[tokio::test(flavor = "multi_thread")]
    async fn resources_document_and_commands() {
        let mut c = Client::start(demo());
        let init = c.init().await;
        assert!(init["result"]["capabilities"]["resources"].is_object(), "{init}");
        assert!(init["result"]["instructions"].as_str().unwrap().contains("doc_inspect"), "{init}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"resources/list"})).await;
        let uris: Vec<&str> = r["result"]["resources"].as_array().unwrap().iter().filter_map(|x| x["uri"].as_str()).collect();
        assert_eq!(uris, [DOCUMENT_URI, COMMANDS_URI], "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":DOCUMENT_URI}})).await;
        assert_eq!(r["result"]["contents"][0]["mimeType"], "application/json", "{r}");
        let v: Value = serde_json::from_str(r["result"]["contents"][0]["text"].as_str().unwrap()).unwrap();
        assert!(v["project"].is_object() && v["sequence"].is_object(), "{v}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":COMMANDS_URI}})).await;
        assert!(r["result"]["contents"][0]["text"].as_str().unwrap().contains("timeline.razor"), "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"filmcraft://nope"}})).await;
        assert!(r["error"]["code"].is_i64(), "{r}");
    }

    /// MCP 2026-07-28 clients (current Claude Code) negotiate per request and reject list and read
    /// results without `resultType`, `ttlMs` and `cacheScope`; sessions that negotiated an older
    /// revision through `initialize` get the old shape.
    #[tokio::test(flavor = "multi_thread")]
    async fn modern_clients_get_result_type_and_cache_hints() {
        let mut c = Client::start(Session::default());
        let meta = json!({
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        });
        let r = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":meta}})).await;
        assert!(r["result"]["supportedVersions"].as_array().unwrap().contains(&json!("2026-07-28")), "{r}");
        for (id, method, params) in [
            (2, "tools/list", json!({"_meta": meta})),
            (3, "resources/list", json!({"_meta": meta})),
            (4, "resources/read", json!({"_meta": meta, "uri": DOCUMENT_URI})),
        ] {
            let r = c.ask(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})).await;
            assert_eq!(r["result"]["resultType"], "complete", "{method}: {r}");
            assert!(r["result"]["ttlMs"].is_u64(), "{method}: {r}");
            assert!(matches!(r["result"]["cacheScope"].as_str(), Some("public" | "private")), "{method}: {r}");
        }
        let mut c = Client::start(Session::default());
        c.init().await;
        for (id, method, params) in [(2, "tools/list", json!({})), (3, "resources/list", json!({})), (4, "resources/read", json!({"uri": DOCUMENT_URI}))] {
            let r = c.ask(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})).await;
            assert!(r["result"].is_object() && r["result"].get("resultType").is_none() && r["result"].get("ttlMs").is_none(), "{method}: {r}");
        }
    }

    /// docs/agents.md § Long exports: a blocking export reports `notifications/progress` for its
    /// token (none without one), answers `ping` meanwhile, and `notifications/cancelled` stops
    /// it, deletes the partial file and sends no response.
    #[tokio::test(flavor = "multi_thread")]
    async fn export_progress_and_cancel() {
        let dir = std::env::temp_dir().join(format!("filmcraft-mcp-progress-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut c = Client::start(demo());
        c.init().await;
        let export = |id: u64, name: &str, token: Option<Value>, seconds: f64| {
            let path = dir.join(name).to_string_lossy().to_string();
            let params = json!({"path": path, "format": "h264", "width": 640, "height": 360, "audio": false, "range": "custom", "startSeconds": 0, "endSeconds": seconds, "wait": true});
            let mut call =
                json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"file.exportMedia","params":params}}});
            if let Some(t) = token {
                call["params"]["_meta"] = json!({"progressToken": t});
            }
            call.to_string()
        };
        // no token: no notifications, the same response as before
        c.send(&export(1, "quiet.mp4", None, 0.5)).await;
        let m = c.next().await;
        assert_eq!((m["id"].as_u64(), &m["result"]["isError"]), (Some(1), &json!(false)), "{m}");
        // progress, strictly increasing, with a total; ping answered while the export runs. The
        // job is polled every 100 ms, so a fast machine finishes a short export within one poll:
        // a longer one is tried until the export outlasts a few polls.
        let (mut notes, mut pong) = (0, false);
        for (attempt, seconds) in [1.0, 4.0, 16.0].into_iter().enumerate() {
            let (id, ping, name) = (20 + attempt as u64, 30 + attempt as u64, format!("a{attempt}.mp4"));
            c.send(&export(id, &name, Some(json!("tok")), seconds)).await;
            let (mut last, mut pinged) = (-1.0, false);
            (notes, pong) = (0, false);
            let done = loop {
                let m = c.next().await;
                if m["method"] == "notifications/progress" {
                    assert_eq!(m["params"]["progressToken"], "tok", "{m}");
                    let p = m["params"]["progress"].as_f64().unwrap();
                    assert!(p > last && m["params"]["total"].as_f64().unwrap() >= p, "{m}");
                    last = p;
                    notes += 1;
                    if !pinged {
                        pinged = true;
                        c.send(&json!({"jsonrpc": "2.0", "id": ping, "method": "ping"}).to_string()).await;
                    }
                } else if m["id"] == ping {
                    pong = true;
                } else if m["id"] == id {
                    break m;
                }
            };
            assert_eq!(done["result"]["isError"], false, "{done}");
            assert!(notes >= 1, "a finished export reports its progress at least once");
            assert!(dir.join(&name).exists());
            if notes >= 2 && pong {
                break;
            }
            // the ping's answer is still on its way: read it so it cannot be taken for a later reply
            while pinged && !pong {
                pong = c.next().await["id"] == ping;
            }
            pong = false;
        }
        assert!(notes >= 2 && pong, "{notes} notifications, ping answered during the export: {pong}");
        // cancel after the first notification: no response, no partial file
        c.send(&export(4, "b.mp4", Some(json!(7)), 20.0)).await;
        loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "finished before the cancel: {m}");
            if m["method"] == "notifications/progress" {
                break;
            }
        }
        c.send(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":4}}"#).await;
        let t0 = std::time::Instant::now();
        while dir.join("b.mp4").exists() || t0.elapsed() < std::time::Duration::from_millis(500) {
            assert!(t0.elapsed().as_secs() < 30, "partial file still there");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        c.send(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"jobs.list"}}}"#).await;
        let m = loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "no response for a cancelled request: {m}");
            if m["id"] == 5 {
                break m;
            }
        };
        let jobs: Value = serde_json::from_str(m["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let last = jobs.as_array().and_then(|a| a.last()).cloned().unwrap_or_default();
        assert_eq!(last["result"]["error"], "cancelled", "{jobs}");
        assert!(!dir.join("b.mp4").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A command that panicked used to poison the session lock, and every later call failed with
    /// "session lock poisoned" until the server was restarted.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_poisoned_session_lock_keeps_serving() {
        let m = FilmcraftMcp::headless(demo());
        let Backend::Headless(s) = &*m.backend else { panic!("headless backend") };
        let s = s.clone();
        let _ = std::thread::spawn(move || {
            let _g = s.lock().unwrap();
            panic!("a command panicked");
        })
        .join();
        let v = m.run("sequence.inspect", json!({})).await.unwrap();
        assert!(v.is_object(), "{v}");
        let r = m.render_frame(Parameters(RenderParams { seconds: Some(0.5), max_side: Some(64) })).await.unwrap();
        assert_ne!(r.is_error, Some(true));
    }
}
