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
    /// With `query`: select every search result (the agent clicks the first tile and
    /// Shift+clicks the last) instead of up to max_items tiles.
    #[serde(default)]
    pub all: Option<bool>,
    /// Skip picked items created before this date (YYYY-MM-DD); they are not downloaded.
    #[serde(default)]
    pub since: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PhotosAddParams {
    /// Media item ids of already-downloaded items (from googlephotos_pick or googlephotos_cached).
    pub ids: Vec<String>,
}

/// The most ids one googlephotos_add call takes.
const MAX_ADD_IDS: usize = 2000;

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

impl FilmcraftMcp {
    pub fn headless(session: Session) -> Self {
        Self { backend: Arc::new(Backend::Headless(Arc::new(Mutex::new(session)))), tool_router: Self::tool_router() }
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
        description = "Open the Google Photos picker in the user's browser, wait for a selection and press Done, download the picks (cached under the app's data dir) and import them into the open project with file.import. Google offers apps no server-side search of the library (retired 2025-03-31): the choice happens in the picker. Either the user chooses (tell them what to pick), or — pass `query` — a local vision agent (Qwen2.5-VL via Ollama) drives the picker itself: it types the query into the picker's own search, selects matching video tiles and presses Done; needs `ollama pull qwen2.5vl:7b` plus Screen Recording and Accessibility for this process, and its report comes back as `autopick`. If the wait runs out the result is `status: pending` with a `session` and `pickerUri`; call again with that `session` to keep waiting (with short client tool timeouts, start with wait_seconds 0, then resume). Videos download whole, so they can take a while. One-time setup: `filmcraft-cli photos auth` (see crates/gphotos/README.md).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn googlephotos_pick(&self, Parameters(p): Parameters<PhotosPickParams>) -> Result<CallToolResult, McpError> {
        let gp = match gphotos() {
            Ok(gp) => gp,
            Err(e) => return Ok(fail(e)),
        };
        let since = p.since.filter(|d| !d.trim().is_empty());
        if let Some(d) = since.as_deref().filter(|d| !filmcraft_gphotos::is_day(d)) {
            return Ok(fail(format!("`since` wants a date like 2020-01-01, not `{d}`")));
        }
        let opts = filmcraft_gphotos::PickOptions {
            max_items: p.max_items,
            wait: std::time::Duration::from_secs(p.wait_seconds.unwrap_or(if p.query.is_some() { 900 } else { 300 })),
            resume: p.session.filter(|s| !s.trim().is_empty()),
            open_browser: true,
            since,
        };
        let query = p.query.filter(|q| !q.trim().is_empty());
        let max_for_agent = p.max_items.unwrap_or(6);
        let select_all = p.all.unwrap_or(false);
        // blocking HTTP and sleeps: keep them off the async threads
        let outcome = match tokio::task::spawn_blocking(move || {
            let prepared = gp.prepare(&opts, &mut |_| {})?;
            let agent =
                query.map(|query| filmcraft_screenagent::agent::spawn(filmcraft_screenagent::agent::Request { query, max_items: max_for_agent, select_all }));
            gp.complete(prepared, opts.wait, &mut |_| {}).map(|o| (o, agent))
        })
        .await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Ok(fail(e)),
            Err(e) => return Ok(fail(format!("internal error: {e}"))),
        };
        let (outcome, agent) = outcome;
        let mut autopick: Value = Value::Null;
        let picked = match outcome {
            filmcraft_gphotos::PickOutcome::Done(picked) => {
                if let Some(h) = agent {
                    autopick = match h.join() {
                        Ok(Ok(s)) => serde_json::to_value(&s).unwrap_or(Value::Null),
                        Ok(Err(e)) => json!({"error": e}),
                        Err(_) => json!({"error": "the autopick agent thread died"}),
                    };
                }
                picked
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
        let paths: Vec<String> = picked.fetched.iter().map(|f| f.path.to_string_lossy().into_owned()).collect();
        let cached = picked.fetched.iter().filter(|f| f.was_cached).count();
        let items: Vec<Value> = picked
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
        let imported = self.run("file.import", json!({"paths": paths})).await;
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
        description = "Import already-downloaded Google Photos items (ids from googlephotos_pick or googlephotos_cached) into the open project with file.import. Items never picked are not reachable: Google shares only what the user picks, so use googlephotos_pick for new ones.",
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
        wrap(self.run("file.import", json!({"paths": paths})).await.map(|imported| json!({"imported": imported, "paths": paths})))
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

const INSTRUCTIONS: &str = "FilmCraft video editor (Premiere Pro-class). Every edit is an engine command: `command_list` to discover ids/params, `command_run` to execute (undoable; `command_batch` runs several). `doc_inspect` (or `project_inspect`/`sequence_inspect`) returns ids you can pass to commands; `render_preview` shows the result. In bridge mode the `ui_*` tools drive the live app: `ui_elements` lists clickable ids, `ui_click`/`ui_drag`/`ui_key` operate it, `ui_screenshot` shows it. Time is in ticks: 254016000000 per second (commands also accept `seconds`, `frame` or `timecode`). For plain-language requests, `autocut_plan` turns a prompt into a validated command batch (dry run) and `autocut_edit` runs it; local models (Ollama) handle routine edits, and a needsSupervisor result means you plan that one yourself with the command tools. Cloud media: `googlephotos_pick` opens the Google Photos picker — the user chooses items, or pass `query` and a local vision agent (Qwen2.5-VL via Ollama, plus Screen Recording/Accessibility grants) drives the picker itself: it types the query into the picker's search, selects matching videos and presses Done; `googlephotos_cached`/`googlephotos_add` re-import earlier picks (after the one-time `filmcraft-cli photos auth`).";

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
        assert!(schema["required"].as_array().is_none_or(|r| r.is_empty()), "{schema}");
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
