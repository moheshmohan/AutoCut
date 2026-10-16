//! AutoCut showcases in the app: find every clip that shows a subject, plan a counted, narrated
//! showcase of them (entries and voice-over), review it, build it — as engine commands, so the
//! Showcase panel, the CLI, the control channel and MCP agents share one path.
//!
//! | command | does |
//! |---|---|
//! | `autocut.analyze` | job: which clips show `subject` (of `items`, filmed `since` a date), what it is, when; the report is kept in the session |
//! | `autocut.showcasePlan` | job: group the report's clips into entries (or take reviewed `entries`) and write the voice-over; the plan is kept |
//! | `autocut.makeShowcase` | job: both, one after the other (Sequence ▸ Make a Showcase…, and AutoCut prompts like "make a video of every robot since 2020") |
//! | `autocut.showcaseBuild` | job: build the plan (the kept one, or `plan`) into a new sequence; `export` it (YouTube 1080p, chapters and a QA report beside it) and `upload` the export to Google Photos |
//! | `autocut.showcaseInspect` | the kept report (a summary, or all of it with `report: true`), plan and last build, and the latest job |
//! | `autocut.showcaseSet` | keep a `report` or `plan` from elsewhere (or `reportPath` / `planPath`: the CLI's `analyze --out` and `showcase --script-out` files); `null` clears one |
//!
//! A plan is `{subject, title, subtitle?, voice?, mood, entries, script: {intro, lines, outro}}`;
//! each entry is the CLI plan's entry (`number`, `label`, `kind`, `description`, `unix`, `shots`)
//! plus the review's `include` (default true) and `mergePrevious` (default false), and
//! `script.lines` follows `entries` one to one (excluded and merged-in entries' lines unused).
//!
//! The engine only runs the jobs and keeps what they make. The work — vision and script models,
//! grouping, the build — lives in `filmcraft-autocut`, a layer above, which the host (desktop
//! app, CLI, MCP server) installs as [`AutoCut::host`]. Without one (the web build) the job
//! commands are disabled and say so.
//!
//! A job edits only through engine commands: it hands them to the session over a [`CommandPort`]
//! and the session's owner runs them between its own work ([`Session::pump_port`]) — the UI once
//! per frame (the port wakes it), a `wait: true` call while it waits, the MCP server on a pump
//! thread. So every edit of a showcase is an ordinary command, in History like a person's.
//!
//! One showcase job runs at a time. Progress, status and cancellation are a job's like an
//! export's (`jobs.list`, `jobs.cancel`); a finished job's outcome is its `jobs.list` entry's
//! `showcase` (and the answer's, with `wait: true`).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{EngineError, Result, Session};

/// What the job commands say without a host.
const NOT_HERE: &str = "AutoCut showcases are not available in this app";
/// Longest subject ("robot", "vintage car").
pub const MAX_SUBJECT_CHARS: usize = 80;
/// Longest title, subtitle, voice id or mood.
pub const MAX_TEXT_CHARS: usize = 400;
/// Most media items one analysis is asked about.
pub const MAX_ITEMS: usize = 10_000;
/// Most entries a plan holds.
pub const MAX_ENTRIES: usize = 2_000;
/// Largest report or plan file `autocut.showcaseSet` reads.
const MAX_FILE_BYTES: u64 = 64 << 20;
/// Most side-by-side identity comparisons an analysis may ask for.
const MAX_COMPARE: u64 = 5_000;
/// How long the port waits for a job's next request once one came (it usually follows at once):
/// a frame runs a short burst of a job's commands, not one.
const FOLLOW_UP: Duration = Duration::from_millis(1);

/// The four showcase jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskKind {
    Analyze,
    Plan,
    Make,
    Build,
}

impl TaskKind {
    pub fn command(self) -> &'static str {
        match self {
            TaskKind::Analyze => "autocut.analyze",
            TaskKind::Plan => "autocut.showcasePlan",
            TaskKind::Make => "autocut.makeShowcase",
            TaskKind::Build => "autocut.showcaseBuild",
        }
    }

    /// The job's label (Progress panel, `jobs.list`).
    fn label(self, subject: &str) -> String {
        match self {
            TaskKind::Analyze => format!("Finding every {subject}"),
            TaskKind::Plan => "Writing the showcase script".into(),
            TaskKind::Make => format!("Making a showcase of every {subject}"),
            TaskKind::Build => "Building the showcase".into(),
        }
    }
}

/// What a job asks the host to do: the command's (checked) parameters, and the report and plan
/// the session keeps, as they were when the job started.
#[derive(Clone, Debug)]
pub struct Task {
    pub kind: TaskKind,
    pub params: Value,
    pub report: Option<Value>,
    pub plan: Option<Value>,
}

/// Runs showcase jobs, installed by the host in [`AutoCut::host`].
pub trait AutoCutHost: Send + Sync {
    /// Whether a job can start now; `Err` says why not. Called while drawing (menus), so it must
    /// be cheap: no network.
    fn ready(&self) -> std::result::Result<(), String> {
        Ok(())
    }
    /// Do `task` (blocking, on the job's thread), editing only through `port`. Put a line in
    /// `progress.status` (and counts in `done` / `total`) as it goes; stop with an error once
    /// `progress.cancel` is set. `Ok` is `{"report"?, "plan"?, "built"?, "summary"?, "status"?}`:
    /// the session keeps `report`, `plan` and `built`, and the job's outcome is the rest.
    fn run(&self, task: Task, port: &CommandPort, progress: &Arc<filmcraft_export::Progress>) -> std::result::Result<Value, String>;
}

/// What the session keeps of showcases (`autocut.showcaseInspect`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Showcase {
    /// The latest analysis report: `{subject, model?, since?, clips: [ClipFacts]}`.
    pub report: Option<Value>,
    /// The latest plan (see the module docs).
    pub plan: Option<Value>,
    /// What the latest build made.
    pub built: Option<Value>,
    /// Bumped when `report` changes (the Showcase panel regroups).
    pub report_revision: u64,
    /// Bumped when `plan` changes (the Showcase panel reloads its draft).
    pub plan_revision: u64,
    /// The latest showcase job and what it does.
    pub job: Option<(u64, TaskKind)>,
}

/// Wakes the session's owner when a job sends a request: the UI asks for a repaint, the MCP
/// server's pump thread wakes.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

enum Request {
    Command {
        id: String,
        params: Value,
        reply: Sender<std::result::Result<Value, String>>,
    },
    /// The active sequence's frame at `seconds`, at most `max_side` on its longest side: RGBA8
    /// over black.
    Render {
        seconds: f64,
        max_side: u32,
        reply: Sender<Option<(u32, u32, Vec<u8>)>>,
    },
}

/// The session's end of the command port ([`Session::pump_port`]).
pub struct Port {
    tx: Sender<Request>,
    rx: Mutex<Receiver<Request>>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Default for Port {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self { tx, rx: Mutex::new(rx), waker: Arc::default() }
    }
}

impl Port {
    /// A job's end of the port; `progress` (the job's) stops its waits when cancelled.
    pub fn handle(&self, progress: Option<Arc<filmcraft_export::Progress>>) -> CommandPort {
        CommandPort { tx: self.tx.clone(), waker: self.waker.clone(), progress }
    }

    /// Call `waker` whenever a job sends a request (replacing any earlier one).
    pub fn set_waker(&self, waker: Waker) {
        *lock(&self.waker) = Some(waker);
    }

    pub fn has_waker(&self) -> bool {
        lock(&self.waker).is_some()
    }

    fn next(&self, wait: Option<Duration>) -> Option<Request> {
        let rx = lock(&self.rx);
        match wait {
            // (no clock on the web: no waiting there, and no jobs either)
            Some(d) if !cfg!(target_arch = "wasm32") => rx.recv_timeout(d).ok(),
            _ => rx.try_recv().ok(),
        }
    }
}

/// A job's way to run engine commands on the session that started it: each call waits until
/// the session's owner has run it ([`Session::pump_port`]).
#[derive(Clone)]
pub struct CommandPort {
    tx: Sender<Request>,
    waker: Arc<Mutex<Option<Waker>>>,
    progress: Option<Arc<filmcraft_export::Progress>>,
}

impl CommandPort {
    /// Run command `id` with `params` on the session and wait for its answer. Fails at once when
    /// the job was cancelled, and when the session is gone.
    pub fn call(&self, id: &str, params: Value) -> std::result::Result<Value, String> {
        self.check()?;
        let (reply, answer) = channel();
        self.send(Request::Command { id: id.to_string(), params, reply })?;
        self.wait(&answer).and_then(|r| r)
    }

    /// The active sequence's frame at `seconds` as RGBA8 over black, its longest side at most
    /// `max_side`: `(width, height, pixels)`, or `None` when there is no sequence or it does not
    /// render.
    pub fn render(&self, seconds: f64, max_side: u32) -> Option<(u32, u32, Vec<u8>)> {
        self.check().ok()?;
        let (reply, answer) = channel();
        self.send(Request::Render { seconds, max_side, reply }).ok()?;
        self.wait(&answer).ok().flatten()
    }

    /// Ask the session to cancel job `job` (a long command this job started), also once this job
    /// is cancelled itself; does not wait.
    pub fn cancel_job(&self, job: u64) {
        let (reply, _) = channel();
        let _ = self.send(Request::Command { id: "jobs.cancel".into(), params: json!({"job": job}), reply });
    }

    /// Whether the job this port serves was cancelled.
    pub fn cancelled(&self) -> bool {
        self.progress.as_ref().is_some_and(|p| p.cancel.load(Ordering::Relaxed))
    }

    fn check(&self) -> std::result::Result<(), String> {
        if self.cancelled() { Err("cancelled".into()) } else { Ok(()) }
    }

    fn send(&self, r: Request) -> std::result::Result<(), String> {
        self.tx.send(r).map_err(|_| "the session closed".to_string())?;
        let waker = lock(&self.waker).clone();
        if let Some(w) = waker {
            w();
        }
        Ok(())
    }

    fn wait<T>(&self, answer: &Receiver<T>) -> std::result::Result<T, String> {
        loop {
            match answer.recv_timeout(Duration::from_millis(200)) {
                Ok(v) => return Ok(v),
                Err(RecvTimeoutError::Timeout) => self.check()?,
                Err(RecvTimeoutError::Disconnected) => return Err("the session closed".into()),
            }
        }
    }
}

/// A job's outcome, filled when it finishes (`showcase` in its `jobs.list` entry).
pub type Outcome = Arc<Mutex<Option<Value>>>;

/// AutoCut in a session: the host that runs showcase jobs, the command port they edit through,
/// what they made, and each job's outcome.
#[derive(Default)]
pub struct AutoCut {
    /// Runs the jobs (None = not available: the job commands are disabled). Hosts install
    /// filmcraft-autocut's here; tests a fake.
    pub host: Option<Arc<dyn AutoCutHost>>,
    pub port: Port,
    pub state: Arc<Mutex<Showcase>>,
    pub outcomes: HashMap<u64, Outcome>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    let job = |id: &'static str, label: &'static str, menu: &'static [&'static str], params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| {
        CommandSpec { id, label, menu, shortcut: None, params, enabled: can_run, run, journal: false }
    };
    vec![
        job(
            "autocut.analyze",
            "Find Every Clip of a Subject",
            &[],
            r#"{"subject":str,"items":[id]?,"since":"YYYY|YYYY-MM|YYYY-MM-DD"?,"frames":1..8=6,"compare":u32=150,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Analyze, p),
        ),
        job(
            "autocut.showcasePlan",
            "Write Showcase Script",
            &[],
            r#"{"subject":str?,"title":str?,"subtitle":str?,"voice":str?,"mood":str?,"minCoverage":0..1=0.3,"entries":[Entry]?,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Plan, p),
        ),
        job(
            "autocut.makeShowcase",
            "Make a Showcase…",
            &["Sequence"],
            r#"{"subject":str?,"prompt":str?,"items":[id]?,"since":"YYYY|YYYY-MM|YYYY-MM-DD"?,"title":str?,"voice":str?,"mood":str?,"frames":1..8=6,"compare":u32=150,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Make, p),
        ),
        job("autocut.showcaseBuild", "Build Showcase", &[], r#"{"plan":Plan?,"export":str?,"upload":bool=false,"qa":bool?,"wait":bool=false}"#, |s, p| {
            start(s, TaskKind::Build, p)
        }),
        CommandSpec {
            id: "autocut.showcaseInspect",
            label: "Inspect Showcase",
            menu: &[],
            shortcut: None,
            params: r#"{"report":bool=false}"#,
            enabled: always,
            run: inspect,
            journal: false,
        },
        CommandSpec {
            id: "autocut.showcaseSet",
            label: "Set Showcase Plan",
            menu: &[],
            shortcut: None,
            params: r#"{"report":obj|null?,"plan":obj|null?,"reportPath":str?,"planPath":str?}"#,
            enabled: always,
            run: set,
            journal: false,
        },
    ]
}

/// The job commands can run: a host is installed and ready.
pub fn can_run(s: &Session) -> std::result::Result<(), String> {
    s.autocut.host.as_ref().ok_or_else(|| NOT_HERE.to_string())?.ready()
}

/// The running showcase job, if any.
pub fn running(s: &Session) -> Option<(u64, TaskKind)> {
    let job = lock(&s.autocut.state).job?;
    let j = s.jobs.iter().find(|j| j.id == job.0)?;
    (!j.progress.finished.load(Ordering::Relaxed)).then_some(job)
}

/// `text` trimmed, or `None` when empty; an error when it is not a string or longer than `max`
/// characters.
fn text_p(cmd: &str, p: &Value, key: &str, max: usize) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(t)) => {
            let t = t.trim();
            if t.chars().count() > max {
                return Err(bad(cmd, format!("`{key}` is longer than {max} characters")));
            }
            Ok((!t.is_empty()).then(|| t.to_string()))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

/// An unsigned whole number in `lo..=hi`, or `None`.
fn count_p(cmd: &str, p: &Value, key: &str, lo: u64, hi: u64) -> Result<Option<u64>> {
    let Some(v) = p.get(key).filter(|v| !v.is_null()) else { return Ok(None) };
    let n = v.as_u64().or_else(|| v.as_f64().filter(|f| f.is_finite() && f.fract() == 0.0 && *f >= 0.0).map(|f| f as u64));
    match n {
        Some(n) if (lo..=hi).contains(&n) => Ok(Some(n)),
        _ => Err(bad(cmd, format!("`{key}` must be a whole number from {lo} to {hi}"))),
    }
}

/// `since` as Unix seconds (UTC midnight): `2020`, `"2020"`, `"2020-06"` or `"2020-06-15"`.
pub fn parse_since(v: &Value) -> std::result::Result<i64, String> {
    let text = match v {
        Value::Number(n) => n.as_u64().map(|y| y.to_string()).unwrap_or_default(),
        Value::String(s) => s.trim().to_string(),
        _ => String::new(),
    };
    let parts: Vec<&str> = text.split('-').collect();
    let num = |i: usize, lo: i64, hi: i64| -> Option<i64> {
        match parts.get(i) {
            None => Some(lo),
            Some(p) if !p.is_empty() && p.len() <= 4 && p.chars().all(|c| c.is_ascii_digit()) => p.parse::<i64>().ok().filter(|n| (lo..=hi).contains(n)),
            Some(_) => None,
        }
    };
    let date = (parts.len() <= 3 && parts.first().is_some_and(|y| y.len() == 4)).then(|| Some((num(0, 1900, 2200)?, num(1, 1, 12)?, num(2, 1, 31)?))).flatten();
    let (y, m, d) = date.ok_or_else(|| format!("`since` wants a year or a date like 2020, 2020-06 or 2020-06-15, not {v}"))?;
    Ok(crate::media_date::days_from_civil(y, m, d) * 86_400)
}

/// The checked parameters of a job command (what the host gets).
fn job_params(cmd: &str, kind: TaskKind, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let mut out = serde_json::Map::new();
    fn put(out: &mut serde_json::Map<String, Value>, k: &str, v: Option<Value>) {
        if let Some(v) = v {
            out.insert(k.to_string(), v);
        }
    }
    if matches!(kind, TaskKind::Analyze | TaskKind::Make) {
        let subject = text_p(cmd, p, "subject", MAX_SUBJECT_CHARS)?;
        let prompt = text_p(cmd, p, "prompt", MAX_TEXT_CHARS)?;
        if subject.is_none() && (kind == TaskKind::Analyze || prompt.is_none()) {
            let what = if kind == TaskKind::Make {
                "`subject` (what to look for, e.g. \"robot\") or a `prompt`"
            } else {
                "`subject`: what to look for, e.g. \"robot\""
            };
            return Err(bad(cmd, format!("need {what}")));
        }
        put(&mut out, "subject", subject.map(Value::String));
        put(&mut out, "prompt", prompt.map(Value::String));
        if let Some(items) = p.get("items").filter(|v| !v.is_null()) {
            let list = items.as_array().ok_or_else(|| bad(cmd, "`items` must be a list of media item ids"))?;
            if list.len() > MAX_ITEMS {
                return Err(bad(cmd, format!("at most {MAX_ITEMS} items")));
            }
            let ids: Vec<u64> = list.iter().map(Value::as_u64).collect::<Option<_>>().ok_or_else(|| bad(cmd, "`items` must be media item ids"))?;
            if ids.is_empty() {
                return Err(bad(cmd, "`items` is empty: leave it out to look at every clip"));
            }
            put(&mut out, "items", Some(json!(ids)));
        }
        if let Some(since) = p.get("since").filter(|v| !v.is_null()) {
            let t = parse_since(since).map_err(|e| bad(cmd, e))?;
            put(&mut out, "since", Some(since.clone()));
            put(&mut out, "sinceUnix", Some(json!(t)));
        }
        put(&mut out, "frames", count_p(cmd, p, "frames", 1, 8)?.map(|n| json!(n)));
        put(&mut out, "compare", count_p(cmd, p, "compare", 0, MAX_COMPARE)?.map(|n| json!(n)));
    }
    if kind == TaskKind::Plan {
        put(&mut out, "subject", text_p(cmd, p, "subject", MAX_SUBJECT_CHARS)?.map(Value::String));
    }
    if matches!(kind, TaskKind::Plan | TaskKind::Make) {
        // given but empty is kept: "" is "no narration" (or "no subtitle"), not "as before"
        for key in ["title", "subtitle", "voice", "mood"] {
            let given = p.get(key).is_some_and(Value::is_string);
            put(&mut out, key, text_p(cmd, p, key, MAX_TEXT_CHARS)?.or_else(|| given.then(String::new)).map(Value::String));
        }
    }
    if kind == TaskKind::Plan {
        if let Some(c) = p.get("minCoverage").filter(|v| !v.is_null()) {
            let c = c.as_f64().filter(|c| (0.0..=1.0).contains(c)).ok_or_else(|| bad(cmd, "`minCoverage` must be a number from 0 to 1"))?;
            put(&mut out, "minCoverage", Some(json!(c)));
        }
        if let Some(e) = p.get("entries").filter(|v| !v.is_null()) {
            put(&mut out, "entries", Some(entries_p(cmd, e)?));
        }
    }
    if kind == TaskKind::Build {
        if let Some(plan) = p.get("plan").filter(|v| !v.is_null()) {
            check_plan(cmd, plan)?;
            put(&mut out, "plan", Some(plan.clone()));
        }
        let export = text_p(cmd, p, "export", 4096)?;
        let upload = bool_p(p, "upload").unwrap_or(false);
        if upload && export.is_none() {
            return Err(bad(cmd, "`upload` sends the export to Google Photos: give an `export` path too"));
        }
        put(&mut out, "export", export.map(Value::String));
        put(&mut out, "upload", Some(json!(upload)));
        if let Some(q) = p.get("qa").filter(|v| !v.is_null()) {
            put(&mut out, "qa", Some(json!(q.as_bool().ok_or_else(|| bad(cmd, "`qa` must be true or false"))?)));
        }
    }
    Ok(Value::Object(out))
}

/// Reviewed entries: a list of at most [`MAX_ENTRIES`] objects.
fn entries_p(cmd: &str, v: &Value) -> Result<Value> {
    let list = v.as_array().ok_or_else(|| bad(cmd, "`entries` must be a list"))?;
    if list.len() > MAX_ENTRIES {
        return Err(bad(cmd, format!("at most {MAX_ENTRIES} entries")));
    }
    if !list.iter().all(Value::is_object) {
        return Err(bad(cmd, "each entry must be an object (as a plan's `entries`)"));
    }
    Ok(v.clone())
}

/// A plan's shape: an object with `entries`.
fn check_plan(cmd: &str, plan: &Value) -> Result<()> {
    if !plan.is_object() {
        return Err(bad(cmd, "`plan` must be an object with `entries` and `script`"));
    }
    entries_p(cmd, plan.get("entries").unwrap_or(&Value::Null))
        .map_err(|_| bad(cmd, format!("`plan` needs `entries`: a list of at most {MAX_ENTRIES} entries")))?;
    match plan.get("script") {
        None | Some(Value::Null) | Some(Value::Object(_)) => Ok(()),
        Some(_) => Err(bad(cmd, "the plan's `script` must be an object: {intro, lines, outro}")),
    }
}

/// A report's shape: an object with `clips`.
fn check_report(cmd: &str, report: &Value) -> Result<()> {
    match report.get("clips") {
        Some(Value::Array(_)) => Ok(()),
        _ => Err(bad(cmd, "a report is an object with `clips` (as `analyze --out` writes)")),
    }
}

/// Start showcase job `kind`.
fn start(s: &mut Session, kind: TaskKind, p: &Value) -> Result<Value> {
    let cmd = kind.command();
    let host = s.autocut.host.clone().ok_or_else(|| bad(cmd, NOT_HERE))?;
    if cfg!(target_arch = "wasm32") {
        return Err(bad(cmd, "showcases are not available on the web"));
    }
    if let Some((id, k)) = running(s) {
        return Err(bad(cmd, format!("a showcase job is already running ({}, job {id}); wait for it or cancel it (jobs.cancel)", k.command())));
    }
    let params = job_params(cmd, kind, p)?;
    let (report, plan) = {
        let st = lock(&s.autocut.state);
        (st.report.clone(), st.plan.clone())
    };
    match kind {
        TaskKind::Plan if report.is_none() && params.get("entries").is_none() => {
            return Err(bad(cmd, "nothing to plan yet: find the clips first (autocut.analyze), or pass reviewed `entries`"));
        }
        TaskKind::Build if plan.is_none() && params.get("plan").is_none() => {
            return Err(bad(cmd, "nothing to build yet: write a plan first (autocut.showcasePlan), or pass a `plan`"));
        }
        TaskKind::Build if params["upload"] == json!(true) => crate::photos::can_upload(s).map_err(|e| bad(cmd, format!("upload: {e}")))?,
        _ => {}
    }
    let subject = params["subject"].as_str().unwrap_or("subject").to_string();
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, kind.label(&subject));
    *lock(&job.progress.status) = "Starting…".into();
    let id = job.id;
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.autocut.outcomes.insert(id, outcome.clone());
    let state = s.autocut.state.clone();
    let port = s.autocut.port.handle(Some(prog.clone()));
    let task = Task { kind, params, report, plan };
    let run = crate::export_tools::guard_job(prog.clone(), res.clone(), move || {
        let t0 = web_time::Instant::now();
        let r = host.run(task, &port, &prog);
        let r = if prog.cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { r };
        finish(&prog, &res, &state, &outcome, r, t0);
    });
    s.jobs.push(job);
    lock(&s.autocut.state).job = Some((id, kind));
    if let Err(e) = std::thread::Builder::new().name("filmcraft-autocut".into()).spawn(run) {
        // no worker: the job ends here, so it never shows as running
        if let Some(j) = s.jobs.iter().find(|j| j.id == id) {
            *lock(&j.progress.error) = Some(e.to_string());
            j.progress.finished.store(true, Ordering::Relaxed);
            *lock(&j.result) = Some(Err(e.to_string()));
        }
        return Err(EngineError::Other(format!("{cmd}: could not start the job: {e}")));
    }
    let mut out = json!({"job": id});
    if wait {
        wait_for(s, id);
        if let Some(j) = s.jobs.iter().find(|j| j.id == id) {
            let v = j.to_json();
            if let Some(e) = v["result"]["error"].as_str() {
                return Err(EngineError::Other(format!("{cmd} failed: {e}")));
            }
            out["result"] = v["result"].clone();
            out["showcase"] = outcome_json(s, id);
        }
    }
    Ok(out)
}

/// Run the job's requests until it finishes (a `wait: true` call: nobody else pumps the port
/// meanwhile).
fn wait_for(s: &mut Session, job: u64) {
    loop {
        let done = s.jobs.iter().find(|j| j.id == job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if done {
            break;
        }
        if let Some(r) = s.autocut.port.next(Some(Duration::from_millis(20))) {
            serve(s, r);
        }
    }
}

/// End a job: keep what it made, then the result that marks it finished.
fn finish(
    prog: &filmcraft_export::Progress,
    res: &crate::narration::JobResult,
    state: &Mutex<Showcase>,
    outcome: &Outcome,
    r: std::result::Result<Value, String>,
    t0: web_time::Instant,
) {
    let r = match r {
        Ok(mut v) => {
            {
                let mut st = lock(state);
                if let Some(report) = v.get("report").filter(|r| r.is_object()) {
                    st.report = Some(report.clone());
                    st.report_revision = st.report_revision.wrapping_add(1);
                }
                if let Some(plan) = v.get("plan").filter(|p| p.is_object()) {
                    st.plan = Some(plan.clone());
                    st.plan_revision = st.plan_revision.wrapping_add(1);
                }
                if let Some(built) = v.get("built").filter(|b| !b.is_null()) {
                    st.built = Some(built.clone());
                }
            }
            // the whole report stays in the session; the outcome has its summary
            if let Some(o) = v.as_object_mut() {
                o.remove("report");
            }
            let status = v["status"].as_str().unwrap_or("Done").to_string();
            let path = v["built"]["export"]["path"].as_str().unwrap_or_default().to_string();
            *lock(outcome) = Some(v);
            *lock(&prog.status) = status;
            Ok(filmcraft_export::Report { path, frames: 0, seconds: t0.elapsed().as_secs_f64(), bytes: 0, render_fps: 0.0, extra_files: Vec::new() })
        }
        Err(e) => {
            *lock(&prog.status) = e.clone();
            *lock(&prog.error) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *lock(res) = Some(r);
}

/// Job `job`'s outcome (`null` while it runs, after a failure, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.autocut.outcomes.get(&job).and_then(|o| lock(o).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is a showcase job.
pub fn is_showcase(s: &Session, job: u64) -> bool {
    s.autocut.outcomes.contains_key(&job)
}

/// Run one request from a job.
fn serve(s: &mut Session, r: Request) {
    match r {
        Request::Command { id, params, reply } => {
            let out = s.execute(&id, params).map_err(|e| e.to_string());
            let _ = reply.send(out);
        }
        Request::Render { seconds, max_side, reply } => {
            let frame = (|| {
                let (w, h) = s.active_sequence().map(|q| (q.settings.width, q.settings.height))?;
                let scale = (max_side.max(1) as f32 / w.max(h).max(1) as f32).min(1.0);
                let t = filmcraft_time::Tick::from_seconds_f64(if seconds.is_finite() { seconds.max(0.0) } else { 0.0 });
                let img = s.try_render_program_at(scale, t).ok()?;
                Some((u32::try_from(img.w).ok()?, u32::try_from(img.h).ok()?, img.over_black_rgba8()))
            })();
            let _ = reply.send(frame);
        }
    }
}

impl Session {
    /// Run what background jobs asked for over the command port ([`CommandPort`]), for about
    /// `budget` at most: the requests waiting, and those that follow right after them. Hosts
    /// call it regularly (the UI once per frame, through [`Session::poll_persistence`]); returns
    /// how many ran.
    pub fn pump_port(&mut self, budget: Duration) -> usize {
        let t0 = web_time::Instant::now();
        let mut n = 0;
        while let Some(r) = self.autocut.port.next((n > 0).then_some(FOLLOW_UP)) {
            serve(self, r);
            n += 1;
            if t0.elapsed() >= budget {
                break;
            }
        }
        n
    }

    /// Wake the session's owner when a job sends a request (see [`Port::set_waker`]).
    pub fn set_port_waker(&mut self, waker: Waker) {
        self.autocut.port.set_waker(waker);
    }
}

/// A report as `autocut.showcaseInspect` shows it by default: how many clips, how many show the
/// subject.
fn report_summary(report: &Value) -> Value {
    let clips = report["clips"].as_array().map(Vec::as_slice).unwrap_or_default();
    json!({
        "subject": report["subject"],
        "since": report.get("since").cloned().unwrap_or(Value::Null),
        "clips": clips.len(),
        "showing": clips.iter().filter(|c| c["present"].as_bool() == Some(true)).count(),
    })
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let whole = bool_p(p, "report").unwrap_or(false);
    let st = lock(&s.autocut.state).clone();
    let job = st.job.and_then(|(id, kind)| {
        let j = s.jobs.iter().find(|j| j.id == id)?;
        let mut v = j.to_json();
        v["command"] = json!(kind.command());
        v["showcase"] = outcome_json(s, id);
        Some(v)
    });
    Ok(json!({
        "available": can_run(s).is_ok(),
        "report": st.report.as_ref().map(|r| if whole { r.clone() } else { report_summary(r) }),
        "plan": st.plan,
        "built": st.built,
        "job": job,
        "reportRevision": st.report_revision,
        "planRevision": st.plan_revision,
    }))
}

/// A JSON file of at most [`MAX_FILE_BYTES`].
fn read_json(s: &Session, cmd: &str, path: &str) -> Result<Value> {
    let size = s.services.file_size(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if size > MAX_FILE_BYTES {
        return Err(bad(cmd, format!("{path} is {size} bytes; at most {MAX_FILE_BYTES}")));
    }
    let bytes = s.services.read_file(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("{path}: not JSON ({e})")))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "autocut.showcaseSet";
    if !p.is_object() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    if let Some((id, _)) = running(s) {
        return Err(bad(cmd, format!("a showcase job is running (job {id}); wait for it or cancel it first")));
    }
    // (key in params, its file key): Some(Some(v)) sets, Some(None) clears, None leaves
    let pick = |key: &str, path_key: &str| -> Result<Option<Option<Value>>> {
        if let Some(path) = str_p(p, path_key) {
            return read_json(s, cmd, path).map(|v| Some(Some(v)));
        }
        Ok(p.get(key).map(|v| (!v.is_null()).then(|| v.clone())))
    };
    let report = pick("report", "reportPath")?;
    let plan = pick("plan", "planPath")?;
    if report.is_none() && plan.is_none() {
        return Err(bad(cmd, "pass `report` and/or `plan` (or `reportPath` / `planPath`; null clears)"));
    }
    if let Some(Some(r)) = &report {
        check_report(cmd, r)?;
    }
    if let Some(Some(pl)) = &plan {
        check_plan(cmd, pl)?;
    }
    let mut st = lock(&s.autocut.state);
    if let Some(r) = report {
        st.report = r;
        st.report_revision = st.report_revision.wrapping_add(1);
    }
    if let Some(pl) = plan {
        st.plan = pl;
        st.plan_revision = st.plan_revision.wrapping_add(1);
    }
    Ok(
        json!({"report": st.report.as_ref().map(report_summary), "plan": st.plan.is_some(), "reportRevision": st.report_revision, "planRevision": st.plan_revision}),
    )
}
