//! AutoCut showcases in the app: find every clip that shows a subject, plan a counted, narrated
//! showcase of them (entries and voice-over), review it, build it — as engine commands, so the
//! Showcase panel, the CLI, the control channel and MCP agents share one path.
//!
//! | command | does |
//! |---|---|
//! | `autocut.analyze` | job: which clips and photos show `subject` (or what a `prompt` asks for; of `items`, or of the Google Photos `album`, filmed `since` a date; `photos: false` for videos only), what it is, when; the report is kept in the session |
//! | `autocut.showcasePlan` | job: group the report's clips into entries (or take reviewed `entries`) and write the voice-over; the plan is kept |
//! | `autocut.makeShowcase` | job: both, one after the other (Sequence ▸ Make a Showcase…, and AutoCut prompts like "make a video of every robot since 2020") |
//! | `autocut.showcaseBuild` | job: build the plan (the kept one, or `plan`) into a new sequence; `export` it (YouTube 1080p, chapters and a QA report beside it) and `upload` the export to Google Photos (into `uploadAlbum` too) |
//! | `autocut.showcaseInspect` | the kept report (a summary, or all of it with `report: true`), plan and last build, and the latest job |
//! | `autocut.showcaseSet` | keep a `report` or `plan` from elsewhere (or `reportPath` / `planPath`: the CLI's `analyze --out` and `showcase --script-out` files); `null` clears one |
//! | `autocut.malayalamScript` | job: the plan's (the kept one, `plan` or `planPath`) English intro, lines and outro in natural spoken Malayalam, written by the local script model (or `model`) from `docs/malayalam-style.md`, its examples and the user's approved pairs; every line keeps its English's facts, gets the safe fixes and the lint (`script.lintMalayalam`) and goes back to the model once with what to fix (`revise: false`: not). The outcome's `malayalam` is the lines file (`{intro, outro, lines: {label: …}}`), written to `out` too, and `lines` what is left to look at |
//! | `autocut.trailer` | job: a vertical trailer (9:16, also 1:1 / 4:5) of a finished showcase: the show `sequence`, its `plan` (or `planPath`, or the kept plan) and its build report (`report` / `reportPath`, or the last build). A hook (the most striking robot, or `hook`) under a big title ("48 robots · 2013 → 2025"), then a chronological montage of `robots` robots cut on the beats of generated music, each on its best moment, followed by the reframe, its name and year inside the `platform`'s safe zone; the show's first narration sentence over the hook (`hookVoice`); the show's redaction kept; an end card (`logo`, `endText`, `channel`). `export` writes it with the Social Vertical preset, with a cover, a caption, a frame sheet and QA beside it |
//! | `autocut.shorts` | job: long video → shorts. The transcript of `sequence` (or of media `item`, put in a new sequence) is cut into sentences; runs of them `minSeconds`..`maxSeconds` long are scored for interest (and, with the local model, for the strength of their opening hook), and the best `count` that do not overlap each become a vertical sequence (`sequence.autoReframe` to `aspect`, following the speaker) with bold captions from the transcript (`captions`) and an optional `title` card; `export` (a folder) exports each with its captions burned in, and `uploadYoutube` sends each export to YouTube, private unless `youtube.privacy` says otherwise |
//!
//! A plan is `{subject, title, subtitle?, voice?, mood, entries, script: {intro, lines, outro}}`;
//! each entry is the CLI plan's entry (`number`, `label`, `kind`, `description`, `unix`, `shots`)
//! plus the review's `include` (default true) and `mergePrevious` (default false), and
//! `script.lines` follows `entries` one to one (excluded and merged-in entries' lines unused).
//!
//! With `album` (a Google Photos album's exact title) an analysis takes its clips from that
//! album: the host picks all of it, imports it (`file.import` with `reuse`: what the project
//! already holds is not imported twice) and analyses exactly those clips — the album is the
//! seed the user curates in Google Photos. Hosts that cannot pick albums say so
//! ([`AutoCutHost::albums_ready`]) before a job starts.
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
/// What a job with `album` says when the host cannot pick Google Photos albums.
pub const NO_ALBUMS: &str = "picking a Google Photos album is not available in this app";
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
    /// Long video → shorts (`autocut.shorts`).
    Shorts,
    /// The plan's English voice-over in natural spoken Malayalam (`autocut.malayalamScript`).
    MalayalamScript,
    /// A vertical trailer of a finished showcase (`autocut.trailer`).
    Trailer,
}

impl TaskKind {
    pub fn command(self) -> &'static str {
        match self {
            TaskKind::Analyze => "autocut.analyze",
            TaskKind::Plan => "autocut.showcasePlan",
            TaskKind::Make => "autocut.makeShowcase",
            TaskKind::Build => "autocut.showcaseBuild",
            TaskKind::Shorts => "autocut.shorts",
            TaskKind::MalayalamScript => "autocut.malayalamScript",
            TaskKind::Trailer => "autocut.trailer",
        }
    }

    /// The job's label (Progress panel, `jobs.list`).
    fn label(self, subject: &str) -> String {
        match self {
            TaskKind::Analyze => format!("Finding every {subject}"),
            TaskKind::Plan => "Writing the showcase script".into(),
            TaskKind::Make => format!("Making a showcase of every {subject}"),
            TaskKind::Build => "Building the showcase".into(),
            TaskKind::Shorts => "Making shorts".into(),
            TaskKind::MalayalamScript => "Writing the Malayalam script".into(),
            TaskKind::Trailer => "Making a trailer".into(),
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
    /// Whether a job can pick a Google Photos `album` now (signed in, …); `Err` says why not.
    /// Cheap, no network: a job with `album` is refused before it starts when this fails.
    fn albums_ready(&self) -> std::result::Result<(), String> {
        Err(NO_ALBUMS.into())
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
            r#"{"subject":str?,"prompt":str?,"items":[id]?,"album":str?,"photos":bool=true,"since":"YYYY|YYYY-MM|YYYY-MM-DD"?,"frames":1..8=6,"screen":0..8=0,"compare":u32=150,"fresh":bool=false,"wait":bool=false}"#,
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
            r#"{"subject":str?,"prompt":str?,"items":[id]?,"album":str?,"photos":bool=true,"since":"YYYY|YYYY-MM|YYYY-MM-DD"?,"title":str?,"voice":str?,"mood":str?,"frames":1..8=6,"screen":0..8=0,"compare":u32=150,"fresh":bool=false,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Make, p),
        ),
        job(
            "autocut.showcaseBuild",
            "Build Showcase",
            &[],
            r#"{"plan":Plan?,"export":str?,"upload":bool=false,"uploadAlbum":str?,"qa":bool?,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Build, p),
        ),
        job(
            "autocut.shorts",
            "Make Shorts…",
            &[],
            r#"{"item":id?,"sequence":id?,"count":1..20=3,"minSeconds":3..180=15,"maxSeconds":5..180=60,"aspect":"9:16"|"1:1"|"4:5"|"16:9"="9:16","motion":"smooth"|"responsive"?,"fallback":"blur"|"letterbox"|"center"="blur","platform":"youtube"|"reels"|"tiktok"|"vertical"="youtube","captions":bool=true,"title":bool=false,"topic":str?,"analysisFps":1..60=30,"export":dir?,"uploadYoutube":bool=false,"youtube":{"channel":str?,"privacy":"private"|"unlisted"|"public"="private","tags":[str]?,"playlist":str?,"description":str?}?,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Shorts, p),
        ),
        job(
            "autocut.trailer",
            "Make a Trailer…",
            &[],
            r#"{"sequence":id?,"plan":Plan?,"planPath":str?,"report":obj?,"reportPath":str?,"seconds":10..90=34,"aspect":"9:16"|"1:1"|"4:5"="9:16","platform":"reels"|"youtube"|"tiktok"|"vertical"="reels","robots":3..30?,"hook":str?,"hookAt":seconds?,"at":{"<label>":seconds|{"item":id?,"seconds":seconds}}?,"title":str?,"hookVoice":bool?,"logo":str?,"channel":str?,"endText":str="Full video on YouTube","link":str?,"exclude":[str]?,"include":[str]?,"bpm":80..180=124,"mood":str="energetic","seed":u64=7,"redact":"faces,plates"|"faces"|"plates"|false?,"fallback":"blur"|"letterbox"|"center"="blur","export":str?,"preset":str?,"qa":bool?,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::Trailer, p),
        ),
        job(
            "autocut.malayalamScript",
            "Write Malayalam Script",
            &[],
            r#"{"plan":Plan?,"planPath":str?,"out":str?,"model":str?,"examples":str?,"revise":bool=true,"maxWords":4..60=14,"wait":bool=false}"#,
            |s, p| start(s, TaskKind::MalayalamScript, p),
        ),
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
        if subject.is_none() && prompt.is_none() {
            return Err(bad(cmd, "need `subject` (what to look for, e.g. \"robot\") or a `prompt` (\"a video of every robot since 2020\")"));
        }
        put(&mut out, "subject", subject.map(Value::String));
        put(&mut out, "prompt", prompt.map(Value::String));
        if let Some(album) = text_p(cmd, p, "album", crate::photos::MAX_ALBUM_TITLE_CHARS)? {
            if album.chars().any(char::is_control) {
                return Err(bad(cmd, "`album` must be an album title on one line"));
            }
            if p.get("items").is_some_and(|v| !v.is_null()) {
                return Err(bad(cmd, "`album` decides which clips are analysed: leave out `items`"));
            }
            put(&mut out, "album", Some(Value::String(album)));
        } else if p.get("album").is_some_and(Value::is_string) {
            return Err(bad(cmd, "`album` is empty: give the Google Photos album's title"));
        }
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
        put(&mut out, "screen", count_p(cmd, p, "screen", 0, 8)?.map(|n| json!(n)));
        put(&mut out, "compare", count_p(cmd, p, "compare", 0, MAX_COMPARE)?.map(|n| json!(n)));
        // analyse again every clip the analysis cache already knows
        put(&mut out, "fresh", bool_p(p, "fresh").filter(|f| *f).map(Value::Bool));
        // photos are kept by default: only "videos only" is passed on
        match p.get("photos") {
            None | Some(Value::Null) | Some(Value::Bool(true)) => {}
            Some(Value::Bool(false)) => put(&mut out, "photos", Some(Value::Bool(false))),
            Some(_) => return Err(bad(cmd, "`photos` is true (photos too, the default) or false (videos only)")),
        }
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
        let upload_album = text_p(cmd, p, "uploadAlbum", crate::photos::MAX_ALBUM_TITLE_CHARS)?;
        if upload_album.is_some() && !upload {
            return Err(bad(cmd, "`uploadAlbum` names the album the upload goes into: set `upload` too"));
        }
        put(&mut out, "export", export.map(Value::String));
        put(&mut out, "upload", Some(json!(upload)));
        put(&mut out, "uploadAlbum", upload_album.map(Value::String));
        if let Some(q) = p.get("qa").filter(|v| !v.is_null()) {
            put(&mut out, "qa", Some(json!(q.as_bool().ok_or_else(|| bad(cmd, "`qa` must be true or false"))?)));
        }
    }
    if kind == TaskKind::Shorts {
        shorts_params(cmd, p, &mut out)?;
    }
    if kind == TaskKind::Trailer {
        trailer_params(cmd, p, &mut out)?;
    }
    if kind == TaskKind::MalayalamScript {
        if let Some(plan) = p.get("plan").filter(|v| !v.is_null()) {
            check_plan(cmd, plan)?;
            put(&mut out, "plan", Some(plan.clone()));
        }
        if p.get("plan").is_some_and(|v| !v.is_null()) && p.get("planPath").is_some_and(|v| !v.is_null()) {
            return Err(bad(cmd, "give `plan` or `planPath`, not both"));
        }
        put(&mut out, "planPath", text_p(cmd, p, "planPath", 4096)?.map(Value::String));
        put(&mut out, "out", text_p(cmd, p, "out", 4096)?.map(Value::String));
        put(&mut out, "examples", text_p(cmd, p, "examples", 4096)?.map(Value::String));
        let model = text_p(cmd, p, "model", MAX_SUBJECT_CHARS)?;
        if model.as_deref().is_some_and(|m| m.chars().any(|c| c.is_whitespace() || c.is_control())) {
            return Err(bad(cmd, "`model` is an Ollama model name like qwen3:8b"));
        }
        put(&mut out, "model", model.map(Value::String));
        if let Some(r) = p.get("revise").filter(|v| !v.is_null()) {
            put(&mut out, "revise", Some(json!(r.as_bool().ok_or_else(|| bad(cmd, "`revise` must be true or false"))?)));
        }
        put(&mut out, "maxWords", count_p(cmd, p, "maxWords", 4, 60)?.map(|n| json!(n)));
    }
    Ok(Value::Object(out))
}

/// Most shorts one job makes.
pub const MAX_SHORTS: u64 = 20;

/// The checked parameters of `autocut.shorts` (see the module docs).
fn shorts_params(cmd: &str, p: &Value, out: &mut serde_json::Map<String, Value>) -> Result<()> {
    let id = |k: &str| -> Result<Option<u64>> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => v.as_u64().map(Some).ok_or_else(|| bad(cmd, format!("`{k}` must be an id"))),
        }
    };
    let (item, sequence) = (id("item")?, id("sequence")?);
    if item.is_some() && sequence.is_some() {
        return Err(bad(cmd, "give `item` (a media clip) or `sequence`, not both"));
    }
    if let Some(i) = item {
        out.insert("item".into(), json!(i));
    }
    if let Some(q) = sequence {
        out.insert("sequence".into(), json!(q));
    }
    out.insert("count".into(), json!(count_p(cmd, p, "count", 1, MAX_SHORTS)?.unwrap_or(3)));
    let secs = |k: &str, lo: f64, hi: f64, default: f64| -> Result<f64> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => {
                v.as_f64().filter(|f| f.is_finite() && (lo..=hi).contains(f)).ok_or_else(|| bad(cmd, format!("`{k}` is a number of seconds from {lo} to {hi}")))
            }
        }
    };
    let (min, max) = (secs("minSeconds", 3.0, 180.0, 15.0)?, secs("maxSeconds", 5.0, 180.0, 60.0)?);
    if max < min + 2.0 {
        return Err(bad(cmd, "`maxSeconds` must be at least 2 seconds more than `minSeconds`"));
    }
    out.insert("minSeconds".into(), json!(min));
    out.insert("maxSeconds".into(), json!(max));
    let aspect = text_p(cmd, p, "aspect", 16)?.unwrap_or_else(|| "9:16".into());
    if filmcraft_render::reframe::parse_aspect(&aspect).is_none() {
        return Err(bad(cmd, format!("`aspect` is \"9:16\", \"1:1\", \"4:5\" or \"16:9\", not \"{aspect}\"")));
    }
    out.insert("aspect".into(), json!(aspect));
    if let Some(m) = text_p(cmd, p, "motion", 32)? {
        let f = filmcraft_render::reframe::Follow::from_name(&m).ok_or_else(|| bad(cmd, "`motion` is \"smooth\" or \"responsive\""))?;
        out.insert("motion".into(), json!(f.name()));
    }
    // where nobody is followed: the whole picture over a blurred copy of itself, by default
    let fallback = match text_p(cmd, p, "fallback", 32)? {
        None => filmcraft_render::reframe::Fallback::Blur,
        Some(f) => filmcraft_render::reframe::Fallback::from_name(&f).ok_or_else(|| bad(cmd, "`fallback` is \"blur\", \"letterbox\" or \"center\""))?,
    };
    out.insert("fallback".into(), json!(fallback.name()));
    // whose interface the captions and the title keep clear of
    let platform = match crate::safe_zones::platform_p(cmd, p, "platform")? {
        None => crate::safe_zones::Platform::Youtube,
        Some(Some(pl)) => pl,
        Some(None) => return Err(bad(cmd, "`platform` is \"youtube\", \"reels\", \"tiktok\" or \"vertical\"")),
    };
    out.insert("platform".into(), json!(platform.name()));
    for (k, default) in [("captions", true), ("title", false), ("uploadYoutube", false)] {
        let v = match p.get(k).filter(|v| !v.is_null()) {
            None => default,
            Some(v) => v.as_bool().ok_or_else(|| bad(cmd, format!("`{k}` must be true or false")))?,
        };
        out.insert(k.into(), json!(v));
    }
    if let Some(t) = text_p(cmd, p, "topic", MAX_TEXT_CHARS)? {
        out.insert("topic".into(), json!(t));
    }
    if let Some(f) = p.get("analysisFps").filter(|v| !v.is_null()) {
        let f = f
            .as_f64()
            .filter(|f| f.is_finite() && (1.0..=60.0).contains(f))
            .ok_or_else(|| bad(cmd, "`analysisFps` is a number of frames a second from 1 to 60"))?;
        out.insert("analysisFps".into(), json!(f));
    }
    let export = text_p(cmd, p, "export", 4096)?;
    if out.get("uploadYoutube") == Some(&json!(true)) && export.is_none() {
        return Err(bad(cmd, "`uploadYoutube` sends the exports to YouTube: give an `export` folder too"));
    }
    if let Some(e) = export {
        if e.chars().any(char::is_control) {
            return Err(bad(cmd, "`export` is a folder path"));
        }
        out.insert("export".into(), json!(e));
    }
    if let Some(y) = p.get("youtube").filter(|v| !v.is_null()) {
        let o = y.as_object().ok_or_else(|| bad(cmd, "`youtube` is an object: {channel?, privacy?, tags?, playlist?, description?}"))?;
        let mut keep = serde_json::Map::new();
        for (k, v) in o {
            let ok = match (k.as_str(), v) {
                ("channel" | "playlist" | "description" | "category", Value::String(t)) => t.chars().count() <= 5000,
                ("privacy", Value::String(t)) => matches!(t.as_str(), "private" | "unlisted" | "public"),
                ("tags", Value::Array(a)) => a.len() <= 500 && a.iter().all(Value::is_string),
                ("madeForKids" | "notifySubscribers", Value::Bool(_)) => true,
                _ => false,
            };
            if !ok {
                return Err(bad(
                    cmd,
                    format!(
                        "`youtube.{k}` is not an option this takes (channel, privacy, tags, playlist, description, category, madeForKids, notifySubscribers)"
                    ),
                ));
            }
            keep.insert(k.clone(), v.clone());
        }
        out.insert("youtube".into(), Value::Object(keep));
    }
    Ok(())
}

/// Longest label, title or channel a trailer takes, and most labels in `exclude` / `include`.
pub const MAX_TRAILER_TEXT: usize = 200;
pub const MAX_TRAILER_LABELS: usize = 100;

/// The checked parameters of `autocut.trailer` (see the module docs).
fn trailer_params(cmd: &str, p: &Value, out: &mut serde_json::Map<String, Value>) -> Result<()> {
    if let Some(q) = p.get("sequence").filter(|v| !v.is_null()) {
        out.insert("sequence".into(), json!(q.as_u64().ok_or_else(|| bad(cmd, "`sequence` must be an id"))?));
    }
    if let Some(plan) = p.get("plan").filter(|v| !v.is_null()) {
        check_plan(cmd, plan)?;
        out.insert("plan".into(), plan.clone());
    }
    if p.get("plan").is_some_and(|v| !v.is_null()) && p.get("planPath").is_some_and(|v| !v.is_null()) {
        return Err(bad(cmd, "give `plan` or `planPath`, not both"));
    }
    if let Some(report) = p.get("report").filter(|v| !v.is_null()) {
        if !report.is_object() {
            return Err(bad(cmd, "`report` is the showcase's build report (an object: what `showcase` prints, or a build's outcome)"));
        }
        out.insert("report".into(), report.clone());
    }
    if p.get("report").is_some_and(|v| !v.is_null()) && p.get("reportPath").is_some_and(|v| !v.is_null()) {
        return Err(bad(cmd, "give `report` or `reportPath`, not both"));
    }
    // a path, a label or a line of text: one line each
    let line = |key: &str, max: usize| -> Result<Option<String>> {
        let t = text_p(cmd, p, key, max)?;
        if t.as_deref().is_some_and(|t| t.chars().any(char::is_control)) {
            return Err(bad(cmd, format!("`{key}` must be on one line")));
        }
        Ok(t)
    };
    for key in ["planPath", "reportPath", "logo", "export"] {
        if let Some(v) = line(key, 4096)? {
            out.insert(key.into(), json!(v));
        }
    }
    for key in ["hook", "title", "channel", "endText", "link", "preset"] {
        if let Some(v) = line(key, MAX_TRAILER_TEXT)? {
            out.insert(key.into(), json!(v));
        }
    }
    if let Some(m) = line("mood", 40)? {
        out.insert("mood".into(), json!(m));
    }
    let num = |key: &str, lo: f64, hi: f64, default: f64| -> Result<f64> {
        match p.get(key).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|f| f.is_finite() && (lo..=hi).contains(f)).ok_or_else(|| bad(cmd, format!("`{key}` is a number from {lo} to {hi}"))),
        }
    };
    out.insert("seconds".into(), json!(num("seconds", 10.0, 90.0, 34.0)?));
    out.insert("bpm".into(), json!(num("bpm", 80.0, 180.0, 124.0)?));
    if p.get("hookAt").is_some_and(|v| !v.is_null()) {
        out.insert("hookAt".into(), json!(num("hookAt", 0.0, 86_400.0, 0.0)?));
    }
    // where robots' shots start: {"<label>": seconds | {"item": id, "seconds": s}}
    if let Some(at) = p.get("at").filter(|v| !v.is_null()) {
        let o = at.as_object().ok_or_else(|| bad(cmd, "`at` is {\"<entry label>\": seconds or {item, seconds}}"))?;
        if o.len() > MAX_TRAILER_LABELS {
            return Err(bad(cmd, format!("`at` names at most {MAX_TRAILER_LABELS} entries")));
        }
        let secs = |v: &Value| v.as_f64().filter(|s| s.is_finite() && (0.0..=86_400.0).contains(s));
        let mut keep = serde_json::Map::new();
        for (label, v) in o {
            let l = label.trim();
            if l.is_empty() || l.chars().count() > MAX_TRAILER_TEXT || l.chars().any(char::is_control) {
                return Err(bad(cmd, "`at`'s keys are entry labels, each on one line"));
            }
            let ok = match v {
                Value::Number(_) => secs(v).map(|s| json!(s)),
                Value::Object(x) => match (x.get("seconds").and_then(secs), x.get("item")) {
                    (Some(s), None | Some(Value::Null)) => Some(json!({"seconds": s})),
                    (Some(s), Some(i)) => i.as_u64().map(|i| json!({"item": i, "seconds": s})),
                    _ => None,
                },
                _ => None,
            };
            keep.insert(l.to_string(), ok.ok_or_else(|| bad(cmd, format!("`at.{l}` is seconds into the clip (0–86400), or {{item, seconds}}")))?);
        }
        out.insert("at".into(), Value::Object(keep));
    }
    if let Some(n) = count_p(cmd, p, "robots", 3, 30)? {
        out.insert("robots".into(), json!(n));
    }
    if let Some(seed) = p.get("seed").filter(|v| !v.is_null()) {
        out.insert("seed".into(), json!(seed.as_u64().ok_or_else(|| bad(cmd, "`seed` is a whole number"))?));
    }
    let aspect = text_p(cmd, p, "aspect", 16)?.unwrap_or_else(|| "9:16".into());
    if !matches!(aspect.as_str(), "9:16" | "1:1" | "4:5") {
        return Err(bad(cmd, format!("`aspect` is \"9:16\", \"1:1\" or \"4:5\", not \"{}\"", aspect.chars().take(16).collect::<String>())));
    }
    out.insert("aspect".into(), json!(aspect));
    let platform = match crate::safe_zones::platform_p(cmd, p, "platform")? {
        None => crate::safe_zones::Platform::Reels,
        Some(Some(pl)) => pl,
        Some(None) => return Err(bad(cmd, "`platform` is \"reels\", \"youtube\", \"tiktok\" or \"vertical\"")),
    };
    out.insert("platform".into(), json!(platform.name()));
    let fallback = match text_p(cmd, p, "fallback", 32)? {
        None => filmcraft_render::reframe::Fallback::Blur,
        Some(f) => filmcraft_render::reframe::Fallback::from_name(&f).ok_or_else(|| bad(cmd, "`fallback` is \"blur\", \"letterbox\" or \"center\""))?,
    };
    out.insert("fallback".into(), json!(fallback.name()));
    for key in ["hookVoice", "qa"] {
        if let Some(v) = p.get(key).filter(|v| !v.is_null()) {
            out.insert(key.into(), json!(v.as_bool().ok_or_else(|| bad(cmd, format!("`{key}` must be true or false")))?));
        }
    }
    match p.get("redact") {
        None | Some(Value::Null) => {}
        Some(Value::Bool(false)) => {
            out.insert("redact".into(), json!(false));
        }
        Some(Value::String(r)) => {
            let kinds: Vec<String> = r.split(',').map(|k| k.trim().to_ascii_lowercase()).filter(|k| !k.is_empty()).collect();
            if kinds.is_empty() || kinds.len() > 2 || !kinds.iter().all(|k| matches!(k.as_str(), "faces" | "plates")) {
                return Err(bad(cmd, "`redact` is \"faces,plates\", \"faces\", \"plates\" or false"));
            }
            out.insert("redact".into(), json!(kinds.join(",")));
        }
        Some(_) => return Err(bad(cmd, "`redact` is \"faces,plates\", \"faces\", \"plates\" or false")),
    }
    for key in ["exclude", "include"] {
        let Some(v) = p.get(key).filter(|v| !v.is_null()) else { continue };
        let list = v.as_array().ok_or_else(|| bad(cmd, format!("`{key}` is a list of entry labels")))?;
        if list.len() > MAX_TRAILER_LABELS {
            return Err(bad(cmd, format!("`{key}` names at most {MAX_TRAILER_LABELS} entries")));
        }
        let mut labels = Vec::new();
        for l in list {
            let t = l.as_str().map(str::trim).filter(|t| !t.is_empty() && t.chars().count() <= MAX_TRAILER_TEXT && !t.chars().any(char::is_control));
            labels.push(t.ok_or_else(|| bad(cmd, format!("`{key}` is a list of entry labels, each on one line")))?.to_string());
        }
        out.insert(key.into(), json!(labels));
    }
    Ok(())
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
    match plan.get("entries") {
        Some(e @ Value::Array(_)) => entries_p(cmd, e)?,
        _ => return Err(bad(cmd, format!("`plan` needs `entries`: a list of at most {MAX_ENTRIES} entries"))),
    };
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
    let mut params = job_params(cmd, kind, p)?;
    let (report, plan) = {
        let st = lock(&s.autocut.state);
        (st.report.clone(), st.plan.clone())
    };
    // a plan from a file (the CLI's `showcase --script-out`, `showcase-plan --out`)
    if let Some(path) = params.get("planPath").and_then(Value::as_str).map(str::to_string) {
        let v = read_json(s, cmd, &path)?;
        check_plan(cmd, &v)?;
        params["plan"] = v;
    }
    // a trailer's build report from a file (what the CLI's `showcase` prints), else the last build
    if kind == TaskKind::Trailer {
        if let Some(path) = params.get("reportPath").and_then(Value::as_str).map(str::to_string) {
            let v = read_json(s, cmd, &path)?;
            if !v.is_object() {
                return Err(bad(cmd, format!("{path}: not a showcase build report")));
            }
            params["report"] = v;
        } else if params.get("report").is_none()
            && let Some(built) = lock(&s.autocut.state).built.clone()
        {
            params["report"] = built;
        }
    }
    match kind {
        TaskKind::MalayalamScript if plan.is_none() && params.get("plan").is_none() => {
            return Err(bad(cmd, "nothing to translate yet: write a plan first (autocut.showcasePlan), or pass a `plan` or `planPath`"));
        }
        TaskKind::Plan if report.is_none() && params.get("entries").is_none() => {
            return Err(bad(cmd, "nothing to plan yet: find the clips first (autocut.analyze), or pass reviewed `entries`"));
        }
        TaskKind::Build if plan.is_none() && params.get("plan").is_none() => {
            return Err(bad(cmd, "nothing to build yet: write a plan first (autocut.showcasePlan), or pass a `plan`"));
        }
        TaskKind::Shorts if params["uploadYoutube"] == json!(true) => {
            crate::youtube::can_upload(s).map_err(|e| bad(cmd, format!("uploadYoutube: {e}")))?;
        }
        TaskKind::Build if params["upload"] == json!(true) => {
            crate::photos::can_upload(s).map_err(|e| bad(cmd, format!("upload: {e}")))?;
            if params.get("uploadAlbum").is_some() {
                crate::photos::can_use_albums(s).map_err(|e| bad(cmd, format!("uploadAlbum: {e}")))?;
            }
        }
        _ => {}
    }
    if params.get("album").is_some() {
        host.albums_ready().map_err(|e| bad(cmd, format!("album: {e}")))?;
    }
    let subject = params["subject"].as_str().unwrap_or("clip the prompt asks for").to_string();
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
