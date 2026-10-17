//! Publish to Instagram: connect the user's professional (Creator or Business) account, see it
//! and its publishing limits, disconnect it, and publish a video as a Reel — a background job
//! with byte progress and cancellation like an export (`jobs.list`, `jobs.cancel`).
//!
//! | command | what it does |
//! |---|---|
//! | `instagram.connect {wait?}` | job: Instagram's business login in the browser (the app set up once with `filmcraft-cli instagram setup`); the code is exchanged for a 60-day token kept in the Keychain |
//! | `instagram.account` | the connected account, when its sign-in runs out, this machine's count of posts and containers in the last 24 h (from disk, no network) |
//! | `instagram.limits` | Instagram's own count of posts in the last 24 h (`content_publishing_limit`) beside this machine's |
//! | `instagram.disconnect` | forget the account's sign-in here |
//! | `instagram.publish {path?, videoUrl?, caption?, shareToFeed?, coverSeconds?, coverUrl?, wait?}` | job: a Reel container, the video sent to it (or fetched by Instagram from `videoUrl`), its processing awaited (at most 5 minutes), then published |
//!
//! The caption, cover and source are checked here before any job starts ([`check_caption`],
//! [`parse_publish`]), so a bad parameter never spends one of the day's containers. The calls
//! themselves live in `filmcraft-instagram`, a layer above, which the host installs as
//! [`InstagramState::publisher`]; without one (the web build, tests) the commands are disabled
//! and say so. A finished job's answer (the account connected, or the Reel published) is in its
//! `jobs.list` entry as `instagram`.
//!
//! Tokens and the app secret never pass through these commands: they are entered in the CLI
//! (standard input or a hidden prompt) and kept in the Keychain by `filmcraft-instagram`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p};
use crate::{EngineError, Result, Session};

/// Instagram's limit on a caption, in characters.
pub const MAX_CAPTION_CHARS: usize = 2200;
/// Instagram's limit on hashtags in a caption.
pub const MAX_HASHTAGS: usize = 30;
/// Instagram's limit on @mentions in a caption.
pub const MAX_MENTIONS: usize = 20;
/// The longest Reel Instagram takes, and the shortest (seconds).
pub const MAX_REEL_SECONDS: f64 = 15.0 * 60.0;
pub const MIN_REEL_SECONDS: f64 = 3.0;
/// The largest Reel file Instagram takes.
pub const MAX_REEL_BYTES: u64 = 300 * 1024 * 1024;
/// The longest URL taken (a `videoUrl` or `coverUrl`).
pub const MAX_URL_CHARS: usize = 2048;
const MAX_PATH_CHARS: usize = 4096;

/// What the commands say without a publisher.
const NOT_HERE: &str = "Instagram is not available in this app";

/// One Reel, as the publisher gets it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReelRequest {
    /// The local file: sent to Instagram directly (a resumable upload), and what the pre-flight
    /// looks at.
    pub path: Option<String>,
    /// A public `https://` URL Instagram fetches the video from instead (the documented way for
    /// apps on Instagram Login); with `path` too, `path` is only checked.
    pub video_url: Option<String>,
    pub caption: Option<String>,
    /// Show it in the profile's feed grid as well as the Reels tab (Instagram's default: yes).
    pub share_to_feed: bool,
    /// The cover: the frame this far into the video (milliseconds; `thumb_offset`).
    pub cover_ms: Option<u64>,
    /// The cover: a public `https://` JPEG (≤ 8 MB, 9:16 recommended; `cover_url`, which wins
    /// over `cover_ms`).
    pub cover_url: Option<String>,
    /// Upload and let Instagram process it, but do not publish: the container waits 23 hours,
    /// and the same request without this publishes it without sending the video again.
    #[serde(default)]
    pub upload_only: bool,
}

impl ReelRequest {
    /// The file or URL in messages.
    pub fn source(&self) -> String {
        match (&self.path, &self.video_url) {
            (Some(p), _) => std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.clone()),
            (None, Some(u)) => u.clone(),
            (None, None) => "the video".into(),
        }
    }
}

/// Instagram for the engine, installed by the host in [`InstagramState::publisher`].
pub trait InstagramPublisher: Send + Sync {
    /// Whether a browser sign-in can start: the app is set up (`instagram setup`). Cheap (no
    /// network): it decides whether `instagram.connect` is enabled.
    fn connect_ready(&self) -> std::result::Result<(), String>;
    /// Whether a publish can start: an account is connected and its sign-in has not run out.
    /// Cheap (no network): it is asked while drawing.
    fn ready(&self) -> std::result::Result<(), String>;
    /// The connected account, its sign-in's expiry and the local limit counts (no network).
    fn account(&self) -> std::result::Result<Value, String>;
    /// Sign in (blocking: the browser, at most a few minutes), reporting what it waits for in
    /// `progress.status`. `Ok` is the account connected.
    fn connect(&self, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
    /// Forget the account's sign-in here.
    fn disconnect(&self) -> std::result::Result<Value, String>;
    /// Instagram's count of posts in the last 24 h, beside this machine's (network).
    fn limits(&self) -> std::result::Result<Value, String>;
    /// What can be checked before the job starts (no network): the account, the file's size and
    /// type, the local limit counts.
    fn check(&self, req: &ReelRequest) -> std::result::Result<Value, String>;
    /// Publish (blocking): bytes in `progress.done` / `progress.total`, a line in
    /// `progress.status`; stop with an error once `progress.cancel` is set. `Ok` is the Reel.
    fn publish(&self, req: &ReelRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
}

/// A job's answer, filled when it finishes.
pub type Outcome = Arc<Mutex<Option<Value>>>;

/// The engine's Instagram state: the host's publisher and what its jobs answered.
#[derive(Default)]
pub struct InstagramState {
    /// `None` = not available: the commands are disabled.
    pub publisher: Option<Arc<dyn InstagramPublisher>>,
    /// Job id → its answer (`instagram` in `jobs.list`).
    pub outcomes: HashMap<u64, Outcome>,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "instagram.connect",
            label: "Connect Instagram",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: can_connect,
            run: connect,
            journal: false,
        },
        CommandSpec {
            id: "instagram.account",
            label: "Instagram Account",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: installed,
            run: account,
            journal: false,
        },
        CommandSpec {
            id: "instagram.limits",
            label: "Instagram Publishing Limits",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: can_publish,
            run: limits,
            journal: false,
        },
        CommandSpec {
            id: "instagram.disconnect",
            label: "Disconnect Instagram",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: can_publish,
            run: disconnect,
            journal: false,
        },
        CommandSpec {
            id: "instagram.publish",
            label: "Publish to Instagram as a Reel",
            menu: &[],
            shortcut: None,
            params: r#"{"path":str?,"videoUrl":str?,"caption":str?,"shareToFeed":bool=true,"coverSeconds":f64?,"coverUrl":str?,"uploadOnly":bool=false,"wait":bool=false}"#,
            enabled: can_publish,
            run: publish,
            journal: false,
        },
    ]
}

fn publisher(s: &Session) -> std::result::Result<&Arc<dyn InstagramPublisher>, String> {
    s.instagram.publisher.as_ref().ok_or_else(|| NOT_HERE.to_string())
}

fn installed(s: &Session) -> std::result::Result<(), String> {
    publisher(s).map(drop)
}

/// `instagram.connect` can run: a publisher is installed and the app is set up.
pub fn can_connect(s: &Session) -> std::result::Result<(), String> {
    publisher(s)?.connect_ready()
}

/// `instagram.publish` (and the account commands) can run: an account is connected.
pub fn can_publish(s: &Session) -> std::result::Result<(), String> {
    publisher(s)?.ready()
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// -------------------------------------------------------------------------------------------
// checks (shared with filmcraft-instagram and the CLI)
// -------------------------------------------------------------------------------------------

/// Hashtags (`#word`) and mentions (`@name`) in a caption, as Instagram counts them: a `#` or
/// `@` at the start or after a space or punctuation, followed by a letter, digit or `_`.
pub fn caption_counts(c: &str) -> (usize, usize) {
    let mut tags = 0usize;
    let mut mentions = 0usize;
    let mut prev: Option<char> = None;
    let mut it = c.chars().peekable();
    while let Some(ch) = it.next() {
        let starts = prev.is_none_or(|p| !(p.is_alphanumeric() || p == '_' || p == '#' || p == '@'));
        if starts && (ch == '#' || ch == '@') && it.peek().is_some_and(|n| n.is_alphanumeric() || *n == '_') {
            if ch == '#' {
                tags = tags.saturating_add(1);
            } else {
                mentions = mentions.saturating_add(1);
            }
        }
        prev = Some(ch);
    }
    (tags, mentions)
}

/// A caption Instagram takes (trimmed; empty is none): at most [`MAX_CAPTION_CHARS`] characters,
/// [`MAX_HASHTAGS`] hashtags and [`MAX_MENTIONS`] mentions, no control characters but line breaks
/// and tabs.
pub fn check_caption(c: Option<&str>) -> std::result::Result<Option<String>, String> {
    let Some(c) = c.map(str::trim).filter(|c| !c.is_empty()) else { return Ok(None) };
    let c = c.replace("\r\n", "\n");
    let n = c.chars().count();
    if n > MAX_CAPTION_CHARS {
        return Err(format!("the caption is {n} characters; Instagram takes at most {MAX_CAPTION_CHARS}"));
    }
    if c.chars().any(|ch| ch.is_control() && ch != '\n' && ch != '\t') {
        return Err("the caption contains a control character".into());
    }
    let (tags, mentions) = caption_counts(&c);
    if tags > MAX_HASHTAGS {
        return Err(format!("the caption has {tags} hashtags; Instagram takes at most {MAX_HASHTAGS}"));
    }
    if mentions > MAX_MENTIONS {
        return Err(format!("the caption mentions {mentions} accounts; Instagram takes at most {MAX_MENTIONS} @mentions"));
    }
    Ok(Some(c))
}

/// A public `https://` URL Instagram can fetch: no spaces or control characters, a host, not
/// a loopback or private address (Instagram's servers fetch it, so `localhost` can never work).
pub fn check_public_url(what: &str, u: &str) -> std::result::Result<String, String> {
    let u = u.trim();
    let bad_url = || format!("the {what} `{}` is not a public https:// URL Instagram can fetch", clip(u, 60));
    if u.chars().count() > MAX_URL_CHARS || u.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(bad_url());
    }
    let Some(rest) = u.strip_prefix("https://") else { return Err(bad_url()) };
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = host.rsplit_once('@').map_or(host, |(_, h)| h);
    let name = host.split(':').next().unwrap_or_default().to_ascii_lowercase();
    let private = name.is_empty()
        || name == "localhost"
        || name.ends_with(".localhost")
        || name.ends_with(".local")
        || name.starts_with("127.")
        || name.starts_with("10.")
        || name.starts_with("192.168.")
        || name.starts_with('[')
        || name == "0.0.0.0"
        || (name.starts_with("172.") && name.split('.').nth(1).and_then(|o| o.parse::<u8>().ok()).is_some_and(|o| (16..=31).contains(&o)))
        || !name.contains('.');
    if private {
        return Err(format!("the {what} must be on a public server Instagram can reach, not `{}`", clip(&name, 60)));
    }
    Ok(u.to_string())
}

// -------------------------------------------------------------------------------------------
// parameters
// -------------------------------------------------------------------------------------------

fn opt_str(cmd: &str, p: &Value, key: &str) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.trim().to_string()).filter(|s| !s.is_empty())),
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

fn opt_bool(cmd: &str, p: &Value, key: &str, default: bool) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
    }
}

/// `instagram.publish`'s parameters (also `publish.run`'s for an Instagram Reel) as a checked
/// [`ReelRequest`].
pub fn parse_publish(cmd: &str, p: &Value) -> Result<ReelRequest> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let path = match opt_str(cmd, p, "path")? {
        Some(path) if path.chars().count() > MAX_PATH_CHARS || path.contains('\0') => return Err(bad(cmd, "`path` is not a usable path to a video")),
        other => other,
    };
    let video_url = opt_str(cmd, p, "videoUrl")?.map(|u| check_public_url("videoUrl", &u)).transpose().map_err(|e| bad(cmd, e))?;
    if path.is_none() && video_url.is_none() {
        return Err(bad(cmd, "need `path` (the video to send) or `videoUrl` (a public https:// URL Instagram fetches it from)"));
    }
    let caption = match p.get("caption") {
        None | Some(Value::Null) => None,
        Some(Value::String(c)) => check_caption(Some(c)).map_err(|e| bad(cmd, e))?,
        Some(_) => return Err(bad(cmd, "`caption` must be a string")),
    };
    let cover_ms = match p.get("coverSeconds") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let secs = v
                .as_f64()
                .filter(|s| s.is_finite() && *s >= 0.0 && *s <= MAX_REEL_SECONDS)
                .ok_or_else(|| bad(cmd, format!("`coverSeconds` must be a number of seconds into the video, 0 to {MAX_REEL_SECONDS:.0}")))?;
            Some((secs * 1000.0).round() as u64)
        }
    };
    let cover_url = opt_str(cmd, p, "coverUrl")?.map(|u| check_public_url("coverUrl", &u)).transpose().map_err(|e| bad(cmd, e))?;
    Ok(ReelRequest {
        path,
        video_url,
        caption,
        share_to_feed: opt_bool(cmd, p, "shareToFeed", true)?,
        cover_ms,
        cover_url,
        upload_only: opt_bool(cmd, p, "uploadOnly", false)?,
    })
}

// -------------------------------------------------------------------------------------------
// commands
// -------------------------------------------------------------------------------------------

fn account(s: &mut Session, _p: &Value) -> Result<Value> {
    let cmd = "instagram.account";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.account().map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn limits(s: &mut Session, _p: &Value) -> Result<Value> {
    let cmd = "instagram.limits";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.limits().map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn disconnect(s: &mut Session, _p: &Value) -> Result<Value> {
    let cmd = "instagram.disconnect";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.disconnect().map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn connect(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "instagram.connect";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.connect_ready().map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, "Connecting Instagram".into());
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting for the Instagram sign-in in your browser".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.instagram.outcomes.insert(job.id, outcome.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-instagram-connect", move || {
        let t0 = web_time::Instant::now();
        let r = up.connect(&prog).map(|v| {
            let name = v["account"]["username"].as_str().map(|u| format!("@{u}")).unwrap_or_else(|| "the account".into());
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
            format!("Connected {name}")
        });
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    answer(s, id, wait, json!({"job": id}))
}

/// Start an `instagram.publish` job for a checked request (also `publish.run`'s).
pub(crate) fn start_publish(s: &mut Session, cmd: &str, req: ReelRequest, wait: bool) -> Result<Value> {
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    let bytes = match &req.path {
        Some(path) => {
            let n = s.services.file_size(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
            if n == 0 {
                return Err(bad(cmd, format!("{path} is empty")));
            }
            if n > MAX_REEL_BYTES && req.video_url.is_none() {
                return Err(bad(cmd, format!("{path} is {} MB; Instagram takes Reels of at most 300 MB", n / (1024 * 1024))));
            }
            n
        }
        None => 0,
    };
    up.ready().map_err(|e| bad(cmd, e))?;
    let checked = up.check(&req).map_err(|e| bad(cmd, e))?;
    let to = checked["account"]["username"].as_str().map(|u| format!(" (@{u})")).unwrap_or_default();
    let job = crate::narration::new_job(s, format!("Publishing {} to Instagram{to}", req.source()));
    job.progress.total.store(bytes, std::sync::atomic::Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting to publish".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.instagram.outcomes.insert(job.id, outcome.clone());
    let path = req.path.clone().unwrap_or_default();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-instagram-publish", move || {
        let t0 = web_time::Instant::now();
        let r = up.publish(&req, &prog);
        finish_publish(&prog, &res, &outcome, req.path.as_deref().unwrap_or_default(), bytes, r, t0);
    })?;
    answer(s, id, wait, json!({"job": id, "path": path, "checked": checked}))
}

fn publish(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "instagram.publish";
    publisher(s).map_err(|e| bad(cmd, e))?;
    let req = parse_publish(cmd, p)?;
    start_publish(s, cmd, req, bool_p(p, "wait").unwrap_or(false))
}

/// With `wait`, the finished job's result and answer; a failure is the command's error.
fn answer(s: &Session, id: u64, wait: bool, mut out: Value) -> Result<Value> {
    if wait && let Some(j) = s.jobs.iter().find(|j| j.id == id) {
        let v = j.to_json();
        if let Some(e) = v["result"]["error"].as_str() {
            return Err(EngineError::Other(e.to_string()));
        }
        out["result"] = v["result"].clone();
        out["instagram"] = outcome_json(s, id);
    }
    Ok(out)
}

/// End a publish job: the Reel first, then the result that marks it finished.
fn finish_publish(
    prog: &filmcraft_export::Progress,
    res: &crate::narration::JobResult,
    outcome: &Outcome,
    path: &str,
    bytes: u64,
    r: std::result::Result<Value, String>,
    t0: web_time::Instant,
) {
    use std::sync::atomic::Ordering;
    let r = match r {
        Ok(reel) => {
            let done = if reel["published"] == false { "Uploaded to Instagram (not published)" } else { "Published to Instagram" };
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(reel);
            prog.done.store(bytes, Ordering::Relaxed);
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = done.into();
            let seconds = t0.elapsed().as_secs_f64();
            Ok(filmcraft_export::Report { path: path.to_string(), frames: 0, seconds, bytes, render_fps: 0.0, extra_files: Vec::new() })
        }
        Err(e) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
            *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
}

/// Job `job`'s answer (`null` while it runs, after a failure, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.instagram.outcomes.get(&job).and_then(|o| o.lock().unwrap_or_else(PoisonError::into_inner).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is an Instagram job (a sign-in or a publish).
pub fn is_job(s: &Session, job: u64) -> bool {
    s.instagram.outcomes.contains_key(&job)
}
