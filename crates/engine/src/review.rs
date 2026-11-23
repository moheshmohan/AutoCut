//! Review pointed at frames (PX8): comments pinned to a time, a clip and a box on the picture,
//! kept in an inbox in the project's build folder, for an agent to look at, fix and close.
//! (The idea of frame-pinned comments for an agent comes from huashu-motion, MIT; this is
//! written fresh.)
//!
//! | command | does |
//! |---|---|
//! | `review.comment` | appends a comment: a time (frame), the clip and box it points at, the text, who wrote it |
//! | `review.inbox` | lists the comments (open, done or all; since a time); with `wait`, returns as soon as a new open comment arrives or the timeout passes |
//! | `review.look` | renders a comment's frame with its box drawn, and says what is there: clips, graphics and their text, captions, the nearest cuts, a transition, `sequence.checkMotion` findings |
//! | `review.done` | marks a comment done, with a note of what was changed |
//! | `review.delete` | removes a comment's text and box; its id is kept (a deleted record), so it is never given to another comment |
//!
//! **The inbox** is `build/review/inbox.jsonl` beside the saved project (the build folder
//! `delivery.preflight` reads `build/decisions.json` from): one JSON object a line, at most
//! [`MAX_COMMENTS`] comments and [`MAX_INBOX_BYTES`] bytes. A line that is not a comment (or a
//! deleted record) is skipped with a note (`damaged`) and written back unchanged, never a crash.
//! A record holds what the comment says and where it points (`id`, `sequence`, `time` in ticks,
//! `seconds`, `timecode`, `clip?`, `box?`, `text`, `author`, `created`, `status`, `note?`,
//! `doneAt?`): no folder paths, nothing personal beyond what was typed. A comment is not part of
//! the project: it is no undo step, and `review.delete` is how one is taken back.
//!
//! **Waiting** never holds the UI or a server: `job: true` waits on a worker (a job in
//! `jobs.list`, its answer under `review` once finished, `jobs.cancel` stops it), which is what
//! MCP's `command_run` and the app's control channel use for `wait: true`; called in-process
//! (the CLI's `exec`), `wait: true` blocks that call, polling the file every quarter second.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_project::{ItemId, ItemKind, Project, Sequence};
use filmcraft_time::{Tick, TimeDisplay, format_time};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::commands::{CommandSpec, always, bad, has_seq};
use crate::{EngineError, Result, Services, Session};

/// The folder of the inbox (and of the looks and packets) in the build folder.
pub const FOLDER: &str = "review";
/// The inbox's file name in [`FOLDER`].
pub const INBOX: &str = "inbox.jsonl";
/// Largest inbox read or written.
pub const MAX_INBOX_BYTES: u64 = 4 << 20;
/// Most comments (open and done) an inbox holds.
pub const MAX_COMMENTS: usize = 2000;
/// Most lines of any kind read from an inbox.
const MAX_LINES: usize = 10_000;
/// Longest comment text and done note, characters.
pub const MAX_TEXT: usize = 2000;
pub const MAX_NOTE: usize = 1000;
/// Longest author label, characters.
const MAX_AUTHOR: usize = 48;
/// Most damaged lines described in an answer (all are counted).
const MAX_DAMAGED_NOTES: usize = 20;
/// Longest wait, seconds, and the default.
pub const MAX_TIMEOUT: f64 = 600.0;
const DEFAULT_TIMEOUT: f64 = 60.0;
/// How often a wait reads the inbox, milliseconds.
const POLL_MS: u64 = 250;
/// Most finished waits kept for `jobs.list`.
const MAX_OUTCOMES: usize = 16;
/// The long side of a look's frame by default, pixels.
const LOOK_SIDE: f64 = 960.0;

const CMD_COMMENT: &str = "review.comment";
const CMD_INBOX: &str = "review.inbox";
const CMD_LOOK: &str = "review.look";
const CMD_DONE: &str = "review.done";
const CMD_DELETE: &str = "review.delete";

type Run = fn(&mut Session, &Value) -> Result<Value>;
type Enabled = fn(&Session) -> std::result::Result<(), String>;

fn spec(id: &'static str, label: &'static str, params: &'static str, enabled: Enabled, run: Run) -> CommandSpec {
    CommandSpec { id, label, params, run, enabled, menu: &[], shortcut: None, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            CMD_COMMENT,
            "Comment on Frame",
            r#"{"time":ticks?|"seconds":f64?|"frame":int?|"timecode":str? (default the playhead; the frame it falls in),"sequence":id? (default the active one),"clip":clipId? (what the comment points at),"box":[x,y,w,h]? (sequence pixels; w and h 0: a point),"text":str (at most 2000 characters),"author":str="agent" ("user" when it is the user's own words or came from the app)}"#,
            has_seq,
            comment,
        ),
        spec(
            CMD_INBOX,
            "Review Inbox",
            r#"{"status":"open"|"done"|"all"="open","since":utc? (RFC 3339: only comments created after it; each answer's `now` is the next call's since),"wait":bool=false (return as soon as a new open comment arrives, or at the timeout),"timeout":seconds 0..600=60,"job":bool=false (wait on a worker: answers {job} at once; jobs.list has the comments under `review` when it finishes)}"#,
            always,
            inbox,
        ),
        spec(
            CMD_LOOK,
            "Look at a Review Comment",
            r#"{"id":str (a comment id, e.g. "c3"),"scale":0.05..1? (default a long side of 960 px),"out":path? (the PNG; default build/review/look-<id>.png),"inline":bool=false (also pngBase64)}"#,
            always,
            look,
        ),
        spec(CMD_DONE, "Mark Review Comment Done", r#"{"id":str,"note":str (what was changed, at most 1000 characters)}"#, always, done),
        spec(CMD_DELETE, "Delete Review Comment", r#"{"id":str}"#, always, delete),
    ]
}

// ---------------------------------------------------------------------------------------------
// the records

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Open,
    Done,
}

/// One comment, as stored in the inbox (and answered).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Comment {
    pub id: String,
    /// The sequence its time is on.
    pub sequence: u64,
    /// Ticks: the start of the frame it points at.
    pub time: i64,
    pub seconds: f64,
    pub timecode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<u64>,
    /// `[x, y, w, h]` in sequence pixels (w and h 0: a point).
    #[serde(default, rename = "box", skip_serializing_if = "Option::is_none")]
    pub area: Option<[f64; 4]>,
    pub text: String,
    pub author: String,
    /// UTC, RFC 3339 with milliseconds.
    pub created: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_at: Option<String>,
}

impl Comment {
    /// Why a parsed record is not a comment this inbox keeps.
    fn problem(&self) -> Option<String> {
        if id_number(&self.id).is_none() {
            return Some(format!("`{}` is not a comment id (c and a number)", short(&self.id, 24)));
        }
        if self.time < 0 {
            return Some("a negative time".into());
        }
        if self.text.trim().is_empty() || self.text.chars().count() > MAX_TEXT {
            return Some(format!("its text is empty or longer than {MAX_TEXT} characters"));
        }
        if self.author.is_empty() || self.author.chars().count() > MAX_AUTHOR {
            return Some("its author is empty or too long".into());
        }
        if self.area.is_some_and(|b| b.iter().any(|v| !v.is_finite() || *v < 0.0)) {
            return Some("its box has a value that is not a finite, non-negative number".into());
        }
        if parse_utc(&self.created).is_none() {
            return Some("its `created` is not an RFC 3339 time".into());
        }
        if self.note.as_ref().is_some_and(|n| n.chars().count() > MAX_NOTE) {
            return Some(format!("its note is longer than {MAX_NOTE} characters"));
        }
        None
    }
}

/// What is left of a deleted comment: its id, so it is never given again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Deleted {
    id: String,
    deleted: String,
}

#[derive(Clone, Debug)]
enum Line {
    Comment(Box<Comment>),
    Deleted(Deleted),
    /// A line that is neither, kept as it was.
    Damaged(String),
}

/// An inbox as read: its lines in order, and notes on the damaged ones.
#[derive(Clone, Debug, Default)]
pub struct Inbox {
    lines: Vec<Line>,
    damaged: Vec<Value>,
    damaged_count: usize,
}

impl Inbox {
    pub fn comments(&self) -> impl Iterator<Item = &Comment> {
        self.lines.iter().filter_map(|l| if let Line::Comment(c) = l { Some(&**c) } else { None })
    }
    fn find(&self, id: &str) -> Option<&Comment> {
        self.comments().find(|c| c.id == id)
    }
    fn find_mut(&mut self, id: &str) -> Option<&mut Comment> {
        self.lines.iter_mut().find_map(|l| if let Line::Comment(c) = l { (c.id == id).then_some(&mut **c) } else { None })
    }
    fn was_deleted(&self, id: &str) -> bool {
        self.lines.iter().any(|l| matches!(l, Line::Deleted(d) if d.id == id))
    }
    fn open(&self) -> usize {
        self.comments().filter(|c| c.status == Status::Open).count()
    }
    /// The next id: one past the highest ever given (deleted ones included).
    fn next_id(&self) -> Option<String> {
        let top = self
            .lines
            .iter()
            .filter_map(|l| match l {
                Line::Comment(c) => id_number(&c.id),
                Line::Deleted(d) => id_number(&d.id),
                Line::Damaged(_) => None,
            })
            .max()
            .unwrap_or(0);
        top.checked_add(1).map(|n| format!("c{n}"))
    }
    fn damaged_json(&self) -> Value {
        json!({"count": self.damaged_count, "lines": self.damaged, "note": "lines that are not comments were skipped and are kept in the file as they were"})
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        for l in &self.lines {
            let text = match l {
                Line::Comment(c) => serde_json::to_string(c).unwrap_or_default(),
                Line::Deleted(d) => serde_json::to_string(d).unwrap_or_default(),
                Line::Damaged(raw) => raw.clone(),
            };
            if !text.is_empty() {
                out.push_str(&text);
                out.push('\n');
            }
        }
        out.into_bytes()
    }
}

/// `c12` → 12.
fn id_number(id: &str) -> Option<u64> {
    let digits = id.strip_prefix('c')?;
    if digits.is_empty() || digits.len() > 9 || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The lines of an inbox file.
pub fn parse(text: &str) -> Inbox {
    let mut inbox = Inbox::default();
    let mut seen = BTreeSet::new();
    for (n, raw) in text.lines().enumerate() {
        if raw.trim().is_empty() {
            continue;
        }
        if inbox.lines.len() >= MAX_LINES {
            inbox.damaged_count = inbox.damaged_count.saturating_add(1);
            if inbox.damaged.len() < MAX_DAMAGED_NOTES {
                inbox.damaged.push(json!({"line": n + 1, "reason": format!("past the first {MAX_LINES} lines")}));
            }
            inbox.lines.push(Line::Damaged(raw.to_string()));
            continue;
        }
        let line = if let Ok(d) = serde_json::from_str::<Deleted>(raw) {
            match id_number(&d.id) {
                None => Err("a deleted record without a comment id".to_string()),
                Some(_) if !seen.insert(d.id.clone()) => Err(format!("{} is there twice", d.id)),
                Some(_) => Ok(Line::Deleted(d)),
            }
        } else {
            match serde_json::from_str::<Comment>(raw) {
                Ok(c) => match c.problem() {
                    None if !seen.insert(c.id.clone()) => Err(format!("{} is there twice", c.id)),
                    None => Ok(Line::Comment(Box::new(c))),
                    Some(why) => Err(why),
                },
                Err(e) => Err(format!("not a comment: {}", short(&e.to_string(), 120))),
            }
        };
        match line {
            Ok(l) => inbox.lines.push(l),
            Err(reason) => {
                inbox.damaged_count = inbox.damaged_count.saturating_add(1);
                if inbox.damaged.len() < MAX_DAMAGED_NOTES {
                    inbox.damaged.push(json!({"line": n + 1, "reason": reason}));
                }
                inbox.lines.push(Line::Damaged(raw.to_string()));
            }
        }
    }
    inbox
}

// ---------------------------------------------------------------------------------------------
// the file

/// The review folder beside the saved project (`build/review`), `None` when it is not saved.
pub fn folder(s: &Session) -> Option<String> {
    crate::copy_lint::beside_project(s, FOLDER)
}

/// The inbox beside the saved project, `None` when it is not saved.
pub fn inbox_path(s: &Session) -> Option<String> {
    folder(s).map(|d| std::path::Path::new(&d).join(INBOX).display().to_string())
}

fn require_inbox(s: &Session, cmd: &str) -> Result<String> {
    inbox_path(s).ok_or_else(|| bad(cmd, "the project is not saved: the review inbox lives in build/review beside it, so save it first"))
}

/// Read the inbox at `path` (missing: empty).
pub fn read(services: &dyn Services, path: &str) -> std::result::Result<Inbox, String> {
    match services.file_size(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Inbox::default()),
        Err(e) => return Err(format!("{path}: {e}")),
        Ok(n) if n > MAX_INBOX_BYTES => return Err(format!("{path} is {n} bytes, over the inbox's {MAX_INBOX_BYTES} byte limit")),
        Ok(_) => {}
    }
    let bytes = services.read_file_bounded(path, MAX_INBOX_BYTES).map_err(|e| format!("{path}: {e}"))?;
    Ok(parse(&String::from_utf8_lossy(&bytes)))
}

fn write(services: &dyn Services, path: &str, inbox: &Inbox) -> std::result::Result<(), String> {
    let bytes = inbox.to_bytes();
    if bytes.len() as u64 > MAX_INBOX_BYTES {
        return Err(format!("the inbox would pass its {MAX_INBOX_BYTES} byte limit: mark comments done or delete some"));
    }
    if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    services.write_file(path, &bytes).map_err(|e| format!("{path}: {e}"))
}

fn read_for(s: &Session, cmd: &str, path: &str) -> Result<Inbox> {
    read(&*s.services, path).map_err(|e| bad(cmd, format!("the review inbox cannot be read: {e}")))
}

/// The open comments of the saved project's inbox, for markers: empty when the project is not
/// saved or the inbox cannot be read.
pub fn open_comments(s: &Session) -> Vec<Comment> {
    let Some(path) = inbox_path(s) else { return Vec::new() };
    read(&*s.services, &path).map(|i| i.comments().filter(|c| c.status == Status::Open).cloned().collect()).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// time

/// Now, UTC, RFC 3339 with milliseconds.
pub(crate) fn utc_now() -> String {
    let ms = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    chrono::DateTime::from_timestamp_millis(i64::try_from(ms).unwrap_or(i64::MAX))
        .map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

pub(crate) fn parse_utc(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s.trim()).ok().map(|d| d.with_timezone(&chrono::Utc))
}

pub(crate) fn r3(x: f64) -> f64 {
    if x.is_finite() { (x * 1000.0).round() / 1000.0 } else { 0.0 }
}

fn r2(x: f64) -> f64 {
    if x.is_finite() { (x * 100.0).round() / 100.0 } else { 0.0 }
}

/// At most `n` characters of `s`.
pub(crate) fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n.saturating_sub(1)).collect::<String>()) }
}

/// `t` as the Program monitor shows it.
pub(crate) fn timecode(q: &Sequence, t: Tick) -> String {
    format_time(t, q.settings.frame_rate.sane(), q.settings.drop_frame, TimeDisplay::Timecode, 48000)
}

/// Time fields for an answer: ticks, seconds, timecode.
pub(crate) fn when(q: &Sequence, t: Tick) -> Value {
    json!({"time": t.0, "seconds": r3(t.seconds()), "timecode": timecode(q, t)})
}

// ---------------------------------------------------------------------------------------------
// parameters

pub(crate) fn keys(p: &Value, cmd: &str, allowed: &[&str]) -> Result<()> {
    if p.is_null() {
        return Ok(());
    }
    let o = p.as_object().ok_or_else(|| bad(cmd, "parameters must be an object"))?;
    if let Some(k) = o.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(bad(cmd, format!("unknown parameter `{}`: {}", short(k, 64), allowed.join(", "))));
    }
    Ok(())
}

pub(crate) fn bool_or(p: &Value, key: &str, default: bool, cmd: &str) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, format!("`{key}` is true or false"))),
    }
}

fn id_p(p: &Value, cmd: &str) -> Result<String> {
    let id = p.get("id").and_then(Value::as_str).map(str::trim).ok_or_else(|| bad(cmd, "`id` is a comment id, e.g. \"c3\""))?;
    if id_number(id).is_none() {
        return Err(bad(cmd, format!("`{}` is not a comment id (c and a number, e.g. \"c3\")", short(id, 24))));
    }
    Ok(id.to_string())
}

/// Text from the caller: trimmed, control characters other than line breaks dropped, tabs as
/// spaces; at most `max` characters, never empty.
pub(crate) fn text_p(p: &Value, key: &str, max: usize, cmd: &str) -> Result<String> {
    let raw = p.get(key).and_then(Value::as_str).ok_or_else(|| bad(cmd, format!("`{key}` is the text")))?;
    let clean: String = raw.chars().map(|c| if c == '\t' { ' ' } else { c }).filter(|c| *c == '\n' || !c.is_control()).collect();
    let clean = clean.trim().to_string();
    if clean.is_empty() {
        return Err(bad(cmd, format!("`{key}` is empty")));
    }
    let n = clean.chars().count();
    if n > max {
        return Err(bad(cmd, format!("`{key}` is {n} characters; at most {max}")));
    }
    Ok(clean)
}

fn author_p(p: &Value, cmd: &str) -> Result<String> {
    let Some(v) = p.get("author").filter(|v| !v.is_null()) else { return Ok("agent".into()) };
    let a = v.as_str().map(str::trim).ok_or_else(|| bad(cmd, "`author` is a short label: \"user\" or the agent's name"))?;
    let ok = !a.is_empty() && a.chars().count() <= MAX_AUTHOR && a.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.' | ':' | '/'));
    if !ok {
        return Err(bad(cmd, format!("`author` is a label of at most {MAX_AUTHOR} letters, digits, spaces and - _ . : /")));
    }
    Ok(a.to_string())
}

/// The sequence named by `sequence`, else the active one.
pub(crate) fn sequence_p(s: &Session, p: &Value, cmd: &str) -> Result<ItemId> {
    match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(cmd, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(cmd, "no such sequence"))?;
            Ok(id)
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence),
    }
}

/// A box inside a `w`×`h` frame (half a pixel over an edge is let in, then kept inside).
fn box_p(p: &Value, (w, h): (u32, u32), cmd: &str) -> Result<Option<[f64; 4]>> {
    let Some(v) = p.get("box").filter(|v| !v.is_null()) else { return Ok(None) };
    let wrong = || bad(cmd, "`box` is [x, y, w, h] in sequence pixels: finite, not negative (w and h 0 for a point)");
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(wrong)?;
    let mut b = [0.0; 4];
    for (slot, x) in b.iter_mut().zip(a) {
        *slot = x.as_f64().filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(wrong)?;
    }
    let (fw, fh) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let [x, y, bw, bh] = b;
    if x > fw + 0.5 || y > fh + 0.5 || x + bw > fw + 0.5 || y + bh > fh + 0.5 {
        return Err(bad(cmd, format!("`box` must lie inside the {w}×{h} frame")));
    }
    let x = x.min(fw);
    let y = y.min(fh);
    Ok(Some([r2(x), r2(y), r2(bw.min(fw - x)), r2(bh.min(fh - y))]))
}

/// `time` / `frame` / `seconds` / `timecode`, any of them given; `None` when none is.
fn time_given(p: &Value, q: &Sequence, cmd: &str) -> Result<Option<Tick>> {
    let named = ["time", "frame", "seconds", "timecode"].iter().filter(|k| p.get(**k).is_some_and(|v| !v.is_null())).count();
    if named == 0 {
        return Ok(None);
    }
    if named > 1 {
        return Err(bad(cmd, "give the time one way: time (ticks), frame, seconds or timecode"));
    }
    if p.get("seconds").and_then(Value::as_f64).is_some_and(|x| !x.is_finite() || x.abs() > 1e9) {
        return Err(bad(cmd, "`seconds` must be a finite number"));
    }
    crate::commands::time_p_at(p, "", q.settings.frame_rate.sane(), q.settings.drop_frame, Tick::ZERO)
        .map(Some)
        .ok_or_else(|| bad(cmd, "the time is ticks (`time`), a `frame`, `seconds` or a `timecode`"))
}

// ---------------------------------------------------------------------------------------------
// review.comment

fn comment(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_COMMENT, &["time", "frame", "seconds", "timecode", "sequence", "clip", "box", "text", "author"])?;
    let path = require_inbox(s, CMD_COMMENT)?;
    let seq = sequence_p(s, p, CMD_COMMENT)?;
    let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?;
    let end = q.duration();
    if end.0 <= 0 {
        return Err(bad(CMD_COMMENT, "the sequence is empty: there is no frame to comment on"));
    }
    let rate = q.settings.frame_rate.sane();
    let playhead = s.state.playheads.get(&seq).copied().unwrap_or_default();
    let t = time_given(p, q, CMD_COMMENT)?.unwrap_or(playhead);
    if t < Tick::ZERO {
        return Err(bad(CMD_COMMENT, "the time must not be negative"));
    }
    if t >= end {
        return Err(bad(CMD_COMMENT, format!("the time is after the sequence's last frame (it ends at {})", timecode(q, end))));
    }
    // the start of the frame it falls in
    let t = rate.snap(t).max(Tick::ZERO);
    let mut warnings = Vec::new();
    let clip = match p.get("clip").filter(|v| !v.is_null()) {
        Some(v) => {
            let c = v.as_u64().ok_or_else(|| bad(CMD_COMMENT, "`clip` is a clip id"))?;
            let (_, it) = q.find_item(filmcraft_project::ClipId(c)).ok_or_else(|| bad(CMD_COMMENT, format!("clip {c} is not in this sequence")))?;
            if !(it.start <= t && t < it.end()) {
                warnings.push(format!("clip {c} is not under {}", timecode(q, t)));
            }
            Some(c)
        }
        None => None,
    };
    let area = box_p(p, (q.settings.width, q.settings.height), CMD_COMMENT)?;
    let text = text_p(p, "text", MAX_TEXT, CMD_COMMENT)?;
    let author = author_p(p, CMD_COMMENT)?;
    let tc = timecode(q, t);
    let mut inbox = read_for(s, CMD_COMMENT, &path)?;
    if inbox.comments().count() >= MAX_COMMENTS {
        return Err(bad(CMD_COMMENT, format!("the inbox holds {MAX_COMMENTS} comments: delete some first (review.delete)")));
    }
    let id = inbox.next_id().ok_or_else(|| bad(CMD_COMMENT, "no comment id is left"))?;
    let c = Comment {
        id,
        sequence: seq.0,
        time: t.0,
        seconds: r3(t.seconds()),
        timecode: tc,
        clip,
        area,
        text,
        author,
        created: utc_now(),
        status: Status::Open,
        note: None,
        done_at: None,
    };
    inbox.lines.push(Line::Comment(Box::new(c.clone())));
    write(&*s.services, &path, &inbox).map_err(|e| bad(CMD_COMMENT, e))?;
    let mut out = json!({
        "comment": c,
        "inbox": path,
        "open": inbox.open(),
        "undoable": false,
        "note": "kept in the review inbox, not the project: it is not an undo step; review.delete {id} removes it",
    });
    if !warnings.is_empty() {
        out["warnings"] = json!(warnings);
    }
    if inbox.damaged_count > 0 {
        out["damaged"] = inbox.damaged_json();
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// review.inbox

/// Which comments an inbox answer lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    Open,
    Done,
    All,
}

/// What a wait looks for and what its answer lists.
#[derive(Clone, Debug)]
struct Query {
    which: Which,
    since: Option<chrono::DateTime<chrono::Utc>>,
    /// Ids there when the wait began: a new comment is one not among them (or created after
    /// `since`, when that is given).
    baseline: BTreeSet<String>,
}

impl Query {
    fn lists(&self, c: &Comment) -> bool {
        let status = match self.which {
            Which::Open => c.status == Status::Open,
            Which::Done => c.status == Status::Done,
            Which::All => true,
        };
        status && self.since.is_none_or(|t| parse_utc(&c.created).is_some_and(|c| c > t))
    }
    fn is_new(&self, c: &Comment) -> bool {
        c.status == Status::Open
            && match self.since {
                Some(t) => parse_utc(&c.created).is_some_and(|c| c > t),
                None => !self.baseline.contains(&c.id),
            }
    }
    fn answer(&self, path: &str, inbox: &Inbox) -> Value {
        let comments: Vec<&Comment> = inbox.comments().filter(|c| self.lists(c)).collect();
        let done = inbox.comments().filter(|c| c.status == Status::Done).count();
        let mut out = json!({
            "inbox": path,
            "comments": comments,
            "open": inbox.open(),
            "done": done,
            "now": utc_now(),
        });
        if inbox.damaged_count > 0 {
            out["damaged"] = inbox.damaged_json();
        }
        out
    }
}

fn inbox(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_INBOX, &["status", "since", "wait", "timeout", "job"])?;
    let path = require_inbox(s, CMD_INBOX)?;
    let which = match p.get("status").filter(|v| !v.is_null()) {
        None => Which::Open,
        Some(v) => match v.as_str() {
            Some("open") => Which::Open,
            Some("done") => Which::Done,
            Some("all") => Which::All,
            _ => return Err(bad(CMD_INBOX, "`status` is open, done or all")),
        },
    };
    let since = match p.get("since").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_str().and_then(parse_utc).ok_or_else(|| bad(CMD_INBOX, "`since` is a UTC time in RFC 3339, e.g. 2026-11-23T10:00:00.000Z"))?),
    };
    let job = bool_or(p, "job", false, CMD_INBOX)?;
    let wait = bool_or(p, "wait", false, CMD_INBOX)? || job;
    let timeout = match p.get("timeout").filter(|v| !v.is_null()) {
        None => DEFAULT_TIMEOUT,
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && (0.0..=MAX_TIMEOUT).contains(x))
            .ok_or_else(|| bad(CMD_INBOX, format!("`timeout` is seconds, 0 to {MAX_TIMEOUT}")))?,
    };
    let start = read_for(s, CMD_INBOX, &path)?;
    let query = Query { which, since, baseline: start.comments().map(|c| c.id.clone()).collect() };
    if !wait {
        return Ok(query.answer(&path, &start));
    }
    if cfg!(target_arch = "wasm32") {
        return Err(bad(CMD_INBOX, "waiting needs the desktop app or the command line; call it without `wait`"));
    }
    let services = s.services.clone();
    if !job {
        return wait_for(&*services, &path, &query, timeout, None).map_err(|e| bad(CMD_INBOX, e));
    }
    let job = crate::narration::new_job(s, "Waiting for review comments".into());
    let steps = (timeout * 1000.0 / POLL_MS as f64).ceil().max(1.0) as u64;
    job.progress.total.store(steps, Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting for a review comment".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome: Arc<Mutex<Option<Value>>> = Arc::default();
    s.review.outcomes.insert(job.id, outcome.clone());
    while s.review.outcomes.len() > MAX_OUTCOMES {
        let Some(old) = s.review.outcomes.keys().next().copied() else { break };
        s.review.outcomes.remove(&old);
    }
    let thread_path = path.clone();
    let id = crate::narration::spawn(s, job, false, "filmcraft-review-inbox", move || {
        let t0 = web_time::Instant::now();
        let r = wait_for(&*services, &thread_path, &query, timeout, Some(&prog)).map(|v| {
            let n = v["new"].as_array().map_or(0, Vec::len);
            let status = if v["cancelled"] == true {
                "Cancelled".to_string()
            } else if n == 0 {
                "No new review comment".to_string()
            } else {
                format!("{n} new review comment{}", if n == 1 { "" } else { "s" })
            };
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
            status
        });
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    Ok(json!({
        "job": id, "poll": "jobs.list", "cancel": "jobs.cancel", "inbox": path, "timeout": timeout,
        "note": "jobs.list shows the comments under `review` once a new one arrives or the timeout passes",
    }))
}

/// Read the inbox every [`POLL_MS`] until a new open comment is there or `timeout` seconds pass
/// (or the job is cancelled); the answer with `new`, `waited` and `timedOut`.
fn wait_for(
    services: &dyn Services,
    path: &str,
    query: &Query,
    timeout: f64,
    progress: Option<&filmcraft_export::Progress>,
) -> std::result::Result<Value, String> {
    let t0 = web_time::Instant::now();
    let limit = std::time::Duration::from_secs_f64(timeout.clamp(0.0, MAX_TIMEOUT));
    let mut last_error = None;
    loop {
        let inbox = match read(services, path) {
            Ok(i) => Some(i),
            Err(e) => {
                last_error = Some(e);
                None
            }
        };
        let fresh: Vec<String> = inbox.as_ref().map(|i| i.comments().filter(|c| query.is_new(c)).map(|c| c.id.clone()).collect()).unwrap_or_default();
        let elapsed = t0.elapsed();
        let cancelled = progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed));
        if !fresh.is_empty() || elapsed >= limit || cancelled {
            let Some(inbox) = inbox else {
                return Err(format!("the review inbox cannot be read: {}", last_error.unwrap_or_default()));
            };
            let mut out = query.answer(path, &inbox);
            out["new"] = json!(fresh);
            out["waited"] = json!(r3(elapsed.as_secs_f64()));
            out["timedOut"] = json!(fresh.is_empty() && !cancelled);
            if cancelled {
                out["cancelled"] = json!(true);
            }
            return Ok(out);
        }
        if let Some(p) = progress {
            let step = (elapsed.as_millis() / u128::from(POLL_MS)) as u64;
            p.done.store(step.min(p.total.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
        let left = limit.saturating_sub(elapsed);
        std::thread::sleep(left.min(std::time::Duration::from_millis(POLL_MS)));
    }
}

/// Waits (`review.inbox` with `job: true`) and what the finished ones answered.
#[derive(Default)]
pub struct ReviewJobs {
    pub outcomes: BTreeMap<u64, Arc<Mutex<Option<Value>>>>,
}

/// A finished wait's answer (`null` while it waits or after a failure).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    let o = s.review.outcomes.get(&job)?;
    Some(o.lock().unwrap_or_else(PoisonError::into_inner).clone().unwrap_or(Value::Null))
}

// ---------------------------------------------------------------------------------------------
// review.done and review.delete

fn done(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_DONE, &["id", "note"])?;
    let path = require_inbox(s, CMD_DONE)?;
    let id = id_p(p, CMD_DONE)?;
    let note = text_p(p, "note", MAX_NOTE, CMD_DONE)?;
    let mut inbox = read_for(s, CMD_DONE, &path)?;
    if inbox.was_deleted(&id) {
        return Err(bad(CMD_DONE, format!("{id} was deleted")));
    }
    let Some(c) = inbox.find_mut(&id) else { return Err(bad(CMD_DONE, format!("there is no comment {id}"))) };
    let again = c.status == Status::Done;
    c.status = Status::Done;
    c.note = Some(note);
    c.done_at = Some(utc_now());
    let c = c.clone();
    write(&*s.services, &path, &inbox).map_err(|e| bad(CMD_DONE, e))?;
    Ok(json!({"comment": c, "again": again, "open": inbox.open(), "inbox": path}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_DELETE, &["id"])?;
    let path = require_inbox(s, CMD_DELETE)?;
    let id = id_p(p, CMD_DELETE)?;
    let mut inbox = read_for(s, CMD_DELETE, &path)?;
    if inbox.was_deleted(&id) {
        return Err(bad(CMD_DELETE, format!("{id} was already deleted")));
    }
    let Some(at) = inbox.lines.iter().position(|l| matches!(l, Line::Comment(c) if c.id == id)) else {
        return Err(bad(CMD_DELETE, format!("there is no comment {id}")));
    };
    let Some(slot) = inbox.lines.get_mut(at) else { return Err(bad(CMD_DELETE, format!("there is no comment {id}"))) };
    *slot = Line::Deleted(Deleted { id: id.clone(), deleted: utc_now() });
    write(&*s.services, &path, &inbox).map_err(|e| bad(CMD_DELETE, e))?;
    Ok(json!({"deleted": id, "open": inbox.open(), "inbox": path, "note": "its text and box are gone; the id stays taken, so no other comment gets it"}))
}

// ---------------------------------------------------------------------------------------------
// review.look

/// The box's colour on a look, and the darker edge around it.
const BOX_RGB: [u8; 3] = [255, 196, 40];
const EDGE_RGB: [u8; 3] = [16, 16, 16];

fn look(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_LOOK, &["id", "scale", "out", "inline"])?;
    let path = require_inbox(s, CMD_LOOK)?;
    let id = id_p(p, CMD_LOOK)?;
    let inline = bool_or(p, "inline", false, CMD_LOOK)?;
    let inbox = read_for(s, CMD_LOOK, &path)?;
    if inbox.was_deleted(&id) {
        return Err(bad(CMD_LOOK, format!("{id} was deleted")));
    }
    let c = inbox.find(&id).cloned().ok_or_else(|| bad(CMD_LOOK, format!("there is no comment {id}")))?;
    let seq = ItemId(c.sequence);
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or_else(|| bad(CMD_LOOK, format!("the sequence of {id} is no longer in the project")))?;
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let scale = match p.get("scale").filter(|v| !v.is_null()) {
        Some(v) => {
            v.as_f64().filter(|x| x.is_finite() && (0.05..=1.0).contains(x)).ok_or_else(|| bad(CMD_LOOK, "`scale` is the frame's share, 0.05 to 1"))? as f32
        }
        None => (LOOK_SIDE / f64::from(w.max(h))).min(1.0) as f32,
    };
    let out = match p.get("out").filter(|v| !v.is_null()) {
        Some(v) => {
            let o = v
                .as_str()
                .map(str::trim)
                .filter(|o| !o.is_empty() && o.len() <= 4096 && !o.contains('\0'))
                .ok_or_else(|| bad(CMD_LOOK, "`out` is the PNG's path"))?;
            crate::export_tools::expand_home(o)
        }
        None => {
            let dir = folder(s).ok_or_else(|| bad(CMD_LOOK, "the project is not saved"))?;
            std::path::Path::new(&dir).join(format!("look-{id}.png")).display().to_string()
        }
    };
    crate::output_safety::protect(s, &project, &out, CMD_LOOK)?;
    let end = q.duration();
    let mut notes = Vec::new();
    let mut t = Tick(c.time);
    if end.0 <= 0 {
        return Err(bad(CMD_LOOK, "the sequence is empty now"));
    }
    if t >= end {
        t = Tick(end.0.saturating_sub(q.settings.frame_rate.sane().frame_duration().0.max(1))).max(Tick::ZERO);
        notes.push(format!("the sequence is shorter now: its last frame ({}) is shown", timecode(q, t)));
    }
    if c.clip.is_some_and(|clip| q.find_item(filmcraft_project::ClipId(clip)).is_none()) {
        notes.push("the clip it points at is no longer in the sequence".to_string());
    }
    // the frame, as the Program monitor shows it (captions on)
    let (ow, oh) = filmcraft_render::output_size(q, scale);
    let (Ok(ow32), Ok(oh32)) = (u32::try_from(ow), u32::try_from(oh)) else { return Err(bad(CMD_LOOK, "the frame is too large")) };
    filmcraft_project::validate_frame_size(ow32, oh32).map_err(|e| bad(CMD_LOOK, format!("cannot render a {ow}x{oh} frame: {e}")))?;
    let img = {
        let provider = s.media.provider(project.clone(), s.services.clone());
        let opts = filmcraft_render::RenderOptions { scale, captions: true, ..Default::default() };
        filmcraft_render::render_sequence(&project, seq, t, opts, &provider)
    };
    let (iw, ih) = (u32::try_from(img.w).unwrap_or(0), u32::try_from(img.h).unwrap_or(0));
    let mut rgba = img.over_black_rgba8();
    if iw == 0 || ih == 0 || rgba.len() as u64 != u64::from(iw) * u64::from(ih) * 4 {
        return Err(bad(CMD_LOOK, "the frame could not be rendered"));
    }
    let drawn = c.area.map(|b| draw_box(&mut rgba, iw, ih, b, f64::from(iw) / f64::from(w), f64::from(ih) / f64::from(h)));
    let png = filmcraft_export::encode_png(rgba, iw, ih).map_err(|e| bad(CMD_LOOK, e.to_string()))?;
    if let Some(dir) = std::path::Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| bad(CMD_LOOK, format!("{}: {e}", dir.display())))?;
    }
    s.services.write_file(&out, &png).map_err(|e| bad(CMD_LOOK, format!("{out}: {e}")))?;
    let mut context = context_at(&project, q, t);
    // what sequence.checkMotion says about this moment (timeline only, nothing rendered)
    context["motion"] = match s.execute("sequence.checkMotion", json!({"sequence": seq.0})) {
        Ok(v) => motion_near(&v, t, q.settings.frame_rate.sane().frame_duration()),
        Err(e) => json!({"error": e.to_string()}),
    };
    let mut frame = json!({"path": out, "width": iw, "height": ih, "scale": r3(f64::from(scale))});
    if let Some(b) = drawn {
        frame["box"] = json!(b);
    }
    for (k, v) in when(q, t).as_object().cloned().unwrap_or_default() {
        frame[k] = v;
    }
    let mut answer = json!({"comment": c, "frame": frame, "context": context});
    if !notes.is_empty() {
        answer["notes"] = json!(notes);
    }
    if inline {
        answer["pngBase64"] = json!(filmcraft_project::gtemplate::base64_encode(&png));
    }
    Ok(answer)
}

/// Draw the box `b` (sequence pixels; `kx`, `ky` image pixels per sequence pixel) on the straight
/// RGBA picture: an outline with a dark edge, or a ring and cross for a point. The box in image
/// pixels.
fn draw_box(rgba: &mut [u8], w: u32, h: u32, b: [f64; 4], kx: f64, ky: f64) -> [u32; 4] {
    let (wi, hi) = (i64::from(w), i64::from(h));
    let thick = (i64::from(w.min(h)) / 270).clamp(2, 8);
    let mut put = |x: i64, y: i64, c: [u8; 3]| {
        if x < 0 || y < 0 || x >= wi || y >= hi {
            return;
        }
        let at = ((y as u64) * u64::from(w) + x as u64) as usize * 4;
        if let Some(px) = rgba.get_mut(at..at + 4) {
            px[0] = c[0];
            px[1] = c[1];
            px[2] = c[2];
            px[3] = 255;
        }
    };
    let px = |v: f64, k: f64, max: i64| -> i64 { ((v * k).round() as i64).clamp(0, max) };
    let (x0, y0) = (px(b[0], kx, wi - 1), px(b[1], ky, hi - 1));
    let (x1, y1) = (px(b[0] + b[2], kx, wi - 1), px(b[1] + b[3], ky, hi - 1));
    if x1 - x0 < 2 && y1 - y0 < 2 {
        // a point: a ring and a cross
        let r = (i64::from(w.min(h)) / 30).clamp(6, 60);
        for (rr, c) in [(r + thick, EDGE_RGB), (r, BOX_RGB)] {
            let steps = (rr * 8).clamp(32, 2048);
            for k in 0..steps {
                let a = k as f64 / steps as f64 * std::f64::consts::TAU;
                for d in 0..thick {
                    let rd = (rr - d) as f64;
                    put(x0 + (a.cos() * rd).round() as i64, y0 + (a.sin() * rd).round() as i64, c);
                }
            }
        }
        for d in -(r / 2)..=(r / 2) {
            for e in 0..thick.max(2) / 2 + 1 {
                put(x0 + d, y0 + e, BOX_RGB);
                put(x0 + e, y0 + d, BOX_RGB);
            }
        }
        return [x0 as u32, y0 as u32, 0, 0];
    }
    // the dark edge just outside, then the outline
    for (grow, c, n) in [(thick, EDGE_RGB, 1), (0, BOX_RGB, thick)] {
        let (ax, ay, bx, by) = (x0 - grow, y0 - grow, x1 + grow, y1 + grow);
        for d in 0..n {
            for x in ax..=bx {
                put(x, ay + d, c);
                put(x, by - d, c);
            }
            for y in ay..=by {
                put(ax + d, y, c);
                put(bx - d, y, c);
            }
        }
    }
    [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32]
}

fn kind_name(project: &Project, item: ItemId) -> &'static str {
    match project.item(item).map(|i| &i.kind) {
        Some(ItemKind::Media(_)) => "media",
        Some(ItemKind::Sequence(_)) => "sequence",
        Some(ItemKind::Subclip { .. }) => "subclip",
        Some(ItemKind::AdjustmentLayer { .. }) => "adjustment",
        Some(ItemKind::Graphic { .. }) => "graphic",
        None => "missing",
    }
}

/// What is on screen and around time `t` of `q`: the clips under it (graphics with their text),
/// the captions, the nearest cuts, a transition it is inside, the markers on it.
pub(crate) fn context_at(project: &Project, q: &Sequence, t: Tick) -> Value {
    const MAX: usize = 64;
    let mut clips = Vec::new();
    let tracks = q
        .video_tracks
        .iter()
        .enumerate()
        .map(|(i, tr)| (format!("V{}", i + 1), tr))
        .chain(q.audio_tracks.iter().enumerate().map(|(i, tr)| (format!("A{}", i + 1), tr)));
    for (name, tr) in tracks {
        for it in tr.items.iter().filter(|it| it.start <= t && t < it.end()) {
            if clips.len() >= MAX {
                break;
            }
            let mut v = json!({
                "clip": it.id.0, "name": it.name, "track": name, "kind": kind_name(project, it.item), "item": it.item.0,
                "start": it.start.0, "end": it.end().0, "seconds": r3(it.start.seconds()), "until": r3(it.end().seconds()),
                "enabled": it.enabled && tr.enabled,
            });
            if tr.locked {
                v["locked"] = json!(true);
            }
            clips.push(v);
        }
    }
    let mut text: Vec<Value> = Vec::new();
    for (ti, it, layer, line) in crate::craft_check::title_layers(project, q, 256) {
        if it.start <= t && t < it.end() && text.len() < MAX {
            text.push(json!({"clip": it.id.0, "track": format!("V{}", ti + 1), "layer": layer, "text": line}));
        }
    }
    let mut captions = Vec::new();
    for (ci, tr) in q.caption_tracks.iter().enumerate() {
        for c in tr.captions.iter().filter(|c| c.start <= t && t < c.end()) {
            if captions.len() >= MAX {
                break;
            }
            captions.push(json!({"track": format!("C{}", ci + 1), "caption": c.id.0, "text": c.plain_lines().join("\n"), "shown": tr.enabled, "start": c.start.0, "end": c.end().0}));
        }
    }
    let end = q.duration();
    let mut cuts = BTreeMap::<i64, Vec<u64>>::new();
    for tr in q.video_tracks.iter().filter(|t| t.enabled) {
        for it in tr.items.iter().filter(|i| i.enabled) {
            for at in [it.start.0, it.end().0].into_iter().filter(|x| *x > 0 && *x < end.0) {
                let ids = cuts.entry(at).or_default();
                if ids.len() < 8 {
                    ids.push(it.id.0);
                }
            }
        }
    }
    let cut = |(at, ids): (&i64, &Vec<u64>)| {
        let mut v = when(q, Tick(*at));
        v["away"] = json!(r3(Tick(*at - t.0).seconds()));
        v["clips"] = json!(ids);
        v
    };
    let before = cuts.range(..=t.0).next_back().map(cut);
    let after = cuts.range(t.0.saturating_add(1)..).next().map(cut);
    let mut transitions = Vec::new();
    for (i, tr) in q.video_tracks.iter().enumerate() {
        for x in tr.transitions.iter().filter(|x| x.start <= t && t < x.end()) {
            transitions.push(json!({
                "transition": x.id.0, "track": format!("V{}", i + 1), "effect": x.effect.effect, "start": x.start.0, "end": x.end().0,
                "progress": r3(x.progress(t)), "from": x.from.map(|c| c.0), "to": x.to.map(|c| c.0),
            }));
        }
    }
    let frame = q.settings.frame_rate.sane().frame_duration();
    let markers: Vec<Value> = q
        .markers
        .iter()
        .filter(|m| m.start <= t && t < Tick(m.start.0.saturating_add(m.duration.max(frame).0)))
        .take(MAX)
        .map(|m| json!({"name": m.name, "comment": m.comment, "start": m.start.0}))
        .collect();
    let mut out = Map::new();
    out.insert("clips".into(), json!(clips));
    out.insert("text".into(), json!(text));
    out.insert("captions".into(), json!(captions));
    out.insert("cuts".into(), json!({"before": before, "after": after}));
    out.insert("transition".into(), transitions.into_iter().next().unwrap_or(Value::Null));
    out.insert("markers".into(), json!(markers));
    Value::Object(out)
}

/// `sequence.checkMotion`'s findings that cover `t` (a finding without an end covers half a
/// second each side), with the check's counts.
fn motion_near(v: &Value, t: Tick, frame: Tick) -> Value {
    let half = Tick::from_seconds_f64(0.5).0;
    let near: Vec<&Value> = v["findings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| {
            let Some(a) = f["time"].as_i64() else { return false };
            match f["end"].as_i64() {
                Some(b) => a.saturating_sub(frame.0) <= t.0 && t.0 <= b.saturating_add(frame.0),
                None => a.saturating_sub(half) <= t.0 && t.0 <= a.saturating_add(half),
            }
        })
        .take(32)
        .collect();
    json!({"findings": near, "inSequence": v["findings"].as_array().map_or(0, Vec::len), "note": "sequence.checkMotion's findings at this moment; warnings only"})
}

#[cfg(test)]
#[path = "review_tests.rs"]
mod tests;
