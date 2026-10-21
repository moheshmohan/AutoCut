//! Feedback bundles: `app.feedbackBundle {path?}` writes one folder an agent (or a person) can
//! read to diagnose a problem, with secrets and personal paths taken out.
//!
//! | file | what |
//! |---|---|
//! | `README.txt` | what the bundle is and what was left out |
//! | `manifest.json` | app and build version, OS, when, every file with its size, what was cut or omitted |
//! | `system.json` | the system report (OS, CPU threads, decoders, export formats; the app adds its GPU) |
//! | `history.json` | the undo and redo steps (labels), when steps were made / undone / redone, the last commands run |
//! | `events.json` | the event log (failed commands, warnings) |
//! | `jobs.json` | background jobs (exports, analyses…) and how they ended |
//! | `perf.json` | `perf.stats` (the app's, with playback and frame workers, when it made the bundle) |
//! | `project.json` | `project.inspect` of the open project |
//! | `preferences.json` | the preferences |
//! | `qa.json` | the last AutoCut QA report, when there is one |
//! | `logs/` | the end of the app's logs (`filmcraft.log`, the two before it, session logs) |
//! | `crashes/` | the most recent crash reports (`crash-<day>.log`, written by the panic hook) |
//!
//! **Redaction.** Every file goes through [`redact_text`] / [`redact_json`]: the home directory
//! (and any `/Users/<name>`, `/home/<name>`, `C:\Users\<name>`) becomes `~`; e-mail addresses
//! become `[email]`; values of secret-looking keys (`token`, `secret`, `password`, `apiKey`,
//! `authorization`, `cookie`, `client_secret`, `code`…) in JSON and in `key=value` / `key: value`
//! text become `[redacted]`, as do `Bearer` / `Basic` credentials and strings shaped like OAuth
//! tokens, API keys and JWTs (`ya29.`, `1//`, `AIza…`, `sk-…`, `ghp_…`, `xox…-`, `AKIA…`, `eyJ….….`,
//! long mixed-case random strings). Token stores, the Keychain and OAuth files are never read:
//! only the files above are.
//!
//! **Size.** Each log keeps its last [`LOG_TAIL`] bytes, each crash report its last
//! [`CRASH_TAIL`], at most [`MAX_CRASHES`] of them; JSON parts larger than [`MAX_JSON_PART`] are
//! replaced by a note; the whole bundle stays under [`MAX_BUNDLE`] (parts that would pass it are
//! omitted and listed in the manifest).

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, str_p};
use crate::{Result, Session};

/// The whole bundle's cap.
pub const MAX_BUNDLE: usize = 8 << 20;
/// The end of each log kept.
pub const LOG_TAIL: usize = 1 << 20;
/// The end of each crash report kept.
pub const CRASH_TAIL: usize = 256 << 10;
/// Most crash reports kept (the newest).
pub const MAX_CRASHES: usize = 5;
/// Most session logs kept (the newest).
const MAX_SESSION_LOGS: usize = 3;
/// A JSON part larger than this is replaced by a note.
pub const MAX_JSON_PART: usize = 2 << 20;
/// Largest QA report read.
const MAX_QA_BYTES: u64 = 8 << 20;
/// Commands of the journal kept in `history.json`.
const JOURNAL_TAIL: usize = 300;
/// Deepest JSON nesting redacted (deeper values are replaced).
const MAX_DEPTH: usize = 64;
/// What a removed secret becomes.
const REDACTED: &str = "[redacted]";

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "app.feedbackBundle",
        label: "Save Feedback Bundle",
        menu: &[],
        shortcut: None,
        params: r#"{"path":"folder to write"? (default: <data dir>/Feedback/FilmCraft Feedback <date>),"qaReport":"a QA report (.qa.json) to include"?,"perf":object?,"system":object? (the app passes its own perf.stats and system report)}"#,
        enabled: always,
        run: bundle,
        journal: false,
    }]
}

/// The data directory: where the preferences are, else the default one.
fn data_dir(s: &Session) -> Option<PathBuf> {
    s.prefs_path.as_ref().and_then(|p| p.parent().map(Path::to_path_buf)).or_else(crate::autosave::default_data_dir)
}

/// The home directory to hide.
fn home_dir(s: &Session) -> Option<String> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| v.len() > 1);
    s.services.home_dir().filter(|h| h.len() > 1).or_else(|| env("HOME")).or_else(|| env("USERPROFILE")).map(|h| h.trim_end_matches(['/', '\\']).to_string())
}

/// `YYYY-MM-DD HH-MM-SS` (UTC) of a Unix time.
fn stamp(unix: u64) -> String {
    let days = i64::try_from(unix / 86_400).unwrap_or(0);
    let secs = unix % 86_400;
    // civil date from days since 1970-01-01 (H. Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}-{:02}-{:02}", secs / 3600, (secs / 60) % 60, secs % 60)
}

fn now_unix() -> u64 {
    web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// redaction

/// Key names whose values are secrets (compared lower-case, without `_` / `-`).
fn secret_key(key: &str) -> bool {
    let k: String = key.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    // (a bare `key` is left alone: `prefs.set {"key": …}` names a setting; API keys are caught
    // by their shape)
    const EXACT: [&str; 8] = ["code", "auth", "pwd", "sig", "signature", "session", "sessionid", "otp"];
    const PART: [&str; 13] = [
        "token",
        "secret",
        "password",
        "passwd",
        "passphrase",
        "apikey",
        "credential",
        "authorization",
        "cookie",
        "privatekey",
        "accesskey",
        "bearer",
        "oauth",
    ];
    EXACT.contains(&k.as_str()) || PART.iter().any(|p| k.contains(p))
}

/// Characters of a token-like word.
fn word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '~' | '+' | '/' | '=' | '%' | '@' | ':' | '\\')
}

/// A word that looks like a credential.
fn secret_word(w: &str) -> bool {
    let w = w.trim_matches(|c: char| matches!(c, '.' | ':' | ',' | '='));
    if w.len() < 12 {
        return false;
    }
    const PREFIXES: [&str; 16] =
        ["ya29.", "1//0", "aiza", "sk-", "sk_live_", "sk_test_", "rk_live_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "glpat-", "igqv", "eaab"];
    let lower = w.to_ascii_lowercase();
    if PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    if ["xoxa-", "xoxb-", "xoxp-", "xoxr-", "xoxs-"].iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    if (w.starts_with("AKIA") || w.starts_with("ASIA")) && w.len() >= 20 && w.chars().take(20).all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return true;
    }
    // a JWT: three base64url parts
    if w.starts_with("eyJ") && w.matches('.').count() >= 2 {
        return true;
    }
    // a long random string: mixed case and digits, no path or file-name punctuation
    w.len() >= 32
        && !w.contains(['/', '\\', '.', ':', '@', '~'])
        && w.chars().any(|c| c.is_ascii_uppercase())
        && w.chars().any(|c| c.is_ascii_lowercase())
        && w.chars().any(|c| c.is_ascii_digit())
}

/// An e-mail address.
fn email(w: &str) -> bool {
    let w = w.trim_matches(|c: char| matches!(c, '.' | ':' | ','));
    let Some((user, domain)) = w.split_once('@') else { return false };
    !user.is_empty()
        && user.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '%'))
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
        && domain.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
}

/// Replace home directories with `~`: `home` itself, and `/Users/<name>`, `/home/<name>`,
/// `C:\Users\<name>` (any drive, either slash).
fn hide_homes(text: &str, home: Option<&str>) -> (String, usize) {
    let mut n = 0usize;
    let mut s = match home.filter(|h| h.len() > 1) {
        Some(h) if text.contains(h) => {
            n += text.matches(h).count();
            text.replace(h, "~")
        }
        _ => text.to_string(),
    };
    for prefix in [":\\Users\\", ":/Users/", "\\Users\\", "/Users/", "/home/"] {
        let mut out = String::with_capacity(s.len());
        let mut rest = s.as_str();
        while let Some(i) = rest.find(prefix) {
            let (before, after) = rest.split_at(i);
            let name_start = prefix.len();
            let tail = after.get(name_start..).unwrap_or("");
            let seg = tail.get(..tail.find(['"', '\'', '\n', '\t', ',', ')', ']', '}', '>', '<', '|']).unwrap_or(tail.len())).unwrap_or("");
            // Windows user folders may hold spaces (`C:\Users\Ann Lee\…`): up to the next `\`;
            // elsewhere a name ends at a space too
            let name_len = if prefix.contains('\\') {
                seg.find(['\\', '/']).or_else(|| seg.find(' ')).unwrap_or(seg.len())
            } else {
                seg.find(['/', '\\', ' ']).unwrap_or(seg.len())
            };
            if name_len == 0 || tail.starts_with("Shared") {
                out.push_str(before);
                out.push_str(prefix);
                rest = tail;
                continue;
            }
            // a drive letter before `:\Users\` goes too
            let before = if prefix.starts_with(':') {
                let mut b = before.to_string();
                if b.chars().last().is_some_and(|c| c.is_ascii_alphabetic()) {
                    b.pop();
                }
                b
            } else {
                before.to_string()
            };
            out.push_str(&before);
            out.push('~');
            n += 1;
            rest = tail.get(name_len..).unwrap_or("");
        }
        out.push_str(rest);
        s = out;
    }
    (s, n)
}

/// Redact one text: home directories, e-mail addresses, secret `key=value` / `key: value` pairs,
/// `Bearer` / `Basic` credentials and token-shaped words. Returns the text and how many things
/// were replaced.
pub fn redact_text(text: &str, home: Option<&str>) -> (String, usize) {
    let (s, mut n) = hide_homes(text, home);
    let mut out = String::with_capacity(s.len());
    // the next word is a secret's value: after `token:`, `"password": `, `key =`, `Bearer`
    let mut pending = false;
    let mut chars = s.char_indices().peekable();
    while let Some(&(i, c)) = chars.peek() {
        if !word_char(c) {
            if c == '\n' {
                pending = false;
            }
            out.push(c);
            chars.next();
            continue;
        }
        let mut end = i;
        while let Some(&(j, d)) = chars.peek() {
            if !word_char(d) {
                break;
            }
            end = j + d.len_utf8();
            chars.next();
        }
        let word = s.get(i..end).unwrap_or("");
        let bare = word.trim_end_matches([':', '=']);
        let scheme = matches!(bare.to_ascii_lowercase().as_str(), "bearer" | "basic");
        // `key=value` / `key:value` in one word (URL queries, log fields)
        let split = word.find(['=', ':']).filter(|&k| k > 0 && k + 1 < word.len());
        let inline_secret = split.filter(|&k| secret_key(word.get(..k).unwrap_or("")) && !word.get(k + 1..).unwrap_or("").starts_with("//"));
        if let Some(k) = inline_secret {
            out.push_str(word.get(..=k).unwrap_or(""));
            out.push_str(REDACTED);
            n += 1;
            pending = false;
        } else if bare.is_empty() || (pending && scheme) {
            // `:` / `=` between a key and its value, or the scheme of `Authorization: Bearer x`
            out.push_str(word);
            pending |= scheme;
        } else if pending {
            out.push_str(REDACTED);
            n += 1;
            pending = false;
        } else if email(word) {
            out.push_str("[email]");
            n += 1;
        } else if secret_word(word) {
            out.push_str(REDACTED);
            n += 1;
        } else {
            out.push_str(word);
            pending = scheme || (secret_key(bare) && (word.ends_with([':', '=']) || next_is_assign(&s, end)));
        }
    }
    (out, n)
}

/// Whether the text after `at` is an assignment (`=` or `:` after optional spaces and a quote).
fn next_is_assign(s: &str, at: usize) -> bool {
    let rest = s.get(at..).unwrap_or("");
    let rest = rest.trim_start_matches(['"', '\'']).trim_start_matches([' ', '\t']);
    rest.starts_with('=') || (rest.starts_with(':') && !rest.starts_with("://"))
}

/// Redact a JSON value: secret keys' values become `[redacted]`, every string goes through
/// [`redact_text`]. Nesting deeper than [`MAX_DEPTH`] is cut.
pub fn redact_json(v: &Value, home: Option<&str>, count: &mut usize) -> Value {
    fn go(v: &Value, home: Option<&str>, count: &mut usize, depth: usize) -> Value {
        if depth > MAX_DEPTH {
            return json!("[too deep]");
        }
        match v {
            Value::String(s) => {
                let (t, n) = redact_text(s, home);
                *count += n;
                Value::String(t)
            }
            Value::Array(a) => Value::Array(a.iter().map(|x| go(x, home, count, depth + 1)).collect()),
            Value::Object(o) => Value::Object(
                o.iter()
                    .map(|(k, x)| {
                        let (key, n) = redact_text(k, home);
                        *count += n;
                        if secret_key(k) && !matches!(x, Value::Null | Value::Bool(_)) {
                            *count += 1;
                            (key, json!(REDACTED))
                        } else {
                            (key, go(x, home, count, depth + 1))
                        }
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    go(v, home, count, 0)
}

// ---------------------------------------------------------------------------------------------
// the bundle

/// One file of the bundle.
struct Part {
    name: String,
    bytes: Vec<u8>,
    note: Option<String>,
}

struct Builder {
    home: Option<String>,
    parts: Vec<Part>,
    omitted: Vec<Value>,
    redactions: usize,
    total: usize,
}

impl Builder {
    /// Add a part if it fits the bundle's budget (else list it as omitted).
    fn add(&mut self, name: &str, bytes: Vec<u8>, note: Option<String>) {
        if self.total.saturating_add(bytes.len()) > MAX_BUNDLE {
            self.omitted.push(json!({"file": name, "why": format!("the bundle would pass {} MiB", MAX_BUNDLE >> 20)}));
            return;
        }
        self.total = self.total.saturating_add(bytes.len());
        self.parts.push(Part { name: name.to_string(), bytes, note });
    }
    /// A JSON part, redacted; replaced by a note when larger than [`MAX_JSON_PART`].
    fn json(&mut self, name: &str, v: &Value) {
        let mut n = 0;
        let red = redact_json(v, self.home.as_deref(), &mut n);
        self.redactions += n;
        let bytes = serde_json::to_vec_pretty(&red).unwrap_or_default();
        if bytes.len() > MAX_JSON_PART {
            let note = json!({"omitted": format!("{name} was {} bytes, more than {} MiB", bytes.len(), MAX_JSON_PART >> 20)});
            self.add(name, serde_json::to_vec_pretty(&note).unwrap_or_default(), Some("too large: replaced by a note".into()));
        } else {
            self.add(name, bytes, None);
        }
    }
    /// The end of a text file (at most `tail` bytes, from a line start), redacted.
    fn text_tail(&mut self, s: &Session, name: &str, path: &str, tail: usize) {
        let size = match s.services.file_size(path) {
            Ok(n) => n,
            Err(_) => return,
        };
        let from = size.saturating_sub(tail as u64);
        let Ok(bytes) = s.services.read_range(path, from, tail) else {
            self.omitted.push(json!({"file": name, "why": "could not be read"}));
            return;
        };
        let text = String::from_utf8_lossy(&bytes);
        // start at a whole line when cut
        let text = if from > 0 { text.split_once('\n').map_or(text.as_ref(), |(_, rest)| rest) } else { text.as_ref() };
        let (red, n) = redact_text(text, self.home.as_deref());
        self.redactions += n;
        let note = (from > 0).then(|| format!("the last {} KiB of {} KiB", tail >> 10, size >> 10));
        self.add(name, red.into_bytes(), note);
    }
}

/// The last AutoCut QA report: `qaReport`, else the latest showcase build's, else the one next to
/// the last export (`<export>.qa.json`).
fn qa_report(s: &Session, p: &Value) -> Result<Option<(Value, String)>> {
    let read = |path: &str| -> Result<Value> {
        let size = s.services.file_size(path).map_err(|e| bad("app.feedbackBundle", format!("{path}: {e}")))?;
        if size > MAX_QA_BYTES {
            return Err(bad("app.feedbackBundle", format!("{path} is {size} bytes; at most {MAX_QA_BYTES}")));
        }
        let bytes = s.services.read_file(path).map_err(|e| bad("app.feedbackBundle", format!("{path}: {e}")))?;
        serde_json::from_slice(&bytes).map_err(|e| bad("app.feedbackBundle", format!("{path}: not JSON ({e})")))
    };
    if let Some(path) = str_p(p, "qaReport") {
        return Ok(Some((read(path)?, path.to_string())));
    }
    let latest = s
        .autocut
        .outcomes
        .iter()
        .filter_map(|(id, o)| {
            let v = o.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()?;
            let qa = v.get("qa").filter(|q| q.is_object())?.clone();
            Some((*id, qa))
        })
        .max_by_key(|(id, _)| *id);
    if let Some((id, qa)) = latest {
        return Ok(Some((qa, format!("showcase job {id}"))));
    }
    if let Some(export) = crate::photos::last_export(s) {
        let path = Path::new(&export).with_extension("qa.json").to_string_lossy().to_string();
        if s.services.file_size(&path).is_ok() {
            return Ok(read(&path).ok().map(|v| (v, path)));
        }
    }
    Ok(None)
}

/// An object parameter (the app's perf stats or system report), or None.
fn object_p(p: &Value, key: &str) -> Result<Option<Value>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v @ Value::Object(_)) => Ok(Some(v.clone())),
        Some(_) => Err(bad("app.feedbackBundle", format!("`{key}` must be an object"))),
    }
}

fn run_query(s: &mut Session, id: &str) -> Value {
    match crate::commands::find(id) {
        Some(c) => (c.run)(s, &json!({})).unwrap_or_else(|e| json!({"error": e.to_string()})),
        None => Value::Null,
    }
}

fn bundle(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "app.feedbackBundle";
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let now = now_unix();
    let dir: PathBuf = match p.get("path") {
        None | Some(Value::Null) => {
            let base = data_dir(s).map_or_else(|| crate::temp_dir().join("FilmCraft Feedback"), |d| d.join("Feedback"));
            base.join(format!("FilmCraft Feedback {}", stamp(now)))
        }
        Some(Value::String(path)) if !path.trim().is_empty() && !path.contains('\0') && path.len() <= 4096 => {
            PathBuf::from(crate::export_tools::expand_home(path.trim()))
        }
        Some(_) => return Err(bad(CMD, "`path` is the folder to write")),
    };
    if dir.is_file() {
        return Err(bad(CMD, format!("{} is a file; give a folder", dir.display())));
    }
    let perf = object_p(p, "perf")?;
    let system = object_p(p, "system")?;
    let qa = qa_report(s, p)?;
    let mut b = Builder { home: home_dir(s), parts: Vec::new(), omitted: Vec::new(), redactions: 0, total: 0 };

    // system and version
    let mut sys = system.unwrap_or_else(crate::project_tools::system_report);
    sys["build"] = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
    });
    b.json("system.json", &sys);
    // undo history and the last commands
    let journal: Vec<Value> = s.journal.iter().rev().take(JOURNAL_TAIL).rev().map(|(id, params)| json!({"command": id, "params": params})).collect();
    let history = json!({
        "undo": s.history.undo.iter().map(|h| &h.0).collect::<Vec<_>>(),
        "redo": s.history.redo.iter().map(|h| &h.0).collect::<Vec<_>>(),
        "log": s.history.log,
        "journal": journal,
    });
    b.json("history.json", &history);
    let events = run_query(s, "events.list");
    b.json("events.json", &events);
    let jobs = run_query(s, "jobs.list");
    b.json("jobs.json", &jobs);
    let perf = perf.unwrap_or_else(|| crate::perf::stats(s));
    b.json("perf.json", &perf);
    let mut project = crate::commands::inspect_project(s);
    project["file"] = json!(s.path);
    project["unsavedChanges"] = json!(s.is_dirty());
    b.json("project.json", &project);
    let prefs = serde_json::to_value(&s.prefs).unwrap_or_default();
    b.json("preferences.json", &prefs);
    match &qa {
        Some((report, from)) => b.json("qa.json", &json!({"from": from, "report": report})),
        None => b.omitted.push(json!({"file": "qa.json", "why": "no AutoCut QA report in this session (pass qaReport)"})),
    }
    // crash reports and logs, newest first
    let logs_dir = data_dir(s).map(|d| d.join("Logs"));
    let mut names: Vec<String> = logs_dir.as_ref().and_then(|d| s.services.list_dir(&d.to_string_lossy())).and_then(|r| r.ok()).unwrap_or_default();
    names.sort();
    let crashes: Vec<&String> = names.iter().filter(|n| n.starts_with("crash-") && n.ends_with(".log")).rev().take(MAX_CRASHES).collect();
    let mut sessions: Vec<&String> = names.iter().filter(|n| n.starts_with("session-") && n.ends_with(".log")).collect();
    if let Some(d) = &logs_dir {
        // the newest session logs (by modification time when the host can tell)
        sessions.sort_by_key(|n| std::fs::metadata(d.join(n)).and_then(|m| m.modified()).ok());
        sessions.reverse();
        sessions.truncate(MAX_SESSION_LOGS);
        for n in &crashes {
            b.text_tail(s, &format!("crashes/{n}"), &d.join(n).to_string_lossy(), CRASH_TAIL);
        }
        for n in ["filmcraft.log", "filmcraft.1.log", "filmcraft.2.log"].iter().filter(|n| names.iter().any(|x| x == *n)) {
            b.text_tail(s, &format!("logs/{n}"), &d.join(n).to_string_lossy(), LOG_TAIL);
        }
        for n in &sessions {
            b.text_tail(s, &format!("logs/{n}"), &d.join(n).to_string_lossy(), LOG_TAIL);
        }
    }
    if crashes.is_empty() {
        b.omitted.push(json!({"file": "crashes/", "why": "no crash reports"}));
    }
    if !b.parts.iter().any(|x| x.name.starts_with("logs/")) {
        b.omitted.push(json!({"file": "logs/", "why": match &logs_dir { Some(d) => format!("no logs in {}", redact_text(&d.to_string_lossy(), b.home.as_deref()).0), None => "no data directory".into() }}));
    }

    // README and manifest last (they list the rest)
    let files: Vec<Value> = b.parts.iter().map(|x| json!({"file": x.name, "bytes": x.bytes.len(), "note": x.note})).collect();
    let manifest = json!({
        "format": "filmcraft-feedback-bundle",
        "version": 1,
        "app": format!("FilmCraft {}", env!("CARGO_PKG_VERSION")),
        "build": if cfg!(debug_assertions) { "debug" } else { "release" },
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "createdUnix": now,
        "created": format!("{} UTC", stamp(now)),
        "files": files,
        "omitted": b.omitted,
        "redactions": b.redactions,
        "capBytes": MAX_BUNDLE,
    });
    let readme = format!(
        "FilmCraft feedback bundle\n=========================\n\nFilmCraft {} ({}, {} {}), made {} UTC.\n\nWhat is here: manifest.json lists every file with its size and what was cut or left out.\nsystem.json (OS, CPU, decoders, export formats, GPU), history.json (undo / redo steps, when\nthey were made, the last commands), events.json (failed commands and warnings), jobs.json\n(background jobs), perf.json (performance counters), project.json (the open project's tree),\npreferences.json, qa.json (the last AutoCut QA report), logs/ (the end of the app's logs) and\ncrashes/ (recent crash reports).\n\nPrivacy: the home folder is shown as ~, e-mail addresses as [email], and tokens, passwords,\nAPI keys, cookies and other secrets as [redacted] ({} replacements). Token stores, the Keychain\nand sign-in files were not read. Look the files over before you share them.\n",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) { "debug" } else { "release" },
        std::env::consts::OS,
        std::env::consts::ARCH,
        stamp(now),
        b.redactions,
    );

    // write
    let write_dir = |d: &Path| std::fs::create_dir_all(d).map_err(|e| bad(CMD, format!("{}: {e}", d.display())));
    write_dir(&dir)?;
    let mut written: Vec<Value> = Vec::new();
    let mut total = 0usize;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap_or_default();
    let all = [("README.txt".to_string(), readme.into_bytes()), ("manifest.json".to_string(), manifest_bytes)]
        .into_iter()
        .chain(b.parts.into_iter().map(|x| (x.name, x.bytes)));
    for (name, bytes) in all {
        let path = dir.join(&name);
        if let Some(parent) = path.parent() {
            write_dir(parent)?;
        }
        s.services.write_file(&path.to_string_lossy(), &bytes).map_err(|e| bad(CMD, format!("{}: {e}", path.display())))?;
        total = total.saturating_add(bytes.len());
        written.push(json!({"file": name, "bytes": bytes.len()}));
    }
    Ok(json!({
        "path": dir.to_string_lossy(),
        "files": written,
        "bytes": total,
        "redactions": manifest["redactions"],
        "omitted": manifest["omitted"],
    }))
}

#[cfg(test)]
#[path = "feedback_tests.rs"]
mod tests;
