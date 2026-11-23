//! The taste history (PX10): one record per finished video — when it was made (UTC), the
//! project's file name (only the name: its folders hold the user's paths, and the file goes into
//! bug reports), the pipeline that made it, its six fingerprint axes ([`crate::fingerprint`])
//! and the deal choices it used ([`crate::deal`]). `autocut.deal` reads it so consecutive videos
//! differ; `autocut.fingerprint` compares a sequence with its newest records and, with `record`,
//! adds one.
//!
//! The file is JSON lines, newest last, in the data directory beside the default taste profile
//! (`taste-history.jsonl` next to `taste.json`). It is bounded: at most [`MAX_RECORDS`] records
//! are kept, only the newest [`MAX_BYTES`] of a larger file are read, and a line that does not
//! parse is skipped with a note, never an error. Every text read back is capped and stripped of
//! control characters (the file is not trusted).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Session;

/// The history's file in the data directory.
pub const FILE: &str = "taste-history.jsonl";
/// Most records kept (the oldest are dropped when one is added).
pub const MAX_RECORDS: usize = 2000;
/// Most bytes read from the end of the file.
pub const MAX_BYTES: u64 = 4 * 1024 * 1024;
/// Longest text kept in a record (an id, a pipeline, a file name), characters.
const MAX_TEXT: usize = 120;
/// Most deal choices a record keeps.
const MAX_DEALS: usize = 16;
/// Most notes about skipped lines reported.
const MAX_NOTES: usize = 5;

/// The six axes of a video's fingerprint, as short ids.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Axes {
    /// `coldOpen`, `titleCard`, `montage`, `talking` or `still`.
    pub opening: String,
    /// The titles' dominant entrance (a `graphics.animateText` in-kind) or `none`.
    pub title_entrance: String,
    /// The dominant transition family or `cut`.
    pub transition_family: String,
    /// The colour look (our look, a creator look or a Lumetri preset) or `none`.
    pub color_look: String,
    /// The caption style (a preset, `plain`) or `none`.
    pub caption_style: String,
    /// `endCard`, `logo`, `fadeOut`, `title` or `cut`.
    pub close: String,
}

/// What `autocut.deal` chooses (a record's `deals` keys).
pub const DEAL_KINDS: [&str; 6] = ["look", "titleStyle", "transitionFamily", "music", "register", "endCard"];

/// The axes' names, in order.
pub const AXES: [&str; 6] = ["opening", "titleEntrance", "transitionFamily", "colorLook", "captionStyle", "close"];

impl Axes {
    /// The value of axis `name` (one of [`AXES`]).
    pub fn get(&self, name: &str) -> Option<&str> {
        Some(match name {
            "opening" => &self.opening,
            "titleEntrance" => &self.title_entrance,
            "transitionFamily" => &self.transition_family,
            "colorLook" => &self.color_look,
            "captionStyle" => &self.caption_style,
            "close" => &self.close,
            _ => return None,
        })
    }

    /// The axes both fingerprints share (an axis neither knows does not count).
    pub fn matching(&self, other: &Axes) -> Vec<&'static str> {
        AXES.iter().copied().filter(|a| matches!((self.get(a), other.get(a)), (Some(x), Some(y)) if !x.is_empty() && x == y)).collect()
    }

    fn sanitized(&self) -> Axes {
        Axes {
            opening: text(&self.opening),
            title_entrance: text(&self.title_entrance),
            transition_family: text(&self.transition_family),
            color_look: text(&self.color_look),
            caption_style: text(&self.caption_style),
            close: text(&self.close),
        }
    }
}

/// One finished video.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Record {
    /// When it was recorded: RFC 3339, UTC, whole seconds.
    pub when: String,
    /// The project's file name (no folders), when it was saved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// What made it (`direct`, `quick`, `showcase`, …), when the caller said.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<String>,
    pub axes: Axes,
    /// The deal choices it used: kind (`look`, `titleStyle`, …) → id.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub deals: BTreeMap<String, String>,
}

impl Record {
    /// Usable values (a history file is not trusted).
    pub fn sanitized(&self) -> Record {
        let opt = |s: &Option<String>| s.as_deref().map(text).filter(|t| !t.is_empty());
        Record {
            when: text(&self.when),
            project: opt(&self.project).map(|p| file_name(&p)),
            pipeline: opt(&self.pipeline),
            axes: self.axes.sanitized(),
            deals: self.deals.iter().take(MAX_DEALS).map(|(k, v)| (text(k), text(v))).filter(|(k, v)| !k.is_empty() && !v.is_empty()).collect(),
        }
    }

    /// What it used for deal kind `kind`: its deal choice, else the axis that shows it.
    pub fn used(&self, kind: &str) -> Option<&str> {
        if let Some(v) = self.deals.get(kind) {
            return Some(v.as_str());
        }
        let axis = match kind {
            "look" => &self.axes.color_look,
            "titleStyle" => &self.axes.title_entrance,
            "transitionFamily" => &self.axes.transition_family,
            "endCard" => &self.axes.close,
            "captionStyle" => &self.axes.caption_style,
            "opening" => &self.axes.opening,
            _ => return None,
        };
        (!axis.is_empty()).then_some(axis.as_str())
    }
}

/// `s` without control characters, trimmed, at most [`MAX_TEXT`] characters.
fn text(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(MAX_TEXT).collect::<String>().trim().to_string()
}

/// The last component of a path (either separator), never a folder.
pub fn file_name(path: &str) -> String {
    text(path.rsplit(['/', '\\']).next().unwrap_or_default())
}

/// The history as read: its records (oldest first) and notes on what was skipped.
#[derive(Clone, Debug, Default)]
pub struct History {
    pub path: Option<String>,
    pub records: Vec<Record>,
    pub notes: Vec<String>,
}

impl History {
    /// The newest `n` records, newest first.
    pub fn recent(&self, n: usize) -> Vec<&Record> {
        self.records.iter().rev().take(n).collect()
    }
}

/// The data directory: beside the session's preferences, else the default one.
pub fn data_dir(s: &Session) -> Option<std::path::PathBuf> {
    s.prefs_path.as_ref().and_then(|p| p.parent().map(std::path::Path::to_path_buf)).or_else(crate::autosave::default_data_dir)
}

/// The history file's path.
pub fn path(s: &Session) -> Option<String> {
    data_dir(s).map(|d| d.join(FILE).display().to_string())
}

/// The records in `bytes` (JSON lines), with notes on lines skipped. `partial_first`: the first
/// line may be cut (the read started inside the file) and is dropped.
fn parse(bytes: &[u8], partial_first: bool) -> (Vec<Record>, Vec<String>) {
    let text = String::from_utf8_lossy(bytes);
    let mut records = Vec::new();
    let mut notes = Vec::new();
    let mut skipped = 0usize;
    for (i, line) in text.lines().enumerate() {
        if i == 0 && partial_first {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Record>(line) {
            Ok(r) => records.push(r.sanitized()),
            Err(e) => {
                skipped = skipped.saturating_add(1);
                if notes.len() < MAX_NOTES {
                    notes.push(format!("line {} skipped: {}", i.saturating_add(1), text_short(&e.to_string())));
                }
            }
        }
    }
    if skipped > notes.len() {
        notes.push(format!("{skipped} lines skipped in all"));
    }
    let extra = records.len().saturating_sub(MAX_RECORDS);
    records.drain(..extra);
    (records, notes)
}

fn text_short(s: &str) -> String {
    s.chars().take(160).collect()
}

/// The history of the session's data directory (empty when there is none; never an error).
pub fn read(s: &Session) -> History {
    let Some(path) = path(s) else {
        return History { notes: vec!["no data directory: no history".into()], ..History::default() };
    };
    let Ok(size) = s.services.file_size(&path) else { return History { path: Some(path), ..History::default() } };
    let (bytes, partial) = if size > MAX_BYTES {
        let at = size.saturating_sub(MAX_BYTES);
        (s.services.read_range(&path, at, MAX_BYTES as usize), true)
    } else {
        (s.services.read_file_bounded(&path, MAX_BYTES), false)
    };
    let bytes = match bytes {
        Ok(b) => b,
        Err(e) => {
            return History { notes: vec![format!("the history could not be read: {}", text_short(&e.to_string()))], path: Some(path), ..History::default() };
        }
    };
    let (records, mut notes) = parse(&bytes, partial);
    if partial {
        notes.insert(0, format!("the history is over {} MiB: only its newest part was read", MAX_BYTES / (1024 * 1024)));
    }
    History { path: Some(path), records, notes }
}

/// `record` added to the history (the oldest dropped past [`MAX_RECORDS`]); the history as
/// written.
pub fn append(s: &Session, record: &Record) -> std::result::Result<History, String> {
    let mut h = read(s);
    let path = h.path.clone().ok_or("no data directory for the history")?;
    h.records.push(record.sanitized());
    let extra = h.records.len().saturating_sub(MAX_RECORDS);
    h.records.drain(..extra);
    let mut out = String::new();
    for r in &h.records {
        let line = serde_json::to_string(r).map_err(|e| e.to_string())?;
        out.push_str(&line);
        out.push('\n');
    }
    if let Some(dir) = std::path::Path::new(&path).parent() {
        // (the data directory may not exist yet on a first run)
        let _ = std::fs::create_dir_all(dir);
    }
    s.services.write_file(&path, out.as_bytes()).map_err(|e| format!("{path}: {e}"))?;
    Ok(h)
}

/// Now, as a record's `when` (the clock read through `web-time`, so the web build has one too).
pub fn now() -> String {
    let secs = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| i64::try_from(d.as_secs()).unwrap_or(0)).unwrap_or(0);
    when(secs)
}

/// Unix time `secs` as a record's `when` (RFC 3339, UTC, through chrono).
pub fn when(secs: i64) -> String {
    chrono::DateTime::from_timestamp(secs, 0).map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(look: &str) -> Record {
        Record { when: "2026-11-23T10:00:00Z".into(), axes: Axes { color_look: look.into(), ..Axes::default() }, ..Record::default() }
    }

    #[test]
    fn corrupt_and_hostile_lines_are_skipped_or_capped() {
        let good = serde_json::to_string(&rec("warm")).unwrap();
        let long = format!("{{\"when\":\"x\",\"project\":\"/a/b/{}\\u0007.fcproj\",\"deals\":{{\"look\":\"{}\"}}}}", "n".repeat(10_000), "v".repeat(500));
        let bytes = format!("{good}\n{{not json\n\n[1,2]\n{long}\n{good}\n");
        let (records, notes) = parse(bytes.as_bytes(), false);
        assert_eq!(records.len(), 3);
        assert_eq!(notes.len(), 2, "{notes:?}");
        let r = &records[1];
        assert!(r.project.as_deref().is_some_and(|p| !p.contains('/') && p.chars().count() <= MAX_TEXT));
        assert_eq!(r.deals["look"].chars().count(), MAX_TEXT);
        // a read that starts inside the file drops its first, cut line
        let (records, _) = parse(format!("ed\":1}}\n{good}\n").as_bytes(), true);
        assert_eq!(records.len(), 1);
        // a record's deal choice wins over its axis
        let mut r = rec("warm");
        assert_eq!(r.used("look"), Some("warm"));
        r.deals.insert("look".into(), "teal".into());
        assert_eq!((r.used("look"), r.used("music")), (Some("teal"), None));
        assert_eq!(file_name("C:\\x\\y.fcproj"), "y.fcproj");
    }

    #[test]
    fn the_history_is_bounded() {
        let dir = std::env::temp_dir().join(format!("filmcraft-taste-history-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
        assert!(read(&s).records.is_empty());
        // a file over the cap is read from its end
        let line = serde_json::to_string(&rec("cool")).unwrap();
        let n = (MAX_BYTES as usize / line.len()) + 50;
        std::fs::write(dir.join(FILE), format!("{line}\n").repeat(n)).unwrap();
        let h = read(&s);
        assert!(h.records.len() <= MAX_RECORDS && !h.records.is_empty());
        assert!(h.notes.first().is_some_and(|n| n.contains("newest part")), "{:?}", h.notes);
        let h = append(&s, &rec("warm")).unwrap();
        assert_eq!(h.records.len(), MAX_RECORDS);
        assert_eq!(read(&s).records.last().map(|r| r.axes.color_look.as_str()), Some("warm"));
        assert!(std::fs::metadata(dir.join(FILE)).unwrap().len() < MAX_BYTES);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
