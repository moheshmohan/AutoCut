//! `review`: the review inbox and the finish reviewer from the shell (PX8.4). Each action is one
//! engine command, run in the CLI's session on a project or in the app's over `--bridge`:
//!
//! | action | command | what it does |
//! |---|---|---|
//! | `comment` | `review.comment` | a comment pinned to a frame (and a clip or a box) |
//! | `inbox` | `review.inbox` | the open comments; `wait=true` returns when a new one arrives |
//! | `look` | `review.look` | the comment's frame as a PNG with the box drawn, and its context |
//! | `done` | `review.done` | marks a comment done with a note of what changed |
//! | `delete` | `review.delete` | removes a comment |
//! | `packet` | `review.packet` | the finish reviewer's packet: frame sheet, filmstrips, QA |
//! | `gate` | `review.gate` | records the finish reviewer's answer or the user's own words |
//!
//! Parameters are the command's own, as `key=value` pairs or one JSON object (like `exec`).
//! `look`, `done` and `delete` take the comment id first (`review done c3 note="raised the
//! title"`), and `gate --file answer.txt` reads the reviewer's answer from a file.

use std::io::Read as _;
use std::path::Path;

use serde_json::{Map, Value, json};

use crate::args::Args;
use crate::{Backend, fail, print, usage};

pub const USAGE: &str = "\
review (--project p.fcproj | --bridge addr) <action> [key=value ... | JSON]
  comment text=\"…\" [seconds=12.5 | timecode=00:00:12:12 | frame=300] [clip=ID] [box=[x,y,w,h]] [author=user]
  inbox [status=open|done|all] [since=RFC3339] [wait=true] [timeout=60]
  look <id> [scale=0.5] [out=look.png]
  done <id> note=\"what was changed\"
  delete <id>
  packet [out=DIR] [frames=40] [strips=false] [qa=report.qa.json] [preflight=true] [wait=true]
  gate [--file answer.txt | text=\"…\"] [round=1|2] [packet=DIR] [dryRun=true] | gate user=\"…\" verdict=ship";

/// The actions and the commands they run.
const ACTIONS: [(&str, &str); 7] = [
    ("comment", "review.comment"),
    ("inbox", "review.inbox"),
    ("look", "review.look"),
    ("done", "review.done"),
    ("delete", "review.delete"),
    ("packet", "review.packet"),
    ("gate", "review.gate"),
];

const OPTIONS: &[&str] = &["--project", "--bridge", "--data-dir", "--save", "--save-as", "--json", "--compact", "--file"];

/// Largest reviewer's answer `--file` reads: the gate itself refuses far less.
const MAX_ANSWER_BYTES: u64 = 256 * 1024;

/// The command and its parameters for `review <action> …`.
pub fn params(a: &Args) -> Result<(&'static str, Value), String> {
    if let Some(o) = a.unknown_opt(OPTIONS) {
        return Err(format!("unknown option `{o}`"));
    }
    let names = ACTIONS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ");
    let action = a.pos(1).ok_or_else(|| format!("what to do: {names}"))?;
    let (_, id) = ACTIONS.iter().find(|(n, _)| *n == action).ok_or_else(|| format!("unknown action `{action}`: {names}"))?;
    // look / done / delete: the comment id may come first, bare
    let bare_id = matches!(action, "look" | "done" | "delete").then(|| a.pos(2)).flatten().filter(|s| !s.contains('=') && !s.trim_start().starts_with('{'));
    let from = if bare_id.is_some() { 3 } else { 2 };
    let mut p = match a.params_from(from)? {
        Value::Object(m) => m,
        _ => return Err("parameters are key=value pairs or one JSON object".into()),
    };
    if let Some(c) = bare_id {
        put(&mut p, "id", json!(c), "the comment id")?;
    }
    if let Some(file) = a.opt("--file") {
        if action != "gate" {
            return Err("--file is the reviewer's answer for `gate`".into());
        }
        put(&mut p, "text", json!(read_answer(Path::new(file))?), "the reviewer's answer")?;
    } else if a.flag("--file") {
        return Err("--file wants the path of the reviewer's answer".into());
    }
    // the local backend makes long commands wait unless told not to: the inbox only waits when asked
    if action == "inbox" && !p.contains_key("wait") {
        p.insert("wait".into(), json!(false));
    }
    Ok((id, Value::Object(p)))
}

/// Sets `key` once: given twice (bare and as key=value, or by --file and text=) is a mistake.
fn put(p: &mut Map<String, Value>, key: &str, v: Value, what: &str) -> Result<(), String> {
    if p.contains_key(key) {
        return Err(format!("{what} is given twice"));
    }
    p.insert(key.into(), v);
    Ok(())
}

/// The reviewer's answer from a file, at most [`MAX_ANSWER_BYTES`].
fn read_answer(path: &Path) -> Result<String, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("--file {}: {e}", path.display()))?;
    let mut text = String::new();
    f.take(MAX_ANSWER_BYTES + 1).read_to_string(&mut text).map_err(|e| format!("--file {}: {e}", path.display()))?;
    if u64::try_from(text.len()).unwrap_or(u64::MAX) > MAX_ANSWER_BYTES {
        return Err(format!("--file {}: an answer is at most {} KiB", path.display(), MAX_ANSWER_BYTES / 1024));
    }
    Ok(text)
}

pub async fn run(a: &Args) {
    if a.pos(1).is_none() {
        usage(USAGE);
    }
    let (id, p) = params(a).unwrap_or_else(|e| usage(format!("review: {e}\n{USAGE}")));
    let mut b = Backend::open(a);
    match b.exec(id, p).await {
        Ok(v) => print(a, &v),
        Err(e) => fail(format!("review: {e}")),
    }
    b.finish(a).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn actions_map_to_their_commands() {
        for (action, id) in ACTIONS {
            let (got, _) = params(&args(&["review", action])).unwrap();
            assert_eq!(got, id);
        }
        assert!(params(&args(&["review", "comments"])).unwrap_err().contains("unknown action `comments`"));
        assert!(params(&args(&["review"])).unwrap_err().contains("comment, inbox"));
    }

    #[test]
    fn key_values_and_json_are_the_commands_params() {
        let (_, p) = params(&args(&["review", "comment", "text=Hold this shot longer", "seconds=12.5", "box=[10,20,100,50]"])).unwrap();
        assert_eq!(p, json!({"text": "Hold this shot longer", "seconds": 12.5, "box": [10, 20, 100, 50]}));
        let (_, p) = params(&args(&["review", "inbox", r#"{"status":"all","wait":true}"#])).unwrap();
        assert_eq!(p, json!({"status": "all", "wait": true}));
        // a plain inbox answers at once, even on the local backend that makes long commands wait
        let (_, p) = params(&args(&["review", "inbox", "status=done"])).unwrap();
        assert_eq!(p, json!({"status": "done", "wait": false}));
    }

    #[test]
    fn look_done_and_delete_take_the_id_first() {
        let (_, p) = params(&args(&["review", "done", "c3", "note=raised the title"])).unwrap();
        assert_eq!(p, json!({"id": "c3", "note": "raised the title"}));
        let (id, p) = params(&args(&["review", "look", "c1", "scale=0.5"])).unwrap();
        assert_eq!((id, p), ("review.look", json!({"id": "c1", "scale": 0.5})));
        let (_, p) = params(&args(&["review", "delete", "id=c2"])).unwrap();
        assert_eq!(p, json!({"id": "c2"}));
        assert!(params(&args(&["review", "delete", "c2", "id=c3"])).unwrap_err().contains("twice"));
        // other actions take no bare id
        assert!(params(&args(&["review", "inbox", "c2"])).unwrap_err().contains("key=value"));
    }

    #[test]
    fn gate_reads_the_answer_from_a_file_and_caps_it() {
        let dir = std::env::temp_dir().join(format!("fc-review-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let answer = dir.join("answer.txt");
        std::fs::write(&answer, "FIX 1: the title sits low\nGATE: fix 1\n").unwrap();
        let file = answer.to_str().unwrap();
        let (id, p) = params(&args(&["review", "gate", "--file", file, "round=1"])).unwrap();
        assert_eq!(id, "review.gate");
        assert_eq!(p, json!({"text": "FIX 1: the title sits low\nGATE: fix 1\n", "round": 1}));
        assert!(params(&args(&["review", "gate", "--file", file, "text=x"])).unwrap_err().contains("twice"));
        assert!(params(&args(&["review", "inbox", "--file", file])).unwrap_err().contains("for `gate`"));
        let big = dir.join("big.txt");
        std::fs::write(&big, vec![b'a'; usize::try_from(MAX_ANSWER_BYTES).unwrap() + 1]).unwrap();
        assert!(params(&args(&["review", "gate", "--file", big.to_str().unwrap()])).unwrap_err().contains("at most 256 KiB"));
        assert!(params(&args(&["review", "gate", "--file", dir.join("missing.txt").to_str().unwrap()])).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_options_are_refused() {
        assert!(params(&args(&["review", "inbox", "--wait"])).unwrap_err().contains("unknown option `--wait`"));
    }
}
