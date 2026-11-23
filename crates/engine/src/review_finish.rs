//! The finish reviewer (PX8.3): the evidence a fresh reviewer reads, and its answer checked and
//! recorded. The contract is `docs/finish-reviewer.md`; its brief is [`BRIEF`]. (The ideas of a
//! reviewer in a fresh context, a bounded list of fixes with one thing to keep, and a final gate
//! line come from impeccable (Apache-2.0), hallmark (MIT) and taste-skill (MIT); written fresh.)
//!
//! | command | does |
//! |---|---|
//! | `review.packet` | gathers the evidence into a new folder in `build/review`: the brief, a frame sheet (one frame of each shot), motion filmstrips (`sequence.motionStrips`), the latest QA report (or `delivery.preflight` run now), `sequence.checkMotion`, `build/decisions.json`, the review comments, the rounds so far and a summary of the sequence |
//! | `review.gate` | parses a reviewer's answer (one `GATE:` line, the last; a known verdict; at most eight fixes and one keep) and records it as the next round of the review run in `build/review/rounds.json`; two rounds at most; the user's own words close a run early or after its two rounds |
//!
//! A malformed answer is refused (this is a parser); nothing here blocks a delivery yet (decision
//! D7, warn first): `blocksDelivery` is always false.
use std::collections::BTreeSet;

use filmcraft_project::Sequence;
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, has_seq};
use crate::review::{bool_or, keys, r3, sequence_p, short, text_p, timecode, utc_now};
use crate::{EngineError, Result, Session};

const CMD_PACKET: &str = "review.packet";
const CMD_GATE: &str = "review.gate";
/// The limits of an answer: fixes, keeps, rounds of a run.
pub const MAX_FIXES: usize = 8;
pub const MAX_KEEPS: usize = 1;
pub const MAX_ROUNDS: usize = 2;
/// Longest answer read, bytes.
const MAX_ANSWER: usize = 64 * 1024;
/// Longest fix, keep or summary kept, characters.
const MAX_ITEM: usize = 600;
/// Longest user's words recorded, characters.
const MAX_WORDS: usize = 1000;
/// Most runs `rounds.json` keeps (the oldest go first).
const MAX_RUNS: usize = 50;
/// Largest `rounds.json`, QA report and decisions file read, bytes.
const MAX_ROUNDS_BYTES: u64 = 2 << 20;
const MAX_QA_BYTES: u64 = 16 << 20;
const MAX_DECISIONS_BYTES: u64 = 2 << 20;
/// The rounds file in the review folder.
pub const ROUNDS: &str = "rounds.json";
/// Frames on the packet's sheet: by default, at most.
const FRAMES: usize = 40;
const MAX_FRAMES: usize = 120;
/// Fewest frames a sheet shows (a film of one or two shots gets frames spread over it).
const MIN_FRAMES: usize = 8;
/// The long side of a sheet's tile, pixels; tiles in a row.
const TILE_SIDE: f64 = 320.0;
const TILES_PER_ROW: usize = 8;

/// The brief a fresh reviewer gets first (`brief.md` in every packet; `{round}` is filled in).
pub const BRIEF: &str = "# Finish review

You are the finish reviewer of a short film. You did not make it. Judge it from the evidence in
this folder alone: what you were told about it before this brief does not count.

Read in this order:

1. `comments.json`: what the user pointed at, by frame (`open` first). Every open comment gets an
   answer: a FIX line, or the reason it needs none.
2. `frames.jpg`: one frame of each shot, in order, labelled with its number and time.
3. `strips/`: seven frames of every animated title, transition and overlay (just before, first,
   three between, last, settled) and the findings on them (missing when nothing moves).
4. `qa.json`: the latest quality report (`delivery.preflight`), and `motion.json`:
   `sequence.checkMotion` (eases, entrance and exit timing, stagger, loops, pops).
5. `decisions.json`: what was decided on purpose, with reasons and evidence. Do not reopen it.
6. `sequence.json`: the film's size, rate, length, shots, titles and captions.

Answer with one of four verdicts:

- **ship**: it is ready; nothing left is worth another round.
- **fix**: one to eight changes, the most important first, each on its own line
  `FIX n: <time or comment id> <what to change, and how to tell it is right>`, and at most one
  `KEEP: <what must not change while fixing>`.
- **rebuild**: the cut does not work as it is (structure, order, pacing); say why on the GATE line.
- **recapture**: material is missing or unusable (a shot, a sound, a take); say what on the GATE
  line.

End the answer with exactly one gate line, its last line:

    GATE: ship
    GATE: fix <n> [keep 1]
    GATE: rebuild - <why>
    GATE: recapture - <what>

Limits: at most eight FIX lines and one KEEP line; no new ideas beyond the fixes; two rounds at
most (this packet is round {round}). Only the user's own words end the review early.
";

type Run = fn(&mut Session, &Value) -> Result<Value>;
type Enabled = fn(&Session) -> std::result::Result<(), String>;

fn spec(id: &'static str, label: &'static str, params: &'static str, enabled: Enabled, run: Run) -> CommandSpec {
    CommandSpec { id, label, params, run, enabled, menu: &[], shortcut: None, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            CMD_PACKET,
            "Finish Review Packet",
            r#"{"out":path? (a new or empty folder; default build/review/packet-<n>),"sequence":id? (default the active one),"frames":int 1..120=40 (frames on the sheet: one of each shot, at most this many),"strips":bool=true (motion filmstrips, sequence.motionStrips),"qa":path? (the QA report to include; default the newest *.qa.json in build/ or beside the project),"preflight":bool=false (run delivery.preflight now and include its report: slow),"wait":bool=false (render the filmstrips in the call; else they are a job, its id in `job`)}"#,
            has_seq,
            packet,
        ),
        spec(
            CMD_GATE,
            "Finish Review Gate",
            r#"{"text":str? (the reviewer's whole answer: FIX/KEEP lines, then one last line GATE: ship | fix <n> [keep 1] | rebuild - <why> | recapture - <what>),"round":1|2? (checked against the run: a third round is refused),"packet":path? (the packet it reviewed),"dryRun":bool=false (check it, record nothing),"user":str? (the user's own words: they close the open run early, or end a run that used its two rounds so a new one can start),"verdict":"ship"|"fix"|"rebuild"|"recapture"? (with user: what they decided)} — with neither text nor user: the runs so far"#,
            always,
            gate,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// the answer

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Ship,
    Fix,
    Rebuild,
    Recapture,
}

impl Verdict {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ship" => Some(Self::Ship),
            "fix" => Some(Self::Fix),
            "rebuild" => Some(Self::Rebuild),
            "recapture" => Some(Self::Recapture),
            _ => None,
        }
    }
}

/// A FIX or KEEP line: its text and the comments and times it names.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub times: Vec<String>,
}

/// A reviewer's answer, parsed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    pub verdict: Verdict,
    pub fixes: Vec<Item>,
    pub keep: Vec<Item>,
    /// What the GATE line says after its verdict (the why of a rebuild or recapture).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// A line with markdown decoration (`- `, `> `, `**`, backticks) taken off both ends.
fn bare(line: &str) -> &str {
    line.trim().trim_start_matches(['-', '*', '>', '`', '#', ' ']).trim_end_matches(['*', '`', ' ']).trim()
}

/// `word` (any case), an optional number, then `:` at the start of `line`: the number and the
/// rest.
fn labelled<'a>(line: &'a str, word: &str) -> Option<(Option<usize>, &'a str)> {
    let head = line.get(..word.len())?;
    if !head.eq_ignore_ascii_case(word) {
        return None;
    }
    let rest = line.get(word.len()..)?.trim_start();
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let number = rest.get(..digits).filter(|d| !d.is_empty()).and_then(|d| d.parse().ok());
    let rest = rest.get(digits..)?.trim_start_matches(['*', ' ']);
    let rest = rest.strip_prefix(':').or_else(|| rest.strip_prefix('.')).or_else(|| rest.strip_prefix(')'))?;
    Some((number, rest.trim_start_matches(['*', ' ']).trim()))
}

/// The comment ids (`c12`) and times (`00:01:02:03`, `1:02`, `12.5s`) a line names.
fn refs(text: &str) -> (Vec<String>, Vec<String>) {
    let mut comments = BTreeSet::new();
    let mut times = Vec::new();
    for raw in text.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | '[' | ']' | '—' | '–')) {
        let w = raw.trim_matches(|c: char| matches!(c, '.' | ';' | '"' | '\'' | '`' | '*'));
        if w.len() > 1 && w.len() <= 10 && w.starts_with('c') && w.get(1..).is_some_and(|d| d.bytes().all(|b| b.is_ascii_digit())) {
            comments.insert(w.to_string());
            continue;
        }
        let tc = !w.is_empty()
            && w.split([':', ';']).count() >= 2
            && w.split([':', ';', '.']).all(|p| !p.is_empty() && p.len() <= 3 && p.bytes().all(|b| b.is_ascii_digit()));
        let secs = w.strip_suffix('s').is_some_and(|n| !n.is_empty() && n.parse::<f64>().is_ok_and(|x| x.is_finite() && x >= 0.0));
        if (tc || secs) && times.len() < 8 {
            times.push(w.to_string());
        }
    }
    (comments.into_iter().take(16).collect(), times)
}

fn item(text: &str) -> Item {
    let text = short(text, MAX_ITEM);
    let (comments, times) = refs(&text);
    Item { text, comments, times }
}

/// Parse a reviewer's answer: refused with the reason when it breaks the contract.
pub fn parse_answer(text: &str) -> std::result::Result<Answer, String> {
    if text.len() > MAX_ANSWER {
        return Err(format!("the answer is {} bytes; at most {MAX_ANSWER}", text.len()));
    }
    let lines: Vec<&str> = text.lines().collect();
    let gates: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| bare(l).starts_with("GATE:")).map(|(i, _)| i).collect();
    let Some(&at) = gates.first() else {
        return Err("no `GATE:` line: the answer ends on one, e.g. `GATE: fix 3 keep 1`".into());
    };
    if gates.len() > 1 {
        let which: Vec<String> = gates.iter().take(6).map(|g| (g + 1).to_string()).collect();
        return Err(format!("{} `GATE:` lines (lines {}): exactly one, the last line", gates.len(), which.join(", ")));
    }
    if lines.iter().skip(at + 1).any(|l| !bare(l).is_empty()) {
        return Err("the `GATE:` line is not the last line: it ends the answer".into());
    }
    let gate_line = lines.get(at).map(|l| bare(l)).unwrap_or_default();
    let rest = gate_line.strip_prefix("GATE:").unwrap_or_default().trim_start_matches(['*', ' ']).trim();
    // the verdict and its counts, then a summary after a dash (or a colon)
    let cut = ["—", "–", " -- ", " - ", ":"].iter().filter_map(|sep| rest.find(sep).map(|i| (i, sep.len()))).min();
    let (head, summary) = match cut {
        Some((i, n)) => (rest.get(..i).unwrap_or_default(), rest.get(i + n..).unwrap_or_default().trim()),
        None => (rest, ""),
    };
    let summary = short(summary, MAX_ITEM);
    let words: Vec<String> = head
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| matches!(c, ',' | ';' | '.' | '*' | '`')).to_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    let Some(first) = words.first() else { return Err("the `GATE:` line has no verdict: ship, fix, rebuild or recapture".into()) };
    let verdict = Verdict::parse(first).ok_or_else(|| format!("`{}` is not a verdict: ship, fix, rebuild or recapture", short(first, 24)))?;
    let (mut said_fixes, mut said_keep) = (None, None);
    let grammar = "GATE: ship | fix <n> [keep 1] | rebuild - <why> | recapture - <what>";
    let mut w = words.iter().skip(1).peekable();
    while let Some(word) = w.next() {
        match (verdict, word.as_str()) {
            (Verdict::Fix, n) if said_fixes.is_none() && said_keep.is_none() && n.parse::<usize>().is_ok() => said_fixes = n.parse::<usize>().ok(),
            (Verdict::Fix, "fixes") if said_fixes.is_some() => {}
            (Verdict::Fix | Verdict::Ship, "keep" | "keeps") if said_keep.is_none() => {
                said_keep = Some(match w.peek().and_then(|n| n.parse::<usize>().ok()) {
                    Some(k) => {
                        w.next();
                        k
                    }
                    None => 1,
                });
            }
            _ => return Err(format!("unexpected `{}` on the GATE line: {grammar}", short(word, 24))),
        }
    }
    let mut fixes = Vec::new();
    let mut keep = Vec::new();
    let mut warnings = Vec::new();
    let mut numbers = Vec::new();
    for raw in lines.iter().take(at) {
        let l = bare(raw);
        if let Some((n, t)) = labelled(l, "FIX") {
            if t.is_empty() {
                return Err(format!("FIX {} has no text: say what to change", fixes.len() + 1));
            }
            numbers.push(n);
            fixes.push(item(t));
        } else if let Some((_, t)) = labelled(l, "KEEP") {
            if t.is_empty() {
                return Err("a KEEP line has no text: say what must not change".into());
            }
            keep.push(item(t));
        }
    }
    if keep.len() > MAX_KEEPS {
        return Err(format!("{} KEEP lines: at most {MAX_KEEPS}, the one thing that must not change", keep.len()));
    }
    if said_keep.is_some_and(|k| k != keep.len()) {
        return Err(format!("the GATE line says keep {}, the answer has {} KEEP line(s)", said_keep.unwrap_or_default(), keep.len()));
    }
    match verdict {
        Verdict::Fix => {
            if fixes.is_empty() {
                return Err("a fix answer lists its fixes, one `FIX n: …` line each".into());
            }
            if fixes.len() > MAX_FIXES {
                return Err(format!("{} FIX lines: at most {MAX_FIXES}, keep the ones that matter most", fixes.len()));
            }
            if let Some(n) = said_fixes.filter(|n| *n != fixes.len()) {
                return Err(format!("the GATE line says fix {n}, the answer lists {} FIX line(s)", fixes.len()));
            }
            if numbers.iter().enumerate().any(|(i, n)| n.is_some_and(|n| n != i + 1)) {
                warnings.push("the FIX lines are not numbered 1, 2, 3… in order".into());
            }
            let vague = fixes.iter().filter(|f| f.comments.is_empty() && f.times.is_empty()).count();
            if vague > 0 {
                warnings.push(format!("{vague} FIX line(s) name no time or comment id: say where"));
            }
        }
        Verdict::Ship => {
            if !fixes.is_empty() {
                return Err(format!("a ship answer lists no FIX lines (it lists {}): fixes mean `GATE: fix`", fixes.len()));
            }
        }
        Verdict::Rebuild | Verdict::Recapture => {
            if summary.is_empty() {
                let what = if verdict == Verdict::Rebuild { "why the cut does not work" } else { "what to shoot or fetch again" };
                return Err(format!("say {what} on the GATE line, after a dash: `GATE: {} - …`", first));
            }
            if !fixes.is_empty() {
                warnings.push(format!("{} FIX line(s) are kept as notes: a {first} is not a list of fixes", fixes.len()));
            }
        }
    }
    Ok(Answer { verdict, fixes, keep, summary, warnings })
}

// ---------------------------------------------------------------------------------------------
// the rounds

/// One round of a review run: a reviewer's answer, recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Round {
    pub round: usize,
    #[serde(flatten)]
    pub answer: Answer,
    /// UTC.
    pub at: String,
    /// The packet's folder name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet: Option<String>,
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Closed {
    /// `gate` (a ship), `limit` (two rounds without a ship: the user decides) or `user`.
    pub by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// The user's own words, when they closed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<String>,
    pub at: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewRun {
    pub run: u64,
    pub started: String,
    pub rounds: Vec<Round>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed: Option<Closed>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rounds {
    pub format: String,
    pub version: u32,
    pub runs: Vec<ReviewRun>,
}

impl Default for Rounds {
    fn default() -> Self {
        Rounds { format: "autocut.reviewRounds".into(), version: 1, runs: Vec::new() }
    }
}

impl Rounds {
    /// The open run's next round number (1 when no run is open), or why no round can be
    /// recorded now.
    fn next_round(&self) -> std::result::Result<(u64, usize), String> {
        match self.runs.last() {
            Some(r) if r.closed.is_none() => Ok((r.run, r.rounds.len().saturating_add(1))),
            Some(r) if r.closed.as_ref().is_some_and(|c| c.by == "limit") => Err(format!(
                "review run {} used its {MAX_ROUNDS} rounds without a ship: the user decides now; record their own words with review.gate {{user}} to start another run",
                r.run
            )),
            Some(r) => Ok((r.run.saturating_add(1), 1)),
            None => Ok((1, 1)),
        }
    }
    fn json(&self) -> Value {
        let open = self.runs.last().filter(|r| r.closed.is_none()).map(|r| r.run);
        let next = match self.next_round() {
            Ok((run, round)) => json!({"run": run, "round": round}),
            Err(why) => json!({"refused": why}),
        };
        json!({"runs": self.runs, "openRun": open, "next": next})
    }
}

fn rounds_path(s: &Session, cmd: &str) -> Result<String> {
    let dir =
        crate::review::folder(s).ok_or_else(|| bad(cmd, "the project is not saved: the review rounds live in build/review beside it, so save it first"))?;
    Ok(std::path::Path::new(&dir).join(ROUNDS).display().to_string())
}

fn read_rounds(s: &Session, path: &str, cmd: &str) -> Result<Rounds> {
    match s.services.file_size(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Rounds::default()),
        Err(e) => return Err(bad(cmd, format!("{path}: {e}"))),
        Ok(n) if n > MAX_ROUNDS_BYTES => return Err(bad(cmd, format!("{path} is {n} bytes, over its {MAX_ROUNDS_BYTES} byte limit"))),
        Ok(_) => {}
    }
    let bytes = s.services.read_file_bounded(path, MAX_ROUNDS_BYTES).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("{path} is not a rounds file ({e}); move it aside to start again")))
}

fn write_rounds(s: &Session, path: &str, rounds: &mut Rounds, cmd: &str) -> Result<()> {
    let drop = rounds.runs.len().saturating_sub(MAX_RUNS);
    rounds.runs.drain(..drop);
    let bytes = serde_json::to_vec_pretty(rounds).map_err(|e| bad(cmd, e.to_string()))?;
    if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| bad(cmd, format!("{}: {e}", dir.display())))?;
    }
    s.services.write_file(path, &bytes).map_err(|e| bad(cmd, format!("{path}: {e}")))
}

// ---------------------------------------------------------------------------------------------
// review.gate

fn gate(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_GATE, &["text", "round", "packet", "dryRun", "user", "verdict"])?;
    let path = rounds_path(s, CMD_GATE)?;
    let mut rounds = read_rounds(s, &path, CMD_GATE)?;
    let has = |k: &str| p.get(k).is_some_and(|v| !v.is_null());
    if has("text") && has("user") {
        return Err(bad(CMD_GATE, "give the reviewer's answer (`text`) or the user's own words (`user`), not both"));
    }
    if has("user") {
        return user_words(s, p, &path, &mut rounds);
    }
    if !has("text") {
        if has("verdict") || has("round") {
            return Err(bad(CMD_GATE, "`text` is the reviewer's answer"));
        }
        return Ok(rounds.json());
    }
    if has("verdict") {
        return Err(bad(CMD_GATE, "`verdict` goes with `user`; a reviewer's verdict is on its GATE line"));
    }
    let text = p.get("text").and_then(Value::as_str).ok_or_else(|| bad(CMD_GATE, "`text` is the reviewer's whole answer"))?;
    let answer = parse_answer(text).map_err(|e| bad(CMD_GATE, e))?;
    let (run, round) = rounds.next_round().map_err(|e| bad(CMD_GATE, e))?;
    if let Some(v) = p.get("round").filter(|v| !v.is_null()) {
        let want = v.as_u64().ok_or_else(|| bad(CMD_GATE, "`round` is 1 or 2"))?;
        if want > MAX_ROUNDS as u64 {
            return Err(bad(CMD_GATE, format!("round {want} is refused: {MAX_ROUNDS} rounds at most, then the user decides")));
        }
        if want == 0 || want != round as u64 {
            return Err(bad(CMD_GATE, format!("this answer would be round {round} of review run {run}, not round {want}")));
        }
    }
    if round > MAX_ROUNDS {
        return Err(bad(CMD_GATE, format!("review run {run} has had its {MAX_ROUNDS} rounds: the user decides")));
    }
    let packet = match p.get("packet").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => {
            let raw = v
                .as_str()
                .map(str::trim)
                .filter(|t| !t.is_empty() && t.len() <= 4096 && !t.contains('\0'))
                .ok_or_else(|| bad(CMD_GATE, "`packet` is the packet's folder"))?;
            // its name only: no folder paths in the record
            Some(std::path::Path::new(raw).file_name().map_or_else(|| short(raw, 120), |n| short(&n.to_string_lossy(), 120)))
        }
    };
    let dry = bool_or(p, "dryRun", false, CMD_GATE)?;
    let closed = match answer.verdict {
        Verdict::Ship => Some(Closed { by: "gate".into(), verdict: Some(Verdict::Ship), words: None, at: utc_now() }),
        v if round >= MAX_ROUNDS => Some(Closed { by: "limit".into(), verdict: Some(v), words: None, at: utc_now() }),
        _ => None,
    };
    let next = match (answer.verdict, round >= MAX_ROUNDS) {
        (Verdict::Ship, _) => "ship: deliver the final (delivery.prepare); the review run is closed".to_string(),
        (_, true) => format!("that was round {MAX_ROUNDS}: give the verdict and its list to the user, who decides; a new run starts only after their own words (review.gate {{user}})"),
        (Verdict::Fix, false) => "make the fixes without touching what KEEP names, mark the comments they answer done (review.done), build the review cut again, then a new review.packet and a fresh reviewer for round 2".to_string(),
        (Verdict::Rebuild, false) => "rebuild the cut for the reason given, then a new review.packet and a fresh reviewer for round 2".to_string(),
        (Verdict::Recapture, false) => "get the material named (shoot, fetch or generate it), rebuild the review cut, then a new review.packet and a fresh reviewer for round 2".to_string(),
    };
    let entry = Round { round, answer: answer.clone(), at: utc_now(), packet };
    if !dry {
        match rounds.runs.last_mut().filter(|r| r.run == run && r.closed.is_none()) {
            Some(r) => r.rounds.push(entry.clone()),
            None => rounds.runs.push(ReviewRun { run, started: utc_now(), rounds: vec![entry.clone()], closed: None }),
        }
        if let Some(r) = rounds.runs.last_mut() {
            r.closed = closed.clone();
        }
        write_rounds(s, &path, &mut rounds, CMD_GATE)?;
    }
    Ok(json!({
        "recorded": !dry,
        "run": run,
        "round": round,
        "gate": answer,
        "closed": closed,
        "next": next,
        "blocksDelivery": false,
        "note": "warn first (decision D7): the gate is recorded, nothing here blocks a delivery yet",
        "rounds": path,
    }))
}

/// The user's own words: they close the open run early, or end a run that used its two rounds.
fn user_words(s: &Session, p: &Value, path: &str, rounds: &mut Rounds) -> Result<Value> {
    if p.get("text").is_some_and(|v| !v.is_null()) || p.get("round").is_some_and(|v| !v.is_null()) || p.get("dryRun").is_some_and(|v| !v.is_null()) {
        return Err(bad(CMD_GATE, "`user` goes alone (with `verdict` if they gave one)"));
    }
    let words = text_p(p, "user", MAX_WORDS, CMD_GATE)?;
    let verdict = match p.get("verdict").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_str().and_then(Verdict::parse).ok_or_else(|| bad(CMD_GATE, "`verdict` is ship, fix, rebuild or recapture"))?),
    };
    let last = rounds.runs.last_mut().ok_or_else(|| bad(CMD_GATE, "there is no review run to close"))?;
    let ends = match &last.closed {
        None => "closed early",
        Some(c) if c.by == "limit" => "ended after its rounds",
        Some(_) => return Err(bad(CMD_GATE, format!("review run {} is closed already; the next answer starts a new run", last.run))),
    };
    let run = last.run;
    last.closed = Some(Closed { by: "user".into(), verdict, words: Some(words), at: utc_now() });
    let closed = last.closed.clone();
    write_rounds(s, path, rounds, CMD_GATE)?;
    Ok(
        json!({"run": run, "closed": closed, "note": format!("review run {run} {ends} on the user's own words; the next answer starts a new run"), "blocksDelivery": false}),
    )
}

// ---------------------------------------------------------------------------------------------
// review.packet

/// The cut points of the enabled video tracks of `q`, inside it.
fn cuts(q: &Sequence) -> Vec<Tick> {
    let end = q.duration();
    let mut set = BTreeSet::new();
    for tr in q.video_tracks.iter().filter(|t| t.enabled) {
        for it in tr.items.iter().filter(|i| i.enabled) {
            for at in [it.start, it.end()] {
                if at > Tick::ZERO && at < end {
                    set.insert(at);
                }
            }
        }
    }
    set.into_iter().collect()
}

/// The sheet's times: the middle frame of each shot, at most `max`; a film of few shots gets
/// frames spread over it too.
fn sheet_times(q: &Sequence, max: usize) -> Vec<Tick> {
    let end = q.duration();
    let rate = q.settings.frame_rate.sane();
    let mut bounds = vec![Tick::ZERO];
    bounds.extend(cuts(q));
    bounds.push(end);
    let mut times: Vec<Tick> = bounds.windows(2).filter_map(|w| Some(rate.snap(Tick(w.first()?.0 + (w.get(1)?.0 - w.first()?.0) / 2)))).collect();
    if times.len() > max {
        // evenly picked among the shots
        let n = times.len();
        times = (0..max).filter_map(|k| times.get(k * n / max).copied()).collect();
    }
    let mut seen = BTreeSet::new();
    let mut out: Vec<Tick> = times.into_iter().filter(|t| *t >= Tick::ZERO && *t < end && seen.insert(rate.frame_at(*t))).collect();
    // a film of few shots: frames spread over it too, those furthest from the shots' first
    let want = MIN_FRAMES.min(max);
    if out.len() < want {
        let mut spread: Vec<Tick> = (0..want).map(|k| rate.snap(Tick(((end.0 as f64) * (k as f64 + 0.5) / want as f64) as i64))).collect();
        let away = |t: &Tick, from: &[Tick]| from.iter().map(|o| (o.0 - t.0).saturating_abs()).min().unwrap_or(i64::MAX);
        while out.len() < want && !spread.is_empty() {
            let Some((i, _)) = spread.iter().enumerate().max_by_key(|(i, t)| (away(t, &out), std::cmp::Reverse(*i))) else { break };
            let t = spread.swap_remove(i);
            if t < end && seen.insert(rate.frame_at(t)) {
                out.push(t);
            }
        }
    }
    out.sort();
    out.truncate(max);
    out
}

/// The newest `*.qa.json` in the build folder or beside the project.
fn newest_qa(s: &Session) -> Option<String> {
    let project = s.path.as_deref().filter(|p| !p.is_empty())?;
    let dir = std::path::Path::new(project).parent()?;
    let mut best: Option<(u64, String)> = None;
    for d in [dir.join("build"), dir.to_path_buf()] {
        let d = d.display().to_string();
        let Some(Ok(entries)) = s.services.list_entries(&d) else { continue };
        for e in entries.into_iter().filter(|e| !e.is_dir && e.name.ends_with(".qa.json")).take(10_000) {
            let when = e.modified.unwrap_or(0);
            if best.as_ref().is_none_or(|b| when > b.0) {
                best = Some((when, std::path::Path::new(&d).join(&e.name).display().to_string()));
            }
        }
    }
    best.map(|b| b.1)
}

/// The next free `packet-<n>` in the review folder.
fn next_packet(s: &Session, dir: &str) -> String {
    let taken: BTreeSet<u64> = match s.services.list_dir(dir) {
        Some(Ok(names)) => names.iter().filter_map(|n| n.strip_prefix("packet-")?.parse().ok()).collect(),
        _ => BTreeSet::new(),
    };
    let n = taken.last().map_or(1, |m| m.saturating_add(1));
    std::path::Path::new(dir).join(format!("packet-{n}")).display().to_string()
}

fn packet(s: &mut Session, p: &Value) -> Result<Value> {
    keys(p, CMD_PACKET, &["out", "sequence", "frames", "strips", "qa", "preflight", "wait"])?;
    let review_dir =
        crate::review::folder(s).ok_or_else(|| bad(CMD_PACKET, "the project is not saved: the packet goes in build/review beside it, so save it first"))?;
    let seq = sequence_p(s, p, CMD_PACKET)?;
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    q.check_bounds().map_err(|e| bad(CMD_PACKET, e))?;
    let end = q.duration();
    if end.0 <= 0 {
        return Err(bad(CMD_PACKET, "the sequence is empty"));
    }
    let frames = match p.get("frames").filter(|v| !v.is_null()) {
        None => FRAMES,
        Some(v) => {
            v.as_u64().filter(|n| (1..=MAX_FRAMES as u64).contains(n)).ok_or_else(|| bad(CMD_PACKET, format!("`frames` is 1 to {MAX_FRAMES}")))? as usize
        }
    };
    let want_strips = bool_or(p, "strips", true, CMD_PACKET)?;
    let preflight = bool_or(p, "preflight", false, CMD_PACKET)?;
    let wait = bool_or(p, "wait", false, CMD_PACKET)?;
    let qa_from = match p.get("qa").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(crate::export_tools::expand_home(
            v.as_str()
                .map(str::trim)
                .filter(|t| !t.is_empty() && t.len() <= 4096 && !t.contains('\0'))
                .ok_or_else(|| bad(CMD_PACKET, "`qa` is the path of a QA report"))?,
        )),
    };
    if qa_from.is_some() && preflight {
        return Err(bad(CMD_PACKET, "give `qa` or `preflight`, not both"));
    }
    let out = match p.get("out").filter(|v| !v.is_null()) {
        Some(v) => crate::export_tools::expand_home(
            v.as_str()
                .map(str::trim)
                .filter(|o| !o.is_empty() && o.len() <= 4096 && !o.contains('\0'))
                .ok_or_else(|| bad(CMD_PACKET, "`out` is the packet's folder"))?,
        ),
        None => next_packet(s, &review_dir),
    };
    match s.services.list_dir(&out) {
        Some(Ok(names)) if !names.is_empty() => return Err(bad(CMD_PACKET, format!("{out} is not empty: choose a new or empty folder for the packet"))),
        Some(Err(e)) if e.kind() != std::io::ErrorKind::NotFound => return Err(bad(CMD_PACKET, format!("{out}: {e}"))),
        _ => {}
    }
    let file = |name: &str| std::path::Path::new(&out).join(name).display().to_string();
    for name in ["brief.md", "frames.jpg", "sequence.json", "motion.json", "qa.json", "decisions.json", "comments.json", "rounds.json", "manifest.json"] {
        crate::output_safety::protect(s, &project, &file(name), CMD_PACKET)?;
    }
    std::fs::create_dir_all(&out).map_err(|e| bad(CMD_PACKET, format!("{out}: {e}")))?;
    let mut files: Vec<String> = Vec::new();
    let mut missing: Vec<Value> = Vec::new();
    let write = |s: &Session, files: &mut Vec<String>, name: &str, bytes: &[u8]| -> Result<()> {
        s.services.write_file(&file(name), bytes).map_err(|e| bad(CMD_PACKET, format!("{}: {e}", file(name))))?;
        files.push(name.to_string());
        Ok(())
    };
    let pretty = |v: &Value| serde_json::to_vec_pretty(v).unwrap_or_default();
    // the run and round this packet is for
    let rounds_file = rounds_path(s, CMD_PACKET)?;
    let rounds = read_rounds(s, &rounds_file, CMD_PACKET).unwrap_or_default();
    let (run, round, refused) = match rounds.next_round() {
        Ok((run, round)) => (run, round, None),
        Err(why) => (rounds.runs.last().map_or(1, |r| r.run), MAX_ROUNDS, Some(why)),
    };
    write(s, &mut files, "brief.md", BRIEF.replace("{round}", &round.to_string()).as_bytes())?;
    write(s, &mut files, "rounds.json", &pretty(&rounds.json()))?;
    // the sequence in words
    let rate = q.settings.frame_rate.sane();
    let shot_bounds = cuts(q);
    let titles: Vec<Value> = crate::craft_check::title_layers(&project, q, 200)
        .into_iter()
        .map(|(ti, it, layer, text)| json!({"clip": it.id.0, "track": format!("V{}", ti + 1), "layer": layer, "text": text, "start": timecode(q, it.start), "end": timecode(q, it.end())}))
        .collect();
    let captions: usize = q.caption_tracks.iter().map(|t| t.captions.len()).sum();
    let name = project.item(seq).map(|i| i.name.clone()).unwrap_or_default();
    let summary = json!({
        "sequence": seq.0, "name": name, "width": q.settings.width, "height": q.settings.height, "fps": r3(rate.as_f64()),
        "duration": crate::review::when(q, end), "shots": shot_bounds.len() + 1,
        "cuts": shot_bounds.iter().take(2000).map(|t| timecode(q, *t)).collect::<Vec<_>>(),
        "titles": titles, "captions": captions, "captionTracks": q.caption_tracks.len(),
    });
    write(s, &mut files, "sequence.json", &pretty(&summary))?;
    // one frame of each shot
    let times = sheet_times(q, frames);
    let scale = (TILE_SIDE / f64::from(q.settings.width.max(q.settings.height).max(1))).min(1.0) as f32;
    let (ow, oh) = filmcraft_render::output_size(q, scale);
    let size_ok = u32::try_from(ow).ok().zip(u32::try_from(oh).ok()).is_some_and(|(w, h)| filmcraft_project::validate_frame_size(w, h).is_ok());
    let sheet = if size_ok && !times.is_empty() {
        let rows: Vec<Vec<(filmcraft_render::Image, String)>> = {
            let provider = s.media.provider(project.clone(), s.services.clone());
            let opts = filmcraft_render::RenderOptions { scale, captions: true, ..Default::default() };
            let tiles: Vec<(filmcraft_render::Image, String)> = times
                .iter()
                .enumerate()
                .map(|(n, t)| (filmcraft_render::render_sequence(&project, seq, *t, opts, &provider), format!("{} {}", n + 1, timecode(q, *t))))
                .collect();
            tiles.chunks(TILES_PER_ROW).map(<[_]>::to_vec).collect()
        };
        crate::contact_sheet::write_images(s, CMD_PACKET, &rows, &file("frames.jpg")).map(|v| {
            files.push("frames.jpg".into());
            json!({"file": "frames.jpg", "frames": times.len(), "width": v["width"], "height": v["height"], "times": times.iter().map(|t| timecode(q, *t)).collect::<Vec<_>>()})
        })
    } else {
        Err(bad(CMD_PACKET, "the frame is too large to render"))
    };
    let sheet = sheet.unwrap_or_else(|e| {
        missing.push(json!({"file": "frames.jpg", "why": e.to_string()}));
        Value::Null
    });
    // motion read from the timeline
    match s.execute("sequence.checkMotion", json!({"sequence": seq.0})) {
        Ok(v) => write(s, &mut files, "motion.json", &pretty(&v))?,
        Err(e) => missing.push(json!({"file": "motion.json", "why": e.to_string()})),
    }
    // the QA report
    let qa = if preflight {
        match s.execute("delivery.preflight", json!({"report": file("qa.json")})) {
            Ok(v) => {
                files.push("qa.json".into());
                json!({"from": "delivery.preflight (run now)", "passed": v["passed"], "summary": v["summary"]})
            }
            Err(e) => {
                missing.push(json!({"file": "qa.json", "why": format!("delivery.preflight: {e}")}));
                Value::Null
            }
        }
    } else {
        match qa_from.or_else(|| newest_qa(s)) {
            None => {
                missing.push(json!({"file": "qa.json", "why": "no QA report: run delivery.preflight {report} first, or give `qa` or `preflight: true`"}));
                Value::Null
            }
            Some(src) => match s
                .services
                .read_file_bounded(&src, MAX_QA_BYTES)
                .map_err(|e| e.to_string())
                .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()))
            {
                Ok(v) => {
                    write(s, &mut files, "qa.json", &pretty(&v))?;
                    let from = std::path::Path::new(&src).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    json!({"from": from, "passed": v["passed"], "summary": v["summary"]})
                }
                Err(e) => {
                    missing.push(json!({"file": "qa.json", "why": format!("the QA report could not be read: {e}")}));
                    Value::Null
                }
            },
        }
    };
    // the decisions
    match crate::copy_lint::beside_project(s, "decisions.json") {
        Some(src) if s.services.file_size(&src).is_ok() => {
            match s
                .services
                .read_file_bounded(&src, MAX_DECISIONS_BYTES)
                .map_err(|e| e.to_string())
                .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()))
            {
                Ok(v) => write(s, &mut files, "decisions.json", &pretty(&v))?,
                Err(e) => missing.push(json!({"file": "decisions.json", "why": format!("build/decisions.json could not be read: {e}")})),
            }
        }
        _ => missing.push(json!({"file": "decisions.json", "why": "no build/decisions.json: nothing was waived"})),
    }
    // the comments
    let open_done = match crate::review::inbox_path(s).map(|path| crate::review::read(&*s.services, &path)) {
        Some(Ok(inbox)) => {
            let open: Vec<_> = inbox.comments().filter(|c| c.status == crate::review::Status::Open && c.sequence == seq.0).collect();
            let done: Vec<_> = inbox.comments().filter(|c| c.status == crate::review::Status::Done && c.sequence == seq.0).collect();
            write(s, &mut files, "comments.json", &pretty(&json!({"open": open, "done": done})))?;
            json!({"open": open.len(), "done": done.len()})
        }
        Some(Err(e)) => {
            missing.push(json!({"file": "comments.json", "why": format!("the review inbox could not be read: {e}")}));
            Value::Null
        }
        None => Value::Null,
    };
    // motion filmstrips: a job unless waited for
    let mut job = Value::Null;
    let strips = if want_strips {
        let folder = file("strips");
        match s.execute("sequence.motionStrips", json!({"sequence": seq.0, "out": folder, "wait": wait})) {
            Ok(v) if v["job"].is_u64() && !wait => {
                job = v["job"].clone();
                json!({"folder": "strips", "job": v["job"], "note": "rendering: jobs.list shows it (its sheets and findings under motionStrips)"})
            }
            Ok(v) => {
                json!({"folder": "strips", "sheets": v["sheets"].as_array().map_or(0, Vec::len), "findings": v["findings"].as_array().map_or(0, Vec::len)})
            }
            Err(e) => {
                let why = e.to_string();
                missing.push(json!({"file": "strips/", "why": why}));
                Value::Null
            }
        }
    } else {
        missing.push(json!({"file": "strips/", "why": "strips: false"}));
        Value::Null
    };
    let manifest = json!({
        "format": "autocut.reviewPacket", "version": 1, "created": utc_now(),
        "sequence": seq.0, "run": run, "round": round, "roundRefused": refused,
        "files": files, "missing": missing, "sheet": sheet, "strips": strips, "qa": qa, "comments": open_done,
        "contract": "docs/finish-reviewer.md",
    });
    write(s, &mut files, "manifest.json", &pretty(&manifest))?;
    let mut answer = manifest;
    answer["folder"] = json!(out);
    answer["files"] = json!(files);
    answer["brief"] = json!(file("brief.md"));
    answer["job"] = job;
    answer["next"] = json!(match &answer["roundRefused"] {
        Value::String(why) => why.clone(),
        _ => "give a fresh reviewer this folder (brief.md first) and nothing else; record its answer with review.gate {text, round}".to_string(),
    });
    Ok(answer)
}

#[cfg(test)]
#[path = "review_finish_tests.rs"]
mod tests;
