//! `copy.lint`: the words of a video checked against built-in word lists and device budgets
//! (PX1.3; the idea of a copy lint with budgets after taste-skill and hallmark, every list and
//! number here written fresh for FilmCraft).
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `stockPhrase` | a stock phrase of video copy ("in this video we'll", "let's dive in", "smash that like button") or of marketing copy ("game-changer", "next level") | [`VIDEO_STOCK`], [`GENERAL_STOCK`] |
//! | `filler` | a filler word or a hedge ("basically", "kind of", "I think") | [`FILLER`], [`HEDGES`] |
//! | `describesShot` | narration (caption cues, a script, `text`) says what the picture shows ("here we see", "as you can see") | [`DESCRIBES_SHOT`] |
//! | `overBudget` | a device is used more than its budget per 100 words: exclamation marks, questions, em dashes, emoji, all-caps words; `level: "fail"` from twice the budget plus one | [`BUDGETS`]; a text under 100 words gets the budget of 100 |
//! | `unledgeredFigure` | a figure written in digits is not in the fact ledger (`facts`, or `build/facts.json` beside the saved project) | compared by value |
//!
//! **The person's own phrases win (decision D6).** A phrase in the taste profiles' `phrases`
//! (the project's and the user's default together), a wording rule's `say`, or the `allow`
//! parameter is never flagged: a list match that overlaps one is dropped (`allowed.suppressed`
//! counts them), and devices and figures inside one are not counted.
//!
//! What is read: `text` (checked as narration), `publish` (`title`, `description`, `caption`,
//! `tags`), `script` (a lines file, a showcase plan or plain text, as `script.lintMalayalam`
//! reads it; narration) and `sequence` (every enabled text layer of a graphic clip on an enabled
//! video track, as `sequence.checkCraft` collects titles, and every caption cue of every caption
//! track, hidden ones too since they go out as sidecars; cues are narration). Budgets are counted
//! per unit: the text, the publish title, description and caption each, the script, all titles
//! together and each caption track. Tags get the stock phrase and figure checks only.
//!
//! Approximations: a question counts whether it is rhetorical or not (copy rarely waits for an
//! answer); an all-caps word is one of at least 5 letters (shorter ones are taken as acronyms)
//! or a short shout word ("WOW", "FREE"), and a line with no lower-case letter at all is a title
//! styled in capitals and not counted; emoji are counted by code point ranges (a flag is one, a
//! joined sequence counts its first picture). Figures are digits only (not number words), and
//! these are not figures: clock times and chapter stamps ("0:00"), codes and formats where
//! letters touch the digits ("i3", "4K", "1080p", "3D", but "10km" and "2x" are figures), hashtags,
//! web addresses, and a number after "part", "episode", "step", "day" and the like.
//!
//! Every finding is a warning (decision D7), the `fail` level of a budget too: `copy.lint` never
//! blocks. `publish.preflight` adds it as a `copy` check (warn or pass) and `publish.fixCopy`
//! makes the safe mechanical fixes ([`fix_text`]).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use filmcraft_project::{ItemId, Project, Sequence};
use filmcraft_time::Tick;
use serde_json::{Map, Value, json};

use crate::commands::{CommandSpec, always, bad};
use crate::motion_energy::{bounded, r3};
use crate::{Result, Session};

pub const CMD: &str = "copy.lint";

/// Stock phrases of video copy: (pattern, trimmable). A pattern is words separated by spaces,
/// `a|b` being either word; a trimmable one is a greeting or a call to action, which
/// `publish.fixCopy` may cut when a whole sentence at the start or end of a description is made
/// of them.
pub const VIDEO_STOCK: &[(&str, bool)] = &[
    ("in this video we|we'll|we're|we've|i|i'll|i'm|i've|you|you'll|you're", false),
    ("in today's|todays video", false),
    ("in this episode we|we'll|i|i'll|you|you'll", false),
    ("welcome back to my|the|our channel", true),
    ("welcome to my|the|our channel", true),
    ("welcome back everyone|guys|folks|friends", true),
    ("hey|hi|hello guys|everyone|folks", true),
    ("what's|whats up guys|everyone|folks", true),
    ("let's|lets dive in", true),
    ("let's|lets dive right|straight in", true),
    ("let's|lets dive into", false),
    ("let's|lets jump in", true),
    ("let's|lets jump right|straight in", true),
    ("let's|lets get started", true),
    ("let's|lets get into it", true),
    ("let's|lets get right into it", true),
    ("without further ado", true),
    ("don't|dont forget to subscribe|like|comment|share", true),
    ("do not forget to subscribe|like|comment|share", true),
    ("make sure to|you subscribe|like", true),
    ("like and subscribe", true),
    ("like comment and subscribe", true),
    ("like share and subscribe", true),
    ("smash|hit|tap that|the like button", true),
    ("smash|hit|tap that|the subscribe button", true),
    ("hit|ring|smash|tap that|the bell", true),
    ("hit|ring|smash|tap that|the notification bell", true),
    ("turn on notifications", true),
    ("subscribe for more", true),
    ("leave a comment below", true),
    ("leave a comment down below", true),
    ("drop a comment below", true),
    ("let me|us know in the comments", true),
    ("see you in the next one|video|episode", true),
    ("stay tuned", true),
    ("watch until|till|to the end", true),
    ("before we get started|begin", false),
    ("you won't|wont believe", false),
    ("you need to see this", false),
];

/// Stock phrases of marketing copy (hyphens split words: "game-changer" is "game changer").
pub const GENERAL_STOCK: &[&str] = &[
    "game changer",
    "game changing",
    "next level",
    "mind blowing",
    "mind blown",
    "jaw dropping",
    "a must see",
    "must watch",
    "unlock the secret|secrets|power|potential",
    "unleash the|your",
    "elevate your",
    "delve into",
    "deep dive",
    "fast paced world",
    "at the end of the day",
    "it's|its worth noting",
    "a testament to",
    "the ultimate guide",
    "everything you need to know",
    "look no further",
    "buckle up",
    "ahead of the curve",
    "like never before",
    "cutting edge",
    "state of the art",
    "embark on",
    "in the realm of",
];

/// Filler words: they add length, not meaning.
pub const FILLER: &[&str] = &[
    "basically",
    "literally",
    "actually",
    "honestly",
    "kind of",
    "sort of",
    "pretty much",
    "a little bit",
    "i mean",
    "so yeah",
    "needless to say",
    "at this point in time",
    "the fact that",
];

/// Hedges: they weaken a line the film can stand behind.
pub const HEDGES: &[&str] = &[
    "i think",
    "i guess",
    "i feel like",
    "i suppose",
    "i believe",
    "perhaps",
    "maybe",
    "probably",
    "arguably",
    "somewhat",
    "in my opinion",
    "it seems",
    "to be honest",
    "more or less",
];

/// Narration that says what the picture already shows.
pub const DESCRIBES_SHOT: &[&str] = &[
    "here we see",
    "here we can see",
    "here you see",
    "here you can see",
    "as you can see",
    "as we can see",
    "as you see here",
    "you can see here",
    "we can see here",
    "this is a shot of",
    "here's|here is a shot of",
    "this shot shows",
    "in this shot",
    "this clip shows",
    "in this clip",
    "this footage shows",
    "in this footage",
    "you're|you are looking at",
    "on screen you can see",
    "on the screen you can see",
    "pictured here",
    "shown here",
    "as seen here",
];

/// Words before which "kind of" and "sort of" name a kind, not a hedge ("a kind of printer").
const NOUN_BEFORE: &[&str] = &[
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "some",
    "any",
    "one",
    "every",
    "each",
    "another",
    "no",
    "same",
    "of",
    "new",
    "first",
    "only",
    "right",
    "wrong",
    "different",
    "other",
    "all",
    "its",
    "our",
    "my",
    "your",
    "their",
    "his",
    "her",
];

/// Words a trimmable stock sentence may hold besides its stock phrases ("So, hey everyone!").
const CONNECTORS: &[&str] = &["and", "or", "so", "please", "guys", "everyone", "folks", "also", "then", "now", "too", "okay", "ok", "alright", "well", "oh"];

/// Short all-caps words that shout rather than abbreviate.
const SHOUT: &[&str] = &["wow", "omg", "free", "new", "best", "epic", "huge", "must", "now", "ever", "lol", "yes", "sale"];

/// Letters after digits that make a quantity ("10km", "2x", "4th"); other letters make a code or
/// a format ("4K", "1080p", "3D").
const UNITS: &[&str] = &[
    "km", "m", "cm", "mm", "g", "kg", "mg", "l", "ml", "h", "hr", "hrs", "min", "mins", "s", "sec", "secs", "ms", "mph", "kph", "kmh", "w", "kw", "kwh", "v",
    "mah", "wh", "gb", "tb", "mb", "kb", "x", "c", "f", "in", "ft", "lb", "lbs", "oz", "st", "nd", "rd", "th", "yr", "yrs",
];

/// Words after which a number names a part, not a figure ("part 2", "day 3").
const NUMBERING: &[&str] =
    &["part", "episode", "ep", "chapter", "step", "season", "volume", "vol", "no", "number", "version", "day", "week", "round", "level", "page", "series"];

/// A device and its budget per 100 words.
pub const BUDGETS: &[(&str, f64)] = &[("exclamation", 1.0), ("question", 1.0), ("emDash", 2.0), ("emoji", 2.0), ("allCaps", 1.0)];
/// A text shorter than this gets the budget of this many words.
pub const BUDGET_WORDS: usize = 100;
/// The shortest all-caps word counted (shorter ones are taken as acronyms unless in [`SHOUT`]).
pub const CAPS_LETTERS: usize = 5;

/// Longest text read in one parameter or field, bytes.
const MAX_TEXT_BYTES: usize = 1 << 20;
/// Most bytes linted in one call (past them, segments are left out and `skipped` says so).
const MAX_TOTAL_BYTES: usize = 8 << 20;
/// Most segments (titles, cues, script parts, tags) read.
const MAX_SEGMENTS: usize = 40_000;
const MAX_TAGS: usize = 500;
const MAX_TAG_CHARS: usize = 500;
/// Most ledger entries and allowed phrases, and the longest of each.
const MAX_FACTS: usize = 10_000;
const MAX_FACT_BYTES: usize = 4096;
const MAX_ALLOW: usize = 1000;
const MAX_ALLOW_CHARS: usize = 300;
/// Largest script and ledger files read.
const MAX_SCRIPT_BYTES: u64 = 16 << 20;
const MAX_LEDGER_BYTES: u64 = 4 << 20;
/// Most rounds of trimming one description gets.
const MAX_TRIMS: usize = 64;
/// Characters of context shown on each side of a match.
const CONTEXT: usize = 40;
/// Most findings of one code listed (`counts` has them all).
const MAX_PER_CODE: usize = 200;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Copy",
        menu: &[],
        shortcut: None,
        params: r#"{"text":str? (checked as narration),"sequence":id? (title text layers and caption cues),"publish":{"title":str?,"description":str?,"caption":str?,"tags":[str]?}?,"script":path? (a lines file, a showcase plan or plain text),"facts":[str|number]? (the fact ledger; default build/facts.json beside the saved project),"allow":[str]? (phrases never flagged, with the taste profiles' own phrases)} — at least one of text, sequence, publish or script"#,
        enabled: always,
        run,
        journal: false,
    }]
}

// -------------------------------------------------------------------------------------------
// what is read
// -------------------------------------------------------------------------------------------

/// What a piece of text is, which decides the checks it gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Publish copy: a title, a description, a caption.
    Copy,
    /// Spoken words: caption cues, a script, `text`.
    Narration,
    /// On-screen title text.
    Title,
    /// One tag.
    Tag,
}

impl Kind {
    fn id(self) -> &'static str {
        match self {
            Kind::Copy => "copy",
            Kind::Narration => "narration",
            Kind::Title => "title",
            Kind::Tag => "tag",
        }
    }
}

/// One piece of text.
#[derive(Clone, Debug)]
pub(crate) struct Segment {
    /// The unit its devices are budgeted in: `text`, `title`, `description`, `caption`, `tags`,
    /// `script`, `titles`, `captions:C1`…
    pub unit: String,
    pub kind: Kind,
    pub text: String,
    /// Where it is, put into each of its findings (`time`, `clip`, `track` for a sequence's text,
    /// `part` for a script's).
    pub at: Map<String, Value>,
}

impl Segment {
    fn new(unit: &str, kind: Kind, text: String) -> Self {
        Segment { unit: unit.to_string(), kind, text, at: Map::new() }
    }
}

/// The publish copy of a video.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PublishCopy {
    pub title: Option<String>,
    pub description: Option<String>,
    pub caption: Option<String>,
    pub tags: Option<Vec<String>>,
}

impl PublishCopy {
    pub(crate) fn segments(&self) -> Vec<Segment> {
        let mut out = Vec::new();
        for (unit, v) in [("title", &self.title), ("description", &self.description), ("caption", &self.caption)] {
            if let Some(t) = v {
                out.push(Segment::new(unit, Kind::Copy, t.clone()));
            }
        }
        for (i, t) in self.tags.iter().flatten().enumerate() {
            let mut seg = Segment::new("tags", Kind::Tag, t.clone());
            seg.at.insert("tag".into(), json!(i));
            out.push(seg);
        }
        out
    }

    pub(crate) fn to_json(&self) -> Value {
        let mut v = Map::new();
        for (k, f) in [("title", &self.title), ("description", &self.description), ("caption", &self.caption)] {
            if let Some(t) = f {
                v.insert(k.into(), json!(t));
            }
        }
        if let Some(t) = &self.tags {
            v.insert("tags".into(), json!(t));
        }
        Value::Object(v)
    }
}

fn text_field(cmd: &str, key: &str, v: &Value) -> Result<String> {
    let t = v.as_str().ok_or_else(|| bad(cmd, format!("`{key}` must be a string")))?;
    if t.len() > MAX_TEXT_BYTES {
        return Err(bad(cmd, format!("`{key}` is {} bytes; at most {MAX_TEXT_BYTES}", t.len())));
    }
    Ok(t.to_string())
}

/// The publish copy in `p` (`title`, `description`, `caption`, `tags`; absent or `null` fields
/// are not there): `None` when it has none of them.
pub(crate) fn publish_copy(cmd: &str, p: &Value) -> Result<Option<PublishCopy>> {
    let field = |key: &str| -> Result<Option<String>> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => text_field(cmd, key, v).map(Some),
        }
    };
    let tags = match p.get("tags") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let a = v.as_array().filter(|a| a.len() <= MAX_TAGS).ok_or_else(|| bad(cmd, format!("`tags` is a list of at most {MAX_TAGS} texts")))?;
            Some(
                a.iter()
                    .map(|t| {
                        t.as_str()
                            .filter(|t| t.chars().count() <= MAX_TAG_CHARS)
                            .map(str::to_string)
                            .ok_or_else(|| bad(cmd, format!("each tag is a text of at most {MAX_TAG_CHARS} characters")))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        }
    };
    let c = PublishCopy { title: field("title")?, description: field("description")?, caption: field("caption")?, tags };
    Ok((c != PublishCopy::default()).then_some(c))
}

/// The texts of sequence `q`: its titles (unit `titles`) and every caption cue (unit
/// `captions:C<n>`, narration), each with where it is.
pub(crate) fn sequence_segments(project: &Project, q: &Sequence) -> Vec<Segment> {
    let mut out = Vec::new();
    let place = |seg: &mut Segment, kind: &str, clip: u64, track: String, start: Tick, end: Tick| {
        for (k, v) in [
            ("kind", json!(kind)),
            ("clip", json!(clip)),
            ("track", json!(track)),
            ("time", json!(start.0)),
            ("seconds", json!(r3(start.seconds()))),
            ("end", json!(end.0)),
            ("until", json!(r3(end.seconds()))),
        ] {
            seg.at.insert(k.into(), v);
        }
    };
    for (ti, it, _, text) in crate::craft_check::title_layers(project, q, MAX_SEGMENTS) {
        let mut seg = Segment::new("titles", Kind::Title, text);
        place(&mut seg, "title", it.id.0, format!("V{}", ti + 1), it.start, it.end());
        out.push(seg);
    }
    for (ci, track) in q.caption_tracks.iter().enumerate() {
        for c in &track.captions {
            if out.len() >= MAX_SEGMENTS {
                return out;
            }
            let mut seg = Segment::new(&format!("captions:C{}", ci + 1), Kind::Narration, c.plain_lines().join(" "));
            place(&mut seg, "caption", c.id.0, format!("C{}", ci + 1), c.start, c.end());
            out.push(seg);
        }
    }
    out
}

/// The parts of the script file at `path` (a lines file, a showcase plan or plain text).
fn script_segments(s: &Session, cmd: &str, path: &str) -> Result<Vec<Segment>> {
    let path = crate::export_tools::expand_home(path);
    let bytes = s.services.read_file_bounded(&path, MAX_SCRIPT_BYTES).map_err(|e| bad(cmd, format!("script {path}: {e}")))?;
    let v = match serde_json::from_slice::<Value>(&bytes) {
        Ok(v @ Value::Object(_)) => v,
        // a plain text file (or a bare JSON string) is a script too
        Ok(Value::String(t)) => json!({"text": t}),
        Ok(Value::Array(_)) => return Err(bad(cmd, format!("script {path} is a JSON list: give a lines file, a plan with `script`, or plain text"))),
        _ => json!({"text": String::from_utf8_lossy(&bytes)}),
    };
    let parts = crate::ml_script::malayalam_parts(&v).map_err(|_| {
        bad(cmd, format!("script {path} is not a script: give a lines file ({{\"intro\", \"outro\", \"lines\": {{…}}}}), a plan with `script`, or plain text"))
    })?;
    Ok(parts
        .into_iter()
        .take(MAX_SEGMENTS)
        .map(|p| {
            let mut seg = Segment::new("script", Kind::Narration, p.text);
            seg.at.insert("part".into(), json!(p.key));
            seg
        })
        .collect())
}

// -------------------------------------------------------------------------------------------
// the person's phrases and the fact ledger
// -------------------------------------------------------------------------------------------

/// The phrases never flagged, as words (normalized), and where they came from.
#[derive(Clone, Debug, Default)]
pub(crate) struct Allowed {
    pub phrases: Vec<Vec<String>>,
    pub from_taste: usize,
    pub from_param: usize,
}

/// The phrases `copy.lint` never flags: the taste profiles' own phrases and wording rules' `say`
/// (the project's and the user's default together) and `p.allow`.
pub(crate) fn allowed(s: &Session, cmd: &str, p: &Value) -> Result<Allowed> {
    let mut taste: Vec<String> = Vec::new();
    for t in [s.project.taste.clone().unwrap_or_default().sanitized(), crate::taste::default_taste(s)] {
        taste.extend(t.phrases.into_iter().flatten());
        taste.extend(t.words.into_iter().flatten().map(|w| w.say));
    }
    let param: Vec<String> = match p.get("allow") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => v
            .as_array()
            .filter(|a| a.len() <= MAX_ALLOW)
            .ok_or_else(|| bad(cmd, format!("`allow` is a list of at most {MAX_ALLOW} phrases")))?
            .iter()
            .map(|x| {
                x.as_str()
                    .filter(|t| t.chars().count() <= MAX_ALLOW_CHARS)
                    .map(str::to_string)
                    .ok_or_else(|| bad(cmd, format!("each `allow` phrase is a text of at most {MAX_ALLOW_CHARS} characters")))
            })
            .collect::<Result<_>>()?,
    };
    let words = |list: &[String]| -> Vec<Vec<String>> {
        list.iter().map(|t| tokens(t, &[]).into_iter().map(|t| t.norm).collect::<Vec<_>>()).filter(|w| !w.is_empty()).collect()
    };
    let (a, b) = (words(&taste), words(&param));
    let (from_taste, from_param) = (a.len(), b.len());
    let mut phrases = a;
    phrases.extend(b);
    phrases.sort();
    phrases.dedup();
    Ok(Allowed { phrases, from_taste, from_param })
}

/// The fact ledger: the figures the film may state.
#[derive(Clone, Debug)]
pub(crate) struct Ledger {
    /// `facts` (the parameter) or the file's path.
    pub from: String,
    pub entries: usize,
    pub values: Vec<f64>,
}

impl Ledger {
    fn holds(&self, x: f64) -> bool {
        self.values.iter().any(|v| (v - x).abs() <= 1e-9 * x.abs().max(1.0))
    }
}

/// Where the ledger came from, or why there is none.
#[derive(Clone, Debug)]
pub(crate) enum LedgerState {
    Found(Ledger),
    /// No ledger: why.
    None(String),
    /// A ledger file that could not be used: why.
    Broken(String),
}

/// The ledger of `entries` (texts, numbers or `{text}`).
fn ledger_of(from: String, entries: &[Value]) -> std::result::Result<Ledger, String> {
    if entries.len() > MAX_FACTS {
        return Err(format!("{} ledger entries; at most {MAX_FACTS}", entries.len()));
    }
    let mut values = Vec::new();
    for e in entries {
        match e {
            Value::Number(n) => values.extend(n.as_f64().filter(|v| v.is_finite())),
            Value::String(t) if t.len() <= MAX_FACT_BYTES => values.extend(figures(t, &[]).into_iter().map(|f| f.value)),
            Value::Object(o) => match o.get("text") {
                Some(Value::String(t)) if t.len() <= MAX_FACT_BYTES => values.extend(figures(t, &[]).into_iter().map(|f| f.value)),
                _ => return Err("a ledger entry object needs a `text` of at most 4096 bytes".into()),
            },
            _ => return Err(format!("a ledger entry is a text of at most {MAX_FACT_BYTES} bytes, a number or {{\"text\"}}")),
        }
    }
    Ok(Ledger { from, entries: entries.len(), values })
}

/// The project's ledger file: `build/facts.json` beside the saved project.
pub(crate) fn beside_project(s: &Session, name: &str) -> Option<String> {
    let project = s.path.as_deref().filter(|p| !p.is_empty())?;
    let dir = std::path::Path::new(project).parent()?;
    Some(dir.join("build").join(name).display().to_string())
}

/// The ledger: `p.facts`, else `build/facts.json` beside the saved project.
pub(crate) fn ledger(s: &Session, cmd: &str, p: &Value) -> Result<LedgerState> {
    if let Some(v) = p.get("facts").filter(|v| !v.is_null()) {
        let a = v.as_array().ok_or_else(|| bad(cmd, "`facts` is the fact ledger: a list of texts or numbers"))?;
        return ledger_of("facts".into(), a).map(LedgerState::Found).map_err(|e| bad(cmd, format!("facts: {e}")));
    }
    let Some(path) = beside_project(s, "facts.json") else {
        return Ok(LedgerState::None("no fact ledger: give `facts`, or keep build/facts.json beside the saved project".into()));
    };
    if s.services.file_size(&path).is_err() {
        return Ok(LedgerState::None(format!("no fact ledger: {path} is not there and `facts` was not given")));
    }
    let bytes = match s.services.read_file_bounded(&path, MAX_LEDGER_BYTES) {
        Ok(b) => b,
        Err(e) => return Ok(LedgerState::Broken(format!("{path}: {e}"))),
    };
    let entries = match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Array(a)) => a,
        Ok(Value::Object(mut o)) => match o.remove("facts") {
            Some(Value::Array(a)) => a,
            _ => return Ok(LedgerState::Broken(format!("{path}: the ledger is a list, or an object with a `facts` list"))),
        },
        Ok(_) => return Ok(LedgerState::Broken(format!("{path}: the ledger is a list, or an object with a `facts` list"))),
        Err(e) => return Ok(LedgerState::Broken(format!("{path}: {e}"))),
    };
    Ok(match ledger_of(path.clone(), &entries) {
        Ok(l) => LedgerState::Found(l),
        Err(e) => LedgerState::Broken(format!("{path}: {e}")),
    })
}

// -------------------------------------------------------------------------------------------
// reading text
// -------------------------------------------------------------------------------------------

/// A word: its bytes in the text and its normalized form (lower case, straight apostrophe).
#[derive(Clone, Debug)]
pub(crate) struct Tok {
    pub start: usize,
    pub end: usize,
    pub norm: String,
    /// A hashtag or a mention.
    pub tag: bool,
}

fn apostrophe(c: char) -> bool {
    matches!(c, '\'' | '\u{2019}' | '\u{02BC}')
}

fn inside(spans: &[(usize, usize)], at: usize) -> bool {
    spans.iter().any(|(a, b)| *a <= at && at < *b)
}

fn overlaps(spans: &[(usize, usize)], a: usize, b: usize) -> bool {
    spans.iter().any(|(x, y)| *x < b && a < *y)
}

/// The words of `text` outside `skip` (byte ranges): runs of letters and digits, an apostrophe
/// between two of them kept ("let's").
pub(crate) fn tokens(text: &str, skip: &[(usize, usize)]) -> Vec<Tok> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut k = 0;
    while let Some(&(start, c)) = chars.get(k) {
        if !c.is_alphanumeric() {
            k += 1;
            continue;
        }
        let mut j = k;
        let mut norm = String::new();
        while let Some(&(_, d)) = chars.get(j) {
            let joins = apostrophe(d) && chars.get(j + 1).is_some_and(|(_, n)| n.is_alphanumeric());
            if d.is_alphanumeric() {
                norm.extend(d.to_lowercase());
            } else if joins {
                norm.push('\'');
            } else {
                break;
            }
            j += 1;
        }
        let end = chars.get(j).map_or(text.len(), |(b, _)| *b);
        let tag = k > 0 && chars.get(k - 1).is_some_and(|(_, p)| matches!(p, '#' | '@'));
        if !inside(skip, start) {
            out.push(Tok { start, end, norm, tag });
        }
        k = j.max(k + 1);
    }
    out
}

/// Web addresses in `text`, as byte ranges.
fn urls(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    let chunk = |a: usize, b: usize, out: &mut Vec<(usize, usize)>| {
        let w = text.get(a..b).unwrap_or("").to_ascii_lowercase();
        if w.starts_with("http://") || w.starts_with("https://") || w.starts_with("www.") || w.contains("://") {
            out.push((a, b));
        }
    };
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(a)) => {
                chunk(a, i, &mut out);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(a) = start {
        chunk(a, text.len(), &mut out);
    }
    out
}

/// Whether the text between two words ends a sentence or a clause.
fn breaks(text: &str, a: usize, b: usize) -> bool {
    text.get(a..b.max(a)).is_some_and(|g| g.chars().any(|c| matches!(c, '.' | '!' | '?' | ';' | ':' | '\n' | '\u{2026}' | '|' | '\u{2022}')))
}

/// A figure written in digits.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Figure {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub value: f64,
}

/// The figures of `text` outside `skip` (see the module docs for what is not a figure).
pub(crate) fn figures(text: &str, skip: &[(usize, usize)]) -> Vec<Figure> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let at = |k: usize| chars.get(k).map(|(_, c)| *c);
    let mut out = Vec::new();
    let mut k = 0;
    while let Some(&(start, c)) = chars.get(k) {
        if !c.is_ascii_digit() {
            k += 1;
            continue;
        }
        let prev = k.checked_sub(1).and_then(at);
        let mut j = k;
        let mut dot = false;
        let mut clock = false;
        while let Some(d) = at(j) {
            let digit_next = at(j + 1).is_some_and(|n| n.is_ascii_digit());
            if d.is_ascii_digit() || (d == ',' && digit_next) || (d == '.' && !dot && digit_next) {
                dot |= d == '.';
                j += 1;
            } else if d == ':' && digit_next {
                clock = true;
                j += 1;
            } else {
                break;
            }
        }
        let run_end = chars.get(j).map_or(text.len(), |(b, _)| *b);
        let mut end = run_end;
        // letters right after the digits: a unit makes a quantity, others a code
        let mut m = j;
        let mut suffix = String::new();
        while let Some(l) = at(m).filter(|l| l.is_alphabetic()) {
            suffix.extend(l.to_lowercase());
            m += 1;
        }
        let prev2 = k.checked_sub(2).and_then(at);
        let code = prev.is_some_and(|p| p.is_alphabetic() || matches!(p, '#' | '@' | '_' | '/'))
            || (prev == Some('-') && prev2.is_some_and(char::is_alphabetic))
            || (!suffix.is_empty() && !UNITS.contains(&suffix.as_str()));
        if at(j) == Some('%') {
            end = chars.get(j + 1).map_or(text.len(), |(b, _)| *b);
        }
        // the word before: "part 2", "day 3"
        let before: String = {
            let head = text.get(..start).unwrap_or("");
            let trimmed = head.trim_end_matches(|c: char| c.is_whitespace() || c == '#' || c == '.');
            trimmed.chars().rev().take_while(|c| c.is_alphabetic()).collect::<Vec<_>>().into_iter().rev().flat_map(char::to_lowercase).collect()
        };
        let numbering = NUMBERING.contains(&before.as_str());
        if !clock && !code && !numbering && !inside(skip, start) {
            let digits: String = text.get(start..run_end).unwrap_or("").chars().filter(|c| *c != ',').collect();
            if let Ok(v) = digits.parse::<f64>()
                && v.is_finite()
            {
                out.push(Figure { start, end, text: text.get(start..end).unwrap_or("").to_string(), value: v });
            }
        }
        k = m.max(j).max(k + 1);
    }
    out
}

fn is_emoji(c: char) -> bool {
    let u = c as u32;
    !(0x1F3FB..=0x1F3FF).contains(&u)
        && matches!(u, 0x1F300..=0x1F5FF | 0x1F600..=0x1F64F | 0x1F680..=0x1F6FF | 0x1F900..=0x1F9FF | 0x1FA70..=0x1FAFF | 0x2600..=0x26FF | 0x2700..=0x27BF)
}

/// How often `text` uses each device, outside `skip` (web addresses and the person's phrases).
fn devices(text: &str, toks: &[Tok], skip: &[(usize, usize)]) -> [usize; 5] {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let (mut excl, mut quest, mut dash, mut emoji, mut flags) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut k = 0;
    while let Some(&(b, c)) = chars.get(k) {
        let prev = k.checked_sub(1).and_then(|p| chars.get(p)).map(|(_, c)| *c);
        let next = chars.get(k + 1).map(|(_, c)| *c);
        if inside(skip, b) {
            k += 1;
            continue;
        }
        match c {
            '!' | '\u{FF01}' => excl += 1,
            '\u{203C}' => excl += 2,
            '\u{2049}' => {
                excl += 1;
                quest += 1;
            }
            '?' | '\u{FF1F}' if !matches!(prev, Some('?' | '\u{FF1F}')) => quest += 1,
            '\u{2014}' | '\u{2015}' => dash += 1,
            '\u{2013}' if prev.is_some_and(char::is_whitespace) || next.is_some_and(char::is_whitespace) => dash += 1,
            '-' if next == Some('-') && prev != Some('-') => dash += 1,
            c if (0x1F1E6..=0x1F1FF).contains(&(c as u32)) => flags += 1,
            c if is_emoji(c) && prev != Some('\u{200D}') => emoji += 1,
            _ => {}
        }
        k += 1;
    }
    emoji += flags.div_ceil(2);
    let styled_caps = !text.chars().any(char::is_lowercase);
    let caps = if styled_caps {
        0
    } else {
        toks.iter()
            .filter(|t| !t.tag && !inside(skip, t.start))
            .filter(|t| {
                let w = text.get(t.start..t.end).unwrap_or("");
                let letters = w.chars().filter(|c| c.is_alphabetic()).count();
                letters > 0
                    && !w.chars().any(|c| c.is_ascii_digit())
                    && w.chars().filter(|c| c.is_alphabetic()).all(char::is_uppercase)
                    && (letters >= CAPS_LETTERS || (letters >= 2 && SHOUT.contains(&t.norm.as_str())))
            })
            .count()
    };
    [excl, quest, dash, emoji, caps]
}

// -------------------------------------------------------------------------------------------
// the lists
// -------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum List {
    Stock,
    Filler,
    Shot,
}

struct Pattern {
    list: List,
    /// `video`, `general`, `filler`, `hedge` or `shot`.
    category: &'static str,
    words: Vec<Vec<&'static str>>,
    trim: bool,
    /// "kind of" and "sort of": not after a word that makes them a noun.
    guard: bool,
}

struct Lists {
    patterns: Vec<Pattern>,
    /// Each first word's patterns.
    first: HashMap<&'static str, Vec<usize>>,
    /// The most words a pattern has.
    longest: usize,
}

fn lists() -> &'static Lists {
    static L: OnceLock<Lists> = OnceLock::new();
    L.get_or_init(|| {
        let mut patterns = Vec::new();
        let mut add = |list: List, category: &'static str, text: &'static str, trim: bool| {
            let words: Vec<Vec<&'static str>> = text.split(' ').filter(|w| !w.is_empty()).map(|w| w.split('|').collect()).collect();
            let guard = matches!(words.first().and_then(|w| w.first()), Some(&("kind" | "sort")));
            patterns.push(Pattern { list, category, words, trim, guard });
        };
        for (t, trim) in VIDEO_STOCK {
            add(List::Stock, "video", t, *trim);
        }
        for t in GENERAL_STOCK {
            add(List::Stock, "general", t, false);
        }
        for t in FILLER {
            add(List::Filler, "filler", t, false);
        }
        for t in HEDGES {
            add(List::Filler, "hedge", t, false);
        }
        for t in DESCRIBES_SHOT {
            add(List::Shot, "shot", t, false);
        }
        let mut first: HashMap<&'static str, Vec<usize>> = HashMap::new();
        for (i, p) in patterns.iter().enumerate() {
            for w in p.words.first().into_iter().flatten() {
                first.entry(*w).or_default().push(i);
            }
        }
        let longest = patterns.iter().map(|p| p.words.len()).max().unwrap_or(1);
        Lists { patterns, first, longest }
    })
}

/// The sizes of the lists, as the result reports them.
pub(crate) fn list_sizes() -> Value {
    json!({
        "videoStock": VIDEO_STOCK.len(), "generalStock": GENERAL_STOCK.len(), "filler": FILLER.len(), "hedges": HEDGES.len(),
        "describesShot": DESCRIBES_SHOT.len(), "trimmable": VIDEO_STOCK.iter().filter(|(_, t)| *t).count(),
    })
}

/// A list match: the pattern and the words it spans (`[first, last)`).
#[derive(Clone, Copy, Debug)]
struct Hit {
    pattern: usize,
    first: usize,
    last: usize,
}

/// The matches in `toks` of the patterns `want` takes; one inside a longer match of its own
/// list is left out.
fn hits(text: &str, toks: &[Tok], want: impl Fn(&Pattern) -> bool) -> Vec<Hit> {
    let l = lists();
    let mut out: Vec<Hit> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        let Some(cands) = l.first.get(t.norm.as_str()) else { continue };
        for &pi in cands {
            let Some(p) = l.patterns.get(pi) else { continue };
            if !want(p) {
                continue;
            }
            if p.guard
                && let Some(before) = i.checked_sub(1).and_then(|b| toks.get(b))
                && NOUN_BEFORE.contains(&before.norm.as_str())
                && !breaks(text, before.end, t.start)
            {
                continue;
            }
            let n = p.words.len();
            let ok = n > 0
                && (0..n).all(|k| {
                    let (Some(w), Some(alts)) = (toks.get(i + k), p.words.get(k)) else { return false };
                    alts.contains(&w.norm.as_str()) && (k == 0 || toks.get(i + k - 1).is_some_and(|prev| !breaks(text, prev.end, w.start)))
                });
            if ok {
                out.push(Hit { pattern: pi, first: i, last: i + n });
            }
        }
    }
    // `out` is in order of first word, and a pattern spans at most `longest` words: only the
    // neighbours can hold a match
    let list_of = |h: &Hit| l.patterns.get(h.pattern).map(|p| p.list);
    let holds = |o: &Hit, h: &Hit| list_of(o) == list_of(h) && o.first <= h.first && h.last <= o.last && (o.last - o.first) > (h.last - h.first);
    let contained = |idx: usize, h: &Hit| {
        let mut j = idx;
        while j > 0 {
            j -= 1;
            let Some(o) = out.get(j) else { break };
            if o.first.saturating_add(l.longest) < h.first {
                break;
            }
            if holds(o, h) {
                return true;
            }
        }
        let mut j = idx + 1;
        while let Some(o) = out.get(j) {
            if o.first > h.first {
                break;
            }
            if holds(o, h) {
                return true;
            }
            j += 1;
        }
        false
    };
    out.iter().enumerate().filter(|(idx, h)| !contained(*idx, h)).map(|(_, h)| *h).collect()
}

/// The byte ranges of `allowed`'s phrases in `toks`.
fn allowed_spans(text: &str, toks: &[Tok], allowed: &[Vec<String>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if allowed.is_empty() || toks.is_empty() {
        return out;
    }
    let mut at: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, t) in toks.iter().enumerate() {
        at.entry(t.norm.as_str()).or_default().push(i);
    }
    for phrase in allowed {
        let n = phrase.len();
        let Some(starts) = phrase.first().and_then(|w| at.get(w.as_str())) else { continue };
        for &i in starts {
            let ok = (0..n).all(|k| {
                let (Some(w), Some(want)) = (toks.get(i + k), phrase.get(k)) else { return false };
                w.norm == *want && (k == 0 || toks.get(i + k - 1).is_some_and(|prev| !breaks(text, prev.end, w.start)))
            });
            if ok && let (Some(a), Some(b)) = (toks.get(i), toks.get(i + n - 1)) {
                // the marks the person ends it with are theirs too ("WOW!!")
                let tail: usize =
                    text.get(b.end..).unwrap_or("").chars().take_while(|c| matches!(c, '!' | '?' | '\u{203C}' | '\u{2049}')).map(char::len_utf8).sum();
                out.push((a.start, b.end + tail));
            }
        }
    }
    out
}

/// Up to [`CONTEXT`] characters on each side of `text[a..b]`.
fn context(text: &str, a: usize, b: usize) -> String {
    let before: String = text.get(..a).unwrap_or("").chars().rev().take(CONTEXT).collect::<Vec<_>>().into_iter().rev().collect();
    let after: String = text.get(b..).unwrap_or("").chars().take(CONTEXT).collect();
    format!("{before}{}{after}", text.get(a..b).unwrap_or("")).replace('\n', " ").trim().to_string()
}

// -------------------------------------------------------------------------------------------
// the lint
// -------------------------------------------------------------------------------------------

fn finding(seg: &Segment, code: &str, message: String, extra: Value) -> Value {
    let mut o = Map::new();
    o.insert("code".into(), json!(code));
    o.insert("severity".into(), json!("warn"));
    o.insert("source".into(), json!(seg.unit));
    o.insert("message".into(), json!(message));
    o.extend(seg.at.clone());
    if let Value::Object(extra) = extra {
        o.extend(extra);
    }
    Value::Object(o)
}

/// What one unit holds, for its budgets.
#[derive(Default)]
struct Unit {
    kind: Option<Kind>,
    segments: usize,
    words: usize,
    devices: [usize; 5],
}

/// The lint of `segments`: findings, budgets and what was read.
pub(crate) fn report(segments: &[Segment], allow: &Allowed, ledger: &LedgerState) -> Value {
    let l = lists();
    let mut findings = Vec::new();
    let mut skipped = Vec::new();
    let mut units: BTreeMap<String, Unit> = BTreeMap::new();
    let mut suppressed = 0usize;
    // every finding is counted; at most MAX_PER_CODE of each code are kept (huge inputs would
    // otherwise build huge lists)
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    let mut keep = |f: Value, findings: &mut Vec<Value>| {
        let n = tally.entry(f["code"].as_str().unwrap_or("").to_string()).or_default();
        *n += 1;
        if *n <= MAX_PER_CODE {
            findings.push(f);
        }
    };
    let mut total = 0usize;
    let mut read = 0usize;
    for seg in segments {
        total = total.saturating_add(seg.text.len());
        if total > MAX_TOTAL_BYTES {
            skipped.push(json!({"reason": format!("{} of {} texts read: at most {MAX_TOTAL_BYTES} bytes are linted in one call", read, segments.len())}));
            break;
        }
        read += 1;
        let text = seg.text.as_str();
        let web = urls(text);
        let toks = tokens(text, &web);
        let mine = allowed_spans(text, &toks, &allow.phrases);
        let kind = seg.kind;
        let want = |p: &Pattern| match p.list {
            List::Stock => true,
            List::Filler => kind != Kind::Tag,
            List::Shot => kind == Kind::Narration,
        };
        for h in hits(text, &toks, want) {
            let (Some(p), Some(a), Some(b)) = (l.patterns.get(h.pattern), toks.get(h.first), toks.get(h.last.saturating_sub(1))) else { continue };
            if overlaps(&mine, a.start, b.end) {
                suppressed += 1;
                continue;
            }
            let phrase: String = toks.get(h.first..h.last).unwrap_or(&[]).iter().map(|t| t.norm.as_str()).collect::<Vec<_>>().join(" ");
            let matched = text.get(a.start..b.end).unwrap_or("").to_string();
            let (code, message) = match p.list {
                List::Stock => ("stockPhrase", format!("stock phrase \"{matched}\": say the specific thing instead")),
                List::Filler if p.category == "hedge" => ("filler", format!("hedge \"{matched}\" weakens the line; say it plainly or cut it")),
                List::Filler => ("filler", format!("filler \"{matched}\" adds length, not meaning")),
                List::Shot => ("describesShot", format!("the narration describes the shot (\"{matched}\"): say what it means, the picture shows the rest")),
            };
            keep(
                finding(
                    seg,
                    code,
                    message,
                    json!({"category": p.category, "phrase": phrase, "match": matched, "at": a.start, "context": context(text, a.start, b.end), "trimmable": p.trim}),
                ),
                &mut findings,
            );
        }
        if let LedgerState::Found(lg) = ledger {
            for f in figures(text, &web) {
                if overlaps(&mine, f.start, f.end) {
                    suppressed += 1;
                    continue;
                }
                if !lg.holds(f.value) {
                    keep(
                        finding(
                            seg,
                            "unledgeredFigure",
                            format!("the figure {} is not in the fact ledger: add it with its source, or take it out", f.text),
                            json!({"figure": f.text, "value": f.value, "at": f.start, "context": context(text, f.start, f.end)}),
                        ),
                        &mut findings,
                    );
                }
            }
        }
        let u = units.entry(seg.unit.clone()).or_default();
        u.kind = Some(kind);
        u.segments += 1;
        if kind != Kind::Tag {
            u.words = u.words.saturating_add(toks.len());
            let mut skip = web.clone();
            skip.extend(mine.iter().copied());
            for (n, c) in devices(text, &toks, &skip).iter().enumerate() {
                if let Some(d) = u.devices.get_mut(n) {
                    *d = d.saturating_add(*c);
                }
            }
        }
    }
    let mut budgets = Vec::new();
    for (name, u) in &units {
        if u.kind == Some(Kind::Tag) {
            continue;
        }
        let scaled = u.words.max(BUDGET_WORDS) as f64 / 100.0;
        for (n, (device, per100)) in BUDGETS.iter().enumerate() {
            let count = u.devices.get(n).copied().unwrap_or(0);
            let budget = per100 * scaled;
            let fail_at = 2.0 * budget + 1.0;
            let level = if count as f64 >= fail_at {
                "fail"
            } else if count as f64 > budget {
                "warn"
            } else {
                "ok"
            };
            budgets
                .push(json!({"source": name, "device": device, "count": count, "words": u.words, "budget": r3(budget), "failAt": r3(fail_at), "level": level}));
            if level != "ok" {
                let seg = Segment::new(name, u.kind.unwrap_or(Kind::Copy), String::new());
                keep(
                    finding(
                        &seg,
                        "overBudget",
                        format!(
                            "{count} {} in {} words of {name}: the budget is {:.1} ({per100} per 100 words){}",
                            device_name(device, count),
                            u.words,
                            budget,
                            if level == "fail" {
                                format!(", and {:.1} or more is the fail level (still a warning this round, D7)", fail_at)
                            } else {
                                String::new()
                            }
                        ),
                        json!({"device": device, "count": count, "words": u.words, "budget": r3(budget), "failAt": r3(fail_at), "per100": per100, "level": level}),
                    ),
                    &mut findings,
                );
            }
        }
    }
    let (findings, _) = bounded(findings);
    let counts: Map<String, Value> = tally.into_iter().map(|(k, n)| (k, json!(n))).collect();
    let ledger_json = match ledger {
        LedgerState::Found(lg) => json!({"from": lg.from, "entries": lg.entries, "figures": lg.values.len()}),
        LedgerState::None(why) => json!({"from": null, "why": why}),
        LedgerState::Broken(why) => json!({"from": null, "error": why}),
    };
    match ledger {
        LedgerState::None(why) => skipped.push(json!({"check": "unledgeredFigure", "reason": why})),
        LedgerState::Broken(why) => skipped.push(json!({"check": "unledgeredFigure", "reason": format!("the fact ledger could not be read: {why}")})),
        LedgerState::Found(_) => {}
    }
    let sources: Vec<Value> =
        units.iter().map(|(name, u)| json!({"source": name, "kind": u.kind.map(Kind::id), "texts": u.segments, "words": u.words})).collect();
    json!({
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "budgets": budgets,
        "checked": {"sources": sources, "texts": read, "words": units.values().map(|u| u.words).sum::<usize>()},
        "ledger": ledger_json,
        "allowed": {"phrases": allow.phrases.len(), "fromTaste": allow.from_taste, "fromParam": allow.from_param, "suppressed": suppressed},
        "skipped": skipped,
        "lists": list_sizes(),
        "thresholds": {
            "budgetsPer100Words": BUDGETS.iter().map(|(d, n)| (d.to_string(), json!(n))).collect::<Map<String, Value>>(),
            "budgetWords": BUDGET_WORDS, "failAt": "twice the budget plus one", "capsLetters": CAPS_LETTERS,
        },
    })
}

fn device_name(device: &str, n: usize) -> &'static str {
    match (device, n == 1) {
        ("exclamation", true) => "exclamation mark",
        ("exclamation", false) => "exclamation marks",
        ("question", true) => "question",
        ("question", false) => "questions",
        ("emDash", true) => "em dash",
        ("emDash", false) => "em dashes",
        ("emoji", _) => "emoji",
        (_, true) => "all-caps word",
        (_, false) => "all-caps words",
    }
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let mut segments = Vec::new();
    let mut given = 0;
    if let Some(v) = p.get("text").filter(|v| !v.is_null()) {
        segments.push(Segment::new("text", Kind::Narration, text_field(CMD, "text", v)?));
        given += 1;
    }
    if let Some(v) = p.get("publish").filter(|v| !v.is_null()) {
        if !v.is_object() {
            return Err(bad(CMD, "`publish` is {title?, description?, caption?, tags?}"));
        }
        let c = publish_copy(CMD, v)?.ok_or_else(|| bad(CMD, "`publish` has none of title, description, caption or tags"))?;
        segments.extend(c.segments());
        given += 1;
    }
    if let Some(v) = p.get("script").filter(|v| !v.is_null()) {
        let path = v.as_str().filter(|t| !t.trim().is_empty() && t.len() <= 4096 && !t.contains('\0')).ok_or_else(|| bad(CMD, "`script` is a file path"))?;
        segments.extend(script_segments(s, CMD, path)?);
        given += 1;
    }
    let mut sequence = None;
    if let Some(v) = p.get("sequence").filter(|v| !v.is_null()) {
        let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
        let q = s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
        segments.extend(sequence_segments(&s.project, q));
        sequence = Some(id.0);
        given += 1;
    }
    if given == 0 {
        return Err(bad(CMD, "give at least one of `text`, `sequence`, `publish` or `script`"));
    }
    let allow = allowed(s, CMD, p)?;
    let ledger = ledger(s, CMD, p)?;
    let mut r = report(&segments, &allow, &ledger);
    r["sequence"] = json!(sequence);
    Ok(r)
}

// -------------------------------------------------------------------------------------------
// safe fixes
// -------------------------------------------------------------------------------------------

/// The sentences of `text`, as byte ranges without their surrounding white space: each ends
/// after its run of closing punctuation (followed by white space or the end) or at a new line.
fn sentences(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let closing = |c: char| matches!(c, '.' | '!' | '?' | '\u{2026}' | '\u{203C}' | '\u{2049}' | ')' | '"' | '\'' | '\u{201D}' | '\u{2019}');
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut k = 0;
    let push = |a: usize, b: usize, out: &mut Vec<(usize, usize)>| {
        let span = text.get(a..b).unwrap_or("");
        let end = a + span.trim_end().len();
        if end > a {
            out.push((a, end));
        }
    };
    while let Some(&(b, c)) = chars.get(k) {
        if c == '\n' {
            if let Some(a) = start.take() {
                push(a, b, &mut out);
            }
            k += 1;
            continue;
        }
        if start.is_none() && !c.is_whitespace() {
            start = Some(b);
        }
        if matches!(c, '.' | '!' | '?' | '\u{2026}' | '\u{203C}' | '\u{2049}') {
            let mut m = k + 1;
            while chars.get(m).is_some_and(|(_, d)| closing(*d)) {
                m += 1;
            }
            if chars.get(m).is_none_or(|(_, d)| d.is_whitespace()) {
                let end = chars.get(m).map_or(text.len(), |(e, _)| *e);
                if let Some(a) = start.take() {
                    push(a, end, &mut out);
                }
                k = m;
                continue;
            }
        }
        k += 1;
    }
    if let Some(a) = start {
        push(a, text.len(), &mut out);
    }
    out
}

/// Whether `sentence` is made only of trimmable stock phrases (and joining words), and none of
/// the person's phrases.
fn fluff(sentence: &str, allow: &[Vec<String>]) -> bool {
    let toks = tokens(sentence, &[]);
    if toks.is_empty() || !allowed_spans(sentence, &toks, allow).is_empty() {
        return false;
    }
    let l = lists();
    let found = hits(sentence, &toks, |p| p.list == List::Stock && p.trim);
    if found.is_empty() {
        return false;
    }
    let mut covered = vec![false; toks.len()];
    for h in &found {
        if l.patterns.get(h.pattern).is_some_and(|p| p.trim) {
            for c in covered.iter_mut().take(h.last).skip(h.first) {
                *c = true;
            }
        }
    }
    toks.iter().zip(&covered).all(|(t, c)| *c || CONNECTORS.contains(&t.norm.as_str()))
}

/// `text` with every run of two or more exclamation marks made one (outside `keep`), and how
/// many runs were collapsed.
fn collapse_exclamations(text: &str, keep: &[(usize, usize)]) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut runs = 0;
    let mut it = text.char_indices().peekable();
    while let Some((b, c)) = it.next() {
        if !matches!(c, '!' | '\u{203C}') || inside(keep, b) {
            out.push(c);
            continue;
        }
        let mut weight = if c == '!' { 1 } else { 2 };
        let mut run = String::from(c);
        while let Some(&(_, d)) = it.peek() {
            if !matches!(d, '!' | '\u{203C}') {
                break;
            }
            weight += if d == '!' { 1 } else { 2 };
            run.push(d);
            it.next();
        }
        if weight >= 2 {
            out.push('!');
            runs += 1;
        } else {
            out.push_str(&run);
        }
    }
    (out, runs)
}

/// The safe mechanical fixes of one field: with `trim`, whole sentences at the start or end made
/// only of greetings and calls to action are cut (never the last sentence left); then runs of
/// exclamation marks become one. Nothing else changes: no word is replaced or reordered.
pub(crate) fn fix_text(text: &str, trim: bool, allow: &[Vec<String>]) -> (String, Vec<Value>) {
    let mut fixes = Vec::new();
    let mut t = text.to_string();
    if trim {
        for _ in 0..MAX_TRIMS {
            let spans = sentences(&t);
            if spans.len() < 2 {
                break;
            }
            let (Some(&(a, b)), Some(&(next, _))) = (spans.first(), spans.get(1)) else { break };
            if fluff(t.get(a..b).unwrap_or(""), allow) {
                fixes.push(json!({"fix": "trimmedStart", "removed": t.get(a..b).unwrap_or("")}));
                t = t.get(next..).unwrap_or("").to_string();
                continue;
            }
            let n = spans.len();
            let (Some(&(a, b)), Some(&(_, prev_end))) = (spans.get(n - 1), spans.get(n - 2)) else { break };
            if fluff(t.get(a..b).unwrap_or(""), allow) {
                fixes.push(json!({"fix": "trimmedEnd", "removed": t.get(a..b).unwrap_or("")}));
                t = t.get(..prev_end).unwrap_or("").to_string();
                continue;
            }
            break;
        }
    }
    let keep = {
        let web = urls(&t);
        let toks = tokens(&t, &web);
        let mut k = allowed_spans(&t, &toks, allow);
        k.extend(web);
        k
    };
    let (collapsed, runs) = collapse_exclamations(&t, &keep);
    if runs > 0 {
        fixes.push(json!({"fix": "collapsedExclamations", "runs": runs}));
    }
    (collapsed, fixes)
}

/// The publish copy with its safe fixes (`fix_text`: the description and caption trimmed and
/// collapsed, the title collapsed, tags as they are), what changed, and the lint of the result.
pub(crate) fn fix_publish(s: &Session, cmd: &str, p: &Value, c: &PublishCopy) -> Result<Value> {
    let allow = allowed(s, cmd, p)?;
    let ledger = ledger(s, cmd, p)?;
    let mut fixed = c.clone();
    let mut fixes = Vec::new();
    for (field, value, trim) in [("title", &mut fixed.title, false), ("description", &mut fixed.description, true), ("caption", &mut fixed.caption, true)] {
        if let Some(v) = value.as_mut() {
            let (t, f) = fix_text(v, trim, &allow.phrases);
            for mut x in f {
                x["field"] = json!(field);
                fixes.push(x);
            }
            *v = t;
        }
    }
    let before = report(&c.segments(), &allow, &ledger);
    let after = report(&fixed.segments(), &allow, &ledger);
    Ok(json!({
        "changed": fixed != *c,
        "fixed": fixed.to_json(),
        "fixes": fixes,
        "before": {"findings": before["findings"].as_array().map_or(0, Vec::len), "counts": before["counts"]},
        "lint": after,
        "note": "only safe fixes: stock greetings and calls to action trimmed when a whole sentence at the start or end of a description is made of them, repeated exclamation marks collapsed; the rest is for the writer",
    }))
}

/// The `copy` check `publish.preflight` adds to report `r`: the lint of the publish copy in
/// `p`, a warning (never an error, D7) when it finds anything.
pub(crate) fn publish_leg(s: &Session, cmd: &str, p: &Value, c: &PublishCopy, r: &mut Value) -> Result<()> {
    if !r.is_object() {
        return Err(bad(cmd, "internal: the pre-flight report is not an object"));
    }
    let allow = allowed(s, cmd, p)?;
    let ledger = ledger(s, cmd, p)?;
    let lint = report(&c.segments(), &allow, &ledger);
    let n = lint["findings"].as_array().map_or(0, Vec::len);
    let status = if n == 0 { "pass" } else { "warn" };
    let row = json!({
        "id": "copy", "label": "Title and description", "status": status,
        "value": if n == 0 { "no findings".to_string() } else { format!("{n} finding{}", if n == 1 { "" } else { "s" }) },
        "need": "no stock phrases, devices within budget, figures in the fact ledger",
        "fix": (n > 0).then_some("`publish.fixCopy` with the same copy trims stock greetings and calls to action and collapses repeated exclamation marks; reword the rest (see `copy`)"),
        "fixable": null,
    });
    if let Some(a) = r["checks"].as_array_mut() {
        a.push(row);
    }
    if n > 0 {
        r["warnings"] = json!(r["warnings"].as_u64().unwrap_or(0) + 1);
    }
    r["copy"] = lint;
    Ok(())
}

#[cfg(test)]
#[path = "copy_lint_tests.rs"]
mod tests;
