//! The on-screen text of a quick short ([`write_texts`]): each text slot's text says only what is
//! shown while it is on screen, and is checked before it is used.
//!
//! 1. **Where it is.** A text's window is what plays under it: the picture placements its own
//!    placement overlaps in the template ([`TextSpec::under`]), else the pick its `about` names.
//! 2. **What it may say.** The writer (the local model) is given, for each text, only the Qwen3-VL
//!    descriptions of the shots under it (subjects, actions, the text read on screen, the
//!    paragraph), the kept speech and the topic, and asked for three options, best first.
//! 3. **The words.** Every word of an option that names something ([`content_words`]: not a
//!    creator's stock word, [`GENERIC`]) must come from the topic, the speech or the descriptions
//!    of the shots under it ([`check_words`]); a word from nowhere ("drone" over a serving robot)
//!    rejects the option.
//! 4. **The frames.** An option naming what the descriptions say (and the topic and the speech
//!    do not) is checked against the frames of its window, cut to what the edit's frame shows of
//!    them (`media.checkText`): the in-process Qwen3-VL is asked about each thing it names, one
//!    at a time ("boy" once the boy has left the frame: no), and SigLIP 2 scores it against
//!    those frames, against the other shots' and against a plain alternative made from the
//!    descriptions ([`judge`]). Two texts of one video never say the same thing.
//! 5. **Again, then by rule.** Texts whose options all fail are asked for once more, with what
//!    was rejected and why; then the first deterministic text that passes ([`fallback_options`]:
//!    the subject's name, the text read on screen, the caption, the topic, the slot's own line).
//!
//! Every candidate, its verdict, the scores and the answers go in the report (`texts[].check`).
//! Without the models: without Ollama the deterministic texts; without SigLIP and Qwen3-VL the
//! words alone are checked (the report says so).

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use serde_json::{Value, json};

use super::{ItemSpeech, Pick, ShotInfo, TemplateInfo, chars, finite};
use crate::Executor;
use crate::llm::ChatModel;

/// Kept speech quoted to the writer, characters.
const SPEECH_QUOTE: usize = 700;
/// Options the writer gives for each text.
const OPTIONS: usize = 3;
/// Deterministic texts checked in the first round with the model's options (the rest after).
const FALLBACKS_FIRST: usize = 2;
/// Questions asked about one text's candidates in one round, and about one candidate, at most.
const ASKS_PER_TEXT: usize = 8;
const WORDS_ASKED: usize = 3;
/// A text overlapping a picture placement by less than this (seconds) is not over it.
const UNDER_OVERLAP: f64 = 0.25;
/// SigLIP margins ([`judge`]): a text may match another shot's frames this much better than its
/// own, and the plain alternative this much better, before it is taken as not about its shot.
pub const ELSEWHERE_MARGIN: f64 = 0.015;
pub const ALTERNATIVE_MARGIN: f64 = 0.04;

/// A text slot's place to write.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TextSpec {
    pub slot: String,
    pub index: usize,
    pub prompt: String,
    pub max_chars: usize,
    /// The pick it is about, when the slot says.
    pub about: Option<usize>,
    /// The picks shown while it is on screen (the longest overlap first; its `about` pick when
    /// the template does not say where it is).
    pub under: Vec<usize>,
    pub capitals: bool,
}

impl TextSpec {
    /// The picks its words are grounded in: those under it, and the one it is about.
    fn grounding(&self) -> Vec<usize> {
        let mut v = self.under.clone();
        if let Some(a) = self.about
            && !v.contains(&a)
        {
            v.push(a);
        }
        v
    }
}

/// The texts a template's text slots take.
pub fn text_specs(t: &TemplateInfo, picks: &[Pick]) -> Vec<TextSpec> {
    let mut out = Vec::new();
    for s in t.slots.iter().filter(|s| s.kind == "text") {
        let h = s.ai.clone().unwrap_or_default();
        let limit = s.max_chars.unwrap_or(40).clamp(1, 400) as usize;
        let max_chars = h.max_chars.map_or(limit, |m| (m as usize).min(limit)).max(1);
        let prompt = [h.prompt.as_str(), s.prompt.as_str(), s.name.as_str()].into_iter().find(|p| !p.trim().is_empty()).unwrap_or("A short line").to_string();
        let capitals = prompt.to_lowercase().contains("capital");
        for i in 0..s.count.min(64) {
            let about = (!h.about.is_empty()).then(|| picks.iter().position(|p| p.slot == h.about && p.media == i)).flatten();
            let mut under = under_text(t, &s.placements, i, picks);
            if under.is_empty() {
                under.extend(about);
            }
            out.push(TextSpec { slot: s.id.clone(), index: i, prompt: prompt.clone(), max_chars, about, under, capitals });
        }
    }
    out
}

/// The picks that play while text `i` (of a slot placed at `places`) is on screen: the picture
/// placements its placement overlaps in the template, the longest overlap first (a slot given
/// fewer picks than places reuses them in turn, as `template.apply` does).
fn under_text(t: &TemplateInfo, places: &[super::Place], i: usize, picks: &[Pick]) -> Vec<usize> {
    let Some((a, b)) = places.iter().find(|p| p.media == i).and_then(|p| Some((p.start?, p.start? + p.seconds))) else { return Vec::new() };
    let mut hits: Vec<(f64, usize)> = Vec::new();
    for ps in t.slots.iter().filter(|x| x.picture()) {
        let mut mine: Vec<(usize, usize)> = picks.iter().enumerate().filter(|(_, p)| p.slot == ps.id).map(|(k, p)| (p.media, k)).collect();
        mine.sort_unstable();
        if mine.is_empty() {
            continue;
        }
        for pl in &ps.placements {
            let Some(c) = pl.start else { continue };
            let overlap = b.min(c + pl.seconds) - a.max(c);
            if overlap < UNDER_OVERLAP {
                continue;
            }
            let Some(&(_, k)) = mine.get(pl.media % mine.len()) else { continue };
            match hits.iter_mut().find(|h| h.1 == k) {
                Some(h) => h.0 += overlap,
                None => hits.push((overlap, k)),
            }
        }
    }
    hits.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    hits.into_iter().map(|h| h.1).take(4).collect()
}

/// `t` on one line, without quotes, at most `max` characters (cut at a word when it can be).
pub fn tidy_text(t: &str, max: usize, capitals: bool) -> String {
    let t: String = t.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.trim().trim_matches(['"', '\'', '“', '”', '«', '»']).trim().trim_end_matches(['.', ';', ':']).to_string();
    let t = if capitals { t.to_uppercase() } else { t };
    if t.chars().count() <= max {
        return t;
    }
    let cut: String = t.chars().take(max).collect();
    match cut.rfind(' ') {
        Some(i) if i >= max / 2 => cut.get(..i).unwrap_or(&cut).trim_end_matches([',', '-', '–']).trim().to_string(),
        _ => cut.trim().to_string(),
    }
}

/// A name within `max` characters: the words at its start dropped first (a subject's adjectives
/// go before its noun: "bright orange robot" → "orange robot"), else cut.
fn fit_name(name: &str, max: usize, capitals: bool) -> String {
    let words: Vec<&str> = name.split_whitespace().collect();
    for k in 0..words.len() {
        let rest = words.get(k..).unwrap_or_default().join(" ");
        if rest.chars().count() <= max {
            return tidy_text(&rest, max, capitals);
        }
    }
    tidy_text(name, max, capitals)
}

/// Title case of `s`.
pub(crate) fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn shot_of<'a>(p: &Pick, shots: &'a [ShotInfo]) -> Option<&'a ShotInfo> {
    shots.iter().find(|s| s.item == p.item && s.index == p.shot)
}

/// The slot's own line when nothing is shown to name: a label, a call to action, a question, a
/// closing line, the topic, a hook.
fn slot_line(t: &TextSpec, topic: Option<&str>) -> Option<String> {
    let id = t.slot.to_lowercase();
    let p = t.prompt.to_lowercase();
    let pick = |a: &str, b: &str| if t.index == 0 { a.to_string() } else { b.to_string() };
    let line = if id.contains("label") {
        pick("Before", "After")
    } else if id.contains("cta") || id.contains("outro") && !id.contains("sub") || p.contains("call to action") {
        "Follow for more".into()
    } else if id.contains("question") && !id.contains("sub") {
        "Which one is your favourite?".into()
    } else if id.ends_with("_sub") || p.contains("closing line") || id.contains("closing") {
        "See you in the next one".into()
    } else if let Some(topic) = topic.filter(|t| !t.trim().is_empty()) {
        title_case(topic.trim())
    } else if id.contains("hook") {
        "Watch this".into()
    } else {
        return None;
    };
    Some(tidy_text(&line, t.max_chars, t.capitals))
}

/// Texts without a model, best first: what the shot it is about (its `about` link) shows — its
/// subject, the text read on it, its caption (for the longer slots) — then the topic and the
/// slot's own line. Each within the slot's length.
pub fn fallback_options(t: &TextSpec, picks: &[Pick], shots: &[ShotInfo], topic: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut put = |x: String| {
        let x = tidy_text(&x, t.max_chars, t.capitals);
        if !x.is_empty() && !out.iter().any(|o| o.eq_ignore_ascii_case(&x)) {
            out.push(x);
        }
    };
    let short = t.max_chars <= 20;
    let first = t.about;
    if let Some(d) = first.and_then(|i| picks.get(i)).and_then(|p| shot_of(p, shots)).and_then(ShotInfo::description) {
        let subject = d.subjects.iter().map(|s| s.name.trim()).find(|s| !s.is_empty()).map(str::to_string);
        let read = d.text.iter().map(|x| x.trim()).find(|x| x.chars().count() >= 3).map(str::to_string);
        let caption = Some(d.caption.trim().to_string()).filter(|c| !c.is_empty());
        // a subject's name that says nothing the topic does not ("robot" for "robots at the
        // expo") comes after the topic
        let topic_words: Vec<String> = topic.map(filmcraft_scenes::search::terms).unwrap_or_default();
        let plain = |x: &str| content_words(x).iter().all(|w| topic_words.iter().any(|t| same_word(t, w)));
        if let (Some(x), Some(tp)) = (&subject, topic.map(|x| title_case(x.trim())).filter(|x| !x.is_empty() && x.chars().count() <= t.max_chars))
            && plain(x)
        {
            put(tp);
        }
        if short {
            if let Some(x) = &subject {
                put(fit_name(&title_case(x), t.max_chars, t.capitals));
            }
            if let Some(x) = &read {
                put(title_case(x));
            }
        } else {
            for x in [caption, subject, read].into_iter().flatten() {
                put(title_case(&x));
            }
        }
    }
    if let Some(line) = slot_line(t, topic) {
        // the topic is the line of a hook or a title; a call to action keeps its own
        if let Some(tp) = topic.filter(|x| !x.trim().is_empty()) {
            let tp = title_case(tp.trim());
            if line != tidy_text(&tp, t.max_chars, t.capitals) && t.about.is_some() {
                put(tp);
            }
        }
        put(line);
    }
    out.truncate(6);
    out
}

/// A text without a model ([`fallback_options`]' first).
pub fn fallback_text(t: &TextSpec, picks: &[Pick], shots: &[ShotInfo], topic: Option<&str>) -> Option<String> {
    fallback_options(t, picks, shots, topic).into_iter().next()
}

// ---------------------------------------------------------------------------------------------
// the words
// ---------------------------------------------------------------------------------------------

/// Words that name nothing on screen: what creators write around what they show (a call to
/// action, a hook's verbs, praise, time). A text's other words must come from what it is shown
/// over, the topic or what is said.
pub const GENERIC: &[&str] = &[
    "meet",
    "watch",
    "wait",
    "look",
    "new",
    "next",
    "more",
    "follow",
    "subscribe",
    "like",
    "share",
    "save",
    "comment",
    "see",
    "check",
    "out",
    "part",
    "episode",
    "series",
    "channel",
    "week",
    "weekly",
    "every",
    "day",
    "today",
    "tomorrow",
    "tonight",
    "later",
    "now",
    "here",
    "there",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "our",
    "we",
    "us",
    "my",
    "me",
    "it",
    "they",
    "them",
    "what",
    "how",
    "why",
    "when",
    "who",
    "which",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "not",
    "no",
    "yes",
    "all",
    "any",
    "one",
    "first",
    "last",
    "best",
    "top",
    "most",
    "very",
    "so",
    "too",
    "just",
    "only",
    "also",
    "again",
    "ever",
    "never",
    "always",
    "still",
    "even",
    "really",
    "real",
    "big",
    "huge",
    "amazing",
    "awesome",
    "cool",
    "epic",
    "wow",
    "incredible",
    "insane",
    "crazy",
    "wild",
    "smart",
    "future",
    "futuristic",
    "tech",
    "technology",
    "innovation",
    "innovative",
    "thing",
    "stuff",
    "moment",
    "story",
    "highlight",
    "recap",
    "vibe",
    "magic",
    "world",
    "life",
    "time",
    "explore",
    "discover",
    "learn",
    "find",
    "enjoy",
    "love",
    "fun",
    "great",
    "good",
    "favourite",
    "favorite",
    "ultimate",
    "inside",
    "behind",
    "get",
    "got",
    "make",
    "made",
    "go",
    "going",
    "come",
    "coming",
    "take",
    "watching",
    "tour",
    "welcome",
    "hello",
    "hey",
    "thanks",
    "thank",
    "bye",
    "join",
    "let",
    "ready",
    "begin",
    "start",
    "end",
    "final",
    "reveal",
    "revealed",
    "unveil",
    "unveiled",
    "introducing",
    "intro",
    "outro",
    "full",
    "live",
    "year",
    "don't",
    "miss",
    "stay",
    "tuned",
    "forget",
    "about",
    "for",
    "with",
    "from",
    "into",
    "over",
    "under",
    "up",
    "down",
    "off",
    "but",
    "than",
    "then",
    "its",
    "here's",
    "next-level",
    "level",
    "wonder",
    "wonders",
    "action",
    "meets",
    "say",
    "hi",
    "beyond",
    "era",
    "age",
    "rise",
    "tomorrow's",
    "something",
    "everything",
    "nothing",
    "don",
    "won",
    "isn",
    "aren",
    "doesn",
    "didn",
    "before",
    "after",
    "heres",
    "whats",
    "let's",
    "lets",
    "yet",
    "soon",
    "worth",
    "must",
    "need",
    "needs",
    "want",
    "feel",
    "feels",
    "blow",
    "mind",
    "mindblowing",
    "event",
    "showcase",
    "ahead",
    "close",
    "closer",
    "up-close",
    "looks",
    "shot",
    "scene",
    "clip",
    "video",
    "short",
    "reel",
    "shorts",
    "reels",
];

/// [`GENERIC`] as [`filmcraft_scenes::search::terms`] spells its words.
fn generic() -> &'static [String] {
    static WORDS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    WORDS.get_or_init(|| {
        let mut v: Vec<String> = GENERIC.iter().flat_map(|g| filmcraft_scenes::search::terms(g)).collect();
        v.sort();
        v.dedup();
        v
    })
}

/// The words of `s` that may name something: [`filmcraft_scenes::search::terms`] (lower case,
/// stop words out, a plural `s` dropped) of three letters or more (or with a digit), without the
/// [`GENERIC`] ones.
pub fn content_words(s: &str) -> Vec<String> {
    let g = generic();
    filmcraft_scenes::search::terms(s)
        .into_iter()
        .filter(|w| w.chars().count() >= 3 || w.chars().any(|c| c.is_ascii_digit()))
        .filter(|w| g.binary_search(w).is_err())
        .take(32)
        .collect()
}

/// The word of `text` that `stem` (one of its [`content_words`]) was, as written (lower case).
fn as_written(text: &str, stem: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .find(|w| filmcraft_scenes::search::terms(w).first().is_some_and(|t| t == stem))
        .map_or_else(|| stem.to_string(), str::to_lowercase)
}

/// Whether two stemmed words are one: the same, or one the other's start (robot, robotic).
fn same_word(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (short, long) = if a.chars().count() <= b.chars().count() { (a, b) } else { (b, a) };
    short.chars().count() >= 4 && long.starts_with(short) && long.chars().count() - short.chars().count() <= 4
}

/// Where the words a text may use come from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grounds {
    /// The topic's and the speech's words: they may be said over anything.
    pub told: Vec<String>,
    /// The words of the descriptions of what it is shown over: said only when the frames show them.
    pub shown: Vec<String>,
}

impl Grounds {
    /// The grounds of text `t`.
    pub fn of(t: &TextSpec, picks: &[Pick], shots: &[ShotInfo], speech: &[ItemSpeech], topic: Option<&str>) -> Grounds {
        let mut told: Vec<String> = topic.map(filmcraft_scenes::search::terms).unwrap_or_default();
        let mut shown: Vec<String> = Vec::new();
        for i in t.grounding() {
            let Some(p) = picks.get(i) else { continue };
            if let Some(said) = &p.said {
                told.extend(filmcraft_scenes::search::terms(said));
            }
            if let Some(s) = shot_of(p, shots) {
                shown.extend(filmcraft_scenes::search::terms(&s.shot.text()));
                shown.extend(filmcraft_scenes::search::terms(&p.caption));
            }
        }
        for sp in speech.iter().filter(|s| s.talking()) {
            for k in sp.kept.iter().take(200) {
                told.extend(filmcraft_scenes::search::terms(&k.text));
            }
        }
        told.sort();
        told.dedup();
        told.truncate(4000);
        shown.sort();
        shown.dedup();
        shown.truncate(4000);
        Grounds { told, shown }
    }
}

/// What a text's words are grounded in.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct WordCheck {
    /// Named, but in nothing it is shown over, the topic or the speech.
    pub unknown: Vec<String>,
    /// Only in the descriptions of what it is shown over: the frames must show them.
    pub shown: Vec<String>,
}

/// The words of `text` against `g`.
pub fn check_words(text: &str, g: &Grounds) -> WordCheck {
    let mut out = WordCheck::default();
    for w in content_words(text) {
        if g.told.iter().any(|x| same_word(x, &w)) {
            continue;
        }
        if g.shown.iter().any(|x| same_word(x, &w)) {
            out.shown.push(w);
        } else {
            out.unknown.push(w);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// the frames
// ---------------------------------------------------------------------------------------------

/// What `media.checkText` answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Looked {
    /// Cosine of each text (rows) with each window's frames (columns).
    pub cosine: Option<Vec<Vec<f64>>>,
    /// The describer's answers: `(window, text)` → what each ask was about (one thing the text
    /// names, or the whole text) and `yes`, `no` or `unclear`.
    pub answers: HashMap<(usize, usize), Vec<(Option<String>, String)>>,
    pub embedder: Option<String>,
    pub asker: Option<String>,
    pub notes: Vec<String>,
    /// Questions sent to the describer (none: no text named what only the descriptions say).
    pub asked: usize,
}

impl Looked {
    /// Read an answer defensively (the job's `scenes`, or the answer itself).
    pub fn parse(v: &Value) -> Looked {
        let v = if v.get("scenes").is_some_and(Value::is_object) { &v["scenes"] } else { v };
        let cosine = v["cosine"].as_array().map(|rows| {
            rows.iter().take(256).map(|r| r.as_array().map(|c| c.iter().take(256).map(|x| finite(x).unwrap_or(0.0)).collect()).unwrap_or_default()).collect()
        });
        let mut answers: HashMap<(usize, usize), Vec<(Option<String>, String)>> = HashMap::new();
        for a in v["answers"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(256) {
            let (Some(w), Some(t), Some(ans)) = (a["window"].as_u64(), a["text"].as_u64(), a["answer"].as_str()) else { continue };
            if let (Ok(w), Ok(t)) = (usize::try_from(w), usize::try_from(t)) {
                answers.entry((w, t)).or_default().push((a["about"].as_str().map(|x| chars(x, 60)), chars(ans, 16)));
            }
        }
        let s = |k: &str| v[k].as_str().map(|x| chars(x, 120));
        let notes = v["notes"].as_array().map(|a| a.iter().filter_map(Value::as_str).take(10).map(|n| chars(n, 200)).collect()).unwrap_or_default();
        Looked { cosine, answers, embedder: s("embedder"), asker: s("asker"), notes, asked: 0 }
    }

    fn cos(&self, text: usize, window: usize) -> Option<f64> {
        self.cosine.as_ref()?.get(text)?.get(window).copied()
    }
}

/// A moment the texts are checked against: a pick's span of its media, and where the edit's
/// frame is centred on it.
#[derive(Clone, Debug, PartialEq)]
struct Window {
    item: u64,
    from: f64,
    to: f64,
    focus: Option<[f64; 2]>,
}

/// What was decided about one candidate text.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Verdict {
    pub text: String,
    /// `model`, `model, asked again` or `fallback`.
    pub by: String,
    pub ok: bool,
    pub why: Vec<String>,
    pub words: WordCheck,
    /// SigLIP: its cosine with the frames under it, the best with another shot's, the plain
    /// alternative's with the frames under it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub own: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elsewhere: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternative: Option<f64>,
    /// The describer's answer: does it see what the text names? (`no` when it misses any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// Each thing asked about and the answer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub asked: Vec<(String, String)>,
}

/// Where a candidate's scores are: its row in the check, the windows under its text, the other
/// windows (other clips), the plain alternative's row.
#[derive(Clone, Debug, Default)]
pub struct Spot<'a> {
    pub row: Option<usize>,
    pub under: &'a [usize],
    pub others: &'a [usize],
    pub alternative: Option<usize>,
}

fn r4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// Judge candidate `text` (see the module docs): its words, then — when it names what only the
/// descriptions say — the describer's answer, else SigLIP against the other shots and the plain
/// alternative. Without either model, the words alone decide.
pub fn judge(text: &str, by: &str, g: &Grounds, looked: Option<&Looked>, at: &Spot<'_>) -> Verdict {
    let mut v = Verdict { text: text.to_string(), by: by.to_string(), words: check_words(text, g), ..Verdict::default() };
    if text.trim().is_empty() {
        v.why.push("empty".into());
        return v;
    }
    if !v.words.unknown.is_empty() {
        v.why.push(format!("names {}, which nothing under it, the topic or the speech mentions", v.words.unknown.join(", ")));
        return v;
    }
    // the SigLIP scores, for the report and when the describer cannot say
    if let (Some(l), Some(row)) = (looked, at.row) {
        let own: Vec<f64> = at.under.iter().filter_map(|w| l.cos(row, *w)).collect();
        if !own.is_empty() {
            v.own = Some(r4(own.iter().sum::<f64>() / own.len() as f64));
            v.elsewhere = at.others.iter().filter_map(|w| l.cos(row, *w)).reduce(f64::max).map(r4);
            v.alternative = at.alternative.filter(|a| *a != row).and_then(|a| {
                let alt: Vec<f64> = at.under.iter().filter_map(|w| l.cos(a, *w)).collect();
                (!alt.is_empty()).then(|| r4(alt.iter().sum::<f64>() / alt.len() as f64))
            });
        }
        for (about, ans) in at.under.iter().filter_map(|w| l.answers.get(&(*w, row))).flatten() {
            v.asked.push((about.clone().unwrap_or_else(|| text.to_string()), ans.clone()));
        }
        let says = |a: &str| v.asked.iter().any(|(_, x)| x == a);
        if says("no") {
            v.answer = Some("no".into());
        } else if says("yes") {
            v.answer = Some("yes".into());
        } else if !v.asked.is_empty() {
            v.answer = Some("unclear".into());
        }
    }
    if v.words.shown.is_empty() {
        v.ok = true;
        v.why.push("names only what the topic or the speech says".into());
        return v;
    }
    let named = v.words.shown.join(", ");
    match v.answer.as_deref() {
        Some("no") => {
            let missed: Vec<&str> = v.asked.iter().filter(|(_, a)| a == "no").map(|(w, _)| w.as_str()).collect();
            v.why.push(format!("the describer does not see {} in the frames under it", missed.join(", ")));
            return v;
        }
        Some("yes") => {
            v.ok = true;
            v.why.push(format!("the describer sees {named} in the frames under it"));
            return v;
        }
        _ => {}
    }
    match v.own {
        Some(own) => {
            if let Some(e) = v.elsewhere.filter(|e| own + ELSEWHERE_MARGIN < *e) {
                v.why.push(format!("it matches another shot's frames better than its own (SigLIP {own:.3} < {e:.3})"));
            } else if let Some(a) = v.alternative.filter(|a| own + ALTERNATIVE_MARGIN < *a) {
                v.why.push(format!("a plain description matches its frames much better (SigLIP {own:.3} < {a:.3})"));
            } else {
                v.ok = true;
                v.why.push(format!("SigLIP: it matches its own frames ({own:.3}) as well as any other shot's"));
            }
        }
        None => {
            v.ok = true;
            v.why.push(format!("{named}: in the descriptions of what it is shown over (not looked at: no vision model)"));
        }
    }
    v
}

// ---------------------------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------------------------

const WRITE_SYSTEM: &str = "You write the on-screen text of a short vertical video (YouTube Shorts, Instagram Reels, TikTok). Each text is short, concrete and natural, like a creator would write it: no hashtags, no emoji, no quotation marks. Each text is shown over the shots listed with it: it may name only what those shots show (their subjects, what happens, the text read on them), the topic and what is said. Never name a person, an animal, a vehicle or an object that is not listed with that text, never move something from one text's shots to another's, and never invent names, numbers, prices or claims. A call to action or a closing line names nothing it cannot see. Keep every text within its character limit. Give three different options for each text, the best first. Answer with JSON only.";

/// What the writer is told about the shots under a text.
fn shots_line(t: &TextSpec, picks: &[Pick], shots: &[ShotInfo]) -> String {
    let mut out = String::new();
    for i in t.grounding().into_iter().take(4) {
        let Some(p) = picks.get(i) else { continue };
        let len = if p.still { "a photo".to_string() } else { format!("{:.1} s of {}", (p.out_seconds - p.in_seconds).max(0.0), chars(&p.name, 60)) };
        let mut line = format!("\n   - {len}: {}", chars(&p.caption, 220));
        if let Some(d) = shot_of(p, shots).and_then(ShotInfo::description) {
            let subjects: Vec<&str> = d.subjects.iter().map(|x| x.name.trim()).filter(|n| !n.is_empty()).take(4).collect();
            if !subjects.is_empty() {
                line.push_str(&format!(" | shows: {}", chars(&subjects.join(", "), 120)));
            }
            let actions: Vec<&str> = d.actions.iter().map(|x| x.trim()).filter(|n| !n.is_empty()).take(3).collect();
            if !actions.is_empty() {
                line.push_str(&format!(" | happening: {}", chars(&actions.join("; "), 160)));
            }
            if !d.setting.trim().is_empty() {
                line.push_str(&format!(" | at: {}", chars(d.setting.trim(), 80)));
            }
            let read: Vec<&str> = d.text.iter().map(String::as_str).filter(|x| !x.trim().is_empty()).take(4).collect();
            if !read.is_empty() {
                line.push_str(&format!(" | text on screen: {}", chars(&read.join(", "), 120)));
            }
            if !d.detail.trim().is_empty() {
                line.push_str(&format!(" | in detail: {}", chars(d.detail.trim(), 400)));
            }
        }
        if let Some(said) = &p.said {
            line.push_str(&format!(" | said: \"{}\"", chars(said, 200)));
        }
        out.push_str(&line);
    }
    if out.is_empty() { " (nothing to name: write it without naming anything)".into() } else { out }
}

/// The request for `specs` (with what was rejected before, when asking again).
fn request(
    specs: &[&TextSpec],
    picks: &[Pick],
    shots: &[ShotInfo],
    speech: &[ItemSpeech],
    topic: Option<&str>,
    language: Option<&str>,
    rejected: &HashMap<(String, usize), Vec<Verdict>>,
) -> String {
    let mut c = String::new();
    c.push_str(&format!("Topic: {}\n", topic.filter(|t| !t.trim().is_empty()).unwrap_or("(not given)")));
    if let Some(l) = language.filter(|l| !l.is_empty() && *l != "en" && *l != "auto") {
        c.push_str(&format!("Write the texts in the language with ISO code \"{l}\".\n"));
    }
    let kept: String = speech.iter().filter(|s| s.talking()).flat_map(|s| s.kept.iter().map(|k| k.text.as_str())).collect::<Vec<_>>().join(" ");
    if !kept.trim().is_empty() {
        c.push_str(&format!("What is said (kept takes): \"{}\"\n", chars(kept.trim(), SPEECH_QUOTE)));
    }
    c.push_str("The texts, each with the shots shown under it:\n");
    for (k, t) in specs.iter().enumerate().take(64) {
        let caps = if t.capitals { ", in capitals" } else { "" };
        c.push_str(&format!(
            "{}. slot \"{}\" n {} (at most {} characters{caps}): {}\n   Shown over:{}\n",
            k + 1,
            t.slot,
            t.index + 1,
            t.max_chars,
            chars(&t.prompt, 300),
            shots_line(t, picks, shots)
        ));
        if let Some(r) = rejected.get(&(t.slot.clone(), t.index)).filter(|r| !r.is_empty()) {
            let list: Vec<String> = r.iter().take(6).map(|v| format!("\"{}\" ({})", chars(&v.text, 60), chars(&v.why.join("; "), 160))).collect();
            c.push_str(&format!("   Rejected, do not repeat: {}\n", list.join("; ")));
        }
    }
    c.push_str("Answer: {\"texts\": [{\"slot\": \"…\", \"n\": 1, \"options\": [\"…\", \"…\", \"…\"]}, …]} with one entry per text.");
    c
}

/// The writer's options for `specs` (tidied, at most [`OPTIONS`] each), keyed by slot and index.
async fn write_options(
    m: &dyn ChatModel,
    specs: &[&TextSpec],
    picks: &[Pick],
    shots: &[ShotInfo],
    speech: &[ItemSpeech],
    topic: Option<&str>,
    language: Option<&str>,
    rejected: &HashMap<(String, usize), Vec<Verdict>>,
    notes: &mut Vec<String>,
) -> HashMap<(String, usize), Vec<String>> {
    let mut got: HashMap<(String, usize), Vec<String>> = HashMap::new();
    if specs.is_empty() {
        return got;
    }
    let user = request(specs, picks, shots, speech, topic, language, rejected);
    match m.complete_json(WRITE_SYSTEM, &user, &[]).await {
        Ok(v) => {
            for x in v["texts"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(256) {
                // models change the slot's case and write numbers as text: both are read
                let Some(slot) = x["slot"].as_str() else { continue };
                let n = x["n"].as_u64().or_else(|| x["n"].as_str().and_then(|n| n.trim().parse().ok())).unwrap_or(1);
                let Some(i) = usize::try_from(n).ok().and_then(|n| n.checked_sub(1)) else { continue };
                let same =
                    |a: &str| a.trim().eq_ignore_ascii_case(slot.trim()) || a.replace(['_', '-'], "").eq_ignore_ascii_case(&slot.replace(['_', '-', ' '], ""));
                let Some(t) = specs.iter().find(|t| same(&t.slot) && t.index == i) else { continue };
                let mut texts: Vec<&str> = x["options"].as_array().map(|a| a.iter().filter_map(Value::as_str).take(8).collect()).unwrap_or_default();
                if let Some(one) = x["text"].as_str() {
                    texts.insert(0, one);
                }
                let list = got.entry((t.slot.clone(), i)).or_default();
                for text in texts {
                    let tidy = tidy_text(text, t.max_chars, t.capitals);
                    if !tidy.is_empty() && !list.iter().any(|o| o.eq_ignore_ascii_case(&tidy)) && list.len() < OPTIONS {
                        list.push(tidy);
                    }
                }
            }
            got.retain(|_, l| !l.is_empty());
            if got.is_empty() {
                notes.push(format!("text: {} answered no usable text; written from the descriptions instead", m.id()));
            }
        }
        Err(e) => notes.push(format!("text: {} did not answer ({e}); written from the descriptions instead", m.id())),
    }
    got
}

// ---------------------------------------------------------------------------------------------
// writing and checking
// ---------------------------------------------------------------------------------------------

/// One text as written and checked.
#[derive(Clone, Debug)]
pub struct Written {
    pub spec: TextSpec,
    pub text: Option<String>,
    /// `model`, `model, asked again` or `fallback`.
    pub by: String,
    /// Every candidate looked at, in order, with its verdict.
    pub tried: Vec<Verdict>,
}

impl Written {
    /// The report's entry.
    pub fn to_json(&self, picks: &[Pick]) -> Value {
        let under: Vec<Value> = self
            .spec
            .under
            .iter()
            .filter_map(|i| picks.get(*i))
            .map(|p| json!({"slot": p.slot, "n": p.media + 1, "item": p.item, "name": p.name, "from": p.in_seconds, "to": p.out_seconds}))
            .collect();
        json!({
            "slot": self.spec.slot,
            "n": self.spec.index + 1,
            "text": self.text,
            "by": self.by,
            "under": under,
            "candidates": self.tried,
        })
    }
}

/// The windows of the picks (each pick's span of its media).
fn windows(picks: &[Pick]) -> Vec<Window> {
    picks
        .iter()
        .take(filmcraft_engine::scene_check::MAX_WINDOWS)
        .map(|p| Window { item: p.item, from: p.in_seconds.max(0.0), to: p.out_seconds.max(p.in_seconds).max(0.0), focus: p.focus })
        .collect()
}

/// One ask: a window, a text's row and the one thing of it asked about.
type AskAbout = (usize, usize, String);

/// One round of checking: the texts against the frames of the windows (cut to the edit's frame,
/// `aspect`), the describer asked about each thing in `asks`.
async fn look(ex: &dyn Executor, wins: &[Window], aspect: Option<f64>, texts: &[String], asks: &[AskAbout], notes: &mut Vec<String>) -> Option<Looked> {
    if wins.is_empty() || texts.is_empty() {
        return None;
    }
    let mut p = json!({
        "windows": wins
            .iter()
            .map(|w| {
                let mut x = json!({"item": w.item, "from": w.from, "to": w.to});
                if let Some(f) = w.focus {
                    x["focus"] = json!(f);
                }
                x
            })
            .collect::<Vec<_>>(),
        "texts": texts,
        "asks": asks.iter().map(|(w, t, a)| json!({"window": w, "text": t, "about": a})).collect::<Vec<_>>(),
        "frames": 2,
        "wait": true,
    });
    if let Some(a) = aspect.filter(|a| a.is_finite() && (0.1..=10.0).contains(a)) {
        p["aspect"] = json!(a);
    }
    match ex.execute("media.checkText", p).await {
        Ok(v) => Some(Looked { asked: asks.len(), ..Looked::parse(&v) }),
        Err(e) => {
            notes.push(format!("text check: the frames could not be looked at ({}); the words alone were checked", chars(&e, 200)));
            None
        }
    }
}

/// The candidates of one text in a round: `(text, by)`.
type Candidates = Vec<(String, String)>;

/// Judge every candidate of every open text in one round: words first, the frames for those that
/// need them. Returns each text's verdicts in order.
async fn round(
    ex: Option<&dyn Executor>,
    specs: &[TextSpec],
    open: &[usize],
    cands: &BTreeMap<usize, Candidates>,
    alternatives: &BTreeMap<usize, String>,
    grounds: &[Grounds],
    picks: &[Pick],
    aspect: Option<f64>,
    taken: &mut Vec<String>,
    looked_any: &mut Option<Looked>,
    notes: &mut Vec<String>,
) -> BTreeMap<usize, Vec<Verdict>> {
    let wins = windows(picks);
    // every text once: the candidates and the plain alternatives
    let mut texts: Vec<String> = Vec::new();
    let row = |t: &str, texts: &mut Vec<String>| -> Option<usize> {
        if let Some(k) = texts.iter().position(|x| x == t) {
            return Some(k);
        }
        (texts.len() < filmcraft_engine::scene_check::MAX_TEXTS).then(|| {
            texts.push(t.to_string());
            texts.len() - 1
        })
    };
    let mut asks: Vec<AskAbout> = Vec::new();
    for &si in open {
        let (Some(spec), Some(g)) = (specs.get(si), grounds.get(si)) else { continue };
        let under: Vec<usize> = spec.under.iter().copied().filter(|w| *w < wins.len()).collect();
        if let Some(a) = alternatives.get(&si) {
            row(a, &mut texts);
        }
        let mut asked = 0;
        for (text, _) in cands.get(&si).map(Vec::as_slice).unwrap_or_default() {
            let Some(r) = row(text, &mut texts) else { continue };
            let w = check_words(text, g);
            // only what the descriptions alone say is asked about — each thing on its own, over
            // the text's main shot
            if !w.unknown.is_empty() {
                continue;
            }
            let Some(first) = under.first() else { continue };
            for word in w.shown.iter().take(WORDS_ASKED) {
                if asked >= ASKS_PER_TEXT || asks.len() >= filmcraft_engine::scene_check::MAX_ASKS {
                    break;
                }
                asks.push((*first, r, as_written(text, word)));
                asked += 1;
            }
        }
    }
    let looked = match ex {
        Some(ex) => look(ex, &wins, aspect, &texts, &asks, notes).await,
        None => None,
    };
    let mut out = BTreeMap::new();
    for &si in open {
        let (Some(spec), Some(g)) = (specs.get(si), grounds.get(si)) else { continue };
        let under: Vec<usize> = spec.under.iter().copied().filter(|w| *w < wins.len()).collect();
        // the other shots: other clips' windows (another moment of the same clip may look alike)
        let items: Vec<u64> = under.iter().filter_map(|w| wins.get(*w)).map(|w| w.item).collect();
        let others: Vec<usize> = (0..wins.len()).filter(|w| !under.contains(w) && wins.get(*w).is_some_and(|x| !items.contains(&x.item))).collect();
        let alt = alternatives.get(&si).and_then(|a| texts.iter().position(|x| x == a));
        let mut verdicts = Vec::new();
        for (text, by) in cands.get(&si).map(Vec::as_slice).unwrap_or_default() {
            let at = Spot { row: texts.iter().position(|x| x == text), under: &under, others: &others, alternative: alt };
            let mut v = judge(text, by, g, looked.as_ref(), &at);
            // one text, one place: the same words twice on screen read as a mistake
            let key = text.to_lowercase();
            if v.ok && taken.contains(&key) {
                v.ok = false;
                v.why.push("the same as another text of this video".into());
            }
            let done = v.ok;
            verdicts.push(v);
            if done {
                taken.push(key);
                break;
            }
        }
        out.insert(si, verdicts);
    }
    // what checked the texts in any round: a later round that asked nothing still had the answers
    // of an earlier one
    if let Some(mut l) = looked {
        if let Some(prev) = looked_any.take() {
            l.asker = l.asker.or(prev.asker);
            l.asked = l.asked.saturating_add(prev.asked);
            l.embedder = l.embedder.or(prev.embedder);
            for n in prev.notes {
                if !l.notes.contains(&n) {
                    l.notes.push(n);
                }
            }
        }
        *looked_any = Some(l);
    }
    out
}

/// Write and check the texts (see the module docs). `ex` runs `media.checkText` (none: the words
/// alone are checked) on frames cut to the edit's frame (`aspect`, width / height). Returns each
/// text and how the checking went (for the report).
#[allow(clippy::too_many_arguments)]
pub async fn write_texts(
    ex: Option<&dyn Executor>,
    aspect: Option<f64>,
    model: Option<&dyn ChatModel>,
    specs: &[TextSpec],
    picks: &[Pick],
    shots: &[ShotInfo],
    speech: &[ItemSpeech],
    topic: Option<&str>,
    language: Option<&str>,
    notes: &mut Vec<String>,
) -> (Vec<Written>, Value) {
    let grounds: Vec<Grounds> = specs.iter().map(|t| Grounds::of(t, picks, shots, speech, topic)).collect();
    let fallbacks: Vec<Vec<String>> = specs.iter().map(|t| fallback_options(t, picks, shots, topic)).collect();
    // the plain alternative a text is compared with: what the descriptions name under it
    let alternatives: BTreeMap<usize, String> = fallbacks.iter().enumerate().filter_map(|(i, f)| Some((i, f.first()?.clone()))).collect();
    let mut tried: Vec<Vec<Verdict>> = vec![Vec::new(); specs.len()];
    let mut chosen: Vec<Option<(String, String)>> = vec![None; specs.len()];
    let mut looked: Option<Looked> = None;
    let mut taken: Vec<String> = Vec::new();
    let no_rejects = HashMap::new();
    // round 1: the model's options, then the first deterministic texts
    let all: Vec<&TextSpec> = specs.iter().collect();
    let written = match model {
        Some(m) => write_options(m, &all, picks, shots, speech, topic, language, &no_rejects, notes).await,
        None => HashMap::new(),
    };
    let mut cands: BTreeMap<usize, Candidates> = BTreeMap::new();
    for (i, t) in specs.iter().enumerate() {
        let mut c: Candidates =
            written.get(&(t.slot.clone(), t.index)).map(|l| l.iter().map(|x| (x.clone(), "model".to_string())).collect()).unwrap_or_default();
        if model.is_none() || c.is_empty() {
            c.extend(fallbacks.get(i).map(Vec::as_slice).unwrap_or_default().iter().take(FALLBACKS_FIRST).map(|x| (x.clone(), "fallback".to_string())));
        }
        cands.insert(i, c);
    }
    let open: Vec<usize> = (0..specs.len()).collect();
    let verdicts = round(ex, specs, &open, &cands, &alternatives, &grounds, picks, aspect, &mut taken, &mut looked, notes).await;
    settle(&verdicts, &mut tried, &mut chosen);
    // round 2: the model asked again for the texts it got wrong, with why
    let open: Vec<usize> = (0..specs.len()).filter(|i| chosen.get(*i).is_some_and(Option::is_none)).collect();
    if let (Some(m), false) = (model, open.is_empty()) {
        let again: Vec<&TextSpec> = open.iter().filter_map(|i| specs.get(*i)).collect();
        let rejected: HashMap<(String, usize), Vec<Verdict>> =
            open.iter().filter_map(|i| Some(((specs.get(*i)?.slot.clone(), specs.get(*i)?.index), tried.get(*i)?.clone()))).collect();
        let written = write_options(m, &again, picks, shots, speech, topic, language, &rejected, notes).await;
        let mut cands: BTreeMap<usize, Candidates> = BTreeMap::new();
        for &i in &open {
            let Some(t) = specs.get(i) else { continue };
            let before: Vec<&str> = tried.get(i).map(|v| v.iter().map(|x| x.text.as_str()).collect()).unwrap_or_default();
            let mut c: Candidates = written
                .get(&(t.slot.clone(), t.index))
                .map(|l| {
                    l.iter().filter(|x| !before.iter().any(|b| b.eq_ignore_ascii_case(x))).map(|x| (x.clone(), "model, asked again".to_string())).collect()
                })
                .unwrap_or_default();
            c.extend(
                fallbacks
                    .get(i)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter(|x| !before.iter().any(|b| b.eq_ignore_ascii_case(x)))
                    .map(|x| (x.clone(), "fallback".to_string())),
            );
            cands.insert(i, c);
        }
        let verdicts = round(ex, specs, &open, &cands, &alternatives, &grounds, picks, aspect, &mut taken, &mut looked, notes).await;
        settle(&verdicts, &mut tried, &mut chosen);
    } else if !open.is_empty() {
        // no model: the rest of the deterministic texts
        let mut cands: BTreeMap<usize, Candidates> = BTreeMap::new();
        for &i in &open {
            let c: Candidates =
                fallbacks.get(i).map(Vec::as_slice).unwrap_or_default().iter().skip(FALLBACKS_FIRST).map(|x| (x.clone(), "fallback".to_string())).collect();
            cands.insert(i, c);
        }
        let verdicts = round(ex, specs, &open, &cands, &alternatives, &grounds, picks, aspect, &mut taken, &mut looked, notes).await;
        settle(&verdicts, &mut tried, &mut chosen);
    }
    // what is still open: the slot's own line (it names nothing), else nothing
    let out: Vec<Written> = specs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let tried_i = tried.get(i).cloned().unwrap_or_default();
            match chosen.get(i).cloned().flatten() {
                Some((text, by)) => Written { spec: t.clone(), text: Some(text), by, tried: tried_i },
                None => {
                    let line = slot_line(t, None).filter(|l| content_words(l).is_empty() && !taken.contains(&l.to_lowercase()));
                    if line.is_none() {
                        notes.push(format!("text: no text for `{}` {} passed the check: the template's own text stays", t.slot, t.index + 1));
                    }
                    Written { spec: t.clone(), text: line, by: "fallback".into(), tried: tried_i }
                }
            }
        })
        .collect();
    let how = match &looked {
        Some(l) if l.asker.is_some() && l.embedder.is_some() => "words, the describer's answers and SigLIP",
        Some(l) if l.asker.is_some() => "words and the describer's answers",
        Some(l) if l.embedder.is_some() && l.asked == 0 => "words and SigLIP (nothing to ask the describer)",
        Some(l) if l.embedder.is_some() => "words and SigLIP",
        _ => "words only",
    };
    let summary = json!({
        "how": how,
        "embedder": looked.as_ref().and_then(|l| l.embedder.clone()),
        "asker": looked.as_ref().and_then(|l| l.asker.clone()),
        "asked": looked.as_ref().map_or(0, |l| l.asked),
        "rejected": out.iter().map(|w| w.tried.iter().filter(|v| !v.ok).count()).sum::<usize>(),
        "notes": looked.as_ref().map(|l| l.notes.clone()).unwrap_or_default(),
        "margins": {"elsewhere": ELSEWHERE_MARGIN, "alternative": ALTERNATIVE_MARGIN},
    });
    (out, summary)
}

/// Keep each text's verdicts, and its first passing candidate.
fn settle(verdicts: &BTreeMap<usize, Vec<Verdict>>, tried: &mut [Vec<Verdict>], chosen: &mut [Option<(String, String)>]) {
    for (i, list) in verdicts {
        if let Some(t) = tried.get_mut(*i) {
            t.extend(list.iter().cloned());
        }
        if let (Some(c), Some(v)) = (chosen.get_mut(*i), list.iter().find(|v| v.ok)) {
            *c = Some((v.text.clone(), v.by.clone()));
        }
    }
}
