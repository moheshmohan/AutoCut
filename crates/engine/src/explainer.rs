//! `script.timeline` (PX7.1): an explainer's script, read in its markup, spoken by the narration
//! voice and laid out as a timeline: shot ranges, subtitle blocks, chapter starts, sequence
//! markers and one JSON a shot for its composition. The idea of a narration-driven shot
//! timeline comes from anything2explainer (Vincentwei1021/anything2explainer, PolyForm
//! Noncommercial): only the idea is taken, from its README; the markup, the layout, the numbers
//! and every line here are written fresh.
//!
//! **The markup.**
//!
//! | write | means |
//! |---|---|
//! | `# Title` (any number of `#`, then a space) | a chapter starts; its title is shown on screen (the chapter marker, the chapter bar), never spoken |
//! | a blank line | the paragraph ends: one paragraph is one shot (split into sentence groups when it speaks longer than `maxShotSeconds`) |
//! | `[shot: …]` | a hint for the paragraph's picture (what the composition or card shows: a name, a number, a diagram); on a line of its own (before or inside its paragraph) or inline; never spoken; at most 8 a paragraph |
//! | `[pause 1s]` | a pause the voice makes (the narration's own marker); Malayalam scripts also take the delivery tags (`[warm]`, `[excited]`, `[breath]`…) |
//!
//! Text before the first heading is an untitled first chapter. Refused, with the line: a heading
//! without a title, a chapter without a paragraph, a hint with no paragraph to go with, an
//! unclosed or empty `[shot:`, any other bracket (it would be read aloud), a paragraph of marks
//! only. Names belong on screen, not in the voice: put them in a hint or a heading.
//!
//! **The voice.** Every sentence is spoken on its own by the narration path (`tts.create`'s
//! voices, settings, caches and strict check), so sentence times are exact; `estimate: true`
//! times them from the words instead (2.5 words a second and the pauses), with no voice. Word
//! times inside a sentence are spread by letters (an estimate, said in the result).
//!
//! **The layout.** Shots follow each other with no gap and no overlap: a shot runs from the
//! start of its first sentence (on the frame grid) to the start of the next shot, which is its
//! speech plus the shot gap (the chapter gap before a new chapter, the tail after the last
//! shot), rounded up to a frame. So the shots add up to the timeline, which is the speech and the
//! gaps. Inside a shot the sentences follow with the sentence gap. A subtitle block is at most
//! `captionChars` characters (two lines), timed by its letters inside its sentence, and held to
//! its reading time (3 words a second plus 0.5 s) when there is room before the next one.
//!
//! **What comes back.** `shots` (id `shot-01`…, chapter, start, end, duration, frames, the shown
//! and spoken text, hints, sentences, subtitles), `chapters` (title, start, end, its shots),
//! `subtitles` (with `words`), `markers` (a chapter marker at each chapter start, named by its
//! title, and a comment marker `shot-NN` at each shot start, its comment the shot's first hint)
//! and `narration` (speech, gaps, how many sentences came from the caches). With `out` (a new or
//! empty folder): `timeline.json` and one `shot-NN.json` a shot, times relative to the shot in
//! seconds and frames, for the composition that draws it. With `place`: in one undo step a new
//! sequence (or the active one), each shot's narration as one clip (a narration item `tts.edit`
//! edits) on one audio track, the subtitles on a new caption track and the markers.
//!
//! Bounded: 64 KB of script, 64 chapters, 300 shots, 600 sentences, 3 hours of timeline.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::Arc;

use serde_json::{Map, Value, json};

use filmcraft_project::{CaptionWord, ClipId, ItemId, Label, Marker, MarkerId, MarkerKind, TrackId, TrackKind};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use filmcraft_tts as tts;

use crate::commands::{CommandSpec, always, bad};
use crate::motion_energy::r3;
use crate::{EngineError, Result, Session};

pub const CMD: &str = "script.timeline";
/// Longest script, bytes (the narration's own limit).
pub const MAX_TEXT_BYTES: usize = tts::MAX_TEXT_BYTES;
/// Most chapters, shots and sentences.
pub const MAX_CHAPTERS: usize = 64;
pub const MAX_SHOTS: usize = 300;
pub const MAX_SENTENCES: usize = 600;
/// Most hints a paragraph has, and the longest hint and title, characters.
pub const MAX_HINTS: usize = 8;
pub const MAX_HINT_CHARS: usize = 300;
pub const MAX_TITLE_CHARS: usize = 200;
/// Longest timeline, seconds.
pub const MAX_TOTAL_SECONDS: f64 = 3.0 * 3600.0;
/// Words a second an estimate speaks, and its shortest sentence, seconds.
pub const ESTIMATE_WPS: f64 = 2.5;
const ESTIMATE_MIN_SECONDS: f64 = 0.5;
/// The gaps, seconds: between sentences, after a shot, before a new chapter, after the last shot.
pub const SENTENCE_GAP: f64 = 0.25;
pub const SHOT_GAP: f64 = 0.6;
pub const CHAPTER_GAP: f64 = 1.0;
pub const TAIL: f64 = 1.0;
const MAX_GAP: f64 = 10.0;
/// A paragraph speaking longer than this is split into sentence groups, seconds (and its bounds).
pub const MAX_SHOT_SECONDS: f64 = 12.0;
const SHOT_SECONDS_RANGE: (f64, f64) = (2.0, 120.0);
/// Longest subtitle block, characters (two lines), and its bounds.
pub const CAPTION_CHARS: usize = 84;
const CAPTION_CHARS_RANGE: (usize, usize) = (16, 200);
/// Reading time a subtitle is held to when there is room: words a second and a floor (as
/// `sequence.checkCraft` reads).
const READ_WPS: f64 = 3.0;
const READ_FLOOR: f64 = 0.5;
/// Shot ids and their markers' names start with this.
pub const SHOT_PREFIX: &str = "shot-";
/// The narration settings passed on to the voice.
const VOICE_KEYS: &[&str] = &[
    "voice",
    "language",
    "pitch",
    "pace",
    "delivery",
    "numbers",
    "temperature",
    "emphasis",
    "speed",
    "expressiveness",
    "take",
    "check",
    "checkPolicy",
    "agreement",
    "voiceGate",
    "minSimilarity",
];
const KEYS: &[&str] = &["text", "path", "estimate", "gaps", "maxShotSeconds", "captionChars", "fps", "out", "place"];
/// The format of the files written with `out`.
pub const SHOT_FORMAT: &str = "autocut.explainerShot";
pub const TIMELINE_FORMAT: &str = "autocut.explainerTimeline";

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Script to Timeline",
        menu: &[],
        shortcut: None,
        params: r#"{"text":str? (the script: `# Chapter` headings, blank-line paragraphs, [shot: …] hints, [pause 1s]),"path":path? (a script file instead, at most 64 KB),"voice":str?,"language":"en-US|ml-IN"?,"pitch":str?,"pace":0.5..2?,"delivery":str?,"numbers":str?,"check":"strict|off"?,"checkPolicy":{…}? (as tts.create),"estimate":bool=false (time from the words, no voice),"gaps":{"sentence":s=0.25,"shot":s=0.6,"chapter":s=1,"tail":s=1}?,"maxShotSeconds":2..120=12 (a longer paragraph is split into sentence groups),"captionChars":16..200=84 (longest subtitle block),"fps":f64? (default the place's, else the active sequence's, else 30),"out":path? (a new or empty folder: timeline.json and shot-NN.json),"place":bool|{"sequence":"new"|"active"="new","name":str?,"width":u32?,"height":u32?,"fps":f64?,"at":ticks?|"seconds":f64?,"track":"A1"?,"narration":bool=true,"captions":bool=true,"markers":bool=true,"dir":path? (the narration files)}?}"#,
        enabled: always,
        run,
        journal: true,
    }]
}

// -------------------------------------------------------------------------------------------
// the markup
// -------------------------------------------------------------------------------------------

/// A paragraph: what the voice reads (pause markers and delivery tags kept, hints taken out) and
/// its hints.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Paragraph {
    pub text: String,
    pub hints: Vec<String>,
    /// Its first line, 1-based.
    pub line: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Chapter {
    /// Empty for the untitled first chapter.
    pub title: String,
    pub line: usize,
    pub paragraphs: Vec<Paragraph>,
}

fn clip_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The `[shot: …]` hints of `line` taken out: the rest and the hints.
fn take_hints(line: &str, at: usize) -> std::result::Result<(String, Vec<String>), String> {
    let mut rest = String::new();
    let mut hints = Vec::new();
    let mut tail = line;
    loop {
        let lower = tail.to_lowercase();
        // `to_lowercase` can change byte lengths outside ASCII: only use the position when the
        // prefix is unchanged
        let Some(i) = lower.find("[shot:").filter(|i| tail.get(..*i).is_some_and(|h| h.to_lowercase().len() == *i)) else {
            rest.push_str(tail);
            break;
        };
        rest.push_str(tail.get(..i).unwrap_or(""));
        let after = tail.get(i + "[shot:".len()..).unwrap_or("");
        let Some(close) = after.find(']') else {
            return Err(format!("line {at}: `[shot:` is not closed with `]`"));
        };
        let hint = after.get(..close).unwrap_or("").trim();
        if hint.contains('[') {
            return Err(format!("line {at}: a `[` inside a [shot: …] hint; close the hint first"));
        }
        if hint.is_empty() {
            return Err(format!("line {at}: an empty [shot: ] hint; say what the picture shows, or take it out"));
        }
        if hint.chars().count() > MAX_HINT_CHARS {
            return Err(format!("line {at}: a [shot: …] hint is {} characters; at most {MAX_HINT_CHARS}", hint.chars().count()));
        }
        hints.push(hint.to_string());
        rest.push(' ');
        tail = after.get(close + 1..).unwrap_or("");
    }
    Ok((rest, hints))
}

/// Whether `mark` (`[…]`, brackets included) is a narration marker the voice understands: a
/// pause, or (Malayalam) a delivery tag.
fn voice_mark(mark: &str, malayalam: bool) -> bool {
    let pause = matches!(tts::script::parse(mark).as_slice(), [tts::script::Segment::Pause { .. }]);
    pause || (malayalam && matches!(tts::script::parse_marked(mark).as_slice(), [tts::script::Mark::Tag(_)]))
}

/// Every bracket of `text` (a paragraph without its hints) is a voice marker, else an error.
fn check_brackets(text: &str, at: usize, malayalam: bool) -> std::result::Result<(), String> {
    let spans = tts::script::mark_spans(text);
    let mut covered = vec![false; text.len()];
    for &(a, b) in &spans {
        let mark = text.get(a..b).unwrap_or("");
        if !voice_mark(mark, malayalam) {
            let tags = if malayalam { "" } else { " (delivery tags are for Malayalam scripts)" };
            return Err(format!("line {at}: `{}` is not read here{tags}: use [shot: …] for the picture or [pause 1s] for a pause", clip_chars(mark, 40)));
        }
        for c in covered.iter_mut().take(b).skip(a) {
            *c = true;
        }
    }
    for (i, c) in text.char_indices() {
        if matches!(c, '[' | ']') && !covered.get(i).copied().unwrap_or(false) {
            let from = text.get(i..).unwrap_or("");
            return Err(format!(
                "line {at}: unknown mark `{}`: use [shot: …] for what the picture shows or [pause 1s] for a pause; other brackets would be read aloud",
                clip_chars(from, 40)
            ));
        }
    }
    Ok(())
}

/// Read a script in its markup (see the module docs).
pub(crate) fn parse(text: &str, malayalam: bool) -> std::result::Result<Vec<Chapter>, String> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(format!("the script is {} bytes; at most {MAX_TEXT_BYTES}", text.len()));
    }
    if text.contains('\0') {
        return Err("the script holds a NUL character".into());
    }
    let mut chapters: Vec<Chapter> = Vec::new();
    // (text, hints, first line) of the paragraph being read
    let mut open: Option<(String, Vec<String>, usize)> = None;
    // hints on lines of their own before their paragraph: (hints, line)
    let mut pending: Option<(Vec<String>, usize)> = None;
    fn close(chapters: &mut Vec<Chapter>, open: &mut Option<(String, Vec<String>, usize)>, malayalam: bool) -> std::result::Result<(), String> {
        let Some((text, hints, line)) = open.take() else { return Ok(()) };
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        check_brackets(&text, line, malayalam)?;
        if !tts::script::strip_marks(&text).chars().any(char::is_alphanumeric) {
            return Err(format!("line {line}: the paragraph has no words, only marks; give it words or take it out"));
        }
        if hints.len() > MAX_HINTS {
            return Err(format!("line {line}: the paragraph has {} [shot: …] hints; at most {MAX_HINTS}", hints.len()));
        }
        if chapters.is_empty() {
            chapters.push(Chapter { title: String::new(), line, paragraphs: Vec::new() });
        }
        if let Some(c) = chapters.last_mut() {
            c.paragraphs.push(Paragraph { text, hints, line });
        }
        Ok(())
    }
    let orphan = |p: &Option<(Vec<String>, usize)>| -> std::result::Result<(), String> {
        match p {
            Some((_, line)) => {
                Err(format!("line {line}: a [shot: …] hint with no paragraph after it in its chapter; write the paragraph it shows, or take it out"))
            }
            None => Ok(()),
        }
    };
    for (n, raw) in text.lines().enumerate() {
        let at = n + 1;
        let line = raw.trim();
        // `# Title` is a heading; `#tag` is text
        let heading = line.strip_prefix('#').map(|h| h.trim_start_matches('#')).filter(|t| t.is_empty() || t.starts_with(char::is_whitespace));
        if let Some(title) = heading {
            close(&mut chapters, &mut open, malayalam)?;
            orphan(&pending)?;
            let title = title.trim();
            if title.is_empty() {
                return Err(format!("line {at}: a chapter heading needs a title (`# What a cloud is`)"));
            }
            if title.chars().count() > MAX_TITLE_CHARS {
                return Err(format!("line {at}: the chapter title is {} characters; at most {MAX_TITLE_CHARS}", title.chars().count()));
            }
            if title.contains('[') || title.contains(']') {
                return Err(format!("line {at}: a chapter title takes no marks or hints; put the hint under it"));
            }
            if let Some(c) = chapters.last()
                && c.paragraphs.is_empty()
            {
                return Err(empty_chapter(chapters.len(), c));
            }
            if chapters.len() >= MAX_CHAPTERS {
                return Err(format!("line {at}: more than {MAX_CHAPTERS} chapters"));
            }
            chapters.push(Chapter { title: title.to_string(), line: at, paragraphs: Vec::new() });
            continue;
        }
        if line.is_empty() {
            close(&mut chapters, &mut open, malayalam)?;
            continue;
        }
        let (rest, hints) = take_hints(line, at)?;
        let words = !rest.trim().is_empty();
        match (&mut open, words) {
            (Some((t, h, _)), _) => {
                if words {
                    t.push(' ');
                    t.push_str(rest.trim());
                }
                h.extend(hints);
            }
            (None, true) => {
                let mut h = pending.take().map(|(h, _)| h).unwrap_or_default();
                h.extend(hints);
                open = Some((rest.trim().to_string(), h, at));
            }
            (None, false) => match &mut pending {
                Some((h, _)) => h.extend(hints),
                None => pending = Some((hints, at)),
            },
        }
        if chapters.iter().map(|c| c.paragraphs.len()).sum::<usize>() > MAX_SHOTS {
            return Err(format!("line {at}: more than {MAX_SHOTS} paragraphs"));
        }
    }
    close(&mut chapters, &mut open, malayalam)?;
    orphan(&pending)?;
    match chapters.last() {
        None => return Err("the script says nothing: write paragraphs for the voice, under `# Chapter` headings".into()),
        Some(c) if c.paragraphs.is_empty() => return Err(empty_chapter(chapters.len(), c)),
        Some(_) => {}
    }
    Ok(chapters)
}

fn empty_chapter(n: usize, c: &Chapter) -> String {
    format!("chapter {n} (“{}”, line {}) has no paragraphs: write what the narrator says under it, or take the heading out", c.title, c.line)
}

const ABBREVIATIONS: &[&str] =
    &["mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie", "no", "fig", "approx", "jr", "sr", "prof", "inc", "ltd", "co", "mt", "ca"];

/// The sentences of a paragraph: split after `.`, `!`, `?` or `…` (and closing quotes) before
/// white space and a word that does not start in lower case; not after an abbreviation or an
/// initial.
pub(crate) fn sentences(text: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let terminal = |c: char| matches!(c, '.' | '!' | '?' | '\u{2026}' | '\u{3002}' | '\u{FF01}' | '\u{FF1F}');
    let closer = |c: char| terminal(c) || matches!(c, '"' | '\'' | '\u{201D}' | '\u{2019}' | ')');
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut k = 0usize;
    while let Some(&(b, c)) = chars.get(k) {
        if !terminal(c) {
            k += 1;
            continue;
        }
        let mut j = k + 1;
        while chars.get(j).is_some_and(|(_, d)| closer(*d)) {
            j += 1;
        }
        let spaced = chars.get(j).is_none_or(|(_, d)| d.is_whitespace());
        let mut m = j;
        while chars.get(m).is_some_and(|(_, d)| d.is_whitespace()) {
            m += 1;
        }
        let lower_next = chars.get(m).is_some_and(|(_, d)| d.is_lowercase());
        let word: String =
            text.get(start..b).unwrap_or("").chars().rev().take_while(|c| c.is_alphabetic() || *c == '.').collect::<Vec<_>>().into_iter().rev().collect();
        let word = word.trim_matches('.').replace('.', "").to_lowercase();
        let initial = c == '.' && word.chars().count() == 1 && text.get(..b).and_then(|h| h.chars().last()).is_some_and(char::is_uppercase);
        let abbrev = c == '.' && (ABBREVIATIONS.contains(&word.as_str()) || initial);
        if spaced && !lower_next && !abbrev {
            let end = chars.get(j).map_or(text.len(), |(e, _)| *e);
            let piece = text.get(start..end).unwrap_or("").trim();
            if !piece.is_empty() {
                out.push(piece.to_string());
            }
            start = end;
        }
        k = j;
    }
    let piece = text.get(start..).unwrap_or("").trim();
    if !piece.is_empty() {
        out.push(piece.to_string());
    }
    // a piece with no words (a lone pause marker) joins the sentence before it
    let mut merged: Vec<String> = Vec::new();
    for p in out {
        let words = tts::script::strip_marks(&p).chars().any(char::is_alphanumeric);
        match merged.last_mut() {
            Some(last) if !words => {
                last.push(' ');
                last.push_str(&p);
            }
            _ => merged.push(p),
        }
    }
    if let (Some(first), true) = (merged.first(), merged.len() > 1)
        && !tts::script::strip_marks(first).chars().any(char::is_alphanumeric)
    {
        // a leading lone marker joins the next sentence
        let lead = merged.remove(0);
        if let Some(next) = merged.first_mut() {
            *next = format!("{lead} {next}");
        }
    }
    merged
}

// -------------------------------------------------------------------------------------------
// subtitles
// -------------------------------------------------------------------------------------------

/// How long a piece of text takes to say, relatively: its letters and digits, and one a word.
fn weight(text: &str) -> f64 {
    let letters = text.chars().filter(|c| c.is_alphanumeric()).count();
    (letters + text.split_whitespace().count()).max(1) as f64
}

/// `text` cut into blocks of at most `max` characters at word boundaries, a block ending at a
/// comma, semicolon or colon when it is past 60 % full.
pub(crate) fn blocks(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        let len = cur.chars().count();
        if !cur.is_empty() && len + 1 + w.chars().count() > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
        let full = cur.chars().count() as f64 >= 0.6 * max as f64;
        if full && w.ends_with([',', ';', ':']) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A block in two balanced lines when it is longer than half of `max`.
fn lines(block: &str, max: usize) -> String {
    let n = block.chars().count();
    if n <= max / 2 {
        return block.to_string();
    }
    let mid = n / 2;
    let spaces: Vec<usize> = block.char_indices().filter(|(_, c)| *c == ' ').map(|(i, _)| i).collect();
    let char_at = |byte: usize| block.get(..byte).map_or(0, |h| h.chars().count());
    match spaces.iter().min_by_key(|b| char_at(**b).abs_diff(mid)) {
        Some(&b) => format!("{}\n{}", block.get(..b).unwrap_or(""), block.get(b + 1..).unwrap_or("")),
        None => block.to_string(),
    }
}

// -------------------------------------------------------------------------------------------
// the timeline
// -------------------------------------------------------------------------------------------

/// A spoken sentence.
#[derive(Clone)]
struct Spoken {
    /// What the voice reads (markers kept) and what subtitles show.
    text: String,
    shown: String,
    /// The audio (none when estimated).
    audio: Option<Arc<tts::Audio>>,
    duration: Tick,
    cached: bool,
    check: Value,
}

/// A sentence placed in its shot: offset from the shot's start.
#[derive(Clone)]
struct Placed {
    sentence: usize,
    from: Tick,
    to: Tick,
}

/// A subtitle block with its timeline times and word times.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub text: String,
    pub start: Tick,
    pub end: Tick,
    pub shot: usize,
    pub words: Vec<(String, Tick, Tick)>,
}

#[derive(Clone)]
struct Shot {
    chapter: usize,
    paragraph: usize,
    part: usize,
    parts: usize,
    hints: Vec<String>,
    sentences: Vec<usize>,
    start: Tick,
    end: Tick,
    speech: Tick,
    placed: Vec<Placed>,
}

/// The gaps, seconds.
#[derive(Clone, Copy, Debug)]
struct Gaps {
    sentence: f64,
    shot: f64,
    chapter: f64,
    tail: f64,
}

fn seconds_p(v: Option<&Value>, key: &str, default: f64, (lo, hi): (f64, f64)) -> Result<f64> {
    match v {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x.as_f64().filter(|s| s.is_finite() && (lo..=hi).contains(s)).ok_or_else(|| bad(CMD, format!("`{key}` is a number from {lo} to {hi}"))),
    }
}

fn gaps_p(p: &Value) -> Result<Gaps> {
    let Some(g) = p.get("gaps").filter(|v| !v.is_null()) else {
        return Ok(Gaps { sentence: SENTENCE_GAP, shot: SHOT_GAP, chapter: CHAPTER_GAP, tail: TAIL });
    };
    let o = g.as_object().ok_or_else(|| bad(CMD, "`gaps` is {sentence?, shot?, chapter?, tail?} in seconds"))?;
    if let Some(k) = o.keys().find(|k| !matches!(k.as_str(), "sentence" | "shot" | "chapter" | "tail")) {
        return Err(bad(CMD, format!("`gaps.{}` is not a gap: sentence, shot, chapter or tail", clip_chars(k, 40))));
    }
    let r = (0.0, MAX_GAP);
    Ok(Gaps {
        sentence: seconds_p(o.get("sentence"), "gaps.sentence", SENTENCE_GAP, r)?,
        shot: seconds_p(o.get("shot"), "gaps.shot", SHOT_GAP, r)?,
        chapter: seconds_p(o.get("chapter"), "gaps.chapter", CHAPTER_GAP, r)?,
        tail: seconds_p(o.get("tail"), "gaps.tail", TAIL, r)?,
    })
}

/// `t` rounded up to the frame grid.
fn frame_up(rate: FrameRate, t: Tick) -> Tick {
    let down = rate.snap(t);
    if down >= t { down } else { rate.tick_of(rate.frame_at(t).saturating_add(1)) }
}

/// How long the estimate speaks `text`: its words at [`ESTIMATE_WPS`] and its pauses.
fn estimate(text: &str) -> Tick {
    let mut secs = 0.0;
    for seg in tts::script::parse(text) {
        match seg {
            tts::script::Segment::Speech(t) => secs += t.split_whitespace().count() as f64 / ESTIMATE_WPS,
            tts::script::Segment::Pause { millis } => secs += f64::from(millis) / 1000.0,
        }
    }
    Tick::from_seconds_f64(secs.max(ESTIMATE_MIN_SECONDS))
}

/// The voice's settings in `p` (no text), the language decided once for the whole script.
fn voice_params(p: &Value, malayalam: bool) -> Value {
    let mut v = Map::new();
    for k in VOICE_KEYS {
        if let Some(x) = p.get(*k).filter(|x| !x.is_null()) {
            v.insert((*k).to_string(), x.clone());
        }
    }
    if !v.contains_key("language") {
        v.insert("language".into(), json!(if malayalam { tts::malayalam::TAG } else { "en-US" }));
    }
    Value::Object(v)
}

/// Speak `text` with the voice (or estimate it).
fn speak(s: &Session, voice: &Value, text: &str, estimated: bool) -> Result<Spoken> {
    let shown = crate::narration::shown_text(text);
    if estimated {
        return Ok(Spoken { text: text.to_string(), shown, audio: None, duration: estimate(text), cached: false, check: Value::Null });
    }
    let mut q = voice.clone();
    q["text"] = json!(text);
    let n = crate::narration::settings(CMD, &q, None)?;
    let (audio, cached, check) = crate::narration::synthesize(s, CMD, &q, &n)?;
    let rate = audio.sample_rate.max(1);
    let duration = Tick::from_units(audio.samples.len() as i64, i64::from(rate));
    Ok(Spoken { text: text.to_string(), shown, audio: Some(audio), duration, cached, check })
}

/// The script's text: `text`, or the file at `path`.
fn script_text(s: &Session, p: &Value) -> Result<String> {
    match (p.get("text").filter(|v| !v.is_null()), p.get("path").filter(|v| !v.is_null())) {
        (Some(_), Some(_)) => Err(bad(CMD, "give `text` or `path`, not both")),
        (Some(t), None) => {
            let t = t.as_str().ok_or_else(|| bad(CMD, "`text` is the script, a string"))?;
            if t.len() > MAX_TEXT_BYTES {
                return Err(bad(CMD, format!("the script is {} bytes; at most {MAX_TEXT_BYTES}", t.len())));
            }
            Ok(t.to_string())
        }
        (None, Some(v)) => {
            let path = v.as_str().map(str::trim).filter(|x| !x.is_empty() && x.len() <= 4096).ok_or_else(|| bad(CMD, "`path` is a script file"))?;
            let path = crate::export_tools::expand_home(path);
            let bytes = s.services.read_file_bounded(&path, MAX_TEXT_BYTES as u64).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
            String::from_utf8(bytes).map_err(|_| bad(CMD, format!("{path} is not UTF-8 text")))
        }
        (None, None) => Err(bad(CMD, "need the script: `text` (or a file's `path`)")),
    }
}

/// Where and how to place the timeline.
#[derive(Clone, Debug)]
struct PlaceOpts {
    new_sequence: bool,
    name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f64>,
    at: Tick,
    track: Option<Value>,
    narration: bool,
    captions: bool,
    markers: bool,
    raw: Value,
}

fn flag(o: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, format!("`place.{key}` is true or false"))),
    }
}

fn place_p(p: &Value) -> Result<Option<PlaceOpts>> {
    let v = match p.get("place") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(None),
        Some(Value::Bool(true)) => json!({}),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(_) => return Err(bad(CMD, "`place` is true, or {sequence?, name?, width?, height?, fps?, at?, track?, narration?, captions?, markers?, dir?}")),
    };
    let o = v.as_object().cloned().unwrap_or_default();
    const PLACE_KEYS: &[&str] = &["sequence", "name", "width", "height", "fps", "at", "seconds", "track", "narration", "captions", "markers", "dir"];
    if let Some(k) = o.keys().find(|k| !PLACE_KEYS.contains(&k.as_str())) {
        return Err(bad(CMD, format!("`place.{}` is not a placement setting ({})", clip_chars(k, 40), PLACE_KEYS.join(", "))));
    }
    let new_sequence = match o.get("sequence") {
        None | Some(Value::Null) => true,
        Some(Value::String(x)) if x == "new" => true,
        Some(Value::String(x)) if x == "active" => false,
        Some(_) => return Err(bad(CMD, "`place.sequence` is \"new\" or \"active\"")),
    };
    let side = |k: &str| -> Result<Option<u32>> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                v.as_u64().filter(|n| (16..=16_384).contains(n)).map(|n| Some(n as u32)).ok_or_else(|| bad(CMD, format!("`place.{k}` is 16 to 16384 pixels")))
            }
        }
    };
    let name = match o.get("name") {
        None | Some(Value::Null) => None,
        Some(Value::String(n)) if !n.trim().is_empty() && n.chars().count() <= 200 => Some(n.trim().to_string()),
        Some(_) => return Err(bad(CMD, "`place.name` is a sequence name of at most 200 characters")),
    };
    let fps = match o.get("fps") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().filter(|f| f.is_finite() && (1.0..=240.0).contains(f)).ok_or_else(|| bad(CMD, "`place.fps` is 1 to 240"))?),
    };
    let at = match (o.get("at"), o.get("seconds")) {
        (Some(v), None) if !v.is_null() => Tick(
            v.as_i64()
                .filter(|t| (0..=Tick::from_seconds_f64(24.0 * 3600.0).0).contains(t))
                .ok_or_else(|| bad(CMD, "`place.at` is ticks from 0 to 24 hours"))?,
        ),
        (None, Some(v)) if !v.is_null() => Tick::from_seconds_f64(
            v.as_f64().filter(|s| s.is_finite() && (0.0..=24.0 * 3600.0).contains(s)).ok_or_else(|| bad(CMD, "`place.seconds` is 0 to 24 hours"))?,
        ),
        (Some(_), Some(_)) => return Err(bad(CMD, "give `place.at` or `place.seconds`, not both")),
        _ => Tick::ZERO,
    };
    if new_sequence && o.get("track").is_some_and(|t| !t.is_null()) {
        return Err(bad(CMD, "`place.track` is for the active sequence (`place.sequence: \"active\"`); a new sequence's narration goes on A1"));
    }
    Ok(Some(PlaceOpts {
        new_sequence,
        name,
        width: side("width")?,
        height: side("height")?,
        fps,
        at,
        track: o.get("track").filter(|t| !t.is_null()).cloned(),
        narration: flag(&o, "narration", true)?,
        captions: flag(&o, "captions", true)?,
        markers: flag(&o, "markers", true)?,
        raw: v,
    }))
}

/// The id of shot `i` (0-based).
pub fn shot_id(i: usize) -> String {
    format!("{SHOT_PREFIX}{:02}", i + 1)
}

/// The shot index a marker name names (`shot-07` → 6).
pub fn shot_index(name: &str) -> Option<usize> {
    name.strip_prefix(SHOT_PREFIX)
        .filter(|d| !d.is_empty() && d.len() <= 6 && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse::<usize>().ok())
        .and_then(|n| n.checked_sub(1))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object with the script (`text` or `path`)"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !KEYS.contains(&k.as_str()) && !VOICE_KEYS.contains(&k.as_str())).cloned()) {
        return Err(bad(
            CMD,
            format!(
                "unknown parameter `{}` (the script, the voice's settings as tts.create takes them, estimate, gaps, maxShotSeconds, captionChars, fps, out, place)",
                clip_chars(&k, 64)
            ),
        ));
    }
    let estimated = match p.get("estimate") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`estimate` is true or false"))?,
    };
    let gaps = gaps_p(p)?;
    let max_shot = seconds_p(p.get("maxShotSeconds"), "maxShotSeconds", MAX_SHOT_SECONDS, SHOT_SECONDS_RANGE)?;
    let caption_chars = match p.get("captionChars") {
        None | Some(Value::Null) => CAPTION_CHARS,
        Some(v) => v
            .as_u64()
            .filter(|n| (CAPTION_CHARS_RANGE.0 as u64..=CAPTION_CHARS_RANGE.1 as u64).contains(n))
            .ok_or_else(|| bad(CMD, format!("`captionChars` is {} to {}", CAPTION_CHARS_RANGE.0, CAPTION_CHARS_RANGE.1)))? as usize,
    };
    let place = place_p(p)?;
    let out = match p.get("out").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(crate::export_tools::expand_home(
            v.as_str().map(str::trim).filter(|o| !o.is_empty() && o.len() <= 4096 && !o.contains('\0')).ok_or_else(|| bad(CMD, "`out` is a folder"))?,
        )),
    };
    let fps_given = match p.get("fps").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_f64().filter(|f| f.is_finite() && (1.0..=240.0).contains(f)).ok_or_else(|| bad(CMD, "`fps` is 1 to 240"))?),
    };
    if place.as_ref().is_some_and(|o| !o.new_sequence) && s.active_sequence().is_none() {
        return Err(bad(CMD, "`place.sequence: \"active\"` needs an open sequence"));
    }
    let text = script_text(s, p)?;
    let malayalam = match p.get("language").and_then(Value::as_str) {
        Some(l) => tts::malayalam::is_tag(l),
        None => tts::malayalam::detect(&text),
    };
    let chapters = parse(&text, malayalam).map_err(|e| bad(CMD, e))?;
    // the frame grid: the placement's, else the asked rate, else the active sequence's, else 30
    let rate = match place.as_ref().and_then(|o| o.fps).or(fps_given) {
        Some(f) => FrameRate::from_f64(f).sane(),
        None => match (&place, s.active_sequence()) {
            (Some(o), Some(q)) if !o.new_sequence => q.settings.frame_rate.sane(),
            (None, Some(q)) => q.settings.frame_rate.sane(),
            _ => FrameRate::FPS_30,
        },
    };
    if rate.frame_duration() <= Tick::ZERO {
        return Err(bad(CMD, "the frame rate is not usable"));
    }
    let size = match (&place, s.active_sequence()) {
        (Some(o), _) if o.new_sequence => (o.width.unwrap_or(1920), o.height.unwrap_or(1080)),
        (_, Some(q)) => (q.settings.width, q.settings.height),
        (Some(o), None) => (o.width.unwrap_or(1920), o.height.unwrap_or(1080)),
        (None, None) => (1920, 1080),
    };
    // ---- every sentence spoken
    let voice = voice_params(p, malayalam);
    let mut spoken: Vec<Spoken> = Vec::new();
    // per paragraph: (chapter, paragraph index in the script, its sentences, its hints)
    let mut paragraphs: Vec<(usize, usize, Vec<usize>, Vec<String>)> = Vec::new();
    // every sentence is split out first, so an over-long script is refused before the voice speaks
    let mut said: Vec<String> = Vec::new();
    for (ci, c) in chapters.iter().enumerate() {
        for para in &c.paragraphs {
            let mut mine = Vec::new();
            for sentence in sentences(&para.text) {
                if said.len() >= MAX_SENTENCES {
                    return Err(bad(CMD, format!("the script has more than {MAX_SENTENCES} sentences; split it into parts")));
                }
                mine.push(said.len());
                said.push(sentence);
            }
            paragraphs.push((ci, paragraphs.len(), mine, para.hints.clone()));
        }
    }
    for sentence in &said {
        spoken.push(speak(s, &voice, sentence, estimated)?);
    }
    // one sample rate for the narration files
    let sample_rate = spoken.iter().find_map(|x| x.audio.as_ref().map(|a| a.sample_rate.max(1)));
    if let Some(r) = sample_rate
        && spoken.iter().any(|x| x.audio.as_ref().is_some_and(|a| a.sample_rate.max(1) != r))
    {
        return Err(bad(CMD, "the voice spoke the sentences at different sample rates"));
    }
    let gap_of = |secs: f64| -> Tick {
        // the sentence gap as whole samples, so the narration file and the times agree
        match sample_rate {
            Some(r) => Tick::from_units((secs * f64::from(r)).round() as i64, i64::from(r)),
            None => Tick::from_seconds_f64(secs),
        }
    };
    let sentence_gap = gap_of(gaps.sentence);
    // ---- shots: a paragraph, or its sentence groups when it speaks longer than max_shot
    let mut shots: Vec<Shot> = Vec::new();
    let limit = Tick::from_seconds_f64(max_shot);
    for (ci, pidx, mine, hints) in &paragraphs {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_len = Tick::ZERO;
        for &si in mine {
            let d = spoken.get(si).map_or(Tick::ZERO, |x| x.duration);
            let with = if cur.is_empty() { d } else { cur_len + sentence_gap + d };
            if !cur.is_empty() && with > limit {
                groups.push(std::mem::take(&mut cur));
                cur_len = d;
            } else {
                cur_len = with;
            }
            cur.push(si);
        }
        if !cur.is_empty() {
            groups.push(cur);
        }
        let parts = groups.len();
        for (gi, g) in groups.into_iter().enumerate() {
            if shots.len() >= MAX_SHOTS {
                return Err(bad(CMD, format!("the script makes more than {MAX_SHOTS} shots; split it into parts or raise maxShotSeconds")));
            }
            shots.push(Shot {
                chapter: *ci,
                paragraph: *pidx,
                part: gi + 1,
                parts,
                hints: hints.clone(),
                sentences: g,
                start: Tick::ZERO,
                end: Tick::ZERO,
                speech: Tick::ZERO,
                placed: Vec::new(),
            });
        }
    }
    // ---- the layout
    let mut t = Tick::ZERO;
    let chapter_of: Vec<usize> = shots.iter().map(|x| x.chapter).collect();
    for (i, shot) in shots.iter_mut().enumerate() {
        shot.start = t;
        let mut off = Tick::ZERO;
        for (k, &si) in shot.sentences.iter().enumerate() {
            if k > 0 {
                off += sentence_gap;
            }
            let d = spoken.get(si).map_or(Tick::ZERO, |x| x.duration);
            shot.placed.push(Placed { sentence: si, from: off, to: off + d });
            off += d;
        }
        shot.speech = off;
        let after = match chapter_of.get(i + 1) {
            None => gaps.tail,
            Some(next) if *next != shot.chapter => gaps.chapter,
            Some(_) => gaps.shot,
        };
        let end = frame_up(rate, shot.start + off + Tick::from_seconds_f64(after)).max(shot.start + rate.frame_duration());
        shot.end = end;
        t = end;
        if t.seconds() > MAX_TOTAL_SECONDS {
            return Err(bad(CMD, format!("the timeline runs past {} hours; split the script into parts", MAX_TOTAL_SECONDS / 3600.0)));
        }
    }
    let total = t;
    // ---- subtitles: blocks of each sentence, timed by their letters, held to their reading time
    let mut subs: Vec<Block> = Vec::new();
    for (si_shot, shot) in shots.iter().enumerate() {
        for pl in &shot.placed {
            let Some(sp) = spoken.get(pl.sentence) else { continue };
            let bl = blocks(&sp.shown, caption_chars);
            let total_w: f64 = bl.iter().map(|b| weight(b)).sum::<f64>().max(1.0);
            let span = (pl.to - pl.from).0.max(0) as f64;
            let mut acc = 0.0;
            for b in bl {
                let w = weight(&b);
                let a = shot.start + pl.from + Tick((span * acc / total_w) as i64);
                acc += w;
                let e = shot.start + pl.from + Tick((span * acc / total_w) as i64);
                // words spread by letters inside the block
                let words: Vec<&str> = b.split_whitespace().collect();
                let ww: f64 = words.iter().map(|x| weight(x)).sum::<f64>().max(1.0);
                let len = (e - a).0.max(0) as f64;
                let mut wacc = 0.0;
                let mut timed = Vec::with_capacity(words.len());
                for x in words {
                    let ws = a + Tick((len * wacc / ww) as i64);
                    wacc += weight(x);
                    let we = a + Tick((len * wacc / ww) as i64);
                    timed.push((x.to_string(), ws, we));
                }
                subs.push(Block { text: lines(&b, caption_chars), start: a, end: e, shot: si_shot, words: timed });
            }
        }
    }
    // on the frame grid, held to the reading time when there is room, never overlapping
    let frame = rate.frame_duration();
    let mut timed: Vec<Block> = Vec::with_capacity(subs.len());
    let mut prev_end = Tick::ZERO;
    for (i, b) in subs.iter().enumerate() {
        let next = subs.get(i + 1).map_or(total, |x| rate.snap_nearest(x.start));
        let start = rate.snap_nearest(b.start).max(prev_end);
        if start + frame > total {
            break;
        }
        let words = b.text.split_whitespace().count() as f64;
        let read = start + Tick::from_seconds_f64(words / READ_WPS + READ_FLOOR);
        let end = rate.snap_nearest(b.end.max(read)).min(next.max(start + frame)).max(start + frame).min(total);
        prev_end = end;
        timed.push(Block { start, end, ..b.clone() });
    }
    let subs = timed;
    // ---- what comes back
    let shots_json: Vec<Value> = shots
        .iter()
        .enumerate()
        .map(|(i, sh)| {
            let ch = chapters.get(sh.chapter);
            let shown: Vec<String> = sh.sentences.iter().filter_map(|si| spoken.get(*si).map(|x| x.shown.clone())).collect();
            let said: Vec<String> = sh.sentences.iter().filter_map(|si| spoken.get(*si).map(|x| x.text.clone())).collect();
            let d = sh.end - sh.start;
            json!({
                "id": shot_id(i),
                "index": i,
                "chapter": sh.chapter,
                "chapterTitle": ch.map(|c| c.title.clone()).unwrap_or_default(),
                "paragraph": sh.paragraph,
                "part": sh.part,
                "parts": sh.parts,
                "start": sh.start.0,
                "end": sh.end.0,
                "duration": d.0,
                "seconds": r3(d.seconds()),
                "frames": rate.frame_at(sh.end).saturating_sub(rate.frame_at(sh.start)),
                "speech": sh.speech.0,
                "text": shown.join(" "),
                "spoken": said.join(" "),
                "hints": sh.hints,
                "sentences": sh.placed.iter().map(|pl| json!({
                    "text": spoken.get(pl.sentence).map(|x| x.shown.clone()).unwrap_or_default(),
                    "start": (sh.start + pl.from).0,
                    "end": (sh.start + pl.to).0,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let chapters_json: Vec<Value> = chapters
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let mine: Vec<usize> = shots.iter().enumerate().filter(|(_, x)| x.chapter == ci).map(|(i, _)| i).collect();
            let start = mine.first().and_then(|i| shots.get(*i)).map_or(Tick::ZERO, |x| x.start);
            let end = mine.last().and_then(|i| shots.get(*i)).map_or(Tick::ZERO, |x| x.end);
            json!({"index": ci, "title": c.title, "start": start.0, "end": end.0, "seconds": r3(start.seconds()), "length": r3((end - start).seconds()), "shots": [mine.first(), mine.last()], "line": c.line})
        })
        .collect();
    let subs_json: Vec<Value> = subs
        .iter()
        .map(|b| {
            json!({"text": b.text, "start": b.start.0, "end": b.end.0, "shot": b.shot,
                "words": b.words.iter().map(|(w, a, e)| json!({"text": w, "start": a.0, "end": e.0})).collect::<Vec<_>>()})
        })
        .collect();
    let mut markers: Vec<(Tick, MarkerKind, String, String)> = Vec::new();
    for (ci, c) in chapters.iter().enumerate() {
        if let Some(first) = shots.iter().find(|x| x.chapter == ci) {
            let name = if c.title.is_empty() { format!("Chapter {}", ci + 1) } else { c.title.clone() };
            markers.push((first.start, MarkerKind::Chapter, name, String::new()));
        }
    }
    for (i, sh) in shots.iter().enumerate() {
        markers.push((sh.start, MarkerKind::Comment, shot_id(i), sh.hints.first().cloned().unwrap_or_default()));
    }
    let markers_json: Vec<Value> = markers
        .iter()
        .map(|(t, k, n, c)| json!({"start": t.0, "kind": if *k == MarkerKind::Chapter { "chapter" } else { "shot" }, "name": n, "comment": c}))
        .collect();
    let speech: Tick = spoken.iter().fold(Tick::ZERO, |a, x| a + x.duration);
    let cached = spoken.iter().filter(|x| x.cached).count();
    let checks: Vec<&Value> = spoken.iter().map(|x| &x.check).filter(|c| !c.is_null()).collect();
    let unclear = checks.iter().filter(|c| c["status"] == "unclear").count();
    let voice_name = voice.get("voice").and_then(Value::as_str).map_or_else(|| tts::default_voice_id().to_string(), str::to_string);
    let mut result = json!({
        "duration": total.0,
        "seconds": r3(total.seconds()),
        "fps": r3(rate.as_f64()),
        "size": [size.0, size.1],
        "frames": rate.frame_at(total),
        "voice": if estimated { Value::Null } else { json!(voice_name) },
        "language": voice.get("language").cloned().unwrap_or(Value::Null),
        "estimated": estimated,
        "chapters": chapters_json,
        "shots": shots_json,
        "subtitles": subs_json,
        "markers": markers_json,
        "narration": {
            "speech": speech.0,
            "speechSeconds": r3(speech.seconds()),
            "gaps": (total - speech).0,
            "sentences": spoken.len(),
            "cached": cached,
            "checked": checks.len(),
            "unclear": unclear,
            "timing": if estimated { "estimated from the words (2.5 a second)" } else { "sentence times from the voice; word times spread by letters" },
        },
        "counts": {"chapters": chapters.len(), "shots": shots.len(), "sentences": spoken.len(), "subtitles": subs.len(), "hints": shots.iter().map(|x| x.hints.len()).sum::<usize>()},
        "settings": {"gaps": {"sentence": gaps.sentence, "shot": gaps.shot, "chapter": gaps.chapter, "tail": gaps.tail}, "maxShotSeconds": max_shot, "captionChars": caption_chars},
        "placed": Value::Null,
        "files": Value::Null,
    });
    if unclear > 0 {
        result["warnings"] = json!([format!("{unclear} sentences were not heard clearly by the strict check; listen to them before building")]);
    }
    // ---- the files for the compositions
    if let Some(dir) = &out {
        result["files"] = write_files(s, dir, &result, &shots_json, &subs_json, rate, size)?;
    }
    // ---- on the timeline
    if let Some(opts) = place {
        let placed = s.edit_group("Script Timeline", |s| {
            place_timeline(s, &opts, &chapters, &shots, &spoken, &subs, &markers, sentence_gap, total, size, rate, &voice)
        })?;
        result["placed"] = placed;
    }
    Ok(result)
}

/// `timeline.json` and one `shot-NN.json` a shot into `dir` (a new or empty folder).
fn write_files(s: &Session, dir: &str, result: &Value, shots: &[Value], subs: &[Value], rate: FrameRate, size: (u32, u32)) -> Result<Value> {
    match s.services.list_dir(dir) {
        Some(Ok(names)) if !names.is_empty() => return Err(bad(CMD, format!("{dir} is not empty: choose a new or empty folder for the shot files"))),
        Some(Err(e)) if e.kind() != std::io::ErrorKind::NotFound => return Err(bad(CMD, format!("{dir}: {e}"))),
        _ => {}
    }
    let path_of = |name: &str| std::path::Path::new(dir).join(name).display().to_string();
    let mut paths = Vec::new();
    for sh in shots {
        paths.push(path_of(&format!("{}.json", sh["id"].as_str().unwrap_or("shot"))));
    }
    let timeline = path_of("timeline.json");
    for path in paths.iter().chain(std::iter::once(&timeline)) {
        crate::output_safety::protect(s, &s.project, path, CMD)?;
    }
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(dir).map_err(|e| bad(CMD, format!("{dir}: {e}")))?;
    }
    let secs = |t: i64| r3(Tick(t).seconds());
    for (sh, path) in shots.iter().zip(&paths) {
        let start = sh["start"].as_i64().unwrap_or(0);
        let rel = |t: i64| t.saturating_sub(start);
        let frame = |t: i64| rate.frame_at(Tick(rel(t)));
        let index = sh["index"].as_u64().unwrap_or(0) as usize;
        let mine: Vec<Value> = subs
            .iter()
            .filter(|b| b["shot"].as_u64() == Some(index as u64))
            .map(|b| {
                let (a, e) = (b["start"].as_i64().unwrap_or(0), b["end"].as_i64().unwrap_or(0));
                json!({"text": b["text"], "from": frame(a), "to": frame(e), "seconds": [secs(rel(a)), secs(rel(e))]})
            })
            .collect();
        let sentences: Vec<Value> = sh["sentences"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|x| {
                        let (a, e) = (x["start"].as_i64().unwrap_or(0), x["end"].as_i64().unwrap_or(0));
                        json!({"text": x["text"], "from": frame(a), "to": frame(e), "seconds": [secs(rel(a)), secs(rel(e))]})
                    })
                    .collect()
            })
            .unwrap_or_default();
        let doc = json!({
            "format": SHOT_FORMAT, "version": 1,
            "id": sh["id"], "index": sh["index"],
            "chapter": {"index": sh["chapter"], "title": sh["chapterTitle"]},
            "part": sh["part"], "parts": sh["parts"],
            "text": sh["text"], "spoken": sh["spoken"], "hints": sh["hints"],
            "fps": r3(rate.as_f64()), "size": [size.0, size.1],
            "frames": sh["frames"], "seconds": sh["seconds"],
            "start": sh["start"], "duration": sh["duration"],
            "sentences": sentences, "subtitles": mine,
        });
        let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| bad(CMD, e.to_string()))?;
        s.services.write_file(path, &bytes).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
    }
    let mut whole = result.clone();
    if let Some(o) = whole.as_object_mut() {
        o.insert("format".into(), json!(TIMELINE_FORMAT));
        o.insert("version".into(), json!(1));
        o.remove("placed");
        o.remove("files");
    }
    let bytes = serde_json::to_vec_pretty(&whole).map_err(|e| bad(CMD, e.to_string()))?;
    s.services.write_file(&timeline, &bytes).map_err(|e| bad(CMD, format!("{timeline}: {e}")))?;
    Ok(json!({"dir": dir, "timeline": timeline, "shots": paths}))
}

/// The audio track for narration of `range` in the active sequence: `track` (it must be free),
/// else the first free audio track, else a new one.
fn narration_track(s: &mut Session, track: Option<&Value>, range: TimeRange) -> Result<TrackId> {
    let free = |s: &Session, id: TrackId| {
        s.active_sequence()
            .and_then(|q| q.audio_tracks.iter().find(|t| t.id == id))
            .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
    };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    if let Some(v) = track {
        let id =
            crate::mixer::strip_ref(seq, v).filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id)).ok_or_else(|| bad(CMD, format!("no audio track {v}")))?;
        if !free(s, id) {
            return Err(bad(CMD, format!("audio track {v} is locked or has clips where the narration goes")));
        }
        return Ok(id);
    }
    let ids: Vec<TrackId> = seq.audio_tracks.iter().map(|t| t.id).collect();
    if let Some(id) = ids.iter().copied().find(|id| free(s, *id)) {
        return Ok(id);
    }
    s.execute("sequence.addTracks", json!({"video": 0, "audio": 1, "audioAfter": "last"}))?;
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| !ids.contains(id)))
        .ok_or_else(|| EngineError::Other("could not add an audio track".into()))
}

#[allow(clippy::too_many_arguments)]
fn place_timeline(
    s: &mut Session,
    opts: &PlaceOpts,
    chapters: &[Chapter],
    shots: &[Shot],
    spoken: &[Spoken],
    subs: &[Block],
    markers: &[(Tick, MarkerKind, String, String)],
    sentence_gap: Tick,
    total: Tick,
    size: (u32, u32),
    rate: FrameRate,
    voice: &Value,
) -> Result<Value> {
    if opts.new_sequence {
        let name = opts.name.clone().unwrap_or_else(|| chapters.iter().map(|c| c.title.clone()).find(|t| !t.is_empty()).unwrap_or_else(|| "Explainer".into()));
        s.execute("file.newSequence", json!({"name": name, "width": size.0, "height": size.1, "fps": rate.as_f64(), "video": 3, "audio": 2}))?;
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let at = opts.at;
    let mut out = json!({"sequence": seq_id.0, "at": at.0});
    // ---- the narration: one file and one clip a shot
    if opts.narration {
        let any_audio = spoken.iter().any(|x| x.audio.is_some());
        if !any_audio {
            out["narration"] = json!({"skipped": "an estimate has no voice: run without `estimate` to place the narration"});
        } else {
            let track = narration_track(s, opts.track.as_ref(), TimeRange::new(at, total))?;
            let mut files = Vec::new();
            let mut made: Vec<(ItemId, Tick, Tick, filmcraft_project::Narration)> = Vec::new();
            for sh in shots {
                let mut samples: Vec<f32> = Vec::new();
                let mut rate_hz = tts::SAMPLE_RATE;
                for (k, pl) in sh.placed.iter().enumerate() {
                    let Some(a) = spoken.get(pl.sentence).and_then(|x| x.audio.as_ref()) else { continue };
                    rate_hz = a.sample_rate.max(1);
                    if k > 0 {
                        let gap = sentence_gap.to_units_floor(i64::from(rate_hz)).max(0) as usize;
                        samples.resize(samples.len().saturating_add(gap), 0.0);
                    }
                    samples.extend_from_slice(&a.samples);
                }
                if samples.is_empty() {
                    continue;
                }
                let text: Vec<&str> = sh.sentences.iter().filter_map(|si| spoken.get(*si).map(|x| x.text.as_str())).collect();
                let mut q = voice.clone();
                q["text"] = json!(text.join(" "));
                let mut n = crate::narration::settings(CMD, &q, None)?;
                let dur = Tick::from_units(samples.len() as i64, i64::from(rate_hz));
                n.speech_duration = dur;
                let (item, path) = crate::narration::write_and_import(s, &opts.raw, &samples, rate_hz, None)?;
                files.push(path);
                made.push((item, at + sh.start, dur, n));
            }
            let seq_rate = s.project.sequence(seq_id).map(|q| q.settings.frame_rate).unwrap_or_default();
            let clips = s.edit("Script Narration", |pr, st| {
                let mut ids = Vec::new();
                for (item, start, dur, n) in &made {
                    let mut it = pr
                        .make_track_item(*item, TrackKind::Audio, *start, TimeRange::new(Tick::ZERO, *dur), seq_rate)
                        .ok_or(EngineError::Other("a narration file is not importable".into()))?;
                    it.duration = *dur;
                    ids.push(it.id);
                    let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
                    let t = q.audio_tracks.iter_mut().find(|t| t.id == track).ok_or_else(|| bad(CMD, "the narration track was deleted"))?;
                    let pos = t.items.partition_point(|x| x.start <= *start);
                    t.items.insert(pos, it);
                    pr.narrations.insert(*item, n.clone());
                }
                pr.sequence(seq_id).ok_or(EngineError::NoSequence)?.check().map_err(EngineError::Other)?;
                st.selection.clear();
                Ok(ids)
            })?;
            let label = s.project.sequence(seq_id).map_or(Value::Null, |q| json!(crate::mixer::strip_label(q, track)));
            out["narration"] = json!({"track": label, "clips": clips.iter().map(|c: &ClipId| c.0).collect::<Vec<_>>(), "files": files});
        }
    }
    // ---- the subtitles on a caption track of their own
    if opts.captions && !subs.is_empty() {
        let language = if voice.get("language").and_then(Value::as_str).is_some_and(tts::malayalam::is_tag) { "ml" } else { "en" };
        let r = s.execute("captions.newTrack", json!({"name": "Script", "language": language}))?;
        let tid = TrackId(r["track"].as_u64().ok_or_else(|| EngineError::Other("no caption track was made".into()))?);
        let n = s.edit_sequence("Script Captions", |q, ctx, st| {
            let mut n = 0usize;
            for b in subs {
                let (start, end) = (at + b.start, at + b.end);
                if end <= start {
                    continue;
                }
                let id = filmcraft_edit::captions::add_caption(q, tid, start, end - start, &b.text, ctx)?;
                if let Some(c) = q.caption_track_mut(tid).and_then(|t| t.caption_mut(id)) {
                    let len = end - start;
                    let rel = |t: Tick| (t - b.start).max(Tick::ZERO).min(len);
                    c.words = b.words.iter().map(|(w, a, e)| CaptionWord { text: w.clone(), start: rel(*a), end: rel(*e).max(rel(*a)) }).collect();
                }
                n += 1;
            }
            st.caption_selection.clear();
            Ok(n)
        })?;
        out["captions"] = json!({"track": tid.0, "captions": n});
    }
    // ---- chapter and shot markers
    if opts.markers {
        let n = s.edit_sequence("Script Markers", |q, ctx, _| {
            for (t, kind, name, comment) in markers {
                let color = if *kind == MarkerKind::Chapter { Label::Violet } else { Label::Cerulean };
                q.markers.push(Marker {
                    id: MarkerId(ctx.alloc()),
                    start: at + *t,
                    duration: Tick::ZERO,
                    name: name.clone(),
                    comment: comment.clone(),
                    kind: *kind,
                    color,
                });
            }
            q.markers.sort_by_key(|m| m.start);
            Ok(markers.len())
        })?;
        out["markers"] = json!(n);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "explainer_tests.rs"]
mod tests;
