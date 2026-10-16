//! Showcase videos: "every robot I've filmed, in order" — built from a clip analysis
//! ([`crate::analyze`]) into a narrated 16:9 edit with an on-screen counter.
//!
//! 1. [`group`]: clips that show the subject, oldest first; the clips of one robot are one
//!    entry — the same identity from the analysis ([`crate::identity`]), or in older reports the
//!    same (or, the same day, a generic) label — shown in up to three shots.
//! 2. [`write_script`]: the local LLM writes the voice-over — an intro, one or two first-person
//!    sentences per entry, an outro — with the analysis' facts only; any line it does not
//!    deliver gets a plain template line, so the edit never depends on the model behaving.
//! 3. [`build`]: a new sequence (1920×1080 by default) — intro title, then per entry: narration
//!    (any voice, cloned voices included), shots around each clip's best moment (portrait clips
//!    over a blurred, dimmed copy of themselves), a name/date lower third, a transition from a
//!    rotating set of dramatic ones with a whoosh, the year as a title card with an impact
//!    whenever it changes; the counter (`graphics.counter`) across the whole show; a generated
//!    music bed (`audio.generate`) ducked under the narration; an outro card. Then (optionally)
//!    an export with the YouTube preset.
//!
//! Everything goes through engine commands on the [`Executor`], so the same build drives the
//! headless CLI and the live app.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::analyze::ClipFacts;
use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;

fn ticks(seconds: f64) -> i64 {
    (seconds.max(0.0) * TPS).round() as i64
}

/// `x` seconds on the nearest frame boundary: clips and titles that meet are laid out from
/// the same frame-exact times, so rounding never makes one overlap the next by a frame.
pub fn on_frame(x: f64, fps: f64) -> f64 {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    (x.max(0.0) * fps).round() / fps
}

/// What to build.
#[derive(Clone, Debug)]
pub struct Options {
    pub title: String,
    /// Under the title ("2020 – 2026"); default: the years the entries span.
    pub subtitle: Option<String>,
    /// What the counter counts ("robot" → "ROBOT #3").
    pub subject: String,
    /// Narration voice (`clone:<name>` for a cloned voice); `None`: no narration.
    pub voice: Option<String>,
    pub mood: String,
    pub bpm: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// Clips showing the subject in less than this share of their frames are left out.
    pub min_coverage: f64,
    /// Export here when done (YouTube 1080p preset).
    pub export: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            title: "My Robot Showcase".into(),
            subtitle: None,
            subject: "robot".into(),
            voice: None,
            mood: "epic".into(),
            bpm: 100.0,
            width: 1920,
            height: 1080,
            fps: 30.0,
            min_coverage: 0.3,
            export: None,
        }
    }
}

/// One clip shown in an entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Shot {
    pub item: u64,
    pub name: String,
    pub clip_seconds: f64,
    pub best: f64,
    /// Sampled moments (seconds into the clip) that show the subject…
    #[serde(default)]
    pub seen: Vec<f64>,
    /// …and that do not: windows are cut where it is in view.
    #[serde(default)]
    pub unseen: Vec<f64>,
}

impl Shot {
    fn of(f: &ClipFacts) -> Self {
        let (seen, unseen): (Vec<&crate::analyze::FrameFact>, Vec<&crate::analyze::FrameFact>) = f.frames.iter().partition(|x| x.present);
        Self {
            item: f.item,
            name: f.name.clone(),
            clip_seconds: f.seconds,
            best: f.best,
            seen: seen.iter().map(|x| x.seconds).collect(),
            unseen: unseen.iter().map(|x| x.seconds).collect(),
        }
    }

    /// The stretches of the clip where the subject is in view, `(start, end)` seconds in time
    /// order: each run of sampled moments that show it, reaching halfway to the neighbouring
    /// moments that do not (to the clip's ends past the first and last), inside a small margin.
    /// A clip without samples is one stretch.
    pub fn spans(&self) -> Vec<(f64, f64)> {
        let (lo, hi) = (0.2, (self.clip_seconds - 0.2).max(0.2));
        let mut samples: Vec<(f64, bool)> = self
            .seen
            .iter()
            .map(|t| (*t, true))
            .chain(self.unseen.iter().map(|t| (*t, false)))
            .filter(|(t, _)| t.is_finite())
            .map(|(t, v)| (t.clamp(lo, hi), v))
            .collect();
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        if samples.is_empty() {
            return vec![(lo, hi)];
        }
        let mut out = Vec::new();
        let mut run: Option<usize> = None;
        for i in 0..=samples.len() {
            let seen = samples.get(i).is_some_and(|s| s.1);
            match (seen, run) {
                (true, None) => run = Some(i),
                (false, Some(first)) => {
                    let start = match first.checked_sub(1).and_then(|p| samples.get(p)) {
                        Some(prev) => (prev.0 + samples.get(first).map_or(prev.0, |s| s.0)) / 2.0,
                        None => lo,
                    };
                    let end = match (samples.get(i - 1), samples.get(i)) {
                        (Some(last), Some(next)) => (last.0 + next.0) / 2.0,
                        _ => hi,
                    };
                    if end > start {
                        out.push((start, end));
                    }
                    run = None;
                }
                _ => {}
            }
        }
        out
    }

    /// Where a `len`-second window into this clip starts: where it holds the most moments that
    /// show the subject and the fewest that do not, closest to the best moment on a tie, inside
    /// the clip with a little margin at both ends.
    pub fn window_start(&self, len: f64) -> f64 {
        let lo = 0.2;
        let hi = (self.clip_seconds - len - 0.2).max(lo);
        let around_best = (self.best - len * 0.4).clamp(lo, hi);
        if self.seen.is_empty() && self.unseen.is_empty() {
            return around_best;
        }
        let score = |a: f64| {
            let inside = |x: &&f64| **x >= a && **x <= a + len;
            self.seen.iter().filter(inside).count() as f64 - 1.5 * self.unseen.iter().filter(inside).count() as f64
        };
        let steps = (((hi - lo) / 0.5).ceil() as usize).min(2000);
        let mut best = (score(around_best), around_best);
        for i in 0..=steps {
            let a = (lo + i as f64 * 0.5).min(hi);
            let sc = score(a);
            if sc > best.0 + 1e-9 || ((sc - best.0).abs() < 1e-9 && (a - around_best).abs() < (best.1 - around_best).abs()) {
                best = (sc, a);
            }
        }
        best.1
    }
}

/// One counted subject (a robot), with its shots.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub number: usize,
    pub label: String,
    pub kind: String,
    pub description: String,
    pub unix: Option<i64>,
    pub shots: Vec<Shot>,
}

/// The voice-over.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Script {
    pub intro: String,
    pub lines: Vec<String>,
    pub outro: String,
    /// Lines the model did not deliver, filled from templates.
    #[serde(default)]
    pub templated: usize,
}

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// `(year, month 1..=12, day)` of Unix seconds (UTC).
pub fn ymd(unix: i64) -> (i64, u32, u32) {
    let z = unix.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// "October 2022".
pub fn month_year(unix: Option<i64>) -> String {
    match unix {
        Some(t) => {
            let (y, m, _) = ymd(t);
            format!("{} {y}", MONTHS.get(m.saturating_sub(1) as usize).copied().unwrap_or(""))
        }
        None => String::new(),
    }
}

/// A label that says only what kind of thing it is ("wheeled robot", "robot"), which the model
/// gives when it cannot tell more.
pub(crate) fn generic(label: &str, subject: &str) -> bool {
    let l = label.to_lowercase();
    let words: Vec<&str> = l.split_whitespace().filter(|w| *w != subject && *w != "robot" && *w != "a" && *w != "the").collect();
    words.len() <= 1 && words.iter().all(|w| ["wheeled", "toy", "humanoid", "quadruped", "industrial", "small", "large", "arm", "other"].contains(w))
}

/// Shots one entry shows at most.
const MAX_SHOTS: usize = 3;
/// A robot with the same specific name filmed again within this many days (the next day of an
/// expo) is the same robot: it is counted once.
pub(crate) const SAME_EVENT_DAYS: i64 = 7;

/// A clip's label as an entry shows it: a bare kind (or nothing) becomes "{kind} {subject}".
fn entry_label(f: &ClipFacts, subject: &str) -> String {
    if f.label.trim().is_empty() || f.label.eq_ignore_ascii_case(&f.kind) { format!("{} {subject}", f.kind).trim().to_string() } else { f.label.clone() }
}

/// Whether clip `f` shows the robot of an entry (its clips `members`, oldest first; `last`: the
/// entry just before). With identities from the analysis, the same identity is the same robot
/// and a different one never is; without (an older report), by label against the entry's first
/// clip: the same specific name within [`SAME_EVENT_DAYS`], or, on the same day and for the entry
/// just before, a generic label of the same kind ("wheeled robot").
fn same_robot(members: &[&ClipFacts], f: &ClipFacts, last: bool, subject: &str) -> bool {
    let Some(first) = members.first() else { return false };
    if let Some(id) = f.identity {
        if members.iter().any(|m| m.identity == Some(id)) {
            return true;
        }
        if members.iter().all(|m| m.identity.is_some()) {
            return false;
        }
    }
    let label = entry_label(first, subject);
    let days_apart = match (first.unix, f.unix) {
        (Some(a), Some(b)) => (b.div_euclid(86_400) - a.div_euclid(86_400)).abs(),
        _ => i64::MAX,
    };
    let named = !generic(&label, subject) && label.eq_ignore_ascii_case(f.label.trim()) && days_apart <= SAME_EVENT_DAYS;
    let generic_next = last && days_apart == 0 && generic(&label, subject) && generic(&f.label, subject) && first.kind == f.kind;
    named || generic_next
}

/// Seconds a clip shows the subject, about: its length times the share of frames that show it.
fn in_view(f: &ClipFacts) -> f64 {
    // (a NaN coverage stays NaN through the clamp, and counts as nothing)
    let v = f.coverage.clamp(0.0, 1.0) * f.seconds.max(0.0);
    if v.is_finite() { v } else { 0.0 }
}

/// One counted robot from its clips (oldest first): the (up to) [`MAX_SHOTS`] clips that show it
/// longest, in time order; the most specific label among them all — a name read on the robot
/// ([`crate::analyze::FrameText::names`]), then any specific label, the earliest among equals —
/// with that clip's kind and description; the date it was first filmed.
fn entry(number: usize, members: &[&ClipFacts], subject: &str) -> Option<Entry> {
    let first = members.first()?;
    let rank = |f: &ClipFacts| {
        if f.text.names(&f.label) {
            2
        } else if !generic(&entry_label(f, subject), subject) {
            1
        } else {
            0
        }
    };
    let named = members.iter().enumerate().max_by_key(|(i, f)| (rank(f), std::cmp::Reverse(*i))).map_or(*first, |(_, f)| *f);
    let mut shown: Vec<(usize, &ClipFacts)> = members.iter().copied().enumerate().collect();
    shown.sort_by(|(i, a), (j, b)| in_view(b).total_cmp(&in_view(a)).then(i.cmp(j)));
    shown.truncate(MAX_SHOTS);
    shown.sort_by_key(|(i, _)| *i);
    Some(Entry {
        number,
        label: entry_label(named, subject),
        kind: named.kind.clone(),
        description: named.description.clone(),
        unix: first.unix,
        shots: shown.into_iter().map(|(_, f)| Shot::of(f)).collect(),
    })
}

/// Clips that show the subject → entries, oldest first, one per robot ([`same_robot`]: by the
/// analysis' identities, or by label in reports without them). A robot filmed more than three
/// times shows the clips where it is longest in view.
pub fn group(facts: &[ClipFacts], subject: &str, min_coverage: f64) -> Vec<Entry> {
    let mut clips: Vec<&ClipFacts> = facts.iter().filter(|f| f.present && f.coverage >= min_coverage).collect();
    clips.sort_by_key(|f| (f.unix.unwrap_or(i64::MAX), f.item));
    let mut robots: Vec<Vec<&ClipFacts>> = Vec::new();
    for f in clips {
        let n = robots.len();
        let same = robots.iter().enumerate().rev().find(|(i, members)| same_robot(members, f, *i + 1 == n, subject)).map(|(i, _)| i);
        match same.and_then(|i| robots.get_mut(i)) {
            Some(members) => members.push(f),
            None => robots.push(vec![f]),
        }
    }
    robots.iter().filter(|members| !members.is_empty()).enumerate().filter_map(|(k, members)| entry(k + 1, members, subject)).collect()
}

/// A label as a title shows it: its first letter capitalised ("service robot" → "Service robot").
pub fn display_label(label: &str) -> String {
    let mut c = label.trim().chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn template_line(e: &Entry) -> String {
    let when = month_year(e.unix);
    let what = e.label.trim();
    if when.is_empty() { format!("Next up: the {what}.") } else { format!("{when}: the {what}.") }
}

/// Clean one model sentence: one line, no quotes or numbering, at most ~40 words.
fn tidy(s: &str) -> String {
    let s = s.replace(['\n', '\r', '"', '“', '”'], " ");
    let s = s.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '#' || c == '-').trim();
    s.split_whitespace().take(40).collect::<Vec<_>>().join(" ")
}

/// A model line without the item heading it sometimes copies in front ("March 2020 — Zaku II
/// model kit: I built…" → "I built…"), which would be read out.
fn strip_heading(line: &str, e: &Entry) -> String {
    let Some((head, rest)) = line.split_once(": ") else { return line.to_string() };
    let head_l = head.to_lowercase();
    let copied = head.chars().count() <= 80
        && (head_l.contains(&e.label.to_lowercase()) || (!month_year(e.unix).is_empty() && head_l == month_year(e.unix).to_lowercase()));
    if copied && rest.split_whitespace().count() >= 3 {
        let mut c = rest.trim().chars();
        c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
    } else {
        line.to_string()
    }
}

/// Entries per request: small local models write a handful of lines reliably and fast, where
/// one long answer for a whole show can run past the request timeout.
const SCRIPT_BATCH: usize = 6;

const SCRIPT_SYSTEM: &str = "You write short, natural voice-over lines for a YouTube video. The narrator is the person who filmed the clips, speaking in the first person, warm and a little excited. Use only the facts given; never invent names, brands, places or numbers. Answer with JSON only.";

/// The voice-over for `entries`, written by `model` from the analysis facts: the intro and
/// outro in one request, then the entries' lines a few at a time. Any line the model does not
/// deliver gets a plain template line, so the edit never depends on the model behaving.
pub async fn write_script(model: &dyn ChatModel, opts: &Options, entries: &[Entry], progress: &mut dyn FnMut(&str)) -> Script {
    let years = years(entries);
    let video = format!(
        "The video: \"{}\" — every {} I have filmed{}, in the order I met them ({} in all). A counter on screen numbers them, so do not say the numbers.",
        opts.title,
        opts.subject,
        if years.is_empty() { String::new() } else { format!(" ({years})") },
        entries.len()
    );
    let names: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
    let user = format!(
        "{video}\n\nAmong them: {}.\n\nWrite \"intro\": two sentences that open the video, and \"outro\": two sentences that close it.\n\nJSON: {{\"intro\": \"...\", \"outro\": \"...\"}}",
        names.join(", ")
    );
    let (mut intro, mut outro) = (String::new(), String::new());
    match model.complete_json(SCRIPT_SYSTEM, &user, &[]).await {
        Ok(v) => {
            intro = v.get("intro").and_then(Value::as_str).map(tidy).unwrap_or_default();
            outro = v.get("outro").and_then(Value::as_str).map(tidy).unwrap_or_default();
        }
        Err(e) => progress(&format!("the script model did not write the intro ({e}); using a plain one")),
    }
    let mut written: Vec<Option<String>> = vec![None; entries.len()];
    for (b, batch) in entries.chunks(SCRIPT_BATCH).enumerate() {
        let listing: String =
            batch.iter().enumerate().map(|(i, e)| format!("{}. {} — {}: {}", i + 1, month_year(e.unix), e.label, e.description)).collect::<Vec<_>>().join("\n");
        let user = format!(
            "{video}\n\nWrite \"lines\": exactly {} entries, one per item below in the same order. Each is one or two short spoken sentences (at most 25 words) that say when it was and what it is or does, as I would tell a friend; do not start with the item's heading.\n\nItems:\n{listing}\n\nJSON: {{\"lines\": [\"...\"]}}",
            batch.len()
        );
        match model.complete_json(SCRIPT_SYSTEM, &user, &[]).await {
            Ok(v) => {
                let lines = v.get("lines").and_then(Value::as_array).cloned().unwrap_or_default();
                for (i, e) in batch.iter().enumerate() {
                    let line = lines.get(i).and_then(Value::as_str).map(tidy).map(|l| strip_heading(&l, e)).filter(|l| l.split_whitespace().count() >= 3);
                    if let Some(slot) = written.get_mut(b * SCRIPT_BATCH + i) {
                        *slot = line;
                    }
                }
            }
            Err(e) => {
                progress(&format!("the script model did not write lines {}–{} ({e}); using plain ones", b * SCRIPT_BATCH + 1, b * SCRIPT_BATCH + batch.len()))
            }
        }
    }
    let mut templated = 0;
    let lines: Vec<String> = entries
        .iter()
        .zip(written)
        .map(|(e, w)| match w {
            Some(l) => l,
            None => {
                templated += 1;
                template_line(e)
            }
        })
        .collect();
    if intro.split_whitespace().count() < 4 {
        intro = format!(
            "This is every {} I have filmed{}, in order. Let's count them.",
            opts.subject,
            if years.is_empty() { String::new() } else { format!(" from {years}") }
        );
    }
    if outro.split_whitespace().count() < 4 {
        outro = format!("That's {} {}s so far. Thanks for watching, and see you at the next one.", entries.len(), opts.subject);
    }
    Script { intro, lines, outro, templated }
}

/// "2020 – 2026" for the entries' years.
pub fn years(entries: &[Entry]) -> String {
    let ys: Vec<i64> = entries.iter().filter_map(|e| e.unix).map(|t| ymd(t).0).collect();
    match (ys.iter().min(), ys.iter().max()) {
        (Some(a), Some(b)) if a == b => a.to_string(),
        (Some(a), Some(b)) => format!("{a} – {b}"),
        _ => String::new(),
    }
}

/// Transitions between entries, in rotation (all render on the CPU, no handles beyond a few frames).
const TRANSITIONS: &[(&str, Option<(&str, &str)>)] = &[
    ("cross_zoom", None),
    ("whip", Some(("direction", "From East"))),
    ("flash", None),
    ("push", Some(("direction", "From South"))),
    ("glitch", None),
    ("light_leak", None),
    ("whip", Some(("direction", "From West"))),
    ("zoom_blur", None),
];

/// What [`build`] made.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Built {
    pub sequence: Option<u64>,
    pub seconds: f64,
    pub entries: usize,
    pub narration_lines: usize,
    /// Narration lines taken from the narration cache (not synthesized again).
    pub narration_cached: usize,
    /// Where each part starts: the intro, each year, the outro.
    pub chapters: Vec<Chapter>,
    pub notes: Vec<String>,
    pub export: Option<Value>,
    /// Where each entry and its shots landed on the timeline (what [`crate::qa`] checks).
    pub layout: Vec<PlacedEntry>,
}

/// An entry as built: its span on the timeline and its shots, seconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacedEntry {
    pub number: usize,
    pub label: String,
    pub start: f64,
    pub end: f64,
    pub shots: Vec<PlacedShot>,
}

/// A shot as built: its picture clip (on V2) and media item, and where it plays.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacedShot {
    pub clip: u64,
    pub item: u64,
    pub start: f64,
    pub end: f64,
}

/// A chapter of the video.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Chapter {
    pub seconds: f64,
    pub title: String,
}

/// YouTube's shortest chapter, seconds.
const MIN_CHAPTER: f64 = 10.0;

/// The chapters as YouTube reads them from a video's description ("0:00 Intro", one per line):
/// the first at 0:00 and each at least 10 s long — a shorter one is joined with the one after
/// it ("2022 / 2023"), a short last one with the one before — and nothing when fewer than three
/// remain, since YouTube then ignores them all.
pub fn youtube_chapters(chapters: &[Chapter], total: f64) -> String {
    let mut kept: Vec<Chapter> = Vec::new();
    for c in chapters {
        match kept.last_mut() {
            Some(last) if c.seconds - last.seconds < MIN_CHAPTER => last.title = format!("{} / {}", last.title, c.title),
            _ => kept.push(c.clone()),
        }
    }
    if kept.len() >= 2
        && kept.last().is_some_and(|c| total - c.seconds < MIN_CHAPTER)
        && let Some(short) = kept.pop()
        && let Some(last) = kept.last_mut()
    {
        last.title = format!("{} / {}", last.title, short.title);
    }
    if kept.len() < 3 {
        return String::new();
    }
    kept.iter()
        .enumerate()
        .map(|(i, c)| {
            let t = if i == 0 { 0 } else { c.seconds.max(0.0).floor() as u64 };
            let (h, m, s) = (t / 3600, t / 60 % 60, t % 60);
            let stamp = if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") };
            format!("{stamp} {}", c.title)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    let step = crate::PlannedStep::new(id, p.clone(), "showcase");
    crate::validate::validate_step(&step).map_err(AutoCutError::Invalid)?;
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

/// Media size of an item (from a tiny frame).
async fn media_size(ex: &dyn Executor, item: u64) -> Option<(f64, f64)> {
    let r = ex.execute("media.frame", json!({"item": item, "seconds": 0.0, "maxSide": 64})).await.ok()?;
    Some((r["width"].as_f64()?, r["height"].as_f64()?))
}

struct Tracks {
    /// Frames per second of the sequence: every placement is frame-exact.
    fps: f64,
    bg: String,
    fg: String,
    /// Main titles and lower thirds.
    titles_index: usize,
    /// Text that overlaps a title: subtitles and year cards.
    over_index: usize,
    counter_index: usize,
    clip_audio: String,
    voice: String,
    music: String,
    /// Whooshes.
    sfx: String,
    /// Impacts and risers.
    hits: String,
}

/// A generated sound effect at `t` on `track`, or on any free audio track when that one is busy
/// there (two cues can overlap when entries are short).
async fn sfx(ex: &dyn Executor, kind: &str, duration: f64, t: f64, track: &str, intensity: f64, mix: &mut Mix) -> Result<(), AutoCutError> {
    let p = json!({"kind": kind, "duration": duration, "time": ticks(t), "track": track, "intensity": intensity});
    let r = match run(ex, "audio.generate", p).await {
        Ok(r) => r,
        Err(_) => run(ex, "audio.generate", json!({"kind": kind, "duration": duration, "time": ticks(t), "intensity": intensity})).await?,
    };
    mix.effects.extend(r["clip"].as_u64());
    Ok(())
}

/// The sound clips of the show by role, for the mix.
#[derive(Default)]
struct Mix {
    /// The shots' own sound.
    sounds: Vec<u64>,
    /// Whooshes, impacts, risers.
    effects: Vec<u64>,
}

/// Loudness targets (LUFS) of the mix: the voice on top at a YouTube level, the shots' own
/// sound and the music under it (and ducked further while the voice speaks), the effects between.
const VOICE_LUFS: f64 = -16.0;
const SOUND_LUFS: f64 = -27.0;
const MUSIC_LUFS: f64 = -24.0;
const EFFECTS_LUFS: f64 = -23.0;
/// Ceiling of the limiter on the mix, dBFS.
const CEILING_DB: f64 = -1.0;
/// The shots' own sound when it cannot be measured.
const FALLBACK_SOUND_DB: f64 = -14.0;

/// Mix the show: set each clip's audio type, match its loudness to its role's target, duck the
/// shots' sound and the music under the voice, and put a limiter on the mix so nothing clips.
/// Steps that fail are noted, never fatal: the edit is still usable unmixed.
async fn mix_down(ex: &dyn Executor, mix: &Mix, narrations: &[u64], music: Option<u64>, notes: &mut Vec<String>) {
    let mut step = async |id: &str, p: Value| {
        if let Err(e) = run(ex, id, p).await {
            notes.push(format!("mix: {e}"));
            return false;
        }
        true
    };
    if !narrations.is_empty() {
        step("essentialSound.setType", json!({"clips": narrations, "type": "dialogue"})).await;
        step("essentialSound.autoMatch", json!({"clips": narrations, "target": VOICE_LUFS})).await;
    }
    if !mix.sounds.is_empty() {
        step("essentialSound.applyPreset", json!({"clips": mix.sounds, "preset": "Duck Under Dialogue", "type": "ambience"})).await;
        if !step("essentialSound.autoMatch", json!({"clips": mix.sounds, "target": SOUND_LUFS})).await {
            step("clip.audioGain", json!({"clips": mix.sounds, "mode": "set", "db": FALLBACK_SOUND_DB})).await;
        }
    }
    if !mix.effects.is_empty() {
        step("essentialSound.setType", json!({"clips": mix.effects, "type": "sfx"})).await;
        step("essentialSound.autoMatch", json!({"clips": mix.effects, "target": EFFECTS_LUFS})).await;
    }
    if let Some(m) = music {
        step("essentialSound.applyPreset", json!({"clips": [m], "preset": "Duck Under Dialogue", "type": "music"})).await;
        step("essentialSound.autoMatch", json!({"clips": [m], "target": MUSIC_LUFS})).await;
    }
    if !narrations.is_empty() {
        let ducked: Vec<u64> = mix.sounds.iter().copied().chain(music).collect();
        step("essentialSound.generateDucking", json!({"clips": ducked})).await;
    }
    if step("mixer.addInsert", json!({"strip": "Mix", "effect": "hard_limiter"})).await {
        step("mixer.setInsert", json!({"strip": "Mix", "slot": 0, "params": {"max": CEILING_DB}})).await;
    }
}

/// A placed shot: its foreground, background and sound clips, and where it ends (seconds, on a
/// frame).
struct Placed {
    fg: u64,
    bg: Option<u64>,
    audio: Option<u64>,
    end: f64,
}

/// Place `item`'s `[src, src + dur)` at `t` (both ends on frames): foreground fitted on V2 (with
/// its sound on A1, which [`mix_down`] levels), and for a clip whose shape is not the frame's, a
/// blurred, dimmed copy filling V1 behind it.
async fn place_shot(ex: &dyn Executor, tr: &Tracks, item: u64, t: f64, src: f64, dur: f64, frame_aspect: f64) -> Result<Placed, AutoCutError> {
    let t = on_frame(t, tr.fps);
    let end = on_frame(t + dur, tr.fps).max(t + 1.0 / tr.fps);
    let dur = end - t;
    let r = run(
        ex,
        "timeline.place",
        json!({"item": item, "track": tr.fg, "audioTrack": tr.clip_audio, "time": ticks(t), "sourceIn": ticks(src), "duration": ticks(dur)}),
    )
    .await?;
    let ids: Vec<u64> = r["clips"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    let fg = *ids.first().ok_or_else(|| AutoCutError::Executor("timeline.place placed nothing".into()))?;
    let audio = ids.get(1).copied();
    run(ex, "clip.fitToFrame", json!({"clips": [fg]})).await?;
    let aspect = media_size(ex, item).await.map(|(w, h)| w / h.max(1.0)).unwrap_or(frame_aspect);
    let mut bg = None;
    if (aspect - frame_aspect).abs() > 0.05 {
        let r =
            run(ex, "timeline.place", json!({"item": item, "track": tr.bg, "time": ticks(t), "sourceIn": ticks(src), "duration": ticks(dur), "audio": false}))
                .await?;
        if let Some(b) = r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64) {
            run(ex, "clip.fillFrame", json!({"clips": [b]})).await?;
            run(ex, "effects.apply", json!({"clips": [b], "effect": "gaussian_blur"})).await?;
            run(ex, "effects.setParam", json!({"clip": b, "effect": "gaussian_blur", "param": "blurriness", "value": 60.0})).await?;
            run(ex, "effects.setParam", json!({"clip": b, "effect": "opacity", "param": "opacity", "value": 45.0})).await?;
            bg = Some(b);
        }
    }
    Ok(Placed { fg, bg, audio, end })
}

/// Fade a shot's own sound in at its start (a crossfade when it follows another shot's sound).
async fn fade_in(ex: &dyn Executor, audio: Option<u64>) {
    if let Some(a) = audio {
        run(ex, "sequence.applyAudioTransition", json!({"clip": a, "edge": "in", "effect": "constant_power", "frames": AUDIO_FADE_FRAMES})).await.ok();
    }
}

/// A text title clip on video track `track` (an index), from `t` for `dur` seconds, both ends on
/// frames.
#[allow(clippy::too_many_arguments)]
async fn title(ex: &dyn Executor, track: usize, fps: f64, text: &str, t: f64, dur: f64, size: f64, pos: (f64, f64), props: Value) -> Result<u64, AutoCutError> {
    let (t0, t1) = (on_frame(t, fps), on_frame(t + dur, fps));
    let dur = (t1 - t0).max(1.0 / fps.max(1.0));
    let r =
        run(ex, "graphics.newText", json!({"text": text, "seconds": dur, "time": ticks(t0), "size": size, "position": [pos.0, pos.1], "track": track})).await?;
    let clip = r["clip"].as_u64().ok_or_else(|| AutoCutError::Executor("graphics.newText made no clip".into()))?;
    run(ex, "graphics.set", json!({"clip": clip, "props": props})).await?;
    Ok(clip)
}

/// The narration clips placed, and how many lines came from the narration cache (nothing was
/// synthesized for them: a rebuild re-reads only the lines that changed).
#[derive(Default)]
struct Narrations {
    clips: Vec<u64>,
    cached: usize,
}

/// Narrate `text` at `t` on the voice track; its length in seconds (0 without a voice).
async fn narrate(ex: &dyn Executor, tr: &Tracks, voice: Option<&str>, text: &str, t: f64, narrations: &mut Narrations) -> Result<f64, AutoCutError> {
    let Some(voice) = voice else { return Ok(0.0) };
    let r = run(ex, "tts.create", json!({"text": text, "voice": voice, "time": ticks(t), "track": tr.voice})).await?;
    if let Some(c) = r["clip"].as_u64() {
        narrations.clips.push(c);
    }
    if r["cached"].as_bool() == Some(true) {
        narrations.cached += 1;
    }
    Ok(r["duration"].as_f64().map(|d| d / TPS).unwrap_or(0.0))
}

/// Shortest window cut from a stretch where the subject is in view, seconds.
const MIN_WINDOW: f64 = 1.0;

/// How many shots an entry lasting `seconds` shows: one under 7 s, two under 12 s, else three.
fn shot_count(seconds: f64) -> usize {
    if seconds >= 12.0 {
        3
    } else if seconds >= 7.0 {
        2
    } else {
        1
    }
}

/// Shot windows for an entry lasting `seconds`: `(shot, source start, length)` in order, the
/// time split over its shots, each shot's share cut from the stretches where the subject is in
/// view ([`Shot::spans`]) — the longest first, around the best moment, more than one stretch of
/// a clip when one is too short. The windows can add up to less than `seconds` when the subject
/// is rarely in view; [`whole_windows`] is the fallback that always fills the time.
pub fn windows(e: &Entry, seconds: f64) -> Vec<(Shot, f64, f64)> {
    let shots: Vec<&Shot> = e.shots.iter().take(shot_count(seconds)).collect();
    let each = seconds / shots.len().max(1) as f64;
    let mut out = Vec::new();
    for s in shots {
        let mut spans: Vec<(f64, f64)> = s.spans().into_iter().filter(|(a, b)| b - a >= MIN_WINDOW.min(each)).collect();
        // the longest stretches first, the one holding the best moment first among equals
        spans.sort_by(|x, y| (y.1 - y.0).total_cmp(&(x.1 - x.0)).then_with(|| (s.best < x.0 || s.best > x.1).cmp(&(s.best < y.0 || s.best > y.1))));
        let mut picked: Vec<(f64, f64)> = Vec::new();
        let mut left = each;
        for (a, b) in spans {
            if left < MIN_WINDOW * 0.5 {
                break;
            }
            let len = left.min(b - a);
            let start = (s.best - len * 0.4).clamp(a, (b - len).max(a));
            picked.push((start, len));
            left -= len;
        }
        picked.sort_by(|x, y| x.0.total_cmp(&y.0));
        out.extend(picked.into_iter().map(|(start, len)| (s.clone(), start, len)));
    }
    out
}

/// Windows that fill `seconds` whatever is in view: each shot's share in one window, placed
/// where the subject shows most ([`Shot::window_start`]).
pub fn whole_windows(e: &Entry, seconds: f64) -> Vec<(Shot, f64, f64)> {
    let shots: Vec<&Shot> = e.shots.iter().take(shot_count(seconds)).collect();
    let each = seconds / shots.len().max(1) as f64;
    shots
        .into_iter()
        .map(|s| {
            let usable = (s.clip_seconds - 0.4).max(0.5);
            let len = each.min(usable);
            (s.clone(), s.window_start(len), len)
        })
        .collect()
}

/// Most shots in the intro montage.
const MAX_MONTAGE: usize = 12;

/// `n` shots for the intro montage: entries spread evenly over the whole show (first and last
/// included), each one's first shot.
pub fn montage(entries: &[Entry], n: usize) -> Vec<Shot> {
    let len = entries.len();
    if len == 0 || n == 0 {
        return Vec::new();
    }
    let n = n.min(len);
    let mut picks: Vec<usize> = (0..n).map(|i| if n == 1 { 0 } else { i * (len - 1) / (n - 1) }).collect();
    picks.dedup();
    picks.into_iter().filter_map(|i| entries.get(i).and_then(|e| e.shots.first()).cloned()).collect()
}

/// After the words for an entry, its own picture and sound run on this long before the next one.
const BREATHE: f64 = 2.8;
/// Shortest and longest entry, seconds.
const MIN_ENTRY: f64 = 6.0;
const MAX_ENTRY: f64 = 22.0;
/// Fades on each shot's own sound, frames: no clip's sound starts or stops with a click.
const AUDIO_FADE_FRAMES: i64 = 8;

/// Build the showcase into a new sequence. `progress` gets a line per entry.
pub async fn build(ex: &dyn Executor, opts: &Options, entries: &[Entry], script: &Script, progress: &mut dyn FnMut(&str)) -> Result<Built, AutoCutError> {
    if entries.is_empty() {
        return Err(AutoCutError::Invalid(format!("no clip shows a {} clearly enough to count", opts.subject)));
    }
    let mut built = Built { entries: entries.len(), ..Built::default() };
    let r =
        run(ex, "file.newSequence", json!({"name": opts.title, "width": opts.width, "height": opts.height, "fps": opts.fps, "video": 5, "audio": 5})).await?;
    built.sequence = r["sequence"].as_u64();
    let tr = Tracks {
        fps: opts.fps.clamp(1.0, 240.0),
        bg: "V1".into(),
        fg: "V2".into(),
        titles_index: 2,
        over_index: 3,
        counter_index: 4,
        clip_audio: "A1".into(),
        voice: "A2".into(),
        music: "A3".into(),
        sfx: "A4".into(),
        hits: "A5".into(),
    };
    let (w, h) = (f64::from(opts.width), f64::from(opts.height));
    let frame_aspect = w / h.max(1.0);
    let voice = opts.voice.as_deref();
    let mut narrations = Narrations::default();
    let mut mix = Mix::default();
    let mut fg_firsts: Vec<(u64, Option<u64>)> = Vec::new();
    let title_props = json!({"font_style": "Bold", "align": "center", "stroke": true, "stroke_width": 6, "shadow": true});

    // ---- intro: a cold open — a montage of robots from across the years cut on the beat,
    // the title over it, a riser into an impact
    let intro_voice_at = 2.2;
    let intro_spoken = narrate(ex, &tr, voice, &script.intro, intro_voice_at, &mut narrations).await?;
    // the voice track's next free moment: a line never starts over the last one
    let mut voice_free_at = intro_voice_at + intro_spoken;
    let intro_len = on_frame((intro_voice_at + intro_spoken + 0.8).max(6.0), tr.fps);
    let cut = 2.0 * 60.0 / opts.bpm.clamp(60.0, 180.0);
    let picks = montage(entries, ((intro_len / cut).floor() as usize).clamp(1, MAX_MONTAGE));
    let mut at = 0.0;
    // the picks in turn, the last one to the end of the intro; a clip too short for its turn
    // hands over to the next (round again if need be), so the intro has no black
    let mut i = 0;
    while at < intro_len - 0.5 / tr.fps && i < MAX_MONTAGE * 4 {
        let Some(s) = picks.get(i % picks.len().max(1)) else { break };
        let len = if i + 1 >= picks.len() { intro_len - at } else { cut.min(intro_len - at) };
        let len = len.min((s.clip_seconds - 0.4).max(0.5));
        let shot = place_shot(ex, &tr, s.item, at, s.window_start(len), len, frame_aspect).await?;
        fade_in(ex, shot.audio).await;
        mix.sounds.extend(shot.audio);
        at = shot.end;
        i += 1;
    }
    sfx(ex, "riser", 1.8, 0.0, &tr.hits, 0.7, &mut mix).await?;
    sfx(ex, "impact", 3.0, 1.8, &tr.hits, 0.8, &mut mix).await?;
    title(ex, tr.titles_index, tr.fps, &opts.title, 1.8, intro_len - 1.8, 110.0, (w / 2.0, h * 0.46), title_props.clone()).await?;
    let subtitle = opts.subtitle.clone().unwrap_or_else(|| years(entries));
    if !subtitle.is_empty() {
        title(
            ex,
            tr.over_index,
            tr.fps,
            &subtitle,
            2.3,
            intro_len - 2.3,
            54.0,
            (w / 2.0, h * 0.46 + 95.0),
            json!({"font_style": "Bold", "align": "center", "fill_color": "#ffcc00", "shadow": true}),
        )
        .await?;
    }
    built.chapters.push(Chapter { seconds: 0.0, title: "Intro".into() });
    let mut t = intro_len;
    let mut counter_times = Vec::new();
    let mut last_year: Option<i64> = None;

    // ---- the entries
    for (k, e) in entries.iter().enumerate() {
        let year = e.unix.map(|u| ymd(u).0);
        let new_year = year.is_some() && year != last_year;
        last_year = year.or(last_year);
        let lead = if new_year { 1.4 } else { 0.5 };
        let line = script.lines.get(k).cloned().unwrap_or_else(|| template_line(e));
        let voice_at = (t + lead).max(voice_free_at + 0.15);
        let spoken = narrate(ex, &tr, voice, &line, voice_at, &mut narrations).await?;
        if spoken > 0.0 {
            voice_free_at = voice_at + spoken;
        }
        // the words, then a moment of the robot's own picture and sound before the next one
        let needed = (voice_at - t + spoken + BREATHE).clamp(MIN_ENTRY, MAX_ENTRY);
        let must = voice_at - t + spoken + 0.7;
        // the robot in view; when that is too little to carry the words even at half speed,
        // the clips as they are
        let mut wins = windows(e, needed);
        if wins.iter().map(|(_, _, l)| *l).sum::<f64>() / 0.5 < must {
            wins = whole_windows(e, needed);
        }
        let footage: f64 = wins.iter().map(|(_, _, l)| *l).sum();
        // footage too short for the words: slow the shots to cover them, never below half speed
        // (an entry with words to spare just ends sooner)
        let slow = if footage + 0.05 < must { (footage / must).clamp(0.5, 1.0) } else { 1.0 };
        let mut at = t;
        let mut firsts = None;
        let mut shots = Vec::new();
        for (i, (s, src, len)) in wins.iter().enumerate() {
            let Placed { fg, bg, audio, end } = place_shot(ex, &tr, s.item, at, *src, *len, frame_aspect).await?;
            mix.sounds.extend(audio);
            let mut shot_end = end;
            if slow < 0.999 {
                let clips: Vec<u64> = std::iter::once(fg).chain(bg).chain(audio).collect();
                if let Ok(r) =
                    run(ex, "clip.speedDuration", json!({"clips": clips, "speed": slow * 100.0, "interpolation": "frameBlending", "ripple": false})).await
                {
                    // where the slowed shot really ends (the engine rounds to frames)
                    let slowed = r["clips"].as_array().and_then(|a| a.iter().find(|c| c["clip"].as_u64() == Some(fg))).and_then(|c| c["end"].as_f64());
                    shot_end = slowed.map_or_else(|| on_frame(at + len / slow, tr.fps), |e| e / TPS);
                }
            }
            // after any speed change, which moves the clip's end
            fade_in(ex, audio).await;
            if i == 0 {
                firsts = Some((fg, bg));
            } else {
                // between shots (or stretches of one clip) of an entry: a quick dissolve
                run(ex, "sequence.applyVideoTransition", json!({"clip": fg, "edge": "in", "effect": "cross_dissolve", "frames": 8})).await.ok();
            }
            shots.push(PlacedShot { clip: fg, item: s.item, start: on_frame(at, tr.fps), end: shot_end });
            // a slowed shot can end between frames: the next starts on the following frame
            at = (shot_end * tr.fps - 1e-6).ceil() / tr.fps;
        }
        let entry_end = on_frame(at.max(t + 2.0), tr.fps);
        built.layout.push(PlacedEntry { number: e.number, label: e.label.clone(), start: t, end: entry_end, shots });
        if let Some((fg, bg)) = firsts {
            fg_firsts.push((fg, bg));
            // (the first entry follows the intro montage)
            {
                let (effect, param) =
                    if new_year { ("dip_to_black", None) } else { TRANSITIONS.get(k % TRANSITIONS.len()).copied().unwrap_or(("cross_dissolve", None)) };
                let mut p = json!({"clip": fg, "edge": "in", "effect": effect, "frames": if new_year { 18 } else { 12 }});
                if let Some((name, value)) = param {
                    p["params"] = json!({name: value});
                }
                if let Err(err) = run(ex, "sequence.applyVideoTransition", p).await {
                    built.notes.push(format!("transition {effect} at entry {}: {err}", e.number));
                }
                if let Some(b) = bg {
                    run(ex, "sequence.applyVideoTransition", json!({"clip": b, "edge": "in", "effect": "cross_dissolve", "frames": 12})).await.ok();
                }
                sfx(ex, "whoosh", 1.2, (t - 0.55).max(0.0), &tr.sfx, 0.6, &mut mix).await?;
            }
        }
        if new_year && let Some(y) = year {
            built.chapters.push(Chapter { seconds: t, title: y.to_string() });
            title(ex, tr.over_index, tr.fps, &y.to_string(), t, 1.4, 160.0, (w / 2.0, h * 0.58), title_props.clone()).await?;
            sfx(ex, "impact", 2.0, t, &tr.hits, 0.6, &mut mix).await?;
        }
        // the name and date, bottom left, while the narration introduces it
        let when = month_year(e.unix);
        let name = display_label(&e.label);
        let lower = if when.is_empty() { name } else { format!("{name}\n{when}") };
        title(
            ex,
            tr.titles_index,
            tr.fps,
            &lower,
            t + lead,
            (entry_end - t - lead - 0.3).clamp(1.5, 4.5),
            46.0,
            (w * 0.05, h * 0.82),
            json!({"font_style": "Bold", "align": "left", "background": true, "background_opacity": 55, "background_size": 18, "shadow": false}),
        )
        .await?;
        counter_times.push(t + lead);
        progress(&format!("[{}/{}] {} — {} ({:.1}s)", e.number, entries.len(), when, e.label, entry_end - t));
        t = entry_end;
    }

    // ---- outro: the total, over a blurred shot of the latest one, with the closing words
    built.chapters.push(Chapter { seconds: t, title: "Outro".into() });
    let outro_voice_at = (t + 0.8).max(voice_free_at + 0.15);
    let outro_spoken = narrate(ex, &tr, voice, &script.outro, outro_voice_at, &mut narrations).await?;
    let outro_len = (outro_voice_at - t + outro_spoken + 1.5).max(5.0);
    if let Some(h0) = entries.last().and_then(|e| e.shots.first()) {
        let len = outro_len.min((h0.clip_seconds - 0.4).max(1.0));
        let Placed { fg, audio, .. } = place_shot(ex, &tr, h0.item, t, h0.window_start(len), len, frame_aspect).await?;
        mix.sounds.extend(audio);
        fade_in(ex, audio).await;
        if let Some(a) = audio {
            run(ex, "sequence.applyAudioTransition", json!({"clip": a, "edge": "out", "effect": "constant_power", "frames": (opts.fps * 1.5) as i64}))
                .await
                .ok();
        }
        run(ex, "effects.apply", json!({"clips": [fg], "effect": "gaussian_blur"})).await?;
        run(ex, "effects.setParam", json!({"clip": fg, "effect": "gaussian_blur", "param": "blurriness", "value": 40.0})).await?;
        run(ex, "sequence.applyVideoTransition", json!({"clip": fg, "edge": "in", "effect": "dip_to_black", "frames": 18})).await.ok();
        run(ex, "sequence.applyVideoTransition", json!({"clip": fg, "edge": "out", "effect": "dip_to_black", "frames": 30})).await.ok();
    }
    sfx(ex, "impact", 3.0, t, &tr.hits, 0.7, &mut mix).await?;
    let closing = format!("{} {}S", entries.len(), opts.subject.to_uppercase());
    title(ex, tr.titles_index, tr.fps, &closing, t + 0.3, outro_len - 0.3, 130.0, (w / 2.0, h * 0.48), title_props.clone()).await?;
    title(
        ex,
        tr.over_index,
        tr.fps,
        "and counting",
        t + 0.9,
        outro_len - 0.9,
        56.0,
        (w / 2.0, h * 0.48 + 90.0),
        json!({"font_style": "Bold", "align": "center", "fill_color": "#ffcc00", "shadow": true}),
    )
    .await?;
    let total = t + outro_len;
    built.seconds = total;

    // ---- the counter, across the entries
    let counter_prefix = format!("{} #", opts.subject.to_uppercase());
    run(
        ex,
        "graphics.counter",
        json!({"times": counter_times, "end": t, "prefix": counter_prefix, "corner": "topRight", "size": 60, "color": "#ffffff", "track": tr.counter_index}),
    )
    .await?;

    // ---- music under everything, ducked under the voice, faded out
    let m = run(
        ex,
        "audio.generate",
        json!({"kind": "music", "duration": total + 0.5, "mood": opts.mood, "bpm": opts.bpm, "time": 0, "track": tr.music, "intensity": 0.75}),
    )
    .await?;
    let music = m["clip"].as_u64();
    if let Some(music) = music {
        run(ex, "sequence.applyAudioTransition", json!({"clip": music, "edge": "out", "effect": "constant_power", "frames": (opts.fps * 3.0) as i64}))
            .await
            .ok();
    }
    if !narrations.clips.is_empty() {
        progress(&format!("{} narration lines, {} from the narration cache", narrations.clips.len(), narrations.cached));
    }
    // ---- the mix
    progress("mixing");
    mix_down(ex, &mix, &narrations.clips, music, &mut built.notes).await;
    built.narration_lines = narrations.clips.len();
    built.narration_cached = narrations.cached;
    if let Some(path) = &opts.export {
        progress("exporting (YouTube 1080p)");
        built.export = Some(run(ex, "file.exportMedia", json!({"path": path, "preset": "YouTube 1080p Full HD", "wait": true})).await?);
    }
    Ok(built)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::FrameFact;

    fn fact(item: u64, label: &str, kind: &str, unix: i64, cov: f64) -> ClipFacts {
        ClipFacts {
            item,
            name: format!("c{item}.mp4"),
            seconds: 20.0,
            created: None,
            unix: Some(unix),
            present: cov > 0.0,
            coverage: cov,
            label: label.into(),
            kind: kind.into(),
            description: format!("{label} doing things"),
            best: 6.0,
            frames: vec![FrameFact { seconds: 6.0, present: true, area: 0.3, center: [0.5, 0.5] }],
            ..ClipFacts::default()
        }
    }

    #[test]
    fn dates() {
        assert_eq!(ymd(0), (1970, 1, 1));
        assert_eq!(ymd(1_665_741_668), (2022, 10, 14));
        assert_eq!(month_year(Some(1_665_741_668)), "October 2022");
        assert_eq!(month_year(None), "");
    }

    #[test]
    fn grouping_merges_same_day_same_robot_and_drops_weak_clips() {
        let day = 1_599_600_000; // 2020-09-08
        let facts = vec![
            fact(1, "wheeled robot", "wheeled", day, 0.8),
            fact(2, "wheeled", "wheeled", day + 600, 0.7),
            fact(3, "Pepper humanoid robot", "humanoid", day + 900, 1.0),
            fact(4, "Pepper humanoid robot", "humanoid", day + 86_400 * 30, 1.0), // another day: a new entry
            fact(5, "ghost", "other", day + 1000, 0.1),                           // too weak
            fact(6, "no robot", "", day + 2000, 0.0),
        ];
        let e = group(&facts, "robot", 0.3);
        assert_eq!(e.iter().map(|x| x.shots.len()).collect::<Vec<_>>(), vec![2, 1, 1], "{e:?}");
        assert_eq!(e.iter().map(|x| x.number).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(e[0].label, "wheeled robot");
        assert_eq!(years(&e), "2020");
    }

    /// `fact` with an identity from the analysis.
    fn known(item: u64, label: &str, kind: &str, unix: i64, identity: u32) -> ClipFacts {
        ClipFacts { identity: Some(identity), ..fact(item, label, kind, unix, 1.0) }
    }

    /// One expo: a red wheeled-legged robot filmed four times over three days under four labels
    /// (its name read on day four), another wheeled robot right after it on day one, a Spot.
    fn expo() -> Vec<ClipFacts> {
        let day = 1_665_446_400; // 2022-10-11
        vec![
            known(1, "wheeled robot", "wheeled", day, 1),
            known(2, "wheeled robot", "wheeled", day + 600, 2),
            ClipFacts { coverage: 0.4, ..known(3, "red toy robot", "toy", day + 86_400 * 2, 1) },
            ClipFacts {
                text: crate::analyze::FrameText { on_subject: vec!["ARDET".into(), "Swiss-Mile".into()], nearby: vec!["ETH zürich RSL".into()] },
                ..known(4, "Swiss-Mile ARDET robot", "wheeled", day + 86_400 * 3, 1)
            },
            ClipFacts { seconds: 30.0, ..known(5, "wheeled robot", "wheeled", day + 86_400 * 3 + 60, 1) },
            known(6, "Boston Dynamics Spot", "quadruped", day + 86_400 * 3 + 120, 3),
        ]
    }

    #[test]
    fn identities_merge_across_days_and_keep_different_robots_apart() {
        let e = group(&expo(), "robot", 0.3);
        let shape: Vec<(&str, Vec<u64>)> = e.iter().map(|x| (x.label.as_str(), x.shots.iter().map(|s| s.item).collect())).collect();
        // one robot over three days, named from what was read on it, its three clips longest in
        // view (the short-coverage one left out) in time order; the other wheeled robot of day
        // one, which a generic label would have merged with it, kept apart
        assert_eq!(shape, vec![("Swiss-Mile ARDET robot", vec![1, 4, 5]), ("wheeled robot", vec![2]), ("Boston Dynamics Spot", vec![6])]);
        assert_eq!(e.iter().map(|x| x.number).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(e[0].unix, Some(1_665_446_400), "dated when first filmed");
        assert_eq!((e[0].kind.as_str(), e[0].description.as_str()), ("wheeled", "Swiss-Mile ARDET robot doing things"));
        // hostile lengths and coverages from a report count as nothing in view
        for (coverage, seconds) in [(f64::NAN, 10.0), (0.5, f64::INFINITY), (1.0, -5.0), (f64::INFINITY, f64::NAN)] {
            assert_eq!(in_view(&ClipFacts { coverage, seconds, ..ClipFacts::default() }), 0.0, "{coverage} {seconds}");
        }
        assert_eq!(in_view(&ClipFacts { coverage: 7.0, seconds: 10.0, ..ClipFacts::default() }), 10.0);
    }

    #[test]
    fn reports_without_identities_group_by_label_as_before() {
        // an older report: no `identity` (nor `text`) in the JSON
        let old: Vec<ClipFacts> = expo()
            .into_iter()
            .map(|f| {
                let mut v = serde_json::to_value(&f).unwrap();
                let o = v.as_object_mut().unwrap();
                o.remove("identity");
                o.remove("text");
                serde_json::from_value(v).unwrap()
            })
            .collect();
        assert!(old.iter().all(|f| f.identity.is_none() && f.text.is_empty()));
        let e = group(&old, "robot", 0.3);
        let shape: Vec<Vec<u64>> = e.iter().map(|x| x.shots.iter().map(|s| s.item).collect()).collect();
        // day one's two generic wheeled robots are one by label; the rest stay apart
        assert_eq!(shape, vec![vec![1, 2], vec![3], vec![4], vec![5], vec![6]]);
        // a clip with an identity next to clips without one: by label
        let day = 1_665_446_400;
        let mixed = vec![fact(1, "Pepper humanoid robot", "humanoid", day, 1.0), known(2, "Pepper humanoid robot", "humanoid", day + 600, 7)];
        assert_eq!(group(&mixed, "robot", 0.3).len(), 1);
    }

    #[test]
    fn windows_stay_inside_clips_around_the_best_moment() {
        let e = Entry {
            number: 1,
            label: "x".into(),
            kind: "x".into(),
            description: String::new(),
            unix: None,
            shots: vec![
                Shot { item: 1, name: "a".into(), clip_seconds: 20.0, best: 19.5, seen: vec![], unseen: vec![] },
                Shot { item: 2, name: "b".into(), clip_seconds: 3.0, best: 1.0, seen: vec![], unseen: vec![] },
            ],
        };
        let w = whole_windows(&e, 8.0);
        assert_eq!(w.len(), 2);
        for (s, start, len) in &w {
            assert!(*start >= 0.2 && start + len <= s.clip_seconds - 0.2 + 1e-9, "{start} {len} in {}", s.clip_seconds);
        }
        assert_eq!(whole_windows(&e, 5.0).len(), 1, "short entries show one shot");
        // without samples a clip is one stretch, so the in-view windows are the same
        assert_eq!(windows(&e, 8.0).len(), 2);
    }

    #[test]
    fn stretches_in_view_come_from_the_samples() {
        // statues at 2.0 s and 7.8 s of a 15.5 s clip, not at 4.9, 10.7 and 13.6
        let s = Shot { item: 47, name: "s".into(), clip_seconds: 15.5, best: 7.8, seen: vec![2.0, 7.8], unseen: vec![4.9, 10.7, 13.6] };
        let spans = s.spans();
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert!((spans[0].0 - 0.2).abs() < 1e-9 && (spans[0].1 - 3.45).abs() < 1e-9, "{spans:?}");
        assert!((spans[1].0 - 6.35).abs() < 1e-9 && (spans[1].1 - 9.25).abs() < 1e-9, "{spans:?}");
        // every sample in view: the whole clip; none: nothing
        assert_eq!(Shot { seen: vec![3.0, 9.0], unseen: vec![], ..s.clone() }.spans(), vec![(0.2, 15.3)]);
        assert!(Shot { seen: vec![], unseen: vec![3.0, 9.0], ..s.clone() }.spans().is_empty());
        // hostile samples stay inside the clip
        assert!(Shot { seen: vec![f64::NAN, -5.0, 99.0], unseen: vec![], ..s.clone() }.spans().iter().all(|(a, b)| *a >= 0.2 && *b <= 15.3));
        // a 10 s entry of this clip: both stretches, in order, nothing out of view
        let e = Entry { number: 1, label: "x".into(), kind: "x".into(), description: String::new(), unix: None, shots: vec![s] };
        let w = windows(&e, 10.0);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].1 < w[1].1);
        for (_, start, len) in &w {
            assert!(spans.iter().any(|(a, b)| *start >= a - 1e-9 && start + len <= b + 1e-9), "{start} {len} outside {spans:?}");
        }
        let total: f64 = w.iter().map(|x| x.2).sum();
        assert!((total - 6.15).abs() < 1e-6, "{total}");
    }

    #[test]
    fn windows_go_where_the_subject_is_in_view() {
        // in view from 20 s to 40 s of a minute-long clip; the best moment is at 22 s
        let s = Shot {
            item: 1,
            name: "a".into(),
            clip_seconds: 60.0,
            best: 22.0,
            seen: vec![21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0],
            unseen: vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 42.0, 45.0],
        };
        let a = s.window_start(8.0);
        assert!(a >= 19.0 && a + 8.0 <= 41.0, "{a}");
        // without samples: around the best moment
        let bare = Shot { seen: vec![], unseen: vec![], ..s.clone() };
        assert!((bare.window_start(8.0) - (22.0 - 3.2)).abs() < 1e-9);
        // a clip shorter than the window: from its start
        let short = Shot { clip_seconds: 3.0, best: 1.0, ..s };
        assert!((short.window_start(8.0) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn the_same_named_robot_at_one_event_is_counted_once() {
        let day = 1_665_446_400; // 2022-10-11, an expo's first day
        let facts = vec![
            fact(1, "Boston Dynamics Spot", "quadruped", day, 1.0),
            fact(2, "Pepper humanoid robot", "humanoid", day + 600, 1.0),
            fact(3, "Boston Dynamics Spot", "quadruped", day + 86_400 * 3, 1.0), // day four: the same Spot
            fact(4, "quadruped robot", "quadruped", day + 86_400 * 3 + 60, 1.0), // a generic label: another robot
            fact(5, "Boston Dynamics Spot", "quadruped", day + 86_400 * 3 + 120, 1.0),
            fact(6, "Boston Dynamics Spot", "quadruped", day + 86_400 * 3 + 180, 1.0), // a fourth clip: not shown
            fact(7, "Boston Dynamics Spot", "quadruped", day + 86_400 * 200, 1.0),     // months later: met again
        ];
        let e = group(&facts, "robot", 0.3);
        let shape: Vec<(&str, Vec<u64>)> = e.iter().map(|x| (x.label.as_str(), x.shots.iter().map(|s| s.item).collect())).collect();
        assert_eq!(
            shape,
            vec![("Boston Dynamics Spot", vec![1, 3, 5]), ("Pepper humanoid robot", vec![2]), ("quadruped robot", vec![4]), ("Boston Dynamics Spot", vec![7]),]
        );
        assert_eq!(e[0].shots[0].seen, vec![6.0], "shots carry where the robot is in view");
    }

    #[test]
    fn the_montage_spans_the_whole_show() {
        let day = 1_599_600_000;
        let facts: Vec<ClipFacts> = (0..10).map(|i| fact(i, &format!("robot model {i}"), "humanoid", day + 86_400 * 30 * i as i64, 1.0)).collect();
        let entries = group(&facts, "robot", 0.3);
        let items = |n| montage(&entries, n).iter().map(|s| s.item).collect::<Vec<_>>();
        assert_eq!(items(4), vec![0, 3, 6, 9]);
        assert_eq!(items(1), vec![0]);
        assert_eq!(items(50).len(), 10, "never more shots than entries");
        assert!(montage(&[], 5).is_empty());
    }

    #[test]
    fn chapters_follow_youtube_rules() {
        let c = |s: f64, t: &str| Chapter { seconds: s, title: t.into() };
        let chapters = [c(0.0, "Intro"), c(9.5, "2020"), c(65.0, "2022"), c(70.0, "2023"), c(200.0, "2024"), c(3700.0, "Outro")];
        assert_eq!(youtube_chapters(&chapters, 3720.0), "0:00 Intro / 2020\n1:05 2022 / 2023\n3:20 2024\n1:01:40 Outro");
        // a short last chapter joins the one before
        assert_eq!(youtube_chapters(&[c(0.0, "Intro"), c(20.0, "2020"), c(40.0, "2021"), c(60.0, "Outro")], 65.0), "0:00 Intro\n0:20 2020\n0:40 2021 / Outro");
        // fewer than three: none at all
        assert_eq!(youtube_chapters(&[c(0.0, "Intro"), c(30.0, "Outro")], 60.0), "");
    }

    #[test]
    fn labels_show_capitalised() {
        assert_eq!(display_label(" service robot"), "Service robot");
        assert_eq!(display_label("Boston Dynamics Spot"), "Boston Dynamics Spot");
        assert_eq!(display_label(""), "");
    }

    #[test]
    fn times_land_on_frames() {
        assert!((on_frame(2.3, 30.0) - 69.0 / 30.0).abs() < 1e-12);
        assert!((on_frame(10.016, 30.0) - 10.0).abs() < 1e-12);
        assert_eq!(on_frame(-1.0, 30.0), 0.0);
        assert!((on_frame(1.0, f64::NAN) - 1.0).abs() < 1e-12, "a bad rate falls back to 30 fps");
        // a title ending where the next starts: the same frame
        let (a, b) = (on_frame(2.3 + (7.73 - 2.3), 30.0), on_frame(7.73, 30.0));
        assert_eq!(a, b);
    }

    #[test]
    fn tidy_cleans_model_lines() {
        assert_eq!(tidy("1. \"Hello there\"\n"), "Hello there");
        assert_eq!(tidy(&"word ".repeat(100)).split_whitespace().count(), 40);
    }

    #[tokio::test]
    async fn the_script_falls_back_to_templates() {
        let day = 1_665_741_668;
        let entries = group(
            &[fact(1, "Boston Dynamics Spot", "quadruped", day, 1.0), fact(2, "Pepper humanoid robot", "humanoid", day + 86_400 * 400, 1.0)],
            "robot",
            0.3,
        );
        // no usable intro, then one usable line out of two
        let model = crate::llm::FixedModel::new(vec![
            json!({"intro": "Hi", "outro": "That is all for now, thanks for watching!"}),
            json!({"lines": ["I met Spot at an expo and it danced for the crowd."]}),
        ]);
        let s = write_script(&model, &Options::default(), &entries, &mut |_| {}).await;
        assert_eq!(s.lines.len(), 2);
        assert!(s.lines[0].contains("Spot"));
        assert_eq!(s.lines[1], "November 2023: the Pepper humanoid robot.");
        assert_eq!(s.templated, 1);
        assert!(s.intro.contains("every robot"), "{}", s.intro);
        assert!(s.outro.contains("thanks"));
        // a model that does not answer at all still yields a full script
        let none = crate::llm::FixedModel::new(vec![]);
        let s = write_script(&none, &Options::default(), &entries, &mut |_| {}).await;
        assert_eq!((s.lines.len(), s.templated), (2, 2));
    }

    #[tokio::test]
    async fn the_script_is_asked_for_in_batches_without_copied_headings() {
        let day = 1_599_600_000;
        let facts: Vec<ClipFacts> = (0..8).map(|i| fact(i, &format!("robot model {i}"), "humanoid", day + 86_400 * 30 * i as i64, 1.0)).collect();
        let entries = group(&facts, "robot", 0.3);
        assert_eq!(entries.len(), 8);
        let model = crate::llm::FixedModel::new(vec![
            json!({"intro": "Welcome to every robot I have ever filmed.", "outro": "Thanks for watching all of these robots!"}),
            json!({"lines": (0..6).map(|i| format!("September 2020 — robot model {i}: I saw this one at home.")).collect::<Vec<_>>()}),
            json!({"lines": ["It waved at me from across the room.", "short"]}),
        ]);
        let s = write_script(&model, &Options::default(), &entries, &mut |_| {}).await;
        let asked = model.asked.lock().unwrap();
        assert_eq!(asked.len(), 3, "the intro, then 6 + 2 entries");
        assert!(asked[1].1.contains("exactly 6 entries") && asked[2].1.contains("exactly 2 entries"));
        assert_eq!(s.lines[0], "I saw this one at home.", "the copied heading is not read out");
        assert_eq!(s.lines[6], "It waved at me from across the room.");
        assert_eq!(s.templated, 1, "a too-short line gets a template");
    }
}
