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
//!    (any voice, cloned voices included), shots around each clip's best moment (portrait and 4:3
//!    clips filling the frame around their robot), a name/date lower third, a transition from a
//!    rotating set of dramatic ones with a whoosh, the year as a title card with an impact
//!    whenever it changes; the counter (`graphics.counter`) across the whole show; the music —
//!    a generated bed (`audio.generate`) or the user's track ([`Options::music`], fitted to the
//!    show's length) — ducked under the narration, the cuts on its beat ([`Beats`]); an outro
//!    card; subtitles of the
//!    narration ([`crate::subtitles`]) on a caption track. Then (optionally) an export with the
//!    YouTube preset, and the subtitles beside it as `.srt` and `.vtt`.
//!
//! The script is written to be read: the voice says it with the show's respellings
//! ([`Script::pronounce`], [`Script::say`]), the subtitles show it as written.
//!
//! Everything goes through engine commands on the [`Executor`], so the same build drives the
//! headless CLI and the live app.

use std::collections::BTreeMap;

use filmcraft_engine::project::Delivery;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::analyze::ClipFacts;
use crate::llm::ChatModel;
use crate::look::{self, Look, Role, Theme};
use crate::subtitles::Cue;
use crate::{AutoCutError, Executor};

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;

pub(crate) fn ticks(seconds: f64) -> i64 {
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
    /// A music file to play under the show instead of a generated bed: fitted to the show's
    /// length, levelled and ducked like the bed, and the cuts land on its beats.
    pub music: Option<String>,
    /// Warp Stabilizer on every shot (handheld footage; the analysis decodes each shot once).
    pub stabilize: bool,
    /// Match the colour of an entry's other shots to its first full-frame shot
    /// (`lumetri.applyMatch`): clips of one robot filmed with different cameras or light.
    pub match_color: bool,
    /// Aim for a show this long: each entry breathes longer (more and longer shots, as long as
    /// its robot is in view) or shorter after its words. Never padded with footage where the
    /// robot is out of view, nor with held frames; [`Built::seconds`] says what it came to.
    pub target_minutes: Option<f64>,
    /// Clips whose shape is not the frame's (portrait phone video, 4:3) fill the frame, cropped
    /// around the subject with a slow tilt across it, as a 16:9 video should look; `false`: they
    /// sit whole over a blurred, dimmed copy of themselves.
    pub fill_frame: bool,
    /// A filled shot's crop follows its subject (`clip.autoReframe`: the subject tracker, seeded
    /// from the analysis' box) instead of tilting across a fixed point; shots where it is not
    /// found keep the tilt.
    pub follow_subject: bool,
    /// The look: a theme, the timeline, transitions from the catalogue, loudness, embedded
    /// captions, the SoundFont ([`crate::look`]); the default is today's look.
    pub look: Look,
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
            music: None,
            stabilize: false,
            match_color: false,
            target_minutes: None,
            fill_frame: true,
            follow_subject: true,
            look: Look::default(),
        }
    }
}

/// One clip shown in an entry.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
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
    /// The sampled moment where the subject is largest — the thumbnail's pick; absent in plans
    /// from before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub largest: Option<Moment>,
}

/// A sampled moment of a clip that shows the subject: seconds into the clip, the share of the
/// frame its box covers, and the box's centre (0..1 of the frame, x right, y down).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moment {
    pub seconds: f64,
    pub area: f64,
    #[serde(default = "Moment::middle")]
    pub center: [f64; 2],
}

impl Moment {
    fn middle() -> [f64; 2] {
        [0.5, 0.5]
    }
}

impl Shot {
    fn of(f: &ClipFacts) -> Self {
        let (seen, unseen): (Vec<&crate::analyze::FrameFact>, Vec<&crate::analyze::FrameFact>) = f.frames.iter().partition(|x| x.present);
        let largest = seen.iter().filter(|x| x.seconds.is_finite() && x.area.is_finite() && x.area > 0.0).max_by(|a, b| a.area.total_cmp(&b.area));
        Self {
            item: f.item,
            name: f.name.clone(),
            clip_seconds: f.seconds,
            best: f.best,
            seen: seen.iter().map(|x| x.seconds).collect(),
            unseen: unseen.iter().map(|x| x.seconds).collect(),
            largest: largest.map(|x| Moment {
                seconds: x.seconds,
                area: x.area.min(1.0),
                center: x.center.map(|c| if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.5 }),
            }),
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
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub number: usize,
    pub label: String,
    pub kind: String,
    pub description: String,
    pub unix: Option<i64>,
    pub shots: Vec<Shot>,
    /// Where it was filmed ("GITEX Global 2022, Dubai"): consecutive entries of one event share a
    /// chapter, opened by a title card with the event's name, and its lower third says so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// Checked facts about it (who makes it, when it launched, what it does) the voice-over may
    /// use — and nothing else: the script model is told to invent nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facts: Vec<String>,
    /// A scene's caption ("GITEX 2022 · Dubai World Trade Centre"; its label without one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
}

/// The `kind` of a scene entry: footage that shows no robot (a hall, a stage, a skyline) to
/// open a chapter with — no counter, no name lower third, its words optional.
pub const SCENE: &str = "scene";

impl Entry {
    /// A scene shot ([`SCENE`]), not a robot.
    pub fn is_scene(&self) -> bool {
        self.kind.trim().eq_ignore_ascii_case(SCENE)
    }

    /// What a scene's caption says: its `caption`, else its label, else its event.
    pub fn caption_text(&self) -> String {
        [self.caption.as_deref(), Some(self.label.as_str()), self.event.as_deref()]
            .into_iter()
            .flatten()
            .map(str::trim)
            .find(|s| !s.is_empty())
            .map(|s| s.chars().take(MAX_CAPTION_CHARS).collect())
            .unwrap_or_default()
    }

    /// The chapter it belongs to: its event, else the year it was filmed.
    pub fn section(&self) -> Option<String> {
        self.event.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).or_else(|| self.unix.map(|u| ymd(u).0.to_string()))
    }
}

/// Longest scene caption shown, characters.
const MAX_CAPTION_CHARS: usize = 80;

/// The voice-over, as written to be read: subtitles show it as it is, and the voice reads it
/// with the show's respellings ([`Script::pronounce`]) or a line's own spoken version
/// ([`Script::say`]).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Script {
    pub intro: String,
    pub lines: Vec<String>,
    pub outro: String,
    /// Lines the model did not deliver, filled from templates.
    #[serde(default)]
    pub templated: usize,
    /// How the voice says what the text spells for the eye (`{"Zaku II": "Zaku Two", "NodeMCU":
    /// "Node M C U"}`): every spoken part has these replaced — whole words, case as written,
    /// longest first — and nothing shown does.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pronounce: BTreeMap<String, String>,
    /// What the voice says instead of a part, when a respelling is not enough: by `"intro"`,
    /// `"outro"` or an entry's number (`"3"`). Subtitles still show the part as written.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub say: BTreeMap<String, String>,
    /// How each part is spoken — a narration delivery (`conversational`, `warm`, `excited`,
    /// `amazed`, `curious`, `calm`, `dramatic`, `plain`) — by `"intro"`, `"outro"` or an entry's
    /// number. A part without one (or with an unknown one) is spoken [`Script::delivery_of`]'s
    /// default. Voices that cannot take a delivery read every part plainly.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub delivery: BTreeMap<String, String>,
    /// Spec chips beside an entry's lower third, by entry number (`{"3": ["32.5 kg", "1.6
    /// m/s"]}`): up to three each, shown while its narration plays ([`look::chips_by_entry`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub chips: BTreeMap<String, Vec<String>>,
}

/// How the intro is spoken when the script does not say.
pub const INTRO_DELIVERY: Delivery = Delivery::Excited;
/// How every other part is spoken when the script does not say.
pub const DEFAULT_DELIVERY: Delivery = Delivery::Warm;
/// The deliveries the script model may choose from (plain and dramatic are left to people).
const SCRIPT_DELIVERIES: [Delivery; 6] = [Delivery::Conversational, Delivery::Warm, Delivery::Excited, Delivery::Amazed, Delivery::Curious, Delivery::Calm];

impl Script {
    /// How the part under `key` (`"intro"`, `"outro"` or an entry's number) is spoken.
    pub fn delivery_of(&self, key: &str) -> Delivery {
        self.delivery.get(key).and_then(|d| Delivery::from_id(d)).unwrap_or(if key == "intro" { INTRO_DELIVERY } else { DEFAULT_DELIVERY })
    }

    /// What the voice says for `text`, a part of the script shown as written under `key`
    /// (`"intro"`, `"outro"` or an entry's number): its [`Script::say`] when it has one, then
    /// the show's respellings.
    pub fn spoken(&self, key: &str, text: &str) -> String {
        let said = self.say.get(key).map(String::as_str).filter(|s| !s.trim().is_empty()).unwrap_or(text);
        respell(said, &self.pronounce)
    }
}

/// Most respellings applied, and the longest word replaced (a plan is user input).
const MAX_RESPELLINGS: usize = 2000;
const MAX_RESPELLED: usize = 200;

/// `text` with every whole-word occurrence of a key of `map` replaced by its value: the longest
/// key first at each place, case as written, never inside a longer word ("Go1" in "Go1," but not
/// in "Go10"), and never again in what a replacement wrote.
pub fn respell(text: &str, map: &BTreeMap<String, String>) -> String {
    let mut keys: Vec<(&str, &str)> =
        map.iter().filter(|(k, _)| !k.trim().is_empty() && k.len() <= MAX_RESPELLED).take(MAX_RESPELLINGS).map(|(k, v)| (k.as_str(), v.as_str())).collect();
    if keys.is_empty() {
        return text.to_string();
    }
    keys.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(b.0)));
    let word = |c: char| c.is_alphanumeric();
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    let mut i = 0;
    while let Some(rest) = text.get(i..) {
        let Some(c) = rest.chars().next() else { break };
        let hit = keys.iter().find(|(k, _)| {
            rest.starts_with(k)
                && !(k.starts_with(word) && prev.is_some_and(word))
                && !(k.ends_with(word) && rest.get(k.len()..).and_then(|r| r.chars().next()).is_some_and(word))
        });
        match hit {
            Some((k, v)) => {
                out.push_str(v);
                prev = k.chars().next_back();
                i += k.len();
            }
            None => {
                out.push(c);
                prev = Some(c);
                i += c.len_utf8();
            }
        }
    }
    out
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
    // (a scene comes only from a plan: an analysed clip that calls itself one is a robot)
    let kind = if named.kind.trim().eq_ignore_ascii_case(SCENE) { "other".to_string() } else { named.kind.clone() };
    Some(Entry {
        number,
        label: entry_label(named, subject),
        kind,
        description: named.description.clone(),
        unix: first.unix,
        shots: shown.into_iter().map(|(_, f)| Shot::of(f)).collect(),
        ..Entry::default()
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

/// A line for an entry the script has none for, said as a sentence: "In October 2022, I met
/// the red robot."
pub(crate) fn template_line(e: &Entry) -> String {
    // a scene says nothing unless its plan gives it words
    if e.is_scene() {
        return String::new();
    }
    let when = month_year(e.unix);
    let what = e.label.trim();
    if when.is_empty() { format!("Next up, I met the {what}.") } else { format!("In {when}, I met the {what}.") }
}

/// Clean one model sentence: one line, no quotes or numbering, at most ~40 words.
fn tidy(s: &str) -> String {
    let s = s.replace(['\n', '\r', '"', '“', '”'], " ");
    let s = s.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '#' || c == '-').trim();
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() <= MAX_LINE_WORDS {
        return words.join(" ");
    }
    // too long: end at the last full sentence within the limit, never mid-sentence (a cut line
    // is read out as it is); only a single over-long sentence is cut at the limit
    let kept = &words[..MAX_LINE_WORDS];
    match kept.iter().rposition(|w| w.ends_with(['.', '!', '?'])) {
        Some(i) => kept[..=i].join(" "),
        None => kept.join(" "),
    }
}

/// Most words one voice-over line keeps.
const MAX_LINE_WORDS: usize = 45;

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

/// The second line of a lower third: where and when ("GITEX Global 2022, Dubai · October"; the
/// month alone when the event's name already says the year).
fn lower_third_place(place: Option<&str>, when: &str) -> String {
    match (place, when.is_empty()) {
        (Some(p), false) => {
            let month = when.rsplit_once(' ').filter(|(_, y)| p.contains(y)).map_or(when, |(m, _)| m);
            format!("{p} · {month}")
        }
        (Some(p), true) => p.to_string(),
        (None, _) => when.to_string(),
    }
}

/// One item of the script request: number, date, name, where, what, and its facts.
fn item_listing(n: usize, e: &Entry) -> String {
    let mut s = format!("{n}. {} — {}", month_year(e.unix), e.label);
    if let Some(ev) = e.event.as_deref().map(str::trim).filter(|x| !x.is_empty()) {
        s.push_str(&format!(" (at {ev})"));
    }
    s.push_str(&format!(": {}", e.description));
    let facts: Vec<&str> = e.facts.iter().map(|f| f.trim()).filter(|f| !f.is_empty()).take(6).collect();
    if !facts.is_empty() {
        s.push_str(&format!(" Facts: {}", facts.join(" | ")));
    }
    s
}

/// Entries per request: small local models write a handful of lines reliably and fast, where
/// one long answer for a whole show can run past the request timeout.
const SCRIPT_BATCH: usize = 6;

const SCRIPT_SYSTEM: &str = "You write short, natural voice-over lines for a YouTube video, to be spoken aloud. The narrator is the person who filmed the clips, telling a friend about them in the first person: relaxed, warm and genuinely excited, with contractions (I'd, it's, that's) and sentences of different lengths. Never open a line with a bare date or place followed by a colon or a full stop (\"September 2020, at home: my first robot.\"): work the when and where into a sentence (\"Back in September 2020, I built my first robot at home.\"), and do not start two lines the same way. Use only the facts given; never invent names, brands, places or numbers, and never invent experiences or impressions the facts do not state (trying it, talking to it, how it sounded, what it felt like). Say \"next\" or \"then\", never \"lastly\" or \"finally\", except for the last item of the video. Answer with JSON only.";

/// The voice-over for `entries`, written by `model` from the analysis facts: the intro and
/// outro in one request, then the entries' lines a few at a time. Any line the model does not
/// deliver gets a plain template line, so the edit never depends on the model behaving.
pub async fn write_script(model: &dyn ChatModel, opts: &Options, entries: &[Entry], progress: &mut dyn FnMut(&str)) -> Script {
    let years = years(entries);
    // scenes (no robot) get no words from the model: a plan gives them their own, or none
    let robots: Vec<(usize, &Entry)> = entries.iter().enumerate().filter(|(_, e)| !e.is_scene()).collect();
    let video = format!(
        "The video: \"{}\" — every {} I have filmed{}, in the order I met them ({} in all). A counter on screen numbers them, so do not say the numbers.",
        opts.title,
        opts.subject,
        if years.is_empty() { String::new() } else { format!(" ({years})") },
        robots.len()
    );
    let names: Vec<&str> = robots.iter().map(|(_, e)| e.label.as_str()).collect();
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
    let mut delivery: BTreeMap<String, String> = BTreeMap::new();
    for (b, batch) in robots.chunks(SCRIPT_BATCH).enumerate() {
        let listing: String = batch.iter().enumerate().map(|(i, (_, e))| item_listing(i + 1, e)).collect::<Vec<_>>().join("\n");
        let with_facts = batch.iter().any(|(_, e)| !e.facts.is_empty());
        let ask = if with_facts {
            "Each is two or three short spoken sentences (at most 40 words) that say when and where I saw it and what it is or does, and weave in one or two of its facts naturally, as I would tell a friend; use only the facts listed for that item; do not start with the item's heading."
        } else {
            "Each is one or two short spoken sentences (at most 25 words) that say when it was and what it is or does, as I would tell a friend; do not start with the item's heading."
        };
        let moods: Vec<&str> = SCRIPT_DELIVERIES.iter().map(|d| d.id()).collect();
        let user = format!(
            "{video}\n\nWrite \"lines\": exactly {} entries, one per item below in the same order. {ask} Also write \"deliveries\": for each line, how I say it, one of {} (amazed for something impressive, curious for something odd, calm for something quiet).\n\nItems:\n{listing}\n\nJSON: {{\"lines\": [\"...\"], \"deliveries\": [\"...\"]}}",
            batch.len(),
            moods.join(", ")
        );
        match model.complete_json(SCRIPT_SYSTEM, &user, &[]).await {
            Ok(v) => {
                let lines = v.get("lines").and_then(Value::as_array).cloned().unwrap_or_default();
                let said = v.get("deliveries").and_then(Value::as_array).cloned().unwrap_or_default();
                for (i, (at, e)) in batch.iter().enumerate() {
                    let line = lines.get(i).and_then(Value::as_str).map(tidy).map(|l| strip_heading(&l, e)).filter(|l| l.split_whitespace().count() >= 3);
                    // only a delivery the model may choose, and only for a line it wrote
                    let mood = said.get(i).and_then(Value::as_str).and_then(Delivery::from_id).filter(|d| SCRIPT_DELIVERIES.contains(d));
                    if let (Some(_), Some(d)) = (&line, mood) {
                        delivery.insert(e.number.to_string(), d.id().to_string());
                    }
                    if let Some(slot) = written.get_mut(*at) {
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
            None if e.is_scene() => String::new(),
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
        outro = format!("That's {} {}s so far. Thanks for watching, and see you at the next one.", robots.len(), opts.subject);
    }
    Script { intro, lines, outro, templated, delivery, ..Script::default() }
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
    /// Narration lines read again because the voice did not say them ([`narrate`]).
    pub narration_retakes: usize,
    /// Where each part starts: the intro, each year, the outro.
    pub chapters: Vec<Chapter>,
    pub notes: Vec<String>,
    pub export: Option<Value>,
    /// Where each entry and its shots landed on the timeline (what [`crate::qa`] checks).
    pub layout: Vec<PlacedEntry>,
    /// Each narration line as placed: when it is spoken and its text as shown (not respelled).
    pub narration: Vec<Cue>,
    /// The subtitles ([`crate::subtitles::cues`] of the narration), also on the sequence's
    /// caption track "Subtitles".
    pub subtitles: Vec<Cue>,
    /// The subtitle files written next to the export (`<out>.srt`, `<out>.vtt`).
    pub subtitle_files: Vec<String>,
    /// The music under the show, and how many cuts landed on its beat.
    pub music: Option<Music>,
    /// The YouTube thumbnail written next to the export (`<out>.thumbnail.jpg`).
    pub thumbnail: Option<String>,
    /// The length aimed for ([`Options::target_minutes`]), seconds; [`Built::seconds`] is the
    /// length it came to.
    pub target_seconds: Option<f64>,
    /// Scene entries among [`Built::entries`] ([`Entry::is_scene`]).
    pub scenes: usize,
    /// The theme as drawn — its fonts as installed — when one was asked for ([`Look::style`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<Theme>,
    /// The timeline along the bottom, as laid out and drawn ([`Look::timeline`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeline: Option<look::Timeline>,
    /// Transitions from the catalogue, as placed ([`Look::transitions`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<PlacedTransition>,
    /// What `transitions.snapToBeats` did with them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_snap: Option<Value>,
    /// What `audio.normalizeLoudness` measured and did ([`Look::loudness`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loudness: Option<Value>,
}

/// A transition from the catalogue as placed: its id, effect, where (`within` an entry,
/// `between` entries, into a `chapter`), the sound coupled to it, and the entry it leads into.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlacedTransition {
    pub id: u64,
    pub effect: String,
    pub at: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sound: Option<&'static str>,
    pub entry: usize,
}

/// An entry as built: its span on the timeline and its shots, seconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacedEntry {
    pub number: usize,
    pub label: String,
    pub start: f64,
    pub end: f64,
    pub shots: Vec<PlacedShot>,
    /// A scene ([`Entry::is_scene`]): no robot to look for in its shots.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub scene: bool,
    /// Its name lower third (a robot's) or caption (a scene's), and its spec chips: graphic clips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_third: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chips: Vec<u64>,
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

pub(crate) async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
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
    /// Width over height of the frame.
    aspect: f64,
    /// Warp Stabilizer on the shots ([`Options::stabilize`]).
    stabilize: bool,
    /// Odd-shaped clips fill the frame ([`Options::fill_frame`]).
    fill: bool,
    /// Filled shots follow their subject ([`Options::follow_subject`]).
    follow: bool,
    /// The frame's size, sequence pixels.
    frame: (f64, f64),
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
    /// The overlays' video tracks (indices): the timeline's base, picked-out and marker tracks,
    /// and one for each spec chip.
    timeline: Option<[usize; 3]>,
    chips: Vec<usize>,
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
async fn mix_down(ex: &dyn Executor, mix: &Mix, narrations: &[u64], music: &[u64], limit: bool, notes: &mut Vec<String>) {
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
    if !music.is_empty() {
        step("essentialSound.applyPreset", json!({"clips": music, "preset": "Duck Under Dialogue", "type": "music"})).await;
        step("essentialSound.autoMatch", json!({"clips": music, "target": MUSIC_LUFS})).await;
    }
    if !narrations.is_empty() {
        let ducked: Vec<u64> = mix.sounds.iter().chain(music).copied().collect();
        step("essentialSound.generateDucking", json!({"clips": ducked})).await;
    }
    // (a mix normalised to a loudness target gets its limiter from the normalisation)
    if limit && step("mixer.addInsert", json!({"strip": "Mix", "effect": "hard_limiter"})).await {
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

/// How much larger a shot's picture grows a second in its push-in, and at most.
const PUSH_PER_SECOND: f64 = 0.015;
const PUSH_MAX: f64 = 0.06;

/// A slow push-in over the shot `clip` from `t0` to `t1` (seconds, on frames): its scale from
/// `scale` (as fitted) to [`PUSH_PER_SECOND`] larger a second, at most [`PUSH_MAX`], eased out
/// of the first keyframe and into the last — keyframes on its Motion, which live in the media's
/// time, so a shot slowed afterwards keeps the whole move.
async fn push_in(ex: &dyn Executor, clip: u64, scale: f64, t0: f64, t1: f64, fps: f64) -> Result<(), AutoCutError> {
    // the last keyframe on the last frame, where the move ends
    let last = on_frame(t1 - 1.0 / fps, fps);
    if !(scale.is_finite() && scale > 0.0) || last <= t0 {
        return Ok(());
    }
    let grow = (PUSH_PER_SECOND * (t1 - t0)).min(PUSH_MAX);
    for (at, value) in [(t0, scale), (last, scale * (1.0 + grow))] {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": value, "time": ticks(at), "keyframe": true})).await?;
    }
    for (at, ease) in [(t0, "easeOut"), (last, "easeIn")] {
        run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": "motion", "param": "scale", "time": ticks(at), "interpolation": ease})).await?;
    }
    Ok(())
}

/// How far a filled shot's crop travels across its subject at least, as a share of the
/// picture (a tall subject gets a longer tilt: [`reframe_path`]).
const REFRAME_TILT: f64 = 0.06;

/// Where a filled shot sits, start and end: the Motion position (sequence pixels) that puts the
/// subject's centre `focus` (0..1 of the clip, x right, y down) mid-frame, as far as the picture
/// still covers the `frame`, moving along the axis the fill crops (a tilt down a portrait clip)
/// by [`REFRAME_TILT`], or across the whole subject when it is taller than the frame shows: its
/// extent is taken from `area`, the share of the clip its box covers (a robot as tall as wide
/// in the clip's shape, a little taller), so a big robot is revealed top to bottom over the
/// shot. The picture of a clip of shape `aspect` (width over height) filling the frame is as
/// wide as the frame (a narrower clip) or as tall (a wider one). `None` for sizes that make no
/// sense.
fn reframe_path(aspect: f64, frame: (f64, f64), focus: [f64; 2], area: f64) -> Option<([f64; 2], [f64; 2])> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    let (fw, fh) = frame;
    if !(ok(aspect) && ok(fw) && ok(fh)) {
        return None;
    }
    let shown = if aspect < fw / fh { (fw, fw / aspect) } else { (fh * aspect, fh) };
    // the share of the picture the frame shows on each axis, and where its centre may go
    let vis = [(fw / shown.0).min(1.0), (fh / shown.1).min(1.0)];
    let clamp = |axis: usize, x: f64| {
        let (lo, hi) = (vis[axis] / 2.0, 1.0 - vis[axis] / 2.0);
        if x.is_finite() { x.clamp(lo, hi) } else { 0.5 }
    };
    let centre = [clamp(0, focus[0]), clamp(1, focus[1])];
    let axis = if vis[1] <= vis[0] { 1 } else { 0 };
    // the subject's extent along that axis (0..1), from its box's share of the clip
    let extent = if area.is_finite() && area > 0.0 { (area.min(1.0).sqrt() * 1.3).min(1.0) } else { 0.0 };
    let d = REFRAME_TILT.max((extent - vis[axis]) / 2.0).min((1.0 - vis[axis]) / 2.0).max(0.0);
    let (mut from, mut to) = (centre, centre);
    from[axis] = clamp(axis, centre[axis] - d);
    to[axis] = clamp(axis, centre[axis] + d);
    // the picture's point `f` at the frame's middle: the picture moves the other way
    let pos = |f: [f64; 2]| [fw / 2.0 + (0.5 - f[0]) * shown.0, fh / 2.0 + (0.5 - f[1]) * shown.1];
    Some((pos(from), pos(to)))
}

/// Keyframe a filled shot's position along [`reframe_path`] from `t0` to `t1`, eased like the
/// push-in.
#[allow(clippy::too_many_arguments)]
async fn reframe(ex: &dyn Executor, clip: u64, aspect: f64, frame: (f64, f64), focus: Moment, t0: f64, t1: f64, fps: f64) -> Result<(), AutoCutError> {
    let last = on_frame(t1 - 1.0 / fps, fps);
    let Some((from, to)) = reframe_path(aspect, frame, focus.center, focus.area) else { return Ok(()) };
    if last <= t0 {
        return Ok(());
    }
    for (at, p) in [(t0, from), (last, to)] {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "position", "value": p, "time": ticks(at), "keyframe": true})).await?;
    }
    for (at, ease) in [(t0, "easeOut"), (last, "easeIn")] {
        run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": "motion", "param": "position", "time": ticks(at), "interpolation": ease})).await?;
    }
    Ok(())
}

/// Shots shorter than this (seconds) are not followed.
const FOLLOW_MIN_SECONDS: f64 = 1.2;
/// Frames a second the subject tracker looks at in a shot (a calm crop needs few).
const FOLLOW_FPS: f64 = 8.0;

/// Follow the subject through the filled shot `clip` (the clip's media `[src, src + dur)`, shape
/// `aspect`): `clip.autoReframe` keyframes its position with the tracker (its scale, with the
/// push-in, stays), seeded from `focus` — the analysis' box — when that moment is in the shot,
/// else from what stands out. Whether the subject was followed; a shot where it was not found,
/// or that failed, is left for the tilt ([`reframe`]).
async fn follow_subject(ex: &dyn Executor, clip: u64, aspect: f64, src: f64, dur: f64, focus: Option<Moment>) -> bool {
    let mut p = json!({"clips": [clip], "keepScale": true, "motion": "smooth", "fallback": "center", "analysisFps": FOLLOW_FPS, "wait": true});
    if let Some(m) = focus.filter(|m| m.seconds.is_finite() && m.seconds >= src && m.seconds <= src + dur && m.area.is_finite() && m.area > 0.0) {
        let aspect = if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 };
        let (w, h) = ((m.area.min(1.0) / aspect).sqrt().clamp(0.02, 1.0), (m.area.min(1.0) * aspect).sqrt().clamp(0.02, 1.0));
        let c = m.center.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 });
        p["subject"] = json!({"x": c[0] - w / 2.0, "y": c[1] - h / 2.0, "w": w, "h": h, "seconds": m.seconds});
    }
    match run(ex, "clip.autoReframe", p).await {
        Ok(r) => r["reframe"]["followed"].as_u64().is_some_and(|n| n > 0),
        Err(_) => false,
    }
}

/// Place `item`'s `[src, src + dur)` at `t` (both ends on frames): foreground on V2 (with its
/// sound on A1, which [`mix_down`] levels) with a slow push-in ([`push_in`]) and — when the
/// show is stabilized ([`Options::stabilize`]) and `stabilize` — Warp Stabilizer. A clip whose
/// shape is not the frame's fills it ([`Options::fill_frame`]), its crop on the subject's
/// centre `focus` (0..1 of the clip; the middle without one) and tilting across it
/// ([`reframe`]); otherwise it is fitted whole over a blurred, dimmed copy filling V1 behind
/// it, which stays still.
#[allow(clippy::too_many_arguments)]
async fn place_shot(
    ex: &dyn Executor,
    tr: &Tracks,
    item: u64,
    t: f64,
    src: f64,
    dur: f64,
    stabilize: bool,
    focus: Option<Moment>,
    follow: bool,
) -> Result<Placed, AutoCutError> {
    let frame_aspect = tr.aspect;
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
    let size = media_size(ex, item).await;
    let aspect = size.map(|(w, h)| w / h.max(1.0)).unwrap_or(frame_aspect);
    let odd = (aspect - frame_aspect).abs() > 0.05;
    let fill = odd && tr.fill && size.is_some_and(|(w, h)| w > 0.0 && h > 0.0);
    let fit = run(ex, if fill { "clip.fillFrame" } else { "clip.fitToFrame" }, json!({"clips": [fg]})).await?;
    let scale = fit["clips"].as_array().and_then(|a| a.iter().find(|c| c["clip"].as_u64() == Some(fg))).and_then(|c| c["scale"].as_f64());
    if let Some(scale) = scale {
        // cosmetic, like the transitions: a shot without its push-in (or its reframing) is
        // still right
        push_in(ex, fg, scale, t, end, tr.fps).await.ok();
        // a still (the thumbnail's second) has nothing to follow
        let followed = fill && tr.follow && follow && dur > FOLLOW_MIN_SECONDS && follow_subject(ex, fg, aspect, src, dur, focus).await;
        if fill && !followed {
            let focus = focus.unwrap_or(Moment { seconds: 0.0, area: 0.0, center: Moment::middle() });
            reframe(ex, fg, aspect, tr.frame, focus, t, end, tr.fps).await.ok();
        }
    }
    if stabilize && tr.stabilize {
        run(ex, "effects.apply", json!({"clips": [fg], "effect": "warp_stabilizer"})).await?;
    }
    let mut bg = None;
    if odd && !fill {
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

/// Match the colour of an entry's shots to its first full-frame shot (`lumetri.applyMatch`, the
/// middle of that shot as the reference frame), so the clips of one robot — often from other
/// cameras, days or light — look like one scene. Shots from the reference's own clip are left
/// alone, and so is an entry without a full-frame shot (a portrait reference frame would hold
/// the blurred copy behind it). `full[i]`: shot `i` fills the frame. Failures are noted.
async fn match_colour(ex: &dyn Executor, shots: &[PlacedShot], full: &[bool], notes: &mut Vec<String>) {
    let Some(r) = shots.iter().zip(full).find(|(_, f)| **f).map(|(s, _)| s) else { return };
    let reference = (r.start + r.end) / 2.0;
    for s in shots.iter().filter(|s| s.item != r.item) {
        // the frame of the shot that is matched: its middle (the playhead's, pinned to the clip)
        let mid = (s.start + s.end) / 2.0;
        let matched = match run(ex, "playhead.set", json!({"time": ticks(mid)})).await {
            Ok(_) => run(ex, "lumetri.applyMatch", json!({"clip": s.clip, "referenceTime": ticks(reference), "faceDetection": false})).await,
            Err(e) => Err(e),
        };
        if let Err(e) = matched {
            notes.push(format!("colour match at {mid:.2}s: {e}"));
        }
    }
}

/// How long a lower third takes to slide and fade in (and out), seconds, and how far it slides
/// (a share of the frame's width).
const SLIDE: f64 = 0.4;
const FADE: f64 = 0.25;
const SLIDE_BY: f64 = 0.03;

/// Animate the graphic clip `clip` (on screen from `t0` to `t1`, on frames) as a lower third:
/// in from the left while fading in, decelerating into place; out the same way, accelerating —
/// keyframes on its Motion position (the frame's centre, `w`×`h`) and its Opacity.
pub(crate) async fn slide_in_out(ex: &dyn Executor, clip: u64, t0: f64, t1: f64, (w, h): (f64, f64), fps: f64) -> Result<(), AutoCutError> {
    let last = on_frame(t1 - 1.0 / fps, fps);
    if last - t0 < 2.0 * SLIDE {
        return Ok(());
    }
    let (home, away) = ([w / 2.0, h / 2.0], [w / 2.0 - w * SLIDE_BY, h / 2.0]);
    let (shown, gone) = (on_frame(t0 + SLIDE, fps), on_frame(last - SLIDE, fps));
    let keys = [
        ("motion", "position", t0, json!(away), "linear"),
        ("motion", "position", shown, json!(home), "easeIn"),
        ("motion", "position", gone, json!(home), "easeOut"),
        ("motion", "position", last, json!(away), "linear"),
        ("opacity", "opacity", t0, json!(0.0), "linear"),
        ("opacity", "opacity", on_frame(t0 + FADE, fps), json!(100.0), "linear"),
        ("opacity", "opacity", on_frame(last - FADE, fps), json!(100.0), "linear"),
        ("opacity", "opacity", last, json!(0.0), "linear"),
    ];
    for (effect, param, at, value, _) in &keys {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": effect, "param": param, "value": value, "time": ticks(*at), "keyframe": true})).await?;
    }
    for (effect, param, at, _, ease) in keys.iter().filter(|k| k.4 != "linear") {
        run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": effect, "param": param, "time": ticks(*at), "interpolation": ease})).await?;
    }
    Ok(())
}

/// A text title clip on video track `track` (an index), from `t` for `dur` seconds, both ends on
/// frames.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn title(
    ex: &dyn Executor,
    track: usize,
    fps: f64,
    text: &str,
    t: f64,
    dur: f64,
    size: f64,
    pos: (f64, f64),
    props: Value,
) -> Result<u64, AutoCutError> {
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
    /// Each line placed: when, and its shown text.
    lines: Vec<Cue>,
    /// Lines read again ([`narrate`]); whether what the voice said can be checked (None: not
    /// tried yet).
    retakes: usize,
    can_check: Option<bool>,
    /// The cloned voice's own pitch, Hz (None: not asked yet; Some(None): unknown).
    reference_pitch: Option<Option<f64>>,
    notes: Vec<String>,
}

/// A cloned voice's line is heard back (Whisper) and read again when less than this share of
/// its words come back — a garbled generation scores near nothing, a good one most of them
/// (names a recogniser mishears cost a little).
const HEARD_ENOUGH: f64 = 0.5;
/// Readings of one line at most (the first and two more takes).
const MAX_TAKES: u32 = 3;
/// A reading whose median pitch is more than this times its voice sample's overacted. Live
/// (`median_pitch`), a sample at 84 Hz read 86–111 Hz in good lines (a reading sits a little
/// above its sample, more so when excited), so only a reading pitched far up is caught; one
/// that merely sounds over-excited (116 Hz) is not, and is better handled by its delivery.
const PITCH_LIMIT: f64 = 1.45;

/// The share of `said`'s words (letters only, three or more) that `heard` contains, a word
/// counting when the recogniser wrote it or one starting with its first four letters ("Neolix"
/// for "Neo-licks" does not, "robots" for "robot" does); 1 for a line without such words.
pub(crate) fn heard_share(said: &str, heard: &str) -> f64 {
    let words = |t: &str| -> Vec<String> { t.split(|c: char| !c.is_alphabetic()).filter(|w| w.chars().count() >= 3).map(str::to_lowercase).collect() };
    let said = words(said);
    if said.is_empty() {
        return 1.0;
    }
    let heard = words(heard);
    let stem = |w: &str| w.chars().take(4).collect::<String>();
    let found = said.iter().filter(|w| heard.iter().any(|h| h == *w || (h.chars().count() >= 4 && w.chars().count() >= 4 && stem(h) == stem(w)))).count();
    found as f64 / said.len() as f64
}

/// What the voice said in the narration `item`: transcribed, then read back. `None` when it
/// cannot be heard (no speech model).
async fn heard(ex: &dyn Executor, item: u64) -> Option<String> {
    ex.execute("transcript.generate", json!({"items": [item], "language": "en"})).await.ok()?;
    let r = ex.execute("transcript.inspect", json!({"item": item})).await.ok()?;
    r["text"].as_str().map(str::to_string)
}

/// Narrate a part of the script at `t` on the voice track — the voice reads `say`, the
/// subtitles show `shown` — and return its length in seconds (0 without a voice).
async fn narrate(
    ex: &dyn Executor,
    tr: &Tracks,
    voice: Option<&str>,
    say: &str,
    shown: &str,
    delivery: Delivery,
    t: f64,
    narrations: &mut Narrations,
) -> Result<f64, AutoCutError> {
    let Some(voice) = voice else { return Ok(0.0) };
    let create = async |take: u32| {
        run(ex, "tts.create", json!({"text": say, "voice": voice, "delivery": delivery.id(), "take": take, "time": ticks(t), "track": tr.voice})).await
    };
    let mut r = create(0).await?;
    // a cloned voice can garble a line, or overact it (a strained, much higher voice): hear it
    // back and measure its pitch, and read it again (another take) when it did not say it or
    // came out far above the voice's own pitch, keeping the best reading
    if voice.starts_with("clone:") {
        if narrations.reference_pitch.is_none() {
            let r = ex.execute("tts.clonedVoices", json!({"voice": voice})).await.ok();
            narrations.reference_pitch = Some(r.and_then(|r| r["pitchHz"].as_f64()).filter(|p| p.is_finite() && *p > 0.0));
        }
        let reference = narrations.reference_pitch.flatten();
        let mut best: Option<(f64, u32)> = None;
        for take in 0..MAX_TAKES {
            let Some(item) = r["item"].as_u64() else { break };
            let share = if narrations.can_check == Some(false) {
                1.0
            } else {
                match heard(ex, item).await {
                    Some(text) => {
                        narrations.can_check = Some(true);
                        heard_share(say, &text)
                    }
                    None => {
                        narrations.can_check = Some(false);
                        narrations.notes.push("narration: lines were not heard back (no speech model): `transcript.downloadModel`".into());
                        1.0
                    }
                }
            };
            let pitch = r["pitchHz"].as_f64();
            let overacted = matches!((pitch, reference), (Some(p), Some(rf)) if p > rf * PITCH_LIMIT);
            if share >= HEARD_ENOUGH && !overacted {
                break;
            }
            // a reading that says the line beats one that does not; overacting costs half
            let score = share - if overacted { 0.5 } else { 0.0 };
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, take));
            }
            let why = if share < HEARD_ENOUGH { format!("the voice said {:.0}% of the line", share * 100.0) } else { "the voice overacted it".to_string() };
            let last = take + 1 >= MAX_TAKES;
            // the best reading when none was good enough (cached: nothing is read again)
            let next = if last { best.map_or(take, |(_, k)| k) } else { take + 1 };
            if last && next == take {
                narrations.notes.push(format!("narration at {t:.1}s: {why} in its best take"));
                break;
            }
            if let Some(c) = r["clip"].as_u64() {
                run(ex, "edit.clear", json!({"clips": [c]})).await?;
            }
            r = create(next).await?;
            narrations.retakes += 1;
            if last {
                narrations.notes.push(format!("narration at {t:.1}s: no take was clean ({why} in the last); the best kept"));
                break;
            }
        }
    }
    if let Some(c) = r["clip"].as_u64() {
        narrations.clips.push(c);
    }
    if r["cached"].as_bool() == Some(true) {
        narrations.cached += 1;
    }
    let seconds = r["duration"].as_f64().map(|d| d / TPS).filter(|d| d.is_finite() && *d > 0.0).unwrap_or(0.0);
    if seconds > 0.0 {
        narrations.lines.push(Cue { start: t, end: t + seconds, text: shown.to_string() });
    }
    Ok(seconds)
}

/// The subtitles on a new caption track of the sequence ("Subtitles", shown in the app, never
/// burned into the export), one caption per cue; its id. A step that fails is noted, never
/// fatal: the video is still right without subtitles.
async fn caption_track(ex: &dyn Executor, cues: &[Cue], notes: &mut Vec<String>) -> Option<u64> {
    let track = match run(ex, "captions.newTrack", json!({"format": "Subtitle", "name": "Subtitles", "language": "en"})).await {
        Ok(r) => r["track"].as_u64()?,
        Err(e) => {
            notes.push(format!("subtitles: {e}"));
            return None;
        }
    };
    for c in cues {
        let p = json!({"track": track, "text": c.text, "time": ticks(c.start), "durationSeconds": c.end - c.start});
        if let Err(e) = run(ex, "captions.add", p).await {
            notes.push(format!("subtitles: at {:.2}s: {e}", c.start));
        }
    }
    Some(track)
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
/// a clip when one is too short; what a shot cannot fill goes to the next one. The windows can
/// add up to less than `seconds` when the subject is rarely in view; [`whole_windows`] is the
/// fallback that always fills the time.
pub fn windows(e: &Entry, seconds: f64) -> Vec<(Shot, f64, f64)> {
    in_view_windows(e, seconds).into_iter().map(|w| (w.shot, w.start, w.len)).collect()
}

/// Windows that fill `seconds` whatever is in view: each shot's share in one window, placed
/// where the subject shows most ([`Shot::window_start`]).
pub fn whole_windows(e: &Entry, seconds: f64) -> Vec<(Shot, f64, f64)> {
    filling_windows(e, seconds).into_iter().map(|w| (w.shot, w.start, w.len)).collect()
}

/// A window cut from a shot: from `start` for `len` seconds of its clip, and how long it could
/// run from `start` without leaving its stretch of the subject in view (or, for a window that
/// fills its time whatever is in view, the clip).
#[derive(Clone, Debug)]
struct Window {
    shot: Shot,
    start: f64,
    len: f64,
    reach: f64,
}

/// [`windows`], with each window's reach.
fn in_view_windows(e: &Entry, seconds: f64) -> Vec<Window> {
    let shots: Vec<&Shot> = e.shots.iter().take(shot_count(seconds)).collect();
    let each = seconds / shots.len().max(1) as f64;
    let mut out = Vec::new();
    // time a shot could not fill with its robot in view goes to the next one
    let mut carry = 0.0;
    for s in shots {
        let share = each + carry;
        let mut spans: Vec<(f64, f64)> = s.spans().into_iter().filter(|(a, b)| b - a >= MIN_WINDOW.min(share)).collect();
        // the longest stretches first, the one holding the best moment first among equals
        spans.sort_by(|x, y| (y.1 - y.0).total_cmp(&(x.1 - x.0)).then_with(|| (s.best < x.0 || s.best > x.1).cmp(&(s.best < y.0 || s.best > y.1))));
        let mut picked: Vec<(f64, f64, f64)> = Vec::new();
        let mut left = share;
        for (a, b) in spans {
            if left < MIN_WINDOW * 0.5 {
                break;
            }
            let len = left.min(b - a);
            let start = (s.best - len * 0.4).clamp(a, (b - len).max(a));
            picked.push((start, len, (b - start).max(len)));
            left -= len;
        }
        carry = left.max(0.0);
        picked.sort_by(|x, y| x.0.total_cmp(&y.0));
        out.extend(picked.into_iter().map(|(start, len, reach)| Window { shot: s.clone(), start, len, reach }));
    }
    out
}

/// [`whole_windows`], with each window's reach (to the clip's end, less a margin).
fn filling_windows(e: &Entry, seconds: f64) -> Vec<Window> {
    let shots: Vec<&Shot> = e.shots.iter().take(shot_count(seconds)).collect();
    let each = seconds / shots.len().max(1) as f64;
    shots
        .into_iter()
        .map(|s| {
            let usable = (s.clip_seconds - 0.4).max(0.5);
            let len = each.min(usable);
            let start = s.window_start(len);
            Window { shot: s.clone(), start, len, reach: (s.clip_seconds - 0.2 - start).max(len) }
        })
        .collect()
}

/// The beats of the music under the show, in seconds of the timeline: the ones measured in the
/// track ([`Beats::measured`]), continued at its tempo before the first and past the last (a
/// looped or extended track keeps its beat); or the steady grid of a generated bed from 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    times: Vec<f64>,
    period: f64,
}

/// Most measured beats kept (a long hostile report could hold any number).
const MAX_BEATS: usize = 100_000;

impl Beats {
    /// A steady grid from 0 at `bpm` (40..200 beats a minute, as `audio.generate` plays it).
    pub fn grid(bpm: f64) -> Self {
        let bpm = if bpm.is_finite() { bpm.clamp(40.0, 200.0) } else { 96.0 };
        Self { times: vec![0.0], period: 60.0 / bpm }
    }

    /// Measured beat `times` (seconds of the timeline) at about `period` seconds a beat; `None`
    /// when there are none or the tempo makes no sense (outside 20..300 beats a minute).
    pub fn measured(times: &[f64], period: f64) -> Option<Self> {
        if !period.is_finite() || !(0.2..=3.0).contains(&period) {
            return None;
        }
        let mut t: Vec<f64> = times.iter().copied().filter(|x| x.is_finite() && *x >= 0.0).take(MAX_BEATS).collect();
        t.sort_by(f64::total_cmp);
        t.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        (!t.is_empty()).then_some(Self { times: t, period })
    }

    /// The beat map of generated music (`beats`: exact ticks from its start, as
    /// `audio.generate` returns them) placed at `start` seconds of the timeline; `None` for
    /// fewer than two beats.
    pub fn from_ticks(beats: &[i64], start: f64) -> Option<Self> {
        let times: Vec<f64> = beats.iter().take(MAX_BEATS).map(|t| start + *t as f64 / TPS).collect();
        let period = times.get(1).zip(times.first()).map(|(b, a)| b - a)?;
        Self::measured(&times, period)
    }

    /// Seconds per beat.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Beat `k` of the endless sequence: the measured ones at `0..n`, the tempo's grid around them.
    fn at(&self, k: i64) -> Option<f64> {
        let n = i64::try_from(self.times.len()).ok()?;
        let (first, last) = (*self.times.first()?, *self.times.last()?);
        let t = if k < 0 {
            first + k as f64 * self.period
        } else if k >= n {
            last + (k - n + 1) as f64 * self.period
        } else {
            *self.times.get(usize::try_from(k).ok()?)?
        };
        (t.is_finite() && t >= 0.0).then_some(t)
    }

    /// The index of the last beat at or before `x` (it may be before 0 s, and so not exist).
    fn index_before(&self, x: f64) -> Option<i64> {
        let n = self.times.len();
        let (first, last) = (*self.times.first()?, *self.times.last()?);
        let k = if x < first {
            ((x - first) / self.period).floor()
        } else if x >= last {
            (n - 1) as f64 + ((x - last) / self.period).floor()
        } else {
            self.times.partition_point(|b| *b <= x) as f64 - 1.0
        };
        (k.is_finite() && k.abs() < 1e12).then_some(k as i64)
    }

    /// The beat nearest `t` among those in `[lo, hi]`, if any.
    pub fn nearest(&self, t: f64, lo: f64, hi: f64) -> Option<f64> {
        if !(t.is_finite() && lo.is_finite() && hi.is_finite()) || hi < lo {
            return None;
        }
        let k = self.index_before(t)?;
        let mut best: Option<f64> = None;
        // the beats around `t`, then (when none of them is in range) the first one from `lo`
        let around = (k - 1..=k + 2).filter_map(|i| self.at(i));
        let from_lo = self.index_before(lo).and_then(|i| (i..=i + 1).filter_map(|i| self.at(i)).find(|b| *b >= lo));
        for b in around.chain(from_lo) {
            if b >= lo && b <= hi && best.is_none_or(|x| (b - t).abs() < (x - t).abs()) {
                best = Some(b);
            }
        }
        best
    }

    /// The first beat at or after `t`.
    pub fn next(&self, t: f64) -> Option<f64> {
        let k = self.index_before(t)?;
        (k..=k + 2).filter_map(|i| self.at(i)).find(|b| *b >= t - 1e-9)
    }

    /// The beats from `from` to `to` seconds, at most [`MAX_BEATS`], the tempo's grid going on
    /// past the measured ones.
    pub fn list(&self, from: f64, to: f64) -> Vec<f64> {
        if !(from.is_finite() && to.is_finite()) || to < from {
            return Vec::new();
        }
        let (Some(k0), Some(k1)) = (self.index_before(from.max(0.0)), self.index_before(to)) else { return Vec::new() };
        (k0..=k1.saturating_add(1)).take(MAX_BEATS + 2).filter_map(|k| self.at(k)).filter(|b| *b >= from - 1e-9 && *b <= to).take(MAX_BEATS).collect()
    }

    /// How far `t` is from the nearest beat, seconds.
    pub fn distance(&self, t: f64) -> f64 {
        self.nearest(t, t - self.period * 2.0, t + self.period * 2.0).map_or(f64::INFINITY, |b| (b - t).abs())
    }
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

/// Furthest a cut moves to land on a beat, seconds (and never more than half a beat).
const BEAT_SNAP: f64 = 0.4;

/// Where an entry starting at `t` ends on a beat: its windows `wins` (played at `slow` speed)
/// end at the nearest beat within [`BEAT_SNAP`] — the last window a little shorter (never the
/// entry under `min_len` seconds, which carries its words, nor the window under half a
/// second) or a little longer, only as far as its reach (the robot stays in view). The last
/// window is changed to match; `None` (nothing changed) when no beat is that close.
fn snap_end(wins: &mut [Window], t: f64, slow: f64, min_len: f64, beats: &Beats) -> Option<f64> {
    let slow = if slow.is_finite() { slow.clamp(0.5, 1.0) } else { 1.0 };
    let footage: f64 = wins.iter().map(|w| w.len).sum();
    let last = wins.last_mut()?;
    let end = t + footage / slow;
    let tol = BEAT_SNAP.min(beats.period() * 0.5);
    let shorter = ((last.len - 0.5).max(0.0) / slow).min(end - (t + min_len)).max(0.0).min(tol);
    let longer = ((last.reach - last.len).max(0.0) / slow).min(tol);
    let b = beats.nearest(end, end - shorter, end + longer)?;
    last.len = (last.len + (b - end) * slow).max(0.0);
    Some(b)
}

/// The user's music track (`Options::music`), imported: its item, and its length and beats when
/// the analysis found them (`media.beats`, seconds of the track).
struct MusicFile {
    item: u64,
    path: String,
    seconds: Option<f64>,
    bpm: Option<f64>,
    /// Beats measured in the track, which starts at 0 on the timeline.
    beats: Option<Beats>,
}

/// Import `path` and measure its beats. A track without a beat the analysis can follow is still
/// used (noted), its cuts then not on its beat; a file that does not import is an error.
async fn import_music(ex: &dyn Executor, path: &str, notes: &mut Vec<String>) -> Result<MusicFile, AutoCutError> {
    let r = run(ex, "file.import", json!({"paths": [path]})).await?;
    let item = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| {
        let why = r["errors"].as_array().and_then(|a| a.first()).map_or_else(|| "nothing imported".to_string(), Value::to_string);
        AutoCutError::Invalid(format!("music: {path}: {why}"))
    })?;
    let mut m = MusicFile { item, path: path.to_string(), seconds: None, bpm: None, beats: None };
    match run(ex, "media.beats", json!({"item": item})).await {
        Ok(b) => {
            m.seconds = b["seconds"].as_f64().filter(|s| s.is_finite() && *s > 0.0);
            m.bpm = b["bpm"].as_f64().filter(|x| x.is_finite());
            let times: Vec<f64> = b["beats"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            m.beats = Beats::measured(&times, b["period"].as_f64().unwrap_or(f64::NAN));
        }
        Err(e) => notes.push(format!("music: {path}: {e}; the cuts are not on its beat")),
    }
    Ok(m)
}

/// The music under the show, as built.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Music {
    /// `"generated"`, or the file it came from.
    pub source: String,
    /// Its tempo (as generated, or as measured), when known.
    pub bpm: Option<f64>,
    /// How it fits the show's length: `"generated"`, `"trimmed"` (faded out where the show
    /// ends), `"remixed"` (`clip.remix`: cut at musically similar beats to the length), `"looped"`
    /// (played again from the start) or `"as is"` (its length could not be read).
    pub fit: String,
    /// Cuts (montage cuts, entry and outro starts) within a frame of one of its beats, of all.
    pub cuts_on_beat: usize,
    pub cuts: usize,
    /// A generated bed's genre.
    pub genre: Option<String>,
    /// Where a generated bed's final chord lands on its last bar, seconds of the timeline (it
    /// rings out to the end of the bed: no fade).
    pub ending: Option<f64>,
    /// What played a generated bed: a SoundFont's name, or the built-in oscillators.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruments: Option<String>,
}

/// The energy (0..1) of a generated bed over a show `len` seconds long, as a curve of points
/// spread evenly over it: building through the intro (to `intro_end`), high through the entries,
/// settling for the outro (from `outro_at`) — scaled to the mood's own energy `base`.
pub fn show_energy(len: f64, intro_end: f64, outro_at: f64, base: f64) -> Vec<f64> {
    const POINTS: usize = 32;
    let base = if base.is_finite() { base.clamp(0.05, 1.0) } else { 0.8 };
    let len = if len.is_finite() { len.max(0.0) } else { 0.0 };
    let intro_end = if intro_end.is_finite() { intro_end.clamp(0.0, len) } else { 0.0 };
    let outro_at = if outro_at.is_finite() { outro_at.clamp(intro_end, len) } else { len };
    (0..POINTS)
        .map(|k| {
            let x = k as f64 / (POINTS - 1) as f64 * len;
            let shape = if x < intro_end {
                0.7 + 0.25 * x / intro_end.max(1e-6)
            } else if x < outro_at {
                0.95 + 0.1 * (x - intro_end) / (outro_at - intro_end).max(1e-6)
            } else {
                0.55
            };
            ((base * shape).clamp(0.0, 1.0) * 1000.0).round() / 1000.0
        })
        .collect()
}

/// How much a generated bed's energy comes up while a scene plays without words.
const SCENE_LIFT: f64 = 0.15;

/// `energy` (points spread evenly over a show `len` seconds long, as [`show_energy`] gives
/// them) lifted by [`SCENE_LIFT`] wherever one of `spans` (seconds) plays: the music comes up
/// under a scene without words.
pub fn lift_energy(energy: &mut [f64], len: f64, spans: &[(f64, f64)]) {
    let n = energy.len();
    if n < 2 || !(len.is_finite() && len > 0.0) {
        return;
    }
    for (k, e) in energy.iter_mut().enumerate() {
        let x = k as f64 / (n - 1) as f64 * len;
        if spans.iter().any(|(a, b)| x >= *a && x < *b) {
            *e = ((*e + SCENE_LIFT).clamp(0.0, 1.0) * 1000.0).round() / 1000.0;
        }
    }
}

impl Music {
    /// With the count of `cuts` (seconds) within a frame of a beat of the music as it plays.
    fn with_cuts(self, cuts: &[f64], played: Option<&Beats>, fps: f64) -> Self {
        let near = 1.0 / fps.clamp(1.0, 240.0) + 0.012;
        let on = played.map_or(0, |b| cuts.iter().filter(|c| b.distance(**c) <= near).count());
        Self { cuts: cuts.len(), cuts_on_beat: on, ..self }
    }
}

/// Most copies of a short track looped under a long show.
const MAX_LOOPS: usize = 500;

/// The length of clip `clip` on the music track, seconds (read back from the sequence).
async fn clip_seconds(ex: &dyn Executor, tr: &Tracks, clip: u64) -> Option<f64> {
    let seq = ex.execute("sequence.inspect", json!({})).await.ok()?;
    let index = tr.music.trim_start_matches(['A', 'a']).parse::<usize>().ok()?.checked_sub(1)?;
    let items = seq["audio"].get(index)?["items"].as_array()?;
    let c = items.iter().find(|c| c["clip"].as_u64() == Some(clip))?;
    c["duration"].as_f64().map(|d| d / TPS)
}

/// Lay the user's track under the whole show (`target` seconds) on the music track from 0: cut
/// to length when it is longer (faded out by the caller), remixed to length at its beats when it
/// is shorter (`clip.remix`), else looped. Returns the clips, how it was fitted, and its beats
/// as they now play.
async fn place_music(
    ex: &dyn Executor,
    tr: &Tracks,
    m: &MusicFile,
    target: f64,
    notes: &mut Vec<String>,
) -> Result<(Vec<u64>, &'static str, Option<Beats>), AutoCutError> {
    let place = async |time: f64, duration: Option<f64>| -> Result<u64, AutoCutError> {
        let mut p = json!({"item": m.item, "track": tr.music, "time": ticks(time), "sourceIn": 0});
        if let Some(d) = duration {
            p["duration"] = json!(ticks(d));
        }
        let r = run(ex, "timeline.place", p).await?;
        r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| AutoCutError::Executor("timeline.place placed no music".into()))
    };
    let length = match m.seconds {
        Some(s) if s >= target => return Ok((vec![place(0.0, Some(target)).await?], "trimmed", m.beats.clone())),
        Some(s) => Some(s),
        None => None,
    };
    let first = place(0.0, None).await?;
    let length = match length {
        Some(s) => s,
        None => match clip_seconds(ex, tr, first).await {
            Some(s) if s > 0.0 => s,
            _ => return Ok((vec![first], "as is", None)),
        },
    };
    if length >= target {
        // (a track whose beats were not found, longer than the show)
        let delta = ticks(target) - ticks(length);
        if let Err(e) = run(ex, "timeline.trim", json!({"clip": first, "edge": "out", "mode": "regular", "delta": delta})).await {
            notes.push(format!("music: {e}"));
        }
        return Ok((vec![first], "trimmed", m.beats.clone()));
    }
    // shorter than the show: longer at its own beats, cut where the music is most alike (the
    // planner can find no joints for some lengths with few of them: then with more, freer ones)
    if m.beats.is_some() {
        let mut why = String::new();
        for (segments, variations) in [(50, 50), (100, 50), (100, 100), (0, 100)] {
            match run(ex, "clip.remix", json!({"clip": first, "seconds": target, "segments": segments, "variations": variations})).await {
                Ok(r) => {
                    let beats = m.beats.as_ref().and_then(|b| remixed_beats(b, &r));
                    return Ok((vec![first], "remixed", beats));
                }
                Err(e) => why = e.to_string(),
            }
        }
        notes.push(format!("music: cannot remix it to length ({why}); looping it on its beat"));
    }
    // else played again from the start until the show ends, each time on the beat
    let starts = loop_starts(length, target, m.beats.as_ref(), tr.fps);
    let mut clips = vec![first];
    let mut times: Vec<f64> = Vec::new();
    for (k, s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(target.min(s + length));
        if k == 0 {
            if end < length - 0.5 / tr.fps {
                // the first copy (placed whole) is cut where the second starts
                if let Err(e) = run(ex, "timeline.trim", json!({"clip": first, "edge": "out", "mode": "regular", "delta": ticks(end) - ticks(length)})).await {
                    notes.push(format!("music: {e}"));
                }
            }
        } else {
            let c = place(*s, Some(end - s)).await?;
            run(ex, "sequence.applyAudioTransition", json!({"clip": c, "edge": "in", "effect": "constant_power", "frames": AUDIO_FADE_FRAMES})).await.ok();
            clips.push(c);
        }
        if let Some(b) = &m.beats {
            times.extend(b.times.iter().map(|x| x + s).filter(|x| *x < end));
        }
    }
    let beats = m.beats.as_ref().and_then(|b| Beats::measured(&times, b.period()));
    Ok((clips, "looped", beats))
}

/// Where each copy of a `length`-second track starts when it is looped under a `target`-second
/// show: each next copy starts so that its first beat lands on the track's beat grid (`beats`,
/// measured from the track's start, going on at its tempo past the last) at or just before the
/// end of the copy before, which is cut there — so the beat never stumbles at a seam; without
/// beats, end to end. All on frames of `fps`; at most [`MAX_LOOPS`] copies.
fn loop_starts(length: f64, target: f64, beats: Option<&Beats>, fps: f64) -> Vec<f64> {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let floor = |x: f64| (x * fps + 1e-6).floor() / fps;
    let mut starts = vec![0.0];
    if !(length.is_finite() && length > 1.0 && target.is_finite()) {
        return starts;
    }
    let first_beat = beats.and_then(|b| b.times.first().copied());
    while let Some(&s) = starts.last() {
        let natural = s + length;
        if natural >= target - 0.05 || starts.len() >= MAX_LOOPS {
            break;
        }
        let on_beat = match (beats, first_beat) {
            (Some(b), Some(b0)) => b.nearest(natural + b0, natural + b0 - b.period(), natural + b0).map(|g| floor(g - b0)),
            _ => None,
        };
        starts.push(on_beat.filter(|n| *n > s + 1.0 && *n <= natural).unwrap_or_else(|| floor(natural)));
    }
    starts
}

/// The beats of a track as `clip.remix` plays it (its report `r`): each piece of the remix
/// (`src` and `len` in ticks of the track) plays the track's beats in it, one after the other.
fn remixed_beats(track: &Beats, r: &Value) -> Option<Beats> {
    let mut out = Vec::new();
    let mut at = 0.0;
    for p in r["pieces"].as_array()? {
        let (src, len) = (p["src"].as_f64()? / TPS, p["len"].as_f64()? / TPS);
        if !(src.is_finite() && len.is_finite() && len > 0.0) {
            return None;
        }
        out.extend(track.times.iter().filter(|b| **b >= src && **b < src + len).map(|b| at + b - src));
        at += len;
    }
    Beats::measured(&out, track.period())
}

/// After the words for an entry, its own picture and sound run on this long before the next one.
const BREATHE: f64 = 2.8;
/// Shortest and longest entry, seconds.
const MIN_ENTRY: f64 = 6.0;
const MAX_ENTRY: f64 = 22.0;

/// With a target length ([`Options::target_minutes`]): the least an entry breathes after its
/// words, and the most; the shortest entry, and the longest (its robot in view all along).
const MIN_BREATHE: f64 = 0.8;
const MAX_BREATHE: f64 = 60.0;
const TIGHT_ENTRY: f64 = 3.0;
const MAX_STRETCHED_ENTRY: f64 = 45.0;
/// Words a narration voice says a second, about (the built-in voice says ~3.2, cloned ones
/// fewer): to plan the entries still to come; each one's own words are measured.
const WORDS_PER_SECOND: f64 = 2.8;

/// Seconds the subject is in view in the shots an entry can show ([`Shot::spans`] at least a
/// window long).
fn in_view_seconds(e: &Entry) -> f64 {
    let s: f64 = e.shots.iter().take(MAX_SHOTS).flat_map(|s| s.spans()).map(|(a, b)| b - a).filter(|l| *l >= MIN_WINDOW).sum();
    if s.is_finite() { s.max(0.0) } else { 0.0 }
}

/// An entry's length as a function of how long it breathes after its words: its lead and words
/// (`fixed`) plus that, never under what its words need (`min`) nor over what of its robot is in
/// view (`max`; a `max` under `min` gives way: words are never cut).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stretch {
    pub fixed: f64,
    pub min: f64,
    pub max: f64,
}

impl Stretch {
    /// An entry whose lead and words take `fixed` seconds, with `in_view` seconds of its robot.
    pub fn new(fixed: f64, in_view: f64) -> Self {
        let fixed = if fixed.is_finite() { fixed.max(0.0) } else { 0.0 };
        let min = (fixed + MIN_BREATHE).max(TIGHT_ENTRY);
        let in_view = if in_view.is_finite() { in_view } else { 0.0 };
        Self { fixed, min, max: in_view.min(MAX_STRETCHED_ENTRY).max(min) }
    }

    pub fn at(&self, breathe: f64) -> f64 {
        (self.fixed + breathe).max(self.min).min(self.max)
    }
}

/// The breathing time (the same for every entry) that makes entries of these stretches last
/// `seconds` together, as near as their limits allow: [`MIN_BREATHE`] when even that is too
/// long, [`MAX_BREATHE`] (every entry as long as its robot is in view) when nothing is long
/// enough.
pub fn breathe_for(stretches: &[Stretch], seconds: f64) -> f64 {
    let total = |b: f64| stretches.iter().map(|s| s.at(b)).sum::<f64>();
    if !seconds.is_finite() || total(MIN_BREATHE) >= seconds {
        return MIN_BREATHE;
    }
    if total(MAX_BREATHE) <= seconds {
        return MAX_BREATHE;
    }
    let (mut lo, mut hi) = (MIN_BREATHE, MAX_BREATHE);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if total(mid) < seconds {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Where a chapter starts and its card: at an entry whose event differs from the one before
/// ("GITEX Global 2022, Dubai"), or, for an entry with no event, where the year changes — so a
/// robot met days after an expo, the same year, opens no "2022" card of its own.
pub fn chapter_cards(entries: &[Entry]) -> Vec<Option<String>> {
    let (mut last_event, mut last_year, mut last_unix): (Option<String>, Option<i64>, Option<i64>) = (None, None, None);
    entries
        .iter()
        .map(|e| {
            let event = e.event.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
            let year = e.unix.map(|u| ymd(u).0);
            // an entry long after an event's last one starts a chapter of its own, in the same
            // year too (a robot met days later still belongs to the show)
            let left_event = last_event.is_some() && matches!((e.unix, last_unix), (Some(u), Some(l)) if u.saturating_sub(l) > EVENT_AFTERGLOW);
            let card = match &event {
                Some(ev) if Some(ev) != last_event.as_ref() => Some(ev.clone()),
                Some(_) => None,
                None => year.filter(|y| Some(*y) != last_year || left_event).map(|y| y.to_string()),
            };
            last_event = event;
            last_year = year.or(last_year);
            last_unix = e.unix.or(last_unix);
            card
        })
        .collect()
}

/// After this long (seconds) an entry without an event no longer belongs to the event before it.
const EVENT_AFTERGLOW: i64 = 30 * 86_400;

/// The lead before each entry's words: longer where a chapter starts, as its card comes first.
fn leads(entries: &[Entry]) -> Vec<f64> {
    chapter_cards(entries).iter().map(|c| if c.is_some() { CARD_LEAD } else { 0.5 }).collect()
}

/// How long a chapter card shows before the entry's words, seconds.
const CARD_LEAD: f64 = 1.8;

/// A chapter card's text and size: a year big, an event's name smaller, on up to two lines.
fn card(section: &str, width: f64) -> (String, f64) {
    let s = section.trim();
    if s.chars().count() <= 6 {
        return (s.to_string(), 160.0);
    }
    // "GITEX Global 2022, Dubai" → "GITEX GLOBAL 2022" over "DUBAI"; a part too long for one
    // line is split in two at the space nearest its middle
    let upper = s.to_uppercase();
    let parts: Vec<&str> = match upper.split_once(", ") {
        Some((a, b)) => vec![a, b],
        None => vec![upper.as_str()],
    };
    let lines: Vec<String> = parts.iter().flat_map(|p| balanced(p, CARD_LINE_CHARS)).collect();
    // and the longest line fits the frame (bold capitals are about 0.62 of the size wide)
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(1).max(1) as f64;
    let fit = if width.is_finite() && width > 0.0 { width * 0.88 / (0.62 * longest) } else { 96.0 };
    (lines.join("\n"), fit.clamp(40.0, 96.0))
}

/// Longest chapter card line before it is split, characters.
const CARD_LINE_CHARS: usize = 24;

/// `text` as one line, or two split at the space nearest its middle when longer than `max`.
fn balanced(text: &str, max: usize) -> Vec<String> {
    let n = text.chars().count();
    if n <= max {
        return vec![text.to_string()];
    }
    let mid = n / 2;
    let split = text.char_indices().filter(|(_, c)| *c == ' ').map(|(i, _)| i).min_by_key(|i| text[..*i].chars().count().abs_diff(mid));
    match split {
        Some(i) => vec![text[..i].trim().to_string(), text[i..].trim().to_string()],
        None => vec![text.to_string()],
    }
}
/// Fades on each shot's own sound, frames: no clip's sound starts or stops with a click.
const AUDIO_FADE_FRAMES: i64 = 8;

/// How long a scene without words plays (after its chapter card's lead), seconds.
const SCENE_SECONDS: f64 = 4.5;

/// The next of `list` in turn (`counter` counts the turns); `None` for an empty list (cuts).
fn in_turn<'a>(list: &'a [String], counter: &mut usize) -> Option<&'a String> {
    let pick = list.get(*counter % list.len().max(1));
    *counter += 1;
    pick
}

/// Within-entry cuts with transitions from the catalogue: each cut between an entry's windows
/// (starting at `t`, played at full speed) moved to the nearest beat within [`BEAT_SNAP`] (half a
/// beat at most) — the window before it a little longer (only as far as its reach, the robot
/// still in view) or shorter (never under half a second), the one after it the other way — so
/// the entry keeps its length and every cut in it lands on the music.
fn snap_cuts(wins: &mut [Window], t: f64, beats: &Beats) {
    let tol = BEAT_SNAP.min(beats.period() * 0.5);
    let mut at = t;
    for i in 0..wins.len().saturating_sub(1) {
        let (Some(cur), Some(next)) = (wins.get(i).cloned(), wins.get(i + 1).cloned()) else { break };
        let cut = at + cur.len;
        let earlier = (cur.len - 0.5).max(0.0).min((next.reach - next.len).max(0.0)).min(tol);
        let later = (cur.reach - cur.len).max(0.0).min((next.len - 0.5).max(0.0)).min(tol);
        if let Some(b) = beats.nearest(cut, cut - earlier, cut + later) {
            let d = b - cut;
            if let Some(w) = wins.get_mut(i) {
                w.len += d;
            }
            if let Some(w) = wins.get_mut(i + 1) {
                w.len -= d;
            }
        }
        at += wins.get(i).map_or(0.0, |w| w.len);
    }
}

/// Build the showcase into a new sequence. `progress` gets a line per entry.
pub async fn build(ex: &dyn Executor, opts: &Options, entries: &[Entry], script: &Script, progress: &mut dyn FnMut(&str)) -> Result<Built, AutoCutError> {
    if entries.is_empty() {
        return Err(AutoCutError::Invalid(format!("no clip shows a {} clearly enough to count", opts.subject)));
    }
    let look = &opts.look;
    let robots = entries.iter().filter(|e| !e.is_scene()).count();
    if robots == 0 {
        return Err(AutoCutError::Invalid(format!("only scene entries: a showcase needs at least one {}", opts.subject)));
    }
    let (w, h) = (f64::from(opts.width), f64::from(opts.height));
    // the plan's chips and the timeline's dates are checked before anything is built
    let numbers: Vec<usize> = entries.iter().map(|e| e.number).collect();
    let chips = look::chips_by_entry(&script.chips, &numbers)?;
    if look.timeline {
        let dates: Vec<i64> = entries.iter().filter_map(|e| e.unix).collect();
        look::Scale::new(&dates, (w, h))?;
    }
    let mut built = Built { entries: entries.len(), scenes: entries.len() - robots, ..Built::default() };
    // the overlays get tracks of their own, above the show's five
    let mut video = 5;
    let timeline_tracks = if look.timeline {
        video += 3;
        Some([5, 6, 7])
    } else {
        None
    };
    let chip_tracks: Vec<usize> = if chips.is_empty() {
        Vec::new()
    } else {
        let c = (video..video + look::MAX_CHIPS).collect();
        video += look::MAX_CHIPS;
        c
    };
    let r = run(ex, "file.newSequence", json!({"name": opts.title, "width": opts.width, "height": opts.height, "fps": opts.fps, "video": video, "audio": 5}))
        .await?;
    built.sequence = r["sequence"].as_u64();
    let tr = Tracks {
        fps: opts.fps.clamp(1.0, 240.0),
        aspect: w / h.max(1.0),
        stabilize: opts.stabilize,
        fill: opts.fill_frame,
        follow: opts.follow_subject,
        frame: (w, h),
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
        timeline: timeline_tracks,
        chips: chip_tracks,
    };
    // the theme, its families as installed (Inter for one that is not)
    let theme: Option<Theme> = match &look.style {
        Some(st) => {
            let installed = look::installed_fonts(ex).await;
            let th = st.resolve(installed.as_deref());
            for m in &th.missing {
                built.notes.push(format!("style: the font {m} is not installed; {} stands in", look::FALLBACK_FAMILY));
            }
            progress(&format!("style: {} — {}, {}, {}", th.name, th.title.family, th.body.family, th.number.family));
            Some(th)
        }
        None => None,
    };
    // what the new overlays (the timeline, chips, scene captions) wear without a theme
    let overlay = theme.clone().unwrap_or_else(|| look::Style::classic().resolve(None));
    let voice = opts.voice.as_deref();
    let mut narrations = Narrations::default();
    let mut mix = Mix::default();
    let mut fg_firsts: Vec<(u64, Option<u64>)> = Vec::new();
    let title_props = json!({"font_style": "Bold", "align": "center", "stroke": true, "stroke_width": 6, "shadow": true});

    // ---- the music's beats, which the cuts land on: the user's track (imported now, its beats
    // measured) or the generated bed's steady grid
    let track = match &opts.music {
        Some(path) => Some(import_music(ex, path, &mut built.notes).await?),
        None => None,
    };
    // a generated bed: the engine's own tempo (its beats are exact ticks from 0, the grid the
    // cuts land on) and the mood's energy; an engine without `music.suggest` gives the asked ones
    let bed = if track.is_none() { run(ex, "music.suggest", json!({"mood": opts.mood, "bpm": opts.bpm, "count": 1})).await.ok() } else { None };
    let bed_bpm = bed.as_ref().and_then(|v| v["suggestions"][0]["track"]["bpm"].as_f64()).unwrap_or(opts.bpm);
    let bed_energy = bed.as_ref().and_then(|v| v["suggestions"][0]["style"]["energy"].as_f64()).unwrap_or(0.8);
    let beats = match &track {
        Some(m) => m.beats.clone(),
        None => Some(Beats::grid(bed_bpm)),
    };
    if let Some(m) = &track {
        match &m.beats {
            Some(b) => progress(&format!("music: {} at {:.0} bpm; cutting on its beat", m.path, 60.0 / b.period())),
            None => progress(&format!("music: {} (no beat found: cuts are not on its beat)", m.path)),
        }
    }
    // cut points: the montage's, the entries' and the outro's (how many land on a beat)
    let mut cuts: Vec<f64> = Vec::new();

    // ---- intro: a cold open — a montage of robots from across the years cut on the beat,
    // the title over it, a riser into an impact
    let intro_voice_at = 2.2;
    let intro_spoken =
        narrate(ex, &tr, voice, &script.spoken("intro", &script.intro), &script.intro, script.delivery_of("intro"), intro_voice_at, &mut narrations).await?;
    // the voice track's next free moment: a line never starts over the last one
    let mut voice_free_at = intro_voice_at + intro_spoken;
    let intro_needs = (intro_voice_at + intro_spoken + 0.8).max(6.0);
    // the intro ends on the first beat after it has said its words
    let intro_len = on_frame(beats.as_ref().and_then(|b| b.next(intro_needs)).filter(|x| *x < intro_needs + 2.0).unwrap_or(intro_needs), tr.fps);
    // the montage cuts every two beats (four when the beat is quick)
    let beat = beats.as_ref().map_or(60.0 / opts.bpm.clamp(60.0, 180.0), Beats::period);
    let cut = if beat * 2.0 >= 0.6 { 2.0 * beat } else { 4.0 * beat };
    let picks = montage(entries, ((intro_len / cut).floor() as usize).clamp(1, MAX_MONTAGE));
    let mut at = 0.0;
    // the picks in turn, the last one to the end of the intro; a clip too short for its turn
    // hands over to the next (round again if need be), so the intro has no black
    let mut i = 0;
    while at < intro_len - 0.5 / tr.fps && i < MAX_MONTAGE * 4 {
        let Some(s) = picks.get(i % picks.len().max(1)) else { break };
        let next_cut = beats.as_ref().and_then(|b| b.nearest(at + cut, at + cut * 0.5, at + cut * 1.5)).unwrap_or(at + cut);
        let len = if i + 1 >= picks.len() { intro_len - at } else { (next_cut - at).min(intro_len - at) };
        let len = len.min((s.clip_seconds - 0.4).max(0.5));
        let shot = place_shot(ex, &tr, s.item, at, s.window_start(len), len, true, s.largest, true).await?;
        fade_in(ex, shot.audio).await;
        mix.sounds.extend(shot.audio);
        at = shot.end;
        if at < intro_len - 0.5 / tr.fps {
            cuts.push(at);
        }
        i += 1;
    }
    // the title lands with an impact on a beat, a riser into it
    let hit = beats.as_ref().and_then(|b| b.nearest(1.8, 1.3, 2.4)).map_or(1.8, |x| on_frame(x, tr.fps));
    sfx(ex, "riser", hit.min(1.8), (hit - 1.8).max(0.0), &tr.hits, 0.7, &mut mix).await?;
    sfx(ex, "impact", 3.0, hit, &tr.hits, 0.8, &mut mix).await?;
    let subtitle = opts.subtitle.clone().unwrap_or_else(|| years(entries));
    match &theme {
        Some(th) => look::title_card(ex, th, [tr.titles_index, tr.over_index], tr.fps, (w, h), &opts.title, &subtitle, hit, intro_len).await?,
        None => {
            title(ex, tr.titles_index, tr.fps, &opts.title, hit, intro_len - hit, 110.0, (w / 2.0, h * 0.46), title_props.clone()).await?;
            if !subtitle.is_empty() {
                title(
                    ex,
                    tr.over_index,
                    tr.fps,
                    &subtitle,
                    hit + 0.5,
                    intro_len - hit - 0.5,
                    54.0,
                    (w / 2.0, h * 0.46 + 95.0),
                    json!({"font_style": "Bold", "align": "center", "fill_color": "#ffcc00", "shadow": true}),
                )
                .await?;
            }
        }
    }
    built.chapters.push(Chapter { seconds: 0.0, title: "Intro".into() });
    let mut t = intro_len;
    let mut counter_times = Vec::new();
    let cards = chapter_cards(entries);
    // a target length: each entry's share of it, planned from its words and its robot in view
    let target = opts.target_minutes.filter(|m| m.is_finite() && *m > 0.0).map(|m| (m * 60.0).min(24.0 * 3600.0));
    let words = |text: &str| if voice.is_some() { text.split_whitespace().count() as f64 / WORDS_PER_SECOND } else { 0.0 };
    let planned: Vec<Stretch> = entries
        .iter()
        .zip(leads(entries))
        .enumerate()
        .map(|(k, (e, lead))| {
            let line = script.lines.get(k).cloned().unwrap_or_else(|| template_line(e));
            Stretch::new(lead + words(&script.spoken(&e.number.to_string(), &line)), in_view_seconds(e))
        })
        .collect();
    let outro_estimate = (0.8 + words(&script.spoken("outro", &script.outro)) + 1.5).max(5.0);
    // transitions from the catalogue: each list in turn, about a beat long (two into a chapter)
    let (mut n_within, mut n_between, mut n_chapter) = (0usize, 0usize, 0usize);
    let frames_of = |beats: f64, lo: f64, hi: f64| (beat * beats * tr.fps).round().clamp(lo, hi) as i64;
    // scenes that play without words: the music comes up under them
    let mut quiet_scenes: Vec<(f64, f64)> = Vec::new();

    // ---- the entries
    for (k, e) in entries.iter().enumerate() {
        let scene = e.is_scene();
        let section = cards.get(k).cloned().flatten();
        let new_year = section.is_some();
        let lead = if new_year { CARD_LEAD } else { 0.5 };
        let line = script.lines.get(k).cloned().unwrap_or_else(|| template_line(e));
        let voice_at = (t + lead).max(voice_free_at + 0.15);
        // (a part with no words — a scene's, usually — is not narrated)
        let spoken = if line.trim().is_empty() {
            0.0
        } else {
            narrate(ex, &tr, voice, &script.spoken(&e.number.to_string(), &line), &line, script.delivery_of(&e.number.to_string()), voice_at, &mut narrations)
                .await?
        };
        if spoken > 0.0 {
            voice_free_at = voice_at + spoken;
        }
        // the words, then a moment of the robot's own picture and sound before the next one —
        // with a target length, as long a moment as shares out what is left of it over this
        // entry and the ones still to come (planned from their words; this one's are measured)
        let needed = match target {
            Some(target) => {
                let here = Stretch::new(voice_at - t + spoken, in_view_seconds(e));
                let rest: Vec<Stretch> = std::iter::once(here).chain(planned.iter().skip(k + 1).copied()).collect();
                here.at(breathe_for(&rest, target - t - outro_estimate))
            }
            None if scene && spoken <= 0.0 => (lead + SCENE_SECONDS).clamp(MIN_ENTRY, MAX_ENTRY),
            None => (voice_at - t + spoken + BREATHE).clamp(MIN_ENTRY, MAX_ENTRY),
        };
        let must = voice_at - t + spoken + 0.7;
        // the robot in view; when that is too little to carry the words even at half speed,
        // the clips as they are (a scene has no robot: its clips as they are)
        let mut wins = if scene { filling_windows(e, needed) } else { in_view_windows(e, needed) };
        if !scene && wins.iter().map(|w| w.len).sum::<f64>() / 0.5 < must {
            wins = filling_windows(e, needed);
        }
        let footage: f64 = wins.iter().map(|w| w.len).sum();
        // footage too short for the words: slow the shots to cover them, never below half speed
        // (an entry with words to spare just ends sooner)
        let slow = if footage + 0.05 < must { (footage / must).clamp(0.5, 1.0) } else { 1.0 };
        // with transitions from the catalogue, the cuts within the entry land on beats too
        if look.transitions.is_some()
            && slow >= 0.999
            && let Some(b) = beats.as_ref()
        {
            snap_cuts(&mut wins, t, b);
        }
        // the entry ends on a beat (so the next one, its transition and year card start on one)
        let beat_end = beats.as_ref().and_then(|b| snap_end(&mut wins, t, slow, must.max(2.0), b)).map(|x| on_frame(x, tr.fps));
        cuts.push(t);
        let mut at = t;
        let mut firsts = None;
        let mut shots = Vec::new();
        let mut full_frame = Vec::new();
        let n_wins = wins.len();
        for (i, Window { shot: s, start: src, len, reach }) in wins.iter().enumerate() {
            // the last shot ends on the beat's own frame, whatever rounding came before
            let len = match beat_end {
                Some(end) if i + 1 == n_wins && slow >= 0.999 => (end - at).max(0.5).min(reach.max(0.5)),
                _ => *len,
            };
            // (a scene has no robot to frame or follow)
            let focus = if scene { None } else { s.largest };
            let Placed { fg, bg, audio, end } = place_shot(ex, &tr, s.item, at, *src, len, true, focus, !scene).await?;
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
                match &look.transitions {
                    // between shots (or stretches of one clip) of an entry: a quick dissolve
                    None => {
                        run(ex, "sequence.applyVideoTransition", json!({"clip": fg, "edge": "in", "effect": "cross_dissolve", "frames": 8})).await.ok();
                    }
                    // or the plan's own (none by default: a cut, on the beat)
                    Some(tp) => {
                        if let Some(effect) = in_turn(&tp.within, &mut n_within) {
                            let p = json!({"clip": fg, "edge": "in", "effect": effect, "frames": frames_of(0.5, 6.0, 15.0)});
                            match run(ex, "sequence.applyVideoTransition", p).await {
                                Ok(r) => built.transitions.extend(r["transition"].as_u64().map(|id| PlacedTransition {
                                    id,
                                    effect: effect.clone(),
                                    at: "within",
                                    sound: None,
                                    entry: e.number,
                                })),
                                Err(err) => built.notes.push(format!("transition {effect} in entry {}: {err}", e.number)),
                            }
                        }
                    }
                }
            }
            shots.push(PlacedShot { clip: fg, item: s.item, start: on_frame(at, tr.fps), end: shot_end });
            full_frame.push(bg.is_none());
            // a slowed shot can end between frames: the next starts on the following frame
            at = (shot_end * tr.fps - 1e-6).ceil() / tr.fps;
        }
        // (before this entry's titles and transitions are on the reference frame)
        if opts.match_color {
            match_colour(ex, &shots, &full_frame, &mut built.notes).await;
        }
        let entry_end = on_frame(at.max(t + 2.0), tr.fps);
        if scene && spoken <= 0.0 {
            quiet_scenes.push((t, entry_end));
        }
        let mut placed = PlacedEntry { number: e.number, label: e.label.clone(), start: t, end: entry_end, shots, scene, ..PlacedEntry::default() };
        if let Some((fg, bg)) = firsts {
            fg_firsts.push((fg, bg));
            // (the first entry follows the intro montage)
            match &look.transitions {
                None => {
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
                // from the catalogue, each with its sound (a whoosh between entries, a riser
                // into a chapter, whose card brings the impact)
                Some(tp) => {
                    let (list, where_, sound, frames, counter) = if new_year {
                        (&tp.chapter, "chapter", "riser", frames_of(2.0, 12.0, 40.0), &mut n_chapter)
                    } else {
                        (&tp.between, "between", "whoosh", frames_of(1.0, 8.0, 24.0), &mut n_between)
                    };
                    if let Some(effect) = in_turn(list, counter) {
                        match run(ex, "sequence.applyVideoTransition", json!({"clip": fg, "edge": "in", "effect": effect, "frames": frames})).await {
                            Ok(r) => {
                                if let Some(id) = r["transition"].as_u64() {
                                    let intensity = if new_year { 0.75 } else { 0.6 };
                                    let with_sound =
                                        match run(ex, "transitions.addSound", json!({"transitions": [id], "kind": sound, "intensity": intensity})).await {
                                            Ok(_) => Some(sound),
                                            Err(err) => {
                                                built.notes.push(format!("transition sound at entry {}: {err}", e.number));
                                                None
                                            }
                                        };
                                    built.transitions.push(PlacedTransition { id, effect: effect.clone(), at: where_, sound: with_sound, entry: e.number });
                                }
                            }
                            Err(err) => built.notes.push(format!("transition {effect} at entry {}: {err}", e.number)),
                        }
                    }
                    if let Some(b) = bg {
                        run(ex, "sequence.applyVideoTransition", json!({"clip": b, "edge": "in", "effect": "cross_dissolve", "frames": 12})).await.ok();
                    }
                }
            }
        }
        if new_year && let Some(sec) = &section {
            built.chapters.push(Chapter { seconds: t, title: sec.clone() });
            let (text, size) = card(sec, w);
            match &theme {
                Some(th) => {
                    look::chapter_card(ex, th, tr.over_index, tr.fps, (w, h), &text, size, t, CARD_LEAD).await?;
                }
                None => {
                    title(ex, tr.over_index, tr.fps, &text, t, CARD_LEAD, size, (w / 2.0, h * 0.55), title_props.clone()).await?;
                }
            }
            sfx(ex, "impact", 2.0, t, &tr.hits, 0.6, &mut mix).await?;
        }
        // the name, and when and where, bottom left, while the narration introduces it (a
        // scene: its caption)
        let when = month_year(e.unix);
        let shown_for = (entry_end - t - lead - 0.3).clamp(1.5, 4.5);
        let (t0, t1) = (on_frame(t + lead, tr.fps), on_frame(t + lead + shown_for, tr.fps));
        // (raised a little over the timeline)
        let low = if look.timeline { h * 0.78 } else { h * 0.82 };
        let (shown, beside) = if scene {
            let caption = e.caption_text();
            match &theme {
                Some(th) => {
                    let line = look::Line { text: &caption, role: Role::Body, size: 34.0, color: &th.text, caps: false, opacity: 100.0 };
                    let (c, b) = look::panel_text(ex, th, tr.titles_index, tr.fps, (w, h), &[line], t + lead, shown_for, (w * 0.05, h * 0.82)).await?;
                    (c, Some(b))
                }
                None => {
                    let props =
                        json!({"font_style": "Medium", "align": "left", "background": true, "background_opacity": 45, "background_size": 14, "shadow": false});
                    let c = title(ex, tr.titles_index, tr.fps, &caption, t + lead, shown_for, 40.0, (w * 0.05, low), props).await?;
                    (c, None)
                }
            }
        } else {
            let name = display_label(&e.label);
            let place = e.event.as_deref().map(str::trim).filter(|s| !s.is_empty());
            let second = lower_third_place(place, &when);
            match &theme {
                Some(th) => {
                    let mut lines = vec![look::Line { text: &name, role: Role::Title, size: 54.0, color: &th.text, caps: true, opacity: 100.0 }];
                    if !second.is_empty() {
                        lines.push(look::Line { text: &second, role: Role::Body, size: 27.0, color: &th.text, caps: false, opacity: 82.0 });
                    }
                    let (c, b) = look::panel_text(ex, th, tr.titles_index, tr.fps, (w, h), &lines, t + lead, shown_for, (w * 0.05, h * 0.80)).await?;
                    (c, Some(b))
                }
                None => {
                    let lower = if second.is_empty() { name } else { format!("{name}\n{second}") };
                    let props =
                        json!({"font_style": "Bold", "align": "left", "background": true, "background_opacity": 55, "background_size": 18, "shadow": false});
                    (title(ex, tr.titles_index, tr.fps, &lower, t + lead, shown_for, 46.0, (w * 0.05, low), props).await?, None)
                }
            }
        };
        // it slides and fades in and out (cosmetic: still right without)
        slide_in_out(ex, shown, t0, t1, (w, h), tr.fps).await.ok();
        if scene {
            placed.caption = Some(shown);
        } else {
            placed.lower_third = Some(shown);
            counter_times.push(t + lead);
        }
        // its spec chips beside it, one after the other while its words are said
        if let Some(texts) = chips.get(&e.number) {
            let beside = match beside {
                Some(b) => Some(b),
                // the plain lower third (its background reaches 18 px, or 14 for a caption, past the text)
                None => look::measure(ex, shown, Some(0)).await.map(|b| b.pad(if scene { 14.0 } else { 18.0 }, if scene { 14.0 } else { 18.0 })),
            };
            let c0 = if spoken > 0.0 { voice_at + 0.5 } else { t + lead + 0.6 };
            let c1 = (t + lead + shown_for).max(voice_at + spoken + 0.3).min(entry_end - 0.1);
            match beside {
                Some(b) if c1 - c0 >= 1.2 => match look::chips(ex, &overlay, &tr.chips, tr.fps, (w, h), texts, b, c0, c1).await {
                    Ok(c) => placed.chips = c,
                    Err(err) => built.notes.push(format!("chips of entry {}: {err}", e.number)),
                },
                Some(_) => built.notes.push(format!("chips of entry {}: the entry is too short to show them", e.number)),
                None => built.notes.push(format!("chips of entry {}: its lower third could not be measured", e.number)),
            }
        }
        built.layout.push(placed);
        progress(&format!("[{}/{}] {} — {} ({:.1}s)", e.number, entries.len(), when, if scene { e.caption_text() } else { e.label.clone() }, entry_end - t));
        t = entry_end;
    }

    // ---- outro: the total, over a blurred shot of the latest one, with the closing words
    let outro_at = t;
    built.chapters.push(Chapter { seconds: t, title: "Outro".into() });
    let outro_voice_at = (t + 0.8).max(voice_free_at + 0.15);
    let outro_spoken =
        narrate(ex, &tr, voice, &script.spoken("outro", &script.outro), &script.outro, script.delivery_of("outro"), outro_voice_at, &mut narrations).await?;
    let outro_len = (outro_voice_at - t + outro_spoken + 1.5).max(5.0);
    if let Some(h0) = entries.iter().rev().find(|e| !e.is_scene()).and_then(|e| e.shots.first()) {
        let len = outro_len.min((h0.clip_seconds - 0.4).max(1.0));
        let Placed { fg, audio, .. } = place_shot(ex, &tr, h0.item, t, h0.window_start(len), len, false, h0.largest, true).await?;
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
    let closing = format!("{robots} {}S", opts.subject.to_uppercase());
    match &theme {
        Some(th) => look::outro_card(ex, th, [tr.titles_index, tr.over_index], tr.fps, (w, h), &closing, "and counting", t, outro_len).await?,
        None => {
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
        }
    }
    let total = t + outro_len;
    built.seconds = total;
    built.target_seconds = target;
    if let Some(target) = target {
        let clock = |s: f64| format!("{}:{:02}", (s.max(0.0) / 60.0).floor(), (s.max(0.0) % 60.0).floor());
        progress(&format!("length {} for a target of {}", clock(total), clock(target)));
    }

    // ---- the counter, across the entries
    let counter_prefix = format!("{} #", opts.subject.to_uppercase());
    let counter = run(
        ex,
        "graphics.counter",
        json!({"times": counter_times, "end": t, "prefix": counter_prefix, "corner": "topRight", "size": 60, "color": "#ffffff", "track": tr.counter_index}),
    )
    .await?;
    if let (Some(th), Some(clip)) = (&theme, counter["clip"].as_u64()) {
        // in the theme's number font, without the outline
        let mut props = th.text_props(Role::Number, &th.text, "right");
        props["shadow_blur"] = json!(20);
        if let Err(err) = run(ex, "graphics.set", json!({"clip": clip, "layer": 0, "props": props})).await {
            built.notes.push(format!("style: the counter: {err}"));
        }
    }
    // it steps aside while a scene plays (there is no robot to count)
    if let Some(clip) = counter["clip"].as_u64() {
        let first = counter_times.first().copied().unwrap_or(f64::INFINITY);
        let mut spans: Vec<(f64, f64)> = Vec::new();
        for p in built.layout.iter().filter(|p| p.scene && p.start > first) {
            match spans.last_mut() {
                Some(last) if (p.start - last.1).abs() < 1e-6 => last.1 = p.end,
                _ => spans.push((p.start, p.end)),
            }
        }
        if let Err(err) = look::hide_during(ex, clip, &spans, tr.fps).await {
            built.notes.push(format!("the counter over a scene: {err}"));
        }
    }

    // ---- the timeline along the bottom, from the first entry to the outro
    if let Some(tracks) = tr.timeline {
        let along: Vec<look::TimelineEntry> = built
            .layout
            .iter()
            .zip(entries)
            .map(|(p, e)| look::TimelineEntry { number: e.number, start: p.start, unix: e.unix, event: e.event.clone() })
            .collect();
        match look::plan_timeline(&along, &look.events, intro_len, outro_at, (w, h), tr.fps) {
            Ok(Some(mut tl)) => {
                if let Err(err) = look::draw_timeline(ex, &overlay, &mut tl, tracks, tr.fps, (w, h)).await {
                    built.notes.push(format!("timeline: {err}"));
                }
                progress(&format!(
                    "timeline: {}–{}, {} bands, the marker at {} dates",
                    tl.scale.first_year,
                    tl.scale.last_year,
                    tl.bands.len(),
                    tl.stops.len()
                ));
                built.timeline = Some(tl);
            }
            Ok(None) => built.notes.push("timeline: no entry has a date".into()),
            Err(err) => built.notes.push(format!("timeline: {err}")),
        }
    }

    // ---- music under everything, ducked under the voice, faded out: the user's track fitted to
    // the show, or a generated bed
    cuts.push(t);
    let (music, info, played) = match &track {
        Some(m) => {
            let (clips, fit, played) = place_music(ex, &tr, m, total + 0.5, &mut built.notes).await?;
            let info = Music { source: m.path.clone(), bpm: m.bpm, fit: fit.into(), ..Music::default() }.with_cuts(&cuts, played.as_ref(), tr.fps);
            (clips, info, played)
        }
        None => {
            // arranged to the show's exact length, its energy following the show (up under the
            // scenes without words), ending on a bar with its final chord ringing out (so no
            // fade); the mood's own energy scales it
            let len = total + 0.5;
            let mut energy = show_energy(len, intro_len, outro_at, bed_energy);
            lift_energy(&mut energy, len, &quiet_scenes);
            let mut p = json!({"kind": "music", "duration": len, "mood": opts.mood, "bpm": opts.bpm, "time": 0, "track": tr.music, "energy": energy});
            if let Some(sf) = &look.soundfont {
                p["soundfont"] = json!(sf);
            }
            let m = match run(ex, "audio.generate", p.clone()).await {
                Ok(m) => m,
                // a SoundFont that cannot play: the bed on the built-in instruments, noted
                Err(err) if look.soundfont.is_some() => {
                    built.notes.push(format!("music: {err}; the bed is played without that SoundFont"));
                    if let Some(o) = p.as_object_mut() {
                        o.remove("soundfont");
                    }
                    run(ex, "audio.generate", p).await?
                }
                Err(err) => return Err(err),
            };
            // the bed's own beat map (exact ticks): the grid the cuts were laid on
            let map: Vec<i64> = m["beats"].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
            let played = Beats::from_ticks(&map, 0.0).unwrap_or_else(|| Beats::grid(bed_bpm));
            if let (Some(_), Some(warning)) = (&look.soundfont, m["warning"].as_str()) {
                built.notes.push(format!("music: {warning}"));
            }
            let info = Music {
                instruments: m["soundfont"]["name"].as_str().map(str::to_string),
                source: "generated".into(),
                bpm: m["bpm"].as_f64().or(Some(60.0 / played.period())),
                fit: "generated".into(),
                genre: m["genre"].as_str().map(str::to_string),
                ending: m["ending"].as_i64().map(|t| t as f64 / TPS),
                ..Music::default()
            };
            let info = info.with_cuts(&cuts, Some(&played), tr.fps);
            (m["clip"].as_u64().into_iter().collect::<Vec<u64>>(), info, Some(played))
        }
    };
    // the user's track is faded out where the show ends; a generated bed ends on its own
    if track.is_some()
        && let Some(last) = music.last()
    {
        run(ex, "sequence.applyAudioTransition", json!({"clip": last, "edge": "out", "effect": "constant_power", "frames": (opts.fps * 3.0) as i64}))
            .await
            .ok();
    }
    progress(&format!("music: {} ({}); {} of {} cuts on the beat", info.source, info.fit, info.cuts_on_beat, info.cuts));
    built.music = Some(info);
    // ---- transitions from the catalogue: on the music's beats, each a musical length
    if !built.transitions.is_empty() {
        match &played {
            Some(b) => {
                let ids: Vec<u64> = built.transitions.iter().map(|x| x.id).collect();
                match run(ex, "transitions.snapToBeats", json!({"transitions": ids, "beats": b.list(0.0, total + 0.5)})).await {
                    Ok(r) => {
                        let snapped = r["snapped"].as_array().map_or(0, Vec::len);
                        for s in r["skipped"].as_array().into_iter().flatten() {
                            built.notes.push(format!("transition {}: not on the beat ({})", s["transition"], s["reason"].as_str().unwrap_or("")));
                        }
                        progress(&format!("transitions: {snapped} of {} on the beat", ids.len()));
                        built.beat_snap = Some(json!({"snapped": snapped, "of": ids.len(), "bpm": r["bpm"], "skipped": r["skipped"]}));
                    }
                    Err(err) => built.notes.push(format!("transitions: {err}")),
                }
            }
            None => built.notes.push("transitions: the music has no beat to put them on".into()),
        }
    }
    if !narrations.clips.is_empty() {
        progress(&format!("{} narration lines, {} from the narration cache, {} read again", narrations.clips.len(), narrations.cached, narrations.retakes));
    }
    // ---- the mix
    progress("mixing");
    mix_down(ex, &mix, &narrations.clips, &music, look.loudness.is_none(), &mut built.notes).await;
    // to a destination's loudness, with a true-peak ceiling (instead of the limiter alone)
    if let Some(l) = &look.loudness {
        progress("loudness: normalising the mix");
        match run(ex, "audio.normalizeLoudness", json!({"target": l.target(), "truePeak": look::TRUE_PEAK})).await {
            Ok(r) => {
                progress(&format!(
                    "loudness: {:.1} LUFS, true peak {:.1} dBTP",
                    r["loudnessLufs"].as_f64().unwrap_or(f64::NAN),
                    r["truePeakDbtp"].as_f64().unwrap_or(f64::NAN)
                ));
                built.loudness = Some(r);
            }
            Err(err) => built.notes.push(format!("loudness: {err}")),
        }
    }
    built.narration_lines = narrations.clips.len();
    built.narration_cached = narrations.cached;
    built.narration_retakes = narrations.retakes;
    built.notes.append(&mut narrations.notes);
    // ---- subtitles: what the narration says, as written, on a caption track
    built.subtitles = crate::subtitles::cues(&narrations.lines, tr.fps);
    built.narration = narrations.lines;
    let captions = if built.subtitles.is_empty() { None } else { caption_track(ex, &built.subtitles, &mut built.notes).await };
    built.theme = theme;
    if let Some(path) = &opts.export {
        progress("exporting (YouTube 1080p)");
        let mut p = json!({"path": path, "preset": "YouTube 1080p Full HD", "wait": true});
        // the subtitles in the video too, as closed captions (CEA-608/708)
        if look.embed_captions {
            if captions.is_some() {
                p["embedCaptions"] = json!(true);
            } else {
                built.notes.push("captions: no narration to caption; nothing embedded".into());
            }
        }
        built.export = Some(run(ex, "file.exportMedia", p).await?);
        // the subtitles next to the export, for YouTube (SubRip) and the web (WebVTT)
        if let Some(track) = captions {
            for ext in ["srt", "vtt"] {
                let file = std::path::Path::new(path).with_extension(ext).to_string_lossy().into_owned();
                match run(ex, "captions.export", json!({"path": file, "track": track})).await {
                    Ok(_) => built.subtitle_files.push(file),
                    Err(e) => built.notes.push(format!("subtitles: {e}")),
                }
            }
        }
        // and the thumbnail
        progress("thumbnail");
        let file = std::path::Path::new(path).with_extension("thumbnail.jpg").to_string_lossy().into_owned();
        match thumbnail(ex, opts, entries, built.sequence, &file).await {
            Ok(_) => built.thumbnail = Some(file),
            Err(e) => built.notes.push(format!("thumbnail: {e}")),
        }
    }
    Ok(built)
}

/// Thumbnail size: YouTube's (16:9, at most 2 MB).
pub const THUMBNAIL_SIZE: (u32, u32) = (1280, 720);

/// The thumbnail's frame: across every entry's shots, the sampled moment where the subject is
/// largest ([`Shot::largest`]), a well-known robot ([`crate::analyze::famous`]) counting
/// double and any other named one a third more; `(shot, the moment, entry number)`, the
/// moment's time inside the clip. A shot from a plan without `largest` counts at its best
/// moment, in the middle of the frame, as a modest size.
pub fn thumbnail_frame(entries: &[Entry], subject: &str) -> Option<(Shot, Moment, usize)> {
    let mut pick: Option<(f64, &Shot, Moment, usize)> = None;
    for e in entries.iter().filter(|e| !e.is_scene()) {
        let weight = if crate::analyze::famous(&e.label) {
            2.0
        } else if generic(&e.label, subject) {
            1.0
        } else {
            1.3
        };
        for s in &e.shots {
            let m = s.largest.unwrap_or(Moment { seconds: s.best, area: 0.15, center: Moment::middle() });
            if !(m.seconds.is_finite() && m.area.is_finite()) {
                continue;
            }
            let score = m.area.clamp(0.0, 1.0) * weight;
            if pick.is_none_or(|p| score > p.0) {
                pick = Some((score, s, Moment { seconds: m.seconds.max(0.0).min((s.clip_seconds - 0.1).max(0.0)), ..m }, e.number));
            }
        }
    }
    pick.map(|(_, s, m, n)| (s.clone(), m, n))
}

/// The title on at most two balanced lines, in capitals (as thumbnails shout).
fn thumbnail_lines(title: &str) -> Vec<String> {
    let words: Vec<String> = title.split_whitespace().map(str::to_uppercase).collect();
    let all = words.join(" ");
    if all.chars().count() <= 14 || words.len() < 2 {
        return vec![all];
    }
    let total = all.chars().count();
    let mut best = (usize::MAX, 1);
    let mut a = 0;
    for k in 1..words.len() {
        a += words.get(k - 1).map_or(0, |w| w.chars().count()) + usize::from(k > 1);
        let longer = a.max(total.saturating_sub(a + 1));
        if longer < best.0 {
            best = (longer, k);
        }
    }
    let (first, second) = words.split_at(best.1.min(words.len()));
    vec![first.join(" "), second.join(" ")]
}

/// Write the YouTube thumbnail to `path` (JPEG, or PNG for a `.png` path), 1280×720: the
/// [`thumbnail_frame`] filling the picture (a portrait one over its blurred copy), the show's
/// title over it in large outlined capitals and the count in a corner — composed in a sequence
/// of its own ("<title> — Thumbnail", left in the project to adjust and export again) and
/// exported with `file.exportFrame`. The show's sequence `show` is the active one again
/// afterwards, whatever happened.
pub async fn thumbnail(ex: &dyn Executor, opts: &Options, entries: &[Entry], show: Option<u64>, path: &str) -> Result<Value, AutoCutError> {
    let (shot, moment, _) = thumbnail_frame(entries, &opts.subject).ok_or_else(|| AutoCutError::Invalid("no shot to make a thumbnail from".into()))?;
    let (w, h) = THUMBNAIL_SIZE;
    let fps = opts.fps.clamp(1.0, 240.0);
    let r = run(ex, "file.newSequence", json!({"name": format!("{} — Thumbnail", opts.title), "width": w, "height": h, "fps": fps, "video": 4, "audio": 1}))
        .await?;
    let thumb = r["sequence"].as_u64();
    let robots = entries.iter().filter(|e| !e.is_scene()).count();
    let made = compose_thumbnail(ex, opts, robots, &shot, moment, thumb, path).await;
    if let Some(t) = thumb {
        run(ex, "sequence.close", json!({"item": t})).await.ok();
    }
    if let Some(s) = show {
        run(ex, "sequence.open", json!({"item": s})).await?;
    }
    made
}

async fn compose_thumbnail(
    ex: &dyn Executor,
    opts: &Options,
    count: usize,
    shot: &Shot,
    moment: Moment,
    thumb: Option<u64>,
    path: &str,
) -> Result<Value, AutoCutError> {
    let thumb = thumb.ok_or_else(|| AutoCutError::Executor("file.newSequence made no sequence".into()))?;
    let (w, h) = (f64::from(THUMBNAIL_SIZE.0), f64::from(THUMBNAIL_SIZE.1));
    let fps = opts.fps.clamp(1.0, 240.0);
    let tr = Tracks {
        fps,
        aspect: w / h,
        stabilize: false,
        // the whole picture over its blurred copy: the title goes where the robot is not
        fill: false,
        follow: false,
        frame: (w, h),
        bg: "V1".into(),
        fg: "V2".into(),
        titles_index: 2,
        over_index: 3,
        counter_index: 3,
        clip_audio: "A1".into(),
        voice: "A1".into(),
        music: "A1".into(),
        sfx: "A1".into(),
        hits: "A1".into(),
        timeline: None,
        chips: Vec::new(),
    };
    // the frame: a second of the shot from the moment, its first frame the one exported
    place_shot(ex, &tr, shot.item, 0.0, moment.seconds, 1.0, false, Some(moment), true).await?;
    // the title, as large as two lines across the frame allow, in the half of the picture the
    // robot is not in (low unless it stands low), the count in the opposite corner
    let lines = thumbnail_lines(&opts.title);
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(1).max(1) as f64;
    let size = (w * 0.92 / (0.68 * longest)).clamp(56.0, 140.0);
    let n = lines.len() as f64;
    let low = moment.center.get(1).is_none_or(|y| *y < 0.58);
    let first_baseline = if low { h - 60.0 - (n - 1.0) * size * 1.05 } else { 50.0 + size * 0.75 };
    let props =
        json!({"font_style": "Bold", "align": "center", "fill_color": "#ffffff", "stroke": true, "stroke_width": (size / 11.0).round(), "shadow": true});
    title(ex, tr.titles_index, fps, &lines.join("\n"), 0.0, 1.0, size, (w / 2.0, first_baseline), props).await?;
    let tag = format!("{count} {}S", opts.subject.to_uppercase());
    let tag_props = json!({"font_style": "Bold", "align": "left", "fill_color": "#ffcc00", "stroke": true, "stroke_width": 6, "shadow": true});
    title(ex, tr.over_index, fps, &tag, 0.0, 1.0, 64.0, (48.0, if low { 100.0 } else { h - 48.0 }), tag_props).await?;
    let format = if path.to_ascii_lowercase().ends_with(".png") { "png" } else { "jpeg" };
    run(ex, "file.exportFrame", json!({"sequence": thumb, "time": 0, "path": path, "format": format})).await
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
                Shot { item: 1, name: "a".into(), clip_seconds: 20.0, best: 19.5, seen: vec![], unseen: vec![], largest: None },
                Shot { item: 2, name: "b".into(), clip_seconds: 3.0, best: 1.0, seen: vec![], unseen: vec![], largest: None },
            ],
            ..Entry::default()
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
        let s = Shot { item: 47, name: "s".into(), clip_seconds: 15.5, best: 7.8, seen: vec![2.0, 7.8], unseen: vec![4.9, 10.7, 13.6], largest: None };
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
        let e = Entry { number: 1, label: "x".into(), kind: "x".into(), description: String::new(), unix: None, shots: vec![s], ..Entry::default() };
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
            largest: None,
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
    fn events_make_chapters_and_reach_the_script() {
        let day = 1_665_446_400; // 2022-10-11
        let mut a = Entry { number: 1, label: "Spot".into(), unix: Some(day), event: Some("GITEX Global 2022, Dubai".into()), ..Entry::default() };
        a.facts = vec!["Made by Boston Dynamics".into(), " ".into()];
        let b = Entry { number: 2, label: "Pepper".into(), unix: Some(day + 600), event: Some("GITEX Global 2022, Dubai".into()), ..Entry::default() };
        let c = Entry { number: 3, label: "Go1".into(), unix: Some(day + 86_400 * 7), ..Entry::default() };
        // one card for the event; a robot met days later the same year opens none
        assert_eq!(leads(&[a.clone(), b.clone(), c.clone()]), vec![CARD_LEAD, 0.5, 0.5]);
        // months later the same year: a chapter of its own (the live case: Emrill in July
        // after the Dubai AI Festival in April)
        let later = Entry { number: 5, label: "Cleaner".into(), unix: Some(day + 86_400 * 60), ..Entry::default() };
        assert_eq!(chapter_cards(&[a.clone(), later]), vec![Some("GITEX Global 2022, Dubai".into()), Some("2022".into())]);
        let d = Entry { number: 4, label: "Arm".into(), unix: Some(day + 86_400 * 400), ..Entry::default() };
        assert_eq!(chapter_cards(&[c.clone(), a.clone(), d]), vec![Some("2022".into()), Some("GITEX Global 2022, Dubai".into()), Some("2023".into())]);
        assert_eq!(c.section().as_deref(), Some("2022"));
        assert_eq!(card("2022", 1920.0), ("2022".to_string(), 160.0));
        assert_eq!(card("GITEX Global 2022, Dubai", 1920.0), ("GITEX GLOBAL 2022\nDUBAI".to_string(), 96.0));
        // the live overflow: a long event name runs off a 1080p frame at 96 px
        let (text, size) = card("Our Future(s) by Dubai Future Foundation, 2025", 1920.0);
        assert_eq!(text, "OUR FUTURE(S) BY DUBAI\nFUTURE FOUNDATION\n2025");
        let longest = text.lines().map(|l| l.chars().count()).max().unwrap() as f64;
        assert!(longest * 0.62 * size <= 1920.0 * 0.88 + 1e-9, "{size}");
        // one endless word still gets a size, never below the floor
        let (_, tiny) = card(&"X".repeat(200), 1920.0);
        assert!((40.0..=96.0).contains(&tiny));
        assert_eq!(card("Some long event name here", f64::NAN).1, 96.0);
        // the script request carries where and the (non-empty) facts, and only for that item
        let line = item_listing(1, &a);
        assert!(line.contains("(at GITEX Global 2022, Dubai)") && line.contains("Facts: Made by Boston Dynamics") && !line.contains("|"), "{line}");
        assert!(!item_listing(3, &c).contains("Facts"));
        // old plans without the fields still load
        let old: Entry = serde_json::from_value(json!({"number": 1, "label": "x", "kind": "x", "description": "", "unix": null, "shots": []})).unwrap();
        assert!(old.event.is_none() && old.facts.is_empty());
    }

    #[test]
    fn lower_thirds_say_where_and_when_once() {
        assert_eq!(lower_third_place(Some("GITEX Global 2022, Dubai"), "October 2022"), "GITEX Global 2022, Dubai · October");
        assert_eq!(lower_third_place(Some("Museum of the Future"), "July 2025"), "Museum of the Future · July 2025");
        assert_eq!(lower_third_place(None, "July 2025"), "July 2025");
        assert_eq!(lower_third_place(Some("Expo"), ""), "Expo");
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
    fn the_voice_reads_respellings_and_the_text_stays_as_written() {
        let map: BTreeMap<String, String> =
            [("Zaku II", "Zaku Two"), ("NodeMCU", "Node M C U"), ("Go1", "Go One"), ("DP World", "D P World"), ("OnRobot", "On Robot"), ("Go", "Gee Oh")]
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
        assert_eq!(respell("The Zaku II and a NodeMCU.", &map), "The Zaku Two and a Node M C U.");
        // whole words only, longest key first, case as written
        assert_eq!(respell("Go1, Go10 and Go!", &map), "Go One, Go10 and Gee Oh!");
        assert_eq!(respell("dp world vs DP World's booth", &map), "dp world vs D P World's booth");
        assert_eq!(respell("OnRobotics by OnRobot", &map), "OnRobotics by On Robot");
        // a replacement is never replaced again, and text without keys is untouched
        let echo: BTreeMap<String, String> = [("A".to_string(), "A A".to_string())].into_iter().collect();
        assert_eq!(respell("A B", &echo), "A A B");
        assert_eq!(respell("Grüße, Zaku II!", &map), "Grüße, Zaku Two!");
        assert_eq!(respell("nothing here", &BTreeMap::new()), "nothing here");
        // hostile keys: empty or blank ones are ignored
        let odd: BTreeMap<String, String> = [("".to_string(), "x".to_string()), (" ".to_string(), "y".to_string())].into_iter().collect();
        assert_eq!(respell("a b", &odd), "a b");
        // `say` replaces a part, and is respelled too; a blank one is ignored
        let s = Script {
            intro: "Hi.".into(),
            lines: vec!["The Zaku II.".into()],
            outro: "Bye.".into(),
            pronounce: map,
            say: [("1".to_string(), "My Zaku II model kit.".to_string()), ("outro".to_string(), "  ".to_string())].into_iter().collect(),
            ..Script::default()
        };
        assert_eq!(s.spoken("1", "The Zaku II."), "My Zaku Two model kit.");
        assert_eq!(s.spoken("intro", "Hi."), "Hi.");
        assert_eq!(s.spoken("outro", "Bye."), "Bye.");
        // a plan from before: no `pronounce`, no `say`; and none written when there are none
        let old: Script = serde_json::from_value(json!({"intro": "a", "lines": [], "outro": "b"})).unwrap();
        assert!(old.pronounce.is_empty() && old.say.is_empty());
        let v = serde_json::to_value(&old).unwrap();
        assert!(v.get("pronounce").is_none() && v.get("say").is_none(), "{v}");
        assert!(serde_json::to_value(&s).unwrap()["pronounce"]["Zaku II"] == "Zaku Two");
    }

    #[test]
    fn beats_come_from_the_grid_or_the_track_and_go_on_at_its_tempo() {
        let g = Beats::grid(120.0);
        assert_eq!(g.period(), 0.5);
        assert_eq!(g.nearest(7.1, 6.0, 8.0), Some(7.0));
        assert_eq!(g.nearest(7.3, 6.0, 8.0), Some(7.5));
        assert_eq!(g.nearest(7.3, 7.35, 9.0), Some(7.5));
        assert_eq!(g.nearest(7.3, 6.0, 7.2), Some(7.0));
        assert_eq!(g.nearest(7.3, 7.31, 7.4), None, "no beat in range");
        assert_eq!(g.next(6.01), Some(6.5));
        assert_eq!(g.next(6.0), Some(6.0));
        assert!(g.distance(3.26) - 0.24 < 1e-9);
        assert_eq!(Beats::grid(f64::NAN).period(), 60.0 / 96.0);
        assert_eq!(Beats::grid(1000.0).period(), 60.0 / 200.0, "as audio.generate clamps it");
        // a measured track: its own beats (slightly uneven), then its tempo before and after
        let m = Beats::measured(&[1.02, 1.5, 2.01, 2.49, 3.0, f64::NAN, -1.0, 2.01], 0.5).unwrap();
        assert_eq!(m.nearest(1.9, 1.0, 3.0), Some(2.01));
        assert_eq!(m.nearest(0.6, 0.0, 1.0), Some(0.52), "before the first: back at its tempo");
        assert!((m.nearest(10.2, 9.0, 11.0).unwrap() - 10.0).abs() < 1e-9, "after the last: on at its tempo");
        assert_eq!(m.next(2.2), Some(2.49));
        assert!(m.nearest(0.0, -1.0, 0.3).is_some_and(|b| (b - 0.02).abs() < 1e-9), "no beat before 0 s");
        // nonsense tempos and no beats at all: none
        for p in [0.0, f64::NAN, -0.5, 10.0] {
            assert!(Beats::measured(&[1.0], p).is_none(), "{p}");
        }
        assert!(Beats::measured(&[f64::NAN], 0.5).is_none());
        assert!(g.nearest(f64::INFINITY, 0.0, 1.0).is_none() && g.nearest(1.0, 2.0, 1.0).is_none());
        assert!(g.next(1e300).is_none());
        // a generated bed's exact beat map: the same grid
        let bed = Beats::from_ticks(&[0, TPS as i64 / 2, TPS as i64, TPS as i64 * 3 / 2], 0.0).unwrap();
        assert_eq!(bed.period(), 0.5);
        assert_eq!(bed.nearest(7.3, 6.0, 8.0), g.nearest(7.3, 6.0, 8.0), "it goes on at its tempo");
        assert!(Beats::from_ticks(&[0], 0.0).is_none() && Beats::from_ticks(&[], 0.0).is_none());
        assert!(Beats::from_ticks(&[0, -5, i64::MIN], 0.0).is_none(), "nonsense maps are refused");
    }

    #[test]
    fn a_generated_bed_builds_through_the_show_and_settles_for_the_outro() {
        let e = show_energy(60.0, 8.0, 50.0, 0.8);
        assert_eq!(e.len(), 32);
        assert!(e.iter().all(|x| (0.0..=1.0).contains(x)));
        let (intro, body, outro) = (e[0], e[20], e[31]);
        assert!(intro < body && outro < intro, "{e:?}");
        // a calm mood keeps it calm
        assert!(show_energy(60.0, 8.0, 50.0, 0.3).iter().all(|x| *x <= 0.33));
        // hostile numbers stay in range
        for (len, a, b, base) in [(f64::NAN, 1.0, 2.0, 0.8), (10.0, f64::INFINITY, -1.0, f64::NAN), (-5.0, 100.0, 3.0, 9.0), (0.0, 0.0, 0.0, 0.8)] {
            assert!(show_energy(len, a, b, base).iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
        }
    }

    #[test]
    fn an_entry_ends_on_the_nearest_beat_it_can_reach() {
        let shot = Shot { item: 1, name: "a".into(), clip_seconds: 20.0, best: 5.0, seen: vec![], unseen: vec![], largest: None };
        let w = |len: f64, reach: f64| Window { shot: shot.clone(), start: 2.0, len, reach };
        let g = Beats::grid(120.0);
        // 10 + 7.3 s: the beat at 17.5 is in reach
        let mut wins = vec![w(4.0, 4.0), w(3.3, 6.0)];
        assert_eq!(snap_end(&mut wins, 10.0, 1.0, 5.0, &g), Some(17.5));
        assert!((wins[1].len - 3.5).abs() < 1e-9);
        // the robot leaves the view at once: shorter, to the beat before
        let mut wins = vec![w(4.0, 4.0), w(3.2, 3.2)];
        assert_eq!(snap_end(&mut wins, 10.0, 1.0, 5.0, &g), Some(17.0));
        assert!((wins[1].len - 3.0).abs() < 1e-9);
        // neither: the words need the whole entry, and the robot leaves the view
        let mut wins = vec![w(4.0, 4.0), w(3.2, 3.2)];
        assert_eq!(snap_end(&mut wins, 10.0, 1.0, 7.25, &g), None);
        assert!((wins[1].len - 3.2).abs() < 1e-9, "unchanged");
        // never further than half a beat
        let mut wins = vec![w(4.0, 4.0), w(3.5, 6.0)];
        assert_eq!(snap_end(&mut wins, 10.0, 1.0, 5.0, &Beats::grid(60.0)), None, "half a second from either beat");
        // a slowed entry (half speed): the window changes half as much as the end
        let mut wins = vec![w(3.6, 6.0)];
        assert_eq!(snap_end(&mut wins, 10.0, 0.5, 7.0, &g), Some(17.0));
        assert!((wins[0].len - 3.5).abs() < 1e-9);
        assert!(snap_end(&mut [], 10.0, 1.0, 5.0, &g).is_none());
    }

    #[test]
    fn a_target_length_is_shared_out_as_breathing_time() {
        // words of 3, 5 and 2 s; the robots in view for 20, 8 and 1 s
        let s = [Stretch::new(3.0, 20.0), Stretch::new(5.0, 8.0), Stretch::new(2.0, 1.0)];
        assert_eq!((s[2].min, s[2].max), (TIGHT_ENTRY, TIGHT_ENTRY), "too little in view: as short as its words allow");
        let total = |b: f64| s.iter().map(|x| x.at(b)).sum::<f64>();
        for target in [16.0, 20.0, 25.0, 30.0] {
            let b = breathe_for(&s, target);
            assert!((total(b) - target).abs() < 1e-6, "{target}: {}", total(b));
        }
        // too short a target: as tight as the words allow; too long: as long as the robots are in view
        assert_eq!(breathe_for(&s, 5.0), MIN_BREATHE);
        assert!((total(breathe_for(&s, 5.0)) - (3.8 + 5.8 + 3.0)).abs() < 1e-9);
        assert_eq!(breathe_for(&s, 1000.0), MAX_BREATHE);
        assert!((total(MAX_BREATHE) - (20.0 + 8.0 + 3.0)).abs() < 1e-9, "never more than in view");
        // hostile numbers
        assert_eq!(breathe_for(&s, f64::NAN), MIN_BREATHE);
        assert_eq!(breathe_for(&[], 30.0), MAX_BREATHE);
        let odd = Stretch::new(f64::NAN, f64::INFINITY);
        assert!(odd.at(5.0).is_finite() && odd.max <= MAX_STRETCHED_ENTRY);
    }

    #[test]
    fn time_a_shot_cannot_fill_in_view_goes_to_the_next() {
        // two shots: the first in view for 3 s only, the second for 15 s
        let short = Shot { item: 1, name: "a".into(), clip_seconds: 20.0, best: 5.0, seen: vec![4.0, 5.0, 6.0], unseen: vec![2.0, 8.0, 12.0], largest: None };
        let long = Shot { item: 2, name: "b".into(), clip_seconds: 20.0, best: 10.0, seen: vec![3.0, 6.0, 9.0, 12.0, 15.0], unseen: vec![], largest: None };
        let e = Entry { number: 1, label: "x".into(), kind: "x".into(), description: String::new(), unix: None, shots: vec![short, long], ..Entry::default() };
        assert!((in_view_seconds(&e) - (4.0 + 19.6)).abs() < 1e-9, "{}", in_view_seconds(&e));
        let w = windows(&e, 10.0);
        let total: f64 = w.iter().map(|x| x.2).sum();
        assert!((total - 10.0).abs() < 1e-9, "{w:?}");
        assert!((w[0].2 - 4.0).abs() < 1e-9 && (w[1].2 - 6.0).abs() < 1e-9, "{w:?}");
    }

    #[test]
    fn the_thumbnail_shows_the_largest_robot_a_famous_one_first() {
        let day = 1_600_000_000;
        let with_area = |item: u64, label: &str, unix: i64, area: f64| ClipFacts {
            frames: vec![
                FrameFact { seconds: 3.0, present: true, area: area / 2.0, center: [0.5, 0.5] },
                FrameFact { seconds: 9.0, present: true, area, center: [0.5, 0.5] },
                FrameFact { seconds: 12.0, present: false, area: 0.9, center: [0.5, 0.5] },
            ],
            ..fact(item, label, "humanoid", unix, 1.0)
        };
        let facts = vec![
            with_area(1, "red toy robot", day, 0.45),
            with_area(2, "Boston Dynamics Spot", day + 86_400, 0.3),
            with_area(3, "Zork cleaning robot", day + 86_400 * 2, 0.35),
        ];
        let entries = group(&facts, "robot", 0.3);
        assert_eq!(entries[0].shots[0].largest, Some(Moment { seconds: 9.0, area: 0.45, center: [0.5, 0.5] }), "the largest moment that shows it");
        // Spot (0.3, famous: 0.6) over the bigger toy robot (0.45) and the named one (0.35 × 1.3)
        let (shot, at, n) = thumbnail_frame(&entries, "robot").unwrap();
        assert_eq!((shot.item, at.seconds, n), (2, 9.0, 2));
        // a much larger plain one wins after all
        let facts = vec![with_area(1, "red toy robot", day, 0.8), with_area(2, "Boston Dynamics Spot", day + 86_400, 0.3)];
        assert_eq!(thumbnail_frame(&group(&facts, "robot", 0.3), "robot").map(|p| p.0.item), Some(1));
        // a plan from before `largest`: the best moment; hostile values are skipped
        let mut old = group(&facts, "robot", 0.3);
        for e in &mut old {
            for s in &mut e.shots {
                s.largest = None;
            }
        }
        assert_eq!(thumbnail_frame(&old, "robot").map(|p| (p.0.item, p.1.seconds)), Some((2, 6.0)), "Spot's best moment");
        old[1].shots[0].largest = Some(Moment { seconds: f64::NAN, area: 0.9, center: [0.5, 0.5] });
        assert_eq!(thumbnail_frame(&old, "robot").map(|p| p.0.item), Some(1));
        assert!(thumbnail_frame(&[], "robot").is_none());
        assert!(crate::analyze::famous("Pepper humanoid robot") && !crate::analyze::famous("red toy robot") && !crate::analyze::famous("figure skating robot"));
    }

    #[test]
    fn thumbnail_titles_go_on_two_balanced_lines() {
        assert_eq!(thumbnail_lines("Robots"), vec!["ROBOTS"]);
        assert_eq!(thumbnail_lines("Every Robot I've Met"), vec!["EVERY ROBOT", "I'VE MET"]);
        assert_eq!(thumbnail_lines("  "), vec![""]);
        assert_eq!(thumbnail_lines(&"x".repeat(40)).len(), 1, "one long word stays whole");
    }

    #[test]
    fn a_looped_track_keeps_its_beat_across_the_seams() {
        // 0.51 s a beat from 0.37 s: 20 s of track is not a whole number of beats
        let times: Vec<f64> = (0..39).map(|k| 0.37 + 0.51 * k as f64).collect();
        let b = Beats::measured(&times, 0.51).unwrap();
        let starts = loop_starts(20.0, 50.0, Some(&b), 30.0);
        assert_eq!(starts.len(), 3, "{starts:?}");
        for w in starts.windows(2) {
            let seam = w[1];
            assert!(seam <= w[0] + 20.0 && seam > w[0] + 19.0, "cut a little short, never longer: {starts:?}");
            // the next copy's first beat falls on the grid (within a frame)
            assert!(b.distance(seam + 0.37) <= 1.0 / 30.0 + 1e-9, "{seam}: {}", b.distance(seam + 0.37));
            assert!(((seam * 30.0) - (seam * 30.0).round()).abs() < 1e-6, "on a frame");
        }
        // without beats: end to end; a track as long as the show: once; nonsense: once
        assert_eq!(loop_starts(20.0, 50.0, None, 30.0), vec![0.0, 20.0, 40.0]);
        assert_eq!(loop_starts(20.0, 20.0, Some(&b), 30.0), vec![0.0]);
        assert_eq!(loop_starts(f64::NAN, 50.0, None, 30.0), vec![0.0]);
        assert_eq!(loop_starts(0.5, 50.0, None, 30.0), vec![0.0]);
        assert_eq!(loop_starts(2.0, 1e9, None, 30.0).len(), MAX_LOOPS);
    }

    #[test]
    fn a_remix_plays_the_beats_of_its_pieces() {
        let track = Beats::measured(&(0..40).map(|k| 0.25 + 0.5 * k as f64).collect::<Vec<_>>(), 0.5).unwrap();
        let tick = |s: f64| (s * TPS) as i64;
        // the first 8 s, then from 4 s to the end (20 s): 24 s of music
        let r = json!({"pieces": [{"src": 0, "len": tick(8.0)}, {"src": tick(4.0), "len": tick(16.0)}]});
        let b = remixed_beats(&track, &r).unwrap();
        assert_eq!(b.times.len(), 16 + 32);
        assert!((b.times[16] - 8.25).abs() < 1e-9, "{:?}", &b.times[14..18]);
        assert!(b.times.windows(2).all(|w| (w[1] - w[0] - 0.5).abs() < 1e-6), "an even beat across the joint");
        assert!(remixed_beats(&track, &json!({"pieces": [{"src": "x"}]})).is_none());
        assert!(remixed_beats(&track, &json!({})).is_none());
    }

    #[test]
    fn odd_shaped_shots_fill_the_frame_around_their_subject() {
        let frame = (1920.0, 1080.0);
        let covers = |aspect: f64, p: [f64; 2]| {
            let shown = if aspect < frame.0 / frame.1 { (frame.0, frame.0 / aspect) } else { (frame.1 * aspect, frame.1) };
            let eps = 1e-6;
            p[0] - shown.0 / 2.0 <= eps && p[0] + shown.0 / 2.0 >= frame.0 - eps && p[1] - shown.1 / 2.0 <= eps && p[1] + shown.1 / 2.0 >= frame.1 - eps
        };
        // a portrait phone clip, robot in its upper third: a tilt down across it, frame covered
        let portrait = 1080.0 / 1920.0;
        let (a, b) = reframe_path(portrait, frame, [0.5, 0.3], 0.01).unwrap();
        assert_eq!((a[0], b[0]), (960.0, 960.0), "no pan across a portrait clip");
        assert!(a[1] > b[1], "the picture moves up: the view tilts down ({a:?} → {b:?})");
        let shown_h = 1920.0 / portrait;
        assert!(((540.0 - a[1]) / shown_h + 0.5 - 0.24).abs() < 1e-9, "starts a tilt-width above the robot");
        assert!(covers(portrait, a) && covers(portrait, b));
        // a robot at the very top: held at the edge, never showing past the picture
        for f in [[0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 1.0], [f64::NAN, 0.4]] {
            let (a, b) = reframe_path(portrait, frame, f, 0.9).unwrap();
            assert!(covers(portrait, a) && covers(portrait, b), "{f:?}: {a:?} {b:?}");
        }
        // 4:3 crops a little top and bottom; a wide clip pans
        let (a, b) = reframe_path(4.0 / 3.0, frame, [0.5, 0.5], 0.05).unwrap();
        assert!(covers(4.0 / 3.0, a) && covers(4.0 / 3.0, b) && a[1] != b[1]);
        let (a, b) = reframe_path(21.0 / 9.0, frame, [0.8, 0.5], 0.05).unwrap();
        assert!(covers(21.0 / 9.0, a) && covers(21.0 / 9.0, b) && a[0] != b[0] && a[1] == 540.0);
        // the frame's own shape: no move; nonsense: nothing
        assert_eq!(reframe_path(16.0 / 9.0, frame, [0.2, 0.2], 0.5), Some(([960.0, 540.0], [960.0, 540.0])));
        assert_eq!(reframe_path(f64::NAN, frame, [0.5, 0.5], 0.1), None);
        assert_eq!(reframe_path(0.5, (0.0, 1080.0), [0.5, 0.5], 0.1), None);
        // the live case: a guide robot covering 38 % of a portrait clip is taller than the
        // frame shows, so the tilt runs most of the picture instead of a 6 % drift
        let (small_a, small_b) = reframe_path(portrait, frame, [0.5, 0.43], 0.02).unwrap();
        let (big_a, big_b) = reframe_path(portrait, frame, [0.5, 0.43], 0.38).unwrap();
        assert!((big_a[1] - big_b[1]).abs() > 3.0 * (small_a[1] - small_b[1]).abs(), "{big_a:?} {big_b:?} vs {small_a:?} {small_b:?}");
        assert!(covers(portrait, big_a) && covers(portrait, big_b));
        assert!(reframe_path(portrait, frame, [0.5, 0.5], f64::NAN).is_some(), "an unknown size drifts the least");
    }

    #[test]
    fn a_garbled_reading_is_told_from_a_good_one() {
        // tonight's robot 34: sixteen seconds of audio Whisper heard as "and it's arguing"
        let said = "In August 2025, a Promobot was greeting visitors in the D P World lobby. Promobot robots work in more than forty countries, including at the Dubai Mall and Dubai Police headquarters.";
        assert!(heard_share(said, "and it's arguing.") < 0.1);
        // a good reading, misheard names and all
        let said = "October 2022, and Ghee-tex Global was huge, with more than a hundred and seventy thousand visitors. First up, a robot misting disinfectant as it glided through the crowd.";
        let heard =
            "October 2022 and heat-exclobal was huge with more than 170,000 visitors. First up, a robot misting disinfectant as it glided through the crowd.";
        assert!(heard_share(said, heard) >= 0.7, "{}", heard_share(said, heard));
        // stems count; nothing to check is fine; hostile text is no problem
        assert_eq!(heard_share("Robots everywhere", "robot everywhere"), 1.0);
        assert_eq!(heard_share("42 !!", ""), 1.0);
        assert_eq!(heard_share("Hello there", ""), 0.0);
        assert!(heard_share(&"word ".repeat(10_000), "word").is_finite());
    }

    /// An engine that reads a cloned line badly at first: take 0 is garbled, later takes say it.
    struct Retakes {
        calls: std::sync::Mutex<Vec<(String, Value)>>,
        good_from: u64,
        /// Takes before this one come out strained (far above the voice's 100 Hz).
        calm_from: u64,
    }

    impl crate::Executor for Retakes {
        fn execute(&self, id: &str, p: Value) -> crate::ExecFuture {
            self.calls.lock().unwrap().push((id.to_string(), p.clone()));
            let r = match id {
                "tts.create" => {
                    let take = p["take"].as_u64().unwrap_or(0);
                    let pitch = if take >= self.calm_from { 104.0 } else { 150.0 };
                    Ok(json!({"clip": 10 + take, "item": 100 + take, "duration": 5.0 * TPS, "cached": false, "pitchHz": pitch}))
                }
                "tts.clonedVoices" => Ok(json!({"id": p["voice"], "pitchHz": 100.0})),
                "transcript.generate" => Ok(json!({"items": []})),
                "transcript.inspect" => {
                    let take = p["item"].as_u64().unwrap_or(0) - 100;
                    Ok(json!({"text": if take >= self.good_from { "Pepper greeted every visitor at the stand." } else { "and it's arguing." }}))
                }
                "edit.clear" => Ok(json!({})),
                other => Err(format!("unexpected {other}")),
            };
            Box::pin(async move { r })
        }
    }

    fn voice_tracks() -> Tracks {
        Tracks {
            fps: 30.0,
            aspect: 16.0 / 9.0,
            stabilize: false,
            fill: true,
            follow: false,
            frame: (1920.0, 1080.0),
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
            timeline: None,
            chips: Vec::new(),
        }
    }

    #[tokio::test]
    async fn a_garbled_cloned_line_is_read_again() {
        let say = "Pepper greeted every visitor at the stand.";
        let ex = Retakes { calls: Default::default(), good_from: 1, calm_from: 0 };
        let mut n = Narrations::default();
        let secs = narrate(&ex, &voice_tracks(), Some("clone:me"), say, say, Delivery::Warm, 3.0, &mut n).await.unwrap();
        assert_eq!(secs, 5.0);
        assert_eq!((n.clips.clone(), n.retakes), (vec![11], 1), "the second take is kept");
        let calls = ex.calls.lock().unwrap().clone();
        let cleared: Vec<&Value> = calls.iter().filter(|(c, _)| c == "edit.clear").map(|(_, p)| p).collect();
        assert_eq!(cleared, vec![&json!({"clips": [10]})], "the garbled take is cleared");
        assert_eq!(calls.iter().filter(|(c, p)| c == "tts.create" && p["take"] == 1).count(), 1);

        // never good: three takes, the best kept, and a note
        let ex = Retakes { calls: Default::default(), good_from: 99, calm_from: 0 };
        let mut n = Narrations::default();
        narrate(&ex, &voice_tracks(), Some("clone:me"), say, say, Delivery::Warm, 3.0, &mut n).await.unwrap();
        assert_eq!(n.clips.len(), 1);
        assert!(n.notes.iter().any(|l| l.contains("best")), "{:?}", n.notes);
        assert!(ex.calls.lock().unwrap().iter().filter(|(c, _)| c == "tts.create").count() <= MAX_TAKES as usize + 1);

        // a voice that is not cloned is not heard back
        let ex = Retakes { calls: Default::default(), good_from: 99, calm_from: 0 };
        let mut n = Narrations::default();
        narrate(&ex, &voice_tracks(), Some("basic-male"), say, say, Delivery::Plain, 0.0, &mut n).await.unwrap();
        assert!(ex.calls.lock().unwrap().iter().all(|(c, _)| c == "tts.create"));

        // said right but overacted (150 Hz from a 100 Hz voice): read again, the calm take kept
        let ex = Retakes { calls: Default::default(), good_from: 0, calm_from: 1 };
        let mut n = Narrations::default();
        narrate(&ex, &voice_tracks(), Some("clone:me"), say, say, Delivery::Excited, 3.0, &mut n).await.unwrap();
        assert_eq!((n.clips.clone(), n.retakes), (vec![11], 1));
    }

    #[test]
    fn tidy_cleans_model_lines() {
        assert_eq!(tidy("1. \"Hello there\"\n"), "Hello there");
        assert_eq!(tidy(&"word ".repeat(100)).split_whitespace().count(), MAX_LINE_WORDS, "one endless sentence is cut at the limit");
        // too long: it ends at the last full sentence that fits, never mid-sentence
        let long = format!("One short sentence. Then a second sentence that runs on {}", "and on ".repeat(30));
        assert_eq!(tidy(&long), "One short sentence.");
        let two = format!("{} {}", "First part of it here.", "word ".repeat(50));
        assert_eq!(tidy(&format!("{two} Done.")), "First part of it here.");
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
        assert_eq!(s.lines[1], "In November 2023, I met the Pepper humanoid robot.");
        assert_eq!(s.templated, 1);
        assert!(s.intro.contains("every robot"), "{}", s.intro);
        assert!(s.outro.contains("thanks"));
        // a model that does not answer at all still yields a full script
        let none = crate::llm::FixedModel::new(vec![]);
        let s = write_script(&none, &Options::default(), &entries, &mut |_| {}).await;
        assert_eq!((s.lines.len(), s.templated), (2, 2));
    }

    #[test]
    fn cuts_within_an_entry_move_onto_the_beat_and_keep_its_length() {
        let shot = Shot { item: 1, clip_seconds: 30.0, best: 10.0, ..Shot::default() };
        let w = |start: f64, len: f64, reach: f64| Window { shot: shot.clone(), start, len, reach };
        // at 120 bpm from t = 10: cuts at 12.3 and 15.1 move to 12.5 and 15.0
        let mut wins = vec![w(1.0, 2.3, 5.0), w(8.0, 2.8, 6.0), w(20.0, 3.0, 5.0)];
        snap_cuts(&mut wins, 10.0, &Beats::grid(120.0));
        let lens: Vec<f64> = wins.iter().map(|x| x.len).collect();
        assert!((lens[0] - 2.5).abs() < 1e-9 && (lens[0] + lens[1] - 5.0).abs() < 1e-9, "{lens:?}");
        assert!((lens.iter().sum::<f64>() - 8.1).abs() < 1e-9, "the entry keeps its length: {lens:?}");
        // a window at its reach does not grow, one at half a second does not shrink
        let mut tight = vec![w(1.0, 2.3, 2.3), w(8.0, 0.5, 6.0)];
        snap_cuts(&mut tight, 10.0, &Beats::grid(120.0));
        assert_eq!(tight.iter().map(|x| x.len).collect::<Vec<_>>(), vec![2.3, 0.5]);
        // the beats of a span, the tempo going on past the measured ones
        let b = Beats::measured(&[1.0, 1.5, 2.0], 0.5).unwrap();
        assert_eq!(b.list(0.0, 3.2), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        assert!(b.list(5.0, 1.0).is_empty() && b.list(f64::NAN, 3.0).is_empty());
        assert!(Beats::grid(200.0).list(0.0, 1e9).len() <= MAX_BEATS);
        // the bed comes up under a scene without words
        let mut e = vec![0.5; 11];
        lift_energy(&mut e, 10.0, &[(3.0, 5.5)]);
        assert_eq!(e, vec![0.5, 0.5, 0.5, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5]);
        lift_energy(&mut e, f64::NAN, &[(0.0, 10.0)]);
        assert_eq!(e[0], 0.5);
    }

    #[tokio::test]
    async fn scenes_get_no_words_from_the_model_and_are_never_robots() {
        let day = 1_665_741_668;
        let mut entries = group(
            &[fact(1, "Boston Dynamics Spot", "quadruped", day, 1.0), fact(2, "Pepper humanoid robot", "humanoid", day + 86_400 * 400, 1.0)],
            "robot",
            0.3,
        );
        let hall = Entry { number: 2, label: "the hall".into(), kind: "Scene".into(), unix: Some(day), ..Entry::default() };
        entries.insert(1, hall);
        // the model is asked about the two robots only, and its two lines go to them
        let model = crate::llm::FixedModel::new(vec![
            json!({"intro": "Every robot I met, in the order I met them.", "outro": "That is all for now, thanks for watching!"}),
            json!({"lines": ["I met Spot at an expo and it danced.", "Pepper waved at everyone in the shop."]}),
        ]);
        let s = write_script(&model, &Options::default(), &entries, &mut |_| {}).await;
        assert_eq!(s.lines, vec!["I met Spot at an expo and it danced.".to_string(), String::new(), "Pepper waved at everyone in the shop.".to_string()]);
        assert_eq!(s.templated, 0);
        assert!(entries[1].is_scene() && template_line(&entries[1]).is_empty());
        assert_eq!(entries[1].caption_text(), "the hall");
        // an analysed clip that calls itself a scene is still counted as a robot
        let odd = group(&[fact(3, "a robot", "scene", day, 1.0)], "robot", 0.3);
        assert!(!odd[0].is_scene());
        // the thumbnail never shows a scene
        let only = vec![Entry { shots: vec![Shot { item: 9, clip_seconds: 10.0, ..Shot::default() }], ..entries[1].clone() }, entries[0].clone()];
        assert_eq!(thumbnail_frame(&only, "robot").map(|p| p.2), Some(entries[0].number));
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

    #[tokio::test]
    async fn the_script_model_picks_how_each_line_is_said() {
        let day = 1_599_600_000;
        let facts: Vec<ClipFacts> = (0..4).map(|i| fact(i, &format!("robot model {i}"), "humanoid", day + 86_400 * 30 * i as i64, 1.0)).collect();
        let entries = group(&facts, "robot", 0.3);
        let model = crate::llm::FixedModel::new(vec![
            json!({"intro": "Welcome to every robot I have ever filmed.", "outro": "Thanks for watching all of these robots!"}),
            json!({"lines": ["It waved at me from across the room.", "This one could paint with real brushes.", "x", "It stood very still by the door."],
                   "deliveries": ["amazed", "dramatic", "excited", "nonsense"]}),
        ]);
        let s = write_script(&model, &Options::default(), &entries, &mut |_| {}).await;
        let asked = model.asked.lock().unwrap();
        assert!(asked[1].1.contains("\"deliveries\"") && asked[1].1.contains("curious"), "{}", asked[1].1);
        assert!(asked[1].0.contains("bare date"), "the system prompt asks for spoken sentences");
        // a delivery the model may choose, for a line it wrote; the rest are the defaults
        assert_eq!(s.delivery.get("1").map(String::as_str), Some("amazed"));
        assert_eq!(s.delivery.get("2"), None, "dramatic is left to people");
        assert_eq!(s.delivery.get("3"), None, "the line was templated");
        assert_eq!(s.delivery.get("4"), None, "unknown");
        assert_eq!(s.delivery_of("1"), Delivery::Amazed);
        assert_eq!((s.delivery_of("2"), s.delivery_of("intro"), s.delivery_of("outro")), (DEFAULT_DELIVERY, INTRO_DELIVERY, DEFAULT_DELIVERY));
        // a reviewed plan may say anything a person can choose, and an unknown one is the default
        let plan =
            Script { delivery: [("2".to_string(), "dramatic".to_string()), ("3".to_string(), "loud".to_string())].into_iter().collect(), ..Script::default() };
        assert_eq!((plan.delivery_of("2"), plan.delivery_of("3")), (Delivery::Dramatic, DEFAULT_DELIVERY));
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["delivery"]["2"], "dramatic");
        assert!(serde_json::to_value(Script::default()).unwrap().get("delivery").is_none(), "none written when there are none");
    }
}
