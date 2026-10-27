//! The director (`autocut.direct`): normal and 360° clips → one catchy edit on the beat, shot by
//! shot, as the local model (or, without one, rules) directs it.
//!
//! 1. **Media.** `items`, `paths` (files or folders), `insta360` (recordings by name: imported
//!    from Insta360 Studio's download folder, else downloaded from the user's Insta360 cloud) and,
//!    through the host, a Google Photos `album`; none: the project's movies and photos.
//! 2. **Analysis.** The scene index (`media.analyzeScenes`: cuts, quality, the camera's move, the
//!    describer's subjects and their boxes — 360° media is indexed as its stitched panorama, so a
//!    box is a direction), the speech (`media.understandSpeech`), and for each 360° shot where
//!    things happen round the sphere (`media.salience360`). Every moment is measured on its own
//!    frames and ranked; the unusable and the near-duplicates are left out ([`rank`]).
//! 3. **Plan.** A shot list on the music's beats ([`Plan`]): a hook, the story in filming order
//!    (flat and 360° shots taking turns where both exist), an ending. Each shot is a moment of a
//!    clip, its length in beats and its camera ([`Camera`]): for a 360° shot a move of the 360
//!    vocabulary — a tiny-planet intro, an orbit, a barrel roll, a whip pan between two of the
//!    things it saw, a reveal from the sky, a push in, a drift, a planet outro (`clip.vrMove`), a
//!    subject follow (`clip.follow360`), a hyperlapse (`clip.hyperlapse360`) — for a flat one a
//!    push in, a follow (Auto Reframe) or a speed ramp. The local model directs from the brief and
//!    the shots' facts ([`model_plan`]); its plan is checked and repaired ([`validate`]: every
//!    clip, moment, camera and length); without a model, or when its plan is unusable, rules
//!    direct ([`rules_plan`]). The report says who directed and what was repaired.
//! 4. **Commands.** The plan compiles to engine commands ([`compile`]); a shot's own commands run
//!    on its clip once it is placed. `dryRun` returns them with the plan, building nothing.
//! 5. **Build.** A sequence of the `aspect`; the music first (generated, or `music`), its beats
//!    the grid every cut lands on; the shots placed and moved; each cut made from what its two
//!    shots show and say ([`cuts`]: match cuts, whips where the camera is already moving, J and L
//!    cuts under speech, dissolves across time) in the edit's style, with the style's sounds
//!    (the subtle default: a quiet swish under whips and rolls and a chime under the title, no
//!    light effects); a careful colour pass (`clip.autoColor`: each shot's exposure and white
//!    balance measured and gently corrected, the cameras matched by saturation and contrast, a
//!    look only when asked, backed off where skin or greys got worse); the title in English
//!    inside the platform's safe zone; captions of English speech; each sound levelled and the
//!    mix at −14 LUFS; with `export` the video, a frame sheet, a colour before/after sheet, the
//!    report and QA beside it.
//!
//! Everything works without the models; the report's `fallbacks` say what ran instead.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use filmcraft_engine::project::scenes::{CameraMove, Energy, ShotType, ShotUse};
use filmcraft_engine::safe_zones::Platform;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::quick::{ItemSpeech, MediaItem, ShotInfo};
use crate::showcase::{Beats, on_frame, ticks};
use crate::{AutoCutError, Executor};

/// The length asked for by default, and the shortest and longest, seconds.
pub const DEFAULT_SECONDS: f64 = 30.0;
pub const MIN_SECONDS: f64 = 6.0;
pub const MAX_SECONDS: f64 = 300.0;
/// Frames a second of the edit.
pub const FPS: f64 = 30.0;
/// The context the directing model is given, tokens: the shots' facts of a long library and the
/// plan fit.
pub const CONTEXT_TOKENS: u32 = 16_384;
/// Shots the model is told about, at most (the best of a long library).
pub const MAX_CANDIDATES: usize = 48;
/// Reveals an edit has at most: each starts on the sky.
const MAX_REVEALS: usize = 2;
/// 360° shots looked round for action, at most.
const MAX_SALIENCE: usize = 40;
/// Longest brief, title and on-screen text, characters.
pub const MAX_BRIEF_CHARS: usize = 600;
pub const MAX_TITLE_CHARS: usize = 40;
/// Shots one edit holds, at most.
const MAX_SHOTS: usize = 120;
/// The tempo generated music plays at by default, and the range a brief or the model may ask.
pub const DEFAULT_BPM: f64 = 110.0;
const BPM_RANGE: (f64, f64) = (70.0, 150.0);
/// A whip between two 360° shots: how long each half turns, seconds.
const WHIP_HALF: f64 = 0.3;
/// A shot longer than this gives the director several moments, each about [`PIECE_SECONDS`]
/// long, at most [`MAX_PIECES`].
const SPLIT_SECONDS: f64 = 12.0;
const PIECE_SECONDS: f64 = 8.0;
const MAX_PIECES: usize = 4;
/// Two looks of one 360° shot are different subjects this far apart, degrees.
const LOOKS_APART: f64 = 50.0;
/// The moods generated music knows.
/// The colour pass's looks (`clip.autoColor`): ours, none by default.
const LOOKS: &[&str] = &["none", "punchy", "soft", "warm", "cool", "golden", "teal"];
const MOODS: &[&str] = &["epic", "tense", "uplifting", "calm", "happy", "dark", "dreamy", "energetic", "sad"];
/// Sound levels: the music carries the edit, the clips' own sound under it, the swishes and
/// chimes under that.
const BED_LUFS: f64 = -16.0;
const SOUND_LUFS: f64 = -27.0;
const EFFECTS_LUFS: f64 = -24.0;
/// J and L cuts.
const SPLIT_CMD: &str = filmcraft_engine::split_edit::SPLIT_CMD;

fn invalid(m: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(format!("direct: {}", m.into()))
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|f| f.is_finite())
}

fn chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------------------------

/// What the edit should be.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub brief: String,
    /// `9:16`, `16:9`, `1:1` or `4:5`.
    pub aspect: String,
    pub seconds: f64,
    pub platform: Platform,
    pub mood: Option<String>,
    pub bpm: Option<f64>,
    /// A music file, or none: generated.
    pub music: Option<String>,
    pub title: Option<String>,
    /// Captions of English speech.
    pub captions: bool,
    pub caption_style: String,
    /// Hide faces and plates (off unless asked: the people in a trip video are its subject).
    pub redact: Option<crate::privacy::Redact>,
    pub export: Option<String>,
    pub name: Option<String>,
    pub dry_run: bool,
    pub qa: Option<bool>,
    pub full_detail: usize,
    pub download: bool,
    pub seed: u64,
    /// How the cuts are made and what is heard on them ([`cuts::Style`]).
    pub style: cuts::Style,
    /// No style named (or `auto`): the director chooses it from the shots it planned
    /// ([`cuts::style_from_shots`]), the video's own `lightEffects`, `sounds` and `soundLevel`
    /// (kept in `style_given`) over it.
    pub auto_style: bool,
    pub style_given: Value,
    /// What the edit is about (`focus`): the places or the people; none (`auto`) — the director
    /// reads it from the brief and the footage ([`focus::focus_of`]).
    pub focus: Option<focus::Focus>,
    /// The careful colour pass (`clip.autoColor`): on unless switched off; its strength (0..1),
    /// and a look of our own (`none` by default) at a strength (%).
    pub color: bool,
    pub color_strength: f64,
    pub look: String,
    pub look_strength: f64,
}

/// The layout a brief asks for, from its words ("16:9 recap", "vertical reel").
pub fn aspect_from_brief(brief: &str) -> Option<&'static str> {
    let b = brief.to_ascii_lowercase();
    for (k, v) in [("16:9", "16:9"), ("9:16", "9:16"), ("1:1", "1:1"), ("4:5", "4:5")] {
        if b.contains(k) {
            return Some(v);
        }
    }
    let words: BTreeSet<String> = b.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect();
    let any = |ws: &[&str]| ws.iter().any(|w| words.contains(*w));
    if any(&["square"]) {
        Some("1:1")
    } else if any(&["landscape", "widescreen", "youtube", "recap", "film", "movie", "horizontal"]) {
        Some("16:9")
    } else if any(&["reel", "reels", "short", "shorts", "tiktok", "vertical", "story", "stories", "portrait", "instagram"]) {
        Some("9:16")
    } else {
        None
    }
}

/// The length a brief asks for ("a 60 s reel", "1 minute", "a 30-second recap"), seconds.
pub fn seconds_from_brief(brief: &str) -> Option<f64> {
    let b = brief.to_ascii_lowercase().replace('-', " ");
    let toks: Vec<&str> = b.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).collect();
    for (i, t) in toks.iter().enumerate() {
        // "60s", "90sec", "2min"
        let digits: String = t.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        let rest = t.get(digits.len()..).unwrap_or_default();
        let Ok(n) = digits.parse::<f64>() else { continue };
        let unit = if rest.is_empty() { toks.get(i + 1).copied().unwrap_or_default() } else { rest };
        let secs = if unit.starts_with("min") || unit == "m" {
            n * 60.0
        } else if unit.starts_with("sec") || unit == "s" {
            n
        } else {
            continue;
        };
        if secs.is_finite() && (MIN_SECONDS..=MAX_SECONDS).contains(&secs) {
            return Some(secs);
        }
    }
    None
}

/// Words a brief uses to ask, not to name what it is about.
const BRIEF_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "make",
    "create",
    "edit",
    "cut",
    "build",
    "give",
    "me",
    "us",
    "please",
    "of",
    "about",
    "from",
    "with",
    "for",
    "and",
    "my",
    "into",
    "reel",
    "reels",
    "short",
    "shorts",
    "video",
    "videos",
    "clip",
    "clips",
    "recap",
    "story",
    "tiktok",
    "instagram",
    "youtube",
    "vertical",
    "landscape",
    "square",
    "catchy",
    "fun",
    "cool",
    "quick",
    "nice",
    "s",
    "sec",
    "secs",
    "second",
    "seconds",
    "min",
    "mins",
    "minute",
    "minutes",
    "long",
    "all",
    "footage",
    "edit",
    "montage",
    "highlights",
    "highlight",
    "360",
    "normal",
    "using",
    "use",
];

/// A title from a brief: what it is about ("a 60 s reel of our trip to Dubai" → "Our Trip To
/// Dubai"), at most five words; none when nothing is left.
pub fn title_from_brief(brief: &str) -> Option<String> {
    let words: Vec<String> = brief
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .filter(|w| !w.chars().all(|c| c.is_ascii_digit() || c == ':'))
        .filter(|w| !BRIEF_WORDS.contains(&w.to_ascii_lowercase().as_str()))
        .map(|w| {
            let mut cs = w.chars();
            cs.next().map(|f| f.to_uppercase().chain(cs.flat_map(char::to_lowercase)).collect()).unwrap_or_default()
        })
        .take(5)
        .collect();
    let t = words.join(" ");
    (!t.trim().is_empty() && english(&t)).then(|| chars(&t, MAX_TITLE_CHARS))
}

/// A title as titles are set: an all-lower-case one gets capitals ("weekend in motion" →
/// "Weekend In Motion"); one with capitals of its own stays as written.
pub fn title_case(s: &str) -> String {
    if s.chars().any(char::is_uppercase) {
        return s.to_string();
    }
    s.split(' ')
        .map(|w| {
            let mut cs = w.chars();
            cs.next().map(|f| f.to_uppercase().chain(cs).collect()).unwrap_or_default()
        })
        .collect::<Vec<String>>()
        .join(" ")
}

/// Whether `s` is written in the Latin alphabet (the on-screen text is English).
pub fn english(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii() || matches!(c, '’' | '‘' | '“' | '”' | '–' | '—' | '…' | 'é' | 'è' | 'à' | 'ü' | 'ö' | 'ä' | 'ñ' | 'ç'))
}

/// The frame of a layout, pixels.
pub fn frame_of(aspect: &str) -> (u32, u32) {
    match aspect {
        "16:9" => (1920, 1080),
        "1:1" => (1080, 1080),
        "4:5" => (1080, 1350),
        _ => (1080, 1920),
    }
}

impl Options {
    pub fn from_params(p: &Value) -> Result<Options, AutoCutError> {
        let text = |k: &str| p[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        let brief = chars(&text("brief").unwrap_or_default(), MAX_BRIEF_CHARS);
        let aspect = text("aspect").or_else(|| aspect_from_brief(&brief).map(str::to_string)).unwrap_or_else(|| "9:16".into());
        if !matches!(aspect.as_str(), "9:16" | "1:1" | "4:5" | "16:9") {
            return Err(invalid("aspect: 9:16, 1:1, 4:5 or 16:9"));
        }
        let seconds = finite(&p["seconds"]).or_else(|| seconds_from_brief(&brief)).unwrap_or(DEFAULT_SECONDS).clamp(MIN_SECONDS, MAX_SECONDS);
        let platform = p["platform"].as_str().and_then(Platform::from_name).unwrap_or(Platform::Reels);
        let mood = text("mood").map(|m| m.to_ascii_lowercase());
        if let Some(m) = &mood
            && !MOODS.contains(&m.as_str())
        {
            return Err(invalid(format!("mood: one of {}", MOODS.join(", "))));
        }
        let bpm = finite(&p["bpm"]);
        if bpm.is_some_and(|b| !(BPM_RANGE.0..=BPM_RANGE.1).contains(&b)) {
            return Err(invalid(format!("bpm: {} to {}", BPM_RANGE.0, BPM_RANGE.1)));
        }
        let title = text("title").map(|t| chars(&t, MAX_TITLE_CHARS));
        if title.as_deref().is_some_and(|t| !english(t)) {
            return Err(invalid("title: the on-screen text is English"));
        }
        let look = text("look").map_or_else(|| "none".to_string(), |l| l.to_ascii_lowercase());
        if !LOOKS.contains(&look.as_str()) {
            return Err(invalid(format!("look: one of {}", LOOKS.join(", "))));
        }
        let auto_style = p["style"].as_str().is_none_or(|v| v.trim().is_empty() || v.trim().eq_ignore_ascii_case("auto"));
        let mut style_given = json!({});
        for k in ["lightEffects", "sounds", "soundLevel"] {
            if !p[k].is_null() {
                style_given[k] = p[k].clone();
            }
        }
        let focus = focus::Focus::parse(p["focus"].as_str().unwrap_or_default()).map_err(invalid)?;
        let style =
            if auto_style { cuts::Style::from_params(&style_given, &cuts::Style::default()) } else { cuts::Style::from_params(p, &cuts::Style::default()) };
        let redact = match &p["redact"] {
            Value::Null | Value::Bool(false) => None,
            Value::String(s) if matches!(s.trim().to_ascii_lowercase().as_str(), "off" | "none" | "false" | "") => None,
            v => crate::privacy::Redact::from_plan(Some(v))?,
        };
        Ok(Options {
            brief,
            aspect,
            seconds,
            platform,
            mood,
            bpm,
            music: text("music").filter(|m| !m.eq_ignore_ascii_case("generate")),
            title,
            captions: p["captions"].as_bool().unwrap_or(true),
            caption_style: text("captionStyle").unwrap_or_else(|| "clean".into()),
            redact,
            export: text("export"),
            name: text("name").map(|n| chars(&n, 200)),
            dry_run: p["dryRun"].as_bool().unwrap_or(false),
            qa: p["qa"].as_bool(),
            full_detail: p["fullDetail"].as_u64().map_or(8, |n| n.min(64) as usize),
            download: p["download"].as_bool().unwrap_or(false),
            seed: p["seed"].as_u64().unwrap_or(7),
            style: style.map_err(invalid)?,
            auto_style,
            style_given,
            focus,
            color: p["color"].as_bool().unwrap_or(true),
            color_strength: finite(&p["colorStrength"]).map_or(0.7, |v| v.clamp(0.0, 1.0)),
            look,
            look_strength: finite(&p["lookStrength"]).map_or(25.0, |v| v.clamp(0.0, 100.0)),
        })
    }

    pub fn frame(&self) -> (u32, u32) {
        frame_of(&self.aspect)
    }

    /// A vertical layout, whose text keeps to a platform's safe zone.
    pub fn vertical(&self) -> bool {
        self.aspect != "16:9"
    }
}

// ---------------------------------------------------------------------------------------------
// candidates
// ---------------------------------------------------------------------------------------------

/// What a clip is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Flat,
    #[serde(rename = "360")]
    Vr,
    Photo,
}

/// A direction worth looking at in a 360° shot.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Look {
    pub id: String,
    pub yaw: f64,
    pub pitch: f64,
    /// What is there ("a man waving", "something moving").
    pub what: String,
    /// A named subject the describer saw (else where the motion or detail is).
    pub named: bool,
    /// How big it is on the sphere, degrees across and down (a named subject's box).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<(f64, f64)>,
}

/// One shot the director may use.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub id: String,
    pub item: u64,
    pub name: String,
    pub kind: Kind,
    /// Media seconds of the shot, and its best moment.
    pub start: f64,
    pub end: f64,
    pub key: f64,
    /// The whole item's length, seconds (room for a longer window, a hyperlapse).
    pub media_seconds: f64,
    pub caption: String,
    /// 0 calm … 1 high energy.
    pub energy: f64,
    /// 0..1: sharp, well exposed, steady.
    pub quality: f64,
    /// 0..1: how well it grabs attention.
    pub hook: f64,
    pub wide: bool,
    /// Indoors, by its description: a reveal from the sky would show the ceiling.
    pub indoor: bool,
    pub moving: bool,
    /// Someone talking to the camera; English kept speech in it.
    pub talking: bool,
    pub said: Option<String>,
    /// 360°: the directions worth looking at, best first.
    pub looks: Vec<Look>,
    /// Flat: the subject's box, shares of the picture.
    pub subject: Option<[f64; 4]>,
    /// Flat: width over height of the picture.
    pub aspect: f64,
    pub order: usize,
    /// Its shot in its item's scene index (moments of one shot are not copies of each other).
    pub shot: usize,
    /// How its camera moves (a pan is carried across a match cut, or whipped).
    pub pan: CameraMove,
    /// When it was filmed (Unix seconds), when known: a jump in time is dissolved across.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captured: Option<i64>,
    /// What its picture and sound score ([`rank::Scores`]), and its rank from them.
    pub scores: rank::Scores,
    pub rank: f64,
    /// Its shot's image embedding (near-duplicates are told by it).
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

impl Candidate {
    pub fn seconds(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
    fn look(&self, id: &str) -> Option<&Look> {
        self.looks.iter().find(|l| l.id == id)
    }
    /// A second look far enough from the first to whip to.
    fn second_look(&self) -> Option<&Look> {
        let first = self.looks.first()?;
        self.looks.iter().skip(1).find(|l| angle_between(l.yaw, first.yaw) >= LOOKS_APART)
    }
}

/// Words that put a shot indoors.
const INDOOR_WORDS: &[&str] = &[
    "indoor",
    "indoors",
    "inside",
    "room",
    "lobby",
    "hall",
    "elevator",
    "lift",
    "corridor",
    "hallway",
    "office",
    "kitchen",
    "bedroom",
    "restaurant",
    "cafe",
    "café",
    "mall",
    "store",
    "shop",
    "lounge",
    "airport",
    "terminal",
    "hotel",
    "ceiling",
    "museum",
    "gym",
    "classroom",
    "studio",
    "garage",
    "car interior",
];

/// Whether a description puts a shot indoors.
pub fn indoors(text: &str) -> bool {
    let t = text.to_lowercase();
    let words: BTreeSet<&str> = t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    INDOOR_WORDS.iter().any(|w| if w.contains(' ') { t.contains(w) } else { words.contains(w) })
}

fn angle_between(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn energy_of(s: &ShotInfo) -> f64 {
    match s.description().map(|d| d.energy) {
        Some(Energy::Low) => 0.3,
        Some(Energy::Medium) => 0.6,
        Some(Energy::High) => 0.9,
        _ => (0.3 + f64::from(s.shot.quality.motion)).clamp(0.2, 0.85),
    }
}

/// A box's centre as the direction of a 360° panorama.
fn direction_of(b: [f32; 4]) -> (f64, f64) {
    let (x, y) = (f64::from(b[0]) + f64::from(b[2]) / 2.0, f64::from(b[1]) + f64::from(b[3]) / 2.0);
    filmcraft_engine::project::vr::panorama_direction(x, y)
}

/// The kept English speech of `item` inside `[a, b]` media seconds.
fn said_in(speech: &[ItemSpeech], item: u64, a: f64, b: f64) -> Option<String> {
    let s = speech.iter().find(|s| s.item == item && s.language.starts_with("en"))?;
    let words: Vec<&str> = s.kept.iter().filter(|k| k.end > a && k.start < b).map(|k| k.text.trim()).filter(|t| !t.is_empty()).collect();
    (!words.is_empty()).then(|| chars(&words.join(" "), 200))
}

/// The moments of a shot the director may use: the whole shot, or — longer than
/// [`SPLIT_SECONDS`] — a few pieces of it, each `(start, end, its best moment)`; the shot's own
/// best moment is in the piece that holds it.
pub fn pieces(s: &ShotInfo) -> Vec<(f64, f64, f64)> {
    let secs = s.seconds();
    if s.still || !secs.is_finite() || secs <= SPLIT_SECONDS {
        return vec![(s.start, s.end, s.keyframe)];
    }
    let n = ((secs / PIECE_SECONDS).ceil() as usize).clamp(2, MAX_PIECES);
    let len = secs / n as f64;
    (0..n)
        .map(|k| {
            let (a, b) = (s.start + len * k as f64, s.start + len * (k + 1) as f64);
            let key = if s.keyframe >= a && s.keyframe < b { s.keyframe } else { (a + b) / 2.0 };
            (a, b, key)
        })
        .collect()
}

/// The shots as candidates, a long one as several ([`pieces`]): `vr` the 360° items,
/// `salience` the directions `media.salience360` found per (item, shot, piece), `aspects` the
/// flat items' shapes.
pub fn candidates(
    shots: &[ShotInfo],
    media: &[MediaItem],
    vr: &BTreeSet<u64>,
    salience: &HashMap<(u64, usize, usize), Vec<(f64, f64, String)>>,
    aspects: &HashMap<u64, f64>,
    speech: &[ItemSpeech],
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for s in shots {
        let parts = pieces(s);
        let many = parts.len() > 1;
        for (part, (start, end, key)) in parts.into_iter().enumerate() {
            out.push(candidate(s, out.len() + 1, part, many, (start, end, key), media, vr, salience, aspects, speech));
        }
    }
    out
}

/// One candidate: piece `part` of shot `s` (`many`: the shot has several).
#[allow(clippy::too_many_arguments)]
fn candidate(
    s: &ShotInfo,
    number: usize,
    part: usize,
    many: bool,
    (start, end, key): (f64, f64, f64),
    media: &[MediaItem],
    vr: &BTreeSet<u64>,
    salience: &HashMap<(u64, usize, usize), Vec<(f64, f64, String)>>,
    aspects: &HashMap<u64, f64>,
    speech: &[ItemSpeech],
) -> Candidate {
    {
        let kind = if s.still {
            Kind::Photo
        } else if vr.contains(&s.item) {
            Kind::Vr
        } else {
            Kind::Flat
        };
        let id = format!("c{number}");
        let d = s.description();
        let mut looks: Vec<Look> = Vec::new();
        let mut subject = None;
        if let Some(d) = d {
            for sub in d.subjects.iter().take(6) {
                let Some(b) = sub.bbox.filter(|b| b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0) else { continue };
                if kind == Kind::Vr {
                    let (yaw, pitch) = direction_of(b);
                    if looks.iter().all(|l| angle_between(l.yaw, yaw) >= 20.0) {
                        let size = (f64::from(b[2]) * 360.0, f64::from(b[3]) * 180.0);
                        looks.push(Look { id: String::new(), yaw, pitch: pitch.clamp(-60.0, 60.0), what: chars(&sub.name, 60), named: true, size: Some(size) });
                    }
                } else if subject.is_none() {
                    subject = Some(b.map(f64::from));
                }
            }
        }
        if kind == Kind::Vr {
            for (yaw, pitch, what) in salience.get(&(s.item, s.index, part)).map(Vec::as_slice).unwrap_or_default() {
                if looks.iter().all(|l| angle_between(l.yaw, *yaw) >= 30.0) {
                    looks.push(Look { id: String::new(), yaw: *yaw, pitch: pitch.clamp(-45.0, 45.0), what: what.clone(), named: false, size: None });
                }
            }
            if looks.is_empty() {
                looks.push(Look { id: String::new(), yaw: 0.0, pitch: 0.0, what: "straight ahead".into(), named: false, size: None });
            }
            for (k, l) in looks.iter_mut().enumerate() {
                l.id = format!("{id}{}", char::from(b'a' + (k.min(25) as u8)));
            }
        }
        let uses = d.map(|d| d.uses.clone()).unwrap_or_default();
        let media_seconds = media.iter().find(|m| m.item == s.item).map_or(s.end, |m| m.seconds.max(s.end));
        let said = if s.speech > 0.2 { said_in(speech, s.item, start, end) } else { None };
        let energy = energy_of(s);
        let scores = rank::scores_of(s, said.as_deref(), s.speech, energy);
        let caption = if many { format!("{} (moment {} of the shot)", chars(&s.caption(), 140), part + 1) } else { chars(&s.caption(), 160) };
        Candidate {
            id,
            item: s.item,
            name: s.name.clone(),
            kind,
            start,
            end,
            key,
            media_seconds,
            caption,
            energy,
            quality: scores.quality(),
            hook: f64::from(filmcraft_scenes::best::score(&s.shot, ShotUse::Hook).0.clamp(0.0, 1.0)),
            wide: d.is_some_and(|d| d.shot_type == ShotType::Wide) || uses.contains(&ShotUse::Establishing),
            indoor: indoors(&format!("{} {}", s.caption(), d.map(|d| d.setting.as_str()).unwrap_or_default())),
            moving: s.shot.camera != CameraMove::Static || d.is_some_and(|d| !d.actions.is_empty()),
            talking: s.talking,
            said,
            looks,
            subject,
            aspect: aspects.get(&s.item).copied().filter(|a| a.is_finite() && *a > 0.0).unwrap_or(16.0 / 9.0),
            order: s.order,
            shot: s.index,
            pan: s.shot.camera,
            captured: None,
            rank: scores.rank(kind),
            scores,
            embedding: s.shot.embedding.0.clone(),
        }
    }
}

/// The candidates worth telling the model about: the best by rank and hook, at most `n`, in
/// filming order (every clip keeps at least its best shot).
pub fn shortlist(cands: &[Candidate], n: usize) -> Vec<&Candidate> {
    let worth = |c: &Candidate| 0.6 * c.rank + 0.25 * c.hook + 0.15 * c.energy;
    let mut best_of_item: HashMap<u64, (f64, usize)> = HashMap::new();
    for (i, c) in cands.iter().enumerate() {
        let w = worth(c);
        let e = best_of_item.entry(c.item).or_insert((w, i));
        if w > e.0 {
            *e = (w, i);
        }
    }
    let mut chosen: BTreeSet<usize> = best_of_item.values().map(|(_, i)| *i).collect();
    let mut rest: Vec<(f64, usize)> = cands.iter().enumerate().filter(|(i, _)| !chosen.contains(i)).map(|(i, c)| (worth(c), i)).collect();
    rest.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, i) in rest {
        if chosen.len() >= n {
            break;
        }
        chosen.insert(i);
    }
    let mut out: Vec<&Candidate> = chosen.into_iter().filter_map(|i| cands.get(i)).collect();
    out.sort_by(|a, b| a.order.cmp(&b.order).then(a.start.total_cmp(&b.start)));
    out.truncate(n.max(1));
    out
}

// ---------------------------------------------------------------------------------------------
// the plan
// ---------------------------------------------------------------------------------------------

/// What the camera does in a shot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Camera {
    /// Held, filling the frame.
    Hold,
    /// A slow push in (flat: Ken Burns on its subject; 360°: the view narrowing).
    PushIn,
    /// Kept on its subject (flat: Auto Reframe; 360°: `clip.follow360`).
    Follow,
    /// A speed ramp (`clip.speedRamp`, its length kept).
    SpeedRamp,
    PlanetIn,
    PlanetOut,
    Orbit,
    BarrelRoll,
    WhipPan,
    Reveal,
    Drift,
    Hyperlapse,
}

impl Camera {
    pub const ALL: [Camera; 12] = [
        Camera::Hold,
        Camera::PushIn,
        Camera::Follow,
        Camera::SpeedRamp,
        Camera::PlanetIn,
        Camera::PlanetOut,
        Camera::Orbit,
        Camera::BarrelRoll,
        Camera::WhipPan,
        Camera::Reveal,
        Camera::Drift,
        Camera::Hyperlapse,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Camera::Hold => "hold",
            Camera::PushIn => "pushIn",
            Camera::Follow => "follow",
            Camera::SpeedRamp => "speedRamp",
            Camera::PlanetIn => "planetIn",
            Camera::PlanetOut => "planetOut",
            Camera::Orbit => "orbit",
            Camera::BarrelRoll => "barrelRoll",
            Camera::WhipPan => "whipPan",
            Camera::Reveal => "reveal",
            Camera::Drift => "drift",
            Camera::Hyperlapse => "hyperlapse",
        }
    }

    pub fn from_id(s: &str) -> Option<Camera> {
        let k: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        Camera::ALL.into_iter().find(|c| c.id().to_ascii_lowercase() == k).or(match k.as_str() {
            "tinyplanet" | "planet" | "littleplanet" => Some(Camera::PlanetIn),
            "whip" => Some(Camera::WhipPan),
            "roll" => Some(Camera::BarrelRoll),
            "still" | "static" | "none" => Some(Camera::Hold),
            "zoom" | "zoomin" | "kenburns" => Some(Camera::PushIn),
            "track" | "tracking" => Some(Camera::Follow),
            "ramp" => Some(Camera::SpeedRamp),
            "timelapse" => Some(Camera::Hyperlapse),
            _ => None,
        })
    }

    /// What it means, for the model.
    fn meaning(self) -> &'static str {
        match self {
            Camera::Hold => "held still on the shot",
            Camera::PushIn => "a slow push in",
            Camera::Follow => "the camera keeps a moving subject in frame",
            Camera::SpeedRamp => "a speed ramp: fast, then slow on the action",
            Camera::PlanetIn => "360 only: a tiny planet spinning open into the view (the signature opener)",
            Camera::PlanetOut => "360 only: the view folding into a tiny planet (a closer)",
            Camera::Orbit => "360 only: a full turn round the scene",
            Camera::BarrelRoll => "360 only: the view rolls once (use at most once)",
            Camera::WhipPan => "360 only: a fast pan from `look` to `to` (two things in the scene)",
            Camera::Reveal => "360 only, outdoors only: from the sky down to `look`",
            Camera::Drift => "360 only: a slow turn",
            Camera::Hyperlapse => "360 only: a long stretch sped up and steadied",
        }
    }

    pub fn for_kind(self, k: Kind) -> bool {
        match k {
            Kind::Vr => true,
            Kind::Flat => matches!(self, Camera::Hold | Camera::PushIn | Camera::Follow | Camera::SpeedRamp),
            Kind::Photo => matches!(self, Camera::Hold | Camera::PushIn),
        }
    }

    /// The shortest shot it reads in, seconds.
    pub fn min_seconds(self) -> f64 {
        match self {
            Camera::PlanetIn => 2.4,
            Camera::PlanetOut => 2.0,
            Camera::Orbit => 3.0,
            Camera::Follow | Camera::Hyperlapse => 2.4,
            Camera::BarrelRoll => 1.5,
            Camera::Reveal => 1.8,
            Camera::WhipPan => 1.2,
            Camera::Drift => 1.5,
            Camera::SpeedRamp => 1.6,
            Camera::Hold | Camera::PushIn => 0.6,
        }
    }

    /// How much of its media a second of the shot plays.
    fn speed(self) -> f64 {
        if self == Camera::Hyperlapse { 6.0 } else { 1.0 }
    }
}

/// How a shot comes in ([`cuts::joins`] picks it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Enter {
    #[default]
    Cut,
    /// A cut lined up with the shot before (the subject's place, the pan, the framing).
    Match,
    /// A cross dissolve (into the ending, across a jump in time).
    Dissolve,
    /// A whip between two 360° shots: the one before turns away, this one turns in.
    Whip,
    /// A whip-pan transition, the way the camera moves.
    Swipe,
    /// A zoom punch across a jump in time (energetic).
    Punch,
    /// A light leak across a jump in time (energetic, light effects allowed).
    LightLeak,
    /// A dip to black across days (cinematic).
    DipToBlack,
}

/// One shot of the edit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedShot {
    /// Its candidate's id.
    pub clip: String,
    /// Its media's name (a saved plan's shots are found again by it when it is revised).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub file: String,
    /// Where it starts in the media, seconds.
    pub from: f64,
    pub beats: u32,
    pub camera: Camera,
    /// 360°: where it looks, and where a whip pan goes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    #[serde(default)]
    pub enter: Enter,
    /// A match cut: where its subject sits across the frame (shares), lined up with the shot
    /// before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<f64>,
    /// Its sound starts this many seconds before its picture (a J cut), runs on this long after
    /// it (an L cut).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub lead: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tail: f64,
    /// Why it comes in as it does.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cut: String,
    /// Text over it (English).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Its colour as a revision asked for it, over the colour pass ([`revise::Grade`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<revise::Grade>,
    #[serde(default)]
    pub why: String,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// The edit as directed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub title: String,
    #[serde(default)]
    pub mood: String,
    pub bpm: f64,
    pub shots: Vec<PlannedShot>,
    /// `model:<id>` or `rules`.
    #[serde(default)]
    pub by: String,
    /// What was repaired or decided, and why.
    #[serde(default)]
    pub notes: Vec<String>,
    /// The cutting style it was made in (`subtle`, `energetic`, `cinematic`): a revision keeps it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

impl Plan {
    pub fn beats(&self) -> u32 {
        self.shots.iter().map(|s| s.beats).sum()
    }
}

/// Beats of a beat `beat` seconds long that cover `secs`, even when more than two.
fn beats_for(secs: f64, beat: f64) -> u32 {
    let b = if beat.is_finite() && beat > 0.05 { beat } else { 0.5 };
    let n = (secs / b).ceil().clamp(1.0, 64.0) as u32;
    if n > 2 && n % 2 == 1 { n + 1 } else { n }
}

/// The longest a shot holds, seconds (an orbit, a hyperlapse a little longer): a reel keeps
/// moving.
pub const MAX_SHOT_SECONDS: f64 = 6.0;

/// The beat of a tempo, seconds.
pub fn beat_of(bpm: f64) -> f64 {
    60.0 / bpm.clamp(BPM_RANGE.0, BPM_RANGE.1)
}

/// The mood a brief sounds like.
pub fn mood_from_brief(brief: &str) -> &'static str {
    let b = brief.to_ascii_lowercase();
    let has = |ws: &[&str]| ws.iter().any(|w| b.contains(w));
    if has(&["calm", "relax", "peace", "slow", "sunset", "chill"]) {
        "calm"
    } else if has(&["epic", "adventure", "mountain", "drone", "travel"]) {
        "uplifting"
    } else if has(&["party", "hype", "energy", "sport", "race", "action"]) {
        "energetic"
    } else if has(&["dream", "memories", "memory", "nostalg"]) {
        "dreamy"
    } else {
        "happy"
    }
}

/// Where a shot of `need` media seconds starts in candidate `c`: its best moment a third in,
/// inside the media.
fn window_from(c: &Candidate, need: f64) -> Option<f64> {
    if c.kind == Kind::Photo {
        return Some(0.0);
    }
    if !(need.is_finite() && need > 0.0) || need > c.media_seconds + 1e-6 {
        return None;
    }
    let lo = if c.seconds() >= need { c.start } else { 0.0 };
    let hi = if c.seconds() >= need { c.end - need } else { c.media_seconds - need };
    Some((c.key - need / 3.0).clamp(lo, hi.max(lo)))
}

/// Rules' camera for a body shot of candidate `c` lasting `secs`, given the moves used so far.
fn rules_camera(c: &Candidate, secs: f64, used: &HashMap<Camera, usize>, last: Option<Camera>, frame_aspect: f64) -> Camera {
    let n = |m: Camera| used.get(&m).copied().unwrap_or(0);
    let fresh = |m: Camera| Some(m) != last;
    match c.kind {
        Kind::Photo => Camera::PushIn,
        Kind::Flat => {
            if c.energy >= 0.75 && secs >= Camera::SpeedRamp.min_seconds() && n(Camera::SpeedRamp) < 2 && fresh(Camera::SpeedRamp) {
                Camera::SpeedRamp
            } else if (c.aspect - frame_aspect).abs() / frame_aspect > 0.05 && c.moving && secs >= 1.2 {
                Camera::Follow
            } else {
                Camera::PushIn
            }
        }
        Kind::Vr => {
            let options = [
                (c.second_look().is_some() && n(Camera::WhipPan) < 3 && secs >= Camera::WhipPan.min_seconds(), Camera::WhipPan),
                (c.moving && c.looks.first().is_some_and(|l| l.named) && secs >= Camera::Follow.min_seconds() && n(Camera::Follow) < 3, Camera::Follow),
                (c.media_seconds >= secs.max(2.4) * Camera::Hyperlapse.speed() && c.energy <= 0.5 && n(Camera::Hyperlapse) == 0, Camera::Hyperlapse),
                (c.wide && !c.indoor && n(Camera::Reveal) < 2 && secs >= Camera::Reveal.min_seconds(), Camera::Reveal),
                (c.wide && secs >= Camera::Orbit.min_seconds() && n(Camera::Orbit) < 2, Camera::Orbit),
                (c.energy >= 0.75 && n(Camera::BarrelRoll) == 0 && secs >= Camera::BarrelRoll.min_seconds(), Camera::BarrelRoll),
                // a named subject is kept in view (a push in, tracked: `compile`); a view without one drifts
                (true, if c.looks.first().is_some_and(|l| l.named) || n(Camera::Drift) > n(Camera::PushIn) { Camera::PushIn } else { Camera::Drift }),
            ];
            options.iter().find(|(ok, m)| *ok && fresh(*m)).map_or(Camera::Drift, |(_, m)| *m)
        }
    }
}

/// The plan rules make: the hook (a 360° shot opening as a tiny planet when there is a good
/// one), the story in filming order — flat and 360° shots taking turns where both are left —
/// each 360° shot's camera chosen by what it shows ([`rules_camera`]), and an ending (a 360°
/// shot folding into a planet, else a wide shot dissolved into), every length in beats.
pub fn rules_plan(cands: &[Candidate], o: &Options, beat: f64, bpm: f64) -> Plan {
    let total = ((o.seconds / beat).round() as u32).max(4);
    let frame_aspect = {
        let (w, h) = o.frame();
        f64::from(w) / f64::from(h.max(1))
    };
    let good: Vec<&Candidate> = {
        let ok: Vec<&Candidate> = cands.iter().filter(|c| c.quality >= 0.2).collect();
        if ok.len() >= 3 { ok } else { cands.iter().collect() }
    };
    let mut notes = Vec::new();
    let title = o.title.clone().or_else(|| title_from_brief(&o.brief)).unwrap_or_else(|| "Highlights".into());
    let mood = o.mood.clone().unwrap_or_else(|| mood_from_brief(&o.brief).to_string());
    let mut plan = Plan { title: title.clone(), mood, bpm, shots: Vec::new(), by: "rules".into(), notes: Vec::new(), style: None };
    if good.is_empty() {
        return plan;
    }
    let hook_score = |c: &Candidate| c.hook + 0.3 * c.energy + 0.2 * c.quality;
    let mut by_hook: Vec<&Candidate> = good.clone();
    by_hook.sort_by(|a, b| hook_score(b).total_cmp(&hook_score(a)));
    // the hook: a 360° shot among the strongest opens as a tiny planet
    let hook = by_hook.iter().take(4).find(|c| c.kind == Kind::Vr && window_from(c, Camera::PlanetIn.min_seconds()).is_some()).or(by_hook.first()).copied();
    let Some(hook) = hook else { return plan };
    let hook_cam = if hook.kind == Kind::Vr { Camera::PlanetIn } else { rules_camera(hook, 2.0, &HashMap::new(), None, frame_aspect) };
    let hook_beats = beats_for(hook_cam.min_seconds().max(2.0), beat);
    // the ending: a 360° shot of another clip folds into a planet, else a wide shot
    let end = good
        .iter()
        .rev()
        .find(|c| c.kind == Kind::Vr && c.item != hook.item && window_from(c, Camera::PlanetOut.min_seconds()).is_some())
        .or_else(|| good.iter().rev().find(|c| c.id != hook.id && c.wide))
        .or_else(|| good.iter().rev().find(|c| c.id != hook.id))
        .copied();
    let end_cam = match end {
        Some(c) if c.kind == Kind::Vr => Camera::PlanetOut,
        Some(_) => Camera::PushIn,
        None => Camera::Hold,
    };
    let end_beats = if end.is_some() { beats_for(end_cam.min_seconds().max(2.0), beat) } else { 0 };
    let body_room = total.saturating_sub(hook_beats + end_beats);
    // the body: filming order, the kinds taking turns
    let used_ids: BTreeSet<&str> = [Some(hook.id.as_str()), end.map(|c| c.id.as_str())].into_iter().flatten().collect();
    let mut flat: std::collections::VecDeque<&Candidate> = good.iter().filter(|c| c.kind != Kind::Vr && !used_ids.contains(c.id.as_str())).copied().collect();
    let mut vr: std::collections::VecDeque<&Candidate> = good.iter().filter(|c| c.kind == Kind::Vr && !used_ids.contains(c.id.as_str())).copied().collect();
    let mut used: HashMap<Camera, usize> = HashMap::new();
    used.insert(hook_cam, 1);
    let mut last_vr: Option<Camera> = (hook.kind == Kind::Vr).then_some(hook_cam);
    let mut body: Vec<PlannedShot> = Vec::new();
    let mut room = body_room;
    let mut last_kind = hook.kind;
    let mut last_item = hook.item;
    while room >= 2 && (!flat.is_empty() || !vr.is_empty()) && body.len() < MAX_SHOTS {
        // the other kind when it has shots left, else the one that has; the earliest first
        let pick_vr = match (flat.front(), vr.front()) {
            (Some(f), Some(v)) => {
                if last_kind == Kind::Vr {
                    false
                } else if last_kind == Kind::Flat || last_kind == Kind::Photo {
                    true
                } else {
                    (v.order, v.start) < (f.order, f.start)
                }
            }
            (None, Some(_)) => true,
            _ => false,
        };
        let Some(c) = (if pick_vr { vr.pop_front() } else { flat.pop_front() }) else { break };
        // two shots of one clip in a row only when nothing else is left
        if c.item == last_item && (!flat.is_empty() || !vr.is_empty()) && body.len() + 1 < 64 {
            let q = if pick_vr { &mut vr } else { &mut flat };
            if q.len() > 1 {
                q.insert(1, c);
                continue;
            }
        }
        let base: f64 = if c.kind == Kind::Vr {
            2.0
        } else if c.energy >= 0.6 {
            1.0
        } else {
            1.8
        };
        let cam = rules_camera(c, base.max(1.2), &used, if c.kind == Kind::Vr { last_vr } else { None }, frame_aspect);
        let beats = beats_for(base.max(cam.min_seconds()), beat).min(room.max(2));
        let need = f64::from(beats) * beat * cam.speed();
        let Some(from) = window_from(c, need) else { continue };
        let (look, to) = match (c.kind, cam) {
            (Kind::Vr, Camera::WhipPan) => (c.looks.first().map(|l| l.id.clone()), c.second_look().map(|l| l.id.clone())),
            (Kind::Vr, _) => (c.looks.first().map(|l| l.id.clone()), None),
            _ => (None, None),
        };
        let enter = if c.kind == Kind::Vr && last_kind == Kind::Vr && used.get(&Camera::WhipPan).copied().unwrap_or(0) + whips(&body) < 3 && !body.is_empty() {
            Enter::Whip
        } else {
            Enter::Cut
        };
        *used.entry(cam).or_insert(0) += 1;
        if c.kind == Kind::Vr {
            last_vr = Some(cam);
        }
        last_kind = c.kind;
        last_item = c.item;
        room = room.saturating_sub(beats);
        body.push(PlannedShot {
            clip: c.id.clone(),
            file: String::new(),
            grade: None,
            from,
            beats,
            camera: cam,
            look,
            to,
            enter,
            align: None,
            lead: 0.0,
            tail: 0.0,
            cut: String::new(),
            text: None,
            why: format!("{} in filming order", c.caption),
        });
    }
    // beats left over: the longest-held shots take them, two at a time
    let mut k = 0usize;
    while room >= 1 && !body.is_empty() && k < 256 {
        let i = k % body.len();
        if let Some(s) = body.get_mut(i) {
            let add = room.min(2);
            let c = cands.iter().find(|c| c.id == s.clip);
            let need = f64::from(s.beats + add) * beat * s.camera.speed();
            if let Some(from) = c.and_then(|c| window_from(c, need)) {
                s.beats += add;
                s.from = from;
                room -= add;
            }
        }
        k += 1;
    }
    if room > 0 {
        notes.push(format!("{room} beat(s) left over: the clips have no more footage to hold for them"));
    }
    let hook_need = f64::from(hook_beats) * beat * hook_cam.speed();
    let hook_from = window_from(hook, hook_need).unwrap_or(hook.start);
    let hook_look = hook.looks.first().map(|l| l.id.clone());
    plan.shots.push(PlannedShot {
        clip: hook.id.clone(),
        file: String::new(),
        grade: None,
        from: hook_from,
        beats: hook_beats,
        camera: hook_cam,
        look: hook_look,
        to: None,
        enter: Enter::Cut,
        align: None,
        lead: 0.0,
        tail: 0.0,
        cut: String::new(),
        text: Some(title),
        why: format!("the hook: {}", hook.caption),
    });
    plan.shots.extend(body);
    if let Some(c) = end {
        let need = f64::from(end_beats) * beat;
        if let Some(from) = window_from(c, need) {
            plan.shots.push(PlannedShot {
                clip: c.id.clone(),
                file: String::new(),
                grade: None,
                from,
                beats: end_beats,
                camera: end_cam,
                look: c.looks.first().map(|l| l.id.clone()),
                to: None,
                enter: Enter::Dissolve,
                align: None,
                lead: 0.0,
                tail: 0.0,
                cut: String::new(),
                text: None,
                why: format!("the ending: {}", c.caption),
            });
        }
    }
    plan.notes = notes;
    plan
}

fn whips(shots: &[PlannedShot]) -> usize {
    shots.iter().filter(|s| s.enter == Enter::Whip).count()
}

// ---------------------------------------------------------------------------------------------
// the model's plan
// ---------------------------------------------------------------------------------------------

const DIRECTOR_SYSTEM: &str = "You are a film director cutting a short, catchy social video from the user's own clips: normal clips and 360° clips (a 360° clip films every direction at once; you choose where its virtual camera looks and how it moves). The edit's rhythm is set: a list of slots, each a number of beats of the music. Fill every slot with one clip and a camera. Open with a hook — a 360° clip's tiny planet intro (planetIn) when there is a good one — tell the story in filming order, alternate 360° and normal clips, and close the last slot (a 360° planetOut, or a calm wide shot). Keep it tasteful: each move at most twice, a barrel roll at most once. Use only the clip ids, look ids and camera names you are given. The title and any text are short English. Answer with JSON only.";

/// What a slot of the rhythm is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    Hook,
    Body,
    Ending,
}

/// One slot of the edit's rhythm: its length in beats and what it is for.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Slot {
    pub beats: u32,
    pub role: Role,
}

/// The rhythm of an edit `total` beats long at a beat of `beat` seconds: a hook long enough for
/// a tiny planet to open, body shots alternating about 2.4 s and 1.8 s, an ending; never more
/// slots than `available` moments.
pub fn rhythm(total: u32, beat: f64, available: usize) -> Vec<Slot> {
    let hook = beats_for(2.8, beat);
    let end = beats_for(2.2, beat);
    let pattern = [beats_for(2.4, beat), beats_for(1.8, beat)];
    let mut out = vec![Slot { beats: hook.min(total.max(1)), role: Role::Hook }];
    let mut left = total.saturating_sub(out.iter().map(|s| s.beats).sum::<u32>());
    if left == 0 {
        return out;
    }
    if left <= end || available < 2 {
        out.push(Slot { beats: left, role: Role::Ending });
        return out;
    }
    left -= end;
    let mut k = 0usize;
    let room = available.saturating_sub(2).max(1);
    while left > 0 && k < room.min(MAX_SHOTS) {
        let b = pattern.get(k % 2).copied().unwrap_or(4);
        if left < b + pattern[1].min(pattern[0]) / 2 {
            // the remainder joins this slot rather than making a stub
            out.push(Slot { beats: left, role: Role::Body });
            left = 0;
            break;
        }
        out.push(Slot { beats: b, role: Role::Body });
        left -= b;
        k += 1;
    }
    // more beats than moments: the body slots take them, up to their ceiling
    let mut i = 1usize;
    let mut guard = 0usize;
    while left > 0 && guard < 4096 && out.len() > 1 {
        if let Some(s) = out.get_mut(i)
            && s.beats < beats_for(MAX_SHOT_SECONDS, beat)
        {
            s.beats += 1;
            left -= 1;
        }
        i = if i + 1 >= out.len() { 1 } else { i + 1 };
        guard += 1;
    }
    out.push(Slot { beats: end + left, role: Role::Ending });
    out
}

/// What the model is told, as compact lines rather than one JSON object (a small model given a
/// large JSON input tends to echo it back instead of answering): the brief, the slots, the
/// cameras, one line a clip, the rules and the reply's shape.
pub fn director_prompt(cands: &[&Candidate], o: &Options, beat: f64, slots: &[Slot]) -> String {
    let one_line = |t: &str| t.replace(['\n', '\r', '|'], " ");
    let places = o.focus == Some(focus::Focus::Places);
    let mut out = String::new();
    out.push_str(&format!("Brief: \"{}\"\n", one_line(&o.brief)));
    if places {
        let most = ((slots.len() as f64 * focus::MAX_PEOPLE).floor() as usize).max(1);
        out.push_str(&format!(
            "This is a travel video: show the places — streets, buildings, views, the crowd. The person holding the camera is in many 360° shots: never choose a look at them; frame other people (a look marked \"a person\") in at most {most} of the slots, and for the rest choose looks that are places. Use moments from every part of the trip, in filming order.\n"
        ));
    }
    out.push_str(&format!("The video: {:.0} s, {}, {} slots, a beat is {:.2} s.\n", o.seconds, o.aspect, slots.len(), beat));
    let sl: Vec<String> = slots.iter().enumerate().map(|(k, s)| format!("{}:{}", k + 1, s.beats)).collect();
    out.push_str(&format!("Slots (number:beats; slot 1 is the hook, slot {} the ending): {}\n", slots.len(), sl.join(" ")));
    out.push_str("Cameras (name, minimum beats, what it does):\n");
    for c in Camera::ALL {
        out.push_str(&format!("- {} {} — {}\n", c.id(), beats_for(c.min_seconds(), beat), c.meaning()));
    }
    out.push_str(
        "Clips in filming order (id | kind | file | energy | rank: how good the moment is, 0..1 | notes | what it shows | looks | cameras it may use):\n",
    );
    for c in cands {
        let mut notes: Vec<&str> = Vec::new();
        if c.wide {
            notes.push("wide");
        }
        if c.indoor {
            notes.push("indoors");
        }
        if c.moving {
            notes.push("moving");
        }
        if c.scores.people >= 0.6 {
            notes.push("faces");
        }
        if c.scores.expression >= 0.5 {
            notes.push("smiles");
        }
        if c.scores.speech >= 0.5 {
            notes.push("speech");
        }
        let looks = if c.kind == Kind::Vr {
            let mark = |l: &Look| if places && focus::is_person(&l.what) { " (a person)" } else { "" };
            c.looks.iter().take(4).map(|l| format!("{} {}{}", l.id, one_line(&l.what), mark(l))).collect::<Vec<_>>().join("; ")
        } else {
            "-".into()
        };
        let cams: Vec<&str> = Camera::ALL.iter().filter(|m| m.for_kind(c.kind) && !(**m == Camera::Reveal && c.indoor)).map(|m| m.id()).collect();
        let says = c.said.as_deref().map(|s| format!(" (says: \"{}\")", chars(&one_line(s), 80))).unwrap_or_default();
        out.push_str(&format!(
            "{} | {} | {} | {:.1} | {:.1} | {} | {}{} | {} | {}\n",
            c.id,
            kind_word(c.kind),
            chars(&c.name, 32),
            c.energy,
            c.rank,
            if notes.is_empty() { "-".to_string() } else { notes.join(",") },
            one_line(&c.caption),
            says,
            looks,
            cams.join(",")
        ));
    }
    out.push_str("Rules: one shot per slot, in slot order; a clip id at most once; never the same file in two slots in a row; a slot's camera is one its clip may use, and its minimum beats fit the slot; planetIn only in slot 1, planetOut only in the last slot; a whipPan names `look` and `to`, two different looks of its clip; text is short English, usually only in slot 1.\n");
    out.push_str(&format!(
        "Reply with only this JSON, one entry per slot:\n{{\"title\": \"at most 5 English words\", \"mood\": \"{}\", \"shots\": [{{\"slot\": 1, \"clip\": \"c1\", \"camera\": \"planetIn\", \"look\": \"c1a\", \"to\": null, \"text\": \"the title\"}}]}}\n",
        MOODS.join("|")
    ));
    out
}

/// Ask the model to fill the slots: its raw answer, and how many times it was asked (a reply
/// that is not JSON — cut short, or malformed — is asked for once more).
pub async fn model_plan(model: &dyn ChatModel, cands: &[&Candidate], o: &Options, beat: f64, slots: &[Slot]) -> Result<(Value, usize), AutoCutError> {
    let prompt = director_prompt(cands, o, beat, slots);
    match model.complete_json(DIRECTOR_SYSTEM, &prompt, &[]).await {
        Ok(v) => Ok((v, 1)),
        Err(first) => match model.complete_json(DIRECTOR_SYSTEM, &prompt, &[]).await {
            Ok(v) => Ok((v, 2)),
            Err(_) => Err(first),
        },
    }
}

/// A camera for candidate `c` in a shot of `beats`: `asked` when it fits — the clip's kind, the
/// slot's length, a planet only at the ends, no sky reveal indoors, one barrel roll, a whip with
/// two looks — else the rules' choice that fits, else a push in. The notes say each change.
#[allow(clippy::too_many_arguments)]
fn fit_camera(
    asked: Option<Camera>,
    c: &Candidate,
    beats: u32,
    beat: f64,
    slot: usize,
    last: bool,
    used: &HashMap<Camera, usize>,
    frame_aspect: f64,
    notes: &mut Vec<String>,
) -> Camera {
    let secs = f64::from(beats) * beat;
    let ok = |m: Camera| -> Result<(), String> {
        if !m.for_kind(c.kind) {
            return Err(format!("{} is not for a {} clip", m.id(), kind_word(c.kind)));
        }
        if beats_for(m.min_seconds(), beat) > beats {
            return Err(format!("{} needs more than the slot's {beats} beats", m.id()));
        }
        if m == Camera::PlanetIn && slot != 0 {
            return Err("a tiny planet opens the edit only".into());
        }
        if m == Camera::PlanetOut && !last {
            return Err("a planet outro closes the edit only".into());
        }
        if m == Camera::Reveal && c.indoor {
            return Err("indoors a reveal from the sky shows the ceiling".into());
        }
        if m == Camera::BarrelRoll && used.get(&Camera::BarrelRoll).copied().unwrap_or(0) >= 1 {
            return Err("one barrel roll is enough".into());
        }
        if m == Camera::Reveal && used.get(&Camera::Reveal).copied().unwrap_or(0) >= MAX_REVEALS {
            return Err(format!("{MAX_REVEALS} reveals are enough (each starts on the sky)"));
        }
        if m == Camera::WhipPan && c.second_look().is_none() {
            return Err("a whip pan needs two things to turn between".into());
        }
        if m == Camera::Hyperlapse && window_from(c, secs * m.speed()).is_none() {
            return Err("not enough footage for a hyperlapse".into());
        }
        Ok(())
    };
    if let Some(m) = asked {
        match ok(m) {
            Ok(()) => return m,
            Err(why) => {
                let r = [rules_camera(c, secs, used, None, frame_aspect), Camera::PushIn, Camera::Hold]
                    .into_iter()
                    .find(|r| ok(*r).is_ok())
                    .unwrap_or(Camera::Hold);
                notes.push(format!("slot {} ({}): {why}: {}", slot + 1, c.id, r.id()));
                return r;
            }
        }
    }
    [rules_camera(c, secs, used, None, frame_aspect), Camera::PushIn, Camera::Hold].into_iter().find(|r| ok(*r).is_ok()).unwrap_or(Camera::Hold)
}

/// The model's answer made into a plan on `slots`: each slot's pick (by its `slot`, else its
/// place in the list) kept when its clip exists, has the footage and is not the same moment
/// again or a third shot of one file in a row — its camera fitted ([`fit_camera`]) and its looks
/// checked — else the slot is filled by rules (the next unused clip in filming order, the kinds
/// taking turns; a third moment of one file in a row when nothing else is left; with no moment
/// left at all, the shot before plays on through it). The text must be English. The notes say
/// every repair. `Err` when the answer has no shots at all.
pub fn validate(v: &Value, cands: &[Candidate], o: &Options, beat: f64, bpm: f64, slots: &[Slot]) -> Result<Plan, String> {
    let v = plan_body(v);
    let list = shots_in(v).ok_or_else(|| {
        let keys: Vec<&str> = v.as_object().map(|o| o.keys().map(String::as_str).take(8).collect()).unwrap_or_default();
        format!("the plan has no shots (it answered with {})", if keys.is_empty() { "no object".to_string() } else { keys.join(", ") })
    })?;
    let frame_aspect = {
        let (w, h) = o.frame();
        f64::from(w) / f64::from(h.max(1))
    };
    let mut notes: Vec<String> = Vec::new();
    let mut shots: Vec<PlannedShot> = Vec::new();
    let mut used_ids: BTreeSet<String> = BTreeSet::new();
    let mut cams: HashMap<Camera, usize> = HashMap::new();
    let mut from_model = 0usize;
    let n = slots.len();
    let item_of = |id: &str| cands.iter().find(|c| c.id == id).map(|c| c.item);
    for (k, slot) in slots.iter().enumerate() {
        let last = k + 1 == n;
        let asked = list.iter().find(|s| s["slot"].as_u64() == Some(k as u64 + 1)).or_else(|| list.get(k)).filter(|s| s.is_object());
        let third = |item: u64, shots: &[PlannedShot]| {
            let m = shots.len();
            m >= 2 && shots.get(m - 1).and_then(|s| item_of(&s.clip)) == Some(item) && shots.get(m - 2).and_then(|s| item_of(&s.clip)) == Some(item)
        };
        let need = |m: Camera| f64::from(slot.beats) * beat * m.speed();
        // the model's pick
        let mut pick: Option<(&Candidate, Option<Camera>, Value)> = None;
        if let Some(s) = asked {
            match s["clip"].as_str().and_then(|id| cands.iter().find(|c| c.id == id)) {
                None => notes.push(format!("slot {}: no clip `{}`: filled by rules", k + 1, chars(s["clip"].as_str().unwrap_or("?"), 20))),
                Some(c) if used_ids.contains(&c.id) => notes.push(format!("slot {} ({}): the same moment again: filled by rules", k + 1, c.id)),
                Some(c) if third(c.item, &shots) => notes.push(format!("slot {} ({}): a third shot of one file in a row: filled by rules", k + 1, c.id)),
                Some(c) if window_from(c, need(Camera::Hold)).is_none() => {
                    notes.push(format!("slot {} ({}): the clip is too short for the slot: filled by rules", k + 1, c.id))
                }
                Some(c) => pick = Some((c, s["camera"].as_str().and_then(Camera::from_id), s.clone())),
            }
        }
        // else rules: the next unused clip in filming order, the kinds taking turns
        if pick.is_none() {
            let after = shots.last().and_then(|s| cands.iter().find(|c| c.id == s.clip)).map_or((0, 0.0), |c| (c.order, c.start));
            let last_kind = shots.last().and_then(|s| cands.iter().find(|c| c.id == s.clip)).map(|c| c.kind);
            let fits = |c: &&Candidate| !used_ids.contains(&c.id) && !third(c.item, &shots) && window_from(c, need(Camera::Hold)).is_some();
            let mut later: Vec<&Candidate> = cands.iter().filter(fits).collect();
            later.sort_by(|a, b| {
                let ka = ((a.order, a.start) < after, Some(a.kind) == last_kind, a.order, a.start.to_bits());
                let kb = ((b.order, b.start) < after, Some(b.kind) == last_kind, b.order, b.start.to_bits());
                ka.cmp(&kb)
            });
            if slot.role == Role::Hook {
                later.sort_by(|a, b| (b.kind == Kind::Vr).cmp(&(a.kind == Kind::Vr)).then((b.hook + b.energy).total_cmp(&(a.hook + a.energy))));
            }
            pick = later.first().map(|c| (*c, None, Value::Null));
            if pick.is_none() {
                // only one file's moments left: a third of it in a row rather than a short edit
                let mut rest: Vec<&Candidate> = cands.iter().filter(|c| !used_ids.contains(&c.id) && window_from(c, need(Camera::Hold)).is_some()).collect();
                rest.sort_by_key(|c| (c.order, c.start.to_bits()));
                if let Some(c) = rest.first() {
                    notes.push(format!("slot {} ({}): only one file's moments left: a third of it in a row", k + 1, c.id));
                    pick = Some((*c, None, Value::Null));
                }
            }
        }
        let Some((c, asked_cam, s)) = pick else {
            // nothing new to show: the shot before plays on through the slot's beats when its
            // footage lasts, so the edit keeps its length
            let longer = shots.last_mut().and_then(|p| {
                let c = cands.iter().find(|c| c.id == p.clip)?;
                let beats = p.beats.saturating_add(slot.beats);
                let from = window_from(c, f64::from(beats) * beat * p.camera.speed())?;
                p.beats = beats;
                p.from = round2(from);
                Some(())
            });
            notes.push(match longer {
                Some(()) => format!("slot {}: nothing left to fill it: the shot before plays on", k + 1),
                None => format!("slot {}: nothing left to fill it", k + 1),
            });
            continue;
        };
        if !s.is_null() {
            from_model += 1;
        }
        let cam = fit_camera(asked_cam, c, slot.beats, beat, k, last, &cams, frame_aspect, &mut notes);
        let Some(from) = window_from(c, need(cam)) else { continue };
        let mut look = s["look"].as_str().filter(|l| c.look(l).is_some()).map(str::to_string).or_else(|| c.looks.first().map(|l| l.id.clone()));
        let mut to = None;
        if cam == Camera::WhipPan {
            let far = |a: Option<&str>, b: Option<&str>| match (a.and_then(|x| c.look(x)), b.and_then(|x| c.look(x))) {
                (Some(x), Some(y)) => angle_between(x.yaw, y.yaw) >= LOOKS_APART,
                _ => false,
            };
            to = s["to"].as_str().filter(|t| far(look.as_deref(), Some(t))).map(str::to_string);
            if to.is_none() {
                look = c.looks.first().map(|l| l.id.clone());
                to = c.second_look().map(|l| l.id.clone());
            }
        }
        let text = s["text"].as_str().map(str::trim).filter(|t| !t.is_empty()).map(|t| chars(t, MAX_TITLE_CHARS));
        let text = match text {
            Some(t) if !english(&t) => {
                notes.push(format!("slot {} ({}): its text is not English: left out", k + 1, c.id));
                None
            }
            t => t,
        };
        *cams.entry(cam).or_insert(0) += 1;
        used_ids.insert(c.id.clone());
        // (how each cut is made is picked once the plan is set: [`cuts::joins`])
        let enter = if last && n > 2 { Enter::Dissolve } else { Enter::Cut };
        let why = s["why"].as_str().map(|w| chars(w, 160)).unwrap_or_else(|| if s.is_null() { format!("{} (rules)", c.caption) } else { String::new() });
        shots.push(PlannedShot {
            clip: c.id.clone(),
            file: String::new(),
            grade: None,
            from: round2(from),
            beats: slot.beats,
            camera: cam,
            look: if c.kind == Kind::Vr { look } else { None },
            to,
            enter,
            align: None,
            lead: 0.0,
            tail: 0.0,
            cut: String::new(),
            text,
            why,
        });
    }
    if from_model == 0 {
        return Err("none of the plan's shots could be used".into());
    }
    let title = v["title"].as_str().map(str::trim).filter(|t| !t.is_empty() && english(t)).map(|t| title_case(&chars(t, MAX_TITLE_CHARS)));
    let title = match (&o.title, title) {
        (Some(t), _) => t.clone(),
        (None, Some(t)) => t,
        (None, None) => {
            notes.push("the plan's title was missing or not English: one from the brief".into());
            title_from_brief(&o.brief).unwrap_or_else(|| "Highlights".into())
        }
    };
    if let Some(first) = shots.first_mut() {
        // the hook's text is the title, set as a title
        first.text = Some(first.text.as_deref().map_or_else(|| title.clone(), title_case));
    }
    let mood = o
        .mood
        .clone()
        .or_else(|| v["mood"].as_str().map(str::to_ascii_lowercase).filter(|m| MOODS.contains(&m.as_str())))
        .unwrap_or_else(|| mood_from_brief(&o.brief).to_string());
    Ok(Plan { title, mood, bpm, shots, by: String::new(), notes, style: None })
}

/// The plan in a model's answer: the answer itself, or what it wrapped it in (`answer`, `plan`,
/// `edit`: a model may echo the example's key).
fn plan_body(v: &Value) -> &Value {
    for k in ["answer", "plan", "edit", "result"] {
        if v.get(k).is_some_and(|w| shots_in(w).is_some()) {
            return &v[k];
        }
    }
    v
}

/// A plan's shot list, under the names models give it (or the answer being the list itself).
fn shots_in(v: &Value) -> Option<&Vec<Value>> {
    if let Some(a) = v.as_array().filter(|a| !a.is_empty() && a.iter().all(Value::is_object)) {
        return Some(a);
    }
    ["shots", "shotList", "shot_list", "timeline", "edit"].iter().find_map(|k| v.get(*k).and_then(Value::as_array).filter(|a| !a.is_empty()))
}

fn kind_word(k: Kind) -> &'static str {
    match k {
        Kind::Flat => "normal",
        Kind::Vr => "360°",
        Kind::Photo => "photo",
    }
}

// ---------------------------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------------------------

/// What a step's clip parameter is filled with when it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum On {
    /// Nothing: the parameters are complete.
    Nothing,
    /// The step places its shot's clip (`timeline.place`).
    Place,
    /// `"clip"`: its shot's picture clip.
    Clip,
    /// `"clips": [..]`: its shot's picture clip.
    Clips,
    /// `"clip"`: the last text clip made.
    Text,
}

/// One engine command of the edit.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Step {
    pub id: String,
    pub params: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shot: Option<usize>,
    pub on: On,
}

fn step(id: &str, params: Value, shot: Option<usize>, on: On) -> Step {
    Step { id: id.to_string(), params, shot, on }
}

/// Where each shot sits on the timeline, seconds on frames: back to back from 0, each boundary
/// on its beat.
pub fn windows(plan: &Plan, beats: &Beats) -> Vec<(f64, f64)> {
    let total: u32 = plan.beats();
    let period = beats.period();
    let list = beats.list(0.0, f64::from(total + 4) * period + 1.0);
    let at = |k: u32| -> f64 {
        let k = k as usize;
        match list.get(k) {
            Some(t) => *t,
            None => list.last().copied().unwrap_or(0.0) + (k.saturating_sub(list.len().saturating_sub(1))) as f64 * period,
        }
    };
    let first = list.first().copied().unwrap_or(0.0);
    let mut out = Vec::with_capacity(plan.shots.len());
    let mut cum = 0u32;
    for s in &plan.shots {
        let a = if cum == 0 { 0.0 } else { at(cum) - first };
        cum += s.beats;
        let b = at(cum) - first;
        out.push((on_frame(a, FPS), on_frame(b.max(a + 1.0 / FPS), FPS)));
    }
    out
}

/// The title's place and size on a `frame`: centred in the upper part of the safe zone.
pub fn title_layout(o: &Options, frame: (f64, f64), text: &str) -> ([f64; 2], f64, f64) {
    let (w, h) = frame;
    let (x0, y0, x1, y1) = if o.vertical() {
        let b = crate::trailer::boxes(o.platform, frame);
        (b.title[0], b.title[1], b.title[2], b.title[3])
    } else {
        (0.1 * w, 0.12 * h, 0.9 * w, 0.4 * h)
    };
    let room = (x1 - x0).max(1.0);
    let u = h.min(w * 16.0 / 9.0) / 1080.0;
    // inside the room with a margin (bold capitals run wider than an average letter): a long
    // title is set smaller
    let n = text.chars().count().max(1) as f64;
    let size = (110.0 * u).min(0.9 * room / (0.62 * n)).max(40.0 * u.min(1.0));
    ([(x0 + x1) / 2.0, y0 + 0.55 * (y1 - y0)], size, room)
}

/// The commands of a plan on `windows` (timeline seconds of each shot), for a `frame`: each
/// shot placed on V1 with its sound on A1 (a J or L cut moving it across the cut), its camera
/// set and moved, its way in; the title and other text on V2; the style's sounds on A4. `beat`
/// the music's beat, seconds.
pub fn compile(plan: &Plan, cands: &[Candidate], wins: &[(f64, f64)], o: &Options, frame: (f64, f64), beat: f64) -> Vec<Step> {
    let mut out: Vec<Step> = Vec::new();
    let frame_aspect = if frame.1 > 0.0 { frame.0 / frame.1 } else { 16.0 / 9.0 };
    for (k, (s, w)) in plan.shots.iter().zip(wins).enumerate() {
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        let (t0, t1) = *w;
        let len = (t1 - t0).max(1.0 / FPS);
        let speed = s.camera.speed();
        let mut place = json!({"item": c.item, "track": "V1", "time": ticks(t0), "duration": ticks(len * speed)});
        if c.kind == Kind::Photo {
            place["audio"] = json!(false);
        } else {
            place["sourceIn"] = json!(ticks(s.from));
            place["audioTrack"] = json!("A1");
        }
        out.push(step("timeline.place", place, Some(k), On::Place));
        let me = Some(k);
        // the sound across the cut: a J cut leads into this shot, an L cut runs the last one on
        if s.lead > 0.0 {
            out.push(step(SPLIT_CMD, json!({"lead": round2(s.lead)}), me, On::Clip));
        }
        if let Some(prev) = k.checked_sub(1).and_then(|j| plan.shots.get(j)).filter(|p| p.tail > 0.0) {
            out.push(step(SPLIT_CMD, json!({"tail": round2(prev.tail)}), Some(k - 1), On::Clip));
        }
        let look = s.look.as_deref().and_then(|l| c.look(l));
        match c.kind {
            Kind::Vr => {
                if s.camera == Camera::Hyperlapse {
                    out.push(step("clip.hyperlapse360", json!({"speed": speed * 100.0, "mode": "strong", "wait": true}), me, On::Clips));
                }
                // framed for this frame's shape: the subject's whole box in view (its centre
                // mid-frame, the field of view sized to it), else the shape's natural view
                if let Some(l) = look {
                    let (fov, distance) = filmcraft_engine::project::vr::subject_framing(l.size, frame_aspect);
                    let pitch = filmcraft_engine::project::vr::subject_aim(l.pitch, l.size);
                    // a match cut puts the subject where the last shot's was
                    let yaw = l.yaw - s.align.map_or(0.0, |a| (a - 0.5) * fov);
                    out.push(step(
                        "clip.reframe360",
                        json!({"yaw": yaw.round(), "pitch": pitch.round(), "fov": fov.round(), "distance": round2(distance)}),
                        me,
                        On::Clips,
                    ));
                }
                // every 360° shot steadied from its pictures: a walk sways and bounces though the
                // analysis seldom calls it handheld, and the camera's motion sensor takes out only
                // part of the sway (a planet is all sky and ground; a hyperlapse steadies itself)
                if !matches!(s.camera, Camera::Hyperlapse | Camera::PlanetIn | Camera::PlanetOut) {
                    out.push(step("clip.stabilize360", json!({"mode": "smooth", "wait": true}), me, On::Clips));
                }
                // where the shot's own clip time is: the hyperlapse plays its media faster
                let at = |secs: f64| round2(secs.max(0.0));
                // a named subject is tracked under a camera that does not turn by itself
                if look.is_some_and(|l| l.named) && matches!(s.camera, Camera::Hold | Camera::PushIn | Camera::SpeedRamp) {
                    out.push(step(
                        "clip.follow360",
                        json!({"target": "subject", "at": at((c.key - s.from) / speed), "smooth": 0.7, "wait": true}),
                        me,
                        On::Clip,
                    ));
                }
                match s.camera {
                    Camera::Follow => {
                        out.push(step(
                            "clip.follow360",
                            json!({"target": "subject", "at": at((c.key - s.from) / speed), "smooth": 0.6, "wait": true}),
                            me,
                            On::Clip,
                        ));
                    }
                    Camera::WhipPan => {
                        let to = s.to.as_deref().and_then(|l| c.look(l));
                        let mut p = json!({"move": "whipPan", "at": at(len / 2.0 - 0.2), "seconds": 0.4});
                        if let Some(t) = to {
                            p["yaw"] = json!(t.yaw.round());
                            p["pitch"] = json!(t.pitch.round());
                        }
                        out.push(step("clip.vrMove", p, me, On::Clips));
                    }
                    Camera::Hold | Camera::SpeedRamp | Camera::Hyperlapse => {}
                    Camera::PlanetIn => {
                        out.push(step("clip.vrMove", json!({"move": "planetIn", "at": 0.0, "seconds": round2((len * 0.7).clamp(1.6, 3.0))}), me, On::Clips))
                    }
                    Camera::PlanetOut => {
                        out.push(step("clip.vrMove", json!({"move": "planetOut", "seconds": round2((len * 0.7).clamp(1.4, 2.6))}), me, On::Clips))
                    }
                    Camera::Orbit => out.push(step("clip.vrMove", json!({"move": "orbit", "degrees": if k % 2 == 0 { 360.0 } else { -360.0 }}), me, On::Clips)),
                    Camera::BarrelRoll => {
                        out.push(step("clip.vrMove", json!({"move": "barrelRoll", "at": at(len / 2.0 - 0.45), "seconds": 0.9}), me, On::Clips))
                    }
                    Camera::Reveal => out.push(step("clip.vrMove", json!({"move": "reveal", "seconds": round2((len * 0.6).clamp(1.2, 2.4))}), me, On::Clips)),
                    Camera::Drift => out.push(step("clip.vrMove", json!({"move": "drift", "degrees": if k % 2 == 0 { 25.0 } else { -25.0 }}), me, On::Clips)),
                    Camera::PushIn => out.push(step("clip.vrMove", json!({"move": "pushIn", "amount": 0.12}), me, On::Clips)),
                }
                if s.camera == Camera::SpeedRamp {
                    out.push(step("clip.speedRamp", json!({"preset": "smoothIn", "fit": "length", "ripple": false, "opticalFlow": false}), me, On::Clips));
                }
                // a whip out of this shot into the next 360° one
                if plan.shots.get(k + 1).is_some_and(|nx| nx.enter == Enter::Whip) && !matches!(s.camera, Camera::PlanetOut) && len > 2.0 * WHIP_HALF {
                    out.push(step("clip.vrMove", json!({"move": "whipPan", "at": at(len - WHIP_HALF), "seconds": WHIP_HALF, "degrees": 90.0}), me, On::Clips));
                }
                if s.enter == Enter::Whip
                    && let Some(l) = look
                {
                    out.push(step(
                        "clip.vrMove",
                        json!({"move": "whipPan", "at": 0.0, "seconds": WHIP_HALF, "fromYaw": (l.yaw - 90.0).round(), "yaw": l.yaw.round(), "pitch": l.pitch.round()}),
                        me,
                        On::Clips,
                    ));
                }
            }
            Kind::Flat | Kind::Photo => {
                let subject = c.subject.map(|b| json!(b));
                // a picture of another shape than the frame is cropped: its subject tracked
                // through the shot (Auto Reframe, keeping a push in's scale) so it stays in view
                let reshaped = c.kind == Kind::Flat && (c.aspect - frame_aspect).abs() / frame_aspect > 0.05;
                let track = || {
                    let mut p = json!({"keepScale": true, "motion": "smooth", "fallback": "center", "analysisFps": 8, "wait": true});
                    if let Some(b) = c.subject {
                        p["subject"] = json!({"x": b[0], "y": b[1], "w": b[2], "h": b[3], "seconds": round2(c.key)});
                    }
                    step("clip.autoReframe", p, me, On::Clips)
                };
                match s.camera {
                    Camera::Follow => {
                        out.push(step("clip.fillFrame", json!({}), me, On::Clips));
                        out.push(track());
                    }
                    Camera::Hold => {
                        out.push(step("clip.fillFrame", json!({}), me, On::Clips));
                        if reshaped {
                            out.push(track());
                        }
                    }
                    _ => {
                        let mut p = json!({"move": "in", "fit": "fill"});
                        if let Some(b) = subject {
                            p["subject"] = b;
                        }
                        out.push(step("clip.kenBurns", p, me, On::Clips));
                        if reshaped {
                            out.push(track());
                        }
                    }
                }
                if s.camera == Camera::SpeedRamp {
                    out.push(step("clip.speedRamp", json!({"preset": "smoothIn", "fit": "length", "ripple": false, "opticalFlow": false}), me, On::Clips));
                }
            }
        }
        // the way in: a transition on the cut (its middle on the beat)
        let frames = |beats: f64| ((beat * beats * FPS).round() as i64).clamp(6, 36);
        let transition = match s.enter {
            Enter::Dissolve => Some(json!({"effect": "cross_dissolve", "frames": frames(o.style.dissolve_beats())})),
            Enter::Swipe => {
                // the new picture comes from the side the camera turns to
                let prev = k.checked_sub(1).and_then(|j| plan.shots.get(j)).and_then(|p| cands.iter().find(|x| x.id == p.clip));
                let faster = [prev, Some(c)]
                    .into_iter()
                    .flatten()
                    .filter(|x| matches!(x.pan, CameraMove::PanLeft | CameraMove::PanRight))
                    .max_by(|a, b| a.scores.motion.total_cmp(&b.scores.motion));
                let from_east = faster.is_none_or(|x| x.pan == CameraMove::PanRight);
                Some(json!({"effect": "cam_whip_pan", "frames": frames(0.5), "params": {"direction": if from_east { 1 } else { 3 }}}))
            }
            Enter::Punch => Some(json!({"effect": "cam_punch_in", "frames": frames(0.5)})),
            Enter::LightLeak => Some(json!({"effect": "lf_light_leak", "frames": frames(1.0)})),
            Enter::DipToBlack => Some(json!({"effect": "dip_to_black", "frames": frames(1.5)})),
            Enter::Cut | Enter::Match | Enter::Whip => None,
        };
        if let Some(mut t) = transition.filter(|_| k > 0) {
            t["edge"] = json!("in");
            out.push(step("sequence.applyVideoTransition", t, me, On::Clip));
        }
        // the style's sound under each whip and roll, quiet unless it asks for more
        let sound = |kind: &str, at: f64, secs: f64| json!({"kind": kind, "duration": secs, "time": ticks((at - secs / 2.0).max(0.0)), "track": "A4", "intensity": round2(o.style.sound_level), "seed": o.seed + k as u64});
        if let Some(kind) = o.style.whip_sound() {
            if matches!(s.enter, Enter::Whip | Enter::Swipe) {
                out.push(step("audio.generate", sound(kind, t0, 0.5), None, On::Nothing));
            }
            if matches!(s.camera, Camera::BarrelRoll | Camera::WhipPan) {
                out.push(step("audio.generate", sound(kind, t0 + len / 2.0, 0.5), None, On::Nothing));
            }
        }
        if k + 1 == plan.shots.len()
            && k > 0
            && let Some(kind) = o.style.ending_sound()
        {
            // rising into the ending, done as it begins
            out.push(step("audio.generate", sound(kind, (t0 - 0.6).max(0.0), 1.6), None, On::Nothing));
        }
        // the text over the shot: in from 0.3 s, gone 0.3 s before its end
        if let Some(text) = &s.text {
            let (pos, size, _) = title_layout(o, frame, text);
            let (a, b) = (on_frame(t0 + if k == 0 { 0.4 } else { 0.2 }, FPS), on_frame((t1 - 0.3).max(t0 + 1.0), FPS));
            out.push(step(
                "graphics.newText",
                json!({"text": text, "position": pos, "size": size.round(), "seconds": round2(b - a), "time": ticks(a), "track": "V2"}),
                None,
                On::Nothing,
            ));
            out.push(step(
                "graphics.set",
                json!({"props": {"font_style": "Bold", "fill_color": "#ffffff", "align": "center", "stroke": false, "shadow": true, "shadow_opacity": 70, "shadow_distance": 2, "shadow_blur": 18}}),
                None,
                On::Text,
            ));
            for (t, v) in [(0.0, 0.0), (0.35, 100.0), ((b - a - 0.35).max(0.4), 100.0), ((b - a).max(0.5), 0.0)] {
                out.push(step(
                    "effects.setParam",
                    json!({"effect": "opacity", "param": "opacity", "value": v, "time": ticks(t), "keyframe": true}),
                    None,
                    On::Text,
                ));
            }
            if k == 0
                && let Some(kind) = o.style.title_sound()
            {
                out.push(step(
                    "audio.generate",
                    json!({"kind": kind, "duration": 1.2, "time": ticks(a), "track": "A4", "intensity": round2(o.style.sound_level * 0.75), "seed": o.seed}),
                    None,
                    On::Nothing,
                ));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// the job
// ---------------------------------------------------------------------------------------------

/// What the director made, and how.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub brief: String,
    pub aspect: String,
    pub seconds: f64,
    pub media: usize,
    pub vr_media: usize,
    /// 360° recordings levelled by their camera's motion sensor, and the largest tilt taken out
    /// (degrees).
    pub levelled: usize,
    pub max_tilt: f64,
    pub shots_indexed: usize,
    /// The colour pass: each shot's correction and any backing off, and the before/after sheet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<Value>,
    /// The options it was made with (a revision starts from them), and, revised, from which
    /// report and what changed.
    pub options: Value,
    /// The taste profile it was made in (`autocut.taste`), when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taste: Option<Value>,
    /// The cutting style: which, whether it was asked for or the director chose it, and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<Value>,
    /// What the edit is about (the places or the people), and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_from: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub changes: Vec<String>,
    /// Moments measured on their own frames, and every moment's verdict: its scores, kept or
    /// dropped, and why.
    pub measured: usize,
    pub moments: Vec<rank::Verdict>,
    /// The shots the director chose from: what each shows, and a 360° shot's looks.
    pub candidates: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<Plan>,
    pub commands: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
    /// A dry run's: what the directing model was asked (system and user).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    pub director: String,
    pub models: Value,
    pub fallbacks: Vec<String>,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub music: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mix: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_zones: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qa_summary: Option<String>,
}

/// The media: `items`, `paths`' files imported, `insta360` recordings imported (downloaded from
/// the cloud when not on disk), else the project's; in filming order.
async fn gather(
    ex: &dyn Executor,
    p: &Value,
    notes: &mut Vec<String>,
    lines: &mut dyn FnMut(&str),
) -> Result<(Vec<MediaItem>, HashMap<u64, i64>), AutoCutError> {
    let mut want: Vec<u64> = p["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).take(crate::quick::MAX_MEDIA).collect()).unwrap_or_default();
    let paths: Vec<String> =
        p["paths"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).take(crate::quick::MAX_MEDIA).collect()).unwrap_or_default();
    if !paths.is_empty() {
        let files = crate::quick::expand_paths(&paths)?;
        lines(&format!("importing {} file(s)", files.len()));
        let r = run(ex, "file.import", json!({"paths": files, "reuse": true})).await?;
        want.extend(r["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>()).unwrap_or_default());
        for e in r["errors"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(10) {
            notes.push(format!("import: {}", chars(&e.to_string(), 200)));
        }
    }
    let names: Vec<String> = p["insta360"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).take(100).collect()).unwrap_or_default();
    if !names.is_empty() {
        lines(&format!("bringing in {} Insta360 recording(s)", names.len()));
        let r = run(ex, "insta360.import", json!({"names": names})).await?;
        let mut got = |rows: &Value| {
            for row in rows.as_array().map(Vec::as_slice).unwrap_or_default() {
                match row["item"].as_u64() {
                    Some(i) if row["imported"] == true => want.push(i),
                    _ => notes.push(format!("insta360: {}: {}", row["stem"].as_str().unwrap_or("?"), row["why"].as_str().unwrap_or("not imported"))),
                }
            }
        };
        got(&r["recordings"]);
        let missing: Vec<String> = r["missing"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
        if !missing.is_empty() {
            lines(&format!("downloading {} recording(s) from the Insta360 cloud (Insta360 Studio)", missing.len()));
            match run(ex, "insta360.download", json!({"names": missing, "wait": true})).await {
                Ok(d) => got(&d["recordings"]),
                Err(e) => notes.push(format!("insta360: the cloud download did not run: {e}")),
            }
        }
    }
    let tree = run(ex, "project.inspect", json!({})).await?;
    let all = crate::quick::project_media(&tree);
    let mut media: Vec<MediaItem> = if want.is_empty() {
        all
    } else {
        let mut seen = BTreeSet::new();
        want.iter().filter(|i| seen.insert(**i)).filter_map(|i| all.iter().find(|m| m.item == *i).cloned()).collect()
    };
    if media.is_empty() {
        return Err(invalid("no video or photo to direct (give items, files, a folder or Insta360 recordings)"));
    }
    let mut when: HashMap<u64, i64> = HashMap::new();
    if let Ok(d) = run(ex, "media.captureDate", json!({"items": media.iter().map(|m| m.item).collect::<Vec<_>>()})).await {
        when = d["items"].as_array().map(|a| a.iter().filter_map(|x| Some((x["item"].as_u64()?, x["unix"].as_i64()?))).collect()).unwrap_or_default();
        let mut keyed: Vec<(Option<i64>, usize, u64)> = media.iter().enumerate().map(|(k, m)| (when.get(&m.item).copied(), k, m.item)).collect();
        keyed.sort_by(|a, b| match (a.0, b.0) {
            (Some(x), Some(y)) => x.cmp(&y).then(a.1.cmp(&b.1)),
            _ => a.1.cmp(&b.1),
        });
        for (rank, (_, _, item)) in keyed.iter().enumerate() {
            if let Some(m) = media.iter_mut().find(|m| m.item == *item) {
                m.order = rank;
            }
        }
    }
    Ok((media, when))
}

/// The scene index of `media` (fast, then full detail for the items of the most promising shots
/// when that describer is installed), as shots; fallbacks and the models noted.
async fn index(
    ex: &dyn Executor,
    media: &[MediaItem],
    o: &Options,
    report: &mut Report,
    notes: &mut Vec<String>,
    lines: &mut dyn FnMut(&str),
) -> Vec<ShotInfo> {
    let items: Vec<u64> = media.iter().map(|m| m.item).collect();
    lines(&format!("indexing the scenes of {} item(s)", items.len()));
    let status = run(ex, "scenes.status", json!({})).await.ok();
    let runnable = status.as_ref().is_some_and(|s| s["available"] == true);
    let missing = |set: &str| if runnable { status.as_ref().and_then(|s| s[set]["missingBytes"].as_f64()).unwrap_or(f64::MAX) } else { f64::MAX };
    if missing("fast") > 0.0 {
        report.fallbacks.push("scene models: not installed or not available here: shots are cut and ranked by the picture alone, without descriptions or subjects (add --download)".into());
    }
    if let Err(e) = run(ex, "media.analyzeScenes", json!({"items": items, "detail": "fast", "wait": true})).await {
        report.fallbacks.push(format!("scene index: {e}; each clip counts as one shot"));
    }
    let mut indexes: HashMap<u64, Value> = HashMap::new();
    for m in media {
        // (with the shots' image embeddings: near-duplicates are found by them)
        if let Ok(v) = run(ex, "media.scenes", json!({"item": m.item, "embeddings": true})).await
            && v["analyzed"] == true
        {
            indexes.insert(m.item, v);
        }
    }
    let shots_of =
        |idx: &HashMap<u64, Value>| -> Vec<ShotInfo> { media.iter().flat_map(|m| crate::quick::parse_shots(m, idx.get(&m.item))).take(4000).collect() };
    let mut shots = shots_of(&indexes);
    let described = shots.iter().any(|s| s.description().is_some());
    if o.full_detail > 0 && described && missing("full") <= 0.0 {
        let mut ranked: Vec<(f64, u64)> = shots
            .iter()
            .map(|s| {
                (
                    f64::from(
                        [ShotUse::Hook, ShotUse::Reveal, ShotUse::BRoll].iter().map(|u| filmcraft_scenes::best::score(&s.shot, *u).0).fold(0.0f32, f32::max),
                    ),
                    s.item,
                )
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut chosen: Vec<u64> = Vec::new();
        let mut count = 0usize;
        for (_, item) in ranked {
            let n = shots.iter().filter(|s| s.item == item).count();
            if chosen.contains(&item) || count + n > o.full_detail {
                continue;
            }
            chosen.push(item);
            count += n;
        }
        if !chosen.is_empty() {
            lines(&format!("describing {count} promising shot(s) in full detail"));
            match run(ex, "media.analyzeScenes", json!({"items": chosen, "detail": "full", "wait": true})).await {
                Ok(_) => {
                    for item in &chosen {
                        if let Ok(v) = run(ex, "media.scenes", json!({"item": item, "embeddings": true})).await
                            && v["analyzed"] == true
                        {
                            indexes.insert(*item, v);
                        }
                    }
                    shots = shots_of(&indexes);
                }
                Err(e) => notes.push(format!("full detail: {e}")),
            }
        }
    }
    let mut describer: BTreeSet<String> = BTreeSet::new();
    for v in indexes.values() {
        if let Some(x) = v["describer"].as_str().filter(|x| !x.is_empty()) {
            describer.insert(chars(x, 80));
        }
    }
    report.models["describer"] = json!(describer);
    shots
}

/// The speech of the movies (`media.understandSpeech`), marked on the shots.
async fn speech_of(ex: &dyn Executor, media: &[MediaItem], shots: &mut [ShotInfo], report: &mut Report, lines: &mut dyn FnMut(&str)) -> Vec<ItemSpeech> {
    let movies: Vec<u64> = media.iter().filter(|m| !m.still).map(|m| m.item).collect();
    if movies.is_empty() {
        return Vec::new();
    }
    lines(&format!("listening to {} clip(s)", movies.len()));
    let mut speech = Vec::new();
    match run(ex, "media.understandSpeech", json!({"items": movies, "wait": true})).await {
        Ok(_) => {
            if let Ok(v) = run(ex, "media.inspectSpeech", json!({"items": movies})).await {
                speech = crate::quick::parse_speech(&v);
            }
        }
        Err(e) => {
            let why = e.to_string();
            if !why.contains("none of the clips has audio") {
                report.fallbacks.push(format!("speech: not understood ({}): no captions, and talking is not weighed", chars(&why, 160)));
            }
        }
    }
    crate::quick::mark_speech(shots, &speech);
    speech
}

/// Direct the edit (the `autocut.direct` job; parameters as the engine checked them). `model`
/// directs it; none: rules.
pub async fn make(ex: &dyn Executor, p: &Value, model: Option<&dyn ChatModel>, lines: &mut dyn FnMut(&str)) -> Result<Value, AutoCutError> {
    // a revision: the saved edit's options under the ones given now, its plan changed by the notes
    let previous = match p["revise"].as_str().map(str::trim).filter(|n| !n.is_empty()) {
        Some(_) => Some(revise::load(&p["previous"]).map_err(invalid)?),
        None => None,
    };
    let merged = match &previous {
        Some(pr) => revise::merge(&pr.options, p, pr.plan.bpm),
        None => p.clone(),
    };
    // the taste profile fills what the video's own options leave out
    let taste = crate::taste::resolve(ex).await;
    let own = revise::saved_options(&merged);
    let merged = crate::taste::under(&merged, &crate::taste::director_options(&taste));
    let p = &merged;
    let mut o = Options::from_params(p)?;
    let mut report = Report { brief: o.brief.clone(), aspect: o.aspect.clone(), seconds: o.seconds, models: json!({}), ..Report::default() };
    report.options = own;
    report.taste = (!taste.is_empty()).then(|| json!(taste));
    let mut notes: Vec<String> = Vec::new();
    if o.download {
        for set in ["fast", "full"] {
            lines(&format!("downloading the scene models ({set}) if missing"));
            if let Err(e) = run(ex, "scenes.downloadModel", json!({"model": set, "wait": true})).await {
                notes.push(format!("download: scene models ({set}): {e}"));
            }
        }
    }
    // ---- the media and what is 360°
    let (media, filmed) = gather(ex, p, &mut notes, lines).await?;
    report.media = media.len();
    let items: Vec<u64> = media.iter().map(|m| m.item).collect();
    let props = run(ex, "media.vrProperties", json!({"items": items})).await.unwrap_or(Value::Null);
    let rows = props["items"].as_array().map(Vec::as_slice).unwrap_or_default();
    let vr: BTreeSet<u64> = rows.iter().filter(|x| !x["vr"].is_null()).filter_map(|x| x["item"].as_u64()).collect();
    report.vr_media = vr.len();
    // (an Insta360 recording is levelled by its motion sensor when it is imported)
    let tilts: Vec<f64> = rows.iter().filter(|x| x["vr"]["horizon"]["level"] == true).map(|x| finite(&x["vr"]["horizon"]["maxTilt"]).unwrap_or(0.0)).collect();
    report.levelled = tilts.len();
    report.max_tilt = tilts.iter().copied().fold(0.0, f64::max);
    if !tilts.is_empty() {
        lines(&format!("{} 360° recording(s) levelled by their camera's motion sensor (tilted up to {:.0}°)", tilts.len(), report.max_tilt));
    }
    // ---- the index and the speech
    let mut shots = index(ex, &media, &o, &mut report, &mut notes, lines).await;
    let speech = speech_of(ex, &media, &mut shots, &mut report, lines).await;
    report.shots_indexed = shots.len();
    // ---- where things happen round the sphere, and the flat clips' shapes
    let mut salience: HashMap<(u64, usize, usize), Vec<(f64, f64, String)>> = HashMap::new();
    let vr_pieces: Vec<(&ShotInfo, usize, (f64, f64, f64))> =
        shots.iter().filter(|s| vr.contains(&s.item)).flat_map(|s| pieces(s).into_iter().enumerate().map(move |(k, p)| (s, k, p))).take(MAX_SALIENCE).collect();
    if !vr_pieces.is_empty() {
        lines(&format!("looking round {} 360° moment(s) for the action", vr_pieces.len()));
    }
    for (s, part, (a, b, key)) in vr_pieces {
        if let Ok(v) = run(ex, "media.salience360", json!({"item": s.item, "seconds": key, "from": a, "to": b, "max": 3})).await {
            let dirs = v["directions"].as_array().map(Vec::as_slice).unwrap_or_default();
            let list = dirs
                .iter()
                .filter_map(|d| {
                    let what = if d["by"] == "motion" { "something moving" } else { "something to see" };
                    Some((finite(&d["yaw"])?, finite(&d["pitch"])?, what.to_string()))
                })
                .collect();
            salience.insert((s.item, s.index, part), list);
        }
    }
    let mut aspects: HashMap<u64, f64> = HashMap::new();
    for m in media.iter().filter(|m| !vr.contains(&m.item)) {
        if let Ok(r) = run(ex, "media.frame", json!({"item": m.item, "seconds": 0.0, "maxSide": 64})).await
            && let (Some(w), Some(h)) = (finite(&r["width"]), finite(&r["height"]))
            && h > 0.0
        {
            aspects.insert(m.item, w / h);
        }
    }
    let mut cands = candidates(&shots, &media, &vr, &salience, &aspects, &speech);
    for c in cands.iter_mut() {
        c.captured = filmed.get(&c.item).map(|t| t.saturating_add(c.start as i64));
    }
    // ---- which moments: each ranked (on its own frames where it can be measured), the unusable
    // and the near-duplicates left out
    report.measured = rank::measure(ex, &mut cands, lines).await;
    // ---- what it is about: a travel video shows the places, its people now and then
    let (about, why) = focus::focus_of(o.focus, &o.brief, &cands);
    o.focus = Some(about);
    if about == focus::Focus::Places {
        focus::for_places(&mut cands);
    }
    lines(&format!("about: {why}"));
    report.focus = Some(json!({"focus": about.name(), "why": why}));
    let bpm = o.bpm.unwrap_or(DEFAULT_BPM);
    let beat = beat_of(bpm);
    let total = ((o.seconds / beat).round() as u32).max(4);
    let need = rhythm(total, beat, cands.len()).len() + 2;
    let (mut cands, verdicts) = rank::choose(cands, need);
    if o.focus == Some(focus::Focus::Places) {
        let aside = focus::places_pool(&mut cands, need, need.saturating_sub(2));
        if aside > 0 {
            lines(&format!("{aside} indoor moment(s) set aside: a travel video is outdoors"));
        }
    }
    let dropped = verdicts.iter().filter(|v| !v.kept).count();
    if dropped > 0 {
        lines(&format!("{dropped} moment(s) left out (blurred, dark, shaky or a near-duplicate)"));
    }
    report.moments = verdicts;
    // ---- the plan
    let short = shortlist(&cands, MAX_CANDIDATES);
    let slots = rhythm(total, beat, cands.len());
    report.candidates = short
        .iter()
        .map(|c| {
            let mut v = json!({"clip": c.id, "name": c.name, "kind": c.kind, "from": round2(c.start), "to": round2(c.end), "what": c.caption, "energy": round2(c.energy), "quality": round2(c.quality), "rank": c.rank});
            if c.kind == Kind::Vr {
                v["looks"] = json!(c.looks.iter().map(|l| json!({"look": l.id, "what": l.what, "yaw": l.yaw.round(), "pitch": l.pitch.round()})).collect::<Vec<_>>());
            }
            v
        })
        .collect();
    let mut plan = None;
    if let Some(pr) = &previous {
        // ---- the saved plan, its shots found again, changed by the notes
        let notes_text = p["revise"].as_str().unwrap_or_default().to_string();
        let (mut revised, lost) = revise::remap(&pr.plan, &cands);
        let last_cut = windows(&pr.plan, &Beats::grid(pr.plan.bpm));
        let length = last_cut.last().map_or(o.seconds, |w| w.1);
        // the notes the rules read are taken literally; the model reads only the rest (left
        // the whole notes, it added changes nobody asked for: "warmer" became warmer, brighter
        // and more vivid)
        let (mut ops, mut unread) = revise::rules_ops(&notes_text, length);
        let mut by_model = false;
        if let Some(m) = model.filter(|_| !unread.is_empty()) {
            lines(&format!("revising with {}", m.id()));
            match revise::model_ops(m, &unread.join(". "), &revised, &last_cut, &cands).await {
                Ok((more, bad)) if !more.is_empty() => {
                    ops.extend(more);
                    unread = bad;
                    by_model = true;
                }
                Ok(_) | Err(_) => report.fallbacks.push(format!("revise: {} gave no usable change for the notes the rules could not read", m.id())),
            }
        }
        report.changes = revise::apply(&mut revised, &ops, &cands, &last_cut, beat, &mut o);
        report.changes.extend(unread.into_iter().map(|u| format!("not understood: {u}")));
        report.changes.extend(lost);
        report.revised_from = pr.path.clone();
        revised.by = format!("revised ({})", if by_model { format!("rules and {}", model.map_or_else(String::new, |m| m.id())) } else { "rules".into() });
        lines(&format!("{} change(s) to the saved plan", report.changes.len()));
        plan = Some(revised);
    }
    match model {
        _ if plan.is_some() => {}
        Some(m) => {
            lines(&format!("directing with {}", m.id()));
            match model_plan(m, &short, &o, beat, &slots).await {
                Ok((v, asked)) => match validate(&v, &cands, &o, beat, bpm, &slots) {
                    Ok(mut pl) => {
                        pl.by = format!("model:{}", m.id());
                        if asked > 1 {
                            pl.notes.insert(0, "the model's first answer was not JSON: it was asked again".into());
                        }
                        plan = Some(pl);
                    }
                    Err(why) => report.fallbacks.push(format!("director: {}'s plan was unusable ({why}): rules directed", m.id())),
                },
                Err(e) => report.fallbacks.push(format!("director: {} did not answer ({}): rules directed", m.id(), chars(&e.to_string(), 160))),
            }
        }
        None => report.fallbacks.push("director: no local model (Ollama): rules directed".into()),
    }
    let plan = match plan {
        Some(p) => p,
        None => {
            lines("directing by rules");
            rules_plan(&cands, &o, beat, bpm)
        }
    };
    if plan.shots.is_empty() {
        return Err(invalid("no shot could be planned from these clips"));
    }
    // ---- a travel video: the person holding the camera in a third of the shots at most
    let mut plan = plan;
    if o.focus == Some(focus::Focus::Places) {
        let mut changes = focus::cover_parts(&mut plan, &cands, beat);
        changes.extend(focus::in_time_order(&mut plan, &cands, beat));
        changes.extend(focus::limit_people(&mut plan, &cands));
        for change in changes {
            lines(&format!("repaired: {change}"));
            plan.notes.push(change);
        }
    }
    // ---- the style: when the edit leaves it to the director, chosen from the shots it planned
    // (a revision keeps the one its edit was made in)
    if o.auto_style {
        let kept =
            previous.as_ref().and_then(|pr| pr.plan.style.clone()).and_then(|st| cuts::Style::from_params(&json!({"style": st}), &cuts::Style::default()).ok());
        let (base, why) = match kept {
            Some(s) => {
                let why = format!("{}: as the edit was made", s.pace.name());
                (s, why)
            }
            None => cuts::style_from_shots(&plan, &cands, &o.brief),
        };
        o.style = cuts::Style::from_params(&o.style_given, &base).unwrap_or(base);
        lines(&format!("style: {why}"));
        report.style =
            Some(json!({"style": o.style.pace.name(), "by": "director", "why": why, "lightEffects": o.style.light_effects, "sounds": o.style.sounds}));
    } else {
        report.style = Some(json!({"style": o.style.pace.name(), "by": "asked", "lightEffects": o.style.light_effects, "sounds": o.style.sounds}));
    }
    plan.style = Some(o.style.pace.name().into());
    // ---- how each cut is made: from what the shots show and say, in the edit's style
    cuts::joins(&mut plan, &cands, &speech, beat, &o.style);
    for s in plan.shots.iter_mut() {
        s.file = cands.iter().find(|c| c.id == s.clip).map(|c| c.name.clone()).unwrap_or_default();
        // the profile's wording ("our son", never "my son") in every text on screen
        s.text = s.text.as_deref().map(|t| taste.apply_words(t));
    }
    plan.title = taste.apply_words(&plan.title);
    report.director = plan.by.clone();
    let (fw, fh) = o.frame();
    let frame = (f64::from(fw), f64::from(fh));
    // ---- dry run: the plan and its commands on the beat grid
    if o.dry_run {
        if model.is_some() {
            report.prompt = Some(json!({"system": DIRECTOR_SYSTEM, "user": director_prompt(&short, &o, beat, &slots)}));
        }
        let wins = windows(&plan, &Beats::grid(bpm));
        report.steps = compile(&plan, &cands, &wins, &o, frame, beat);
        report.commands = report.steps.len();
        report.plan = Some(plan);
        report.warnings = notes;
        return Ok(finish_json(&report, "Director: planned (dry run)"));
    }
    // ---- the sequence and the music: its beats are the grid
    let name = o.name.clone().unwrap_or_else(|| format!("{} — Directed", plan.title));
    let r = run(ex, "file.newSequence", json!({"name": name, "width": fw, "height": fh, "fps": FPS, "video": 3, "audio": 4})).await?;
    let seq = r["sequence"].as_u64().ok_or_else(|| AutoCutError::Executor("file.newSequence made no sequence".into()))?;
    report.sequence = Some(seq);
    let total_secs = f64::from(plan.beats()) * beat;
    let (beats, music_clip) = music(ex, &o, &plan, total_secs, &mut report, &mut notes).await;
    let wins = windows(&plan, &beats);
    let steps = compile(&plan, &cands, &wins, &o, frame, beats.period());
    report.commands = steps.len();
    // ---- the shots
    lines(&format!("building {} shot(s), {} command(s)", plan.shots.len(), steps.len()));
    let built = execute(ex, &steps, &mut notes, lines).await;
    // ---- the colour pass: each shot measured and gently balanced, the cameras brought together
    let grades: serde_json::Map<String, Value> = plan
        .shots
        .iter()
        .zip(&built.pictures)
        .filter_map(|(s, pic)| {
            let g = s.grade?;
            Some((pic.as_ref()?.to_string(), json!({"temperature": g.warmth, "exposure": g.brightness, "saturation": g.saturation})))
        })
        .collect();
    if o.color || !grades.is_empty() {
        let clips: Vec<u64> = built.pictures.iter().flatten().copied().collect();
        if !clips.is_empty() {
            lines("balancing the colour of each shot");
            let strength = if o.color { o.color_strength } else { 0.0 };
            let mut cp = json!({"clips": clips, "strength": strength, "match": o.color, "look": o.look, "lookStrength": o.look_strength});
            if !grades.is_empty() {
                cp["offsets"] = Value::Object(grades);
            }
            if let Some(out) = &o.export {
                cp["sheet"] = json!(crate::trailer::beside(&filmcraft_engine::export_tools::expand_home(out), "color.jpg").display().to_string());
            }
            match run(ex, "clip.autoColor", cp).await {
                Ok(v) => report.color = Some(color_summary(&v)),
                Err(e) => report.fallbacks.push(format!("colour: left as shot ({})", chars(&e.to_string(), 160))),
            }
        }
    }
    // ---- captions of English speech
    if o.captions {
        report.captions = Some(captions(ex, &plan, &cands, &wins, &speech, &o, &mut notes).await);
    }
    // ---- privacy, when asked
    if let Some(rd) = &o.redact {
        let clips: Vec<u64> = built.pictures.iter().flatten().copied().collect();
        if !clips.is_empty() {
            match run(ex, "clip.autoRedact", json!({"clips": clips, "kinds": rd.kinds_json(), "verify": rd.verify, "wait": true})).await {
                Ok(v) => report.privacy = Some(v["redact"]["clipsChanged"].clone()),
                Err(e) => report.fallbacks.push(format!("privacy: nothing was hidden ({})", chars(&e.to_string(), 160))),
            }
        }
    }
    // ---- the mix
    report.mix = Some(mix(ex, &built, music_clip, &plan, &cands, &o.style, o.focus, &mut notes).await);
    if let Err(e) = run(ex, "audio.normalizeLoudness", json!({"target": "social", "truePeak": crate::look::TRUE_PEAK})).await {
        notes.push(format!("loudness: {e}"));
    }
    if o.vertical() {
        match run(ex, "sequence.checkSafeZones", json!({"platform": o.platform.name(), "sequence": seq})).await {
            Ok(v) => {
                let overlaps = v["overlaps"].as_array().map_or(0, Vec::len);
                report.safe_zones = Some(json!({"platform": o.platform.name(), "ok": overlaps == 0, "overlaps": overlaps}));
            }
            Err(e) => notes.push(format!("safe zones: {e}")),
        }
    }
    report.plan = Some(plan);
    // ---- the files
    if let Some(out) = o.export.clone() {
        export(ex, &o, seq, &out, &mut report, &mut notes, lines).await;
    }
    report.warnings.extend(notes);
    let status = match &report.export {
        Some(p) => format!(
            "Director: {} shot(s), {:.1} s, directed by {}, exported to {p}",
            report.plan.as_ref().map_or(0, |p| p.shots.len()),
            total_secs,
            report.director
        ),
        None => format!("Director: {} shot(s), {:.1} s, directed by {}", report.plan.as_ref().map_or(0, |p| p.shots.len()), total_secs, report.director),
    };
    if let Some(path) = report.export.clone() {
        let file = crate::trailer::beside(&path, "report.json");
        report.report_file = Some(file.display().to_string());
        if let Err(e) = std::fs::write(&file, format!("{}\n", serde_json::to_string_pretty(&report).unwrap_or_default())) {
            report.warnings.push(format!("report: {}: {e}", file.display()));
            report.report_file = None;
        }
    }
    Ok(finish_json(&report, &status))
}

/// What the colour pass did, for the report: per shot its correction, what it said and any
/// backing off; whether the GPU measured; the before/after sheet.
fn color_summary(v: &Value) -> Value {
    let clips: Vec<Value> = v["clips"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .map(|c| json!({"clip": c["clip"], "applied": c["applied"], "did": c["did"], "backedOff": c["backedOff"], "before": c["before"], "after": c["after"]}))
        .collect();
    json!({"clips": clips, "gpu": v["gpu"], "look": v["look"], "strength": v["strength"], "sheet": v["sheet"]["path"]})
}

fn finish_json(report: &Report, status: &str) -> Value {
    let mut out = serde_json::to_value(report).unwrap_or_else(|_| json!({}));
    out["status"] = json!(status);
    out
}

/// The music on A3 from 0: generated at the plan's tempo and mood to the edit's length (or
/// `music`, placed and measured); its beats, and its clip.
async fn music(ex: &dyn Executor, o: &Options, plan: &Plan, secs: f64, report: &mut Report, notes: &mut Vec<String>) -> (Beats, Option<u64>) {
    let grid = Beats::grid(plan.bpm);
    if let Some(file) = &o.music {
        let imported = run(ex, "file.import", json!({"paths": [filmcraft_engine::export_tools::expand_home(file)], "reuse": true})).await;
        let Some(item) = imported.ok().and_then(|r| r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64)) else {
            notes.push(format!("music: {file} could not be imported: generated instead"));
            return generate_music(ex, o, plan, secs, report, notes).await;
        };
        let placed = run(ex, "timeline.place", json!({"item": item, "audioTrack": "A3", "video": false, "time": 0, "duration": ticks(secs)})).await;
        let clip = placed.ok().and_then(|r| r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64));
        let beats = match run(ex, "media.beats", json!({"item": item})).await {
            Ok(b) => {
                let times: Vec<f64> = b["beats"].as_array().map(|a| a.iter().filter_map(finite).collect()).unwrap_or_default();
                let period = finite(&b["period"]).or_else(|| finite(&b["bpm"]).map(|x| 60.0 / x)).unwrap_or(grid.period());
                Beats::measured(&times, period).unwrap_or(grid)
            }
            Err(e) => {
                notes.push(format!("music: its beats were not measured ({e}): the cuts follow {:.0} bpm", plan.bpm));
                grid
            }
        };
        report.music = Some(json!({"file": file, "item": item, "clip": clip}));
        return (beats, clip);
    }
    generate_music(ex, o, plan, secs, report, notes).await
}

async fn generate_music(ex: &dyn Executor, o: &Options, plan: &Plan, secs: f64, report: &mut Report, notes: &mut Vec<String>) -> (Beats, Option<u64>) {
    let grid = Beats::grid(plan.bpm);
    // exactly the edit's length: generated music ends on a bar with its final chord there
    let p = json!({"kind": "music", "duration": secs, "mood": plan.mood, "bpm": plan.bpm, "seed": o.seed, "time": 0, "track": "A3"});
    match run(ex, "audio.generate", p).await {
        Ok(m) => {
            let map: Vec<i64> = m["beats"].as_array().map(|a| a.iter().filter_map(Value::as_i64).take(100_000).collect()).unwrap_or_default();
            report.music = Some(json!({"generated": true, "mood": plan.mood, "bpm": m["bpm"], "genre": m["genre"], "clip": m["clip"], "seed": o.seed}));
            (Beats::from_ticks(&map, 0.0).unwrap_or(grid), m["clip"].as_u64())
        }
        Err(e) => {
            notes.push(format!("music: {e}: the edit has no music; the cuts follow {:.0} bpm", plan.bpm));
            (grid, None)
        }
    }
}

/// What running the steps made.
#[derive(Clone, Debug, Default)]
pub struct Built {
    /// Each shot's picture clip and its sound (none: not placed).
    pub pictures: Vec<Option<u64>>,
    pub sounds: Vec<Option<u64>>,
    /// The swishes and the chime.
    pub effects: Vec<u64>,
    /// Steps that failed (cosmetic: the edit stands without them).
    pub failed: usize,
}

/// Run the steps in order, each shot's own on its clip; a shot that cannot be placed is left
/// out with its steps, any other failed step is noted.
pub async fn execute(ex: &dyn Executor, steps: &[Step], notes: &mut Vec<String>, lines: &mut dyn FnMut(&str)) -> Built {
    let shots = steps.iter().filter_map(|s| s.shot).max().map_or(0, |m| m + 1);
    let mut b = Built { pictures: vec![None; shots], sounds: vec![None; shots], ..Built::default() };
    let mut text: Option<u64> = None;
    for (i, s) in steps.iter().enumerate() {
        let mut p = s.params.clone();
        let clip = s.shot.and_then(|k| b.pictures.get(k).copied().flatten());
        match s.on {
            On::Nothing | On::Place => {}
            On::Clip | On::Clips => {
                let Some(c) = clip else { continue };
                p[if s.on == On::Clip { "clip" } else { "clips" }] = if s.on == On::Clip { json!(c) } else { json!([c]) };
            }
            On::Text => {
                let Some(t) = text else { continue };
                p["clip"] = json!(t);
            }
        }
        if i % 8 == 0
            && let Some(k) = s.shot
        {
            lines(&format!("shot {}: {}", k + 1, s.id));
        }
        match run(ex, &s.id, p).await {
            Ok(r) => match (s.on, s.id.as_str()) {
                (On::Place, _) => {
                    let ids: Vec<u64> = r["clips"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
                    if let Some(k) = s.shot {
                        if let Some(slot) = b.pictures.get_mut(k) {
                            *slot = ids.first().copied();
                        }
                        if let Some(slot) = b.sounds.get_mut(k) {
                            *slot = ids.get(1).copied();
                        }
                    }
                }
                (_, "graphics.newText") => text = r["clip"].as_u64(),
                (_, "audio.generate") => b.effects.extend(r["clip"].as_u64()),
                _ => {}
            },
            Err(e) => {
                b.failed += 1;
                let what = s.shot.map_or_else(String::new, |k| format!("shot {}: ", k + 1));
                notes.push(format!("{what}{}", chars(&e.to_string(), 200)));
            }
        }
    }
    b
}

/// Captions of the English speech the shots play: one per kept segment, on C1, animated.
async fn captions(
    ex: &dyn Executor,
    plan: &Plan,
    cands: &[Candidate],
    wins: &[(f64, f64)],
    speech: &[ItemSpeech],
    o: &Options,
    notes: &mut Vec<String>,
) -> usize {
    let mut cues: Vec<(f64, f64, String)> = Vec::new();
    let places = o.focus == Some(focus::Focus::Places);
    for (s, (t0, t1)) in plan.shots.iter().zip(wins) {
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        // (a travel video under music: only someone talking on screen is captioned, not words
        // caught in passing)
        if s.camera == Camera::Hyperlapse || c.kind == Kind::Photo || (places && !focus::voiced(o.focus, c, s)) {
            continue;
        }
        let Some(sp) = speech.iter().find(|x| x.item == c.item && x.language.starts_with("en")) else { continue };
        let (a, b) = (s.from, s.from + (t1 - t0));
        for k in sp.kept.iter().filter(|k| k.end > a && k.start < b && !k.text.trim().is_empty()) {
            let start = t0 + (k.start - a).max(0.0);
            let end = (t0 + (k.end - a)).min(*t1);
            // (a travel video: whole sentences only, not words caught in passing)
            let whole = !places || k.text.trim_end_matches(['"', '\'', '”', ')']).ends_with(['.', '!', '?', '…']);
            if end - start >= 0.4 && english(&k.text) && whole {
                cues.push((start, end, chars(k.text.trim(), 120)));
            }
        }
    }
    if cues.is_empty() {
        return 0;
    }
    if let Err(e) = run(ex, "captions.newTrack", json!({"format": "Subtitle", "name": "Captions", "language": "en"})).await {
        notes.push(format!("captions: {e}"));
        return 0;
    }
    let mut n = 0;
    for (a, b, text) in &cues {
        if run(ex, "captions.add", json!({"track": "C1", "text": text, "seconds": a, "durationSeconds": round2(b - a)})).await.is_ok() {
            n += 1;
        }
    }
    let mut style = json!({"track": "C1", "preset": o.caption_style});
    if o.vertical() {
        style["options"] = json!({"platform": o.platform.name()});
    }
    if let Err(e) = run(ex, "captions.setAnimatedStyle", style).await {
        notes.push(format!("captions: {e}"));
    }
    n
}

/// The mix: the music carries the edit, each shot's own sound well under it (someone talking
/// over it: the speech mix, the voice first), the swishes and chime under the music.
#[allow(clippy::too_many_arguments)]
async fn mix(
    ex: &dyn Executor,
    b: &Built,
    music: Option<u64>,
    plan: &Plan,
    cands: &[Candidate],
    style: &cuts::Style,
    about: Option<focus::Focus>,
    notes: &mut Vec<String>,
) -> Value {
    // the cuts' sounds as loud as the style asks (the subtle style's −24 LUFS at its 0.4)
    let effects_lufs = (EFFECTS_LUFS + 10.0 * (style.sound_level - 0.4)).clamp(-32.0, -16.0);
    let sounds: Vec<u64> = b.sounds.iter().flatten().copied().collect();
    let talking: Vec<u64> = plan
        .shots
        .iter()
        .zip(&b.sounds)
        .filter(|(s, _)| cands.iter().find(|c| c.id == s.clip).is_some_and(|c| focus::voiced(about, c, s)))
        .filter_map(|(_, a)| *a)
        .collect();
    if !talking.is_empty() {
        let tracks = crate::mix::MixTracks { sounds: "A1".into(), voice: "A1".into(), music: "A3".into(), effects: vec!["A4".into()] };
        let m = crate::mix::Mix {
            sounds: sounds.iter().filter(|c| !talking.contains(c)).copied().collect(),
            effects: b.effects.clone(),
            narrations: talking,
            music: music.into_iter().collect(),
            enhance_voice: false,
        };
        let v = crate::mix::mix_down(ex, &m, &tracks, crate::mix::MixProfile::Speech, false, notes).await;
        return json!({"profile": "speech", "speech": v});
    }
    let step = async |id: &str, p: Value, notes: &mut Vec<String>| {
        if let Err(e) = run(ex, id, p).await {
            notes.push(format!("mix: {e}"));
        }
    };
    if let Some(m) = music {
        step("essentialSound.setType", json!({"clips": [m], "type": "music"}), notes).await;
        step("essentialSound.autoMatch", json!({"clips": [m], "target": BED_LUFS}), notes).await;
        let mut chain = vec![("parametric_eq", crate::mix::bed_eq(crate::mix::BED_SHELF_DB))];
        chain.extend(crate::mix::bed_dynamics(BED_LUFS));
        crate::mix::inserts(ex, "A3", &chain, notes).await;
    }
    if !sounds.is_empty() {
        step("essentialSound.setType", json!({"clips": sounds, "type": "ambience"}), notes).await;
        step("essentialSound.autoMatch", json!({"clips": sounds, "target": SOUND_LUFS}), notes).await;
        crate::mix::inserts(ex, "A1", &[("parametric_eq", crate::mix::sound_eq())], notes).await;
    }
    if !b.effects.is_empty() {
        step("essentialSound.setType", json!({"clips": b.effects, "type": "sfx"}), notes).await;
        step("essentialSound.autoMatch", json!({"clips": b.effects, "target": effects_lufs}), notes).await;
        crate::mix::inserts(ex, "A4", &[("parametric_eq", crate::mix::effects_eq())], notes).await;
    }
    json!({"profile": "music", "music": BED_LUFS, "sounds": SOUND_LUFS, "effects": effects_lufs})
}

/// Export the edit (captions burned in), a frame sheet and QA beside it.
async fn export(ex: &dyn Executor, o: &Options, seq: u64, out: &str, report: &mut Report, notes: &mut Vec<String>, lines: &mut dyn FnMut(&str)) {
    let out = filmcraft_engine::export_tools::expand_home(out);
    if let Some(dir) = Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty())
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        notes.push(format!("export: {}: {e}", dir.display()));
        return;
    }
    lines(&format!("exporting {out}"));
    let mut ep = json!({"sequence": seq, "path": out, "wait": true, "burnCaptions": true, "vr": false});
    match o.aspect.as_str() {
        "9:16" => ep["preset"] = json!(crate::trailer::VERTICAL_PRESET),
        "16:9" => ep["preset"] = json!("YouTube 1080p Full HD"),
        _ => ep["format"] = json!("h264"),
    }
    // what rendered and encoded it: the counters before and after
    let counters = |v: &Value| -> [f64; 4] {
        [
            finite(&v["export"]["gpu"]["frames"]).unwrap_or(0.0),
            finite(&v["export"]["gpu"]["fallbacks"]).unwrap_or(0.0),
            finite(&v["export"]["hardware"]["frames"]).unwrap_or(0.0),
            finite(&v["decode"]["hardware"]["frames"]).unwrap_or(0.0),
        ]
    };
    let reasons = |v: &Value| -> HashMap<String, f64> {
        v["export"]["gpu"]["cpuWork"].as_object().map(|o| o.iter().filter_map(|(k, n)| Some((k.clone(), n.as_f64()?))).take(64).collect()).unwrap_or_default()
    };
    let stats = run(ex, "perf.stats", json!({})).await.ok();
    let before = stats.as_ref().map(counters);
    let reasons_before = stats.as_ref().map(reasons).unwrap_or_default();
    let r = match run(ex, "file.exportMedia", ep).await {
        Ok(r) => r,
        Err(e) => {
            notes.push(format!("export: {e}"));
            return;
        }
    };
    let stats = run(ex, "perf.stats", json!({})).await.ok();
    let after = stats.as_ref().map(counters);
    let reasons_after = stats.as_ref().map(reasons).unwrap_or_default();
    let written = r["path"].as_str().or_else(|| r["result"]["path"].as_str()).map(str::to_string).unwrap_or_else(|| out.clone());
    let mut render = json!({"seconds": r["result"]["seconds"], "frames": r["result"]["frames"]});
    if let (Some(a), Some(b)) = (before, after) {
        let d = |i: usize| (b.get(i).copied().unwrap_or(0.0) - a.get(i).copied().unwrap_or(0.0)).max(0.0);
        render["gpuFrames"] = json!(d(0));
        render["cpuFallbackFrames"] = json!(d(1));
        render["hardwareEncodedFrames"] = json!(d(2));
        render["hardwareDecodedFrames"] = json!(d(3));
        // why the CPU drew what it drew
        let mut why: Vec<(String, f64)> =
            reasons_after.iter().map(|(k, n)| (k.clone(), n - reasons_before.get(k).copied().unwrap_or(0.0))).filter(|(_, n)| *n > 0.0).collect();
        why.sort_by(|a, b| b.1.total_cmp(&a.1));
        render["cpuWork"] = json!(why.into_iter().take(12).map(|(k, n)| json!({"reason": k, "count": n})).collect::<Vec<_>>());
    }
    report.render = Some(render);
    report.export = Some(written.clone());
    let secs = report.plan.as_ref().map_or(o.seconds, |p| f64::from(p.beats()) * beat_of(p.bpm));
    let times: Vec<f64> =
        (0..crate::quick::SHEET_FRAMES).map(|k| ((k as f64 + 0.5) * secs / crate::quick::SHEET_FRAMES as f64 * 30.0).round() / 30.0).collect();
    let sheet = crate::trailer::beside(&written, "sheet.jpg");
    match crate::trailer::frames_sheet(&written, &times, &sheet).await {
        Ok(()) => report.sheet = Some(sheet.display().to_string()),
        Err(e) => notes.push(format!("sheet: {e}")),
    }
    if o.qa.unwrap_or(true) {
        lines("QA");
        run(ex, "sequence.open", json!({"item": seq})).await.ok();
        let q = crate::qa::Options {
            subject: if o.brief.is_empty() { "subject".into() } else { chars(&o.brief, 80) },
            picture_track: "V1".into(),
            sound_track: String::new(),
            voice_track: String::new(),
            music_track: "A3".into(),
            faces: false,
            platform: o.vertical().then(|| o.platform.name().into()),
            ..crate::qa::Options::default()
        };
        let mut progress = |l: &str| lines(l);
        let mut qa = crate::qa::check(ex, &q, None, &mut progress).await;
        if let Err(e) = crate::qa::write_next_to(&mut qa, Path::new(&written)) {
            notes.push(format!("QA: {e}"));
        }
        report.qa_summary = Some(qa.summary());
    }
}

#[path = "director_cuts.rs"]
pub mod cuts;

#[path = "director_focus.rs"]
pub mod focus;

#[path = "director_rank.rs"]
pub mod rank;

#[path = "director_revise.rs"]
pub mod revise;

#[cfg(test)]
#[path = "director_tests.rs"]
mod tests;
