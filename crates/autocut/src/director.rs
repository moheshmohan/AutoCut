//! The director (`autocut.direct`): normal and 360° clips → one catchy edit on the beat, shot by
//! shot, as the local model (or, without one, rules) directs it.
//!
//! 1. **Media.** `items`, `paths` (files or folders), `insta360` (recordings by name: imported
//!    from Insta360 Studio's download folder, else downloaded from the user's Insta360 cloud) and,
//!    through the host, a Google Photos `album`; none: the project's movies and photos.
//! 2. **Analysis.** The scene index (`media.analyzeScenes`: cuts, quality, the camera's move, the
//!    describer's subjects and their boxes — 360° media is indexed as its stitched panorama, so a
//!    box is a direction), the speech (`media.understandSpeech`), and for each 360° shot where
//!    things happen round the sphere (`media.salience360`).
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
//!    the grid every cut lands on; the shots placed and moved; a swish on each whip and roll and a
//!    soft chime under the title — never a whoosh or a riser; a cross dissolve into the ending;
//!    the title in English inside the platform's safe zone; captions of English speech; each
//!    sound levelled and the mix at −14 LUFS; with `export` the video, a frame sheet, the report
//!    and QA beside it. No colour looks, light leaks or flashes: the footage as shot.
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
/// Shots the model is told about, at most (the best of a long library).
pub const MAX_CANDIDATES: usize = 48;
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
const MOODS: &[&str] = &["epic", "tense", "uplifting", "calm", "happy", "dark", "dreamy", "energetic", "sad"];
/// Sound levels: the music carries the edit, the clips' own sound under it, the swishes and
/// chimes under that.
const BED_LUFS: f64 = -16.0;
const SOUND_LUFS: f64 = -27.0;
const EFFECTS_LUFS: f64 = -24.0;

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
    pub handheld: bool,
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
                        looks.push(Look { id: String::new(), yaw, pitch: pitch.clamp(-60.0, 60.0), what: chars(&sub.name, 60), named: true });
                    }
                } else if subject.is_none() {
                    subject = Some(b.map(f64::from));
                }
            }
        }
        if kind == Kind::Vr {
            for (yaw, pitch, what) in salience.get(&(s.item, s.index, part)).map(Vec::as_slice).unwrap_or_default() {
                if looks.iter().all(|l| angle_between(l.yaw, *yaw) >= 30.0) {
                    looks.push(Look { id: String::new(), yaw: *yaw, pitch: pitch.clamp(-45.0, 45.0), what: what.clone(), named: false });
                }
            }
            if looks.is_empty() {
                looks.push(Look { id: String::new(), yaw: 0.0, pitch: 0.0, what: "straight ahead".into(), named: false });
            }
            for (k, l) in looks.iter_mut().enumerate() {
                l.id = format!("{id}{}", char::from(b'a' + (k.min(25) as u8)));
            }
        }
        let uses = d.map(|d| d.uses.clone()).unwrap_or_default();
        let media_seconds = media.iter().find(|m| m.item == s.item).map_or(s.end, |m| m.seconds.max(s.end));
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
            energy: energy_of(s),
            quality: f64::from(s.shot.quality.score.clamp(0.0, 1.0)),
            hook: f64::from(filmcraft_scenes::best::score(&s.shot, ShotUse::Hook).0.clamp(0.0, 1.0)),
            wide: d.is_some_and(|d| d.shot_type == ShotType::Wide) || uses.contains(&ShotUse::Establishing),
            indoor: indoors(&format!("{} {}", s.caption(), d.map(|d| d.setting.as_str()).unwrap_or_default())),
            moving: s.shot.camera != CameraMove::Static || d.is_some_and(|d| !d.actions.is_empty()),
            handheld: s.shot.camera == CameraMove::Handheld || s.shot.quality.shake > 0.35,
            talking: s.talking,
            said: if s.speech > 0.2 { said_in(speech, s.item, start, end) } else { None },
            looks,
            subject,
            aspect: aspects.get(&s.item).copied().filter(|a| a.is_finite() && *a > 0.0).unwrap_or(16.0 / 9.0),
            order: s.order,
        }
    }
}

/// The candidates worth telling the model about: the best by quality and hook, at most `n`,
/// in filming order (every clip keeps at least its best shot).
pub fn shortlist(cands: &[Candidate], n: usize) -> Vec<&Candidate> {
    let worth = |c: &Candidate| 0.45 * c.quality + 0.35 * c.hook + 0.2 * c.energy + if c.kind == Kind::Vr { 0.1 } else { 0.0 };
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

/// How a shot comes in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Enter {
    #[default]
    Cut,
    /// A cross dissolve (into the ending).
    Dissolve,
    /// A whip between two 360° shots: the one before turns away, this one turns in (a swish).
    Whip,
}

/// One shot of the edit.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlannedShot {
    /// Its candidate's id.
    pub clip: String,
    /// Where it starts in the media, seconds.
    pub from: f64,
    pub beats: u32,
    pub camera: Camera,
    /// 360°: where it looks, and where a whip pan goes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    pub enter: Enter,
    /// Text over it (English).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub why: String,
}

/// The edit as directed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Plan {
    pub title: String,
    pub mood: String,
    pub bpm: f64,
    pub shots: Vec<PlannedShot>,
    /// `model:<id>` or `rules`.
    pub by: String,
    /// What was repaired or decided, and why.
    pub notes: Vec<String>,
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
                (true, if n(Camera::Drift) <= n(Camera::PushIn) { Camera::Drift } else { Camera::PushIn }),
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
    let mut plan = Plan { title: title.clone(), mood, bpm, shots: Vec::new(), by: "rules".into(), notes: Vec::new() };
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
        body.push(PlannedShot { clip: c.id.clone(), from, beats, camera: cam, look, to, enter, text: None, why: format!("{} in filming order", c.caption) });
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
        from: hook_from,
        beats: hook_beats,
        camera: hook_cam,
        look: hook_look,
        to: None,
        enter: Enter::Cut,
        text: Some(title),
        why: format!("the hook: {}", hook.caption),
    });
    plan.shots.extend(body);
    if let Some(c) = end {
        let need = f64::from(end_beats) * beat;
        if let Some(from) = window_from(c, need) {
            plan.shots.push(PlannedShot {
                clip: c.id.clone(),
                from,
                beats: end_beats,
                camera: end_cam,
                look: c.looks.first().map(|l| l.id.clone()),
                to: None,
                enter: Enter::Dissolve,
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

const DIRECTOR_SYSTEM: &str = "You are a film director and editor cutting a short, catchy social video from the user's own clips: normal clips and 360° clips (a 360° clip films every direction at once; you choose where its virtual camera looks and how it moves). Cut on the beat of the music. Open with a hook — a 360° clip's tiny planet intro (planetIn) when there is a good one — then tell the story in filming order, alternating 360° and normal clips, and end on a closer (a 360° planetOut, or a wide shot). Keep it tasteful: each 360° move at most twice, a barrel roll at most once, no gimmicks for their own sake. Vary shot lengths with the energy. Use the clip ids, look ids and camera names you are given, and nothing else. The title and any text are short English. Answer with JSON only.";

/// What the model is told: the brief, the length in beats, the cameras, the clips.
pub fn director_prompt(cands: &[&Candidate], o: &Options, beat: f64, total: u32) -> String {
    let cams: Vec<Value> = Camera::ALL.iter().map(|c| json!({"camera": c.id(), "means": c.meaning(), "minBeats": beats_for(c.min_seconds(), beat)})).collect();
    let clips: Vec<Value> = cands
        .iter()
        .map(|c| {
            let mut v = json!({
                "clip": c.id,
                "kind": c.kind,
                "seconds": round2(c.seconds()),
                "what": c.caption,
                "energy": round2(c.energy),
                "quality": round2(c.quality),
                "hook": round2(c.hook),
            });
            if c.wide {
                v["wide"] = json!(true);
            }
            if c.indoor {
                v["indoors"] = json!(true);
            }
            if c.moving {
                v["moving"] = json!(true);
            }
            if let Some(s) = &c.said {
                v["says"] = json!(chars(s, 100));
            }
            if c.kind == Kind::Vr {
                v["looks"] = json!(c.looks.iter().take(4).map(|l| json!({"look": l.id, "what": l.what, "yaw": l.yaw.round()})).collect::<Vec<_>>());
            }
            v
        })
        .collect();
    let answer = json!({
        "title": "a short English title (at most 5 words)",
        "mood": MOODS.join("|"),
        "shots": [{"clip": "c1", "beats": 8, "camera": "planetIn", "look": "c1a", "to": null, "enter": "cut|dissolve|whip", "text": "optional short English text, or null", "why": "one short reason"}],
    });
    json!({
        "brief": o.brief,
        "seconds": o.seconds,
        "aspect": o.aspect,
        "beatSeconds": round2(beat),
        "totalBeats": total,
        "rules": [
            format!("the shots' beats add up to exactly {total}"),
            "a shot's beats are at least its camera's minBeats",
            "360 cameras only on 360 clips; a normal clip uses hold, pushIn, follow or speedRamp",
            "whipPan needs `look` and `to`: two different looks of that clip",
            "a clip may be used twice only if it is long and the two shots look at different things",
            "enter `whip` only between two 360 shots; `dissolve` into the last shot at most",
        ],
        "cameras": cams,
        "clips": clips,
        "answer": answer,
    })
    .to_string()
}

/// Ask the model for a plan (its raw answer).
pub async fn model_plan(model: &dyn ChatModel, cands: &[&Candidate], o: &Options, beat: f64, total: u32) -> Result<Value, AutoCutError> {
    model.complete_json(DIRECTOR_SYSTEM, &director_prompt(cands, o, beat, total), &[]).await
}

/// The model's plan made sound: unknown clips dropped, cameras fitted to the clip, lengths at
/// least what their camera needs and inside the clip's media, looks checked, a barrel roll at
/// most once, the text English, the total brought to the length asked (shots shortened or
/// dropped from the end, or the longest held for longer, else rules' shots added). The notes
/// say every repair. `Err` when nothing usable is left.
pub fn validate(v: &Value, cands: &[Candidate], o: &Options, beat: f64, bpm: f64) -> Result<Plan, String> {
    let total = ((o.seconds / beat).round() as u32).max(4);
    let v = plan_body(v);
    let list = shots_in(v).ok_or_else(|| {
        let keys: Vec<&str> = v.as_object().map(|o| o.keys().map(String::as_str).take(8).collect()).unwrap_or_default();
        format!("the plan has no shots (it answered with {})", if keys.is_empty() { "no object".to_string() } else { keys.join(", ") })
    })?;
    let mut notes: Vec<String> = Vec::new();
    let mut shots: Vec<PlannedShot> = Vec::new();
    let mut used_windows: Vec<(String, f64, f64, Option<String>)> = Vec::new();
    let mut rolls = 0usize;
    let mut cams: HashMap<Camera, usize> = HashMap::new();
    let frame_aspect = {
        let (w, h) = o.frame();
        f64::from(w) / f64::from(h.max(1))
    };
    for (k, s) in list.iter().enumerate().take(MAX_SHOTS) {
        let Some(c) = s["clip"].as_str().and_then(|id| cands.iter().find(|c| c.id == id)) else {
            notes.push(format!("shot {}: no clip `{}`: left out", k + 1, chars(s["clip"].as_str().unwrap_or("?"), 20)));
            continue;
        };
        let asked = s["camera"].as_str().and_then(Camera::from_id);
        let mut cam = match asked {
            Some(m) if m.for_kind(c.kind) => m,
            Some(m) => {
                let r = rules_camera(c, 2.0, &cams, None, frame_aspect);
                notes.push(format!("shot {} ({}): {} is not for a {} clip: {}", k + 1, c.id, m.id(), kind_word(c.kind), r.id()));
                r
            }
            None => rules_camera(c, 2.0, &cams, None, frame_aspect),
        };
        if cam == Camera::Reveal && c.indoor {
            notes.push(format!("shot {} ({}): indoors a reveal from the sky shows the ceiling: a drift instead", k + 1, c.id));
            cam = Camera::Drift;
        }
        if cam == Camera::BarrelRoll {
            rolls += 1;
            if rolls > 1 {
                notes.push(format!("shot {} ({}): a second barrel roll: a drift instead", k + 1, c.id));
                cam = Camera::Drift;
            }
        }
        let mut look = s["look"].as_str().filter(|l| c.look(l).is_some()).map(str::to_string);
        let mut to = s["to"].as_str().filter(|l| c.look(l).is_some()).map(str::to_string);
        if c.kind == Kind::Vr && look.is_none() {
            look = c.looks.first().map(|l| l.id.clone());
        }
        if cam == Camera::WhipPan {
            let far = |a: &Option<String>, b: &Option<String>| match (a.as_deref().and_then(|x| c.look(x)), b.as_deref().and_then(|x| c.look(x))) {
                (Some(x), Some(y)) => angle_between(x.yaw, y.yaw) >= LOOKS_APART,
                _ => false,
            };
            if !far(&look, &to) {
                match c.second_look() {
                    Some(l2) => {
                        look = c.looks.first().map(|l| l.id.clone());
                        to = Some(l2.id.clone());
                    }
                    None => {
                        notes.push(format!("shot {} ({}): a whip pan needs two things to turn between: a drift instead", k + 1, c.id));
                        cam = Camera::Drift;
                        to = None;
                    }
                }
            }
        } else {
            to = None;
        }
        let min = beats_for(cam.min_seconds(), beat);
        let asked_beats = s["beats"].as_u64().map_or(min, |b| u32::try_from(b.min(64)).unwrap_or(64).max(1));
        let mut beats = asked_beats.max(min);
        if beats != asked_beats {
            notes.push(format!("shot {} ({}): {} needs {min} beats, not {asked_beats}", k + 1, c.id, cam.id()));
        }
        // inside the media: shorter, else another camera, else out
        let mut from = None;
        while beats >= 1 {
            let need = f64::from(beats) * beat * cam.speed();
            let at = finite(&s["from"]).filter(|f| *f >= 0.0 && *f + need <= c.media_seconds + 1e-6 && c.kind != Kind::Photo);
            from = at.or_else(|| window_from(c, need));
            if from.is_some() || beats <= min {
                break;
            }
            beats -= 1;
        }
        let Some(from) = from.or_else(|| {
            if cam == Camera::Hyperlapse {
                cam = Camera::Drift;
                beats = beats_for(Camera::Drift.min_seconds(), beat);
                window_from(c, f64::from(beats) * beat)
            } else {
                None
            }
        }) else {
            notes.push(format!("shot {} ({}): the clip is too short for {} beats: left out", k + 1, c.id, beats));
            continue;
        };
        // the same moment of one clip, looking the same way, once
        let span = (from, from + f64::from(beats) * beat * cam.speed());
        if used_windows.iter().any(|(id, a, b, l)| *id == c.id && span.0 < *b && span.1 > *a && (c.kind != Kind::Vr || *l == look)) {
            notes.push(format!("shot {} ({}): the same moment again: left out", k + 1, c.id));
            continue;
        }
        used_windows.push((c.id.clone(), span.0, span.1, look.clone()));
        *cams.entry(cam).or_insert(0) += 1;
        let text = s["text"].as_str().map(str::trim).filter(|t| !t.is_empty()).map(|t| chars(t, MAX_TITLE_CHARS));
        let text = match text {
            Some(t) if !english(&t) => {
                notes.push(format!("shot {} ({}): its text is not English: left out", k + 1, c.id));
                None
            }
            t => t,
        };
        let enter = match s["enter"].as_str().unwrap_or("cut") {
            "whip" => Enter::Whip,
            "dissolve" => Enter::Dissolve,
            _ => Enter::Cut,
        };
        shots.push(PlannedShot {
            clip: c.id.clone(),
            from: round2(from),
            beats,
            camera: cam,
            look,
            to,
            enter,
            text,
            why: chars(s["why"].as_str().unwrap_or_default(), 160),
        });
    }
    if shots.is_empty() {
        return Err("none of the plan's shots could be used".into());
    }
    // entering shots: a whip only between two 360° shots, a dissolve only into the last
    let kinds: Vec<Kind> = shots.iter().map(|s| cands.iter().find(|c| c.id == s.clip).map_or(Kind::Flat, |c| c.kind)).collect();
    let n = shots.len();
    for i in 0..n {
        let prev = i.checked_sub(1).and_then(|j| kinds.get(j)).copied();
        let me = kinds.get(i).copied();
        if let Some(s) = shots.get_mut(i) {
            let whip_ok = prev == Some(Kind::Vr) && me == Some(Kind::Vr);
            let wrong = i == 0 || (s.enter == Enter::Whip && !whip_ok) || (s.enter == Enter::Dissolve && i + 1 != n);
            if wrong {
                s.enter = Enter::Cut;
            }
        }
    }
    // the length asked
    let mut sum: u32 = shots.iter().map(|s| s.beats).sum();
    while sum > total && shots.len() > 1 {
        // shorten the longest body shot that may be shorter, else drop the last body shot
        let mins: Vec<u32> = shots.iter().map(|s| beats_for(s.camera.min_seconds(), beat)).collect();
        let body = 1..shots.len().saturating_sub(1).max(1);
        let longest =
            body.clone().filter(|&i| shots.get(i).zip(mins.get(i)).is_some_and(|(s, m)| s.beats > *m)).max_by_key(|&i| shots.get(i).map_or(0, |s| s.beats));
        match longest {
            Some(i) => {
                if let Some(s) = shots.get_mut(i) {
                    s.beats -= 1;
                    sum -= 1;
                }
            }
            None => {
                let i = if shots.len() > 2 { shots.len() - 2 } else { shots.len() - 1 };
                let gone = shots.remove(i);
                sum = sum.saturating_sub(gone.beats);
                notes.push(format!("{} left out: the shots ran past {total} beats", gone.clip));
            }
        }
    }
    if sum > total
        && let Some(s) = shots.first_mut()
    {
        s.beats = s.beats.saturating_sub(sum - total).max(1);
        sum = total;
    }
    if sum < total {
        let mut k = 0usize;
        while sum < total && k < 512 {
            let i = k % shots.len();
            if let Some(s) = shots.get_mut(i)
                && let Some(c) = cands.iter().find(|c| c.id == s.clip)
            {
                let need = f64::from(s.beats + 1) * beat * s.camera.speed();
                if s.from + need <= c.media_seconds + 1e-6 || c.kind == Kind::Photo {
                    s.beats += 1;
                    sum += 1;
                }
            }
            k += 1;
        }
        if sum < total {
            notes.push(format!("{} beat(s) short: the clips the plan chose have no more footage", total - sum));
        }
    }
    let title = v["title"].as_str().map(str::trim).filter(|t| !t.is_empty() && english(t)).map(|t| chars(t, MAX_TITLE_CHARS));
    let title = match (&o.title, title) {
        (Some(t), _) => t.clone(),
        (None, Some(t)) => t,
        (None, None) => {
            notes.push("the plan's title was missing or not English: one from the brief".into());
            title_from_brief(&o.brief).unwrap_or_else(|| "Highlights".into())
        }
    };
    if let Some(first) = shots.first_mut()
        && first.text.is_none()
    {
        first.text = Some(title.clone());
    }
    let mood = o
        .mood
        .clone()
        .or_else(|| v["mood"].as_str().map(str::to_ascii_lowercase).filter(|m| MOODS.contains(&m.as_str())))
        .unwrap_or_else(|| mood_from_brief(&o.brief).to_string());
    Ok(Plan { title, mood, bpm, shots, by: String::new(), notes })
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
    // about as wide as the room at most: a long title is set smaller
    let n = text.chars().count().max(1) as f64;
    let size = (120.0 * u).min(room / (0.55 * n)).max(48.0 * u.min(1.0));
    ([(x0 + x1) / 2.0, y0 + 0.55 * (y1 - y0)], size, room)
}

/// The commands of a plan on `windows` (timeline seconds of each shot), for a `frame`: each
/// shot placed on V1 with its sound on A1, its camera set and moved, its way in; the title and
/// other text on V2; the swishes and the chime on A4. `beat` the music's beat, seconds.
pub fn compile(plan: &Plan, cands: &[Candidate], wins: &[(f64, f64)], o: &Options, frame: (f64, f64), beat: f64) -> Vec<Step> {
    let mut out: Vec<Step> = Vec::new();
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
        let look = s.look.as_deref().and_then(|l| c.look(l));
        match c.kind {
            Kind::Vr => {
                if s.camera == Camera::Hyperlapse {
                    out.push(step("clip.hyperlapse360", json!({"speed": speed * 100.0, "mode": "strong", "wait": true}), me, On::Clips));
                }
                if let Some(l) = look {
                    out.push(step("clip.reframe360", json!({"yaw": l.yaw.round(), "pitch": l.pitch.round()}), me, On::Clips));
                }
                if c.handheld && !matches!(s.camera, Camera::Hyperlapse | Camera::PlanetIn | Camera::PlanetOut) {
                    out.push(step("clip.stabilize360", json!({"mode": "smooth", "wait": true}), me, On::Clips));
                }
                // where the shot's own clip time is: the hyperlapse plays its media faster
                let at = |secs: f64| round2(secs.max(0.0));
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
                    out.push(step("clip.speedRamp", json!({"preset": "smoothIn", "fit": "length", "ripple": false}), me, On::Clips));
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
                match s.camera {
                    Camera::Follow => {
                        out.push(step("clip.fillFrame", json!({}), me, On::Clips));
                        let mut p = json!({"keepScale": true, "motion": "smooth", "fallback": "center", "analysisFps": 8, "wait": true});
                        if let Some(b) = c.subject {
                            p["subject"] = json!({"x": b[0], "y": b[1], "w": b[2], "h": b[3], "seconds": round2(c.key)});
                        }
                        out.push(step("clip.autoReframe", p, me, On::Clips));
                    }
                    Camera::Hold => out.push(step("clip.fillFrame", json!({}), me, On::Clips)),
                    _ => {
                        let mut p = json!({"move": "in", "fit": "fill"});
                        if let Some(b) = subject {
                            p["subject"] = b;
                        }
                        out.push(step("clip.kenBurns", p, me, On::Clips));
                    }
                }
                if s.camera == Camera::SpeedRamp {
                    out.push(step("clip.speedRamp", json!({"preset": "smoothIn", "fit": "length", "ripple": false}), me, On::Clips));
                }
            }
        }
        if s.enter == Enter::Dissolve && k > 0 {
            let frames = ((beat * FPS).round() as i64).clamp(8, 24);
            out.push(step("sequence.applyVideoTransition", json!({"edge": "in", "effect": "cross_dissolve", "frames": frames}), me, On::Clip));
        }
        // a swish under each whip and roll, quiet
        let swish = |at: f64| json!({"kind": "swish", "duration": 0.5, "time": ticks((at - 0.25).max(0.0)), "track": "A4", "intensity": 0.4, "seed": o.seed + k as u64});
        if s.enter == Enter::Whip {
            out.push(step("audio.generate", swish(t0), None, On::Nothing));
        }
        if s.camera == Camera::BarrelRoll {
            out.push(step("audio.generate", swish(t0 + len / 2.0), None, On::Nothing));
        }
        if s.camera == Camera::WhipPan {
            out.push(step("audio.generate", swish(t0 + len / 2.0), None, On::Nothing));
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
            if k == 0 {
                out.push(step(
                    "audio.generate",
                    json!({"kind": "chime", "duration": 1.2, "time": ticks(a), "track": "A4", "intensity": 0.3, "seed": o.seed}),
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
    pub shots_indexed: usize,
    /// The shots the director chose from: what each shows, and a 360° shot's looks.
    pub candidates: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<Plan>,
    pub commands: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
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
async fn gather(ex: &dyn Executor, p: &Value, notes: &mut Vec<String>, lines: &mut dyn FnMut(&str)) -> Result<Vec<MediaItem>, AutoCutError> {
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
    if let Ok(d) = run(ex, "media.captureDate", json!({"items": media.iter().map(|m| m.item).collect::<Vec<_>>()})).await {
        let when: HashMap<u64, i64> =
            d["items"].as_array().map(|a| a.iter().filter_map(|x| Some((x["item"].as_u64()?, x["unix"].as_i64()?))).collect()).unwrap_or_default();
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
    Ok(media)
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
        if let Ok(v) = run(ex, "media.scenes", json!({"item": m.item})).await
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
                        if let Ok(v) = run(ex, "media.scenes", json!({"item": item})).await
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
    let o = Options::from_params(p)?;
    let mut report = Report { brief: o.brief.clone(), aspect: o.aspect.clone(), seconds: o.seconds, models: json!({}), ..Report::default() };
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
    let media = gather(ex, p, &mut notes, lines).await?;
    report.media = media.len();
    let items: Vec<u64> = media.iter().map(|m| m.item).collect();
    let vr: BTreeSet<u64> = match run(ex, "media.vrProperties", json!({"items": items})).await {
        Ok(v) => {
            v["items"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter(|x| !x["vr"].is_null()).filter_map(|x| x["item"].as_u64()).collect()
        }
        Err(_) => BTreeSet::new(),
    };
    report.vr_media = vr.len();
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
    let cands = candidates(&shots, &media, &vr, &salience, &aspects, &speech);
    // ---- the plan
    let bpm = o.bpm.unwrap_or(DEFAULT_BPM);
    let beat = beat_of(bpm);
    let total = ((o.seconds / beat).round() as u32).max(4);
    let short = shortlist(&cands, MAX_CANDIDATES);
    report.candidates = short
        .iter()
        .map(|c| {
            let mut v = json!({"clip": c.id, "name": c.name, "kind": c.kind, "from": round2(c.start), "to": round2(c.end), "what": c.caption, "energy": round2(c.energy), "quality": round2(c.quality)});
            if c.kind == Kind::Vr {
                v["looks"] = json!(c.looks.iter().map(|l| json!({"look": l.id, "what": l.what, "yaw": l.yaw.round(), "pitch": l.pitch.round()})).collect::<Vec<_>>());
            }
            v
        })
        .collect();
    let mut plan = None;
    match model {
        Some(m) => {
            lines(&format!("directing with {}", m.id()));
            match model_plan(m, &short, &o, beat, total).await {
                Ok(v) => match validate(&v, &cands, &o, beat, bpm) {
                    Ok(mut pl) => {
                        pl.by = format!("model:{}", m.id());
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
    report.director = plan.by.clone();
    let (fw, fh) = o.frame();
    let frame = (f64::from(fw), f64::from(fh));
    // ---- dry run: the plan and its commands on the beat grid
    if o.dry_run {
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
    report.mix = Some(mix(ex, &built, music_clip, &plan, &cands, &mut notes).await);
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
        let placed = run(ex, "timeline.place", json!({"item": item, "audioTrack": "A3", "video": false, "time": 0, "duration": ticks(secs + 0.5)})).await;
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
    let p = json!({"kind": "music", "duration": secs + 0.5, "mood": plan.mood, "bpm": plan.bpm, "seed": o.seed, "time": 0, "track": "A3"});
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
    for (s, (t0, t1)) in plan.shots.iter().zip(wins) {
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        if s.camera == Camera::Hyperlapse || c.kind == Kind::Photo {
            continue;
        }
        let Some(sp) = speech.iter().find(|x| x.item == c.item && x.language.starts_with("en")) else { continue };
        let (a, b) = (s.from, s.from + (t1 - t0));
        for k in sp.kept.iter().filter(|k| k.end > a && k.start < b && !k.text.trim().is_empty()) {
            let start = t0 + (k.start - a).max(0.0);
            let end = (t0 + (k.end - a)).min(*t1);
            if end - start >= 0.4 && english(&k.text) {
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
async fn mix(ex: &dyn Executor, b: &Built, music: Option<u64>, plan: &Plan, cands: &[Candidate], notes: &mut Vec<String>) -> Value {
    let sounds: Vec<u64> = b.sounds.iter().flatten().copied().collect();
    let talking: Vec<u64> = plan
        .shots
        .iter()
        .zip(&b.sounds)
        .filter(|(s, _)| cands.iter().find(|c| c.id == s.clip).is_some_and(|c| c.talking || c.said.is_some()))
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
        step("essentialSound.autoMatch", json!({"clips": b.effects, "target": EFFECTS_LUFS}), notes).await;
        crate::mix::inserts(ex, "A4", &[("parametric_eq", crate::mix::effects_eq())], notes).await;
    }
    json!({"profile": "music", "music": BED_LUFS, "sounds": SOUND_LUFS, "effects": EFFECTS_LUFS})
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

#[cfg(test)]
#[path = "director_tests.rs"]
mod tests;
