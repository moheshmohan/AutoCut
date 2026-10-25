//! A trailer of a finished showcase (`autocut.trailer`): a short vertical video — 9:16 for
//! Instagram Reels by default, also 1:1 and 4:5 — that sends viewers to the full show on YouTube.
//!
//! **What it reads.** The show's plan (its entries: labels, kinds, dates, shots with their
//! analysed best moments, and each entry's `redact` targets) and the show's build report (where
//! each entry's shots played — the window of each clip — what the privacy pass hid, and the
//! narration's lines); the show's sequence for its narration clip and its privacy blur. Without a
//! plan the entries are read from the sequence itself ([`entries_from_sequence`]: the clips of the
//! picture track, named by the lower thirds over them, without dates).
//!
//! **The cut** ([`timing`]), every length a whole number of beats of the music:
//! 1. *Hook* (about 3 s): the most striking robot ([`striking`]: a humanoid or a robot dog, big in
//!    its picture, filmed upright; or `hook`, from `hookAt`), filling the frame under a big title —
//!    `"48 robots"` and `"2013 → 2025"`, from the show's robot count and years (or `title`). The
//!    show's first narration sentence plays over it (`hookVoice`, on when the show is narrated):
//!    its narration clip, cut at the pause after the first sentence (`media.pauses`), in the
//!    owner's voice. Its words show as Bold animated captions inside the platform's safe zone,
//!    timed from the narration's script (`captions: false` leaves them out, `captionStyle` picks
//!    another look or `plain`, `captionKeywords` has the local model colour keywords and add
//!    emoji; [`crate::caption_style`]); the export burns them in.
//! 2. *Montage*: `robots` robots (12–18 by default, as the length allows), chronological so the
//!    years build up — the earliest and the latest always, and in between the most striking of
//!    each stretch of the show ([`choose`]), one per robot (an "again" visit is the same robot) —
//!    each 1.3–2.4 s, the longer ones first, every cut on a beat. Each shot plays its entry's best
//!    moment inside the window the show played (or where `at` says). A robot the vertical crop
//!    holds fills the frame, followed by `clip.autoReframe` (seeded with the analysis' box); a
//!    wider one is shown whole in a [`band`] over a blurred copy of itself. Its name and year sit
//!    in a panel inside the platform's safe zone.
//! 3. *End card* (about 3 s) from the music's final hit: the channel's logo on a light badge,
//!    `endText` ("Full video on YouTube") and the channel's handle, over a blurred copy of the
//!    hook's footage, all inside the safe zone.
//!
//! **Sound.** Generated music (`music.suggest` → `audio.generate`, `mood` energetic, `bpm` 124, a
//! fixed `seed`), arranged to end on a bar with its final chord where the end card starts (a
//! swish into it, an impact on it); the hook voice over its opening; mixed speech first
//! ([`crate::mix::mix_down`]), the music brought up from a bed's level and limited on its track,
//! the voice's fader down a little, normalised to −14 LUFS, −1 dBTP.
//!
//! **Privacy.** What the show blurred stays blurred: the show's own blur — the masks its privacy
//! pass made, keyed in media time — is carried onto every trailer shot of footage it hid something
//! in (`carry_redaction`), and the trailer's clips get the show's privacy pass again
//! ([`crate::privacy`]: faces and plates — the report says whether the show hid any — and every
//! entry's named targets, each robot's own box left sharp) through the redaction cache, with a face
//! hidden at the size the show hid it ([`face_threshold`]: a portrait clip that filled the 16:9
//! show was three times larger there). QA says where the show hid faces in footage the trailer's
//! own pass hides none in.
//!
//! **Out.** The sequence (`<show> — Reel trailer`), and with `export`: the video (Social Vertical
//! 1080×1920 for 9:16: H.264, AAC 48 kHz, 12 Mb/s, GPU rendering as set), `<out>.cover.jpg` (the
//! hook's frame with its title), `<out>.caption.txt` (a caption with hashtags; "link in bio": a
//! Reel's caption cannot hold a link; `<out>.caption.json` with the link for the bio),
//! `<out>.frames.jpg` (16 frames at most, from the file: the hook, the shots, the end card) and QA
//! (`<out>.qa.json`: joins, black frames, black sides, the subject, privacy, the platform's safe
//! zones, the mix; plus the trailer's own: cuts on the beat, the text and logo in the safe zone,
//! the show's redaction carried over).
//!
//! Every edit is an engine command, so the trailer is an ordinary sequence in History.

use std::path::{Path, PathBuf};

use filmcraft_engine::safe_zones::Platform;
use serde::Serialize;
use serde_json::{Value, json};

use crate::caption_style;
use crate::host::ReviewEntry;
use crate::llm::ChatModel;
use crate::look::{self, Bounds, Role, Theme};
use crate::mix::{self, Mix, MixProfile};
use crate::privacy::{Hidden, LocalRedact, Redact};
use crate::qa::{self, Finding, Severity};
use crate::shot_edit::Fit;
use crate::showcase::{Beats, Entry, Moment, PlacedEntry, PlacedShot, Shot, on_frame, run, ticks};
use crate::subtitles::Cue;
use crate::{AutoCutError, Executor};

/// The trailer's usual, shortest and longest length, seconds.
pub const DEFAULT_SECONDS: f64 = 34.0;
pub const MIN_SECONDS: f64 = 10.0;
pub const MAX_SECONDS: f64 = 90.0;
/// The music's usual tempo, beats a minute.
pub const DEFAULT_BPM: f64 = 124.0;
/// A montage shot's shortest and longest, seconds.
pub const MIN_SHOT: f64 = 1.3;
pub const MAX_SHOT: f64 = 2.4;
/// The hook's and the end card's length aimed at, seconds (each a whole number of beats).
pub const HOOK_SECONDS: f64 = 3.0;
pub const END_CARD_SECONDS: f64 = 3.0;
/// Fewest and most robots in the montage.
pub const MIN_ROBOTS: usize = 3;
pub const MAX_ROBOTS: usize = 30;
/// The usual montage, as the length allows.
pub const AUTO_ROBOTS: (usize, usize) = (12, 18);
/// What the end card says by default.
pub const DEFAULT_END_TEXT: &str = "Full video on YouTube";
/// The trailer's frame rate.
pub const FPS: f64 = 30.0;
/// The export preset of a 9:16 trailer.
pub const VERTICAL_PRESET: &str = "Social Vertical 1080×1920";
/// The face a showcase hides: this share of its frame's height (`clip.autoRedact`'s default).
pub const SHOW_MIN_FACE: f64 = 0.05;
/// The most a showcase pushes in on a shot (its faces grow by that much: [`face_threshold`]).
pub const SHOW_PUSH: f64 = 1.06;
/// Room for error in [`face_threshold`]: a face a tenth smaller than the show's is hidden too.
const FACE_MARGIN: f64 = 0.9;
/// Most entries, shots and labels looked at.
const MAX_ENTRIES: usize = 2000;
const MAX_SHOTS: usize = 64;
/// Longest label shown.
const MAX_LABEL_CHARS: usize = 60;
/// The analysis' frame rate for the reframe (calm crops need few).
const REFRAME_FPS: f64 = 15.0;
/// How far the music comes up from the show's bed level (the speech mix's −24 LUFS), dB.
const MUSIC_LIFT_DB: f64 = 8.0;
/// The music track's own limiter (its peaks before the Mix), dBFS, and the voice's trim, dB.
const MUSIC_CEILING_DB: f64 = -3.5;
const VOICE_TRIM_DB: f64 = -3.5;

fn invalid(m: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(format!("trailer: {}", m.into()))
}

fn finite(v: f64, or: f64) -> f64 {
    if v.is_finite() { v } else { or }
}

/// What to make (`autocut.trailer`'s checked parameters).
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub sequence: Option<u64>,
    pub seconds: f64,
    /// The frame: 1080×1920 (9:16), 1080×1080 (1:1) or 1080×1350 (4:5).
    pub frame: (u32, u32),
    pub platform: Platform,
    pub robots: Option<usize>,
    pub hook: Option<String>,
    /// Where the hook starts in its clip, seconds (default: around its best moment).
    pub hook_at: Option<f64>,
    /// Where a robot's shot starts: its label, the clip (an item of its entry; none: its own) and
    /// the second of it.
    pub at: Vec<At>,
    pub title: Option<String>,
    /// The show's first narration sentence over the hook; `None`: when the show has narration.
    pub hook_voice: Option<bool>,
    pub logo: Option<String>,
    pub channel: Option<String>,
    pub end_text: String,
    pub link: Option<String>,
    pub exclude: Vec<String>,
    pub include: Vec<String>,
    pub bpm: f64,
    pub mood: String,
    pub seed: u64,
    /// `None`: as the show (faces and plates when its report hid any); `Some(None)`: none;
    /// `Some(Some(r))`: these kinds.
    pub redact: Option<Option<Redact>>,
    pub fallback: String,
    pub export: Option<String>,
    pub preset: Option<String>,
    pub qa: Option<bool>,
    /// The hook voice's line as captions; `None`: when there is a hook voice.
    pub captions: Option<bool>,
    /// How the captions look (Bold animated by default; [`caption_style::Look`]).
    pub caption_look: caption_style::Look,
    /// The local model colours keywords and adds emoji to the captions.
    pub caption_keywords: bool,
    /// An opt-in creator look over the hook and the montage (`look`: tech, vhs, clean).
    pub look: Option<crate::creator_look::CreatorLook>,
    /// Zoom pulses on the montage's shots at every other beat (`beatPulse`).
    pub beat_pulse: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sequence: None,
            seconds: DEFAULT_SECONDS,
            frame: (1080, 1920),
            platform: Platform::Reels,
            robots: None,
            hook: None,
            hook_at: None,
            at: Vec::new(),
            title: None,
            hook_voice: None,
            logo: None,
            channel: None,
            end_text: DEFAULT_END_TEXT.into(),
            link: None,
            exclude: Vec::new(),
            include: Vec::new(),
            bpm: DEFAULT_BPM,
            mood: "energetic".into(),
            seed: 7,
            redact: None,
            fallback: "blur".into(),
            export: None,
            preset: None,
            qa: None,
            captions: None,
            caption_look: caption_style::Look::Animated("bold".into()),
            caption_keywords: false,
            look: None,
            beat_pulse: false,
        }
    }
}

/// Where a robot's shot starts (`at`): the robot's entry label, the clip — another item of its
/// entry, or its own when `None` — and the second of it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct At {
    pub label: String,
    pub item: Option<u64>,
    pub seconds: f64,
}

/// `at` from the parameters: `{"<label>": seconds | {"item": id, "seconds": s}}` (at most a
/// hundred; anything else left out).
pub fn at_of(v: &Value) -> Vec<At> {
    let Some(o) = v.as_object() else { return Vec::new() };
    o.iter()
        .take(100)
        .filter_map(|(label, x)| {
            let label = label.trim();
            if label.is_empty() || label.chars().count() > 200 || label.chars().any(char::is_control) {
                return None;
            }
            let (item, seconds) = match x {
                Value::Number(n) => (None, n.as_f64()?),
                Value::Object(_) => (x["item"].as_u64(), x["seconds"].as_f64()?),
                _ => return None,
            };
            (seconds.is_finite() && seconds >= 0.0).then(|| At { label: label.to_string(), item, seconds })
        })
        .collect()
}

/// The frame of an aspect: `9:16`, `1:1` or `4:5`, 1080 wide.
pub fn frame_of(aspect: &str) -> Option<(u32, u32)> {
    match aspect.trim() {
        "9:16" => Some((1080, 1920)),
        "1:1" => Some((1080, 1080)),
        "4:5" => Some((1080, 1350)),
        _ => None,
    }
}

impl Options {
    /// From `autocut.trailer`'s parameters (checked by the engine; checked again here, so a
    /// direct caller cannot slip past it).
    pub fn from_params(p: &Value) -> Result<Self, AutoCutError> {
        let d = Self::default();
        let text = |k: &str| p[k].as_str().map(str::trim).filter(|t| !t.is_empty() && !t.chars().any(char::is_control)).map(str::to_string);
        let labels = |k: &str| -> Vec<String> {
            p[k].as_array()
                .map(|a| a.iter().filter_map(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).take(100).collect())
                .unwrap_or_default()
        };
        let seconds = p["seconds"].as_f64().map_or(Ok(d.seconds), |s| {
            if s.is_finite() && (MIN_SECONDS..=MAX_SECONDS).contains(&s) { Ok(s) } else { Err(invalid(format!("seconds: {MIN_SECONDS} to {MAX_SECONDS}"))) }
        })?;
        let aspect = p["aspect"].as_str().unwrap_or("9:16");
        let frame = frame_of(aspect).ok_or_else(|| invalid("aspect: 9:16, 1:1 or 4:5"))?;
        let platform = match p["platform"].as_str() {
            Some(n) => Platform::from_name(n).ok_or_else(|| invalid("platform: reels, youtube, tiktok or vertical"))?,
            None => Platform::Reels,
        };
        let robots = match p["robots"].as_u64() {
            Some(n) if (MIN_ROBOTS as u64..=MAX_ROBOTS as u64).contains(&n) => Some(n as usize),
            Some(_) => return Err(invalid(format!("robots: {MIN_ROBOTS} to {MAX_ROBOTS}"))),
            None => None,
        };
        let bpm = p["bpm"].as_f64().map_or(Ok(d.bpm), |b| if b.is_finite() && (80.0..=180.0).contains(&b) { Ok(b) } else { Err(invalid("bpm: 80 to 180")) })?;
        let redact = match &p["redact"] {
            Value::Null => None,
            Value::Bool(false) => Some(None),
            Value::String(s) => Some(Some(Redact::parse(s)?)),
            _ => return Err(invalid("redact: \"faces,plates\", \"faces\", \"plates\" or false")),
        };
        Ok(Self {
            sequence: p["sequence"].as_u64(),
            seconds,
            frame,
            platform,
            robots,
            hook: text("hook"),
            hook_at: p["hookAt"].as_f64().filter(|v| v.is_finite() && *v >= 0.0),
            at: at_of(&p["at"]),
            title: text("title"),
            hook_voice: p["hookVoice"].as_bool(),
            logo: text("logo"),
            channel: text("channel"),
            end_text: text("endText").unwrap_or(d.end_text),
            link: text("link"),
            exclude: labels("exclude"),
            include: labels("include"),
            bpm,
            mood: text("mood").unwrap_or(d.mood),
            seed: p["seed"].as_u64().unwrap_or(d.seed),
            redact,
            fallback: text("fallback").unwrap_or(d.fallback),
            export: text("export"),
            preset: text("preset"),
            qa: p["qa"].as_bool(),
            captions: p["captions"].as_bool(),
            caption_look: caption_style::Look::of(p),
            caption_keywords: p["captionKeywords"].as_bool() == Some(true),
            look: crate::creator_look::look_of(p)?,
            beat_pulse: p["beatPulse"].as_bool().unwrap_or(false),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The show

/// What the trailer knows of the show.
#[derive(Clone, Debug, Default)]
pub struct Show {
    /// The show's sequence.
    pub sequence: Option<u64>,
    /// Its name ("Every Robot I've Met").
    pub name: String,
    /// Its entries, as shown (the review's left-out and merged ones taken away).
    pub entries: Vec<Entry>,
    /// Where each entry's shots played ([`crate::showcase::Built::layout`]).
    pub layout: Vec<PlacedEntry>,
    /// What its privacy pass hid, clip by clip.
    pub privacy: Vec<Hidden>,
    /// Its narration lines as placed.
    pub narration: Vec<Cue>,
    /// The plan's look (`style`) and branded outro (`outro`), for the trailer's fonts and its end
    /// card's logo and handle.
    pub style: Value,
    pub outro: Value,
    /// The plan's mix profile.
    pub mix: Value,
    /// The report says the show was redacted (faces or plates hidden).
    pub redacted: bool,
    /// The entries were read from the sequence (no plan).
    pub from_sequence: bool,
    pub notes: Vec<String>,
}

impl Show {
    /// From the show's plan and build report (either may be missing).
    pub fn from_docs(plan: Option<&Value>, report: Option<&Value>) -> Self {
        let mut show = Show::default();
        if let Some(plan) = plan {
            let list = plan["entries"].as_array().map(Vec::as_slice).unwrap_or_default();
            for (i, e) in list.iter().take(MAX_ENTRIES).enumerate() {
                match serde_json::from_value::<ReviewEntry>(e.clone()) {
                    Ok(r) if r.include && !r.merge_previous => show.entries.push(r.entry),
                    Ok(_) => {}
                    Err(err) => show.notes.push(format!("plan: entry {}: {err}", i + 1)),
                }
            }
            show.name = plan["title"].as_str().unwrap_or_default().trim().to_string();
            show.style = plan["style"].clone();
            show.outro = plan["outro"].clone();
            show.mix = plan["mix"].clone();
        }
        if let Some(r) = report {
            show.layout =
                r["layout"].as_array().map(|a| a.iter().take(MAX_ENTRIES).filter_map(|e| serde_json::from_value(e.clone()).ok()).collect()).unwrap_or_default();
            show.privacy =
                r["privacy"].as_array().map(|a| a.iter().take(100_000).filter_map(|e| serde_json::from_value(e.clone()).ok()).collect()).unwrap_or_default();
            show.narration =
                r["narration"].as_array().map(|a| a.iter().take(10_000).filter_map(|e| serde_json::from_value(e.clone()).ok()).collect()).unwrap_or_default();
            show.sequence = r["sequence"].as_u64().or_else(|| r["loudness"]["sequence"].as_u64());
            show.redacted = show.privacy.iter().any(|h| h.faces > 0 || h.plates > 0);
        }
        show
    }

    /// The robots it counts (its entries that are not scenes).
    pub fn robots(&self) -> usize {
        self.entries.iter().filter(|e| !e.is_scene()).count()
    }

    /// The first and last year of its entries.
    pub fn years(&self) -> Option<(i64, i64)> {
        let ys: Vec<i64> = self.entries.iter().filter_map(|e| e.unix).map(|u| crate::showcase::ymd(u).0).collect();
        Some((*ys.iter().min()?, *ys.iter().max()?))
    }
}

/// The text of the graphic clip over the start of a clip on another track: (start, end, name).
type Overlay = (i64, i64, String);

/// Entries read from a show's sequence when there is no plan (`sequence.inspect` of it): each run
/// of clips on its picture track (V2, the showcase's; else the video track with most clips) under
/// one lower third (a graphic on V3 named after its robot) is an entry, in timeline order, without
/// dates or kinds.
pub fn entries_from_sequence(seq: &Value) -> (Vec<Entry>, Vec<PlacedEntry>) {
    const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;
    let tracks = seq["video"].as_array().map(Vec::as_slice).unwrap_or_default();
    let clips_of = |t: &Value| t["items"].as_array().map(Vec::as_slice).unwrap_or_default().to_vec();
    let picture = match tracks.get(1).filter(|t| !clips_of(t).is_empty()) {
        Some(t) => clips_of(t),
        None => tracks.iter().take(256).map(clips_of).max_by_key(Vec::len).unwrap_or_default(),
    };
    let overlays: Vec<Overlay> = tracks
        .get(2)
        .map(clips_of)
        .unwrap_or_default()
        .iter()
        .filter_map(|c| Some((c["start"].as_i64()?, c["end"].as_i64()?, c["name"].as_str()?.trim().to_string())))
        .filter(|(_, _, n)| !n.is_empty())
        .collect();
    let mut entries: Vec<Entry> = Vec::new();
    let mut layout: Vec<PlacedEntry> = Vec::new();
    for c in picture.iter().take(MAX_ENTRIES) {
        let (Some(clip), Some(item), Some(start), Some(end)) = (c["clip"].as_u64(), c["item"].as_u64(), c["start"].as_i64(), c["end"].as_i64()) else {
            continue;
        };
        if end <= start {
            continue;
        }
        let (si, so) = (c["sourceIn"].as_i64().unwrap_or(0) as f64 / TPS, c["sourceOut"].as_i64().unwrap_or(0) as f64 / TPS);
        if !(si.is_finite() && so.is_finite() && so > si) {
            continue;
        }
        let name = c["name"].as_str().unwrap_or_default().to_string();
        let label = overlays.iter().find(|(a, b, _)| (*a >= start && *a < end) || (*a <= start && *b > start)).map(|o| o.2.clone());
        let same = label.is_some() && entries.last().is_some_and(|e| Some(&e.label) == label.as_ref());
        let shot = Shot { item, name: name.clone(), clip_seconds: so, best: (si + so) / 2.0, from: Some(si), to: Some(so), ..Shot::default() };
        let placed = PlacedShot { clip, item, start: start as f64 / TPS, end: end as f64 / TPS, source: Some([si, so]), ..PlacedShot::default() };
        if same {
            if let (Some(e), Some(p)) = (entries.last_mut(), layout.last_mut()) {
                e.shots.push(shot);
                p.shots.push(placed);
                p.end = end as f64 / TPS;
            }
            continue;
        }
        let number = entries.len() + 1;
        let label = label.unwrap_or_else(|| name.rsplit_once('.').map_or(name.clone(), |(stem, _)| stem.to_string()));
        entries.push(Entry { number, label: label.clone(), shots: vec![shot], ..Entry::default() });
        layout.push(PlacedEntry { number, label, start: start as f64 / TPS, end: end as f64 / TPS, shots: vec![placed], ..PlacedEntry::default() });
    }
    (entries, layout)
}

// ---------------------------------------------------------------------------------------------
// Picking the robots

/// One shot a robot can be shown in: a clip, the stretch of it the show played and its best
/// moment there.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ShotPick {
    pub item: u64,
    pub name: String,
    pub clip_seconds: f64,
    /// The stretch of the clip the show played, seconds (the whole clip when not known).
    pub window: [f64; 2],
    /// The analysis' moment where the robot is largest, when it falls in the window.
    pub moment: Option<Moment>,
    /// A photo, not a video.
    pub photo: bool,
    /// The show's clip of it, and how it filled the show's frame.
    pub show_clip: Option<u64>,
    pub show_fit: Option<Fit>,
}

/// A robot the trailer can show.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    pub number: usize,
    pub label: String,
    pub kind: String,
    pub unix: Option<i64>,
    /// How striking it is ([`striking`]).
    pub score: f64,
    /// Its entry hides named things (a sponsor's logo): left for last.
    pub branded: bool,
    pub shot: ShotPick,
}

/// How much a kind of robot catches the eye: a face (humanoids), then a robot dog, then arms and
/// vehicles.
pub fn kind_weight(kind: &str) -> f64 {
    let k = kind.to_ascii_lowercase();
    if k.contains("humanoid") || k.contains("android") {
        1.0
    } else if k.contains("quadruped") || k.contains("dog") {
        0.9
    } else if k.contains("arm") {
        0.6
    } else if k.contains("patrol") || k.contains("vehicle") || k.contains("delivery") || k.contains("service") {
        0.55
    } else {
        0.45
    }
}

/// How striking a shot is: its kind ([`kind_weight`]), how big the robot is in its picture (up to
/// 0.8 for one that fills it), filmed upright (0.25: it fills a vertical frame), less for a photo
/// and much less for an entry that hides a sponsor's logo.
pub fn striking(kind: &str, area: f64, portrait: bool, photo: bool, branded: bool) -> f64 {
    let area = if area.is_finite() { area.clamp(0.0, 1.0) } else { 0.0 };
    let s = kind_weight(kind) + 0.8 * area + if portrait { 0.25 } else { 0.0 } - if photo { 0.2 } else { 0.0 } - if branded { 0.8 } else { 0.0 };
    (s * 1000.0).round() / 1000.0
}

/// The robots of `entries` the trailer can show, each in its best shot: the one (of the shots the
/// show played, `layout`) whose window holds the analysis' largest view of it; scored without
/// knowing which are portrait ([`striking`]; the caller adds that).
pub fn candidates(entries: &[Entry], layout: &[PlacedEntry]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for e in entries.iter().filter(|e| !e.is_scene()).take(MAX_ENTRIES) {
        let placed = layout.iter().find(|p| p.number == e.number);
        let mut options: Vec<ShotPick> = Vec::new();
        for ps in placed.map(|p| p.shots.as_slice()).unwrap_or_default().iter().take(MAX_SHOTS) {
            let shot = e.shots.iter().find(|s| s.item == ps.item);
            options.extend(pick_of(shot, ps.item, ps.source, Some(ps.clip), ps.fit, ps.photo));
        }
        if options.is_empty() {
            for s in e.shots.iter().take(MAX_SHOTS) {
                let w = match (s.from, s.to) {
                    (Some(a), Some(b)) => Some([a, b]),
                    _ => None,
                };
                options.extend(pick_of(Some(s), s.item, w, None, None, s.photo));
            }
        }
        let area = |o: &ShotPick| o.moment.map_or(0.0, |m| finite(m.area, 0.0));
        let Some(best) = options.iter().enumerate().max_by(|a, b| area(a.1).total_cmp(&area(b.1)).then(b.0.cmp(&a.0))).map(|(_, o)| o.clone()) else {
            continue;
        };
        let local = LocalRedact::from_plan(e.redact.as_ref()).unwrap_or_default();
        let shot_targets = e.shots.iter().any(|s| LocalRedact::from_plan(s.redact.as_ref()).is_ok_and(|r| !r.targets.is_empty()));
        let branded = !local.targets.is_empty() || shot_targets;
        let score = striking(&e.kind, area(&best), false, best.photo, branded);
        out.push(Candidate { number: e.number, label: display_label(&e.label), kind: e.kind.clone(), unix: e.unix, score, branded, shot: best });
    }
    out
}

/// A label as shown: one line, at most [`MAX_LABEL_CHARS`].
fn display_label(s: &str) -> String {
    let one: String = s.split(char::is_control).next().unwrap_or_default().trim().chars().take(MAX_LABEL_CHARS).collect();
    one.trim().to_string()
}

/// A shot pick from a plan shot (may be missing) of `item`, the window the show played.
fn pick_of(shot: Option<&Shot>, item: u64, window: Option<[f64; 2]>, show_clip: Option<u64>, show_fit: Option<Fit>, photo: bool) -> Option<ShotPick> {
    let photo = photo || shot.is_some_and(|s| s.photo);
    let clip_seconds = shot.map_or(0.0, |s| finite(s.clip_seconds, 0.0)).max(0.0);
    let ok = |w: [f64; 2]| w[0].is_finite() && w[1].is_finite() && w[1] > w[0] && w[0] >= 0.0;
    let window = match window.filter(|w| ok(*w)) {
        Some(w) => w,
        None if clip_seconds > 0.4 => [0.2, clip_seconds - 0.2],
        None => return None,
    };
    let clip_seconds = if clip_seconds > 0.0 { clip_seconds.max(window[1]) } else { window[1] };
    let moment =
        shot.and_then(|s| s.largest).filter(|m| m.seconds.is_finite() && m.area.is_finite() && m.seconds >= window[0] - 0.25 && m.seconds <= window[1] + 0.25);
    Some(ShotPick { item, name: shot.map(|s| s.name.clone()).unwrap_or_default(), clip_seconds, window, moment, photo, show_clip, show_fit })
}

/// The shot of `item` in entry `e`: the window the show played of it (`layout`), else the plan's.
fn pick_for(e: &Entry, layout: &[PlacedEntry], item: u64) -> Option<ShotPick> {
    let shot = e.shots.iter().find(|s| s.item == item);
    if let Some(ps) = layout.iter().find(|p| p.number == e.number).and_then(|p| p.shots.iter().find(|s| s.item == item)) {
        return pick_of(shot, item, ps.source, Some(ps.clip), ps.fit, ps.photo);
    }
    let s = shot?;
    pick_of(Some(s), item, s.from.zip(s.to).map(|(a, b)| [a, b]), None, None, s.photo)
}

/// A robot's name as a caption gives it: its label without ", again".
fn shown_name(label: &str) -> String {
    let l = label.trim();
    let l = l.strip_suffix(", again").or_else(|| l.strip_suffix(" again")).unwrap_or(l);
    l.trim().trim_end_matches(',').trim().to_string()
}

/// A robot's name for telling visits apart: its label without ", again" (and without case).
fn robot_key(label: &str) -> String {
    let l = label.trim().to_lowercase();
    let l = l.strip_suffix(", again").or_else(|| l.strip_suffix(" again")).unwrap_or(&l);
    l.trim().trim_end_matches(',').trim().to_string()
}

/// Two labels name the same robot: the same key, or one ends with the other as a whole word
/// ("Boston Dynamics Spot" and "Spot, again").
pub fn same_robot(a: &str, b: &str) -> bool {
    let (ka, kb) = (robot_key(a), robot_key(b));
    !ka.is_empty() && !kb.is_empty() && (ka == kb || ka.ends_with(&format!(" {kb}")) || kb.ends_with(&format!(" {ka}")))
}

fn matches_label(c: &Candidate, label: &str) -> bool {
    let l = label.trim();
    c.label.eq_ignore_ascii_case(l) || l.strip_prefix('#').and_then(|n| n.parse::<usize>().ok()) == Some(c.number)
}

/// The hook and the montage, as indices into `cands`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Choice {
    pub hook: usize,
    pub montage: Vec<usize>,
}

/// The hook — `hook` (an entry label, or `#number`), else the most striking robot — and `n`
/// robots for the montage, in date order: neither the hook's robot nor any robot twice ("Pepper,
/// again"), none of `exclude`; the earliest and the latest, every one of `include`, and in
/// between the most striking of each stretch of the show (the robots in date order cut into `n`
/// even stretches), a robot with a hidden sponsor's logo only when nothing else is left.
pub fn choose(cands: &[Candidate], n: usize, hook: Option<&str>, include: &[String], exclude: &[String]) -> Result<Choice, AutoCutError> {
    // a clip too short for a shot (the hook's is the longest) is not shown
    let long_enough = |c: &Candidate, len: f64| c.shot.photo || c.shot.clip_seconds >= len + 0.15;
    let excluded = |c: &Candidate| exclude.iter().any(|x| matches_label(c, x) || same_robot(&c.label, x)) || !long_enough(c, MAX_SHOT);
    for want in hook.into_iter().chain(include.iter().map(String::as_str)) {
        if !cands.iter().any(|c| matches_label(c, want)) {
            let names: Vec<&str> = cands.iter().take(80).map(|c| c.label.as_str()).collect();
            return Err(invalid(format!("no robot is called \"{}\" (the robots: {})", want.chars().take(60).collect::<String>(), names.join(", "))));
        }
    }
    let hook_len = HOOK_SECONDS * 1.2;
    let hook = match hook {
        Some(h) => {
            let i = cands.iter().position(|c| matches_label(c, h));
            if let Some(c) = i.and_then(|i| cands.get(i)).filter(|c| !long_enough(c, hook_len)) {
                return Err(invalid(format!("\"{}\"'s clip is {:.1} s, too short for the hook", c.label, c.shot.clip_seconds)));
            }
            i
        }
        None => cands
            .iter()
            .enumerate()
            .filter(|(_, c)| !excluded(c) && long_enough(c, hook_len))
            .max_by(|a, b| a.1.score.total_cmp(&b.1.score).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i),
    }
    .ok_or_else(|| invalid("no robot to show: every entry is a scene, left out or without footage long enough"))?;
    let hook_label = cands.get(hook).map(|c| c.label.clone()).unwrap_or_default();
    // the rest, in date order (undated ones where the plan has them)
    let mut pool: Vec<usize> =
        (0..cands.len()).filter(|&i| i != hook).filter(|&i| cands.get(i).is_some_and(|c| !excluded(c) && !same_robot(&c.label, &hook_label))).collect();
    pool.sort_by_key(|&i| cands.get(i).and_then(|c| c.unix).unwrap_or(i64::MAX));
    let n = n.min(pool.len());
    let mut chosen: Vec<usize> = Vec::new();
    let taken =
        |chosen: &[usize], i: usize| chosen.iter().any(|&k| k == i || cands.get(k).zip(cands.get(i)).is_some_and(|(a, b)| same_robot(&a.label, &b.label)));
    // the must-haves: the first and the last robot, and the ones asked for
    let mut must: Vec<usize> = Vec::new();
    if n >= 2 {
        must.extend(pool.first().copied());
        must.extend(pool.last().copied());
    }
    for want in include {
        must.extend(pool.iter().copied().find(|&i| cands.get(i).is_some_and(|c| matches_label(c, want))));
    }
    for i in must {
        if chosen.len() < n && !taken(&chosen, i) {
            chosen.push(i);
        }
    }
    // the most striking of each stretch that has no robot yet
    let len = pool.len();
    for k in 0..n {
        if chosen.len() >= n {
            break;
        }
        let (a, b) = (k * len / n.max(1), ((k + 1) * len / n.max(1)).max(k * len / n.max(1) + 1));
        let Some(stretch) = pool.get(a..b.min(len)) else { continue };
        if stretch.iter().any(|i| chosen.contains(i)) {
            continue;
        }
        let best = stretch
            .iter()
            .copied()
            .filter(|&i| !taken(&chosen, i))
            .max_by(|&x, &y| cands.get(x).map_or(0.0, |c| c.score).total_cmp(&cands.get(y).map_or(0.0, |c| c.score)).then(y.cmp(&x)));
        if let Some(i) = best {
            chosen.push(i);
        }
    }
    // stretches left empty (every robot of one was seen already): the most striking left
    let mut rest: Vec<usize> = pool.iter().copied().filter(|&i| !taken(&chosen, i)).collect();
    rest.sort_by(|&x, &y| cands.get(y).map_or(0.0, |c| c.score).total_cmp(&cands.get(x).map_or(0.0, |c| c.score)).then(x.cmp(&y)));
    for i in rest {
        if chosen.len() >= n {
            break;
        }
        if !taken(&chosen, i) {
            chosen.push(i);
        }
    }
    chosen.sort_by_key(|&i| (cands.get(i).and_then(|c| c.unix).unwrap_or(i64::MAX), pool.iter().position(|&p| p == i).unwrap_or(usize::MAX)));
    Ok(Choice { hook, montage: chosen })
}

// ---------------------------------------------------------------------------------------------
// The cut, on the beat

/// Where everything goes, seconds of the timeline: the hook, each montage shot and the end card,
/// every cut on a beat of the music, the end card from its final hit (a bar).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Timing {
    /// Seconds a beat.
    pub beat: f64,
    pub hook: [f64; 2],
    pub shots: Vec<[f64; 2]>,
    pub end_card: [f64; 2],
    pub total: f64,
    /// Beats in the hook and in each shot.
    pub hook_beats: usize,
    pub shot_beats: Vec<usize>,
}

impl Timing {
    /// Every cut: the hook's end and each shot's end (the last is the end card's start).
    pub fn cuts(&self) -> Vec<f64> {
        std::iter::once(self.hook[1]).chain(self.shots.iter().map(|s| s[1])).collect()
    }
}

/// The longest a montage shot may be at `beat` seconds a beat in a `montage`-second montage:
/// [`MAX_SHOT`], or the whole beats that let [`MAX_ROBOTS`] shots fill a long trailer.
pub fn max_shot(beat: f64, montage: f64) -> f64 {
    let m = if montage.is_finite() { montage.clamp(0.0, MAX_SECONDS) } else { 0.0 };
    let need = (m / (MAX_ROBOTS as f64 * beat)).ceil() * beat;
    MAX_SHOT.max(if need.is_finite() { need } else { 0.0 })
}

/// The beats a shot may last at `beat` seconds a beat in a `montage`-second montage:
/// [`MIN_SHOT`]..[`max_shot`] (at least one).
fn beat_range(beat: f64, montage: f64) -> (usize, usize) {
    let lo = (MIN_SHOT / beat - 1e-6).ceil().max(1.0) as usize;
    let hi = (max_shot(beat, montage) / beat + 1e-6).floor().max(1.0) as usize;
    (lo.min(hi), hi.max(lo))
}

/// The end card's length at `beat`: about [`END_CARD_SECONDS`], so that a generated bed `hit +
/// it` long puts its final hit at `hit` (it rings out at least two beats, 1.2–3 s, and less
/// than a bar more).
fn end_card_length(beat: f64) -> f64 {
    let tail_min = (2.0 * beat).clamp(1.2, 3.0);
    let bar = 4.0 * beat;
    END_CARD_SECONDS.clamp(tail_min + 1.0 / FPS, (bar + tail_min - 2.0 / FPS).max(tail_min + 1.0 / FPS))
}

/// How many robots a `seconds`-long trailer at `beat` seconds a beat shows: `want`, else
/// [`AUTO_ROBOTS`] as the length allows; never more than `available` or than fit.
pub fn shot_count(beat: f64, seconds: f64, want: Option<usize>, available: usize) -> usize {
    let beat = if beat.is_finite() && beat > 0.05 { beat.min(3.0) } else { 60.0 / DEFAULT_BPM };
    let montage = (finite(seconds, DEFAULT_SECONDS).clamp(MIN_SECONDS, MAX_SECONDS) - HOOK_SECONDS - END_CARD_SECONDS).max(MIN_SHOT);
    let (lo, hi) = beat_range(beat, montage);
    let beats = (montage / beat).round().max(1.0) as usize;
    let (fewest, most) = (beats.div_ceil(hi).max(1), (beats / lo).max(1));
    // a count asked for is kept (the length follows it); else as many as fill the time
    let n = match want {
        Some(n) => n,
        None => ((beats as f64 / ((lo + hi) as f64 / 2.0)).round() as usize).clamp(AUTO_ROBOTS.0, AUTO_ROBOTS.1).clamp(fewest, most.max(fewest)),
    };
    n.clamp(1, MAX_ROBOTS).min(available.max(1))
}

/// The cut of a `seconds`-long trailer with `n` montage shots on `beats` (the music's, from 0):
/// the hook about [`HOOK_SECONDS`], then the shots — each 1.3–2.4 s, the longer ones first so the
/// cutting quickens — ending on a bar (the bed's final hit), then the end card. Every boundary is
/// a beat (to the frame). `n` is the montage's size (fewer beats when it is too small to fill the
/// time).
pub fn timing(beats: &Beats, seconds: f64, n: usize) -> Result<Timing, AutoCutError> {
    let beat = beats.period();
    if !(beat.is_finite() && beat > 0.05) || n == 0 {
        return Err(invalid("no beat to cut on, or no robot to show"));
    }
    let ideal_hook = ((HOOK_SECONDS / beat).round() as usize).max(2);
    let end = end_card_length(beat);
    let seconds = finite(seconds, DEFAULT_SECONDS).clamp(MIN_SECONDS, MAX_SECONDS);
    let (lo, hi) = beat_range(beat, seconds - HOOK_SECONDS - END_CARD_SECONDS);
    let want = ((seconds - end) / beat).round().max(0.0) as usize;
    // the hit on a bar: hook and montage a whole number of bars, as near the length asked for as
    // the shots allow — the hook a beat or two longer or shorter if it must
    let bars_for = |hook: usize| -> Option<usize> {
        let (a, b) = ((hook + n * lo).div_ceil(4), (hook + n * hi) / 4);
        (a <= b).then(|| ((want as f64 / 4.0).round() as usize).clamp(a, b) * 4)
    };
    let mut hooks = [0i64, -1, 1, -2, 2].map(|d| ideal_hook as i64 + d).into_iter().filter(|h| *h >= 2).filter_map(|h| usize::try_from(h).ok());
    let (hook_beats, total_beats) = match hooks.find_map(|h| bars_for(h).map(|t| (h, t))) {
        Some(found) => found,
        // no whole number of bars fits (a montage of one or two): the hit on a beat
        None => (ideal_hook, want.clamp(ideal_hook + n * lo, ideal_hook + n * hi)),
    };
    let montage = total_beats.saturating_sub(hook_beats);
    let (q, r) = (montage / n, montage % n);
    let shot_beats: Vec<usize> = (0..n).map(|i| if i < r { q + 1 } else { q }).collect();
    let times = beats.list(0.0, (total_beats as f64 + 8.0) * beat + 10.0);
    let at = |k: usize| -> Result<f64, AutoCutError> { times.get(k).copied().map(|t| on_frame(t, FPS)).ok_or_else(|| invalid("the music has too few beats")) };
    let hook = [0.0, at(hook_beats)?];
    let mut shots = Vec::with_capacity(n);
    let mut k = hook_beats;
    for b in &shot_beats {
        shots.push([at(k)?, at(k + b)?]);
        k += b;
    }
    let hit = at(k)?;
    let total = on_frame(hit + end, FPS);
    Ok(Timing { beat, hook, shots, end_card: [hit, total], total, hook_beats, shot_beats })
}

/// Where a shot of `len` seconds starts in its clip: around its best moment (a little before it,
/// so the robot is seen arriving at its best), inside the window the show played when that is long
/// enough, else inside the clip. `(start, inside the show's window)`; `None` when the clip is
/// shorter than `len`.
pub fn window_start(pick: &ShotPick, len: f64) -> Option<(f64, bool)> {
    if !(len.is_finite() && len > 0.0) {
        return None;
    }
    if pick.photo {
        return Some((0.0, true));
    }
    let [a, b] = pick.window;
    let centre = pick.moment.map_or((a + b) / 2.0, |m| m.seconds);
    let start = centre - 0.45 * len;
    if b - a >= len {
        return Some((start.clamp(a, b - len), true));
    }
    let (lo, hi) = (0.05, pick.clip_seconds - 0.05 - len);
    if hi < lo {
        return None;
    }
    // as much of the show's window as it holds, reaching past it on both sides
    let s = (a - (len - (b - a)) / 2.0).clamp(lo, hi);
    Some((s, false))
}

/// A shot shown in a band: its picture `width` pixels wide (its height `width` / the picture's
/// shape), centred on `centre`, over a blurred copy of itself filling the frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Band {
    pub width: f64,
    pub centre: [f64; 2],
}

/// A robot this much narrower than what a filling crop shows is followed by the crop.
pub const CROP_ROOM: f64 = 0.8;
/// The robot fills this share of the frame's width in a band.
pub const BAND_FILL: f64 = 0.9;

/// How a montage shot of a picture `aspect` wide (width over height) is shown in a `frame`:
/// `None` — filling the frame, the crop following its robot (a picture about as narrow as the
/// frame, or a robot that fits in what the crop shows: its box `subject` `[x, y, w, h]`, 0..1 of
/// the picture, at most [`CROP_ROOM`] of it); else a [`Band`]: the picture as wide as shows the
/// whole robot across [`BAND_FILL`] of the frame (from the whole picture to filling the frame;
/// half again the frame's width when the robot's size is not known), the robot in the middle as
/// far as the picture still covers the frame's width, a little above the frame's middle (the
/// label sits under it).
pub fn band(aspect: f64, (fw, fh): (f64, f64), subject: Option<[f64; 4]>) -> Option<Band> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    if !(ok(aspect) && ok(fw) && ok(fh)) || aspect <= fw / fh * 1.15 {
        return None;
    }
    let visible = (fw / fh) / aspect;
    let subject = subject.filter(|b| b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0);
    if let Some(b) = subject
        && b[2] <= visible * CROP_ROOM
    {
        return None;
    }
    let widest = fh * aspect;
    let width = match subject {
        Some(b) => BAND_FILL * fw / b[2].clamp(0.05, 1.0),
        None => 1.5 * fw,
    }
    .clamp(fw, widest.max(fw));
    let cx = subject.map_or(0.5, |b| (b[0] + b[2] / 2.0).clamp(0.0, 1.0));
    let x = (fw / 2.0 + (0.5 - cx) * width).clamp(fw - width / 2.0, width / 2.0);
    let height = width / aspect;
    let y = if height >= fh { fh / 2.0 } else { (fh * 0.46).clamp(height / 2.0, fh - height / 2.0) };
    Some(Band { width, centre: [x, y] })
}

/// The show's privacy blur onto the trailer: for each `(show clip, trailer clips)` of the same
/// footage, the show clip's blur and mosaic effects — the masks its privacy pass made, keyed in
/// media time — copied in the show's sequence and pasted onto the trailer's clips keeping their
/// media times (`edit.pasteAttributes` with `keepTimes`). The trailer is the active sequence again
/// afterwards (an error when it cannot be: nothing more may go into the show). The trailer clips
/// that got them.
async fn carry_redaction(ex: &dyn Executor, show: u64, trailer: u64, pairs: &[(u64, Vec<u64>)], notes: &mut Vec<String>) -> Result<Vec<u64>, AutoCutError> {
    let mut done = Vec::new();
    for (from, to) in pairs.iter().take(MAX_SHOTS) {
        let copied = async {
            run(ex, "sequence.open", json!({"item": show})).await?;
            run(ex, "timeline.select", json!({"clips": [from]})).await?;
            run(ex, "edit.copy", json!({})).await?;
            run(ex, "sequence.open", json!({"item": trailer})).await?;
            let keep_only_effects = json!({
                "clips": to, "motion": false, "opacity": false, "timeRemapping": false, "volume": false, "channelVolume": false,
                "panner": false, "effects": ["gaussian_blur", "mosaic"], "keepTimes": true
            });
            run(ex, "edit.pasteAttributes", keep_only_effects).await
        }
        .await;
        match copied {
            Ok(_) => done.extend(to.iter().copied()),
            Err(e) => notes.push(format!("privacy: the show's blur of clip {from} was not carried over: {e}")),
        }
    }
    run(ex, "sequence.open", json!({"item": trailer})).await?;
    Ok(done)
}

/// The clip `timeline.place` made with `p` (its first).
async fn placed_clip(ex: &dyn Executor, p: Value) -> Result<u64, AutoCutError> {
    let r = run(ex, "timeline.place", p).await?;
    r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| AutoCutError::Executor("timeline.place placed nothing".into()))
}

/// Fill the frame with `clip`: its Motion scale filling it.
async fn fill_scale(ex: &dyn Executor, clip: u64) -> Result<f64, AutoCutError> {
    let r = run(ex, "clip.fillFrame", json!({"clips": [clip]})).await?;
    r["clips"]
        .as_array()
        .and_then(|a| a.iter().find(|c| c["clip"].as_u64() == Some(clip)))
        .and_then(|c| c["scale"].as_f64())
        .filter(|s| s.is_finite() && *s > 0.0)
        .ok_or_else(|| AutoCutError::Executor("clip.fillFrame gave no scale".into()))
}

/// Make `clip` (filling the frame at `fill`) the showcase's blurred, dimmed, enlarged backdrop.
async fn backdrop(ex: &dyn Executor, clip: u64, fill: f64) {
    use filmcraft_engine::reframe::{BACKDROP_BLURRINESS, BACKDROP_OPACITY, BACKDROP_ZOOM};
    // cosmetic: a backdrop that keeps some of its look is still a backdrop
    run(ex, "effects.apply", json!({"clips": [clip], "effect": "gaussian_blur"})).await.ok();
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "gaussian_blur", "param": "blurriness", "value": BACKDROP_BLURRINESS})).await.ok();
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "gaussian_blur", "param": "repeat_edge", "value": true})).await.ok();
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "opacity", "param": "opacity", "value": BACKDROP_OPACITY})).await.ok();
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": fill * BACKDROP_ZOOM})).await.ok();
}

// ---------------------------------------------------------------------------------------------
// Privacy

/// The share of the trailer's frame height a face must reach to be hidden, so that every face the
/// show hid in this footage is hidden in the trailer too: the show hid faces of
/// [`SHOW_MIN_FACE`] of its frame as it showed them (filling its 16:9 frame — `show_fit` fill, a
/// portrait clip three times larger — or whole), pushed in by up to [`SHOW_PUSH`]; the trailer
/// counts them as it shows them when it redacts (filling its frame). `aspect` is the footage's
/// width over height.
pub fn face_threshold(show_fit: Option<Fit>, aspect: f64, show_frame: (f64, f64), frame: (f64, f64)) -> f64 {
    let ratio = |(w, h): (f64, f64)| if w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0 { w / h } else { 16.0 / 9.0 };
    let a = if aspect.is_finite() && aspect > 0.0 { aspect } else { 16.0 / 9.0 };
    let (fs, ft) = (ratio(show_frame), ratio(frame));
    // the picture's height as a share of the frame's: filling it (cover) or whole (contain)
    let cover = |f: f64| (f / a).max(1.0);
    let contain = |f: f64| (f / a).min(1.0);
    let shown = match show_fit {
        Some(Fit::Fill) => cover(fs),
        Some(Fit::Full) => contain(fs),
        None if (a - fs).abs() <= 0.05 => 1.0,
        None => cover(fs),
    };
    let here = cover(ft);
    let k = (here / (shown * SHOW_PUSH)).min(1.0);
    ((SHOW_MIN_FACE * k * FACE_MARGIN) * 1000.0).floor().max(5.0) / 1000.0
}

/// What the show hid in an item, and what the trailer hid: a privacy finding where the show hid
/// faces (or plates) and the trailer's own pass hid none in a shot of it — a warning, or, when
/// the show's own blur was carried onto that shot (`carried`, by `carry_redaction`), a note for the
/// record.
pub fn carried_over(show: &[Hidden], show_layout: &[PlacedEntry], trailer: &[Hidden], shots: &[TrailerShot], carried: &[u64]) -> Vec<Finding> {
    let mut out = Vec::new();
    for s in shots {
        let shown: Vec<&Hidden> = show
            .iter()
            .filter(|h| show_layout.iter().flat_map(|p| p.shots.iter()).any(|ps| ps.item == s.item && (ps.clip == h.clip || h.also.contains(&ps.clip))))
            .collect();
        let (faces, plates) = (shown.iter().map(|h| h.faces).sum::<usize>(), shown.iter().map(|h| h.plates).sum::<usize>());
        if faces + plates == 0 {
            continue;
        }
        let here: Vec<&Hidden> = trailer.iter().filter(|h| h.clip == s.clip || h.also.contains(&s.clip)).collect();
        let hidden = here.iter().map(|h| h.faces + h.plates).sum::<usize>();
        if hidden == 0 {
            let (severity, message) = if carried.contains(&s.clip) {
                (
                    Severity::Info,
                    format!(
                        "the show hid {faces} face(s) and {plates} plate(s) in \"{}\"; its blur is carried onto the trailer's {:.1}–{:.1} s of it (the trailer's own pass found none there)",
                        s.label, s.from, s.to
                    ),
                )
            } else {
                (
                    Severity::Warning,
                    format!(
                        "the show hid {faces} face(s) and {plates} plate(s) in \"{}\"; the trailer's {:.1}–{:.1} s of it hides none — look at it (none may be in view there)",
                        s.label, s.from, s.to
                    ),
                )
            };
            out.push(Finding { check: "privacy", severity, seconds: s.start, until: Some(s.end), entry: None, clip: Some(s.clip), message });
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Text on the frame

/// Where the text goes, frame pixels `[left, top, right, bottom]`: all inside `platform`'s safe
/// area (inset a little), the hook's title at its top, a robot's label at its bottom, the end
/// card's logo and lines in its middle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Boxes {
    pub safe: [f64; 4],
    pub title: [f64; 4],
    pub label: [f64; 4],
    pub logo: [f64; 4],
    pub end_text: [f64; 4],
}

/// The [`Boxes`] of a `w`×`h` frame for `platform`.
pub fn boxes(platform: Platform, (w, h): (f64, f64)) -> Boxes {
    let (w, h) = (if w.is_finite() && w >= 16.0 { w } else { 1080.0 }, if h.is_finite() && h >= 16.0 { h } else { 1920.0 });
    let s = platform.safe_area().in_pixels((w, h));
    let pad = 0.03 * w;
    let [x0, y0, x1, y1] = [s[0] + pad, s[1] + pad, s[2] - pad, s[3] - pad];
    let sh = (y1 - y0).max(1.0);
    let title = [x0, y0, x1, y0 + 0.30 * sh];
    let label = [x0, y1 - 0.20 * sh, x1, y1];
    let side = (0.6 * (x1 - x0)).min(0.48 * sh);
    let cx = (x0 + x1) / 2.0;
    let logo_top = y0 + 0.04 * sh;
    let logo = [cx - side / 2.0, logo_top, cx + side / 2.0, logo_top + side];
    let end_text = [x0, logo[3] + 0.06 * sh, x1, (logo[3] + 0.06 * sh + 0.34 * sh).min(y1)];
    Boxes { safe: s, title, label, logo, end_text }
}

/// Whether `b` lies inside `area` (pixels), give or take a pixel.
pub fn inside(b: [f64; 4], area: [f64; 4]) -> bool {
    b[0] >= area[0] - 1.0 && b[1] >= area[1] - 1.0 && b[2] <= area[2] + 1.0 && b[3] <= area[3] + 1.0
}

/// The hook's title in two lines: `"48 robots"` and `"2013 → 2025"` (or the `title` given, its
/// lines split at " · ").
pub fn hook_title(title: Option<&str>, robots: usize, years: Option<(i64, i64)>, subject: &str) -> (String, Option<String>) {
    if let Some(t) = title.map(str::trim).filter(|t| !t.is_empty()) {
        return match t.split_once(" · ") {
            Some((a, b)) => (a.trim().to_string(), Some(b.trim().to_string()).filter(|b| !b.is_empty())),
            None => (t.to_string(), None),
        };
    }
    let plural = if robots == 1 { subject.to_string() } else { format!("{subject}s") };
    let first = format!("{robots} {plural}");
    let second = years.map(|(a, b)| if a == b { a.to_string() } else { format!("{a} → {b}") });
    (first, second)
}

/// A robot's label: its name and, when known, the year.
pub fn label_lines(c: &Candidate) -> (String, Option<String>) {
    (c.label.clone(), c.unix.map(|u| crate::showcase::ymd(u).0.to_string()))
}

// ---------------------------------------------------------------------------------------------
// Caption

/// Words that make a label a description rather than a robot's name ("Reception robot", "Robot
/// arm game"): a caption names robots by name.
const GENERIC_WORDS: &[&str] = &["robot", "robots", "arm", "arms", "game", "hoop", "van", "dog", "vehicle", "kiosk", "printer"];

/// A label that names a robot (not "My first robot", not "Reception robot").
fn proper_name(name: &str) -> bool {
    let lower = name.trim().to_lowercase();
    let last = lower.split_whitespace().last().unwrap_or_default();
    !lower.is_empty() && !lower.starts_with("my ") && !GENERIC_WORDS.contains(&last)
}

/// A suggested caption: what it is, three of its robots by name — the hook's, then those asked
/// for (`named_first`, the `include` list), then the most striking of the kinds not named yet (a
/// robot dog after a humanoid) — where the full video is ("link in bio": a Reel's caption cannot
/// hold a link), and hashtags from the robots' kinds, the events and places, the channel and the
/// show's language.
pub fn caption(
    title: &(String, Option<String>),
    picked: &[&Candidate],
    named_first: &[String],
    events: &[String],
    channel: Option<&str>,
    language: Option<&str>,
    end_text: &str,
) -> (String, Vec<String>) {
    let mut lines = Vec::new();
    let head = match &title.1 {
        Some(years) => format!("{}, {years}.", title.0),
        None => format!("{}.", title.0),
    };
    lines.push(head);
    // each robot once, by its name ("Pepper", not "Pepper, again")
    let class = |k: &str| kind_weight(k).to_bits();
    let mut rest: Vec<&Candidate> = picked.iter().skip(1).copied().collect();
    rest.sort_by(|a, b| b.score.total_cmp(&a.score));
    let asked: Vec<&Candidate> = named_first.iter().filter_map(|l| rest.iter().copied().find(|c| matches_label(c, l))).collect();
    let mut chosen: Vec<&Candidate> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    fn take<'a>(c: &'a Candidate, chosen: &mut Vec<&'a Candidate>, names: &mut Vec<String>) {
        let name = shown_name(&c.label);
        if names.len() < 3 && proper_name(&name) && !names.iter().any(|n| same_robot(n, &name)) {
            names.push(name);
            chosen.push(c);
        }
    }
    for c in picked.first().copied().into_iter().chain(asked) {
        take(c, &mut chosen, &mut names);
    }
    // then a kind not named yet, the most striking of it, while there is one
    while names.len() < 3 {
        let before = names.len();
        if let Some(c) =
            rest.iter().copied().find(|c| !chosen.iter().any(|k| class(&k.kind) == class(&c.kind)) && proper_name(&shown_name(&c.label)) && !chosen.contains(c))
        {
            take(c, &mut chosen, &mut names);
        }
        if names.len() == before {
            break;
        }
    }
    // and the most striking of any kind
    for c in rest.iter().copied() {
        take(c, &mut chosen, &mut names);
    }
    if !names.is_empty() {
        lines.push(format!("{} and more: every robot I've met, in the order I met them.", names.join(", ")));
    }
    let lang =
        language.filter(|l| l.trim().eq_ignore_ascii_case("ml") || l.trim().to_ascii_lowercase().starts_with("ml-")).map(|_| " (in Malayalam)").unwrap_or("");
    let cta = match channel {
        Some(c) => format!("{end_text}{lang}: {c}, link in bio."),
        None => format!("{end_text}{lang}, link in bio."),
    };
    lines.push(cta);
    let mut tags: Vec<String> = vec!["robots".into(), "robotics".into()];
    let mut add = |t: &str| {
        let t: String = t.chars().filter(|c| c.is_alphanumeric()).collect();
        if !t.is_empty() && t.chars().count() <= 30 && !tags.iter().any(|x| x.eq_ignore_ascii_case(&t)) {
            tags.push(t);
        }
    };
    let kinds: Vec<String> = picked.iter().map(|c| c.kind.to_ascii_lowercase()).collect();
    if kinds.iter().any(|k| k.contains("humanoid")) {
        add("humanoidrobot");
    }
    if kinds.iter().any(|k| k.contains("quadruped") || k.contains("dog")) {
        add("robotdog");
    }
    add("AI");
    add("tech");
    for e in events.iter().take(6) {
        let lower = e.to_ascii_lowercase();
        if lower.contains("gitex") {
            add("GITEX");
        }
        if lower.contains("expo 2020") {
            add("Expo2020");
        }
        if lower.contains("dubai") {
            add("Dubai");
        }
    }
    if !lang.is_empty() {
        add("Malayalam");
    }
    if let Some(c) = channel {
        add(c.trim_start_matches('@'));
    }
    add("reels");
    let hashtags: Vec<String> = tags.into_iter().take(15).map(|t| format!("#{t}")).collect();
    lines.push(String::new());
    lines.push(hashtags.join(" "));
    (lines.join("\n"), hashtags)
}

// ---------------------------------------------------------------------------------------------
// The build

/// A shot as the trailer placed it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrailerShot {
    /// 0 for the hook, then 1.. for the montage.
    pub number: usize,
    pub entry: usize,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year: Option<i64>,
    pub clip: u64,
    pub item: u64,
    /// Where it plays on the timeline, and the stretch of its clip.
    pub start: f64,
    pub end: f64,
    pub from: f64,
    pub to: f64,
    /// The stretch lies inside what the show played.
    pub in_show_window: bool,
    /// Shown in a band (its robot's whole width) over its blurred copy (`background`), not
    /// filling the frame with the crop following the robot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub band: Option<Band>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<u64>,
    /// Its robot was followed (else the whole picture over a blurred copy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub followed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_clip: Option<u64>,
}

async fn media_size(ex: &dyn Executor, item: u64) -> Option<(f64, f64)> {
    let r = ex.execute("media.frame", json!({"item": item, "seconds": 0.0, "maxSide": 64})).await.ok()?;
    let (w, h) = (r["width"].as_f64()?, r["height"].as_f64()?);
    (w > 0.0 && h > 0.0 && w.is_finite() && h.is_finite()).then_some((w, h))
}

/// The hook voice's line (`sentence`, said from `at` for `secs` seconds) as captions on a new
/// caption track of the trailer: Bold animated captions by default (`captionStyle`), inside the
/// platform's safe zone, the words timed from the narration's script (or spread over the line);
/// with `captionKeywords` the model colours keywords and adds emoji. Whether a track was made
/// (the export then burns it in); a step that fails is noted.
async fn hook_captions(ex: &dyn Executor, sentence: &str, at: f64, secs: f64, opts: &Options, model: Option<&dyn ChatModel>, notes: &mut Vec<String>) -> bool {
    let track = match run(ex, "captions.newTrack", json!({"format": "Subtitle", "name": "Captions"})).await {
        Ok(r) => r["track"].as_u64(),
        Err(e) => {
            notes.push(format!("captions: {e}"));
            None
        }
    };
    let Some(track) = track else { return false };
    if let Err(e) = run(ex, "captions.add", json!({"track": track, "text": sentence, "time": ticks(at), "durationSeconds": secs})).await {
        notes.push(format!("captions: {e}"));
        return false;
    }
    if let Err(e) = run(ex, "captions.wordTiming", json!({"track": track, "from": "narration"})).await {
        notes.push(format!("captions: word timing: {e}; the words are spread over the line"));
    }
    let extra = match (model, opts.caption_keywords, &opts.caption_look) {
        (Some(m), true, caption_style::Look::Animated(_)) => Some(caption_style::keywords(m, sentence, notes).await),
        _ => None,
    };
    match caption_style::style_track(ex, track, opts.platform, &opts.caption_look, extra.as_ref()).await {
        Ok(_) => true,
        Err(e) => {
            notes.push(format!("captions: {e}"));
            true
        }
    }
}

/// The show's narration clip that opens it, and how far its first sentence runs: `(item, source
/// in, seconds, the sentence)`. From the show's voice track (A2, the showcase's), its first clip;
/// cut at the pause nearest where the first sentence's share of the line's text ends
/// (`media.pauses`).
async fn first_sentence(ex: &dyn Executor, seq: u64, narration: &[Cue], notes: &mut Vec<String>) -> Option<(u64, f64, f64, String)> {
    const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;
    let q = ex.execute("sequence.inspect", json!({"item": seq})).await.ok()?;
    let a2 = q["audio"].get(1)?["items"].as_array()?;
    let first = a2.iter().take(100_000).filter(|c| c["enabled"].as_bool() != Some(false)).min_by_key(|c| c["start"].as_i64().unwrap_or(i64::MAX))?;
    let item = first["item"].as_u64()?;
    let start = first["start"].as_i64()? as f64 / TPS;
    let src = first["sourceIn"].as_i64().unwrap_or(0) as f64 / TPS;
    let len = first["duration"].as_i64()? as f64 / TPS;
    if !(src.is_finite() && len.is_finite() && len > 0.5 && src >= 0.0) {
        return None;
    }
    let text = narration.iter().find(|c| (c.start - start).abs() < 0.25).map(|c| c.text.clone()).unwrap_or_default();
    // the first sentence's share of the text
    let total = text.chars().count().max(1);
    let end_of_first = text.char_indices().find(|(_, c)| matches!(c, '.' | '!' | '?' | '।' | '…')).map(|(i, c)| i + c.len_utf8());
    let sentence = end_of_first.and_then(|e| text.get(..e)).unwrap_or(&text).trim().to_string();
    let share = (sentence.chars().count() as f64 / total as f64).clamp(0.05, 1.0);
    if share >= 0.98 || len <= 6.0 {
        return Some((item, src, len.min(8.0), sentence));
    }
    let r = match ex.execute("media.pauses", json!({"item": item, "sourceIn": ticks(src), "duration": ticks(len)})).await {
        Ok(r) => r,
        Err(e) => {
            notes.push(format!("hook voice: the pauses of the opening line were not found ({e}); no voice over the hook"));
            return None;
        }
    };
    let span = |k: &str| {
        r[k].as_array().map(|a| a.iter().filter_map(|p| Some((p["start"].as_f64()?, p["end"].as_f64()?))).take(10_000).collect::<Vec<_>>()).unwrap_or_default()
    };
    let (sound, pauses) = (span("sound"), span("pauses"));
    let (s0, s1) = (sound.first().map_or(src, |s| s.0), sound.last().map_or(src + len, |s| s.1));
    let expected = s0 + share * (s1 - s0);
    let cut = pauses.iter().filter(|p| p.0 - s0 >= 1.0).min_by(|a, b| ((a.0 + a.1) / 2.0 - expected).abs().total_cmp(&((b.0 + b.1) / 2.0 - expected).abs()))?;
    // a little of the pause after the last word, never into the next one
    let end = (cut.0 + 0.15).min((cut.0 + cut.1) / 2.0);
    let seconds = end - src;
    (seconds > 0.8 && seconds <= len + 0.01).then_some((item, src, seconds, sentence))
}

/// A graphic text clip of one line on track `track` (an index), centred on `cx` with its
/// baseline at `y`, shrunk to `room` pixels across: its clip and bounds.
#[allow(clippy::too_many_arguments)]
async fn line(
    ex: &dyn Executor,
    track: usize,
    text: &str,
    cx: f64,
    y: f64,
    size: f64,
    room: f64,
    props: Value,
    t: f64,
    dur: f64,
) -> Result<(u64, Bounds), AutoCutError> {
    let clip = look::text_clip(ex, track, FPS, text, [cx, y], size, t, dur, props).await?;
    let b = look::measure(ex, clip, Some(0)).await.unwrap_or_else(|| look::estimate(text, size, [cx, y], "center"));
    let b = look::fit_width(ex, clip, 0, size, b, room).await;
    Ok((clip, b))
}

/// A second line added to graphic `clip`: its bounds.
#[allow(clippy::too_many_arguments)]
async fn add_line(ex: &dyn Executor, clip: u64, text: &str, cx: f64, y: f64, size: f64, room: f64, props: Value) -> Result<Bounds, AutoCutError> {
    let layer = look::add_text(ex, clip, text, [cx, y], size, props).await?;
    let b = look::measure(ex, clip, Some(layer)).await.unwrap_or_else(|| look::estimate(text, size, [cx, y], "center"));
    Ok(look::fit_width(ex, clip, layer, size, b, room).await)
}

fn arr(b: Bounds) -> [f64; 4] {
    [b.left, b.top, b.right, b.bottom]
}

/// Bold text with a dark outline and shadow, readable over any footage.
fn loud(theme: &Theme, role: Role, color: &str, u: f64) -> Value {
    let mut p = theme.text_props(role, color, "center");
    p["stroke"] = json!(true);
    p["stroke_width"] = json!(5.0 * u);
    p["stroke_color"] = json!("#000000");
    p["shadow_opacity"] = json!(80.0);
    p["shadow_blur"] = json!(24.0 * u);
    p
}

/// What the trailer made.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Made {
    pub sequence: Option<u64>,
    pub seconds: f64,
    pub frame: (u32, u32),
    pub platform: String,
    pub bpm: Option<f64>,
    pub timing: Option<Timing>,
    pub shots: Vec<TrailerShot>,
    pub title: Vec<String>,
    pub cuts: usize,
    pub cuts_on_beat: usize,
    pub voice: Option<Value>,
    pub music: Option<Value>,
    pub privacy: Vec<Hidden>,
    /// The trailer clips the show's own blur was carried onto (its masks, in media time).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub carried: Vec<u64>,
    pub safe_zones: Option<Value>,
    pub loudness: Option<Value>,
    pub mix: Option<Value>,
    pub export: Option<Value>,
    pub cover: Option<String>,
    pub caption: Option<String>,
    pub caption_text: Option<String>,
    pub hashtags: Vec<String>,
    pub frames: Option<String>,
    pub qa_summary: Option<String>,
    pub qa: Option<Value>,
    /// The creator look put over the hook and the montage, if asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    /// Zoom pulses keyframed on the montage (`beatPulse`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_pulses: Option<u64>,
    pub notes: Vec<String>,
}

/// `out.mp4` → `out.<ext>` beside it (`cover.jpg`, `caption.txt`, `frames.jpg`).
pub fn beside(out: &str, ext: &str) -> PathBuf {
    Path::new(out).with_extension(ext)
}

/// Make the trailer (the `autocut.trailer` job; parameters as the engine checked them): `plan`
/// is the plan the session keeps (used when the parameters carry none); `qa_model` grounds QA's
/// subject check.
pub async fn make(
    ex: &dyn Executor,
    p: &Value,
    plan: Option<&Value>,
    qa_model: Option<&dyn ChatModel>,
    progress: &mut dyn FnMut(&str),
) -> Result<Value, AutoCutError> {
    let opts = Options::from_params(p)?;
    let plan = p.get("plan").filter(|v| v.is_object()).or(plan);
    let report = p.get("report").filter(|v| v.is_object());
    let mut show = Show::from_docs(plan, report);
    show.sequence = opts.sequence.or(show.sequence);
    let made = build(ex, &opts, &mut show, qa_model, progress).await?;
    let status = format!(
        "Made a {:.1} s trailer: {} robots after the hook, {} of {} cuts on the beat{}",
        made.seconds,
        made.shots.len().saturating_sub(1),
        made.cuts_on_beat,
        made.cuts,
        made.export.as_ref().and_then(|e| e["path"].as_str()).map(|p| format!(", exported to {p}")).unwrap_or_default()
    );
    let mut v = serde_json::to_value(&made).map_err(|e| AutoCutError::Other(format!("trailer: {e}")))?;
    v["status"] = json!(status);
    Ok(v)
}

/// The trailer's tracks (indices): V1 the end card's footage, V2 the shots (V3 once the reframe
/// puts its blurred copies under them), then the labels, the title, the badge and the logo.
const VIDEO_TRACKS: usize = 6;

/// Build it (see the module docs).
pub async fn build(
    ex: &dyn Executor,
    opts: &Options,
    show: &mut Show,
    qa_model: Option<&dyn ChatModel>,
    progress: &mut dyn FnMut(&str),
) -> Result<Made, AutoCutError> {
    let mut notes = std::mem::take(&mut show.notes);
    let (fw, fh) = (f64::from(opts.frame.0), f64::from(opts.frame.1));
    let u = fw.min(fh) / 1080.0;
    // the show's sequence: its name, and its entries when there is no plan
    let show_seq = match show.sequence {
        Some(id) => ex.execute("sequence.inspect", json!({"item": id})).await.ok(),
        None => None,
    };
    if show.name.is_empty() {
        show.name = show_seq.as_ref().and_then(|q| q["name"].as_str()).map(str::to_string).unwrap_or_else(|| "Showcase".into());
    }
    if show.entries.is_empty() {
        let q = show_seq.as_ref().ok_or_else(|| invalid("give the show's plan (plan or planPath) or its sequence"))?;
        let (entries, layout) = entries_from_sequence(q);
        if entries.is_empty() {
            return Err(invalid("the show's sequence has no shots to make a trailer of"));
        }
        notes.push(format!("no plan: {} entries read from the show's sequence (labels from its lower thirds, no dates)", entries.len()));
        show.entries = entries;
        show.layout = layout;
        show.from_sequence = true;
    }
    let show_frame = show_seq
        .as_ref()
        .and_then(|q| Some((q["settings"]["width"].as_f64()?, q["settings"]["height"].as_f64()?)))
        .filter(|(w, h)| *w > 0.0 && *h > 0.0)
        .unwrap_or((1920.0, 1080.0));
    // ---- the robots, scored with how they were filmed
    let mut cands = candidates(&show.entries, &show.layout);
    if cands.is_empty() {
        return Err(invalid("no robot entry with footage in the show"));
    }
    progress(&format!("{} robots in \"{}\"; looking at their footage", cands.len(), show.name));
    let mut aspects: Vec<(u64, f64)> = Vec::new();
    for c in cands.iter_mut() {
        let a = match aspects.iter().find(|(i, _)| *i == c.shot.item) {
            Some((_, a)) => *a,
            None => {
                let a = media_size(ex, c.shot.item).await.map_or(16.0 / 9.0, |(w, h)| w / h);
                aspects.push((c.shot.item, a));
                a
            }
        };
        let area = c.shot.moment.map_or(0.0, |m| m.area);
        c.score = striking(&c.kind, area, a < 0.9, c.shot.photo, c.branded);
    }
    // ---- the music's tempo, and how many robots fit
    let suggest =
        run(ex, "music.suggest", json!({"mood": opts.mood, "bpm": opts.bpm, "count": 1, "seed": opts.seed, "useCase": "trailer", "duration": opts.seconds}))
            .await
            .ok();
    let bpm = suggest.as_ref().and_then(|v| v["suggestions"][0]["track"]["bpm"].as_f64()).filter(|b| b.is_finite() && *b >= 40.0).unwrap_or(opts.bpm);
    let genre = suggest.as_ref().and_then(|v| v["suggestions"][0]["track"]["genre"].as_str()).map(str::to_string);
    let grid = Beats::grid(bpm);
    let n = shot_count(grid.period(), opts.seconds, opts.robots, cands.len().saturating_sub(1));
    let choice = choose(&cands, n, opts.hook.as_deref(), &opts.include, &opts.exclude)?;
    let n = choice.montage.len();
    if n == 0 {
        return Err(invalid("only one robot to show: a trailer needs a few"));
    }
    let mut hook_c = cands.get(choice.hook).cloned().ok_or_else(|| invalid("no hook"))?;
    let mut montage: Vec<Candidate> = choice.montage.iter().filter_map(|&i| cands.get(i).cloned()).collect();
    // ---- where shots start when asked (`at`, the hook's `hookAt`), in another clip of the entry
    let mut starts: Vec<Option<f64>> = Vec::new();
    for (k, c) in std::iter::once(&mut hook_c).chain(montage.iter_mut()).enumerate() {
        let asked =
            opts.at.iter().find(|a| matches_label(c, &a.label)).map(|a| (a.item, a.seconds)).or(if k == 0 { opts.hook_at.map(|s| (None, s)) } else { None });
        let Some((item, seconds)) = asked else {
            starts.push(None);
            continue;
        };
        if let Some(i) = item.filter(|i| *i != c.shot.item) {
            match show.entries.iter().find(|e| e.number == c.number).and_then(|e| pick_for(e, &show.layout, i)) {
                Some(p) => {
                    if !aspects.iter().any(|(x, _)| *x == i) {
                        aspects.push((i, media_size(ex, i).await.map_or(16.0 / 9.0, |(w, h)| w / h)));
                    }
                    c.shot = p;
                }
                None => notes.push(format!("at: \"{}\" has no clip {i}; its own is used", c.label)),
            }
        }
        starts.push(Some(seconds));
    }
    for a in opts.at.iter().filter(|a| !std::iter::once(&hook_c).chain(montage.iter()).any(|c| matches_label(c, &a.label))) {
        notes.push(format!("at: no robot called \"{}\" is in the trailer", a.label));
    }
    progress(&format!("hook: {}; montage: {}", hook_c.label, montage.iter().map(|c| c.label.as_str()).collect::<Vec<_>>().join(", ")));
    let planned = timing(&grid, opts.seconds, n)?;
    // ---- the sequence
    let name = format!("{} — Reel trailer", show.name);
    let r = run(ex, "file.newSequence", json!({"name": name, "width": opts.frame.0, "height": opts.frame.1, "fps": FPS, "video": VIDEO_TRACKS, "audio": 5}))
        .await?;
    let seq = r["sequence"].as_u64();
    let mut made = Made { sequence: seq, frame: opts.frame, platform: opts.platform.name().into(), ..Made::default() };
    // ---- the music first: its exact beats are the grid every cut lands on
    let energy: Vec<f64> = (0..16)
        .map(|k| {
            if k < 2 {
                0.62
            } else if k < 13 {
                0.8 + 0.2 * (k - 2) as f64 / 10.0
            } else {
                0.85
            }
        })
        .collect();
    let mut mp =
        json!({"kind": "music", "duration": planned.total, "mood": opts.mood, "bpm": bpm, "seed": opts.seed, "energy": energy, "time": 0, "track": "A3"});
    if let Some(g) = &genre {
        mp["genre"] = json!(g);
    }
    progress(&format!("music: {} at {bpm:.0} bpm, {:.1} s", opts.mood, planned.total));
    let music = run(ex, "audio.generate", mp).await?;
    let map: Vec<i64> = music["beats"].as_array().map(|a| a.iter().filter_map(Value::as_i64).take(100_000).collect()).unwrap_or_default();
    let beats = Beats::from_ticks(&map, 0.0).unwrap_or(grid);
    let tm = timing(&beats, opts.seconds, n)?;
    if let Some(ending) = music["ending"].as_i64().map(|t| t as f64 / filmcraft_time::TICKS_PER_SECOND as f64)
        && (ending - tm.end_card[0]).abs() > 1.5 / FPS
    {
        notes.push(format!("music: its final hit is at {ending:.2} s, the end card at {:.2} s", tm.end_card[0]));
    }
    made.bpm = Some(60.0 / beats.period());
    made.music = Some(
        json!({"clip": music["clip"], "item": music["item"], "bpm": music["bpm"], "genre": music["genre"], "seed": opts.seed, "ending": music["ending"].as_i64().map(|t| t as f64 / filmcraft_time::TICKS_PER_SECOND as f64)}),
    );
    let music_clip = music["clip"].as_u64();
    // ---- the shots: each filled to the frame (as the privacy pass counts its faces), and how it
    // will be shown — following its robot, or its robot's whole width in a band over a blurred copy
    let mut shots: Vec<TrailerShot> = Vec::new();
    let mut fills: Vec<f64> = Vec::new();
    let all: Vec<(&Candidate, [f64; 2])> = std::iter::once((&hook_c, tm.hook)).chain(montage.iter().zip(tm.shots.iter().copied())).collect();
    for (k, (c, [t0, t1])) in all.iter().enumerate() {
        let len = t1 - t0;
        let (from, inside) = match starts.get(k).copied().flatten() {
            Some(at) if c.shot.clip_seconds > len + 0.1 => {
                let from = at.clamp(0.05, c.shot.clip_seconds - len - 0.05);
                (from, from >= c.shot.window[0] - 1e-6 && from + len <= c.shot.window[1] + 1e-6)
            }
            _ => window_start(&c.shot, len).ok_or_else(|| invalid(format!("{}: its clip is shorter than its {len:.1} s", c.label)))?,
        };
        progress(&format!("[{k}/{n}] {} ({:.1}–{:.1} s of {})", c.label, from, from + len, c.shot.name));
        let place =
            |track: &str| json!({"item": c.shot.item, "track": track, "time": ticks(*t0), "sourceIn": ticks(from), "duration": ticks(len), "audio": false});
        let clip = placed_clip(ex, place("V2")).await?;
        let fill = fill_scale(ex, clip).await?;
        let aspect = aspects.iter().find(|(i, _)| *i == c.shot.item).map_or(16.0 / 9.0, |a| a.1);
        let subject = Shot { largest: c.shot.moment, ..Shot::default() }.subject_box(aspect);
        // the hook fills the frame: the one robot seen up close
        let wide = if k == 0 { None } else { band(aspect, (fw, fh), subject) };
        // a band's blurred copy, made now so that the privacy pass hides its faces too
        let background = match wide {
            Some(_) => match placed_clip(ex, place("V1")).await {
                Ok(b) => {
                    run(ex, "clip.fillFrame", json!({"clips": [b]})).await?;
                    Some(b)
                }
                Err(e) => {
                    notes.push(format!("{}: its blurred copy: {e}", c.label));
                    None
                }
            },
            None => None,
        };
        fills.push(fill);
        let (label, year) = label_lines(c);
        shots.push(TrailerShot {
            number: k,
            entry: c.number,
            label,
            year: year.and_then(|y| y.parse().ok()),
            clip,
            item: c.shot.item,
            start: *t0,
            end: *t1,
            from: (from * 1000.0).round() / 1000.0,
            to: ((from + len) * 1000.0).round() / 1000.0,
            in_show_window: inside,
            band: wide,
            background,
            followed: None,
            label_clip: None,
        });
    }
    if shots.len() < 2 {
        return Err(invalid("too few shots could be placed"));
    }
    // the layout QA and the privacy pass read: one entry a shot (a band's blurred copy with it)
    let layout: Vec<PlacedEntry> = shots
        .iter()
        .map(|s| PlacedEntry {
            number: s.entry,
            label: s.label.clone(),
            start: s.start,
            end: s.end,
            shots: vec![PlacedShot {
                clip: s.clip,
                item: s.item,
                start: s.start,
                end: s.end,
                source: Some([s.from, s.to]),
                fit: s.band.map(|_| Fit::Full),
                background: s.background,
                ..PlacedShot::default()
            }],
            ..PlacedEntry::default()
        })
        .collect();
    // ---- privacy: the show's redaction again, faces hidden at the size the show hid them
    let redact = match &opts.redact {
        Some(r) => r.clone(),
        None if show.redacted => Some(Redact::default()),
        None => None,
    };
    let any_targets = show.entries.iter().any(|e| e.redact.is_some() || e.shots.iter().any(|s| s.redact.is_some()));
    if redact.is_some() || any_targets {
        let mut groups: Vec<(f64, Vec<PlacedEntry>)> = Vec::new();
        for (pe, s) in layout.iter().zip(&shots) {
            let aspect = aspects.iter().find(|(i, _)| *i == s.item).map_or(16.0 / 9.0, |a| a.1);
            let fit = std::iter::once(&hook_c).chain(montage.iter()).find(|c| c.number == s.entry && c.shot.item == s.item).and_then(|c| c.shot.show_fit);
            let th = face_threshold(fit, aspect, show_frame, (fw, fh));
            match groups.iter_mut().find(|(t, _)| (*t - th).abs() < 1e-9) {
                Some((_, g)) => g.push(pe.clone()),
                None => groups.push((th, vec![pe.clone()])),
            }
        }
        for (th, group) in &groups {
            let plan = redact.clone().map(|r| Redact { min_face_height: Some(*th), ..r });
            let mut lines = |l: &str| progress(l);
            let hidden = crate::privacy::redact_show(ex, plan.as_ref(), group, &show.entries, &mut notes, &mut lines).await;
            made.privacy.extend(hidden);
        }
    }
    // ---- and the show's own blur on top: its masks (in media time) onto the trailer's clips of
    // the same footage, so whatever the show hid stays hidden whatever this pass found
    let mut carried: Vec<u64> = Vec::new();
    if let (Some(show_id), Some(trailer_id)) = (show.sequence, seq) {
        let mut pairs: Vec<(u64, Vec<u64>)> = Vec::new();
        for s in &shots {
            let Some(from) =
                std::iter::once(&hook_c).chain(montage.iter()).find(|c| c.number == s.entry && c.shot.item == s.item).and_then(|c| c.shot.show_clip)
            else {
                continue;
            };
            let hid = show.privacy.iter().any(|h| (h.clip == from || h.also.contains(&from)) && h.masks > 0);
            if hid {
                pairs.push((from, std::iter::once(s.clip).chain(s.background).collect()));
            }
        }
        if !pairs.is_empty() {
            progress(&format!("privacy: the show's own blur onto {} shot(s)", pairs.len()));
            carried = carry_redaction(ex, show_id, trailer_id, &pairs, &mut notes).await?;
        }
    }
    made.carried = carried.clone();
    // ---- the bands: the picture as wide as its robot needs, pushing in slowly, over its blurred copy
    for (s, fill) in shots.iter().zip(&fills) {
        let Some(b) = s.band else { continue };
        let aspect = aspects.iter().find(|(i, _)| *i == s.item).map_or(16.0 / 9.0, |a| a.1);
        let scale = fill * b.width / (fh * aspect).max(1.0);
        run(ex, "effects.setParam", json!({"clip": s.clip, "effect": "motion", "param": "position", "value": b.centre})).await?;
        run(ex, "effects.setParam", json!({"clip": s.clip, "effect": "motion", "param": "scale", "value": scale})).await?;
        crate::showcase::push_in(ex, s.clip, scale, s.start, s.end, FPS).await.ok();
        if let Some(bg) = s.background {
            backdrop(ex, bg, *fill).await;
        }
    }
    // ---- following each robot in the shots that fill the frame
    let mut subjects: Vec<Value> = Vec::new();
    for s in shots.iter().filter(|s| s.band.is_none()) {
        let Some(c) = std::iter::once(&hook_c).chain(montage.iter()).find(|c| c.number == s.entry) else { continue };
        let aspect = aspects.iter().find(|(i, _)| *i == s.item).map_or(16.0 / 9.0, |a| a.1);
        let shot = Shot { largest: c.shot.moment, ..Shot::default() };
        if let (Some(b), Some(m)) = (shot.subject_box(aspect), c.shot.moment)
            && m.seconds >= s.from - 0.5
            && m.seconds <= s.to + 0.5
        {
            subjects.push(json!({"x": b[0], "y": b[1], "w": b[2], "h": b[3], "clip": s.clip, "seconds": m.seconds.clamp(s.from, s.to)}));
        }
    }
    let followers: Vec<u64> = shots.iter().filter(|s| s.band.is_none()).map(|s| s.clip).collect();
    if !followers.is_empty() {
        progress("reframing: following each robot");
        let mut rp = json!({"clips": followers, "motion": "smooth", "fallback": opts.fallback, "analysisFps": REFRAME_FPS, "wait": true});
        if !subjects.is_empty() {
            rp["subject"] = Value::Array(subjects);
        }
        match run(ex, "clip.autoReframe", rp).await {
            Ok(r) => {
                for c in r["reframe"]["clips"].as_array().map(Vec::as_slice).unwrap_or_default() {
                    if let Some(s) = shots.iter_mut().find(|s| Some(s.clip) == c["clip"].as_u64()) {
                        s.followed = c["followed"].as_bool();
                    }
                }
            }
            Err(e) => notes.push(format!("reframe: {e}; the shots keep a centred crop")),
        }
    }
    // where the shots are now (the blurred copies' track went in under them)
    let q = run(ex, "sequence.inspect", json!({})).await?;
    let first_clip = shots.first().map(|s| s.clip);
    let shots_track = q["video"]
        .as_array()
        .and_then(|t| t.iter().position(|tr| tr["items"].as_array().is_some_and(|a| a.iter().any(|c| c["clip"].as_u64() == first_clip))))
        .unwrap_or(1);
    let video_tracks = q["video"].as_array().map_or(VIDEO_TRACKS, Vec::len);
    let [labels_track, title_track, badge_track, logo_track] = [1, 2, 3, 4].map(|k| (shots_track + k).min(video_tracks.saturating_sub(1)));
    // ---- the words on screen, inside the safe zone
    let bx = boxes(opts.platform, (fw, fh));
    let theme = {
        let style = look::Style::from_plan(&show.style).ok().flatten().unwrap_or_else(look::Style::tech);
        let installed = look::installed_fonts(ex).await;
        style.resolve(installed.as_deref())
    };
    let (t1, t2) = hook_title(opts.title.as_deref(), show.robots(), show.years(), "robot");
    made.title = std::iter::once(t1.clone()).chain(t2.clone()).collect();
    let mut texts: Vec<(String, [f64; 4])> = Vec::new();
    {
        let [x0, y0, x1, _] = bx.title;
        let (cx, room) = ((x0 + x1) / 2.0, x1 - x0);
        let size1 = 176.0 * u;
        let mut props = loud(&theme, Role::Title, &theme.text, u);
        props["caps"] = json!("All Caps");
        props["tracking"] = json!(20.0);
        let (clip, b1) = line(ex, title_track, &t1, cx, y0 + size1 * 0.78, size1, room, props, tm.hook[0], tm.hook[1] - tm.hook[0]).await?;
        let mut b = b1;
        if let Some(t2) = &t2 {
            let size2 = 84.0 * u;
            let mut props = loud(&theme, Role::Number, &theme.accent, u);
            props["tracking"] = json!(60.0);
            let b2 = add_line(ex, clip, t2, cx, b1.bottom + 26.0 * u + size2 * 0.74, size2, room, props).await?;
            b = b.union(b2);
        }
        texts.push(("title".into(), arr(b)));
    }
    for s in shots.iter_mut().filter(|s| s.number > 0) {
        let [x0, _, x1, y1] = bx.label;
        let (cx, room) = ((x0 + x1) / 2.0, x1 - x0 - 48.0 * u);
        let (name_size, year_size) = (76.0 * u, 54.0 * u);
        let (in_at, out_at) = (s.start + 2.0 / FPS, s.end - 1.0 / FPS);
        let year_y = y1 - 28.0 * u;
        let name_y = if s.year.is_some() { year_y - year_size - 18.0 * u } else { year_y };
        let mut props = theme.text_props(Role::Title, &theme.text, "center");
        props["caps"] = json!("All Caps");
        props["tracking"] = json!(20.0);
        let (clip, mut b) = line(ex, labels_track, &s.label, cx, name_y, name_size, room, props, in_at, out_at - in_at).await?;
        if let Some(y) = s.year {
            let mut props = theme.text_props(Role::Number, &theme.accent, "center");
            props["tracking"] = json!(80.0);
            b = b.union(add_line(ex, clip, &y.to_string(), cx, year_y, year_size, room, props).await?);
        }
        // a dark panel behind the words, inside the safe zone
        let panel = Bounds {
            left: (b.left - 24.0 * u).max(x0),
            top: (b.top - 18.0 * u).max(bx.label[1]),
            right: (b.right + 24.0 * u).min(x1),
            bottom: (b.bottom + 18.0 * u).min(y1),
        };
        if let Ok(back) = look::add_shape(
            ex,
            clip,
            "rectangle",
            panel.centre(),
            panel.size(),
            json!({"fill_color": theme.panel_rgba(), "stroke": false, "corner_radius": 14.0 * u}),
        )
        .await
        {
            run(ex, "graphics.sendToBack", json!({"clip": clip, "layer": back})).await.ok();
        }
        s.label_clip = Some(clip);
        texts.push((format!("label {}", s.label), arr(panel.union(b))));
    }
    // ---- the end card: the hook's footage blurred, the logo on its badge, the words
    let [e0, e1] = tm.end_card;
    let end_len = e1 - e0;
    {
        let item = hook_c.shot.item;
        let from = (hook_c.shot.window[0]).max(0.0);
        let room_in_clip = (hook_c.shot.clip_seconds - from - 0.05).max(0.0);
        let dur = end_len.min(room_in_clip.max(0.1));
        match run(
            ex,
            "timeline.place",
            json!({"item": item, "track": "V1", "time": ticks(e0), "sourceIn": ticks(from), "duration": ticks(dur), "audio": false}),
        )
        .await
        {
            Ok(r) => {
                if let Some(b) = r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64) {
                    run(ex, "clip.fillFrame", json!({"clips": [b]})).await.ok();
                    run(ex, "effects.apply", json!({"clips": [b], "effect": "gaussian_blur"})).await.ok();
                    run(ex, "effects.setParam", json!({"clip": b, "effect": "gaussian_blur", "param": "blurriness", "value": 70.0})).await.ok();
                    run(ex, "effects.setParam", json!({"clip": b, "effect": "gaussian_blur", "param": "repeat_edge", "value": true})).await.ok();
                    run(ex, "effects.setParam", json!({"clip": b, "effect": "opacity", "param": "opacity", "value": 40.0})).await.ok();
                }
            }
            Err(e) => notes.push(format!("end card: its footage: {e}")),
        }
    }
    let logo_path = opts.logo.clone().or_else(|| show.outro["logo"].as_str().map(str::to_string));
    let channel = opts.channel.clone().or_else(|| show.outro["handle"].as_str().or_else(|| show.outro["channel"].as_str()).map(str::to_string));
    let [lx0, ly0, lx1, ly1] = bx.logo;
    let (lc, side) = ([(lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0], (lx1 - lx0).min(ly1 - ly0));
    let mut logo_box: Option<[f64; 4]> = None;
    if let Some(path) = &logo_path {
        match place_logo(ex, logo_track, path, e0, end_len, lc, side * 0.86).await {
            Ok((clip, scale)) => {
                let badge = look::shape_clip(
                    ex,
                    badge_track,
                    FPS,
                    "ellipse",
                    lc,
                    [side, side],
                    e0,
                    end_len,
                    json!({"fill_color": "#f7f4ff", "stroke": false, "shadow": true, "shadow_color": "#7a2cff", "shadow_opacity": 85.0, "shadow_distance": 0.0, "shadow_size": 10.0 * u, "shadow_blur": 40.0 * u}),
                )
                .await;
                // the logo and its badge pop in on the hit
                pop(ex, clip, e0, scale).await;
                if let Ok(b) = badge {
                    pop(ex, b, e0, 100.0).await;
                }
                logo_box = Some(bx.logo);
            }
            Err(e) => notes.push(format!("end card: the logo {path}: {e}; left out")),
        }
    }
    {
        let [x0, y0, x1, _] = bx.end_text;
        let (cx, room) = ((x0 + x1) / 2.0, x1 - x0);
        let top = if logo_box.is_some() { y0 } else { bx.logo[1] + 0.3 * (bx.logo[3] - bx.logo[1]) };
        let size1 = 92.0 * u;
        let props = loud(&theme, Role::Title, &theme.text, u);
        let (clip, b1) = line(ex, labels_track, &opts.end_text, cx, top + size1 * 0.78, size1, room, props, e0 + 0.15, end_len - 0.15).await?;
        let mut b = b1;
        if let Some(ch) = &channel {
            let size2 = 84.0 * u;
            let props = loud(&theme, Role::Number, &theme.accent, u);
            b = b.union(add_line(ex, clip, ch, cx, b1.bottom + 30.0 * u + size2 * 0.74, size2, room, props).await?);
        }
        texts.push(("end card".into(), arr(b)));
    }
    if let Some(b) = logo_box {
        texts.push(("logo".into(), b));
    }
    // ---- the hook's voice: the show's first sentence, in its narrator's voice
    let mut voice_clips: Vec<u64> = Vec::new();
    let mut captioned = false;
    let has_narration = !show.narration.is_empty();
    if opts.hook_voice.unwrap_or(has_narration) {
        match show.sequence {
            Some(id) => match first_sentence(ex, id, &show.narration, &mut notes).await {
                Some((item, src, secs, sentence)) => {
                    let at = 2.0 / FPS;
                    let secs = secs.min(tm.end_card[0] - at - 0.2);
                    match run(
                        ex,
                        "timeline.place",
                        json!({"item": item, "track": "A2", "time": ticks(at), "sourceIn": ticks(src), "duration": ticks(secs), "video": false}),
                    )
                    .await
                    {
                        Ok(r) => {
                            if let Some(c) = r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64) {
                                run(ex, "sequence.applyAudioTransition", json!({"clip": c, "edge": "out", "effect": "constant_power", "frames": 4})).await.ok();
                                voice_clips.push(c);
                            }
                            made.voice = Some(json!({"item": item, "from": src, "seconds": secs, "at": at, "text": sentence}));
                            // the line it says, as Bold animated captions (or `captionStyle`)
                            if opts.captions != Some(false) {
                                captioned = hook_captions(ex, &sentence, at, secs, opts, qa_model, &mut notes).await;
                            }
                        }
                        Err(e) => notes.push(format!("hook voice: {e}")),
                    }
                }
                None => notes.push("hook voice: the show's opening narration was not found; the hook plays with music only".into()),
            },
            None => notes.push("hook voice: no show sequence to take the narration from".into()),
        }
    }
    // ---- opt-in: a creator look over the hook and the montage, zoom pulses on the montage's
    // beats (off unless asked: the trailer is as before without them)
    if let Some(l) = opts.look.filter(|l| *l != crate::creator_look::CreatorLook::Clean) {
        progress(&format!("the {} look", l.name()));
        match crate::creator_look::apply_look(ex, l, 0.0, tm.end_card[0], crate::creator_look::hex_color(&theme.accent)).await {
            Ok(_) => made.look = Some(l.name().to_string()),
            Err(e) => notes.push(format!("look {}: {e}", l.name())),
        }
    }
    if opts.beat_pulse {
        let clips: Vec<u64> = shots.iter().filter(|s| s.number >= 1).map(|s| s.clip).collect();
        let from = tm.shots.first().map_or(tm.hook[1], |s| s[0]);
        let on = beats.list(from, tm.end_card[0]);
        match crate::creator_look::beat_pulse(ex, &clips, Some(&on)).await {
            Ok(r) => made.beat_pulses = Some(r["clips"].as_array().map_or(0, |c| c.iter().filter_map(|x| x["pulses"].as_u64()).sum())),
            Err(e) => notes.push(format!("beat pulse: {e}")),
        }
    }
    // ---- a swish into the montage, another into the end card and an impact on its hit
    let mut fx: Vec<u64> = Vec::new();
    let hit = tm.end_card[0];
    for (kind, dur, at, track, intensity) in [
        ("swish", 0.5, tm.hook[1] - 0.25, "A4", 0.5),
        ("swish", 0.5_f64.min(hit), hit - 0.5_f64.min(hit), "A5", 0.6),
        ("impact", end_len.clamp(1.0, 3.0), hit, "A5", 0.8),
    ] {
        match run(
            ex,
            "audio.generate",
            json!({"kind": kind, "duration": dur, "time": ticks(at.max(0.0)), "track": track, "intensity": intensity, "seed": opts.seed}),
        )
        .await
        {
            Ok(r) => fx.extend(r["clip"].as_u64()),
            Err(e) => notes.push(format!("{kind}: {e}")),
        }
    }
    // ---- the mix: speech first, then the platform's loudness
    progress("mixing");
    let profile = MixProfile::from_plan(&show.mix).unwrap_or_default();
    let mix = Mix { sounds: Vec::new(), effects: fx, narrations: voice_clips, music: music_clip.into_iter().collect(), enhance_voice: false };
    let stems = mix::MixTracks { sounds: "A1".into(), voice: "A2".into(), music: "A3".into(), effects: vec!["A4".into(), "A5".into()] };
    made.mix = Some(mix::mix_down(ex, &mix, &stems, profile, false, &mut notes).await);
    // the music carries a trailer: up from the bed level the show's mix gives it (still ducked
    // under the voice), so normalising the whole does not push the voice into the limiter
    if let Some(m) = music_clip
        && let Err(e) = run(ex, "clip.audioGain", json!({"clips": [m], "mode": "adjust", "db": MUSIC_LIFT_DB})).await
    {
        notes.push(format!("music level: {e}"));
    }
    // its peaks held on its own track, and the voice a little under the bed's level (it stands
    // far above the ducked music still), so the Mix's limiter has next to nothing to do
    mix::inserts(ex, "A3", &[("hard_limiter", json!({"max": MUSIC_CEILING_DB, "boost": 0.0}))], &mut notes).await;
    if !mix.narrations.is_empty() {
        // after the voice's limiter (its track's fader): the peaks it lets through come down too
        let fader = made.mix.as_ref().and_then(|m| m["voice"]["faderDb"].as_f64()).filter(|f| f.is_finite()).unwrap_or(0.0);
        if let Err(e) = run(ex, "mixer.setStrip", json!({"strip": "A2", "volumeDb": fader + VOICE_TRIM_DB})).await {
            notes.push(format!("voice level: {e}"));
        }
    }
    match run(ex, "audio.normalizeLoudness", json!({"target": "social", "truePeak": look::TRUE_PEAK})).await {
        Ok(r) => made.loudness = Some(r),
        Err(e) => notes.push(format!("loudness: {e}")),
    }
    // ---- the platform's safe zones
    match run(ex, "sequence.checkSafeZones", json!({"platform": opts.platform.name()})).await {
        Ok(v) => {
            let overlaps = v["overlaps"].as_array().map_or(0, Vec::len);
            let off: Vec<&String> = texts.iter().filter(|(_, b)| !inside(*b, bx.safe)).map(|(n, _)| n).collect();
            for n in &off {
                notes.push(format!("safe zones: the {n} reaches out of {}'s safe area", opts.platform.label()));
            }
            made.safe_zones = Some(
                json!({"platform": opts.platform.name(), "ok": overlaps == 0 && off.is_empty(), "overlaps": overlaps, "outside": off, "texts": texts.iter().map(|(n, b)| json!({"what": n, "box": b})).collect::<Vec<_>>(), "safe": bx.safe}),
            );
        }
        Err(e) => notes.push(format!("safe zones: {e}")),
    }
    let cuts = tm.cuts();
    made.cuts = cuts.len();
    made.cuts_on_beat = cuts.iter().filter(|c| beats.distance(**c) <= 1.0 / FPS + 0.012).count();
    made.seconds = tm.total;
    made.shots = shots;
    made.timing = Some(tm);
    // ---- the files
    if let Some(out) = opts.export.clone() {
        progress(&format!("exporting {out}"));
        let mut ep = json!({"sequence": seq, "path": out, "wait": true, "burnCaptions": captioned});
        match opts.preset.clone().or_else(|| (opts.frame == (1080, 1920)).then(|| VERTICAL_PRESET.to_string())) {
            Some(pr) => ep["preset"] = json!(pr),
            None => ep["format"] = json!("h264"),
        }
        let r = run(ex, "file.exportMedia", ep).await?;
        let written = r["path"].as_str().or_else(|| r["result"]["path"].as_str()).map(str::to_string).unwrap_or_else(|| out.clone());
        made.export = Some(json!({"path": written, "result": r["result"], "preset": r["preset"]}));
        // the cover: the hook's frame with its title
        let at = on_frame(made.timing.as_ref().map_or(1.0, |t| t.hook[0] + 0.6 * (t.hook[1] - t.hook[0])), FPS);
        let cover = beside(&written, "cover.jpg");
        match ex.render(at, opts.frame.0.max(opts.frame.1)).await.map(|png| jpeg(&png, &cover)) {
            Some(Ok(())) => made.cover = Some(cover.display().to_string()),
            Some(Err(e)) => notes.push(format!("cover: {e}")),
            None => notes.push("cover: the frame could not be rendered".into()),
        }
        // the caption
        let events: Vec<String> = show.entries.iter().filter_map(|e| e.event.clone().or_else(|| e.place.clone())).collect();
        let picked: Vec<&Candidate> = std::iter::once(&hook_c).chain(montage.iter()).collect();
        let language = plan_language(show);
        let (text, tags) = caption(&(t1.clone(), t2.clone()), &picked, &opts.include, &events, channel.as_deref(), language.as_deref(), &opts.end_text);
        // the caption as it is pasted, and beside it the same with the link for the profile's bio
        let cap = beside(&written, "caption.txt");
        match std::fs::write(&cap, format!("{text}\n")) {
            Ok(()) => made.caption = Some(cap.display().to_string()),
            Err(e) => notes.push(format!("caption: {}: {e}", cap.display())),
        }
        let info = json!({"caption": text, "hashtags": tags, "link": opts.link, "linkGoesIn": "the profile's bio (a Reel's caption shows no links)", "robots": picked.iter().map(|c| &c.label).collect::<Vec<_>>()});
        let cap_json = beside(&written, "caption.json");
        if let Err(e) = std::fs::write(&cap_json, format!("{}\n", serde_json::to_string_pretty(&info).unwrap_or_default())) {
            notes.push(format!("caption: {}: {e}", cap_json.display()));
        }
        made.caption_text = Some(text);
        made.hashtags = tags;
        // the frame sheet, from the file: the hook, the end card and the shots between (all of
        // them up to [`SHEET_FRAMES`], else spread evenly over the montage)
        let times: Vec<f64> = made.timing.as_ref().map(sheet_times).unwrap_or_default();
        let sheet = beside(&written, "frames.jpg");
        match frames_sheet(&written, &times, &sheet).await {
            Ok(()) => made.frames = Some(sheet.display().to_string()),
            Err(e) => notes.push(format!("frames: {e}")),
        }
    }
    // ---- QA
    let written = made.export.as_ref().and_then(|e| e["path"].as_str()).map(str::to_string);
    if opts.qa.unwrap_or(written.is_some()) {
        let shots_label = format!("V{}", shots_track + 1);
        let q = qa::Options {
            subject: "robot".into(),
            entries: layout.clone(),
            picture_track: shots_label,
            privacy: made.privacy.clone(),
            // the faces the trailer still shows, when it hides faces
            faces: redact.as_ref().is_some_and(|r| r.faces),
            platform: Some(opts.platform.name().into()),
            // a trailer's music is meant to come up after the voice: the voice is measured against
            // the music while it speaks (the mix check's speech over music), not against what follows
            voice_margin: -60.0,
            ..qa::Options::default()
        };
        let mut report = qa::check(ex, &q, qa_model, progress).await;
        report.findings.extend(carried_over(&show.privacy, &show.layout, &made.privacy, &made.shots, &made.carried));
        let missed = cuts.iter().filter(|c| beats.distance(**c) > 1.0 / FPS + 0.012).count();
        if missed > 0 {
            report.findings.push(Finding {
                check: "beats",
                severity: Severity::Warning,
                seconds: 0.0,
                until: None,
                entry: None,
                clip: None,
                message: format!("{missed} of {} cuts are off the music's beat", cuts.len()),
            });
        }
        if let Some(sz) = &made.safe_zones
            && sz["ok"] == json!(false)
        {
            report.findings.push(Finding {
                check: "safe_zones",
                severity: Severity::Warning,
                seconds: 0.0,
                until: None,
                entry: None,
                clip: None,
                message: format!("text or the logo reaches out of {}'s safe area: {}", opts.platform.label(), sz["outside"]),
            });
        }
        report.tally();
        if let Some(path) = &written
            && let Err(e) = qa::write_next_to(&mut report, Path::new(path))
        {
            notes.push(format!("QA: {e}"));
        }
        progress(&format!("QA: {}", report.summary()));
        made.qa_summary = Some(report.summary());
        made.qa = serde_json::to_value(&report).ok();
    }
    made.notes = notes;
    Ok(made)
}

/// The plan's narration language, for the caption (from the show's report narration being in
/// Malayalam script, when the plan does not say).
fn plan_language(show: &Show) -> Option<String> {
    let malayalam = show.narration.iter().take(5).any(|c| c.text.chars().any(|ch| ('\u{0D00}'..='\u{0D7F}').contains(&ch)));
    malayalam.then(|| "ml".to_string())
}

/// Scale `clip` (at `scale` %) from 85 % of its size to all of it over the first quarter second
/// from `t`.
async fn pop(ex: &dyn Executor, clip: u64, t: f64, scale: f64) {
    for (at, k) in [(t, 0.85), (on_frame(t + 0.27, FPS), 1.0)] {
        if run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": scale * k, "time": ticks(at), "keyframe": true}))
            .await
            .is_err()
        {
            return;
        }
    }
    run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": "motion", "param": "scale", "time": ticks(t), "interpolation": "easeOut"})).await.ok();
}

/// Import the logo image at `path` and place it on track `track` (an index) from `t` for `dur`
/// seconds, `px` across its longer side, centred on `c`: its clip and Motion scale.
#[allow(clippy::too_many_arguments)]
async fn place_logo(ex: &dyn Executor, track: usize, path: &str, t: f64, dur: f64, c: [f64; 2], px: f64) -> Result<(u64, f64), AutoCutError> {
    let r = run(ex, "file.import", json!({"paths": [path], "reuse": true})).await?;
    let item =
        r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| {
            AutoCutError::Other(r["errors"].as_array().and_then(|a| a.first()).map_or_else(|| "nothing imported".to_string(), Value::to_string))
        })?;
    let (t0, t1) = (on_frame(t, FPS), on_frame(t + dur, FPS));
    let r = run(ex, "timeline.place", json!({"item": item, "track": format!("V{}", track + 1), "time": ticks(t0), "duration": ticks(t1 - t0), "audio": false}))
        .await?;
    let clip =
        r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| AutoCutError::Executor("timeline.place placed no logo".into()))?;
    let fit = run(ex, "clip.fitToFrame", json!({"clips": [clip]})).await?;
    let fitted = fit["clips"].as_array().and_then(|a| a.iter().find(|x| x["clip"].as_u64() == Some(clip))).and_then(|x| x["scale"].as_f64());
    let (iw, ih) = media_size(ex, item).await.unwrap_or((1.0, 1.0));
    let q = ex.execute("sequence.inspect", json!({})).await.ok();
    let (fw, fh) = q.and_then(|s| Some((s["settings"]["width"].as_f64()?, s["settings"]["height"].as_f64()?))).unwrap_or((1080.0, 1920.0));
    let longer = iw.max(ih) * (fw / iw).min(fh / ih);
    let scale = match fitted {
        Some(s) if s.is_finite() && s > 0.0 && longer > 0.0 => s * px / longer,
        _ => 100.0,
    };
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": scale})).await?;
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "position", "value": c})).await?;
    Ok((clip, scale))
}

/// Most frames on the review sheet.
pub const SHEET_FRAMES: usize = 16;

/// The review sheet's moments: the middle of the hook, of the shots (every one when they fit in
/// [`SHEET_FRAMES`] with the hook and the end card, else that many spread evenly) and of the end
/// card (a little after, the logo in).
pub fn sheet_times(t: &Timing) -> Vec<f64> {
    let room = SHEET_FRAMES - 2;
    let n = t.shots.len();
    let picks: Vec<usize> = if n <= room { (0..n).collect() } else { (0..room).map(|k| k * (n - 1) / (room - 1).max(1)).collect() };
    std::iter::once((t.hook[0] + t.hook[1]) / 2.0)
        .chain(picks.iter().filter_map(|&i| t.shots.get(i)).map(|s| (s[0] + s[1]) / 2.0))
        .chain(std::iter::once((t.end_card[0] + t.end_card[1]) / 2.0 + 0.3))
        .collect()
}

/// A PNG frame as a JPEG at `path`.
fn jpeg(png: &[u8], path: &Path) -> Result<(), String> {
    let img = image::load_from_memory(png).map_err(|e| e.to_string())?.to_rgb8();
    let mut out = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92).encode_image(&img).map_err(|e| e.to_string())?;
    std::fs::write(path, out.into_inner()).map_err(|e| format!("{}: {e}", path.display()))
}

/// A sheet of the exported file's frames at `times` (`media.contactSheet` in a session of its
/// own, so the project is not touched), time-labelled, at `out`.
pub(crate) async fn frames_sheet(video: &str, times: &[f64], out: &Path) -> Result<(), String> {
    let (video, out, times) = (video.to_string(), out.display().to_string(), times.to_vec());
    tokio::task::spawn_blocking(move || {
        let mut s = filmcraft_engine::Session::default();
        let r = s.execute("file.import", json!({"paths": [video]})).map_err(|e| e.to_string())?;
        let item = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or("the export could not be read back")?;
        let columns = times.len().div_ceil(2).clamp(1, 10);
        s.execute("media.contactSheet", json!({"items": [{"item": item, "at": times}], "path": out, "labels": "time", "scale": 0.25, "columns": columns}))
            .map(drop)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

#[cfg(test)]
#[path = "trailer_tests.rs"]
mod tests;
