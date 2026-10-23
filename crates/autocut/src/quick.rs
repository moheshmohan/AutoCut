//! Quick templates (`autocut.quick`): an album, a folder or the selected clips → a finished
//! vertical Short or Reel, a template picked and filled from what the clips show and say.
//!
//! 1. **Media.** `items`, and `paths` (files, or folders whose media files are taken, sorted by
//!    name) imported with `file.import` (what the project holds is reused); none given: every
//!    movie and photo of the project.
//! 2. **Index.** `media.analyzeScenes` (`detail: "fast"`) for every item, then `detail: "full"`
//!    again only for the items holding the most promising shots, at most [`FULL_DETAIL_SHOTS`]
//!    shots in all, when the full describer is installed — the richer descriptions where they
//!    matter without paying for them everywhere. Without the scene models the index still has the
//!    cuts (the luma and histogram detector) and the picture quality; without any index each clip
//!    is one shot.
//! 3. **Speech.** `media.understandSpeech` for the movies (the clips without sound are skipped):
//!    where speech is, the words, the takes kept, a talking-head hint.
//! 4. **Template.** `template` names one, or `"auto"` ([`choose_template`]) scores the templates
//!    that have the `aspect`: the AI-ready ones first, how many shots they need against how many
//!    there are, whether they want someone talking (and whether anyone does), the shots' energy
//!    against the slots', the length against `seconds` and the topic against their tags.
//! 5. **Shots.** Each picture slot's media ([`assign`]) gets the moment that suits it best: the
//!    scene index's score for the slot's `purpose` (`media.bestShots`' rules), its `query`
//!    (`media.searchScenes`), `energy`, `speech`, `minQuality` and the shot's length; no moment
//!    is used twice (a long shot gives several), picks of one clip are kept few, and the slots
//!    that ask for it keep the filming order. A talking slot takes stretches of kept speech.
//! 6. **Trim and frame.** Each pick plays from its best moment (the keyframe, or the start of the
//!    kept speech), framed around the subject's box for the vertical crop.
//! 7. **Text.** The text slots are written by the local model from the shots' descriptions (the
//!    full ones' paragraphs), the kept speech and `topic` ([`write_texts`]); without a model, or
//!    where it says nothing usable, from the descriptions and the topic ([`fallback_text`]).
//! 8. **Apply and finish.** `template.apply` (music generated at a tempo that brings the template
//!    to `seconds` when it cuts on the beat, or `music`), the placeholders of optional slots left
//!    empty removed, `look` swapped in, the captions animated in the `platform`'s safe zone, faces
//!    and plates hidden (`redact`; never the speaker's own face), the safe zones checked, and with
//!    `export` the video, `<out>.sheet.jpg` (12 frames with their times), `<out>.report.json` and
//!    QA (`<out>.qa.json`).
//!
//! Everything works without the models: what was missing and what was used instead is in the
//! report's `fallbacks`. The engine commands that run models take the GPU lock themselves (the
//! scene index, speech understanding); the job does not hold it around them (a second take of
//! the lock would wait on itself), and Ollama queues its own requests.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use filmcraft_engine::project::scenes::{CameraMove, Energy, Shot, ShotDescription, ShotQuality, ShotUse};
use filmcraft_engine::safe_zones::Platform;
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::Serialize;
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

/// The length asked for by default, seconds.
pub const DEFAULT_SECONDS: f64 = 30.0;
/// Shots described again in full detail, at most.
pub const FULL_DETAIL_SHOTS: usize = 12;
/// Media one quick short looks at, at most.
pub const MAX_MEDIA: usize = 500;
/// Shots ranked, at most (a long library is cut down to its best).
const MAX_SHOTS: usize = 2000;
/// Picks one template asks for, at most.
const MAX_PICKS: usize = 256;
/// Two picks of one clip may overlap this much, seconds.
const OVERLAP: f64 = 0.25;
/// Media files a folder gives, at most.
const MAX_FOLDER_FILES: usize = 500;
/// The extensions of media files taken from a folder.
const MEDIA_EXTENSIONS: &[&str] =
    &["mp4", "mov", "m4v", "mkv", "webm", "avi", "mts", "m2ts", "mxf", "jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "webp", "dng"];
/// The tempo a generated bed may be slowed or sped to, beats a minute.
const BPM_RANGE: (f64, f64) = (72.0, 168.0);
/// Frames on the review sheet.
pub const SHEET_FRAMES: usize = 12;

fn invalid(m: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(format!("quick: {}", m.into()))
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|f| f.is_finite())
}

fn secs_of_ticks(v: &Value) -> Option<f64> {
    v.as_i64().map(|t| t as f64 / TICKS_PER_SECOND as f64).filter(|f| f.is_finite())
}

/// At most `n` characters of `s`.
fn chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// A layout's frame shape, width / height (`9:16` → 0.5625), when it reads as one.
pub fn aspect_ratio(aspect: &str) -> Option<f64> {
    let (w, h) = aspect.split_once(':')?;
    let (w, h) = (w.trim().parse::<f64>().ok()?, h.trim().parse::<f64>().ok()?);
    (w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0).then(|| w / h).filter(|r| (0.1..=10.0).contains(r))
}

// ---------------------------------------------------------------------------------------------
// the media
// ---------------------------------------------------------------------------------------------

/// Whether `path` names a media file by its extension.
pub fn is_media_file(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| MEDIA_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// `paths` with each folder replaced by its media files (sorted by name, at most
/// [`MAX_FOLDER_FILES`] a folder; hidden files left out); files stay as they are.
pub fn expand_paths(paths: &[String]) -> Result<Vec<String>, AutoCutError> {
    let mut out = Vec::new();
    for p in paths.iter().take(MAX_MEDIA) {
        let path = PathBuf::from(filmcraft_engine::export_tools::expand_home(p));
        if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.is_file() && is_media_file(f) && !f.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')))
                .collect();
            files.sort();
            if files.is_empty() {
                return Err(invalid(format!("{} has no video or photo files", path.display())));
            }
            out.extend(files.into_iter().take(MAX_FOLDER_FILES).map(|f| f.to_string_lossy().to_string()));
        } else {
            out.push(path.to_string_lossy().to_string());
        }
        if out.len() >= MAX_MEDIA {
            out.truncate(MAX_MEDIA);
            break;
        }
    }
    Ok(out)
}

/// A media item of the project: its kind and length.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MediaItem {
    pub item: u64,
    pub name: String,
    pub still: bool,
    /// Seconds (0 for a still).
    pub seconds: f64,
    /// Filming order (capture time, else import order).
    pub order: usize,
}

/// The movies and photos of a `project.inspect` answer, in bin order.
pub fn project_media(tree: &Value) -> Vec<MediaItem> {
    fn walk(v: &Value, out: &mut Vec<MediaItem>, depth: usize) {
        if depth > 64 || out.len() >= MAX_MEDIA {
            return;
        }
        let kind = v["type"].as_str().unwrap_or_default();
        if matches!(kind, "Movie" | "Still Image")
            && let Some(item) = v["item"].as_u64()
        {
            let still = kind == "Still Image";
            let seconds = if still { 0.0 } else { secs_of_ticks(&v["duration"]).unwrap_or(0.0).max(0.0) };
            out.push(MediaItem { item, name: chars(v["name"].as_str().unwrap_or_default(), 200), still, seconds, order: out.len() });
        }
        for c in v["children"].as_array().map(Vec::as_slice).unwrap_or_default() {
            walk(c, out, depth + 1);
        }
    }
    let mut out = Vec::new();
    walk(&tree["root"], &mut out, 0);
    out
}

// ---------------------------------------------------------------------------------------------
// shots and speech
// ---------------------------------------------------------------------------------------------

/// One shot of the scene index, ready to rank.
#[derive(Clone, Debug)]
pub struct ShotInfo {
    pub item: u64,
    pub name: String,
    pub still: bool,
    /// Its index in the item's scene index.
    pub index: usize,
    /// Media seconds.
    pub start: f64,
    pub end: f64,
    /// Its best moment (the first keyframe; the middle without one).
    pub keyframe: f64,
    /// The index's data as the ranking reads it.
    pub shot: Shot,
    /// Filming order of its item.
    pub order: usize,
    /// The share of it with kept speech (0..1).
    pub speech: f64,
    /// Its item is someone talking to the camera.
    pub talking: bool,
    /// Described in full detail.
    pub full: bool,
}

impl ShotInfo {
    pub fn seconds(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
    pub fn description(&self) -> Option<&ShotDescription> {
        self.shot.description.as_ref()
    }
    /// What it shows, in words.
    pub fn caption(&self) -> String {
        self.description().map(|d| d.caption.trim().to_string()).filter(|c| !c.is_empty()).unwrap_or_else(|| self.name.clone())
    }
}

/// The shots of one `media.scenes` answer for `m` (a shot without a sane span is left out; an
/// item that has no index is one shot, its whole length).
pub fn parse_shots(m: &MediaItem, scenes: Option<&Value>) -> Vec<ShotInfo> {
    let mut out = Vec::new();
    let full = scenes.is_some_and(|v| v["detail"] == "full");
    let shots = scenes.and_then(|v| v["shots"].as_array()).map(Vec::as_slice).unwrap_or_default();
    for (k, s) in shots.iter().enumerate().take(MAX_SHOTS) {
        let (Some(a), Some(b)) = (finite(&s["start"]), finite(&s["end"])) else { continue };
        if a < 0.0 || b < a || b > 86_400.0 {
            continue;
        }
        let key = s["keyframes"].as_array().and_then(|k| k.first()).and_then(finite).filter(|t| *t >= a && *t <= b).unwrap_or((a + b) / 2.0);
        let quality: ShotQuality = serde_json::from_value(s["quality"].clone()).unwrap_or_default();
        let camera: CameraMove = serde_json::from_value(s["camera"].clone()).unwrap_or_default();
        let description: Option<ShotDescription> = s.get("description").filter(|d| d.is_object()).and_then(|d| serde_json::from_value(d.clone()).ok());
        let shot = Shot {
            start: Tick::from_seconds_f64(a),
            end: Tick::from_seconds_f64(b),
            keyframes: vec![Tick::from_seconds_f64(key)],
            quality,
            camera,
            description,
            ..Shot::default()
        };
        let index = s["shot"].as_u64().and_then(|n| usize::try_from(n).ok()).unwrap_or(k);
        out.push(ShotInfo {
            item: m.item,
            name: m.name.clone(),
            still: m.still,
            index,
            start: a,
            end: b,
            keyframe: key,
            shot,
            order: m.order,
            speech: 0.0,
            talking: false,
            full,
        });
    }
    if out.is_empty() {
        let end = if m.still { 0.0 } else { m.seconds.max(0.0) };
        let shot = Shot { start: Tick::ZERO, end: Tick::from_seconds_f64(end), keyframes: vec![Tick::from_seconds_f64(end / 2.0)], ..Shot::default() };
        out.push(ShotInfo {
            item: m.item,
            name: m.name.clone(),
            still: m.still,
            index: 0,
            start: 0.0,
            end,
            keyframe: end / 2.0,
            shot,
            order: m.order,
            speech: 0.0,
            talking: false,
            full: false,
        });
    }
    out
}

/// A stretch of kept speech.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Said {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// How cleanly it was said, 0..1.
    pub fluency: f64,
}

/// What speech understanding found in one item.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ItemSpeech {
    pub item: u64,
    pub language: String,
    /// The recogniser and how speech was found.
    pub source: String,
    pub vad: String,
    /// `talkingHead`, `voice`, `bRoll` or `unknown`.
    pub shot: String,
    pub share: f64,
    /// The kept segments (the takes a retake drops left out), in order.
    pub kept: Vec<Said>,
    /// Takes dropped.
    pub dropped: usize,
}

impl ItemSpeech {
    /// Seconds of kept speech.
    pub fn kept_seconds(&self) -> f64 {
        self.kept.iter().map(|s| (s.end - s.start).max(0.0)).sum()
    }
    /// Words of the kept speech.
    pub fn words(&self) -> usize {
        self.kept.iter().map(|s| s.text.split_whitespace().count()).sum()
    }
    /// Someone talking to the camera (a face on screen) with something to say: at least
    /// [`TALK_SECONDS`] and [`TALK_WORDS`] of believable speech.
    pub fn talking(&self) -> bool {
        self.shot == "talkingHead" && self.kept_seconds() >= TALK_SECONDS && self.words() >= TALK_WORDS
    }
}

/// The least believable speech that makes a clip someone talking, seconds and words.
pub const TALK_SECONDS: f64 = 6.0;
pub const TALK_WORDS: usize = 12;

/// What a recogniser writes for noise, music or a crowd: never speech to build on.
const HALLUCINATIONS: &[&str] =
    &["thank you", "thanks for watching", "thank you for watching", "thank you so much", "you", "bye", "the", "so", "okay", "please subscribe", "subscribe"];

/// Whether a recognised segment is believable speech: confident enough, not one of the phrases a
/// recogniser writes for noise, not one word over and over ("mga mga mga").
pub fn believable(text: &str, confidence: f64) -> bool {
    let words: Vec<String> =
        text.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).filter(|w| !w.is_empty()).take(2000).collect();
    if words.is_empty() || confidence < 0.35 {
        return false;
    }
    if HALLUCINATIONS.contains(&words.join(" ").as_str()) {
        return false;
    }
    let distinct: BTreeSet<&String> = words.iter().collect();
    !(words.len() >= 3 && (distinct.len() as f64) < 0.5 * words.len() as f64)
}

/// The analyses of a `media.inspectSpeech` answer.
pub fn parse_speech(v: &Value) -> Vec<ItemSpeech> {
    let mut out = Vec::new();
    for a in v["items"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(MAX_MEDIA) {
        let Some(item) = a["item"].as_u64() else { continue };
        let dropped: BTreeSet<u64> = a["droppedTakes"].as_array().map(|d| d.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let mut kept = Vec::new();
        for (k, s) in a["segments"].as_array().map(Vec::as_slice).unwrap_or_default().iter().enumerate().take(20_000) {
            let index = s["index"].as_u64().unwrap_or(k as u64);
            if dropped.contains(&index) {
                continue;
            }
            let (Some(x), Some(y)) = (finite(&s["start"]), finite(&s["end"])) else { continue };
            if x < 0.0 || y <= x {
                continue;
            }
            let text = s["text"].as_str().unwrap_or_default();
            if !believable(text, finite(&s["confidence"]).unwrap_or(1.0)) {
                continue;
            }
            kept.push(Said {
                start: x,
                end: y,
                text: chars(s["text"].as_str().unwrap_or_default().trim(), 1000),
                fluency: finite(&s["fluency"]).unwrap_or(0.5).clamp(0.0, 1.0),
            });
        }
        kept.sort_by(|a, b| a.start.total_cmp(&b.start));
        out.push(ItemSpeech {
            item,
            language: chars(a["language"].as_str().unwrap_or_default(), 16),
            source: chars(a["source"].as_str().unwrap_or_default(), 64),
            vad: chars(a["vad"].as_str().unwrap_or_default(), 16),
            shot: chars(a["shot"].as_str().unwrap_or("unknown"), 16),
            share: finite(&a["speechShare"]).unwrap_or(0.0).clamp(0.0, 1.0),
            kept,
            dropped: dropped.len(),
        });
    }
    out
}

/// Mark each shot with the share of it that is kept speech and whether its item is a talking
/// head.
pub fn mark_speech(shots: &mut [ShotInfo], speech: &[ItemSpeech]) {
    let by_item: HashMap<u64, &ItemSpeech> = speech.iter().map(|s| (s.item, s)).collect();
    for s in shots.iter_mut() {
        let Some(sp) = by_item.get(&s.item) else { continue };
        s.talking = sp.talking();
        let len = s.seconds();
        if len <= 0.0 {
            continue;
        }
        let said: f64 = sp.kept.iter().map(|k| (k.end.min(s.end) - k.start.max(s.start)).max(0.0)).sum();
        s.speech = (said / len).clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------------------------
// templates
// ---------------------------------------------------------------------------------------------

/// A slot's AI hints as `template.list` / `template.inspect` show them.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Hints {
    pub purpose: Option<String>,
    pub query: String,
    pub energy: Option<String>,
    pub speech: Option<String>,
    pub min_quality: Option<f64>,
    pub chronological: bool,
    pub prompt: String,
    pub max_chars: Option<u32>,
    pub about: String,
}

impl Hints {
    pub fn of(v: &Value) -> Option<Hints> {
        let o = v.as_object()?;
        let s = |k: &str| o.get(k).and_then(Value::as_str).map(|x| chars(x.trim(), 1000)).filter(|x| !x.is_empty());
        Some(Hints {
            purpose: s("purpose"),
            query: s("query").unwrap_or_default(),
            energy: s("energy"),
            speech: s("speech"),
            min_quality: o.get("minQuality").and_then(finite).map(|q| q.clamp(0.0, 1.0)),
            chronological: o.get("chronological").and_then(Value::as_bool).unwrap_or(false),
            prompt: s("prompt").unwrap_or_default(),
            max_chars: o.get("maxChars").and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok()),
            about: s("about").unwrap_or_default(),
        })
    }
    fn wants_speech(&self) -> bool {
        self.speech.as_deref() == Some("want")
    }
}

/// A slot of a template.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SlotInfo {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub prompt: String,
    pub count: usize,
    pub min_count: usize,
    pub max_chars: Option<u32>,
    pub ai: Option<Hints>,
    /// `template.inspect`: each placement's media index, where it starts in the template and how
    /// long it is (seconds, at the template's own tempo).
    pub placements: Vec<Place>,
}

/// Where one placement of a slot is in the template.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Place {
    pub media: usize,
    /// Seconds into the template (None when the template did not say).
    pub start: Option<f64>,
    pub seconds: f64,
}

impl SlotInfo {
    pub fn picture(&self) -> bool {
        matches!(self.kind.as_str(), "video" | "photo")
    }
    fn hints(&self) -> Hints {
        self.ai.clone().unwrap_or_default()
    }
    /// The longest placement of media `m`, seconds.
    pub fn need(&self, m: usize) -> f64 {
        self.placements.iter().filter(|p| p.media == m).map(|p| p.seconds).fold(0.0, f64::max)
    }
}

/// A template as the choice and the fill read it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TemplateInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub seconds: f64,
    pub aspects: Vec<String>,
    pub ai_ready: bool,
    pub captions: bool,
    pub bpm: Option<f64>,
    pub slots: Vec<SlotInfo>,
}

impl TemplateInfo {
    /// Its picture slots' media, the picks it needs.
    pub fn needs(&self) -> usize {
        self.slots.iter().filter(|s| s.picture()).map(|s| s.count).sum()
    }
    /// It wants someone talking: a slot that wants speech, or captions of what is said.
    pub fn wants_speech(&self) -> bool {
        self.captions || self.slots.iter().any(|s| s.ai.as_ref().is_some_and(Hints::wants_speech))
    }
    /// The mean energy its picture slots ask for (0 calm … 1 high), when they say.
    pub fn energy(&self) -> Option<f64> {
        let e: Vec<f64> = self.slots.iter().filter_map(|s| s.ai.as_ref()?.energy.as_deref().map(energy_level)).collect();
        (!e.is_empty()).then(|| e.iter().sum::<f64>() / e.len() as f64)
    }
}

fn energy_level(s: &str) -> f64 {
    match s {
        "low" => 0.0,
        "high" => 1.0,
        _ => 0.5,
    }
}

/// One template of a `template.list` (or a `template.inspect`) answer.
pub fn parse_template(t: &Value) -> Option<TemplateInfo> {
    let id = t["id"].as_str()?.to_string();
    let strs =
        |v: &Value| -> Vec<String> { v.as_array().map(|a| a.iter().filter_map(Value::as_str).take(64).map(|s| chars(s, 200)).collect()).unwrap_or_default() };
    let slots = t["slots"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .take(64)
        .filter_map(|s| {
            let count = s["count"].as_u64().map_or(1, |n| n.clamp(1, 256) as usize);
            let placements = s["placements"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .take(256)
                        .map(|p| Place {
                            media: p["media"].as_u64().map_or(0, |m| m.min(255) as usize),
                            start: secs_of_ticks(&p["start"]).filter(|s| (0.0..=86_400.0).contains(s)),
                            seconds: finite(&p["seconds"]).unwrap_or(0.0).clamp(0.0, 3600.0),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Some(SlotInfo {
                id: s["id"].as_str()?.to_string(),
                kind: s["kind"].as_str().unwrap_or_default().to_string(),
                name: chars(s["name"].as_str().unwrap_or_default(), 200),
                prompt: chars(s["prompt"].as_str().unwrap_or_default(), 1000),
                count,
                min_count: s["minCount"].as_u64().map_or(0, |n| (n as usize).min(count)),
                max_chars: s["constraints"]["maxChars"].as_u64().and_then(|n| u32::try_from(n).ok()),
                ai: Hints::of(&s["ai"]),
                placements,
            })
        })
        .collect();
    Some(TemplateInfo {
        id,
        name: chars(t["name"].as_str().unwrap_or_default(), 200),
        description: chars(t["description"].as_str().unwrap_or_default(), 2000),
        category: chars(t["category"].as_str().unwrap_or_default(), 100),
        tags: strs(&t["tags"]),
        seconds: finite(&t["seconds"]).unwrap_or(0.0).clamp(0.0, 3600.0),
        aspects: strs(&t["aspects"]),
        ai_ready: t["aiReady"].as_bool().unwrap_or(false),
        captions: t["captions"].as_bool().unwrap_or(false),
        bpm: finite(&t["beatGrid"]["bpm"]).filter(|b| (20.0..=300.0).contains(b)),
        slots,
    })
}

/// What the media is like, for the template choice.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Facts {
    pub items: usize,
    pub videos: usize,
    pub stills: usize,
    pub shots: usize,
    /// Shots with a description.
    pub described: usize,
    /// Someone talks to the camera (with this much kept speech, seconds).
    pub talking: bool,
    pub speech_seconds: f64,
    /// The described shots' mean energy (0 calm … 1 high).
    pub energy: Option<f64>,
    /// Words of the topic and of the shots' subjects.
    pub words: Vec<String>,
}

/// The facts of `shots` and `speech`, with `topic`'s words.
pub fn facts(media: &[MediaItem], shots: &[ShotInfo], speech: &[ItemSpeech], topic: Option<&str>) -> Facts {
    let energies: Vec<f64> =
        shots.iter().filter_map(|s| s.description()).map(|d| d.energy).filter(|e| *e != Energy::Unknown).map(|e| f64::from(e.level())).collect();
    let mut words: Vec<String> = topic.map(words_of).unwrap_or_default();
    for s in shots.iter().filter_map(|s| s.description()).flat_map(|d| d.subjects.iter()).take(200) {
        for w in words_of(&s.name) {
            if !words.contains(&w) {
                words.push(w);
            }
        }
    }
    words.truncate(64);
    let talkers: Vec<&ItemSpeech> = speech.iter().filter(|s| s.talking()).collect();
    Facts {
        items: media.len(),
        videos: media.iter().filter(|m| !m.still).count(),
        stills: media.iter().filter(|m| m.still).count(),
        shots: shots.len(),
        described: shots.iter().filter(|s| s.description().is_some()).count(),
        talking: !talkers.is_empty(),
        speech_seconds: talkers.iter().map(|s| s.kept_seconds()).sum(),
        energy: (!energies.is_empty()).then(|| energies.iter().sum::<f64>() / energies.len() as f64),
        words,
    }
}

/// Lower-case words of `s` without a plural `s`, of three letters or more, not the commonest.
pub fn words_of(s: &str) -> Vec<String> {
    const STOP: &[&str] = &["the", "and", "with", "for", "from", "this", "that", "its", "are", "was", "has", "into", "over", "video", "short", "shot"];
    let mut out = Vec::new();
    for w in s.split(|c: char| !c.is_alphanumeric()) {
        let w = w.to_lowercase();
        let w = w.strip_suffix('s').filter(|r| r.chars().count() >= 3).map(str::to_string).unwrap_or(w);
        if w.chars().count() >= 3 && !STOP.contains(&w.as_str()) && !out.contains(&w) {
            out.push(w);
        }
        if out.len() >= 32 {
            break;
        }
    }
    out
}

/// A template's score and why.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Scored {
    pub id: String,
    pub name: String,
    pub score: f64,
    pub reasons: Vec<String>,
    /// The length it will have (its music's tempo moved to come nearest `seconds`).
    pub seconds: f64,
}

/// The tempo bringing a `bpm` template of `seconds` nearest `target` seconds (each cut keeps its
/// beat number), within [`BPM_RANGE`], and the length it gives.
pub fn stretch(bpm: f64, seconds: f64, target: f64) -> (f64, f64) {
    if !(bpm.is_finite() && bpm > 0.0 && seconds.is_finite() && seconds > 0.0 && target.is_finite() && target > 0.0) {
        return (bpm, seconds);
    }
    let want = (bpm * seconds / target).clamp(BPM_RANGE.0, BPM_RANGE.1);
    let want = (want * 10.0).round() / 10.0;
    (want, seconds * bpm / want)
}

/// Score template `t` for media `f` (see the module docs): higher is better.
pub fn score_template(t: &TemplateInfo, f: &Facts, target: f64) -> Scored {
    let mut score = 0.0;
    let mut why = Vec::new();
    if t.ai_ready {
        score += 1.0;
        why.push("made to fill itself (AI hints on every slot)".to_string());
    }
    // enough shots for its places
    let need = t.needs().max(1);
    let r = f.shots as f64 / need as f64;
    if r >= 1.0 {
        score += 0.6;
        why.push(format!("{} shot(s) for its {need} place(s)", f.shots));
        if need <= 2 && f.shots >= 10 && !t.wants_speech() {
            score -= 0.4;
            why.push(format!("uses only {need} of {} shots", f.shots));
        }
    } else if r >= 0.5 {
        score += 0.6 * r - 0.1;
        why.push(format!("{} shot(s) for {need} places: some come from one long shot", f.shots));
    } else {
        score -= 0.5;
        why.push(format!("only {} shot(s) for {need} places", f.shots));
    }
    // someone talking
    match (t.wants_speech(), f.talking) {
        (true, true) => {
            score += 1.2;
            why.push(format!("someone talks to the camera ({:.0} s kept), and it is made for that", f.speech_seconds));
        }
        (true, false) => {
            score -= 1.5;
            why.push("it wants someone talking to the camera; nobody does".into());
        }
        (false, true) => {
            score -= 0.3;
            why.push("it does not use the speech".into());
        }
        (false, false) => {}
    }
    // energy
    if let (Some(te), Some(me)) = (t.energy(), f.energy) {
        let close = 1.0 - (te - me).abs();
        score += 0.4 * close;
        if close >= 0.75 {
            why.push(format!("the shots' energy ({}) suits its slots", energy_words(me)));
        }
    }
    // length
    let seconds = match t.bpm {
        Some(b) => stretch(b, t.seconds, target).1,
        None => t.seconds,
    };
    let off = ((seconds - target).abs() / target.max(1.0)).min(1.0);
    score += 0.4 * (1.0 - off) - 0.2;
    why.push(if t.bpm.is_some() && (seconds - t.seconds).abs() > 0.5 {
        format!("{:.0} s at its tempo, {:.0} s with the music's tempo moved ({target:.0} s asked)", t.seconds, seconds)
    } else {
        format!("{seconds:.0} s ({target:.0} s asked)")
    });
    // the topic and the subjects
    let mine: BTreeSet<String> = t
        .tags
        .iter()
        .chain(std::iter::once(&t.name))
        .chain(std::iter::once(&t.category))
        .chain(std::iter::once(&t.description))
        .flat_map(|s| words_of(s))
        .collect();
    let hits: Vec<&String> = f.words.iter().filter(|w| mine.contains(*w)).take(3).collect();
    if !hits.is_empty() {
        score += 0.25 * hits.len() as f64;
        why.push(format!("about {}", hits.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    }
    // photos in video slots hold still
    if f.videos == 0 && t.slots.iter().any(|s| s.kind == "video") {
        score -= 0.3;
        why.push("photos only: its video slots hold them still".into());
    }
    Scored { id: t.id.clone(), name: t.name.clone(), score: (score * 1000.0).round() / 1000.0, reasons: why, seconds: (seconds * 10.0).round() / 10.0 }
}

fn energy_words(e: f64) -> &'static str {
    if e >= 0.7 {
        "high"
    } else if e <= 0.3 {
        "calm"
    } else {
        "medium"
    }
}

/// The template choice: the best and the runners-up.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Choice {
    pub chosen: Scored,
    pub auto: bool,
    pub others: Vec<Scored>,
}

/// Choose among `templates` the best for `f` in layout `aspect` (the AI-ready ones when any has
/// the layout; templates without it are left out).
pub fn choose_template(templates: &[TemplateInfo], f: &Facts, aspect: &str, target: f64) -> Result<Choice, AutoCutError> {
    let fits: Vec<&TemplateInfo> = templates.iter().filter(|t| t.aspects.iter().any(|a| a == aspect) && t.needs() > 0).collect();
    if fits.is_empty() {
        return Err(invalid(format!("no template has a {aspect} layout")));
    }
    let mut scored: Vec<Scored> = fits.iter().map(|t| score_template(t, f, target)).collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    let chosen = scored.first().cloned().ok_or_else(|| invalid("no template"))?;
    let others = scored.into_iter().skip(1).take(4).collect();
    Ok(Choice { chosen, auto: true, others })
}

// ---------------------------------------------------------------------------------------------
// shots for the slots
// ---------------------------------------------------------------------------------------------

/// One media of a picture slot to fill.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PickSpec {
    pub slot: String,
    pub media: usize,
    /// How long it plays, seconds.
    pub need: f64,
    pub kind: String,
    #[serde(skip)]
    pub hints: Hints,
    /// Where it starts in the template (its slot's order, then its media index).
    pub order: usize,
}

/// The picks a template's picture slots need, in the template's order (logo slots are left to
/// the person filling it).
pub fn pick_specs(t: &TemplateInfo) -> Vec<PickSpec> {
    let mut out = Vec::new();
    for s in t.slots.iter().filter(|s| s.picture()) {
        for m in 0..s.count {
            if out.len() >= MAX_PICKS {
                return out;
            }
            out.push(PickSpec { slot: s.id.clone(), media: m, need: s.need(m).max(0.1), kind: s.kind.clone(), hints: s.hints(), order: out.len() });
        }
    }
    out
}

/// A moment chosen for a pick.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Pick {
    pub slot: String,
    pub media: usize,
    pub item: u64,
    pub name: String,
    pub shot: usize,
    /// Where it plays from and to in the media, seconds (stills: 0).
    #[serde(rename = "inSeconds")]
    pub in_seconds: f64,
    #[serde(rename = "outSeconds")]
    pub out_seconds: f64,
    /// The picture's focus for the crop (`[x, y]`, fractions), when the subject's place is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<[f64; 2]>,
    pub score: f64,
    pub why: Vec<String>,
    pub caption: String,
    /// A stretch of kept speech (a talking slot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub said: Option<String>,
    /// The subject's box in the picture (`[x, y, w, h]`, fractions) as the description saw it at
    /// `keyframe` (media seconds): kept sharp by the privacy pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<[f64; 4]>,
    #[serde(skip)]
    pub keyframe: f64,
    #[serde(skip)]
    pub still: bool,
    #[serde(skip)]
    pub order: usize,
}

/// One way to fill a pick: a moment of a shot.
#[derive(Clone, Debug)]
struct Candidate<'a> {
    shot: &'a ShotInfo,
    /// The media span it plays.
    from: f64,
    to: f64,
    /// Its score before what was picked already, and why.
    score: f64,
    why: Vec<String>,
    said: Option<String>,
}

fn purpose_use(p: &str) -> ShotUse {
    match p {
        "outro" => ShotUse::Establishing,
        other => ShotUse::from_words(other).unwrap_or(ShotUse::Unknown),
    }
}

/// The moments of `s` for a pick `need` seconds long: its best moment, and in a long shot a few
/// more spread over it.
fn windows(s: &ShotInfo, need: f64) -> Vec<(f64, f64, bool)> {
    if s.still {
        return vec![(0.0, 0.0, true)];
    }
    let len = s.seconds();
    let place = |centre: f64| {
        if len <= need {
            (s.start, s.start + need.min(len.max(0.1)))
        } else {
            let a = (centre - need / 2.0).clamp(s.start, s.end - need);
            (a, a + need)
        }
    };
    let mut out = vec![{
        let (a, b) = place(s.keyframe);
        (a, b, true)
    }];
    if len >= 3.0 * need && need > 0.0 {
        let k = ((len / (1.5 * need)).floor() as usize).clamp(2, 4);
        for j in 0..k {
            let c = s.start + len * (j as f64 + 0.5) / k as f64;
            let (a, b) = place(c);
            if out.iter().all(|(x, y, _)| (b.min(*y) - a.max(*x)) <= OVERLAP) {
                out.push((a, b, false));
            }
        }
    }
    out
}

/// How well shot `s` (played over `[from, to]`) suits pick `p`, and why.
fn suit(p: &PickSpec, s: &ShotInfo, search: &HashMap<(u64, usize), f64>, topic: &[String]) -> (f64, Vec<String>) {
    let h = &p.hints;
    let mut why = Vec::new();
    let (mut v, base_why) = match h.purpose.as_deref().map(purpose_use).filter(|u| *u != ShotUse::Unknown) {
        Some(u) => {
            let (sc, w) = filmcraft_scenes::best::score(&s.shot, u);
            (f64::from(sc), w)
        }
        None => (f64::from(s.shot.quality.score.clamp(0.0, 1.0)) * 0.6 + 0.2, Vec::new()),
    };
    why.extend(base_why);
    if !h.query.is_empty()
        && let Some(q) = search.get(&(s.item, s.index))
    {
        v += 0.35 * q.clamp(0.0, 1.0);
        if *q >= 0.5 {
            why.push(format!("matches \"{}\"", chars(&h.query, 40)));
        }
    }
    let energy = s.description().map_or(Energy::Unknown, |d| d.energy);
    if let (Some(want), true) = (h.energy.as_deref(), energy != Energy::Unknown) {
        let d = (energy_level(want) - f64::from(energy.level())).abs();
        if d < 0.25 {
            v += 0.08;
            why.push(format!("{} energy", energy.name()));
        } else if d > 0.75 {
            v -= 0.08;
        }
    }
    match h.speech.as_deref() {
        Some("want") => {
            if s.speech >= 0.5 {
                v += 0.35;
                why.push("someone speaking".into());
            } else if s.speech < 0.2 {
                v -= 0.4;
            }
            if s.talking {
                v += 0.15;
            }
        }
        Some("avoid") => {
            if s.speech > 0.5 || s.talking {
                v -= 0.3;
            } else {
                v += 0.05;
                why.push("no one talking".into());
            }
        }
        _ => {}
    }
    if let Some(min) = h.min_quality
        && f64::from(s.shot.quality.score) < min
    {
        v -= 0.25;
    }
    if !s.still && s.seconds() < 0.7 * p.need {
        v -= 0.15 * (1.0 - s.seconds() / p.need.max(0.1)).clamp(0.0, 1.0);
        why.push("shorter than its place".into());
    }
    match (p.kind.as_str(), s.still) {
        ("photo", false) => v -= 0.1,
        ("video", true) => {
            v -= 0.15;
            why.push("a photo, held".into());
        }
        _ => {}
    }
    if !topic.is_empty() {
        let text =
            s.description().map(|d| format!("{} {}", d.caption, d.subjects.iter().map(|x| x.name.as_str()).collect::<Vec<_>>().join(" "))).unwrap_or_default();
        let w = words_of(&text);
        if topic.iter().any(|t| w.contains(t)) {
            v += 0.1;
            why.push("about the topic".into());
        }
    }
    (v, why)
}

/// Stretches of kept speech of `speech` (in `shots`' items) about `need` seconds long: from each
/// kept segment, it and the ones after it while they fit.
fn speech_moments<'a>(shots: &'a [ShotInfo], speech: &[ItemSpeech], need: f64) -> Vec<Candidate<'a>> {
    let mut out = Vec::new();
    for sp in speech.iter().filter(|s| s.talking()) {
        for (i, first) in sp.kept.iter().enumerate().take(400) {
            let mut end = first.end;
            let mut text = first.text.clone();
            let mut fl = vec![first.fluency];
            for next in sp.kept.iter().skip(i + 1) {
                if next.end - first.start > need + 0.5 || next.start - end > 1.5 {
                    break;
                }
                end = next.end;
                fl.push(next.fluency);
                text.push(' ');
                text.push_str(&next.text);
            }
            let from = (first.start - 0.12).max(0.0);
            let Some(shot) =
                shots.iter().find(|s| s.item == sp.item && s.start <= from + 0.05 && s.end > from).or_else(|| shots.iter().find(|s| s.item == sp.item))
            else {
                continue;
            };
            let cover = ((end - from) / need.max(0.1)).min(1.0);
            let fluency = fl.iter().sum::<f64>() / fl.len().max(1) as f64;
            out.push(Candidate {
                shot,
                from,
                to: from + need,
                score: 0.4 + 0.4 * fluency + 0.4 * cover,
                why: vec![format!("{:.0} s of kept speech", end - from), format!("fluency {fluency:.2}")],
                said: Some(chars(&text, 300)),
            });
        }
    }
    out
}

/// The order picks choose in: those a single shot stands for (a hook, a reveal, a place, an ending,
/// someone talking) before the many cutaways.
fn rank(p: &PickSpec) -> u8 {
    let key = matches!(p.hints.purpose.as_deref(), Some("hook" | "reveal" | "establishing" | "outro" | "talking-head")) || p.hints.wants_speech();
    if key { 0 } else { 1 }
}

/// The first subject's box of a shot's description, when it has one.
fn subject_box(s: &ShotInfo) -> Option<[f64; 4]> {
    let b = s.description()?.subjects.iter().find_map(|x| x.bbox)?.map(f64::from);
    (b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0 && b[2] <= 1.5 && b[3] <= 1.5).then(|| b.map(|v| (v.clamp(-1.0, 2.0) * 1e4).round() / 1e4))
}

/// Whether moment `[a, b]` of item `item` clashes with a pick already made.
fn clashes(picked: &[Pick], item: u64, still: bool, a: f64, b: f64) -> bool {
    picked.iter().any(|p| p.item == item && (still || (b.min(p.out_seconds) - a.max(p.in_seconds)) > OVERLAP))
}

/// Fill `specs` from `shots` (see the module docs): each pick the best moment not used yet, the
/// best pairs first. `search` holds each slot query's scores per `(item, shot)`.
pub fn assign(
    specs: &[PickSpec],
    shots: &[ShotInfo],
    speech: &[ItemSpeech],
    search: &HashMap<String, HashMap<(u64, usize), f64>>,
    topic: Option<&str>,
) -> Vec<Pick> {
    let topic_words = topic.map(words_of).unwrap_or_default();
    let empty = HashMap::new();
    // every pick's candidates, scored on their own
    let mut cands: Vec<Vec<Candidate<'_>>> = Vec::with_capacity(specs.len());
    for p in specs {
        let q = search.get(&p.hints.query).unwrap_or(&empty);
        let mut list: Vec<Candidate<'_>> = Vec::new();
        if p.hints.wants_speech() {
            list.extend(speech_moments(shots, speech, p.need));
        }
        for s in shots {
            let (score, why) = suit(p, s, q, &topic_words);
            for (a, b, main) in windows(s, p.need) {
                let mut why = why.clone();
                if !main {
                    why.push("another moment of a long shot".into());
                }
                list.push(Candidate { shot: s, from: a, to: b, score: if main { score } else { score - 0.04 }, why, said: None });
            }
        }
        cands.push(list);
    }
    let mut picked: Vec<Pick> = Vec::new();
    let mut done = vec![false; specs.len()];
    while picked.len() < specs.len() {
        let mut best: Option<(usize, usize, f64)> = None;
        // the picks one shot stands for (a hook, a reveal, someone talking…) choose first
        let first = specs.iter().enumerate().filter(|(i, _)| !done.get(*i).copied().unwrap_or(true)).map(|(_, s)| rank(s)).min().unwrap_or(1);
        for (pi, list) in cands.iter().enumerate() {
            if done.get(pi).copied().unwrap_or(true) || specs.get(pi).is_none_or(|s| rank(s) != first) {
                continue;
            }
            for (ci, c) in list.iter().enumerate() {
                if clashes(&picked, c.shot.item, c.shot.still, c.from, c.to) {
                    continue;
                }
                // a few picks of one clip at most (none after the hook or the reveal showed it), and
                // not the same picture twice
                let same_item: f64 = picked
                    .iter()
                    .filter(|p| p.item == c.shot.item)
                    .map(|p| if specs.iter().find(|s| s.slot == p.slot).is_some_and(|s| rank(s) == 0) { 2.5 } else { 1.0 })
                    .sum();
                let caption = c.shot.caption();
                let same_caption = picked.iter().any(|p| !caption.is_empty() && p.caption == caption && p.item != c.shot.item);
                let v = c.score - 0.12 * same_item - if same_caption { 0.1 } else { 0.0 };
                if best.is_none_or(|(_, _, b)| v > b) {
                    best = Some((pi, ci, v));
                }
            }
        }
        let Some((pi, ci, v)) = best else { break };
        let (Some(spec), Some(c)) = (specs.get(pi), cands.get(pi).and_then(|l| l.get(ci))) else { break };
        if let Some(d) = done.get_mut(pi) {
            *d = true;
        }
        let s = c.shot;
        let focus = focus_of(s);
        let (a, b) = if s.still { (0.0, 0.0) } else { (c.from.max(0.0), c.to.max(c.from)) };
        picked.push(Pick {
            slot: spec.slot.clone(),
            media: spec.media,
            item: s.item,
            name: s.name.clone(),
            shot: s.index,
            in_seconds: (a * 1000.0).round() / 1000.0,
            out_seconds: (b * 1000.0).round() / 1000.0,
            focus,
            score: (v * 1000.0).round() / 1000.0,
            why: c.why.clone(),
            caption: s.caption(),
            said: c.said.clone(),
            subject: subject_box(s),
            keyframe: s.keyframe,
            still: s.still,
            order: spec.order,
        });
    }
    in_filming_order(&mut picked, specs, shots);
    picked.sort_by_key(|p| p.order);
    picked
}

/// The picks of slots that keep the filming order, in the template's order, get their moments
/// in the order they were filmed (each pick keeps its place).
fn in_filming_order(picked: &mut [Pick], specs: &[PickSpec], shots: &[ShotInfo]) {
    let chrono: BTreeSet<&str> = specs.iter().filter(|s| s.hints.chronological).map(|s| s.slot.as_str()).collect();
    if chrono.is_empty() {
        return;
    }
    let order_of = |item: u64| shots.iter().find(|s| s.item == item).map_or(usize::MAX, |s| s.order);
    let mut idx: Vec<usize> = picked.iter().enumerate().filter(|(_, p)| chrono.contains(p.slot.as_str())).map(|(i, _)| i).collect();
    idx.sort_by_key(|i| picked.get(*i).map_or(usize::MAX, |p| p.order));
    // the moments, in filming order
    let mut moments: Vec<Pick> = idx.iter().filter_map(|i| picked.get(*i).cloned()).collect();
    moments.sort_by(|a, b| order_of(a.item).cmp(&order_of(b.item)).then(a.in_seconds.total_cmp(&b.in_seconds)));
    for (i, m) in idx.iter().zip(moments) {
        if let Some(p) = picked.get_mut(*i) {
            let (slot, media, order) = (p.slot.clone(), p.media, p.order);
            *p = Pick { slot, media, order, ..m };
            p.why.push("in filming order".into());
        }
    }
}

/// The crop's focus: the middle of the shot's (first) subject, or of a face up high for a
/// talking head; none when not known.
fn focus_of(s: &ShotInfo) -> Option<[f64; 2]> {
    let d = s.description()?;
    if let Some(b) = d.subjects.iter().find_map(|x| x.bbox) {
        let [x, y, w, h] = b.map(f64::from);
        if [x, y, w, h].iter().all(|v| v.is_finite()) && w > 0.0 && h > 0.0 {
            let r = |v: f64| (v.clamp(0.0, 1.0) * 1e4).round() / 1e4;
            return Some([r(x + w / 2.0), r(y + h / 2.0)]);
        }
    }
    (d.faces_prominent || s.talking).then_some([0.5, 0.4])
}

/// `template.apply`'s value for a pick.
pub fn fill_value(p: &Pick) -> Value {
    let mut v = json!({"item": p.item});
    if !p.still {
        v["inSeconds"] = json!(p.in_seconds);
    }
    if let Some(f) = p.focus {
        v["focus"] = json!(f);
    }
    v
}

// ---------------------------------------------------------------------------------------------
// text
// ---------------------------------------------------------------------------------------------

#[path = "quick_text.rs"]
pub mod text;
use text::title_case;
pub use text::{
    GENERIC, Grounds, Looked, Spot, TextSpec, Verdict, WordCheck, Written, check_words, content_words, fallback_options, fallback_text, judge, text_specs,
    tidy_text, write_texts,
};

// ---------------------------------------------------------------------------------------------
// the job
// ---------------------------------------------------------------------------------------------

/// The job's options, as the engine checked them.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub template: Option<String>,
    pub aspect: String,
    pub platform: Platform,
    pub seconds: f64,
    pub topic: Option<String>,
    pub language: Option<String>,
    pub look: Option<crate::creator_look::CreatorLook>,
    /// A music file, or none: generated.
    pub music: Option<String>,
    pub redact: Option<crate::privacy::Redact>,
    pub caption_style: String,
    pub export: Option<String>,
    pub name: Option<String>,
    pub download: bool,
    pub full_detail: usize,
    pub qa: Option<bool>,
}

impl Options {
    pub fn from_params(p: &Value) -> Result<Options, AutoCutError> {
        let text = |k: &str| p[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        let template = text("template").filter(|t| !t.eq_ignore_ascii_case("auto"));
        let aspect = text("aspect").unwrap_or_else(|| "9:16".into());
        if !matches!(aspect.as_str(), "9:16" | "1:1" | "4:5" | "16:9") {
            return Err(invalid("aspect: 9:16, 1:1, 4:5 or 16:9"));
        }
        let platform = p["platform"].as_str().and_then(Platform::from_name).unwrap_or(Platform::Reels);
        let seconds = finite(&p["seconds"]).unwrap_or(DEFAULT_SECONDS).clamp(5.0, 180.0);
        let music = text("music").filter(|m| !m.eq_ignore_ascii_case("generate"));
        let redact = match &p["redact"] {
            Value::Null => Some(crate::privacy::Redact::default()),
            Value::Bool(false) => None,
            Value::String(s) if matches!(s.trim().to_ascii_lowercase().as_str(), "off" | "none" | "false" | "") => None,
            v => crate::privacy::Redact::from_plan(Some(v))?,
        };
        Ok(Options {
            template,
            aspect,
            platform,
            seconds,
            topic: text("topic").map(|t| chars(&t, 400)),
            language: text("language").filter(|l| l != "auto"),
            look: crate::creator_look::look_of(p)?,
            music,
            redact,
            caption_style: text("captionStyle").unwrap_or_else(|| "bold".into()),
            export: text("export"),
            name: text("name").map(|n| chars(&n, 200)),
            download: p["download"].as_bool().unwrap_or(false),
            full_detail: p["fullDetail"].as_u64().map_or(FULL_DETAIL_SHOTS, |n| n.min(64) as usize),
            qa: p["qa"].as_bool(),
        })
    }
}

/// What the models were, and what ran without them.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelsUsed {
    pub boundaries: BTreeSet<String>,
    pub describer: BTreeSet<String>,
    pub embedder: BTreeSet<String>,
    pub aesthetic: BTreeSet<String>,
    #[serde(rename = "fullDetail")]
    pub full_detail: usize,
    pub speech: BTreeSet<String>,
    pub vad: BTreeSet<String>,
    pub writer: Option<String>,
    /// What checked the texts against their frames (SigLIP 2, Qwen3-VL).
    pub checker: Vec<String>,
    pub redaction: Option<String>,
}

/// The job's report: the template and why, each pick and why, the texts, what was missing.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub template: Option<Choice>,
    pub sequence: Option<u64>,
    pub seconds: f64,
    pub media: usize,
    pub shots: usize,
    pub facts: Facts,
    pub picks: Vec<Pick>,
    pub texts: Vec<Value>,
    /// How the texts were checked: by what, how many candidates were rejected.
    #[serde(rename = "textCheck", skip_serializing_if = "Option::is_none")]
    pub text_check: Option<Value>,
    pub models: ModelsUsed,
    pub fallbacks: Vec<String>,
    pub warnings: Vec<String>,
    #[serde(rename = "safeZones", skip_serializing_if = "Option::is_none")]
    pub safe_zones: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    #[serde(rename = "reportFile", skip_serializing_if = "Option::is_none")]
    pub report_file: Option<String>,
    #[serde(rename = "qaSummary", skip_serializing_if = "Option::is_none")]
    pub qa_summary: Option<String>,
}

/// The media items of the job: `items`, the files of `paths` imported, else the project's.
async fn gather(ex: &dyn Executor, p: &Value, lines: &mut dyn FnMut(&str)) -> Result<Vec<MediaItem>, AutoCutError> {
    let mut want: Vec<u64> = p["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).take(MAX_MEDIA).collect()).unwrap_or_default();
    let paths: Vec<String> =
        p["paths"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).take(MAX_MEDIA).collect()).unwrap_or_default();
    if !paths.is_empty() {
        let files = expand_paths(&paths)?;
        lines(&format!("[1/8] importing {} file(s)", files.len()));
        let r = run(ex, "file.import", json!({"paths": files, "reuse": true})).await?;
        let got: Vec<u64> = r["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        if got.is_empty() {
            let why = r["errors"].as_array().and_then(|a| a.first()).map_or_else(|| "nothing could be imported".to_string(), Value::to_string);
            return Err(invalid(why));
        }
        want.extend(got);
    }
    let tree = run(ex, "project.inspect", json!({})).await?;
    let all = project_media(&tree);
    let mut media: Vec<MediaItem> = if want.is_empty() {
        all
    } else {
        let mut seen = BTreeSet::new();
        want.iter().filter(|i| seen.insert(**i)).filter_map(|i| all.iter().find(|m| m.item == *i).cloned()).collect()
    };
    if media.is_empty() {
        return Err(invalid("no video or photo to make a short of (give items, files or a folder)"));
    }
    // filming order: capture dates where the files say, else the order given
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
    } else {
        for (k, m) in media.iter_mut().enumerate() {
            m.order = k;
        }
    }
    Ok(media)
}

/// Download what `--download` asks for: the scene models, the speech models for `language`, the
/// face and plate detectors.
async fn download(ex: &dyn Executor, opts: &Options, lines: &mut dyn FnMut(&str), notes: &mut Vec<String>) {
    for set in ["fast", "full"] {
        lines(&format!("downloading the scene models ({set}) if missing"));
        if let Err(e) = run(ex, "scenes.downloadModel", json!({"model": set, "wait": true})).await {
            notes.push(format!("download: scene models ({set}): {e}"));
        }
    }
    let recogniser = match opts.language.as_deref() {
        Some("ml") => "whisper-medium-ml",
        Some("en") => "whisper-large-v3-turbo",
        _ => "whisper-base",
    };
    for m in [recogniser, "silero-vad", "ecapa-voxceleb"] {
        let installed = run(ex, "transcript.models", json!({}))
            .await
            .ok()
            .is_some_and(|v| v["models"].as_array().is_some_and(|a| a.iter().any(|x| x["id"] == m && x["installed"] == true)));
        if installed {
            continue;
        }
        lines(&format!("downloading the speech model {m}"));
        if let Err(e) = run(ex, "transcript.downloadModel", json!({"model": m})).await {
            notes.push(format!("download: {m}: {e}"));
        }
    }
    if opts.redact.is_some()
        && let Err(e) = run(ex, "redact.downloadModels", json!({"wait": true})).await
    {
        notes.push(format!("download: the face and plate detectors: {e}"));
    }
}

/// Make the short (the `autocut.quick` job; parameters as the engine checked them). `writer`
/// writes the text slots (none: from the descriptions).
pub async fn make(ex: &dyn Executor, p: &Value, writer: Option<&dyn ChatModel>, lines: &mut dyn FnMut(&str)) -> Result<Value, AutoCutError> {
    let opts = Options::from_params(p)?;
    let mut report = Report { seconds: opts.seconds, ..Report::default() };
    let mut notes: Vec<String> = Vec::new();
    if opts.download {
        download(ex, &opts, lines, &mut notes).await;
    }
    // ---- 1. the media
    let media = gather(ex, p, lines).await?;
    report.media = media.len();
    let items: Vec<u64> = media.iter().map(|m| m.item).collect();
    // ---- 2. the index: fast for all, full for the best few
    lines(&format!("[2/8] indexing the scenes of {} item(s)", items.len()));
    let status = run(ex, "scenes.status", json!({})).await.ok();
    // the models can run here (this build has them, or the host's): otherwise they count as missing
    let runnable = status.as_ref().is_some_and(|s| s["available"] == true);
    let missing = |set: &str| if runnable { status.as_ref().and_then(|s| s[set]["missingBytes"].as_f64()).unwrap_or(f64::MAX) } else { f64::MAX };
    if missing("fast") > 0.0 {
        report.fallbacks.push(if missing("fast") < f64::MAX {
            format!(
                "scene models: {:.1} GB of the fast set are not downloaded: shots are cut by the picture's changes and ranked by its quality without descriptions (add --download)",
                missing("fast") / 1e9
            )
        } else {
            "scene models: not available here: shots ranked by the picture's quality alone".into()
        });
    }
    match run(ex, "media.analyzeScenes", json!({"items": items, "detail": "fast", "wait": true})).await {
        Ok(v) => {
            for n in v["scenes"]["notes"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(10) {
                if let Some(n) = n.as_str() {
                    notes.push(format!("scenes: {n}"));
                }
            }
        }
        Err(e) => report.fallbacks.push(format!("scene index: {e}; each clip counts as one shot")),
    }
    let mut indexes: HashMap<u64, Value> = HashMap::new();
    for m in &media {
        if let Ok(v) = run(ex, "media.scenes", json!({"item": m.item})).await
            && v["analyzed"] == true
        {
            indexes.insert(m.item, v);
        }
    }
    let shots_now =
        |indexes: &HashMap<u64, Value>| -> Vec<ShotInfo> { media.iter().flat_map(|m| parse_shots(m, indexes.get(&m.item))).take(MAX_SHOTS).collect() };
    let mut shots = shots_now(&indexes);
    let described = shots.iter().any(|s| s.description().is_some());
    if !described && missing("fast") <= 0.0 {
        report.fallbacks.push("scene index: no shot was described: shots ranked by the picture's quality alone".into());
    }
    if opts.full_detail > 0 && described && missing("full") <= 0.0 {
        // the items of the most promising shots, at most `full_detail` shots in all
        let mut ranked: Vec<(f64, u64)> = shots
            .iter()
            .map(|s| {
                let best = [ShotUse::Hook, ShotUse::Reveal, ShotUse::BRoll].iter().map(|u| filmcraft_scenes::best::score(&s.shot, *u).0).fold(0.0f32, f32::max);
                (f64::from(best), s.item)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut chosen: Vec<u64> = Vec::new();
        let mut count = 0usize;
        for (_, item) in ranked {
            if chosen.contains(&item) {
                continue;
            }
            let n = shots.iter().filter(|s| s.item == item).count();
            if count + n > opts.full_detail && !chosen.is_empty() {
                continue;
            }
            if count + n > opts.full_detail {
                continue;
            }
            chosen.push(item);
            count += n;
        }
        if !chosen.is_empty() {
            lines(&format!("[2/8] describing {count} promising shot(s) in full detail"));
            match run(ex, "media.analyzeScenes", json!({"items": chosen, "detail": "full", "wait": true})).await {
                Ok(_) => {
                    for item in &chosen {
                        if let Ok(v) = run(ex, "media.scenes", json!({"item": item})).await
                            && v["analyzed"] == true
                        {
                            indexes.insert(*item, v);
                        }
                    }
                    shots = shots_now(&indexes);
                }
                Err(e) => notes.push(format!("full detail: {e}")),
            }
        }
    } else if opts.full_detail > 0 && described {
        report.fallbacks.push("full detail: the 4B describer is not downloaded: the fast descriptions are used (add --download)".into());
    }
    for v in indexes.values() {
        let put = |k: &str, set: &mut BTreeSet<String>| {
            if let Some(x) = v[k].as_str().filter(|x| !x.is_empty()) {
                set.insert(chars(x, 80));
            }
        };
        put("boundaries", &mut report.models.boundaries);
        put("describer", &mut report.models.describer);
        put("embedder", &mut report.models.embedder);
        put("aesthetic", &mut report.models.aesthetic);
    }
    report.models.full_detail = shots.iter().filter(|s| s.full).count();
    // ---- 3. speech
    let movies: Vec<u64> = media.iter().filter(|m| !m.still).map(|m| m.item).collect();
    let mut speech: Vec<ItemSpeech> = Vec::new();
    if !movies.is_empty() {
        lines(&format!("[3/8] understanding the speech of {} clip(s)", movies.len()));
        let mut up = json!({"items": movies, "wait": true});
        if let Some(l) = &opts.language {
            up["language"] = json!(l);
        }
        match run(ex, "media.understandSpeech", up).await {
            Ok(_) => {
                if let Ok(v) = run(ex, "media.inspectSpeech", json!({"items": movies})).await {
                    speech = parse_speech(&v);
                }
            }
            Err(e) => {
                let why = e.to_string();
                if !why.contains("none of the clips has audio") {
                    report
                        .fallbacks
                        .push(format!("speech: not understood ({}): talking-head templates are not chosen and no captions are made", chars(&why, 200)));
                }
            }
        }
    }
    for s in &speech {
        if !s.source.is_empty() {
            report.models.speech.insert(s.source.clone());
        }
        if !s.vad.is_empty() {
            report.models.vad.insert(s.vad.clone());
        }
    }
    mark_speech(&mut shots, &speech);
    report.shots = shots.len();
    report.facts = facts(&media, &shots, &speech, opts.topic.as_deref());
    // ---- 4. the template
    lines("[4/8] choosing the template");
    let list = run(ex, "template.list", json!({})).await?;
    let templates: Vec<TemplateInfo> = list["templates"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(parse_template).collect();
    let choice = match &opts.template {
        Some(key) => {
            let t = run(ex, "template.inspect", json!({"template": key})).await?;
            let info = parse_template(&t).ok_or_else(|| invalid(format!("template {key} could not be read")))?;
            if !info.aspects.contains(&opts.aspect) {
                return Err(invalid(format!("{} has no {} layout (it has {})", info.name, opts.aspect, info.aspects.join(", "))));
            }
            let mut s = score_template(&info, &report.facts, opts.seconds);
            s.reasons.insert(0, "asked for".into());
            Choice { chosen: s, auto: false, others: Vec::new() }
        }
        None => choose_template(&templates, &report.facts, &opts.aspect, opts.seconds)?,
    };
    let chosen_id = choice.chosen.id.clone();
    lines(&format!("[4/8] {} — {}", choice.chosen.name, choice.chosen.reasons.join("; ")));
    let inspected = run(ex, "template.inspect", json!({"template": chosen_id})).await?;
    let t = parse_template(&inspected).ok_or_else(|| invalid("the chosen template could not be read"))?;
    report.template = Some(choice);
    if !t.ai_ready {
        notes.push(format!("{} has no AI hints: its shots are picked by quality alone", t.name));
    }
    // ---- 5. shots for the slots
    lines("[5/8] picking a shot for every slot");
    let specs = pick_specs(&t);
    let mut search: HashMap<String, HashMap<(u64, usize), f64>> = HashMap::new();
    for q in specs.iter().map(|s| s.hints.query.clone()).filter(|q| !q.is_empty()).collect::<BTreeSet<_>>() {
        if let Ok(v) = run(ex, "media.searchScenes", json!({"query": q, "items": items, "limit": 500})).await {
            let hits = v["hits"].as_array().map(Vec::as_slice).unwrap_or_default();
            let top = hits.iter().filter_map(|h| finite(&h["score"])).fold(0.0f64, f64::max).max(1e-6);
            let map =
                hits.iter().filter_map(|h| Some(((h["item"].as_u64()?, usize::try_from(h["shot"].as_u64()?).ok()?), finite(&h["score"])? / top))).collect();
            search.insert(q, map);
        }
    }
    let picks = assign(&specs, &shots, &speech, &search, opts.topic.as_deref());
    if picks.len() < specs.len() {
        notes.push(format!("{} of {} places filled: the template repeats the others in turn", picks.len(), specs.len()));
    }
    report.picks = picks.clone();
    // ---- 6. the text
    lines("[6/8] writing the text");
    report.models.writer = writer.map(|w| w.id());
    if writer.is_none() {
        report.fallbacks.push("text: no local model (Ollama): written from the shots' descriptions and the topic".into());
    }
    let tspecs = text_specs(&t, &picks);
    let (texts, checked) = write_texts(
        Some(ex),
        aspect_ratio(&opts.aspect),
        writer,
        &tspecs,
        &picks,
        &shots,
        &speech,
        opts.topic.as_deref(),
        opts.language.as_deref(),
        &mut notes,
    )
    .await;
    match checked["how"].as_str() {
        Some("words only") if !tspecs.is_empty() => report.fallbacks.push(
            "text check: no vision model (SigLIP 2, Qwen3-VL): each text's words were checked against the descriptions of the shots under it, not its frames"
                .into(),
        ),
        Some("words and SigLIP") => report.fallbacks.push("text check: no describer (Qwen3-VL) to ask: SigLIP 2's scores decided what the words name".into()),
        _ => {}
    }
    report.models.checker = [checked["embedder"].as_str(), checked["asker"].as_str()].into_iter().flatten().map(str::to_string).collect();
    report.text_check = Some(checked);
    // ---- 7. apply
    lines("[7/8] building the short");
    let mut slots = serde_json::Map::new();
    for s in t.slots.iter().filter(|s| s.picture()) {
        let mut list: Vec<(usize, Value)> = picks.iter().filter(|p| p.slot == s.id).map(|p| (p.media, fill_value(p))).collect();
        list.sort_by_key(|(m, _)| *m);
        if !list.is_empty() {
            slots.insert(s.id.clone(), Value::Array(list.into_iter().map(|(_, v)| v).collect()));
        }
    }
    let mut by_slot: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for w in &texts {
        report.texts.push(w.to_json(&picks));
        let Some(text) = &w.text else {
            notes.push(format!("text: nothing to put in `{}` {}: the template's own text stays", w.spec.slot, w.spec.index + 1));
            continue;
        };
        by_slot.entry(w.spec.slot.clone()).or_default().push((w.spec.index, text.clone()));
    }
    for (slot, mut list) in by_slot {
        list.sort_by_key(|(i, _)| *i);
        // texts fill in order: a missing one ends the list (the template's own text shows after)
        let mut vals = Vec::new();
        for (k, (i, text)) in list.into_iter().enumerate() {
            if i != k {
                break;
            }
            vals.push(json!(text));
        }
        slots.insert(slot, Value::Array(vals));
    }
    // the music: a file, or a bed at the tempo that brings the template to the length asked
    let music_slot = t.slots.iter().find(|s| s.kind == "music").map(|s| s.id.clone());
    let mut generate_music = true;
    let mut music_beats: Option<Value> = None;
    if let (Some(ms), Some(file)) = (&music_slot, &opts.music) {
        slots.insert(ms.clone(), json!(filmcraft_engine::export_tools::expand_home(file)));
    } else if let (Some(ms), Some(bpm)) = (&music_slot, t.bpm) {
        let (want, length) = stretch(bpm, t.seconds, opts.seconds);
        if (length - t.seconds).abs() > 0.5 {
            let (mood, genre) = mood_of(&inspected, ms);
            let gp = json!({"kind": "music", "duration": length + 1.5, "bpm": want, "mood": mood, "genre": genre, "seed": 7, "place": false});
            match run(ex, "audio.generate", gp).await {
                Ok(r) => match r["item"].as_u64() {
                    Some(item) => {
                        slots.insert(ms.clone(), json!(item));
                        // its exact beats: measuring a slow bed may hear twice the tempo
                        music_beats = r["beats"].as_array().filter(|b| b.len() >= 2).map(|b| Value::Array(b.clone()));
                        notes.push(format!("music: generated at {want} bpm (the template's {bpm}) for about {length:.0} s"));
                    }
                    None => notes.push("music: the generated bed made no item; the template's own is used".into()),
                },
                Err(e) => notes.push(format!("music: {e}; the template's own bed is used")),
            }
        }
    } else if music_slot.is_none() {
        generate_music = false;
    }
    let name = opts.name.clone().unwrap_or_else(|| format!("{} — Quick Short", opts.topic.as_deref().map(title_case).unwrap_or_else(|| t.name.clone())));
    let applied = run(ex, "template.apply", {
        let mut ap = json!({"template": chosen_id, "variant": opts.aspect, "slots": Value::Object(slots), "name": name, "generateMusic": generate_music});
        if let Some(b) = music_beats {
            ap["musicBeats"] = b;
        }
        ap
    })
    .await?;
    let seq = applied["sequence"].as_u64().ok_or_else(|| AutoCutError::Executor("template.apply made no sequence".into()))?;
    report.sequence = Some(seq);
    report.seconds = finite(&applied["seconds"]).unwrap_or(t.seconds);
    for w in applied["warnings"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(40) {
        if let Some(w) = w.as_str() {
            report.warnings.push(w.to_string());
        }
    }
    // ---- 8. finish
    lines("[8/8] finishing");
    finish(ex, &opts, &t, &applied, seq, &picks, &mut report, &mut notes, lines).await;
    report.warnings.extend(notes);
    if let Some(path) = report.export.clone() {
        let file = crate::trailer::beside(&path, "report.json");
        report.report_file = Some(file.display().to_string());
        let text = serde_json::to_string_pretty(&report).unwrap_or_default();
        if let Err(e) = std::fs::write(&file, format!("{text}\n")) {
            report.warnings.push(format!("report: {}: {e}", file.display()));
            report.report_file = None;
        }
    }
    let chosen = report.template.as_ref().map(|c| c.chosen.name.clone()).unwrap_or_default();
    let status = match &report.export {
        Some(p) => format!("Quick Short: {chosen}, {:.1} s, {} shot(s) placed, exported to {p}", report.seconds, report.picks.len()),
        None => format!("Quick Short: {chosen}, {:.1} s, {} shot(s) placed", report.seconds, report.picks.len()),
    };
    let mut out = serde_json::to_value(&report).unwrap_or_else(|_| json!({}));
    out["status"] = json!(status);
    Ok(out)
}

/// The mood and genre the template's music slot generates with (`default.generate`).
fn mood_of(inspected: &Value, slot: &str) -> (String, String) {
    let g = inspected["slots"].as_array().and_then(|a| a.iter().find(|s| s["id"] == slot)).map(|s| s["default"]["generate"].clone()).unwrap_or(Value::Null);
    (g["mood"].as_str().unwrap_or("energetic").to_string(), g["genre"].as_str().unwrap_or("electronic").to_string())
}

/// Select `clips` and remove them (`edit.clear`).
async fn remove(ex: &dyn Executor, clips: &[u64]) -> Result<(), AutoCutError> {
    if clips.is_empty() {
        return Ok(());
    }
    run(ex, "timeline.select", json!({"clips": clips})).await?;
    run(ex, "edit.clear", json!({"clips": clips})).await.map(drop)
}

/// After the apply: empty placeholders out, the look, the captions, privacy, safe zones, the
/// files.
#[allow(clippy::too_many_arguments)]
async fn finish(
    ex: &dyn Executor,
    opts: &Options,
    t: &TemplateInfo,
    applied: &Value,
    seq: u64,
    picks: &[Pick],
    report: &mut Report,
    notes: &mut Vec<String>,
    lines: &mut dyn FnMut(&str),
) {
    let placed = applied["slots"].as_array().map(Vec::as_slice).unwrap_or_default();
    let clips_of = |pred: &dyn Fn(&SlotInfo) -> bool, filled: Option<bool>| -> Vec<u64> {
        placed
            .iter()
            .filter(|x| t.slots.iter().any(|s| x["slot"] == s.id.as_str() && pred(s)))
            .filter(|x| filled.is_none_or(|f| x["media"].is_null() != f))
            .flat_map(|x| x["clips"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(Value::as_u64))
            .collect()
    };
    // optional picture slots nobody filled: their placeholders go
    let empty = clips_of(&|s: &SlotInfo| s.picture() && s.min_count == 0, Some(false));
    if !empty.is_empty() {
        match remove(ex, &empty).await {
            Ok(()) => notes.push(format!("{} placeholder clip(s) of optional slots removed (nothing suited them)", empty.len())),
            Err(e) => notes.push(format!("placeholders: {e}")),
        }
    }
    let info = run(ex, "sequence.inspect", json!({"item": seq})).await.ok();
    let seconds = info.as_ref().and_then(|v| secs_of_ticks(&v["duration"])).unwrap_or(report.seconds);
    report.seconds = (seconds * 100.0).round() / 100.0;
    // the look: the template's own, or the one asked for
    if let Some(look) = opts.look {
        let own: Vec<u64> = info
            .as_ref()
            .and_then(|v| v["video"].as_array())
            .map(|tracks| {
                tracks
                    .iter()
                    .flat_map(|tr| tr["items"].as_array().map(Vec::as_slice).unwrap_or_default().iter())
                    .filter(|c| c["name"].as_str().is_some_and(|n| n.starts_with("Look: ")))
                    .filter_map(|c| c["clip"].as_u64())
                    .collect()
            })
            .unwrap_or_default();
        if let Err(e) = remove(ex, &own).await {
            notes.push(format!("look: the template's own look stays ({e})"));
        }
        run(ex, "sequence.open", json!({"item": seq})).await.ok();
        if let Err(e) = crate::creator_look::apply_look(ex, look, 0.0, seconds, filmcraft_engine::project::effect::HUD_COLOR).await {
            notes.push(format!("look {}: {e}", look.name()));
        }
    }
    // the captions: animated, in the platform's safe zone
    if t.captions {
        let has = info
            .as_ref()
            .is_some_and(|v| v["captionTracks"].as_array().is_some_and(|a| !a.is_empty()) || v["captions"].as_array().is_some_and(|a| !a.is_empty()));
        let caps = applied["captions"]["captions"].as_u64().unwrap_or(0);
        if caps > 0 || has {
            let r =
                run(ex, "captions.setAnimatedStyle", json!({"track": "C1", "preset": opts.caption_style, "options": {"platform": opts.platform.name()}})).await;
            if let Err(e) = r {
                notes.push(format!("captions: {e}"));
            }
        }
        if caps == 0 {
            report.fallbacks.push("captions: nothing was transcribed (no speech model, or no speech): the captions track is empty".into());
        }
    }
    // privacy: faces and plates in the picture slots; never the speaker's own face
    if let Some(r) = &opts.redact {
        let talking = |s: &SlotInfo| s.ai.as_ref().is_some_and(Hints::wants_speech);
        let others = clips_of(&|s: &SlotInfo| s.picture() && !talking(s), Some(true));
        let speakers = clips_of(&|s: &SlotInfo| s.picture() && talking(s), Some(true));
        // each pick's subject (a robot's face, a screen) stays sharp: its box, where it was seen
        let mut exclude: Vec<Value> = Vec::new();
        for x in placed.iter().filter(|x| !x["media"].is_null()) {
            let (Some(slot), Some(m)) = (x["slot"].as_str(), x["media"].as_u64()) else { continue };
            let Some(pick) = picks.iter().find(|p| p.slot == slot && p.media as u64 == m) else { continue };
            let Some([bx, by, bw, bh]) = pick.subject else { continue };
            for clip in x["clips"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(Value::as_u64).take(16) {
                if exclude.len() < 1000 {
                    exclude.push(json!({"x": bx, "y": by, "w": bw, "h": bh, "clip": clip, "seconds": pick.keyframe.max(0.0)}));
                }
            }
        }
        let mut runs: Vec<(Vec<u64>, Value)> = Vec::new();
        if !others.is_empty() {
            runs.push((others, r.kinds_json()));
        }
        if !speakers.is_empty() && r.plates {
            runs.push((speakers, json!(["plates"])));
            notes.push("privacy: the speaker's own face is left sharp (plates are still hidden)".into());
        }
        let mut hidden = json!({"faces": 0, "plates": 0, "clips": 0, "verify": null});
        for (clips, kinds) in runs {
            lines(&format!("privacy: looking for {} in {} clip(s)", kinds, clips.len()));
            let mut rp = json!({"clips": clips, "kinds": kinds, "verify": r.verify, "wait": true});
            if !exclude.is_empty() {
                rp["exclude"] = json!(exclude);
            }
            match run(ex, "clip.autoRedact", rp).await {
                Ok(v) => {
                    let o = &v["redact"];
                    let list = o["clips"].as_array().map(Vec::as_slice).unwrap_or_default();
                    let sum = |k: &str| list.iter().filter_map(|c| c[k].as_u64()).sum::<u64>();
                    hidden["faces"] = json!(hidden["faces"].as_u64().unwrap_or(0) + sum("faces"));
                    hidden["plates"] = json!(hidden["plates"].as_u64().unwrap_or(0) + sum("plates"));
                    hidden["clips"] = json!(hidden["clips"].as_u64().unwrap_or(0) + o["clipsChanged"].as_u64().unwrap_or(0));
                    if !o["verify"]["state"].is_null() {
                        hidden["verify"] = o["verify"]["state"].clone();
                        report.models.redaction = o["verify"]["state"].as_str().map(|s| format!("detectors + robot check: {s}"));
                    }
                    if matches!(o["verify"]["state"].as_str(), Some("skipped" | "failed")) {
                        report.fallbacks.push(format!(
                            "privacy: the check of who is a person did not run ({}): every face found is hidden",
                            chars(o["verify"]["reason"].as_str().unwrap_or(""), 160)
                        ));
                    }
                }
                Err(e) => report
                    .fallbacks
                    .push(format!("privacy: nothing was hidden ({}): `filmcraft-cli quick --download` fetches the detectors", chars(&e.to_string(), 200))),
            }
        }
        report.privacy = Some(hidden);
    }
    // the mix at the social loudness (−14 LUFS), its peaks under −1 dBTP
    if let Err(e) = run(ex, "audio.normalizeLoudness", json!({"target": "social", "truePeak": crate::look::TRUE_PEAK})).await {
        notes.push(format!("loudness: {e}"));
    }
    // what still reaches into the platform's interface is said, not hidden
    match run(ex, "sequence.checkSafeZones", json!({"platform": opts.platform.name(), "sequence": seq})).await {
        Ok(v) => {
            let overlaps = v["overlaps"].as_array().map(Vec::as_slice).unwrap_or_default();
            for o in overlaps.iter().take(10) {
                notes.push(format!(
                    "safe zones: \"{}\" at {:.1} s reaches into {}'s {} zone",
                    chars(o["text"].as_str().unwrap_or_default(), 50),
                    finite(&o["seconds"]).unwrap_or(0.0),
                    opts.platform.label(),
                    o["zones"].as_array().map(|z| z.iter().filter_map(Value::as_str).take(4).collect::<Vec<_>>().join(" and ")).unwrap_or_default()
                ));
            }
            report.safe_zones = Some(json!({"platform": opts.platform.name(), "ok": overlaps.is_empty(), "overlaps": overlaps.len()}));
        }
        Err(e) => notes.push(format!("safe zones: {e}")),
    }
    // the files
    let Some(out) = opts.export.clone() else { return };
    let out = filmcraft_engine::export_tools::expand_home(&out);
    if let Some(dir) = Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty())
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        notes.push(format!("export: {}: {e}", dir.display()));
        return;
    }
    lines(&format!("exporting {out}"));
    let mut ep = json!({"sequence": seq, "path": out, "wait": true, "burnCaptions": true});
    match applied["exportPreset"].as_str() {
        Some(pr) => ep["preset"] = json!(pr),
        None => ep["format"] = json!("h264"),
    }
    let written = match run(ex, "file.exportMedia", ep).await {
        Ok(r) => r["path"].as_str().or_else(|| r["result"]["path"].as_str()).map(str::to_string).unwrap_or_else(|| out.clone()),
        Err(e) => {
            notes.push(format!("export: {e}"));
            return;
        }
    };
    report.export = Some(written.clone());
    let times: Vec<f64> = (0..SHEET_FRAMES).map(|k| ((k as f64 + 0.5) * seconds / SHEET_FRAMES as f64 * 30.0).round() / 30.0).collect();
    let sheet = crate::trailer::beside(&written, "sheet.jpg");
    match crate::trailer::frames_sheet(&written, &times, &sheet).await {
        Ok(()) => report.sheet = Some(sheet.display().to_string()),
        Err(e) => notes.push(format!("sheet: {e}")),
    }
    if opts.qa.unwrap_or(true) {
        lines("QA");
        run(ex, "sequence.open", json!({"item": seq})).await.ok();
        // the template's sound: someone talking on A1 and the music on A2, else the music on A1
        let talks = t.slots.iter().any(|s| s.picture() && s.ai.as_ref().is_some_and(Hints::wants_speech));
        let (voice, music) = if talks { ("A1", "A2") } else { ("", "A1") };
        let q = crate::qa::Options {
            subject: opts.topic.clone().unwrap_or_else(|| "subject".into()),
            picture_track: "V1".into(),
            sound_track: String::new(),
            voice_track: voice.into(),
            music_track: music.into(),
            faces: opts.redact.as_ref().is_some_and(|r| r.faces) && !picks.iter().any(|p| p.said.is_some()),
            platform: Some(opts.platform.name().into()),
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
#[path = "quick_tests.rs"]
mod tests;
