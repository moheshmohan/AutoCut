//! The template manifest (`template.json`) and its strict reader.
//!
//! The reader does not use serde's derive: it walks the JSON by hand so that every problem —
//! a missing field, a wrong type, a value out of range, an unknown key — is reported with the
//! path of the field it is about (`slots[2].constraints.maxDuration`), and all of them at once.
//! Writing uses serde (camelCase), and the two are kept in step by round-trip tests.

use filmcraft_project::CaptionStyle;
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{FORMAT, FORMAT_VERSION, Issues, SEQUENCE_FILE, TemplateError};

/// Largest manifest file, bytes.
pub const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
/// Most slots a template may have.
pub const MAX_SLOTS: usize = 64;
/// Most placements of one slot.
pub const MAX_PLACEMENTS: usize = 256;
/// Most media one slot takes.
pub const MAX_COUNT: u32 = 256;
/// Most aspect variants.
pub const MAX_VARIANTS: usize = 8;
/// Most position hints of one variant.
pub const MAX_HINTS: usize = 1024;
/// Most beats in a beat grid.
pub const MAX_BEATS: usize = 20_000;
/// Most bundled assets.
pub const MAX_ASSETS: usize = 256;
/// Longest template, seconds.
pub const MAX_TEMPLATE_SECONDS: i64 = 60 * 60;
/// Longest template, ticks.
pub const MAX_TEMPLATE_TICKS: i64 = MAX_TEMPLATE_SECONDS * TICKS_PER_SECOND;
/// Longest short text field (ids, names, licences…), bytes.
const MAX_SHORT: usize = 256;
/// Longest long text field (descriptions, default texts), bytes.
const MAX_LONG: usize = 16 * 1024;

/// What a slot takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SlotKind {
    Video,
    Photo,
    Text,
    Logo,
    Music,
    /// Captions of what is said in other slots' media (`captions.source`): transcribed when the
    /// template is applied (Whisper), or given as cues. Shown on the template's first caption
    /// track in its caption style; it has no placements.
    Captions,
}

impl SlotKind {
    pub const ALL: [SlotKind; 6] = [SlotKind::Video, SlotKind::Photo, SlotKind::Text, SlotKind::Logo, SlotKind::Music, SlotKind::Captions];
    pub fn id(self) -> &'static str {
        match self {
            SlotKind::Video => "video",
            SlotKind::Photo => "photo",
            SlotKind::Text => "text",
            SlotKind::Logo => "logo",
            SlotKind::Music => "music",
            SlotKind::Captions => "captions",
        }
    }
    pub fn from_id(s: &str) -> Option<SlotKind> {
        SlotKind::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s))
    }
    /// Slots whose placements are pictures on video tracks.
    pub fn is_visual(self) -> bool {
        matches!(self, SlotKind::Video | SlotKind::Photo | SlotKind::Logo)
    }
}

/// The shape of media a slot wants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Orientation {
    #[default]
    Any,
    Landscape,
    Portrait,
    Square,
}

impl Orientation {
    pub fn id(self) -> &'static str {
        match self {
            Orientation::Any => "any",
            Orientation::Landscape => "landscape",
            Orientation::Portrait => "portrait",
            Orientation::Square => "square",
        }
    }
    /// Whether a `w`×`h` picture has this orientation (square: within 5 %).
    pub fn accepts(self, w: f64, h: f64) -> bool {
        if !(w > 0.0 && h > 0.0) {
            return self == Orientation::Any;
        }
        let r = w / h;
        match self {
            Orientation::Any => true,
            Orientation::Square => (r - 1.0).abs() <= 0.05,
            Orientation::Landscape => r > 1.05,
            Orientation::Portrait => r < 1.0 / 1.05,
        }
    }
}

/// What happens when a slot's media is shorter than its placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ShortFit {
    /// Play the media again from its start until the placement is full.
    #[default]
    Loop,
    /// Slow the media down to fill the placement (no slower than `minSpeed`; past that, loop).
    Stretch,
    /// Play the media, then hold its last frame.
    Hold,
    /// Music: Remix it to the length at musically similar beats (past what Remix can do, loop).
    Remix,
    /// Refuse media that is too short.
    Error,
}

impl ShortFit {
    pub fn id(self) -> &'static str {
        match self {
            ShortFit::Loop => "loop",
            ShortFit::Stretch => "stretch",
            ShortFit::Hold => "hold",
            ShortFit::Remix => "remix",
            ShortFit::Error => "error",
        }
    }
}

/// Which part of a longer source fills the placement when the caller does not say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Portion {
    #[default]
    Start,
    Center,
    End,
}

/// How a picture is framed in the variant's frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Framing {
    /// Cover the whole frame, cropping what is outside (around the focus point).
    #[default]
    Fill,
    /// Fit inside the frame, whole.
    Fit,
    /// Source pixels as they are.
    None,
}

impl Framing {
    pub fn id(self) -> &'static str {
        match self {
            Framing::Fill => "fill",
            Framing::Fit => "fit",
            Framing::None => "none",
        }
    }
}

/// A slot's auto-fit policy: media conforms to the slot, never the reverse.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fit {
    /// Media shorter than the placement.
    pub short: ShortFit,
    /// Which part of longer media is used (the caller's `in` point wins).
    pub portion: Portion,
    /// How pictures are framed.
    pub frame: Framing,
    /// The slowest `stretch` may play the media (0.5 = half speed).
    pub min_speed: f64,
}

impl Default for Fit {
    fn default() -> Self {
        Fit { short: ShortFit::Loop, portion: Portion::Start, frame: Framing::Fill, min_speed: 0.5 }
    }
}

/// What media a slot accepts.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Constraints {
    /// Shortest and longest each placement may become when the template is re-timed (a new
    /// track's beats move the cuts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_duration: Option<Tick>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_duration: Option<Tick>,
    pub orientation: Orientation,
    /// Wanted picture aspect (`[16, 9]`), within 5 %.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect: Option<[u32; 2]>,
    /// A picture of the wrong orientation or aspect is an error instead of a warning (it is
    /// otherwise framed by the slot's `frame` rule, cropping what does not fit).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
    /// Text slots: the longest text, characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<u32>,
}

/// Where a slot's media appears: one clip of the template sequence.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placement {
    /// The clip (id in the template sequence).
    pub clip: u64,
    /// Which of the slot's media fills it (0-based; several placements may share one, and with
    /// fewer media than the slot's `count` they are reused in turn).
    pub media: u32,
    /// Text slots: the text layer (its uid); default the clip's first text layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<u64>,
    /// Picture slots: the part of the frame the picture fills (`[x, y, width, height]`,
    /// fractions of the frame): framed in that box by the slot's `frame` rule and cropped to it —
    /// a collage cell, one side of a split screen, a picture-in-picture. Default: the whole frame.
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub area: Option<[f64; 4]>,
}

/// What an unfilled slot becomes.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SlotDefault {
    /// Text slots: this text (else the template's own placeholder text stays).
    Text(String),
    /// Music slots: a bed generated in the app (`audio.generate`) at the template's length, in
    /// `mood` (and `genre`), at `bpm`.
    Generate {
        mood: String,
        bpm: f64,
        seed: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        genre: Option<String>,
    },
}

/// Moods a generated music bed may have (`audio.generate`'s).
pub const MOODS: [&str; 9] = ["epic", "tense", "uplifting", "calm", "happy", "dark", "dreamy", "energetic", "sad"];
/// Genres a generated music bed may have (`audio.generate`'s).
pub const GENRES: [&str; 6] = ["cinematic", "electronic", "pop", "ambient", "lofi", "piano"];

/// How a captions slot captions its sources.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionSpec {
    /// The slots (video or music) whose speech is captioned.
    pub source: Vec<String>,
    /// Longest caption line, characters.
    pub max_chars: u32,
    /// Lines per caption (1 or 2).
    pub lines: u32,
}

impl Default for CaptionSpec {
    fn default() -> Self {
        CaptionSpec { source: Vec::new(), max_chars: 32, lines: 1 }
    }
}

/// A typed placeholder.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Slot {
    /// Stable id (`[a-z0-9_-]`), unique in the template; callers fill slots by it.
    pub id: String,
    pub kind: SlotKind,
    pub name: String,
    /// What to put here, for the person filling it.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    /// Fill order in the slot picker.
    pub order: u32,
    /// How many media the slot takes.
    pub count: u32,
    /// The fewest media accepted (0 = the slot is optional and keeps its placeholder or default).
    pub min_count: u32,
    pub placements: Vec<Placement>,
    pub constraints: Constraints,
    pub fit: Fit,
    /// The cuts at this slot's placements snap to the beat grid.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub beat_sync: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<SlotDefault>,
    /// Captions slots: what is captioned and how.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captions: Option<CaptionSpec>,
}

/// A graphic layer's place in one variant.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerHint {
    /// The layer's uid.
    pub layer: u64,
    /// Position as a fraction of the frame (`[0.5, 0.8]`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Text size (pixels in this variant's frame).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
}

/// Where a clip goes in one variant (the Magic-Resize model: one design, per-layout hints).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hint {
    pub clip: u64,
    /// Where the picture's focus lands, as a fraction of the frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Size as a percentage of the template's own in this layout (140 = 1.4 times larger).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame: Option<Framing>,
    /// The box the picture fills in this layout (fractions of the frame; see [`Placement::area`]).
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub area: Option<[f64; 4]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerHint>,
}

/// One aspect layout of the template.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Variant {
    /// `16:9`, `9:16`, `1:1`, `4:5`… (unique).
    pub id: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Export preset for this layout (else the template's).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_preset: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<Hint>,
}

/// The template's grade.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Look {
    /// A Lumetri preset by name (`lumetri.presets`).
    Lumetri(String),
    /// An effect preset by name (`presets.list`).
    Preset(String),
}

/// How cuts follow a new track's beats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum QuantizeMode {
    /// Each cut keeps its beat number: a slower song makes a longer video.
    #[default]
    Beats,
    /// Each cut moves to the beat nearest it: the length stays.
    Nearest,
}

/// The music's beats, precomputed as timeline ticks (from `media.beats` or a steady tempo).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BeatGrid {
    pub bpm: f64,
    pub beats: Vec<Tick>,
    pub mode: QuantizeMode,
}

impl BeatGrid {
    /// A steady grid from `offset` at `bpm` covering `[0, until]` (bpm clamped to 20..300).
    pub fn steady(bpm: f64, offset: Tick, until: Tick) -> BeatGrid {
        let bpm = if bpm.is_finite() { bpm.clamp(20.0, 300.0) } else { 120.0 };
        let period = (TICKS_PER_SECOND as f64 * 60.0 / bpm).round() as i64;
        let mut beats = Vec::new();
        let mut k = 0i64;
        while beats.len() < MAX_BEATS {
            let t = Tick(offset.0.saturating_add(k.saturating_mul(period)));
            if t > until {
                break;
            }
            beats.push(t);
            k += 1;
        }
        BeatGrid { bpm, beats, mode: QuantizeMode::Beats }
    }
}

/// A bundled file and where it came from (the package's attribution manifest).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetEntry {
    /// Path in the package (`assets/whoosh.wav`).
    pub path: String,
    /// Lower-case hex SHA-256 of the file.
    pub sha256: String,
    pub title: String,
    pub author: String,
    /// URL, or "original work".
    pub source: String,
    /// SPDX id (no NC / ND licences).
    pub license: String,
}

/// `template.json`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    /// Stable id (`builtin:…` for the built-in starters, `user:…` for saved ones).
    pub id: String,
    pub name: String,
    pub description: String,
    /// The template's own version (`1.0.0`).
    pub version: String,
    pub author: String,
    /// SPDX licence of the template.
    pub license: String,
    pub category: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// The sequence file in the package (`sequence.json`).
    pub sequence_file: String,
    /// The template's sequence: its item id in the sequence file.
    pub sequence: u64,
    /// Length of the template sequence.
    pub duration: Tick,
    pub variants: Vec<Variant>,
    pub default_variant: String,
    pub slots: Vec<Slot>,
    /// The moment its thumbnail shows (default: a third of the way in).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poster: Option<Tick>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub look: Option<Look>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption_style: Option<CaptionStyle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_preset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_grid: Option<BeatGrid>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<AssetEntry>,
}

impl Manifest {
    /// Pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    pub fn slot(&self, id: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.id == id)
    }
    /// A variant by id (`9:16`) or name, case-insensitively; `None` picks the default.
    pub fn variant(&self, key: Option<&str>) -> Option<&Variant> {
        let key = key.unwrap_or(&self.default_variant);
        self.variants.iter().find(|v| v.id.eq_ignore_ascii_case(key)).or_else(|| self.variants.iter().find(|v| v.name.eq_ignore_ascii_case(key)))
    }
    /// The slot and placement index of a template clip.
    pub fn placement_of(&self, clip: u64) -> Option<(&Slot, usize)> {
        self.slots.iter().find_map(|s| s.placements.iter().position(|p| p.clip == clip).map(|i| (s, i)))
    }
    /// The moment the thumbnail shows: `poster`, else a third of the way in.
    pub fn poster_time(&self) -> Tick {
        self.poster.unwrap_or(Tick(self.duration.0 / 3)).max(Tick::ZERO).min(self.duration.max(Tick::ZERO))
    }
    /// Slots that take media (video, photo, logo, music).
    pub fn media_slots(&self) -> usize {
        self.slots.iter().filter(|s| !matches!(s.kind, SlotKind::Text | SlotKind::Captions)).count()
    }
    /// How many clips or photos it takes: the counts of its picture slots (video, photo, logo).
    pub fn media_count(&self) -> usize {
        self.slots.iter().filter(|s| s.kind.is_visual()).map(|s| s.count as usize).sum()
    }
    /// The captions slot, if the template has one.
    pub fn captions_slot(&self) -> Option<&Slot> {
        self.slots.iter().find(|s| s.kind == SlotKind::Captions)
    }
}

// ------------------------------------------------------------------------------------------------
// The reader
// ------------------------------------------------------------------------------------------------

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// A JSON object being read: remembers which keys were read, so the rest are reported unknown.
pub(crate) struct Obj<'a> {
    path: String,
    map: &'a Map<String, Value>,
    used: Vec<&'static str>,
}

impl<'a> Obj<'a> {
    pub(crate) fn of(v: &'a Value, path: &str, is: &mut Issues) -> Option<Obj<'a>> {
        match v {
            Value::Object(map) => Some(Obj { path: path.to_string(), map, used: Vec::new() }),
            other => {
                is.push(path, format!("expected an object, found {}", kind_of(other)));
                None
            }
        }
    }
    pub(crate) fn field(&self, k: &str) -> String {
        if self.path.is_empty() { k.to_string() } else { format!("{}.{k}", self.path) }
    }
    fn get(&mut self, k: &'static str) -> Option<&'a Value> {
        self.used.push(k);
        self.map.get(k).filter(|v| !v.is_null())
    }
    fn missing(&self, is: &mut Issues, k: &str) {
        is.push(self.field(k), "required field is missing");
    }
    pub(crate) fn string(&mut self, is: &mut Issues, k: &'static str, max: usize, required: bool) -> Option<String> {
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(Value::String(s)) if s.len() > max => {
                is.push(self.field(k), format!("longer than {max} bytes"));
                None
            }
            Some(Value::String(s)) => Some(s.clone()),
            Some(o) => {
                is.push(self.field(k), format!("expected a string, found {}", kind_of(o)));
                None
            }
        }
    }
    pub(crate) fn u64(&mut self, is: &mut Issues, k: &'static str, min: u64, max: u64, required: bool) -> Option<u64> {
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(v) => match v.as_u64() {
                Some(n) if (min..=max).contains(&n) => Some(n),
                Some(n) => {
                    is.push(self.field(k), format!("{n} is out of range ({min} to {max})"));
                    None
                }
                None => {
                    is.push(self.field(k), format!("expected a whole number from {min} to {max}, found {}", show(v)));
                    None
                }
            },
        }
    }
    pub(crate) fn u32(&mut self, is: &mut Issues, k: &'static str, min: u32, max: u32, required: bool) -> Option<u32> {
        self.u64(is, k, u64::from(min), u64::from(max), required).and_then(|n| u32::try_from(n).ok())
    }
    pub(crate) fn tick(&mut self, is: &mut Issues, k: &'static str, min: i64, max: i64, required: bool) -> Option<Tick> {
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(v) => match v.as_i64() {
                Some(n) if (min..=max).contains(&n) => Some(Tick(n)),
                Some(n) => {
                    is.push(self.field(k), format!("{n} ticks is out of range ({min} to {max})"));
                    None
                }
                None => {
                    is.push(self.field(k), format!("expected whole ticks (254016000000 a second), found {}", show(v)));
                    None
                }
            },
        }
    }
    pub(crate) fn f64(&mut self, is: &mut Issues, k: &'static str, min: f64, max: f64, required: bool) -> Option<f64> {
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() && x >= min && x <= max => Some(x),
                Some(x) => {
                    is.push(self.field(k), format!("{x} is out of range ({min} to {max})"));
                    None
                }
                None => {
                    is.push(self.field(k), format!("expected a number, found {}", kind_of(v)));
                    None
                }
            },
        }
    }
    pub(crate) fn bool(&mut self, is: &mut Issues, k: &'static str) -> Option<bool> {
        match self.get(k) {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(o) => {
                is.push(self.field(k), format!("expected true or false, found {}", kind_of(o)));
                None
            }
        }
    }
    pub(crate) fn array(&mut self, is: &mut Issues, k: &'static str, max: usize, required: bool) -> Option<&'a Vec<Value>> {
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(Value::Array(a)) if a.len() > max => {
                is.push(self.field(k), format!("{} entries (at most {max})", a.len()));
                None
            }
            Some(Value::Array(a)) => Some(a),
            Some(o) => {
                is.push(self.field(k), format!("expected an array, found {}", kind_of(o)));
                None
            }
        }
    }
    pub(crate) fn object(&mut self, is: &mut Issues, k: &'static str, required: bool) -> Option<Obj<'a>> {
        let path = self.field(k);
        match self.get(k) {
            None => {
                if required {
                    self.missing(is, k);
                }
                None
            }
            Some(v) => Obj::of(v, &path, is),
        }
    }
    pub(crate) fn raw(&mut self, k: &'static str) -> Option<&'a Value> {
        self.get(k)
    }
    pub(crate) fn choice<T: Copy>(&mut self, is: &mut Issues, k: &'static str, options: &[(&str, T)]) -> Option<T> {
        let s = self.string(is, k, MAX_SHORT, false)?;
        match options.iter().find(|(n, _)| n.eq_ignore_ascii_case(&s)) {
            Some((_, v)) => Some(*v),
            None => {
                let names: Vec<&str> = options.iter().map(|(n, _)| *n).collect();
                is.push(self.field(k), format!("`{s}` is not one of {}", names.join(", ")));
                None
            }
        }
    }
    /// Report every key that was not read.
    pub(crate) fn finish(self, is: &mut Issues) {
        for k in self.map.keys() {
            if !self.used.iter().any(|u| u == k) {
                is.push(self.field(k), "unknown field");
            }
        }
    }
}

fn show(v: &Value) -> String {
    match v {
        Value::Number(n) => n.to_string(),
        o => kind_of(o).to_string(),
    }
}

/// `[a-z0-9_-]`, 1–64 characters.
pub fn is_slot_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Template ids: `[a-z0-9:._-]`, 1–128 characters.
pub fn is_template_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-:.".contains(&b))
}

/// Licences that are not open enough to ship (AGENTS.md §1: no NC / ND, nothing proprietary).
pub fn licence_refused(l: &str) -> bool {
    let l = l.to_ascii_lowercase();
    l.is_empty() || l.contains("-nc") || l.contains("-nd") || l.contains("proprietary") || l.contains("all rights reserved") || l.contains("adobe")
}

fn pair_f64(v: &Value, field: &str, is: &mut Issues, lo: f64, hi: f64) -> Option<[f64; 2]> {
    let a = v.as_array().filter(|a| a.len() == 2);
    let xy = a.and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]));
    match xy {
        Some([x, y]) if x.is_finite() && y.is_finite() && (lo..=hi).contains(&x) && (lo..=hi).contains(&y) => Some([x, y]),
        _ => {
            is.push(field, format!("expected [x, y] with numbers from {lo} to {hi}"));
            None
        }
    }
}

/// A box `[x, y, width, height]` in fractions of the frame: the corner within -1..2, the size
/// more than 0 and at most 2.
fn read_box(v: &Value, field: &str, is: &mut Issues) -> Option<[f64; 4]> {
    let a = v.as_array().filter(|a| a.len() == 4);
    let b = a.and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?, a.get(3)?.as_f64()?]));
    match b {
        Some([x, y, w, h])
            if [x, y, w, h].iter().all(|n| n.is_finite())
                && (-1.0..=2.0).contains(&x)
                && (-1.0..=2.0).contains(&y)
                && w > 0.0
                && h > 0.0
                && w <= 2.0
                && h <= 2.0 =>
        {
            Some([x, y, w, h])
        }
        _ => {
            is.push(field, "expected [x, y, width, height] in fractions of the frame (corner -1 to 2, size more than 0 and at most 2)");
            None
        }
    }
}

/// Parse an aspect such as `16:9` (or `[16, 9]`); terms 1–100.
pub fn parse_aspect(v: &Value) -> Option<[u32; 2]> {
    let (a, b) = match v {
        Value::String(s) => {
            let (a, b) = s.split_once(':').or_else(|| s.split_once('x'))?;
            (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?)
        }
        Value::Array(x) if x.len() == 2 => (u32::try_from(x.first()?.as_u64()?).ok()?, u32::try_from(x.get(1)?.as_u64()?).ok()?),
        _ => return None,
    };
    ((1..=100).contains(&a) && (1..=100).contains(&b)).then_some([a, b])
}

const ORIENTATIONS: [(&str, Orientation); 4] =
    [("any", Orientation::Any), ("landscape", Orientation::Landscape), ("portrait", Orientation::Portrait), ("square", Orientation::Square)];
const SHORT_FITS: [(&str, ShortFit); 5] =
    [("loop", ShortFit::Loop), ("stretch", ShortFit::Stretch), ("hold", ShortFit::Hold), ("remix", ShortFit::Remix), ("error", ShortFit::Error)];
const PORTIONS: [(&str, Portion); 3] = [("start", Portion::Start), ("center", Portion::Center), ("end", Portion::End)];
const FRAMINGS: [(&str, Framing); 3] = [("fill", Framing::Fill), ("fit", Framing::Fit), ("none", Framing::None)];
const MODES: [(&str, QuantizeMode); 2] = [("beats", QuantizeMode::Beats), ("nearest", QuantizeMode::Nearest)];

/// The default fit of a slot kind.
pub fn default_fit(kind: SlotKind) -> Fit {
    match kind {
        SlotKind::Video => Fit { short: ShortFit::Stretch, portion: Portion::Center, ..Fit::default() },
        SlotKind::Photo => Fit::default(),
        SlotKind::Logo => Fit { frame: Framing::Fit, ..Fit::default() },
        SlotKind::Music => Fit { short: ShortFit::Remix, frame: Framing::None, ..Fit::default() },
        SlotKind::Text | SlotKind::Captions => Fit { frame: Framing::None, ..Fit::default() },
    }
}

fn read_fit(o: Option<Obj<'_>>, kind: SlotKind, is: &mut Issues) -> Fit {
    let mut f = default_fit(kind);
    let Some(mut o) = o else { return f };
    if let Some(v) = o.choice(is, "short", &SHORT_FITS) {
        f.short = v;
    }
    if let Some(v) = o.choice(is, "portion", &PORTIONS) {
        f.portion = v;
    }
    if let Some(v) = o.choice(is, "frame", &FRAMINGS) {
        f.frame = v;
    }
    if let Some(v) = o.f64(is, "minSpeed", 0.1, 1.0, false) {
        f.min_speed = v;
    }
    if f.short == ShortFit::Remix && kind != SlotKind::Music {
        is.push(o.field("short"), "`remix` is for music slots");
    }
    o.finish(is);
    f
}

fn read_constraints(o: Option<Obj<'_>>, kind: SlotKind, is: &mut Issues) -> Constraints {
    let mut c = Constraints::default();
    let Some(mut o) = o else { return c };
    c.min_duration = o.tick(is, "minDuration", 1, MAX_TEMPLATE_TICKS, false);
    c.max_duration = o.tick(is, "maxDuration", 1, MAX_TEMPLATE_TICKS, false);
    if let (Some(a), Some(b)) = (c.min_duration, c.max_duration)
        && a > b
    {
        is.push(o.field("minDuration"), "is longer than maxDuration");
    }
    if let Some(v) = o.choice(is, "orientation", &ORIENTATIONS) {
        c.orientation = v;
    }
    if let Some(v) = o.raw("aspect") {
        match parse_aspect(v) {
            Some(a) => c.aspect = Some(a),
            None => is.push(o.field("aspect"), "expected \"W:H\" with whole numbers from 1 to 100"),
        }
    }
    c.strict = o.bool(is, "strict").unwrap_or(false);
    c.max_chars = o.u32(is, "maxChars", 1, 10_000, false);
    if c.max_chars.is_some() && kind != SlotKind::Text {
        is.push(o.field("maxChars"), "only text slots take maxChars");
    }
    if (c.aspect.is_some() || c.orientation != Orientation::Any) && !kind.is_visual() {
        is.push(o.field("orientation"), "only video, photo and logo slots take an orientation or aspect");
    }
    o.finish(is);
    c
}

fn read_default(o: Option<Obj<'_>>, kind: SlotKind, is: &mut Issues) -> Option<SlotDefault> {
    let mut o = o?;
    let text = o.string(is, "text", MAX_LONG, false);
    let generate = o.object(is, "generate", false);
    let r = match (text, generate) {
        (Some(t), None) if kind == SlotKind::Text => Some(SlotDefault::Text(t)),
        (Some(_), None) => {
            is.push(o.field("text"), "only text slots have a default text");
            None
        }
        (None, Some(mut g)) => {
            let mood = g.string(is, "mood", 32, false).unwrap_or_else(|| "uplifting".into());
            if !MOODS.contains(&mood.as_str()) {
                is.push(g.field("mood"), format!("`{mood}` is not one of {}", MOODS.join(", ")));
            }
            let bpm = g.f64(is, "bpm", 40.0, 200.0, false).unwrap_or(120.0);
            let seed = g.u64(is, "seed", 0, u64::from(u32::MAX), false).unwrap_or(1);
            let genre = g.string(is, "genre", 32, false);
            if let Some(x) = &genre
                && !GENRES.contains(&x.as_str())
            {
                is.push(g.field("genre"), format!("`{x}` is not one of {}", GENRES.join(", ")));
            }
            g.finish(is);
            if kind != SlotKind::Music {
                is.push(o.field("generate"), "only music slots can generate a default");
            }
            Some(SlotDefault::Generate { mood, bpm, seed, genre })
        }
        (Some(_), Some(_)) => {
            is.push(o.field("text"), "give `text` or `generate`, not both");
            None
        }
        (None, None) => {
            is.push(o.field("text"), "expected `text` or `generate`");
            None
        }
    };
    o.finish(is);
    r
}

fn read_slot(v: &Value, path: &str, is: &mut Issues) -> Option<Slot> {
    let mut o = Obj::of(v, path, is)?;
    let id = o.string(is, "id", 64, true);
    if let Some(id) = &id
        && !is_slot_id(id)
    {
        is.push(o.field("id"), format!("`{id}`: use lower-case letters, digits, `_` and `-`"));
    }
    let kind_s = o.string(is, "kind", 32, true);
    let kind = kind_s.as_deref().and_then(|k| {
        let r = SlotKind::from_id(k);
        if r.is_none() {
            is.push(o.field("kind"), format!("`{k}` is not one of video, photo, text, logo, music"));
        }
        r
    });
    let name = o.string(is, "name", MAX_SHORT, false);
    let prompt = o.string(is, "prompt", MAX_LONG, false).unwrap_or_default();
    let order = o.u32(is, "order", 0, 10_000, false).unwrap_or(0);
    let count = o.u32(is, "count", 1, MAX_COUNT, false).unwrap_or(1);
    let min_count = o.u32(is, "minCount", 0, MAX_COUNT, false).unwrap_or(1);
    if min_count > count {
        is.push(o.field("minCount"), format!("{min_count} is more than count ({count})"));
    }
    let mut placements = Vec::new();
    let captions_kind = kind == Some(SlotKind::Captions);
    if let Some(a) = o.array(is, "placements", MAX_PLACEMENTS, !captions_kind) {
        if a.is_empty() && !captions_kind {
            is.push(o.field("placements"), "a slot needs at least one placement");
        }
        if !a.is_empty() && captions_kind {
            is.push(o.field("placements"), "a captions slot has no placements (its captions go on the first caption track)");
        }
        for (i, p) in a.iter().enumerate() {
            let pp = format!("{}[{i}]", o.field("placements"));
            let Some(mut po) = Obj::of(p, &pp, is) else { continue };
            let clip = po.u64(is, "clip", 1, u64::MAX >> 2, true);
            let media = po.u32(is, "media", 0, MAX_COUNT - 1, false).unwrap_or(i as u32 % count.max(1));
            if media >= count {
                is.push(po.field("media"), format!("media {media} is past the slot's count ({count})"));
            }
            let layer = po.u64(is, "layer", 1, u64::MAX >> 2, false);
            if layer.is_some() && kind.is_some_and(|k| k != SlotKind::Text) {
                is.push(po.field("layer"), "only text slots name a layer");
            }
            let area = po.raw("box").and_then(|b| read_box(b, &po.field("box"), is));
            if area.is_some() && kind.is_some_and(|k| !k.is_visual()) {
                is.push(po.field("box"), "only video, photo and logo slots take a box");
            }
            po.finish(is);
            if let Some(clip) = clip {
                if placements.iter().any(|x: &Placement| x.clip == clip && x.layer == layer) {
                    is.push(po_field(&pp, "clip"), format!("clip {clip} is placed twice"));
                }
                placements.push(Placement { clip, media, layer, area });
            }
        }
        for m in 0..count {
            if !placements.is_empty() && !placements.iter().any(|p| p.media == m) {
                is.push(o.field("count"), format!("media {m} has no placement"));
                break;
            }
        }
    }
    let kind = kind?;
    if matches!(kind, SlotKind::Music | SlotKind::Captions) && count != 1 {
        is.push(o.field("count"), format!("a {} slot takes one", kind.id()));
    }
    let captions = match o.object(is, "captions", false) {
        Some(mut c) => {
            let mut spec = CaptionSpec::default();
            if let Some(a) = c.array(is, "source", 16, true) {
                for (i, x) in a.iter().enumerate() {
                    match x.as_str() {
                        Some(sid) if is_slot_id(sid) => spec.source.push(sid.to_string()),
                        _ => is.push(format!("{}[{i}]", c.field("source")), "expected a slot id"),
                    }
                }
                if a.is_empty() {
                    is.push(c.field("source"), "name the slots whose speech is captioned");
                }
            }
            spec.max_chars = c.u32(is, "maxChars", 8, 200, false).unwrap_or(spec.max_chars);
            spec.lines = c.u32(is, "lines", 1, 2, false).unwrap_or(spec.lines);
            c.finish(is);
            if kind != SlotKind::Captions {
                is.push(o.field("captions"), "only captions slots take `captions`");
            }
            Some(spec)
        }
        None => {
            if kind == SlotKind::Captions {
                is.push(o.field("captions"), "a captions slot needs `captions` ({\"source\": [slot ids]})");
            }
            None
        }
    };
    let constraints = read_constraints(o.object(is, "constraints", false), kind, is);
    let fit = read_fit(o.object(is, "fit", false), kind, is);
    let beat_sync = o.bool(is, "beatSync").unwrap_or(false);
    if beat_sync && !kind.is_visual() {
        is.push(o.field("beatSync"), "only video, photo and logo slots cut on the beat");
    }
    let default = read_default(o.object(is, "default", false), kind, is);
    o.finish(is);
    Some(Slot {
        id: id?,
        kind,
        name: name.unwrap_or_else(|| kind.id().into()),
        prompt,
        order,
        count,
        min_count,
        placements,
        constraints,
        fit,
        beat_sync,
        default,
        captions,
    })
}

fn po_field(path: &str, k: &str) -> String {
    format!("{path}.{k}")
}

fn read_hint(v: &Value, path: &str, is: &mut Issues) -> Option<Hint> {
    let mut o = Obj::of(v, path, is)?;
    let clip = o.u64(is, "clip", 1, u64::MAX >> 2, true);
    let position = o.raw("position").and_then(|p| pair_f64(p, &o.field("position"), is, -1.0, 2.0));
    let scale = o.f64(is, "scale", 1.0, 1000.0, false);
    let frame = o.choice(is, "frame", &FRAMINGS);
    let area = o.raw("box").and_then(|b| read_box(b, &o.field("box"), is));
    let mut layers = Vec::new();
    if let Some(a) = o.array(is, "layers", 256, false) {
        for (i, l) in a.iter().enumerate() {
            let lp = format!("{}[{i}]", o.field("layers"));
            let Some(mut lo) = Obj::of(l, &lp, is) else { continue };
            let layer = lo.u64(is, "layer", 1, u64::MAX >> 2, true);
            let position = lo.raw("position").and_then(|p| pair_f64(p, &lo.field("position"), is, -1.0, 2.0));
            let size = lo.f64(is, "size", 1.0, 2000.0, false);
            lo.finish(is);
            if let Some(layer) = layer {
                layers.push(LayerHint { layer, position, size });
            }
        }
    }
    o.finish(is);
    Some(Hint { clip: clip?, position, scale, frame, area, layers })
}

fn read_variant(v: &Value, path: &str, is: &mut Issues) -> Option<Variant> {
    let mut o = Obj::of(v, path, is)?;
    let id = o.string(is, "id", 32, true);
    let name = o.string(is, "name", MAX_SHORT, false);
    let width = o.u32(is, "width", 16, 8192, true);
    let height = o.u32(is, "height", 16, 8192, true);
    let export_preset = o.string(is, "exportPreset", MAX_SHORT, false);
    let mut hints = Vec::new();
    if let Some(a) = o.array(is, "hints", MAX_HINTS, false) {
        for (i, h) in a.iter().enumerate() {
            if let Some(h) = read_hint(h, &format!("{}[{i}]", o.field("hints")), is) {
                if hints.iter().any(|x: &Hint| x.clip == h.clip) {
                    is.push(format!("{}[{i}].clip", o.field("hints")), format!("clip {} has two hints", h.clip));
                }
                hints.push(h);
            }
        }
    }
    o.finish(is);
    let id = id?;
    if id.trim().is_empty() {
        is.push(format!("{path}.id"), "is empty");
    }
    Some(Variant { name: name.unwrap_or_else(|| id.clone()), id, width: width?, height: height?, export_preset, hints })
}

fn read_caption_style(v: &Value, path: &str, is: &mut Issues) -> Option<CaptionStyle> {
    let known = serde_json::to_value(CaptionStyle::default()).ok()?;
    let (Some(map), Some(known)) = (v.as_object(), known.as_object()) else {
        is.push(path, format!("expected an object, found {}", kind_of(v)));
        return None;
    };
    for k in map.keys() {
        if !known.contains_key(k) {
            is.push(format!("{path}.{k}"), "unknown field");
        }
    }
    let st: CaptionStyle = match serde_json::from_value(v.clone()) {
        Ok(s) => s,
        Err(e) => {
            is.push(path, e.to_string());
            return None;
        }
    };
    let mut ok = true;
    let mut check = |k: &str, x: f32, lo: f32, hi: f32| {
        if !(x.is_finite() && x >= lo && x <= hi) {
            is.push(format!("{path}.{k}"), format!("{x} is out of range ({lo} to {hi})"));
            ok = false;
        }
    };
    check("size", st.size, 4.0, 500.0);
    check("margin", st.margin, 0.0, 0.5);
    check("line_spacing", st.line_spacing, 0.5, 4.0);
    check("outline", st.outline, 0.0, 50.0);
    if st.font.len() > 128 {
        is.push(format!("{path}.font"), "longer than 128 bytes");
        ok = false;
    }
    ok.then_some(st)
}

/// An author's edit of one slot of kind `kind` (`template.save`'s `slots`): the slot fields that
/// are not about placements — `name`, `prompt`, `order`, `minCount`, `beatSync`, `fit`,
/// `constraints`, `default` — read and checked like the manifest's; unknown fields are errors.
pub fn read_slot_edit(v: &Value, kind: SlotKind, path: &str) -> Result<crate::save::SlotEdit, TemplateError> {
    let mut is = Issues::default();
    let Some(mut o) = Obj::of(v, path, &mut is) else { return Err(TemplateError::Invalid(is.0)) };
    let name = o.string(&mut is, "name", MAX_SHORT, false);
    let prompt = o.string(&mut is, "prompt", MAX_LONG, false);
    let order = o.u32(&mut is, "order", 0, 10_000, false);
    let min_count = o.u32(&mut is, "minCount", 0, MAX_COUNT, false);
    let beat_sync = o.bool(&mut is, "beatSync");
    let fit = o.object(&mut is, "fit", false).map(|f| read_fit(Some(f), kind, &mut is));
    let constraints = o.object(&mut is, "constraints", false).map(|c| read_constraints(Some(c), kind, &mut is));
    let default = o.object(&mut is, "default", false).and_then(|d| read_default(Some(d), kind, &mut is));
    o.finish(&mut is);
    if name.as_deref().is_some_and(|n| n.trim().is_empty()) {
        is.push(format!("{path}.name"), "is empty");
    }
    is.into_result()?;
    Ok(crate::save::SlotEdit { name, prompt, order, min_count, beat_sync, fit, constraints, default })
}

/// Read and check a manifest. All problems are reported at once, each with its field.
pub fn parse(bytes: &[u8]) -> Result<Manifest, TemplateError> {
    if crate::zip::is_zip(bytes) {
        return Err(TemplateError::NotTemplate("this is a ZIP archive; open it as a template package, not as a manifest".into()));
    }
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(TemplateError::NotTemplate(format!("the manifest is {} bytes (at most {MAX_MANIFEST_BYTES})", bytes.len())));
    }
    let text = std::str::from_utf8(bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes)).map_err(|_| TemplateError::NotTemplate("not UTF-8 text".into()))?;
    let v: Value = serde_json::from_str(text).map_err(|e| TemplateError::NotTemplate(format!("not JSON: {e}")))?;
    from_value(&v)
}

/// [`parse`] for JSON already read.
pub fn from_value(v: &Value) -> Result<Manifest, TemplateError> {
    let mut is = Issues::default();
    let Some(mut o) = Obj::of(v, "", &mut is) else { return Err(TemplateError::NotTemplate("the manifest is not a JSON object".into())) };
    match o.string(&mut is, "format", MAX_SHORT, false) {
        Some(f) if f == FORMAT => {}
        Some(f) => return Err(TemplateError::NotTemplate(format!("`format` is \"{f}\", not \"{FORMAT}\""))),
        None => return Err(TemplateError::NotTemplate(format!("no `format` (\"{FORMAT}\")"))),
    }
    match o.raw("formatVersion").map(|x| x.as_u64()) {
        Some(Some(n)) if n > u64::from(FORMAT_VERSION) => return Err(TemplateError::TooNew(u32::try_from(n).unwrap_or(u32::MAX))),
        Some(Some(n)) if n >= 1 => {}
        Some(_) => is.push("formatVersion", format!("expected 1 to {FORMAT_VERSION}")),
        None => is.push("formatVersion", "required field is missing"),
    }
    let id = o.string(&mut is, "id", 128, true);
    if let Some(id) = &id
        && !is_template_id(id)
    {
        is.push("id", format!("`{id}`: use lower-case letters, digits and `:._-`"));
    }
    let name = o.string(&mut is, "name", MAX_SHORT, true);
    if name.as_deref().is_some_and(|n| n.trim().is_empty()) {
        is.push("name", "is empty");
    }
    let description = o.string(&mut is, "description", MAX_LONG, false).unwrap_or_default();
    let version = o.string(&mut is, "version", 32, false).unwrap_or_else(|| "1.0.0".into());
    let author = o.string(&mut is, "author", MAX_SHORT, true).unwrap_or_default();
    let license = o.string(&mut is, "license", MAX_SHORT, true).unwrap_or_default();
    if o.map.contains_key("license") && licence_refused(&license) {
        is.push("license", format!("`{license}`: templates must be openly licensed (no NC / ND or proprietary licences)"));
    }
    let category = o.string(&mut is, "category", MAX_SHORT, false).unwrap_or_default();
    let mut tags = Vec::new();
    if let Some(a) = o.array(&mut is, "tags", 32, false) {
        for (i, t) in a.iter().enumerate() {
            match t.as_str() {
                Some(s) if s.len() <= 64 => tags.push(s.to_string()),
                _ => is.push(format!("tags[{i}]"), "expected a string of at most 64 bytes"),
            }
        }
    }
    let sequence_file = o.string(&mut is, "sequenceFile", MAX_SHORT, false).unwrap_or_else(|| SEQUENCE_FILE.into());
    if let Err(why) = crate::zip::check_entry_name(&sequence_file) {
        is.push("sequenceFile", why);
    } else if sequence_file.starts_with(crate::ASSETS_DIR) || sequence_file == crate::MANIFEST_FILE {
        is.push("sequenceFile", "must not be the manifest or inside assets/");
    }
    let sequence = o.u64(&mut is, "sequence", 1, u64::MAX >> 2, true);
    let duration = o.tick(&mut is, "duration", 1, MAX_TEMPLATE_TICKS, true);
    let mut variants = Vec::new();
    if let Some(a) = o.array(&mut is, "variants", MAX_VARIANTS, true) {
        if a.is_empty() {
            is.push("variants", "a template needs at least one variant");
        }
        for (i, v) in a.iter().enumerate() {
            if let Some(v) = read_variant(v, &format!("variants[{i}]"), &mut is) {
                if variants.iter().any(|x: &Variant| x.id.eq_ignore_ascii_case(&v.id)) {
                    is.push(format!("variants[{i}].id"), format!("`{}` is used twice", v.id));
                }
                variants.push(v);
            }
        }
    }
    let default_variant = o.string(&mut is, "defaultVariant", 32, false).or_else(|| variants.first().map(|v| v.id.clone())).unwrap_or_default();
    if !variants.is_empty() && !variants.iter().any(|v| v.id == default_variant) {
        is.push("defaultVariant", format!("`{default_variant}` is not a variant"));
    }
    let mut slots = Vec::new();
    if let Some(a) = o.array(&mut is, "slots", MAX_SLOTS, true) {
        for (i, s) in a.iter().enumerate() {
            if let Some(s) = read_slot(s, &format!("slots[{i}]"), &mut is) {
                if slots.iter().any(|x: &Slot| x.id == s.id) {
                    is.push(format!("slots[{i}].id"), format!("`{}` is used twice", s.id));
                }
                for p in &s.placements {
                    // two text slots may share a graphic clip, each with its own layer
                    let clash = |q: &Placement| q.clip == p.clip && (q.layer.is_none() || p.layer.is_none() || q.layer == p.layer);
                    if slots.iter().any(|x: &Slot| x.placements.iter().any(clash)) {
                        is.push(format!("slots[{i}].placements"), format!("clip {} is already placed by another slot", p.clip));
                    }
                }
                slots.push(s);
            }
        }
    }
    let total: usize = slots.iter().map(|s| s.placements.len()).sum();
    if total > 4 * MAX_PLACEMENTS {
        is.push("slots", format!("{total} placements in all (at most {})", 4 * MAX_PLACEMENTS));
    }
    // captions: one slot at most, captioning video or music slots of this template
    let mut captions_slots = 0;
    for (i, s) in slots.iter().enumerate() {
        let Some(c) = &s.captions else { continue };
        captions_slots += 1;
        if captions_slots > 1 {
            is.push(format!("slots[{i}].kind"), "a template has one captions slot at most");
        }
        for (k, src) in c.source.iter().enumerate() {
            match slots.iter().find(|x| x.id == *src) {
                Some(x) if matches!(x.kind, SlotKind::Video | SlotKind::Music) => {}
                Some(x) => {
                    is.push(format!("slots[{i}].captions.source[{k}]"), format!("`{src}` is a {} slot; captions come from video or music slots", x.kind.id()))
                }
                None => is.push(format!("slots[{i}].captions.source[{k}]"), format!("the template has no slot `{src}`")),
            }
        }
    }
    let poster = o.tick(&mut is, "poster", 0, MAX_TEMPLATE_TICKS, false);
    if let (Some(p), Some(d)) = (poster, duration)
        && p > d
    {
        is.push("poster", "is after the end of the template");
    }
    let look = o.object(&mut is, "look", false).and_then(|mut l| {
        let lumetri = l.string(&mut is, "lumetri", MAX_SHORT, false);
        let preset = l.string(&mut is, "preset", MAX_SHORT, false);
        let field = l.field("lumetri");
        l.finish(&mut is);
        match (lumetri, preset) {
            (Some(n), None) => Some(Look::Lumetri(n)),
            (None, Some(n)) => Some(Look::Preset(n)),
            _ => {
                is.push(field, "give `lumetri` (a Lumetri preset name) or `preset` (an effect preset name)");
                None
            }
        }
    });
    let caption_style = o.raw("captionStyle").and_then(|v| read_caption_style(v, "captionStyle", &mut is));
    let export_preset = o.string(&mut is, "exportPreset", MAX_SHORT, false);
    let beat_grid = o.object(&mut is, "beatGrid", false).and_then(|mut g| {
        let bpm = g.f64(&mut is, "bpm", 20.0, 300.0, true);
        let mode = g.choice(&mut is, "mode", &MODES).unwrap_or_default();
        let mut beats = Vec::new();
        if let Some(a) = g.array(&mut is, "beats", MAX_BEATS, true) {
            let mut last: Option<i64> = None;
            for (i, b) in a.iter().enumerate() {
                match b.as_i64() {
                    Some(t) if (0..=MAX_TEMPLATE_TICKS).contains(&t) && last.is_none_or(|l| t > l) => {
                        last = Some(t);
                        beats.push(Tick(t));
                    }
                    Some(t) if (0..=MAX_TEMPLATE_TICKS).contains(&t) => {
                        is.push(format!("{}[{i}]", g.field("beats")), "beats must be in increasing order");
                        break;
                    }
                    _ => {
                        is.push(format!("{}[{i}]", g.field("beats")), format!("expected ticks from 0 to {MAX_TEMPLATE_TICKS}"));
                        break;
                    }
                }
            }
            if beats.len() < 2 && !a.is_empty() {
                is.push(g.field("beats"), "a beat grid needs at least two beats");
            }
        }
        g.finish(&mut is);
        Some(BeatGrid { bpm: bpm?, beats, mode })
    });
    if beat_grid.is_none() && slots.iter().any(|s| s.beat_sync) {
        is.push("beatGrid", "slots cut on the beat but the template has no beat grid");
    }
    let mut assets = Vec::new();
    if let Some(a) = o.array(&mut is, "assets", MAX_ASSETS, false) {
        for (i, x) in a.iter().enumerate() {
            let path = format!("assets[{i}]");
            let Some(mut ao) = Obj::of(x, &path, &mut is) else { continue };
            let p = ao.string(&mut is, "path", 255, true);
            let sha = ao.string(&mut is, "sha256", 64, true);
            let title = ao.string(&mut is, "title", MAX_SHORT, true);
            let author = ao.string(&mut is, "author", MAX_SHORT, true);
            let source = ao.string(&mut is, "source", MAX_LONG, true);
            let lic = ao.string(&mut is, "license", MAX_SHORT, true);
            ao.finish(&mut is);
            if let Some(p) = &p {
                if let Err(why) = crate::zip::check_entry_name(p) {
                    is.push(format!("{path}.path"), why);
                } else if !p.starts_with(crate::ASSETS_DIR) {
                    is.push(format!("{path}.path"), "bundled files live in assets/");
                }
                if assets.iter().any(|a: &AssetEntry| a.path == *p) {
                    is.push(format!("{path}.path"), "listed twice");
                }
            }
            if let Some(s) = &sha
                && (s.len() != 64 || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
            {
                is.push(format!("{path}.sha256"), "expected 64 lower-case hex digits");
            }
            if let Some(l) = &lic
                && licence_refused(l)
            {
                is.push(format!("{path}.license"), format!("`{l}` is not an open licence (no NC / ND or proprietary assets)"));
            }
            if let (Some(path), Some(sha256), Some(title), Some(author), Some(source), Some(license)) = (p, sha, title, author, source, lic) {
                assets.push(AssetEntry { path, sha256, title, author, source, license });
            }
        }
    }
    o.finish(&mut is);
    is.into_result()?;
    let missing = || TemplateError::Invalid(vec![crate::Issue { field: String::new(), message: "incomplete manifest".into() }]);
    Ok(Manifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        id: id.ok_or_else(missing)?,
        name: name.ok_or_else(missing)?,
        description,
        version,
        author,
        license,
        category,
        tags,
        sequence_file,
        sequence: sequence.ok_or_else(missing)?,
        duration: duration.ok_or_else(missing)?,
        variants,
        default_variant,
        slots,
        poster,
        look,
        caption_style,
        export_preset,
        beat_grid,
        assets,
    })
}
