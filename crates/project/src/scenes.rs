//! Scene indexes of media items (`media.analyzeScenes`): the shots of a clip, what each shows,
//! how good it looks, and an embedding to search it by.
//!
//! A [`SceneIndex`] belongs to one media item (`Project::scenes`, keyed by the item id) and lists
//! its [`Shot`]s in **media time**, so it stays valid however the clip is trimmed or reused. Each
//! shot carries cheap picture measures ([`ShotQuality`]: sharpness, exposure, shake, and an
//! aesthetic score when the model for it is installed), the camera's measured motion
//! ([`CameraMove`]), a structured description from a vision-language model
//! ([`ShotDescription`], when one ran) and an image embedding ([`Embedding`], when the
//! embedding model ran) that text queries are ranked against.
//!
//! Everything here is derived from the media and can be made again (the engine also keeps it in a
//! cache by file content and model in the data directory), so the field is plain serde with
//! defaults: projects without it read as before, and builds without it drop it when saving, which
//! loses nothing that cannot be recomputed. Values read from a file are not trusted: numbers are
//! brought into range by [`SceneIndex::sanitized`] before use.

use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};

/// Version of the analysis an index was made by (bumped when what it measures changes).
pub const SCENE_INDEX_VERSION: u32 = 1;
/// Most shots an index keeps.
pub const MAX_SHOTS: usize = 10_000;
/// Most keyframes per shot.
pub const MAX_KEYFRAMES: usize = 8;
/// Most entries in a description's lists (subjects, actions, text lines, uses).
pub const MAX_LIST: usize = 16;
/// Longest string kept in a description field (characters).
pub const MAX_TEXT: usize = 2_000;
/// Largest embedding kept (floats).
pub const MAX_EMBEDDING: usize = 4_096;

/// The scene index of one media item.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SceneIndex {
    /// [`SCENE_INDEX_VERSION`] when it was made.
    pub version: u32,
    /// How the shots were found: `transnetv2` (the neural shot-boundary model), `signature`
    /// (luma grid and colour histograms, the fallback), `still` (a photo is one shot).
    pub boundaries: String,
    /// The model that described the shots (`qwen3-vl-2b`), empty when none ran.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub describer: String,
    /// The model the embeddings come from (`siglip2-base-224`), empty when none ran.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub embedder: String,
    /// The aesthetic model (`clip-vit-l14-aesthetic`), empty when none ran.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub aesthetic: String,
    /// The detail the shots were described at (`fast` or `full`).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub detail: String,
    /// Length of the media the index covers.
    pub duration: Tick,
    /// Unix seconds when it was made.
    pub created: i64,
    /// Shots in time order, not overlapping.
    pub shots: Vec<Shot>,
}

/// One shot: a continuous take between two cuts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Shot {
    /// Media time of its first frame.
    pub start: Tick,
    /// Media time just after its last frame (exclusive).
    pub end: Tick,
    /// Media times of the frames looked at (described, embedded, measured).
    pub keyframes: Vec<Tick>,
    pub quality: ShotQuality,
    /// How the camera moves, measured from the frames.
    pub camera: CameraMove,
    /// What a vision-language model saw, when one ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<ShotDescription>,
    /// The image embedding of its keyframes (their mean, unit length), when one ran.
    #[serde(skip_serializing_if = "Embedding::is_empty")]
    pub embedding: Embedding,
}

impl Shot {
    pub fn seconds(&self) -> f64 {
        (self.end - self.start).seconds().max(0.0)
    }

    /// The middle of the shot (media time).
    pub fn middle(&self) -> Tick {
        Tick(self.start.0.saturating_add((self.end.0.saturating_sub(self.start.0)) / 2))
    }

    /// Everything said about the shot, as one lower-case text for word search.
    pub fn text(&self) -> String {
        let Some(d) = &self.description else { return String::new() };
        let mut s = String::new();
        let mut put = |t: &str| {
            if !t.is_empty() {
                s.push_str(t);
                s.push(' ');
            }
        };
        put(&d.caption);
        put(&d.detail);
        put(&d.setting);
        put(&d.mood);
        for x in &d.subjects {
            put(&x.name);
        }
        for x in d.actions.iter().chain(&d.text) {
            put(x);
        }
        put(d.shot_type.name());
        s.to_lowercase()
    }
}

/// Cheap measures of how a shot looks. Shares and scores are 0..1.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ShotQuality {
    /// Sharpness score, 0 (blurred) … 1 (crisp), from the variance of the Laplacian.
    pub sharpness: f32,
    /// The variance of the Laplacian of the luma (0..255 scale, on a 512-px frame).
    pub laplacian: f32,
    /// Mean luma, 0..1.
    pub brightness: f32,
    /// Share of pixels crushed to black (luma ≤ 2 %).
    pub dark: f32,
    /// Share of pixels blown to white (luma ≥ 98 %).
    pub blown: f32,
    /// Exposure score, 0 (badly under- or over-exposed) … 1 (well exposed).
    pub exposure: f32,
    /// Camera shake, 0 (steady) … 1 (very shaky): the jitter of the frame-to-frame motion.
    pub shake: f32,
    /// Mean camera motion, frame widths per second.
    pub motion: f32,
    /// LAION aesthetic score (about 1 … 10), when the model ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aesthetic: Option<f32>,
    /// Overall score, 0..1: sharpness, exposure, steadiness and the aesthetic score.
    pub score: f32,
}

/// The camera's motion in a shot, from the frame-to-frame motion of the picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CameraMove {
    #[default]
    Static,
    PanLeft,
    PanRight,
    TiltUp,
    TiltDown,
    /// Moving in more than one direction, or following something.
    Moving,
    /// Shaky hand-held framing without a steady direction.
    Handheld,
    /// A value from a newer build.
    #[serde(other)]
    Unknown,
}

impl CameraMove {
    pub fn name(self) -> &'static str {
        match self {
            CameraMove::Static => "static",
            CameraMove::PanLeft => "pan-left",
            CameraMove::PanRight => "pan-right",
            CameraMove::TiltUp => "tilt-up",
            CameraMove::TiltDown => "tilt-down",
            CameraMove::Moving => "moving",
            CameraMove::Handheld => "handheld",
            CameraMove::Unknown => "unknown",
        }
    }
}

/// How much of the scene a shot frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShotType {
    Wide,
    Medium,
    CloseUp,
    ExtremeCloseUp,
    /// A detail of an object (hands, a dial, a label).
    Insert,
    /// A screen recording or a filmed screen.
    Screen,
    #[default]
    #[serde(other)]
    Unknown,
}

impl ShotType {
    pub const ALL: [ShotType; 6] = [ShotType::Wide, ShotType::Medium, ShotType::CloseUp, ShotType::ExtremeCloseUp, ShotType::Insert, ShotType::Screen];

    pub fn name(self) -> &'static str {
        match self {
            ShotType::Wide => "wide",
            ShotType::Medium => "medium",
            ShotType::CloseUp => "close-up",
            ShotType::ExtremeCloseUp => "extreme-close-up",
            ShotType::Insert => "insert",
            ShotType::Screen => "screen",
            ShotType::Unknown => "unknown",
        }
    }

    /// A model's word for a shot type ("Close up", "ECU", "establishing wide shot"…).
    pub fn from_words(s: &str) -> ShotType {
        let k: String = s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        if k.is_empty() {
            return ShotType::Unknown;
        }
        if k.contains("extreme") || k == "ecu" || k == "xcu" || k.contains("macro") {
            ShotType::ExtremeCloseUp
        } else if k.contains("close") || k == "cu" {
            ShotType::CloseUp
        } else if k.contains("screen") || k.contains("monitor") || k.contains("display") {
            ShotType::Screen
        } else if k.contains("insert") || k.contains("detail") || k.contains("cutaway") {
            ShotType::Insert
        } else if k.contains("medium") || k.contains("mid") || k == "ms" || k.contains("cowboy") || k.contains("waist") {
            ShotType::Medium
        } else if k.contains("wide") || k.contains("long") || k.contains("establishing") || k.contains("full") || k == "ws" || k == "ls" {
            ShotType::Wide
        } else {
            ShotType::Unknown
        }
    }
}

/// What a shot is good for in an edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShotUse {
    /// Grabs attention in the first seconds.
    Hook,
    /// Cutaway footage over narration.
    BRoll,
    /// Shows the subject for the first time, or a change.
    Reveal,
    /// Someone speaking to the camera.
    TalkingHead,
    /// Sets the place.
    Establishing,
    #[serde(other)]
    Unknown,
}

impl ShotUse {
    pub const ALL: [ShotUse; 5] = [ShotUse::Hook, ShotUse::BRoll, ShotUse::Reveal, ShotUse::TalkingHead, ShotUse::Establishing];

    pub fn name(self) -> &'static str {
        match self {
            ShotUse::Hook => "hook",
            ShotUse::BRoll => "b-roll",
            ShotUse::Reveal => "reveal",
            ShotUse::TalkingHead => "talking-head",
            ShotUse::Establishing => "establishing",
            ShotUse::Unknown => "unknown",
        }
    }

    /// A use from a model's or a caller's words ("B-roll", "broll", "talking head"…).
    pub fn from_words(s: &str) -> Option<ShotUse> {
        let k: String = s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        match k.as_str() {
            "hook" | "opener" | "opening" | "intro" => Some(ShotUse::Hook),
            "broll" | "cutaway" | "cutaways" | "filler" => Some(ShotUse::BRoll),
            "reveal" => Some(ShotUse::Reveal),
            "talkinghead" | "interview" | "piecetocamera" | "selfie" => Some(ShotUse::TalkingHead),
            "establishing" | "establishingshot" | "establisher" => Some(ShotUse::Establishing),
            _ => None,
        }
    }
}

/// How lively a shot feels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Energy {
    Low,
    Medium,
    High,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Energy {
    pub fn name(self) -> &'static str {
        match self {
            Energy::Low => "low",
            Energy::Medium => "medium",
            Energy::High => "high",
            Energy::Unknown => "unknown",
        }
    }

    pub fn from_words(s: &str) -> Energy {
        let k = s.trim().to_lowercase();
        if k.contains("high") || k.contains("energetic") || k.contains("intense") || k.contains("fast") {
            Energy::High
        } else if k.contains("low") || k.contains("calm") || k.contains("slow") || k.contains("quiet") {
            Energy::Low
        } else if k.contains("medium") || k.contains("moderate") || k.contains("mid") {
            Energy::Medium
        } else {
            Energy::Unknown
        }
    }

    /// 0 (calm) … 1 (high); unknown is in the middle.
    pub fn level(self) -> f32 {
        match self {
            Energy::Low => 0.0,
            Energy::Medium | Energy::Unknown => 0.5,
            Energy::High => 1.0,
        }
    }
}

/// Someone or something a shot shows.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Subject {
    pub name: String,
    /// Where it is, `[x, y, w, h]` as shares of the picture, when the model said.
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
}

/// What a vision-language model saw in a shot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ShotDescription {
    pub shot_type: ShotType,
    pub subjects: Vec<Subject>,
    /// What happens ("a robot waves", "people walk past").
    pub actions: Vec<String>,
    /// Where it is ("a busy exhibition hall").
    pub setting: String,
    pub mood: String,
    pub energy: Energy,
    /// Text legible in the picture.
    pub text: Vec<String>,
    /// People in the picture.
    pub people: u32,
    /// A face is large enough to recognise.
    pub faces_prominent: bool,
    /// One sentence.
    pub caption: String,
    /// A paragraph (the `full` detail only).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub detail: String,
    /// What it is good for in an edit.
    pub uses: Vec<ShotUse>,
}

/// An embedding vector. Stored as base64 of little-endian IEEE half floats (two bytes a value),
/// so a 768-value embedding costs about 2 KB of project file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Cosine similarity with `other` (0 when either is empty, of another length, or zero).
    pub fn cosine(&self, other: &[f32]) -> f32 {
        if self.0.is_empty() || self.0.len() != other.len() {
            return 0.0;
        }
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in self.0.iter().zip(other) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        if na <= 0.0 || nb <= 0.0 || !dot.is_finite() {
            return 0.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// Rounded to what is stored (half floats), so an index saved and read back is the same.
    pub fn quantized(mut self) -> Self {
        self.0.iter_mut().for_each(|v| *v = f16_to_f32(f32_to_f16(*v)));
        self
    }

    /// Unit length (left as is when zero or not finite).
    pub fn normalized(mut self) -> Self {
        let n: f64 = self.0.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
        if n > 0.0 && n.is_finite() {
            self.0.iter_mut().for_each(|v| *v = (f64::from(*v) / n) as f32);
        }
        self
    }
}

impl Serialize for Embedding {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(self.0.len() * 2);
        for v in &self.0 {
            bytes.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
        s.serialize_str(&crate::gtemplate::base64_encode(&bytes))
    }
}

impl<'de> Deserialize<'de> for Embedding {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        // a damaged embedding reads as none: it is only an index
        let Some(bytes) = crate::gtemplate::base64_decode(&s) else { return Ok(Embedding::default()) };
        let v: Vec<f32> = bytes.as_chunks::<2>().0.iter().take(MAX_EMBEDDING).map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Ok(Embedding::default());
        }
        Ok(Embedding(v))
    }
}

/// IEEE single → half, rounding to nearest even (infinities and NaN kept; out of range: ±inf).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal half
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = u32::from(rest > mid || (rest == mid && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let round = u32::from(rest > 0x1000 || (rest == 0x1000 && half & 1 == 1));
    // a carry into the exponent is right (it rounds up to the next power of two, or to infinity)
    sign | (half + round) as u16
}

/// IEEE half → single.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

fn clip_text(s: &mut String) {
    if s.chars().count() > MAX_TEXT {
        *s = s.chars().take(MAX_TEXT).collect();
    }
}

impl ShotQuality {
    fn sanitize(&mut self) {
        for v in [&mut self.sharpness, &mut self.brightness, &mut self.dark, &mut self.blown, &mut self.exposure, &mut self.shake, &mut self.score] {
            *v = unit(*v);
        }
        for v in [&mut self.laplacian, &mut self.motion] {
            *v = if v.is_finite() { v.clamp(0.0, 1e9) } else { 0.0 };
        }
        self.aesthetic = self.aesthetic.filter(|a| a.is_finite()).map(|a| a.clamp(0.0, 10.0));
    }
}

impl ShotDescription {
    fn sanitize(&mut self) {
        self.subjects.truncate(MAX_LIST);
        for s in &mut self.subjects {
            clip_text(&mut s.name);
            s.bbox = s.bbox.filter(|b| b.iter().all(|v| v.is_finite())).map(|b| b.map(unit));
        }
        for list in [&mut self.actions, &mut self.text] {
            list.truncate(MAX_LIST);
            list.iter_mut().for_each(clip_text);
        }
        self.uses.truncate(MAX_LIST);
        for s in [&mut self.setting, &mut self.mood, &mut self.caption, &mut self.detail] {
            clip_text(s);
        }
        self.people = self.people.min(10_000);
    }
}

impl SceneIndex {
    /// The index with every value from a file brought into range: shots capped, sorted, within
    /// the media and not inverted; keyframes inside their shot; scores in 0..1; strings and lists
    /// capped; embeddings of another length than the first dropped.
    pub fn sanitized(mut self) -> Self {
        self.shots.truncate(MAX_SHOTS);
        let dur = self.duration.max(Tick::ZERO);
        self.duration = dur;
        self.shots.retain(|s| s.end > s.start && s.start >= Tick::ZERO);
        if dur > Tick::ZERO {
            for s in &mut self.shots {
                s.end = s.end.min(dur);
            }
            self.shots.retain(|s| s.end > s.start);
        }
        self.shots.sort_by_key(|s| s.start);
        let width = self.shots.iter().find(|s| !s.embedding.is_empty()).map(|s| s.embedding.0.len());
        for s in &mut self.shots {
            s.keyframes.truncate(MAX_KEYFRAMES);
            let (a, b) = (s.start, s.end);
            s.keyframes.retain(|k| *k >= a && *k < b);
            s.quality.sanitize();
            if let Some(d) = &mut s.description {
                d.sanitize();
            }
            if Some(s.embedding.0.len()) != width {
                s.embedding = Embedding::default();
            }
        }
        for s in [&mut self.boundaries, &mut self.describer, &mut self.embedder, &mut self.aesthetic, &mut self.detail] {
            clip_text(s);
        }
        self
    }

    /// The shot at media time `t`, if any.
    pub fn shot_at(&self, t: Tick) -> Option<&Shot> {
        self.shots.iter().find(|s| t >= s.start && t < s.end)
    }

    /// Whether a model described the shots.
    pub fn described(&self) -> bool {
        self.shots.iter().any(|s| s.description.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_time::TICKS_PER_SECOND;

    fn secs(s: f64) -> Tick {
        Tick::from_seconds_f64(s)
    }

    fn sample() -> SceneIndex {
        SceneIndex {
            version: SCENE_INDEX_VERSION,
            boundaries: "transnetv2".into(),
            describer: "qwen3-vl-2b".into(),
            embedder: "siglip2-base-224".into(),
            aesthetic: String::new(),
            detail: "fast".into(),
            duration: secs(10.0),
            created: 1_760_000_000,
            shots: vec![
                Shot {
                    start: Tick::ZERO,
                    end: secs(4.0),
                    keyframes: vec![secs(2.0)],
                    quality: ShotQuality { sharpness: 0.8, exposure: 0.9, score: 0.7, aesthetic: Some(5.5), ..Default::default() },
                    camera: CameraMove::PanLeft,
                    description: Some(ShotDescription {
                        shot_type: ShotType::Wide,
                        subjects: vec![Subject { name: "white robot".into(), bbox: Some([0.1, 0.2, 0.3, 0.4]) }],
                        actions: vec!["waves".into()],
                        setting: "exhibition hall".into(),
                        mood: "curious".into(),
                        energy: Energy::High,
                        text: vec!["GITEX".into()],
                        people: 3,
                        faces_prominent: true,
                        caption: "A white robot waves at visitors.".into(),
                        detail: String::new(),
                        uses: vec![ShotUse::Hook, ShotUse::Reveal],
                    }),
                    embedding: Embedding(vec![0.5, -0.25, 1.0, 0.0]),
                },
                Shot { start: secs(4.0), end: secs(10.0), keyframes: vec![secs(7.0)], ..Default::default() },
            ],
        }
    }

    #[test]
    fn round_trips_through_json() {
        let idx = sample();
        let text = serde_json::to_string(&idx).unwrap();
        assert!(
            text.contains("\"shotType\":\"wide\"") && text.contains("\"camera\":\"pan-left\"") && text.contains("\"uses\":[\"hook\",\"reveal\"]"),
            "{text}"
        );
        assert!(text.contains("\"box\":[0.1,0.2,0.3,0.4]"), "{text}");
        let back: SceneIndex = serde_json::from_str(&text).unwrap();
        assert_eq!(back, idx, "half floats hold these values exactly");
        // empty parts are left out
        assert!(!text.contains("\"detail\":\"\""));
    }

    #[test]
    fn old_and_foreign_documents_read() {
        // nothing at all
        let idx: SceneIndex = serde_json::from_str("{}").unwrap();
        assert!(idx.shots.is_empty());
        // values from a newer build, and unknown fields, are tolerated
        let v = r#"{"version":9,"shots":[{"start":0,"end":100,"camera":"dolly-zoom","future":1,
            "description":{"shotType":"bird's-eye","uses":["hook","montage"],"energy":"frantic"},"embedding":"%%%"}]}"#;
        let idx: SceneIndex = serde_json::from_str(v).unwrap();
        let s = &idx.shots[0];
        assert_eq!(s.camera, CameraMove::Unknown);
        let d = s.description.as_ref().unwrap();
        assert_eq!(d.shot_type, ShotType::Unknown);
        assert_eq!(d.uses, vec![ShotUse::Hook, ShotUse::Unknown]);
        assert_eq!(d.energy, Energy::Unknown);
        assert!(s.embedding.is_empty(), "a damaged embedding reads as none");
    }

    #[test]
    fn hostile_values_are_brought_into_range() {
        let mut idx = sample();
        idx.shots.push(Shot { start: secs(20.0), end: secs(5.0), ..Default::default() });
        idx.shots.push(Shot { start: secs(-1.0), end: secs(1.0), ..Default::default() });
        idx.shots[0].quality.sharpness = f32::NAN;
        idx.shots[0].quality.exposure = 7.0;
        idx.shots[0].quality.aesthetic = Some(f32::INFINITY);
        idx.shots[0].keyframes = vec![secs(9.0), secs(1.0)];
        idx.shots[1].embedding = Embedding(vec![1.0; 3]);
        idx.shots[1].end = secs(99.0);
        let d = idx.shots[0].description.as_mut().unwrap();
        d.caption = "x".repeat(10 * MAX_TEXT);
        d.actions = vec!["a".into(); 100];
        d.subjects[0].bbox = Some([f32::NAN, 2.0, -1.0, 0.5]);
        let s = idx.sanitized();
        assert_eq!(s.shots.len(), 2, "inverted and negative shots dropped");
        let q = &s.shots[0].quality;
        assert_eq!((q.sharpness, q.exposure, q.aesthetic), (0.0, 1.0, None));
        assert_eq!(s.shots[0].keyframes, vec![secs(1.0)], "keyframes stay inside their shot");
        assert_eq!(s.shots[1].end, secs(10.0), "shots end with the media");
        assert!(s.shots[1].embedding.is_empty(), "an embedding of another width is dropped");
        let d = s.shots[0].description.as_ref().unwrap();
        assert_eq!(d.caption.chars().count(), MAX_TEXT);
        assert_eq!(d.actions.len(), MAX_LIST);
        assert_eq!(d.subjects[0].bbox, None);
    }

    #[test]
    fn half_floats_convert_both_ways() {
        for v in [0.0f32, -0.0, 1.0, -2.5, 0.333_251_95, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{v}");
        }
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(f32_to_f16(1e-9)), 0.0);
        // rounding to nearest
        let x = 0.1f32;
        assert!((f16_to_f32(f32_to_f16(x)) - x).abs() < 1e-4);
        // every half survives the trip
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x}");
        }
    }

    #[test]
    fn words_map_to_vocabulary() {
        assert_eq!(ShotType::from_words("Extreme close-up"), ShotType::ExtremeCloseUp);
        assert_eq!(ShotType::from_words("close up"), ShotType::CloseUp);
        assert_eq!(ShotType::from_words("Medium shot"), ShotType::Medium);
        assert_eq!(ShotType::from_words("wide establishing shot"), ShotType::Wide);
        assert_eq!(ShotType::from_words("screen recording"), ShotType::Screen);
        assert_eq!(ShotType::from_words("insert"), ShotType::Insert);
        assert_eq!(ShotType::from_words("??"), ShotType::Unknown);
        assert_eq!(ShotUse::from_words("B-roll"), Some(ShotUse::BRoll));
        assert_eq!(ShotUse::from_words("talking head"), Some(ShotUse::TalkingHead));
        assert_eq!(ShotUse::from_words("cake"), None);
        assert_eq!(Energy::from_words("High energy"), Energy::High);
        assert_eq!(Energy::from_words("calm"), Energy::Low);
    }

    #[test]
    fn shot_helpers() {
        let idx = sample();
        assert_eq!(idx.shot_at(secs(5.0)).map(|s| s.start), Some(secs(4.0)));
        assert!(idx.shot_at(secs(10.0)).is_none());
        assert!(idx.described());
        let s = &idx.shots[0];
        assert_eq!(s.middle(), Tick(2 * TICKS_PER_SECOND));
        assert!(s.text().contains("white robot") && s.text().contains("gitex") && s.text().contains("wide"));
        assert_eq!(idx.shots[1].text(), "");
        let e = Embedding(vec![3.0, 4.0]).normalized();
        assert_eq!(e.0, vec![0.6, 0.8]);
        assert!((e.cosine(&[0.6, 0.8]) - 1.0).abs() < 1e-6);
        assert_eq!(e.cosine(&[1.0]), 0.0);
        assert_eq!(Embedding::default().cosine(&[]), 0.0);
        assert_eq!(Embedding(vec![0.0, 0.0]).normalized().0, vec![0.0, 0.0]);
    }
}
