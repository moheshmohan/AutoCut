//! Caption tracks.
//!
//! Premiere keeps captions in their own track area above the video tracks. A [`CaptionTrack`] has a
//! format (Subtitle, CEA-608, CEA-708, Teletext), a track style (font, size, colour, background box,
//! position) and a list of [`Caption`] blocks with exact in/out times in sequence ticks. Captions on
//! one track never overlap (like track items); [`CaptionTrack::check`] verifies it.
//!
//! Caption text is kept as written (lines separated by `\n`, inline `<i>`/`<b>`/`<u>` tags and
//! WebVTT cue settings preserved) so files round-trip; renderers strip markup for display.

use filmcraft_time::{Tick, TimeRange};
use serde::{Deserialize, Serialize};

use crate::{ClipId, TrackId};

/// Caption stream format of a track (Premiere's "Caption Format" choices).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaptionFormat {
    /// Open/sidecar subtitles (SRT, WebVTT).
    #[default]
    Subtitle,
    /// CEA-608 (line 21) closed captions.
    Cea608,
    /// CEA-708 (DTVCC) closed captions.
    Cea708,
    /// EBU Teletext subtitles.
    Teletext,
}

impl CaptionFormat {
    pub const ALL: [CaptionFormat; 4] = [CaptionFormat::Subtitle, CaptionFormat::Cea608, CaptionFormat::Cea708, CaptionFormat::Teletext];
    pub fn label(self) -> &'static str {
        match self {
            CaptionFormat::Subtitle => "Subtitle",
            CaptionFormat::Cea608 => "CEA-608",
            CaptionFormat::Cea708 => "CEA-708",
            CaptionFormat::Teletext => "Teletext",
        }
    }
    pub fn from_name(s: &str) -> Option<CaptionFormat> {
        let n = s.to_ascii_lowercase().replace(['-', ' ', '_'], "");
        Self::ALL.iter().copied().find(|f| f.label().to_ascii_lowercase().replace(['-', ' '], "") == n)
    }
    /// Characters per line limit used when wrapping (608 is fixed at 32 columns).
    pub fn max_columns(self) -> Option<usize> {
        match self {
            CaptionFormat::Cea608 => Some(32),
            CaptionFormat::Teletext => Some(40),
            CaptionFormat::Cea708 => Some(42),
            CaptionFormat::Subtitle => None,
        }
    }
}

/// Horizontal text alignment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaptionAlign {
    Left,
    #[default]
    Center,
    Right,
}

/// Vertical placement of the caption block in the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaptionAnchor {
    Top,
    Middle,
    #[default]
    Bottom,
}

/// Track-level caption style (Premiere's caption track style / Essential Graphics text settings).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptionStyle {
    /// Font family name (bundled, craft-fonts or system); unknown or empty names fall back to Inter.
    pub font: String,
    /// Font size in pixels for a 1080-line frame (scaled with the frame height).
    pub size: f32,
    /// Text colour, sRGB + alpha.
    pub color: [u8; 4],
    /// Draw a background box behind each line.
    pub background: bool,
    /// Background box colour, sRGB + alpha.
    pub background_color: [u8; 4],
    pub align: CaptionAlign,
    pub anchor: CaptionAnchor,
    /// Distance from the anchored frame edge, as a fraction of the frame height.
    pub margin: f32,
    /// Line height as a multiple of the font size.
    pub line_spacing: f32,
    /// Text outline width in pixels at 1080 lines (0 = none).
    pub outline: f32,
    pub outline_color: [u8; 4],
    /// Horizontal shift of the captions, a share of the frame width (positive = right): centres
    /// them on a platform's safe span rather than on the frame (see `safe_zones`).
    #[serde(skip_serializing_if = "is_zero_f32")]
    pub x_offset: f32,
    /// Word-by-word animated captions ([`AnimatedCaptions`]); None draws each caption whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub animated: Option<AnimatedCaptions>,
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            font: "Inter".into(),
            size: 54.0,
            color: [255, 255, 255, 255],
            background: true,
            background_color: [0, 0, 0, 191],
            align: CaptionAlign::Center,
            anchor: CaptionAnchor::Bottom,
            margin: 0.08,
            line_spacing: 1.25,
            outline: 0.0,
            outline_color: [0, 0, 0, 255],
            x_offset: 0.0,
            animated: None,
        }
    }
}

/// The look of animated captions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CaptionPreset {
    /// Uppercase heavy sans with a black stroke, 2–4 words at a time, the spoken word in the
    /// active colour and popping to 1.1×.
    #[default]
    Bold,
    /// Sentence case with a soft shadow; the spoken word in the active colour.
    Clean,
    /// A neon glow around the spoken word.
    Glow,
    /// The spoken word on a rounded box of the active colour.
    Box,
    /// The fill sweeps through the words in the active colour as they are spoken.
    Karaoke,
    /// Classic subtitles: the whole caption on up to two lines with a soft shadow.
    TwoLine,
}

impl CaptionPreset {
    pub const ALL: [CaptionPreset; 6] =
        [CaptionPreset::Bold, CaptionPreset::Clean, CaptionPreset::Glow, CaptionPreset::Box, CaptionPreset::Karaoke, CaptionPreset::TwoLine];
    /// The name parameters take.
    pub fn name(self) -> &'static str {
        match self {
            CaptionPreset::Bold => "bold",
            CaptionPreset::Clean => "clean",
            CaptionPreset::Glow => "glow",
            CaptionPreset::Box => "box",
            CaptionPreset::Karaoke => "karaoke",
            CaptionPreset::TwoLine => "twoLine",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            CaptionPreset::Bold => "Bold",
            CaptionPreset::Clean => "Clean",
            CaptionPreset::Glow => "Glow",
            CaptionPreset::Box => "Box",
            CaptionPreset::Karaoke => "Karaoke",
            CaptionPreset::TwoLine => "Two-Line Subtitle",
        }
    }
    /// A preset by name or label (case, spaces, `-` and `_` do not matter; `subtitle` and
    /// `classic` are the two-line subtitle).
    pub fn from_name(s: &str) -> Option<CaptionPreset> {
        let k: String = s.chars().filter(|c| !matches!(c, ' ' | '-' | '_')).flat_map(char::to_lowercase).collect();
        if matches!(k.as_str(), "subtitle" | "subtitles" | "classic" | "twolines") {
            return Some(CaptionPreset::TwoLine);
        }
        Self::ALL.into_iter().find(|p| {
            let l: String = p.label().chars().filter(|c| !matches!(c, ' ' | '-')).flat_map(char::to_lowercase).collect();
            p.name().to_ascii_lowercase() == k || l == k
        })
    }
    /// Words on screen at a time (the two-line subtitle shows the whole caption: 0).
    pub fn words_per_page(self) -> u32 {
        match self {
            CaptionPreset::Bold | CaptionPreset::Glow | CaptionPreset::Box => 3,
            CaptionPreset::Clean => 4,
            CaptionPreset::Karaoke => 6,
            CaptionPreset::TwoLine => 0,
        }
    }
    /// Whether the text is set in capitals.
    pub fn uppercase(self) -> bool {
        matches!(self, CaptionPreset::Bold | CaptionPreset::Glow)
    }
    /// The spoken word's colour.
    pub fn active_color(self) -> [u8; 4] {
        match self {
            CaptionPreset::Bold => [255, 214, 10, 255],
            CaptionPreset::Clean => [125, 211, 252, 255],
            CaptionPreset::Glow => [80, 250, 255, 255],
            CaptionPreset::Box => [139, 92, 246, 255],
            CaptionPreset::Karaoke => [250, 204, 21, 255],
            CaptionPreset::TwoLine => [255, 255, 255, 255],
        }
    }
    /// How much the spoken word grows (1 = not at all).
    pub fn pop(self) -> f32 {
        match self {
            CaptionPreset::Bold => 1.1,
            CaptionPreset::Box | CaptionPreset::Glow => 1.05,
            _ => 1.0,
        }
    }
    /// The track style the preset starts from: size (1080-line pixels), outline, background box,
    /// placement. `captions.setAnimatedStyle` applies it unless told otherwise.
    pub fn base_style(self) -> CaptionStyle {
        let d = CaptionStyle::default();
        let (size, outline, line_spacing) = match self {
            CaptionPreset::Bold => (108.0, 9.0, 1.08),
            CaptionPreset::Glow => (100.0, 0.0, 1.1),
            CaptionPreset::Box => (96.0, 0.0, 1.25),
            CaptionPreset::Karaoke => (88.0, 6.0, 1.15),
            CaptionPreset::Clean => (80.0, 0.0, 1.2),
            CaptionPreset::TwoLine => (60.0, 2.5, 1.25),
        };
        CaptionStyle {
            size,
            outline,
            line_spacing,
            background: false,
            outline_color: [0, 0, 0, 255],
            anchor: CaptionAnchor::Bottom,
            margin: if self == CaptionPreset::TwoLine { 0.08 } else { 0.12 },
            ..d
        }
    }
}

/// A word with an emoji shown after it (animated captions).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WordEmoji {
    pub word: String,
    pub emoji: String,
}

/// Word-by-word animated captions: a few words on screen at a time (a page), the word being
/// spoken highlighted (see [`CaptionPreset`]). Drawn by `filmcraft_captions::animated` in the
/// preview and the export alike. Timing comes from the captions' [`Caption::words`] (or, without
/// them, is spread over each caption by the words' lengths).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnimatedCaptions {
    pub preset: CaptionPreset,
    /// Words on a page, 1–8 (0 = the preset's; the two-line subtitle shows whole captions).
    pub words_per_page: u32,
    /// The spoken word's colour (sRGB + alpha).
    pub active_color: [u8; 4],
    /// The colour of keywords.
    pub keyword_color: [u8; 4],
    /// Words drawn in the keyword colour (compared without case and punctuation).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Emoji shown after words.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub emoji: Vec<WordEmoji>,
    /// Set in capitals (None = the preset's).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uppercase: Option<bool>,
    /// The platform whose safe zone the captions keep to: they are centred on the middle of its
    /// safe span and as wide as it at most, and the anchor margin counts from its safe area. None
    /// = the whole frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<crate::safe_zones::Platform>,
    /// How much the spoken word grows (1 = not at all; 0 = the preset's).
    pub pop: f32,
}

impl Default for AnimatedCaptions {
    fn default() -> Self {
        AnimatedCaptions::preset(CaptionPreset::Bold)
    }
}

impl AnimatedCaptions {
    pub fn preset(p: CaptionPreset) -> AnimatedCaptions {
        AnimatedCaptions {
            preset: p,
            words_per_page: p.words_per_page(),
            active_color: p.active_color(),
            keyword_color: [74, 222, 128, 255],
            keywords: Vec::new(),
            emoji: Vec::new(),
            uppercase: None,
            platform: None,
            pop: p.pop(),
        }
    }
    /// Words per page, 1–8 (0 = a whole caption a page).
    pub fn page_words(&self) -> usize {
        if self.preset == CaptionPreset::TwoLine && self.words_per_page == 0 {
            return 0;
        }
        let n = if self.words_per_page == 0 { self.preset.words_per_page() } else { self.words_per_page };
        n.clamp(1, 8) as usize
    }
    pub fn is_uppercase(&self) -> bool {
        self.uppercase.unwrap_or(self.preset.uppercase())
    }
    /// The spoken word's scale, 1–1.5.
    pub fn pop_scale(&self) -> f32 {
        let p = if self.pop == 0.0 { self.preset.pop() } else { self.pop };
        if p.is_finite() { p.clamp(1.0, 1.5) } else { 1.0 }
    }
    /// Whether `word` is a keyword.
    pub fn is_keyword(&self, word: &str) -> bool {
        let w = crate::transcript::normalize_word(word);
        !w.is_empty() && self.keywords.iter().any(|k| crate::transcript::normalize_word(k) == w)
    }
    /// The emoji after `word`, if any.
    pub fn emoji_for(&self, word: &str) -> Option<&str> {
        let w = crate::transcript::normalize_word(word);
        if w.is_empty() {
            return None;
        }
        self.emoji.iter().find(|e| crate::transcript::normalize_word(&e.word) == w).map(|e| e.emoji.as_str()).filter(|e| !e.trim().is_empty())
    }
}

/// A word of a caption with its timing, relative to the caption's start (so moving the caption
/// moves its words).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CaptionWord {
    pub text: String,
    /// Offset of the word's start from the caption's start.
    pub start: Tick,
    /// Offset of its end.
    pub end: Tick,
}

/// A word of a caption as shown, with its timeline times.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start: Tick,
    pub end: Tick,
    /// The written line of the caption it is on (0 = the first).
    pub line: usize,
}

/// Most words of one caption looked at.
pub const MAX_CAPTION_WORDS: usize = 2000;

/// Saturating tick arithmetic for word times (a project file's are hostile).
fn add(a: Tick, b: Tick) -> Tick {
    Tick(a.0.saturating_add(b.0))
}
fn sub(a: Tick, b: Tick) -> Tick {
    Tick(a.0.saturating_sub(b.0))
}

/// One caption block on a caption track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Caption {
    pub id: ClipId,
    /// Timeline start (sequence ticks).
    pub start: Tick,
    pub duration: Tick,
    /// Caption text; lines separated by `\n`.
    pub text: String,
    /// Speaker name (WebVTT `<v Name>` voice span).
    #[serde(default)]
    pub speaker: Option<String>,
    /// WebVTT cue identifier, kept for round trips.
    #[serde(default)]
    pub cue_id: Option<String>,
    /// WebVTT cue settings (`line:90% align:start`…), kept verbatim.
    #[serde(default)]
    pub settings: String,
    /// When each word is spoken (from a transcript, a narration or ASS karaoke), relative to
    /// `start`; empty when unknown.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<CaptionWord>,
}

impl Caption {
    pub fn end(&self) -> Tick {
        self.start + self.duration
    }
    pub fn range(&self) -> TimeRange {
        TimeRange::new(self.start, self.duration)
    }
    /// The caption's text with markup removed and entities decoded (what is drawn).
    pub fn plain_lines(&self) -> Vec<String> {
        plain_text(&self.text).lines().map(str::to_string).collect()
    }

    /// The words as shown (the plain text split at spaces, line by line) with their timeline
    /// times: from [`Caption::words`] when they match the text word for word; otherwise spread
    /// over the span the timed words cover (or the whole caption) in proportion to each word's
    /// length. Times stay inside the caption and never go backwards. At most
    /// [`MAX_CAPTION_WORDS`].
    pub fn timed_words(&self) -> Vec<TimedWord> {
        let mut shown: Vec<(String, usize)> = Vec::new();
        for (li, line) in self.plain_lines().iter().enumerate() {
            for w in line.split_whitespace() {
                if shown.len() >= MAX_CAPTION_WORDS {
                    break;
                }
                shown.push((w.to_string(), li));
            }
        }
        if shown.is_empty() {
            return Vec::new();
        }
        let (a, b) = (self.start, add(self.start, self.duration).max(self.start));
        let clamp = |t: Tick| t.max(a).min(b);
        let timed: Vec<(Tick, Tick)> = self.words.iter().map(|w| (clamp(add(self.start, w.start)), clamp(add(self.start, w.end.max(w.start))))).collect();
        let mut out = Vec::with_capacity(shown.len());
        if timed.len() == shown.len() {
            let mut last = a;
            for ((text, line), (s, e)) in shown.into_iter().zip(timed) {
                let s = s.max(last);
                let e = e.max(s);
                last = s;
                out.push(TimedWord { text, start: s, end: e, line });
            }
            return out;
        }
        // spread over the timed span (or the caption) by length: a word takes its characters
        // plus one (the pause after it)
        let (sa, sb) = match (timed.first(), timed.last()) {
            (Some(f), Some(l)) if l.1 > f.0 => (f.0, l.1),
            _ => (a, b),
        };
        let weights: Vec<u64> = shown.iter().map(|(w, _)| w.chars().count() as u64 + 1).collect();
        let total: u64 = weights.iter().sum::<u64>().max(1);
        let span = i128::from(sb.0.saturating_sub(sa.0).max(0));
        let mut acc = 0u64;
        for ((text, line), wgt) in shown.into_iter().zip(weights) {
            let s = add(sa, Tick((span * i128::from(acc) / i128::from(total)) as i64));
            acc += wgt;
            let e = add(sa, Tick((span * i128::from(acc) / i128::from(total)) as i64));
            out.push(TimedWord { text, start: s, end: e, line });
        }
        out
    }

    /// Set the words' timeline times (each `(text, start, end)`), stored relative to the
    /// caption's start and kept inside it.
    pub fn set_word_times(&mut self, words: &[(String, Tick, Tick)]) {
        let d = self.duration.max(Tick::ZERO);
        self.words = words
            .iter()
            .take(MAX_CAPTION_WORDS)
            .map(|(text, s, e)| {
                let rs = sub(*s, self.start).max(Tick::ZERO).min(d);
                let re = sub(*e, self.start).max(rs).min(d);
                CaptionWord { text: text.clone(), start: rs, end: re }
            })
            .collect();
    }

    /// Keep the words on their timeline times after the caption's start moved from `old_start`
    /// without its content moving (a trimmed head, the right half of a split): offsets are
    /// shifted and words now outside the caption dropped.
    pub fn rebase_words(&mut self, old_start: Tick) {
        if self.words.is_empty() {
            return;
        }
        let shift = sub(old_start, self.start);
        let d = self.duration;
        self.words.retain_mut(|w| {
            w.start = add(w.start, shift);
            w.end = add(w.end, shift);
            w.end > Tick::ZERO && w.start < d
        });
        for w in &mut self.words {
            w.start = w.start.max(Tick::ZERO);
            w.end = w.end.min(d).max(w.start);
        }
    }
}

/// A caption track (shown in the caption area above the video tracks).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptionTrack {
    pub id: TrackId,
    pub name: String,
    pub format: CaptionFormat,
    /// BCP-47 language tag (`en`, `fr-CA`…).
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub style: CaptionStyle,
    /// Track output (eye): shown in the Program monitor and burned in on export.
    pub enabled: bool,
    pub locked: bool,
    /// Follows inserts / extracts on other tracks.
    #[serde(default = "yes")]
    pub sync_lock: bool,
    /// Captions sorted by start, non-overlapping.
    pub captions: Vec<Caption>,
    /// WebVTT header blocks (STYLE / REGION / NOTE) kept for round trips.
    #[serde(default)]
    pub vtt_blocks: Vec<String>,
}

fn yes() -> bool {
    true
}

impl CaptionTrack {
    pub fn new(id: TrackId, name: String, format: CaptionFormat) -> Self {
        Self {
            id,
            name,
            format,
            language: "en".into(),
            style: CaptionStyle::default(),
            enabled: true,
            locked: false,
            sync_lock: true,
            captions: Vec::new(),
            vtt_blocks: Vec::new(),
        }
    }
    pub fn end(&self) -> Tick {
        self.captions.iter().map(Caption::end).max().unwrap_or(Tick::ZERO)
    }
    /// Caption covering `t`.
    pub fn caption_at(&self, t: Tick) -> Option<&Caption> {
        let idx = self.captions.partition_point(|c| c.start <= t);
        idx.checked_sub(1).map(|i| &self.captions[i]).filter(|c| t < c.end())
    }
    pub fn caption(&self, id: ClipId) -> Option<&Caption> {
        self.captions.iter().find(|c| c.id == id)
    }
    pub fn caption_mut(&mut self, id: ClipId) -> Option<&mut Caption> {
        self.captions.iter_mut().find(|c| c.id == id)
    }
    pub fn sort(&mut self) {
        self.captions.sort_by_key(|c| c.start);
    }
    /// Invariant check: captions sorted, positive duration, no overlaps.
    pub fn check(&self) -> Result<(), String> {
        self.check_bounds()?;
        for c in &self.captions {
            if c.duration.0 <= 0 {
                return Err(format!("{}: caption {:?} has non-positive duration", self.name, c.id));
            }
        }
        for w in self.captions.windows(2) {
            if w[0].end() > w[1].start {
                return Err(format!("{}: captions {:?} and {:?} overlap", self.name, w[0].id, w[1].id));
            }
        }
        Ok(())
    }
    /// Only the time bounds that keep caption arithmetic from overflowing (see `Track::check_bounds`).
    pub fn check_bounds(&self) -> Result<(), String> {
        for c in &self.captions {
            if !crate::bounded_time_range(c.start, c.duration) {
                return Err(format!("{}: caption {:?} is outside supported time bounds", self.name, c.id));
            }
        }
        Ok(())
    }
}

/// The named entities caption text may carry, as a WebVTT import keeps them.
const ENTITIES: [(&str, char); 8] =
    [("amp;", '&'), ("lt;", '<'), ("gt;", '>'), ("nbsp;", '\u{a0}'), ("quot;", '"'), ("apos;", '\''), ("lrm;", '\u{200e}'), ("rlm;", '\u{200f}')];

/// The entity at the start of `s`, as the character it names and its length in bytes. Caption
/// text keeps a literal `<` or `&` as `&lt;` / `&amp;` (as a WebVTT import keeps its entities) so
/// that its style tags stay tags; [`plain_text`] and the caption writers decode them here.
pub fn entity_at(s: &str) -> Option<(char, usize)> {
    let rest = s.strip_prefix('&')?;
    ENTITIES.iter().find(|(name, _)| rest.starts_with(name)).map(|(name, ch)| (*ch, name.len() + 1))
}

/// `s` with its entities decoded and its tags kept: for a format that has tags but no entities.
pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(ch) = rest.chars().next() {
        if let Some((decoded, len)) = entity_at(rest) {
            out.push(decoded);
            rest = rest.get(len..).unwrap_or("");
        } else {
            out.push(ch);
            rest = rest.get(ch.len_utf8()..).unwrap_or("");
        }
    }
    out
}

/// Strip `<…>` tags and decode the common HTML/WebVTT entities.
pub fn plain_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '<' => {
                // A tag runs to the next '>'.  Scan forward once; preserving an unterminated
                // tag as literal text must not rescan every later opening bracket.
                let mut literal = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '>' {
                        closed = true;
                        break;
                    }
                    literal.push(next);
                }
                if !closed {
                    out.push('<');
                    out.push_str(&literal);
                }
            }
            '{' if chars.peek() == Some(&'\\') => {
                // SSA-style override blocks some SRT files carry: {\an8}
                let mut literal = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    literal.push(next);
                }
                if !closed {
                    out.push('{');
                    out.push_str(&literal);
                }
            }
            '&' => {
                let rest: String = std::iter::once('&').chain(chars.clone().take(8)).collect();
                if let Some((ch, len)) = entity_at(&rest) {
                    for _ in 1..len {
                        chars.next();
                    }
                    out.push(ch);
                } else {
                    out.push('&');
                }
            }
            '\r' => {}
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain() {
        assert_eq!(plain_text("<i>Hello</i> &amp; <b>bye</b>"), "Hello & bye");
        assert_eq!(plain_text("{\\an8}Top"), "Top");
        assert_eq!(plain_text("a < b"), "a < b");
        assert_eq!(plain_text("<v Bob>Hi</v>"), "Hi");
    }

    #[test]
    fn entities() {
        assert_eq!(entity_at("&lt;b"), Some(('<', 4)));
        assert_eq!(entity_at("&eacute;"), None);
        assert_eq!(entity_at("&"), None);
        assert_eq!(decode_entities("1 &lt; 2 &amp; <i>x</i> &#60; &"), "1 < 2 & <i>x</i> &#60; &");
        assert_eq!(plain_text("1 &lt; 2 &amp;lt; &nbsp;&"), "1 < 2 &lt; \u{a0}&");
    }

    #[test]
    fn malformed_caption_markup_is_preserved_without_repeated_suffix_scans() {
        let angle = "<".repeat(10_000);
        let ssa = format!("{{\\{}", "{".repeat(10_000));
        assert_eq!(plain_text(&angle), angle);
        assert_eq!(plain_text(&ssa), ssa);
    }

    #[test]
    fn caption_at() {
        let mut t = CaptionTrack::new(TrackId(1), "Subtitle".into(), CaptionFormat::Subtitle);
        for (i, s) in [0i64, 100, 300].iter().enumerate() {
            t.captions.push(Caption {
                id: ClipId(i as u64),
                start: Tick(*s),
                duration: Tick(50),
                text: format!("c{i}"),
                speaker: None,
                cue_id: None,
                settings: String::new(),
                words: Vec::new(),
            });
        }
        assert_eq!(t.caption_at(Tick(120)).unwrap().text, "c1");
        assert!(t.caption_at(Tick(160)).is_none());
        assert!(t.check().is_ok());
        assert_eq!(CaptionFormat::from_name("cea608"), Some(CaptionFormat::Cea608));
        assert_eq!(CaptionFormat::from_name("CEA-708"), Some(CaptionFormat::Cea708));
    }

    #[test]
    fn corrupt_caption_times_do_not_overflow_the_overlap_check() {
        let mut track = CaptionTrack::new(TrackId(1), "Subtitles".into(), CaptionFormat::Subtitle);
        for (id, start) in [(1, i64::MAX), (2, 10)] {
            track.captions.push(Caption {
                id: ClipId(id),
                start: Tick(start),
                duration: Tick(50),
                text: String::new(),
                speaker: None,
                cue_id: None,
                settings: String::new(),
                words: Vec::new(),
            });
        }
        let result = std::panic::catch_unwind(|| track.check());
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    fn cap(start: i64, dur: i64, text: &str) -> Caption {
        Caption {
            id: ClipId(1),
            start: Tick(start),
            duration: Tick(dur),
            text: text.into(),
            speaker: None,
            cue_id: None,
            settings: String::new(),
            words: Vec::new(),
        }
    }

    #[test]
    fn word_timing_round_trips_and_follows_moves() {
        let mut c = cap(1000, 900, "Hello big\nworld");
        c.set_word_times(&[("Hello".into(), Tick(1000), Tick(1200)), ("big".into(), Tick(1300), Tick(1500)), ("world".into(), Tick(1600), Tick(1900))]);
        assert_eq!(c.words[1], CaptionWord { text: "big".into(), start: Tick(300), end: Tick(500) });
        // serde: written only when present, read back exactly; old projects have none
        let j = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<Caption>(&j).unwrap(), c);
        let old: Caption = serde_json::from_str(r#"{"id":1,"start":0,"duration":10,"text":"x","settings":""}"#).unwrap();
        assert!(old.words.is_empty());
        assert!(!serde_json::to_string(&old).unwrap().contains("words"));
        // timeline times, with the line each word is on
        let w = c.timed_words();
        assert_eq!(w.iter().map(|w| (w.start.0, w.end.0, w.line)).collect::<Vec<_>>(), vec![(1000, 1200, 0), (1300, 1500, 0), (1600, 1900, 1)]);
        // moving the caption moves its words
        c.start = Tick(5000);
        assert_eq!(c.timed_words()[2].start, Tick(5600));
        // trimming the head keeps the words where they were on the timeline
        c.start = Tick(5400);
        c.duration = Tick(500);
        c.rebase_words(Tick(5000));
        assert_eq!(c.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["big", "world"]);
        assert_eq!(c.words[0].start, Tick(0), "a word cut by the trim starts with the caption");
        assert_eq!(c.words[1].start, Tick(200));
    }

    #[test]
    fn words_without_timing_spread_by_length() {
        let c = cap(0, 1100, "a bbbb cccc");
        let w = c.timed_words();
        // weights 2, 5, 5 of 12 over 1100 ticks
        assert_eq!(w.iter().map(|w| (w.start.0, w.end.0)).collect::<Vec<_>>(), vec![(0, 183), (183, 641), (641, 1100)]);
        // timing that no longer matches the text (edited): spread over the timed span
        let mut e = cap(0, 1000, "one two three");
        e.set_word_times(&[("one".into(), Tick(100), Tick(300)), ("two".into(), Tick(400), Tick(800))]);
        let w = e.timed_words();
        assert_eq!(w.len(), 3);
        assert_eq!((w[0].start, w[2].end), (Tick(100), Tick(800)));
        assert!(w.windows(2).all(|p| p[0].end <= p[1].start + Tick(1)));
        // Malayalam words split at spaces like any other
        assert_eq!(cap(0, 100, "ഇത് ഒരു പരീക്ഷണം").timed_words().len(), 3);
        assert!(cap(0, 100, "  \n ").timed_words().is_empty());
    }

    #[test]
    fn hostile_word_times_stay_inside_the_caption() {
        let mut c = cap(i64::MAX - 10, 100, "a b");
        c.words = vec![
            CaptionWord { text: "a".into(), start: Tick(i64::MIN), end: Tick(i64::MAX) },
            CaptionWord { text: "b".into(), start: Tick(i64::MAX), end: Tick(-5) },
        ];
        let w = std::panic::catch_unwind(|| c.timed_words()).expect("no panic");
        assert!(w.iter().all(|w| w.start >= c.start && w.end >= w.start));
        let mut d = c.clone();
        assert!(std::panic::catch_unwind(move || d.rebase_words(Tick(i64::MIN))).is_ok());
        let mut e = cap(0, 100, "x");
        e.set_word_times(&[("x".into(), Tick(i64::MIN), Tick(i64::MAX))]);
        assert_eq!((e.words[0].start, e.words[0].end), (Tick(0), Tick(100)));
    }

    #[test]
    fn animated_caption_presets() {
        for p in CaptionPreset::ALL {
            assert_eq!(CaptionPreset::from_name(p.name()), Some(p));
            assert_eq!(CaptionPreset::from_name(p.label()), Some(p));
            let a = AnimatedCaptions::preset(p);
            let j = serde_json::to_value(&a).unwrap();
            assert_eq!(serde_json::from_value::<AnimatedCaptions>(j).unwrap(), a);
        }
        assert_eq!(CaptionPreset::from_name("two-line"), Some(CaptionPreset::TwoLine));
        assert_eq!(CaptionPreset::from_name("subtitle"), Some(CaptionPreset::TwoLine));
        assert_eq!(CaptionPreset::from_name("neon"), None);
        let bold = AnimatedCaptions::preset(CaptionPreset::Bold);
        assert!(bold.is_uppercase() && (2..=4).contains(&bold.page_words()) && (bold.pop_scale() - 1.1).abs() < 1e-6);
        assert_eq!(AnimatedCaptions::preset(CaptionPreset::TwoLine).page_words(), 0);
        let mut k = bold.clone();
        k.keywords = vec!["Money".into()];
        k.emoji = vec![WordEmoji { word: "fire".into(), emoji: "🔥".into() }];
        assert!(k.is_keyword("money!") && !k.is_keyword("honey"));
        assert_eq!(k.emoji_for("FIRE,"), Some("🔥"));
        // a style without animation reads and writes as before
        let s = CaptionStyle::default();
        let j = serde_json::to_string(&s).unwrap();
        assert!(!j.contains("animated") && !j.contains("xOffset") && !j.contains("x_offset"));
        let back: CaptionStyle = serde_json::from_str(&j).unwrap();
        assert_eq!(back, s);
        let with = CaptionStyle { animated: Some(k.clone()), x_offset: -0.065, ..Default::default() };
        assert_eq!(serde_json::from_str::<CaptionStyle>(&serde_json::to_string(&with).unwrap()).unwrap(), with);
        // hostile numbers
        let wild = AnimatedCaptions { words_per_page: u32::MAX, pop: f32::NAN, ..bold };
        assert_eq!(wild.page_words(), 8);
        assert_eq!(wild.pop_scale(), 1.0);
    }
}
