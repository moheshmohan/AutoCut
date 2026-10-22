//! The captions AutoCut puts on shorts and trailers.
//!
//! By default they are **Bold animated captions** (`captions.setAnimatedStyle`): uppercase heavy
//! text with a black stroke, 2–4 words on screen at a time, the spoken word in yellow popping to
//! 1.1×, kept inside the target platform's safe zone and centred on the middle of its safe span
//! (on YouTube Shorts its right margin is wider than its left, so the captions sit a little left
//! of the frame's middle and use the whole safe width). `captionStyle` picks another animated
//! preset (`clean`, `glow`, `box`, `karaoke`, `twoLine`) or `plain`: the outlined whole captions
//! of before, now centred on the safe span too (an `xOffset`), so they may be as wide as it.
//!
//! With `captionKeywords` the local model ([`keywords`]) picks a few words to colour and a few
//! emoji to follow words; what it answers is checked against the caption text (a word it made up,
//! or an "emoji" that is a letter, is dropped). Off by default.

use std::collections::BTreeMap;

use filmcraft_engine::safe_zones::Platform;
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

/// How AutoCut captions look.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Look {
    /// An animated preset by name (`bold`, …).
    Animated(String),
    /// Outlined whole captions, centred on the platform's safe span.
    Plain,
}

impl Look {
    /// From a job's `captionStyle` (the engine checked it): `plain`, or a preset; Bold when absent.
    pub fn of(p: &Value) -> Look {
        match p["captionStyle"].as_str().map(str::trim) {
            Some("plain") => Look::Plain,
            Some(s) if !s.is_empty() => Look::Animated(s.to_string()),
            _ => Look::Animated("bold".into()),
        }
    }
}

/// Room between plain captions and the platform's bottom zone, a share of the frame's height.
pub const CAPTION_GAP: f64 = 0.02;

/// The x offset (a share of the frame width) that centres captions on `platform`'s safe span at
/// the rows just above its bottom zone (where captions go).
pub fn safe_offset(platform: Platform) -> f64 {
    let bottom = 1.0 - platform.margins().bottom - CAPTION_GAP;
    let (l, r) = platform.safe_span(bottom - 0.1, bottom);
    ((l + r) / 2.0 - 0.5).clamp(-0.5, 0.5)
}

/// Style caption track `track` for `platform` with `look`; `extra` are the keywords and emoji to
/// colour and add (animated looks only).
pub async fn style_track(ex: &dyn Executor, track: u64, platform: Platform, look: &Look, extra: Option<&Highlights>) -> Result<Value, AutoCutError> {
    match look {
        Look::Animated(preset) => {
            let mut options = json!({"platform": platform.name()});
            if let Some(h) = extra {
                options["keywords"] = json!(h.keywords);
                options["emoji"] = json!(h.emoji);
            }
            run(ex, "captions.setAnimatedStyle", json!({"track": track, "preset": preset, "options": options})).await
        }
        Look::Plain => {
            let margin = (platform.margins().bottom + CAPTION_GAP).min(0.45);
            let style = json!({"track": track, "size": crate::shorts::CAPTION_SIZE, "color": "#ffffff", "background": false, "outline": 7, "outlineColor": "#000000", "align": "center", "anchor": "bottom", "margin": margin, "lineSpacing": 1.1, "xOffset": safe_offset(platform)});
            run(ex, "captions.setStyle", style).await
        }
    }
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

/// Words to colour and emoji to follow words, as the model picked them (checked).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Highlights {
    pub keywords: Vec<String>,
    pub emoji: BTreeMap<String, String>,
}

/// Most keywords and emoji per short.
pub const MAX_KEYWORDS: usize = 4;
pub const MAX_EMOJI: usize = 3;

const KEYWORDS_SYSTEM: &str = "You style the captions of short vertical videos. From the spoken text, pick the few words a viewer should notice (numbers, names, the strongest nouns and verbs) to colour, and at most a few emoji to show after words where an emoji plainly fits the word. Use only words that appear in the text. Answer with JSON only.";

/// A word as compared: lower case, without the punctuation around it.
fn bare(w: &str) -> String {
    w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

/// Whether `s` is a short emoji (pictographs, symbols and their joiners, no letters or digits).
pub fn is_emoji(s: &str) -> bool {
    let n = s.chars().count();
    (1..=8).contains(&n)
        && s.chars().all(|c| matches!(c as u32, 0x2190..=0x2BFF | 0x1F000..=0x1FAFF | 0xFE0F | 0x200D | 0x20E3))
        && s.chars().any(|c| c as u32 >= 0x2190 && c as u32 != 0xFE0F && c as u32 != 0x200D)
}

/// Check a model's answer against the caption `text`: keywords that are words of it, emoji that
/// are emoji after words of it, at most [`MAX_KEYWORDS`] and [`MAX_EMOJI`].
pub fn check_highlights(answer: &Value, text: &str) -> Highlights {
    let words: Vec<String> = text.split_whitespace().map(bare).filter(|w| !w.is_empty()).collect();
    let mut h = Highlights::default();
    for k in answer["keywords"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(Value::as_str).take(50) {
        let k = bare(k);
        if !k.is_empty() && words.contains(&k) && !h.keywords.contains(&k) && h.keywords.len() < MAX_KEYWORDS {
            h.keywords.push(k);
        }
    }
    let pairs: Vec<(String, String)> = match &answer["emoji"] {
        Value::Array(xs) => xs.iter().take(50).filter_map(|x| Some((x["word"].as_str()?.to_string(), x["emoji"].as_str()?.to_string()))).collect(),
        Value::Object(m) => m.iter().take(50).filter_map(|(w, e)| Some((w.clone(), e.as_str()?.to_string()))).collect(),
        _ => Vec::new(),
    };
    for (w, e) in pairs {
        let (w, e) = (bare(&w), e.trim().to_string());
        if words.contains(&w) && is_emoji(&e) && h.emoji.len() < MAX_EMOJI {
            h.emoji.entry(w).or_insert(e);
        }
    }
    h
}

/// Ask `model` for the keywords and emoji of captions saying `text` (see the module docs). A
/// model that fails gives none, with a note.
pub async fn keywords(model: &dyn ChatModel, text: &str, notes: &mut Vec<String>) -> Highlights {
    let text: String = text.chars().take(2000).collect();
    let user = format!(
        "Spoken text: \"{text}\"\nPick up to {MAX_KEYWORDS} keywords and up to {MAX_EMOJI} emoji.\nAnswer: {{\"keywords\": [\"word\", ...], \"emoji\": [{{\"word\": \"word\", \"emoji\": \"🔥\"}}, ...]}}"
    );
    match model.complete_json(KEYWORDS_SYSTEM, &user, &[]).await {
        Ok(v) => check_highlights(&v, &text),
        Err(e) => {
            notes.push(format!("caption keywords: {e}"));
            Highlights::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_from_params() {
        assert_eq!(Look::of(&json!({})), Look::Animated("bold".into()));
        assert_eq!(Look::of(&json!({"captionStyle": "plain"})), Look::Plain);
        assert_eq!(Look::of(&json!({"captionStyle": "karaoke"})), Look::Animated("karaoke".into()));
    }

    #[test]
    fn the_offset_centres_on_the_safe_span() {
        // YouTube: 5 % left, 18 % right → the middle is 6.5 % left of the frame's
        assert!((safe_offset(Platform::Youtube) + 0.065).abs() < 1e-9, "{}", safe_offset(Platform::Youtube));
        // Reels is symmetric; TikTok's button column pulls the middle left low in the frame
        assert!(safe_offset(Platform::Reels).abs() < 1e-9);
        assert!(safe_offset(Platform::TikTok) < -0.03);
    }

    #[test]
    fn model_answers_are_checked_against_the_text() {
        let text = "We saved $1,600 in one month, and the fire alarm never went off!";
        let a = json!({
            "keywords": ["saved", "MONTH!", "invented", "fire", "alarm", "never", "off"],
            "emoji": [{"word": "fire", "emoji": "🔥"}, {"word": "month", "emoji": "x"}, {"word": "ghost", "emoji": "👻"}, {"word": "saved", "emoji": "💰"}, {"word": "alarm", "emoji": "🚨"}, {"word": "off", "emoji": "❌"}]
        });
        let h = check_highlights(&a, text);
        assert_eq!(h.keywords, ["saved", "month", "fire", "alarm"], "real words only, at most four");
        assert_eq!(h.emoji.len(), 3);
        assert_eq!(h.emoji.get("fire").map(String::as_str), Some("🔥"));
        assert!(!h.emoji.contains_key("ghost") && !h.emoji.contains_key("month"));
        // nonsense and hostile answers give nothing
        for bad in [json!(null), json!({"keywords": "fire"}), json!({"emoji": 5}), json!({"keywords": [1, 2, {"a": 1}]})] {
            assert_eq!(check_highlights(&bad, text), Highlights::default(), "{bad}");
        }
        assert!(is_emoji("❤️") && is_emoji("👩‍👧") && !is_emoji("a") && !is_emoji("") && !is_emoji("🔥🔥🔥🔥🔥🔥🔥🔥🔥"));
    }

    #[tokio::test]
    async fn the_model_is_asked_and_a_failure_is_a_note() {
        let m = crate::llm::FixedModel::new(vec![json!({"keywords": ["money"], "emoji": {"money": "💰"}})]);
        let mut notes = Vec::new();
        let h = keywords(&m, "save money now", &mut notes).await;
        assert_eq!(h.keywords, ["money"]);
        assert_eq!(h.emoji.get("money").map(String::as_str), Some("💰"));
        assert!(m.asked.lock().unwrap()[0].1.contains("save money now"));
        let h = keywords(&m, "again", &mut notes).await;
        assert_eq!(h, Highlights::default());
        assert_eq!(notes.len(), 1, "{notes:?}");
    }
}
