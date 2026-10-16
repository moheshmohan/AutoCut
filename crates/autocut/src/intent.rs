//! Prompt classification: what the user asked for, in a shape the planners can execute.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::ChatModel;
use crate::{AutoCutError, extract_json};

/// Every intent the local planner can execute without help. `custom` means "not one of these"
/// and is what triggers supervisor escalation.
pub const INTENTS: &[&str] = &[
    "remove_silence",
    "remove_fillers",
    "add_captions",
    "cut_range",
    "keep_range",
    "highlight_reel",
    "enhance_voice",
    "add_voice_over",
    "export_video",
    "slow_motion",
    "showcase",
    "custom",
];

/// What the classifier returned for a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    /// One of [`INTENTS`].
    pub intent: String,
    /// 0..=1; below [`crate::Config::confidence_threshold`] the prompt escalates.
    #[serde(default)]
    pub confidence: f32,
    /// Intent parameters, as the planner handlers expect them.
    #[serde(default)]
    pub params: Value,
    /// One line on why, shown to the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

impl Classification {
    /// A `custom` classification with zero confidence (the "couldn't understand" fallback).
    pub fn custom() -> Self {
        Self { intent: "custom".into(), confidence: 0.0, params: Value::Null, rationale: None }
    }

    /// Clamp the confidence and force unknown intents to `custom`.
    pub fn normalized(mut self) -> Self {
        if !INTENTS.contains(&self.intent.as_str()) {
            self.intent = "custom".into();
            self.confidence = 0.0;
        }
        self.confidence = self.confidence.clamp(0.0, 1.0);
        if !self.params.is_object() {
            self.params = Value::Object(serde_json::Map::new());
        }
        self
    }
}

const SYSTEM: &str = "\
You classify video-editing requests for a non-linear video editor. Answer with ONE JSON object and nothing else:
{\"intent\": string, \"confidence\": number 0..1, \"params\": object, \"rationale\": one short sentence}

Intents and their params:
- remove_silence: cut silent/paused parts. params: {\"minSeconds\"?: number (pause length to cut, default 0.5), \"keepSeconds\"?: number (pause length to keep, default 0.2)}
- remove_fillers: cut filler words (um, uh, like, you know). params: {\"fillers\"?: string[] (defaults apply when omitted)}
- add_captions: captions/subtitles from the spoken words. params: {\"maxChars\"?: int (per caption, default 42), \"lines\"?: 1|2 (default 2), \"style\"?: \"default\"|\"tiktok\"|\"bold\"|\"clean\"}
- cut_range: remove a time range. params: {\"start\": seconds number, \"end\": seconds number, \"mode\"?: \"extract\" (close the gap, default) | \"lift\" (leave a gap)}
- keep_range: keep ONLY a time range. params: {\"start\": seconds, \"end\": seconds}
- highlight_reel: keep the most interesting spoken moments, roughly targetSeconds long. params: {\"targetSeconds\"?: number (default 60), \"topic\"?: string (what 'interesting' means here)}
- enhance_voice: clean up / improve spoken audio (noise, hum, de-ess, loudness). params: {\"preset\"?: \"clean\"|\"balanced\"|\"podcast\"|\"radio\"|\"hall\", \"noise\"?: 0-10, \"deess\"?: 0-10, \"duckMusic\"?: bool}
- add_voice_over: synthesize a narration track from a script. params: {\"text\": string (the script), \"voice\"?: string, \"fit\"?: bool (stretch delivery to cover the video)}
- export_video: render/export the sequence to a file. params: {\"destination\": \"youtube\"|\"youtube4k\"|\"tiktok\"|\"reels\"|\"vertical\"|\"vimeo\"|\"prores\"|\"gif\"|\"audio\"|\"aac\", \"path\"?: string (only when the user named a file), \"burnCaptions\"?: bool, \"scale\"?: number}
- slow_motion: change playback speed (smooth by default). params: {\"factor\": number (0.5 = half speed), \"interpolation\"?: \"opticalFlow\" (default, smooth) |\"frameBlending\"|\"frameSampling\"}
- showcase: a new video of every clip that shows one kind of subject, counted in date order (\"make a video of every robot since 2020\"). params: {\"subject\": string (singular, e.g. \"robot\"), \"since\"?: \"YYYY\" | \"YYYY-MM-DD\", \"title\"?: string}
- custom: anything else.

Pick a specific intent ONLY when the request clearly and completely matches it. Use `custom` when
the request asks for effects, styling or a look (e.g. 'cinematic', 'sparkles', color grading),
combines several edits, mentions subjects on screen rather than speech or time ranges (except a
showcase of every clip of one subject), needs
reordering or new material, or is at all ambiguous — `custom` is always safe. Times are seconds
from the start of the video. Classify, don't plan: pick the single best intent.";

/// Classify `prompt` with the local model.
pub async fn classify(model: &dyn ChatModel, prompt: &str) -> Result<Classification, AutoCutError> {
    let v = model.complete_json(SYSTEM, prompt, &[]).await?;
    // tolerate a model that double-encodes the object as a string inside its JSON answer
    let parsed: Classification = serde_json::from_value(v.clone()).or_else(|_| {
        extract_json(&v.to_string())
            .and_then(|v| serde_json::from_value(v).ok())
            .ok_or_else(|| AutoCutError::Llm("the classifier answered unusable JSON".into()))
    })?;
    Ok(parsed.normalized())
}

/// Words that end a showcase's subject ("every robot *I've filmed*", "all the cars *since* 2019").
const SUBJECT_STOP: &[&str] = &[
    "i", "i've", "ive", "i'd", "i'm", "we", "we've", "you", "that", "which", "who", "since", "from", "after", "before", "in", "on", "at", "of", "for", "to",
    "and", "with", "filmed", "shot", "recorded", "seen", "met", "captured", "spotted", "ever", "so", "far", "this", "these", "my", "year", "years", "please",
    "into", "as", "video", "videos", "clip", "clips", "footage",
];

/// Subjects that are edits, not things on screen ("remove every silence", "all the pauses").
const NOT_SUBJECTS: &[&str] = &[
    "silence",
    "pause",
    "filler",
    "um",
    "uh",
    "caption",
    "subtitle",
    "cut",
    "second",
    "frame",
    "word",
    "scene",
    "shot",
    "transition",
    "sentence",
    "breath",
    "gap",
    "mistake",
    "take",
    "track",
    "marker",
    "effect",
    "time",
    "minute",
    "one",
    "thing",
];

/// Words that make a prompt ask for a new video.
const SHOWCASE_WORDS: &[&str] = &["video", "showcase", "montage", "compilation", "reel", "supercut", "countdown", "movie", "film"];

/// `robots` → `robot`, `foxes` → `fox`, `butterflies` → `butterfly`.
fn singular(w: &str) -> String {
    if let Some(stem) = w.strip_suffix("ies").filter(|s| s.len() >= 2) {
        return format!("{stem}y");
    }
    if let Some(stem) = w.strip_suffix("es").filter(|s| s.ends_with('x') || s.ends_with("ch") || s.ends_with("sh") || s.ends_with("ss")) {
        return stem.to_string();
    }
    match w.strip_suffix('s') {
        Some(stem) if stem.len() >= 2 && !stem.ends_with('s') => stem.to_string(),
        _ => w.to_string(),
    }
}

/// A `since` date in a prompt: a year (1900–2199), or `YYYY-MM` / `YYYY-MM-DD`.
fn date_word(w: &str) -> Option<String> {
    let year = w.get(..4).filter(|y| y.chars().all(|c| c.is_ascii_digit()) && (y.starts_with("19") || y.starts_with("20") || y.starts_with("21")))?;
    let rest = w.get(4..).unwrap_or_default();
    let ok = rest.is_empty() || (rest.starts_with('-') && rest.len() <= 6 && rest.chars().all(|c| c.is_ascii_digit() || c == '-'));
    ok.then(|| format!("{year}{rest}"))
}

/// A showcase request recognised without a model: "every" or "all (the|my)" and a subject, in a
/// prompt that asks for a video ("make a video of every robot since 2020", "a montage of all my
/// vintage cars"). Conservative: edits ("remove every silence") and prompts without a video word
/// are left to the classifier.
pub fn showcase_request(prompt: &str) -> Option<Classification> {
    let words: Vec<String> = prompt
        .chars()
        .take(400)
        .collect::<String>()
        .to_lowercase()
        .replace('’', "'")
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-').to_string())
        .filter(|w| !w.is_empty())
        .collect();
    if !words.iter().any(|w| SHOWCASE_WORDS.contains(&singular(w).as_str())) {
        return None;
    }
    let at = words.iter().position(|w| w == "every" || w == "all")?;
    let plural = words.get(at).is_some_and(|w| w == "all");
    let mut i = at + 1;
    while words.get(i).is_some_and(|w| matches!(w.as_str(), "the" | "my" | "of" | "our" | "single")) {
        i += 1;
    }
    let mut subject: Vec<String> = Vec::new();
    while let Some(w) = words.get(i) {
        if subject.len() >= 3 || SUBJECT_STOP.contains(&w.as_str()) || w.chars().any(|c| c.is_ascii_digit()) {
            break;
        }
        subject.push(w.clone());
        i += 1;
    }
    let last = subject.pop()?;
    subject.push(if plural { singular(&last) } else { last });
    if subject.iter().any(|w| NOT_SUBJECTS.contains(&singular(w).as_str())) {
        return None;
    }
    let since = words.windows(2).find_map(|p| match (p.first().map(String::as_str), p.get(1)) {
        (Some("since" | "from" | "after"), Some(w)) => date_word(w),
        _ => None,
    });
    let mut params = serde_json::json!({"subject": subject.join(" ")});
    if let Some(s) = since {
        params["since"] = Value::String(s);
    }
    Some(Classification { intent: "showcase".into(), confidence: 1.0, params, rationale: Some("a video of every clip of one subject".into()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FixedModel;
    use serde_json::json;

    #[test]
    fn showcase_requests_are_recognised_without_a_model() {
        let c = showcase_request("make a video of every robot since 2020").unwrap();
        assert_eq!((c.intent.as_str(), c.confidence), ("showcase", 1.0));
        assert_eq!(c.params, json!({"subject": "robot", "since": "2020"}));
        let c = showcase_request("Make a montage of all the robot dogs I've filmed from 2019-06!").unwrap();
        assert_eq!(c.params, json!({"subject": "robot dog", "since": "2019-06"}));
        let c = showcase_request("a compilation of every vintage car I’ve seen").unwrap();
        assert_eq!(c.params, json!({"subject": "vintage car"}));
        assert_eq!(showcase_request("showcase all my butterflies").unwrap().params["subject"], "butterfly");
        assert_eq!(showcase_request("video of all the foxes").unwrap().params["subject"], "fox");
        // edits, and prompts that ask for no video, are the classifier's
        for p in [
            "remove every silence from the video",
            "cut all the pauses in this video",
            "add captions to every clip of the video",
            "every robot since 2020",
            "make a video",
            "make a video of every",
            "a video of every 2020",
            "",
        ] {
            assert!(showcase_request(p).is_none(), "{p}");
        }
        assert_eq!(showcase_request("a video from my trip of every robot since 2021").unwrap().params["since"], "2021");
        // a hostile length is cut, not choked on
        assert!(showcase_request(&format!("make a video of every robot since 2020 {}", "x ".repeat(100_000))).is_some());
    }

    #[tokio::test]
    async fn parses_and_normalizes() {
        let m = FixedModel::new(vec![json!({"intent": "cut_range", "confidence": 1.4, "params": {"start": 10, "end": 20}, "rationale": "cut the intro"})]);
        let c = classify(&m, "cut out the intro from 10 to 20 seconds").await.unwrap();
        assert_eq!(c.intent, "cut_range");
        assert_eq!(c.confidence, 1.0);
        assert_eq!(c.params["start"], 10);

        let m = FixedModel::new(vec![json!({"intent": "make_me_a_movie", "params": null})]);
        let c = classify(&m, "xyz").await.unwrap();
        assert_eq!((c.intent.as_str(), c.confidence), ("custom", 0.0));
        assert!(c.params.is_object());
    }

    #[test]
    fn every_catalogued_intent_is_listed() {
        for i in INTENTS {
            assert!(!i.is_empty());
        }
        assert!(INTENTS.contains(&"custom"));
    }
}
