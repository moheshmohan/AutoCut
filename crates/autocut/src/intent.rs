//! Prompt classification: what the user asked for, in a shape the planners can execute.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::ChatModel;
use crate::{AutoCutError, extract_json};

/// Every intent the local planner can execute without help. `custom` means "not one of these"
/// and is what triggers supervisor escalation.
pub const INTENTS: &[&str] = &["remove_silence", "remove_fillers", "add_captions", "cut_range", "keep_range", "highlight_reel", "custom"];

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
- add_captions: captions/subtitles from the spoken words. params: {\"maxChars\"?: int (per caption, default 42), \"lines\"?: 1|2 (default 2)}
- cut_range: remove a time range. params: {\"start\": seconds number, \"end\": seconds number, \"mode\"?: \"extract\" (close the gap, default) | \"lift\" (leave a gap)}
- keep_range: keep ONLY a time range. params: {\"start\": seconds, \"end\": seconds}
- highlight_reel: keep the most interesting spoken moments, roughly targetSeconds long. params: {\"targetSeconds\"?: number (default 60), \"topic\"?: string (what 'interesting' means here)}
- custom: anything else, multi-step requests, or anything unclear.

Times are seconds from the start of the video. Classify, don't plan: pick the single best intent.";

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FixedModel;
    use serde_json::json;

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
