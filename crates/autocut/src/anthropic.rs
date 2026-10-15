//! The cloud supervisor: Anthropic's Messages API, for prompts the local models can't plan.

use std::io::Read;

use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, extract_json};

const URL: &str = "https://api.anthropic.com/v1/messages";
const VERSION: &str = "2023-06-01";

/// A Claude model behind an API key.
#[derive(Clone)]
pub struct Anthropic {
    pub model: String,
    pub api_key: String,
    max_tokens: u32,
    agent: ureq::Agent,
}

impl Anthropic {
    pub fn new(model: &str, api_key: &str) -> Self {
        Self { model: model.to_string(), api_key: api_key.to_string(), max_tokens: 4096, agent: crate::ollama::http_agent() }
    }
}

impl ChatModel for Anthropic {
    fn id(&self) -> String {
        format!("anthropic:{}", self.model)
    }

    fn complete_json(&self, system: &str, user: &str, _images: &[Vec<u8>]) -> Result<Value, AutoCutError> {
        let body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}]
        });
        let resp = self
            .agent
            .post(URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", VERSION)
            .send_json(&body)
            .map_err(|e| AutoCutError::Llm(format!("Anthropic request failed: {e}")))?;
        let status = resp.status();
        let mut text = String::new();
        resp.into_body().into_reader().take(1 << 24).read_to_string(&mut text).map_err(|e| AutoCutError::Llm(e.to_string()))?;
        if status != 200 {
            // never include the key; the body never contains it, but be safe with what we echo
            return Err(AutoCutError::Llm(format!("Anthropic answered {status}: {}", truncate(&text, 400))));
        }
        parse_response(&text).ok_or_else(|| AutoCutError::Llm(format!("Anthropic did not answer JSON: {}", truncate(&text, 200))))
    }
}

/// The first text block of a Messages response, parsed as (or mined for) a JSON object.
pub fn parse_response(body: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(body).ok()?;
    let text = v["content"].as_array()?.iter().find_map(|b| b["text"].as_str())?;
    extract_json(text)
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_first_text_block() {
        let body = r#"{"content":[{"type":"text","text":"Here you go:\n{\"steps\":[{\"id\":\"playhead.set\"}]}"}]}"#;
        let v = parse_response(body).unwrap();
        assert_eq!(v["steps"][0]["id"], "playhead.set");
        assert!(parse_response(r#"{"content":[]}"#).is_none());
    }
}
