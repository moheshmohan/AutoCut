//! Eyes and hands for the automation agent.
//!
//! Google retired the Photos Library API's library-wide read scopes on 2025-03-31, so the
//! only way to read a user's library is the Picker — a web page Google renders and the user
//! fills in. This crate turns that page into an agent-drivable surface: it captures the
//! browser window, grounds the UI elements in the screenshot with a local vision model
//! (Qwen2.5-VL through Ollama), and drives them — click, type, scroll — inside that one
//! window, with hard budgets so it can never loop forever.
//!
//! - [`capture`]: the target window's bounds and a screenshot of it (subprocess, macOS).
//! - [`act`]: input synthesis into the target app, clamped to the window's rectangle.
//! - [`ground`]: the vision model call — `inventory` lists elements, `find` locates one.
//! - [`verify`]: pixel-diff of two captures, so the agent knows a click landed.
//! - [`picker`]: the Google Photos picker's visual semantics (duration badges, Done).
//! - [`agent`]: the budgeted see–decide–act loop that selects media and presses Done — over
//!   one search or many in one picking session — and closes the picker tab it drove.
//! - [`locations`]: where an album's items were taken, read from the Google Photos page's Info
//!   panel item by item (Google strips locations from what its API and picker hand out).
//! - [`tabs`]: the picker's browser tabs — reuse one for the next session, close them, answer
//!   "Leave site?", and `photos tidy`.
//!
//! Everything is ordinary subprocesses and HTTP: no `unsafe` (the workspace forbids it),
//! no browser embedding, and no model weights in the repo — the model is served by the
//! Ollama server AutoCut already requires. See README.md for the model-licensing notes.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod act;
pub mod agent;
pub mod capture;
pub mod ground;
pub mod locations;
pub mod picker;
pub mod tabs;
pub mod verify;

use std::time::Duration;

/// Everything the screen agent can fail with, all human-readable and actionable.
#[derive(Debug, thiserror::Error)]
pub enum ScreenError {
    #[error("this build cannot drive the screen ({0}); the screenagent needs macOS")]
    Unsupported(String),
    #[error(
        "cannot read the {0} window: {1} — grant Screen Recording to the app running this command (System Settings ▸ Privacy & Security ▸ Screen Recording), then run it again"
    )]
    Capture(String, String),
    #[error(
        "cannot send input to {0}: {1} — grant Accessibility to the app running this command (System Settings ▸ Privacy & Security ▸ Accessibility), then run it again"
    )]
    Input(String, String),
    #[error("the vision model is unreachable: {0} — is `ollama serve` running, and is the model pulled (`ollama pull {1}`)?")]
    Model(String, String),
    #[error("the vision model's reply was unusable: {0}")]
    Parse(String),
    #[error("the agent stopped making progress: {0}")]
    Stalled(String),
    #[error("{0}")]
    Other(String),
}

/// Where the vision model lives and which one to ask. `AUTOCUT_OLLAMA_HOST` is shared with
/// the AutoCut classifier (one server); `AUTOCUT_SCREEN_MODEL` picks the grounding model.
#[derive(Debug, Clone)]
pub struct Config {
    pub ollama_host: String,
    pub model: String,
    pub http_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_host: "http://127.0.0.1:11434".into(),
            // Qwen2.5-VL grounds UI screenshots natively (bounding boxes as JSON) and is
            // Apache-2.0; `qwen2.5vl:7b` is the quality/size sweet spot on an Apple-silicon
            // Mac. See README.md for why not ShowUI-2B or OmniParser.
            model: "qwen2.5vl:7b".into(),
            http_timeout: Duration::from_secs(240),
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let d = Self::default();
        let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        Self {
            ollama_host: env("AUTOCUT_OLLAMA_HOST").unwrap_or(d.ollama_host).trim_end_matches('/').into(),
            model: env("AUTOCUT_SCREEN_MODEL").unwrap_or(d.model),
            http_timeout: d.http_timeout,
        }
    }
}

/// Standard base64 (for the Ollama `images` array).
pub(crate) fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Every complete JSON object in `s`, in order, at most `cap` of them. Vision models wrap
/// JSON in prose or code fences, answer with an array where an object was asked for, and
/// get cut off by the token limit; scanning for balanced `{…}` objects copes with all three
/// (an unclosed outer object still yields the complete objects inside it).
pub(crate) fn json_objects(s: &str, cap: usize) -> Vec<serde_json::Value> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() && out.len() < cap {
        if bytes.get(i) != Some(&b'{') {
            i += 1;
            continue;
        }
        let parsed = balanced_end(bytes, i).and_then(|end| {
            let v = serde_json::from_slice::<serde_json::Value>(bytes.get(i..=end)?).ok()?;
            Some((v, end))
        });
        match parsed {
            Some((v, end)) => {
                out.push(v);
                i = end + 1;
            }
            None => i += 1, // truncated or malformed: look for complete objects inside it
        }
    }
    out
}

/// Index of the `}` closing the `{` at `start`, string-aware; `None` when it never closes.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth = depth.saturating_add(1),
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_env() {
        let c = Config::default();
        assert_eq!(c.model, "qwen2.5vl:7b");
        assert_eq!(c.ollama_host, "http://127.0.0.1:11434");
        // whatever the environment holds, from_env always yields a usable, non-empty config
        let e = Config::from_env();
        assert!(!e.model.is_empty());
        assert!(!e.ollama_host.is_empty());
    }

    #[test]
    fn json_objects_survive_prose_fences_arrays_and_truncation() {
        let one = json_objects(r#"{"a":1}"#, 8);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0]["a"].as_i64(), Some(1));
        // a fenced array of objects, the shape Qwen2.5-VL answers grounding asks with
        let fenced =
            json_objects("Sure!\n```json\n[\n {\"bbox_2d\": [1, 2, 3, 4], \"label\": \"0:06\"},\n {\"bbox_2d\": [5, 6, 7, 8], \"label\": \"x\"}\n]\n```", 8);
        assert_eq!(fenced.len(), 2);
        assert_eq!(fenced[1]["bbox_2d"][0].as_i64(), Some(5));
        // cut off mid-array: the complete objects survive
        let cut = json_objects(r#"{"elements": [{"label": "a"}, {"label": "b"}, {"label": "#, 8);
        assert_eq!(cut.iter().filter_map(|v| v["label"].as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        // braces inside strings don't confuse it; prose is empty; the cap holds
        assert_eq!(json_objects(r#"{"s":"}{ not a brace"}"#, 8)[0]["s"].as_str(), Some("}{ not a brace"));
        assert!(json_objects("There are none.", 8).is_empty());
        assert_eq!(json_objects(&"{}".repeat(100), 5).len(), 5);
    }
}
