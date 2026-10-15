//! The Ollama client: local text and vision models over its HTTP API (`/api/chat`).

use std::io::Read;
use std::time::Duration;

use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, extract_json};

/// An Ollama endpoint plus the model to talk to.
#[derive(Clone)]
pub struct Ollama {
    pub host: String,
    pub model: String,
    agent: ureq::Agent,
    /// A short-timeout agent for liveness/model checks.
    fast: ureq::Agent,
}

impl Ollama {
    pub fn new(host: &str, model: &str, timeout: Duration) -> Self {
        Self { host: host.trim_end_matches('/').to_string(), model: model.to_string(), agent: agent_with(timeout), fast: agent_with(Duration::from_secs(5)) }
    }

    /// Whether the server answers and has the configured model (`GET /api/tags`).
    pub fn available(&self) -> Result<bool, AutoCutError> {
        let body = self.get_json("/api/tags", &self.fast)?;
        let name = self.model.split(':').next().unwrap_or(&self.model);
        let listed = body["models"].as_array().is_some_and(|ms| {
            ms.iter().any(|m| {
                let full = m["name"].as_str().unwrap_or_default();
                full == self.model || full.split(':').next().unwrap_or(full) == name
            })
        });
        Ok(listed)
    }

    fn get_json(&self, path: &str, agent: &ureq::Agent) -> Result<Value, AutoCutError> {
        let url = format!("{}{path}", self.host);
        let resp = agent.get(&url).call().map_err(|e| AutoCutError::Llm(format!("Ollama at {} is unreachable: {e}", self.host)))?;
        let mut body = String::new();
        resp.into_body().into_reader().take(1 << 22).read_to_string(&mut body).map_err(|e| AutoCutError::Llm(e.to_string()))?;
        serde_json::from_str(&body).map_err(|e| AutoCutError::Llm(format!("Ollama sent invalid JSON: {e}")))
    }

    fn chat(&self, system: &str, user: &str, images: &[Vec<u8>]) -> Result<Value, AutoCutError> {
        let mut message = json!({"role": "user", "content": user});
        if !images.is_empty() {
            message["images"] = Value::Array(images.iter().map(|png| Value::String(crate::b64(png))).collect());
        }
        let body = json!({
            "model": self.model,
            "stream": false,
            "format": "json",
            "messages": [{"role": "system", "content": system}, message],
            "options": {"temperature": 0.1}
        });
        let url = format!("{}/api/chat", self.host);
        let resp = self.agent.post(&url).send_json(&body).map_err(|e| AutoCutError::Llm(format!("Ollama ({}) failed: {e}", self.model)))?;
        let mut text = String::new();
        resp.into_body().into_reader().take(1 << 24).read_to_string(&mut text).map_err(|e| AutoCutError::Llm(e.to_string()))?;
        parse_chat(&text).ok_or_else(|| AutoCutError::Llm(format!("Ollama ({}) did not answer JSON: {}", self.model, truncate(&text, 200))))
    }
}

impl ChatModel for Ollama {
    fn id(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn complete_json(&self, system: &str, user: &str, images: &[Vec<u8>]) -> crate::llm::JsonFuture {
        // blocking HTTP on a blocking thread; Ollama is cheap to clone
        let me = self.clone();
        let (system, user, images) = (system.to_string(), user.to_string(), images.to_vec());
        Box::pin(async move {
            match tokio::task::spawn_blocking(move || me.chat(&system, &user, &images)).await {
                Ok(r) => r,
                Err(e) => Err(AutoCutError::Llm(format!("internal error: {e}"))),
            }
        })
    }
}

/// The model's reply text out of a `/api/chat` (non-streaming) response body.
pub fn parse_chat(body: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(body).ok()?;
    let content = v["message"]["content"].as_str()?;
    extract_json(content)
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The shared HTTP agent: pure-Rust TLS (rustls + platform verifier), as the speech downloader
/// configures it, so http://localhost and https:// hosts both work. Shared with the Anthropic
/// client.
pub fn http_agent() -> ureq::Agent {
    agent_with(Duration::from_secs(240))
}

fn agent_with(global_timeout: Duration) -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(global_timeout))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_reply_content_as_json() {
        assert_eq!(
            parse_chat(r#"{"message":{"role":"assistant","content":"{\"intent\":\"add_captions\",\"confidence\":0.9}"}}"#).map(|v| v["intent"].clone()),
            Some(json!("add_captions"))
        );
        assert_eq!(parse_chat(r#"{"message":{"content":"Sure: {\"a\":1} done"},"done":true}"#), Some(json!({"a": 1})));
        assert_eq!(parse_chat(r#"{"message":{"content":"no json"}}"#), None);
        assert_eq!(parse_chat("not json"), None);
    }
}
