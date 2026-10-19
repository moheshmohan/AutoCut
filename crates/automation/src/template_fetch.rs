//! Downloads for `template.install {url}`: pure-Rust HTTPS (the agent the Google Photos client
//! uses: rustls with the OS's certificate verification, https only, so a redirect to plain http
//! fails), the link checked first, the body read in pieces up to the engine's cap and stopped
//! when the job is cancelled. Nothing downloaded is run: the engine checks the package whole
//! before writing it.

use std::io::Read as _;
use std::sync::Arc;
use std::time::Duration;

use filmcraft_engine::video_template_share::{TemplateFetcher, check_url};

/// The longest a download may take.
const TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// [`TemplateFetcher`] over HTTPS.
pub struct HttpsFetcher;

impl TemplateFetcher for HttpsFetcher {
    fn fetch(&self, url: &str, max_bytes: u64, cancelled: &dyn Fn() -> bool) -> Result<Vec<u8>, String> {
        check_url(url)?;
        let agent = filmcraft_gphotos::api::agent_with_timeout(TIMEOUT);
        let mut resp = agent.get(url).call().map_err(|e| format!("downloading {url}: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("downloading {url}: the server answered {status}"));
        }
        let declared = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse::<u64>().ok());
        if declared.is_some_and(|n| n > max_bytes) {
            return Err(format!("the package is {} bytes (at most {max_bytes})", declared.unwrap_or(0)));
        }
        let mut reader = resp.body_mut().with_config().limit(max_bytes.saturating_add(1)).reader();
        let mut out: Vec<u8> = Vec::with_capacity(usize::try_from(declared.unwrap_or(0).min(16 << 20)).unwrap_or(0));
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            if cancelled() {
                return Err("cancelled".into());
            }
            let n = reader.read(&mut buf).map_err(|e| format!("downloading {url}: {e}"))?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(buf.get(..n).unwrap_or_default());
            if out.len() as u64 > max_bytes {
                return Err(format!("the package is larger than {max_bytes} bytes"));
            }
        }
        Ok(out)
    }
}

/// The fetcher the app, the CLI and the MCP server install (`session.sharing.fetcher`).
pub fn fetcher() -> Arc<dyn TemplateFetcher> {
    Arc::new(HttpsFetcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_are_checked_before_any_connection() {
        let f = HttpsFetcher;
        for url in ["http://example.org/t.zip", "https://user:pw@example.org/t.zip", "ftp://x/y", "https:// spaced/x.zip"] {
            let e = f.fetch(url, 1024, &|| false).unwrap_err();
            assert!(!e.contains("downloading"), "{url}: refused before connecting: {e}");
        }
    }
}
