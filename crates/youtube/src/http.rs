//! How the YouTube calls talk HTTP: a [`Transport`] (ureq with pure-Rust TLS in the product, the
//! same over plain HTTP to a mock server on `127.0.0.1` in the tests), requests and capped
//! responses.

use std::io::Read as _;
use std::time::Duration;

/// Bodies read from YouTube (small JSON); anything longer is cut off.
pub const MAX_BODY_BYTES: u64 = 1 << 20;
/// Response headers kept (Google sends a dozen or so).
const MAX_HEADERS: usize = 256;
/// The longest a request takes: an 8 MiB chunk over a slow line still fits.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
}

/// One request.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub method: Method,
    pub url: &'a str,
    pub headers: Vec<(&'static str, String)>,
    pub body: &'a [u8],
}

impl Request<'_> {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// A response: the status, the headers, and the body (at most [`MAX_BODY_BYTES`], lossy UTF-8).
#[derive(Debug, Clone, Default)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Response {
    /// A header by name, case-insensitively, trimmed.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.trim())
    }
}

/// How the calls talk HTTP. `Err` is a failure with no response (connection dropped, timeout):
/// an upload retries it. HTTP error statuses (and YouTube's `308 Resume Incomplete`) come back
/// as responses.
pub trait Transport: Send + Sync {
    fn send(&self, req: &Request<'_>) -> Result<Response, String>;
}

/// The real transport over a ureq agent.
pub struct UreqTransport {
    agent: ureq::Agent,
}

impl Default for UreqTransport {
    fn default() -> Self {
        Self { agent: https_agent(REQUEST_TIMEOUT) }
    }
}

impl UreqTransport {
    /// Over `agent` (the tests give one that may talk plain HTTP to their mock).
    pub fn with_agent(agent: ureq::Agent) -> Self {
        Self { agent }
    }
}

impl Transport for UreqTransport {
    fn send(&self, req: &Request<'_>) -> Result<Response, String> {
        let sent = match req.method {
            Method::Get => {
                let mut b = self.agent.get(req.url);
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                b.call()
            }
            Method::Post | Method::Put => {
                let mut b = if req.method == Method::Post { self.agent.post(req.url) } else { self.agent.put(req.url) };
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                b.send(req.body)
            }
        };
        let resp = sent.map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        let headers = resp.headers().iter().take(MAX_HEADERS).filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string()))).collect();
        let mut body = Vec::new();
        resp.into_body().into_reader().take(MAX_BODY_BYTES).read_to_end(&mut body).map_err(|e| e.to_string())?;
        Ok(Response { status, headers, body: String::from_utf8_lossy(&body).into_owned() })
    }
}

/// The product's agent: pure-Rust TLS with the OS's certificate verification, https only, status
/// codes as answers (not errors), no redirects followed (YouTube's `308 Resume Incomplete` is an
/// answer, not a redirect), bounded waits.
pub fn https_agent(timeout: Duration) -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .http_status_as_error(false)
        .https_only(true)
        .max_redirects(0)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(timeout))
        .build()
        .into()
}

/// An agent for a mock server on `127.0.0.1` (tests): plain HTTP allowed, otherwise as
/// [`https_agent`].
pub fn local_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder().http_status_as_error(false).max_redirects(0).timeout_global(Some(timeout)).build().into()
}

/// `scheme://host[:port]` of a URL (lowercase), `None` when it has none.
pub fn origin(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if scheme.is_empty() || host.is_empty() {
        return None;
    }
    Some(format!("{}://{}", scheme.to_ascii_lowercase(), host.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins() {
        assert_eq!(origin("https://www.googleapis.com/upload/youtube/v3/videos?x=1").as_deref(), Some("https://www.googleapis.com"));
        assert_eq!(origin("http://127.0.0.1:5000/a").as_deref(), Some("http://127.0.0.1:5000"));
        assert_eq!(origin("HTTPS://WWW.GoogleAPIs.com").as_deref(), Some("https://www.googleapis.com"));
        assert_eq!(origin("nonsense"), None);
        assert_eq!(origin("https:///path"), None);
    }
}
