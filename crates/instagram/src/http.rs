//! How the Instagram calls talk HTTP: a [`Transport`] (ureq with pure-Rust TLS in the product,
//! the same over plain HTTP to a mock server on `127.0.0.1` in the tests), requests whose body
//! may stream from a file (the video, never whole in memory), and capped responses.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

/// Bodies read from Instagram (small JSON); anything longer is cut off.
pub const MAX_BODY_BYTES: u64 = 1 << 20;
/// Response headers kept.
const MAX_HEADERS: usize = 256;
/// The longest a request takes: a 300 MB video over a slow line still fits.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);
/// How much of the file one read takes (and how often progress is reported).
const READ_CHUNK: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

/// What a request carries.
pub enum Body<'a> {
    None,
    /// `application/x-www-form-urlencoded` fields.
    Form(Vec<(&'static str, String)>),
    /// `len` bytes of the file at `path` from `offset`, streamed. `progress(sent)` is told how
    /// many have gone; `false` stops the request (the transport answers `Err(CANCELLED)`).
    File {
        path: &'a Path,
        offset: u64,
        len: u64,
        progress: &'a mut dyn FnMut(u64) -> bool,
    },
}

/// One request. Header values are never logged; secrets go in headers or form fields, except
/// where Meta's token endpoints take them in the query (see `auth.rs`).
pub struct Request<'a> {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(&'static str, String)>,
    pub body: Body<'a>,
}

impl Request<'_> {
    pub fn get(url: String) -> Self {
        Request { method: Method::Get, url, headers: Vec::new(), body: Body::None }
    }

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

/// What a transport answers when a file body's `progress` stopped it.
pub const CANCELLED: &str = "cancelled";

/// How the calls talk HTTP. `Err` is a failure with no response (connection dropped, timeout,
/// or [`CANCELLED`]): an upload retries it. HTTP error statuses come back as responses.
pub trait Transport: Send + Sync {
    fn send(&self, req: Request<'_>) -> Result<Response, String>;
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

/// A file section read in pieces, telling `progress` as it goes.
struct FileReader<'a> {
    file: std::io::Take<std::fs::File>,
    sent: u64,
    progress: &'a mut dyn FnMut(u64) -> bool,
    cancelled: bool,
}

impl Read for FileReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cancelled {
            return Err(std::io::Error::other(CANCELLED));
        }
        let n = buf.len().min(READ_CHUNK);
        let (head, _) = buf.split_at_mut(n);
        let got = self.file.read(head)?;
        self.sent = self.sent.saturating_add(got as u64);
        if !(self.progress)(self.sent) {
            self.cancelled = true;
            return Err(std::io::Error::other(CANCELLED));
        }
        Ok(got)
    }
}

impl Transport for UreqTransport {
    fn send(&self, req: Request<'_>) -> Result<Response, String> {
        let mut cancelled = false;
        let sent = match req.method {
            Method::Get => {
                let mut b = self.agent.get(&req.url);
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                b.call()
            }
            Method::Post => {
                let mut b = self.agent.post(&req.url);
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                match req.body {
                    Body::None => b.send_empty(),
                    Body::Form(fields) => b.send_form(fields.iter().map(|(k, v)| (*k, v.as_str()))),
                    Body::File { path, offset, len, progress } => {
                        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
                        f.seek(SeekFrom::Start(offset)).map_err(|e| format!("{}: {e}", path.display()))?;
                        let mut reader = FileReader { file: f.take(len), sent: 0, progress, cancelled: false };
                        let r = b.header("Content-Length", len.to_string()).send(ureq::SendBody::from_reader(&mut reader));
                        cancelled = reader.cancelled;
                        r
                    }
                }
            }
        };
        if cancelled {
            return Err(CANCELLED.into());
        }
        let resp = sent.map_err(describe)?;
        let status = resp.status().as_u16();
        let headers = resp.headers().iter().take(MAX_HEADERS).filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string()))).collect();
        let mut body = Vec::new();
        resp.into_body().into_reader().take(MAX_BODY_BYTES).read_to_end(&mut body).map_err(|e| format!("the answer broke off: {}", e.kind()))?;
        Ok(Response { status, headers, body: String::from_utf8_lossy(&body).into_owned() })
    }
}

/// A transport failure in words, without the URL (Meta's token calls carry secrets in it).
fn describe(e: ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => "the connection timed out".into(),
        ureq::Error::Io(io) => format!("the connection failed ({})", io.kind()),
        ureq::Error::HostNotFound => "the host was not found (offline?)".into(),
        ureq::Error::ConnectionFailed => "the connection failed".into(),
        ureq::Error::Tls(_) => "the secure connection failed".into(),
        _ => "the request failed".into(),
    }
}

/// The product's agent: pure-Rust TLS with the OS's certificate verification, https only, status
/// codes as answers (not errors), no redirects followed, bounded waits.
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

/// `scheme://host[:port]` of a URL (lowercase), `None` when it has none or carries user info.
pub fn origin(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if scheme.is_empty() || host.is_empty() || host.contains('@') {
        return None;
    }
    Some(format!("{}://{}", scheme.to_ascii_lowercase(), host.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins() {
        assert_eq!(origin("https://rupload.facebook.com/ig-api-upload/v25.0/1").as_deref(), Some("https://rupload.facebook.com"));
        assert_eq!(origin("http://127.0.0.1:5000/a").as_deref(), Some("http://127.0.0.1:5000"));
        assert_eq!(origin("HTTPS://RUpload.Facebook.com"), Some("https://rupload.facebook.com".into()));
        assert_eq!(origin("https://evil@rupload.facebook.com/x"), None, "user info");
        assert_eq!(origin("nonsense"), None);
        assert_eq!(origin("https:///path"), None);
    }

    #[test]
    fn file_bodies_report_progress_and_stop() {
        let dir = std::env::temp_dir().join(format!("filmcraft-ig-http-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        std::fs::write(&p, vec![1u8; 600 * 1024]).unwrap();
        let mut f = std::fs::File::open(&p).unwrap();
        f.seek(SeekFrom::Start(100)).unwrap();
        let mut seen = Vec::new();
        let mut progress = |n: u64| {
            seen.push(n);
            n < 400 * 1024
        };
        let mut r = FileReader { file: f.take(600 * 1024 - 100), sent: 0, progress: &mut progress, cancelled: false };
        let mut out = Vec::new();
        let e = r.read_to_end(&mut out).unwrap_err();
        assert_eq!(e.to_string(), CANCELLED);
        assert!(r.cancelled);
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
