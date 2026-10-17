//! Sending the video to a Reel container: Meta's resumable upload protocol on
//! `rupload.facebook.com`, and the containers kept across runs.
//!
//! 1. The container is made with `upload_type=resumable` ([`crate::api::create_reel`]).
//! 2. `POST https://rupload.facebook.com/ig-api-upload/<version>/<container>` with
//!    `Authorization: OAuth <token>`, `offset: <first byte>`, `file_size: <bytes>` and the bytes
//!    from `offset` on, streamed from disk (never whole in memory). Instagram answers
//!    `{"success": true}`, or `{"debug_info": {"retriable": …, "type": …, "message": …}}`.
//! 3. A dropped connection, a `5xx` or a retriable failure waits (doubling), asks the container
//!    how many bytes Instagram holds (`video_status.uploading_phase.bytes_transferred`, where
//!    Instagram reports it) and sends from there; without that number it sends the whole file
//!    again. At most [`MAX_FAILURES_IN_A_ROW`] failures in a row, [`MAX_FAILURES`] in all.
//! 4. **Across runs.** The container is kept (owner-only) in `<data dir>/instagram/sessions/`
//!    under a key of (account, file path, size, modification time, caption, cover, feed) until
//!    it is published. Running the same publish again within 23 hours — after a cancel, a crash
//!    or a container still processing — resumes that container instead of making another (each
//!    counts against the 400 a day).
//!
//! The upload URI receives the token, so it must be on the rupload host: one Instagram names
//! elsewhere is not used (the container's own URI is built instead).

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::api::{self, Ctx, pause};
use crate::http::{Body, Method, Request};
use crate::{IgError, clean, write_private};

/// Failures in a row before the upload gives up.
pub const MAX_FAILURES_IN_A_ROW: u32 = 5;
/// Failures in all.
pub const MAX_FAILURES: u32 = 12;
/// Kept containers older than this are not trusted (Instagram expires them at 24 h).
pub const KEEP_SECS: u64 = 23 * 60 * 60;

/// How the upload waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadOptions {
    /// The first wait after a failure (doubled each time, at most 30 s).
    pub backoff: Duration,
}

impl Default for UploadOptions {
    fn default() -> Self {
        Self { backoff: Duration::from_secs(1) }
    }
}

/// How an upload went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sent {
    /// Times it carried on from where Instagram had got to.
    pub resumed: u32,
    /// Times it had to send the whole file again.
    pub restarted: u32,
}

/// The upload URI for a container: Instagram's when it is on the rupload host, else built.
pub fn upload_uri(rupload: &str, version: &str, container: &str, named: Option<&str>) -> String {
    let own = format!("{}/{version}/{container}", rupload.trim_end_matches('/'));
    match named {
        Some(u)
            if crate::http::origin(u).is_some_and(|o| Some(o) == crate::http::origin(rupload))
                && u.len() <= 512
                && !u.chars().any(|c| c.is_whitespace() || c.is_control())
                && u.trim_end_matches('/').ends_with(&format!("/{container}")) =>
        {
            u.to_string()
        }
        _ => own,
    }
}

/// Send `path` (`total` bytes) from byte `start` to the container at `uri`.
/// `progress(sent, total, status)` returning `false` cancels (the container is kept for the next
/// run).
#[allow(clippy::too_many_arguments)]
pub fn send(
    ctx: &Ctx<'_>,
    uri: &str,
    container: &str,
    path: &Path,
    total: u64,
    start: u64,
    opts: &UploadOptions,
    progress: api::Progress<'_>,
) -> Result<Sent, IgError> {
    let mut sent = Sent::default();
    let mut offset = start.min(total);
    if offset > 0 {
        sent.resumed += 1;
    }
    let (mut in_a_row, mut failures) = (0u32, 0u32);
    let mut wait = opts.backoff;
    loop {
        if !progress(offset, total, "Sending the video to Instagram") {
            return Err(IgError::Cancelled);
        }
        let mut cancelled = false;
        let mut report = |n: u64| {
            let at = offset.saturating_add(n).min(total);
            let go = progress(at, total, "Sending the video to Instagram");
            cancelled |= !go;
            go
        };
        let req = Request {
            method: Method::Post,
            url: uri.to_string(),
            headers: vec![
                ("Authorization", format!("OAuth {}", ctx.token)),
                ("offset", offset.to_string()),
                ("file_size", total.to_string()),
                ("Content-Type", "application/octet-stream".into()),
            ],
            body: Body::File { path, offset, len: total.saturating_sub(offset), progress: &mut report },
        };
        let outcome = ctx.http.send(req);
        if cancelled {
            return Err(IgError::Cancelled);
        }
        let why = match outcome {
            Ok(r) if r.status == 200 => {
                let v: Value = serde_json::from_str(&r.body).unwrap_or(Value::Null);
                if v.get("success").and_then(Value::as_bool) == Some(true) || (v.get("debug_info").is_none() && v.get("error").is_none()) {
                    progress(total, total, "Sent; Instagram is processing the video");
                    return Ok(sent);
                }
                debug_failure(&v)?
            }
            Ok(r) => {
                let v: Value = serde_json::from_str(&r.body).unwrap_or(Value::Null);
                if v.get("debug_info").is_some() {
                    debug_failure(&v)?
                } else {
                    let e = api::GraphError::parse(&r);
                    if !(e.retryable() || matches!(r.status, 408 | 429)) {
                        return Err(match api::graph_error(&e, "the upload", ctx) {
                            IgError::Api(m) => IgError::Upload(m),
                            other => other,
                        });
                    }
                    format!("{} ({})", clean(&e.message, 120), r.status)
                }
            }
            Err(e) if e == crate::http::CANCELLED => return Err(IgError::Cancelled),
            Err(e) => e,
        };
        in_a_row += 1;
        failures += 1;
        if in_a_row > MAX_FAILURES_IN_A_ROW || failures > MAX_FAILURES {
            return Err(IgError::Upload(format!(
                "the upload kept failing ({why}); run the same publish again to carry on from where it stopped (the container is kept for 23 hours)"
            )));
        }
        let mut stop = || !progress(offset, total, &format!("The connection failed ({}); trying again", clean(&why, 80)));
        pause(wait, &mut stop)?;
        wait = (wait * 2).min(Duration::from_secs(30));
        // where did Instagram get to?
        let held = api::upload_status(ctx, container, &mut stop).ok().and_then(|s| s.bytes_received);
        match held {
            // a count past the file is not believed; one below what was sent is sent again from there
            Some(n) if n <= total => {
                if n > offset {
                    in_a_row = 0;
                }
                offset = n;
                sent.resumed += 1;
                if n >= total {
                    progress(total, total, "Sent; Instagram is processing the video");
                    return Ok(sent);
                }
            }
            _ => {
                offset = 0;
                sent.restarted += 1;
            }
        }
    }
}

/// A `debug_info` answer: retriable is a reason to try again (`Ok(why)`), anything else the error.
fn debug_failure(v: &Value) -> Result<String, IgError> {
    let d = v.get("debug_info").cloned().unwrap_or(Value::Null);
    let message = clean(d.get("message").and_then(Value::as_str).unwrap_or("no reason given"), 200);
    let kind = clean(d.get("type").and_then(Value::as_str).unwrap_or_default(), 60);
    if d.get("retriable").and_then(Value::as_bool) == Some(true) {
        return Ok(format!("{kind}: {message}"));
    }
    let lower = message.to_ascii_lowercase();
    if lower.contains("unauthorized") || lower.contains("permission") || lower.contains("oauth") {
        return Err(IgError::Upload(format!(
            "Instagram refused the upload ({kind}: {message}). Its documentation offers direct (resumable) uploads only to apps on Facebook Login for Business; with Instagram Login, publish from a public address instead: `--video-url https://…` (`videoUrl`)"
        )));
    }
    Err(IgError::Upload(format!("Instagram refused the upload ({kind}: {message})")))
}

/// A container kept across runs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Kept {
    pub container: String,
    pub uri: String,
    pub created_at: u64,
    /// Every byte was sent.
    pub uploaded: bool,
}

/// The kept containers (`<data dir>/instagram/sessions/<key>.json`).
pub struct Sessions {
    dir: PathBuf,
}

impl Sessions {
    pub fn new(dir: &Path) -> Self {
        Self { dir: dir.join("sessions") }
    }

    /// The key of one publish: the account, the file (path, size, modification time) and what
    /// the container says.
    pub fn key(user_id: &str, path: &Path, size: u64, req: &filmcraft_engine::instagram::ReelRequest) -> String {
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut h = Sha256::new();
        for part in [
            user_id.to_string(),
            path.to_string_lossy().into_owned(),
            size.to_string(),
            mtime.to_string(),
            req.caption.clone().unwrap_or_default(),
            req.cover_ms.map(|m| m.to_string()).unwrap_or_default(),
            req.cover_url.clone().unwrap_or_default(),
            req.share_to_feed.to_string(),
        ] {
            h.update(part.as_bytes());
            h.update([0u8]);
        }
        h.finalize().iter().take(16).map(|b| format!("{b:02x}")).collect()
    }

    fn file(&self, key: &str) -> Option<PathBuf> {
        (key.len() == 32 && key.bytes().all(|b| b.is_ascii_hexdigit())).then(|| self.dir.join(format!("{key}.json")))
    }

    /// The container kept under `key`, when it is young enough to trust.
    pub fn get(&self, key: &str, now: u64) -> Option<Kept> {
        let path = self.file(key)?;
        let bytes = std::fs::read(&path).ok().filter(|b| b.len() < 8192)?;
        let k: Kept = serde_json::from_slice(&bytes).ok()?;
        let fresh = k.created_at <= now && now - k.created_at < KEEP_SECS;
        let sane = !k.container.is_empty() && k.container.len() <= 32 && k.container.bytes().all(|b| b.is_ascii_digit());
        if fresh && sane {
            Some(k)
        } else {
            let _ = std::fs::remove_file(&path);
            None
        }
    }

    pub fn put(&self, key: &str, k: &Kept) {
        if let Some(path) = self.file(key) {
            // losing it only costs a new container next time
            let _ = write_private(&path, &serde_json::to_vec(k).unwrap_or_default());
        }
    }

    pub fn remove(&self, key: &str) {
        if let Some(path) = self.file(key) {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_uris_stay_on_the_rupload_host() {
        let r = "https://rupload.facebook.com/ig-api-upload";
        assert_eq!(upload_uri(r, "v25.0", "123", None), "https://rupload.facebook.com/ig-api-upload/v25.0/123");
        assert_eq!(
            upload_uri(r, "v25.0", "123", Some("https://rupload.facebook.com/ig-api-upload/v26.0/123")),
            "https://rupload.facebook.com/ig-api-upload/v26.0/123"
        );
        for hostile in [
            "https://evil.example/ig-api-upload/v25.0/123",
            "http://rupload.facebook.com/ig-api-upload/v25.0/123",
            "https://rupload.facebook.com/ig-api-upload/v25.0/999",
            "https://user@rupload.facebook.com/ig-api-upload/v25.0/123",
            "https://rupload.facebook.com/ig-api-upload/v25.0/123 x",
        ] {
            assert_eq!(upload_uri(r, "v25.0", "123", Some(hostile)), "https://rupload.facebook.com/ig-api-upload/v25.0/123", "{hostile}");
        }
    }

    #[test]
    fn kept_containers_expire_and_damage_is_dropped() {
        let d = std::env::temp_dir().join(format!("filmcraft-ig-sessions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let s = Sessions::new(&d);
        let req = filmcraft_engine::instagram::ReelRequest { caption: Some("hi".into()), share_to_feed: true, ..Default::default() };
        let key = Sessions::key("1784", Path::new("/x/a.mp4"), 10, &req);
        assert_eq!(key.len(), 32);
        let other = Sessions::key("1784", Path::new("/x/a.mp4"), 10, &filmcraft_engine::instagram::ReelRequest { caption: Some("ho".into()), ..req.clone() });
        assert_ne!(key, other, "another caption is another container");
        s.put(&key, &Kept { container: "1789".into(), uri: "u".into(), created_at: 1000, uploaded: true });
        assert_eq!(s.get(&key, 1000 + 60).unwrap().container, "1789");
        assert!(s.get(&key, 1000 + KEEP_SECS + 1).is_none(), "too old");
        assert!(s.get(&key, 1000).is_none(), "and it was removed");
        s.put(&key, &Kept { container: "../../x".into(), created_at: 5, ..Default::default() });
        assert!(s.get(&key, 6).is_none(), "a damaged container id is not trusted");
        assert!(s.get("../etc", 6).is_none() && s.file("zz").is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn debug_answers_in_words() {
        let v = serde_json::json!({"debug_info": {"retriable": true, "type": "Timeout", "message": "slow"}});
        assert!(debug_failure(&v).unwrap().contains("slow"));
        let v = serde_json::json!({"debug_info": {"retriable": false, "type": "ProcessingFailedError", "message": "unauthorized user request"}});
        let e = debug_failure(&v).unwrap_err();
        assert!(e.to_string().contains("--video-url"), "{e}");
        let v = serde_json::json!({"debug_info": {"retriable": "yes", "message": 5}});
        assert!(debug_failure(&v).is_err());
    }
}
