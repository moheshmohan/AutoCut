//! YouTube's resumable upload (`videos.insert`, `uploadType=resumable`;
//! developers.google.com/youtube/v3/guides/using_resumable_upload_protocol):
//!
//! 1. `POST {upload}/videos?uploadType=resumable&part=snippet,status` with the video's metadata
//!    as JSON, `X-Upload-Content-Length` and `X-Upload-Content-Type`: YouTube answers with the
//!    session URI in `Location` (1 call from the day's 100-upload bucket).
//! 2. The file, streamed from disk in chunks ([`CHUNK`], a multiple of 256 KiB; one chunk in
//!    memory at a time): `PUT <session>` with `Content-Range: bytes a-b/total`. Each answer is
//!    `308 Resume Incomplete` with `Range: bytes=0-n` (what arrived) until the last, which is
//!    `200`/`201` and the video resource.
//! 3. A dropped chunk (no answer, 5xx, 429) waits (1 s, doubling, at most 32 s, or the
//!    `Retry-After`), asks where YouTube got to (`PUT` with `Content-Range: bytes */total`) and
//!    resumes from there; a `404` means the session expired, and the upload starts again from
//!    the first byte (once). At most 5 failures in a row and 20 in all.
//! 4. The session URI is kept on disk ([`Sessions`]) until the upload ends, so running the same
//!    upload again (same channel, file and metadata) after a crash or a lost connection asks
//!    YouTube where it got to and sends only the rest — without spending another upload.
//!
//! Answers are hostile: the session URI must be on YouTube's upload host (it receives the bearer
//! token), a reported range must start at 0 and never pass what was sent, and the video id must
//! be short and URL-safe.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{ApiError, Ctx, Progress, Retry, api_error, retry_after};
use crate::http::{Method, Request, Response};
use crate::media::LocalVideo;
use crate::quota::Call;
use crate::{Endpoints, YtError, clean};

/// Chunk sizes are multiples of this (YouTube's rule; the last chunk is exempt).
pub const GRANULE: u64 = 256 * 1024;
/// The chunk size: 8 MiB, the most of the file held in memory at once.
pub const CHUNK: u64 = 8 << 20;
/// The largest chunk sent.
pub const MAX_CHUNK: u64 = 64 << 20;
/// How long a kept session is trusted (YouTube's last about a week).
const SESSION_DAYS: u64 = 6;
/// The longest session URI taken.
const MAX_URL_CHARS: usize = 4096;

/// How an upload runs.
#[derive(Debug, Clone)]
pub struct UploadOptions {
    /// Preferred chunk size; rounded down to a multiple of [`GRANULE`] (at least one).
    pub chunk: u64,
    /// Consecutive failed requests tolerated; four times as many in total.
    pub max_retries: u32,
    /// The first wait before a retry, doubled each time (at most 32 s).
    pub backoff: Duration,
    /// Fresh sessions after YouTube dropped one (404).
    pub max_restarts: u32,
}

impl Default for UploadOptions {
    fn default() -> Self {
        Self { chunk: CHUNK, max_retries: crate::api::MAX_RETRIES, backoff: Duration::from_secs(1), max_restarts: 1 }
    }
}

/// The chunk size for `want`: a multiple of [`GRANULE`], at least one, at most [`MAX_CHUNK`].
pub fn chunk_size(want: u64) -> u64 {
    (want.min(MAX_CHUNK) / GRANULE).max(1).saturating_mul(GRANULE)
}

/// The video YouTube made.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Video {
    pub id: String,
    pub title: String,
    pub channel_id: String,
    /// What YouTube set (`private`, `unlisted`, `public`).
    pub privacy_status: String,
    /// `uploaded`, `processed`, …
    pub upload_status: String,
}

/// How the bytes went.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sent {
    /// Times the upload resumed after a dropped chunk.
    pub resumed: u32,
    /// Times YouTube dropped the session and the upload started again.
    pub restarted: u32,
    /// It went on from a session an earlier run left.
    pub from_earlier: bool,
}

/// A video resource (the last chunk's answer).
pub fn parse_video(body: &str) -> Result<Video, YtError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| YtError::Upload(format!("YouTube finished the upload but sent an unreadable answer: {}", clean(&e.to_string(), 120))))?;
    let s = |p: &str| v.pointer(p).and_then(Value::as_str).unwrap_or_default();
    let id = s("/id");
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err(YtError::Upload("YouTube finished the upload but sent no usable video id; look for it in YouTube Studio".into()));
    }
    let video = Video {
        id: id.to_string(),
        title: clean(s("/snippet/title"), 150),
        channel_id: clean(s("/snippet/channelId"), 64),
        privacy_status: clean(s("/status/privacyStatus"), 20),
        upload_status: clean(s("/status/uploadStatus"), 20),
    };
    match video.upload_status.as_str() {
        "rejected" => Err(YtError::Upload(format!(
            "YouTube rejected the video {} ({}); see it in YouTube Studio",
            video.id,
            clean(if s("/status/rejectionReason").is_empty() { "no reason given" } else { s("/status/rejectionReason") }, 60)
        ))),
        "failed" => Err(YtError::Upload(format!(
            "YouTube could not process the video {} ({}); see it in YouTube Studio",
            video.id,
            clean(if s("/status/failureReason").is_empty() { "no reason given" } else { s("/status/failureReason") }, 60)
        ))),
        _ => Ok(video),
    }
}

/// The session URI gets the bearer token: only YouTube's upload host (the endpoint's own origin,
/// or an https Google API host) is accepted.
pub fn check_session_url(url: &str, endpoints: &Endpoints) -> Result<(), YtError> {
    let bad = || YtError::Upload(format!("YouTube sent an upload address that is not YouTube's ({}); not sending the file there", clean(url, 80)));
    if url.is_empty() || url.chars().count() > MAX_URL_CHARS || url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(bad());
    }
    let origin = crate::http::origin(url).ok_or_else(bad)?;
    if crate::http::origin(&endpoints.upload).as_deref() == Some(origin.as_str()) {
        return Ok(());
    }
    let host = origin.strip_prefix("https://").ok_or_else(bad)?;
    let google = host.strip_suffix(".googleapis.com").is_some_and(|h| !h.is_empty() && !h.ends_with('.'));
    // no user info, no port: `https://evil.example@www.googleapis.com` goes to evil.example in some parsers
    if !google || host.contains(['@', ':', '\\']) {
        return Err(bad());
    }
    Ok(())
}

/// What arrived: `Range: bytes=0-n` is n + 1 bytes, no `Range` none. It must start at 0 and
/// not pass `sent_to` (what was ever sent) or the file.
pub fn parse_range(resp: &Response, sent_to: u64, total: u64) -> Result<u64, YtError> {
    let Some(r) = resp.header("Range") else { return Ok(0) };
    let bad = || YtError::Upload(format!("YouTube reported an unreadable upload range (`{}`); run it again to resume", clean(r, 40)));
    let span = r.strip_prefix("bytes=").ok_or_else(bad)?;
    let (a, b) = span.split_once('-').ok_or_else(bad)?;
    let (a, b): (u64, u64) = (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
    let got = b.checked_add(1).ok_or_else(bad)?;
    if a != 0 || got > sent_to || got > total {
        return Err(YtError::Upload(format!("YouTube reports {got} bytes received, but {sent_to} of {total} were sent; run it again to resume")));
    }
    Ok(got)
}

/// Where an upload stands after a status check.
enum Status {
    Received(u64),
    Done(Video),
    Expired,
}

/// Upload sessions kept between runs, by a key of (channel, file, size, time, metadata).
pub struct Sessions {
    dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Kept {
    url: String,
    total: u64,
    created: u64,
}

impl Sessions {
    pub fn new(dir: &Path) -> Self {
        Self { dir: dir.join("sessions") }
    }

    /// The key of an upload: the same channel, file (path, size, modification time) and
    /// metadata resume; anything else is a new upload.
    pub fn key(channel: &str, path: &Path, bytes: u64, body: &str) -> String {
        use sha2::Digest as _;
        let mtime =
            std::fs::metadata(path).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        let mut h = sha2::Sha256::new();
        for part in [channel, &path.to_string_lossy(), &bytes.to_string(), &mtime.to_string(), body] {
            h.update(part.as_bytes());
            h.update([0]);
        }
        h.finalize().iter().take(16).map(|b| format!("{b:02x}")).collect()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    fn get(&self, key: &str, total: u64, endpoints: &Endpoints) -> Option<String> {
        let b = std::fs::File::open(self.path(key)).ok().and_then(|f| {
            let mut b = Vec::new();
            f.take(16 * 1024).read_to_end(&mut b).ok()?;
            Some(b)
        })?;
        let k: Kept = serde_json::from_slice(&b).ok()?;
        let fresh = crate::now().saturating_sub(k.created) < SESSION_DAYS * 86_400;
        (fresh && k.total == total && check_session_url(&k.url, endpoints).is_ok()).then_some(k.url)
    }

    fn put(&self, key: &str, url: &str, total: u64) {
        let k = Kept { url: url.to_string(), total, created: crate::now() };
        if std::fs::create_dir_all(&self.dir).is_ok() {
            let _ = crate::write_private(&self.path(key), &serde_json::to_vec(&k).unwrap_or_default());
        }
    }

    fn remove(&self, key: &str) {
        let _ = std::fs::remove_file(self.path(key));
    }
}

/// The metadata of `videos.insert` as JSON.
pub fn insert_body(meta: &filmcraft_engine::youtube::VideoMeta, description: Option<&str>) -> String {
    let mut snippet = serde_json::json!({"title": meta.title, "categoryId": meta.category});
    if let Some(d) = description {
        snippet["description"] = Value::String(d.to_string());
    }
    if !meta.tags.is_empty() {
        snippet["tags"] = serde_json::json!(meta.tags);
    }
    let mut status = serde_json::json!({"privacyStatus": meta.privacy.as_str(), "selfDeclaredMadeForKids": meta.made_for_kids});
    if let Some(at) = &meta.publish_at {
        status["publishAt"] = Value::String(at.clone());
    }
    serde_json::json!({"snippet": snippet, "status": status}).to_string()
}

/// One upload.
pub struct Job<'a> {
    pub ctx: &'a Ctx<'a>,
    pub path: &'a Path,
    pub video: &'a LocalVideo,
    /// [`insert_body`].
    pub body: &'a str,
    pub notify_subscribers: bool,
    /// Where the session is kept between runs, and its key.
    pub keep: Option<(&'a Sessions, String)>,
    pub opts: &'a UploadOptions,
}

/// Upload the file and return the video YouTube made. `progress(sent, total, status)` is called
/// between chunks and while waiting to retry; returning `false` cancels (the session is kept,
/// so the same upload resumes later).
pub fn upload(job: &Job<'_>, progress: Progress<'_>) -> Result<(Video, Sent), YtError> {
    let total = job.video.bytes;
    let mut file = std::fs::File::open(job.path).map_err(|e| YtError::Invalid(format!("{}: {e}", job.path.display())))?;
    let mut retry = Retry::new(job.opts.max_retries, job.opts.backoff);
    let mut sent = Sent::default();
    let r = run(job, &mut file, total, &mut retry, &mut sent, progress);
    if let Some((s, key)) = &job.keep {
        match &r {
            // a cancelled or interrupted upload keeps its session: the next run resumes it
            Err(YtError::Cancelled) | Err(YtError::Upload(_)) => {}
            _ => s.remove(key),
        }
    }
    r.map(|v| (v, sent))
}

fn run(job: &Job<'_>, file: &mut (impl Read + Seek), total: u64, retry: &mut Retry, sent: &mut Sent, progress: Progress<'_>) -> Result<Video, YtError> {
    let ctx = job.ctx;
    let chunk = chunk_size(job.opts.chunk);
    let cap = usize::try_from(chunk.min(total)).map_err(|_| YtError::Upload("the chunk does not fit in memory".into()))?;
    let mut buf = vec![0u8; cap];
    let earlier = job.keep.as_ref().and_then(|(s, key)| s.get(key, total, ctx.endpoints));
    let mut session = match earlier {
        Some(url) => {
            sent.from_earlier = true;
            url
        }
        None => start(job, total, retry, progress)?,
    };
    let mut offset = 0u64;
    // the furthest byte ever sent: YouTube cannot have received more (an earlier run may have
    // sent anything up to the whole file)
    let mut sent_to = if sent.from_earlier { total } else { 0 };
    let mut ask = sent.from_earlier;
    let mut refreshed = false;
    let pieces = total.div_ceil(chunk);
    let steps = pieces
        .saturating_add(u64::from(retry.total_cap()).saturating_mul(3))
        .saturating_add(u64::from(job.opts.max_restarts).saturating_mul(pieces.saturating_add(2)))
        .saturating_add(4);
    for _ in 0..steps {
        if ask {
            ask = false;
            if !progress(offset, total, "Asking YouTube where the upload got to") {
                return Err(YtError::Cancelled);
            }
            match status(job, &session, total, sent_to, retry, progress)? {
                Status::Received(at) => {
                    offset = at;
                    sent_to = sent_to.max(at);
                }
                Status::Done(v) => return Ok(v),
                Status::Expired => {
                    restart(job, &mut session, total, retry, sent, progress)?;
                    (offset, sent_to) = (0, 0);
                }
            }
            continue;
        }
        if !progress(offset, total, &format!("Uploading to YouTube ({})", ctx.channel)) {
            return Err(YtError::Cancelled);
        }
        let len = chunk.min(total.saturating_sub(offset));
        if len == 0 {
            // everything arrived but no video yet: ask for it
            ask = true;
            continue;
        }
        let n = usize::try_from(len).map_err(|_| YtError::Upload("the chunk does not fit in memory".into()))?;
        let piece = buf.get_mut(..n).ok_or_else(|| YtError::Upload("the chunk does not fit in its buffer".into()))?;
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(piece))
            .map_err(|e| YtError::Upload(format!("reading the file at byte {offset}: {e} (did it change during the upload?)")))?;
        let last = offset.saturating_add(len).saturating_sub(1);
        let req = Request {
            method: Method::Put,
            url: &session,
            headers: vec![
                ("Authorization", format!("Bearer {}", (ctx.bearer)(false)?)),
                ("Content-Range", format!("bytes {offset}-{last}/{total}")),
                ("Content-Type", job.video.mime.to_string()),
            ],
            body: piece,
        };
        sent_to = sent_to.max(offset.saturating_add(len));
        match ctx.http.send(&req) {
            Ok(resp) if resp.status == 308 => {
                let at = parse_range(&resp, sent_to, total)?;
                retry.succeeded();
                refreshed = false;
                offset = at;
            }
            Ok(resp) if matches!(resp.status, 200 | 201) => return parse_video(&resp.body),
            Ok(resp) if matches!(resp.status, 404 | 410) => {
                restart(job, &mut session, total, retry, sent, progress)?;
                (offset, sent_to) = (0, 0);
            }
            Ok(resp) if resp.status == 401 && !refreshed => {
                refreshed = true;
                (ctx.bearer)(true)?;
                ask = true;
            }
            Ok(resp) => {
                let e = ApiError::parse(resp.status, &resp.body);
                if !e.retryable() {
                    return Err(api_error(&e, "the upload", ctx));
                }
                retry.pause(offset, total, &format!("YouTube answered {}", resp.status), retry_after(&resp), progress)?;
                sent.resumed = sent.resumed.saturating_add(1);
                ask = true;
            }
            Err(e) => {
                retry.pause(offset, total, &format!("connection: {}", clean(&e, 120)), None, progress)?;
                sent.resumed = sent.resumed.saturating_add(1);
                ask = true;
            }
        }
    }
    Err(YtError::Upload(format!("the upload did not finish after {steps} requests; run it again to resume")))
}

/// A new session after YouTube dropped one, at most `max_restarts` times.
fn restart(job: &Job<'_>, session: &mut String, total: u64, retry: &mut Retry, sent: &mut Sent, progress: Progress<'_>) -> Result<(), YtError> {
    if sent.restarted >= job.opts.max_restarts {
        if let Some((s, key)) = &job.keep {
            s.remove(key);
        }
        return Err(YtError::Upload("YouTube dropped the upload session again; upload the file again".into()));
    }
    sent.restarted = sent.restarted.saturating_add(1);
    if !progress(0, total, "YouTube dropped the upload session; starting again") {
        return Err(YtError::Cancelled);
    }
    *session = start(job, total, retry, progress)?;
    Ok(())
}

/// Open a session (`videos.insert`, `uploadType=resumable`) and keep it.
fn start(job: &Job<'_>, total: u64, retry: &mut Retry, progress: Progress<'_>) -> Result<String, YtError> {
    let ctx = job.ctx;
    if !progress(0, total, &format!("Starting the upload to {}", ctx.channel)) {
        return Err(YtError::Cancelled);
    }
    let mut url = format!("{}/videos?uploadType=resumable&part=snippet%2Cstatus", ctx.endpoints.upload);
    if !job.notify_subscribers {
        url.push_str("&notifySubscribers=false");
    }
    let mut refreshed = false;
    for _ in 0..=retry.total_cap().saturating_add(2) {
        let req = Request {
            method: Method::Post,
            url: &url,
            headers: vec![
                ("Authorization", format!("Bearer {}", (ctx.bearer)(false)?)),
                ("Content-Type", "application/json; charset=UTF-8".into()),
                ("X-Upload-Content-Length", total.to_string()),
                ("X-Upload-Content-Type", job.video.mime.to_string()),
            ],
            body: job.body.as_bytes(),
        };
        match ctx.http.send(&req) {
            Ok(resp) if (200..300).contains(&resp.status) => {
                ctx.quota.record(Call::VideosInsert);
                let loc = resp
                    .header("Location")
                    .filter(|l| !l.is_empty())
                    .ok_or_else(|| YtError::Upload("YouTube accepted the upload but named no address to send it to; try again".into()))?;
                check_session_url(loc, ctx.endpoints)?;
                if let Some((s, key)) = &job.keep {
                    s.put(key, loc, total);
                }
                return Ok(loc.to_string());
            }
            Ok(resp) if resp.status == 401 && !refreshed => {
                refreshed = true;
                (ctx.bearer)(true)?;
            }
            Ok(resp) => {
                let e = ApiError::parse(resp.status, &resp.body);
                if !e.retryable() {
                    return Err(api_error(&e, "the upload", ctx));
                }
                retry.pause(0, total, &format!("YouTube answered {}", resp.status), retry_after(&resp), progress)?;
            }
            Err(e) => retry.pause(0, total, &format!("connection: {}", clean(&e, 120)), None, progress)?,
        }
    }
    Err(YtError::Upload("YouTube did not open an upload session; try again".into()))
}

/// Ask where the upload got to (`Content-Range: bytes */total`).
fn status(job: &Job<'_>, session: &str, total: u64, sent_to: u64, retry: &mut Retry, progress: Progress<'_>) -> Result<Status, YtError> {
    let ctx = job.ctx;
    let mut refreshed = false;
    for _ in 0..=retry.total_cap().saturating_add(2) {
        let req = Request {
            method: Method::Put,
            url: session,
            headers: vec![("Authorization", format!("Bearer {}", (ctx.bearer)(false)?)), ("Content-Range", format!("bytes */{total}"))],
            body: &[],
        };
        match ctx.http.send(&req) {
            Ok(resp) if resp.status == 308 => {
                retry.succeeded();
                return parse_range(&resp, sent_to, total).map(Status::Received);
            }
            Ok(resp) if matches!(resp.status, 200 | 201) => return parse_video(&resp.body).map(Status::Done),
            Ok(resp) if matches!(resp.status, 404 | 410) => return Ok(Status::Expired),
            Ok(resp) if resp.status == 401 && !refreshed => {
                refreshed = true;
                (ctx.bearer)(true)?;
            }
            Ok(resp) => {
                let e = ApiError::parse(resp.status, &resp.body);
                if !e.retryable() {
                    return Err(api_error(&e, "the upload", ctx));
                }
                retry.pause(sent_to, total, &format!("YouTube answered {}", resp.status), retry_after(&resp), progress)?;
            }
            Err(e) => retry.pause(sent_to, total, &format!("connection: {}", clean(&e, 120)), None, progress)?,
        }
    }
    Err(YtError::Upload("YouTube did not say where the upload got to; run it again to resume".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(range: Option<&str>) -> Response {
        Response { status: 308, headers: range.map(|r| vec![("Range".to_string(), r.to_string())]).unwrap_or_default(), body: String::new() }
    }

    #[test]
    fn ranges_are_checked() {
        assert_eq!(parse_range(&resp(None), 10, 10).unwrap(), 0);
        assert_eq!(parse_range(&resp(Some("bytes=0-262143")), 262_144, 1_000_000).unwrap(), 262_144);
        for bad in ["bytes=1-5", "bytes=0-", "0-5", "bytes=0-99", "bytes=0-18446744073709551615", "bytes=a-b", "bytes=-5"] {
            assert!(parse_range(&resp(Some(bad)), 50, 1000).is_err(), "{bad}");
        }
    }

    #[test]
    fn session_urls_must_be_youtubes() {
        let g = Endpoints::google();
        assert!(check_session_url("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&upload_id=xyz", &g).is_ok());
        assert!(check_session_url("https://youtube.googleapis.com/upload/youtube/v3/videos?upload_id=1", &g).is_ok());
        for bad in [
            "",
            "http://www.googleapis.com/upload",
            "https://evil.example/upload",
            "https://www.googleapis.com.evil.example/x",
            "https://evil.example@www.googleapis.com/x",
            "https://www.googleapis.com:8443/x",
            "https://.googleapis.com/x",
            "https://www.googleapis.com/a b",
            "https://www.googleapis.com/\n",
        ] {
            assert!(check_session_url(bad, &g).is_err(), "{bad}");
        }
        let mock = Endpoints::at("http://127.0.0.1:5000");
        assert!(check_session_url("http://127.0.0.1:5000/upload/session/1", &mock).is_ok());
        assert!(check_session_url("http://127.0.0.1:5001/upload/session/1", &mock).is_err());
    }

    #[test]
    fn videos_are_read_carefully() {
        let v = parse_video(r#"{"id":"dQw4w9WgXcQ","snippet":{"title":"T","channelId":"UCx"},"status":{"privacyStatus":"private","uploadStatus":"uploaded"}}"#)
            .unwrap();
        assert_eq!((v.id.as_str(), v.privacy_status.as_str(), v.upload_status.as_str()), ("dQw4w9WgXcQ", "private", "uploaded"));
        assert!(parse_video(r#"{"id":"x","status":{"uploadStatus":"rejected","rejectionReason":"duplicate"}}"#).unwrap_err().to_string().contains("duplicate"));
        assert!(parse_video(r#"{"id":"x","status":{"uploadStatus":"failed"}}"#).unwrap_err().to_string().contains("could not process"));
        for bad in ["", "{}", r#"{"id":"a/b"}"#, r#"{"id":5}"#, &format!(r#"{{"id":"{}"}}"#, "a".repeat(65))] {
            assert!(parse_video(bad).is_err(), "{bad}");
        }
        assert_eq!(chunk_size(0), GRANULE);
        assert_eq!(chunk_size(CHUNK + 1), CHUNK);
        assert_eq!(chunk_size(u64::MAX), MAX_CHUNK);
    }

    #[test]
    fn the_insert_body_says_what_youtube_wants() {
        use filmcraft_engine::youtube::{Privacy, VideoMeta};
        let meta = VideoMeta {
            title: "Robots".into(),
            tags: vec!["a".into()],
            privacy: Privacy::Private,
            publish_at: Some("2026-11-01T18:00:00Z".into()),
            ..Default::default()
        };
        let v: Value = serde_json::from_str(&insert_body(&meta, Some("d"))).unwrap();
        assert_eq!(v["snippet"]["title"], "Robots");
        assert_eq!(v["snippet"]["categoryId"], "22");
        assert_eq!(v["snippet"]["description"], "d");
        assert_eq!(v["status"]["privacyStatus"], "private");
        assert_eq!(v["status"]["selfDeclaredMadeForKids"], false);
        assert_eq!(v["status"]["publishAt"], "2026-11-01T18:00:00Z");
        let v: Value = serde_json::from_str(&insert_body(&VideoMeta { title: "T".into(), ..Default::default() }, None)).unwrap();
        assert!(v["snippet"].get("description").is_none() && v["snippet"].get("tags").is_none() && v["status"].get("publishAt").is_none());
    }
}
