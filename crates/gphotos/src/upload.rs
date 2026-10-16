//! Upload a local file (typically a finished export) into the user's Google Photos library.
//!
//! The Library API still serves app uploads after the 2025-03-31 scope retirement: with the
//! append-only scope ([`crate::auth::UPLOAD_SCOPE`]) an app may add media to the library. It is
//! requested only by `photos auth --upload` (incremental consent), so picker-only users never
//! see it. The upload follows Google's resumable protocol:
//!
//! 1. `POST /v1/uploads` with `X-Goog-Upload-Protocol: resumable`, `X-Goog-Upload-Command:
//!    start`, the MIME type and the size: Google answers with the session URL
//!    (`X-Goog-Upload-URL`) and the chunk granularity (`X-Goog-Upload-Chunk-Granularity`).
//! 2. The bytes, streamed from disk in chunks ([`CHUNK`], a multiple of the granularity; at most
//!    one chunk is in memory), each at its `X-Goog-Upload-Offset`, the last one with
//!    `upload, finalize`: its body is the upload token. After a dropped chunk the upload asks
//!    where Google got to (`X-Goog-Upload-Command: query`) and resumes from there.
//! 3. `POST /v1/mediaItems:batchCreate` turns the token into a media item (and, with an
//!    `albumId`, adds it to that album: one this app created, see [`crate::albums`]).
//!
//! Every request goes through [`Transport`], so the tests drive the whole protocol with a fake.
//! Server answers are hostile input: the session URL must be an https Google API host (it gets
//! the bearer token), the granularity is capped, and reported offsets are checked against what
//! was sent. Retries are capped (consecutive and in total) and cancellation is checked between
//! chunks and while waiting to retry.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::GPhotosError;
use crate::auth::{UPLOAD_CONSENT, truncate};

/// Where an upload starts.
pub const UPLOADS_URL: &str = "https://photoslibrary.googleapis.com/v1/uploads";
/// Where uploaded bytes become media items.
pub const BATCH_CREATE_URL: &str = "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate";
/// Chunk sizes are multiples of this when Google names no granularity (its usual one).
pub const GRANULARITY: u64 = 256 << 10;
/// The chunk size: 8 MiB, the most of the file held in memory at once.
pub const CHUNK: u64 = 8 << 20;
/// The largest chunk a server-sent granularity may force; anything larger is refused.
pub const MAX_CHUNK: u64 = 64 << 20;
/// Google Photos' limits: videos under 10 GB, photos under 200 MB.
pub const MAX_VIDEO_BYTES: u64 = 10_000_000_000;
pub const MAX_PHOTO_BYTES: u64 = 200_000_000;
/// Google's limit on a media item's description, in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 1000;
/// Consecutive failed requests (dropped chunks, 5xx, 429) before the upload gives up.
pub const MAX_RETRIES: u32 = 5;
/// The longest wait between retries.
const MAX_BACKOFF: Duration = Duration::from_secs(32);
/// How often a wait between retries checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(100);
/// Caps on what Google sends back.
const MAX_TOKEN_CHARS: usize = 8192;
const MAX_URL_CHARS: usize = 4096;
const MAX_ID_CHARS: usize = 512;
/// The longest file name sent to Google.
const MAX_NAME_CHARS: usize = 255;
/// Per-request limit of the real transport: an 8 MiB chunk over a slow line still fits.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// Bodies read from Google (tokens and small JSON); anything longer is cut off.
const MAX_BODY_BYTES: u64 = 1 << 20;
/// Response headers kept (Google sends a dozen or so).
const MAX_HEADERS: usize = 256;

// ---------------------------------------------------------------------------------------------
// transport
// ---------------------------------------------------------------------------------------------

/// An HTTP method of the Library API calls (the upload protocol only POSTs; listing albums GETs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    Get,
    #[default]
    Post,
}

/// One Library API request: an upload protocol step (always a POST) or an album call.
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

/// A response: the status, the headers, and the body (capped, lossy UTF-8).
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

/// How the upload talks HTTP. `Err` is a failure with no response (connection dropped, timeout):
/// the upload retries it. HTTP error statuses come back as responses.
pub trait Transport: Send + Sync {
    fn send(&self, req: &Request<'_>) -> Result<Response, String>;
}

/// The real transport: ureq with pure-Rust TLS, https only, half an hour per request.
pub struct UreqTransport {
    agent: ureq::Agent,
}

impl Default for UreqTransport {
    fn default() -> Self {
        Self { agent: crate::api::agent_with_timeout(REQUEST_TIMEOUT) }
    }
}

impl UreqTransport {
    /// A transport with another limit per request (album calls are small).
    pub fn with_timeout(timeout: Duration) -> Self {
        Self { agent: crate::api::agent_with_timeout(timeout) }
    }
}

impl Transport for UreqTransport {
    fn send(&self, req: &Request<'_>) -> Result<Response, String> {
        let sent = match req.method {
            Method::Post => {
                let mut b = self.agent.post(req.url);
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                b.send(req.body)
            }
            Method::Get => {
                let mut b = self.agent.get(req.url);
                for (k, v) in &req.headers {
                    b = b.header(*k, v.as_str());
                }
                b.call()
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

// ---------------------------------------------------------------------------------------------
// what may be uploaded
// ---------------------------------------------------------------------------------------------

/// Photo or video, for Google's size limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Photo,
    Video,
}

/// What Google Photos takes, by extension.
const TYPES: &[(&str, &str, Kind)] = &[
    ("jpg", "image/jpeg", Kind::Photo),
    ("jpeg", "image/jpeg", Kind::Photo),
    ("png", "image/png", Kind::Photo),
    ("gif", "image/gif", Kind::Photo),
    ("webp", "image/webp", Kind::Photo),
    ("heic", "image/heic", Kind::Photo),
    ("heif", "image/heif", Kind::Photo),
    ("avif", "image/avif", Kind::Photo),
    ("bmp", "image/bmp", Kind::Photo),
    ("tif", "image/tiff", Kind::Photo),
    ("tiff", "image/tiff", Kind::Photo),
    ("ico", "image/x-icon", Kind::Photo),
    ("mp4", "video/mp4", Kind::Video),
    ("m4v", "video/x-m4v", Kind::Video),
    ("mov", "video/quicktime", Kind::Video),
    ("3gp", "video/3gpp", Kind::Video),
    ("3g2", "video/3gpp2", Kind::Video),
    ("asf", "video/x-ms-asf", Kind::Video),
    ("wmv", "video/x-ms-wmv", Kind::Video),
    ("avi", "video/x-msvideo", Kind::Video),
    ("divx", "video/divx", Kind::Video),
    ("mkv", "video/x-matroska", Kind::Video),
    ("mpg", "video/mpeg", Kind::Video),
    ("mpeg", "video/mpeg", Kind::Video),
    ("mod", "video/mpeg", Kind::Video),
    ("tod", "video/mpeg", Kind::Video),
    ("mts", "video/mp2t", Kind::Video),
    ("m2t", "video/mp2t", Kind::Video),
    ("m2ts", "video/mp2t", Kind::Video),
];

/// The MIME type and kind Google Photos takes `path` as, or why it does not.
pub fn media_type(path: &Path) -> Result<(&'static str, Kind), GPhotosError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    TYPES.iter().find(|(e, _, _)| *e == ext).map(|(_, mime, kind)| (*mime, *kind)).ok_or_else(|| {
        let what = if ext.is_empty() { "a file without an extension".to_string() } else { format!("a .{ext} file") };
        GPhotosError::Upload(format!(
            "Google Photos does not take {what}: upload a video (MP4, MOV, M4V, MKV, AVI, …) or a photo (JPEG, PNG, HEIC, WebP, GIF, TIFF, …)"
        ))
    })
}

/// A file about to be uploaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalMedia {
    pub bytes: u64,
    pub mime: &'static str,
    pub kind: Kind,
    /// The name it gets in Google Photos.
    pub file_name: String,
}

/// Check `path` before anything is sent: a regular, non-empty file of a type Google Photos
/// takes, within its size limit.
pub fn check_file(path: &Path) -> Result<LocalMedia, GPhotosError> {
    let (mime, kind) = media_type(path)?;
    let meta = std::fs::metadata(path).map_err(|e| GPhotosError::Upload(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(GPhotosError::Upload(format!("{} is not a file", path.display())));
    }
    let bytes = meta.len();
    if bytes == 0 {
        return Err(GPhotosError::Upload(format!("{} is empty", path.display())));
    }
    let (limit, what) = match kind {
        Kind::Video => (MAX_VIDEO_BYTES, "videos up to 10 GB"),
        Kind::Photo => (MAX_PHOTO_BYTES, "photos up to 200 MB"),
    };
    if bytes > limit {
        return Err(GPhotosError::Upload(format!("{} is too large ({bytes} bytes): Google Photos takes {what}", path.display())));
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(LocalMedia { bytes, mime, kind, file_name: clean_name(&name).unwrap_or_else(|| format!("FilmCraft export.{}", ext_of(path))) })
}

fn ext_of(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// A file name as Google gets it: no control characters, at most [`MAX_NAME_CHARS`].
fn clean_name(name: &str) -> Option<String> {
    let n: String = name.chars().filter(|c| !c.is_control()).take(MAX_NAME_CHARS).collect();
    let n = n.trim().to_string();
    (!n.is_empty()).then_some(n)
}

/// A description Google takes (trimmed; empty is none), or why not.
pub fn check_description(d: Option<&str>) -> Result<Option<String>, GPhotosError> {
    let Some(d) = d.map(str::trim).filter(|d| !d.is_empty()) else { return Ok(None) };
    let n = d.chars().count();
    if n > MAX_DESCRIPTION_CHARS {
        return Err(GPhotosError::Upload(format!("the description is {n} characters; Google Photos takes at most {MAX_DESCRIPTION_CHARS}")));
    }
    Ok(Some(d.to_string()))
}

// ---------------------------------------------------------------------------------------------
// the upload
// ---------------------------------------------------------------------------------------------

/// How an upload runs.
#[derive(Debug, Clone)]
pub struct UploadOptions {
    /// Shown under the item in Google Photos (at most 1000 characters).
    pub description: Option<String>,
    /// The item's file name (default: the local file's).
    pub file_name: Option<String>,
    /// Preferred chunk size; rounded down to Google's granularity (at least one granule).
    pub chunk: u64,
    /// Consecutive failed requests tolerated; four times as many in total.
    pub max_retries: u32,
    /// The first wait before a retry, doubled each time (at most 32 s).
    pub backoff: Duration,
    /// Put the new item into this album as well (`batchCreate`'s `albumId`). Google takes only
    /// an album this app created. [`crate::GooglePhotos::upload`] takes a title or an id here
    /// and resolves it (creating the album when missing); [`upload_file`] wants the id.
    pub album: Option<String>,
}

impl Default for UploadOptions {
    fn default() -> Self {
        Self { description: None, file_name: None, chunk: CHUNK, max_retries: MAX_RETRIES, backoff: Duration::from_secs(1), album: None }
    }
}

/// A finished upload: the new media item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uploaded {
    pub media_item_id: String,
    /// The item in Google Photos (only ever an `https://photos.google.com/` link).
    pub product_url: Option<String>,
    pub file_name: String,
    pub mime_type: String,
    pub bytes: u64,
    pub description: Option<String>,
    /// How many times the upload resumed after a dropped chunk.
    pub resumed: u32,
    /// The album it went into as well ([`UploadOptions::album`]), when one was asked for.
    pub album: Option<crate::albums::Album>,
    /// That album was created for this upload (it did not exist yet).
    pub album_created: bool,
}

impl Uploaded {
    pub fn to_json(&self) -> Value {
        json!({
            "mediaItemId": self.media_item_id, "productUrl": self.product_url, "fileName": self.file_name,
            "mimeType": self.mime_type, "bytes": self.bytes, "description": self.description, "resumed": self.resumed,
            "album": self.album.as_ref().map(crate::albums::Album::to_json), "albumCreated": self.album_created,
        })
    }
}

/// Upload `path` and create its media item. `bearer` gives a current access token (it is
/// asked before every request, so a long upload outlives the token's hour). `progress(sent,
/// total, status)` is called between chunks and while waiting to retry; returning `false`
/// cancels the upload ([`GPhotosError::Cancelled`]).
pub fn upload_file(
    http: &dyn Transport,
    bearer: &dyn Fn() -> Result<String, GPhotosError>,
    path: &Path,
    opts: &UploadOptions,
    progress: &mut dyn FnMut(u64, u64, &str) -> bool,
) -> Result<Uploaded, GPhotosError> {
    let media = check_file(path)?;
    let description = check_description(opts.description.as_deref())?;
    let album = opts.album.as_deref().map(crate::albums::check_id).transpose()?;
    let file_name = opts.file_name.as_deref().and_then(clean_name).unwrap_or_else(|| media.file_name.clone());
    let total = media.bytes;
    if !progress(0, total, "Starting the upload") {
        return Err(GPhotosError::Cancelled);
    }
    let mut file = std::fs::File::open(path).map_err(|e| GPhotosError::Upload(format!("{}: {e}", path.display())))?;
    let mut retry = Retry::new(opts);
    let session = start(http, bearer, &media, &mut retry, progress)?;
    let sent = send_bytes(http, bearer, &mut file, total, &session, opts.chunk, &mut retry, progress)?;
    let adding = if album.is_some() { "Adding it to your Google Photos library and the album" } else { "Adding it to your Google Photos library" };
    if !progress(total, total, adding) {
        return Err(GPhotosError::Cancelled);
    }
    let item = batch_create(http, bearer, &sent.token, &file_name, description.as_deref(), album.as_deref(), &mut retry, total, progress)?;
    Ok(Uploaded {
        media_item_id: item.id,
        product_url: item.product_url,
        file_name: if item.filename.is_empty() { file_name } else { item.filename },
        mime_type: media.mime.to_string(),
        bytes: total,
        description,
        resumed: sent.resumed,
        album: album.map(crate::albums::Album::with_id),
        album_created: false,
    })
}

/// The resumable session Google opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadSession {
    pub url: String,
    pub granularity: u64,
}

/// `X-Goog-Upload-Command: start`.
fn start(
    http: &dyn Transport,
    bearer: &dyn Fn() -> Result<String, GPhotosError>,
    media: &LocalMedia,
    retry: &mut Retry,
    progress: &mut dyn FnMut(u64, u64, &str) -> bool,
) -> Result<UploadSession, GPhotosError> {
    loop {
        let req = Request {
            method: Method::Post,
            url: UPLOADS_URL,
            headers: vec![
                ("Authorization", format!("Bearer {}", bearer()?)),
                ("X-Goog-Upload-Command", "start".into()),
                ("X-Goog-Upload-Content-Type", media.mime.into()),
                ("X-Goog-Upload-Protocol", "resumable".into()),
                ("X-Goog-Upload-Raw-Size", media.bytes.to_string()),
            ],
            body: &[],
        };
        match classify(http.send(&req)) {
            Step::Done(resp) => return parse_session(&resp),
            Step::Retry(why) => retry.pause(0, media.bytes, &why, progress)?,
            Step::Fail(e) => return Err(e),
        }
    }
}

/// The session URL and granularity from a `start` answer.
pub fn parse_session(resp: &Response) -> Result<UploadSession, GPhotosError> {
    let url = resp
        .header("X-Goog-Upload-URL")
        .filter(|u| !u.is_empty())
        .ok_or_else(|| GPhotosError::Upload("Google accepted the upload but sent no X-Goog-Upload-URL to send it to; try again".into()))?;
    check_upload_url(url)?;
    let granularity = match resp.header("X-Goog-Upload-Chunk-Granularity").map(str::parse::<u64>) {
        // absent or unreadable: Google's documented granularity
        None | Some(Err(_)) | Some(Ok(0)) => GRANULARITY,
        Some(Ok(g)) if g > MAX_CHUNK => {
            return Err(GPhotosError::Upload(format!("Google asked for upload chunks of {g} bytes; at most {MAX_CHUNK} are sent at once")));
        }
        Some(Ok(g)) => g,
    };
    Ok(UploadSession { url: url.to_string(), granularity })
}

/// The session URL gets the bearer token: only https Google API hosts are accepted.
pub fn check_upload_url(url: &str) -> Result<(), GPhotosError> {
    let bad =
        || GPhotosError::Upload(format!("Google sent an upload address that is not a Google API one ({}); not sending the file there", truncate(url, 80)));
    if url.chars().count() > MAX_URL_CHARS || url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(bad());
    }
    let rest = url.strip_prefix("https://").ok_or_else(bad)?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default().to_ascii_lowercase();
    let google = host == "photoslibrary.googleapis.com" || host.strip_suffix(".googleapis.com").is_some_and(|h| !h.is_empty() && !h.ends_with('.'));
    // no user info, no port: `https://evil.example@photoslibrary.googleapis.com` would still go to evil.example in some parsers
    if !google || host.contains(['@', ':', '\\']) {
        return Err(bad());
    }
    Ok(())
}

/// The chunk size for a granularity: `want` rounded down to a multiple of it, at least one
/// granule, at most [`MAX_CHUNK`].
pub fn chunk_size(want: u64, granularity: u64) -> u64 {
    let g = granularity.clamp(1, MAX_CHUNK);
    let n = (want.min(MAX_CHUNK) / g).max(1);
    n.saturating_mul(g).min(MAX_CHUNK).max(g)
}

/// The bytes have arrived: the upload token, and how often the upload resumed.
struct Sent {
    token: String,
    resumed: u32,
}

/// Send the file from offset 0 to the end, chunk by chunk, resuming after dropped chunks.
fn send_bytes(
    http: &dyn Transport,
    bearer: &dyn Fn() -> Result<String, GPhotosError>,
    file: &mut (impl Read + Seek),
    total: u64,
    session: &UploadSession,
    want_chunk: u64,
    retry: &mut Retry,
    progress: &mut dyn FnMut(u64, u64, &str) -> bool,
) -> Result<Sent, GPhotosError> {
    let chunk = chunk_size(want_chunk, session.granularity);
    let cap = usize::try_from(chunk.min(total)).map_err(|_| GPhotosError::Upload("the chunk does not fit in memory".into()))?;
    let mut buf = vec![0u8; cap];
    let mut offset = 0u64;
    // the furthest byte ever sent: Google cannot have received more
    let mut sent_to = 0u64;
    let mut resumed = 0u32;
    // each step sends a chunk; a failure costs its own step plus, when Google kept part of the
    // chunk, one more to realign. Failures are capped by `retry`, so this bounds the loop.
    let steps = total.div_ceil(chunk).saturating_add(u64::from(retry.total_cap()).saturating_mul(2)).saturating_add(1);
    for _ in 0..steps {
        if !progress(offset, total, "Uploading to Google Photos") {
            return Err(GPhotosError::Cancelled);
        }
        let len = chunk.min(total.saturating_sub(offset));
        let n = usize::try_from(len).map_err(|_| GPhotosError::Upload("the chunk does not fit in memory".into()))?;
        let piece = buf.get_mut(..n).ok_or_else(|| GPhotosError::Upload("the chunk does not fit in its buffer".into()))?;
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(piece))
            .map_err(|e| GPhotosError::Upload(format!("reading the file at byte {offset}: {e} (did it change during the upload?)")))?;
        let last = offset.saturating_add(len) >= total;
        let req = Request {
            method: Method::Post,
            url: &session.url,
            headers: vec![
                ("Authorization", format!("Bearer {}", bearer()?)),
                ("X-Goog-Upload-Command", if last { "upload, finalize" } else { "upload" }.into()),
                ("X-Goog-Upload-Offset", offset.to_string()),
            ],
            body: piece,
        };
        sent_to = sent_to.max(offset.saturating_add(len));
        match classify(http.send(&req)) {
            Step::Done(resp) => {
                retry.succeeded();
                if last {
                    return Ok(Sent { token: parse_token(&resp.body)?, resumed });
                }
                offset = offset.saturating_add(len);
            }
            Step::Retry(why) => {
                retry.pause(offset, total, &why, progress)?;
                match query(http, bearer, session, total, sent_to, retry, progress)? {
                    Received::Active(at) => {
                        offset = at;
                        resumed = resumed.saturating_add(1);
                    }
                    Received::Final(token) => return Ok(Sent { token, resumed: resumed.saturating_add(1) }),
                }
            }
            Step::Fail(e) => return Err(e),
        }
    }
    Err(GPhotosError::Upload(format!("the upload did not finish after {steps} requests; try again")))
}

/// Where an interrupted upload stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Received {
    /// Google has this many bytes; send the rest from there.
    Active(u64),
    /// Google has everything and finalized the upload: here is the token.
    Final(String),
}

/// `X-Goog-Upload-Command: query`.
fn query(
    http: &dyn Transport,
    bearer: &dyn Fn() -> Result<String, GPhotosError>,
    session: &UploadSession,
    total: u64,
    sent_to: u64,
    retry: &mut Retry,
    progress: &mut dyn FnMut(u64, u64, &str) -> bool,
) -> Result<Received, GPhotosError> {
    loop {
        let req = Request {
            method: Method::Post,
            url: &session.url,
            headers: vec![("Authorization", format!("Bearer {}", bearer()?)), ("X-Goog-Upload-Command", "query".into())],
            body: &[],
        };
        match classify(http.send(&req)) {
            Step::Done(resp) => return parse_query(&resp, total, sent_to),
            Step::Retry(why) => retry.pause(sent_to, total, &why, progress)?,
            Step::Fail(e) => return Err(e),
        }
    }
}

/// A `query` answer: `X-Goog-Upload-Status` and `X-Goog-Upload-Size-Received`, checked against
/// the file size and what was sent.
pub fn parse_query(resp: &Response, total: u64, sent_to: u64) -> Result<Received, GPhotosError> {
    let status = resp.header("X-Goog-Upload-Status").unwrap_or_default().to_ascii_lowercase();
    match status.as_str() {
        "final" => Ok(Received::Final(parse_token(&resp.body)?)),
        "active" => {
            let got = resp.header("X-Goog-Upload-Size-Received").unwrap_or_default();
            let at: u64 = got
                .parse()
                .map_err(|_| GPhotosError::Upload(format!("Google did not say how much of the upload arrived (`{}`); try again", truncate(got, 40))))?;
            if at > sent_to || at > total {
                return Err(GPhotosError::Upload(format!("Google reports {at} bytes received, but {sent_to} of {total} were sent; try again")));
            }
            Ok(Received::Active(at))
        }
        "cancelled" => Err(GPhotosError::Upload("Google cancelled the upload session; start the upload again".into())),
        other => Err(GPhotosError::Upload(format!("Google answered the upload query with status `{}`; try again", truncate(other, 40)))),
    }
}

/// The upload token: the whole body of the finalizing answer.
pub fn parse_token(body: &str) -> Result<String, GPhotosError> {
    let t = body.trim();
    if t.is_empty() {
        return Err(GPhotosError::Upload("Google finished the upload but sent no upload token; try again".into()));
    }
    if t.chars().count() > MAX_TOKEN_CHARS || !t.chars().all(|c| c.is_ascii_graphic()) {
        return Err(GPhotosError::Upload("Google finished the upload with an unreadable upload token; try again".into()));
    }
    Ok(t.to_string())
}

/// The created media item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatedItem {
    pub id: String,
    pub product_url: Option<String>,
    pub filename: String,
}

/// `mediaItems:batchCreate` for the one uploaded file.
fn batch_create(
    http: &dyn Transport,
    bearer: &dyn Fn() -> Result<String, GPhotosError>,
    token: &str,
    file_name: &str,
    description: Option<&str>,
    album: Option<&str>,
    retry: &mut Retry,
    total: u64,
    progress: &mut dyn FnMut(u64, u64, &str) -> bool,
) -> Result<CreatedItem, GPhotosError> {
    let body = batch_create_body(token, file_name, description, album).to_string();
    loop {
        let req = Request {
            method: Method::Post,
            url: BATCH_CREATE_URL,
            headers: vec![("Authorization", format!("Bearer {}", bearer()?)), ("Content-Type", "application/json".into())],
            body: body.as_bytes(),
        };
        match classify(http.send(&req)) {
            Step::Done(resp) => return parse_batch_create(&resp.body),
            Step::Retry(why) => retry.pause(total, total, &why, progress)?,
            Step::Fail(e) => return Err(e),
        }
    }
}

/// The `batchCreate` request body; with `album`, the item also goes into that (app-created)
/// album, at its end.
pub fn batch_create_body(token: &str, file_name: &str, description: Option<&str>, album: Option<&str>) -> Value {
    let mut item = json!({"simpleMediaItem": {"uploadToken": token, "fileName": file_name}});
    if let Some(d) = description {
        item["description"] = json!(d);
    }
    let mut body = json!({"newMediaItems": [item]});
    if let Some(a) = album {
        body["albumId"] = json!(a);
    }
    body
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct BatchJson {
    new_media_item_results: Vec<ResultJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct ResultJson {
    status: Option<StatusJson>,
    media_item: Option<ItemJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct StatusJson {
    code: i64,
    message: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct ItemJson {
    id: String,
    product_url: String,
    filename: String,
}

/// A `batchCreate` answer: the item's status (code 0 or none is success) and the new item.
pub fn parse_batch_create(text: &str) -> Result<CreatedItem, GPhotosError> {
    let b: BatchJson = serde_json::from_str(text).map_err(|e| GPhotosError::Upload(format!("Google sent an unreadable media item answer: {e}")))?;
    let r = b.new_media_item_results.into_iter().next().ok_or_else(|| GPhotosError::Upload("Google created no media item for the upload".into()))?;
    if let Some(s) = r.status.filter(|s| s.code != 0) {
        let msg = if s.message.is_empty() { format!("status {}", s.code) } else { truncate(&s.message, 300).to_string() };
        return Err(GPhotosError::Upload(format!("Google Photos did not add the file: {msg}")));
    }
    let item = r.media_item.unwrap_or_default();
    if item.id.is_empty() || item.id.chars().count() > MAX_ID_CHARS || item.id.chars().any(char::is_control) {
        return Err(GPhotosError::Upload("Google answered without a usable media item id".into()));
    }
    let product_url =
        Some(item.product_url).filter(|u| u.starts_with("https://photos.google.com/") && u.len() <= MAX_URL_CHARS && !u.contains(char::is_whitespace));
    Ok(CreatedItem { id: item.id, product_url, filename: clean_name(&item.filename).unwrap_or_default() })
}

// ---------------------------------------------------------------------------------------------
// errors and retries
// ---------------------------------------------------------------------------------------------

/// What to do with one request's outcome.
enum Step {
    Done(Response),
    Retry(String),
    Fail(GPhotosError),
}

fn classify(r: Result<Response, String>) -> Step {
    match r {
        Err(e) => Step::Retry(format!("connection: {}", truncate(&e, 200))),
        Ok(resp) if (200..300).contains(&resp.status) => Step::Done(resp),
        Ok(resp) if matches!(resp.status, 408 | 429 | 500..=599) => Step::Retry(format!("Google answered {}", resp.status)),
        Ok(resp) => Step::Fail(upload_error(resp.status, &resp.body)),
    }
}

/// A readable error for a failed upload request, with the fix for the usual causes: the
/// Photos Library API not enabled in the user's Google Cloud project, a sign-in without the
/// upload permission, a file Google refuses.
pub fn upload_error(status: u16, text: &str) -> GPhotosError {
    let err = serde_json::from_str::<Value>(text).ok().and_then(|v| v.get("error").cloned()).unwrap_or(Value::Null);
    let message = err.get("message").and_then(Value::as_str).unwrap_or_else(|| truncate(text.trim(), 300));
    let kind = err.get("status").and_then(Value::as_str).unwrap_or_default();
    let reasons: Vec<&str> =
        err.get("details").and_then(Value::as_array).into_iter().flatten().take(16).filter_map(|d| d.get("reason").and_then(Value::as_str)).collect();
    let has = |r: &str| reasons.contains(&r);
    if has("SERVICE_DISABLED") || message.contains("has not been used") || message.contains("is disabled") {
        return GPhotosError::Setup(format!(
            "the Photos Library API is not enabled in your Google Cloud project: open APIs & Services → Library, search \"Photos Library API\", press Enable, wait a few minutes and upload again (Google said: {})",
            truncate(message, 200)
        ));
    }
    if status == 401 || kind == "UNAUTHENTICATED" || has("ACCESS_TOKEN_SCOPE_INSUFFICIENT") || message.contains("insufficient authentication scopes") {
        return GPhotosError::Consent(UPLOAD_CONSENT.into());
    }
    if status == 413 {
        return GPhotosError::Upload("Google Photos refused the file as too large (videos up to 10 GB, photos up to 200 MB)".into());
    }
    if matches!(status, 404 | 410) {
        return GPhotosError::Upload(format!("Google no longer knows this upload (the session expired or was cancelled); upload again ({status})"));
    }
    let label = if kind.is_empty() { status.to_string() } else { format!("{status} {kind}") };
    GPhotosError::Upload(format!("Google answered {label}: {message}"))
}

/// The retry budget: consecutive failures up to `max_retries`, four times that in total, with
/// exponential backoff that can be cancelled.
struct Retry {
    max: u32,
    total_max: u32,
    consecutive: u32,
    total: u32,
    backoff: Duration,
}

impl Retry {
    fn new(opts: &UploadOptions) -> Self {
        let max = opts.max_retries.min(50);
        Self { max, total_max: max.saturating_mul(4), consecutive: 0, total: 0, backoff: opts.backoff.min(MAX_BACKOFF) }
    }

    fn total_cap(&self) -> u32 {
        self.total_max
    }

    fn succeeded(&mut self) {
        self.consecutive = 0;
    }

    /// Count a failure and wait before the retry; an error once the budget is spent or when the
    /// caller cancels.
    fn pause(&mut self, sent: u64, total: u64, why: &str, progress: &mut dyn FnMut(u64, u64, &str) -> bool) -> Result<(), GPhotosError> {
        if self.consecutive >= self.max || self.total >= self.total_max {
            return Err(GPhotosError::Upload(format!("the upload kept failing ({why}); gave up after {} retries", self.total)));
        }
        self.consecutive += 1;
        self.total += 1;
        let wait = self.backoff.saturating_mul(1u32 << (self.consecutive - 1).min(5)).min(MAX_BACKOFF);
        let status = format!("Retrying the upload ({why})");
        let start = std::time::Instant::now();
        loop {
            if !progress(sent, total, &status) {
                return Err(GPhotosError::Cancelled);
            }
            let left = wait.saturating_sub(start.elapsed());
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(CANCEL_POLL));
        }
    }
}

#[cfg(test)]
mod tests;
