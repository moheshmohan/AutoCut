//! The YouTube Data API calls besides the upload itself — `channels.list?mine=true`,
//! `playlists.list?mine=true`, `playlistItems.insert`, `thumbnails.set` — with the retry policy
//! they share with the upload, and Google's errors turned into messages a user can act on.
//!
//! Answers are hostile input: bodies are capped ([`crate::http::MAX_BODY_BYTES`]), ids must be
//! short and URL-safe, text is cut short and stripped of control characters, a page token must be
//! printable ASCII, and every loop is bounded.

use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::http::{Method, Request, Response, Transport};
use crate::quota::{Call, QuotaBook};
use crate::{Endpoints, YtError, clean};

/// Consecutive failed requests (dropped connections, 5xx, 429) before a call gives up.
pub const MAX_RETRIES: u32 = 5;
/// The longest wait between retries.
const MAX_BACKOFF: Duration = Duration::from_secs(32);
/// The longest `Retry-After` honoured.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
/// How often a wait checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(100);
/// Playlists read at most (pages of 50).
const MAX_PLAYLIST_PAGES: usize = 20;

/// What a call needs: the transport, where Google is, the quota book, a current access token
/// (`true` forces a refresh, after a 401), and who it is for (messages).
pub struct Ctx<'a> {
    pub http: &'a dyn Transport,
    pub endpoints: &'a Endpoints,
    pub quota: &'a QuotaBook,
    pub bearer: &'a dyn Fn(bool) -> Result<String, YtError>,
    /// "H4hacks (@h4hacks)".
    pub channel: String,
    /// The `youtube auth` that signs this channel in again.
    pub sign_in: &'static str,
    /// The first wait before a retry (doubled each time).
    pub backoff: Duration,
}

/// Progress: `(bytes sent, total, status)`; `false` cancels.
pub type Progress<'a> = &'a mut dyn FnMut(u64, u64, &str) -> bool;

// -------------------------------------------------------------------------------------------
// errors
// -------------------------------------------------------------------------------------------

/// A Google error answer, read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    /// `errors[].reason` and `details[].reason` (`quotaExceeded`, `SERVICE_DISABLED`, …).
    pub reasons: Vec<String>,
    /// `error.status` (`PERMISSION_DENIED`, …).
    pub kind: String,
    pub message: String,
}

impl ApiError {
    pub fn parse(status: u16, body: &str) -> Self {
        let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        let err = v.get("error").cloned().unwrap_or(Value::Null);
        let message = match err.get("message").and_then(Value::as_str) {
            Some(m) => clean(m, 300),
            // `{"error": "invalid_grant"}` or a page of HTML
            None => clean(err.as_str().unwrap_or(body.trim()), 200),
        };
        let mut reasons = Vec::new();
        for list in ["errors", "details"] {
            for e in err.get(list).and_then(Value::as_array).into_iter().flatten().take(16) {
                if let Some(r) = e.get("reason").and_then(Value::as_str) {
                    reasons.push(clean(r, 60));
                }
            }
        }
        Self { status, reasons, kind: clean(err.get("status").and_then(Value::as_str).unwrap_or_default(), 60), message }
    }

    pub fn has(&self, reason: &str) -> bool {
        self.reasons.iter().any(|r| r == reason)
    }

    /// Worth retrying after a pause: Google's 5xx, 429 and rate limits.
    pub fn retryable(&self) -> bool {
        matches!(self.status, 408 | 429 | 500 | 502 | 503 | 504)
            || self.has("rateLimitExceeded")
            || self.has("userRateLimitExceeded")
            || self.has("backendError")
            || self.has("uploadRateLimitExceeded")
    }
}

/// The error a user acts on for a refused call (`what`: "the upload", "the thumbnail", …).
pub fn api_error(e: &ApiError, what: &str, ctx: &Ctx<'_>) -> YtError {
    let msg = if e.message.is_empty() { format!("status {}", e.status) } else { e.message.clone() };
    if e.has("quotaExceeded") || e.has("dailyLimitExceeded") || (e.has("RATE_LIMIT_EXCEEDED") && msg.contains("quota")) {
        ctx.quota.exceeded("quotaExceeded");
        let now = crate::now();
        return YtError::Quota(format!(
            "YouTube's daily quota for your Google Cloud project is used up (10,000 units a day, and 100 uploads in their own bucket); it resets at midnight Pacific time, in {}. \
             More can be asked for in Google Cloud Console ▸ APIs & Services ▸ YouTube Data API v3 ▸ Quotas",
            crate::quota::describe_wait(crate::quota::next_reset(now).saturating_sub(now))
        ));
    }
    if e.has("uploadLimitExceeded") {
        ctx.quota.exceeded("uploadLimitExceeded");
        return YtError::Quota(format!(
            "{} has reached YouTube's limit on uploads for now (YouTube limits how many videos a channel uploads in a day); try again tomorrow, or upload to another channel",
            ctx.channel
        ));
    }
    if e.has("SERVICE_DISABLED") || e.has("accessNotConfigured") || msg.contains("has not been used in project") || msg.contains("it is disabled") {
        return YtError::Setup(format!(
            "the YouTube Data API v3 is not enabled in your Google Cloud project: open APIs & Services ▸ Library, search \"YouTube Data API v3\", press Enable, wait a few minutes and try again (Google said: {})",
            clean(&msg, 200)
        ));
    }
    if e.has("youtubeSignupRequired") {
        return YtError::Channel(
            "this Google account has no YouTube channel: create one at https://www.youtube.com/create_channel, then connect again (`filmcraft-cli youtube auth`)".into(),
        );
    }
    if e.status == 401 || e.kind == "UNAUTHENTICATED" || e.has("authError") {
        return YtError::Auth(format!(
            "Google refused the sign-in of {} ({}); sign in again: `{}` and pick {} in Google's account chooser",
            ctx.channel,
            clean(&msg, 120),
            ctx.sign_in,
            ctx.channel
        ));
    }
    if e.has("insufficientPermissions") || e.has("ACCESS_TOKEN_SCOPE_INSUFFICIENT") || msg.contains("insufficient authentication scopes") {
        return YtError::Consent(format!(
            "{}'s sign-in lacks the permission {what} needs: run `{}` and pick {} in Google's account chooser",
            ctx.channel,
            if what.contains("playlist") { crate::auth::SIGN_IN_PLAYLISTS } else { crate::auth::SIGN_IN },
            ctx.channel
        ));
    }
    for (reason, field) in [
        ("invalidTitle", "title"),
        ("invalidDescription", "description"),
        ("invalidTags", "tags"),
        ("invalidCategoryId", "category"),
        ("invalidPublishAt", "scheduled publishing time (publishAt: in the future, privacy private)"),
        ("invalidFilename", "file name"),
        ("invalidVideoMetadata", "metadata"),
        ("forbiddenPrivacySetting", "privacy setting"),
        ("forbiddenLicenseSetting", "licence"),
        ("defaultLanguageNotSet", "metadata (no default language)"),
    ] {
        if e.has(reason) {
            return YtError::Invalid(format!("YouTube refused the video's {field}: {msg}"));
        }
    }
    let label = if e.reasons.is_empty() { e.status.to_string() } else { format!("{} {}", e.status, e.reasons.first().map(String::as_str).unwrap_or_default()) };
    if e.status >= 500 {
        return YtError::Api(format!("YouTube could not take {what} right now ({label}: {msg}); try again later"));
    }
    YtError::Api(format!("YouTube refused {what} ({label}: {msg})"))
}

// -------------------------------------------------------------------------------------------
// retries
// -------------------------------------------------------------------------------------------

/// The retry budget: consecutive failures up to `max`, four times that in total, with
/// exponential backoff that can be cancelled.
pub struct Retry {
    max: u32,
    total_max: u32,
    consecutive: u32,
    total: u32,
    backoff: Duration,
}

impl Retry {
    pub fn new(max: u32, backoff: Duration) -> Self {
        let max = max.min(50);
        Self { max, total_max: max.saturating_mul(4), consecutive: 0, total: 0, backoff: backoff.min(MAX_BACKOFF) }
    }

    pub fn total_cap(&self) -> u32 {
        self.total_max
    }

    /// Failures so far.
    pub fn failures(&self) -> u32 {
        self.total
    }

    pub fn succeeded(&mut self) {
        self.consecutive = 0;
    }

    /// Count a failure and wait before the retry (`after`: the server's `Retry-After`); an error
    /// once the budget is spent or when the caller cancels.
    pub fn pause(&mut self, sent: u64, total: u64, why: &str, after: Option<Duration>, progress: Progress<'_>) -> Result<(), YtError> {
        if self.consecutive >= self.max || self.total >= self.total_max {
            return Err(YtError::Upload(format!("YouTube kept failing ({why}); gave up after {} retries — run it again to resume", self.total)));
        }
        self.consecutive = self.consecutive.saturating_add(1);
        self.total = self.total.saturating_add(1);
        let doubling = self.backoff.saturating_mul(1u32 << self.consecutive.saturating_sub(1).min(5)).min(MAX_BACKOFF);
        let wait = after.map_or(doubling, |a| a.min(MAX_RETRY_AFTER).max(doubling));
        let status = format!("Retrying ({why})");
        let start = Instant::now();
        loop {
            if !progress(sent, total, &status) {
                return Err(YtError::Cancelled);
            }
            let left = wait.saturating_sub(start.elapsed());
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(CANCEL_POLL));
        }
    }
}

/// A `Retry-After` of whole seconds.
pub fn retry_after(resp: &Response) -> Option<Duration> {
    resp.header("Retry-After")?.parse::<u64>().ok().map(Duration::from_secs)
}

/// What to do with one answer.
pub enum Step {
    Done(Response),
    /// Try again after a pause (why, the server's wait).
    Retry(String, Option<Duration>),
    /// The access token was refused: refresh it and try once more.
    Refresh(Response),
    Fail(ApiError),
}

pub fn classify(r: Result<Response, String>) -> Step {
    match r {
        Err(e) => Step::Retry(format!("connection: {}", clean(&e, 200)), None),
        Ok(resp) if (200..300).contains(&resp.status) => Step::Done(resp),
        Ok(resp) if resp.status == 401 => Step::Refresh(resp),
        Ok(resp) => {
            let e = ApiError::parse(resp.status, &resp.body);
            if e.retryable() { Step::Retry(format!("YouTube answered {}", resp.status), retry_after(&resp)) } else { Step::Fail(e) }
        }
    }
}

/// One API call with the shared policy: the bearer token (refreshed once after a 401), retries
/// with backoff, the quota counted when it succeeds, errors mapped.
pub fn call(
    ctx: &Ctx<'_>,
    method: Method,
    url: &str,
    body: &[u8],
    content_type: Option<&str>,
    what: &str,
    quota: Call,
    retry: &mut Retry,
    progress: Progress<'_>,
) -> Result<Response, YtError> {
    let mut refreshed = false;
    // bounded by the retry budget, plus the one refresh
    for _ in 0..=retry.total_cap().saturating_add(2) {
        let mut headers = vec![("Authorization", format!("Bearer {}", (ctx.bearer)(false)?))];
        if let Some(ct) = content_type {
            headers.push(("Content-Type", ct.to_string()));
        }
        let req = Request { method, url, headers, body };
        match classify(ctx.http.send(&req)) {
            Step::Done(resp) => {
                retry.succeeded();
                ctx.quota.record(quota);
                return Ok(resp);
            }
            Step::Refresh(_) if !refreshed => {
                refreshed = true;
                (ctx.bearer)(true)?;
            }
            Step::Refresh(resp) => return Err(api_error(&ApiError::parse(resp.status, &resp.body), what, ctx)),
            Step::Retry(why, after) => retry.pause(0, 0, &why, after, progress)?,
            Step::Fail(e) => return Err(api_error(&e, what, ctx)),
        }
    }
    Err(YtError::Api(format!("YouTube did not take {what} after {} tries", retry.failures())))
}

// -------------------------------------------------------------------------------------------
// channels
// -------------------------------------------------------------------------------------------

/// The channel a token acts as.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelInfo {
    pub id: String,
    pub title: String,
    pub handle: String,
    pub long_uploads: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ListJson {
    items: Vec<Value>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
}

/// A `channels.list?mine=true` answer: the channel, `None` when the account has none.
pub fn parse_channels(body: &str) -> Result<Option<ChannelInfo>, YtError> {
    let list: ListJson =
        serde_json::from_str(body).map_err(|e| YtError::Api(format!("YouTube sent an unreadable channel list: {}", clean(&e.to_string(), 120))))?;
    let Some(item) = list.items.first() else { return Ok(None) };
    let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
    if !crate::store::valid_channel_id(id) {
        return Err(YtError::Api("YouTube answered with a channel without a usable id".into()));
    }
    let s = |p: &str| item.pointer(p).and_then(Value::as_str).unwrap_or_default();
    let title = clean(s("/snippet/title"), 150);
    let handle = clean(s("/snippet/customUrl"), 100);
    let handle = if handle.is_empty() || handle.starts_with('@') { handle } else { format!("@{handle}") };
    let long = clean(s("/status/longUploadsStatus"), 20);
    Ok(Some(ChannelInfo { id: id.to_string(), title: if title.is_empty() { id.to_string() } else { title }, handle, long_uploads: long }))
}

/// The channel `ctx`'s token acts as (`channels.list?part=snippet,status&mine=true`, 1 unit).
pub fn my_channel(ctx: &Ctx<'_>, progress: Progress<'_>) -> Result<Option<ChannelInfo>, YtError> {
    let url = format!("{}/channels?part=snippet%2Cstatus&mine=true", ctx.endpoints.api);
    let mut retry = Retry::new(3, ctx.backoff);
    let resp = call(ctx, Method::Get, &url, &[], None, "the channel lookup", Call::ChannelsList, &mut retry, progress)?;
    parse_channels(&resp.body)
}

// -------------------------------------------------------------------------------------------
// playlists
// -------------------------------------------------------------------------------------------

/// A playlist of the channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Playlist {
    pub id: String,
    pub title: String,
}

/// A playlist (or video) id: 1–128 URL-safe characters.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 128 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// One `playlists.list` page: the playlists and the next page's token.
pub fn parse_playlists(body: &str) -> Result<(Vec<Playlist>, Option<String>), YtError> {
    let list: ListJson =
        serde_json::from_str(body).map_err(|e| YtError::Api(format!("YouTube sent an unreadable playlist list: {}", clean(&e.to_string(), 120))))?;
    let items = list
        .items
        .iter()
        .take(50)
        .filter_map(|i| {
            let id = i.get("id").and_then(Value::as_str).filter(|id| valid_id(id))?;
            Some(Playlist { id: id.to_string(), title: clean(i.pointer("/snippet/title").and_then(Value::as_str).unwrap_or_default(), 150) })
        })
        .collect();
    let next = list.next_page_token.filter(|t| !t.is_empty() && t.len() <= 256 && t.bytes().all(|b| b.is_ascii_graphic()));
    Ok((items, next))
}

/// The channel's playlist `reference` names (its id, or its exact title in any case), from
/// `playlists.list?mine=true` (1 unit a page of 50).
pub fn find_playlist(ctx: &Ctx<'_>, reference: &str, progress: Progress<'_>) -> Result<Playlist, YtError> {
    let r = reference.trim();
    let mut all: Vec<Playlist> = Vec::new();
    let mut token: Option<String> = None;
    let mut retry = Retry::new(3, ctx.backoff);
    for _ in 0..MAX_PLAYLIST_PAGES {
        let mut url = format!("{}/playlists?part=snippet&mine=true&maxResults=50", ctx.endpoints.api);
        if let Some(t) = &token {
            url.push_str("&pageToken=");
            url.push_str(&filmcraft_oauth::urlencode(t));
        }
        let resp = call(ctx, Method::Get, &url, &[], None, "the playlist lookup", Call::PlaylistsList, &mut retry, &mut *progress)?;
        let (page, next) = parse_playlists(&resp.body)?;
        all.extend(page);
        if let Some(p) = all.iter().find(|p| p.id == r) {
            return Ok(p.clone());
        }
        match next {
            Some(t) if Some(&t) != token.as_ref() => token = Some(t),
            _ => break,
        }
    }
    let by_title: Vec<&Playlist> = all.iter().filter(|p| p.title.trim().eq_ignore_ascii_case(r)).collect();
    match by_title.as_slice() {
        [p] => Ok((*p).clone()),
        [] => {
            let some: Vec<&str> = all.iter().take(10).map(|p| p.title.as_str()).collect();
            Err(YtError::Invalid(format!(
                "{} has no playlist `{}`{}; make it in YouTube Studio first",
                ctx.channel,
                clean(r, 60),
                if some.is_empty() { String::new() } else { format!(" (it has: {})", some.join(", ")) }
            )))
        }
        _ => Err(YtError::Invalid(format!("{} has {} playlists titled `{}`: name one by its id", ctx.channel, by_title.len(), clean(r, 60)))),
    }
}

/// Add a video to a playlist (`playlistItems.insert`, 50 units).
pub fn add_to_playlist(ctx: &Ctx<'_>, playlist: &str, video: &str, progress: Progress<'_>) -> Result<(), YtError> {
    let url = format!("{}/playlistItems?part=snippet", ctx.endpoints.api);
    let body = json!({"snippet": {"playlistId": playlist, "resourceId": {"kind": "youtube#video", "videoId": video}}}).to_string();
    let mut retry = Retry::new(3, ctx.backoff);
    call(
        ctx,
        Method::Post,
        &url,
        body.as_bytes(),
        Some("application/json; charset=UTF-8"),
        "adding to the playlist",
        Call::PlaylistItemsInsert,
        &mut retry,
        progress,
    )
    .map(drop)
}

/// Set a video's thumbnail (`thumbnails.set`, 50 units). A channel that may not use custom
/// thumbnails gets a message saying how to allow them.
pub fn set_thumbnail(ctx: &Ctx<'_>, video: &str, image: &[u8], mime: &str, progress: Progress<'_>) -> Result<(), YtError> {
    let url = format!("{}/thumbnails/set?videoId={}&uploadType=media", ctx.endpoints.upload, filmcraft_oauth::urlencode(video));
    let mut retry = Retry::new(3, ctx.backoff);
    match call(ctx, Method::Post, &url, image, Some(mime), "the thumbnail", Call::ThumbnailsSet, &mut retry, progress) {
        Ok(_) => Ok(()),
        Err(YtError::Api(m)) if m.contains("403") || m.contains("forbidden") => Err(YtError::Consent(format!(
            "{} cannot set custom thumbnails yet: YouTube turns them on once the channel is verified (https://www.youtube.com/verify, a phone number); then set the thumbnail in YouTube Studio ({m})",
            ctx.channel
        ))),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_read_from_googles_json() {
        let body = r#"{"error":{"code":403,"message":"The request cannot be completed because you have exceeded your <a>quota</a>.","errors":[{"message":"x","domain":"youtube.quota","reason":"quotaExceeded"}],"status":"PERMISSION_DENIED"}}"#;
        let e = ApiError::parse(403, body);
        assert!(e.has("quotaExceeded") && e.kind == "PERMISSION_DENIED" && !e.retryable(), "{e:?}");
        let e = ApiError::parse(
            403,
            r#"{"error":{"message":"disabled","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"SERVICE_DISABLED"}]}}"#,
        );
        assert!(e.has("SERVICE_DISABLED"));
        let e = ApiError::parse(502, "<html>\u{7}Bad gateway</html>");
        assert!(e.retryable() && !e.message.contains('\u{7}'), "{e:?}");
        assert!(ApiError::parse(403, r#"{"error":{"errors":[{"reason":"rateLimitExceeded"}]}}"#).retryable());
        let e = ApiError::parse(
            400,
            &format!(r#"{{"error":{{"message":"{}","errors":[{}]}}}}"#, "m".repeat(5000), r#"{"reason":"r"},"#.repeat(40) + r#"{"reason":"r"}"#),
        );
        assert!(e.message.chars().count() <= 301 && e.reasons.len() <= 32, "{}", e.message.len());
        for odd in ["", "null", "[]", r#"{"error":5}"#, r#"{"error":{"errors":"x","message":7}}"#] {
            let _ = ApiError::parse(400, odd);
        }
    }

    #[test]
    fn channels_and_playlists_are_read_carefully() {
        let ch = parse_channels(
            r#"{"items":[{"id":"UCabc_-1","snippet":{"title":"H4hacks\u0007","customUrl":"h4hacks"},"status":{"longUploadsStatus":"allowed"}}]}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!((ch.id.as_str(), ch.title.as_str(), ch.handle.as_str(), ch.long_uploads.as_str()), ("UCabc_-1", "H4hacks", "@h4hacks", "allowed"));
        assert_eq!(parse_channels(r#"{"items":[]}"#).unwrap(), None);
        assert_eq!(parse_channels(r#"{}"#).unwrap(), None);
        for bad in [r#"{"items":[{"id":"../x"}]}"#, r#"{"items":[{}]}"#, r#"{"items":[{"id":7}]}"#, "nope", r#"{"items":5}"#] {
            assert!(parse_channels(bad).is_err(), "{bad}");
        }
        let (p, next) =
            parse_playlists(r#"{"items":[{"id":"PLa","snippet":{"title":"Robots"}},{"id":"bad id"},{"id":"PLb"}],"nextPageToken":"CAUQAA"}"#).unwrap();
        assert_eq!(p, [Playlist { id: "PLa".into(), title: "Robots".into() }, Playlist { id: "PLb".into(), title: String::new() }]);
        assert_eq!(next.as_deref(), Some("CAUQAA"));
        assert_eq!(parse_playlists(r#"{"nextPageToken":"a b"}"#).unwrap().1, None);
        assert!(parse_playlists("[").is_err());
    }
}
