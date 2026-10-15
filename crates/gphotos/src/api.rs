//! The Google Photos Picker API client: picking sessions, the picked media items, and the
//! polling rules.
//!
//! The Picker is the only way left for an app to read a user's library: Google retired the
//! Library API's library-wide read scopes on 2025-03-31, and that API now answers only for
//! media the app uploaded itself. With the Picker the app opens a session, the user chooses
//! items in Google Photos' own UI, and the app may then list and download exactly those items.
//! There is no server-side search or browsing.

use std::io::Read;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{GPhotosError, GooglePhotos};

const API: &str = "https://photospicker.googleapis.com/v1";
/// Google's cap on one session's selection (`pickingConfig.maxItemCount` defaults to it).
pub const MAX_PICK: u32 = 2000;
const PAGE_SIZE: u32 = 100;
/// Enough pages for a full selection, plus slack; a misbehaving token chain stops here.
const MAX_PAGES: u32 = MAX_PICK / PAGE_SIZE + 5;
/// Used when Google sends no `pollInterval`.
pub const DEFAULT_POLL: Duration = Duration::from_secs(3);
const MIN_POLL: Duration = Duration::from_secs(1);
const MAX_POLL: Duration = Duration::from_secs(30);
/// Longest Duration we accept from Google (a week); anything longer is clamped.
const MAX_DURATION_SECS: f64 = 7.0 * 24.0 * 3600.0;
/// API answers are small JSON; anything bigger is cut off (and then fails to parse).
const MAX_JSON_BYTES: u64 = 1 << 22;

/// A picking session: the link the user opens, and Google's polling advice.
#[derive(Debug, Clone, PartialEq)]
pub struct PickingSession {
    pub id: String,
    /// The Google Photos link (always `https://`; empty if Google sent anything else).
    pub picker_uri: String,
    /// Google's recommended time between polls.
    pub poll_interval: Option<Duration>,
    /// Time left before Google stops waiting for a selection; zero means stop now.
    pub timeout_in: Option<Duration>,
    /// The user has finished picking; the items can be listed.
    pub media_items_set: bool,
    /// RFC 3339.
    pub expire_time: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct SessionJson {
    id: String,
    picker_uri: String,
    polling_config: Option<PollingJson>,
    media_items_set: bool,
    expire_time: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct PollingJson {
    poll_interval: Option<String>,
    timeout_in: Option<String>,
}

impl PickingSession {
    /// A `PickingSession` resource as Google sends it.
    pub fn parse(text: &str) -> Result<Self, GPhotosError> {
        let s: SessionJson = serde_json::from_str(text).map_err(|e| GPhotosError::Api(format!("Google sent an unreadable picking session: {e}")))?;
        if s.id.is_empty() || s.id.len() > 512 {
            return Err(GPhotosError::Api("Google sent a picking session without a usable id".into()));
        }
        let polling = s.polling_config.unwrap_or_default();
        Ok(Self {
            id: s.id,
            picker_uri: if s.picker_uri.starts_with("https://") { s.picker_uri } else { String::new() },
            poll_interval: polling.poll_interval.as_deref().and_then(parse_duration),
            timeout_in: polling.timeout_in.as_deref().and_then(parse_duration),
            media_items_set: s.media_items_set,
            expire_time: s.expire_time,
        })
    }

    /// The picker link with `/autoclose`, so Google Photos closes its tab when the user is done.
    pub fn autoclose_uri(&self) -> String {
        if self.picker_uri.is_empty() {
            return String::new();
        }
        let (base, query) = match self.picker_uri.split_once('?') {
            Some((b, q)) => (b, Some(q)),
            None => (self.picker_uri.as_str(), None),
        };
        let base = base.trim_end_matches('/');
        let base = if base.ends_with("/autoclose") { base.to_string() } else { format!("{base}/autoclose") };
        match query {
            Some(q) => format!("{base}?{q}"),
            None => base,
        }
    }
}

/// A protobuf JSON `Duration` (`"3.5s"`, `"0s"`, `"1800s"`). Negative, non-finite or
/// malformed values are `None`; huge ones are clamped to a week.
pub fn parse_duration(s: &str) -> Option<Duration> {
    let secs: f64 = s.trim().strip_suffix('s')?.trim().parse().ok()?;
    if secs.is_nan() || secs < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(secs.min(MAX_DURATION_SECS)).ok()
}

/// What to do after a `sessions.get`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poll {
    /// The user has finished; list the items.
    Ready,
    /// Ask again after this long.
    Wait(Duration),
    /// Google stopped waiting: the session timed out without a selection.
    Expired,
    /// Our own wait ran out; the session is still alive and can be resumed.
    OutOfTime,
}

/// The polling rule: follow Google's interval (clamped to 1–30 s), never sleep past Google's
/// timeout or our own budget.
pub fn next_poll(s: &PickingSession, waited: Duration, budget: Duration) -> Poll {
    if s.media_items_set {
        return Poll::Ready;
    }
    if s.timeout_in.is_some_and(|t| t.is_zero()) {
        return Poll::Expired;
    }
    let Some(left) = budget.checked_sub(waited).filter(|d| !d.is_zero()) else {
        return Poll::OutOfTime;
    };
    let mut wait = s.poll_interval.unwrap_or(DEFAULT_POLL).clamp(MIN_POLL, MAX_POLL).min(left);
    if let Some(t) = s.timeout_in {
        wait = wait.min(t);
    }
    Poll::Wait(wait)
}

/// Photo or video, from the item's `type` (or its MIME type when that is unspecified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Photo,
    Video,
    Unknown,
}

impl MediaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MediaKind::Photo => "photo",
            MediaKind::Video => "video",
            MediaKind::Unknown => "unknown",
        }
    }
}

/// One item the user picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickedItem {
    pub id: String,
    pub filename: String,
    pub mime_type: String,
    pub kind: MediaKind,
    /// RFC 3339 creation time.
    pub created: String,
    /// Valid for ~60 minutes and only with the bearer token; never shown to users.
    pub base_url: String,
    /// A video's `processingStatus` (`READY` once downloadable).
    pub video_status: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct ItemsPageJson {
    media_items: Option<Vec<ItemJson>>,
    next_page_token: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct ItemJson {
    id: String,
    create_time: String,
    #[serde(rename = "type")]
    kind: String,
    media_file: Option<FileJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct FileJson {
    base_url: String,
    mime_type: String,
    filename: String,
    media_file_metadata: Option<MetadataJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct MetadataJson {
    video_metadata: Option<VideoJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct VideoJson {
    processing_status: Option<String>,
}

/// One page of `mediaItems.list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemsPage {
    pub items: Vec<PickedItem>,
    pub next_page_token: Option<String>,
}

/// A `mediaItems.list` response.
pub fn parse_items_page(text: &str) -> Result<ItemsPage, GPhotosError> {
    let page: ItemsPageJson = serde_json::from_str(text).map_err(|e| GPhotosError::Api(format!("Google sent an unreadable media item list: {e}")))?;
    let items = page
        .media_items
        .unwrap_or_default()
        .into_iter()
        .map(|m| {
            let file = m.media_file.unwrap_or_default();
            let kind = match m.kind.as_str() {
                "PHOTO" => MediaKind::Photo,
                "VIDEO" => MediaKind::Video,
                _ if file.mime_type.starts_with("video/") => MediaKind::Video,
                _ if file.mime_type.starts_with("image/") => MediaKind::Photo,
                _ => MediaKind::Unknown,
            };
            PickedItem {
                id: m.id,
                filename: file.filename,
                mime_type: file.mime_type,
                kind,
                created: m.create_time,
                base_url: file.base_url,
                video_status: file.media_file_metadata.and_then(|md| md.video_metadata).and_then(|v| v.processing_status),
            }
        })
        .collect();
    Ok(ItemsPage { items, next_page_token: page.next_page_token.filter(|t| !t.is_empty()) })
}

impl PickedItem {
    /// What the CLI and MCP report per item (no `baseUrl`: it is short-lived and needs the token).
    pub fn summary(&self) -> Value {
        json!({"id": self.id, "filename": self.filename, "mimeType": self.mime_type, "kind": self.kind.as_str(), "created": self.created})
    }

    /// A label for messages.
    pub fn display_name(&self) -> &str {
        if self.filename.is_empty() { &self.id } else { &self.filename }
    }
}

impl GooglePhotos {
    /// `sessions.create`: a new picking session, at most `max_items` (1–2000) selectable.
    pub fn create_session(&self, max_items: Option<u32>) -> Result<PickingSession, GPhotosError> {
        let body = match max_items {
            Some(n) => json!({"pickingConfig": {"maxItemCount": n.clamp(1, MAX_PICK).to_string()}}),
            None => json!({}),
        };
        let text = self.call("POST", &format!("{API}/sessions"), Some(&body))?;
        let s = PickingSession::parse(&text)?;
        if s.picker_uri.is_empty() {
            return Err(GPhotosError::Api("Google sent a picking session without an https picker link".into()));
        }
        Ok(s)
    }

    /// `sessions.get`: the session's state and polling advice.
    pub fn session(&self, id: &str) -> Result<PickingSession, GPhotosError> {
        PickingSession::parse(&self.call("GET", &format!("{API}/sessions/{}", crate::auth::urlencode(id)), None)?)
    }

    /// `sessions.delete`: Google limits open sessions, so every finished one is deleted.
    pub fn delete_session(&self, id: &str) -> Result<(), GPhotosError> {
        self.call("DELETE", &format!("{API}/sessions/{}", crate::auth::urlencode(id)), None).map(drop)
    }

    /// `mediaItems.list`, every page: the items the user picked in `session_id`.
    pub fn picked_items(&self, session_id: &str) -> Result<Vec<PickedItem>, GPhotosError> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let mut url = format!("{API}/mediaItems?sessionId={}&pageSize={PAGE_SIZE}", crate::auth::urlencode(session_id));
            if let Some(t) = &token {
                url.push_str(&format!("&pageToken={}", crate::auth::urlencode(t)));
            }
            let page = parse_items_page(&self.call("GET", &url, None)?)?;
            out.extend(page.items);
            token = page.next_page_token;
            if token.is_none() || out.len() >= MAX_PICK as usize {
                break;
            }
        }
        out.truncate(MAX_PICK as usize);
        Ok(out)
    }

    /// One authorized API call; the body text on 200, Google's error message otherwise.
    fn call(&self, method: &str, url: &str, body: Option<&Value>) -> Result<String, GPhotosError> {
        let auth = format!("Bearer {}", self.access_token()?);
        let agent = self.agent();
        let sent = match (method, body) {
            ("POST", Some(b)) => agent.post(url).header("Authorization", &auth).send_json(b),
            ("POST", None) => agent.post(url).header("Authorization", &auth).send_empty(),
            ("DELETE", _) => agent.delete(url).header("Authorization", &auth).call(),
            _ => agent.get(url).header("Authorization", &auth).call(),
        };
        let resp = sent.map_err(|e| GPhotosError::Api(format!("cannot reach Google Photos: {e}")))?;
        let status = resp.status().as_u16();
        let mut text = String::new();
        resp.into_body().into_reader().take(MAX_JSON_BYTES).read_to_string(&mut text).map_err(|e| GPhotosError::Api(e.to_string()))?;
        if status != 200 {
            return Err(GPhotosError::Api(api_error(status, &text)));
        }
        Ok(text)
    }
}

/// A readable message from Google's error JSON (`{"error": {"code", "message", "status"}}`),
/// with the fix for the usual setup mistakes.
pub fn api_error(status: u16, text: &str) -> String {
    let err = serde_json::from_str::<Value>(text).ok().and_then(|v| v.get("error").cloned()).unwrap_or(Value::Null);
    let message = err.get("message").and_then(Value::as_str).unwrap_or_else(|| crate::auth::truncate(text, 300));
    let kind = err.get("status").and_then(Value::as_str).unwrap_or_default();
    let hint = if message.contains("has not been used") || message.contains("is disabled") || kind == "SERVICE_DISABLED" {
        " — enable the Photos Picker API in the Google Cloud project"
    } else if status == 401 || kind == "UNAUTHENTICATED" || message.contains("insufficient authentication scopes") {
        " — run `filmcraft-cli photos auth` again"
    } else {
        ""
    };
    if kind.is_empty() { format!("Google answered {status}: {message}{hint}") } else { format!("Google answered {status} {kind}: {message}{hint}") }
}

/// The shared HTTP agent (pure-Rust TLS), as the other crates configure ureq. HTTP errors come
/// back as responses so Google's error text reaches the user.
pub fn http_agent() -> ureq::Agent {
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
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(json: &str) -> PickingSession {
        PickingSession::parse(json).unwrap()
    }

    #[test]
    fn sessions_parse_from_google_json() {
        let s = session(
            r#"{"id":"0a1b-c2d3","pickerUri":"https://photos.google.com/picker/AbC","pollingConfig":{"pollInterval":"5s","timeoutIn":"1799.5s"},
                "expireTime":"2026-10-15T20:00:00.000Z","mediaItemsSet":false,"pickingConfig":{"maxItemCount":"50"}}"#,
        );
        assert_eq!(s.id, "0a1b-c2d3");
        assert_eq!(s.poll_interval, Some(Duration::from_secs(5)));
        assert_eq!(s.timeout_in, Some(Duration::from_millis(1_799_500)));
        assert!(!s.media_items_set);
        assert_eq!(s.expire_time.as_deref(), Some("2026-10-15T20:00:00.000Z"));
        assert_eq!(s.autoclose_uri(), "https://photos.google.com/picker/AbC/autoclose");

        // once picked, Google drops the polling config
        let done = session(r#"{"id":"x","pickerUri":"https://photos.google.com/picker/AbC/","mediaItemsSet":true}"#);
        assert!(done.media_items_set && done.poll_interval.is_none() && done.timeout_in.is_none());
        assert_eq!(done.autoclose_uri(), "https://photos.google.com/picker/AbC/autoclose");
        assert_eq!(session(r#"{"id":"x","pickerUri":"https://p.example/a/autoclose?x=1"}"#).autoclose_uri(), "https://p.example/a/autoclose?x=1");
        assert_eq!(session(r#"{"id":"x","pickerUri":"https://p.example/a?hl=en"}"#).autoclose_uri(), "https://p.example/a/autoclose?hl=en");
    }

    #[test]
    fn hostile_sessions_are_errors_or_harmless() {
        assert!(PickingSession::parse("{}").is_err(), "no id");
        assert!(PickingSession::parse(r#"{"id":""}"#).is_err());
        assert!(PickingSession::parse(&format!(r#"{{"id":"{}"}}"#, "x".repeat(600))).is_err());
        assert!(PickingSession::parse(r#"{"id":7}"#).is_err());
        assert!(PickingSession::parse("[").is_err());
        // a non-https picker link is dropped, never handed to the OS opener
        for uri in ["-a Calculator", "file:///etc/passwd", "javascript:alert(1)", "http://photos.google.com/x"] {
            let s = session(&format!(r#"{{"id":"x","pickerUri":"{uri}"}}"#));
            assert_eq!((s.picker_uri.as_str(), s.autoclose_uri().as_str()), ("", ""), "{uri}");
        }
        let s = session(r#"{"id":"x","pollingConfig":{"pollInterval":"-3s","timeoutIn":"soon"}}"#);
        assert_eq!((s.poll_interval, s.timeout_in), (None, None));
        // truncations of a real answer never panic
        let full =
            r#"{"id":"abc","pickerUri":"https://photos.google.com/picker/x","pollingConfig":{"pollInterval":"3s","timeoutIn":"1800s"},"mediaItemsSet":false}"#;
        for n in 0..full.len() {
            let _ = PickingSession::parse(full.get(..n).unwrap_or_default());
        }
    }

    #[test]
    fn durations_follow_the_protobuf_json_form() {
        assert_eq!(parse_duration("3.5s"), Some(Duration::from_millis(3500)));
        assert_eq!(parse_duration("0s"), Some(Duration::ZERO));
        assert_eq!(parse_duration(" 1800s "), Some(Duration::from_secs(1800)));
        assert_eq!(parse_duration("0.000340012s"), Some(Duration::from_nanos(340_012)));
        assert_eq!(parse_duration("1e99s"), Some(Duration::from_secs(7 * 24 * 3600)), "clamped");
        assert_eq!(parse_duration("infs"), Some(Duration::from_secs(7 * 24 * 3600)));
        for bad in ["", "s", "3", "3m", "-1s", "NaNs", "abc", "--3s"] {
            assert_eq!(parse_duration(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn polling_follows_google_within_our_budget() {
        let mut s = session(r#"{"id":"x","pollingConfig":{"pollInterval":"5s","timeoutIn":"600s"}}"#);
        let min = Duration::from_secs(60);
        assert_eq!(next_poll(&s, Duration::ZERO, 10 * min), Poll::Wait(Duration::from_secs(5)));
        assert_eq!(next_poll(&s, Duration::from_secs(598), Duration::from_secs(600)), Poll::Wait(Duration::from_secs(2)), "never past our budget");
        assert_eq!(next_poll(&s, Duration::from_secs(600), Duration::from_secs(600)), Poll::OutOfTime);
        assert_eq!(next_poll(&s, Duration::ZERO, Duration::ZERO), Poll::OutOfTime, "a zero budget returns at once");

        s.poll_interval = Some(Duration::from_millis(10));
        assert_eq!(next_poll(&s, Duration::ZERO, min), Poll::Wait(Duration::from_secs(1)), "at least a second");
        s.poll_interval = Some(Duration::from_secs(3600));
        assert_eq!(next_poll(&s, Duration::ZERO, 10 * min), Poll::Wait(Duration::from_secs(30)), "at most 30 s");
        s.poll_interval = None;
        assert_eq!(next_poll(&s, Duration::ZERO, min), Poll::Wait(DEFAULT_POLL));

        s.timeout_in = Some(Duration::from_millis(400));
        assert_eq!(next_poll(&s, Duration::ZERO, min), Poll::Wait(Duration::from_millis(400)), "never past Google's timeout");
        s.timeout_in = Some(Duration::ZERO);
        assert_eq!(next_poll(&s, Duration::ZERO, min), Poll::Expired);
        s.media_items_set = true;
        assert_eq!(next_poll(&s, Duration::from_secs(9999), Duration::ZERO), Poll::Ready, "a finished pick wins");
    }

    #[test]
    fn media_item_pages_parse() {
        let page = parse_items_page(
            r#"{"mediaItems":[
                {"id":"AF1Qip-photo","createTime":"2024-07-01T18:30:00Z","type":"PHOTO",
                 "mediaFile":{"baseUrl":"https://lh3.googleusercontent.com/ppa/abc","mimeType":"image/jpeg","filename":"IMG_0001.JPG",
                   "mediaFileMetadata":{"width":4032,"height":3024,"cameraMake":"Apple","photoMetadata":{"focalLength":4.2,"isoEquivalent":50,"exposureTime":"0.008s"}}}},
                {"id":"AF1Qip-video","createTime":"2024-07-01T18:31:00Z","type":"VIDEO",
                 "mediaFile":{"baseUrl":"https://lh3.googleusercontent.com/ppa/def","mimeType":"video/mp4","filename":"PXL_20240701.mp4",
                   "mediaFileMetadata":{"width":1920,"height":1080,"videoMetadata":{"fps":29.97,"processingStatus":"READY"}}}}
              ],"nextPageToken":"page2"}"#,
        )
        .unwrap();
        assert_eq!(page.next_page_token.as_deref(), Some("page2"));
        let [photo, video] = page.items.as_slice() else { panic!("two items: {page:?}") };
        assert_eq!((photo.kind, photo.filename.as_str(), photo.mime_type.as_str()), (MediaKind::Photo, "IMG_0001.JPG", "image/jpeg"));
        assert_eq!(photo.base_url, "https://lh3.googleusercontent.com/ppa/abc");
        assert_eq!(photo.video_status, None);
        assert_eq!((video.kind, video.video_status.as_deref()), (MediaKind::Video, Some("READY")));
        assert_eq!(
            video.summary(),
            json!({"id":"AF1Qip-video","filename":"PXL_20240701.mp4","mimeType":"video/mp4","kind":"video","created":"2024-07-01T18:31:00Z"})
        );
        assert!(video.summary().get("baseUrl").is_none(), "short-lived URLs stay internal");

        let last =
            parse_items_page(r#"{"mediaItems":[{"id":"x","type":"TYPE_UNSPECIFIED","mediaFile":{"mimeType":"video/quicktime"}}],"nextPageToken":""}"#).unwrap();
        assert_eq!(last.next_page_token, None, "an empty token ends the listing");
        assert_eq!(last.items.first().map(|i| (i.kind, i.display_name())), Some((MediaKind::Video, "x")));
        assert_eq!(parse_items_page("{}").unwrap(), ItemsPage { items: vec![], next_page_token: None });
        assert!(parse_items_page(r#"{"mediaItems":{}}"#).is_err());
        assert!(parse_items_page(r#"{"mediaItems":[{"id":3}]}"#).is_err());
    }

    #[test]
    fn google_errors_read_well() {
        let denied = r#"{"error":{"code":403,"message":"Request had insufficient authentication scopes.","status":"PERMISSION_DENIED"}}"#;
        let msg = api_error(403, denied);
        assert!(msg.contains("PERMISSION_DENIED") && msg.contains("photos auth"), "{msg}");
        let disabled =
            r#"{"error":{"code":403,"message":"Photos Picker API has not been used in project 1 before or it is disabled.","status":"PERMISSION_DENIED"}}"#;
        assert!(api_error(403, disabled).contains("enable the Photos Picker API"));
        assert_eq!(api_error(502, "<html>bad gateway</html>"), "Google answered 502: <html>bad gateway</html>");
        assert!(api_error(400, &"é".repeat(1000)).chars().count() < 400, "long bodies are cut at a char boundary");
    }
}
