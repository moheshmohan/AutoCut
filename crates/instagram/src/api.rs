//! The Instagram Graph API calls (graph.instagram.com, the Instagram API with Instagram Login):
//! `/me`, Reel containers, their `status_code`, `media_publish`, `content_publishing_limit` and
//! the published media's permalink — with the retry policy they share with the upload, and
//! Meta's errors turned into messages a user can act on.
//!
//! Answers are hostile input: bodies are capped ([`crate::http::MAX_BODY_BYTES`]), ids must be
//! short digit strings, text is cut short and stripped of control characters, usage headers are
//! parsed defensively, and every loop is bounded.

use std::time::{Duration, Instant};

use serde_json::Value;

use crate::http::{Body, Method, Request, Response, Transport};
use crate::{Endpoints, IgError, clean};

/// Consecutive failed requests (dropped connections, 5xx, transient errors) before a call gives up.
pub const MAX_RETRIES: u32 = 4;
/// The longest wait between retries.
const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// How often a wait checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// What a call needs: the transport, where Instagram is, the access token, the account (messages).
pub struct Ctx<'a> {
    pub http: &'a dyn Transport,
    pub endpoints: &'a Endpoints,
    pub token: &'a str,
    /// "@moheshmohan".
    pub account: String,
    /// The first wait before a retry (doubled each time).
    pub backoff: Duration,
}

/// Progress: `(bytes sent, total, status)`; `false` cancels.
pub type Progress<'a> = &'a mut dyn FnMut(u64, u64, &str) -> bool;

// -------------------------------------------------------------------------------------------
// errors
// -------------------------------------------------------------------------------------------

/// A Graph API error answer, read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphError {
    pub status: u16,
    pub code: i64,
    pub subcode: i64,
    /// `OAuthException`, `IGApiException`, …
    pub kind: String,
    pub message: String,
    /// What Meta suggests saying to the user, when it says.
    pub user_message: String,
    pub transient: bool,
    /// Minutes until the app or account may call again (from the usage headers).
    pub regain_minutes: Option<u64>,
}

impl GraphError {
    pub fn parse(r: &Response) -> Self {
        let v: Value = serde_json::from_str(&r.body).unwrap_or(Value::Null);
        let e = v.get("error").cloned().unwrap_or(Value::Null);
        let message = match e.get("message").and_then(Value::as_str) {
            Some(m) => clean(m, 300),
            None => clean(e.as_str().unwrap_or(r.body.trim()), 200),
        };
        let user_message = [e.get("error_user_title"), e.get("error_user_msg")]
            .iter()
            .filter_map(|x| x.and_then(Value::as_str))
            .map(|s| clean(s, 200))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(": ");
        Self {
            status: r.status,
            code: e.get("code").and_then(Value::as_i64).unwrap_or(0),
            subcode: e.get("error_subcode").and_then(Value::as_i64).unwrap_or(0),
            kind: clean(e.get("type").and_then(Value::as_str).unwrap_or_default(), 60),
            message,
            user_message,
            transient: e.get("is_transient").and_then(Value::as_bool).unwrap_or(false),
            regain_minutes: regain_minutes(r),
        }
    }

    /// Worth retrying after a pause: 5xx, and Meta's temporary errors.
    pub fn retryable(&self) -> bool {
        matches!(self.status, 500 | 502 | 503 | 504)
            || self.transient
            || matches!(self.code, 1 | 2 | -1 | -2)
            || matches!(self.subcode, 2207001 | 2207003 | 2207032 | 2207053)
    }

    /// Meta's rate limits (app, account, business use case).
    pub fn rate_limited(&self) -> bool {
        self.status == 429 || matches!(self.code, 4 | 17 | 32 | 613 | 80002 | 80008)
    }
}

/// `estimated_time_to_regain_access` (minutes) from `X-App-Usage` / `X-Business-Use-Case-Usage`,
/// when either says one (and it is sane).
pub fn regain_minutes(r: &Response) -> Option<u64> {
    let mut best: Option<u64> = None;
    for h in ["x-business-use-case-usage", "x-app-usage", "x-ad-account-usage"] {
        let Some(text) = r.header(h).filter(|t| t.len() <= 16 * 1024) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(text) else { continue };
        let mut found = Vec::new();
        collect_regain(&v, 0, &mut found);
        for m in found.into_iter().filter(|m| *m > 0 && *m <= 7 * 24 * 60) {
            best = Some(best.map_or(m, |b: u64| b.max(m)));
        }
    }
    best
}

fn collect_regain(v: &Value, depth: usize, out: &mut Vec<u64>) {
    if depth > 4 || out.len() > 64 {
        return;
    }
    match v {
        Value::Object(o) => {
            for (k, x) in o.iter().take(64) {
                if k == "estimated_time_to_regain_access" {
                    if let Some(n) = x.as_u64() {
                        out.push(n);
                    }
                } else {
                    collect_regain(x, depth + 1, out);
                }
            }
        }
        Value::Array(a) => a.iter().take(64).for_each(|x| collect_regain(x, depth + 1, out)),
        _ => {}
    }
}

/// What a subcode means, in words, with the fix (Meta's error-code reference).
pub fn subcode_words(subcode: i64) -> Option<&'static str> {
    Some(match subcode {
        2207001 | 2207032 | 2207053 => "Instagram failed to process the video (a temporary server problem); publish it again",
        2207003 => "Instagram timed out fetching the video; publish it again",
        2207004 => "the cover image is larger than 8 MB",
        2207005 => "Instagram does not take that image format (JPEG only)",
        2207006 => "Instagram could not find the media (a permission or an expired sign-in); connect again and publish again",
        2207008 | 2207020 => "the upload container expired (24 hours unpublished, or Instagram dropped it); publish again: a new one is made",
        2207009 => "the aspect ratio is outside what Instagram takes",
        2207010 => "the caption is too long (2,200 characters, 30 hashtags and 20 @mentions at most)",
        2207026 => {
            "Instagram does not take this video's format: it wants MP4 or MOV with H.264 or HEVC video and AAC audio (the pre-flight's fixed copy is one)"
        }
        2207027 => "the video is not processed yet",
        2207042 => "the account reached Instagram's limit of API-published posts for 24 hours; publish again tomorrow",
        2207050 => {
            "Instagram has restricted the account (inactive, checkpointed or restricted): open the Instagram app, complete what it asks, then publish again"
        }
        2207051 => "Instagram blocked the action as possible spam; wait a while, and if it is not spam, tell Instagram from the app",
        2207052 => "Instagram could not fetch the video from its URL: it must be a public https:// address that serves the file itself",
        2207057 => "the cover frame is past the end of the video; pick a cover time within it",
        _ => return None,
    })
}

/// The error a user acts on for a refused call (`what`: "the container", "the publish", …).
pub fn graph_error(e: &GraphError, what: &str, ctx: &Ctx<'_>) -> IgError {
    let said = if !e.user_message.is_empty() {
        e.user_message.clone()
    } else if e.message.is_empty() {
        format!("status {}", e.status)
    } else {
        e.message.clone()
    };
    if e.code == 190 || e.code == 102 || (e.status == 401 && e.code == 0) {
        return IgError::Auth(format!(
            "Instagram no longer accepts the sign-in of {} ({}); connect again: `{}` (or paste a new token: `{}`)",
            ctx.account,
            clean(&said, 160),
            crate::auth::SIGN_IN,
            crate::auth::PASTE_TOKEN
        ));
    }
    if e.code == 10 || (200..300).contains(&e.code) {
        return IgError::Consent(format!(
            "{} has not allowed FilmCraft to publish ({}): connect again and allow \"content publishing\" (instagram_business_content_publish); in development mode the account must also be an accepted Instagram Tester of the app",
            ctx.account,
            clean(&said, 160)
        ));
    }
    if e.subcode == 2207042 || e.code == 9 {
        return IgError::Limit(format!("{}: {}", ctx.account, subcode_words(2207042).unwrap_or_default()));
    }
    if e.rate_limited() {
        let when = e.regain_minutes.map(|m| format!("in about {}", crate::limits::describe_wait(m.saturating_mul(60)))).unwrap_or_else(|| "in an hour".into());
        return IgError::Limit(format!("Instagram's API rate limit was reached for {what} ({}); try again {when}", clean(&said, 120)));
    }
    if let Some(words) = subcode_words(e.subcode) {
        let err = format!("{words} (Instagram said: {})", clean(&said, 160));
        return match e.subcode {
            2207026 | 2207052 | 2207057 | 2207010 | 2207004 | 2207005 | 2207009 => IgError::Invalid(err),
            2207008 | 2207020 => IgError::Processing(err),
            2207050 | 2207051 => IgError::Api(err),
            _ => IgError::Api(err),
        };
    }
    IgError::Api(format!(
        "Instagram refused {what}: {} (code {}{})",
        clean(&said, 200),
        e.code,
        if e.subcode != 0 { format!("/{}", e.subcode) } else { String::new() }
    ))
}

// -------------------------------------------------------------------------------------------
// calls
// -------------------------------------------------------------------------------------------

/// Wait `d`, checking `cancelled` every [`CANCEL_POLL`].
pub fn pause(d: Duration, cancelled: &mut dyn FnMut() -> bool) -> Result<(), IgError> {
    let until = Instant::now() + d;
    while Instant::now() < until {
        if cancelled() {
            return Err(IgError::Cancelled);
        }
        std::thread::sleep(CANCEL_POLL.min(until.saturating_duration_since(Instant::now())));
    }
    if cancelled() { Err(IgError::Cancelled) } else { Ok(()) }
}

/// One Graph call with retries: `GET` (`form` empty) or form `POST` to `{graph}/{version}/{path}`,
/// the token as a bearer header. `Ok` is the 200 answer's JSON.
pub fn call(
    ctx: &Ctx<'_>,
    method: Method,
    path: &str,
    form: &[(&'static str, String)],
    what: &str,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Value, IgError> {
    let url = format!("{}/{}/{}", ctx.endpoints.graph, ctx.endpoints.version, path.trim_start_matches('/'));
    let mut wait = ctx.backoff;
    let mut last = String::new();
    for _ in 0..=MAX_RETRIES {
        if cancelled() {
            return Err(IgError::Cancelled);
        }
        let body = if method == Method::Post { Body::Form(form.to_vec()) } else { Body::None };
        let req = Request {
            method,
            url: url.clone(),
            headers: vec![("Authorization", format!("Bearer {}", ctx.token)), ("Accept", "application/json".into())],
            body,
        };
        match ctx.http.send(req) {
            Ok(r) if r.status == 200 => {
                return serde_json::from_str(&r.body).map_err(|_| IgError::Api(format!("Instagram's answer to {what} is not JSON: {}", clean(&r.body, 120))));
            }
            Ok(r) => {
                let e = GraphError::parse(&r);
                if !e.retryable() {
                    return Err(graph_error(&e, what, ctx));
                }
                last = format!("{} ({})", clean(&e.message, 120), e.status);
            }
            Err(e) if e == crate::http::CANCELLED => return Err(IgError::Cancelled),
            Err(e) => last = e,
        }
        pause(wait, cancelled)?;
        wait = (wait * 2).min(MAX_BACKOFF);
    }
    Err(IgError::Api(format!("Instagram could not take {what} right now ({last}); try again in a few minutes")))
}

/// A digits-only id from an answer, or an error naming what was expected.
pub fn id_of(v: &Value, key: &str, what: &str) -> Result<String, IgError> {
    let id = match v.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.as_u64().map(|n| n.to_string()).unwrap_or_default(),
        _ => String::new(),
    };
    if id.is_empty() || id.len() > 32 || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(IgError::Api(format!("Instagram's answer to {what} carries no usable `{key}`: {}", clean(&v.to_string(), 120))));
    }
    Ok(id)
}

/// Who the token belongs to (`/me`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Me {
    pub user_id: String,
    pub username: String,
    pub name: String,
    pub account_type: String,
}

pub fn me(ctx: &Ctx<'_>, cancelled: &mut dyn FnMut() -> bool) -> Result<Me, IgError> {
    let v = call(ctx, Method::Get, "me?fields=user_id,username,name,account_type", &[], "the account lookup", cancelled)?;
    // `user_id` is the professional account id content is published under; `id` the app-scoped one
    let user_id = id_of(&v, "user_id", "the account lookup").or_else(|_| id_of(&v, "id", "the account lookup"))?;
    let username = clean(v.get("username").and_then(Value::as_str).unwrap_or_default(), 30);
    if !username.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_') {
        return Err(IgError::Api("Instagram's account lookup answered an odd username".into()));
    }
    let account_type = clean(v.get("account_type").and_then(Value::as_str).unwrap_or_default(), 30);
    if account_type.eq_ignore_ascii_case("PERSONAL") {
        return Err(IgError::Consent(format!(
            "@{username} is a personal account: Instagram publishes through its API only for professional accounts — switch to a Creator or Business account in Instagram ▸ Settings ▸ Account type and tools, then connect again"
        )));
    }
    Ok(Me { user_id, username, name: clean(v.get("name").and_then(Value::as_str).unwrap_or_default(), 80), account_type })
}

/// What a container is made from.
pub enum Source<'a> {
    /// A resumable upload (the bytes follow at `rupload`).
    Upload,
    /// A public URL Instagram fetches.
    Url(&'a str),
}

/// A new Reel container: `(container id, the upload URI Instagram named, if any)`.
pub fn create_reel(
    ctx: &Ctx<'_>,
    user_id: &str,
    source: Source<'_>,
    req: &filmcraft_engine::instagram::ReelRequest,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<(String, Option<String>), IgError> {
    let mut form: Vec<(&'static str, String)> = vec![("media_type", "REELS".into())];
    match source {
        Source::Upload => form.push(("upload_type", "resumable".into())),
        Source::Url(u) => form.push(("video_url", u.to_string())),
    }
    if let Some(c) = &req.caption {
        form.push(("caption", c.clone()));
    }
    form.push(("share_to_feed", if req.share_to_feed { "true" } else { "false" }.into()));
    if let Some(u) = &req.cover_url {
        form.push(("cover_url", u.clone()));
    } else if let Some(ms) = req.cover_ms {
        form.push(("thumb_offset", ms.to_string()));
    }
    let v = call(ctx, Method::Post, &format!("{user_id}/media"), &form, "the Reel container", cancelled)?;
    let id = id_of(&v, "id", "the Reel container")?;
    let uri = v.get("uri").and_then(Value::as_str).map(str::to_string);
    Ok((id, uri))
}

/// A container's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// `IN_PROGRESS`, `FINISHED`, `PUBLISHED`, `ERROR`, `EXPIRED` (or what else Instagram said).
    pub code: String,
    /// Instagram's detail (an error subcode when `ERROR`).
    pub detail: String,
    /// `video_status.uploading_phase.bytes_transferred`, when Instagram reports it.
    pub bytes_received: Option<u64>,
}

pub fn status(ctx: &Ctx<'_>, container: &str, cancelled: &mut dyn FnMut() -> bool) -> Result<Status, IgError> {
    let v = call(ctx, Method::Get, &format!("{container}?fields=status_code,status"), &[], "the container's status", cancelled)?;
    Ok(read_status(&v))
}

/// The container's status and how many bytes Instagram holds (`video_status`, which Instagram
/// documents for Facebook Login; asked for separately so a host that does not know the field
/// still answers the status).
pub fn upload_status(ctx: &Ctx<'_>, container: &str, cancelled: &mut dyn FnMut() -> bool) -> Result<Status, IgError> {
    match call(ctx, Method::Get, &format!("{container}?fields=status_code,status,video_status"), &[], "the upload's status", cancelled) {
        Ok(v) => Ok(read_status(&v)),
        Err(IgError::Api(_)) => status(ctx, container, cancelled),
        Err(e) => Err(e),
    }
}

pub fn read_status(v: &Value) -> Status {
    let code = clean(v.get("status_code").and_then(Value::as_str).unwrap_or_default(), 30).to_ascii_uppercase();
    let detail = clean(v.get("status").and_then(Value::as_str).unwrap_or_default(), 200);
    let bytes_received = v
        .get("video_status")
        .and_then(|s| s.get("uploading_phase"))
        .and_then(|p| p.get("bytes_transferred"))
        .and_then(|b| b.as_u64().or_else(|| b.as_str()?.parse().ok()));
    Status { code, detail, bytes_received }
}

/// A subcode in a container's `status` text ("Error: 2207026", "…subcode 2207026…").
pub fn subcode_in(text: &str) -> Option<i64> {
    let digits: Vec<&str> = text.split(|c: char| !c.is_ascii_digit()).filter(|d| d.len() == 7 && d.starts_with("2207")).collect();
    digits.first().and_then(|d| d.parse().ok())
}

/// Publish a finished container; the new media id.
pub fn publish(ctx: &Ctx<'_>, user_id: &str, container: &str, cancelled: &mut dyn FnMut() -> bool) -> Result<String, IgError> {
    let v = call(ctx, Method::Post, &format!("{user_id}/media_publish"), &[("creation_id", container.to_string())], "the publish", cancelled)?;
    id_of(&v, "id", "the publish")
}

/// Instagram's count of posts in the last 24 h.
pub fn publishing_limit(ctx: &Ctx<'_>, user_id: &str, now: u64, cancelled: &mut dyn FnMut() -> bool) -> Result<crate::limits::Reported, IgError> {
    let v = call(ctx, Method::Get, &format!("{user_id}/content_publishing_limit?fields=config,quota_usage"), &[], "the publishing limit", cancelled)?;
    crate::limits::parse_reported(&v.to_string(), now)
        .ok_or_else(|| IgError::Api(format!("Instagram's publishing limit answer is not readable: {}", clean(&v.to_string(), 120))))
}

/// The published media's link (best effort).
pub fn permalink(ctx: &Ctx<'_>, media: &str, cancelled: &mut dyn FnMut() -> bool) -> Option<String> {
    let v = call(ctx, Method::Get, &format!("{media}?fields=permalink,shortcode"), &[], "the media lookup", cancelled).ok()?;
    let p = v.get("permalink").and_then(Value::as_str)?;
    (p.starts_with("https://www.instagram.com/") && p.len() <= 300 && !p.chars().any(|c| c.is_whitespace() || c.is_control())).then(|| p.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(status: u16, body: &str, headers: &[(&str, &str)]) -> Response {
        Response { status, headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: body.into() }
    }

    fn ctx<'a>(http: &'a dyn Transport, e: &'a Endpoints) -> Ctx<'a> {
        Ctx { http, endpoints: e, token: "t", account: "@moheshmohan".into(), backoff: Duration::from_millis(1) }
    }

    struct Nothing;
    impl Transport for Nothing {
        fn send(&self, _: Request<'_>) -> Result<Response, String> {
            Err("no network".into())
        }
    }

    #[test]
    fn errors_in_words() {
        let e = Endpoints::instagram();
        let c = ctx(&Nothing, &e);
        let g = |body: &str| graph_error(&GraphError::parse(&resp(400, body, &[])), "the publish", &c);
        assert!(matches!(g(r#"{"error":{"message":"Error validating access token","type":"OAuthException","code":190}}"#), IgError::Auth(_)));
        assert!(matches!(g(r#"{"error":{"message":"perm","code":10}}"#), IgError::Consent(_)));
        let lim = g(r#"{"error":{"message":"limit","code":9,"error_subcode":2207042}}"#);
        assert!(matches!(lim, IgError::Limit(_)) && lim.to_string().contains("tomorrow"), "{lim}");
        let fmt = g(
            r#"{"error":{"message":"bad","code":352,"error_subcode":2207026,"error_user_title":"Unsupported format","error_user_msg":"The video format is not supported."}}"#,
        );
        assert!(matches!(fmt, IgError::Invalid(_)) && fmt.to_string().contains("H.264") && fmt.to_string().contains("Unsupported format"), "{fmt}");
        let rate = graph_error(
            &GraphError::parse(&resp(
                400,
                r#"{"error":{"message":"Application request limit reached","code":4}}"#,
                &[("X-App-Usage", r#"{"call_count":100,"estimated_time_to_regain_access":42}"#)],
            )),
            "the container",
            &c,
        );
        assert!(matches!(rate, IgError::Limit(_)) && rate.to_string().contains("42 min"), "{rate}");
        let other = g(r#"{"error":{"message":"weird","code":12345}}"#);
        assert!(other.to_string().contains("code 12345"), "{other}");
        let html = g("<html>\u{7}bad gateway</html>");
        assert!(!html.to_string().contains('\u{7}'));
        assert!(GraphError::parse(&resp(503, "", &[])).retryable());
        assert!(GraphError::parse(&resp(400, r#"{"error":{"code":2,"is_transient":true}}"#, &[])).retryable());
        assert!(!GraphError::parse(&resp(400, r#"{"error":{"code":100}}"#, &[])).retryable());
    }

    #[test]
    fn usage_headers_are_hostile() {
        assert_eq!(regain_minutes(&resp(400, "", &[("x-app-usage", "not json")])), None);
        assert_eq!(regain_minutes(&resp(400, "", &[("x-app-usage", r#"{"estimated_time_to_regain_access":-3}"#)])), None);
        assert_eq!(regain_minutes(&resp(400, "", &[("x-app-usage", r#"{"estimated_time_to_regain_access":999999999}"#)])), None);
        let nested = r#"{"1784":[{"type":"instagram","estimated_time_to_regain_access":15},{"estimated_time_to_regain_access":30}]}"#;
        assert_eq!(regain_minutes(&resp(400, "", &[("X-Business-Use-Case-Usage", nested)])), Some(30));
        let deep = format!("{}{}{}", "[".repeat(200), "1", "]".repeat(200));
        assert_eq!(regain_minutes(&resp(400, "", &[("x-app-usage", &deep)])), None);
    }

    #[test]
    fn statuses_and_ids_are_read_defensively() {
        let s = read_status(
            &serde_json::json!({"status_code": "error", "status": "Error: 2207026\u{7}", "video_status": {"uploading_phase": {"bytes_transferred": "5000"}}}),
        );
        assert_eq!((s.code.as_str(), s.bytes_received), ("ERROR", Some(5000)));
        assert_eq!(subcode_in(&s.detail), Some(2207026));
        assert_eq!(subcode_in("no code"), None);
        assert_eq!(read_status(&Value::Null).code, "");
        assert!(id_of(&serde_json::json!({"id": "17889615691921648"}), "id", "x").is_ok());
        assert_eq!(id_of(&serde_json::json!({"id": 17889615691921648u64}), "id", "x").unwrap(), "17889615691921648");
        for bad in [
            serde_json::json!({"id": "../x"}),
            serde_json::json!({"id": ""}),
            serde_json::json!({"id": -1}),
            serde_json::json!({}),
            serde_json::json!({"id": "1".repeat(40)}),
        ] {
            assert!(id_of(&bad, "id", "x").is_err(), "{bad}");
        }
    }

    #[test]
    fn calls_give_up_after_the_retries() {
        let e = Endpoints::instagram();
        let c = ctx(&Nothing, &e);
        let err = call(&c, Method::Get, "me", &[], "the account lookup", &mut || false).unwrap_err();
        assert!(err.to_string().contains("could not take the account lookup") && err.to_string().contains("no network"), "{err}");
        let mut n = 0;
        let err = call(&c, Method::Get, "me", &[], "x", &mut || {
            n += 1;
            n > 2
        })
        .unwrap_err();
        assert_eq!(err, IgError::Cancelled);
    }
}
