//! The OAuth 2.0 installed-app flow: browser consent, loopback redirect, token refresh, and a
//! permission-restricted token store. The Photos Picker's read-only scope is always requested;
//! the Library API's append-only scope only when the user asks to upload (`photos auth
//! --upload`), as incremental consent on top of what was granted before.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::GPhotosError;

/// The one scope FilmCraft asks for: read the items the user hands over in the Google Photos
/// picker. (Google retired the Library API's library-wide read scopes on 2025-03-31.)
pub const SCOPE: &str = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly";
/// Add media to the user's library (and read back only what this app added): what uploading
/// exports needs. Requested only by `photos auth --upload`, so picker-only users never see it.
pub const UPLOAD_SCOPE: &str = "https://www.googleapis.com/auth/photoslibrary.appendonly";
/// What to tell a user whose sign-in lacks [`UPLOAD_SCOPE`].
pub const UPLOAD_CONSENT: &str = "Google Photos needs your consent before FilmCraft can add files to your library: run `filmcraft-cli photos auth --upload` and leave the \"add to your Google Photos library\" permission ticked";
/// Read back the albums (and media items) this app created — `albums.list`, `albums.get` and
/// `mediaItems.search` take nothing else since 2025-03-31. Creating albums and adding to them
/// is [`UPLOAD_SCOPE`]'s; this one lets FilmCraft find an album it made earlier (by title),
/// so album commands are idempotent. Requested only by `photos auth --albums`, with the upload
/// scope (Google: "intended to be requested together with the photoslibrary.appendonly scope").
pub const ALBUM_SCOPE: &str = "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata";
/// What to tell a user whose sign-in lacks [`ALBUM_SCOPE`] or [`UPLOAD_SCOPE`] for an album
/// command.
pub const ALBUMS_CONSENT: &str = "Google Photos needs your consent before FilmCraft can make albums and find the ones it made: run `filmcraft-cli photos auth --albums` and leave both Google Photos permissions (add to your library; view what FilmCraft added) ticked";
/// The loopback redirect. Desktop OAuth clients accept loopback redirects on any port, so the
/// Google Cloud console needs no redirect entry.
pub const REDIRECT_URI: &str = "http://127.0.0.1:8417";
const LISTEN: &str = "127.0.0.1:8417";
const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
/// How long [`consent`] waits for the browser to come back.
const CONSENT_WAIT: Duration = Duration::from_secs(300);
/// Connections that are not the redirect (favicon, speculative preconnects) before giving up.
const MAX_STRAY_CONNECTIONS: usize = 64;

/// The `installed` object of a downloaded `client_secret.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClientSecret {
    pub client_id: String,
    pub client_secret: String,
}

impl ClientSecret {
    pub fn load(dir: &Path) -> Result<Self, GPhotosError> {
        let path = dir.join("client_secret.json");
        let text = std::fs::read_to_string(&path).map_err(|_| {
            GPhotosError::Setup(format!(
                "no {} — create a Google Cloud project, enable the Photos Picker API, add an OAuth Desktop client, and save the downloaded secret there (see crates/gphotos/README.md)",
                path.display()
            ))
        })?;
        let v: Value = serde_json::from_str(&text).map_err(|e| GPhotosError::Setup(format!("client_secret.json is not valid JSON: {e}")))?;
        let installed = v
            .get("installed")
            .or_else(|| v.get("web"))
            .ok_or_else(|| GPhotosError::Setup("client_secret.json has no `installed` object (create a Desktop client, not a Web one)".into()))?;
        serde_json::from_value(installed.clone()).map_err(|e| GPhotosError::Setup(format!("client_secret.json is missing a field: {e}")))
    }
}

/// Stored tokens; the access token lives ~1 h, the refresh token until revoked.
#[derive(Debug, Clone)]
pub struct Tokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix seconds after which the access token must be refreshed.
    pub expires_at: u64,
    /// The scopes Google granted, space-separated.
    pub scope: String,
}

/// `tokens.json` on disk, and the token endpoint's answer (`expires_in` instead of
/// `expires_at`); every field optional so an odd file gives an error, not a crash.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TokenJson {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: u64,
    expires_in: u64,
    scope: String,
}

impl Tokens {
    pub fn load(dir: &Path) -> Result<Self, GPhotosError> {
        let path = dir.join("tokens.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|_| GPhotosError::Setup(format!("not signed in — run `filmcraft-cli photos auth` first (no {})", path.display())))?;
        let t: TokenJson = serde_json::from_str(&text).map_err(|e| GPhotosError::Auth(format!("tokens.json is corrupt: {e}; delete it and re-auth")))?;
        let tokens = Self { access_token: t.access_token, refresh_token: t.refresh_token, expires_at: t.expires_at, scope: t.scope };
        if !tokens.grants(SCOPE) {
            return Err(GPhotosError::Setup(
                "the saved Google sign-in is for the retired Photos Library permission — run `filmcraft-cli photos auth` again to grant the Photos Picker one"
                    .into(),
            ));
        }
        Ok(tokens)
    }

    /// Owner-only, like the rest of the data dir's private state.
    pub fn save(&self, dir: &Path) -> Result<(), GPhotosError> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("tokens.json");
        let v = json!({"access_token": self.access_token, "refresh_token": self.refresh_token, "expires_at": self.expires_at, "scope": self.scope});
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    /// Refresh a little early so a call never starts on a dying token.
    pub fn needs_refresh(&self) -> bool {
        now().saturating_add(60) >= self.expires_at
    }

    /// Whether Google granted `scope` (consent screens let the user untick scopes).
    pub fn grants(&self, scope: &str) -> bool {
        self.scope.split_whitespace().any(|s| s == scope)
    }

    /// The upload permission ([`UPLOAD_SCOPE`]), or the consent error naming the command that
    /// grants it.
    pub fn check_upload(&self) -> Result<(), GPhotosError> {
        if self.grants(UPLOAD_SCOPE) { Ok(()) } else { Err(GPhotosError::Consent(UPLOAD_CONSENT.into())) }
    }

    /// The album permissions ([`UPLOAD_SCOPE`] to create and add, [`ALBUM_SCOPE`] to find what
    /// FilmCraft made), or the consent error naming the command that grants them.
    pub fn check_albums(&self) -> Result<(), GPhotosError> {
        if self.grants(UPLOAD_SCOPE) && self.grants(ALBUM_SCOPE) { Ok(()) } else { Err(GPhotosError::Consent(ALBUMS_CONSENT.into())) }
    }
}

/// What a consent asks Google for: the picker alone, the picker plus uploads, or both plus the
/// albums FilmCraft makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Access {
    #[default]
    Pick,
    PickAndUpload,
    /// `photos auth --albums`: uploads, and reading back the albums (and items) this app made.
    PickUploadAndAlbums,
}

impl Access {
    pub fn scopes(self) -> &'static [&'static str] {
        match self {
            Access::Pick => &[SCOPE],
            Access::PickAndUpload => &[SCOPE, UPLOAD_SCOPE],
            Access::PickUploadAndAlbums => &[SCOPE, UPLOAD_SCOPE, ALBUM_SCOPE],
        }
    }

    /// The consent URL: every scope of `self`, offline access (a refresh token), and
    /// `include_granted_scopes` so the new token keeps what an earlier consent granted.
    pub fn consent_url(self, client_id: &str, state: &str) -> String {
        format!(
            "{AUTH_URL}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&include_granted_scopes=true&prompt=consent&state={state}",
            urlencode(client_id),
            urlencode(REDIRECT_URI),
            urlencode(&self.scopes().join(" ")),
        )
    }
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Run the consent flow for `access`: open the browser, catch the redirect on the loopback port
/// (for at most five minutes), exchange the code. Returns the fresh tokens, stored. Progress goes
/// to stderr: stdout may be a protocol stream.
pub fn consent(secret: &ClientSecret, dir: &Path, access: Access) -> Result<Tokens, GPhotosError> {
    let state = random_state();
    let url = access.consent_url(&secret.client_id, &state);
    let listener =
        std::net::TcpListener::bind(LISTEN).map_err(|e| GPhotosError::Auth(format!("cannot listen on {REDIRECT_URI} (another auth running?): {e}")))?;
    listener.set_nonblocking(true).map_err(GPhotosError::from)?;
    if crate::open_in_browser(&url) {
        eprintln!("opened the browser for Google consent; waiting for the redirect to {REDIRECT_URI}…");
    } else {
        eprintln!("open this URL in a browser to sign in:\n{url}");
    }

    let deadline = Instant::now() + CONSENT_WAIT;
    let mut strays = 0usize;
    while Instant::now() < deadline && strays < MAX_STRAY_CONNECTIONS {
        match listener.accept() {
            Ok((stream, _)) => match read_redirect(stream, &state)? {
                Some(code) => {
                    let tokens = exchange_code(secret, &code, access)?;
                    tokens.save(dir)?;
                    return Ok(tokens);
                }
                None => strays += 1,
            },
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => strays += 1,
        }
    }
    Err(GPhotosError::Auth(format!("no sign-in came back within {} minutes; run `filmcraft-cli photos auth` again", CONSENT_WAIT.as_secs() / 60)))
}

/// A hard-to-guess `state` from the std hasher's per-process random keys.
fn random_state() -> String {
    use std::hash::{BuildHasher as _, Hasher as _};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(now());
    h.write_u32(std::process::id());
    let a = h.finish();
    h.write_u64(a);
    format!("{a:016x}{:016x}", h.finish())
}

/// One loopback request. `Some(code)` for the OAuth redirect, `None` for anything else (the
/// browser's favicon fetch, a speculative preconnect that never speaks), errors for a
/// declined consent or a mismatched state.
fn read_redirect(stream: TcpStream, want_state: &str) -> Result<Option<String>, GPhotosError> {
    stream.set_nonblocking(false).ok();
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let Ok(clone) = stream.try_clone() else { return Ok(None) };
    // a request line plus headers; a browser never sends more to a redirect
    let mut reader = BufReader::new(clone.take(32 * 1024));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return Ok(None);
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or_default().to_string();
    let query = path.split_once('?').map(|(_, q)| q.to_string()).unwrap_or_default();
    let params: Vec<(String, String)> = query.split('&').filter(|p| !p.is_empty()).map(urldecode_pair).collect();
    // drain headers so the browser sees the response
    let mut line = String::new();
    while reader.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" && line != "\n" {
        line.clear();
    }
    let get = |k: &str| params.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
    let respond = |status: &str, text: &str| {
        let body = format!("<html><body style=\"font-family:system-ui;padding:3em\"><h2>{text}</h2><p>You can close this tab.</p></body></html>");
        let mut w = &stream;
        // the browser may already have gone; the flow's outcome does not depend on it
        let _ = write!(w, "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    };
    if get("state").is_none() && get("code").is_none() && get("error").is_none() {
        respond("404 Not Found", "Not the sign-in redirect");
        return Ok(None);
    }
    if get("state").as_deref() != Some(want_state) {
        respond("400 Bad Request", "Stale sign-in response");
        return Err(GPhotosError::Auth("the redirect's state did not match this request; try again".into()));
    }
    if let Some(err) = get("error") {
        respond("200 OK", "Consent was declined");
        return Err(GPhotosError::Auth(format!("Google returned `{err}` (consent declined or scope not granted)")));
    }
    let Some(code) = get("code").filter(|c| !c.is_empty()) else {
        respond("400 Bad Request", "Missing code");
        return Err(GPhotosError::Auth("the redirect carried no authorization code".into()));
    };
    respond("200 OK", "Signed in to Google Photos");
    Ok(Some(code))
}

/// Exchange the authorization code for tokens (this is when the refresh token arrives).
fn exchange_code(secret: &ClientSecret, code: &str, access: Access) -> Result<Tokens, GPhotosError> {
    let body = json!({
        "code": code, "client_id": secret.client_id, "client_secret": secret.client_secret,
        "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
    });
    let tokens = token_request(&crate::api::http_agent(), &body)?;
    if tokens.refresh_token.is_none() {
        return Err(GPhotosError::Auth("Google returned no refresh token; re-run auth (prompt=consent forces one)".into()));
    }
    check_granted(&tokens, access)?;
    Ok(tokens)
}

/// Every scope `access` asked for made it into the token (consent screens let the user untick
/// them).
fn check_granted(tokens: &Tokens, access: Access) -> Result<(), GPhotosError> {
    if !tokens.grants(SCOPE) {
        return Err(GPhotosError::Auth("Google did not grant the Photos Picker permission; run auth again and leave its checkbox ticked".into()));
    }
    if access == Access::PickAndUpload && !tokens.grants(UPLOAD_SCOPE) {
        return Err(GPhotosError::Auth(
            "Google did not grant the permission to add to your Google Photos library; run `filmcraft-cli photos auth --upload` again and leave it ticked"
                .into(),
        ));
    }
    if access == Access::PickUploadAndAlbums && !(tokens.grants(UPLOAD_SCOPE) && tokens.grants(ALBUM_SCOPE)) {
        return Err(GPhotosError::Auth(
            "Google did not grant both album permissions (add to your library; view what FilmCraft added); run `filmcraft-cli photos auth --albums` again and leave them ticked"
                .into(),
        ));
    }
    Ok(())
}

/// Use the refresh token for a new access token.
pub fn refresh(agent: &ureq::Agent, secret: &ClientSecret, tokens: &mut Tokens) -> Result<(), GPhotosError> {
    let Some(refresh_token) = tokens.refresh_token.clone() else {
        return Err(GPhotosError::Auth("the access token expired and there is no refresh token; re-run `filmcraft-cli photos auth`".into()));
    };
    let body = json!({"refresh_token": refresh_token, "client_id": secret.client_id, "client_secret": secret.client_secret, "grant_type": "refresh_token"});
    let fresh = token_request(agent, &body)?;
    tokens.access_token = fresh.access_token;
    tokens.expires_at = fresh.expires_at;
    if fresh.refresh_token.is_some() {
        tokens.refresh_token = fresh.refresh_token;
    }
    if !fresh.scope.is_empty() {
        tokens.scope = fresh.scope;
    }
    Ok(())
}

fn token_request(agent: &ureq::Agent, body: &Value) -> Result<Tokens, GPhotosError> {
    let resp = agent.post(TOKEN_URL).send_json(body).map_err(|e| GPhotosError::Auth(format!("the token endpoint failed: {e}")))?;
    let status = resp.status();
    let mut text = String::new();
    resp.into_body().into_reader().take(1 << 16).read_to_string(&mut text).map_err(GPhotosError::from)?;
    if status != 200 {
        if text.contains("invalid_grant") {
            return Err(GPhotosError::Auth(
                "Google no longer accepts the saved sign-in (sign-ins of apps in Testing lapse after 7 days); run `filmcraft-cli photos auth` again".into(),
            ));
        }
        return Err(GPhotosError::Auth(format!("the token endpoint answered {status}: {}", truncate(&text, 300))));
    }
    parse_token_response(&text)
}

/// The token endpoint's JSON. An absent `scope` is read as the requested one (Google always
/// sends it; if it ever does not, the API's own 403 is the honest answer).
fn parse_token_response(text: &str) -> Result<Tokens, GPhotosError> {
    let t: TokenJson = serde_json::from_str(text).map_err(|e| GPhotosError::Auth(format!("the token endpoint sent invalid JSON: {e}")))?;
    if t.access_token.is_empty() {
        return Err(GPhotosError::Auth("the token endpoint sent no access token".into()));
    }
    let lifetime = if t.expires_in == 0 { 3600 } else { t.expires_in };
    Ok(Tokens {
        access_token: t.access_token,
        refresh_token: t.refresh_token,
        expires_at: now().saturating_add(lifetime),
        scope: if t.scope.is_empty() { SCOPE.to_string() } else { t.scope },
    })
}

/// The query-string pieces of a redirect URL, decoded.
fn urldecode_pair(p: &str) -> (String, String) {
    let (k, v) = p.split_once('=').unwrap_or((p, ""));
    (urldecode(k), urldecode(v))
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        match b {
            b'%' => {
                let hex = |b: u8| (b as char).to_digit(16);
                if let (Some(h), Some(l)) = (bytes.get(i + 1).and_then(|b| hex(*b)), bytes.get(i + 2).and_then(|b| hex(*b))) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub(crate) fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => s.get(..i).unwrap_or(s),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_query_pairs() {
        assert_eq!(urldecode_pair("code=4%2F0Ax"), ("code".into(), "4/0Ax".into()));
        assert_eq!(urldecode("a+b%20c"), "a b c");
        assert_eq!(urldecode("dangling%2"), "dangling%2");
        assert_eq!(urlencode("a b/c?"), "a%20b%2Fc%3F");
        assert_eq!(urlencode(SCOPE), "https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fphotospicker.mediaitems.readonly");
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gphotos-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tokens_round_trip_owner_only() {
        let dir = temp_dir("tokens");
        let t = Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 123, scope: SCOPE.into() };
        t.save(&dir).unwrap();
        let back = Tokens::load(&dir).unwrap();
        assert_eq!((back.access_token, back.refresh_token, back.expires_at, back.scope), ("a".into(), Some("r".into()), 123, SCOPE.to_string()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(dir.join("tokens.json")).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "owner-only");
        }
        assert!(Tokens::load(&dir.join("nope")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn library_era_tokens_ask_for_a_fresh_sign_in() {
        let dir = temp_dir("old-scope");
        // what the pre-2025 code saved: no scope field at all
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","refresh_token":"r","expires_at":1}"#).unwrap();
        let err = Tokens::load(&dir).unwrap_err().to_string();
        assert!(err.contains("photos auth"), "{err}");
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","scope":"https://www.googleapis.com/auth/photoslibrary.readonly"}"#).unwrap();
        assert!(Tokens::load(&dir).is_err());
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","expires_at":-5}"#).unwrap();
        assert!(Tokens::load(&dir).is_err(), "corrupt numbers are an error, not a crash");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn token_responses_parse_and_check_the_scope() {
        let t = parse_token_response(&format!(r#"{{"access_token":"ya29","expires_in":3599,"refresh_token":"1//r","scope":"{SCOPE}","token_type":"Bearer"}}"#))
            .unwrap();
        assert!(t.grants(SCOPE) && t.refresh_token.as_deref() == Some("1//r"));
        assert!(t.expires_at >= now() + 3500);
        let unticked = parse_token_response(r#"{"access_token":"ya29","expires_in":3599,"scope":"openid"}"#).unwrap();
        assert!(!unticked.grants(SCOPE));
        assert!(parse_token_response(r#"{"expires_in":3599}"#).is_err(), "no access token");
        assert!(parse_token_response(r#"{"access_token":"a","expires_in":18446744073709551615}"#).is_ok(), "huge lifetimes saturate");
        assert!(parse_token_response("not json").is_err());
    }

    #[test]
    fn refresh_window_is_early() {
        let t = Tokens { access_token: "a".into(), refresh_token: None, expires_at: now() + 30, scope: SCOPE.into() };
        assert!(t.needs_refresh(), "refreshes a minute ahead");
        let t = Tokens { access_token: "a".into(), refresh_token: None, expires_at: now() + 600, scope: SCOPE.into() };
        assert!(!t.needs_refresh());
    }

    #[test]
    fn loopback_ignores_strays_and_checks_state() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let send = |req: &'static str| {
            std::thread::spawn(move || {
                let mut s = TcpStream::connect(addr).unwrap();
                s.write_all(req.as_bytes()).unwrap();
                let mut out = String::new();
                let _ = s.read_to_string(&mut out);
                out
            })
        };
        let favicon = send("GET /favicon.ico HTTP/1.1\r\nHost: x\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1").unwrap(), None);
        assert!(favicon.join().unwrap().starts_with("HTTP/1.1 404"));

        let good = send("GET /?state=s1&code=4%2F0Ax&scope=x HTTP/1.1\r\nHost: x\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1").unwrap().as_deref(), Some("4/0Ax"));
        assert!(good.join().unwrap().contains("Signed in"));

        let forged = send("GET /?state=other&code=c HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1").is_err());
        forged.join().unwrap();

        let declined = send("GET /?state=s1&error=access_denied HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1").unwrap_err().to_string().contains("access_denied"));
        declined.join().unwrap();
    }

    #[test]
    fn states_differ() {
        assert_ne!(random_state(), random_state());
        assert_eq!(random_state().len(), 32);
    }

    /// The upload scope is asked for only with `--upload`, as incremental consent that keeps
    /// what was granted before.
    #[test]
    fn consent_urls_ask_for_uploads_only_when_wanted() {
        let pick = Access::Pick.consent_url("id.apps.googleusercontent.com", "st");
        assert!(pick.starts_with(AUTH_URL) && pick.contains("include_granted_scopes=true") && pick.contains("access_type=offline"), "{pick}");
        assert!(pick.contains(&format!("scope={}&", urlencode(SCOPE))), "{pick}");
        assert!(!pick.contains("photoslibrary"), "picker-only users are never asked for the Library API: {pick}");
        let up = Access::PickAndUpload.consent_url("id", "st");
        assert!(up.contains(&format!("scope={}%20{}&", urlencode(SCOPE), urlencode(UPLOAD_SCOPE))), "{up}");
        assert!(up.contains("include_granted_scopes=true") && up.ends_with("&state=st"), "{up}");
        assert_eq!(Access::default(), Access::Pick);
    }

    #[test]
    fn upload_permission_is_checked() {
        let t = |scope: &str| Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: u64::MAX, scope: scope.into() };
        let picker = t(SCOPE);
        let e = picker.check_upload().unwrap_err();
        assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("filmcraft-cli photos auth --upload"), "{e}");
        assert!(t(&format!("{SCOPE} {UPLOAD_SCOPE}")).check_upload().is_ok());
        assert!(t(&format!("{SCOPE} {UPLOAD_SCOPE}x")).check_upload().is_err(), "whole scopes only");
        // what a consent must have granted
        assert!(check_granted(&picker, Access::Pick).is_ok());
        assert!(check_granted(&picker, Access::PickAndUpload).unwrap_err().to_string().contains("--upload"));
        assert!(check_granted(&t(UPLOAD_SCOPE), Access::PickAndUpload).unwrap_err().to_string().contains("Picker"));
        assert!(check_granted(&t(&format!("{UPLOAD_SCOPE} {SCOPE}")), Access::PickAndUpload).is_ok());
        // incremental consent: the token endpoint lists every granted scope
        let both =
            parse_token_response(&format!(r#"{{"access_token":"ya29","expires_in":3599,"refresh_token":"r","scope":"{UPLOAD_SCOPE} {SCOPE}"}}"#)).unwrap();
        assert!(both.check_upload().is_ok() && both.grants(SCOPE));
    }

    /// Albums need the upload scope (create, add) and the app-created read scope (find what
    /// FilmCraft made); `--albums` asks for both on top of the picker, incrementally.
    #[test]
    fn album_permission_is_checked_and_asked_for_only_with_albums() {
        let t = |scope: &str| Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: u64::MAX, scope: scope.into() };
        for lacking in [SCOPE.to_string(), format!("{SCOPE} {UPLOAD_SCOPE}"), format!("{SCOPE} {ALBUM_SCOPE}")] {
            let e = t(&lacking).check_albums().unwrap_err();
            assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("filmcraft-cli photos auth --albums"), "{lacking}: {e}");
        }
        let all = t(&format!("{SCOPE} {UPLOAD_SCOPE} {ALBUM_SCOPE}"));
        assert!(all.check_albums().is_ok() && all.check_upload().is_ok());
        assert!(check_granted(&all, Access::PickUploadAndAlbums).is_ok());
        let e = check_granted(&t(&format!("{SCOPE} {UPLOAD_SCOPE}")), Access::PickUploadAndAlbums).unwrap_err();
        assert!(e.to_string().contains("--albums"), "{e}");
        let url = Access::PickUploadAndAlbums.consent_url("id", "st");
        assert!(url.contains(&format!("scope={}%20{}%20{}&", urlencode(SCOPE), urlencode(UPLOAD_SCOPE), urlencode(ALBUM_SCOPE))), "{url}");
        assert!(url.contains("include_granted_scopes=true"), "{url}");
        // the other consents never ask for it
        for a in [Access::Pick, Access::PickAndUpload] {
            assert!(!a.consent_url("id", "st").contains("appcreateddata"), "{a:?}");
        }
        // and it is never one of the retired library-wide scopes
        assert!(!ALBUM_SCOPE.ends_with("photoslibrary.readonly") && !ALBUM_SCOPE.ends_with("photoslibrary"));
    }
}
