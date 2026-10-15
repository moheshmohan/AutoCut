//! The OAuth 2.0 installed-app flow: browser consent, loopback redirect, token refresh, and a
//! permission-restricted token store. Only the read-only Photos Library scope is requested.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::GPhotosError;

pub const SCOPE: &str = "https://www.googleapis.com/auth/photoslibrary.readonly";
/// The loopback redirect; the same URI must be on the OAuth client in Google Cloud Console.
pub const REDIRECT_URI: &str = "http://localhost:8417";
const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

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
                "no {} — create a Google Cloud project, enable the Photos Library API, add an OAuth Desktop client with redirect URI {REDIRECT_URI}, and save the downloaded secret there (see crates/gphotos/README.md)",
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

/// Stored tokens; the access token lives ~1 h, the refresh token indefinitely.
#[derive(Debug, Clone)]
pub struct Tokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix seconds after which the access token must be refreshed.
    pub expires_at: u64,
}

impl Tokens {
    pub fn load(dir: &Path) -> Result<Self, GPhotosError> {
        let path = dir.join("tokens.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|_| GPhotosError::Setup(format!("not signed in — run `filmcraft-cli photos auth` first (no {})", path.display())))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| GPhotosError::Auth(format!("tokens.json is corrupt: {e}; delete it and re-auth")))?;
        Ok(Self {
            access_token: v["access_token"].as_str().unwrap_or_default().to_string(),
            refresh_token: v["refresh_token"].as_str().map(str::to_string),
            expires_at: v["expires_at"].as_u64().unwrap_or(0),
        })
    }

    /// Owner-only, like the rest of the data dir's private state.
    pub fn save(&self, dir: &Path) -> Result<(), GPhotosError> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("tokens.json");
        let v = json!({"access_token": self.access_token, "refresh_token": self.refresh_token, "expires_at": self.expires_at});
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
        now() + 60 >= self.expires_at
    }
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Run the consent flow: open the browser, catch the redirect on the loopback port, exchange
/// the code. Returns the fresh tokens, stored.
pub fn consent(secret: &ClientSecret, dir: &Path) -> Result<Tokens, GPhotosError> {
    let state = format!("{:016x}", std::process::id() as u64 ^ (now() << 17));
    let url = format!(
        "{AUTH_URL}?client_id={}&redirect_uri={}&response_type=code&scope={SCOPE}&access_type=offline&prompt=consent&state={state}",
        urlencode(&secret.client_id),
        urlencode(REDIRECT_URI),
    );
    println!("opening the browser for Google consent (redirect {REDIRECT_URI})…");
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&url).status();
    }
    #[cfg(not(target_os = "macos"))]
    {
        println!("open this URL in a browser:\n{url}");
    }

    let listener = std::net::TcpListener::bind("127.0.0.1:8417")
        .map_err(|e| GPhotosError::Auth(format!("cannot listen on {REDIRECT_URI} (another auth running?): {e}")))?;
    let (stream, _) = listener.accept().map_err(|e| GPhotosError::Auth(format!("waiting for the redirect failed: {e}")))?;
    let code = read_redirect(stream, &state)?;
    let tokens = exchange_code(secret, &code)?;
    tokens.save(dir)?;
    Ok(tokens)
}

/// One redirect request: verify the state, pull `code` (or the error), answer with a page the
/// user can close.
fn read_redirect(stream: std::net::TcpStream, want_state: &str) -> Result<String, GPhotosError> {
    stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
    let mut reader = BufReader::new(stream.try_clone().map_err(GPhotosError::from)?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(GPhotosError::from)?;
    let path = request_line.split_whitespace().nth(1).unwrap_or_default().to_string();
    let query = path.split_once('?').map(|(_, q)| q.to_string()).unwrap_or_default();
    let mut params: Vec<(String, String)> = query.split('&').filter(|p| !p.is_empty()).map(urldecode_pair).collect();
    // drain headers so the browser sees the response
    let mut sink = String::new();
    while reader.read_line(&mut sink).is_ok() && sink != "\r\n" && sink != "\n" {
        sink.clear();
    }
    let get = |k: &str| params.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
    let respond = |text: &str| -> Result<(), GPhotosError> {
        let body = format!("<html><body style=\"font-family:system-ui;padding:3em\"><h2>{text}</h2><p>You can close this tab.</p></body></html>");
        let mut w = stream;
        write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
            .map_err(GPhotosError::from)
    };
    if let Some(err) = get("error") {
        respond("Consent was declined")?;
        return Err(GPhotosError::Auth(format!("Google returned `{err}` (consent declined or scope not granted)")));
    }
    let state = get("state").unwrap_or_default();
    if state != want_state {
        respond("Stale response")?;
        return Err(GPhotosError::Auth("the redirect's state did not match this request; try again".into()));
    }
    let Some(code) = get("code") else {
        respond("Missing code")?;
        return Err(GPhotosError::Auth("the redirect carried no authorization code".into()));
    };
    params.clear();
    respond("Signed in to Google Photos")?;
    Ok(code)
}

/// Exchange the authorization code for tokens (this is when the refresh token arrives).
fn exchange_code(secret: &ClientSecret, code: &str) -> Result<Tokens, GPhotosError> {
    let body = json!({
        "code": code, "client_id": secret.client_id, "client_secret": secret.client_secret,
        "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
    });
    let tokens = token_request(&crate::api::http_agent(), secret, &body)?;
    if tokens.refresh_token.is_none() {
        return Err(GPhotosError::Auth("Google returned no refresh token; re-run auth (prompt=consent forces one)".into()));
    }
    Ok(tokens)
}

/// Use the refresh token for a new access token.
pub fn refresh(agent: &ureq::Agent, secret: &ClientSecret, tokens: &mut Tokens) -> Result<(), GPhotosError> {
    let Some(refresh_token) = tokens.refresh_token.clone() else {
        return Err(GPhotosError::Auth("the access token expired and there is no refresh token; re-run `filmcraft-cli photos auth`".into()));
    };
    let body = json!({"refresh_token": refresh_token, "client_id": secret.client_id, "client_secret": secret.client_secret, "grant_type": "refresh_token"});
    let fresh = token_request(agent, secret, &body)?;
    tokens.access_token = fresh.access_token;
    tokens.expires_at = fresh.expires_at;
    if fresh.refresh_token.is_some() {
        tokens.refresh_token = fresh.refresh_token;
    }
    Ok(())
}

fn token_request(agent: &ureq::Agent, secret: &ClientSecret, body: &Value) -> Result<Tokens, GPhotosError> {
    let resp = agent.post(TOKEN_URL).send_json(body).map_err(|e| GPhotosError::Auth(format!("the token endpoint failed: {e}")))?;
    let status = resp.status();
    let mut text = String::new();
    resp.into_body().into_reader().take(1 << 16).read_to_string(&mut text).map_err(GPhotosError::from)?;
    if status != 200 {
        return Err(GPhotosError::Auth(format!("the token endpoint answered {status}: {}", truncate(&text, 300))));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| GPhotosError::Auth(format!("the token endpoint sent invalid JSON: {e}")))?;
    let _ = secret;
    Ok(Tokens {
        access_token: v["access_token"].as_str().unwrap_or_default().to_string(),
        refresh_token: v["refresh_token"].as_str().map(str::to_string),
        expires_at: now() + v["expires_in"].as_u64().unwrap_or(3600),
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
    while i < bytes.len() {
        match bytes[i] {
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

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
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
    }

    #[test]
    fn tokens_round_trip_owner_only() {
        let dir = std::env::temp_dir().join(format!("gphotos-tokens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 123 };
        t.save(&dir).unwrap();
        let back = Tokens::load(&dir).unwrap();
        assert_eq!((back.access_token, back.refresh_token, back.expires_at), ("a".into(), Some("r".into()), 123));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(dir.join("tokens.json")).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "owner-only");
        }
        assert!(!Tokens::load(&dir.join("nope")).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refresh_window_is_early() {
        let t = Tokens { access_token: "a".into(), refresh_token: None, expires_at: now() + 30 };
        assert!(t.needs_refresh(), "refreshes a minute ahead");
        let t = Tokens { access_token: "a".into(), refresh_token: None, expires_at: now() + 600 };
        assert!(!t.needs_refresh());
    }
}
