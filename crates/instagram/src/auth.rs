//! Signing in to Instagram ("Business Login for Instagram") and keeping the sign-in alive.
//!
//! - **The app.** The user's own Meta app's Instagram app id and its redirect URI live in
//!   `<data dir>/instagram/app.json`; its **app secret** in the Keychain item **FilmCraft
//!   Instagram app** (an owner-only `app_secret` file where there is no Keychain), entered once
//!   with `filmcraft-cli instagram setup` from standard input or a hidden prompt — never from
//!   the command line, never printed.
//! - **The browser.** `https://www.instagram.com/oauth/authorize` with `client_id`,
//!   `redirect_uri` (exactly as registered in the App Dashboard), `response_type=code`, the
//!   scopes `instagram_business_basic,instagram_business_content_publish` and a random `state`.
//!   Instagram documents no PKCE for this flow, so none is sent; the code is worth nothing
//!   without the app secret. An `http://localhost:<port>/` (or `127.0.0.1`) redirect is caught
//!   by a listener on that port; any other redirect (an `https://` page) is pasted back: the
//!   user copies the address the browser ended on.
//! - **The tokens.** The code (valid an hour, once; Instagram appends `#_`, which is not part of
//!   it) is exchanged at `https://api.instagram.com/oauth/access_token` for a short-lived token
//!   (an hour), then at `graph.instagram.com/access_token?grant_type=ig_exchange_token` for a
//!   long-lived one (60 days). `graph.instagram.com/refresh_access_token?grant_type=ig_refresh_token`
//!   renews a long-lived token that is at least 24 hours old and not expired, for another 60 days;
//!   one not refreshed for 60 days is gone. Meta takes the secret and tokens of these two calls in
//!   the query string; no error message here ever quotes a URL.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use filmcraft_oauth::{FileStore, SecretStore, keychain::account_for, store::system_store, urldecode, urlencode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http::{Body, Method, Request, Transport};
use crate::{Endpoints, IgError, clean, write_private};

/// The permissions asked for: the profile (`/me`) and publishing.
pub const SCOPES: &[&str] = &["instagram_business_basic", "instagram_business_content_publish"];
pub const PUBLISH_PERMISSION: &str = "instagram_business_content_publish";
/// The Keychain item (service) of the app secret.
pub const SECRET_SERVICE: &str = "FilmCraft Instagram app";
/// The redirect URI `instagram setup` registers unless told otherwise.
pub const DEFAULT_REDIRECT: &str = "http://localhost:8714/";
/// How long a browser sign-in waits.
pub const CONSENT_WAIT: Duration = Duration::from_secs(10 * 60);
/// What a token answer without `expires_in` is taken to last (a long-lived token: 60 days).
pub const LONG_LIVED_SECS: u64 = 60 * 24 * 60 * 60;
/// A long-lived token must be this old before Instagram refreshes it.
pub const REFRESH_AFTER_SECS: u64 = 24 * 60 * 60;
/// The longest token taken (Instagram's are a few hundred characters).
const MAX_TOKEN: usize = 2048;

pub const SIGN_IN: &str = "filmcraft-cli instagram auth";
pub const PASTE_TOKEN: &str = "filmcraft-cli instagram token";
pub const SETUP: &str = "filmcraft-cli instagram setup";

/// The non-secret half of the app (`app.json`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppConfig {
    /// The **Instagram** app id (App Dashboard ▸ Instagram ▸ API setup with Instagram login),
    /// not the Meta app id.
    pub app_id: String,
    pub redirect_uri: String,
}

/// An app id: digits only.
pub fn check_app_id(id: &str) -> Result<String, String> {
    let id = id.trim();
    if id.len() < 5 || id.len() > 25 || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "`{}` is not an Instagram app id (digits only: App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Instagram app ID)",
            clean(id, 30)
        ));
    }
    Ok(id.to_string())
}

/// An app secret: letters and digits (Meta's are 32 hexadecimal characters).
pub fn check_secret(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.len() < 16 || s.len() > 256 || !s.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err("that is not an Instagram app secret (letters and digits, 32 of them, from App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Instagram app secret ▸ Show)".into());
    }
    Ok(s.to_string())
}

/// A token pasted from the App Dashboard: printable ASCII without spaces.
pub fn check_token(t: &str) -> Result<String, String> {
    let t = t.trim();
    if t.len() < 20 || t.len() > MAX_TOKEN || !t.bytes().all(|b| b.is_ascii_graphic()) {
        return Err("that is not an Instagram access token (one line of letters, digits and punctuation, from App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Generate access tokens)".into());
    }
    Ok(t.to_string())
}

/// Where the redirect comes back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Redirect {
    /// `http://localhost:<port>/…` or `http://127.0.0.1:<port>/…`: a listener catches it.
    Loopback(u16),
    /// Anything else: the user pastes the address the browser ended on.
    Paste,
}

/// A redirect URI FilmCraft can use: `http://localhost:<port>/…`, `http://127.0.0.1:<port>/…`, or
/// any `https://` URL (pasted back). Exactly as registered in the App Dashboard.
pub fn check_redirect(uri: &str) -> Result<(String, Redirect), String> {
    let u = uri.trim();
    let bad_uri =
        || format!("`{}` is not a redirect URI FilmCraft can use: http://localhost:<port>/ (caught here) or an https:// address (pasted back)", clean(u, 80));
    if u.is_empty() || u.len() > 512 || u.chars().any(|c| c.is_whitespace() || c.is_control()) || u.contains('#') {
        return Err(bad_uri());
    }
    if let Some(rest) = u.strip_prefix("http://") {
        let host = rest.split(['/', '?']).next().unwrap_or_default();
        let (name, port) = host.rsplit_once(':').ok_or_else(|| format!("{}: give the port, e.g. {DEFAULT_REDIRECT}", bad_uri()))?;
        if !(name.eq_ignore_ascii_case("localhost") || name == "127.0.0.1") {
            return Err(bad_uri());
        }
        let port: u16 = port.parse().ok().filter(|p| *p >= 1024).ok_or_else(bad_uri)?;
        return Ok((u.to_string(), Redirect::Loopback(port)));
    }
    if u.starts_with("https://") && u.len() > "https://".len() {
        return Ok((u.to_string(), Redirect::Paste));
    }
    Err(bad_uri())
}

/// The app's id and secret (`app.json` beside the secret store).
pub struct AppStore {
    dir: PathBuf,
    secure: Option<Arc<dyn SecretStore>>,
}

impl AppStore {
    /// The Keychain (unless `FILMCRAFT_SECRET_STORE=file`), else owner-only files in `dir`.
    pub fn system(dir: &Path) -> Self {
        Self { dir: dir.to_path_buf(), secure: system_store(SECRET_SERVICE) }
    }

    pub fn with_store(dir: &Path, store: Option<Arc<dyn SecretStore>>) -> Self {
        Self { dir: dir.to_path_buf(), secure: store }
    }

    fn config_path(&self) -> PathBuf {
        self.dir.join("app.json")
    }

    /// Where the secret is kept, for messages.
    pub fn place(&self) -> String {
        match &self.secure {
            Some(s) => s.name(),
            None => format!("{} (readable only by you)", self.dir.join("app_secret").display()),
        }
    }

    /// The app config, `None` before `instagram setup`.
    pub fn config(&self) -> Result<Option<AppConfig>, IgError> {
        let path = self.config_path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(IgError::Setup(format!("cannot read {}: {e}", path.display()))),
        };
        let c: AppConfig = serde_json::from_slice(bytes.get(..bytes.len().min(64 * 1024)).unwrap_or_default())
            .map_err(|e| IgError::Setup(format!("{} is damaged ({e}); run `{SETUP}` again", path.display())))?;
        check_app_id(&c.app_id).map_err(|e| IgError::Setup(format!("{}: {e}; run `{SETUP}` again", path.display())))?;
        check_redirect(&c.redirect_uri).map_err(|e| IgError::Setup(format!("{}: {e}; run `{SETUP}` again", path.display())))?;
        Ok(Some(c))
    }

    /// Whether the app is set up (from disk, no Keychain: asked while drawing).
    pub fn ready(&self) -> Result<(), String> {
        match self.config() {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(format!(
                "Instagram's business login is not set up: run `{SETUP}` (your Instagram app id, its secret and the redirect URI registered in the Meta App Dashboard), or paste a token from the dashboard: `{PASTE_TOKEN}`"
            )),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Keep the app id, redirect URI and secret (the secret read back to be sure).
    pub fn save(&self, config: &AppConfig, secret: &str) -> Result<String, IgError> {
        let secret = check_secret(secret).map_err(IgError::Setup)?;
        check_app_id(&config.app_id).map_err(IgError::Setup)?;
        check_redirect(&config.redirect_uri).map_err(IgError::Setup)?;
        match &self.secure {
            Some(store) => {
                let account = account_for(&self.dir);
                store.set(&account, secret.as_bytes()).map_err(|e| IgError::Setup(format!("the app secret could not be kept in {}: {e}", store.name())))?;
                match store.get(&account) {
                    Ok(Some(b)) if b == secret.as_bytes() => {}
                    _ => {
                        return Err(IgError::Setup(format!(
                            "{} did not keep the app secret; set FILMCRAFT_SECRET_STORE=file to keep it in an owner-only file",
                            store.name()
                        )));
                    }
                }
                let _ = FileStore::new(&self.dir).delete("app_secret");
            }
            None => FileStore::new(&self.dir).set("app_secret", secret.as_bytes()).map_err(|e| IgError::Setup(e.0))?,
        }
        let path = self.config_path();
        write_private(&path, &serde_json::to_vec_pretty(config).unwrap_or_default())
            .map_err(|e| IgError::Setup(format!("cannot write {}: {e}", path.display())))?;
        Ok(self.place())
    }

    /// The app secret, for the two calls that need it.
    pub fn secret(&self) -> Result<String, IgError> {
        let got = match &self.secure {
            Some(store) => store.get(&account_for(&self.dir)).map_err(|e| {
                IgError::Setup(format!("{e}. Unlock the Keychain (log in at the Mac), or set FILMCRAFT_SECRET_STORE=file and run `{SETUP}` again"))
            })?,
            None => FileStore::new(&self.dir).get("app_secret").map_err(|e| IgError::Setup(e.0))?,
        };
        let bytes = got.ok_or_else(|| IgError::Setup(format!("the Instagram app secret is not kept here; run `{SETUP}`")))?;
        check_secret(&String::from_utf8_lossy(&bytes)).map_err(|_| IgError::Setup(format!("the kept Instagram app secret is damaged; run `{SETUP}` again")))
    }

    /// Forget the app (its id, redirect and secret).
    pub fn remove(&self) -> Result<(), IgError> {
        if let Some(store) = &self.secure {
            store.delete(&account_for(&self.dir)).map_err(|e| IgError::Setup(e.0))?;
        }
        FileStore::new(&self.dir).delete("app_secret").map_err(|e| IgError::Setup(e.0))?;
        match std::fs::remove_file(self.config_path()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(IgError::Setup(format!("cannot remove {}: {e}", self.config_path().display()))),
            _ => Ok(()),
        }
    }
}

/// The consent URL.
pub fn authorize_url(endpoints: &Endpoints, app_id: &str, redirect_uri: &str, state: &str) -> String {
    format!(
        "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}",
        endpoints.authorize,
        urlencode(app_id),
        urlencode(redirect_uri),
        urlencode(&SCOPES.join(",")),
        urlencode(state)
    )
}

/// The code from the address the browser ended on (pasted back): `code` and `state` from its
/// query, Instagram's `#_` dropped. A declined consent, a missing code or another sign-in's
/// `state` is an error.
pub fn code_from_url(pasted: &str, want_state: &str) -> Result<String, IgError> {
    let pasted = pasted.trim();
    let no_fragment = pasted.split('#').next().unwrap_or_default();
    let query = no_fragment.split_once('?').map(|(_, q)| q).unwrap_or(no_fragment);
    let params: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .take(64)
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (urldecode(k), urldecode(v))
        })
        .collect();
    let get = |k: &str| params.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
    if let Some(err) = get("error") {
        let why = get("error_reason").or(get("error_description")).unwrap_or(err);
        return Err(IgError::Auth(format!("the sign-in was declined or failed ({}); nothing was connected", clean(why, 80))));
    }
    if get("state") != Some(want_state) {
        return Err(IgError::Auth(format!(
            "that address did not come from this sign-in (its state does not match); copy the address the browser shows after you allow FilmCraft, and run `{SIGN_IN}` again if it was an older one"
        )));
    }
    let code = get("code")
        .map(|c| c.trim_end_matches("#_"))
        .filter(|c| !c.is_empty() && c.len() <= 4096)
        .ok_or_else(|| IgError::Auth("that address carries no sign-in code".into()))?;
    Ok(code.to_string())
}

/// A token answer, whatever its shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub access_token: String,
    /// Seconds it lasts (`None`: not said).
    pub expires_in: Option<u64>,
    pub user_id: Option<String>,
    /// Comma-separated.
    pub permissions: Option<String>,
}

/// Read a token answer: `{"access_token", "expires_in", "token_type"}`, or the code exchange's
/// `{"data": [{"access_token", "user_id", "permissions"}]}` (older answers are flat). Hostile:
/// sizes and types are checked; a token that is not one is refused.
pub fn parse_tokens(body: &str) -> Result<Tokens, IgError> {
    let v: Value = serde_json::from_str(body).map_err(|_| IgError::Api(format!("Instagram's token answer is not JSON: {}", clean(body, 120))))?;
    let d = match v.get("data") {
        Some(Value::Array(a)) => a.first().cloned().unwrap_or(Value::Null),
        _ => v,
    };
    let access_token = d.get("access_token").and_then(Value::as_str).unwrap_or_default();
    let access_token = check_token(access_token).map_err(|_| IgError::Api("Instagram's token answer carries no usable access token".into()))?;
    if let Some(t) = d.get("token_type").and_then(Value::as_str)
        && !t.eq_ignore_ascii_case("bearer")
    {
        return Err(IgError::Api(format!("Instagram sent a `{}` token, not a bearer token", clean(t, 30))));
    }
    let expires_in = match d.get("expires_in") {
        Some(Value::Number(n)) => n.as_u64().filter(|s| *s > 0 && *s <= 10 * 365 * 86_400),
        Some(Value::String(s)) => s.trim().parse::<u64>().ok().filter(|s| *s > 0 && *s <= 10 * 365 * 86_400),
        _ => None,
    };
    let user_id = match d.get("user_id") {
        Some(Value::Number(n)) => n.as_u64().map(|n| n.to_string()),
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
    .filter(|s| !s.is_empty() && s.len() <= 32 && s.bytes().all(|b| b.is_ascii_digit()));
    let permissions = match d.get("permissions") {
        Some(Value::String(s)) => Some(clean(s, 1000)),
        Some(Value::Array(a)) => Some(a.iter().filter_map(Value::as_str).take(64).map(|s| clean(s, 80)).collect::<Vec<_>>().join(",")),
        _ => None,
    };
    Ok(Tokens { access_token, expires_in, user_id, permissions })
}

/// Instagram's answer to a refused token request, in words.
pub fn token_error(status: u16, body: &str, what: &str) -> IgError {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    // `{"error_type":"OAuthException","code":400,"error_message":"…"}` (api.instagram.com) or
    // `{"error":{"message":"…","type":"OAuthException","code":190}}` (graph.instagram.com)
    let message = v
        .get("error_message")
        .and_then(Value::as_str)
        .or_else(|| v.get("error").and_then(|e| e.get("message")).and_then(Value::as_str))
        .or_else(|| v.get("error_description").and_then(Value::as_str))
        .map(|m| clean(m, 200))
        .unwrap_or_else(|| clean(body, 120));
    let code = v.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64).unwrap_or(0);
    let lower = message.to_ascii_lowercase();
    if lower.contains("client secret") || lower.contains("client_secret") || lower.contains("invalid platform app") || lower.contains("invalid client") {
        return IgError::Setup(format!("Instagram refused the app ({message}): check the Instagram app id and secret with `{SETUP}`"));
    }
    if lower.contains("redirect") {
        return IgError::Setup(format!(
            "Instagram refused the redirect URI ({message}): it must be exactly one of the URIs in App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Set up Instagram business login ▸ OAuth redirect URIs (a trailing / counts); `{SETUP}` sets the one FilmCraft uses"
        ));
    }
    if code == 190 || lower.contains("expired") || lower.contains("session has been invalidated") || lower.contains("authorization code has been used") {
        return IgError::Auth(format!(
            "Instagram no longer accepts that {what} ({message}); connect again: `{SIGN_IN}` (or paste a new token: `{PASTE_TOKEN}`)"
        ));
    }
    IgError::Api(format!("Instagram refused the {what} ({status}: {message})"))
}

/// Exchange a code for a short-lived token (an hour).
pub fn exchange_code(http: &dyn Transport, endpoints: &Endpoints, app: &AppConfig, secret: &str, code: &str) -> Result<Tokens, IgError> {
    let req = Request {
        method: Method::Post,
        url: endpoints.token.clone(),
        headers: vec![("Accept", "application/json".into())],
        body: Body::Form(vec![
            ("client_id", app.app_id.clone()),
            ("client_secret", secret.to_string()),
            ("grant_type", "authorization_code".into()),
            ("redirect_uri", app.redirect_uri.clone()),
            ("code", code.to_string()),
        ]),
    };
    let r = http.send(req).map_err(|e| IgError::Api(format!("could not reach Instagram to finish the sign-in: {e}")))?;
    if r.status != 200 {
        return Err(token_error(r.status, &r.body, "sign-in code"));
    }
    parse_tokens(&r.body)
}

/// Exchange a short-lived token for a long-lived one (60 days).
pub fn long_lived(http: &dyn Transport, endpoints: &Endpoints, secret: &str, short: &str) -> Result<Tokens, IgError> {
    let url = format!("{}/access_token?grant_type=ig_exchange_token&client_secret={}&access_token={}", endpoints.graph, urlencode(secret), urlencode(short));
    let r = http.send(Request::get(url)).map_err(|e| IgError::Api(format!("could not reach Instagram for the 60-day token: {e}")))?;
    if r.status != 200 {
        return Err(token_error(r.status, &r.body, "short-lived token"));
    }
    parse_tokens(&r.body)
}

/// Refresh a long-lived token (at least 24 hours old, not expired) for another 60 days.
pub fn refresh(http: &dyn Transport, endpoints: &Endpoints, token: &str) -> Result<Tokens, IgError> {
    let url = format!("{}/refresh_access_token?grant_type=ig_refresh_token&access_token={}", endpoints.graph, urlencode(token));
    let r = http.send(Request::get(url)).map_err(|e| IgError::Api(format!("could not reach Instagram to refresh the sign-in: {e}")))?;
    if r.status != 200 {
        return Err(token_error(r.status, &r.body, "sign-in"));
    }
    parse_tokens(&r.body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirects_are_checked() {
        assert_eq!(check_redirect("http://localhost:8714/").unwrap().1, Redirect::Loopback(8714));
        assert_eq!(check_redirect(" http://127.0.0.1:50000/cb ").unwrap(), ("http://127.0.0.1:50000/cb".into(), Redirect::Loopback(50000)));
        assert_eq!(check_redirect("https://example.com/ig").unwrap().1, Redirect::Paste);
        for bad in [
            "",
            "http://example.com:80/",
            "http://localhost/",
            "http://localhost:80/",
            "http://localhost:99999/",
            "ftp://x",
            "https://",
            "https://a b",
            "http://localhost:8714/#x",
            "javascript:alert(1)",
        ] {
            assert!(check_redirect(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn ids_secrets_and_tokens_are_checked() {
        assert_eq!(check_app_id(" 2133883020551415 ").unwrap(), "2133883020551415");
        for bad in ["", "abc", "12", "2133883020551415x", &"1".repeat(40)] {
            assert!(check_app_id(bad).is_err(), "{bad:?}");
        }
        assert!(check_secret("0123456789abcdef0123456789abcdef").is_ok());
        for bad in ["", "short", "0123456789abcdef 0123456789abcdef", "0123456789abcdef0123456789abcde!"] {
            assert!(check_secret(bad).is_err(), "{bad:?}");
        }
        assert!(check_token("IGAAx0123456789abcdefghijklmnop").is_ok());
        assert!(check_token("IGAA x").is_err() && check_token("tiny").is_err() && check_token(&"a".repeat(5000)).is_err());
        // a check never echoes the secret
        assert!(!check_secret("0123456789abcdef0123456789abcde!").unwrap_err().contains("0123"));
    }

    #[test]
    fn the_consent_url_and_the_pasted_address() {
        let e = Endpoints::instagram();
        let u = authorize_url(&e, "2133883020551415", "http://localhost:8714/", "st/1");
        assert!(
            u.starts_with(
                "https://www.instagram.com/oauth/authorize?client_id=2133883020551415&redirect_uri=http%3A%2F%2Flocalhost%3A8714%2F&response_type=code&"
            ),
            "{u}"
        );
        assert!(u.contains("scope=instagram_business_basic%2Cinstagram_business_content_publish") && u.contains("state=st%2F1"), "{u}");
        assert_eq!(code_from_url("https://example.com/ig?code=AQB%2Fx&state=s1#_", "s1").unwrap(), "AQB/x");
        assert!(code_from_url("code=abc#_&state=s1", "s1").unwrap_err().to_string().contains("state"), "the fragment ends the query");
        assert!(code_from_url("https://x/?code=abc&state=other", "s1").unwrap_err().to_string().contains("state does not match"));
        assert!(code_from_url("https://x/?error=access_denied&error_reason=user_denied&state=s1", "s1").unwrap_err().to_string().contains("user_denied"));
        assert!(code_from_url("https://x/?state=s1", "s1").unwrap_err().to_string().contains("no sign-in code"));
        assert!(code_from_url("", "s1").is_err());
    }

    #[test]
    fn token_answers_of_every_shape() {
        let t = parse_tokens(r#"{"data":[{"access_token":"IGAAshort0123456789abcdef","user_id":"17841400000000000","permissions":"instagram_business_basic,instagram_business_content_publish"}]}"#).unwrap();
        assert_eq!((t.user_id.as_deref(), t.expires_in), (Some("17841400000000000"), None));
        assert!(t.permissions.unwrap().contains("content_publish"));
        let t = parse_tokens(r#"{"access_token":"IGAAshort0123456789abcdef","user_id":17841400000000000,"permissions":["a","b"]}"#).unwrap();
        assert_eq!((t.user_id.as_deref(), t.permissions.as_deref()), (Some("17841400000000000"), Some("a,b")));
        let t = parse_tokens(r#"{"access_token":"IGAAlong0123456789abcdef","token_type":"bearer","expires_in":5183944}"#).unwrap();
        assert_eq!(t.expires_in, Some(5183944));
        for odd in [
            r#"{"expires_in":-1,"access_token":"IGAAlong0123456789abcdef"}"#,
            r#"{"expires_in":"x","access_token":"IGAAlong0123456789abcdef"}"#,
            r#"{"expires_in":1e40,"access_token":"IGAAlong0123456789abcdef"}"#,
        ] {
            assert_eq!(parse_tokens(odd).unwrap().expires_in, None, "{odd}");
        }
        for bad in [
            "",
            "null",
            "[]",
            r#"{"data":[]}"#,
            r#"{"access_token":""}"#,
            r#"{"access_token":"has space in it 0123456789"}"#,
            r#"{"access_token":"IGAAlong0123456789abcdef","token_type":"mac"}"#,
        ] {
            assert!(parse_tokens(bad).is_err(), "{bad}");
        }
        assert!(parse_tokens(r#"{"access_token":"IGAAlong0123456789abcdef","user_id":"12x"}"#).unwrap().user_id.is_none());
    }

    #[test]
    fn token_errors_in_words() {
        let e = token_error(400, r#"{"error_type":"OAuthException","code":400,"error_message":"Error validating client secret."}"#, "sign-in code");
        assert!(matches!(e, IgError::Setup(_)) && e.to_string().contains(SETUP), "{e}");
        let e = token_error(
            400,
            r#"{"error_type":"OAuthException","code":400,"error_message":"Error validating verification code. Please make sure your redirect_uri is identical"}"#,
            "sign-in code",
        );
        assert!(e.to_string().contains("OAuth redirect URIs"), "{e}");
        let e = token_error(400, r#"{"error":{"message":"Error validating access token: Session has expired","type":"OAuthException","code":190}}"#, "sign-in");
        assert!(matches!(e, IgError::Auth(_)) && e.to_string().contains(SIGN_IN), "{e}");
        let e = token_error(500, "<html>oops</html>", "sign-in");
        assert!(matches!(e, IgError::Api(_)) && e.to_string().contains("500"), "{e}");
    }

    #[test]
    fn the_app_store_keeps_the_secret_apart() {
        let d = std::env::temp_dir().join(format!("filmcraft-ig-app-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let mem: Arc<dyn SecretStore> = Arc::new(filmcraft_oauth::MemoryStore::new());
        let s = AppStore::with_store(&d, Some(mem.clone()));
        assert!(s.ready().unwrap_err().contains(SETUP));
        assert!(s.secret().is_err());
        let cfg = AppConfig { app_id: "2133883020551415".into(), redirect_uri: DEFAULT_REDIRECT.into() };
        assert!(s.save(&cfg, "not a secret").is_err());
        s.save(&cfg, "0123456789abcdef0123456789abcdef").unwrap();
        assert!(s.ready().is_ok());
        assert_eq!(s.secret().unwrap(), "0123456789abcdef0123456789abcdef");
        let on_disk = std::fs::read_to_string(d.join("app.json")).unwrap();
        assert!(!on_disk.contains("0123456789abcdef") && !d.join("app_secret").exists(), "the secret is only in the store");
        // without a secure store: an owner-only file
        let f = AppStore::with_store(&d.join("files"), None);
        f.save(&cfg, "0123456789abcdef0123456789abcdef").unwrap();
        assert_eq!(f.secret().unwrap(), "0123456789abcdef0123456789abcdef");
        std::fs::write(d.join("app.json"), b"{broken").unwrap();
        assert!(s.ready().unwrap_err().contains("damaged"));
        s.remove().unwrap();
        assert!(s.secret().is_err() && mem.get(&account_for(&d)).unwrap().is_none());
        std::fs::remove_dir_all(&d).ok();
    }
}
