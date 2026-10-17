//! The tokens of one sign-in, as stored (the Keychain item and `tokens.json` hold exactly this
//! JSON) and as read from a token endpoint's answer.

use serde::{Deserialize, Serialize};

use crate::{OAuthError, now};

/// Refresh this long before the access token runs out, so a call never starts on a dying token.
pub const REFRESH_MARGIN_SECS: u64 = 60;
/// What an answer without `expires_in` is taken to last (Google's access tokens last an hour).
const DEFAULT_LIFETIME_SECS: u64 = 3600;
/// Longer than any token answer or stored token set (Google's tokens are under 3 KB in all).
pub const MAX_TOKEN_JSON: usize = 64 * 1024;

/// An access token (short-lived), the refresh token (long-lived, until revoked or lapsed) and
/// the scopes granted. Every field is optional on reading, so an odd file is an error or an
/// empty field, never a crash.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenSet {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix seconds after which the access token must be refreshed.
    pub expires_at: u64,
    /// The scopes the provider granted, space-separated (a consent screen lets the user untick
    /// some).
    pub scope: String,
}

/// Tokens never reach a log or an error message.
impl std::fmt::Debug for TokenSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenSet")
            .field("access_token", &if self.access_token.is_empty() { "" } else { "<redacted>" })
            .field("refresh_token", &self.refresh_token.as_ref().map(|_| "<redacted>"))
            .field("expires_at", &self.expires_at)
            .field("scope", &self.scope)
            .finish()
    }
}

impl TokenSet {
    /// Whether the access token is gone or about to be ([`REFRESH_MARGIN_SECS`]).
    pub fn needs_refresh(&self) -> bool {
        self.needs_refresh_at(now())
    }

    pub fn needs_refresh_at(&self, now: u64) -> bool {
        self.access_token.is_empty() || now.saturating_add(REFRESH_MARGIN_SECS) >= self.expires_at
    }

    /// Whether `scope` was granted (whole scopes only).
    pub fn grants(&self, scope: &str) -> bool {
        self.scope.split_whitespace().any(|s| s == scope)
    }

    /// The stored form: compact JSON.
    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// The stored form back; `what` names the place in the error ("tokens.json").
    pub fn from_json(bytes: &[u8], what: &str) -> Result<Self, OAuthError> {
        if bytes.len() > MAX_TOKEN_JSON {
            return Err(OAuthError::Store(format!("{what} is {} bytes, more than a sign-in ever takes; sign in again", bytes.len())));
        }
        serde_json::from_slice(bytes).map_err(|e| OAuthError::Store(format!("{what} is corrupt ({e}); sign in again")))
    }

    /// A refresh's answer on top of what was stored: a new access token and expiry, the refresh
    /// token kept unless a new one came, the scopes kept unless the answer lists them.
    pub fn refreshed(&self, fresh: TokenSet) -> TokenSet {
        TokenSet {
            access_token: fresh.access_token,
            refresh_token: fresh.refresh_token.or_else(|| self.refresh_token.clone()),
            expires_at: fresh.expires_at,
            scope: if fresh.scope.is_empty() { self.scope.clone() } else { fresh.scope },
        }
    }
}

/// A token endpoint's JSON (RFC 6749 §5.1 and §5.2).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TokenAnswer {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<serde_json::Value>,
    scope: String,
    token_type: String,
}

/// The `error` and `error_description` of a refused token request.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(crate) struct ErrorAnswer {
    pub error: String,
    pub error_description: String,
}

/// A successful token endpoint answer as a [`TokenSet`], `expires_in` counted from `now`. An
/// absent `scope` stays empty (the caller knows what it asked for).
pub fn parse_token_answer(text: &str, now: u64) -> Result<TokenSet, OAuthError> {
    if text.len() > MAX_TOKEN_JSON {
        return Err(OAuthError::Token("the token endpoint's answer is too large".into()));
    }
    let t: TokenAnswer = serde_json::from_str(text).map_err(|e| OAuthError::Token(format!("the token endpoint sent unreadable JSON: {e}")))?;
    if t.access_token.is_empty() {
        return Err(OAuthError::Token("the token endpoint sent no access token".into()));
    }
    if !t.token_type.is_empty() && !t.token_type.eq_ignore_ascii_case("bearer") {
        return Err(OAuthError::Token(format!("the token endpoint sent a `{}` token, not a bearer token", crate::truncate(&t.token_type, 40))));
    }
    // a number (sometimes sent as a string); anything else, zero or negative is the default
    let lifetime = match &t.expires_in {
        Some(serde_json::Value::Number(n)) => n.as_u64(),
        Some(serde_json::Value::String(s)) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
    .filter(|&s| s > 0)
    .unwrap_or(DEFAULT_LIFETIME_SECS);
    Ok(TokenSet {
        access_token: t.access_token,
        refresh_token: t.refresh_token.filter(|r| !r.is_empty()),
        expires_at: now.saturating_add(lifetime),
        scope: t.scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_parse() {
        let t = parse_token_answer(r#"{"access_token":"ya29","expires_in":3599,"refresh_token":"1//r","scope":"a b","token_type":"Bearer"}"#, 1000).unwrap();
        assert_eq!((t.access_token.as_str(), t.refresh_token.as_deref(), t.expires_at), ("ya29", Some("1//r"), 4599));
        assert!(t.grants("a") && t.grants("b") && !t.grants("a b") && !t.grants("c"));
        let t = parse_token_answer(r#"{"access_token":"x","expires_in":"120"}"#, 0).unwrap();
        assert_eq!(t.expires_at, 120);
        for odd in [r#""expires_in":-5"#, r#""expires_in":0"#, r#""expires_in":1.5"#, r#""expires_in":null"#, r#""expires_in":[1]"#] {
            let t = parse_token_answer(&format!(r#"{{"access_token":"x",{odd}}}"#), 10).unwrap();
            assert_eq!(t.expires_at, 10 + DEFAULT_LIFETIME_SECS, "{odd}");
        }
        assert_eq!(parse_token_answer(r#"{"access_token":"a","expires_in":18446744073709551615}"#, 5).unwrap().expires_at, u64::MAX, "saturates");
        assert!(parse_token_answer(r#"{"expires_in":3599}"#, 0).is_err(), "no access token");
        assert!(parse_token_answer(r#"{"access_token":"a","token_type":"mac"}"#, 0).is_err());
        assert!(parse_token_answer("not json", 0).is_err());
        assert!(parse_token_answer(&format!(r#"{{"access_token":"{}"}}"#, "a".repeat(MAX_TOKEN_JSON)), 0).is_err());
        assert_eq!(parse_token_answer(r#"{"access_token":"a","refresh_token":""}"#, 0).unwrap().refresh_token, None);
    }

    #[test]
    fn stored_form_round_trips_and_odd_files_are_errors() {
        let t = TokenSet { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 123, scope: "s".into() };
        assert_eq!(TokenSet::from_json(&t.to_json(), "x").unwrap(), t);
        // the pre-oauth-crate tokens.json
        let old = TokenSet::from_json(br#"{"access_token":"a","refresh_token":"r","expires_at":1,"scope":"s"}"#, "tokens.json").unwrap();
        assert_eq!(old.refresh_token.as_deref(), Some("r"));
        assert_eq!(TokenSet::from_json(b"{}", "x").unwrap(), TokenSet::default());
        let e = TokenSet::from_json(br#"{"access_token":"a","expires_at":-5}"#, "tokens.json").unwrap_err();
        assert!(e.to_string().contains("tokens.json is corrupt"), "{e}");
        assert!(TokenSet::from_json(b"{not json", "x").is_err());
        assert!(TokenSet::from_json(&vec![b' '; MAX_TOKEN_JSON + 1], "x").is_err());
    }

    #[test]
    fn debug_never_shows_tokens() {
        let t = TokenSet { access_token: "ya29.secret".into(), refresh_token: Some("1//secret".into()), expires_at: 1, scope: "s".into() };
        let d = format!("{t:?}");
        assert!(!d.contains("secret") && d.contains("redacted"), "{d}");
    }

    #[test]
    fn refresh_timing_and_merge() {
        let t = TokenSet { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 1000, scope: "s1 s2".into() };
        assert!(!t.needs_refresh_at(900));
        assert!(t.needs_refresh_at(940), "a minute early");
        assert!(t.needs_refresh_at(u64::MAX));
        assert!(TokenSet { access_token: String::new(), ..t.clone() }.needs_refresh_at(0));
        let m = t.refreshed(TokenSet { access_token: "b".into(), refresh_token: None, expires_at: 5000, scope: String::new() });
        assert_eq!((m.access_token.as_str(), m.refresh_token.as_deref(), m.expires_at, m.scope.as_str()), ("b", Some("r"), 5000, "s1 s2"));
        let m = t.refreshed(TokenSet { access_token: "c".into(), refresh_token: Some("r2".into()), expires_at: 1, scope: "s1".into() });
        assert_eq!((m.refresh_token.as_deref(), m.scope.as_str()), (Some("r2"), "s1"));
    }
}
