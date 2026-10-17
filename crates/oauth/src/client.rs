//! The authorization-code grant with PKCE for a native app (RFC 6749 §4.1, RFC 7636, RFC 8252):
//! consent in the system browser, the loopback redirect, the code exchange; then refresh and
//! revocation.
//!
//! Token requests are `application/x-www-form-urlencoded` POSTs (RFC 6749 §4.1.3, §6); answers
//! are read at most [`crate::token::MAX_TOKEN_JSON`] bytes and treated as hostile.

use std::io::Read as _;
use std::time::Duration;

use crate::loopback::{DonePage, Loopback};
use crate::pkce::{self, Pkce};
use crate::token::{ErrorAnswer, MAX_TOKEN_JSON, TokenSet, parse_token_answer};
use crate::{OAuthError, now, truncate, urlencode};

/// The longest a sign-in waits for the browser to come back.
pub const MAX_WAIT: Duration = Duration::from_secs(1800);

/// An OAuth provider's endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// Its name in messages ("Google").
    pub name: String,
    pub auth_url: String,
    pub token_url: String,
    /// RFC 7009 revocation, where the provider has it.
    pub revoke_url: Option<String>,
    /// Why a saved sign-in may stop working, said with [`OAuthError::SignInAgain`].
    pub lapse_note: String,
}

impl Provider {
    /// Google's endpoints (Google Photos; YouTube next).
    pub fn google() -> Self {
        Self {
            name: "Google".into(),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            revoke_url: Some("https://oauth2.googleapis.com/revoke".into()),
            lapse_note: " — it expired or was revoked (Google ends the sign-ins of a Google Cloud app in Testing mode after 7 days, and changing the account's password or removing the app's access ends them too)".into(),
        }
    }
}

/// An OAuth client registered with a [`Provider`].
#[derive(Clone, PartialEq, Eq)]
pub struct OAuthClient {
    pub provider: Provider,
    pub client_id: String,
    /// Google's Desktop clients have a "secret" that is not secret (it ships with the app) but
    /// that the token endpoint still requires; other providers' public clients have none.
    pub client_secret: Option<String>,
    /// What the user runs to sign in again (`filmcraft-cli photos auth`): every error that only
    /// a new sign-in fixes names it.
    pub sign_in_command: String,
}

impl std::fmt::Debug for OAuthClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthClient")
            .field("provider", &self.provider.name)
            .field("client_id", &self.client_id)
            .field("client_secret", &self.client_secret.as_ref().map(|_| "<redacted>"))
            .field("sign_in_command", &self.sign_in_command)
            .finish()
    }
}

/// What a sign-in asks for.
#[derive(Debug, Clone)]
pub struct SignInRequest {
    pub scopes: Vec<String>,
    /// Provider-specific consent parameters (Google: `access_type=offline`,
    /// `include_granted_scopes=true`, `prompt=consent`).
    pub extra: Vec<(String, String)>,
    /// How long to wait for the browser (capped at [`MAX_WAIT`]).
    pub wait: Duration,
    /// What the browser tab says afterwards.
    pub page: DonePage,
}

impl OAuthClient {
    /// The consent URL: `response_type=code`, the scopes, `state`, the S256 challenge, and
    /// `extra`.
    pub fn authorization_url(&self, redirect_uri: &str, scopes: &[String], state: &str, challenge: &str, extra: &[(String, String)]) -> String {
        let mut url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&code_challenge={}&code_challenge_method={}",
            self.provider.auth_url,
            urlencode(&self.client_id),
            urlencode(redirect_uri),
            urlencode(&scopes.join(" ")),
            urlencode(state),
            urlencode(challenge),
            pkce::METHOD,
        );
        for (k, v) in extra {
            url.push('&');
            url.push_str(&urlencode(k));
            url.push('=');
            url.push_str(&urlencode(v));
        }
        url
    }

    /// The whole sign-in: a loopback listener on a random port, a fresh `state` and PKCE
    /// verifier, the consent page through `open` (the system browser; `false` when it could not
    /// open it, and the link then goes to `progress` for the user to open), the redirect, the
    /// code exchange. The tokens are returned, not stored.
    pub fn sign_in(
        &self,
        agent: &ureq::Agent,
        req: &SignInRequest,
        open: &mut dyn FnMut(&str) -> bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<TokenSet, OAuthError> {
        let listener = Loopback::bind()?;
        let redirect_uri = listener.redirect_uri();
        let state = pkce::random_state()?;
        let pkce = Pkce::generate()?;
        let url = self.authorization_url(&redirect_uri, &req.scopes, &state, &pkce.challenge(), &req.extra);
        if open(&url) {
            progress(&format!("opened the browser to sign in to {}; waiting for it to come back to {redirect_uri}", req.page.service));
        } else {
            progress(&format!("open this link in a browser to sign in to {}:\n{url}", req.page.service));
        }
        let code = listener.wait_for_code(&state, req.wait.min(MAX_WAIT), &req.page).map_err(|e| match e {
            OAuthError::SignIn(m) => OAuthError::SignIn(format!("{m}; run `{}` again", self.sign_in_command)),
            other => other,
        })?;
        let mut tokens = self.exchange_code(agent, &code, &redirect_uri, &pkce)?;
        if tokens.scope.is_empty() {
            // the provider did not list them (Google always does): what was asked for
            tokens.scope = req.scopes.join(" ");
        }
        Ok(tokens)
    }

    /// Exchange an authorization code (with the verifier of the challenge it was issued for).
    pub fn exchange_code(&self, agent: &ureq::Agent, code: &str, redirect_uri: &str, pkce: &Pkce) -> Result<TokenSet, OAuthError> {
        let mut form = vec![
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", self.client_id.as_str()),
            ("code_verifier", pkce.verifier()),
        ];
        if let Some(s) = &self.client_secret {
            form.push(("client_secret", s.as_str()));
        }
        match self.token_request(agent, &self.provider.token_url, &form) {
            Err(OAuthError::SignInAgain { .. }) => Err(OAuthError::SignIn(format!(
                "{} refused the sign-in code (it expired, was used already, or does not match this sign-in); run `{}` again",
                self.provider.name, self.sign_in_command
            ))),
            other => other,
        }
    }

    /// A new access token for `tokens`' refresh token; the answer merged onto `tokens` (see
    /// [`TokenSet::refreshed`]). A refused refresh token is [`OAuthError::SignInAgain`].
    pub fn refresh(&self, agent: &ureq::Agent, tokens: &TokenSet) -> Result<TokenSet, OAuthError> {
        let Some(refresh_token) = tokens.refresh_token.as_deref().filter(|r| !r.is_empty()) else {
            return Err(self.sign_in_again(" — it was saved without a refresh token"));
        };
        let mut form = vec![("grant_type", "refresh_token"), ("refresh_token", refresh_token), ("client_id", self.client_id.as_str())];
        if let Some(s) = &self.client_secret {
            form.push(("client_secret", s.as_str()));
        }
        let fresh = self.token_request(agent, &self.provider.token_url, &form)?;
        Ok(tokens.refreshed(fresh))
    }

    /// Revoke a token at the provider (RFC 7009); revoking a refresh token ends the whole grant.
    /// A token the provider no longer knows counts as revoked.
    pub fn revoke(&self, agent: &ureq::Agent, token: &str) -> Result<(), OAuthError> {
        let Some(url) = &self.provider.revoke_url else {
            return Err(OAuthError::Client(format!("{} has no revocation endpoint; remove FilmCraft's access in your account settings", self.provider.name)));
        };
        let resp = agent
            .post(url)
            .send_form([("token", token)])
            .map_err(|e| OAuthError::Token(format!("could not reach {}'s revocation endpoint: {e}", self.provider.name)))?;
        let status = resp.status().as_u16();
        let text = read_capped(resp)?;
        if status == 200 {
            return Ok(());
        }
        let err: ErrorAnswer = serde_json::from_str(&text).unwrap_or_default();
        if status == 400 && err.error == "invalid_token" {
            return Ok(());
        }
        Err(OAuthError::Token(format!("{} did not revoke the sign-in ({status}{})", self.provider.name, describe(&err))))
    }

    fn sign_in_again(&self, why: &str) -> OAuthError {
        OAuthError::SignInAgain {
            provider: self.provider.name.clone(),
            command: self.sign_in_command.clone(),
            why: if why.is_empty() { self.provider.lapse_note.clone() } else { why.to_string() },
        }
    }

    /// One token request; the answer as tokens, or the error a user can act on.
    fn token_request(&self, agent: &ureq::Agent, url: &str, form: &[(&str, &str)]) -> Result<TokenSet, OAuthError> {
        let resp = agent
            .post(url)
            .header("Accept", "application/json")
            .send_form(form.iter().copied())
            .map_err(|e| OAuthError::Token(format!("could not reach {}'s token endpoint: {e}", self.provider.name)))?;
        let status = resp.status().as_u16();
        let text = read_capped(resp)?;
        if status == 200 {
            return parse_token_answer(&text, now());
        }
        let err: ErrorAnswer = serde_json::from_str(&text).unwrap_or_default();
        match err.error.as_str() {
            // RFC 6749 §5.2: the grant (refresh token or code) is invalid, expired or revoked
            "invalid_grant" => Err(self.sign_in_again("")),
            "invalid_client" | "unauthorized_client" => Err(OAuthError::Client(format!(
                "{} refused FilmCraft's OAuth client{} — check the client file (a Desktop client of your own project), then run `{}`",
                self.provider.name,
                describe(&err),
                self.sign_in_command
            ))),
            _ if err.error.is_empty() => {
                Err(OAuthError::Token(format!("{}'s token endpoint answered {status}: {}", self.provider.name, truncate(text.trim(), 200))))
            }
            _ => Err(OAuthError::Token(format!("{}'s token endpoint answered {status}{}", self.provider.name, describe(&err)))),
        }
    }
}

/// ` (error: description)`, both cut short and stripped of control characters.
fn describe(err: &ErrorAnswer) -> String {
    let clean = |s: &str, n: usize| truncate(s, n).chars().filter(|c| !c.is_control()).collect::<String>();
    match (err.error.is_empty(), err.error_description.is_empty()) {
        (true, _) => String::new(),
        (false, true) => format!(" ({})", clean(&err.error, 60)),
        (false, false) => format!(" ({}: {})", clean(&err.error, 60), clean(&err.error_description, 200)),
    }
}

fn read_capped(resp: ureq::http::Response<ureq::Body>) -> Result<String, OAuthError> {
    let mut bytes = Vec::new();
    resp.into_body()
        .into_reader()
        .take(MAX_TOKEN_JSON as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| OAuthError::Token(format!("the token endpoint's answer broke off: {e}")))?;
    if bytes.len() > MAX_TOKEN_JSON {
        return Err(OAuthError::Token("the token endpoint's answer is too large".into()));
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> OAuthClient {
        OAuthClient {
            provider: Provider::google(),
            client_id: "id.apps.googleusercontent.com".into(),
            client_secret: Some("shh".into()),
            sign_in_command: "filmcraft-cli photos auth".into(),
        }
    }

    #[test]
    fn the_consent_url_carries_pkce_and_state() {
        let url = client().authorization_url(
            "http://127.0.0.1:5555",
            &["a b".into(), "c".into()],
            "st/1",
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            &[("access_type".into(), "offline".into()), ("include_granted_scopes".into(), "true".into())],
        );
        assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?response_type=code&"), "{url}");
        for part in [
            "client_id=id.apps.googleusercontent.com",
            "redirect_uri=http%3A%2F%2F127.0.0.1%3A5555",
            "scope=a%20b%20c",
            "state=st%2F1",
            "code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "code_challenge_method=S256",
            "access_type=offline",
            "include_granted_scopes=true",
        ] {
            assert!(url.contains(part), "{part} in {url}");
        }
        assert!(!url.contains("shh"), "the client secret never goes in a URL");
    }

    #[test]
    fn errors_are_described_briefly() {
        let e = ErrorAnswer { error: "invalid_request".into(), error_description: format!("bad\u{7}{}", "x".repeat(500)) };
        let d = describe(&e);
        assert!(d.starts_with(" (invalid_request: bad") && !d.contains('\u{7}') && d.len() < 300, "{d}");
        assert_eq!(describe(&ErrorAnswer::default()), "");
        assert!(!format!("{:?}", client()).contains("shh"));
    }

    #[test]
    fn refresh_without_a_refresh_token_asks_for_a_sign_in() {
        let t = TokenSet { access_token: "a".into(), refresh_token: None, expires_at: 0, scope: String::new() };
        let e = client().refresh(&ureq::Agent::new_with_defaults(), &t).unwrap_err();
        assert!(matches!(e, OAuthError::SignInAgain { .. }) && e.to_string().contains("`filmcraft-cli photos auth`"), "{e}");
    }
}
