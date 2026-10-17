//! OAuth 2.0 for desktop sign-ins, shared by every FilmCraft integration that signs in to an
//! account (Google Photos now; YouTube next).
//!
//! - [`client`]: the authorization-code flow with PKCE S256 (RFC 7636) for native apps
//!   (RFC 8252): the consent page in the system browser, the redirect caught by a
//!   [`loopback`] listener on `127.0.0.1` with a random port, `state` checked, the code
//!   exchanged together with the PKCE verifier; token refresh, and an `invalid_grant` refresh
//!   turned into [`OAuthError::SignInAgain`] that names the command to run; revocation.
//! - [`pkce`]: verifiers, S256 challenges and `state` values from the OS's secure random source.
//! - [`token`]: the token set (access token, refresh token, expiry, granted scopes), with a
//!   `Debug` that never prints a token.
//! - [`store`]: the [`store::SecretStore`] trait and its implementations: the macOS Keychain
//!   ([`keychain::KeychainStore`]), an owner-only file ([`store::FileStore`]) and memory (tests).
//! - [`vault`]: where one integration's tokens live: the Keychain when there is one, the
//!   owner-only `tokens.json` otherwise; an existing `tokens.json` is moved into the Keychain on
//!   first use and removed only after the Keychain copy reads back the same.
//!
//! Nothing here prints: callers report progress (stdout may be a protocol stream).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod client;
pub mod keychain;
pub mod loopback;
pub mod pkce;
pub mod store;
pub mod token;
pub mod vault;

pub use client::{OAuthClient, Provider, SignInRequest};
pub use store::{FileStore, MemoryStore, SecretStore, StoreError};
pub use token::TokenSet;
pub use vault::{Place, Stored, Summary, TokenVault};

use std::time::SystemTime;

/// Everything that can go wrong signing in or keeping a sign-in, each with a message a user can
/// act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OAuthError {
    /// The browser half of a sign-in: consent declined, a forged or stale redirect, no answer in
    /// time, the loopback port unavailable.
    #[error("{0}")]
    SignIn(String),
    /// The provider no longer accepts the saved sign-in (`invalid_grant` on refresh): it
    /// expired or was revoked, and only a new sign-in helps. The message names the command.
    #[error("{provider} no longer accepts the saved sign-in{why}; sign in again: `{command}`")]
    SignInAgain {
        provider: String,
        /// The command that signs in again (`filmcraft-cli photos auth`).
        command: String,
        /// Why that happens, from the provider (" — it expired or was revoked (…)"); may be empty.
        why: String,
    },
    /// The provider refused the OAuth client itself (`invalid_client`, `unauthorized_client`).
    #[error("{0}")]
    Client(String),
    /// The token endpoint failed otherwise: unreachable, a server error, an unreadable answer.
    #[error("{0}")]
    Token(String),
    /// Keeping or reading the tokens failed (Keychain, file).
    #[error("{0}")]
    Store(String),
}

/// Unix seconds now (0 if the clock is before 1970).
pub fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Percent-encode everything but RFC 3986's unreserved characters.
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Decode `%XX` and `+` (a query string); malformed escapes are kept as they are.
pub fn urldecode(s: &str) -> String {
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

/// At most `n` characters of `s` (for quoting a server's answer in an error).
pub fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => s.get(..i).unwrap_or(s),
        None => s,
    }
}

/// Open an https link in the system browser, detached and silent (stdout may be an MCP stream).
/// Anything that is not `https://` is refused, so a hostile link can never become an opener flag
/// or a local file.
pub fn open_in_browser(url: &str) -> bool {
    if !url.starts_with("https://") || url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    #[cfg(target_os = "macos")]
    let mut cmd = std::process::Command::new("open");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler");
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = std::process::Command::new("xdg-open");
    #[cfg(not(any(unix, windows)))]
    return false;
    #[cfg(any(unix, windows))]
    {
        use std::process::Stdio;
        match cmd.arg(url).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
            Ok(mut child) => {
                // reap the opener without blocking (xdg-open can outlive the call)
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                true
            }
            Err(_) => false,
        }
    }
}

/// The HTTPS agent for token endpoints: pure-Rust TLS with the OS's certificate verification,
/// https only, status codes as answers (not errors), bounded waits.
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
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_global(Some(std::time::Duration::from_secs(60)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_coding() {
        assert_eq!(urldecode("4%2F0Ax"), "4/0Ax");
        assert_eq!(urldecode("a+b%20c"), "a b c");
        assert_eq!(urldecode("dangling%2"), "dangling%2");
        assert_eq!(urldecode("%zz%"), "%zz%");
        assert_eq!(urlencode("a b/c?~"), "a%20b%2Fc%3F~");
        assert_eq!(urlencode("é"), "%C3%A9");
        assert_eq!(truncate("héllo", 2), "hé");
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn the_opener_refuses_anything_but_https() {
        for bad in ["", "-a Calculator", "file:///etc/passwd", "http://example.com", "https://x.example/a b", "https://x\n", "javascript:alert(1)"] {
            assert!(!open_in_browser(bad), "{bad:?}");
        }
    }

    #[test]
    fn sign_in_again_names_the_command() {
        let e = OAuthError::SignInAgain { provider: "Google".into(), command: "filmcraft-cli photos auth".into(), why: " — it lapsed".into() };
        assert_eq!(e.to_string(), "Google no longer accepts the saved sign-in — it lapsed; sign in again: `filmcraft-cli photos auth`");
    }
}
