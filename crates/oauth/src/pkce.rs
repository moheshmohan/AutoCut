//! Proof Key for Code Exchange (RFC 7636) with the S256 method, and the random `state` that ties
//! a redirect to the request that caused it.
//!
//! The verifier is 32 bytes from the OS's secure random source, base64url-encoded without
//! padding: 43 characters of the unreserved set, as §7.1 recommends. The challenge is
//! `BASE64URL(SHA256(ASCII(verifier)))` (§4.2).

use sha2::{Digest, Sha256};

use crate::OAuthError;

/// The only challenge method FilmCraft sends (`plain` would give the verifier away).
pub const METHOD: &str = "S256";

/// A PKCE verifier; its challenge goes in the consent URL, the verifier itself only to the token
/// endpoint.
#[derive(Clone, PartialEq, Eq)]
pub struct Pkce {
    verifier: String,
}

impl std::fmt::Debug for Pkce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pkce").field("verifier", &"<redacted>").finish()
    }
}

impl Pkce {
    /// A fresh verifier from 32 random bytes.
    pub fn generate() -> Result<Self, OAuthError> {
        Ok(Self { verifier: base64url(&random_bytes::<32>()?) })
    }

    /// A given verifier, checked against RFC 7636 §4.1: 43 to 128 characters of
    /// `[A-Z] / [a-z] / [0-9] / "-" / "." / "_" / "~"`.
    pub fn from_verifier(verifier: &str) -> Result<Self, OAuthError> {
        let n = verifier.len();
        if !(43..=128).contains(&n) {
            return Err(OAuthError::SignIn(format!("a PKCE verifier has 43 to 128 characters, not {n}")));
        }
        if !verifier.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')) {
            return Err(OAuthError::SignIn("a PKCE verifier holds only letters, digits, `-`, `.`, `_` and `~`".into()));
        }
        Ok(Self { verifier: verifier.to_string() })
    }

    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    /// `BASE64URL(SHA256(verifier))`, for `code_challenge` with `code_challenge_method=S256`.
    pub fn challenge(&self) -> String {
        challenge_s256(&self.verifier)
    }
}

/// The S256 challenge of `verifier`.
pub fn challenge_s256(verifier: &str) -> String {
    base64url(&Sha256::digest(verifier.as_bytes()))
}

/// A `state` value: 24 random bytes, base64url (32 characters).
pub fn random_state() -> Result<String, OAuthError> {
    Ok(base64url(&random_bytes::<24>()?))
}

fn random_bytes<const N: usize>() -> Result<[u8; N], OAuthError> {
    let mut b = [0u8; N];
    getrandom::fill(&mut b).map_err(|e| OAuthError::SignIn(format!("the system's random number source failed: {e}")))?;
    Ok(b)
}

const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Base64url without padding (RFC 4648 §5).
pub fn base64url(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3).saturating_mul(4));
    for chunk in bytes.chunks(3) {
        let b = [chunk.first().copied().unwrap_or(0), chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        let sextets = [(n >> 18) & 63, (n >> 12) & 63, (n >> 6) & 63, n & 63];
        // one input byte → two characters, two → three, three → four
        for s in sextets.iter().take(chunk.len() + 1) {
            if let Some(&c) = B64URL.get(*s as usize) {
                out.push(c as char);
            }
        }
    }
    out
}

/// The bytes of unpadded (or padded) base64url text; `None` for anything else.
pub fn base64url_decode(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    if text.len() % 4 == 1 {
        return None;
    }
    let value = |c: u8| B64URL.iter().position(|&x| x == c).map(|p| p as u32);
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 2);
    for chunk in text.as_bytes().chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= value(c)? << (18 - 6 * i as u32);
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend(bytes.iter().take(chunk.len().saturating_sub(1)));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 7636 Appendix B.
    #[test]
    fn rfc7636_appendix_b() {
        let octets =
            [116, 24, 223, 180, 151, 153, 224, 37, 79, 250, 96, 125, 216, 173, 187, 186, 22, 212, 37, 77, 105, 214, 191, 240, 91, 88, 5, 88, 83, 132, 141, 121];
        assert_eq!(base64url(&octets), "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk");
        let p = Pkce::from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk").unwrap();
        assert_eq!(p.challenge(), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        assert_eq!(METHOD, "S256");
    }

    /// RFC 4648 §10, in the URL-safe alphabet without padding.
    #[test]
    fn rfc4648_vectors() {
        for (plain, coded) in [("", ""), ("f", "Zg"), ("fo", "Zm8"), ("foo", "Zm9v"), ("foob", "Zm9vYg"), ("fooba", "Zm9vYmE"), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64url(plain.as_bytes()), coded);
            assert_eq!(base64url_decode(coded).unwrap(), plain.as_bytes());
        }
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
        assert_eq!(base64url_decode("-_8").unwrap(), [0xfb, 0xff]);
        assert_eq!(base64url_decode("Zg==").unwrap(), b"f");
        for bad in ["Z", "Zm9vY", "Zm+v", "Zm/v", "a b"] {
            assert!(base64url_decode(bad).is_none(), "{bad}");
        }
        // round trip over every byte value and length
        let all: Vec<u8> = (0..=255).collect();
        for n in 0..all.len() {
            assert_eq!(base64url_decode(&base64url(&all[..n])).unwrap(), &all[..n]);
        }
    }

    #[test]
    fn generated_verifiers_are_valid_and_differ() {
        let a = Pkce::generate().unwrap();
        let b = Pkce::generate().unwrap();
        assert_ne!(a, b);
        assert_eq!(a.verifier().len(), 43);
        assert!(Pkce::from_verifier(a.verifier()).is_ok());
        assert_eq!(a.challenge().len(), 43);
        assert_ne!(a.challenge(), a.verifier());
        assert!(!format!("{a:?}").contains(a.verifier()), "Debug never shows the verifier");
        let s = random_state().unwrap();
        assert_eq!(s.len(), 32);
        assert_ne!(s, random_state().unwrap());
    }

    #[test]
    fn verifiers_are_checked() {
        assert!(Pkce::from_verifier(&"a".repeat(42)).is_err());
        assert!(Pkce::from_verifier(&"a".repeat(43)).is_ok());
        assert!(Pkce::from_verifier(&"a".repeat(128)).is_ok());
        assert!(Pkce::from_verifier(&"a".repeat(129)).is_err());
        assert!(Pkce::from_verifier(&format!("{}+", "a".repeat(43))).is_err());
        assert!(Pkce::from_verifier(&format!("{}é", "a".repeat(43))).is_err());
        assert!(Pkce::from_verifier(&format!("{}-._~", "a".repeat(43))).is_ok());
    }
}
