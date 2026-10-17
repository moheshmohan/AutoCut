//! Google Photos' sign-in, on the shared OAuth module ([`filmcraft_oauth`]): consent in the
//! system browser with PKCE (S256) and a loopback redirect on a random port, token refresh, and
//! the tokens kept in the macOS Keychain (an owner-only `tokens.json` elsewhere, or when the
//! Keychain is unavailable). A `tokens.json` from before the Keychain moves into it on first
//! use.
//!
//! The Photos Picker's read-only scope is always requested; the Library API's append-only scope
//! only when the user asks to upload (`photos auth --upload`), and the app-created read scope
//! with `--albums`, as incremental consent on top of what was granted before.

use std::path::Path;
use std::time::Duration;

use filmcraft_oauth::loopback::DonePage;
use filmcraft_oauth::{OAuthClient, OAuthError, Provider, SignInRequest, Stored, Summary, TokenVault};
use serde::Deserialize;
use serde_json::Value;

pub use filmcraft_oauth::TokenSet as Tokens;
pub use filmcraft_oauth::{truncate, urlencode};

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
/// The Keychain item's name (Keychain Access lists it so); its account is the data directory.
pub const KEYCHAIN_SERVICE: &str = "FilmCraft Google Photos";
/// How long [`consent`] waits for the browser to come back.
const CONSENT_WAIT: Duration = Duration::from_secs(300);

/// The `installed` object of a downloaded `client_secret.json`.
#[derive(Clone, Deserialize)]
pub struct ClientSecret {
    pub client_id: String,
    pub client_secret: String,
}

impl std::fmt::Debug for ClientSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientSecret").field("client_id", &self.client_id).field("client_secret", &"<redacted>").finish()
    }
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

    /// The OAuth client for Google's endpoints; `sign_in_command` is what errors tell the user
    /// to run.
    pub fn client(&self, sign_in_command: &str) -> OAuthClient {
        OAuthClient {
            provider: Provider::google(),
            client_id: self.client_id.clone(),
            client_secret: Some(self.client_secret.clone()),
            sign_in_command: sign_in_command.to_string(),
        }
    }
}

/// The scope checks on a sign-in: the tokens themselves, or the [`Summary`] readiness reads
/// without the Keychain.
pub trait Grants {
    /// Whether Google granted `scope` (consent screens let the user untick scopes).
    fn granted(&self, scope: &str) -> bool;

    /// The upload permission ([`UPLOAD_SCOPE`]), or the consent error naming the command that
    /// grants it.
    fn check_upload(&self) -> Result<(), GPhotosError> {
        if self.granted(UPLOAD_SCOPE) { Ok(()) } else { Err(GPhotosError::Consent(UPLOAD_CONSENT.into())) }
    }

    /// The album permissions ([`UPLOAD_SCOPE`] to create and add, [`ALBUM_SCOPE`] to find what
    /// FilmCraft made), or the consent error naming the command that grants them.
    fn check_albums(&self) -> Result<(), GPhotosError> {
        if self.granted(UPLOAD_SCOPE) && self.granted(ALBUM_SCOPE) { Ok(()) } else { Err(GPhotosError::Consent(ALBUMS_CONSENT.into())) }
    }

    /// The `photos auth` that signs in again with what this sign-in has.
    fn sign_in_command(&self) -> &'static str {
        Access::for_grants(self).sign_in_command()
    }
}

impl Grants for Tokens {
    fn granted(&self, scope: &str) -> bool {
        self.grants(scope)
    }
}

impl Grants for Summary {
    fn granted(&self, scope: &str) -> bool {
        self.grants(scope)
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

    /// The command that asks for this access.
    pub fn sign_in_command(self) -> &'static str {
        match self {
            Access::Pick => "filmcraft-cli photos auth",
            Access::PickAndUpload => "filmcraft-cli photos auth --upload",
            Access::PickUploadAndAlbums => "filmcraft-cli photos auth --albums",
        }
    }

    /// The widest access a sign-in already has (what to ask for again when it lapses).
    pub fn for_grants(g: &(impl Grants + ?Sized)) -> Self {
        if g.check_albums().is_ok() {
            Access::PickUploadAndAlbums
        } else if g.granted(UPLOAD_SCOPE) {
            Access::PickAndUpload
        } else {
            Access::Pick
        }
    }

    /// The consent: every scope of `self`, offline access (a refresh token), consent asked
    /// again (so Google sends a refresh token every time), and `include_granted_scopes` so the
    /// new token keeps what an earlier consent granted.
    pub fn sign_in_request(self) -> SignInRequest {
        SignInRequest {
            scopes: self.scopes().iter().map(|s| s.to_string()).collect(),
            extra: [("access_type", "offline"), ("include_granted_scopes", "true"), ("prompt", "consent")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            wait: CONSENT_WAIT,
            page: DonePage { app: "FilmCraft".into(), service: "Google Photos".into() },
        }
    }
}

/// Where this data directory's sign-in lives: the Keychain item [`KEYCHAIN_SERVICE`] on a Mac,
/// `dir/tokens.json` (owner-only) elsewhere or with `FILMCRAFT_SECRET_STORE=file`.
pub fn vault(dir: &Path) -> TokenVault {
    TokenVault::system(dir, KEYCHAIN_SERVICE)
}

/// An OAuth failure as the Google Photos error a user acts on.
pub fn from_oauth(e: OAuthError) -> GPhotosError {
    match e {
        OAuthError::Client(m) | OAuthError::Store(m) => GPhotosError::Setup(m),
        other => GPhotosError::Auth(other.to_string()),
    }
}

/// The saved tokens (moving a plaintext `tokens.json` into the Keychain on first use) and where
/// they are; a setup error naming `photos auth` when there are none, or when they are for the
/// retired Library permission.
pub fn load(vault: &TokenVault) -> Result<(Tokens, Stored), GPhotosError> {
    let Some((tokens, stored)) = vault.load().map_err(from_oauth)? else {
        return Err(GPhotosError::Setup("not signed in — run `filmcraft-cli photos auth` first".into()));
    };
    if !tokens.grants(SCOPE) {
        return Err(GPhotosError::Setup(
            "the saved Google sign-in is for the retired Photos Library permission — run `filmcraft-cli photos auth` again to grant the Photos Picker one"
                .into(),
        ));
    }
    Ok((tokens, stored))
}

/// What the saved sign-in grants, from disk alone (no Keychain, no network): readiness asks it
/// while drawing. `Err` says what to run.
pub fn granted(dir: &Path) -> Result<Summary, String> {
    match vault(dir).summary() {
        Ok(Some(s)) if s.grants(SCOPE) => Ok(s),
        Ok(Some(_)) => Err("the saved Google sign-in is for the retired Photos Library permission — run `filmcraft-cli photos auth` again".into()),
        Ok(None) => Err("not signed in — run `filmcraft-cli photos auth` first".into()),
        Err(e) => Err(e.to_string()),
    }
}

/// Run the consent flow for `access`: open the browser, catch the redirect on the loopback
/// listener (for at most five minutes), exchange the code with the PKCE verifier, check what was
/// granted, keep the tokens. Progress goes to stderr: stdout may be a protocol stream.
pub fn consent(secret: &ClientSecret, dir: &Path, access: Access) -> Result<(Tokens, Stored), GPhotosError> {
    let client = secret.client(access.sign_in_command());
    let tokens = client
        .sign_in(&filmcraft_oauth::http_agent(), &access.sign_in_request(), &mut |url| crate::open_in_browser(url), &mut |m| eprintln!("{m}"))
        .map_err(from_oauth)?;
    if tokens.refresh_token.is_none() {
        return Err(GPhotosError::Auth(format!("Google returned no refresh token; run `{}` again", access.sign_in_command())));
    }
    check_granted(&tokens, access)?;
    let stored = vault(dir).save(&tokens).map_err(from_oauth)?;
    Ok((tokens, stored))
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

/// Use the refresh token for a new access token. A lapsed or revoked sign-in says to sign in
/// again with the `photos auth` that grants what it had.
pub fn refresh(agent: &ureq::Agent, secret: &ClientSecret, tokens: &mut Tokens) -> Result<(), GPhotosError> {
    *tokens = secret.client(tokens.sign_in_command()).refresh(agent, tokens).map_err(from_oauth)?;
    Ok(())
}

/// What [`sign_out`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedOut {
    /// There was a sign-in to remove.
    pub was_signed_in: bool,
    /// Google confirmed the revocation; `Err` says why not (the local copy is gone anyway).
    pub revoked: Result<(), String>,
}

/// Revoke the saved sign-in at Google (best effort) and forget it here (Keychain item,
/// `tokens.json`, `signin.json`).
pub fn sign_out(agent: &ureq::Agent, dir: &Path) -> Result<SignedOut, GPhotosError> {
    const BY_HAND: &str = "remove FilmCraft's access at https://myaccount.google.com/connections";
    let vault = vault(dir);
    // an unreadable sign-in (a corrupt file, a locked Keychain) is still forgotten here
    let (was_signed_in, revoked) = match vault.load() {
        Ok(Some((t, _))) => {
            let token = t.refresh_token.as_deref().unwrap_or(&t.access_token);
            let client = OAuthClient {
                provider: Provider::google(),
                client_id: String::new(),
                client_secret: None,
                sign_in_command: Access::Pick.sign_in_command().into(),
            };
            (true, client.revoke(agent, token).map_err(|e| format!("{e}; {BY_HAND}")))
        }
        Ok(None) => (false, Ok(())),
        Err(e) => (true, Err(format!("the saved sign-in could not be read ({e}), so Google was not asked to revoke it; {BY_HAND}"))),
    };
    vault.remove().map_err(from_oauth)?;
    Ok(SignedOut { was_signed_in, revoked })
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_oauth::{MemoryStore, Place};
    use std::sync::Arc;

    fn t(scope: &str) -> Tokens {
        Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: u64::MAX, scope: scope.into() }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gphotos-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn library_era_tokens_ask_for_a_fresh_sign_in() {
        let dir = temp_dir("old-scope");
        let vault = TokenVault::file_only(&dir);
        // what the pre-2025 code saved: no scope field at all
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","refresh_token":"r","expires_at":1}"#).unwrap();
        let err = load(&vault).unwrap_err().to_string();
        assert!(err.contains("photos auth"), "{err}");
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","scope":"https://www.googleapis.com/auth/photoslibrary.readonly"}"#).unwrap();
        assert!(load(&vault).is_err());
        std::fs::write(dir.join("tokens.json"), r#"{"access_token":"a","expires_at":-5}"#).unwrap();
        assert!(load(&vault).is_err(), "corrupt numbers are an error, not a crash");
        std::fs::remove_file(dir.join("tokens.json")).unwrap();
        assert!(load(&vault).unwrap_err().to_string().contains("not signed in"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `tokens.json` from before the Keychain: moved into the (here: fake) secret store on the
    /// first load, the plaintext file gone, readiness still answered from disk.
    #[test]
    fn a_plaintext_sign_in_moves_into_the_secret_store() {
        let dir = temp_dir("migrate");
        std::fs::write(dir.join("tokens.json"), format!(r#"{{"access_token":"ya29","refresh_token":"1//r","expires_at":1,"scope":"{SCOPE} {UPLOAD_SCOPE}"}}"#))
            .unwrap();
        let vault = TokenVault::with_store(&dir, Arc::new(MemoryStore::new()));
        let (tokens, stored) = load(&vault).unwrap();
        assert!(tokens.check_upload().is_ok() && tokens.refresh_token.as_deref() == Some("1//r"));
        assert_eq!(stored.place, Place::Secure("memory".into()));
        assert!(!dir.join("tokens.json").exists());
        let summary = vault.summary().unwrap().unwrap();
        assert!(summary.check_upload().is_ok() && summary.check_albums().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn readiness_reads_the_grants_from_disk() {
        let dir = temp_dir("granted");
        assert!(granted(&dir).unwrap_err().contains("not signed in"));
        std::fs::write(dir.join("tokens.json"), "{not json").unwrap();
        assert!(granted(&dir).unwrap_err().contains("corrupt"));
        TokenVault::file_only(&dir).save(&t(&format!("{SCOPE} {UPLOAD_SCOPE}"))).unwrap();
        assert!(granted(&dir).unwrap().check_upload().is_ok());
        TokenVault::file_only(&dir).save(&t("https://www.googleapis.com/auth/photoslibrary.readonly")).unwrap();
        assert!(granted(&dir).unwrap_err().contains("retired"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The upload scope is asked for only with `--upload`, as incremental consent that keeps
    /// what was granted before.
    #[test]
    fn consents_ask_for_uploads_only_when_wanted() {
        let secret = ClientSecret { client_id: "id.apps.googleusercontent.com".into(), client_secret: "s".into() };
        let url = |a: Access| {
            secret.client(a.sign_in_command()).authorization_url("http://127.0.0.1:5000", &a.sign_in_request().scopes, "st", "ch", &a.sign_in_request().extra)
        };
        let pick = url(Access::Pick);
        assert!(pick.starts_with("https://accounts.google.com/o/oauth2/v2/auth?"), "{pick}");
        for part in ["include_granted_scopes=true", "access_type=offline", "prompt=consent", "code_challenge_method=S256", "state=st"] {
            assert!(pick.contains(part), "{part}: {pick}");
        }
        assert!(pick.contains(&format!("scope={}&", urlencode(SCOPE))), "{pick}");
        assert!(!pick.contains("photoslibrary"), "picker-only users are never asked for the Library API: {pick}");
        let up = url(Access::PickAndUpload);
        assert!(up.contains(&format!("scope={}%20{}&", urlencode(SCOPE), urlencode(UPLOAD_SCOPE))), "{up}");
        assert!(up.contains("include_granted_scopes=true"), "{up}");
        assert_eq!(Access::default(), Access::Pick);
        assert_eq!(Access::Pick.sign_in_request().wait, CONSENT_WAIT);
        assert_eq!(Access::PickAndUpload.sign_in_request().page.service, "Google Photos");
    }

    #[test]
    fn upload_permission_is_checked() {
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
    }

    /// Albums need the upload scope (create, add) and the app-created read scope (find what
    /// FilmCraft made); `--albums` asks for both on top of the picker, incrementally.
    #[test]
    fn album_permission_is_checked_and_asked_for_only_with_albums() {
        for lacking in [SCOPE.to_string(), format!("{SCOPE} {UPLOAD_SCOPE}"), format!("{SCOPE} {ALBUM_SCOPE}")] {
            let e = t(&lacking).check_albums().unwrap_err();
            assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("filmcraft-cli photos auth --albums"), "{lacking}: {e}");
        }
        let all = t(&format!("{SCOPE} {UPLOAD_SCOPE} {ALBUM_SCOPE}"));
        assert!(all.check_albums().is_ok() && all.check_upload().is_ok());
        assert!(check_granted(&all, Access::PickUploadAndAlbums).is_ok());
        let e = check_granted(&t(&format!("{SCOPE} {UPLOAD_SCOPE}")), Access::PickUploadAndAlbums).unwrap_err();
        assert!(e.to_string().contains("--albums"), "{e}");
        let scopes = Access::PickUploadAndAlbums.sign_in_request().scopes;
        assert_eq!(scopes, [SCOPE, UPLOAD_SCOPE, ALBUM_SCOPE]);
        // the other consents never ask for it
        for a in [Access::Pick, Access::PickAndUpload] {
            assert!(!a.sign_in_request().scopes.iter().any(|s| s.contains("appcreateddata")), "{a:?}");
        }
        // and it is never one of the retired library-wide scopes
        assert!(!ALBUM_SCOPE.ends_with("photoslibrary.readonly") && !ALBUM_SCOPE.ends_with("photoslibrary"));
    }

    /// A lapsed sign-in is answered with the `photos auth` that grants what it had.
    #[test]
    fn signing_in_again_keeps_the_access_it_had() {
        assert_eq!(t(SCOPE).sign_in_command(), "filmcraft-cli photos auth");
        assert_eq!(t(&format!("{SCOPE} {UPLOAD_SCOPE}")).sign_in_command(), "filmcraft-cli photos auth --upload");
        assert_eq!(t(&format!("{SCOPE} {UPLOAD_SCOPE} {ALBUM_SCOPE}")).sign_in_command(), "filmcraft-cli photos auth --albums");
        let e = from_oauth(OAuthError::SignInAgain { provider: "Google".into(), command: "filmcraft-cli photos auth --upload".into(), why: String::new() });
        assert!(matches!(e, GPhotosError::Auth(_)) && e.to_string().contains("sign in again: `filmcraft-cli photos auth --upload`"), "{e}");
        assert!(matches!(from_oauth(OAuthError::Store("x".into())), GPhotosError::Setup(_)));
        assert!(!format!("{:?}", ClientSecret { client_id: "i".into(), client_secret: "hush".into() }).contains("hush"));
    }
}
