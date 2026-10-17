//! YouTube's sign-in, on the shared OAuth module ([`filmcraft_oauth`]): consent in the system
//! browser with PKCE (S256) and a loopback redirect on a random port; the OAuth client is the
//! user's own Desktop client — the one Google Photos uses (`google-photos/client_secret.json`),
//! or one in `youtube/`.
//!
//! Each sign-in connects **one channel**: with YouTube's scopes Google's account chooser lists
//! the Google account and every channel it manages (Brand Accounts included), and the token it
//! returns acts as the channel picked there. So connecting several channels is several sign-ins,
//! each kept apart ([`crate::store`]).

use std::path::{Path, PathBuf};
use std::time::Duration;

use filmcraft_oauth::loopback::DonePage;
use filmcraft_oauth::{OAuthClient, OAuthError, Provider, SignInRequest};
use serde::Deserialize;
use serde_json::Value;

use crate::YtError;

/// Upload videos (and set their thumbnails).
pub const UPLOAD_SCOPE: &str = "https://www.googleapis.com/auth/youtube.upload";
/// Read the channel (`channels.list?mine=true`, its playlists).
pub const READ_SCOPE: &str = "https://www.googleapis.com/auth/youtube.readonly";
/// Manage the account's videos, playlists and captions: what `playlistItems.insert` needs.
/// Asked for only by `youtube auth --playlists`.
pub const MANAGE_SCOPE: &str = "https://www.googleapis.com/auth/youtube.force-ssl";
/// What signs in.
pub const SIGN_IN: &str = "filmcraft-cli youtube auth";
/// What signs in with the playlist permission.
pub const SIGN_IN_PLAYLISTS: &str = "filmcraft-cli youtube auth --playlists";
/// How long a sign-in waits for the browser to come back.
pub const CONSENT_WAIT: Duration = Duration::from_secs(300);
/// The client file.
pub const CLIENT_FILE: &str = "client_secret.json";
/// The largest client file read.
const MAX_CLIENT_FILE: u64 = 64 * 1024;

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
    /// The first `client_secret.json` of `dirs` (`youtube/`, then `google-photos/`), and where it was.
    pub fn find(dirs: &[PathBuf]) -> Result<(Self, PathBuf), YtError> {
        for d in dirs {
            let path = d.join(CLIENT_FILE);
            if path.is_file() {
                return Ok((Self::load(&path)?, path));
            }
        }
        Err(YtError::Setup(missing_client(dirs)))
    }

    /// Whether one of `dirs` has a client file (readiness: no parsing).
    pub fn present(dirs: &[PathBuf]) -> Result<PathBuf, String> {
        dirs.iter().map(|d| d.join(CLIENT_FILE)).find(|p| p.is_file()).ok_or_else(|| missing_client(dirs))
    }

    pub fn load(path: &Path) -> Result<Self, YtError> {
        use std::io::Read as _;
        let mut text = String::new();
        std::fs::File::open(path)
            .and_then(|f| f.take(MAX_CLIENT_FILE).read_to_string(&mut text))
            .map_err(|e| YtError::Setup(format!("{}: {e}", path.display())))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| YtError::Setup(format!("{} is not valid JSON: {e}", path.display())))?;
        let installed = v
            .get("installed")
            .or_else(|| v.get("web"))
            .ok_or_else(|| YtError::Setup(format!("{} has no `installed` object (create a Desktop client, not a Web one)", path.display())))?;
        let s: Self = serde_json::from_value(installed.clone()).map_err(|e| YtError::Setup(format!("{} is missing a field: {e}", path.display())))?;
        if s.client_id.trim().is_empty() {
            return Err(YtError::Setup(format!("{} has an empty client_id", path.display())));
        }
        Ok(s)
    }

    /// The OAuth client; `sign_in_command` is what errors tell the user to run.
    pub fn client(&self, provider: Provider, sign_in_command: &str) -> OAuthClient {
        OAuthClient {
            provider,
            client_id: self.client_id.clone(),
            client_secret: Some(self.client_secret.clone()),
            sign_in_command: sign_in_command.to_string(),
        }
    }
}

fn missing_client(dirs: &[PathBuf]) -> String {
    let places: Vec<String> = dirs.iter().map(|d| d.join(CLIENT_FILE).display().to_string()).collect();
    format!(
        "no OAuth client for YouTube: save your Google Cloud project's Desktop client as {} (the Google Photos one works: enable the YouTube Data API v3 in the same project; see crates/youtube/README.md)",
        places.join(" or ")
    )
}

/// The scopes of a sign-in.
pub fn scopes(playlists: bool) -> Vec<String> {
    let mut s = vec![UPLOAD_SCOPE.to_string(), READ_SCOPE.to_string()];
    if playlists {
        s.push(MANAGE_SCOPE.to_string());
    }
    s
}

/// The `youtube auth` that grants what a sign-in with `scope` had.
pub fn sign_in_command(scope: &str) -> &'static str {
    if scope.split_whitespace().any(|s| s == MANAGE_SCOPE) { SIGN_IN_PLAYLISTS } else { SIGN_IN }
}

/// The consent: the scopes, offline access (a refresh token), and Google's account chooser
/// every time (`prompt=select_account consent`), where the channel is picked.
pub fn sign_in_request(playlists: bool, wait: Duration) -> SignInRequest {
    SignInRequest {
        scopes: scopes(playlists),
        extra: [("access_type", "offline"), ("include_granted_scopes", "true"), ("prompt", "select_account consent")]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        wait,
        page: DonePage { app: "FilmCraft".into(), service: "YouTube".into() },
    }
}

/// Every scope asked for was granted (consent screens let the user untick some).
pub fn check_granted(granted: &str, playlists: bool) -> Result<(), YtError> {
    let has = |s: &str| granted.split_whitespace().any(|g| g == s);
    if !has(UPLOAD_SCOPE) || !has(READ_SCOPE) {
        return Err(YtError::Consent(format!(
            "Google did not grant both YouTube permissions (manage your videos; view your account); run `{}` again and leave them ticked",
            if playlists { SIGN_IN_PLAYLISTS } else { SIGN_IN }
        )));
    }
    if playlists && !has(MANAGE_SCOPE) {
        return Err(YtError::Consent(format!(
            "Google did not grant the permission to manage your YouTube account (for playlists); run `{SIGN_IN_PLAYLISTS}` again and leave it ticked"
        )));
    }
    Ok(())
}

/// An OAuth failure as the error a user acts on.
pub fn from_oauth(e: OAuthError, channel: Option<&str>) -> YtError {
    match e {
        OAuthError::Client(m) | OAuthError::Store(m) => YtError::Setup(m),
        OAuthError::SignInAgain { .. } => {
            let pick = channel.map(|c| format!(" (pick {c} in Google's account chooser)")).unwrap_or_default();
            YtError::Auth(format!("{e}{pick}"))
        }
        other => YtError::Auth(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_consent_asks_for_youtube_and_the_chooser() {
        let secret = ClientSecret { client_id: "id.apps.googleusercontent.com".into(), client_secret: "hush".into() };
        let req = sign_in_request(false, CONSENT_WAIT);
        let url = secret.client(Provider::google(), SIGN_IN).authorization_url("http://127.0.0.1:5000", &req.scopes, "st", "ch", &req.extra);
        for part in ["youtube.upload", "youtube.readonly", "prompt=select_account%20consent", "access_type=offline", "code_challenge_method=S256"] {
            assert!(url.contains(part), "{part}: {url}");
        }
        assert!(!url.contains("force-ssl") && !url.contains("hush"), "{url}");
        assert!(sign_in_request(true, CONSENT_WAIT).scopes.iter().any(|s| s == MANAGE_SCOPE));
        assert_eq!(req.page.service, "YouTube");
        assert!(!format!("{secret:?}").contains("hush"));
    }

    #[test]
    fn grants_are_checked() {
        let both = format!("{UPLOAD_SCOPE} {READ_SCOPE}");
        assert!(check_granted(&both, false).is_ok());
        assert!(check_granted(UPLOAD_SCOPE, false).unwrap_err().to_string().contains("youtube auth"));
        assert!(check_granted(&both, true).unwrap_err().to_string().contains("--playlists"));
        assert!(check_granted(&format!("{both} {MANAGE_SCOPE}"), true).is_ok());
        assert_eq!(sign_in_command(&format!("{both} {MANAGE_SCOPE}")), SIGN_IN_PLAYLISTS);
        assert_eq!(sign_in_command(&both), SIGN_IN);
        let e = from_oauth(OAuthError::SignInAgain { provider: "Google".into(), command: SIGN_IN.into(), why: String::new() }, Some("H4hacks"));
        assert!(matches!(e, YtError::Auth(_)) && e.to_string().contains("pick H4hacks"), "{e}");
    }

    #[test]
    fn client_files_are_found_and_checked() {
        let base = std::env::temp_dir().join(format!("filmcraft-yt-client-{}", std::process::id()));
        let (yt, gp) = (base.join("youtube"), base.join("google-photos"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&yt).unwrap();
        std::fs::create_dir_all(&gp).unwrap();
        let dirs = vec![yt.clone(), gp.clone()];
        let e = ClientSecret::find(&dirs).unwrap_err().to_string();
        assert!(e.contains("youtube/client_secret.json") && e.contains("google-photos/client_secret.json"), "{e}");
        assert!(ClientSecret::present(&dirs).is_err());
        std::fs::write(gp.join(CLIENT_FILE), r#"{"installed":{"client_id":"gp.apps","client_secret":"s"}}"#).unwrap();
        let (s, at) = ClientSecret::find(&dirs).unwrap();
        assert_eq!((s.client_id.as_str(), at), ("gp.apps", gp.join(CLIENT_FILE)));
        std::fs::write(yt.join(CLIENT_FILE), r#"{"installed":{"client_id":"yt.apps","client_secret":"s"}}"#).unwrap();
        assert_eq!(ClientSecret::find(&dirs).unwrap().0.client_id, "yt.apps", "youtube's own comes first");
        for bad in ["", "{", "[]", r#"{"other":{}}"#, r#"{"installed":{"client_id":"x"}}"#, r#"{"installed":{"client_id":" ","client_secret":"s"}}"#] {
            std::fs::write(yt.join(CLIENT_FILE), bad).unwrap();
            assert!(matches!(ClientSecret::find(&dirs), Err(YtError::Setup(_))), "{bad}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
