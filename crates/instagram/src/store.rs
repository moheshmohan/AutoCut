//! Where the connected account lives: `<data dir>/instagram/account.json` (who it is, how it was
//! connected, when its token was last refreshed and runs out — no secret) and its token in a
//! [`TokenVault`] over `<data dir>/instagram/account/` (the Keychain item **FilmCraft Instagram**
//! on a Mac, an owner-only `tokens.json` elsewhere).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use filmcraft_oauth::{SecretStore, TokenVault};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{IgError, write_private};

/// The Keychain item (service) of the account's token.
pub const TOKEN_SERVICE: &str = "FilmCraft Instagram";

/// How the account was connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Via {
    /// Instagram's business login in the browser (`instagram auth`).
    #[default]
    Login,
    /// A token generated in the Meta App Dashboard and pasted (`instagram token`).
    Token,
}

/// The connected account (no secret).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Account {
    /// The Instagram professional account id content is published under (`user_id` of `/me`).
    pub user_id: String,
    pub username: String,
    pub name: String,
    /// `BUSINESS` or `MEDIA_CREATOR`.
    pub account_type: String,
    pub via: Via,
    /// The permissions granted, comma-separated as Instagram lists them.
    pub permissions: String,
    pub connected_at: u64,
    /// When the token was issued or last refreshed (Unix seconds).
    pub refreshed_at: u64,
    /// When it runs out unless refreshed.
    pub expires_at: u64,
}

impl Account {
    /// "@moheshmohan".
    pub fn label(&self) -> String {
        if self.username.is_empty() { format!("account {}", self.user_id) } else { format!("@{}", self.username) }
    }

    pub fn grants(&self, permission: &str) -> bool {
        self.permissions.split([',', ' ']).any(|p| p.trim() == permission)
    }

    pub fn to_json(&self, now: u64) -> Value {
        let left = self.expires_at.saturating_sub(now);
        json!({
            "userId": self.user_id, "username": self.username, "name": self.name, "accountType": self.account_type,
            "via": self.via, "permissions": self.permissions.split([',', ' ']).filter(|p| !p.trim().is_empty()).map(str::trim).collect::<Vec<_>>(),
            "connectedAt": self.connected_at, "refreshedAt": self.refreshed_at, "expiresAt": self.expires_at,
            "expiresInDays": left / 86_400, "expired": self.expires_at <= now,
        })
    }
}

/// Which secret store the token goes to.
#[derive(Clone)]
pub enum Secrets {
    /// The Keychain on a Mac (unless `FILMCRAFT_SECRET_STORE=file`), owner-only files elsewhere.
    System,
    /// Owner-only files only.
    Files,
    /// This store (tests: memory).
    Given(Arc<dyn SecretStore>),
}

/// The account file and its token vault.
pub struct AccountStore {
    dir: PathBuf,
    secrets: Secrets,
}

/// Instagram user ids are numbers; anything else in the file is damage.
pub fn check_user_id(id: &str) -> Result<(), IgError> {
    if id.is_empty() || id.len() > 32 || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(IgError::Api(format!("Instagram answered with an account id that is not one (`{}`)", crate::clean(id, 40))));
    }
    Ok(())
}

impl AccountStore {
    pub fn new(dir: &Path, secrets: Secrets) -> Self {
        Self { dir: dir.to_path_buf(), secrets }
    }

    fn file(&self) -> PathBuf {
        self.dir.join("account.json")
    }

    /// The account's token vault.
    pub fn vault(&self) -> TokenVault {
        let d = self.dir.join("account");
        match &self.secrets {
            Secrets::System => TokenVault::system(&d, TOKEN_SERVICE),
            Secrets::Files => TokenVault::file_only(&d),
            Secrets::Given(s) => TokenVault::with_store(&d, s.clone()),
        }
    }

    /// The connected account, `None` when there is none.
    pub fn load(&self) -> Result<Option<Account>, IgError> {
        let path = self.file();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(IgError::Setup(format!("cannot read {}: {e}", path.display()))),
        };
        if bytes.len() > 64 * 1024 {
            return Err(IgError::Setup(format!("{} is too large to be FilmCraft's; remove it and connect again", path.display())));
        }
        let a: Account = serde_json::from_slice(&bytes)
            .map_err(|e| IgError::Setup(format!("{} is damaged ({e}); remove it and connect again: `filmcraft-cli instagram auth`", path.display())))?;
        check_user_id(&a.user_id).map_err(|_| IgError::Setup(format!("{} names no account; connect again: `filmcraft-cli instagram auth`", path.display())))?;
        Ok(Some(a))
    }

    pub fn save(&self, a: &Account) -> Result<(), IgError> {
        let path = self.file();
        write_private(&path, &serde_json::to_vec_pretty(a).unwrap_or_default()).map_err(|e| IgError::Setup(format!("cannot write {}: {e}", path.display())))
    }

    /// Forget the account here: its token and its file.
    pub fn remove(&self) -> Result<(), IgError> {
        let r = self.vault().remove().map_err(|e| IgError::Setup(e.to_string()));
        match std::fs::remove_file(self.file()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(IgError::Setup(format!("cannot remove {}: {e}", self.file().display()))),
            _ => {}
        }
        let _ = std::fs::remove_dir(self.dir.join("account"));
        r
    }

    /// Whether a token is kept for the account (from disk; never the Keychain: asked while drawing).
    pub fn has_token(&self) -> bool {
        self.vault().summary().ok().flatten().is_some()
    }
}
