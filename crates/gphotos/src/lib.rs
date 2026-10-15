//! Google Photos for the editor: browse, search and import cloud media.
//!
//! - [`auth`]: the OAuth 2.0 installed-app flow (browser + loopback redirect) and the token
//!   store, under `<data dir>/google-photos/`.
//! - [`api`]: the Library API client — albums, `mediaItems.search` (Google filters + local
//!   filename matching), item lookup.
//! - [`cache`]: downloads originals into a per-account cache so imports are instant the second
//!   time.
//!
//! Setup (one time): create a Google Cloud project, enable the Photos Library API, add an
//! OAuth *Desktop* client with redirect URI `http://localhost:8417`, keep yourself as a Testing
//! user, and save the downloaded `client_secret.json` into `<data dir>/google-photos/`. Then
//! run `filmcraft-cli photos auth`. See README.md for the step-by-step.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod api;
pub mod auth;
pub mod cache;

use std::path::PathBuf;

/// Everything Google-Photos-related lives here: `client_secret.json`, `tokens.json`, and the
/// downloaded-media cache.
pub fn data_dir() -> PathBuf {
    filmcraft_engine::autosave::default_data_dir().unwrap_or_else(|| PathBuf::from(".")).join("google-photos")
}

/// A configured client: credentials, fresh tokens, and the HTTP agent.
#[derive(Clone)]
pub struct GooglePhotos {
    pub secret: auth::ClientSecret,
    tokens: std::sync::Arc<std::sync::Mutex<auth::Tokens>>,
    agent: ureq::Agent,
}

impl GooglePhotos {
    /// Load credentials and tokens from the data dir, refreshing the access token if needed.
    pub fn configured() -> Result<Self, GPhotosError> {
        let dir = data_dir();
        let secret = auth::ClientSecret::load(&dir)?;
        let mut tokens = auth::Tokens::load(&dir)?;
        let agent = api::http_agent();
        if tokens.needs_refresh() {
            auth::refresh(&agent, &secret, &mut tokens)?;
            tokens.save(&dir)?;
        }
        Ok(Self { secret, tokens: std::sync::Arc::new(std::sync::Mutex::new(tokens)), agent })
    }

    /// An access token that is refreshed on demand (callers hold the lock only while copying).
    pub fn access_token(&self) -> Result<String, GPhotosError> {
        let mut tokens = self.tokens.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if tokens.needs_refresh() {
            auth::refresh(&self.agent, &self.secret, &mut tokens)?;
            tokens.save(&data_dir())?;
        }
        Ok(tokens.access_token.clone())
    }

    pub fn agent(&self) -> ureq::Agent {
        self.agent.clone()
    }
}

/// Google Photos failures, all human-readable.
#[derive(Debug, thiserror::Error)]
pub enum GPhotosError {
    #[error("{0}")]
    Setup(String),
    #[error("auth: {0}")]
    Auth(String),
    #[error("api: {0}")]
    Api(String),
    #[error("download: {0}")]
    Download(String),
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for GPhotosError {
    fn from(e: std::io::Error) -> Self {
        GPhotosError::Io(e.to_string())
    }
}
