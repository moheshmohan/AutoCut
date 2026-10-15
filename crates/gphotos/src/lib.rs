//! Google Photos for the editor: pick cloud media in Google's own picker and import it.
//!
//! - [`auth`]: the OAuth 2.0 installed-app flow (browser + loopback redirect) and the token
//!   store, under `<data dir>/google-photos/`. The one scope is the Photos Picker's read-only
//!   one.
//! - [`api`]: the Photos Picker API client — picking sessions, polling, the picked items.
//! - [`cache`]: downloads originals into a local cache so an item is fetched only once.
//! - [`GooglePhotos::pick`]: the whole flow — open a session, send the user to the picker,
//!   wait (bounded) for their selection, download it, delete the session.
//!
//! Google retired the Library API's library-wide read scopes on 2025-03-31, so there is no
//! server-side search or album browsing any more: the user chooses in the picker.
//!
//! Setup (one time): create a Google Cloud project, enable the Photos Picker API, add an OAuth
//! *Desktop* client, keep yourself as a Testing user, and save the downloaded
//! `client_secret.json` into `<data dir>/google-photos/`. Then run `filmcraft-cli photos auth`.
//! See README.md for the step-by-step.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod api;
pub mod auth;
pub mod cache;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use api::{PickedItem, PickingSession, Poll};

/// Everything Google-Photos-related lives here: `client_secret.json`, `tokens.json`, and the
/// downloaded-media cache.
pub fn data_dir() -> PathBuf {
    filmcraft_engine::autosave::default_data_dir().unwrap_or_else(|| PathBuf::from(".")).join("google-photos")
}

/// The longest a single [`GooglePhotos::pick`] waits for the user.
pub const MAX_WAIT: Duration = Duration::from_secs(3600);
/// A hard stop for the polling loop, whatever the clock does.
const MAX_POLLS: usize = 10_000;
/// Consecutive failed polls (network blips) tolerated before giving up.
const MAX_POLL_FAILURES: u32 = 3;

/// A configured client: credentials, fresh tokens, and the HTTP agent.
#[derive(Clone)]
pub struct GooglePhotos {
    pub secret: auth::ClientSecret,
    tokens: std::sync::Arc<std::sync::Mutex<auth::Tokens>>,
    agent: ureq::Agent,
}

/// How [`GooglePhotos::pick`] runs.
#[derive(Debug, Clone)]
pub struct PickOptions {
    /// At most this many items selectable (1–2000; Google's default is 2000).
    pub max_items: Option<u32>,
    /// How long to wait for the user (capped at [`MAX_WAIT`]); zero returns at once with the
    /// session pending, for a caller that resumes it later.
    pub wait: Duration,
    /// Resume this session (from an earlier [`PickOutcome::Pending`]) instead of opening one.
    pub resume: Option<String>,
    /// Open the picker in the default browser (otherwise only report the link).
    pub open_browser: bool,
}

impl Default for PickOptions {
    fn default() -> Self {
        Self { max_items: None, wait: Duration::from_secs(600), resume: None, open_browser: true }
    }
}

/// One picked item, now on disk.
#[derive(Debug, Clone)]
pub struct Fetched {
    pub item: PickedItem,
    pub path: PathBuf,
    /// It was already in the cache: nothing was downloaded.
    pub was_cached: bool,
}

/// The result of a finished pick.
#[derive(Debug, Clone, Default)]
pub struct Picked {
    pub fetched: Vec<Fetched>,
    /// `"<filename>: <reason>"` for each item that could not be downloaded.
    pub failed: Vec<String>,
    /// Google confirmed the session's deletion (it expires on its own otherwise).
    pub session_deleted: bool,
}

#[derive(Debug, Clone)]
pub enum PickOutcome {
    /// The user finished picking; the items are downloaded (or were cached).
    Done(Picked),
    /// The wait ran out before the user pressed Done. The session stays open: pass its id
    /// back as [`PickOptions::resume`] to keep waiting.
    Pending(PickingSession),
}

/// A picking session that is open (and, with `open_browser`, already on screen) — the
/// intermission between [`GooglePhotos::prepare`] and [`GooglePhotos::complete`] where a
/// caller may drive the picker UI itself.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub session: PickingSession,
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

    /// The whole pick: open (or resume) a picking session, send the user to Google Photos,
    /// wait for their selection within `opts.wait`, download every picked item into the
    /// cache, and delete the session. `progress` gets one human-readable line per step (the
    /// CLI prints them to stderr; stdout may be a protocol stream, so this never prints).
    pub fn pick(&self, opts: &PickOptions, progress: &mut dyn FnMut(&str)) -> Result<PickOutcome, GPhotosError> {
        let prepared = self.prepare(opts, progress)?;
        self.complete(prepared, opts.wait, progress)
    }

    /// Step one of [`Self::pick`]: open (or resume) a picking session and surface the picker
    /// link. Split out so a caller can interpose — e.g. let a screen agent drive the picker
    /// UI between this and [`Self::complete`] — while `pick` remains the one-call form.
    pub fn prepare(&self, opts: &PickOptions, progress: &mut dyn FnMut(&str)) -> Result<Prepared, GPhotosError> {
        let session = match &opts.resume {
            Some(id) => {
                let s = self.session(id)?;
                if !s.media_items_set {
                    progress(&format!("still waiting for your selection in Google Photos: {}", s.autoclose_uri()));
                }
                s
            }
            None => {
                let s = self.create_session(opts.max_items)?;
                let uri = s.autoclose_uri();
                if opts.open_browser && open_in_browser(&uri) {
                    progress(&format!("opened Google Photos in your browser — choose photos/videos and press Done ({uri})"));
                } else {
                    progress(&format!("open this link, choose photos/videos and press Done: {uri}"));
                }
                s
            }
        };
        Ok(Prepared { session })
    }

    /// Step two of [`Self::pick`]: wait for the selection within `wait`, download every
    /// picked item into the cache, and delete the session.
    pub fn complete(&self, prepared: Prepared, wait: Duration, progress: &mut dyn FnMut(&str)) -> Result<PickOutcome, GPhotosError> {
        let id = prepared.session.id.clone();
        let session = match self.wait_for_selection(prepared.session, wait.min(MAX_WAIT)) {
            Ok(Waited::Ready(s)) => s,
            Ok(Waited::OutOfTime(s)) => return Ok(PickOutcome::Pending(s)),
            Ok(Waited::Expired) => {
                let _ = self.delete_session(&id);
                return Err(GPhotosError::Api("the Google Photos picker timed out before anything was picked; start a new pick".into()));
            }
            Err(e) => {
                let _ = self.delete_session(&id);
                return Err(e);
            }
        };
        let fetched = self.fetch_picked(&session.id, progress);
        let session_deleted = self.delete_session(&session.id).is_ok();
        let mut picked = fetched?;
        picked.session_deleted = session_deleted;
        Ok(PickOutcome::Done(picked))
    }

    /// Poll `sessions.get` as Google advises until the user is done, Google gives up, or
    /// `budget` runs out.
    fn wait_for_selection(&self, mut session: PickingSession, budget: Duration) -> Result<Waited, GPhotosError> {
        let start = Instant::now();
        let mut failures = 0u32;
        for _ in 0..MAX_POLLS {
            match api::next_poll(&session, start.elapsed(), budget) {
                Poll::Ready => return Ok(Waited::Ready(session)),
                Poll::Expired => return Ok(Waited::Expired),
                Poll::OutOfTime => return Ok(Waited::OutOfTime(session)),
                Poll::Wait(d) => std::thread::sleep(d),
            }
            match self.session(&session.id) {
                Ok(s) => {
                    session = s;
                    failures = 0;
                }
                Err(_) if failures < MAX_POLL_FAILURES => failures += 1,
                Err(e) => return Err(e),
            }
        }
        Ok(Waited::OutOfTime(session))
    }

    /// List the session's picked items and bring each one into the cache.
    fn fetch_picked(&self, session_id: &str, progress: &mut dyn FnMut(&str)) -> Result<Picked, GPhotosError> {
        let items = self.picked_items(session_id)?;
        progress(&format!("{} item(s) picked", items.len()));
        let total = items.len();
        let mut picked = Picked::default();
        for (i, item) in items.into_iter().enumerate() {
            let was_cached = cache::cached_path(&item).is_some_and(|p| p.is_file()) || cache::find(&item.id).is_some();
            progress(&format!("[{}/{total}] {} {}", i + 1, if was_cached { "cached" } else { "downloading" }, item.display_name()));
            match self.local_copy(&item) {
                Ok(path) => picked.fetched.push(Fetched { item, path, was_cached }),
                Err(e) => picked.failed.push(format!("{}: {e}", item.display_name())),
            }
        }
        Ok(picked)
    }
}

enum Waited {
    Ready(PickingSession),
    Expired,
    OutOfTime(PickingSession),
}

/// Open an https link in the default browser, detached and silent (stdout may be an MCP
/// stream). Anything that is not `https://` is refused, so a hostile link can never become an
/// opener flag or a local file.
pub fn open_in_browser(url: &str) -> bool {
    if !url.starts_with("https://") || url.chars().any(char::is_whitespace) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_opener_refuses_anything_but_https() {
        for bad in ["", "-a Calculator", "file:///etc/passwd", "http://example.com", "https://x.example/a b", "javascript:alert(1)"] {
            assert!(!open_in_browser(bad), "{bad:?}");
        }
    }

    #[test]
    fn default_wait_is_bounded() {
        let o = PickOptions::default();
        assert!(o.wait <= MAX_WAIT && o.open_browser && o.resume.is_none());
    }
}
