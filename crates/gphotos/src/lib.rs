//! Google Photos for the editor: pick cloud media in Google's own picker and import it, and
//! upload finished exports to the user's library.
//!
//! - [`auth`]: the OAuth 2.0 installed-app flow (browser + loopback redirect) and the token
//!   store, under `<data dir>/google-photos/`. The Photos Picker's read-only scope always; the
//!   Library API's append-only scope only after `photos auth --upload` (incremental consent).
//! - [`api`]: the Photos Picker API client — picking sessions, polling, the picked items.
//! - [`cache`]: downloads originals into a local cache so an item is fetched only once.
//! - [`upload`]: the Library API's resumable upload and `mediaItems:batchCreate`, behind a
//!   [`upload::Transport`] the tests fake.
//! - [`albums`]: albums this app creates (create, list, find by title, add its uploads, a note
//!   at the top), over the same transport; `photos auth --albums` grants what they need.
//! - [`GooglePhotos::pick`]: the whole flow — open a session, send the user to the picker,
//!   wait (bounded) for their selection, download it, delete the session.
//! - [`GooglePhotos::upload`] and [`engine_uploader`]: upload a file (into an album, too); the
//!   latter is what hosts install as the engine's `photos.*` uploader.
//!
//! Google retired the Library API's library-wide read scopes on 2025-03-31, so there is no
//! server-side search or browsing of the user's library any more: the user chooses in the
//! picker. App uploads still work with the append-only scope, and an app may read back only the
//! albums and items it created itself (so items a user adds by hand to an album FilmCraft made
//! are reachable only through the picker).
//!
//! Setup (one time): create a Google Cloud project, enable the Photos Picker API (and, to upload,
//! the Photos Library API), add an OAuth *Desktop* client, keep yourself as a Testing user, and
//! save the downloaded `client_secret.json` into `<data dir>/google-photos/`. Then run
//! `filmcraft-cli photos auth` (`--upload` to allow uploads). See README.md for the step-by-step.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod albums;
pub mod api;
pub mod auth;
pub mod cache;
pub mod upload;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
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
    /// Skip picked items created before this date (`YYYY-MM-DD`): they are not downloaded.
    /// Lets a caller select a whole search result and keep only the recent part of it.
    pub since: Option<String>,
}

impl Default for PickOptions {
    fn default() -> Self {
        Self { max_items: None, wait: Duration::from_secs(600), resume: None, open_browser: true, since: None }
    }
}

/// Whether an item created at `created` (RFC 3339) is on or after `since` (`YYYY-MM-DD`).
/// An item with no usable date is kept: a missing timestamp is not a reason to drop media.
pub fn created_on_or_after(created: &str, since: &str) -> bool {
    match (parse_day(created), parse_day(since)) {
        (Some(c), Some(s)) => c >= s,
        _ => true,
    }
}

/// Whether `s` is a `YYYY-MM-DD` date (what [`PickOptions::since`] takes).
pub fn is_day(s: &str) -> bool {
    s.trim().len() == 10 && parse_day(s).is_some()
}

/// `(year, month, day)` from the first ten characters of an RFC 3339 / `YYYY-MM-DD` string.
fn parse_day(s: &str) -> Option<(u32, u32, u32)> {
    let d = s.trim().get(..10)?;
    let mut it = d.split('-');
    let y = it.next()?.parse().ok()?;
    let m = it.next()?.parse().ok()?;
    let dd = it.next()?.parse().ok()?;
    (it.next().is_none() && (1..=12).contains(&m) && (1..=31).contains(&dd)).then_some((y, m, dd))
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
    /// `"<filename>: created <date>"` for each item left out by [`PickOptions::since`].
    pub skipped: Vec<String>,
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
    /// [`PickOptions::since`], applied when the picks are fetched.
    pub since: Option<String>,
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
        Ok(Prepared { session, since: opts.since.clone().filter(|d| !d.trim().is_empty()) })
    }

    /// Step two of [`Self::pick`]: wait for the selection within `wait`, download every
    /// picked item into the cache, and delete the session.
    pub fn complete(&self, prepared: Prepared, wait: Duration, progress: &mut dyn FnMut(&str)) -> Result<PickOutcome, GPhotosError> {
        self.complete_unless(prepared, wait, &mut || false, progress)
    }

    /// [`Self::complete`], but the wait for the selection ends as soon as `give_up()` says so
    /// (asked after every poll while nothing is picked) — e.g. the screen agent driving the
    /// picker finished without pressing Done. The session is then [`PickOutcome::Pending`],
    /// still open for the user to finish by hand, instead of waiting out `wait`.
    pub fn complete_unless(
        &self,
        prepared: Prepared,
        wait: Duration,
        give_up: &mut dyn FnMut() -> bool,
        progress: &mut dyn FnMut(&str),
    ) -> Result<PickOutcome, GPhotosError> {
        let id = prepared.session.id.clone();
        let session = match self.wait_for_selection(prepared.session, wait.min(MAX_WAIT), give_up) {
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
        let fetched = self.fetch_picked(&session.id, prepared.since.as_deref(), progress);
        let session_deleted = self.delete_session(&session.id).is_ok();
        let mut picked = fetched?;
        picked.session_deleted = session_deleted;
        Ok(PickOutcome::Done(picked))
    }

    /// Poll `sessions.get` as Google advises until the user is done, Google gives up, or
    /// `budget` runs out, or `give_up()` while nothing is picked.
    fn wait_for_selection(&self, mut session: PickingSession, budget: Duration, give_up: &mut dyn FnMut() -> bool) -> Result<Waited, GPhotosError> {
        let start = Instant::now();
        let mut failures = 0u32;
        for _ in 0..MAX_POLLS {
            match api::next_poll(&session, start.elapsed(), budget) {
                Poll::Ready => return Ok(Waited::Ready(session)),
                Poll::Expired => return Ok(Waited::Expired),
                Poll::OutOfTime => return Ok(Waited::OutOfTime(session)),
                Poll::Wait(_) if give_up() => return Ok(Waited::OutOfTime(session)),
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
    fn fetch_picked(&self, session_id: &str, since: Option<&str>, progress: &mut dyn FnMut(&str)) -> Result<Picked, GPhotosError> {
        let items = self.picked_items(session_id)?;
        progress(&format!("{} item(s) picked", items.len()));
        let mut picked = Picked::default();
        let (items, older): (Vec<PickedItem>, Vec<PickedItem>) = items.into_iter().partition(|i| since.is_none_or(|d| created_on_or_after(&i.created, d)));
        if let Some(d) = since
            && !older.is_empty()
        {
            progress(&format!("skipping {} item(s) created before {d}", older.len()));
        }
        picked.skipped = older.iter().map(|i| format!("{}: created {}", i.display_name(), i.created.get(..10).unwrap_or(&i.created))).collect();
        let total = items.len();
        for (i, item) in items.into_iter().enumerate() {
            let was_cached = cache::cached_path(&item).is_some_and(|p| p.is_file()) || cache::find(&item.id).is_some();
            progress(&format!("[{}/{total}] {} {}", i + 1, if was_cached { "cached" } else { "downloading" }, item.display_name()));
            match self.local_copy(&item) {
                Ok(path) => {
                    // Google's re-encoded downloads drop the capture date: keep it beside the file
                    cache::write_sidecar(&item, &path);
                    picked.fetched.push(Fetched { item, path, was_cached })
                }
                Err(e) => picked.failed.push(format!("{}: {e}", item.display_name())),
            }
        }
        Ok(picked)
    }
}

impl GooglePhotos {
    /// Whether the saved sign-in includes the upload permission.
    pub fn can_upload(&self) -> bool {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).grants(auth::UPLOAD_SCOPE)
    }

    /// Whether the saved sign-in includes the album permissions (`photos auth --albums`).
    pub fn can_use_albums(&self) -> bool {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).check_albums().is_ok()
    }

    /// Upload `path` into the user's library (see [`upload`]). Without the upload permission
    /// this is a [`GPhotosError::Consent`] naming `filmcraft-cli photos auth --upload`; nothing
    /// is sent. With [`upload::UploadOptions::album`] (a title or an id) the item goes into that
    /// album too: an album FilmCraft made earlier with that exact title, else a new one (that
    /// needs `photos auth --albums`; it is resolved before any byte is sent).
    /// `progress(sent, total, status)` returning `false` cancels.
    pub fn upload(
        &self,
        path: &Path,
        opts: &upload::UploadOptions,
        progress: &mut dyn FnMut(u64, u64, &str) -> bool,
    ) -> Result<upload::Uploaded, GPhotosError> {
        self.upload_with(&upload::UreqTransport::default(), path, opts, progress)
    }

    /// [`Self::upload`] over another transport.
    pub fn upload_with(
        &self,
        http: &dyn upload::Transport,
        path: &Path,
        opts: &upload::UploadOptions,
        progress: &mut dyn FnMut(u64, u64, &str) -> bool,
    ) -> Result<upload::Uploaded, GPhotosError> {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).check_upload()?;
        let bearer = || self.access_token();
        let Some(album) = opts.album.as_deref().map(str::trim).filter(|a| !a.is_empty()) else {
            let plain = upload::UploadOptions { album: None, ..opts.clone() };
            return upload::upload_file(http, &bearer, path, &plain, progress);
        };
        // the file first: a file Google would refuse must not leave a new empty album behind
        upload::check_file(path)?;
        upload::check_description(opts.description.as_deref())?;
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).check_albums()?;
        if !progress(0, 0, "Finding the Google Photos album") {
            return Err(GPhotosError::Cancelled);
        }
        let mut calls = albums::Albums::new(http, &bearer);
        calls.backoff = opts.backoff;
        let ensured = calls.ensure(album, None)?;
        let into = upload::UploadOptions { album: Some(ensured.album.id.clone()), ..opts.clone() };
        let mut up = upload::upload_file(http, &bearer, path, &into, progress)?;
        up.album = Some(ensured.album);
        up.album_created = ensured.created;
        Ok(up)
    }

    /// Run `f` with the album calls over `http`, once the sign-in is known to have the album
    /// permissions ([`GPhotosError::Consent`] naming `photos auth --albums` otherwise; nothing
    /// is sent).
    pub fn albums_with<T>(&self, http: &dyn upload::Transport, f: impl FnOnce(&albums::Albums<'_>) -> Result<T, GPhotosError>) -> Result<T, GPhotosError> {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).check_albums()?;
        let bearer = || self.access_token();
        f(&albums::Albums::new(http, &bearer))
    }

    /// [`Self::albums_with`] over the real transport.
    pub fn albums<T>(&self, f: impl FnOnce(&albums::Albums<'_>) -> Result<T, GPhotosError>) -> Result<T, GPhotosError> {
        self.albums_with(&upload::UreqTransport::with_timeout(albums::TIMEOUT), f)
    }
}

/// Whether an upload can start, from what is on disk in `dir` (no network): the OAuth client,
/// a sign-in, and the upload permission. `Err` says what to do.
pub fn upload_readiness(dir: &Path) -> Result<(), String> {
    if !dir.join("client_secret.json").is_file() {
        return Err(format!(
            "Google Photos is not connected: save your OAuth client as {} (see crates/gphotos/README.md), then run `filmcraft-cli photos auth --upload`",
            dir.join("client_secret.json").display()
        ));
    }
    let tokens = auth::Tokens::load(dir).map_err(|_| "Google Photos is not connected: run `filmcraft-cli photos auth --upload`".to_string())?;
    tokens.check_upload().map_err(|e| e.to_string())
}

/// Whether album commands can run, from what is on disk in `dir` (no network): the OAuth
/// client, a sign-in, and the album permissions. `Err` says what to do.
pub fn album_readiness(dir: &Path) -> Result<(), String> {
    if !dir.join("client_secret.json").is_file() {
        return Err(format!(
            "Google Photos is not connected: save your OAuth client as {} (see crates/gphotos/README.md), then run `filmcraft-cli photos auth --albums`",
            dir.join("client_secret.json").display()
        ));
    }
    let tokens = auth::Tokens::load(dir).map_err(|_| "Google Photos is not connected: run `filmcraft-cli photos auth --albums`".to_string())?;
    tokens.check_albums().map_err(|e| e.to_string())
}

/// How long [`EngineUploader::ready`] trusts its last look at the disk (it runs while drawing).
const READY_TTL: Duration = Duration::from_secs(2);

/// The engine's `photos.*` uploader over the data dir's sign-in.
pub struct EngineUploader {
    dir: PathBuf,
    ready: Mutex<Option<(Instant, Result<(), String>)>>,
    albums_ready: Mutex<Option<(Instant, Result<(), String>)>>,
}

/// The uploader hosts install as `Session::photos`.
pub fn engine_uploader() -> Arc<dyn filmcraft_engine::photos::PhotosUploader> {
    Arc::new(EngineUploader::new(data_dir()))
}

impl EngineUploader {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, ready: Mutex::new(None), albums_ready: Mutex::new(None) }
    }

    /// An upload (into `album` when given) reporting into `progress`.
    fn upload_into(
        &self,
        path: &str,
        description: Option<&str>,
        album: Option<&str>,
        progress: &filmcraft_engine::export::Progress,
    ) -> Result<serde_json::Value, String> {
        use std::sync::atomic::Ordering;
        let gp = GooglePhotos::configured().map_err(|e| e.to_string())?;
        let opts = upload::UploadOptions { description: description.map(str::to_string), album: album.map(str::to_string), ..Default::default() };
        let mut report = |sent: u64, total: u64, status: &str| {
            // the album lookup reports no size yet: keep the file's size from the command
            if total > 0 {
                progress.total.store(total, Ordering::Relaxed);
                progress.done.store(sent, Ordering::Relaxed);
            }
            *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = status.to_string();
            !progress.cancel.load(Ordering::Relaxed)
        };
        gp.upload(Path::new(path), &opts, &mut report).map(|u| u.to_json()).map_err(|e| e.to_string())
    }
}

/// A readiness answer cached for [`READY_TTL`] (it is asked while drawing).
fn cached_readiness(slot: &Mutex<Option<(Instant, Result<(), String>)>>, check: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    let mut cached = slot.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((at, r)) = &*cached
        && at.elapsed() < READY_TTL
    {
        return r.clone();
    }
    let r = check();
    *cached = Some((Instant::now(), r.clone()));
    r
}

impl filmcraft_engine::photos::PhotosUploader for EngineUploader {
    fn ready(&self) -> Result<(), String> {
        cached_readiness(&self.ready, || upload_readiness(&self.dir))
    }

    fn accepts(&self, path: &str) -> Result<(), String> {
        upload::media_type(Path::new(path)).map(drop).map_err(|e| e.to_string())
    }

    fn upload(&self, path: &str, description: Option<&str>, progress: &filmcraft_engine::export::Progress) -> Result<serde_json::Value, String> {
        self.upload_into(path, description, None, progress)
    }

    fn albums_ready(&self) -> Result<(), String> {
        cached_readiness(&self.albums_ready, || album_readiness(&self.dir))
    }

    fn upload_to_album(
        &self,
        path: &str,
        description: Option<&str>,
        album: &str,
        progress: &filmcraft_engine::export::Progress,
    ) -> Result<serde_json::Value, String> {
        self.upload_into(path, description, Some(album), progress)
    }

    fn create_album(&self, title: &str, description: Option<&str>) -> Result<serde_json::Value, String> {
        let gp = GooglePhotos::configured().map_err(|e| e.to_string())?;
        gp.albums(|a| a.ensure(title, description)).map(|e| e.to_json()).map_err(|e| e.to_string())
    }

    fn albums(&self) -> Result<serde_json::Value, String> {
        let gp = GooglePhotos::configured().map_err(|e| e.to_string())?;
        gp.albums(|a| a.list()).map(|l| l.to_json()).map_err(|e| e.to_string())
    }

    fn add_to_album(&self, album: &str, media_item_ids: &[String]) -> Result<serde_json::Value, String> {
        let gp = GooglePhotos::configured().map_err(|e| e.to_string())?;
        gp.albums(|a| a.add(album, media_item_ids)).map(|a| a.to_json()).map_err(|e| e.to_string())
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
    /// The sign-in lacks a permission; the message names the command that grants it.
    #[error("{0}")]
    Consent(String),
    #[error("upload: {0}")]
    Upload(String),
    #[error("album: {0}")]
    Album(String),
    /// The caller stopped it.
    #[error("cancelled")]
    Cancelled,
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
    #[test]
    fn since_keeps_items_on_or_after_the_date() {
        assert!(created_on_or_after("2020-01-01T00:00:01Z", "2020-01-01"));
        assert!(created_on_or_after("2024-05-17T09:30:00Z", "2020-01-01"));
        assert!(!created_on_or_after("2019-12-31T23:59:59Z", "2020-01-01"));
        assert!(!created_on_or_after("2013-07-27T10:00:00Z", "2020-01-01"));
        // unknown or malformed dates are kept, never dropped; a malformed `since` keeps all
        for c in ["", "garbage", "2020-13-01T00:00:00Z", "20"] {
            assert!(created_on_or_after(c, "2020-01-01"), "{c}");
        }
        assert!(created_on_or_after("2010-01-01T00:00:00Z", "not a date"));
        assert!(is_day("2020-01-01") && is_day(" 2024-12-31 "));
        for no in ["2020-1-1", "2020-13-01", "20200101", "2020-01-01T00:00:00Z", ""] {
            assert!(!is_day(no), "{no}");
        }
    }

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

    /// Answers in order; every request is counted (and its method, URL and body kept).
    struct Scripted(Mutex<(Vec<upload::Response>, usize)>, Mutex<Vec<(upload::Method, String, String)>>);

    impl Scripted {
        fn new(answers: Vec<upload::Response>) -> Self {
            Self(Mutex::new((answers, 0)), Mutex::new(Vec::new()))
        }
    }

    impl upload::Transport for Scripted {
        fn send(&self, req: &upload::Request<'_>) -> Result<upload::Response, String> {
            self.1.lock().unwrap().push((req.method, req.url.to_string(), String::from_utf8_lossy(req.body).into_owned()));
            let mut s = self.0.lock().unwrap();
            s.1 += 1;
            if s.0.is_empty() { Err("no more answers".into()) } else { Ok(s.0.remove(0)) }
        }
    }

    fn answer(body: &str) -> upload::Response {
        upload::Response { status: 200, headers: vec![], body: body.into() }
    }

    #[test]
    fn uploading_into_an_album_finds_or_creates_it_first() {
        let dir = temp("album");
        let file = dir.join("showcase.mp4");
        std::fs::write(&file, b"0123456789").unwrap();
        let opts = upload::UploadOptions { backoff: Duration::ZERO, album: Some("Robot showcase".into()), ..Default::default() };
        // upload permission only: the album needs `--albums`, and nothing is sent
        let http = Scripted::new(Vec::new());
        let gp = signed_in(&format!("{} {}", auth::SCOPE, auth::UPLOAD_SCOPE));
        assert!(!gp.can_use_albums());
        let e = gp.upload_with(&http, &file, &opts, &mut |_, _, _| true).unwrap_err();
        assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("photos auth --albums"), "{e}");
        assert_eq!(http.0.lock().unwrap().1, 0);
        // a file Google refuses: no album is made for it
        let wav = dir.join("mix.wav");
        std::fs::write(&wav, b"RIFF").unwrap();
        let gp = signed_in(&format!("{} {} {}", auth::SCOPE, auth::UPLOAD_SCOPE, auth::ALBUM_SCOPE));
        assert!(gp.can_use_albums());
        assert!(gp.upload_with(&http, &wav, &opts, &mut |_, _, _| true).is_err());
        assert_eq!(http.0.lock().unwrap().1, 0);
        // the album is listed (none), created, then the upload goes into it
        let album = r#"{"id":"AF1QipAlbum00000000000000000000000000000000000000001","title":"Robot showcase","productUrl":"https://photos.google.com/lr/album/x","isWriteable":true}"#;
        let session = "https://photoslibrary.googleapis.com/v1/uploads?upload_id=x";
        let http = Scripted::new(vec![
            answer("{}"),
            answer(album),
            upload::Response { status: 200, headers: vec![("X-Goog-Upload-URL".into(), session.into())], body: String::new() },
            answer("tok"),
            answer(r#"{"newMediaItemResults":[{"status":{"message":"Success"},"mediaItem":{"id":"M1"}}]}"#),
        ]);
        let mut statuses = Vec::new();
        let up = gp
            .upload_with(&http, &file, &opts, &mut |_, _, s| {
                statuses.push(s.to_string());
                true
            })
            .unwrap();
        let seen = http.1.lock().unwrap().clone();
        assert_eq!(seen.len(), 5);
        assert_eq!((seen[0].0, seen[0].1.as_str()), (upload::Method::Get, "https://photoslibrary.googleapis.com/v1/albums?pageSize=50"));
        assert_eq!((seen[1].0, seen[1].1.as_str()), (upload::Method::Post, albums::ALBUMS_URL));
        let batch: serde_json::Value = serde_json::from_str(&seen[4].2).unwrap();
        assert_eq!(batch["albumId"], "AF1QipAlbum00000000000000000000000000000000000000001");
        assert!(up.album_created);
        let a = up.album.as_ref().unwrap();
        assert_eq!((a.title.as_str(), a.product_url.as_deref()), ("Robot showcase", Some("https://photos.google.com/lr/album/x")));
        assert_eq!(up.to_json()["albumCreated"], true);
        assert!(statuses.first().is_some_and(|s| s.contains("album")), "{statuses:?}");
        // cancelled while finding the album: nothing sent
        let http = Scripted::new(Vec::new());
        assert!(matches!(gp.upload_with(&http, &file, &opts, &mut |_, _, _| false), Err(GPhotosError::Cancelled)));
        assert_eq!(http.0.lock().unwrap().1, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn album_readiness_comes_from_the_disk() {
        let dir = temp("album-ready");
        assert!(album_readiness(&dir).unwrap_err().contains("photos auth --albums"));
        std::fs::write(dir.join("client_secret.json"), r#"{"installed":{"client_id":"id","client_secret":"s"}}"#).unwrap();
        let t = |scope: &str| auth::Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 1, scope: scope.into() };
        t(&format!("{} {}", auth::SCOPE, auth::UPLOAD_SCOPE)).save(&dir).unwrap();
        let e = album_readiness(&dir).unwrap_err();
        assert!(e.contains("consent") && e.contains("photos auth --albums"), "{e}");
        assert_eq!(upload_readiness(&dir), Ok(()), "uploads alone still work");
        use filmcraft_engine::photos::PhotosUploader as _;
        let up = EngineUploader::new(dir.clone());
        assert!(up.albums_ready().unwrap_err().contains("--albums"));
        assert_eq!(up.ready(), Ok(()));
        t(&format!("{} {} {}", auth::SCOPE, auth::UPLOAD_SCOPE, auth::ALBUM_SCOPE)).save(&dir).unwrap();
        assert_eq!(album_readiness(&dir), Ok(()));
        assert_eq!(EngineUploader::new(dir.clone()).albums_ready(), Ok(()));
        std::fs::remove_dir_all(&dir).ok();
    }

    fn signed_in(scope: &str) -> GooglePhotos {
        let tokens = auth::Tokens { access_token: "ya29".into(), refresh_token: Some("r".into()), expires_at: u64::MAX, scope: scope.into() };
        GooglePhotos {
            secret: auth::ClientSecret { client_id: "id".into(), client_secret: "s".into() },
            tokens: Arc::new(Mutex::new(tokens)),
            agent: api::http_agent(),
        }
    }

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gphotos-lib-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn uploading_needs_the_upload_permission() {
        let dir = temp("consent");
        let file = dir.join("export.mp4");
        std::fs::write(&file, b"0123456789").unwrap();
        let opts = upload::UploadOptions { backoff: Duration::ZERO, ..Default::default() };
        // a picker-only sign-in: the consent error, and nothing goes out
        let http = Scripted::new(Vec::new());
        let gp = signed_in(auth::SCOPE);
        assert!(!gp.can_upload());
        let e = gp.upload_with(&http, &file, &opts, &mut |_, _, _| true).unwrap_err();
        assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("photos auth --upload"), "{e}");
        assert_eq!(http.0.lock().unwrap().1, 0);
        // with it: start, one chunk, the media item
        let gp = signed_in(&format!("{} {}", auth::SCOPE, auth::UPLOAD_SCOPE));
        assert!(gp.can_upload());
        let session = "https://photoslibrary.googleapis.com/v1/uploads?upload_id=x";
        let answers = vec![
            upload::Response { status: 200, headers: vec![("X-Goog-Upload-URL".into(), session.into())], body: String::new() },
            upload::Response { status: 200, headers: vec![], body: "tok".into() },
            upload::Response {
                status: 200,
                headers: vec![],
                body: r#"{"newMediaItemResults":[{"status":{"message":"Success"},"mediaItem":{"id":"M1"}}]}"#.into(),
            },
        ];
        let http = Scripted::new(answers);
        let up = gp.upload_with(&http, &file, &opts, &mut |_, _, _| true).unwrap();
        assert_eq!((up.media_item_id.as_str(), up.bytes, up.product_url), ("M1", 10, None));
        assert_eq!(http.0.lock().unwrap().1, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn readiness_comes_from_the_disk() {
        use filmcraft_engine::photos::PhotosUploader as _;
        let dir = temp("ready");
        let e = upload_readiness(&dir).unwrap_err();
        assert!(e.contains("not connected") && e.contains("client_secret.json"), "{e}");
        std::fs::write(dir.join("client_secret.json"), r#"{"installed":{"client_id":"id","client_secret":"s"}}"#).unwrap();
        let e = upload_readiness(&dir).unwrap_err();
        assert!(e.contains("not connected") && e.contains("photos auth --upload"), "{e}");
        let t = |scope: &str| auth::Tokens { access_token: "a".into(), refresh_token: Some("r".into()), expires_at: 1, scope: scope.into() };
        t(auth::SCOPE).save(&dir).unwrap();
        let e = upload_readiness(&dir).unwrap_err();
        assert!(e.contains("consent") && e.contains("photos auth --upload"), "{e}");
        std::fs::write(dir.join("tokens.json"), "{not json").unwrap();
        assert!(upload_readiness(&dir).unwrap_err().contains("not connected"));
        t(&format!("{} {}", auth::SCOPE, auth::UPLOAD_SCOPE)).save(&dir).unwrap();
        assert_eq!(upload_readiness(&dir), Ok(()));
        let up = EngineUploader::new(dir.clone());
        assert_eq!(up.ready(), Ok(()));
        assert!(up.accepts("/x/showcase.mp4").is_ok());
        assert!(up.accepts("/x/mix.wav").unwrap_err().contains(".wav"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
