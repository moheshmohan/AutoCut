//! YouTube for the editor: connect channels, upload finished exports to one of them — from the
//! CLI, over MCP and in the app.
//!
//! - [`auth`]: the sign-in, on the shared OAuth module (`filmcraft-oauth`: PKCE, the system
//!   browser, a loopback redirect on a random port, refresh, revocation) with the user's own
//!   Desktop OAuth client (the Google Photos one). Scopes `youtube.upload` + `youtube.readonly`;
//!   `youtube.force-ssl` only for playlists (`youtube auth --playlists`).
//! - [`store`]: the connected channels — one sign-in per channel, each its own Keychain account
//!   (`FilmCraft YouTube`), so a Google account's several channels (Brand Accounts) are each
//!   connected by their own sign-in — and the default channel.
//! - [`upload`]: the resumable upload (chunks, resume after a dropped connection or a crash,
//!   backoff, a dropped session started again), behind the [`http::Transport`] the tests drive
//!   against a mock Google on `127.0.0.1`.
//! - [`api`]: the channel lookup, playlists, thumbnails, the retry policy, Google's errors in
//!   words (`quotaExceeded`, `uploadLimitExceeded`, a lapsed sign-in, the API not enabled…).
//! - [`quota`]: the day's quota as this machine spent it (Pacific-time days, as Google counts).
//! - [`media`]: the checks before a byte is sent — the file, the thumbnail, the chapters a
//!   showcase export leaves beside itself, the Shorts hint.
//! - [`YouTube`]: all of it over one data directory; [`engine_publisher`] is what hosts install
//!   as the engine's `youtube.*` publisher.
//!
//! Nothing here prints: callers report progress (stdout may be a protocol stream).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod api;
pub mod auth;
pub mod http;
pub mod media;
pub mod quota;
pub mod store;
pub mod upload;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use filmcraft_engine::youtube::{ConnectOptions, UploadRequest};
use filmcraft_oauth::{OAuthClient, Provider, TokenSet, TokenVault};
use serde_json::{Value, json};

pub use filmcraft_oauth::now;
pub use store::{Channel, Secrets};

use api::Ctx;
use auth::ClientSecret;
use http::Transport;
use quota::QuotaBook;
use store::Channels;

/// YouTube failures, each a message a user can act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum YtError {
    /// The OAuth client file, the API not enabled in the project, the Keychain, a damaged file.
    #[error("{0}")]
    Setup(String),
    /// The sign-in is gone (lapsed, revoked): the message names the command that signs in again.
    #[error("{0}")]
    Auth(String),
    /// The sign-in lacks a permission (or the channel a feature): the message says how to get it.
    #[error("{0}")]
    Consent(String),
    /// The day's quota or the channel's upload limit ran out.
    #[error("{0}")]
    Quota(String),
    /// A file, a thumbnail or metadata YouTube does not take.
    #[error("{0}")]
    Invalid(String),
    /// No such connected channel, no default, an account without a channel.
    #[error("{0}")]
    Channel(String),
    /// YouTube answered otherwise.
    #[error("{0}")]
    Api(String),
    /// The upload protocol failed (the session dropped twice, odd answers, retries spent).
    #[error("{0}")]
    Upload(String),
    /// The caller stopped it.
    #[error("cancelled")]
    Cancelled,
}

/// At most `n` characters of `s`, without control characters, trimmed (for quoting an answer).
pub fn clean(s: &str, n: usize) -> String {
    s.chars().filter(|c| !c.is_control()).take(n).collect::<String>().trim().to_string()
}

/// Owner-only, atomic.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    filmcraft_oauth::store::write_private(path, bytes)
}

/// Where Google's endpoints are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoints {
    /// OAuth: consent, token, revocation.
    pub provider: Provider,
    /// `https://www.googleapis.com/youtube/v3`
    pub api: String,
    /// `https://www.googleapis.com/upload/youtube/v3`
    pub upload: String,
}

impl Endpoints {
    pub fn google() -> Self {
        Self {
            provider: Provider::google(),
            api: "https://www.googleapis.com/youtube/v3".into(),
            upload: "https://www.googleapis.com/upload/youtube/v3".into(),
        }
    }

    /// Everything under `base` (a mock server in tests): `/youtube/v3`, `/upload/youtube/v3`,
    /// `/token`, `/revoke`, `/auth`.
    pub fn at(base: &str) -> Self {
        let base = base.trim_end_matches('/');
        Self {
            provider: Provider {
                auth_url: format!("{base}/auth"),
                token_url: format!("{base}/token"),
                revoke_url: Some(format!("{base}/revoke")),
                ..Provider::google()
            },
            api: format!("{base}/youtube/v3"),
            upload: format!("{base}/upload/youtube/v3"),
        }
    }
}

/// FilmCraft's data directory (as the app's).
pub fn data_root() -> PathBuf {
    filmcraft_engine::autosave::default_data_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// `<data dir>/youtube`: the channel list, the quota count, kept upload sessions (and, without
/// a Keychain, the channels' `tokens.json`).
pub fn data_dir() -> PathBuf {
    data_root().join("youtube")
}

/// A channel just connected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connected {
    pub channel: Channel,
    pub is_default: bool,
    /// It was connected before (the sign-in was replaced).
    pub replaced: bool,
    /// Where the sign-in is kept ("the macOS Keychain …").
    pub place: String,
    pub note: Option<String>,
}

impl Connected {
    pub fn to_json(&self) -> Value {
        json!({"channel": self.channel.to_json(self.is_default), "default": self.is_default, "replaced": self.replaced, "kept": self.place, "note": self.note})
    }
}

/// A channel just disconnected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnected {
    pub channel: Channel,
    /// Google confirmed the revocation; `Err` says why not (the local copy is gone anyway).
    pub revoked: Result<(), String>,
    /// The default now.
    pub default: Option<Channel>,
}

impl Disconnected {
    pub fn to_json(&self) -> Value {
        json!({
            "channel": self.channel.to_json(false), "revoked": self.revoked.is_ok(), "revokeError": self.revoked.as_ref().err(),
            "default": self.default.as_ref().map(|c| c.to_json(true)),
        })
    }
}

/// What can be known before a byte is sent.
#[derive(Debug, Clone, PartialEq)]
pub struct Preflight {
    pub channel: Channel,
    pub video: media::LocalVideo,
    pub thumbnail: Option<media::Thumbnail>,
    /// The description sent (with the chapters, when they were appended).
    pub description: Option<String>,
    /// Chapters appended to the description (their count).
    pub chapters: Option<usize>,
    pub short: bool,
    pub shorts_note: String,
    pub notes: Vec<String>,
}

impl Preflight {
    pub fn to_json(&self) -> Value {
        json!({
            "channel": self.channel.to_json(false), "bytes": self.video.bytes, "mimeType": self.video.mime,
            "thumbnail": self.thumbnail.as_ref().map(|t| t.path.display().to_string()), "chapters": self.chapters,
            "shorts": {"likely": self.short, "note": self.shorts_note,
                       "width": self.video.shape.map(|s| s.width), "height": self.video.shape.map(|s| s.height), "seconds": self.video.shape.map(|s| s.seconds)},
            "notes": self.notes,
        })
    }
}

/// A finished upload.
#[derive(Debug, Clone, PartialEq)]
pub struct Uploaded {
    pub video: upload::Video,
    pub channel: Channel,
    pub privacy: String,
    pub bytes: u64,
    pub sent: upload::Sent,
    pub preflight: Preflight,
    /// The thumbnail: set, or why not (`None`: none was asked for).
    pub thumbnail: Option<Result<String, String>>,
    /// The playlist (id, title): added, or why not.
    pub playlist: Option<(api::Playlist, Result<(), String>)>,
    pub notes: Vec<String>,
    pub quota: quota::QuotaDay,
}

impl Uploaded {
    pub fn url(&self) -> String {
        format!("https://www.youtube.com/watch?v={}", self.video.id)
    }

    pub fn to_json(&self) -> Value {
        let mut notes = self.preflight.notes.clone();
        notes.extend(self.notes.iter().cloned());
        json!({
            "videoId": self.video.id, "url": self.url(), "studioUrl": format!("https://studio.youtube.com/video/{}/edit", self.video.id),
            "title": self.video.title, "channel": self.channel.to_json(false), "privacy": self.privacy,
            "privacyStatus": self.video.privacy_status, "uploadStatus": self.video.upload_status, "bytes": self.bytes,
            "resumed": self.sent.resumed, "restarted": self.sent.restarted, "resumedEarlierUpload": self.sent.from_earlier,
            "chapters": self.preflight.chapters,
            "shorts": {"likely": self.preflight.short, "note": self.preflight.shorts_note},
            "thumbnail": self.thumbnail.as_ref().map(|r| match r {
                Ok(p) => json!({"set": true, "path": p}),
                Err(e) => json!({"set": false, "error": e}),
            }),
            "playlist": self.playlist.as_ref().map(|(p, r)| json!({"id": p.id, "title": p.title, "added": r.is_ok(), "error": r.as_ref().err()})),
            "notes": notes,
            "quota": self.quota.to_json(now()),
        })
    }
}

/// A channel's sign-in, refreshed on demand.
struct Signed {
    vault: TokenVault,
    tokens: Mutex<TokenSet>,
    client: OAuthClient,
    label: String,
}

/// YouTube over one data directory.
pub struct YouTube {
    dir: PathBuf,
    secret_dirs: Vec<PathBuf>,
    endpoints: Endpoints,
    http: Arc<dyn Transport>,
    token_agent: ureq::Agent,
    channels: Channels,
    quota: QuotaBook,
    opener: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// How long a sign-in waits for the browser.
    pub sign_in_wait: Duration,
    pub upload_opts: upload::UploadOptions,
}

impl Default for YouTube {
    fn default() -> Self {
        Self::new()
    }
}

impl YouTube {
    /// Over the app's data directory, Google's endpoints, the Keychain.
    pub fn new() -> Self {
        Self::in_data_dir(&data_root())
    }

    /// Over `root` (a FilmCraft data directory): `root/youtube`, and the OAuth client from
    /// there or from `root/google-photos`.
    pub fn in_data_dir(root: &Path) -> Self {
        let dir = root.join("youtube");
        Self {
            secret_dirs: vec![dir.clone(), root.join("google-photos")],
            endpoints: Endpoints::google(),
            http: Arc::new(http::UreqTransport::default()),
            token_agent: filmcraft_oauth::http_agent(),
            channels: Channels::new(&dir, Secrets::System),
            quota: QuotaBook::new(&dir),
            opener: Arc::new(filmcraft_oauth::open_in_browser),
            sign_in_wait: auth::CONSENT_WAIT,
            upload_opts: upload::UploadOptions::default(),
            dir,
        }
    }

    pub fn with_endpoints(mut self, endpoints: Endpoints) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// The API calls over `http`, the token calls over `token_agent` (tests: plain HTTP to a mock).
    pub fn with_transport(mut self, http: Arc<dyn Transport>, token_agent: ureq::Agent) -> Self {
        self.http = http;
        self.token_agent = token_agent;
        self
    }

    pub fn with_secrets(mut self, secrets: Secrets) -> Self {
        self.channels = Channels::new(&self.dir, secrets);
        self
    }

    /// What opens the consent page (tests: a stand-in browser).
    pub fn with_opener(mut self, open: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        self.opener = Arc::new(open);
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn store(&self) -> &Channels {
        &self.channels
    }

    pub fn quota(&self) -> quota::QuotaDay {
        self.quota.today()
    }

    /// Whether a sign-in can start (the OAuth client file is there), from disk.
    pub fn connect_readiness(&self) -> Result<(), String> {
        ClientSecret::present(&self.secret_dirs).map(drop)
    }

    /// Whether an upload can start (the client file and a connected channel), from disk.
    pub fn upload_readiness(&self) -> Result<(), String> {
        self.connect_readiness()?;
        let book = self.channels.load().map_err(|e| e.to_string())?;
        if book.channels.is_empty() {
            return Err("YouTube is not connected: run `filmcraft-cli youtube auth` and pick the channel in Google's account chooser".into());
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // channels
    // ---------------------------------------------------------------------------------------

    /// Sign in: the browser opens Google's consent with its account chooser, where the channel
    /// is picked; `channels.list?mine=true` names it; its sign-in is kept apart from the other
    /// channels'. The first channel connected (or any with `make_default`) becomes the default.
    pub fn connect(&self, opts: ConnectOptions, progress: &mut dyn FnMut(&str)) -> Result<Connected, YtError> {
        let (secret, _) = ClientSecret::find(&self.secret_dirs)?;
        let cmd = if opts.playlists { auth::SIGN_IN_PLAYLISTS } else { auth::SIGN_IN };
        let client = secret.client(self.endpoints.provider.clone(), cmd);
        progress(
            "In Google's account chooser, pick the channel to connect: your Google account and the channels it manages (Brand Accounts) are listed; each sign-in connects one channel",
        );
        let req = auth::sign_in_request(opts.playlists, self.sign_in_wait);
        let tokens = client.sign_in(&self.token_agent, &req, &mut |u| (self.opener)(u), &mut |m| progress(m)).map_err(|e| auth::from_oauth(e, None))?;
        if tokens.refresh_token.is_none() {
            return Err(YtError::Auth(format!("Google returned no refresh token; run `{cmd}` again")));
        }
        auth::check_granted(&tokens.scope, opts.playlists)?;
        progress("Signed in; asking YouTube which channel it is");
        let access = tokens.access_token.clone();
        let bearer = move |_: bool| Ok::<String, YtError>(access.clone());
        let ctx = Ctx {
            http: &*self.http,
            endpoints: &self.endpoints,
            quota: &self.quota,
            bearer: &bearer,
            channel: "the channel picked".into(),
            sign_in: cmd,
            backoff: self.upload_opts.backoff,
        };
        let info = match api::my_channel(&ctx, &mut |_, _, _| true)? {
            Some(info) => info,
            None => {
                // nothing to keep: give the grant back
                let _ = client.revoke(&self.token_agent, tokens.refresh_token.as_deref().unwrap_or(&tokens.access_token));
                return Err(YtError::Channel(
                    "the account picked has no YouTube channel: create one at https://www.youtube.com/create_channel, or connect again and pick a channel in Google's account chooser".into(),
                ));
            }
        };
        let mut book = self.channels.load()?;
        let replaced = book.channels.iter().any(|c| c.id == info.id);
        if !replaced && book.channels.len() >= store::MAX_CHANNELS {
            return Err(YtError::Channel(format!("{} channels are connected, the most FilmCraft keeps; disconnect one first", store::MAX_CHANNELS)));
        }
        let stored = self.channels.vault(&info.id)?.save(&tokens).map_err(|e| auth::from_oauth(e, None))?;
        let channel = Channel {
            id: info.id.clone(),
            title: info.title,
            handle: info.handle,
            long_uploads: info.long_uploads,
            scope: tokens.scope.clone(),
            connected_at: now(),
        };
        book.channels.retain(|c| c.id != channel.id);
        book.channels.push(channel.clone());
        if opts.make_default || book.default.is_none() {
            book.default = Some(channel.id.clone());
        }
        self.channels.save(&book)?;
        Ok(Connected {
            is_default: book.default.as_deref() == Some(channel.id.as_str()),
            channel,
            replaced,
            place: stored.place.to_string(),
            note: stored.note,
        })
    }

    /// The connected channels, the default, today's quota, the client file: from disk.
    pub fn listing(&self) -> Result<Value, YtError> {
        let book = self.channels.load()?;
        let channels: Vec<Value> = book
            .channels
            .iter()
            .map(|c| {
                let mut v = c.to_json(book.default.as_deref() == Some(c.id.as_str()));
                // the sign-in's note on disk (never the Keychain: this is asked while drawing)
                let signed_in = self.channels.vault(&c.id).ok().and_then(|v| v.summary().ok().flatten()).is_some();
                v["signedIn"] = json!(signed_in);
                v
            })
            .collect();
        let client = ClientSecret::present(&self.secret_dirs);
        Ok(json!({
            "channels": channels,
            "default": book.default_channel().map(|c| c.to_json(true)),
            "quota": self.quota.today().to_json(now()),
            "clientFile": client.as_ref().ok().map(|p| p.display().to_string()),
            "setup": client.err(),
        }))
    }

    /// Make the channel `reference` names the default.
    pub fn set_default(&self, reference: &str) -> Result<Channel, YtError> {
        let mut book = self.channels.load()?;
        let ch = book.find(Some(reference))?.clone();
        book.default = Some(ch.id.clone());
        self.channels.save(&book)?;
        Ok(ch)
    }

    /// Revoke the channel's sign-in at Google (best effort) and forget it here: its Keychain
    /// item, its files, its entry. Another channel becomes the default if it was.
    pub fn disconnect(&self, reference: &str) -> Result<Disconnected, YtError> {
        const BY_HAND: &str = "remove FilmCraft's access at https://myaccount.google.com/connections (for a Brand Account, switch to it there first)";
        let mut book = self.channels.load()?;
        let ch = book.find(Some(reference))?.clone();
        let vault = self.channels.vault(&ch.id)?;
        let revoker =
            OAuthClient { provider: self.endpoints.provider.clone(), client_id: String::new(), client_secret: None, sign_in_command: auth::SIGN_IN.into() };
        let revoked = match vault.load() {
            Ok(Some((t, _))) => revoker.revoke(&self.token_agent, t.refresh_token.as_deref().unwrap_or(&t.access_token)).map_err(|e| format!("{e}; {BY_HAND}")),
            Ok(None) => Ok(()),
            Err(e) => Err(format!("its sign-in could not be read ({e}), so Google was not asked to revoke it; {BY_HAND}")),
        };
        let removed = vault.remove();
        if let Ok(dir) = self.channels.channel_dir(&ch.id) {
            let _ = std::fs::remove_dir_all(dir);
        }
        book.channels.retain(|c| c.id != ch.id);
        if book.default.as_deref() == Some(ch.id.as_str()) {
            book.default = book.channels.first().map(|c| c.id.clone());
        }
        self.channels.save(&book)?;
        removed.map_err(|e| auth::from_oauth(e, None))?;
        Ok(Disconnected { channel: ch, revoked, default: book.default_channel().cloned() })
    }

    /// A channel's sign-in, refreshed if it is about to run out.
    fn signed(&self, ch: &Channel) -> Result<Signed, YtError> {
        let label = ch.label();
        let (secret, _) = ClientSecret::find(&self.secret_dirs)?;
        let vault = self.channels.vault(&ch.id)?;
        let Some((mut tokens, _)) = vault.load().map_err(|e| auth::from_oauth(e, Some(&label)))? else {
            return Err(YtError::Auth(format!(
                "{label} has no saved sign-in here; run `{}` and pick it in Google's account chooser",
                auth::sign_in_command(&ch.scope)
            )));
        };
        let client = secret.client(self.endpoints.provider.clone(), auth::sign_in_command(&tokens.scope));
        if tokens.needs_refresh() {
            tokens = client.refresh(&self.token_agent, &tokens).map_err(|e| auth::from_oauth(e, Some(&label)))?;
            vault.save(&tokens).map_err(|e| auth::from_oauth(e, Some(&label)))?;
        }
        Ok(Signed { vault, tokens: Mutex::new(tokens), client, label })
    }

    /// A current access token for `s` (`force`: refresh now, after YouTube refused one).
    fn bearer(&self, s: &Signed, force: bool) -> Result<String, YtError> {
        let mut t = s.tokens.lock().unwrap_or_else(PoisonError::into_inner);
        if force || t.needs_refresh() {
            *t = s.client.refresh(&self.token_agent, &t).map_err(|e| auth::from_oauth(e, Some(&s.label)))?;
            s.vault.save(&t).map_err(|e| auth::from_oauth(e, Some(&s.label)))?;
        }
        Ok(t.access_token.clone())
    }

    // ---------------------------------------------------------------------------------------
    // uploads
    // ---------------------------------------------------------------------------------------

    /// Everything that can be checked before a byte is sent, without the network: the client
    /// file, the channel and its sign-in, the playlist permission, the file, its length against
    /// the channel's limit, the thumbnail, the chapters, the Shorts hint.
    pub fn preflight(&self, req: &UploadRequest) -> Result<Preflight, YtError> {
        ClientSecret::find(&self.secret_dirs)?;
        let book = self.channels.load()?;
        let channel = book.find(req.channel.as_deref())?.clone();
        let label = channel.label();
        let summary = self.channels.vault(&channel.id)?.summary().map_err(|e| auth::from_oauth(e, Some(&label)))?;
        if summary.is_none() {
            return Err(YtError::Auth(format!(
                "{label} has no saved sign-in here; run `{}` and pick it in Google's account chooser",
                auth::sign_in_command(&channel.scope)
            )));
        }
        if req.playlist.is_some() && !channel.grants(auth::MANAGE_SCOPE) {
            return Err(YtError::Consent(format!(
                "adding to a playlist needs the permission to manage {label}'s videos and playlists: run `{}` and pick {label} in Google's account chooser",
                auth::SIGN_IN_PLAYLISTS
            )));
        }
        let path = Path::new(&req.path);
        let video = media::check_video(path)?;
        let mut notes = Vec::new();
        if let Some(s) = video.shape
            && s.seconds > media::LONG_UPLOAD_SECONDS
        {
            match channel.long_uploads.as_str() {
                "allowed" => {}
                "" => notes.push(format!("{:.0} minutes long: YouTube takes videos over 15 minutes only from verified channels", s.seconds / 60.0)),
                other => {
                    return Err(YtError::Invalid(format!(
                        "{} is {:.0} minutes long, and {label} may upload only up to 15 minutes (longUploadsStatus: {other}): verify the channel at https://www.youtube.com/verify, then connect it again",
                        path.display(),
                        s.seconds / 60.0
                    )));
                }
            }
        }
        let thumbnail = match &req.thumbnail {
            Some(t) => Some(media::check_thumbnail(Path::new(t))?),
            None if req.sidecars && media::thumbnail_beside(path).is_file() => match media::check_thumbnail(&media::thumbnail_beside(path)) {
                Ok(t) => Some(t),
                Err(e) => {
                    notes.push(format!("the thumbnail beside the video was not used: {e}"));
                    None
                }
            },
            None => None,
        };
        let mut description = req.meta.description.clone();
        let mut chapters = None;
        if req.sidecars {
            match media::read_chapters(&media::chapters_beside(path)) {
                Ok(Some(c)) => match media::with_chapters(description.as_deref(), &c) {
                    Ok(d) => {
                        chapters = Some(c.lines().count());
                        description = Some(d);
                    }
                    Err(e) => notes.push(e),
                },
                Ok(None) => {}
                Err(e) => notes.push(format!("chapters not added: {e}")),
            }
        }
        let (short, shorts_note) = media::shorts_note(video.shape.as_ref());
        Ok(Preflight { channel, video, thumbnail, description, chapters, short, shorts_note, notes })
    }

    /// Upload `req.path` to the channel `req.channel` names (the default when none), then set its
    /// thumbnail and add it to the playlist. A thumbnail or playlist that fails does not undo the
    /// upload: it is reported in the answer. `progress(sent, total, status)` returning `false`
    /// cancels (the upload session is kept, so the same upload resumes later).
    pub fn upload(&self, req: &UploadRequest, progress: &mut dyn FnMut(u64, u64, &str) -> bool) -> Result<Uploaded, YtError> {
        let pre = self.preflight(req)?;
        let total = pre.video.bytes;
        if !progress(0, total, &format!("Signing in to {}", pre.channel.label())) {
            return Err(YtError::Cancelled);
        }
        let signed = self.signed(&pre.channel)?;
        let bearer = |force: bool| self.bearer(&signed, force);
        let ctx = Ctx {
            http: &*self.http,
            endpoints: &self.endpoints,
            quota: &self.quota,
            bearer: &bearer,
            channel: pre.channel.label(),
            sign_in: auth::sign_in_command(&pre.channel.scope),
            backoff: self.upload_opts.backoff,
        };
        // the playlist before any byte: a wrong name must not leave a video behind
        let playlist = match &req.playlist {
            Some(p) => {
                if !progress(0, total, "Finding the playlist") {
                    return Err(YtError::Cancelled);
                }
                Some(api::find_playlist(&ctx, p, &mut *progress)?)
            }
            None => None,
        };
        let body = upload::insert_body(&req.meta, pre.description.as_deref());
        let path = Path::new(&req.path);
        let sessions = upload::Sessions::new(&self.dir);
        let key = upload::Sessions::key(&pre.channel.id, path, total, &body);
        let job = upload::Job {
            ctx: &ctx,
            path,
            video: &pre.video,
            body: &body,
            notify_subscribers: req.meta.notify_subscribers,
            keep: Some((&sessions, key)),
            opts: &self.upload_opts,
        };
        let (video, sent) = upload::upload(&job, &mut *progress)?;
        let mut notes = Vec::new();
        let asked = req.meta.privacy.as_str();
        if asked != "private" && video.privacy_status == "private" {
            notes.push(format!(
                "YouTube kept the video private, not {asked}: it may lock uploads from a Google Cloud project it has not audited to private; change it in YouTube Studio"
            ));
        }
        // after the video: failures here are reported, not raised (the video is up)
        let mut keep_going = |status: &str| progress(total, total, status);
        let thumbnail = pre.thumbnail.as_ref().map(|t| {
            if !keep_going("Setting the thumbnail") {
                return Err("cancelled before the thumbnail was set".to_string());
            }
            let bytes = std::fs::read(&t.path).map_err(|e| format!("{}: {e}", t.path.display()))?;
            api::set_thumbnail(&ctx, &video.id, &bytes, t.mime, &mut |_, _, _| true).map(|()| t.path.display().to_string()).map_err(|e| e.to_string())
        });
        let playlist = playlist.map(|p| {
            if !keep_going("Adding it to the playlist") {
                return (p, Err("cancelled before it was added".to_string()));
            }
            let r = api::add_to_playlist(&ctx, &p.id, &video.id, &mut |_, _, _| true).map_err(|e| e.to_string());
            (p, r)
        });
        Ok(Uploaded {
            channel: pre.channel.clone(),
            privacy: asked.to_string(),
            bytes: total,
            sent,
            thumbnail,
            playlist,
            notes,
            quota: self.quota.today(),
            preflight: pre,
            video,
        })
    }
}

// -------------------------------------------------------------------------------------------
// the engine's publisher
// -------------------------------------------------------------------------------------------

/// How long [`EnginePublisher::ready`] trusts its last look at the disk (it runs while drawing).
const READY_TTL: Duration = Duration::from_secs(2);

type Cached = Mutex<Option<(Instant, Result<(), String>)>>;

/// The engine's `youtube.*` publisher over a data directory.
pub struct EnginePublisher {
    yt: YouTube,
    ready: Cached,
    connect_ready: Cached,
}

/// The publisher hosts install as `Session::youtube.publisher`.
pub fn engine_publisher() -> Arc<dyn filmcraft_engine::youtube::YouTubePublisher> {
    Arc::new(EnginePublisher::new(YouTube::new()))
}

impl EnginePublisher {
    pub fn new(yt: YouTube) -> Self {
        Self { yt, ready: Mutex::new(None), connect_ready: Mutex::new(None) }
    }

    fn forget_readiness(&self) {
        *self.ready.lock().unwrap_or_else(PoisonError::into_inner) = None;
        *self.connect_ready.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

fn cached(slot: &Cached, check: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    let mut c = slot.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((at, r)) = &*c
        && at.elapsed() < READY_TTL
    {
        return r.clone();
    }
    let r = check();
    *c = Some((Instant::now(), r.clone()));
    r
}

impl filmcraft_engine::youtube::YouTubePublisher for EnginePublisher {
    fn connect_ready(&self) -> Result<(), String> {
        cached(&self.connect_ready, || self.yt.connect_readiness())
    }

    fn ready(&self) -> Result<(), String> {
        cached(&self.ready, || self.yt.upload_readiness())
    }

    fn channels(&self) -> Result<Value, String> {
        self.yt.listing().map_err(|e| e.to_string())
    }

    fn connect(&self, opts: ConnectOptions, progress: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        let mut report = |m: &str| *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = clean(m, 300);
        let r = self.yt.connect(opts, &mut report).map(|c| c.to_json()).map_err(|e| e.to_string());
        self.forget_readiness();
        r
    }

    fn set_default(&self, channel: &str) -> Result<Value, String> {
        self.yt.set_default(channel).map(|c| json!({"default": c.to_json(true)})).map_err(|e| e.to_string())
    }

    fn disconnect(&self, channel: &str) -> Result<Value, String> {
        let r = self.yt.disconnect(channel).map(|d| d.to_json()).map_err(|e| e.to_string());
        self.forget_readiness();
        r
    }

    fn check(&self, req: &UploadRequest) -> Result<Value, String> {
        self.yt.preflight(req).map(|p| p.to_json()).map_err(|e| e.to_string())
    }

    fn upload(&self, req: &UploadRequest, progress: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        use std::sync::atomic::Ordering;
        let mut report = |sent: u64, total: u64, status: &str| {
            if total > 0 {
                progress.total.store(total, Ordering::Relaxed);
                progress.done.store(sent, Ordering::Relaxed);
            }
            *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = status.to_string();
            !progress.cancel.load(Ordering::Relaxed)
        };
        self.yt.upload(req, &mut report).map(|u| u.to_json()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cleaned_and_endpoints_built() {
        assert_eq!(clean(" a\u{7}b\nc ", 10), "abc");
        assert_eq!(clean("héllo", 2), "hé");
        let e = Endpoints::at("http://127.0.0.1:9/");
        assert_eq!((e.api.as_str(), e.upload.as_str()), ("http://127.0.0.1:9/youtube/v3", "http://127.0.0.1:9/upload/youtube/v3"));
        assert_eq!(e.provider.token_url, "http://127.0.0.1:9/token");
        assert_eq!(Endpoints::google().provider, Provider::google());
    }
}
