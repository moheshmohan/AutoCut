//! Instagram for the editor: connect the user's professional account and publish finished
//! exports as Reels — from the CLI, over MCP and in the app (roadmap B4).
//!
//! - [`auth`]: Instagram's business login with the user's own Meta app (its Instagram app id and
//!   secret, set up once; the secret kept in the Keychain), or a 60-day token pasted from the
//!   App Dashboard; the long-lived token refreshed once it is a day old, so it never lapses while
//!   FilmCraft is used.
//! - [`store`]: the connected account (`account.json`, no secret) and its token
//!   ([`filmcraft_oauth::TokenVault`]: the Keychain item **FilmCraft Instagram**).
//! - [`api`]: `/me`, Reel containers, their status, `media_publish`, `content_publishing_limit`,
//!   Meta's errors in words.
//! - [`upload`]: the video sent to its container with Meta's resumable upload protocol
//!   (`rupload.facebook.com`), resumed after a dropped connection and across runs.
//! - [`limits`]: the 24-hour publishing limits as this machine spent them.
//! - [`Instagram`]: all of it over one data directory; [`engine_publisher`] is what hosts install
//!   as the engine's `instagram.*` publisher.
//!
//! Nothing here prints: callers report progress (stdout may be a protocol stream). No token or
//! secret ever reaches a message, a log or a file outside the secret store.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod api;
pub mod auth;
pub mod http;
pub mod limits;
pub mod store;
pub mod upload;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use filmcraft_engine::instagram::ReelRequest;
use filmcraft_oauth::TokenSet;
use filmcraft_oauth::loopback::{DonePage, Loopback};
use serde_json::{Value, json};

pub use filmcraft_oauth::now;
pub use store::{Account, Secrets, Via};

use api::Ctx;
use auth::{AppConfig, AppStore, Redirect};
use http::Transport;
use limits::LimitBook;
use store::AccountStore;

/// Instagram failures, each a message a user can act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IgError {
    /// The app is not set up, the Keychain, a damaged file.
    #[error("{0}")]
    Setup(String),
    /// The sign-in is gone (expired, revoked, declined): the message says how to connect again.
    #[error("{0}")]
    Auth(String),
    /// A permission is missing, or the account cannot publish (a personal account).
    #[error("{0}")]
    Consent(String),
    /// A publishing or API rate limit.
    #[error("{0}")]
    Limit(String),
    /// A file, caption, cover or URL Instagram does not take.
    #[error("{0}")]
    Invalid(String),
    /// Instagram could not process the video (container `ERROR` / `EXPIRED`, or still busy).
    #[error("{0}")]
    Processing(String),
    /// Instagram answered otherwise.
    #[error("{0}")]
    Api(String),
    /// Sending the video failed.
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

/// The Graph API version FilmCraft calls (Meta keeps a version for two years).
pub const API_VERSION: &str = "v25.0";

/// Where Instagram's endpoints are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoints {
    /// `https://www.instagram.com/oauth/authorize`
    pub authorize: String,
    /// `https://api.instagram.com/oauth/access_token`
    pub token: String,
    /// `https://graph.instagram.com`
    pub graph: String,
    pub version: String,
    /// `https://rupload.facebook.com/ig-api-upload`
    pub rupload: String,
}

impl Endpoints {
    pub fn instagram() -> Self {
        Self {
            authorize: "https://www.instagram.com/oauth/authorize".into(),
            token: "https://api.instagram.com/oauth/access_token".into(),
            graph: "https://graph.instagram.com".into(),
            version: API_VERSION.into(),
            rupload: "https://rupload.facebook.com/ig-api-upload".into(),
        }
    }

    /// Everything under `base` (a mock server in tests).
    pub fn at(base: &str) -> Self {
        let base = base.trim_end_matches('/');
        Self {
            authorize: format!("{base}/oauth/authorize"),
            token: format!("{base}/oauth/access_token"),
            graph: base.to_string(),
            version: API_VERSION.into(),
            rupload: format!("{base}/ig-api-upload"),
        }
    }
}

/// FilmCraft's data directory (as the app's).
pub fn data_root() -> PathBuf {
    filmcraft_engine::autosave::default_data_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// How long processing is awaited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollOptions {
    /// The first wait (doubling up to `every`).
    pub first: Duration,
    /// The longest wait between two looks (Meta suggests once a minute).
    pub every: Duration,
    /// The longest processing is awaited in all (Meta: no more than 5 minutes).
    pub limit: Duration,
}

impl Default for PollOptions {
    fn default() -> Self {
        Self { first: Duration::from_secs(5), every: Duration::from_secs(60), limit: Duration::from_secs(5 * 60) }
    }
}

/// An account just connected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connected {
    pub account: Account,
    /// It replaced a connected account.
    pub replaced: Option<String>,
    /// Where the token is kept ("the macOS Keychain …").
    pub place: String,
    pub note: Option<String>,
}

impl Connected {
    pub fn to_json(&self) -> Value {
        json!({"account": self.account.to_json(now()), "replaced": self.replaced, "kept": self.place, "note": self.note})
    }
}

/// What can be known before a byte is sent.
#[derive(Debug, Clone, PartialEq)]
pub struct Preflight {
    pub account: Account,
    /// The file's size (0 for a URL).
    pub bytes: u64,
    pub notes: Vec<String>,
}

impl Preflight {
    pub fn to_json(&self) -> Value {
        json!({"account": self.account.to_json(now()), "bytes": self.bytes, "notes": self.notes})
    }
}

/// A published Reel.
#[derive(Debug, Clone, PartialEq)]
pub struct Published {
    /// `None`: uploaded and processed only (`upload_only`), not published.
    pub media_id: Option<String>,
    pub permalink: Option<String>,
    pub container: String,
    pub account: Account,
    /// `upload` (sent from the file) or `url` (fetched by Instagram).
    pub source: &'static str,
    pub bytes: u64,
    pub sent: upload::Sent,
    /// A container kept from an earlier run was used.
    pub from_earlier: bool,
    pub processing_seconds: f64,
    pub notes: Vec<String>,
    pub limits: Value,
}

impl Published {
    pub fn to_json(&self) -> Value {
        json!({
            "published": self.media_id.is_some(), "mediaId": self.media_id, "permalink": self.permalink, "url": self.permalink, "container": self.container,
            "account": self.account.to_json(now()), "source": self.source, "bytes": self.bytes,
            "resumed": self.sent.resumed, "restarted": self.sent.restarted, "resumedEarlierUpload": self.from_earlier,
            "processingSeconds": (self.processing_seconds * 10.0).round() / 10.0, "notes": self.notes, "limits": self.limits,
        })
    }
}

/// Instagram over one data directory.
pub struct Instagram {
    dir: PathBuf,
    endpoints: Endpoints,
    http: Arc<dyn Transport>,
    app: AppStore,
    accounts: AccountStore,
    limits: LimitBook,
    opener: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// How long a browser sign-in waits.
    pub sign_in_wait: Duration,
    pub upload_opts: upload::UploadOptions,
    pub poll: PollOptions,
    /// The first wait before an API call is retried.
    pub backoff: Duration,
}

impl Default for Instagram {
    fn default() -> Self {
        Self::new()
    }
}

impl Instagram {
    /// Over the app's data directory, Instagram's endpoints, the Keychain.
    pub fn new() -> Self {
        Self::in_data_dir(&data_root())
    }

    /// Over `root` (a FilmCraft data directory): `root/instagram`.
    pub fn in_data_dir(root: &Path) -> Self {
        let dir = root.join("instagram");
        Self {
            endpoints: Endpoints::instagram(),
            http: Arc::new(http::UreqTransport::default()),
            app: AppStore::system(&dir),
            accounts: AccountStore::new(&dir, Secrets::System),
            limits: LimitBook::new(&dir),
            opener: Arc::new(filmcraft_oauth::open_in_browser),
            sign_in_wait: auth::CONSENT_WAIT,
            upload_opts: upload::UploadOptions::default(),
            poll: PollOptions::default(),
            backoff: Duration::from_secs(1),
            dir,
        }
    }

    pub fn with_endpoints(mut self, endpoints: Endpoints) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// The calls over `http` (tests: plain HTTP to a mock).
    pub fn with_transport(mut self, http: Arc<dyn Transport>) -> Self {
        self.http = http;
        self
    }

    /// Where the app secret and the token go (tests: memory).
    pub fn with_secrets(mut self, secrets: Secrets) -> Self {
        self.app = match &secrets {
            Secrets::System => AppStore::system(&self.dir),
            Secrets::Files => AppStore::with_store(&self.dir, None),
            Secrets::Given(s) => AppStore::with_store(&self.dir, Some(s.clone())),
        };
        self.accounts = AccountStore::new(&self.dir, secrets);
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

    pub fn accounts(&self) -> &AccountStore {
        &self.accounts
    }

    pub fn limit_book(&self) -> &LimitBook {
        &self.limits
    }

    fn ctx<'a>(&'a self, token: &'a str, account: String) -> Ctx<'a> {
        Ctx { http: &*self.http, endpoints: &self.endpoints, token, account, backoff: self.backoff }
    }

    // ---------------------------------------------------------------------------------------
    // the app and the account
    // ---------------------------------------------------------------------------------------

    /// Keep the app's id, redirect URI and secret (`instagram setup`). `Ok` names where the
    /// secret went.
    pub fn setup(&self, app_id: &str, redirect_uri: &str, secret: &str) -> Result<String, IgError> {
        let app_id = auth::check_app_id(app_id).map_err(IgError::Setup)?;
        let (redirect_uri, _) = auth::check_redirect(redirect_uri).map_err(IgError::Setup)?;
        self.app.save(&AppConfig { app_id, redirect_uri }, secret)
    }

    /// The app config, `None` before setup.
    pub fn app_config(&self) -> Result<Option<AppConfig>, IgError> {
        self.app.config()
    }

    /// Whether a browser sign-in can start (the app is set up), from disk.
    pub fn connect_readiness(&self) -> Result<(), String> {
        self.app.ready()
    }

    /// Whether a publish can start (an account with a live token), from disk.
    pub fn publish_readiness(&self) -> Result<(), String> {
        let a = self.accounts.load().map_err(|e| e.to_string())?.ok_or_else(|| {
            format!("Instagram is not connected: run `{}` (or paste a token from the Meta App Dashboard: `{}`)", auth::SIGN_IN, auth::PASTE_TOKEN)
        })?;
        if !self.accounts.has_token() {
            return Err(format!("{} has no saved sign-in here; connect again: `{}`", a.label(), auth::SIGN_IN));
        }
        if a.expires_at <= now() {
            return Err(expired(&a).to_string());
        }
        Ok(())
    }

    /// Sign in in the browser. With a loopback redirect the listener catches the code; with any
    /// other, `paste` is shown the consent URL and returns the address the browser ended on.
    pub fn connect(&self, paste: Option<&mut dyn FnMut(&str) -> Option<String>>, progress: &mut dyn FnMut(&str)) -> Result<Connected, IgError> {
        let app = self.app.config()?.ok_or_else(|| IgError::Setup(self.app.ready().err().unwrap_or_default()))?;
        let secret = self.app.secret()?;
        let (_, redirect) = auth::check_redirect(&app.redirect_uri).map_err(IgError::Setup)?;
        let state = filmcraft_oauth::pkce::random_state().map_err(|e| IgError::Setup(e.to_string()))?;
        let url = auth::authorize_url(&self.endpoints, &app.app_id, &app.redirect_uri, &state);
        let code = match redirect {
            Redirect::Loopback(port) => {
                // listen before the browser opens, so a quick consent is not missed
                let listener = Loopback::bind_port(port).map_err(|e| IgError::Auth(e.to_string()))?;
                if (self.opener)(&url) {
                    progress("opened the browser: log in to Instagram as the professional account and allow FilmCraft; waiting for it to come back");
                } else {
                    progress(&format!("open this link in a browser to connect Instagram:\n{url}"));
                }
                let page = DonePage { app: "FilmCraft".into(), service: "Instagram".into() };
                listener
                    .wait_for_code(&state, self.sign_in_wait.min(filmcraft_oauth::client::MAX_WAIT), &page)
                    .map_err(|e| IgError::Auth(format!("{e}; run `{}` again", auth::SIGN_IN)))?
            }
            Redirect::Paste => {
                let Some(paste) = paste else {
                    return Err(IgError::Setup(format!(
                        "the registered redirect URI ({}) is a web page, so the sign-in's last step is pasting its address: run `{}` in a terminal, or register http://localhost:<port>/ and run `{}` with it",
                        app.redirect_uri,
                        auth::SIGN_IN,
                        auth::SETUP
                    )));
                };
                let opened = (self.opener)(&url);
                progress(if opened {
                    "opened the browser: log in to Instagram and allow FilmCraft"
                } else {
                    "open the link below in a browser and allow FilmCraft"
                });
                let back = paste(&url).ok_or_else(|| IgError::Auth("no address was pasted; nothing was connected".into()))?;
                auth::code_from_url(&back, &state)?
            }
        };
        progress("signed in; asking Instagram for a 60-day token");
        let short = auth::exchange_code(&*self.http, &self.endpoints, &app, &secret, &code)?;
        let long = auth::long_lived(&*self.http, &self.endpoints, &secret, &short.access_token)?;
        if let Some(p) = &short.permissions
            && !p.split(',').any(|x| x.trim() == auth::PUBLISH_PERMISSION)
        {
            return Err(IgError::Consent(format!(
                "the sign-in did not allow content publishing (granted: {p}); connect again and leave \"content publishing\" ticked: `{}`",
                auth::SIGN_IN
            )));
        }
        self.keep(long.access_token, long.expires_in, short.permissions.unwrap_or_default(), Via::Login, None, progress)
    }

    /// Connect with a token generated in the App Dashboard (long-lived, 60 days).
    pub fn connect_token(&self, token: &str, progress: &mut dyn FnMut(&str)) -> Result<Connected, IgError> {
        let token = auth::check_token(token).map_err(IgError::Invalid)?;
        progress("asking Instagram which account the token is for");
        let me = api::me(&self.ctx(&token, "the pasted token".into()), &mut || false)?;
        // a token from the dashboard may be days old: a refresh tells exactly when it ends (and
        // works only once it is a day old — otherwise the 60 days count from now)
        let (token, expires_in) = match auth::refresh(&*self.http, &self.endpoints, &token) {
            Ok(t) => {
                progress("the token was refreshed: it lasts 60 days from today");
                (t.access_token, t.expires_in)
            }
            Err(_) => (token, None),
        };
        self.keep(token, expires_in, String::new(), Via::Token, Some(me), progress)
    }

    /// Keep `token` and the account it is (`me`, or asked now).
    fn keep(
        &self,
        token: String,
        expires_in: Option<u64>,
        permissions: String,
        via: Via,
        me: Option<api::Me>,
        progress: &mut dyn FnMut(&str),
    ) -> Result<Connected, IgError> {
        let me = match me {
            Some(m) => m,
            None => {
                progress("asking Instagram which account it is");
                api::me(&self.ctx(&token, "the account".into()), &mut || false)?
            }
        };
        let t = now();
        let expires_at = t.saturating_add(expires_in.unwrap_or(auth::LONG_LIVED_SECS));
        let previous = self.accounts.load().ok().flatten();
        let tokens = TokenSet { access_token: token, refresh_token: None, expires_at, scope: permissions.replace(',', " ") };
        let stored = self.accounts.vault().save(&tokens).map_err(|e| IgError::Setup(e.to_string()))?;
        let account = Account {
            user_id: me.user_id,
            username: me.username,
            name: me.name,
            account_type: me.account_type,
            via,
            permissions,
            connected_at: t,
            refreshed_at: t,
            expires_at,
        };
        self.accounts.save(&account)?;
        Ok(Connected {
            replaced: previous.filter(|p| p.user_id != account.user_id).map(|p| p.label()),
            account,
            place: stored.place.to_string(),
            note: stored.note,
        })
    }

    /// Forget the account here (its token and file). Instagram has no revocation call for these
    /// tokens: the message says where the user removes the app's access.
    pub fn disconnect(&self) -> Result<Value, IgError> {
        let a = self.accounts.load()?.ok_or_else(|| IgError::Setup("no Instagram account is connected".into()))?;
        self.accounts.remove()?;
        Ok(json!({
            "disconnected": a.to_json(now()),
            "note": "the sign-in is forgotten here; to end it at Instagram too, remove the app in Instagram ▸ Settings ▸ Website permissions ▸ Apps and websites",
        }))
    }

    /// The account, the app and the limit counts, from disk.
    pub fn listing(&self) -> Result<Value, IgError> {
        let t = now();
        let account = self.accounts.load()?;
        let app = self.app.config().ok().flatten();
        Ok(json!({
            "account": account.as_ref().map(|a| a.to_json(t)),
            "signedIn": account.is_some() && self.accounts.has_token(),
            "ready": self.publish_readiness().err(),
            "app": app.map(|a| json!({"appId": a.app_id, "redirectUri": a.redirect_uri})),
            "setup": self.app.ready().err(),
            "limits": self.limits.to_json(t),
        }))
    }

    /// The account and a live token, refreshed once it is a day old (a refresh that fails for
    /// a passing reason leaves the current token in use, and says so).
    fn signed(&self, cancelled: &mut dyn FnMut() -> bool) -> Result<(Account, String, Vec<String>), IgError> {
        let mut a = self.accounts.load()?.ok_or_else(|| IgError::Setup(format!("Instagram is not connected: run `{}`", auth::SIGN_IN)))?;
        let Some((mut tokens, _)) = self.accounts.vault().load().map_err(|e| IgError::Setup(e.to_string()))? else {
            return Err(IgError::Auth(format!("{} has no saved sign-in here; connect again: `{}`", a.label(), auth::SIGN_IN)));
        };
        let t = now();
        if a.expires_at <= t || tokens.expires_at <= t {
            return Err(expired(&a));
        }
        let mut notes = Vec::new();
        if t.saturating_sub(a.refreshed_at) >= auth::REFRESH_AFTER_SECS && !cancelled() {
            match auth::refresh(&*self.http, &self.endpoints, &tokens.access_token) {
                Ok(fresh) => {
                    tokens = TokenSet {
                        access_token: fresh.access_token,
                        expires_at: t.saturating_add(fresh.expires_in.unwrap_or(auth::LONG_LIVED_SECS)),
                        ..tokens
                    };
                    self.accounts.vault().save(&tokens).map_err(|e| IgError::Setup(e.to_string()))?;
                    a.refreshed_at = t;
                    a.expires_at = tokens.expires_at;
                    self.accounts.save(&a)?;
                }
                Err(IgError::Auth(m)) => return Err(IgError::Auth(m)),
                Err(e) => {
                    notes.push(format!("the sign-in could not be refreshed now ({e}); it still works for {} days", a.expires_at.saturating_sub(t) / 86_400))
                }
            }
        }
        Ok((a, tokens.access_token, notes))
    }

    /// Refresh the token now (it must be a day old), `instagram refresh`.
    pub fn refresh_now(&self) -> Result<Account, IgError> {
        let mut a = self.accounts.load()?.ok_or_else(|| IgError::Setup(format!("Instagram is not connected: run `{}`", auth::SIGN_IN)))?;
        let Some((tokens, _)) = self.accounts.vault().load().map_err(|e| IgError::Setup(e.to_string()))? else {
            return Err(IgError::Auth(format!("{} has no saved sign-in here; connect again: `{}`", a.label(), auth::SIGN_IN)));
        };
        let t = now();
        if a.expires_at <= t {
            return Err(expired(&a));
        }
        let fresh = auth::refresh(&*self.http, &self.endpoints, &tokens.access_token)?;
        let tokens = TokenSet { access_token: fresh.access_token, expires_at: t.saturating_add(fresh.expires_in.unwrap_or(auth::LONG_LIVED_SECS)), ..tokens };
        self.accounts.vault().save(&tokens).map_err(|e| IgError::Setup(e.to_string()))?;
        a.refreshed_at = t;
        a.expires_at = tokens.expires_at;
        self.accounts.save(&a)?;
        Ok(a)
    }

    /// Instagram's count of posts in the last 24 h beside this machine's (network).
    pub fn limits(&self) -> Result<Value, IgError> {
        let (a, token, notes) = self.signed(&mut || false)?;
        let ctx = self.ctx(&token, a.label());
        let t = now();
        let reported = api::publishing_limit(&ctx, &a.user_id, t, &mut || false)?;
        self.limits.record_reported(t, reported);
        let mut v = self.limits.to_json(t);
        v["account"] = json!(a.label());
        v["notes"] = json!(notes);
        Ok(v)
    }

    // ---------------------------------------------------------------------------------------
    // publishing
    // ---------------------------------------------------------------------------------------

    /// Everything that can be checked without the network: the account and its token, the file
    /// (an MP4 or MOV of at most 300 MB) or URL, this machine's limit counts.
    pub fn preflight(&self, req: &ReelRequest) -> Result<Preflight, IgError> {
        let account = self.accounts.load()?.ok_or_else(|| IgError::Setup(self.publish_readiness().err().unwrap_or_default()))?;
        self.publish_readiness().map_err(IgError::Auth)?;
        let mut notes = Vec::new();
        let mut bytes = 0;
        if let Some(p) = &req.path {
            let path = Path::new(p);
            let meta = std::fs::metadata(path).map_err(|e| IgError::Invalid(format!("{}: {e}", path.display())))?;
            if !meta.is_file() || meta.len() == 0 {
                return Err(IgError::Invalid(format!("{} is not a video file", path.display())));
            }
            let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            if !matches!(ext.as_str(), "mp4" | "mov" | "m4v") {
                return Err(IgError::Invalid(format!(
                    "{} is not an MP4 or MOV: Instagram takes Reels in those containers (export with an H.264 preset, or use the pre-flight's fixed copy)",
                    path.display()
                )));
            }
            bytes = meta.len();
            if bytes > filmcraft_engine::instagram::MAX_REEL_BYTES && req.video_url.is_none() {
                return Err(IgError::Invalid(format!("{} is {} MB; Instagram takes Reels of at most 300 MB", path.display(), bytes / (1024 * 1024))));
            }
        }
        if let Some(u) = &req.video_url {
            filmcraft_engine::instagram::check_public_url("videoUrl", u).map_err(IgError::Invalid)?;
            notes.push("Instagram fetches the video from its URL; it must stay reachable until the Reel is published".into());
        }
        let t = now();
        self.limits.check_container(t).map_err(IgError::Limit)?;
        self.limits.check_post(t, None).map_err(IgError::Limit)?;
        if account.expires_at.saturating_sub(t) < 7 * 86_400 {
            notes.push(format!(
                "the sign-in runs out in {} days unless it is refreshed (publishing refreshes it)",
                account.expires_at.saturating_sub(t) / 86_400
            ));
        }
        Ok(Preflight { account, bytes, notes })
    }

    /// Publish a Reel: a container (or the one kept from an earlier run of this publish), the
    /// video sent to it (or fetched from `video_url`), processing awaited (bounded), then
    /// `media_publish`. `progress(sent, total, status)` returning `false` cancels; a container
    /// already made is kept for the next run.
    pub fn publish(&self, req: &ReelRequest, progress: &mut dyn FnMut(u64, u64, &str) -> bool) -> Result<Published, IgError> {
        let pre = self.preflight(req)?;
        let total = pre.bytes;
        if !progress(0, total, &format!("Signing in to Instagram as {}", pre.account.label())) {
            return Err(IgError::Cancelled);
        }
        let (account, token, mut notes) = self.signed(&mut || false)?;
        notes.extend(pre.notes.iter().cloned());
        let ctx = self.ctx(&token, account.label());
        let sessions = upload::Sessions::new(&self.dir);
        let mut sent = upload::Sent::default();
        let mut from_earlier = false;
        let (container, key, source) = match (&req.path, &req.video_url) {
            (Some(p), None) => {
                let path = Path::new(p);
                let key = upload::Sessions::key(&account.user_id, path, total, req);
                let mut kept = None;
                if let Some(k) = sessions.get(&key, now()) {
                    let mut stop = || !progress(0, total, "Looking at the upload kept from an earlier run");
                    match api::status(&ctx, &k.container, &mut stop) {
                        Ok(s) if s.code == "FINISHED" || s.code == "IN_PROGRESS" => kept = Some(k),
                        Ok(s) if s.code == "PUBLISHED" => {
                            sessions.remove(&key);
                            return Err(IgError::Processing(format!(
                                "this exact publish already went out (container {}); change the caption or the file to post it again",
                                k.container
                            )));
                        }
                        Err(IgError::Cancelled) => return Err(IgError::Cancelled),
                        Err(e @ IgError::Auth(_)) => return Err(e),
                        _ => sessions.remove(&key),
                    }
                }
                let k = match kept {
                    Some(k) => {
                        from_earlier = true;
                        k
                    }
                    None => {
                        self.limits.check_container(now()).map_err(IgError::Limit)?;
                        let mut stop = || !progress(0, total, "Making the Reel container");
                        let (id, named) = api::create_reel(&ctx, &account.user_id, api::Source::Upload, req, &mut stop)?;
                        self.limits.record_container(now());
                        let uri = upload::upload_uri(&self.endpoints.rupload, &self.endpoints.version, &id, named.as_deref());
                        let k = upload::Kept { container: id, uri, created_at: now(), uploaded: false };
                        sessions.put(&key, &k);
                        k
                    }
                };
                if !k.uploaded {
                    // a kept container: carry on from where Instagram got to (when it says)
                    let start = if from_earlier {
                        let mut stop = || !progress(0, total, "Asking Instagram where the earlier upload stopped");
                        match api::upload_status(&ctx, &k.container, &mut stop) {
                            Ok(s) => s.bytes_received.filter(|n| *n < total).unwrap_or(0),
                            Err(IgError::Cancelled) => return Err(IgError::Cancelled),
                            Err(_) => 0,
                        }
                    } else {
                        0
                    };
                    sent = upload::send(&ctx, &k.uri, &k.container, path, total, start, &self.upload_opts, &mut *progress)?;
                    sessions.put(&key, &upload::Kept { uploaded: true, ..k.clone() });
                }
                (k.container, Some(key), "upload")
            }
            (_, Some(u)) => {
                self.limits.check_container(now()).map_err(IgError::Limit)?;
                let mut stop = || !progress(0, total, "Making the Reel container (Instagram fetches the video from its URL)");
                let (id, _) = api::create_reel(&ctx, &account.user_id, api::Source::Url(u), req, &mut stop)?;
                self.limits.record_container(now());
                (id, None, "url")
            }
            (None, None) => return Err(IgError::Invalid("nothing to publish: give a file or a public video URL".into())),
        };
        // processing
        let started = Instant::now();
        let mut wait = self.poll.first;
        loop {
            let mut stop = || !progress(total, total, "Instagram is processing the video");
            let s = api::status(&ctx, &container, &mut stop)?;
            match s.code.as_str() {
                "FINISHED" => break,
                "PUBLISHED" => {
                    if let Some(k) = &key {
                        sessions.remove(k);
                    }
                    return Err(IgError::Processing(format!("container {container} was already published")));
                }
                "ERROR" | "EXPIRED" => {
                    if let Some(k) = &key {
                        sessions.remove(k);
                    }
                    let why = api::subcode_in(&s.detail).and_then(api::subcode_words).map(str::to_string).unwrap_or_else(|| {
                        if s.code == "EXPIRED" {
                            "the container expired before it was published (24 hours)".into()
                        } else {
                            "Instagram could not process the video".into()
                        }
                    });
                    let detail = if s.detail.is_empty() { String::new() } else { format!(" (Instagram said: {})", s.detail) };
                    return Err(IgError::Processing(format!("{why}{detail}; run the pre-flight (it names what Instagram does not take), then publish again")));
                }
                _ => {}
            }
            let elapsed = started.elapsed();
            if elapsed >= self.poll.limit {
                return Err(IgError::Processing(format!(
                    "Instagram is still processing the video after {} minutes; run the same publish again later to finish it (the upload is kept for 23 hours)",
                    self.poll.limit.as_secs().div_ceil(60)
                )));
            }
            let secs = elapsed.as_secs();
            let mut stop = || !progress(total, total, &format!("Instagram is processing the video ({secs} s)"));
            api::pause(wait.min(self.poll.limit.saturating_sub(elapsed)), &mut stop)?;
            wait = (wait * 2).min(self.poll.every);
        }
        let processing_seconds = started.elapsed().as_secs_f64();
        if req.upload_only {
            // staged: the container waits (kept for 23 hours); the same publish without
            // `upload_only` publishes it without sending the video again
            progress(total, total, "Uploaded and processed; not published (upload only)");
            return Ok(Published {
                media_id: None,
                permalink: None,
                container,
                account,
                source,
                bytes: total,
                sent,
                from_earlier,
                processing_seconds,
                notes,
                limits: self.limits.to_json(now()),
            });
        }
        // the limit, as Instagram counts it, before the post
        let t = now();
        let mut stop = || !progress(total, total, "Checking today's publishing limit");
        let fresh = match api::publishing_limit(&ctx, &account.user_id, t, &mut stop) {
            Ok(r) => {
                self.limits.record_reported(t, r.clone());
                Some(r)
            }
            Err(IgError::Cancelled) => return Err(IgError::Cancelled),
            Err(e) => {
                notes.push(format!("the publishing limit could not be read ({e}); publishing anyway"));
                None
            }
        };
        self.limits
            .check_post(t, fresh.as_ref())
            .map_err(|e| IgError::Limit(format!("{e}; the uploaded video is kept for 23 hours: run the same publish again then")))?;
        if !progress(total, total, "Publishing the Reel") {
            return Err(IgError::Cancelled);
        }
        let media_id = api::publish(&ctx, &account.user_id, &container, &mut || false)?;
        self.limits.record_post(now());
        if let Some(k) = &key {
            sessions.remove(k);
        }
        let permalink = api::permalink(&ctx, &media_id, &mut || false);
        Ok(Published {
            media_id: Some(media_id),
            permalink,
            container,
            account,
            source,
            bytes: total,
            sent,
            from_earlier,
            processing_seconds,
            notes,
            limits: self.limits.to_json(now()),
        })
    }
}

fn expired(a: &Account) -> IgError {
    IgError::Auth(format!(
        "the Instagram sign-in of {} ran out (tokens last 60 days unless refreshed, and FilmCraft refreshes them only while it is used); connect again: `{}` (or paste a new token: `{}`)",
        a.label(),
        auth::SIGN_IN,
        auth::PASTE_TOKEN
    ))
}

// -------------------------------------------------------------------------------------------
// the engine's publisher
// -------------------------------------------------------------------------------------------

/// How long [`EnginePublisher::ready`] trusts its last look at the disk (it runs while drawing).
const READY_TTL: Duration = Duration::from_secs(2);

type Cached = Mutex<Option<(Instant, Result<(), String>)>>;

/// The engine's `instagram.*` publisher over a data directory.
pub struct EnginePublisher {
    ig: Instagram,
    ready: Cached,
    connect_ready: Cached,
}

/// The publisher hosts install as `Session::instagram.publisher`.
pub fn engine_publisher() -> Arc<dyn filmcraft_engine::instagram::InstagramPublisher> {
    Arc::new(EnginePublisher::new(Instagram::new()))
}

impl EnginePublisher {
    pub fn new(ig: Instagram) -> Self {
        Self { ig, ready: Mutex::new(None), connect_ready: Mutex::new(None) }
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

impl filmcraft_engine::instagram::InstagramPublisher for EnginePublisher {
    fn connect_ready(&self) -> Result<(), String> {
        cached(&self.connect_ready, || self.ig.connect_readiness())
    }

    fn ready(&self) -> Result<(), String> {
        cached(&self.ready, || self.ig.publish_readiness())
    }

    fn account(&self) -> Result<Value, String> {
        self.ig.listing().map_err(|e| e.to_string())
    }

    fn connect(&self, progress: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        let mut report = |m: &str| *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = clean(m, 300);
        let r = self.ig.connect(None, &mut report).map(|c| c.to_json()).map_err(|e| e.to_string());
        self.forget_readiness();
        r
    }

    fn disconnect(&self) -> Result<Value, String> {
        let r = self.ig.disconnect().map_err(|e| e.to_string());
        self.forget_readiness();
        r
    }

    fn limits(&self) -> Result<Value, String> {
        self.ig.limits().map_err(|e| e.to_string())
    }

    fn check(&self, req: &ReelRequest) -> Result<Value, String> {
        self.ig.preflight(req).map(|p| p.to_json()).map_err(|e| e.to_string())
    }

    fn publish(&self, req: &ReelRequest, progress: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        use std::sync::atomic::Ordering;
        let mut report = |sent: u64, total: u64, status: &str| {
            if total > 0 {
                progress.total.store(total, Ordering::Relaxed);
                progress.done.store(sent, Ordering::Relaxed);
            }
            *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = status.to_string();
            !progress.cancel.load(Ordering::Relaxed)
        };
        let r = self.ig.publish(req, &mut report).map(|p| p.to_json()).map_err(|e| e.to_string());
        self.forget_readiness();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cleaned_and_endpoints_built() {
        assert_eq!(clean(" a\u{7}b\nc ", 10), "abc");
        let e = Endpoints::at("http://127.0.0.1:9/");
        assert_eq!(
            (e.graph.as_str(), e.token.as_str(), e.rupload.as_str()),
            ("http://127.0.0.1:9", "http://127.0.0.1:9/oauth/access_token", "http://127.0.0.1:9/ig-api-upload")
        );
        assert!(Endpoints::instagram().authorize.starts_with("https://www.instagram.com/"));
    }
}
