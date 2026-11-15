//! Publish to YouTube: connect channels, list them, choose the default, disconnect one, and
//! upload a file (typically a finished export) as a background job with byte progress and
//! cancellation like an export (`jobs.list`, `jobs.cancel`).
//!
//! | command | what it does |
//! |---|---|
//! | `youtube.connect {default?, playlists?, wait?}` | job: sign in in the browser; Google's account chooser picks the channel (a Brand Account's channel too); the channel is named by `channels.list?mine=true` and its sign-in kept apart from the others |
//! | `youtube.channels` | the connected channels, the default and today's quota use (from disk, no network) |
//! | `youtube.setDefault {channel}` | the channel uploads go to when none is named |
//! | `youtube.disconnect {channel}` | revoke the channel's sign-in at Google and forget it here |
//! | `youtube.upload {path, title, channel?, description?, tags?, category?, privacy?, madeForKids?, publishAt?, playlist?, thumbnail?, sidecars?, notifySubscribers?, wait?}` | job: the resumable upload, then the thumbnail and the playlist |
//!
//! A channel is named by its id (`UC…`), its handle (`@name`) or its title. The video's
//! metadata is checked here, before any job starts, against YouTube's limits ([`check_title`],
//! [`check_description`], [`check_tags`], [`check_category`], [`check_publish_at`]), so a bad
//! parameter never costs quota.
//!
//! The engine only runs them. The calls themselves (OAuth, the YouTube Data API) live in
//! `filmcraft-youtube`, a layer above, which the host (desktop app, CLI, MCP server) installs as
//! [`YouTubeState::publisher`]. Without one (the web build, tests) the commands are disabled and
//! say so. A finished job's answer (the channel connected, or the video uploaded) is in its
//! `jobs.list` entry as `youtube`, and in the command's answer with `wait: true`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p};
use crate::{EngineError, Result, Session};

/// YouTube's limit on a title, in characters.
pub const MAX_TITLE_CHARS: usize = 100;
/// YouTube's limit on a description, in bytes (UTF-8).
pub const MAX_DESCRIPTION_BYTES: usize = 5000;
/// YouTube's limit on the tags together, in characters: commas between them count, and a tag
/// with a space counts its quotation marks.
pub const MAX_TAGS_CHARS: usize = 500;
/// The most tags taken (each is at least one character, plus its comma).
pub const MAX_TAGS: usize = 250;
/// The longest channel reference (id, handle or title) and playlist reference taken.
pub const MAX_REF_CHARS: usize = 200;
/// The longest path taken.
const MAX_PATH_CHARS: usize = 4096;

/// What the commands say without a publisher.
const NOT_HERE: &str = "YouTube is not available in this app";

/// The video categories a channel can pick (YouTube's assignable ones in most regions): id and
/// name. `category` takes either.
pub const CATEGORIES: &[(&str, &str)] = &[
    ("1", "Film & Animation"),
    ("2", "Autos & Vehicles"),
    ("10", "Music"),
    ("15", "Pets & Animals"),
    ("17", "Sports"),
    ("19", "Travel & Events"),
    ("20", "Gaming"),
    ("22", "People & Blogs"),
    ("23", "Comedy"),
    ("24", "Entertainment"),
    ("25", "News & Politics"),
    ("26", "Howto & Style"),
    ("27", "Education"),
    ("28", "Science & Technology"),
    ("29", "Nonprofits & Activism"),
];
/// The category of an upload that names none (YouTube's own default).
pub const DEFAULT_CATEGORY: &str = "22";

/// Who can watch the video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Privacy {
    #[default]
    Private,
    Unlisted,
    Public,
}

impl Privacy {
    pub fn as_str(self) -> &'static str {
        match self {
            Privacy::Private => "private",
            Privacy::Unlisted => "unlisted",
            Privacy::Public => "public",
        }
    }

    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "private" => Ok(Privacy::Private),
            "unlisted" => Ok(Privacy::Unlisted),
            "public" => Ok(Privacy::Public),
            other => Err(format!("privacy `{}` is not one of private, unlisted, public", clip(other, 40))),
        }
    }
}

/// A video's metadata, checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoMeta {
    pub title: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    /// A category id (`"22"`).
    pub category: String,
    pub privacy: Privacy,
    /// `status.selfDeclaredMadeForKids` (false unless said).
    pub made_for_kids: bool,
    /// When a private video goes public (RFC 3339), YouTube's scheduled publishing.
    pub publish_at: Option<String>,
    /// Tell the channel's subscribers (YouTube's default: yes).
    pub notify_subscribers: bool,
}

impl Default for VideoMeta {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: None,
            tags: Vec::new(),
            category: DEFAULT_CATEGORY.into(),
            privacy: Privacy::Private,
            made_for_kids: false,
            publish_at: None,
            notify_subscribers: true,
        }
    }
}

/// One upload, as the publisher gets it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadRequest {
    pub path: String,
    /// Id, `@handle` or title; `None` is the default channel.
    pub channel: Option<String>,
    pub meta: VideoMeta,
    /// A playlist of the channel (id or exact title) to add the video to.
    pub playlist: Option<String>,
    /// A JPEG or PNG to set as the video's thumbnail.
    pub thumbnail: Option<String>,
    /// Take `<file>.chapters.txt` (appended to the description) and `<file>.thumbnail.jpg` (the
    /// thumbnail, unless one is given) from beside the file, as a showcase export leaves them.
    pub sidecars: bool,
    /// Upload even when the QA report beside the file (`<file>.qa.json`) lists unblurred faces
    /// of people (its privacy errors); otherwise the upload is refused before a byte is sent.
    #[serde(default)]
    pub allow_privacy_findings: bool,
}

/// What a sign-in asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConnectOptions {
    /// Make the connected channel the default.
    pub make_default: bool,
    /// Also ask for the permission to add videos to playlists (`youtube.force-ssl`).
    pub playlists: bool,
}

/// YouTube for the engine, installed by the host in [`YouTubeState::publisher`].
pub trait YouTubePublisher: Send + Sync {
    /// Whether a sign-in can start: the OAuth client file is there. `Err` says what to do.
    /// Cheap (no network): it decides whether `youtube.connect` is enabled.
    fn connect_ready(&self) -> std::result::Result<(), String>;
    /// Whether an upload can start: a channel is connected. `Err` names the command that
    /// connects one. Cheap (no network): it is asked while drawing.
    fn ready(&self) -> std::result::Result<(), String>;
    /// The connected channels, the default and today's quota use, from disk (no network).
    fn channels(&self) -> std::result::Result<Value, String>;
    /// Sign in (blocking: the browser and Google's account chooser, at most a few minutes),
    /// reporting what it waits for in `progress.status`. `Ok` is the connected channel.
    fn connect(&self, opts: ConnectOptions, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
    /// Make `channel` the default. `Ok` is the channel.
    fn set_default(&self, channel: &str) -> std::result::Result<Value, String>;
    /// Revoke `channel`'s sign-in at Google (best effort) and forget it here.
    fn disconnect(&self, channel: &str) -> std::result::Result<Value, String>;
    /// What can be checked before the job starts (no network): the file, the thumbnail, the
    /// channel, the permissions. `Ok` may carry notes (`{"channel", "shorts", …}`).
    fn check(&self, req: &UploadRequest) -> std::result::Result<Value, String>;
    /// Upload (blocking): bytes in `progress.done` / `progress.total`, a line in
    /// `progress.status`; stop with an error once `progress.cancel` is set. `Ok` is the video.
    fn upload(&self, req: &UploadRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
}

/// A job's answer, filled when it finishes.
pub type Outcome = Arc<Mutex<Option<Value>>>;

/// The engine's YouTube state: the host's publisher and what its jobs answered.
#[derive(Default)]
pub struct YouTubeState {
    /// `None` = not available: the commands are disabled.
    pub publisher: Option<Arc<dyn YouTubePublisher>>,
    /// Job id → its answer (`youtube` in `jobs.list`).
    pub outcomes: HashMap<u64, Outcome>,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "youtube.connect",
            label: "Connect a YouTube Channel",
            menu: &[],
            shortcut: None,
            params: r#"{"default":bool=false,"playlists":bool=false,"wait":bool=false}"#,
            enabled: can_connect,
            run: connect,
            journal: false,
        },
        CommandSpec {
            id: "youtube.channels",
            label: "YouTube Channels",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: installed,
            run: channels,
            journal: false,
        },
        CommandSpec {
            id: "youtube.setDefault",
            label: "Set the Default YouTube Channel",
            menu: &[],
            shortcut: None,
            params: r#"{"channel":str}"#,
            enabled: can_upload,
            run: set_default,
            journal: false,
        },
        CommandSpec {
            id: "youtube.disconnect",
            label: "Disconnect a YouTube Channel",
            menu: &[],
            shortcut: None,
            params: r#"{"channel":str}"#,
            enabled: can_upload,
            run: disconnect,
            journal: false,
        },
        CommandSpec {
            id: "youtube.upload",
            label: "Upload to YouTube",
            menu: &[],
            shortcut: None,
            params: r#"{"path":str,"title":str,"channel":str?,"description":str?,"tags":[str]?,"category":str?,"privacy":"private"|"unlisted"|"public"="private","madeForKids":bool=false,"publishAt":str?,"playlist":str?,"thumbnail":str?,"sidecars":bool=true,"notifySubscribers":bool=true,"allowPrivacyFindings":bool=false,"wait":bool=false}"#,
            enabled: can_upload,
            run: upload,
            journal: false,
        },
    ]
}

fn publisher(s: &Session) -> std::result::Result<&Arc<dyn YouTubePublisher>, String> {
    s.youtube.publisher.as_ref().ok_or_else(|| NOT_HERE.to_string())
}

fn installed(s: &Session) -> std::result::Result<(), String> {
    publisher(s).map(drop)
}

/// `youtube.connect` can run: a publisher is installed and the OAuth client file is there.
pub fn can_connect(s: &Session) -> std::result::Result<(), String> {
    publisher(s)?.connect_ready()
}

/// `youtube.upload` (and the channel commands) can run: a channel is connected.
pub fn can_upload(s: &Session) -> std::result::Result<(), String> {
    publisher(s)?.ready()
}

/// At most `n` characters of `s`.
fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// -------------------------------------------------------------------------------------------
// metadata checks (shared with filmcraft-youtube and the CLI)
// -------------------------------------------------------------------------------------------

/// A title YouTube takes: trimmed, not empty, at most [`MAX_TITLE_CHARS`] characters, no `<` or
/// `>` and no control characters (a line break included).
pub fn check_title(t: &str) -> std::result::Result<String, String> {
    let t = t.trim();
    if t.is_empty() {
        return Err("the title is empty; YouTube needs one".into());
    }
    let n = t.chars().count();
    if n > MAX_TITLE_CHARS {
        return Err(format!("the title is {n} characters; YouTube takes at most {MAX_TITLE_CHARS}"));
    }
    if t.contains(['<', '>']) {
        return Err("the title contains < or >, which YouTube does not take".into());
    }
    if t.chars().any(char::is_control) {
        return Err("the title contains a line break or another control character".into());
    }
    Ok(t.to_string())
}

/// A description YouTube takes (trimmed; empty is none): at most [`MAX_DESCRIPTION_BYTES`]
/// bytes, no `<` or `>`, no control characters but line breaks and tabs.
pub fn check_description(d: Option<&str>) -> std::result::Result<Option<String>, String> {
    let Some(d) = d.map(str::trim).filter(|d| !d.is_empty()) else { return Ok(None) };
    let d = d.replace("\r\n", "\n");
    if d.len() > MAX_DESCRIPTION_BYTES {
        return Err(format!("the description is {} bytes; YouTube takes at most {MAX_DESCRIPTION_BYTES}", d.len()));
    }
    if d.contains(['<', '>']) {
        return Err("the description contains < or >, which YouTube does not take".into());
    }
    if d.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
        return Err("the description contains a control character".into());
    }
    Ok(Some(d))
}

/// How YouTube counts tags against [`MAX_TAGS_CHARS`]: the commas between them, and quotation
/// marks around a tag with a space.
pub fn tags_length(tags: &[String]) -> usize {
    let quoted = |t: &String| t.chars().count().saturating_add(if t.contains(' ') { 2 } else { 0 });
    tags.iter().map(quoted).sum::<usize>().saturating_add(tags.len().saturating_sub(1))
}

/// Tags YouTube takes: each trimmed (empty ones and repeats dropped), no `<`, `>`, `,` or control
/// characters, within [`MAX_TAGS_CHARS`] together.
pub fn check_tags(tags: &[String]) -> std::result::Result<Vec<String>, String> {
    if tags.len() > MAX_TAGS {
        return Err(format!("{} tags; YouTube takes at most {MAX_TAGS_CHARS} characters of tags in all", tags.len()));
    }
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if t.is_empty() {
            continue;
        }
        if t.contains(['<', '>', ',']) || t.chars().any(char::is_control) {
            return Err(format!("the tag `{}` contains <, >, a comma or a control character", clip(&t, 40)));
        }
        if !out.iter().any(|o| o.eq_ignore_ascii_case(&t)) {
            out.push(t);
        }
    }
    let n = tags_length(&out);
    if n > MAX_TAGS_CHARS {
        return Err(format!("the tags come to {n} characters (commas, and quotes around tags with spaces, count); YouTube takes at most {MAX_TAGS_CHARS}"));
    }
    Ok(out)
}

/// A category id from an id (`28`) or a name (`Science & Technology`, any case).
pub fn check_category(c: Option<&str>) -> std::result::Result<String, String> {
    let Some(c) = c.map(str::trim).filter(|c| !c.is_empty()) else { return Ok(DEFAULT_CATEGORY.into()) };
    if let Some((id, _)) = CATEGORIES.iter().find(|(id, name)| *id == c || name.eq_ignore_ascii_case(c)) {
        return Ok((*id).to_string());
    }
    // another region's assignable category: an id YouTube checks itself
    if !c.is_empty() && c.len() <= 4 && c.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(c.to_string());
    }
    let names: Vec<String> = CATEGORIES.iter().map(|(id, n)| format!("{id} {n}")).collect();
    Err(format!("category `{}` is not a YouTube category; use an id or a name: {}", clip(c, 40), names.join(", ")))
}

/// A scheduled publishing time: RFC 3339 (`2026-11-01T18:00:00Z`, or with an offset).
pub fn check_publish_at(t: &str) -> std::result::Result<String, String> {
    let t = t.trim();
    let bad = || format!("publishAt `{}` is not a date and time like 2026-11-01T18:00:00Z", clip(t, 40));
    let b = t.as_bytes();
    let digits = |r: std::ops::Range<usize>| b.get(r).is_some_and(|s| !s.is_empty() && s.iter().all(u8::is_ascii_digit));
    let at = |i: usize, c: u8| b.get(i) == Some(&c);
    if b.len() > 40 || !(digits(0..4) && at(4, b'-') && digits(5..7) && at(7, b'-') && digits(8..10) && (at(10, b'T') || at(10, b't'))) {
        return Err(bad());
    }
    if !(digits(11..13) && at(13, b':') && digits(14..16) && at(16, b':') && digits(17..19)) {
        return Err(bad());
    }
    let num = |r: std::ops::Range<usize>| t.get(r).and_then(|s| s.parse::<u32>().ok()).unwrap_or(u32::MAX);
    if !(1..=12).contains(&num(5..7)) || !(1..=31).contains(&num(8..10)) || num(11..13) > 23 || num(14..16) > 59 || num(17..19) > 60 {
        return Err(bad());
    }
    let mut i = 19;
    if at(i, b'.') {
        i += 1;
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return Err(bad());
        }
    }
    let zone = t.get(i..).unwrap_or_default();
    let zb = zone.as_bytes();
    let ok_zone = zone == "Z"
        || zone == "z"
        || (zb.len() == 6
            && matches!(zb.first(), Some(b'+' | b'-'))
            && zb.get(1..3).is_some_and(|s| s.iter().all(u8::is_ascii_digit))
            && zb.get(3) == Some(&b':')
            && zb.get(4..6).is_some_and(|s| s.iter().all(u8::is_ascii_digit)));
    if !ok_zone {
        return Err(bad());
    }
    Ok(t.to_string())
}

/// A channel or playlist reference: trimmed, not empty, at most [`MAX_REF_CHARS`] characters,
/// no control characters.
pub fn check_ref(what: &str, r: &str) -> std::result::Result<String, String> {
    let r = r.trim();
    if r.is_empty() {
        return Err(format!("the {what} is empty"));
    }
    if r.chars().count() > MAX_REF_CHARS || r.chars().any(char::is_control) {
        return Err(format!("the {what} `{}` is not a {what} id, handle or title", clip(r, 40)));
    }
    Ok(r.to_string())
}

// -------------------------------------------------------------------------------------------
// parameters
// -------------------------------------------------------------------------------------------

/// An optional string parameter (trimmed; empty is absent).
fn opt_str(cmd: &str, p: &Value, key: &str) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.trim().to_string()).filter(|s| !s.is_empty())),
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

fn opt_bool(cmd: &str, p: &Value, key: &str, default: bool) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
    }
}

fn path_param(cmd: &str, p: &Value, key: &str, what: &str) -> Result<Option<String>> {
    let Some(path) = opt_str(cmd, p, key)? else { return Ok(None) };
    if path.chars().count() > MAX_PATH_CHARS || path.contains('\0') {
        return Err(bad(cmd, format!("`{key}` is not a usable path to {what}")));
    }
    Ok(Some(path))
}

/// `youtube.upload`'s parameters as a checked [`UploadRequest`].
pub fn parse_upload(cmd: &str, p: &Value) -> Result<UploadRequest> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let path = path_param(cmd, p, "path", "the file to upload")?.ok_or_else(|| bad(cmd, "need `path`: the file to upload"))?;
    let title = match p.get("title") {
        Some(Value::String(t)) => check_title(t).map_err(|e| bad(cmd, e))?,
        None | Some(Value::Null) => return Err(bad(cmd, "need `title`: the video's title (at most 100 characters)")),
        Some(_) => return Err(bad(cmd, "`title` must be a string")),
    };
    let description = match p.get("description") {
        None | Some(Value::Null) => None,
        Some(Value::String(d)) => check_description(Some(d)).map_err(|e| bad(cmd, e))?,
        Some(_) => return Err(bad(cmd, "`description` must be a string")),
    };
    let tags = match p.get("tags") {
        None | Some(Value::Null) => Vec::new(),
        // "a, b, c" as one string, as a shell passes it
        Some(Value::String(s)) => check_tags(&s.split(',').map(str::to_string).collect::<Vec<_>>()).map_err(|e| bad(cmd, e))?,
        Some(Value::Array(a)) => {
            if a.len() > MAX_TAGS {
                return Err(bad(cmd, format!("{} tags; YouTube takes at most {MAX_TAGS_CHARS} characters of tags in all", a.len())));
            }
            let list = a.iter().map(|t| t.as_str().map(str::to_string).ok_or_else(|| bad(cmd, "every tag must be a string"))).collect::<Result<Vec<_>>>()?;
            check_tags(&list).map_err(|e| bad(cmd, e))?
        }
        Some(_) => return Err(bad(cmd, "`tags` must be a list of strings")),
    };
    let category = check_category(opt_str(cmd, p, "category")?.as_deref()).map_err(|e| bad(cmd, e))?;
    let privacy = match opt_str(cmd, p, "privacy")? {
        Some(s) => Privacy::parse(&s).map_err(|e| bad(cmd, e))?,
        None => Privacy::Private,
    };
    let publish_at = opt_str(cmd, p, "publishAt")?.map(|t| check_publish_at(&t)).transpose().map_err(|e| bad(cmd, e))?;
    if publish_at.is_some() && privacy != Privacy::Private {
        return Err(bad(cmd, "`publishAt` schedules a private video to go public: leave `privacy` private"));
    }
    let channel = opt_str(cmd, p, "channel")?.map(|c| check_ref("channel", &c)).transpose().map_err(|e| bad(cmd, e))?;
    let playlist = opt_str(cmd, p, "playlist")?.map(|c| check_ref("playlist", &c)).transpose().map_err(|e| bad(cmd, e))?;
    let thumbnail = path_param(cmd, p, "thumbnail", "a JPEG or PNG thumbnail")?;
    Ok(UploadRequest {
        path,
        channel,
        meta: VideoMeta {
            title,
            description,
            tags,
            category,
            privacy,
            made_for_kids: opt_bool(cmd, p, "madeForKids", false)?,
            publish_at,
            notify_subscribers: opt_bool(cmd, p, "notifySubscribers", true)?,
        },
        playlist,
        thumbnail,
        sidecars: opt_bool(cmd, p, "sidecars", true)?,
        allow_privacy_findings: opt_bool(cmd, p, "allowPrivacyFindings", false)?,
    })
}

fn channel_param(cmd: &str, p: &Value) -> Result<String> {
    let c = opt_str(cmd, p, "channel")?.ok_or_else(|| bad(cmd, "need `channel`: its id (UC…), @handle or title (see youtube.channels)"))?;
    check_ref("channel", &c).map_err(|e| bad(cmd, e))
}

// -------------------------------------------------------------------------------------------
// commands
// -------------------------------------------------------------------------------------------

fn channels(s: &mut Session, _p: &Value) -> Result<Value> {
    let cmd = "youtube.channels";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.channels().map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn set_default(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "youtube.setDefault";
    let channel = channel_param(cmd, p)?;
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.set_default(&channel).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn disconnect(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "youtube.disconnect";
    let channel = channel_param(cmd, p)?;
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.disconnect(&channel).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn connect(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "youtube.connect";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    up.connect_ready().map_err(|e| bad(cmd, e))?;
    let opts = ConnectOptions { make_default: opt_bool(cmd, p, "default", false)?, playlists: opt_bool(cmd, p, "playlists", false)? };
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, "Connecting a YouTube channel".into());
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting for the Google sign-in in your browser".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.youtube.outcomes.insert(job.id, outcome.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-youtube-connect", move || {
        let t0 = web_time::Instant::now();
        let r = up.connect(opts, &prog);
        let r = r.map(|v| {
            let title = v["channel"]["title"].as_str().unwrap_or("the channel").to_string();
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
            format!("Connected {title}")
        });
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    answer(s, id, wait, json!({"job": id}))
}

fn upload(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "youtube.upload";
    let up = publisher(s).map_err(|e| bad(cmd, e))?.clone();
    let req = parse_upload(cmd, p)?;
    let bytes = s.services.file_size(&req.path).map_err(|e| bad(cmd, format!("{}: {e}", req.path)))?;
    if bytes == 0 {
        return Err(bad(cmd, format!("{} is empty", req.path)));
    }
    up.ready().map_err(|e| bad(cmd, e))?;
    let checked = up.check(&req).map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let name = std::path::Path::new(&req.path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| req.path.clone());
    let to = checked["channel"]["title"].as_str().map(|t| format!(" ({t})")).unwrap_or_default();
    let job = crate::narration::new_job(s, format!("Uploading {name} to YouTube{to}"));
    job.progress.total.store(bytes, std::sync::atomic::Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting to upload".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.youtube.outcomes.insert(job.id, outcome.clone());
    let path = req.path.clone();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-youtube-upload", move || {
        let t0 = web_time::Instant::now();
        let r = up.upload(&req, &prog);
        finish_upload(&prog, &res, &outcome, &req.path, bytes, r, t0);
    })?;
    answer(s, id, wait, json!({"job": id, "path": path, "checked": checked}))
}

/// With `wait`, the finished job's result and answer; a failure is the command's error.
fn answer(s: &Session, id: u64, wait: bool, mut out: Value) -> Result<Value> {
    if wait && let Some(j) = s.jobs.iter().find(|j| j.id == id) {
        let v = j.to_json();
        if let Some(e) = v["result"]["error"].as_str() {
            return Err(EngineError::Other(e.to_string()));
        }
        out["result"] = v["result"].clone();
        out["youtube"] = outcome_json(s, id);
    }
    Ok(out)
}

/// End an upload job: the video first, then the result that marks it finished.
fn finish_upload(
    prog: &filmcraft_export::Progress,
    res: &crate::narration::JobResult,
    outcome: &Outcome,
    path: &str,
    bytes: u64,
    r: std::result::Result<Value, String>,
    t0: web_time::Instant,
) {
    use std::sync::atomic::Ordering;
    let r = match r {
        Ok(video) => {
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(video);
            prog.done.store(bytes, Ordering::Relaxed);
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Uploaded to YouTube".into();
            let seconds = t0.elapsed().as_secs_f64();
            Ok(filmcraft_export::Report { path: path.to_string(), frames: 0, seconds, bytes, render_fps: 0.0, ..Default::default() })
        }
        Err(e) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
            *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
}

/// Job `job`'s answer (`null` while it runs, after a failure, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.youtube.outcomes.get(&job).and_then(|o| o.lock().unwrap_or_else(PoisonError::into_inner).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is a YouTube job (a sign-in or an upload).
pub fn is_job(s: &Session, job: u64) -> bool {
    s.youtube.outcomes.contains_key(&job)
}
