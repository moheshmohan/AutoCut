//! The MCP server. Tool names are snake_case.

use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine as _;
use filmcraft_autocut::Executor as _;
use filmcraft_engine::Session;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock as Content};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AutomationError, BridgeClient, png_rgba};

pub enum Backend {
    Headless(Arc<Mutex<Session>>),
    Bridge(Arc<BridgeClient>),
}

/// A panic payload as text.
fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// A failed blocking task: a panic is reported as an internal error and the session stays usable.
fn join_error(e: tokio::task::JoinError) -> AutomationError {
    match e.try_into_panic() {
        Ok(p) => AutomationError::Other(format!("internal error: {}", panic_message(p))),
        Err(e) => AutomationError::Other(e.to_string()),
    }
}

#[derive(Clone)]
pub struct FilmcraftMcp {
    backend: Arc<Backend>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListParams {
    /// Only commands whose id or label contains this text (case-insensitive).
    #[serde(default)]
    pub filter: Option<String>,
    /// Only commands that can run right now.
    #[serde(default)]
    pub enabled_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunParams {
    /// Command id, e.g. `sequence.addEdit`, `timeline.trim`, `effects.apply`, `file.import`.
    pub id: String,
    /// Parameters object (see `params` in `command_list`).
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchStep {
    /// Command id.
    pub id: String,
    /// Parameters object.
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchParams {
    /// Commands to run in order.
    pub steps: Vec<BatchStep>,
    /// Stop at the first failing step (default true).
    #[serde(default)]
    pub stop_on_error: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct RenderParams {
    /// Timeline time in seconds (default: the playhead).
    #[serde(default)]
    pub seconds: Option<f64>,
    /// Longest side of the PNG (default 960).
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShotParams {
    /// Panel to crop to (e.g. `Timeline`, `Program`); omit for the whole window.
    #[serde(default)]
    pub panel: Option<String>,
    /// Longest side of the PNG (default 1600).
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClickParams {
    /// Element id from `ui_elements` (e.g. `timeline.track.V1.lock`, `tools.Razor`, `project.item.12`).
    #[serde(default)]
    pub id: Option<String>,
    /// Or screen coordinates in points.
    #[serde(default)]
    pub x: Option<f64>,
    #[serde(default)]
    pub y: Option<f64>,
    /// left | right | middle
    #[serde(default)]
    pub button: Option<String>,
    /// Number of clicks (0–10; defaults to 1). Use 2 for double-click. Larger values are rejected.
    #[serde(default)]
    pub count: Option<u32>,
    /// {"shift":bool,"alt":bool,"command":bool,"ctrl":bool}
    #[serde(default)]
    pub modifiers: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DragParams {
    /// Start: {"id":..} or {"x":..,"y":..} (optionally "fx"/"fy" fractions inside the element).
    pub from: Value,
    /// End point, same forms.
    pub to: Value,
    /// Movement steps (0–1024; defaults to 12; 0 or 1 uses 2). Larger values are rejected.
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub modifiers: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KeyParams {
    /// Key with optional modifiers, e.g. `Space`, `Cmd+K`, `Shift+Delete`, `I`, `L`.
    pub key: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TypeParams {
    pub text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MediaImportParams {
    /// Absolute paths to media files (MP4/MOV, WAV/MP3/FLAC/AIFF/Ogg, PNG/JPEG/…).
    pub paths: Vec<String>,
    /// Bin ID to import into (optional).
    #[serde(default)]
    pub bin: Option<u64>,
    /// Import numbered stills as an image sequence (optional, default false).
    #[serde(default)]
    pub image_sequence: Option<bool>,
}

/// `file.import` params for a `media_import` call.
fn media_import_params(p: MediaImportParams) -> Value {
    let mut params = json!({"paths": p.paths});
    if let Some(bin) = p.bin {
        params["bin"] = json!(bin);
    }
    if let Some(seq) = p.image_sequence {
        params["imageSequence"] = json!(seq);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ElementsParams {
    /// Only element ids starting with this prefix (e.g. `timeline.clip.`).
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ControlParams {
    /// Any control-channel method (e.g. `ui.set`, `ui.scroll`, `ui.timeline.hit`, `ui.playback`).
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PhotosPickParams {
    /// At most this many items selectable in the picker (1–2000; default 2000).
    #[serde(default)]
    pub max_items: Option<u32>,
    /// Seconds to wait for the user to press Done (default 300, at most 3600). 0 opens the
    /// picker and returns at once with a pending session to resume.
    #[serde(default)]
    pub wait_seconds: Option<u64>,
    /// Resume this pending session (from an earlier `status: pending` result) instead of
    /// opening a new picker.
    #[serde(default)]
    pub session: Option<String>,
    /// When set, a local vision agent drives the picker itself: it types this query into the
    /// picker's search field, selects matching video tiles (up to max_items or 6), and
    /// presses Done — no human needed. Requires the Ollama grounding model
    /// (AUTOCUT_SCREEN_MODEL, default qwen2.5vl:7b — `ollama pull qwen2.5vl:7b`) and Screen
    /// Recording + Accessibility for the process running this server. The agent is
    /// best-effort: its report comes back as `autopick` in the result, and the pick itself
    /// is still authoritative.
    #[serde(default)]
    pub query: Option<String>,
    /// More searches (with or without `query`): the agent types them one after another into
    /// ONE picking session in ONE browser tab — the selection accumulates — and presses Done
    /// once; many run in as few sessions as the item cap (max_items, up to 2000) allows, each
    /// taking up to 20 searches, and searches that did not fit come back as `left` for another
    /// call. A search with no results is skipped. Date searches ("videos October 2022") are
    /// selected by the picker's date groups. Use with `all` for whole results.
    #[serde(default)]
    pub queries: Option<Vec<String>>,
    /// With `query`/`queries`: select every search result (the date groups of a date search,
    /// else the first tile and a Shift+click on the last) instead of up to max_items tiles.
    #[serde(default)]
    pub all: Option<bool>,
    /// Skip picked items created before this date (YYYY-MM-DD); they are not downloaded.
    #[serde(default)]
    pub since: Option<String>,
    /// Instead of `query`: pick every item of the user's Google Photos album with this exact
    /// title (the album is the seed of the video — the user adds what they want to it in
    /// Google Photos). The vision agent searches the picker for the exact title — Google's
    /// picker answers an album's exact title with the album's items — and selects that whole
    /// result; a search that shows nothing new, or a result too big for an album, selects
    /// nothing and reports why in `autopick`. Same prerequisites as `query`.
    #[serde(default)]
    pub album: Option<String>,
    /// With `album`: download only its videos (the album is selected whole, photos too — the
    /// picker cannot select by type — and its photos are skipped).
    #[serde(default)]
    pub videos_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PhotosAddParams {
    /// Media item ids of already-downloaded items (from googlephotos_pick or googlephotos_cached).
    pub ids: Vec<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct Insta360Params {
    /// Recordings by name: full stems (`VID_20260913_114003_00_093`), Studio's short names
    /// (`VID_093`) or file names. For `insta360_list` and `insta360_import` optional.
    #[serde(default)]
    pub names: Vec<String>,
    /// Only recordings shot on or after this date (YYYY-MM-DD, camera time).
    #[serde(default)]
    pub since: Option<String>,
    /// Only recordings shot on or before this date (YYYY-MM-DD).
    #[serde(default)]
    pub until: Option<String>,
    /// Folders to look in (default: Insta360 Studio's download folder and mounted camera cards).
    #[serde(default)]
    pub folders: Vec<String>,
    /// `insta360_import`: every recording found (when no names or dates are given).
    #[serde(default)]
    pub all: bool,
}

/// The most ids one googlephotos_add call takes.
const MAX_ADD_IDS: usize = 2000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PhotosUploadParams {
    /// Absolute path of the file to upload, typically a finished export (MP4/MOV/M4V/MKV/AVI…
    /// videos up to 10 GB, JPEG/PNG/HEIC/WebP/GIF/TIFF… photos up to 200 MB).
    pub path: String,
    /// Shown under the item in Google Photos (at most 1000 characters).
    #[serde(default)]
    pub description: Option<String>,
    /// Also put the item into this album (a title, or an album id): the album FilmCraft made
    /// with exactly that title, created when there is none. The user can then add more to it
    /// by hand in Google Photos. Needs `filmcraft-cli photos auth --albums` once.
    #[serde(default)]
    pub album: Option<String>,
}

/// `photos.upload` params for a `googlephotos_upload` call (a long job: progress and cancel).
fn photos_upload_params(p: PhotosUploadParams) -> Value {
    let mut params = json!({"path": p.path, "wait": true});
    if let Some(d) = p.description {
        params["description"] = json!(d);
    }
    if let Some(a) = p.album {
        params["album"] = json!(a);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct YouTubeUploadParams {
    /// Absolute path of the video to upload, typically a finished export (MP4, MOV, WebM, MKV…).
    pub path: String,
    /// The video's title (at most 100 characters, no < or >).
    pub title: String,
    /// The channel: its id (UC…), @handle or title (`youtube.channels` lists the connected
    /// ones); default: the default channel.
    #[serde(default)]
    pub channel: Option<String>,
    /// At most 5000 bytes, no < or >. A showcase's `<file>.chapters.txt` is appended.
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// A category id (28) or name (Science & Technology); default 22 (People & Blogs).
    #[serde(default)]
    pub category: Option<String>,
    /// private (default), unlisted or public.
    #[serde(default)]
    pub privacy: Option<String>,
    /// A playlist of the channel (its title or id); needs `youtube auth --playlists`.
    #[serde(default)]
    pub playlist: Option<String>,
    /// A JPEG or PNG thumbnail (default: `<file>.thumbnail.jpg` beside the video, if any).
    #[serde(default)]
    pub thumbnail: Option<String>,
    /// Made for kids (YouTube's selfDeclaredMadeForKids); default false.
    #[serde(default, rename = "madeForKids")]
    pub made_for_kids: Option<bool>,
}

/// `youtube.upload` params for a `youtube_upload` call (a long job: progress and cancel).
fn youtube_upload_params(p: YouTubeUploadParams) -> Value {
    let mut params = json!({"path": p.path, "title": p.title, "wait": true});
    for (k, v) in [
        ("channel", p.channel),
        ("description", p.description),
        ("category", p.category),
        ("privacy", p.privacy),
        ("playlist", p.playlist),
        ("thumbnail", p.thumbnail),
    ] {
        if let Some(v) = v {
            params[k] = json!(v);
        }
    }
    if let Some(t) = p.tags {
        params["tags"] = json!(t);
    }
    if let Some(k) = p.made_for_kids {
        params["madeForKids"] = json!(k);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PublishRunParams {
    /// youtube, youtubeShort or instagramReel.
    pub destination: String,
    /// Absolute path of the video, typically a finished export.
    #[serde(default)]
    pub path: Option<String>,
    /// YouTube: the title (default: the file's name), description, tags, privacy (private by
    /// default), channel (id, @handle or title), playlist.
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub privacy: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub playlist: Option<String>,
    /// Instagram Reel: the caption, whether it also shows in the profile grid, the cover frame
    /// (seconds in), a public videoUrl instead of sending the file, uploadOnly (stage it).
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default, rename = "shareToFeed")]
    pub share_to_feed: Option<bool>,
    #[serde(default, rename = "coverSeconds")]
    pub cover_seconds: Option<f64>,
    #[serde(default, rename = "videoUrl")]
    pub video_url: Option<String>,
    #[serde(default, rename = "uploadOnly")]
    pub upload_only: Option<bool>,
}

/// `publish.run` params for a `publish_run` call (a long job: progress and cancel).
fn publish_run_params(p: PublishRunParams) -> Value {
    let mut params = json!({"destination": p.destination, "wait": true});
    for (k, v) in [
        ("path", p.path),
        ("title", p.title),
        ("description", p.description),
        ("privacy", p.privacy),
        ("channel", p.channel),
        ("playlist", p.playlist),
        ("caption", p.caption),
        ("videoUrl", p.video_url),
    ] {
        if let Some(v) = v {
            params[k] = json!(v);
        }
    }
    if let Some(t) = p.tags {
        params["tags"] = json!(t);
    }
    if let Some(f) = p.share_to_feed {
        params["shareToFeed"] = json!(f);
    }
    if let Some(c) = p.cover_seconds {
        params["coverSeconds"] = json!(c);
    }
    if let Some(u) = p.upload_only {
        params["uploadOnly"] = json!(u);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct InstagramPublishParams {
    /// Absolute path of the video to publish (MP4 or MOV, typically a finished export): sent
    /// from this machine with Meta's resumable upload.
    #[serde(default)]
    pub path: Option<String>,
    /// Instead of (or besides) `path`: a public https:// address Instagram fetches the video from.
    #[serde(default, rename = "videoUrl")]
    pub video_url: Option<String>,
    /// At most 2,200 characters, 30 hashtags and 20 @mentions.
    #[serde(default)]
    pub caption: Option<String>,
    /// Show it in the profile grid as well as the Reels tab (default true).
    #[serde(default, rename = "shareToFeed")]
    pub share_to_feed: Option<bool>,
    /// The cover: the frame this many seconds in (default: the first frame).
    #[serde(default, rename = "coverSeconds")]
    pub cover_seconds: Option<f64>,
    /// The cover: a public https:// JPEG (8 MB at most, 9:16 recommended); wins over `coverSeconds`.
    #[serde(default, rename = "coverUrl")]
    pub cover_url: Option<String>,
}

/// `instagram.publish` params for an `instagram_publish` call (a long job: progress and cancel).
fn instagram_publish_params(p: InstagramPublishParams) -> Value {
    let mut params = json!({"wait": true});
    for (k, v) in [("path", p.path), ("videoUrl", p.video_url), ("caption", p.caption), ("coverUrl", p.cover_url)] {
        if let Some(v) = v {
            params[k] = json!(v);
        }
    }
    if let Some(f) = p.share_to_feed {
        params["shareToFeed"] = json!(f);
    }
    if let Some(c) = p.cover_seconds {
        params["coverSeconds"] = json!(c);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseAnalyzeParams {
    /// What to look for in every clip, singular: "robot", "vintage car". Or give a `prompt`.
    #[serde(default)]
    pub subject: Option<String>,
    /// Instead of `subject`: the video to make, in words ("a video of every robot since 2020"):
    /// the subject (and a date and title, when it names them) is read from it — by built-in
    /// rules, else the local classifier.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Only these media items (ids from project_inspect); default: every video and photo item.
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// The clips come from this Google Photos album (its exact title) instead of the project:
    /// the album the user filled in Google Photos is picked whole (the local screen agent opens
    /// it in Google's picker and selects all of it — Google lets apps read only albums they
    /// made, so the picker is the way in), its videos and photos downloaded and imported (files the
    /// project already holds are reused, not imported twice), and only those are analysed. Not
    /// with `items`. Needs `filmcraft-cli photos auth` once, the grounding model
    /// (`ollama pull qwen2.5vl:7b`) and Screen Recording + Accessibility for this process.
    #[serde(default)]
    pub album: Option<String>,
    /// Photos too (the default): an album's photos are downloaded (full resolution) and a robot
    /// that was only photographed is shown in its photos, with a slow pan and zoom; `false`
    /// leaves photos out (videos only).
    #[serde(default)]
    pub photos: Option<bool>,
    /// Only clips filmed on or after this date: "2020", "2020-06" or "2020-06-15" (clips without a
    /// capture date are left out).
    #[serde(default)]
    pub since: Option<String>,
    /// Frames sampled per clip, 1–8 (default 6).
    #[serde(default)]
    pub frames: Option<u32>,
    /// Look at this many spread-out frames of a clip first and skip the rest when none shows the
    /// subject — much faster over many clips that mostly do not (default 0: off).
    #[serde(default)]
    pub screen: Option<u32>,
    /// Most side-by-side comparisons for telling the same robot across clips (default 150; 0:
    /// by name only).
    #[serde(default)]
    pub compare: Option<u32>,
    /// Analyse every clip again. By default a clip analysed before — the same file (its Google
    /// Photos id, or its size and content), subject, model and frames/screen — comes from the
    /// analysis cache in the data directory (`cached` in the summary counts them).
    #[serde(default)]
    pub fresh: Option<bool>,
    /// Also group the clips into entries and write the voice-over (`autocut.makeShowcase`),
    /// default false: the result then holds the plan to review.
    #[serde(default)]
    pub plan: Option<bool>,
    /// With `plan`: the video's title (default "Every <Subject> I've Filmed").
    #[serde(default)]
    pub title: Option<String>,
    /// With `plan`: the narration voice (`clone:<name>` for a cloned one; "" none).
    #[serde(default)]
    pub voice: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct TrailerParams {
    /// The finished show's sequence (its narration is taken from it; without a plan, its entries).
    #[serde(default)]
    pub sequence: Option<u64>,
    /// The show's plan file (entries, shots, dates, redact targets; default: the kept plan).
    #[serde(default)]
    pub plan_path: Option<String>,
    /// The show's build report file (what `filmcraft-cli showcase` prints: where each shot played,
    /// what the privacy pass hid; default: the last build).
    #[serde(default)]
    pub report_path: Option<String>,
    /// Length, 10–90 seconds (default 34).
    #[serde(default)]
    pub seconds: Option<f64>,
    /// "9:16" (default), "1:1" or "4:5".
    #[serde(default)]
    pub aspect: Option<String>,
    /// Whose safe zones the words keep inside: "reels" (default), "youtube", "tiktok", "vertical".
    #[serde(default)]
    pub platform: Option<String>,
    /// Robots in the montage after the hook, 3–30 (default: 12–18 as the length allows).
    #[serde(default)]
    pub robots: Option<u32>,
    /// The entry label of the hook's robot (default: the most striking).
    #[serde(default)]
    pub hook: Option<String>,
    /// Where the hook starts in its clip, seconds (default: around its best moment).
    #[serde(default)]
    pub hook_at: Option<f64>,
    /// Where robots' shots start: {"<entry label>": seconds into its clip, or {"item": another
    /// clip of the entry, "seconds": …}}.
    #[serde(default)]
    pub at: Option<serde_json::Map<String, Value>>,
    /// The hook's title (default "<N> robots · <first year> → <last year>"; " · " splits lines).
    #[serde(default)]
    pub title: Option<String>,
    /// The show's first narration sentence over the hook (default: when the show is narrated).
    #[serde(default)]
    pub hook_voice: Option<bool>,
    /// The end card's logo image (default: the plan's outro logo).
    #[serde(default)]
    pub logo: Option<String>,
    /// The channel's handle on the end card (default: the plan's outro handle).
    #[serde(default)]
    pub channel: Option<String>,
    /// What the end card says (default "Full video on YouTube").
    #[serde(default)]
    pub end_text: Option<String>,
    /// The full video's address, for the caption file (Reels' captions show no links).
    #[serde(default)]
    pub link: Option<String>,
    /// Entry labels to leave out, and to keep in.
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// The music's tempo (default 124), mood (default "energetic") and seed (default 7).
    #[serde(default)]
    pub bpm: Option<f64>,
    #[serde(default)]
    pub mood: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// "faces,plates", "faces", "plates", or "off" (default: as the show was redacted).
    #[serde(default)]
    pub redact: Option<String>,
    /// Export the trailer here (MP4; a cover, caption, frame sheet and QA beside it).
    #[serde(default)]
    pub export: Option<String>,
}

/// `autocut.trailer` params for a `trailer_make` call (a long job).
fn trailer_params(p: TrailerParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    let text = |t: Option<String>| t.filter(|t| !t.trim().is_empty()).map(Value::String);
    put("sequence", p.sequence.map(|v| json!(v)));
    put("planPath", text(p.plan_path));
    put("reportPath", text(p.report_path));
    put("seconds", p.seconds.map(|v| json!(v)));
    put("aspect", text(p.aspect));
    put("platform", text(p.platform));
    put("robots", p.robots.map(|v| json!(v)));
    put("hook", text(p.hook));
    put("hookAt", p.hook_at.map(|v| json!(v)));
    put("at", p.at.filter(|a| !a.is_empty()).map(Value::Object));
    put("title", text(p.title));
    put("hookVoice", p.hook_voice.map(Value::Bool));
    put("logo", text(p.logo));
    put("channel", text(p.channel));
    put("endText", text(p.end_text));
    put("link", text(p.link));
    put("exclude", p.exclude.map(|v| json!(v)));
    put("include", p.include.map(|v| json!(v)));
    put("bpm", p.bpm.map(|v| json!(v)));
    put("mood", text(p.mood));
    put("seed", p.seed.map(|v| json!(v)));
    put(
        "redact",
        p.redact
            .filter(|r| !r.trim().is_empty())
            .map(|r| if matches!(r.trim().to_ascii_lowercase().as_str(), "off" | "none" | "false") { json!(false) } else { json!(r) }),
    );
    put("export", text(p.export));
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShortsParams {
    /// A media item (video with speech) to make shorts of; it is put in a new sequence. Not with
    /// `sequence`.
    #[serde(default)]
    pub item: Option<u64>,
    /// The sequence to make shorts of (default: the active one).
    #[serde(default)]
    pub sequence: Option<u64>,
    /// How many shorts, 1–20 (default 3).
    #[serde(default)]
    pub count: Option<u32>,
    /// Shortest short, seconds (default 15).
    #[serde(default)]
    pub min_seconds: Option<f64>,
    /// Longest short, seconds (default 60).
    #[serde(default)]
    pub max_seconds: Option<f64>,
    /// "9:16" (default), "1:1", "4:5" or "16:9".
    #[serde(default)]
    pub aspect: Option<String>,
    /// How the crop follows the speaker: "smooth" (default) or "responsive".
    #[serde(default)]
    pub motion: Option<String>,
    /// Where nobody is followed: "blur" (default: the whole picture over a blurred, dimmed copy
    /// of itself), "letterbox" (black bars) or "center" (a centred crop).
    #[serde(default)]
    pub fallback: Option<String>,
    /// Whose interface the captions and the title keep clear of: "youtube" (default), "reels",
    /// "tiktok" or "vertical" (all three).
    #[serde(default)]
    pub platform: Option<String>,
    /// Words that make a passage more interesting (e.g. "robots launch").
    #[serde(default)]
    pub topic: Option<String>,
    /// Captions from the transcript (default true).
    #[serde(default)]
    pub captions: Option<bool>,
    /// A title card over each short's first seconds (default false).
    #[serde(default)]
    pub title: Option<bool>,
    /// Export each short into this folder (MP4, captions burned in).
    #[serde(default)]
    pub export: Option<String>,
    /// Upload each export to YouTube (needs `export`; private unless `youtube_privacy`).
    #[serde(default)]
    pub upload_youtube: Option<bool>,
    /// The YouTube channel (id, @handle or title; default: the default channel).
    #[serde(default)]
    pub youtube_channel: Option<String>,
    /// "private" (default), "unlisted" or "public".
    #[serde(default)]
    pub youtube_privacy: Option<String>,
    /// Tags for the uploads.
    #[serde(default)]
    pub youtube_tags: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct SpeechUnderstandParams {
    /// Media items to understand (default: the Project panel selection, else the active
    /// sequence's clips).
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// "en", "ml" or "auto" (default: detected). For "ml" the Malayalam model and for "en"
    /// Whisper large-v3-turbo hear, when downloaded.
    #[serde(default)]
    pub language: Option<String>,
    /// A speech model id (`transcript.models`), instead of the language's.
    #[serde(default)]
    pub model: Option<String>,
    /// Tell the speakers apart (default true).
    #[serde(default)]
    pub diarize: Option<bool>,
    /// Look for a face on screen for the talking-head hint (default true).
    #[serde(default)]
    pub faces: Option<bool>,
    /// Shortest pause reported, seconds (default 0.5).
    #[serde(default)]
    pub min_pause: Option<f64>,
}

/// `media.understandSpeech` params for a `speech_understand` call (a long job).
fn speech_understand_params(p: SpeechUnderstandParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("items", p.items.map(|v| json!(v)));
    put("language", p.language.map(Value::String));
    put("model", p.model.map(Value::String));
    put("diarize", p.diarize.map(Value::Bool));
    put("faces", p.faces.map(Value::Bool));
    put("minPause", p.min_pause.map(|v| json!(v)));
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct RoughCutParams {
    /// Media items (raw talking clips) to cut, in order. Or `paths`.
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// Files to import and cut, in order. Or `items`.
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    /// "en", "ml" or "auto" (default: detected).
    #[serde(default)]
    pub language: Option<String>,
    /// Cut silences longer than `min_pause` (default true).
    #[serde(default)]
    pub remove_pauses: Option<bool>,
    /// Cut filler words (English um/uh…, Malayalam ആ/അതായത്/എന്ന് പറഞ്ഞാൽ…; default true).
    #[serde(default)]
    pub remove_fillers: Option<bool>,
    /// Keep only the most fluent take of a line said more than once (default true).
    #[serde(default)]
    pub remove_retakes: Option<bool>,
    /// Also cut hesitations: speech the recogniser wrote no word for (default false).
    #[serde(default)]
    pub remove_hesitations: Option<bool>,
    /// Shortest silence cut, seconds (default 0.6).
    #[serde(default)]
    pub min_pause: Option<f64>,
    /// Silence kept next to the words at each cut, seconds (default 0.12).
    #[serde(default)]
    pub keep_pause: Option<f64>,
    /// The new sequence's name.
    #[serde(default)]
    pub name: Option<String>,
    /// Export the cut to this file (MP4).
    #[serde(default)]
    pub export: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct DirectParams {
    /// What to make, in the user's words: "a 30 s reel of our weekend", "a 16:9 recap of the
    /// trip" (the layout, length, title and mood are read from it unless given).
    #[serde(default)]
    pub brief: Option<String>,
    /// Media items to direct (default: the project's videos and photos). Or `paths`, `insta360`,
    /// `album`.
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// Files or folders (their videos and photos) to import and direct.
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    /// Insta360 recordings by name ("VID_20260913_114003_00_093" or Studio's "VID_093"):
    /// imported from Insta360 Studio's download folder, else downloaded from the user's Insta360
    /// cloud by driving Studio (tell the user first: it moves the mouse).
    #[serde(default)]
    pub insta360: Option<Vec<String>>,
    /// A Google Photos album's exact title: all of it picked and imported first.
    #[serde(default)]
    pub album: Option<String>,
    /// "9:16", "16:9", "1:1" or "4:5" (default: from the brief, else 9:16).
    #[serde(default)]
    pub aspect: Option<String>,
    /// The length, seconds (6–300; default: from the brief, else 30).
    #[serde(default)]
    pub seconds: Option<f64>,
    /// Whose interface the text keeps clear of: "reels" (the default), "shorts", "tiktok", "vertical".
    #[serde(default)]
    pub platform: Option<String>,
    /// The music's mood (epic, tense, uplifting, calm, happy, dark, dreamy, energetic, sad).
    #[serde(default)]
    pub mood: Option<String>,
    /// The music's tempo, 70–150 (default 110): every cut lands on its beats.
    #[serde(default)]
    pub bpm: Option<f64>,
    /// A music file, or "generate" (the default).
    #[serde(default)]
    pub music: Option<String>,
    /// The English title over the hook (default: the model's, else from the brief).
    #[serde(default)]
    pub title: Option<String>,
    /// Captions of English speech (default true).
    #[serde(default)]
    pub captions: Option<bool>,
    /// "faces,plates", "faces", "plates" or "off" (the default: the people are the subject).
    #[serde(default)]
    pub redact: Option<String>,
    /// The new sequence's name.
    #[serde(default)]
    pub name: Option<String>,
    /// Plan only: the plan and the engine commands it compiles to, nothing built.
    #[serde(default)]
    pub dry_run: Option<bool>,
    /// Export the edit to this file (.mp4), with `<out>.sheet.jpg`, `<out>.report.json` and QA.
    #[serde(default)]
    pub export: Option<String>,
    /// Download the missing scene models first — only after the user agreed (several GB).
    #[serde(default)]
    pub download: Option<bool>,
    /// How the cuts are made: "subtle" (cuts, match cuts, a few whips, dissolves across time),
    /// "energetic" (more whips, punch-ins) or "cinematic" (long dissolves, dips to black). Left
    /// out (or "auto"), the taste profile's, else the director chooses it from the shots it
    /// planned (the report's `style` says which and why).
    #[serde(default)]
    pub style: Option<String>,
    /// What the edit is about: "places" (a travel video: the places first, the person holding
    /// the camera in at most a third of the shots) or "people". Left out (or "auto"), read from
    /// the brief and the footage.
    #[serde(default)]
    pub focus: Option<String>,
    /// Light effects (light leaks) allowed in the energetic style (default: the style's; a style
    /// the director chooses allows them when most shots are not at night).
    #[serde(default)]
    pub light_effects: Option<bool>,
    /// The sounds heard on the cuts: any of "swish", "chime", "whoosh", "impact", "riser",
    /// "shutter", "pop", "thump" (default: the style's; the subtle style's are a swish and a
    /// chime). An empty list: none.
    #[serde(default)]
    pub sounds: Option<Vec<String>>,
    /// How loud those sounds are, 0..1 (default 0.4, quiet).
    #[serde(default)]
    pub sound_level: Option<f64>,
    /// The careful colour pass: each shot's exposure and white balance measured and gently
    /// corrected, the cameras matched (default true; false leaves the footage as shot).
    #[serde(default)]
    pub color: Option<bool>,
    /// How far the colour pass corrects, 0..1 (default 0.7).
    #[serde(default)]
    pub color_strength: Option<f64>,
    /// A look of our own over the balance: "none" (the default), "punchy", "soft", "warm",
    /// "cool", "golden", "teal". Only when the user asks for one.
    #[serde(default)]
    pub look: Option<String>,
    /// The look's strength, % (default 25).
    #[serde(default)]
    pub look_strength: Option<f64>,
    /// Revise an edit already made: the viewer's notes ("1:10 is too long", "warmer at the end",
    /// "fewer barrel rolls"), turned into changes to its saved plan and rebuilt (the report lists
    /// the changes). Needs `previous`.
    #[serde(default)]
    pub revise: Option<String>,
    /// The report of the edit to revise (`<out>.report.json` of the earlier export).
    #[serde(default)]
    pub previous: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct QuickShortParams {
    /// Media items to make it of (default: the project's videos and photos). Or `paths`.
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// Files or folders (their videos and photos) to import and make it of. Or `items`.
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    /// "auto" (the default: chosen from what the clips show and say) or a template's id or name
    /// (`template_list`).
    #[serde(default)]
    pub template: Option<String>,
    /// "9:16" (the default), "1:1", "4:5" or "16:9".
    #[serde(default)]
    pub aspect: Option<String>,
    /// Whose interface the text keeps clear of: "reels" (the default), "shorts", "tiktok" or
    /// "vertical" (all three).
    #[serde(default)]
    pub platform: Option<String>,
    /// The length wanted, seconds (default 30; a template cut on the beat gets music at the tempo
    /// that brings it nearest).
    #[serde(default)]
    pub seconds: Option<f64>,
    /// What it is about, in a few words: steers the template, the shots and the text.
    #[serde(default)]
    pub topic: Option<String>,
    /// The language spoken and written: "en", "ml" or "auto".
    #[serde(default)]
    pub language: Option<String>,
    /// A creator look instead of the template's: "tech", "vhs" or "clean".
    #[serde(default)]
    pub look: Option<String>,
    /// A music file, or "generate" (the default).
    #[serde(default)]
    pub music: Option<String>,
    /// What to hide: "faces,plates" (the default), "faces", "plates" or "off".
    #[serde(default)]
    pub redact: Option<String>,
    /// The new sequence's name.
    #[serde(default)]
    pub name: Option<String>,
    /// Export the short to this file (.mp4), with `<out>.sheet.jpg`, `<out>.report.json` and QA.
    #[serde(default)]
    pub export: Option<String>,
    /// Download the missing scene, speech and privacy models first — only after the user agreed
    /// (several GB; `scenes.status`, `transcript.models` list them).
    #[serde(default)]
    pub download: Option<bool>,
}

/// `autocut.quick` params for a `quick_short` call (a long job).
fn quick_short_params(p: QuickShortParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("items", p.items.map(|v| json!(v)));
    put("paths", p.paths.map(|v| json!(v)));
    put("template", p.template.map(Value::String));
    put("aspect", p.aspect.map(Value::String));
    put("platform", p.platform.map(Value::String));
    put("seconds", p.seconds.map(|v| json!(v)));
    put("topic", p.topic.map(Value::String));
    put("language", p.language.map(Value::String));
    put("look", p.look.map(Value::String));
    put("music", p.music.map(Value::String));
    put("redact", p.redact.map(Value::String));
    put("name", p.name.map(Value::String));
    put("export", p.export.map(Value::String));
    put("download", p.download.map(Value::Bool));
    params
}

/// `autocut.direct` params for an `autocut_direct` call (a long job).
fn direct_params(p: DirectParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("brief", p.brief.map(Value::String));
    put("items", p.items.map(|v| json!(v)));
    put("paths", p.paths.map(|v| json!(v)));
    put("insta360", p.insta360.map(|v| json!(v)));
    put("album", p.album.map(Value::String));
    put("aspect", p.aspect.map(Value::String));
    put("seconds", p.seconds.map(|v| json!(v)));
    put("platform", p.platform.map(Value::String));
    put("mood", p.mood.map(Value::String));
    put("bpm", p.bpm.map(|v| json!(v)));
    put("music", p.music.map(Value::String));
    put("title", p.title.map(Value::String));
    put("captions", p.captions.map(Value::Bool));
    put("redact", p.redact.map(Value::String));
    put("name", p.name.map(Value::String));
    put("dryRun", p.dry_run.map(Value::Bool));
    put("export", p.export.map(Value::String));
    put("download", p.download.map(Value::Bool));
    put("style", p.style.map(Value::String));
    put("focus", p.focus.map(Value::String));
    put("lightEffects", p.light_effects.map(Value::Bool));
    put("sounds", p.sounds.map(|v| json!(v)));
    put("soundLevel", p.sound_level.map(|v| json!(v)));
    put("color", p.color.map(Value::Bool));
    put("colorStrength", p.color_strength.map(|v| json!(v)));
    put("look", p.look.map(Value::String));
    put("lookStrength", p.look_strength.map(|v| json!(v)));
    put("revise", p.revise.map(Value::String));
    put("previous", p.previous.map(Value::String));
    params
}

/// `autocut.roughCut` params for a `rough_cut` call (a long job).
fn rough_cut_params(p: RoughCutParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("items", p.items.map(|v| json!(v)));
    put("paths", p.paths.map(|v| json!(v)));
    put("language", p.language.map(Value::String));
    put("removePauses", p.remove_pauses.map(Value::Bool));
    put("removeFillers", p.remove_fillers.map(Value::Bool));
    put("removeRetakes", p.remove_retakes.map(Value::Bool));
    put("removeHesitations", p.remove_hesitations.map(Value::Bool));
    put("minPause", p.min_pause.map(|v| json!(v)));
    put("keepPause", p.keep_pause.map(|v| json!(v)));
    put("name", p.name.map(Value::String));
    put("export", p.export.map(Value::String));
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ScenesAnalyzeParams {
    /// Media items to index (default: the Project panel's selection, else every video and photo
    /// item of the project).
    #[serde(default)]
    pub items: Option<Vec<u64>>,
    /// "fast" (the default: Qwen3-VL 2B, one keyframe a shot) or "full" (Qwen3-VL 4B, two or
    /// three keyframes a shot, a paragraph of detail).
    #[serde(default)]
    pub detail: Option<String>,
    /// Describe the shots with the vision-language model (default true).
    #[serde(default)]
    pub describe: Option<bool>,
    /// Analyse again, ignoring the scene cache.
    #[serde(default)]
    pub fresh: Option<bool>,
}

/// `media.analyzeScenes` params for a `scenes_analyze` call (a long job).
fn scenes_analyze_params(p: ScenesAnalyzeParams) -> Value {
    let mut params = json!({"wait": true});
    if let Some(i) = p.items {
        params["items"] = json!(i);
    }
    if let Some(d) = p.detail.filter(|d| !d.trim().is_empty()) {
        params["detail"] = json!(d);
    }
    if let Some(d) = p.describe {
        params["describe"] = json!(d);
    }
    if let Some(f) = p.fresh {
        params["fresh"] = json!(f);
    }
    params
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScenesSearchParams {
    /// What to find, in words: "a robot waving at the camera", "crowd at a booth", "sunset".
    pub query: String,
    /// Most shots returned (default 20).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Only these media items (default: every indexed item).
    #[serde(default)]
    pub items: Option<Vec<u64>>,
}

/// `autocut.shorts` params for a `shorts_make` call (a long job).
fn shorts_params(p: ShortsParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("item", p.item.map(|v| json!(v)));
    put("sequence", p.sequence.map(|v| json!(v)));
    put("count", p.count.map(|v| json!(v)));
    put("minSeconds", p.min_seconds.map(|v| json!(v)));
    put("maxSeconds", p.max_seconds.map(|v| json!(v)));
    put("aspect", p.aspect.map(Value::String));
    put("motion", p.motion.map(Value::String));
    put("fallback", p.fallback.map(Value::String));
    put("platform", p.platform.map(Value::String));
    put("topic", p.topic.filter(|t| !t.trim().is_empty()).map(Value::String));
    put("captions", p.captions.map(Value::Bool));
    put("title", p.title.map(Value::Bool));
    put("export", p.export.map(Value::String));
    put("uploadYoutube", p.upload_youtube.map(Value::Bool));
    let mut y = serde_json::Map::new();
    if let Some(c) = p.youtube_channel {
        y.insert("channel".into(), json!(c));
    }
    if let Some(v) = p.youtube_privacy {
        y.insert("privacy".into(), json!(v));
    }
    if let Some(t) = p.youtube_tags {
        y.insert("tags".into(), json!(t));
    }
    put("youtube", (!y.is_empty()).then_some(Value::Object(y)));
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct AutoReframeParams {
    /// The sequence to reframe (default: the active one).
    #[serde(default)]
    pub sequence: Option<u64>,
    /// "9:16" (default), "1:1", "4:5" or "16:9".
    #[serde(default)]
    pub aspect: Option<String>,
    /// Boxes around the subject, 0..1 of the picture: {x, y, w, h} (top-left corner and size),
    /// with `clip` (one clip) and `seconds` (its media time) when known — the vision model's.
    #[serde(default)]
    pub subject: Option<Vec<Value>>,
    /// "smooth" (default) or "responsive".
    #[serde(default)]
    pub motion: Option<String>,
    /// Clips with nobody to follow: "letterbox" (default), "center", or "blur" (the whole picture
    /// over a blurred, dimmed copy of itself, also for the stretches of a followed clip where the
    /// subject is lost).
    #[serde(default)]
    pub fallback: Option<String>,
    /// Only this part of the sequence, seconds.
    #[serde(default)]
    pub start_seconds: Option<f64>,
    #[serde(default)]
    pub end_seconds: Option<f64>,
    /// The new sequence's name (default "<name> (9:16)").
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct AutoRedactParams {
    /// These clips (default: every video clip of `sequence`).
    #[serde(default)]
    pub clips: Option<Vec<u64>>,
    /// The sequence whose clips are redacted (default: the active one; ignored with `clips`).
    #[serde(default)]
    pub sequence: Option<u64>,
    /// What to hide: ["faces", "plates"] (default both).
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
    /// "blur" (default) or "mosaic".
    #[serde(default)]
    pub style: Option<String>,
    /// 1..100 (default 60).
    #[serde(default)]
    pub strength: Option<f64>,
    /// Share of the box added on every side, 0..1 (default 0.15).
    #[serde(default)]
    pub padding: Option<f64>,
    /// Smallest face hidden, share of the picture height (default 0.05).
    #[serde(default)]
    pub min_face_height: Option<f64>,
    /// Smallest plate hidden, share of the picture width (default 0.025).
    #[serde(default)]
    pub min_plate_width: Option<f64>,
    /// Areas never hidden, 0..1 of the clip's picture: {x, y, w, h, clip?, seconds?} (a robot's box).
    #[serde(default)]
    pub exclude: Option<Vec<Value>>,
    /// Ask the local vision model which faces are people and which plates are vehicles' (default true).
    #[serde(default)]
    pub verify: Option<bool>,
    /// Report the tracks without changing anything.
    #[serde(default)]
    pub dry_run: Option<bool>,
    /// Things named in words to hide too ("DP World logo", "the dpworld.com floor sign"), found by
    /// the local vision model; each a description, or {text, clips} for some clips only.
    #[serde(default)]
    pub targets: Option<Vec<Value>>,
    /// Seconds between the vision model's looks for the targets (default 1).
    #[serde(default)]
    pub target_every: Option<f64>,
    /// The profile pass: people and their poses find the faces in profile or half hidden that the
    /// face detector misses (default: on once its models are downloaded).
    #[serde(default)]
    pub profiles: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct VoiceParams {
    /// These audio clips (a video clip's linked audio counts; default: the selection).
    #[serde(default)]
    pub clips: Option<Vec<u64>>,
    /// 0..100 (default: 80 for isolate_voice, 70 for enhance_voice).
    #[serde(default)]
    pub amount: Option<f64>,
    /// enhance_voice: the loudness each clip is matched to, measured through its effects: a
    /// number (-60..0), "mix" (the Dialogue target in Preferences) or "none" (default -16).
    #[serde(default)]
    pub lufs: Option<Value>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct VoiceFileParams {
    /// The media file whose sound is processed.
    pub path: String,
    /// Where the result goes: a .wav (24-bit) or .m4a (AAC) file.
    pub out: String,
    /// "enhance" (default: Enhance Voice) or "isolate" (Voice Isolation alone).
    #[serde(default)]
    pub mode: Option<String>,
    /// 0..100 (default 70 for enhance, 80 for isolate).
    #[serde(default)]
    pub amount: Option<f64>,
    /// The loudness of the result: a number (-60..0), "mix" or "none" (default -16).
    #[serde(default)]
    pub lufs: Option<Value>,
    /// AAC bitrate for .m4a, 32..512 kbps (default 160).
    #[serde(default)]
    pub bitrate: Option<u64>,
}

/// Engine params of a voice tool call.
fn voice_params(p: VoiceParams) -> Value {
    let mut v = json!({});
    if let Some(c) = p.clips {
        v["clips"] = json!(c);
    }
    if let Some(a) = p.amount {
        v["amount"] = json!(a);
    }
    if let Some(l) = p.lufs {
        v["lufs"] = l;
    }
    v
}

/// `clip.autoRedact` (with `clips`) or `sequence.autoRedact` params for an `auto_redact` call.
fn auto_redact_params(p: AutoRedactParams) -> (&'static str, Value) {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    let id = if p.clips.is_some() { "clip.autoRedact" } else { "sequence.autoRedact" };
    put("clips", p.clips.map(|c| json!(c)));
    if id == "sequence.autoRedact" {
        put("sequence", p.sequence.map(|v| json!(v)));
    }
    put("kinds", p.kinds.map(|k| json!(k)));
    put("style", p.style.map(Value::String));
    put("strength", p.strength.map(|v| json!(v)));
    put("padding", p.padding.map(|v| json!(v)));
    put("minFaceHeight", p.min_face_height.map(|v| json!(v)));
    put("minPlateWidth", p.min_plate_width.map(|v| json!(v)));
    put("exclude", p.exclude.map(Value::Array));
    put("verify", p.verify.map(Value::Bool));
    put("dryRun", p.dry_run.map(Value::Bool));
    put("targets", p.targets.map(Value::Array));
    put("targetEvery", p.target_every.map(|v| json!(v)));
    put("profiles", p.profiles.map(Value::Bool));
    (id, params)
}

/// `sequence.autoReframe` params for a `sequence_auto_reframe` call (a long job).
fn auto_reframe_params(p: AutoReframeParams) -> Value {
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    put("sequence", p.sequence.map(|v| json!(v)));
    put("aspect", p.aspect.map(Value::String));
    put("subject", p.subject.map(Value::Array));
    put("motion", p.motion.map(Value::String));
    put("fallback", p.fallback.map(Value::String));
    put("name", p.name.map(Value::String));
    if p.start_seconds.is_some() || p.end_seconds.is_some() {
        put("range", Some(json!({"startSeconds": p.start_seconds, "endSeconds": p.end_seconds})));
    }
    params
}

/// `autocut.analyze` (or, with `plan`, `autocut.makeShowcase`) and its params, as a long job.
fn showcase_analyze_params(p: ShowcaseAnalyzeParams) -> (&'static str, Value) {
    let make = p.plan.unwrap_or(false);
    let mut params = json!({"wait": true});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params[k] = v;
        }
    };
    let text = |v: Option<String>| v.filter(|t| !t.trim().is_empty()).map(Value::String);
    put("subject", text(p.subject));
    put("prompt", text(p.prompt));
    put("album", text(p.album));
    put("photos", p.photos.map(Value::Bool));
    put("items", p.items.map(|i| json!(i)));
    put("since", p.since.map(Value::String));
    put("frames", p.frames.map(|n| json!(n)));
    put("screen", p.screen.map(|n| json!(n)));
    put("compare", p.compare.map(|n| json!(n)));
    put("fresh", p.fresh.filter(|f| *f).map(Value::Bool));
    if make {
        put("title", p.title.map(Value::String));
        put("voice", p.voice.map(Value::String));
    }
    (if make { "autocut.makeShowcase" } else { "autocut.analyze" }, params)
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcasePlanParams {
    /// The video's title (default: the plan's, else "Every <Subject> I've Filmed").
    #[serde(default)]
    pub title: Option<String>,
    /// Under the title on the opening card (default: the years the entries span).
    #[serde(default)]
    pub subtitle: Option<String>,
    /// Narration voice: a built-in id, or `clone:<name>` for a cloned voice; "" no narration.
    #[serde(default)]
    pub voice: Option<String>,
    /// The music bed's mood: epic, tense, uplifting, calm, happy, dark, dreamy, energetic or sad
    /// (synonyms such as "suspense" or "chill" work too; `music.taxonomy` lists them).
    #[serde(default)]
    pub mood: Option<String>,
    /// Clips showing the subject in less of their frames are left out (0–1, default 0.3).
    #[serde(default)]
    pub min_coverage: Option<f64>,
    /// Reviewed entries (a plan's `entries`, each with `include` and `mergePrevious`): the voice-over
    /// is written for exactly these. Default: the kept analysis, grouped afresh.
    #[serde(default)]
    pub entries: Option<Vec<Value>>,
    /// A long, narrated show: the share of entries the voice-over speaks over, 0..1 (default
    /// 1); the rest play under the music with their labels, as a storyteller leaves many shots
    /// alone.
    #[serde(default)]
    pub narrate: Option<f64>,
    /// Use the person's own voice: clips where they talk over their own footage give short
    /// bites, heard and captioned in English, telling their entries instead of the voice-over.
    #[serde(default)]
    pub bites: Option<bool>,
}

fn showcase_plan_params(p: ShowcasePlanParams) -> Value {
    let mut params = json!({"wait": true});
    for (k, v) in [("title", p.title), ("subtitle", p.subtitle), ("voice", p.voice), ("mood", p.mood)] {
        if let Some(v) = v {
            params[k] = json!(v);
        }
    }
    if let Some(c) = p.min_coverage {
        params["minCoverage"] = json!(c);
    }
    if let Some(e) = p.entries {
        params["entries"] = json!(e);
    }
    if let Some(n) = p.narrate {
        params["narrate"] = json!(n);
    }
    if let Some(b) = p.bites {
        params["bites"] = json!(b);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseBuildParams {
    /// The plan to build (as showcase_plan returns it, edited); default: the kept plan.
    #[serde(default)]
    pub plan: Option<Value>,
    /// Export the built showcase here (YouTube 1080p; chapters and a QA report beside it).
    #[serde(default)]
    pub export: Option<String>,
    /// Then upload the export to the user's Google Photos library (its own job; needs `export`
    /// and `filmcraft-cli photos auth --upload` once).
    #[serde(default)]
    pub upload: Option<bool>,
    /// With `upload`: also put the export into this album (FilmCraft's album of that title,
    /// made when missing; needs `filmcraft-cli photos auth --albums` once).
    #[serde(default)]
    pub upload_album: Option<String>,
    /// Check the built show (gaps, black frames, the mix, the subject in every shot); default:
    /// with an export.
    #[serde(default)]
    pub qa: Option<bool>,
    /// Check the exported narration for breaths in the middle of a phrase (default true: its
    /// transcript, `media.checkPauses`; the result's `pauses`).
    #[serde(default)]
    pub check_pauses: Option<bool>,
}

fn showcase_build_params(p: ShowcaseBuildParams) -> Value {
    let mut params = json!({"wait": true, "upload": p.upload.unwrap_or(false)});
    if let Some(plan) = p.plan {
        params["plan"] = plan;
    }
    if let Some(e) = p.export {
        params["export"] = json!(e);
    }
    if let Some(a) = p.upload_album.filter(|a| !a.trim().is_empty()) {
        params["uploadAlbum"] = json!(a);
    }
    if let Some(q) = p.qa {
        params["qa"] = json!(q);
    }
    if let Some(c) = p.check_pauses {
        params["checkPauses"] = json!(c);
    }
    params
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct FeedbackParams {
    /// The folder to write (absolute). Default: <data dir>/Feedback/FilmCraft Feedback <date>.
    #[serde(default)]
    pub path: Option<String>,
    /// An AutoCut QA report (`<export>.qa.json`) to include; by default the session's last one.
    #[serde(default)]
    pub qa_report: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowcaseInspectParams {
    /// The whole analysis report (every clip's facts) instead of its summary.
    #[serde(default)]
    pub report: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct TemplateListParams {
    /// Only templates whose name, description, id or tags contain this text.
    #[serde(default)]
    pub query: Option<String>,
    /// One template in full (id, name or path of a .fcvideo-template folder or .zip): its slots with
    /// their placements, variants, beat grid and manifest.
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TemplateApplyParams {
    /// Template id (`builtin:photo-beat-slideshow`, `builtin:channel-intro`, `builtin:top-3-listicle`),
    /// name, or path of a .fcvideo-template folder or .zip.
    pub template: String,
    /// Aspect variant: `16:9`, `9:16`… (default: the template's own).
    #[serde(default)]
    pub variant: Option<String>,
    /// What fills each slot, by slot id: a file path, a project item id, a text (text slots), or
    /// {"path"|"item"|"text", "inSeconds"?, "focus"?: [x, y], "focusByVariant"?: {"9:16": [x, y]}},
    /// or a list of them for slots that take several. Unfilled slots keep their placeholders.
    #[serde(default)]
    pub slots: Option<Value>,
    /// Name of the new sequence.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct RecipeListParams {
    /// One recipe in full (id or name): its parameters, its steps, and the steps as they would
    /// run with `params` (checked, nothing runs).
    #[serde(default)]
    pub recipe: Option<String>,
    /// With `recipe`: parameter values to plan with.
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecipeRunParams {
    /// Recipe id (`builtin:talking-head-clean-up`, `builtin:vertical-short`,
    /// `builtin:template-publish-youtube`, or a saved `user:…`) or name; see recipe_list.
    pub recipe: String,
    /// Parameter values by parameter id (recipe_list shows them); the rest keep their defaults.
    #[serde(default)]
    pub params: Option<Value>,
    /// Check and plan the steps only; nothing runs (default false).
    #[serde(default)]
    pub dry_run: Option<bool>,
}

/// Signed-in Google Photos, or the tool-shaped setup error.
fn gphotos() -> Result<filmcraft_gphotos::GooglePhotos, AutomationError> {
    filmcraft_gphotos::GooglePhotos::configured().map_err(|e| AutomationError::Other(e.to_string()))
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct AutocutParams {
    /// What to do, in plain words, e.g. "remove the silences and add captions" or "cut 10 to 25 seconds".
    pub prompt: String,
    /// Caption sampled frames with the local vision model and include them in the returned
    /// document when the prompt is handed to you (default false; slower).
    #[serde(default)]
    pub frames: Option<bool>,
    /// Plan only; nothing executes (autocut_edit only, default false).
    #[serde(default)]
    pub dry_run: Option<bool>,
}

/// Shared body of the AutoCut tools: run the pipeline and return the report as JSON.
async fn autocut_report(ex: &AutocutExecutor, prompt: &str, opts: filmcraft_autocut::runner::Options) -> Result<Value, AutomationError> {
    let cfg = filmcraft_autocut::config::Config::from_env();
    let runners = filmcraft_autocut::runner::Runners::from_config(&cfg);
    filmcraft_autocut::runner::run_with(&runners, prompt, ex, &cfg, opts)
        .await
        .map_err(|e| AutomationError::Other(e.to_string()))
        .and_then(|r| serde_json::to_value(&r).map_err(|e| AutomationError::Other(e.to_string())))
}

/// The most searches one `googlephotos_pick` call takes.
const MAX_PICK_QUERIES: usize = 1000;

/// The files of a pick to import, what to report of each item (no short-lived download link),
/// and how many came from the cache.
fn picked_items(picked: &filmcraft_gphotos::Picked) -> (Vec<String>, Vec<Value>, usize) {
    let paths = picked.fetched.iter().map(|f| f.path.to_string_lossy().into_owned()).collect();
    let items = picked
        .fetched
        .iter()
        .map(|f| {
            let mut v = f.item.summary();
            if let Some(o) = v.as_object_mut() {
                o.insert("path".into(), json!(f.path.to_string_lossy()));
            }
            v
        })
        .collect();
    (paths, items, picked.fetched.iter().filter(|f| f.was_cached).count())
}

fn insta360_params(p: &Insta360Params) -> Value {
    let mut m = serde_json::Map::new();
    if !p.names.is_empty() {
        m.insert("names".into(), json!(p.names));
    }
    if !p.folders.is_empty() {
        m.insert("folders".into(), json!(p.folders));
    }
    if let Some(s) = &p.since {
        m.insert("since".into(), json!(s));
    }
    if let Some(u) = &p.until {
        m.insert("until".into(), json!(u));
    }
    if p.all {
        m.insert("all".into(), json!(true));
    }
    Value::Object(m)
}

fn ok_json(v: &Value) -> CallToolResult {
    CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(v).unwrap_or_default())])
}
fn fail(e: impl std::fmt::Display) -> CallToolResult {
    CallToolResult::error(vec![Content::text(e.to_string())])
}
fn png_result(png: &[u8], note: &str) -> CallToolResult {
    CallToolResult::success(vec![Content::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"), Content::text(note.to_string())])
}
fn wrap(r: Result<Value, AutomationError>) -> Result<CallToolResult, McpError> {
    Ok(match r {
        Ok(v) => ok_json(&v),
        Err(e) => fail(e),
    })
}

/// `Executor` over either MCP backend: the same dispatch `FilmcraftMcp::run` uses, as a
/// standalone handle AutoCut can plan against.
#[derive(Clone)]
pub struct AutocutExecutor(Arc<Backend>);

impl AutocutExecutor {
    /// A bridge-mode executor over a (cloned) [`BridgeClient`], for callers that already have
    /// one (the CLI's `--bridge` backend).
    pub fn bridge(client: BridgeClient) -> Self {
        Self(Arc::new(Backend::Bridge(Arc::new(client))))
    }
}

impl filmcraft_autocut::Executor for AutocutExecutor {
    fn execute(&self, id: &str, params: Value) -> filmcraft_autocut::ExecFuture {
        match &*self.0 {
            Backend::Headless(s) if id.starts_with("autocut.") && crate::long_job::is_long(id, &params) => {
                Box::pin(long_headless(s.clone(), id.to_string(), params))
            }
            Backend::Headless(s) => {
                let s = s.clone();
                let id = id.to_string();
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        // A command that panicked earlier poisoned the lock; the session is still usable.
                        let mut g = s.lock().unwrap_or_else(PoisonError::into_inner);
                        g.execute(&id, params).map_err(|e| e.to_string())
                    })
                    .await
                    .map_err(|e| format!("internal error: {e}"))?
                })
            }
            Backend::Bridge(b) => {
                let b = b.clone();
                let id = id.to_string();
                Box::pin(async move { b.execute(&id, params).await.map_err(|e| e.to_string()) })
            }
        }
    }

    fn render(&self, seconds: f64, max_side: u32) -> filmcraft_autocut::RenderFuture {
        match &*self.0 {
            Backend::Headless(s) => {
                let s = s.clone();
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        let g = s.lock().unwrap_or_else(PoisonError::into_inner);
                        let (w, h) = g.active_sequence().map(|q| (q.settings.width, q.settings.height))?;
                        let scale = (max_side as f32 / w.max(h).max(1) as f32).min(1.0);
                        let img = g.try_render_program_at(scale, filmcraft_time::Tick::from_seconds_f64(seconds)).ok()?;
                        render_png(img.w as u32, img.h as u32, img.over_black_rgba8(), max_side)
                    })
                    .await
                    .ok()
                    .flatten()
                })
            }
            // the live app renders through its UI; engine.execute can't reach the compositor,
            // so bridge mode plans without eyes for now
            Backend::Bridge(_) => Box::pin(async move { None }),
        }
    }
}

/// Same encoding as `filmcraft_autocut`'s `SessionExecutor::render`: RGBA8 → downscaled PNG.
fn render_png(w: u32, h: u32, rgba: Vec<u8>, max_side: u32) -> Option<Vec<u8>> {
    let mut img = image::RgbaImage::from_raw(w, h, rgba)?;
    if max_side > 0 && w.max(h) > max_side {
        let s = max_side as f32 / w.max(h) as f32;
        img = image::imageops::resize(&img, ((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1), image::imageops::FilterType::Triangle);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

/// Run what background jobs (AutoCut showcases) send a headless session over its command port,
/// as the app's UI does once per frame: a thread woken by the port that holds the session lock
/// only for a burst of requests, so the server answers other calls while a job runs. It ends
/// with the session (whose waker holds `woken`'s sender).
fn pump_port(session: std::sync::Weak<Mutex<Session>>, woken: std::sync::mpsc::Receiver<()>) {
    use std::sync::mpsc::RecvTimeoutError;
    let pump = move || loop {
        match woken.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(()) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        let Some(s) = session.upgrade() else { return };
        s.lock().unwrap_or_else(PoisonError::into_inner).pump_port(std::time::Duration::from_millis(50));
    };
    // a failed spawn leaves `wait: true` calls working (they run the port themselves)
    let _ = std::thread::Builder::new().name("filmcraft-mcp-port".into()).spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(pump));
    });
}

/// A long showcase command (`wait: true`) on the headless session, started without waiting and
/// its job polled, as the bridge does: the session lock is free meanwhile, so the port's pump
/// thread can run the job's commands. The answer is the engine's `wait: true` answer.
async fn long_headless(s: Arc<Mutex<Session>>, id: String, mut params: Value) -> Result<Value, String> {
    let exec = |id: String, params: Value| {
        let s = s.clone();
        async move {
            tokio::task::spawn_blocking(move || s.lock().unwrap_or_else(PoisonError::into_inner).execute(&id, params).map_err(|e| e.to_string()))
                .await
                .map_err(|e| format!("internal error: {e}"))?
        }
    };
    if let Some(p) = params.as_object_mut() {
        p.insert("wait".into(), json!(false));
    }
    let mut start = exec(id.clone(), params).await?;
    let Some(job) = start.get("job").and_then(Value::as_u64) else { return Ok(start) };
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let jobs = exec("jobs.list".into(), json!({})).await?;
        let Some(j) = jobs.as_array().and_then(|a| a.iter().find(|j| j["id"].as_u64() == Some(job))).cloned() else {
            return Err(format!("`{id}`: job {job} is gone"));
        };
        if j["finished"].as_bool() == Some(true) {
            if let Some(e) = j["result"]["error"].as_str() {
                return Err(format!("`{id}` failed: {e}"));
            }
            if let Some(o) = start.as_object_mut() {
                o.insert("result".into(), j["result"].clone());
                for k in crate::long_job::JOB_EXTRAS {
                    if let Some(v) = j.get(*k) {
                        o.insert((*k).into(), v.clone());
                    }
                }
            }
            return Ok(start);
        }
    }
}

impl FilmcraftMcp {
    pub fn headless(mut session: Session) -> Self {
        let (wake, woken) = std::sync::mpsc::channel::<()>();
        session.set_port_waker(Arc::new(move || {
            let _ = wake.send(());
        }));
        let shared = Arc::new(Mutex::new(session));
        pump_port(Arc::downgrade(&shared), woken);
        Self { backend: Arc::new(Backend::Headless(shared)), tool_router: Self::tool_router() }
    }
    pub fn bridge(addr: &str) -> Result<Self, AutomationError> {
        Ok(Self { backend: Arc::new(Backend::Bridge(Arc::new(BridgeClient::new(addr)?))), tool_router: Self::tool_router() })
    }

    /// An [`AutocutExecutor`] handle onto this server's backend.
    pub fn executor(&self) -> AutocutExecutor {
        AutocutExecutor(self.backend.clone())
    }

    pub async fn serve_stdio(self) -> Result<(), AutomationError> {
        self.serve_io(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serve MCP over any line stream. A line that isn't JSON gets a JSON-RPC parse error
    /// (`-32700`, id null) instead of being dropped silently; the rest go to the rmcp service. One
    /// writer task owns `output`, so replies never interleave.
    pub async fn serve_io(
        self,
        input: impl tokio::io::AsyncRead + Unpin + Send + 'static,
        output: impl tokio::io::AsyncWrite + Unpin + Send + 'static,
    ) -> Result<(), AutomationError> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let (mut to_service, service_in) = tokio::io::duplex(1 << 20);
        let (service_out, from_service) = tokio::io::duplex(1 << 20);
        let (err_tx, mut err_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let reader = tokio::spawn(async move {
            let mut lines = BufReader::new(input).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(_) => {
                        if to_service.write_all(format!("{line}\n").as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let reply = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                        let _ = err_tx.send(reply.to_string());
                    }
                }
            }
            // EOF: closing the pipe ends the service
            let _ = to_service.shutdown().await;
        });
        let writer = tokio::spawn(async move {
            let mut output = output;
            let mut replies = BufReader::new(from_service).lines();
            loop {
                let line = tokio::select! {
                    l = replies.next_line() => match l { Ok(Some(l)) => l, _ => break },
                    Some(e) = err_rx.recv() => e,
                };
                if output.write_all(format!("{line}\n").as_bytes()).await.is_err() || output.flush().await.is_err() {
                    break;
                }
            }
        });
        let running = self.serve((service_in, service_out)).await.map_err(|e| AutomationError::Other(format!("MCP init: {e}")))?;
        let r = running.waiting().await.map_err(|e| AutomationError::Other(e.to_string()));
        reader.abort();
        let _ = writer.await;
        r.map(|_| ())
    }

    /// Run an engine command on whichever backend.
    pub async fn run(&self, id: &str, params: Value) -> Result<Value, AutomationError> {
        self.executor().execute(id, params).await.map_err(AutomationError::Other)
    }

    /// The project tree plus the active sequence (null when there is none).
    ///
    /// `project.inspect` and `sequence.inspect` are two separate calls (two lock scopes in
    /// headless mode, two round trips to the app in bridge mode), so the active sequence can
    /// change between them if another client, or the app's own UI, switches sequence tabs in
    /// that window (#1477). Pinning `sequence.inspect` to the `activeSequence` id this
    /// `project.inspect` just reported keeps the pair consistent: the returned `sequence`, when
    /// not null, always describes the sequence the returned `project.activeSequence` names.
    async fn document(&self) -> Result<Value, AutomationError> {
        let project = self.run("project.inspect", json!({})).await?;
        // `ItemId` is a u64 (`item_p` parses the `item` param with `u64_p`): read it as one too,
        // not `as_i64`, so an id above `i64::MAX` is still looked up instead of silently going null.
        let active = project.get("activeSequence").and_then(Value::as_u64);
        let sequence = match active {
            Some(id) => self.run("sequence.inspect", json!({"item": id})).await.ok(),
            None => None,
        };
        Ok(json!({"project": project, "sequence": sequence}))
    }

    fn bridge_client(&self) -> Option<Arc<BridgeClient>> {
        match &*self.backend {
            Backend::Bridge(b) => Some(b.clone()),
            Backend::Headless(_) => None,
        }
    }

    async fn ui(&self, method: &str, params: Value) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => wrap(b.call(method, params).await),
            None => Ok(fail(
                "this tool drives the live app: run `filmcraft-cli mcp --bridge 127.0.0.1:<port>` with the app started as `filmcraft --control <port>`",
            )),
        }
    }
}

#[tool_router]
impl FilmcraftMcp {
    #[tool(
        title = "List commands",
        description = "List every command (id, label, menu path, shortcut, parameter doc, enabled now). Menus, shortcuts, panels and timeline gestures all map to these ids.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn command_list(&self, Parameters(p): Parameters<ListParams>) -> Result<CallToolResult, McpError> {
        let r = self.run("command.list", json!({})).await.map(|v| {
            let f = p.filter.unwrap_or_default().to_ascii_lowercase();
            let only = p.enabled_only.unwrap_or(false);
            Value::Array(
                v.as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|c| {
                        let id = c["id"].as_str().unwrap_or("").to_ascii_lowercase();
                        let label = c["label"].as_str().unwrap_or("").to_ascii_lowercase();
                        (f.is_empty() || id.contains(&f) || label.contains(&f)) && (!only || c["enabled"].as_bool() == Some(true))
                    })
                    .collect(),
            )
        });
        wrap(r)
    }

    #[tool(
        title = "Run a command",
        description = "Run a command by id with JSON params, e.g. {\"id\":\"timeline.razor\",\"params\":{\"seconds\":3.5}} or {\"id\":\"effects.apply\",\"params\":{\"effect\":\"Gaussian Blur\"}}. Undoable edits land in History.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn command_run(&self, Parameters(p): Parameters<RunParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run(&p.id, p.params.unwrap_or(json!({}))).await)
    }

    #[tool(
        title = "Run several commands",
        description = "Run several commands in order: {steps: [{id, params?}], stop_on_error?: true}. Returns {completed, failed, results: [{ok, result | error}]}; each edit is its own undo step.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn command_batch(&self, Parameters(p): Parameters<BatchParams>) -> Result<CallToolResult, McpError> {
        let stop = p.stop_on_error.unwrap_or(true);
        let (mut completed, mut failed, mut results) = (0u32, 0u32, Vec::new());
        for st in p.steps {
            match self.run(&st.id, st.params.unwrap_or(json!({}))).await {
                Ok(v) => {
                    completed += 1;
                    results.push(json!({"ok": true, "result": v}));
                }
                Err(e) => {
                    failed += 1;
                    results.push(json!({"ok": false, "error": e.to_string()}));
                    if stop {
                        break;
                    }
                }
            }
        }
        let v = json!({"completed": completed, "failed": failed, "results": results});
        Ok(if failed > 0 { CallToolResult::error(vec![Content::text(v.to_string())]) } else { ok_json(&v) })
    }

    #[tool(
        title = "Inspect the document",
        description = "The project as JSON: the tree (as project_inspect) plus the active sequence (as sequence_inspect; null when there is none).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn doc_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.document().await)
    }

    #[tool(
        title = "Render a preview",
        description = "The active sequence's frame at `seconds` (default: the playhead) as PNG, same as render_frame (bridge mode: Program monitor screenshot).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn render_preview(&self, p: Parameters<RenderParams>) -> Result<CallToolResult, McpError> {
        self.render_frame(p).await
    }

    #[tool(
        title = "Inspect the project tree",
        description = "Project tree (bins and items with ids, types, durations), active sequence, source clip.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn project_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.run("project.inspect", json!({})).await)
    }

    #[tool(
        title = "Inspect the active sequence",
        description = "The active sequence as JSON: settings, tracks, clips (ids, start/duration in ticks and frames, effects), transitions, markers, in/out, playhead, selection. 254016000000 ticks = 1 second.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn sequence_inspect(&self) -> Result<CallToolResult, McpError> {
        wrap(self.run("sequence.inspect", json!({})).await)
    }

    #[tool(
        title = "Import media",
        description = "Import media files by absolute path (MP4/MOV, WAV/MP3/FLAC/AIFF/Ogg, PNG/JPEG/…).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn media_import(&self, Parameters(p): Parameters<MediaImportParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("file.import", media_import_params(p)).await)
    }

    #[tool(
        title = "AutoCut: plan an edit",
        description = "Turn a plain-language prompt (\"remove the silences\", \"cut 0:10 to 0:25\") into a validated batch of engine commands (dry run: nothing executes). Local models classify and plan routine edits. When the report has needsSupervisor, the prompt is beyond the routine catalog and YOU are the supervisor: plan the edit yourself with command_list / command_run / command_batch (render_preview to check) using the returned document; pass frames: true to also get vision captions of sampled frames.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = true)
    )]
    async fn autocut_plan(&self, Parameters(p): Parameters<AutocutParams>) -> Result<CallToolResult, McpError> {
        wrap(
            autocut_report(
                &self.executor(),
                &p.prompt,
                filmcraft_autocut::runner::Options { mode: filmcraft_autocut::runner::Mode::PlanOnly, frames: p.frames.unwrap_or(false) },
            )
            .await,
        )
    }

    #[tool(
        title = "AutoCut: edit",
        description = "Turn a plain-language prompt into engine commands and run them (each an ordinary undoable edit; `edit.undo` reverts). For prompts beyond the routine catalog the report comes back with needsSupervisor and nothing runs: plan it yourself with command_list / command_run / command_batch as in autocut_plan. Configure models with AUTOCUT_* environment variables.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = true)
    )]
    async fn autocut_edit(&self, Parameters(p): Parameters<AutocutParams>) -> Result<CallToolResult, McpError> {
        let mode = if p.dry_run.unwrap_or(false) { filmcraft_autocut::runner::Mode::PlanOnly } else { filmcraft_autocut::runner::Mode::Execute };
        wrap(autocut_report(&self.executor(), &p.prompt, filmcraft_autocut::runner::Options { mode, frames: p.frames.unwrap_or(false) }).await)
    }

    #[tool(
        title = "Google Photos: pick and import",
        description = "Open the Google Photos picker in the user's browser, wait for a selection and press Done, download the picks (cached under the app's data dir) and import them into the open project with file.import (a file the project already holds is reused, not imported twice). Google offers apps no server-side search of the library (retired 2025-03-31): the choice happens in the picker. Either the user chooses (tell them what to pick), or — pass `query` — a local vision agent (Qwen2.5-VL via Ollama) drives the picker itself: it types the query into the picker's own search, selects matching video tiles and presses Done; needs `ollama pull qwen2.5vl:7b` plus Screen Recording and Accessibility for this process, and its report comes back as `autopick`. Pass `queries` (and `all`) to run many searches — a list, or a sweep of a whole history such as one \"videos <Month> <Year>\" per month — in ONE picking session and ONE browser tab (the selection accumulates across searches; date searches are selected by the picker's date groups; a search with no results is skipped; Done once), in as few sessions as the item cap allows; searches that did not fit come back as `left`. The agent closes the picker tab when it finishes, and a session it gave up on is deleted. Pass `album` instead to use a whole Google Photos album as the seed (the user curates it in Google Photos): the agent searches the picker for that exact title (Google's picker answers an album's exact title with the album's items) and selects that whole result (`videos_only` downloads only its videos); a search that shows nothing new, or a result too big for an album, selects nothing. Items a user adds to an album are reachable only this way (the Library API shows apps only what they uploaded). If the wait runs out the result is `status: pending` with a `session` and `pickerUri`; call again with that `session` to keep waiting (with short client tool timeouts, start with wait_seconds 0, then resume). Videos download whole, so they can take a while. One-time setup: `filmcraft-cli photos auth` (see crates/gphotos/README.md).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn googlephotos_pick(&self, Parameters(p): Parameters<PhotosPickParams>) -> Result<CallToolResult, McpError> {
        let searching = p.query.as_deref().is_some_and(|q| !q.trim().is_empty()) || p.queries.as_ref().is_some_and(|q| q.iter().any(|q| !q.trim().is_empty()));
        if p.album.as_deref().is_some_and(|a| !a.trim().is_empty()) && searching {
            return Ok(fail("pass `query`/`queries` (searches) or `album` (a whole album), not both"));
        }
        let asked = p.query.iter().count().saturating_add(p.queries.as_ref().map_or(0, Vec::len));
        if asked > MAX_PICK_QUERIES {
            return Ok(fail(format!("at most {MAX_PICK_QUERIES} searches per call (got {asked}); the rest can go in the next call")));
        }
        let gp = match gphotos() {
            Ok(gp) => gp,
            Err(e) => return Ok(fail(e)),
        };
        let since = p.since.filter(|d| !d.trim().is_empty());
        if let Some(d) = since.as_deref().filter(|d| !filmcraft_gphotos::is_day(d)) {
            return Ok(fail(format!("`since` wants a date like 2020-01-01, not `{d}`")));
        }
        let queries = filmcraft_screenagent::agent::clean_queries(p.query.into_iter().chain(p.queries.unwrap_or_default()).collect());
        let album = p.album.map(|a| a.trim().to_string()).filter(|a| !a.is_empty());
        if let Some(a) = album.as_deref()
            && let Err(e) = filmcraft_gphotos::albums::clean_title(a)
        {
            return Ok(fail(format!("`album`: {e}")));
        }
        let videos_only = p.videos_only.unwrap_or(false);
        if videos_only && album.is_none() {
            return Ok(fail("`videos_only` goes with `album` (`query` always picks videos)"));
        }
        let select_all = p.all.unwrap_or(false);
        let max_items = |default_tiles: u32| p.max_items.unwrap_or(if select_all { filmcraft_gphotos::api::MAX_PICK } else { default_tiles });
        if queries.len() > 1 {
            if p.session.as_deref().is_some_and(|s| !s.trim().is_empty()) {
                return Ok(fail("`session` resumes one pick; it does not go with many `queries`"));
            }
            let req = crate::photos::Searches { queries, select_all, max_items: max_items(6), per_session: crate::photos::PER_SESSION };
            let opts = filmcraft_gphotos::PickOptions {
                max_items: None,
                wait: std::time::Duration::from_secs(p.wait_seconds.unwrap_or(900)),
                resume: None,
                open_browser: true,
                since,
                videos_only: false,
            };
            let swept = match tokio::task::spawn_blocking(move || crate::photos::pick_searches(&gp, &opts, &req, &mut |_| {}, &|| false)).await {
                Ok(s) => s,
                Err(e) => return Ok(fail(format!("internal error: {e}"))),
            };
            let (paths, items, cached) = picked_items(&swept.picked);
            let imported = if paths.is_empty() {
                Err(crate::AutomationError::Other("nothing was picked".into()))
            } else {
                self.run("file.import", json!({"paths": paths, "reuse": true})).await
            };
            let v = json!({
                "status": "done", "items": items, "downloaded": swept.picked.fetched.len().saturating_sub(cached), "cached": cached,
                "failed": swept.picked.failed, "skipped": swept.picked.skipped, "imported": imported.as_ref().ok(),
                "importError": imported.as_ref().err().map(|e| e.to_string()),
                "sessions": swept.sessions, "left": swept.left, "tooBig": swept.too_big, "stopped": swept.stopped,
                "next": if swept.left.is_empty() { Value::Null } else if swept.too_big.is_empty() {
                    json!("call googlephotos_pick again with `queries` = `left` (and the same `all`) for the searches no session got to")
                } else {
                    json!("call googlephotos_pick again with `queries` = `left` (and the same `all`); split each of `tooBig` first (by day, or narrower searches): its results alone are more than a session holds")
                },
            });
            let ok = imported.is_ok() && swept.picked.failed.is_empty() && swept.stopped.is_none();
            return Ok(if ok { ok_json(&v) } else { CallToolResult::error(vec![Content::text(v.to_string())]) });
        }
        let opts = filmcraft_gphotos::PickOptions {
            max_items: p.max_items,
            wait: std::time::Duration::from_secs(p.wait_seconds.unwrap_or(if !queries.is_empty() || album.is_some() { 900 } else { 300 })),
            resume: p.session.filter(|s| !s.trim().is_empty()),
            open_browser: true,
            since,
            videos_only,
        };
        let request = match (queries.first(), album) {
            (Some(_), _) => Some(filmcraft_screenagent::agent::Request { queries, max_items: max_items(6), select_all, album: None, videos_only: true }),
            (None, Some(title)) => Some(filmcraft_screenagent::agent::Request {
                max_items: p.max_items.unwrap_or(filmcraft_gphotos::api::MAX_PICK),
                ..crate::photos::album_request(&title, videos_only)
            }),
            (None, None) => None,
        };
        // blocking HTTP, sleeps and the agent: keep them off the async threads. The wait ends
        // early when the agent finishes without pressing Done
        let driven = match tokio::task::spawn_blocking(move || crate::photos::pick_driven(&gp, &opts, request, &mut |_| {}, &|| false)).await {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => return Ok(fail(e)),
            Err(e) => return Ok(fail(format!("internal error: {e}"))),
        };
        let autopick: Value = match &driven.agent {
            Some(Ok(s)) => serde_json::to_value(s).unwrap_or(Value::Null),
            Some(Err(e)) => json!({"error": e}),
            None => Value::Null,
        };
        if driven.nothing_to_pick() {
            return Ok(fail(json!({
                "status": "done", "error": "the agent found nothing to pick (no matching album or search result), so nothing was selected",
                "autopick": autopick,
            })));
        }
        let picked = match driven.outcome {
            filmcraft_gphotos::PickOutcome::Done(picked) => picked,
            filmcraft_gphotos::PickOutcome::Pending(_) if driven.abandoned => {
                return Ok(fail(json!({
                    "status": "done", "error": "the agent did not finish the pick in time; its picking session was deleted and its tab closed",
                    "autopick": autopick,
                })));
            }
            filmcraft_gphotos::PickOutcome::Pending(s) => {
                return Ok(ok_json(&json!({
                    "status": "pending", "session": s.id, "pickerUri": s.autoclose_uri(), "expires": s.expire_time,
                    "next": "no selection is confirmed yet; give the user pickerUri if the browser did not open, then call googlephotos_pick again with this session",
                })));
            }
        };
        if picked.fetched.is_empty() {
            return Ok(fail(json!({"status": "done", "error": "nothing was picked or nothing could be downloaded", "failed": picked.failed})));
        }
        let (paths, items, cached) = picked_items(&picked);
        let imported = self.run("file.import", json!({"paths": paths, "reuse": true})).await;
        let ok = imported.is_ok() && picked.failed.is_empty();
        let v = json!({
            "status": "done", "items": items, "downloaded": picked.fetched.len().saturating_sub(cached), "cached": cached,
            "failed": picked.failed, "skipped": picked.skipped, "imported": imported.as_ref().ok(), "importError": imported.as_ref().err().map(|e| e.to_string()),
            "autopick": autopick,
        });
        Ok(if ok { ok_json(&v) } else { CallToolResult::error(vec![Content::text(v.to_string())]) })
    }

    #[tool(
        title = "Insta360: list recordings",
        description = "The user's Insta360 360° recordings on disk — where Insta360 Studio downloads its cloud files (~/Downloads/Insta360 Download) and any mounted camera card — newest first: stem (VID_20260913_114003_00_093), when it was shot (camera time), path, size, the camera's .lrv proxy, whether it has finished downloading, the camera model. Filter with names, since/until.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn insta360_list(&self, Parameters(p): Parameters<Insta360Params>) -> Result<CallToolResult, McpError> {
        wrap(self.run("insta360.list", insta360_params(&p)).await)
    }

    #[tool(
        title = "Insta360: import recordings",
        description = "Import Insta360 recordings (names, or since/until, or all) as 360° media: VR Properties from the file (dual fisheye: both lenses decoded side by side), the camera's .lrv attached as the proxy, the two lenses aligned along their seam. Files the project already holds are reused. Then `clip.reframe360` (command_run) looks into them with a virtual camera: yaw/pitch/roll/fov/distance, keyframes, presets natural|wide|tinyPlanet|rabbitHole|sphere; `clip.stabilize360` steadies the view; `clip.follow360` follows a subject.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn insta360_import(&self, Parameters(p): Parameters<Insta360Params>) -> Result<CallToolResult, McpError> {
        wrap(self.run("insta360.import", insta360_params(&p)).await)
    }

    #[tool(
        title = "Insta360: download from the cloud",
        description = "Download recordings (names: full stems like VID_20260913_114003_00_093, or Studio's short VID_093) from the user's Insta360 cloud through Insta360 Studio, which must be open and signed in: the local screen agent (Qwen2.5-VL via Ollama, Screen Recording and Accessibility granted) opens Studio's Cloud Media in List View, right-clicks each recording, chooses Download, checks the dialog's file name and presses OK (Studio's own download folder); then it waits for each file to finish and imports it as insta360_import does. It moves the mouse in Studio's window (the `chrome` lock: one agent at a time) — tell the user before running it. Long: progress comes as MCP progress notifications.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = true)
    )]
    async fn insta360_download(&self, Parameters(p): Parameters<Insta360Params>) -> Result<CallToolResult, McpError> {
        if p.names.is_empty() {
            return Ok(fail("no names given: the recordings to download (insta360_list shows what is already here)"));
        }
        let mut v = insta360_params(&p);
        v["wait"] = json!(true);
        wrap(self.run("insta360.download", v).await)
    }

    #[tool(
        title = "Google Photos: list downloaded items",
        description = "The Google Photos originals already downloaded by earlier picks: {id, filename, bytes, path}. Pass ids to googlephotos_add to import them again without the picker.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn googlephotos_cached(&self) -> Result<CallToolResult, McpError> {
        let list = tokio::task::spawn_blocking(filmcraft_gphotos::cache::list).await.unwrap_or_default();
        Ok(ok_json(&Value::Array(list)))
    }

    #[tool(
        title = "Google Photos: import downloaded items",
        description = "Import already-downloaded Google Photos items (ids from googlephotos_pick or googlephotos_cached) into the open project with file.import (files the project already holds are reused, not imported twice). Items never picked are not reachable: Google shares only what the user picks, so use googlephotos_pick for new ones.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn googlephotos_add(&self, Parameters(p): Parameters<PhotosAddParams>) -> Result<CallToolResult, McpError> {
        if p.ids.is_empty() {
            return Ok(fail("no media item ids given"));
        }
        if p.ids.len() > MAX_ADD_IDS {
            return Ok(fail(format!("at most {MAX_ADD_IDS} ids per call")));
        }
        let (paths, missing) = match tokio::task::spawn_blocking(move || {
            let mut paths = Vec::new();
            let mut missing = Vec::new();
            for id in p.ids {
                match filmcraft_gphotos::cache::find(&id) {
                    Some(path) => paths.push(path.to_string_lossy().into_owned()),
                    None => missing.push(id),
                }
            }
            (paths, missing)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(fail(format!("internal error: {e}"))),
        };
        if !missing.is_empty() {
            return Ok(fail(json!({"error": "not downloaded yet; pick these with googlephotos_pick", "missing": missing})));
        }
        wrap(self.run("file.import", json!({"paths": paths, "reuse": true})).await.map(|imported| json!({"imported": imported, "paths": paths})))
    }

    #[tool(
        title = "Google Photos: upload a file",
        description = "Upload a local file — typically a finished export (`file.exportMedia`) — to the user's Google Photos library, resumably in 8 MiB chunks, and return the new media item (mediaItemId, productUrl). With `album` (a title or id) the item also goes into that album: the one FilmCraft made with exactly that title, created when missing, so the user can add more to it by hand for the next iteration (read those back with googlephotos_pick `album`). Runs as the `photos.upload` engine job: progress is reported as MCP progress notifications (bytes) and cancelling the request stops the upload (the local file is never touched). Videos up to 10 GB, photos up to 200 MB. Needs the user's one-time consent to add to their library (`filmcraft-cli photos auth --upload`; `--albums` for albums) and the Photos Library API enabled in their Google Cloud project; without them the error says what to run. Albums are also engine commands: `photos.createAlbum`, `photos.albums`, `photos.addToAlbum` (command_run).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn googlephotos_upload(
        &self,
        Parameters(p): Parameters<PhotosUploadParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if p.path.trim().is_empty() {
            return Ok(fail("`path` is empty: pass the file to upload"));
        }
        Ok(self.run_long("photos.upload", photos_upload_params(p), &context).await)
    }

    #[tool(
        title = "YouTube: upload a video",
        description = "Upload a video — typically a finished export (`file.exportMedia`, or a showcase's) — to one of the user's connected YouTube channels (`channel`: id, @handle or title; default: the default channel), resumably in 8 MiB chunks, and return the video (videoId, url, studioUrl, privacyStatus, the Shorts hint, the thumbnail and playlist outcome, today's quota). Private by default; YouTube may keep uploads from an unaudited Google Cloud project private whatever is asked. A showcase export's `<file>.chapters.txt` is appended to the description and `<file>.thumbnail.jpg` becomes the thumbnail. Runs as the `youtube.upload` engine job: progress notifications in bytes, cancelling stops it (running the same upload again resumes it). Channels are connected by the user once per channel with `filmcraft-cli youtube auth` (or the `youtube.connect` command: it opens the browser, where Google's account chooser picks the channel); `youtube.channels`, `youtube.setDefault`, `youtube.disconnect` (command_run) manage them. The quota: 100 uploads and 10,000 units a day per Google Cloud project, reset at midnight Pacific.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn youtube_upload(
        &self,
        Parameters(p): Parameters<YouTubeUploadParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if p.path.trim().is_empty() {
            return Ok(fail("`path` is empty: pass the video to upload"));
        }
        Ok(self.run_long("youtube.upload", youtube_upload_params(p), &context).await)
    }

    #[tool(
        title = "Publish to a destination (checked first)",
        description = "Publish a video — typically a finished export — to a destination: `youtube`, `youtubeShort` or `instagramReel`. The shared pre-flight runs first, before any byte leaves the machine (container, codecs, 4:2:0, frame rate, width, aspect — 9:16 for Reels, vertical or square for a Short — bitrate, duration, size, audio rate and channels, moov placement): a failure stops it with the fix; `publish.preflight` (command_run) shows the whole checklist with loudness (−14 LUFS), and `publish.fixCopy` writes a copy that passes where an export preset can (re-encoded H.264/AAC, or only the moov moved). Then the destination's job runs (`youtube.upload` or `instagram.publish`): progress notifications in bytes, cancellable; the answer carries `youtube` (the video) or `instagram` (the Reel). `publish.destinations` lists the destinations and whether each is connected.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn publish_run(
        &self,
        Parameters(p): Parameters<PublishRunParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if p.destination.trim().is_empty() {
            return Ok(fail("`destination` is empty: youtube, youtubeShort or instagramReel"));
        }
        Ok(self.run_long("publish.run", publish_run_params(p), &context).await)
    }

    #[tool(
        title = "Instagram: publish a Reel",
        description = "Publish a video — typically a finished export — as a Reel on the user's connected Instagram professional account: a container, the file sent from this machine with Meta's resumable upload (or `videoUrl`, a public https:// address Instagram fetches it from), Instagram's processing awaited (at most 5 minutes), then published; returns the Reel (mediaId, permalink) and the 24-hour limits (100 posts, 400 containers). Caption: at most 2,200 characters, 30 hashtags, 20 @mentions; `coverSeconds` picks the cover frame. Runs as the `instagram.publish` engine job: progress notifications in bytes, cancelling stops it (running the same publish again carries on with the same container). The account is connected by the user once with `filmcraft-cli instagram auth` (or `instagram token`); `instagram.account` / `instagram.limits` (command_run) show it.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn instagram_publish(
        &self,
        Parameters(p): Parameters<InstagramPublishParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if p.path.as_deref().is_none_or(|x| x.trim().is_empty()) && p.video_url.as_deref().is_none_or(|x| x.trim().is_empty()) {
            return Ok(fail("pass `path` (the video to send) or `videoUrl` (a public https:// address)"));
        }
        Ok(self.run_long("instagram.publish", instagram_publish_params(p), &context).await)
    }

    #[tool(
        title = "Showcase: find every clip of a subject",
        description = "Step 1 of a showcase (\"every robot I've filmed, in order\": a counted, narrated video). A local vision model (Ollama, AUTOCUT_ANALYSIS_MODEL) looks at every video clip (or `items`, filmed `since` a date) for `subject` (or what a `prompt` asks for), names and describes what it finds, reads capture dates and tells which clips show the same one. With `album`, the clips are a Google Photos album the user filled: the whole album is picked through Google's picker by the local screen agent, imported (no duplicates) and only its clips are analysed — the one-call way to turn \"the clips in my album X\" into a video (then plan: true, review, showcase_build). Runs as the `autocut.analyze` engine job (progress notifications, cancellable); the report is kept in the session (showcase_inspect). With plan: true it goes on to step 2 (`autocut.makeShowcase`) and returns the plan. In the app the Showcase panel shows the job and the plan.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_analyze(
        &self,
        Parameters(p): Parameters<ShowcaseAnalyzeParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let given = |v: &Option<String>| v.as_deref().is_some_and(|t| !t.trim().is_empty());
        if !given(&p.subject) && !given(&p.prompt) {
            return Ok(fail("say what to look for: a `subject` (e.g. \"robot\") or a `prompt` (\"a video of every robot since 2020\")"));
        }
        let (id, params) = showcase_analyze_params(p);
        Ok(self.run_long(id, params, &context).await)
    }

    #[tool(
        title = "Showcase: plan the entries and the voice-over",
        description = "Step 2 of a showcase: group the analysed clips into entries (one per robot, oldest first, up to three shots each) and have the local script model write the voice-over — or, given reviewed `entries` (drop one with include: false, fold one into the one before with mergePrevious: true, fix a label or description), write it for exactly those. Runs as the `autocut.showcasePlan` engine job; returns the plan ({title, voice, mood, entries, script: {intro, lines, outro}}, lines one per entry). Review it with the user before building: wrong subjects, labels, pronunciations.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_plan(
        &self,
        Parameters(p): Parameters<ShowcasePlanParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.showcasePlan", showcase_plan_params(p), &context).await)
    }

    #[tool(
        title = "Showcase: build (and export)",
        description = "Step 3 of a showcase: build the plan (the kept one, or `plan` as edited) into a new 16:9 sequence — cold open, each entry with its narration, shots where the subject is in view, a lower third, transitions, the year cards, an on-screen counter, music ducked under the voice — all as ordinary undoable engine commands. With `export`, export it (YouTube 1080p) with chapters and a QA report beside it; with `upload`, then send the export to Google Photos (into `upload_album` too). Runs as the `autocut.showcaseBuild` engine job (progress, cancellable).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = true)
    )]
    async fn showcase_build(
        &self,
        Parameters(p): Parameters<ShowcaseBuildParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.showcaseBuild", showcase_build_params(p), &context).await)
    }

    #[tool(
        title = "Scenes: index clips shot by shot",
        description = "Build the scene index of media items (the `media.analyzeScenes` engine job: progress notifications, cancellable; one undo step): each clip's shots (TransNetV2 cuts, else the luma/histogram detector), per shot its sharpness, exposure, camera move and shake, a LAION aesthetic score, a SigLIP 2 embedding for search, and a structured description from a local Qwen3-VL — shot type, subjects with boxes, actions, setting, mood, energy, legible text, people and prominent faces, a caption, a paragraph with detail \"full\", and uses (hook, b-roll, reveal, talking-head, establishing). Everything runs locally (the GPU on macOS); models missing are skipped — `scenes.status` lists them with sizes and licences and `scenes.downloadModel` fetches them after the user agreed (command_run). Indexes are kept with the project and in a cache by file content, so footage is indexed once. Read them with `media.scenes`, `scenes_search`, `media.bestShots` (command_run).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn scenes_analyze(
        &self,
        Parameters(p): Parameters<ScenesAnalyzeParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("media.analyzeScenes", scenes_analyze_params(p), &context).await)
    }

    #[tool(
        title = "Scenes: find shots by what they show",
        description = "Search the scene index (`scenes_analyze` first) for shots matching words: ranked by how much they look like it (SigLIP 2 image–text similarity), what their descriptions say and what is said in them (transcripts). Returns {hits: [{item, name, shot, start, end, keyframe, caption, camera, quality, score, visual, text, speech}]}, times in media seconds: put a hit on the timeline with its item and range.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn scenes_search(&self, Parameters(p): Parameters<ScenesSearchParams>) -> Result<CallToolResult, McpError> {
        if p.query.trim().is_empty() {
            return Ok(fail("`query` is empty: say what to find"));
        }
        let mut params = json!({"query": p.query});
        if let Some(l) = p.limit {
            params["limit"] = json!(l);
        }
        if let Some(i) = p.items {
            params["items"] = json!(i);
        }
        wrap(self.run("media.searchScenes", params).await)
    }

    #[tool(
        title = "Shorts: make vertical shorts from a long video",
        description = "Long video → shorts (the `autocut.shorts` engine job: progress, cancellable). The transcript of `sequence` (default the active one) or of media `item` (put in a new sequence) — Whisper's, made first when there is none — is cut into sentences; runs of them `min_seconds`..`max_seconds` long are scored for interest and for the strength of their opening hook (the local Ollama model rates the best openings and titles them when it is there), and the best `count` that do not overlap each become a vertical sequence: `sequence.autoReframe` to `aspect` (the speaker followed by the subject tracker, Motion keyframes; where nobody is found, the whole picture over a blurred, dimmed copy of itself — `fallback` \"letterbox\" for black bars), big outlined captions from the transcript, an optional `title` card. With `export` (a folder) each is exported with the captions burned in; with `upload_youtube` each export goes to YouTube, private unless `youtube_privacy`. Returns each short's sequence, range, score, hook, title and export.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = true)
    )]
    async fn shorts_make(
        &self,
        Parameters(p): Parameters<ShortsParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.shorts", shorts_params(p), &context).await)
    }

    #[tool(
        title = "Speech: understand raw talking clips",
        description = "Understand the speech of media clips (the `media.understandSpeech` engine job: progress, cancellable). Per clip: where speech is (Silero VAD, else an energy detector), the words with times and confidences (Whisper: the Malayalam model for language \"ml\", large-v3-turbo for \"en\", when downloaded), who speaks (ECAPA speaker embeddings, else MFCC clustering), segments (sentences) with speaker, text, words, confidence and fluency, pauses, filler words (English and Malayalam) and hesitations, retakes — a line said again or a start abandoned — with the take to keep, and a talking-head / voice / b-roll hint (a face on screen, when the face detector is downloaded). Stored with the project beside each clip's transcript (one undo step); `media.inspectSpeech` reads it back, `transcript.removeRetakes` and `sequence.roughCut` edit with it.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn speech_understand(
        &self,
        Parameters(p): Parameters<SpeechUnderstandParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("media.understandSpeech", speech_understand_params(p), &context).await)
    }

    #[tool(
        title = "Rough cut: a clean talking edit from raw clips",
        description = "Raw talking clips → a clean talking edit (the `autocut.roughCut` engine job: progress, cancellable). The clips (`items`, or `paths` to import) are understood (as `speech_understand`), then their kept speech goes on a new sequence back to back: silences longer than `min_pause` (0.6 s), filler words and abandoned takes (every take of a repeated line but the most fluent) are cut, every cut between words with `keep_pause` (0.12 s) of silence kept, no gaps. With `export` (an .mp4 path) the cut is rendered. Returns the analyses, the new sequence, the ranges kept and what was removed. One undo step undoes the cut.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn rough_cut(
        &self,
        Parameters(p): Parameters<RoughCutParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.roughCut", rough_cut_params(p), &context).await)
    }

    #[tool(
        title = "Quick Short: a vertical Short or Reel from a template filled automatically",
        description = "An album, a folder or clips → a finished vertical Short or Reel (the `autocut.quick` engine job: progress, cancellable). The media (`items`, or `paths`: files or folders to import; default the project's videos and photos) is indexed shot by shot (`media.analyzeScenes`: cuts, quality, a local Qwen3-VL description; the most promising shots described again in full detail), its speech understood (`media.understandSpeech`), a template chosen (`template` \"auto\": the AI-ready Shorts starters first — Tech Reveal, Event Recap Short, Talking-head Tip, Gadget Showcase, Before / After Short, Day in 30s — by the shots, whether someone talks to the camera, their energy, `seconds` and `topic`; or an id or name) and filled: each slot's shot by its AI hints, no moment used twice, trimmed to its best moment and framed on its subject for the vertical crop, the text written by the local model (Ollama) from the shots and what is said, or from the descriptions. Music generated at the tempo that brings it near `seconds` (or `music`), the `look`, Bold animated captions of the kept speech in the `platform`'s safe zone, faces and plates hidden (`redact`; never the speaker's own face). With `export`: the video, `<out>.sheet.jpg`, `<out>.report.json` and QA. Returns the report: the template and why (with the runners-up), each shot and why (scores and reasons), the texts and who wrote them, the models used and the fallbacks where a model was missing. Missing models are never downloaded unless `download` (ask the user first).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn quick_short(
        &self,
        Parameters(p): Parameters<QuickShortParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.quick", quick_short_params(p), &context).await)
    }

    #[tool(
        title = "Director: normal and 360° clips directed into one catchy edit",
        description = "Normal and 360° clips → one catchy edit on the beat (the `autocut.direct` engine job: progress, cancellable). The media (`items`, `paths`, Insta360 recordings by name in `insta360` — imported from Insta360 Studio's download folder, else fetched from the user's Insta360 cloud by driving Studio, so tell the user first — a Google Photos `album`, or the project's) is indexed shot by shot (cuts, quality, the local Qwen3-VL's subjects; 360° media as its stitched panorama, so a subject's box is a direction), its speech understood, and each 360° shot looked round for where the action is (`media.salience360`). The local model (Ollama) directs from `brief`: a hook (a 360° clip's tiny-planet intro when there is a good one), the shots in filming order alternating 360° and normal, an ending; each shot's length in beats and its camera — 360°: planetIn, planetOut, orbit, barrelRoll (once at most), whipPan between two things the shot saw, reveal from the sky, drift, pushIn, follow (`clip.follow360`), hyperlapse (`clip.hyperlapse360`); normal: hold, pushIn (Ken Burns on its subject), follow (Auto Reframe), speedRamp. The plan is checked and repaired (unknown clips, cameras on the wrong kind, lengths, the total, English text) and rules direct without a model; the report's `director` says which and `plan.notes` every repair. `dry_run`: the plan and the engine commands it compiles to, nothing built. Built on generated music's beats (or `music`, its beats measured), with a soft swish on each whip and roll and a chime under the English title — never a whoosh or riser, no light leaks or colour tints — a cross dissolve into the ending, captions of English speech, each sound levelled and −14 LUFS. 360° shots render through the GPU's virtual camera. With `export`: the video, `<out>.sheet.jpg`, `<out>.report.json` and QA. Faces stay sharp unless `redact` asks. Missing models are never downloaded unless `download` (ask the user first).",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn autocut_direct(
        &self,
        Parameters(p): Parameters<DirectParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.direct", direct_params(p), &context).await)
    }

    #[tool(
        title = "Trailer: a vertical trailer of a finished showcase",
        description = "A finished showcase → a vertical trailer for Instagram Reels (the `autocut.trailer` engine job: progress, cancellable). From the show's `sequence`, its plan (`plan_path`, default the kept plan) and build report (`report_path`, what `filmcraft-cli showcase` prints; default the last build): a hook — the most striking robot (or `hook`, an entry label) under a big title (\"48 robots · 2013 → 2025\", or `title`) with the show's first narration sentence over it (`hook_voice`) — then `robots` robots in date order (12–18 by default), each 1.3–2.4 s on its best moment inside the window the show played, reframed following the robot (`clip.autoReframe`, the analysis' box; the whole picture over a blurred copy where it is lost), its name and year inside the `platform`'s safe zone (reels by default), every cut on a beat of generated music (`bpm` 124, `mood` energetic, a fixed `seed`), and an end card on the music's final hit: the `logo`, `end_text` (\"Full video on YouTube\") and the `channel` handle (the plan's outro by default). The show's redaction is kept (faces hidden at the size the show hid them, the plan's named targets). Mixed speech first at −14 LUFS, −1 dBTP. With `export` (an .mp4 path, Social Vertical 1080×1920): the video, `<out>.cover.jpg`, `<out>.caption.txt` (hashtags, \"link in bio\"), `<out>.frames.jpg` and QA (`<out>.qa.json`). Returns the sequence, each shot (robot, year, window, followed), the cuts on the beat, the safe-zone check and QA's summary.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn trailer_make(
        &self,
        Parameters(p): Parameters<TrailerParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("autocut.trailer", trailer_params(p), &context).await)
    }

    #[tool(
        title = "Auto Reframe Sequence",
        description = "A copy of a sequence in another shape — `aspect` \"9:16\" (default), \"1:1\", \"4:5\" or \"16:9\" — whose clips follow their subject (`sequence.autoReframe`, one undo step): a pure-Rust tracker (Median Flow on pyramidal Lucas–Kanade, re-detection by template after occlusions and cuts) seeded from `subject` boxes (0..1 of the picture: {x, y, w, h, clip?, seconds?}, e.g. the vision model's) or the salient region, smoothed by a critically damped zero-phase follow (`motion`: smooth keeps a calm camera window, responsive follows closely) into Motion Position/Scale keyframes; clips with nobody to follow are letterboxed (`fallback: \"center\"` crops centred instead; `\"blur\"` fits them whole over a blurred, dimmed copy of themselves, and eases a followed clip out to that wherever its subject is lost for a second or more). `range` {startSeconds, endSeconds} reframes only that part. Runs as an engine job with progress; returns the new sequence and, per clip, whether its subject was followed.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn sequence_auto_reframe(
        &self,
        Parameters(p): Parameters<AutoReframeParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        Ok(self.run_long("sequence.autoReframe", auto_reframe_params(p), &context).await)
    }

    #[tool(
        title = "Auto Redact Faces and Plates",
        description = "Hide clearly recognisable faces and legible vehicle number plates behind a tracked blur (`clip.autoRedact` with `clips`, else `sequence.autoRedact`; one undo step). Local detectors (YuNet faces, LPD-YuNet plates, and for faces in profile or half hidden the D-FINE person detector with RTMPose poses, `profiles`, on by default once downloaded; download them once with `redact.downloadModels` after the user agrees: 4.4 MB + 35.8 MB, MIT / Apache-2.0) run on every analysed frame; detections become smoothed tracks, faces smaller than `min_face_height` of the picture (5 %) are left alone, and with `verify` the local vision model sees one crop per track so humanoid robots, mannequins, posters, screens and signs stay sharp. `exclude` boxes are never hidden. `targets` names logos, brands or signs to hide too: the vision model looks for them every `target_every` seconds and at shot changes, a tracker follows them between looks, a rectangle mask each. Each clip gets one Gaussian Blur (or Mosaic) with a keyframed ellipse / quad mask per track, ahead of its other effects, so it follows the clip's motion, scale and crop. `dry_run` reports the tracks only. Runs as an engine job with progress; returns per clip what was hidden and why the rest was not.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn auto_redact(
        &self,
        Parameters(p): Parameters<AutoRedactParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let (id, params) = auto_redact_params(p);
        Ok(self.run_long(id, params, &context).await)
    }

    #[tool(
        title = "Isolate Voice",
        description = "Keep the speech of audio clips and remove the rest — noise, crowds, music, hum, room (`audio.isolateVoice`; one undo step). Puts a Voice Isolation effect first in each clip's chain (or sets the one it has) at `amount` 0..100 (80): at 0 the clip is unchanged, at 100 it is the model's output alone. Runs DeepFilterNet 3 (pure Rust, MIT/Apache-2.0) once `audio.downloadVoiceModel` installed it after the user agreed (8 MB; `audio.voiceModelStatus` says), else a DSP noise reduction; the result says which (`neural`). 40 ms of latency, compensated.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn isolate_voice(&self, Parameters(p): Parameters<VoiceParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("audio.isolateVoice", voice_params(p)).await)
    }

    #[tool(
        title = "Enhance Voice",
        description = "The one-knob voice boost for clips that keep their own speech (`audio.enhanceVoice`; one undo step): an Enhance Voice effect at `amount` 0..100 (70) — voice isolation, a rumble high-pass, a gentle presence and consonant lift, a de-esser, an expander between words, a 3:1 compressor and a -1 dBFS limiter — then each clip's gain set so it measures `lufs` through its effects (-16 LUFS; \"mix\": the Dialogue target; \"none\": leave the level). Not for synthetic narration, which needs none of it.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn enhance_voice(&self, Parameters(p): Parameters<VoiceParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("audio.enhanceVoice", voice_params(p)).await)
    }

    #[tool(
        title = "Enhance Voice in a File",
        description = "Process a media file's sound straight to a .wav or .m4a file (`audio.enhanceFile`): `mode` \"enhance\" (Enhance Voice, default) or \"isolate\" (Voice Isolation alone), `amount`, the level to `lufs` (-16) under a -1 dBFS true-peak limiter. Returns the loudness, noise floor (between words) and 1-4 kHz share before and after, which model ran and how fast. Files of at most 30 minutes.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn enhance_audio_file(&self, Parameters(p): Parameters<VoiceFileParams>) -> Result<CallToolResult, McpError> {
        let mut v = json!({"path": p.path, "out": p.out});
        if let Some(m) = p.mode {
            v["mode"] = json!(m);
        }
        if let Some(a) = p.amount {
            v["amount"] = json!(a);
        }
        if let Some(l) = p.lufs {
            v["lufs"] = l;
        }
        if let Some(b) = p.bitrate {
            v["bitrate"] = json!(b);
        }
        wrap(self.run("audio.enhanceFile", v).await)
    }

    #[tool(
        title = "Showcase: inspect",
        description = "What the session keeps of the showcase: the analysis report (a summary, or every clip with report: true), the plan, the last build, and the latest showcase job (running or done).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn showcase_inspect(&self, Parameters(p): Parameters<ShowcaseInspectParams>) -> Result<CallToolResult, McpError> {
        wrap(self.run("autocut.showcaseInspect", json!({"report": p.report.unwrap_or(false)})).await)
    }

    #[tool(
        title = "Save a feedback bundle",
        description = "Write one folder to diagnose a problem from (`app.feedbackBundle`): manifest.json (app version, build, OS, every file), system.json, history.json (undo / redo steps with times, the last commands), events.json, jobs.json, perf.json (perf.stats; the running app's, with playback and frame workers, in bridge mode), project.json (project.inspect), preferences.json, qa.json (the last AutoCut QA report), logs/ (the end of the app's logs) and crashes/ (recent crash reports). Secrets (tokens, passwords, API keys, cookies) and the home folder are taken out; capped at 8 MiB. Returns the folder and its files: read them with your file tools.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn feedback_bundle(&self, Parameters(p): Parameters<FeedbackParams>) -> Result<CallToolResult, McpError> {
        let mut params = json!({});
        if let Some(path) = p.path {
            params["path"] = json!(path);
        }
        if let Some(qa) = p.qa_report {
            params["qaReport"] = json!(qa);
        }
        wrap(self.run("app.feedbackBundle", params).await)
    }

    #[tool(
        title = "Video templates: list",
        description = "The video templates (whole-sequence starters with media slots — 21 built in: photo-beat slideshow, channel intro, top 3 / top 5, talking head with auto-captions, product promo, travel vlog intro and outro, before/after, lower thirds + chapters, end card, tutorial with webcam, podcast, collage, quote card, countdown, event recap, day in my life, unboxing, Q&A, robot showcase — and the user's own from <data dir>/Video Templates): id, name, length, variants (16:9, 9:16, 1:1, 4:5), slots (kind, count, required). Pass `template` for one in full, with every placement's time and the beat grid. Then template_apply (a `captions` slot is transcribed from its sources).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn template_list(&self, Parameters(p): Parameters<TemplateListParams>) -> Result<CallToolResult, McpError> {
        match p.template {
            Some(t) => wrap(self.run("template.inspect", json!({"template": t})).await),
            None => wrap(self.run("template.list", p.query.map(|q| json!({"query": q})).unwrap_or(json!({}))).await),
        }
    }

    #[tool(
        title = "Video templates: apply",
        description = "Make a new sequence from a video template (`template.apply`): media conforms to each slot (trimmed, looped, stretched or held; music Remixed), effects, transitions and titles stay with the slots, cuts land on the music's beats (re-measured when you give your own track; an unfilled music slot gets a generated bed), the variant sets the frame (16:9 or 9:16) and the template's look, caption style and export preset come along. Files are imported (once each). One undo step; returns the sequence, what filled each placement and how, and warnings.",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn template_apply(&self, Parameters(p): Parameters<TemplateApplyParams>) -> Result<CallToolResult, McpError> {
        let mut params = json!({"template": p.template});
        if let Some(v) = p.variant {
            params["variant"] = json!(v);
        }
        if let Some(s) = p.slots {
            params["slots"] = s;
        }
        if let Some(n) = p.name {
            params["name"] = json!(n);
        }
        wrap(self.run("template.apply", params).await)
    }

    #[tool(
        title = "Recipes: list",
        description = "The recipes: one-click chains of engine commands and AutoCut intents with parameters (built in: talking-head clean-up — pauses, fillers, voice, captions; vertical short from a long clip — highlights, 9:16, captions, export; apply a template and publish privately to YouTube — publishing off unless asked), and the user's own from <data dir>/Recipes. Pass `recipe` for one in full: its parameters (kinds, defaults, choices) and its steps as they would run with `params`. Then recipe_run. Recipes are saved with command_run `recipe.save` or recorded with `recipe.record`.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn recipe_list(&self, Parameters(p): Parameters<RecipeListParams>) -> Result<CallToolResult, McpError> {
        match p.recipe {
            Some(r) => {
                let mut q = json!({"name": r});
                if let Some(v) = p.params {
                    q["params"] = v;
                }
                wrap(self.run("recipe.inspect", q).await)
            }
            None => wrap(self.run("recipe.list", json!({})).await),
        }
    }

    #[tool(
        title = "Recipes: run",
        description = "Run a recipe on the open project (`recipe.run`, an engine job: progress, cancellable). Every step is checked before it runs — an AutoCut intent is planned against the open sequence and its commands checked against the registry and the recipe allowlist; a command that starts a job (an export, a publish) is waited for; a failing step stops the run and the answer says which and why. Each edit is an ordinary undoable command. Returns each step's commands and results (`recipe`). `dry_run` plans only.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = true)
    )]
    async fn recipe_run(
        &self,
        Parameters(p): Parameters<RecipeRunParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let mut q = json!({"name": p.recipe});
        if let Some(v) = p.params {
            q["params"] = v;
        }
        if p.dry_run.unwrap_or(false) {
            q["dryRun"] = json!(true);
            return wrap(self.run("recipe.run", q).await);
        }
        q["wait"] = json!(true);
        Ok(self.run_long("recipe.run", q, &context).await)
    }

    #[tool(
        title = "Render a frame",
        description = "Render the active sequence's frame (at `seconds` or the playhead) and return it as PNG.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn render_frame(&self, Parameters(p): Parameters<RenderParams>) -> Result<CallToolResult, McpError> {
        let max = p.max_side.unwrap_or(960);
        match &*self.backend {
            Backend::Headless(s) => {
                let s = s.clone();
                let r = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, AutomationError> {
                    let g = s.lock().unwrap_or_else(PoisonError::into_inner);
                    // read-only: render at `seconds` without moving the playhead (#114)
                    let t = p.seconds.map_or_else(|| g.playhead(), filmcraft_time::Tick::from_seconds_f64);
                    let (w, h) =
                        g.active_sequence().map(|q| (q.settings.width, q.settings.height)).ok_or_else(|| AutomationError::Other("no sequence".into()))?;
                    let scale = render_scale(max, w, h);
                    let img = g.try_render_program_at(scale, t).map_err(|e| AutomationError::Other(e.to_string()))?;
                    png_rgba(img.w as u32, img.h as u32, img.over_black_rgba8(), max)
                })
                .await
                .unwrap_or_else(|e| Err(join_error(e)));
                Ok(match r {
                    Ok(png) => png_result(&png, "rendered frame"),
                    Err(e) => fail(e),
                })
            }
            Backend::Bridge(b) => {
                let Some(sec) = p.seconds else {
                    return self.screenshot(&b.clone(), Some("Program".into()), max).await;
                };
                // The Program monitor shows the playhead's frame, so the live playhead has to move
                // for the screenshot; put it (and the selection, which may follow it) back after,
                // so the tool stays read-only (#114).
                let before = match self.run("sequence.inspect", json!({})).await {
                    Ok(v) => v,
                    Err(e) => return Ok(fail(e)),
                };
                if let Err(e) = self.run("playhead.set", json!({"seconds": sec})).await {
                    return Ok(fail(e));
                }
                let shot = self.screenshot(&b.clone(), Some("Program".into()), max).await;
                if let Some(t) = before.get("playhead").and_then(Value::as_i64) {
                    let _ = self.run("playhead.set", json!({"time": t})).await;
                }
                // only when it changed: selecting also leaves trim mode
                let now = self.run("sequence.inspect", json!({})).await.ok();
                if let Some(sel) = before.get("selection").filter(|v| v.is_array())
                    && now.as_ref().and_then(|n| n.get("selection")) != Some(sel)
                {
                    let _ = self.run("timeline.select", json!({"clips": sel})).await;
                }
                shot
            }
        }
    }

    #[tool(
        title = "Live app: UI state",
        description = "Live app: UI state (tool, workspace, panels, timeline zoom, playback, selection, fps).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_inspect(&self) -> Result<CallToolResult, McpError> {
        self.ui("ui.inspect", json!({})).await
    }

    #[tool(
        title = "Live app: on-screen elements",
        description = "Live app: every on-screen interactive element with id, label and rect (points). Use ids with ui_click/ui_drag.",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_elements(&self, Parameters(p): Parameters<ElementsParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.elements", json!({"prefix": p.prefix.unwrap_or_default()})).await
    }

    /// Forward a click to the live editor; an omitted count is sent as null and uses its default.
    #[tool(
        title = "Live app: click",
        description = "Live app: click an element by id or at x,y (points). Supports right/middle button, double-click (count=2) and modifiers.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_click(&self, Parameters(p): Parameters<ClickParams>) -> Result<CallToolResult, McpError> {
        let mut v = json!({"button": p.button, "count": p.count, "modifiers": p.modifiers});
        if let Some(id) = p.id {
            v["id"] = json!(id);
        } else {
            v["x"] = json!(p.x);
            v["y"] = json!(p.y);
        }
        self.ui("ui.click", v).await
    }

    /// Forward a drag to the live editor, which applies the step default and validates its bounds.
    #[tool(
        title = "Live app: drag",
        description = "Live app: press-drag-release from one point/element to another (move clips, trim edges, scrub, resize panels, drop project items onto tracks).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_drag(&self, Parameters(p): Parameters<DragParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.drag", json!({"from": p.from, "to": p.to, "steps": p.steps, "modifiers": p.modifiers})).await
    }

    #[tool(
        title = "Live app: press a key",
        description = "Live app: press a key or shortcut (e.g. `Space`, `L`, `Cmd+K`, `Shift+Delete`, `Cmd+Z`).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_key(&self, Parameters(p): Parameters<KeyParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.key", json!({"key": p.key})).await
    }

    #[tool(
        title = "Live app: type text",
        description = "Live app: type text into the focused field.",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_type(&self, Parameters(p): Parameters<TypeParams>) -> Result<CallToolResult, McpError> {
        self.ui("ui.type", json!({"text": p.text})).await
    }

    #[tool(
        title = "Live app: screenshot",
        description = "Live app: screenshot of the window or one panel (PNG).",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn ui_screenshot(&self, Parameters(p): Parameters<ShotParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => self.screenshot(&b, p.panel, p.max_side.unwrap_or(1600)).await,
            None => Ok(fail("ui_screenshot needs bridge mode")),
        }
    }

    #[tool(
        title = "Live app: control-channel call",
        description = "Live app: call any control-channel method directly (ui.set, ui.scroll, ui.timeline.hit, ui.timeline.locate, ui.playback, ui.panel.show, ui.menu.list, …).",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false)
    )]
    async fn ui_control(&self, Parameters(p): Parameters<ControlParams>) -> Result<CallToolResult, McpError> {
        self.ui(&p.method, p.params.unwrap_or(json!({}))).await
    }
}

impl FilmcraftMcp {
    async fn screenshot(&self, b: &BridgeClient, panel: Option<String>, max: u32) -> Result<CallToolResult, McpError> {
        // The desktop bridge needs a pathname, so use a freshly-created,
        // owner-only directory rather than exposing image bytes at a
        // predictable shared-temp filename.
        let dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(e) => return Ok(fail(format!("screenshot: {e}"))),
        };
        let path = dir.path().join("screenshot.png");
        if let Err(e) = b.call("ui.screenshot", json!({"path": path.to_string_lossy(), "panel": panel})).await {
            return Ok(fail(e));
        }
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => return Ok(fail(format!("screenshot: {e}"))),
        };
        let bytes = match image::load_from_memory(&bytes) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                png_rgba(w, h, rgba.into_raw(), max).unwrap_or(bytes)
            }
            Err(_) => bytes,
        };
        Ok(png_result(&bytes, "screenshot of the live FilmCraft window"))
    }
}

impl FilmcraftMcp {
    /// The first argument `tool` doesn't take, as a message naming the ones it does.
    fn unknown_arg(&self, tool: &str, args: Option<&rmcp::model::JsonObject>) -> Option<String> {
        let def = self.tool_router.get(tool)?;
        let props = def.input_schema.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
        let bad = args?.keys().find(|k| !props.contains_key(*k))?;
        let accepted: Vec<&str> = props.keys().map(String::as_str).collect();
        let accepted = if accepted.is_empty() { "no arguments".to_string() } else { accepted.join(", ") };
        Some(format!("unknown argument \"{bad}\" for {tool}; expected: {accepted}"))
    }
}

const INSTRUCTIONS: &str = "FilmCraft video editor (Premiere Pro-class). Every edit is an engine command: `command_list` to discover ids/params, `command_run` to execute (undoable; `command_batch` runs several). `doc_inspect` (or `project_inspect`/`sequence_inspect`) returns ids you can pass to commands; `render_preview` shows the result. In bridge mode the `ui_*` tools drive the live app: `ui_elements` lists clickable ids, `ui_click`/`ui_drag`/`ui_key` operate it, `ui_screenshot` shows it. Time is in ticks: 254016000000 per second (commands also accept `seconds`, `frame` or `timecode`). For plain-language requests, `autocut_plan` turns a prompt into a validated command batch (dry run) and `autocut_edit` runs it; local models (Ollama) handle routine edits, and a needsSupervisor result means you plan that one yourself with the command tools. Cloud media: `googlephotos_pick` opens the Google Photos picker — the user chooses items, or pass `query` and a local vision agent (Qwen2.5-VL via Ollama, plus Screen Recording/Accessibility grants) drives the picker itself: it types the query into the picker's search, selects matching videos and presses Done (pass `queries` with `all` for many searches — a list, or one per month of a whole history — in one session and one tab, `left` returning what did not fit), or pass `album` and it picks that whole album (the user's curated seed); `googlephotos_cached`/`googlephotos_add` re-import earlier picks (after the one-time `filmcraft-cli photos auth`). `googlephotos_upload` sends a finished export to the user's Google Photos library (a long job with progress; needs `filmcraft-cli photos auth --upload` once); with `album` it also goes into that album (FilmCraft's album of that title, created when missing; needs `photos auth --albums`), and `photos.createAlbum` / `photos.albums` / `photos.addToAlbum` (command_run) manage the albums FilmCraft made. 360° video: `insta360_list` / `insta360_import` bring the user's Insta360 recordings (from Insta360 Studio's download folder or a camera card) in as 360° media (the camera's .lrv attached as the proxy, the two lenses aligned), and `insta360_download` fetches named recordings from their Insta360 cloud by driving Insta360 Studio with the local vision agent (tell the user first: it moves the mouse); `clip.reframe360` (command_run) looks into a 360° clip through a keyframable virtual camera (yaw, pitch, roll, fov, distance; presets natural, wide, tinyPlanet, rabbitHole, sphere), `clip.stabilize360` steadies it (the camera's turn measured from the pictures; mode smooth, light, strong or lock), `clip.follow360` turns it to follow a subject (target subject, box or face), `clip.vrMove` writes a camera move (planetIn, planetOut, orbit, barrelRoll, whipPan, reveal, pushIn, crashZoom, drift) as editable keyframes, `clip.hyperlapse360` speeds and steadies a long stretch, `media.salience360` says where the action is round the sphere, and `media.vrProperties` / `media.align360` set and refine how a picture maps onto the sphere. `autocut_direct` turns normal and 360° clips into one catchy edit on the beat, the local model directing every shot and camera move (rules without a model; `dry_run` shows the plan and its commands first). `youtube_upload` publishes a video to one of the user's connected YouTube channels (private by default; `youtube.channels` lists them, `youtube.connect` connects another — the user picks it in Google's account chooser — `youtube.setDefault` / `youtube.disconnect` via command_run). `publish_run` publishes to a destination (youtube, youtubeShort, instagramReel) after the shared pre-flight (`publish.preflight` shows the checklist, `publish.fixCopy` writes a copy that passes); `instagram_publish` publishes a video as a Reel on the user's connected Instagram professional account (`instagram.account`, `instagram.limits` via command_run; the user connects it once with `filmcraft-cli instagram auth`). Showcases (\"make a video of every robot since 2020\"): `showcase_analyze` finds every clip of a subject or of what a `prompt` asks for (plan: true also plans; with `album`, the clips are that whole Google Photos album, picked and imported first), `showcase_plan` groups them into entries and writes the voice-over, the user reviews the plan with you, `showcase_build` builds (and exports) it; `showcase_inspect` shows what is kept. Motion tracking (command_run): `track.create` follows an object boxed on a clip (`subject` or flat-surface `points` tracking, both ways, kept in the project), `track.attach` makes a clip, graphic layer, effect point or mask follow it (keyframes, re-written by `track.refine`), and `track.label` / `track.blurObject` / `track.spotlight` track and add a callout, a privacy blur or a spotlight in one step; `clip.cameraTrack` keeps a subject, box or face framed with a gentle push-in. Vertical video: `sequence_auto_reframe` makes a 9:16 (1:1, 4:5) copy of a sequence whose clips follow their subject (a local tracker, Motion keyframes), and `shorts_make` turns a long talk into scored, reframed, captioned shorts (exported, and uploaded to YouTube privately when asked); `trailer_make` cuts a finished showcase into a vertical trailer for Reels (a hook, robots in date order on the beat, an end card; the show's redaction kept). Talking clips: `speech_understand` finds the words, speakers, pauses, fillers and retakes of raw clips (English or Malayalam), and `rough_cut` turns them into a clean talking edit (pauses, fillers and abandoned takes cut between words; exported when asked). Quick templates: `quick_short` turns an album, a folder or the selected clips into a finished vertical Short or Reel — it indexes the scenes, understands the speech, picks a template (the AI-ready Shorts starters, or `template`), fills every slot from the scene index and the speech, writes the text with the local model and exports with a report of every choice; it runs without any model (and says what it fell back on), and downloads models only with `download` after the user agreed. Voice: `isolate_voice` keeps clips' speech and removes noise, crowds and music, `enhance_voice` is the one-knob voice boost for clips that keep their own speech (isolation, EQ, de-esser, compression, a loudness match), `enhance_audio_file` does either straight on a file (`audio.voiceModelStatus` says whether the neural model is installed; `audio.downloadVoiceModel` only after the user agreed). Scenes: `scenes_analyze` indexes clips shot by shot (cuts, sharpness, exposure, camera move, an aesthetic score and a local Qwen3-VL description of each shot; `scenes.status` lists the models, `scenes.downloadModel` only after the user agreed), and `scenes_search` finds shots by what they show or what is said in them (`media.bestShots` and `media.scenes` via command_run). Privacy: `auto_redact` hides visitors' faces and number plates behind a tracked blur, robots and signs left sharp (`redact.status` says whether the detectors are installed; `redact.downloadModels` only after the user agreed). Music (command_run): `music.suggest` {mood: epic|tense|uplifting|calm|happy|dark|dreamy|energetic|sad or a synonym, genre: cinematic|electronic|pop|ambient|lofi|piano, useCase, duration, energy} ranks styles and returns `audio.generate` params; `music.preview` renders exactly what `audio.generate` {kind: music, …, seed, energy (0–1 or a curve), levels/mute per stem (drums, bass, melody, pad), stems, soundfont} would place, for the user to hear first; music plays on the built-in oscillators or a SoundFont (`soundfont`: auto, oscillators, generaluser-gs or an .sf2 path; `music.useSoundfont` keeps the choice) with the same arrangement and beat map either way. `music.soundfonts` shows GeneralUser GS (size, source, licence text, the author's note on sample provenance); download it only after the user read and agreed: `music.downloadSoundfont` {consent: true}. `music.generateNeural` (ACE-Step, once `music.downloadNeuralModel` installed it) makes music from a description.";

/// Resources: the project (as `doc_inspect`) and the command catalog (as `command_list`).
const DOCUMENT_URI: &str = "filmcraft://document";
const COMMANDS_URI: &str = "filmcraft://commands";

/// MCP 2026-07-28 clients (current Claude Code) require `ttlMs` and `cacheScope` on list and read
/// results; rmcp fills `resultType` and leaves the cache hints to the server, as the generated
/// `tools/list` does. The resource list never changes; the document does, so reads aren't cached.
fn cache_hints(context: &rmcp::service::RequestContext<rmcp::RoleServer>, ttl_ms: u64) -> Option<(u64, rmcp::model::CacheScope)> {
    context.protocol_version().is_some_and(|v| v >= rmcp::model::ProtocolVersion::V_2026_07_28).then_some((ttl_ms, rmcp::model::CacheScope::Private))
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for FilmcraftMcp {
    fn get_info(&self) -> rmcp::model::ServerConfig {
        rmcp::model::ServerConfig::new(rmcp::model::ServerCapabilities::builder().enable_tools().enable_resources().build())
            .with_server_info(rmcp::model::Implementation::new("filmcraft", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS.to_string())
    }

    /// Strict arguments and long exports in front of the tool router.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        if let Some(m) = self.unknown_arg(&request.name, request.arguments.as_ref()) {
            return Err(McpError::invalid_params(m, None));
        }
        // A blocking export reports progress and can be cancelled (docs/agents.md § Long exports).
        // Bridge mode too: the app runs it as a job, so it stays responsive (#91, #92).
        if request.name == "command_run"
            && let Some(a) = &request.arguments
            && let Some(id) = a.get("id").and_then(Value::as_str)
        {
            let params = a.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
            if crate::long_job::is_long(id, &params) {
                return Ok(self.run_long(id, params, &context).await.into());
            }
        }
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    async fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListResourcesResult, McpError> {
        use rmcp::model::Resource;
        let mut r = rmcp::model::ListResourcesResult::with_all_items(vec![
            Resource::new(DOCUMENT_URI, "document")
                .with_title("Project")
                .with_description("The project tree and the active sequence (same as doc_inspect).")
                .with_mime_type("application/json"),
            Resource::new(COMMANDS_URI, "commands")
                .with_title("Command catalog")
                .with_description("Every command with id, label, menu path, shortcut, params and enabled state (same as command_list).")
                .with_mime_type("application/json"),
        ]);
        if let Some((ttl, scope)) = cache_hints(&context, 600_000) {
            r = r.with_ttl_ms(ttl).with_cache_scope(scope);
        }
        Ok(r)
    }

    async fn read_resource(
        &self,
        request: rmcp::model::ReadResourceRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ReadResourceResponse, McpError> {
        let v = match request.uri.as_str() {
            DOCUMENT_URI => self.document().await,
            COMMANDS_URI => self.run("command.list", json!({})).await,
            other => return Err(McpError::resource_not_found(format!("resource not found: {other}"), None)),
        }
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let text = serde_json::to_string_pretty(&v).map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let mut r = rmcp::model::ReadResourceResult::new(vec![rmcp::model::ResourceContents::text(text, request.uri).with_mime_type("application/json")]);
        if let Some((ttl, scope)) = cache_hints(&context, 0) {
            r = r.with_ttl_ms(ttl).with_cache_scope(scope);
        }
        Ok(r.into())
    }
}

/// Render scale that fits the longest side of a `w` x `h` sequence into `max_side` (never upscales; 0 = no limit, as in `png_rgba`).
fn render_scale(max_side: u32, w: u32, h: u32) -> f32 {
    if max_side == 0 {
        return 1.0;
    }
    (max_side as f32 / w.max(h).max(1) as f32).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_scale_zero_means_no_limit() {
        assert_eq!(render_scale(0, 1920, 1080), 1.0);
        assert_eq!(render_scale(960, 1920, 1080), 0.5);
        assert_eq!(render_scale(960, 640, 360), 1.0);
        assert_eq!(render_scale(960, 0, 0), 1.0);
    }

    #[tokio::test]
    async fn headless_commands_and_render() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let m = FilmcraftMcp::headless(s);
        let v = m.run("sequence.addEditAllTracks", json!({"seconds": 2.0})).await.unwrap();
        assert!(v["cuts"].as_u64().unwrap() >= 2);
        let r = m.render_frame(Parameters(RenderParams { seconds: Some(1.0), max_side: Some(320) })).await.unwrap();
        assert_ne!(r.is_error, Some(true));
    }

    /// `feedback_bundle` writes the bundle where it is told and lists its files; a bad path is an
    /// error result.
    #[tokio::test(flavor = "multi_thread")]
    async fn feedback_bundle_tool_writes_the_folder() {
        let dir = std::env::temp_dir().join(format!("fc-mcp-feedback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = demo();
        s.prefs_path = Some(dir.join("data").join("preferences.json"));
        let mut c = Client::start(s);
        c.init().await;
        let out = dir.join("bundle");
        let r = c.call(1, "feedback_bundle", json!({"path": out.to_string_lossy()})).await;
        assert_eq!(r["result"]["isError"], false, "{r}");
        let v = tool_json(&r);
        assert_eq!(v["path"].as_str(), out.to_str());
        let files: Vec<&str> = v["files"].as_array().unwrap().iter().filter_map(|f| f["file"].as_str()).collect();
        for want in ["README.txt", "manifest.json", "history.json", "perf.json", "project.json"] {
            assert!(files.contains(&want) && out.join(want).is_file(), "{want}: {files:?}");
        }
        let bad = c.call(2, "feedback_bundle", json!({"path": ""})).await;
        assert_eq!(bad["result"]["isError"], true, "{bad}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `recipe_list` / `recipe_run`: the built-ins are listed, one is inspected, a saved recipe
    /// (a command and a planned AutoCut intent) runs as a job over MCP — the port's pump thread
    /// serves its commands — and the answer carries each step; a bad name is an error.
    #[tokio::test(flavor = "multi_thread")]
    async fn recipe_tools_list_and_run() {
        let mut s = demo();
        s.recipes.planner = Some(filmcraft_autocut::recipes::planner());
        let mut c = Client::start(s);
        c.init().await;
        let l = tool_json(&c.call(1, "recipe_list", json!({})).await);
        assert!(l["recipes"].as_array().unwrap().iter().any(|r| r["id"] == "builtin:vertical-short"), "{l}");
        let one = tool_json(&c.call(2, "recipe_list", json!({"recipe": "builtin:talking-head-clean-up"})).await);
        assert_eq!(one["ready"], true, "{one}");
        let steps = json!([{"command": "markers.add", "params": {"name": "mcp"}}, {"intent": "enhance_voice", "params": {"loudness": false}}]);
        let save = c.call(3, "command_run", json!({"id": "recipe.save", "params": {"name": "Marks", "steps": steps}})).await;
        assert_eq!(save["result"]["isError"], false, "{save}");
        let r = c.call(4, "recipe_run", json!({"recipe": "Marks"})).await;
        assert_eq!(r["result"]["isError"], false, "{r}");
        let v = tool_json(&r);
        assert_eq!(v["recipe"]["ok"], true, "{v}");
        assert_eq!(v["recipe"]["steps"].as_array().unwrap().len(), 2);
        let bad = c.call(5, "recipe_run", json!({"recipe": "no such recipe"})).await;
        assert_eq!(bad["result"]["isError"], true, "{bad}");
        let dry = tool_json(&c.call(6, "recipe_run", json!({"recipe": "builtin:vertical-short", "dry_run": true})).await);
        assert!(dry["plan"].is_array(), "{dry}");
    }

    /// `template_list` / `template_apply`: the starters are listed, one is inspected, and applying
    /// one makes the new sequence the active one.
    #[tokio::test]
    async fn template_tools_list_and_apply() {
        let m = FilmcraftMcp::headless(Session::default());
        let l = m.template_list(Parameters(TemplateListParams::default())).await.unwrap();
        assert_ne!(l.is_error, Some(true));
        let one = m.template_list(Parameters(TemplateListParams { query: None, template: Some("builtin:top-3-listicle".into()) })).await.unwrap();
        assert_ne!(one.is_error, Some(true));
        let p = TemplateApplyParams {
            template: "builtin:top-3-listicle".into(),
            variant: Some("9:16".into()),
            slots: Some(json!({"title": "Best Bots", "labels": ["A", "B", "C"]})),
            name: Some("Countdown".into()),
        };
        let r = m.template_apply(Parameters(p)).await.unwrap();
        assert_ne!(r.is_error, Some(true), "{r:?}");
        let q = m.run("sequence.inspect", json!({})).await.unwrap();
        assert_eq!(q["settings"]["width"], 1080, "{q}");
        let bad = m.template_apply(Parameters(TemplateApplyParams { template: "nope".into(), variant: None, slots: None, name: None })).await.unwrap();
        assert_eq!(bad.is_error, Some(true));
    }

    /// `render_frame` / `render_preview` are annotated read-only (#114): rendering at `seconds`
    /// leaves the playhead, and the selection that follows it, where they were, and still renders
    /// the frame at `seconds`.
    #[tokio::test]
    async fn rendering_at_a_time_does_not_move_the_playhead() {
        let m = FilmcraftMcp::headless(demo());
        m.run("sequence.selectionFollowsPlayhead", json!({"on": true})).await.unwrap();
        m.run("playhead.set", json!({"seconds": 1.0})).await.unwrap();
        let before = m.run("sequence.inspect", json!({})).await.unwrap();
        let png = |r: CallToolResult| serde_json::to_string(&r.content).unwrap();
        let at_playhead = png(m.render_frame(Parameters(RenderParams { seconds: None, max_side: Some(64) })).await.unwrap());
        for seconds in [5.0, 8.0] {
            let r = m.render_frame(Parameters(RenderParams { seconds: Some(seconds), max_side: Some(64) })).await.unwrap();
            assert_ne!(png(r), at_playhead, "renders the frame at {seconds} s");
            m.render_preview(Parameters(RenderParams { seconds: Some(seconds), max_side: Some(64) })).await.unwrap();
        }
        let after = m.run("sequence.inspect", json!({})).await.unwrap();
        assert_eq!(after["playhead"], before["playhead"]);
        assert_eq!(after["selection"], before["selection"]);
    }

    /// One request/response exchange over `serve_io`, as an MCP client sees it.
    struct Client {
        tx: tokio::io::DuplexStream,
        rx: tokio::io::Lines<tokio::io::BufReader<tokio::io::DuplexStream>>,
    }

    impl Client {
        fn start(s: Session) -> Self {
            use tokio::io::AsyncBufReadExt;
            let (tx, server_in) = tokio::io::duplex(1 << 22);
            let (server_out, rx) = tokio::io::duplex(1 << 22);
            tokio::spawn(FilmcraftMcp::headless(s).serve_io(server_in, server_out));
            Self { tx, rx: tokio::io::BufReader::new(rx).lines() }
        }
        async fn send(&mut self, line: &str) {
            use tokio::io::AsyncWriteExt;
            self.tx.write_all(format!("{line}\n").as_bytes()).await.unwrap();
        }
        async fn next(&mut self) -> Value {
            serde_json::from_str(&self.rx.next_line().await.unwrap().unwrap()).unwrap()
        }
        async fn ask(&mut self, msg: Value) -> Value {
            self.send(&msg.to_string()).await;
            self.next().await
        }
        async fn init(&mut self) -> Value {
            let r = self.ask(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})).await;
            self.send(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string()).await;
            r
        }
        async fn call(&mut self, id: u64, name: &str, arguments: Value) -> Value {
            self.ask(json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":arguments}})).await
        }
    }

    /// `googlephotos_pick` exposes the optional autopick `query`; its schema stays offline-checkable.
    #[test]
    fn googlephotos_pick_schema_has_query() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("googlephotos_pick") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        let Some(props) = schema["properties"].as_object() else { panic!("schema: {schema}") };
        assert!(props.contains_key("query"), "properties: {props:?}");
        assert!(props.contains_key("all") && props.contains_key("since"), "properties: {props:?}");
        assert!(props.contains_key("session"));
        assert!(props.contains_key("max_items"));
        assert!(props.contains_key("album") && props.contains_key("videos_only"), "properties: {props:?}");
        assert_eq!(schema["properties"]["queries"]["type"], json!(["array", "null"]), "{schema}");
        assert!(schema["required"].as_array().is_none_or(|r| r.is_empty()), "{schema}");
    }

    /// `query` and `album` are two different picks: both at once is refused before anything
    /// (even the sign-in) is looked at.
    #[tokio::test]
    async fn googlephotos_pick_takes_a_query_or_an_album() {
        let mut c = Client::start(Session::default());
        c.init().await;
        let r = c.call(1, "googlephotos_pick", json!({"query": "robots", "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not both"), "{r}");
        // many searches and an album: the same refusal
        let r = c.call(2, "googlephotos_pick", json!({"queries": ["videos October 2022", "videos November 2022"], "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not both"), "{r}");
        // too many searches for one call: refused before the sign-in is looked at
        let many: Vec<String> = (0..=MAX_PICK_QUERIES).map(|i| format!("q{i}")).collect();
        let r = c.call(3, "googlephotos_pick", json!({"queries": many})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("at most"), "{r}");
    }

    /// `googlephotos_upload` takes a required `path` and an optional `description` and `album`,
    /// and runs `photos.upload` as a long job (progress, cancel).
    #[test]
    fn googlephotos_upload_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("googlephotos_upload") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        let Some(props) = schema["properties"].as_object() else { panic!("schema: {schema}") };
        assert_eq!(props.keys().map(String::as_str).collect::<Vec<_>>(), ["album", "description", "path"], "{schema}");
        assert_eq!(schema["properties"]["path"]["type"], "string", "{schema}");
        assert_eq!(schema["required"], json!(["path"]), "{schema}");
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4", "description": "Robots"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "description": "Robots", "wait": true}));
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4", "album": "Robot showcase"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "album": "Robot showcase", "wait": true}));
        let p: PhotosUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4"})).unwrap();
        assert_eq!(photos_upload_params(p), json!({"path": "/x/out.mp4", "wait": true}));
        assert!(serde_json::from_value::<PhotosUploadParams>(json!({"description": "no path"})).is_err());
        assert!(crate::long_job::is_long("photos.upload", &json!({"wait": true})));
    }

    /// `youtube_upload` takes `path` and `title` (required) and the metadata, and runs
    /// `youtube.upload` as a long job.
    #[test]
    fn youtube_upload_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("youtube_upload") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        assert_eq!(schema["required"], json!(["path", "title"]), "{schema}");
        assert!(schema["properties"]["madeForKids"].is_object(), "{schema}");
        let p: YouTubeUploadParams = serde_json::from_value(json!({"path": "/x/out.mp4", "title": "Robots", "tags": ["a"], "channel": "@H4hacks"})).unwrap();
        assert_eq!(youtube_upload_params(p), json!({"path": "/x/out.mp4", "title": "Robots", "tags": ["a"], "channel": "@H4hacks", "wait": true}));
        assert!(serde_json::from_value::<YouTubeUploadParams>(json!({"path": "/x"})).is_err(), "a title is needed");
        assert!(crate::long_job::is_long("youtube.upload", &json!({"wait": true})) && crate::long_job::is_long("youtube.connect", &json!({"wait": true})));
        assert!(crate::long_job::JOB_EXTRAS.contains(&"youtube"));
    }

    /// `publish_run` takes a destination and its options, and runs `publish.run` as a long job.
    #[test]
    fn publish_run_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("publish_run") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        assert_eq!(schema["required"], json!(["destination"]), "{schema}");
        let p: PublishRunParams =
            serde_json::from_value(json!({"destination": "instagramReel", "path": "/x/a.mp4", "caption": "c", "uploadOnly": true})).unwrap();
        assert_eq!(publish_run_params(p), json!({"destination": "instagramReel", "path": "/x/a.mp4", "caption": "c", "uploadOnly": true, "wait": true}));
        assert!(crate::long_job::is_long("publish.run", &json!({"wait": true})) && crate::long_job::is_long("publish.fixCopy", &json!({"wait": true})));
        assert!(crate::long_job::JOB_EXTRAS.contains(&"publish"));
    }

    /// `instagram_publish` takes a file or a URL and the Reel's options, and runs
    /// `instagram.publish` as a long job.
    #[test]
    fn instagram_publish_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(tool) = m.tool_router.get("instagram_publish") else { panic!("tool missing") };
        let schema = Value::Object((*tool.input_schema).clone());
        assert!(schema["properties"]["videoUrl"].is_object() && schema["properties"]["coverSeconds"].is_object(), "{schema}");
        let p: InstagramPublishParams =
            serde_json::from_value(json!({"path": "/x/out.mp4", "caption": "Robots #ai", "shareToFeed": false, "coverSeconds": 1.5})).unwrap();
        assert_eq!(
            instagram_publish_params(p),
            json!({"path": "/x/out.mp4", "caption": "Robots #ai", "shareToFeed": false, "coverSeconds": 1.5, "wait": true})
        );
        assert!(crate::long_job::is_long("instagram.publish", &json!({"wait": true})) && crate::long_job::is_long("instagram.connect", &json!({"wait": true})));
        assert!(crate::long_job::JOB_EXTRAS.contains(&"instagram"));
    }

    /// Over MCP, an upload is the engine job: the media item comes back in the answer, and a
    /// host without an uploader (or without consent) gets the reason, not a crash.
    #[tokio::test]
    async fn googlephotos_upload_runs_the_engine_job() {
        struct Fake;
        impl filmcraft_engine::photos::PhotosUploader for Fake {
            fn ready(&self) -> Result<(), String> {
                Ok(())
            }
            fn upload(&self, _: &str, d: Option<&str>, p: &filmcraft_engine::export::Progress) -> Result<Value, String> {
                p.done.store(p.total.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                Ok(json!({"mediaItemId": "M1", "description": d}))
            }
        }
        let file = std::env::temp_dir().join(format!("fc-mcp-upload-{}.mp4", std::process::id()));
        std::fs::write(&file, b"0123456789").unwrap();
        let path = file.to_string_lossy().into_owned();

        let mut c = Client::start(Session::default());
        c.init().await;
        let r = c.call(1, "googlephotos_upload", json!({"path": path})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("not available"), "{r}");
        let r = c.call(2, "googlephotos_upload", json!({"path": path, "folder": "x"})).await;
        assert!(r["error"]["message"].as_str().unwrap().contains("unknown argument \"folder\""), "{r}");

        let mut s = Session::default();
        s.photos = Some(std::sync::Arc::new(Fake));
        let mut c = Client::start(s);
        c.init().await;
        let r = c.call(3, "googlephotos_upload", json!({"path": path, "description": "Robots"})).await;
        let text = r["result"]["content"][0]["text"].as_str().unwrap_or_default();
        let v: Value = serde_json::from_str(text).unwrap_or_else(|_| panic!("{r}"));
        assert_eq!(v["upload"], json!({"mediaItemId": "M1", "description": "Robots"}), "{v}");
        assert_eq!((v["result"]["bytes"].as_u64(), v["path"].as_str()), (Some(10), Some(path.as_str())), "{v}");
        let r = c.call(4, "googlephotos_upload", json!({"path": "/no/such/export.mp4"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        // an album needs an uploader that does albums: the reason, not a crash
        let r = c.call(5, "googlephotos_upload", json!({"path": path, "album": "Robots"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("albums are not available"), "{r}");
        assert!(file.exists(), "the uploaded file is left alone");
        std::fs::remove_file(&file).ok();
    }

    /// An upload reports byte progress and stops on `notifications/cancelled`; unlike an
    /// export's partial output, the file being uploaded is the user's and stays.
    #[tokio::test(flavor = "multi_thread")]
    async fn googlephotos_upload_progress_and_cancel() {
        struct Slow;
        impl filmcraft_engine::photos::PhotosUploader for Slow {
            fn ready(&self) -> Result<(), String> {
                Ok(())
            }
            fn upload(&self, _: &str, _: Option<&str>, p: &filmcraft_engine::export::Progress) -> Result<Value, String> {
                use std::sync::atomic::Ordering;
                for i in 1..=4000u64 {
                    if p.cancel.load(Ordering::Relaxed) {
                        return Err("cancelled".into());
                    }
                    p.done.store(i.min(p.total.load(Ordering::Relaxed)), Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err("never cancelled".into())
            }
        }
        let file = std::env::temp_dir().join(format!("fc-mcp-upload-cancel-{}.mp4", std::process::id()));
        std::fs::write(&file, vec![1u8; 100_000]).unwrap();
        let mut s = Session::default();
        s.photos = Some(std::sync::Arc::new(Slow));
        let mut c = Client::start(s);
        c.init().await;
        let call = json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"googlephotos_upload","arguments":{"path":file.to_string_lossy()},"_meta":{"progressToken":"up"}}});
        c.send(&call.to_string()).await;
        loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "finished before the cancel: {m}");
            if m["method"] == "notifications/progress" {
                assert_eq!((m["params"]["progressToken"].as_str(), m["params"]["total"].as_f64()), (Some("up"), Some(100_000.0)), "{m}");
                break;
            }
        }
        c.send(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":4}}"#).await;
        let t0 = std::time::Instant::now();
        let last = loop {
            assert!(t0.elapsed().as_secs() < 30, "the upload never stopped");
            c.send(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"jobs.list"}}}"#).await;
            let m = loop {
                let m = c.next().await;
                assert_ne!(m["id"], 4, "no response for a cancelled request: {m}");
                if m["id"] == 5 {
                    break m;
                }
            };
            let jobs: Value = serde_json::from_str(m["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
            let last = jobs.as_array().and_then(|a| a.last()).cloned().unwrap_or_default();
            if last["finished"] == true {
                break last;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        };
        assert_eq!(last["result"]["error"], "cancelled", "{last}");
        // give a (wrong) partial-file cleanup its chance, then check the file is still there
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(file.exists(), "cancelling an upload must not delete the user's file");
        std::fs::remove_file(&file).ok();
    }

    /// The schema of a tool, offline.
    fn schema(tool: &str) -> Value {
        let m = FilmcraftMcp::headless(Session::default());
        let Some(t) = m.tool_router.get(tool) else { panic!("{tool} missing") };
        Value::Object((*t.input_schema).clone())
    }

    fn props(schema: &Value) -> Vec<String> {
        let mut p: Vec<String> = schema["properties"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        p.sort();
        p
    }

    /// The vertical-video tools' arguments and the engine jobs they run.
    #[test]
    fn shorts_and_reframe_tool_params() {
        let s = schema("shorts_make");
        assert!(props(&s).contains(&"upload_youtube".to_string()) && props(&s).contains(&"count".to_string()), "{s}");
        let p: ShortsParams =
            serde_json::from_value(json!({"item": 4, "count": 2, "export": "/x", "upload_youtube": true, "youtube_privacy": "unlisted", "topic": " "}))
                .unwrap();
        assert_eq!(shorts_params(p), json!({"item": 4, "count": 2, "export": "/x", "uploadYoutube": true, "youtube": {"privacy": "unlisted"}, "wait": true}));
        let p: ShortsParams = serde_json::from_value(json!({"platform": "tiktok", "fallback": "letterbox"})).unwrap();
        assert_eq!(shorts_params(p), json!({"platform": "tiktok", "fallback": "letterbox", "wait": true}));
        let p: AutoReframeParams =
            serde_json::from_value(json!({"aspect": "1:1", "start_seconds": 2.0, "end_seconds": 9.5, "subject": [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]}))
                .unwrap();
        assert_eq!(
            auto_reframe_params(p),
            json!({"aspect": "1:1", "range": {"startSeconds": 2.0, "endSeconds": 9.5}, "subject": [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}], "wait": true})
        );
        assert!(crate::long_job::is_long("autocut.shorts", &json!({"wait": true})) && crate::long_job::is_long("sequence.autoReframe", &json!({"wait": true})));
    }

    /// The trailer tool's arguments and the engine job it runs.
    #[test]
    fn trailer_tool_params() {
        let s = schema("trailer_make");
        for k in ["plan_path", "report_path", "hook", "robots", "exclude", "logo", "channel", "export"] {
            assert!(props(&s).contains(&k.to_string()), "{k} in {s}");
        }
        let p: TrailerParams = serde_json::from_value(json!({
            "sequence": 9755, "plan_path": "/v5/plan.json", "report_path": "/v5/build.json", "seconds": 34, "robots": 14,
            "hook": "Ameca, again", "exclude": ["Promobot"], "hook_voice": false, "channel": "@H4hacks", "redact": "off", "title": " ", "export": "/m/reel.mp4"
        }))
        .unwrap();
        assert_eq!(
            trailer_params(p),
            json!({"sequence": 9755, "planPath": "/v5/plan.json", "reportPath": "/v5/build.json", "seconds": 34.0, "robots": 14, "hook": "Ameca, again",
                   "exclude": ["Promobot"], "hookVoice": false, "channel": "@H4hacks", "redact": false, "export": "/m/reel.mp4", "wait": true})
        );
        let p: TrailerParams = serde_json::from_value(json!({"redact": "faces"})).unwrap();
        assert_eq!(trailer_params(p), json!({"redact": "faces", "wait": true}));
        assert!(crate::long_job::is_long("autocut.trailer", &json!({"wait": true})));
    }

    #[test]
    fn auto_redact_tool_params() {
        let s = schema("auto_redact");
        assert!(props(&s).contains(&"dry_run".to_string()) && props(&s).contains(&"exclude".to_string()), "{s}");
        let p: AutoRedactParams =
            serde_json::from_value(json!({"clips": [3, 4], "kinds": ["faces"], "min_face_height": 0.1, "dry_run": true, "sequence": 9})).unwrap();
        assert_eq!(
            auto_redact_params(p),
            ("clip.autoRedact", json!({"clips": [3, 4], "kinds": ["faces"], "minFaceHeight": 0.1, "dryRun": true, "wait": true}))
        );
        let p: AutoRedactParams =
            serde_json::from_value(json!({"clips": [1], "kinds": [], "targets": ["DP World logo", {"text": "a sign", "clips": [1]}], "target_every": 0.5}))
                .unwrap();
        assert_eq!(
            auto_redact_params(p),
            (
                "clip.autoRedact",
                json!({"clips": [1], "kinds": [], "targets": ["DP World logo", {"text": "a sign", "clips": [1]}], "targetEvery": 0.5, "wait": true})
            )
        );
        let p: AutoRedactParams = serde_json::from_value(
            json!({"sequence": 9, "style": "mosaic", "verify": false, "exclude": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}], "profiles": false}),
        )
        .unwrap();
        assert_eq!(
            auto_redact_params(p),
            (
                "sequence.autoRedact",
                json!({"sequence": 9, "style": "mosaic", "verify": false, "exclude": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}], "profiles": false, "wait": true})
            )
        );
        assert!(props(&s).contains(&"profiles".to_string()), "{s}");
        assert_eq!(auto_redact_params(AutoRedactParams::default()), ("sequence.autoRedact", json!({"wait": true})));
        for id in ["clip.autoRedact", "sequence.autoRedact", "redact.downloadModels"] {
            assert!(crate::long_job::is_long(id, &json!({"wait": true})), "{id}");
        }
        assert!(crate::long_job::JOB_EXTRAS.contains(&"redact"));
    }

    /// The showcase tools' arguments and the engine jobs they run (as long jobs: progress, cancel).
    #[test]
    fn showcase_tool_schemas_and_params() {
        let s = schema("showcase_analyze");
        assert_eq!(
            props(&s),
            ["album", "compare", "frames", "fresh", "items", "photos", "plan", "prompt", "screen", "since", "subject", "title", "voice"],
            "{s}"
        );
        assert!(s["required"].as_array().is_none_or(|r| r.is_empty()), "a subject or a prompt: {s}");
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "since": "2020", "items": [5, 6]})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.analyze", json!({"subject": "robot", "since": "2020", "items": [5, 6], "wait": true})));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "plan": true, "title": "Robots", "voice": ""})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.makeShowcase", json!({"subject": "robot", "title": "Robots", "voice": "", "wait": true})));
        // title and voice only matter with a plan
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "title": "Robots"})).unwrap();
        assert!(showcase_analyze_params(p).1.get("title").is_none());
        // a Google Photos album as the seed, with the video asked for in words
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"prompt": "every robot since 2020", "album": "Robot showcase", "subject": " "})).unwrap();
        assert_eq!(showcase_analyze_params(p), ("autocut.analyze", json!({"prompt": "every robot since 2020", "album": "Robot showcase", "wait": true})));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "fresh": true})).unwrap();
        assert_eq!(showcase_analyze_params(p).1, json!({"subject": "robot", "fresh": true, "wait": true}));
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"subject": "robot", "fresh": false})).unwrap();
        assert!(showcase_analyze_params(p).1.get("fresh").is_none());
        // videos only: the album's photos left out
        let p: ShowcaseAnalyzeParams = serde_json::from_value(json!({"album": "Robots", "subject": "robot", "photos": false})).unwrap();
        assert_eq!(showcase_analyze_params(p).1, json!({"album": "Robots", "subject": "robot", "photos": false, "wait": true}));

        let s = schema("showcase_plan");
        assert_eq!(props(&s), ["bites", "entries", "min_coverage", "mood", "narrate", "subtitle", "title", "voice"], "{s}");
        assert!(s["required"].as_array().is_none_or(|r| r.is_empty()), "{s}");
        let p: ShowcasePlanParams = serde_json::from_value(json!({"title": "T", "min_coverage": 0.5, "entries": [{"label": "x"}]})).unwrap();
        assert_eq!(showcase_plan_params(p), json!({"title": "T", "minCoverage": 0.5, "entries": [{"label": "x"}], "wait": true}));
        // a long, narrated show
        let p: ShowcasePlanParams = serde_json::from_value(json!({"narrate": 0.5, "bites": true})).unwrap();
        assert_eq!(showcase_plan_params(p), json!({"narrate": 0.5, "bites": true, "wait": true}));

        let s = schema("showcase_build");
        assert_eq!(props(&s), ["check_pauses", "export", "plan", "qa", "upload", "upload_album"], "{s}");
        let p: ShowcaseBuildParams = serde_json::from_value(json!({"export": "/x/robots.mp4", "check_pauses": false})).unwrap();
        assert_eq!(showcase_build_params(p), json!({"export": "/x/robots.mp4", "checkPauses": false, "upload": false, "wait": true}));
        let p: ShowcaseBuildParams = serde_json::from_value(json!({"export": "/x/robots.mp4", "upload": true})).unwrap();
        assert_eq!(showcase_build_params(p), json!({"export": "/x/robots.mp4", "upload": true, "wait": true}));
        let p: ShowcaseBuildParams = serde_json::from_value(json!({"export": "/x/robots.mp4", "upload": true, "upload_album": "Showcases"})).unwrap();
        assert_eq!(showcase_build_params(p), json!({"export": "/x/robots.mp4", "upload": true, "uploadAlbum": "Showcases", "wait": true}));
        assert_eq!(showcase_build_params(ShowcaseBuildParams::default()), json!({"upload": false, "wait": true}));

        assert_eq!(props(&schema("showcase_inspect")), ["report"]);
        for id in ["autocut.analyze", "autocut.showcasePlan", "autocut.makeShowcase", "autocut.showcaseBuild"] {
            assert!(crate::long_job::is_long(id, &json!({"wait": true})), "{id}");
        }
        assert!(crate::long_job::JOB_EXTRAS.contains(&"showcase"));
    }

    /// A demo session that runs showcase jobs with scripted models.
    fn showcase_demo() -> Session {
        let mut s = demo();
        let host = filmcraft_autocut::host::ShowcaseHost::with_models(filmcraft_autocut::host::Models::scripted());
        s.autocut.host = Some(std::sync::Arc::new(host));
        s
    }

    fn tool_json(r: &Value) -> Value {
        let text = r["result"]["content"][0]["text"].as_str().unwrap_or_default();
        serde_json::from_str(text).unwrap_or_else(|_| panic!("{r}"))
    }

    /// Analyse, plan, review and build over MCP: the jobs run in the background (the port's pump
    /// thread runs their commands) while the server reports progress and keeps answering.
    #[tokio::test(flavor = "multi_thread")]
    async fn showcase_tools_analyze_plan_and_build() {
        let mut c = Client::start(showcase_demo());
        c.init().await;
        let call = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"showcase_analyze","arguments":{"subject":"robot","plan":true,"title":"Robots","voice":""},"_meta":{"progressToken":"sc"}}});
        c.send(&call.to_string()).await;
        let (mut notes, mut pinged, mut pong) = (0, false, false);
        let r = loop {
            let m = c.next().await;
            if m["method"] == "notifications/progress" {
                notes += 1;
                if !pinged {
                    pinged = true;
                    c.send(&json!({"jsonrpc": "2.0", "id": 99, "method": "ping"}).to_string()).await;
                }
            } else if m["id"] == 99 {
                pong = true;
            } else if m["id"] == 1 {
                break m;
            }
        };
        assert_eq!(r["result"]["isError"], false, "{r}");
        let v = tool_json(&r);
        let plan = v["showcase"]["plan"].clone();
        assert_eq!(plan["entries"].as_array().map(Vec::len), Some(6), "{v}");
        assert!(notes >= 1, "progress was reported");
        if pinged && !pong {
            pong = c.next().await["id"] == 99;
        }
        assert!(!pinged || pong, "the server answered while the job ran");

        let r = c.call(2, "showcase_inspect", json!({})).await;
        let v = tool_json(&r);
        assert_eq!((v["report"]["showing"].as_u64(), v["planRevision"].as_u64()), (Some(6), Some(1)), "{v}");

        // review: keep two robots, write the words for them
        let mut entries = plan["entries"].as_array().cloned().unwrap();
        for e in entries.iter_mut().skip(2) {
            e["include"] = json!(false);
        }
        let r = c.call(3, "showcase_plan", json!({"entries": entries})).await;
        let v = tool_json(&r);
        let mut plan = v["showcase"]["plan"].clone();
        assert_eq!(plan["title"], "Robots", "{v}");
        plan["width"] = json!(320);
        plan["height"] = json!(180);
        let r = c.call(4, "showcase_build", json!({"plan": plan})).await;
        let v = tool_json(&r);
        assert_eq!(v["showcase"]["built"]["entries"], 2, "{v}");
        let r = c.call(5, "showcase_analyze", json!({"subject": " "})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        // an album needs a host that picks albums: this one says so before any job starts
        let r = c.call(7, "showcase_analyze", json!({"prompt": "a video of every robot", "album": "Robot showcase"})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("album") && r.to_string().contains("not available"), "{r}");
        let r = c.call(6, "showcase_build", json!({"plan": {"entries": []}})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(r.to_string().contains("no entry is included"), "{r}");
    }

    /// The AutoCut prompt "make a video of every robot" starts the showcase job over MCP too,
    /// and its plan comes back in the report.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_showcase_prompt_over_mcp() {
        let m = FilmcraftMcp::headless(showcase_demo());
        let r = m.autocut_edit(Parameters(AutocutParams { prompt: "make a video of every robot".into(), frames: None, dry_run: None })).await.unwrap();
        assert_ne!(r.is_error, Some(true), "{r:?}");
        let text = serde_json::to_value(&r.content).unwrap()[0]["text"].as_str().unwrap_or_default().to_string();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["intent"], "showcase", "{v}");
        assert_eq!(v["results"][0]["result"]["showcase"]["plan"]["entries"].as_array().map(Vec::len), Some(6), "{v}");
    }

    /// #499: `media_import` takes `paths` (a required array), and forwards `bin` / `image_sequence` to `file.import`.
    #[test]
    fn media_import_schema_and_params() {
        let m = FilmcraftMcp::headless(Session::default());
        let def = m.tool_router.get("media_import").unwrap();
        let schema = Value::Object((*def.input_schema).clone());
        assert_eq!(schema["properties"]["paths"]["type"], "array", "{schema}");
        assert!(schema["required"].as_array().unwrap().iter().any(|r| r == "paths"), "{schema}");
        let p: MediaImportParams = serde_json::from_value(json!({"paths": ["/a.png"], "bin": 7, "image_sequence": true})).unwrap();
        assert_eq!(media_import_params(p), json!({"paths": ["/a.png"], "bin": 7, "imageSequence": true}));
        let p: MediaImportParams = serde_json::from_value(json!({"paths": ["/a.wav"]})).unwrap();
        assert_eq!(media_import_params(p), json!({"paths": ["/a.wav"]}));
        assert!(serde_json::from_value::<MediaImportParams>(json!({"text": "/a.wav"})).is_err());
    }

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    /// Every listed tool has a title and the four hints, and the core tools are listed.
    #[tokio::test(flavor = "multi_thread")]
    async fn tools_are_titled_and_annotated() {
        let mut c = Client::start(demo());
        let init = c.init().await;
        assert_eq!(init["result"]["serverInfo"]["name"], "filmcraft");
        let tools = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})).await["result"]["tools"].as_array().cloned().unwrap();
        for t in &tools {
            assert!(t["title"].is_string(), "{} has no title", t["name"]);
            for h in ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] {
                assert!(t["annotations"][h].is_boolean(), "{} lacks {h}", t["name"]);
            }
        }
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        for want in [
            "command_list",
            "command_run",
            "command_batch",
            "doc_inspect",
            "render_preview",
            "project_inspect",
            "render_frame",
            "ui_click",
            "autocut_plan",
            "autocut_edit",
            "showcase_analyze",
            "showcase_plan",
            "showcase_build",
            "showcase_inspect",
            "shorts_make",
            "trailer_make",
            "speech_understand",
            "rough_cut",
            "quick_short",
            "autocut_direct",
            "sequence_auto_reframe",
            "auto_redact",
            "isolate_voice",
            "enhance_voice",
            "enhance_audio_file",
            "template_list",
            "template_apply",
            "recipe_list",
            "recipe_run",
            "feedback_bundle",
        ] {
            assert!(names.contains(&want), "{want} not listed");
        }
        let ro = |n: &str| tools.iter().find(|t| t["name"] == n).map(|t| t["annotations"]["readOnlyHint"].clone());
        assert_eq!((ro("command_list"), ro("command_run")), (Some(json!(true)), Some(json!(false))));
    }

    /// Strict arguments, malformed JSON, batch, doc_inspect and render_preview over the raw line
    /// protocol.
    #[tokio::test(flavor = "multi_thread")]
    async fn strict_arguments_and_parse_errors() {
        let mut c = Client::start(demo());
        c.init().await;
        let r = c.call(1, "command_list", json!({"filter": "razor", "enabled_only": true})).await;
        assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("timeline.razor"), "{r}");
        // an unknown argument names itself and the accepted ones
        let r = c.call(2, "command_list", json!({"filtr": "x"})).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        let m = r["error"]["message"].as_str().unwrap();
        assert!(m.contains("\"filtr\"") && m.contains("filter") && m.contains("enabled_only"), "{m}");
        let r = c.call(3, "doc_inspect", json!({"verbose": true})).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        // command params stay free-form
        let r = c.call(4, "command_run", json!({"id": "playhead.set", "params": {"seconds": 0.5}})).await;
        assert_eq!(r["result"]["isError"], false, "{r}");
        // malformed JSON: a parse error, and the server keeps serving
        c.send("{\"jsonrpc\": \"2.0\", \"id\": 5, BROKEN").await;
        let r = c.next().await;
        assert_eq!((r["error"]["code"].as_i64(), r["id"].is_null()), (Some(-32700), true), "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":6,"method":"ping"})).await;
        assert!(r["result"].is_object(), "{r}");
        // batch: stops at the first failure and reports both
        let r =
            c.call(7, "command_batch", json!({"steps": [{"id": "playhead.set", "params": {"seconds": 0.5}}, {"id": "no.such"}, {"id": "playhead.set"}]})).await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        let v: Value = serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!((v["completed"].as_u64(), v["failed"].as_u64(), v["results"].as_array().map(Vec::len)), (Some(1), Some(1), Some(2)), "{v}");
        let r = c.call(8, "doc_inspect", json!({})).await;
        let v: Value = serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert!(v["project"].is_object() && v["sequence"].is_object(), "{v}");
        let r = c.call(9, "render_preview", json!({"max_side": 64})).await;
        assert_eq!(r["result"]["content"][0]["type"], "image", "{r}");
    }

    /// `filmcraft://document` and `filmcraft://commands` are listed and readable as JSON.
    #[tokio::test(flavor = "multi_thread")]
    async fn resources_document_and_commands() {
        let mut c = Client::start(demo());
        let init = c.init().await;
        assert!(init["result"]["capabilities"]["resources"].is_object(), "{init}");
        assert!(init["result"]["instructions"].as_str().unwrap().contains("doc_inspect"), "{init}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"resources/list"})).await;
        let uris: Vec<&str> = r["result"]["resources"].as_array().unwrap().iter().filter_map(|x| x["uri"].as_str()).collect();
        assert_eq!(uris, [DOCUMENT_URI, COMMANDS_URI], "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":DOCUMENT_URI}})).await;
        assert_eq!(r["result"]["contents"][0]["mimeType"], "application/json", "{r}");
        let v: Value = serde_json::from_str(r["result"]["contents"][0]["text"].as_str().unwrap()).unwrap();
        assert!(v["project"].is_object() && v["sequence"].is_object(), "{v}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":COMMANDS_URI}})).await;
        assert!(r["result"]["contents"][0]["text"].as_str().unwrap().contains("timeline.razor"), "{r}");
        let r = c.ask(json!({"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"filmcraft://nope"}})).await;
        assert!(r["error"]["code"].is_i64(), "{r}");
    }

    /// MCP 2026-07-28 clients (current Claude Code) negotiate per request and reject list and read
    /// results without `resultType`, `ttlMs` and `cacheScope`; sessions that negotiated an older
    /// revision through `initialize` get the old shape.
    #[tokio::test(flavor = "multi_thread")]
    async fn modern_clients_get_result_type_and_cache_hints() {
        let mut c = Client::start(Session::default());
        let meta = json!({
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        });
        let r = c.ask(json!({"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":meta}})).await;
        assert!(r["result"]["supportedVersions"].as_array().unwrap().contains(&json!("2026-07-28")), "{r}");
        for (id, method, params) in [
            (2, "tools/list", json!({"_meta": meta})),
            (3, "resources/list", json!({"_meta": meta})),
            (4, "resources/read", json!({"_meta": meta, "uri": DOCUMENT_URI})),
        ] {
            let r = c.ask(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})).await;
            assert_eq!(r["result"]["resultType"], "complete", "{method}: {r}");
            assert!(r["result"]["ttlMs"].is_u64(), "{method}: {r}");
            assert!(matches!(r["result"]["cacheScope"].as_str(), Some("public" | "private")), "{method}: {r}");
        }
        let mut c = Client::start(Session::default());
        c.init().await;
        for (id, method, params) in [(2, "tools/list", json!({})), (3, "resources/list", json!({})), (4, "resources/read", json!({"uri": DOCUMENT_URI}))] {
            let r = c.ask(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})).await;
            assert!(r["result"].is_object() && r["result"].get("resultType").is_none() && r["result"].get("ttlMs").is_none(), "{method}: {r}");
        }
    }

    /// docs/agents.md § Long exports: a blocking export reports `notifications/progress` for its
    /// token (none without one), answers `ping` meanwhile, and `notifications/cancelled` stops
    /// it, deletes the partial file and sends no response.
    #[tokio::test(flavor = "multi_thread")]
    async fn export_progress_and_cancel() {
        let dir = std::env::temp_dir().join(format!("filmcraft-mcp-progress-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut c = Client::start(demo());
        c.init().await;
        let export = |id: u64, name: &str, token: Option<Value>, seconds: f64| {
            let path = dir.join(name).to_string_lossy().to_string();
            let params = json!({"path": path, "format": "h264", "width": 640, "height": 360, "audio": false, "range": "custom", "startSeconds": 0, "endSeconds": seconds, "wait": true});
            let mut call =
                json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"file.exportMedia","params":params}}});
            if let Some(t) = token {
                call["params"]["_meta"] = json!({"progressToken": t});
            }
            call.to_string()
        };
        // no token: no notifications, the same response as before
        c.send(&export(1, "quiet.mp4", None, 0.5)).await;
        let m = c.next().await;
        assert_eq!((m["id"].as_u64(), &m["result"]["isError"]), (Some(1), &json!(false)), "{m}");
        // progress, strictly increasing, with a total; ping answered while the export runs. The
        // job is polled every 100 ms, so a fast machine finishes a short export within one poll:
        // a longer one is tried until the export outlasts a few polls.
        let (mut notes, mut pong) = (0, false);
        for (attempt, seconds) in [1.0, 4.0, 16.0].into_iter().enumerate() {
            let (id, ping, name) = (20 + attempt as u64, 30 + attempt as u64, format!("a{attempt}.mp4"));
            c.send(&export(id, &name, Some(json!("tok")), seconds)).await;
            let (mut last, mut pinged) = (-1.0, false);
            (notes, pong) = (0, false);
            let done = loop {
                let m = c.next().await;
                if m["method"] == "notifications/progress" {
                    assert_eq!(m["params"]["progressToken"], "tok", "{m}");
                    let p = m["params"]["progress"].as_f64().unwrap();
                    assert!(p > last && m["params"]["total"].as_f64().unwrap() >= p, "{m}");
                    last = p;
                    notes += 1;
                    if !pinged {
                        pinged = true;
                        c.send(&json!({"jsonrpc": "2.0", "id": ping, "method": "ping"}).to_string()).await;
                    }
                } else if m["id"] == ping {
                    pong = true;
                } else if m["id"] == id {
                    break m;
                }
            };
            assert_eq!(done["result"]["isError"], false, "{done}");
            assert!(notes >= 1, "a finished export reports its progress at least once");
            assert!(dir.join(&name).exists());
            if notes >= 2 && pong {
                break;
            }
            // the ping's answer is still on its way: read it so it cannot be taken for a later reply
            while pinged && !pong {
                pong = c.next().await["id"] == ping;
            }
            pong = false;
        }
        assert!(notes >= 2 && pong, "{notes} notifications, ping answered during the export: {pong}");
        // cancel after the first notification: no response, no partial file
        c.send(&export(4, "b.mp4", Some(json!(7)), 20.0)).await;
        loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "finished before the cancel: {m}");
            if m["method"] == "notifications/progress" {
                break;
            }
        }
        c.send(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":4}}"#).await;
        let t0 = std::time::Instant::now();
        while dir.join("b.mp4").exists() || t0.elapsed() < std::time::Duration::from_millis(500) {
            assert!(t0.elapsed().as_secs() < 30, "partial file still there");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        c.send(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"jobs.list"}}}"#).await;
        let m = loop {
            let m = c.next().await;
            assert_ne!(m["id"], 4, "no response for a cancelled request: {m}");
            if m["id"] == 5 {
                break m;
            }
        };
        let jobs: Value = serde_json::from_str(m["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let last = jobs.as_array().and_then(|a| a.last()).cloned().unwrap_or_default();
        assert_eq!(last["result"]["error"], "cancelled", "{jobs}");
        assert!(!dir.join("b.mp4").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A command that panicked used to poison the session lock, and every later call failed with
    /// "session lock poisoned" until the server was restarted.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_poisoned_session_lock_keeps_serving() {
        let m = FilmcraftMcp::headless(demo());
        let Backend::Headless(s) = &*m.backend else { panic!("headless backend") };
        let s = s.clone();
        let _ = std::thread::spawn(move || {
            let _g = s.lock().unwrap();
            panic!("a command panicked");
        })
        .join();
        let v = m.run("sequence.inspect", json!({})).await.unwrap();
        assert!(v.is_object(), "{v}");
        let r = m.render_frame(Parameters(RenderParams { seconds: Some(0.5), max_side: Some(64) })).await.unwrap();
        assert_ne!(r.is_error, Some(true));
    }

    /// `doc_inspect` composes `project.inspect` and `sequence.inspect` as two separate round
    /// trips to the app. If the active sequence changes between them (another client, or the
    /// app's own UI, switching tabs in that window), an unscoped `sequence.inspect` answers for
    /// whichever sequence is active *now*, not the one `project.inspect` just reported, so the
    /// pair comes back inconsistent (#1477, reproduced with a real bridge client and a flipping
    /// `sequence.open` in `race_test.py`: ~300/300 calls mismatched). Pinning the second call to
    /// the id the first one reported keeps `sequence`, when not null, always describing the
    /// sequence `project.activeSequence` names.
    #[tokio::test]
    async fn doc_inspect_pins_the_sequence_to_the_reported_active_id() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (s, _) = listener.accept().await.unwrap();
            let (r, mut w) = s.into_split();
            let mut lines = tokio::io::BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let req: Value = serde_json::from_str(&line).unwrap();
                let result = match req["params"]["command"].as_str().unwrap_or_default() {
                    "project.inspect" => json!({"activeSequence": 21}),
                    // The app's own active sequence has since moved on to 54; only an explicit
                    // `item` still gets 21's data (an unscoped call would wrongly answer for 54).
                    "sequence.inspect" => json!({"id": req["params"]["params"]["item"].as_i64().unwrap_or(54)}),
                    _ => Value::Null,
                };
                let line = format!("{}\n", json!({"id": req["id"], "ok": true, "result": result}));
                w.write_all(line.as_bytes()).await.unwrap();
            }
        });
        let m = FilmcraftMcp::bridge(&addr).unwrap();
        let v = m.document().await.unwrap();
        assert_eq!(v["project"]["activeSequence"], json!(21), "{v}");
        assert_eq!(v["sequence"]["id"], json!(21), "sequence must describe the reported active id, not whatever is active now: {v}");
    }

    /// `activeSequence` is a `u64` (`ItemId`'s representation; `item_p` parses the `item` param
    /// with `u64_p`), so an id above `i64::MAX` must still round-trip through `document()`:
    /// reading it with `as_i64` would silently return `None` and make `sequence` go null instead
    /// of inspecting the (valid) id project.inspect reported.
    #[tokio::test]
    async fn doc_inspect_handles_an_active_sequence_id_above_i64_max() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        const HUGE: u64 = i64::MAX as u64 + 1;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (s, _) = listener.accept().await.unwrap();
            let (r, mut w) = s.into_split();
            let mut lines = tokio::io::BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let req: Value = serde_json::from_str(&line).unwrap();
                let result = match req["params"]["command"].as_str().unwrap_or_default() {
                    "project.inspect" => json!({"activeSequence": HUGE}),
                    // Echoes back whatever `item` it was actually asked for, so the test can
                    // check that `document()` asked for `HUGE` rather than nothing at all.
                    "sequence.inspect" => json!({"id": req["params"]["params"]["item"]}),
                    _ => Value::Null,
                };
                let line = format!("{}\n", json!({"id": req["id"], "ok": true, "result": result}));
                w.write_all(line.as_bytes()).await.unwrap();
            }
        });
        let m = FilmcraftMcp::bridge(&addr).unwrap();
        let v = m.document().await.unwrap();
        assert_eq!(v["project"]["activeSequence"].as_u64(), Some(HUGE), "{v}");
        assert_eq!(v["sequence"]["id"].as_u64(), Some(HUGE), "an id above i64::MAX must still be inspected, not dropped to a null sequence: {v}");
    }
}
