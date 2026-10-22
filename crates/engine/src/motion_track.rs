//! Motion tracking (CapCut-style): follow an object through a clip, keep the track in the
//! project ([`filmcraft_project::motion_track`]) and make other things follow it.
//!
//! | command | does |
//! |---|---|
//! | `track.create` | job: track the object in `box` (`[x, y, w, h]`, the top-left corner and size as shares of the clip's picture; with `space: "frame"` of the sequence frame as shown) drawn at `at` (sequence seconds, default the playhead) on `clip` (default the selected video clip, else the topmost one under the playhead), `direction` `forward` / `backward` / `both` (default) through the media the clip shows, `method` `subject` (default: the whole object, Median Flow with template re-detection) or `points` (a flat surface: KLT corners and a RANSAC similarity — position, scale and rotation); one undo step ("Track Object") when it finishes |
//! | `track.list` | the project's tracks: clip, frames, the share seen, lost stretches, attachments |
//! | `track.inspect` | one track's samples (media and sequence seconds, the box in the picture and where it shows in the frame), lost stretches and attachments |
//! | `track.delete` | a track (what follows it keeps its keyframes); one undo step |
//! | `track.refine` | job: re-seed from a corrected `box` at `at` and re-run from there (away from the seed, or `direction`), the samples on the other side kept; every attachment re-written; one undo step ("Refine Track") |
//! | `track.attach` | make `target` follow the track — `{clip}` (a sticker, picture or graphic clip's Motion), `{clip, layer}` (a graphic layer: index or uid), `{clip, effect, param}` (an effect's point, e.g. a blur, mosaic, lens flare or light centre) or `{clip, effect, mask}` (a mask path) — with `follow` `position` (default), `positionScale` or `positionScaleRotation` and `offset` `"keep"` (default: where it is now), `"center"` or `[dx, dy]` sequence pixels; one undo step |
//! | `track.detach` | stop an attachment (`clear: true` also removes the keyframes it wrote) |
//! | `track.rebake` | re-write a track's attachments (after the clips were trimmed, moved or sped up) |
//! | `track.blurObject` | track (as `track.create`), then a blurred (`style: "blur"`) or pixelated (`"mosaic"`) mask follows the object: privacy for one object or logo; one undo step ("Blur Object") |
//! | `track.label` | track, then a callout follows the object: `text` on a graphic clip over it with a leader line and a dot (`style` `callout` / `tag` / `plain`, `side` `auto` — where it stays in the frame the longest as the object moves — / `above` / `right` / `left` / `below`); one undo step ("Label Object") |
//! | `track.spotlight` | track, then the surroundings are darkened (`dim` %) outside a soft ellipse following the object; one undo step ("Spotlight Object") |
//! | `track.add` | the same on a track already made (`track`): `what` `label` (with `text`, `style`, `side`, `size`, `accent`), `blur`, `mosaic` or `spotlight` (with `dim`); one undo step |
//!
//! **Baked keyframes, linked.** An attachment writes ordinary keyframes on its target in the
//! target's media time (one a sequence frame where the track and the target overlap, reduced with
//! Ramer–Douglas–Peucker to within half a pixel) and keeps an [`Attachment`] record on the track.
//! The renderer, Effect Controls, the graph editor, export, interchange and the web build need
//! nothing new and the result can be hand-tweaked; the record keeps it linked: `track.refine`
//! re-writes every attachment from the new samples, and `track.rebake` does after edits to the
//! clips. (A live link evaluated at render time would have to reach every evaluator — Motion,
//! graphic layers, effect points, masks, the GPU path and the preview cache keys — and would show
//! values in Effect Controls that are not there.)
//!
//! **Time and space.** The samples are in the tracked clip's media time and source picture; each
//! sequence frame `t` the clip shows media time `clip.source_time_at(t)` (speed, reverse, trims and
//! frame holds included) and the object's centre maps through the clip's Motion
//! (`filmcraft_render::motion_matrix`) into sequence pixels, its rotation and scale adding the
//! Motion's. A target's value is that point (plus the offset, turning and growing with the object
//! when it follows rotation and scale) mapped into the target's own space: Motion Position is in
//! sequence pixels; a graphic layer's position and an effect's point in the target clip's pixels
//! (the inverse of its Motion); a mask path likewise. Keyframes go at the target's
//! `effect_time_at(t)`.
//!
//! Tracking runs in a background job (`jobs.list` shows its progress; `jobs.cancel` stops it and
//! nothing changes); `wait: true` finishes it before the command answers. Every parameter is
//! checked: a box outside the picture, of no size or NaN, a clip with no picture, an unknown
//! track or target are errors, never a panic.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::motion_track::{MAX_ATTACHMENTS, MAX_NAME_CHARS};
use filmcraft_project::{
    AttachTarget, Attachment, ClipId, EffectInstance, FollowMode, ItemId, ItemKind, Mask, MaskPath, MotionTrack, MotionTrackId, Param, ParamKind, ParamValue,
    Project, Sequence, TrackDirection, TrackItem, TrackKind, TrackSample, TrackingMethod,
};
use filmcraft_render::object_track::{ANALYSIS_SIDE, ObjectTracker};
use filmcraft_render::subject::{SubjectBox, downscale_rgba};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, has_seq, u64_p};
use crate::{EngineError, Result, Session};

/// Most frames one tracking job analyses (a longer clip is analysed at a lower rate).
pub const MAX_FRAMES: usize = 100_000;
/// Fastest analysis rate, frames a second (faster media is analysed at this rate).
const MAX_ANALYSIS_FPS: f64 = 60.0;
/// Most tracks a project keeps.
pub const MAX_TRACKS: usize = 10_000;
/// Most sequence frames one attachment writes.
const MAX_BAKE_FRAMES: usize = 200_000;
/// Finished jobs' outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 256;
/// Longest label text, characters.
const MAX_TEXT_CHARS: usize = 500;
/// How closely baked keyframes reproduce the per-frame values: pixels, percent, degrees.
const TOLERANCE: [f64; 4] = [0.5, 0.5, 0.25, 0.25];
/// Largest offset or coordinate a command takes, pixels.
const MAX_COORD: f64 = 1e6;

const CREATE: &str = "track.create";
const REFINE: &str = "track.refine";
const ATTACH: &str = "track.attach";
const BLUR: &str = "track.blurObject";
const LABEL: &str = "track.label";
const SPOTLIGHT: &str = "track.spotlight";
const ADD: &str = "track.add";

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal: true }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            CREATE,
            "Track Object",
            concat!(
                "{",
                r#""clip":id?,"box":[x,y,w,h]|{"x","y","w","h"},"space":"picture"|"frame"="picture","at":seconds?,"direction":"forward"|"backward"|"both"="both","method":"subject"|"points"="subject","name":str?,"wait":bool=false"#,
                "}"
            ),
            create,
        ),
        CommandSpec {
            id: "track.list",
            label: "List Motion Tracks",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?}"#,
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "track.inspect",
            label: "Inspect Motion Track",
            menu: &[],
            shortcut: None,
            params: r#"{"track":id,"samples":bool=true}"#,
            enabled: always,
            run: inspect,
            journal: false,
        },
        CommandSpec {
            id: "track.delete",
            label: "Delete Motion Track",
            menu: &[],
            shortcut: None,
            params: r#"{"track":id}"#,
            enabled: always,
            run: delete,
            journal: true,
        },
        spec(
            REFINE,
            "Refine Motion Track",
            r#"{"track":id,"box":[x,y,w,h]|{"x","y","w","h"},"space":"picture"|"frame"="picture","at":seconds?,"direction":"forward"|"backward"|"both"?,"wait":bool=false}"#,
            refine,
        ),
        spec(
            ATTACH,
            "Attach to Motion Track",
            r#"{"track":id,"target":{"clip":id}|{"clip":id,"layer":n|uid}|{"clip":id,"effect":index|id,"param":str}|{"clip":id,"effect":index|id?,"mask":n},"follow":"position"|"positionScale"|"positionScaleRotation"="position","offset":"keep"|"center"|[dx,dy]="keep","at":seconds?}"#,
            attach,
        ),
        spec("track.detach", "Detach from Motion Track", r#"{"track":id,"attachment":id,"clear":bool=false}"#, detach),
        spec("track.rebake", "Update Motion Track Attachments", r#"{"track":id}"#, rebake),
        spec(
            ADD,
            "Add to Motion Track",
            r##"{"track":id,"what":"label"|"blur"|"mosaic"|"spotlight","text":str?,"style":str?,"side":"auto"|"above"|"right"|"left"|"below"?,"size":px?,"accent":"#rrggbb"?,"amount":0..100?,"shape":"ellipse"|"rectangle"?,"feather":px?,"dim":0..100?,"at":seconds?}"##,
            add,
        ),
        spec(
            BLUR,
            "Blur Tracked Object",
            concat!(
                "{",
                r#""clip":id?,"box":[x,y,w,h]|{"x","y","w","h"},"space":"picture"|"frame"="picture","at":seconds?,"direction":"forward"|"backward"|"both"="both","method":"subject"|"points"="subject","name":str?,"wait":bool=false"#,
                r#","style":"blur"|"mosaic"="blur","amount":0..100?,"shape":"ellipse"|"rectangle"="ellipse","feather":px?}"#
            ),
            blur_object,
        ),
        spec(
            LABEL,
            "Label Tracked Object",
            concat!(
                "{",
                r#""clip":id?,"box":[x,y,w,h]|{"x","y","w","h"},"space":"picture"|"frame"="picture","at":seconds?,"direction":"forward"|"backward"|"both"="both","method":"subject"|"points"="subject","name":str?,"wait":bool=false"#,
                r##","text":str,"style":"callout"|"tag"|"plain"="callout","side":"auto"|"above"|"right"|"left"|"below"="auto","size":px?,"accent":"#rrggbb"?,"videoTrack":index?,"from":seconds?,"to":seconds?}"##
            ),
            label,
        ),
        spec(
            SPOTLIGHT,
            "Spotlight Tracked Object",
            concat!(
                "{",
                r#""clip":id?,"box":[x,y,w,h]|{"x","y","w","h"},"space":"picture"|"frame"="picture","at":seconds?,"direction":"forward"|"backward"|"both"="both","method":"subject"|"points"="subject","name":str?,"wait":bool=false"#,
                r#","dim":0..100=60,"feather":px?}"#
            ),
            spotlight,
        ),
    ]
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// session state
// ---------------------------------------------------------------------------------------------

type Results = Arc<Mutex<Option<std::result::Result<Vec<TrackSample>, String>>>>;

/// What a finished job does with its samples.
enum Action {
    /// A new track (its samples still empty), then what follows it.
    Create { track: Box<MotionTrack>, then: Then },
    /// New samples for a track from `from` on (`forward`), up to it, or (both) all of them.
    Refine { id: MotionTrackId, from: Tick, side: TrackDirection },
}

/// What follows a new track as soon as it is made.
enum Then {
    Nothing,
    Blur(BlurSpec),
    Label(LabelSpec),
    Spotlight(SpotSpec),
}

struct PendingTrack {
    job: u64,
    cmd: &'static str,
    clip: ClipId,
    action: Action,
    results: Results,
}

/// The session's tracking jobs and what the finished ones made.
#[derive(Default)]
pub struct MotionTracks {
    pending: Vec<PendingTrack>,
    outcomes: HashMap<u64, Value>,
}

impl MotionTracks {
    /// The running tracking jobs on `clip`: their job ids.
    pub fn jobs_on(&self, clip: ClipId) -> Vec<u64> {
        self.pending.iter().filter(|p| p.clip == clip).map(|p| p.job).collect()
    }
    /// Every running tracking job: (job, clip).
    pub fn running(&self) -> Vec<(u64, ClipId)> {
        self.pending.iter().map(|p| (p.job, p.clip)).collect()
    }
}

/// What a finished tracking job made (`track` in its `jobs.list` entry).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.motion_tracks.outcomes.get(&job).cloned()
}

// ---------------------------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------------------------

fn track_id_p(cmd: &str, p: &Value) -> Result<MotionTrackId> {
    u64_p(p, "track").map(MotionTrackId).ok_or_else(|| bad(cmd, "`track` is a motion track id (`track.list`)"))
}

fn get_track(s: &Session, cmd: &str, id: MotionTrackId) -> Result<Arc<MotionTrack>> {
    s.project.motion_tracks.get(&id).cloned().ok_or_else(|| bad(cmd, format!("no motion track {}", id.0)))
}

/// `box`: `[x, y, w, h]` or `{x, y, w, h}`, finite numbers.
fn box_p(cmd: &str, p: &Value) -> Result<[f64; 4]> {
    let v = p.get("box").filter(|v| !v.is_null()).ok_or_else(|| bad(cmd, "need `box`: [x, y, w, h] (the top-left corner and size, 0..1 of the picture)"))?;
    let nums: Option<Vec<f64>> = match v {
        Value::Array(a) if a.len() == 4 => a.iter().map(Value::as_f64).collect(),
        Value::Object(_) => ["x", "y", "w", "h"].iter().map(|k| v.get(*k).and_then(Value::as_f64)).collect(),
        _ => None,
    };
    let nums = nums.filter(|n| n.iter().all(|v| v.is_finite())).ok_or_else(|| bad(cmd, "`box` is [x, y, w, h]: four finite numbers"))?;
    match nums.as_slice() {
        [x, y, w, h] => Ok([*x, *y, *w, *h]),
        _ => Err(bad(cmd, "`box` is [x, y, w, h]")),
    }
}

/// A text parameter: `None` when absent or null, an error when it is not text.
fn opt_str<'a>(cmd: &str, p: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(bad(cmd, format!("`{key}` is text"))),
    }
}

fn method_p(cmd: &str, p: &Value) -> Result<TrackingMethod> {
    match opt_str(cmd, p, "method")? {
        None => Ok(TrackingMethod::Subject),
        Some(m) => TrackingMethod::from_name(m).ok_or_else(|| bad(cmd, format!("`method` is \"subject\" or \"points\", not \"{m}\""))),
    }
}

/// Round to `digits` decimals: samples and offsets are stored rounded (the nearest double to a
/// short decimal), so a project file reads back exactly what was saved, and stays small.
fn round_to(v: f64, digits: i32) -> f64 {
    let k = 10f64.powi(digits);
    let r = (v * k).round() / k;
    if r.is_finite() { r } else { v }
}

fn direction_p(cmd: &str, p: &Value) -> Result<Option<TrackDirection>> {
    match p.get("direction").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(Value::String(d)) => {
            TrackDirection::from_name(d).map(Some).ok_or_else(|| bad(cmd, format!("`direction` is \"forward\", \"backward\" or \"both\", not \"{d}\"")))
        }
        Some(_) => Err(bad(cmd, "`direction` is \"forward\", \"backward\" or \"both\"")),
    }
}

/// `at`: sequence seconds (finite, within ±10⁷ s); `None` when not given.
fn at_p(cmd: &str, p: &Value) -> Result<Option<Tick>> {
    match p.get("at").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            let s = v.as_f64().filter(|s| s.is_finite() && s.abs() <= 1e7).ok_or_else(|| bad(cmd, "`at` is a time in seconds on the sequence"))?;
            Ok(Some(Tick::from_seconds_f64(s)))
        }
    }
}

fn name_p(cmd: &str, p: &Value) -> Result<Option<String>> {
    match p.get("name").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(Value::String(n)) if !n.trim().is_empty() && n.chars().count() <= MAX_NAME_CHARS && !n.chars().any(char::is_control) => {
            Ok(Some(n.trim().to_string()))
        }
        Some(_) => Err(bad(cmd, format!("`name` is a name on one line, at most {MAX_NAME_CHARS} characters"))),
    }
}

/// Whether a clip shows a picture that can be tracked (media with video, or a subclip of it).
fn trackable(project: &Project, it: &TrackItem) -> bool {
    match project.item(it.item).map(|i| &i.kind) {
        Some(ItemKind::Media(m)) => m.info.has_video(),
        Some(ItemKind::Subclip { .. }) => project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.has_video()),
        _ => false,
    }
}

/// The clip to track: `clip`, else the first selected trackable video clip, else the topmost one
/// under the playhead.
fn tracked_clip(s: &Session, cmd: &str, p: &Value) -> Result<(ItemId, TrackItem)> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let on_video = |c: ClipId| q.find_item(c).filter(|(t, _)| q.track(*t).is_some_and(|t| t.kind == TrackKind::Video)).map(|(_, i)| i.clone());
    let it = match p.get("clip").filter(|v| !v.is_null()) {
        Some(v) => {
            let c = ClipId(v.as_u64().ok_or_else(|| bad(cmd, "`clip` is a clip id"))?);
            on_video(c).ok_or_else(|| bad(cmd, "no such video clip in the active sequence"))?
        }
        None => {
            let ph = s.playhead();
            s.state
                .selection
                .iter()
                .filter_map(|c| on_video(*c))
                .find(|i| trackable(&s.project, i))
                .or_else(|| q.video_tracks.iter().rev().filter_map(|t| t.item_at(ph)).find(|i| trackable(&s.project, i)).cloned())
                .ok_or_else(|| bad(cmd, "select a video clip to track (or pass `clip`)"))?
        }
    };
    if !trackable(&s.project, &it) {
        return Err(bad(cmd, "the clip has no picture to track (track a clip of video or still media)"));
    }
    Ok((seq_id, it))
}

/// A box in shares of the clip's picture from `box` and `space`, at sequence time `at`.
fn seed_box(s: &Session, cmd: &str, p: &Value, q: &Sequence, it: &TrackItem, at: Tick) -> Result<TrackSample> {
    let [x, y, w, h] = box_p(cmd, p)?;
    let size = s.project.source_size(it.item).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    let (sw, sh) = (f64::from(size.0.max(1)), f64::from(size.1.max(1)));
    let mt = it.source_time_at(at);
    let corners = match opt_str(cmd, p, "space")?.unwrap_or("picture") {
        "picture" => [x, y, w, h],
        "frame" => {
            // the frame as shown → the clip's pixels through the inverse of its Motion
            let m = filmcraft_render::motion_matrix(q, it, size, s.project.source_par(it.item), it.effect_time_at(at));
            let inv = m.inverse().ok_or_else(|| bad(cmd, "the clip is scaled to nothing: no box can be drawn on it"))?;
            let (fw, fh) = (f64::from(q.settings.width), f64::from(q.settings.height));
            let pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)].map(|(a, b)| inv.apply(Vec2::new(a * fw, b * fh)));
            let (lo, hi) = pts.iter().fold((Vec2::new(f64::INFINITY, f64::INFINITY), Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY)), |(lo, hi), p| {
                (Vec2::new(lo.x.min(p.x), lo.y.min(p.y)), Vec2::new(hi.x.max(p.x), hi.y.max(p.y)))
            });
            [lo.x / sw, lo.y / sh, (hi.x - lo.x) / sw, (hi.y - lo.y) / sh]
        }
        other => return Err(bad(cmd, format!("`space` is \"picture\" or \"frame\", not \"{other}\""))),
    };
    if !(corners[2] > 0.0 && corners[3] > 0.0) {
        return Err(bad(cmd, "`box` has no size: w and h must be above zero"));
    }
    let r = |v: f64| round_to(v, 6);
    TrackSample::from_corner(mt, r(corners[0]), r(corners[1]), r(corners[2]), r(corners[3]))
        .map(|s| TrackSample { cx: r(s.cx), cy: r(s.cy), w: r(s.w), h: r(s.h), ..s })
        .ok_or_else(|| bad(cmd, "`box` is outside the picture (or too small to track)"))
}

/// A tracking request, checked: the clip, the seed and the frames to analyse.
struct Request {
    seq: ItemId,
    clip: TrackItem,
    at: Tick,
    seed: TrackSample,
    method: TrackingMethod,
    direction: TrackDirection,
    name: String,
    src: filmcraft_media::SharedSource,
    scale: f32,
    /// Media times from the seed on, the seed's included.
    forward: Vec<Tick>,
    backward: Vec<Tick>,
}

impl Request {
    fn frames(&self) -> usize {
        self.forward.len() + self.backward.len()
    }
}

/// Media times of the frames to analyse from `mt0` towards `end` (inclusive), `step` apart.
fn steps(mt0: Tick, end: Tick, step: Tick, max: usize) -> Vec<Tick> {
    let mut out = Vec::new();
    let step = step.0.max(1);
    let forward = end >= mt0;
    let more = |t: Tick| if forward { t <= end } else { t >= end };
    let mut t = mt0;
    while out.len() < max && more(t) {
        out.push(t);
        let Some(next) = (if forward { t.0.checked_add(step) } else { t.0.checked_sub(step) }) else { break };
        t = Tick(next);
    }
    out
}

/// The frames a track of `it` from media time `mt0` covers: every frame of the media the clip
/// shows (at most [`MAX_ANALYSIS_FPS`] a second), forward and/or backward.
fn frames_for(project: &Project, it: &TrackItem, mt0: Tick, dir: TrackDirection) -> (Vec<Tick>, Vec<Tick>) {
    let rate = project.resolve_media(it.item).map(|(_, m, _)| m.frame_rate()).unwrap_or_default().sane();
    let fd = rate.frame_duration().max(Tick(1));
    let mut step = Tick(fd.0.saturating_mul((rate.as_f64() / MAX_ANALYSIS_FPS).ceil().max(1.0) as i64));
    let last = it.end() - Tick(1);
    let (a, b) = (it.source_time_at(it.start), it.source_time_at(last.max(it.start)));
    let (lo, hi) = (a.min(b), a.max(b));
    let count = |step: Tick| -> usize {
        let n = |span: i64| usize::try_from(span / step.0.max(1)).unwrap_or(usize::MAX).saturating_add(1);
        (if dir.forward() { n((hi - mt0).0.max(0)) } else { 0 }).saturating_add(if dir.backward() { n((mt0 - lo).0.max(0)) } else { 0 })
    };
    while count(step) > MAX_FRAMES {
        step = Tick(step.0.saturating_mul(2));
    }
    let fwd = if dir.forward() { steps(mt0, hi, step, MAX_FRAMES) } else { Vec::new() };
    let back = if dir.backward() { steps(mt0, lo, step, MAX_FRAMES) } else { Vec::new() };
    (fwd, back)
}

fn request(s: &Session, cmd: &str, p: &Value) -> Result<Request> {
    if !p.is_object() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    if s.project.motion_tracks.len() >= MAX_TRACKS {
        return Err(bad(cmd, format!("the project has {MAX_TRACKS} motion tracks already: delete some first")));
    }
    let (seq, clip) = tracked_clip(s, cmd, p)?;
    let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?;
    let at = match at_p(cmd, p)? {
        Some(t) if t < clip.start || t >= clip.end() => return Err(bad(cmd, "`at` is outside the clip")),
        Some(t) => t,
        None => s.playhead().clamp(clip.start, (clip.end() - Tick(1)).max(clip.start)),
    };
    let seed = seed_box(s, cmd, p, q, &clip, at)?;
    let method = method_p(cmd, p)?;
    let direction = direction_p(cmd, p)?.unwrap_or_default();
    let n = s.project.motion_tracks.values().filter(|t| t.clip == clip.id).count() + 1;
    let name = name_p(cmd, p)?.unwrap_or_else(|| format!("{} track {n}", clip.name));
    let src = s.source(clip.item).ok_or_else(|| EngineError::Other(format!("{cmd}: the clip's media is not available")))?;
    let (w, h) = s.project.source_size(clip.item).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    let scale = (ANALYSIS_SIDE as f64 / f64::from(w.max(h).max(1))).min(1.0) as f32;
    let (forward, backward) = frames_for(&s.project, &clip, seed.t, direction);
    if forward.is_empty() && backward.is_empty() {
        return Err(bad(cmd, "the clip has no frames to track"));
    }
    Ok(Request { seq, clip, at, seed, method, direction, name, src, scale, forward, backward })
}

// ---------------------------------------------------------------------------------------------
// the job
// ---------------------------------------------------------------------------------------------

/// Follow the object from `seed` through `frames` (media times, in tracking order).
fn track_frames(
    src: &filmcraft_media::SharedSource,
    scale: f32,
    method: TrackingMethod,
    seed: &TrackSample,
    frames: &[Tick],
    tick: &mut dyn FnMut() -> bool,
) -> Option<Vec<TrackSample>> {
    let b = SubjectBox::new(seed.cx, seed.cy, seed.w, seed.h)?;
    let mut tr = ObjectTracker::new(method, b);
    let mut out = Vec::with_capacity(frames.len());
    for &mt in frames {
        if !tick() {
            return None;
        }
        let obs = match src.video_frame(filmcraft_media::FrameRequest { time: mt, scale }) {
            Ok(f) => {
                let rgba = f.to_rgba8();
                match downscale_rgba(&rgba, f.width as usize, f.height as usize, ANALYSIS_SIDE) {
                    Some((small, w, h)) => tr.push(&small, w, h, mt.seconds()),
                    None => tr.push(&[], 0, 0, mt.seconds()),
                }
            }
            // a frame that does not decode is a frame where the object is not seen
            Err(_) => tr.push(&[], 0, 0, mt.seconds()),
        };
        let o = obs.bbox;
        let (r6, r4) = (|v: f64| round_to(v, 6), |v: f64| round_to(v, 4));
        let sample = TrackSample {
            t: mt,
            cx: r6(o.cx),
            cy: r6(o.cy),
            w: r6(o.w),
            h: r6(o.h),
            rotation: r4(o.rotation),
            scale: r6(o.scale),
            confidence: (obs.confidence * 1000.0).round() / 1000.0,
            lost: obs.lost,
        };
        out.push(if sample.is_valid() { sample } else { TrackSample { t: mt, lost: true, confidence: 0.0, ..*seed } });
    }
    Some(out)
}

/// Start the tracking job for `req`; `action` says what its samples become.
fn start(s: &mut Session, cmd: &'static str, req: Request, action: Action, p: &Value) -> Result<Value> {
    let total = req.frames();
    let clip = req.clip.id;
    let label = match &action {
        Action::Create { track, .. } => format!("Track {}", track.name),
        Action::Refine { .. } => format!("Refine {}", req.name),
    };
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(total as u64, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let Request { src, scale, method, seed, forward, backward, .. } = req;
    let run = move || {
        let t0 = web_time::Instant::now();
        let mut done = 0u64;
        let mut tick = || {
            if prog.cancel.load(Ordering::Relaxed) {
                return false;
            }
            done += 1;
            prog.done.store(done, Ordering::Relaxed);
            if done.is_multiple_of(8) {
                *lock(&prog.status) = format!("Tracking: frame {done} of {total}");
            }
            true
        };
        let fwd = track_frames(&src, scale, method, &seed, &forward, &mut tick);
        let back = fwd.as_ref().and_then(|_| track_frames(&src, scale, method, &seed, &backward, &mut tick));
        let made = match (fwd, back) {
            (Some(mut f), Some(b)) => {
                // the seed frame once (the forward run's)
                f.extend(b.into_iter().skip(usize::from(!forward.is_empty())));
                f.sort_by_key(|x| x.t);
                f.dedup_by_key(|x| x.t);
                Some(f)
            }
            _ => None,
        };
        let secs = t0.elapsed().as_secs_f64();
        let lost = made.as_ref().map_or(0, |m| m.iter().filter(|x| x.lost).count());
        *lock(&prog.status) = match &made {
            None => "Stopped: nothing was changed".into(),
            Some(m) => format!("Tracked {} frame(s) in {secs:.1}s, {lost} without the object", m.len()),
        };
        *lock(&out) = Some(made.ok_or_else(|| "stopped".to_string()));
        *lock(&res) = Some(Ok(filmcraft_export::Report {
            path: String::new(),
            frames: done,
            seconds: secs,
            bytes: 0,
            render_fps: done as f64 / secs.max(1e-6),
            extra_files: Vec::new(),
        }));
        prog.finished.store(true, Ordering::Relaxed);
    };
    s.motion_tracks.pending.push(PendingTrack { job: id, cmd, clip, action, results });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-motion-track", run)?;
    let mut answer = json!({"job": id, "frames": total});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.motion_tracks.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["track"] = o.clone();
        }
    }
    Ok(answer)
}

/// Apply finished jobs (one undo step each) and keep what they made for `jobs.list`. Called each
/// UI frame (through [`Session::poll_persistence`]), by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.motion_tracks.pending.len() {
        let Some(pj) = s.motion_tracks.pending.get(i) else { break };
        let ready = lock(&pj.results).is_some();
        let job = s.jobs.iter().find(|j| j.id == pj.job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !ready && !finished {
            i += 1;
            continue;
        }
        let pj = s.motion_tracks.pending.remove(i);
        let results = lock(&pj.results).take();
        let outcome = match results {
            Some(Ok(samples)) if !cancelled => match apply(s, pj.action, samples) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(pj.cmd, format!("Motion tracking: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            _ => json!({"error": "stopped: nothing was changed"}),
        };
        s.motion_tracks.outcomes.insert(pj.job, outcome);
        if s.motion_tracks.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.motion_tracks.outcomes.keys().min().copied()
        {
            s.motion_tracks.outcomes.remove(&old);
        }
    }
}

fn apply(s: &mut Session, action: Action, samples: Vec<TrackSample>) -> Result<Value> {
    if samples.is_empty() {
        return Err(EngineError::Other("no frame could be tracked".into()));
    }
    match action {
        Action::Create { mut track, then } => {
            track.samples = samples;
            let label = match then {
                Then::Nothing => "Track Object",
                Then::Blur(_) => "Blur Object",
                Then::Label(_) => "Label Object",
                Then::Spotlight(_) => "Spotlight Object",
            };
            let (id, extra) = s.edit(label, move |pr, st| {
                if pr.sequence(track.sequence).and_then(|q| q.find_item(track.clip)).is_none() {
                    return Err(EngineError::Other("the tracked clip is no longer in its sequence".into()));
                }
                track.id = MotionTrackId(pr.alloc_id());
                let extra = match then {
                    Then::Nothing => Value::Null,
                    Then::Blur(b) => add_blur(pr, &mut track, &b)?,
                    Then::Label(l) => add_label(pr, st, &mut track, &l)?,
                    Then::Spotlight(sp) => add_spotlight(pr, &mut track, &sp)?,
                };
                let id = track.id;
                pr.motion_tracks.insert(id, Arc::new(*track));
                Ok((id, extra))
            })?;
            let mut v = track_json(s, id, false)?;
            if !extra.is_null() {
                v["made"] = extra;
            }
            Ok(v)
        }
        Action::Refine { id, from, side } => {
            s.edit("Refine Track", move |pr, _| {
                let mut t = pr.motion_tracks.get(&id).map(|t| (**t).clone()).ok_or_else(|| EngineError::Other("the track is gone".into()))?;
                match side {
                    TrackDirection::Forward => t.splice(from, true, samples),
                    TrackDirection::Backward => t.splice(from, false, samples),
                    TrackDirection::Both => {
                        t.samples.clear();
                        t.splice(from, true, samples);
                    }
                }
                let attachments = std::mem::take(&mut t.attachments);
                for a in &attachments {
                    bake(pr, &t, a)?;
                }
                t.attachments = attachments;
                pr.motion_tracks.insert(id, Arc::new(t));
                Ok(())
            })?;
            track_json(s, id, false)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------------------------

fn new_track(req: &Request) -> MotionTrack {
    MotionTrack {
        id: MotionTrackId(0),
        name: req.name.clone(),
        item: req.clip.item,
        sequence: req.seq,
        clip: req.clip.id,
        method: req.method,
        direction: req.direction,
        seed: req.seed,
        samples: Vec::new(),
        attachments: Vec::new(),
    }
}

fn create(s: &mut Session, p: &Value) -> Result<Value> {
    let req = request(s, CREATE, p)?;
    let track = Box::new(new_track(&req));
    start(s, CREATE, req, Action::Create { track, then: Then::Nothing }, p)
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let only = u64_p(p, "clip").map(ClipId);
    let ids: Vec<MotionTrackId> = s.project.motion_tracks.values().filter(|t| only.is_none_or(|c| c == t.clip)).map(|t| t.id).collect();
    let tracks: Vec<Value> = ids.into_iter().filter_map(|id| track_json(s, id, false).ok()).collect();
    let running: Vec<Value> = s.motion_tracks.running().into_iter().map(|(job, clip)| json!({"job": job, "clip": clip.0})).collect();
    Ok(json!({"tracks": tracks, "running": running}))
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let id = track_id_p("track.inspect", p)?;
    get_track(s, "track.inspect", id)?;
    track_json(s, id, bool_p(p, "samples").unwrap_or(true))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    let id = track_id_p("track.delete", p)?;
    get_track(s, "track.delete", id)?;
    s.edit("Delete Motion Track", |pr, _| {
        pr.motion_tracks.remove(&id);
        Ok(())
    })?;
    Ok(json!({"deleted": id.0}))
}

fn refine(s: &mut Session, p: &Value) -> Result<Value> {
    let id = track_id_p(REFINE, p)?;
    let t = get_track(s, REFINE, id)?;
    if s.motion_tracks.pending.iter().any(|j| matches!(j.action, Action::Refine { id: r, .. } if r == id)) {
        return Err(bad(REFINE, "this track is being refined already"));
    }
    if s.state.active_sequence != Some(t.sequence) {
        return Err(bad(REFINE, "open the sequence the track was made in first"));
    }
    let mut q = p.clone();
    if let Some(o) = q.as_object_mut() {
        o.insert("clip".into(), json!(t.clip.0));
        o.insert("method".into(), json!(t.method.name()));
        o.remove("direction");
    }
    let mut req = request(s, REFINE, &q)?;
    if req.clip.item != t.item {
        return Err(bad(REFINE, "the tracked clip now shows other media: make a new track"));
    }
    // away from the seed, unless asked
    let side = match direction_p(REFINE, p)? {
        Some(d) => d,
        None if req.seed.t > t.seed.t => TrackDirection::Forward,
        None if req.seed.t < t.seed.t => TrackDirection::Backward,
        None => TrackDirection::Both,
    };
    let (forward, backward) = frames_for(&s.project, &req.clip, req.seed.t, side);
    req.forward = forward;
    req.backward = backward;
    req.name = t.name.clone();
    let from = req.seed.t;
    start(s, REFINE, req, Action::Refine { id, from, side }, p)
}

/// What a command asked to attach.
#[derive(Clone, Debug)]
enum TargetSpec {
    Clip(ClipId),
    Layer(ClipId, Value),
    Effect(ClipId, Value, String),
    Mask(ClipId, Option<Value>, usize),
}

#[derive(Clone, Copy, Debug)]
enum OffsetSpec {
    Keep,
    Center,
    Px(Vec2),
}

fn target_p(cmd: &str, p: &Value) -> Result<TargetSpec> {
    let t = p
        .get("target")
        .filter(|v| v.is_object())
        .ok_or_else(|| bad(cmd, "`target` is {clip}, {clip, layer}, {clip, effect, param} or {clip, effect, mask}"))?;
    let clip = ClipId(t.get("clip").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`target.clip` is a clip id"))?);
    if let Some(m) = t.get("mask").filter(|v| !v.is_null()) {
        let m = m.as_u64().and_then(|m| usize::try_from(m).ok()).ok_or_else(|| bad(cmd, "`target.mask` is a mask index"))?;
        return Ok(TargetSpec::Mask(clip, t.get("effect").filter(|v| !v.is_null()).cloned(), m));
    }
    if let Some(prm) = t.get("param").filter(|v| !v.is_null()) {
        let prm = prm.as_str().filter(|s| !s.is_empty() && s.len() <= 64).ok_or_else(|| bad(cmd, "`target.param` is a parameter id"))?;
        let e = t.get("effect").filter(|v| !v.is_null()).cloned().ok_or_else(|| bad(cmd, "`target.effect` (index or id) goes with `target.param`"))?;
        return Ok(TargetSpec::Effect(clip, e, prm.to_string()));
    }
    if let Some(l) = t.get("layer").filter(|v| !v.is_null()) {
        return Ok(TargetSpec::Layer(clip, l.clone()));
    }
    Ok(TargetSpec::Clip(clip))
}

fn follow_p(cmd: &str, p: &Value) -> Result<FollowMode> {
    match opt_str(cmd, p, "follow")? {
        None => Ok(FollowMode::Position),
        Some(f) => {
            FollowMode::from_name(f).ok_or_else(|| bad(cmd, format!("`follow` is \"position\", \"positionScale\" or \"positionScaleRotation\", not \"{f}\"")))
        }
    }
}

fn offset_p(cmd: &str, p: &Value) -> Result<OffsetSpec> {
    match p.get("offset").filter(|v| !v.is_null()) {
        None => Ok(OffsetSpec::Keep),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("keep") => Ok(OffsetSpec::Keep),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("center") || s.eq_ignore_ascii_case("centre") => Ok(OffsetSpec::Center),
        Some(Value::Array(a)) if a.len() == 2 => {
            let v: Option<Vec<f64>> = a.iter().map(|x| x.as_f64().filter(|x| x.is_finite() && x.abs() <= MAX_COORD)).collect();
            let v = v.ok_or_else(|| bad(cmd, "`offset` [dx, dy] takes finite pixels"))?;
            Ok(OffsetSpec::Px(Vec2::new(v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0))))
        }
        Some(_) => Err(bad(cmd, "`offset` is \"keep\", \"center\" or [dx, dy] in sequence pixels")),
    }
}

fn attach(s: &mut Session, p: &Value) -> Result<Value> {
    let id = track_id_p(ATTACH, p)?;
    let t = get_track(s, ATTACH, id)?;
    if t.attachments.len() >= MAX_ATTACHMENTS {
        return Err(bad(ATTACH, format!("a track drives at most {MAX_ATTACHMENTS} things")));
    }
    let target = target_p(ATTACH, p)?;
    let follow = follow_p(ATTACH, p)?;
    let offset = offset_p(ATTACH, p)?;
    let at = at_p(ATTACH, p)?.unwrap_or_else(|| s.playhead());
    let made = s.edit("Attach to Motion Track", |pr, _| {
        let mut t = (*t).clone();
        let a = attach_in(pr, &mut t, &target, follow, offset, at)?;
        let keys = bake(pr, &t, &a)?;
        let v = json!({"attachment": a.id, "target": a.target.kind_name(), "clip": a.target.clip().0, "keyframes": keys});
        t.attachments.push(a);
        pr.motion_tracks.insert(t.id, Arc::new(t));
        Ok(v)
    })?;
    Ok(made)
}

fn detach(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "track.detach";
    let id = track_id_p(cmd, p)?;
    let t = get_track(s, cmd, id)?;
    let aid = u64_p(p, "attachment").ok_or_else(|| bad(cmd, "`attachment` is an attachment id (`track.inspect`)"))?;
    let a = t.attachments.iter().find(|a| a.id == aid).cloned().ok_or_else(|| bad(cmd, format!("no attachment {aid} on this track")))?;
    let clear = bool_p(p, "clear").unwrap_or(false);
    let ph = s.playhead();
    s.edit("Detach from Motion Track", |pr, _| {
        let mut t = (*t).clone();
        t.attachments.retain(|x| x.id != aid);
        if clear {
            clear_keys(pr, t.sequence, &a, ph);
        }
        pr.motion_tracks.insert(t.id, Arc::new(t));
        Ok(())
    })?;
    Ok(json!({"detached": aid, "cleared": clear}))
}

fn rebake(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "track.rebake";
    let id = track_id_p(cmd, p)?;
    let t = get_track(s, cmd, id)?;
    if t.attachments.is_empty() {
        return Ok(json!({"attachments": 0}));
    }
    let n = s.edit("Update Motion Track Attachments", |pr, _| {
        let mut keys = 0;
        for a in &t.attachments {
            keys += bake(pr, &t, a)?;
        }
        Ok(keys)
    })?;
    Ok(json!({"attachments": t.attachments.len(), "keyframes": n}))
}

// ---------------------------------------------------------------------------------------------
// convenience commands: track, then something follows
// ---------------------------------------------------------------------------------------------

/// `track.blurObject`'s look.
struct BlurSpec {
    mosaic: bool,
    amount: Option<f64>,
    rectangle: bool,
    feather: Option<f64>,
    at: Tick,
}

struct LabelSpec {
    text: String,
    style: LabelStyle,
    side: Option<Side>,
    size: Option<f64>,
    accent: [f32; 4],
    at: Tick,
    /// The video track (0-based) the label goes on; `None`: the first free one above the clip.
    video_track: Option<usize>,
    /// Shown only from / until these sequence times (inside the tracked clip).
    from: Option<Tick>,
    to: Option<Tick>,
}

/// Where a callout sits from its object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Above,
    Right,
    Left,
    Below,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelStyle {
    Callout,
    Tag,
    Plain,
}

struct SpotSpec {
    dim: f64,
    feather: Option<f64>,
    at: Tick,
}

fn opt_range(cmd: &str, p: &Value, key: &str, lo: f64, hi: f64) -> Result<Option<f64>> {
    match p.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            v.as_f64().filter(|x| x.is_finite() && (lo..=hi).contains(x)).map(Some).ok_or_else(|| bad(cmd, format!("`{key}` is a number from {lo} to {hi}")))
        }
    }
}

fn blur_spec(cmd: &str, p: &Value, mosaic: Option<bool>, at: Tick) -> Result<BlurSpec> {
    let mosaic = match (mosaic, opt_str(cmd, p, "style")?.unwrap_or("blur")) {
        (Some(m), _) => m,
        (None, "blur") => false,
        (None, "mosaic" | "pixelate") => true,
        (None, o) => return Err(bad(cmd, format!("`style` is \"blur\" or \"mosaic\", not \"{o}\""))),
    };
    let rectangle = match opt_str(cmd, p, "shape")?.unwrap_or("ellipse") {
        "ellipse" => false,
        "rectangle" | "rect" => true,
        o => return Err(bad(cmd, format!("`shape` is \"ellipse\" or \"rectangle\", not \"{o}\""))),
    };
    Ok(BlurSpec { mosaic, amount: opt_range(cmd, p, "amount", 0.0, 100.0)?, rectangle, feather: opt_range(cmd, p, "feather", 0.0, 1000.0)?, at })
}

fn blur_object(s: &mut Session, p: &Value) -> Result<Value> {
    let req = request(s, BLUR, p)?;
    let spec = blur_spec(BLUR, p, None, req.at)?;
    let track = Box::new(new_track(&req));
    start(s, BLUR, req, Action::Create { track, then: Then::Blur(spec) }, p)
}

fn hex_color(s: &str) -> Option<[f32; 4]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let c = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
    Some([c(0)?, c(2)?, c(4)?, 1.0])
}

fn label_spec(cmd: &str, p: &Value, at: Tick) -> Result<LabelSpec> {
    let text = match p.get("text") {
        Some(Value::String(t)) if !t.trim().is_empty() && t.chars().count() <= MAX_TEXT_CHARS => t.trim().to_string(),
        _ => return Err(bad(cmd, format!("`text` is the label, 1 to {MAX_TEXT_CHARS} characters"))),
    };
    let style = match opt_str(cmd, p, "style")?.unwrap_or("callout") {
        "callout" => LabelStyle::Callout,
        "tag" => LabelStyle::Tag,
        "plain" => LabelStyle::Plain,
        o => return Err(bad(cmd, format!("`style` is \"callout\", \"tag\" or \"plain\", not \"{o}\""))),
    };
    let side = match opt_str(cmd, p, "side")?.unwrap_or("auto") {
        "auto" => None,
        "right" => Some(Side::Right),
        "left" => Some(Side::Left),
        "above" => Some(Side::Above),
        "below" => Some(Side::Below),
        o => return Err(bad(cmd, format!("`side` is \"auto\", \"left\", \"right\", \"above\" or \"below\", not \"{o}\""))),
    };
    let accent = match opt_str(cmd, p, "accent")? {
        None => [1.0, 0.78, 0.18, 1.0],
        Some(c) => hex_color(c).ok_or_else(|| bad(cmd, "`accent` is a colour \"#rrggbb\""))?,
    };
    let size = opt_range(cmd, p, "size", 4.0, 1000.0)?;
    let video_track = match p.get("videoTrack").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| *n < 1000).ok_or_else(|| bad(cmd, "`videoTrack` is a video track index (0 = V1)"))?,
        ),
    };
    let secs = |k: &str| -> Result<Option<Tick>> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && x.abs() <= 1e7)
                .map(|x| Some(Tick::from_seconds_f64(x)))
                .ok_or_else(|| bad(cmd, format!("`{k}` is a time in seconds on the sequence"))),
        }
    };
    let (from, to) = (secs("from")?, secs("to")?);
    if let (Some(a), Some(b)) = (from, to)
        && b <= a
    {
        return Err(bad(cmd, "`to` must come after `from`"));
    }
    Ok(LabelSpec { text, style, side, size, accent, at, video_track, from, to })
}

fn label(s: &mut Session, p: &Value) -> Result<Value> {
    // checked before the (slower) tracking request
    label_spec(LABEL, p, Tick::ZERO)?;
    let req = request(s, LABEL, p)?;
    let spec = label_spec(LABEL, p, req.at)?;
    let track = Box::new(new_track(&req));
    start(s, LABEL, req, Action::Create { track, then: Then::Label(spec) }, p)
}

fn spot_spec(cmd: &str, p: &Value, at: Tick) -> Result<SpotSpec> {
    Ok(SpotSpec { dim: opt_range(cmd, p, "dim", 0.0, 100.0)?.unwrap_or(60.0), feather: opt_range(cmd, p, "feather", 0.0, 2000.0)?, at })
}

/// `track.add`: a label, blur, mosaic or spotlight on a track already made, at once.
fn add(s: &mut Session, p: &Value) -> Result<Value> {
    let id = track_id_p(ADD, p)?;
    let t = get_track(s, ADD, id)?;
    if t.attachments.len() >= MAX_ATTACHMENTS {
        return Err(bad(ADD, format!("a track drives at most {MAX_ATTACHMENTS} things")));
    }
    // when: `at`, else the playhead where the tracked clip shows, else the seed's moment
    let clip = s
        .project
        .sequence(t.sequence)
        .and_then(|q| q.find_item(t.clip))
        .map(|(_, c)| c.clone())
        .ok_or_else(|| bad(ADD, "the tracked clip is no longer in its sequence"))?;
    let at = match at_p(ADD, p)? {
        Some(a) => a,
        None if s.playhead() >= clip.start && s.playhead() < clip.end() => s.playhead(),
        None => sequence_time_of(&clip, t.seed.t).unwrap_or(clip.start),
    };
    let then = match opt_str(ADD, p, "what")? {
        Some("label" | "text" | "callout") => Then::Label(label_spec(ADD, p, at)?),
        Some("blur") => Then::Blur(blur_spec(ADD, p, Some(false), at)?),
        Some("mosaic") => Then::Blur(blur_spec(ADD, p, Some(true), at)?),
        Some("spotlight") => Then::Spotlight(spot_spec(ADD, p, at)?),
        Some(w) => return Err(bad(ADD, format!("`what` is \"label\", \"blur\", \"mosaic\" or \"spotlight\", not \"{w}\""))),
        None => return Err(bad(ADD, "need `what`: \"label\", \"blur\", \"mosaic\" or \"spotlight\"")),
    };
    let label = match then {
        Then::Label(_) => "Label Object",
        Then::Blur(_) => "Blur Object",
        _ => "Spotlight Object",
    };
    let made = s.edit(label, |pr, st| {
        let mut t = (*t).clone();
        let v = match then {
            Then::Label(l) => add_label(pr, st, &mut t, &l)?,
            Then::Blur(b) => add_blur(pr, &mut t, &b)?,
            Then::Spotlight(sp) => add_spotlight(pr, &mut t, &sp)?,
            Then::Nothing => Value::Null,
        };
        pr.motion_tracks.insert(t.id, Arc::new(t));
        Ok(v)
    })?;
    Ok(made)
}

fn spotlight(s: &mut Session, p: &Value) -> Result<Value> {
    let req = request(s, SPOTLIGHT, p)?;
    let spec = spot_spec(SPOTLIGHT, p, req.at)?;
    let track = Box::new(new_track(&req));
    start(s, SPOTLIGHT, req, Action::Create { track, then: Then::Spotlight(spec) }, p)
}

/// Insert `e` before the clip's intrinsic effects (so it works on the picture before Motion),
/// returning its index.
fn insert_effect(it: &mut TrackItem, e: EffectInstance) -> usize {
    let at = it.effects.iter().position(|x| x.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
    it.effects.insert(at, e);
    at
}

/// The object at the seed in the tracked clip's pixels: centre, half size, rotation.
fn seed_in_pixels(pr: &Project, t: &MotionTrack) -> Option<(Vec2, Vec2, f64)> {
    let (w, h) = pr.source_size(t.item)?;
    let pose = t.pose_at(t.seed.t)?;
    let (w, h) = (f64::from(w), f64::from(h));
    Some((Vec2::new(pose.cx * w, pose.cy * h), Vec2::new(pose.w * w / 2.0, pose.h * h / 2.0), pose.rotation))
}

/// A mask shape around the object at the seed (clip pixels), `pad` times its size.
fn object_mask(pr: &Project, t: &MotionTrack, rectangle: bool, pad: f64) -> Option<(MaskPath, Vec2)> {
    let (c, half, rot) = seed_in_pixels(pr, t)?;
    let half = half * pad;
    let local = if rectangle { MaskPath::rect(-half.x, -half.y, half.x, half.y) } else { MaskPath::ellipse(Vec2::ZERO, half) };
    Some((local.transformed(&Affine::translate(c.x, c.y).then_apply(&Affine::rotate_deg(rot))), half))
}

fn follow_for(t: &MotionTrack) -> FollowMode {
    match t.method {
        TrackingMethod::Points => FollowMode::PositionScaleRotation,
        TrackingMethod::Subject => FollowMode::PositionScale,
    }
}

/// A masked effect on the tracked clip whose mask follows the object; `(effect index, mask
/// index)`.
fn masked_effect(pr: &mut Project, t: &mut MotionTrack, e: EffectInstance, mask: Mask, at: Tick) -> Result<(usize, Value)> {
    let q = pr.sequence_mut(t.sequence).ok_or(EngineError::NoSequence)?;
    let (_, it) = q.find_item_mut(t.clip).ok_or_else(|| EngineError::Other("the tracked clip is gone".into()))?;
    let mut e = e;
    e.masks.push(mask);
    let idx = insert_effect(it, e);
    let spec = TargetSpec::Mask(t.clip, Some(json!(idx)), 0);
    let a = attach_in(pr, t, &spec, follow_for(t), OffsetSpec::Center, at)?;
    let keys = bake(pr, t, &a)?;
    t.attachments.push(a);
    Ok((idx, json!({"effect": idx, "keyframes": keys})))
}

fn add_blur(pr: &mut Project, t: &mut MotionTrack, b: &BlurSpec) -> Result<Value> {
    let (path, half) = object_mask(pr, t, b.rectangle, 1.15).ok_or_else(|| EngineError::Other("the tracked clip has no picture".into()))?;
    let size = (half.x.min(half.y) * 2.0).max(4.0);
    let amount = b.amount.unwrap_or(60.0) / 100.0;
    let e = if b.mosaic {
        let (w, _) = pr.source_size(t.item).unwrap_or((1920, 1080));
        // blocks about a sixth of the object across, whatever the picture's size
        let block = (size / (6.0 + 18.0 * (1.0 - amount))).max(2.0);
        let mut e = filmcraft_project::effect::find_effect("mosaic").ok_or_else(|| EngineError::Other("no Mosaic effect".into()))?.instance();
        let n = (f64::from(w) / block).clamp(1.0, 4000.0).round();
        e.params.insert("horizontal".into(), Param::new(ParamValue::Float(n)));
        e.params.insert("vertical".into(), Param::new(ParamValue::Float(n)));
        e
    } else {
        let mut e = filmcraft_project::effect::find_effect("gaussian_blur").ok_or_else(|| EngineError::Other("no Gaussian Blur effect".into()))?.instance();
        e.params.insert("blurriness".into(), Param::new(ParamValue::Float((size * 0.5 * amount.max(0.05)).clamp(4.0, 2000.0))));
        e.params.insert("repeat_edge".into(), Param::new(ParamValue::Bool(true)));
        e
    };
    let mut m = Mask::new("Tracked object", path);
    let feather = b.feather.unwrap_or((size * 0.12).clamp(1.0, 200.0));
    m.feather = Param::new(ParamValue::Float(feather));
    let (idx, v) = masked_effect(pr, t, e, m, b.at)?;
    Ok(json!({"kind": if b.mosaic { "mosaic" } else { "blur" }, "clip": t.clip.0, "effect": idx, "keyframes": v["keyframes"]}))
}

fn add_spotlight(pr: &mut Project, t: &mut MotionTrack, sp: &SpotSpec) -> Result<Value> {
    let (path, half) = object_mask(pr, t, false, 1.35).ok_or_else(|| EngineError::Other("the tracked clip has no picture".into()))?;
    let mut e = filmcraft_project::effect::find_effect("levels").ok_or_else(|| EngineError::Other("no Levels effect".into()))?.instance();
    e.params.insert("out_white".into(), Param::new(ParamValue::Float((255.0 * (1.0 - sp.dim / 100.0)).clamp(0.0, 255.0))));
    let mut m = Mask::new("Spotlight", path);
    m.inverted = true;
    let size = half.x.min(half.y) * 2.0;
    m.feather = Param::new(ParamValue::Float(sp.feather.unwrap_or((size * 0.35).clamp(4.0, 600.0))));
    let (idx, v) = masked_effect(pr, t, e, m, sp.at)?;
    Ok(json!({"kind": "spotlight", "clip": t.clip.0, "effect": idx, "keyframes": v["keyframes"]}))
}

/// The label's graphic clip: a dot on the object, a leader line and the text, laid out around the
/// canvas centre; the clip's Motion then follows the object (its centre on the object's).
fn add_label(pr: &mut Project, st: &mut crate::EditorState, t: &mut MotionTrack, l: &LabelSpec) -> Result<Value> {
    let q = pr.sequence(t.sequence).ok_or(EngineError::NoSequence)?;
    let (_, clip) = q.find_item(t.clip).ok_or_else(|| EngineError::Other("the tracked clip is gone".into()))?;
    let clip = clip.clone();
    let (fw, fh, rate) = (q.settings.width, q.settings.height, q.settings.frame_rate);
    let size = pr.source_size(t.item).ok_or_else(|| EngineError::Other("the tracked clip has no picture".into()))?;
    let view = View::new(pr, q, &clip, t, size);
    let shown = view.at(l.at).ok_or_else(|| EngineError::Other("the object is not on screen there".into()))?;
    // laid out for a 1080-pixel short side, scaled with the frame
    let u = f64::from(fw.min(fh)) / 1080.0;
    let text_size = l.size.unwrap_or(44.0 * u);
    let pad = 14.0 * u;
    let (fwf, fhf) = (f64::from(fw), f64::from(fh));
    // where the object shows over the clip (every few frames), to keep the label in the frame
    let fd = rate.sane().frame_duration().max(Tick(1));
    let step = Tick(fd.0.saturating_mul(((clip.duration.0 / fd.0.max(1)) / 200).max(1)));
    let mut path: Vec<Vec2> = Vec::new();
    let mut tt = clip.start;
    while tt < clip.end() && path.len() < 400 {
        if let Some(o) = view.at(tt) {
            path.push(o.pos);
        }
        let Some(next) = tt.0.checked_add(step.0) else { break };
        tt = Tick(next);
    }
    if path.is_empty() {
        path.push(shown.pos);
    }
    // the text's box, estimated (a bold sans is about 0.58 em a character)
    let chars = l.text.lines().map(|x| x.chars().count()).max().unwrap_or(1).max(1) as f64;
    let lines = l.text.lines().count().max(1) as f64;
    let (tw, th) = ((chars * 0.58 * text_size + 2.0 * pad).min(fwf * 0.95), lines * text_size * 1.25 + 2.0 * pad);
    let reach = (shown.size.y * 0.35).max(60.0 * u) + 30.0 * u;
    // per side: the leader's end and the label box (left, top, right, bottom), from the object
    let layout = |side: Side| -> (Vec2, [f64; 4]) {
        match side {
            Side::Above => {
                let end = Vec2::new(0.0, -(shown.size.y / 2.0).max(reach * 0.6) - 20.0 * u);
                (end, [-tw / 2.0, end.y - th, tw / 2.0, end.y])
            }
            Side::Below => {
                let end = Vec2::new(0.0, (shown.size.y / 2.0).max(reach * 0.6) + 20.0 * u);
                (end, [-tw / 2.0, end.y, tw / 2.0, end.y + th])
            }
            Side::Right => {
                let end = Vec2::new(reach * 0.8, -reach);
                (end, [end.x + 6.0 * u, end.y - th / 2.0, end.x + 6.0 * u + tw, end.y + th / 2.0])
            }
            Side::Left => {
                let end = Vec2::new(-reach * 0.8, -reach);
                (end, [end.x - 6.0 * u - tw, end.y - th / 2.0, end.x - 6.0 * u, end.y + th / 2.0])
            }
        }
    };
    let margin = 8.0 * u;
    let inside = |side: Side| -> usize {
        let (_, b) = layout(side);
        path.iter().filter(|p| p.x + b[0] >= margin && p.x + b[2] <= fwf - margin && p.y + b[1] >= margin && p.y + b[3] <= fhf - margin).count()
    };
    // asked for, else the side that keeps it in the frame the longest (above first on ties:
    // it reads best on a tall frame; then the side with more room)
    let roomier = if shown.pos.x < fwf / 2.0 { [Side::Right, Side::Left] } else { [Side::Left, Side::Right] };
    let side = l.side.unwrap_or_else(|| {
        let order = [Side::Above, roomier[0], roomier[1], Side::Below];
        let mut best = (0usize, Side::Above);
        for sd in order {
            let n = inside(sd);
            if n > best.0 {
                best = (n, sd);
            }
        }
        best.1
    });
    let (end_off, bx) = layout(side);
    let centre = Vec2::new(fwf / 2.0, fhf / 2.0);
    let end = centre + end_off;
    let mut layers: Vec<EffectInstance> = Vec::new();
    let accent = l.accent;
    if l.style != LabelStyle::Tag {
        // the leader: a thin bar from the dot to the label, and the dot
        let d = end - centre;
        let len = d.length().max(1.0);
        let mut bar = filmcraft_project::graphic::new_shape_layer(0, centre + d * 0.5, Vec2::new(len, (3.0 * u).max(1.5)), Vec::new());
        set(&mut bar, "fill_color", ParamValue::Color(accent));
        set(&mut bar, "rotation", ParamValue::Float(d.y.atan2(d.x).to_degrees()));
        set(&mut bar, "name", ParamValue::Text("Leader".into()));
        layers.push(bar);
        let r = (9.0 * u).max(3.0);
        let mut dot = filmcraft_project::graphic::new_shape_layer(1, centre, Vec2::new(2.0 * r, 2.0 * r), Vec::new());
        set(&mut dot, "fill_color", ParamValue::Color(accent));
        set(&mut dot, "stroke", ParamValue::Bool(true));
        set(&mut dot, "stroke_color", ParamValue::Color([0.0, 0.0, 0.0, 0.6]));
        set(&mut dot, "stroke_width", ParamValue::Float((2.0 * u).max(1.0)));
        set(&mut dot, "name", ParamValue::Text("Dot".into()));
        layers.push(dot);
    }
    // the text's anchor: on its first baseline, at its alignment point
    let first_baseline = bx[1] + pad + text_size * 0.95;
    let (pos, align) = match side {
        Side::Above | Side::Below => (Vec2::new(centre.x, centre.y + first_baseline), 1),
        Side::Right => (Vec2::new(centre.x + bx[0] + pad, centre.y + first_baseline), 0),
        Side::Left => (Vec2::new(centre.x + bx[2] - pad, centre.y + first_baseline), 2),
    };
    let mut text = filmcraft_project::graphic::new_text_layer(&l.text, pos, text_size);
    set(&mut text, "align", ParamValue::Choice(align));
    set(&mut text, "font_style", ParamValue::Text("Bold".into()));
    set(&mut text, "name", ParamValue::Text("Label".into()));
    match l.style {
        LabelStyle::Callout => {
            set(&mut text, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
            set(&mut text, "background", ParamValue::Bool(true));
            set(&mut text, "background_color", ParamValue::Color([0.06, 0.07, 0.09, 1.0]));
            set(&mut text, "background_opacity", ParamValue::Float(82.0));
            set(&mut text, "background_size", ParamValue::Float(pad));
            set(&mut text, "background_radius", ParamValue::Float(pad));
            set(&mut text, "stroke", ParamValue::Bool(false));
        }
        LabelStyle::Tag => {
            set(&mut text, "fill_color", ParamValue::Color([0.05, 0.05, 0.06, 1.0]));
            set(&mut text, "background", ParamValue::Bool(true));
            set(&mut text, "background_color", ParamValue::Color(accent));
            set(&mut text, "background_opacity", ParamValue::Float(100.0));
            set(&mut text, "background_size", ParamValue::Float(pad * 0.8));
            set(&mut text, "background_radius", ParamValue::Float(text_size));
        }
        LabelStyle::Plain => {
            set(&mut text, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
            set(&mut text, "shadow", ParamValue::Bool(true));
            set(&mut text, "shadow_opacity", ParamValue::Float(85.0));
            set(&mut text, "shadow_blur", ParamValue::Float(12.0 * u));
            set(&mut text, "shadow_distance", ParamValue::Float(3.0 * u));
        }
    }
    layers.push(text);
    // the graphic clip over the tracked one (or the part of it asked for), on the first free
    // video track above it (or the one asked for)
    let fd = rate.sane().frame_duration().max(Tick(1));
    let start = rate.sane().snap(l.from.map_or(clip.start, |f| f.max(clip.start)));
    let end = l.to.map_or(clip.end(), |t| t.min(clip.end()));
    if end <= start {
        return Err(EngineError::Other("the label's `from`..`to` is not inside the tracked clip".into()));
    }
    let src = crate::graphics::graphic_source(pr, fw, fh, rate);
    let mut gi = pr
        .make_track_item(src, TrackKind::Video, start, TimeRange::new(Tick::ZERO, (end - start).max(fd)), rate)
        .ok_or_else(|| EngineError::Other("can't make the label clip".into()))?;
    gi.name = format!("Label: {}", l.text.chars().take(40).collect::<String>());
    gi.effects.extend(layers);
    for e in &mut gi.effects {
        filmcraft_project::resolve_auto_points(e, (fw, fh), (fw, fh));
    }
    let label_id = gi.id;
    let track_id = filmcraft_project::TrackId(pr.alloc_id());
    let q = pr.sequence_mut(t.sequence).ok_or(EngineError::NoSequence)?;
    let range = gi.range();
    let above = q.video_tracks.iter().position(|tr| tr.items.iter().any(|i| i.id == t.clip)).map_or(0, |k| k + 1);
    let free = |tr: &filmcraft_project::Track| !tr.locked && !tr.items.iter().any(|i| i.range().overlaps(&range));
    let idx = match l.video_track {
        Some(k) if q.video_tracks.get(k).is_some_and(free) => k,
        Some(k) if k < q.video_tracks.len() => return Err(EngineError::Other(format!("V{} is not free for the label", k + 1))),
        Some(k) => return Err(EngineError::Other(format!("there is no V{}", k + 1))),
        None => match (above..q.video_tracks.len()).find(|&k| q.video_tracks.get(k).is_some_and(free)) {
            Some(k) => k,
            None => {
                let n = q.video_tracks.len() + 1;
                q.video_tracks.push(filmcraft_project::Track::new(track_id, TrackKind::Video, format!("Video {n}")));
                q.video_tracks.len() - 1
            }
        },
    };
    let tr = q.video_tracks.get_mut(idx).ok_or_else(|| EngineError::Other("no track for the label".into()))?;
    tr.items.push(gi);
    tr.sort();
    q.check().map_err(EngineError::Other)?;
    let a = attach_in(pr, t, &TargetSpec::Clip(label_id), FollowMode::Position, OffsetSpec::Center, l.at)?;
    let keys = bake(pr, t, &a)?;
    t.attachments.push(a);
    st.selection = vec![label_id];
    Ok(json!({"kind": "label", "clip": label_id.0, "track": format!("V{}", idx + 1), "keyframes": keys}))
}

fn set(e: &mut EffectInstance, id: &str, v: ParamValue) {
    e.params.insert(id.to_string(), Param::new(v));
}

// ---------------------------------------------------------------------------------------------
// where the object shows
// ---------------------------------------------------------------------------------------------

/// The tracked object as shown in the sequence through its clip.
struct View<'a> {
    seq: &'a Sequence,
    clip: &'a TrackItem,
    track: &'a MotionTrack,
    size: (u32, u32),
    par: Option<(u32, u32)>,
}

/// The object in one sequence frame: its centre (sequence pixels), rotation (degrees) and scale
/// (the track's times the clip's Motion), its box's size as shown, and whether it was lost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    pub pos: Vec2,
    pub rot: f64,
    pub scale: f64,
    pub size: Vec2,
    pub lost: bool,
}

impl<'a> View<'a> {
    fn new(pr: &Project, seq: &'a Sequence, clip: &'a TrackItem, track: &'a MotionTrack, size: (u32, u32)) -> Self {
        View { seq, clip, track, size, par: pr.source_par(clip.item) }
    }

    fn at(&self, t: Tick) -> Option<Shown> {
        let mt = self.clip.source_time_at(t);
        let pose = self.track.pose_at(mt)?;
        let m = filmcraft_render::motion_matrix(self.seq, self.clip, self.size, self.par, self.clip.effect_time_at(t));
        let (w, h) = (f64::from(self.size.0), f64::from(self.size.1));
        let pos = m.apply(Vec2::new(pose.cx * w, pose.cy * h));
        let mscale = m.determinant().abs().sqrt();
        let mrot = m.b.atan2(m.a).to_degrees();
        let (sx, sy) = ((m.a * m.a + m.b * m.b).sqrt(), (m.c * m.c + m.d * m.d).sqrt());
        let shown = Shown { pos, rot: pose.rotation + mrot, scale: pose.scale * mscale, size: Vec2::new(pose.w * w * sx, pose.h * h * sy), lost: pose.lost };
        let ok = [shown.pos.x, shown.pos.y, shown.rot, shown.scale, shown.size.x, shown.size.y].iter().all(|v| v.is_finite()) && shown.scale > 0.0;
        ok.then_some(shown)
    }
}

/// The sequence time at which `clip` shows media time `mt` (the first, for a clip that shows it
/// more than once), when it does.
fn sequence_time_of(clip: &TrackItem, mt: Tick) -> Option<Tick> {
    if clip.frame_hold.is_some() {
        return None;
    }
    let speed = if clip.speed.is_finite() && clip.speed != 0.0 { clip.speed.abs() } else { return None };
    let rel = if clip.reverse { (clip.source_out() - Tick(1) - mt).0 as f64 / speed } else { (mt - clip.source_in).0 as f64 / speed };
    if !rel.is_finite() {
        return None;
    }
    let t = clip.start + Tick(rel.round() as i64);
    (t >= clip.start && t < clip.end()).then_some(t)
}

/// A track's path as shown in its sequence, for the Program monitor: (sequence time, where the
/// object shows, lost) at every sample the clip shows.
pub fn shown_path(project: &Project, track: &MotionTrack) -> Vec<(Tick, Shown)> {
    let Some(q) = project.sequence(track.sequence) else { return Vec::new() };
    let Some((_, clip)) = q.find_item(track.clip) else { return Vec::new() };
    let Some(size) = project.source_size(clip.item) else { return Vec::new() };
    let view = View::new(project, q, clip, track, size);
    track.samples.iter().filter_map(|s| sequence_time_of(clip, s.t)).filter_map(|t| view.at(t).map(|o| (t, o))).collect()
}

/// The object in the active sequence at sequence time `t`, for the monitor overlay.
pub fn shown_at(project: &Project, track: &MotionTrack, t: Tick) -> Option<Shown> {
    let q = project.sequence(track.sequence)?;
    let (_, clip) = q.find_item(track.clip)?;
    if t < clip.start || t >= clip.end() {
        return None;
    }
    let size = project.source_size(clip.item)?;
    View::new(project, q, clip, track, size).at(t)
}

fn track_json(s: &Session, id: MotionTrackId, samples: bool) -> Result<Value> {
    let t = s.project.motion_tracks.get(&id).ok_or_else(|| bad("track", format!("no motion track {}", id.0)))?;
    let secs = |t: Tick| (t.seconds() * 1000.0).round() / 1000.0;
    let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
    let clip = s.project.sequence(t.sequence).and_then(|q| q.find_item(t.clip)).map(|(_, c)| c.clone());
    let mut v = json!({
        "id": t.id.0,
        "name": t.name,
        "clip": t.clip.0,
        "sequence": t.sequence.0,
        "item": t.item.0,
        "clipMissing": clip.is_none(),
        "method": t.method.name(),
        "direction": t.direction.name(),
        "seed": {"seconds": secs(t.seed.t), "box": [r3(t.seed.cx - t.seed.w / 2.0), r3(t.seed.cy - t.seed.h / 2.0), r3(t.seed.w), r3(t.seed.h)]},
        "frames": t.samples.len(),
        "seen": r3(t.seen_share()),
        "range": t.range().map(|(a, b)| json!([secs(a), secs(b)])).unwrap_or(Value::Null),
        "lost": t.lost_spans().iter().map(|(a, b)| json!([secs(*a), secs(*b)])).collect::<Vec<_>>(),
        "attachments": t.attachments.iter().map(|a| json!({
            "id": a.id, "target": a.target.kind_name(), "clip": a.target.clip().0, "follow": a.follow.name(),
            "offset": [r3(a.offset[0]), r3(a.offset[1])],
            "detail": match &a.target {
                AttachTarget::Clip { .. } => Value::Null,
                AttachTarget::GraphicLayer { layer, .. } => json!({"layer": layer}),
                AttachTarget::EffectParam { effect, param, .. } => json!({"effect": effect, "param": param}),
                AttachTarget::Mask { effect, mask, .. } => json!({"effect": effect, "mask": mask}),
            },
        })).collect::<Vec<_>>(),
    });
    if samples {
        let shown: HashMap<i64, Shown> = match &clip {
            Some(c) => {
                let q = s.project.sequence(t.sequence);
                let size = s.project.source_size(t.item);
                match (q, size) {
                    (Some(q), Some(size)) => {
                        let view = View::new(&s.project, q, c, t, size);
                        t.samples.iter().filter_map(|x| sequence_time_of(c, x.t).and_then(|st| view.at(st)).map(|o| (x.t.0, o))).collect()
                    }
                    _ => HashMap::new(),
                }
            }
            None => HashMap::new(),
        };
        v["samples"] = t
            .samples
            .iter()
            .map(|x| {
                let mut o = json!({
                    "seconds": secs(x.t),
                    "box": [r3(x.cx - x.w / 2.0), r3(x.cy - x.h / 2.0), r3(x.w), r3(x.h)],
                    "rotation": r3(x.rotation), "scale": r3(x.scale), "confidence": r3(f64::from(x.confidence)), "lost": x.lost,
                });
                if let (Some(c), Some(sh)) = (&clip, shown.get(&x.t.0)) {
                    o["sequenceSeconds"] = json!(sequence_time_of(c, x.t).map(secs));
                    o["frame"] = json!([r3(sh.pos.x), r3(sh.pos.y)]);
                }
                o
            })
            .collect();
    }
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// attaching and baking
// ---------------------------------------------------------------------------------------------

/// The parameter values a target has at sequence time `t`, and where its point is in the
/// sequence.
struct TargetNow {
    point: Vec2,
    base_scale: f64,
    base_rotation: f64,
}

/// The clip's Motion matrix (source pixels → sequence pixels) at sequence time `t`.
fn clip_matrix(pr: &Project, q: &Sequence, it: &TrackItem, t: Tick) -> Option<Affine> {
    let size = pr.source_size(it.item).unwrap_or((q.settings.width, q.settings.height));
    let m = filmcraft_render::motion_matrix(q, it, size, pr.source_par(it.item), it.effect_time_at(t));
    m.inverse().map(|_| m)
}

fn layer_index(it: &TrackItem, layer: &Value) -> Option<usize> {
    let layers = filmcraft_project::graphic::layer_indices(&it.effects);
    match layer {
        Value::Number(n) => {
            let n = n.as_u64()?;
            // a uid first (they are large), else an index among the layers
            layers
                .iter()
                .copied()
                .find(|&i| it.effects.get(i).and_then(|e| e.layer.as_ref()).is_some_and(|x| x.uid == n && n != 0))
                .or_else(|| layers.get(usize::try_from(n).ok()?).copied())
        }
        Value::String(name) => layers
            .iter()
            .enumerate()
            .find(|(k, i)| it.effects.get(**i).is_some_and(|e| filmcraft_project::graphic::layer_display_name(e, *k).eq_ignore_ascii_case(name)))
            .map(|(_, i)| *i),
        _ => None,
    }
}

fn effect_index(it: &TrackItem, e: &Value) -> Option<usize> {
    match e {
        Value::Number(n) => usize::try_from(n.as_u64()?).ok().filter(|i| *i < it.effects.len()),
        Value::String(id) => it.effects.iter().position(|x| &x.effect == id),
        _ => None,
    }
}

/// Resolve `spec` to an attachment of `t` made at sequence time `at` (clamped into where the
/// track and the target overlap); assigns a graphic layer its uid if it has none.
fn attach_in(pr: &mut Project, t: &mut MotionTrack, spec: &TargetSpec, follow: FollowMode, offset: OffsetSpec, at: Tick) -> Result<Attachment> {
    let cmd = ATTACH;
    let target_clip = match spec {
        TargetSpec::Clip(c) | TargetSpec::Layer(c, _) | TargetSpec::Effect(c, _, _) | TargetSpec::Mask(c, _, _) => *c,
    };
    // a graphic layer needs a uid to be found again
    let mut layer_uid: Option<u64> = None;
    if let TargetSpec::Layer(c, l) = spec {
        let fresh = pr.next_id;
        let mut used = false;
        {
            let q = pr.sequence_mut(t.sequence).ok_or(EngineError::NoSequence)?;
            let (_, it) = q.find_item_mut(*c).ok_or_else(|| bad(cmd, "no such target clip in the track's sequence"))?;
            let li = layer_index(it, l).ok_or_else(|| bad(cmd, "no such graphic layer on the target clip"))?;
            let e = it.effects.get_mut(li).ok_or_else(|| bad(cmd, "no such graphic layer"))?;
            let extra = e.layer.get_or_insert_with(Default::default);
            if extra.uid == 0 {
                extra.uid = fresh;
                used = true;
            }
            layer_uid = Some(extra.uid);
        }
        if used {
            pr.next_id = pr.next_id.saturating_add(1);
        }
    }
    let q = pr.sequence(t.sequence).ok_or_else(|| bad(cmd, "the track's sequence is gone"))?;
    let (_, clip) = q.find_item(t.clip).ok_or_else(|| bad(cmd, "the tracked clip is no longer in its sequence"))?;
    let (_, target) = q.find_item(target_clip).ok_or_else(|| bad(cmd, "no such target clip in the track's sequence"))?;
    if q.find_item(target_clip).and_then(|(tr, _)| q.track(tr)).is_some_and(|tr| tr.kind != TrackKind::Video) {
        return Err(bad(cmd, "the target is an audio clip"));
    }
    let (a, b) = (clip.start.max(target.start), clip.end().min(target.end()));
    if b <= a {
        return Err(bad(cmd, "the target and the tracked clip are never on screen together"));
    }
    let at = at.clamp(a, b - Tick(1));
    let size = pr.source_size(clip.item).ok_or_else(|| bad(cmd, "the tracked clip has no picture"))?;
    let shown = View::new(pr, q, clip, t, size).at(at).ok_or_else(|| bad(cmd, "the track has no samples"))?;
    let m = clip_matrix(pr, q, target, at).ok_or_else(|| bad(cmd, "the target is scaled to nothing"))?;
    let tmt = target.effect_time_at(at);
    let centre_of = |it: &TrackItem| -> Vec2 {
        let (w, h) = pr.source_size(it.item).unwrap_or((q.settings.width, q.settings.height));
        Vec2::new(f64::from(w) / 2.0, f64::from(h) / 2.0)
    };
    let (kind, now) = match spec {
        TargetSpec::Clip(c) => {
            let mo = target.effect("motion").ok_or_else(|| bad(cmd, "the target has no Motion"))?;
            let p = mo.vec2_at("position", tmt);
            let point = if p.x.is_finite() && p.y.is_finite() { p } else { Vec2::new(f64::from(q.settings.width) / 2.0, f64::from(q.settings.height) / 2.0) };
            (AttachTarget::Clip { clip: *c }, TargetNow { point, base_scale: mo.f64_at("scale", tmt), base_rotation: mo.f64_at("rotation", tmt) })
        }
        TargetSpec::Layer(c, l) => {
            let li = layer_index(target, l).ok_or_else(|| bad(cmd, "no such graphic layer on the target clip"))?;
            let e = target.effects.get(li).ok_or_else(|| bad(cmd, "no such graphic layer"))?;
            let p = e.vec2_at("position", tmt);
            let local = if p.x.is_finite() && p.y.is_finite() { p } else { centre_of(target) };
            let uid = layer_uid.unwrap_or(0);
            (
                AttachTarget::GraphicLayer { clip: *c, layer: uid },
                TargetNow { point: m.apply(local), base_scale: e.f64_at("scale", tmt), base_rotation: e.f64_at("rotation", tmt) },
            )
        }
        TargetSpec::Effect(c, e, prm) => {
            let ei = effect_index(target, e).ok_or_else(|| bad(cmd, "no such effect on the target clip"))?;
            let fx = target.effects.get(ei).ok_or_else(|| bad(cmd, "no such effect"))?;
            let def = fx.def().and_then(|d| d.param(prm)).ok_or_else(|| bad(cmd, format!("the effect has no parameter `{prm}`")))?;
            if def.kind != ParamKind::Point || !def.animatable {
                return Err(bad(cmd, format!("`{prm}` is not a point that can be animated")));
            }
            let p = fx.vec2_at(prm, tmt);
            let local = if p.x.is_finite() && p.y.is_finite() { p } else { centre_of(target) };
            (AttachTarget::EffectParam { clip: *c, effect: ei, param: prm.clone() }, TargetNow { point: m.apply(local), base_scale: 100.0, base_rotation: 0.0 })
        }
        TargetSpec::Mask(c, e, mi) => {
            let ei = match e {
                Some(e) => effect_index(target, e).ok_or_else(|| bad(cmd, "no such effect on the target clip"))?,
                None => target.effects.iter().position(|x| x.effect == "opacity").ok_or_else(|| bad(cmd, "the target has no Opacity effect"))?,
            };
            let mask = target.effects.get(ei).and_then(|x| x.masks.get(*mi)).ok_or_else(|| bad(cmd, format!("no mask {mi} on that effect")))?;
            // the path at `at`, in sequence pixels relative to the object's centre
            let base = mask.path_at(tmt).transformed(&Affine::translate(-shown.pos.x, -shown.pos.y).then_apply(&m));
            if base.vertices.is_empty() || base.vertices.iter().any(|v| !(v.p.x.is_finite() && v.p.y.is_finite())) {
                return Err(bad(cmd, "the mask has no path"));
            }
            (AttachTarget::Mask { clip: *c, effect: ei, mask: *mi, base }, TargetNow { point: shown.pos, base_scale: 100.0, base_rotation: 0.0 })
        }
    };
    let off = match (offset, &kind) {
        (_, AttachTarget::Mask { .. }) | (OffsetSpec::Center, _) => Vec2::ZERO,
        (OffsetSpec::Keep, _) => now.point - shown.pos,
        (OffsetSpec::Px(v), _) => v,
    };
    if !(off.x.is_finite() && off.y.is_finite()) {
        return Err(bad(cmd, "the offset is not a number"));
    }
    let finite = |v: f64, d: f64| if v.is_finite() { round_to(v, 6) } else { d };
    Ok(Attachment {
        id: t.next_attachment_id(),
        target: kind,
        follow,
        offset: [round_to(off.x, 4), round_to(off.y, 4)],
        at,
        ref_scale: finite(shown.scale, 1.0),
        ref_rotation: finite(shown.rot, 0.0),
        base_scale: finite(now.base_scale, 100.0),
        base_rotation: finite(now.base_rotation, 0.0),
    })
}

/// One sequence frame of an attachment: the attached point (sequence pixels), the object's
/// scale and rotation relative to the attachment's reference.
#[derive(Clone, Copy, Debug)]
struct Row {
    t: Tick,
    point: Vec2,
    rel_scale: f64,
    rel_rot: f64,
}

/// Ramer–Douglas–Peucker over `rows` (point x, y, scale ×100, rotation): the indices to keep so
/// that linear interpolation stays within [`TOLERANCE`] of every row in every dimension.
fn simplify(rows: &[Row]) -> Vec<usize> {
    let n = rows.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let val = |r: &Row| [r.point.x, r.point.y, r.rel_scale * 100.0, r.rel_rot];
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ra, rb) = (rows[a], rows[b]);
        let (ta, tb) = (ra.t.0 as f64, rb.t.0 as f64);
        let (va, vb) = (val(&ra), val(&rb));
        let mut worst = (0.0f64, a);
        for (i, r) in rows.iter().enumerate().take(b).skip(a + 1) {
            let u = if tb > ta { (r.t.0 as f64 - ta) / (tb - ta) } else { 0.5 };
            let v = val(r);
            let e = (0..4).map(|k| (v[k] - (va[k] + (vb[k] - va[k]) * u)).abs() / TOLERANCE[k]).fold(0.0, f64::max);
            if e > worst.0 {
                worst = (e, i);
            }
        }
        if worst.0 > 1.0 {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// Write `keys` (time, value) on `prm` as its whole animation (one key: a static value).
fn put_keys(prm: &mut Param, keys: Vec<(Tick, ParamValue)>) {
    let mut keys = keys;
    keys.sort_by_key(|k| k.0);
    keys.dedup_by_key(|k| k.0);
    prm.keyframes.clear();
    match keys.as_slice() {
        [] => {}
        [(_, v)] => prm.value = v.clone(),
        _ => {
            if let Some((_, v)) = keys.first() {
                prm.value = v.clone();
            }
            for (t, v) in keys {
                prm.put_keyframe(t, v);
            }
        }
    }
}

fn param_entry<'e>(e: &'e mut EffectInstance, id: &str, default: ParamValue) -> &'e mut Param {
    e.params.entry(id.to_string()).or_insert_with(|| Param::new(default))
}

/// Write an attachment's keyframes from the track's samples; how many were written.
fn bake(pr: &mut Project, t: &MotionTrack, a: &Attachment) -> Result<usize> {
    let cmd = "track";
    let q = pr.sequence(t.sequence).ok_or_else(|| bad(cmd, "the track's sequence is gone"))?;
    let (_, clip) = q.find_item(t.clip).ok_or_else(|| bad(cmd, "the tracked clip is no longer in its sequence"))?;
    let (_, target) = q.find_item(a.target.clip()).ok_or_else(|| bad(cmd, "the attached clip is no longer in the sequence"))?;
    let (start, end) = (clip.start.max(target.start), clip.end().min(target.end()));
    if end <= start {
        return Err(bad(cmd, "the attached clip and the tracked clip are no longer on screen together"));
    }
    let size = pr.source_size(clip.item).ok_or_else(|| bad(cmd, "the tracked clip has no picture"))?;
    let fd = q.settings.frame_rate.sane().frame_duration().max(Tick(1));
    let view = View::new(pr, q, clip, t, size);
    let off = Vec2::new(a.offset[0], a.offset[1]);
    let mut rows: Vec<Row> = Vec::new();
    let mut tt = start;
    while tt < end && rows.len() < MAX_BAKE_FRAMES {
        if let Some(o) = view.at(tt) {
            let rel_scale = if a.follow.scale() { (o.scale / a.ref_scale).clamp(1e-3, 1e3) } else { 1.0 };
            let rel_rot = if a.follow.rotation() { o.rot - a.ref_rotation } else { 0.0 };
            let point = o.pos + Affine::rotate_deg(rel_rot).apply(off * rel_scale);
            rows.push(Row { t: tt, point, rel_scale, rel_rot });
        }
        let Some(next) = tt.0.checked_add(fd.0) else { break };
        tt = Tick(next);
    }
    if rows.is_empty() {
        return Err(bad(cmd, "the track has no samples where the attached clip is"));
    }
    let kept: Vec<Row> = simplify(&rows).into_iter().filter_map(|i| rows.get(i).copied()).collect();
    // in the target's own space and media time
    let target = target.clone();
    let mats: Vec<(Tick, Affine)> = kept.iter().map(|r| (target.effect_time_at(r.t), clip_matrix(pr, q, &target, r.t).unwrap_or(Affine::IDENTITY))).collect();
    let n = kept.len();
    let seq_id = t.sequence;
    let qm = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
    let (_, it) = qm.find_item_mut(a.target.clip()).ok_or_else(|| bad(cmd, "the attached clip is gone"))?;
    let scale_keys = |base: f64| -> Vec<(Tick, ParamValue)> {
        kept.iter().zip(&mats).map(|(r, (mt, _))| (*mt, ParamValue::Float((base * r.rel_scale).clamp(0.0, 1e5)))).collect()
    };
    let rot_keys = |base: f64| -> Vec<(Tick, ParamValue)> { kept.iter().zip(&mats).map(|(r, (mt, _))| (*mt, ParamValue::Float(base + r.rel_rot))).collect() };
    match &a.target {
        AttachTarget::Clip { .. } => {
            let mo = it.effect_mut("motion").ok_or_else(|| bad(cmd, "the attached clip has no Motion"))?;
            mo.enabled = true;
            put_keys(
                param_entry(mo, "position", ParamValue::Vec2(Vec2::ZERO)),
                kept.iter().zip(&mats).map(|(r, (mt, _))| (*mt, ParamValue::Vec2(r.point))).collect(),
            );
            if a.follow.scale() {
                let uniform = mo.param("uniform_scale").and_then(|p| p.value.as_bool()).unwrap_or(true);
                let base_w = mo.f64_at("scale_width", a.at);
                put_keys(param_entry(mo, "scale", ParamValue::Float(100.0)), scale_keys(a.base_scale));
                if !uniform {
                    put_keys(param_entry(mo, "scale_width", ParamValue::Float(100.0)), scale_keys(base_w));
                }
            }
            if a.follow.rotation() {
                put_keys(param_entry(mo, "rotation", ParamValue::Float(0.0)), rot_keys(a.base_rotation));
            }
        }
        AttachTarget::GraphicLayer { layer, .. } => {
            let li = layer_index(it, &json!(layer)).filter(|_| *layer != 0).ok_or_else(|| bad(cmd, "the attached graphic layer is gone"))?;
            let e = it.effects.get_mut(li).ok_or_else(|| bad(cmd, "the attached graphic layer is gone"))?;
            let pts: Vec<(Tick, ParamValue)> =
                kept.iter().zip(&mats).map(|(r, (mt, m))| (*mt, ParamValue::Vec2(m.inverse().unwrap_or(Affine::IDENTITY).apply(r.point)))).collect();
            put_keys(param_entry(e, "position", ParamValue::Vec2(Vec2::ZERO)), pts);
            if a.follow.scale() {
                put_keys(param_entry(e, "scale", ParamValue::Float(100.0)), scale_keys(a.base_scale));
            }
            if a.follow.rotation() {
                put_keys(param_entry(e, "rotation", ParamValue::Float(0.0)), rot_keys(a.base_rotation));
            }
        }
        AttachTarget::EffectParam { effect, param, .. } => {
            let e = it.effects.get_mut(*effect).ok_or_else(|| bad(cmd, "the attached effect is gone"))?;
            if e.def().and_then(|d| d.param(param)).is_none_or(|d| d.kind != ParamKind::Point) {
                return Err(bad(cmd, format!("the attached effect has no point `{param}`")));
            }
            let pts: Vec<(Tick, ParamValue)> =
                kept.iter().zip(&mats).map(|(r, (mt, m))| (*mt, ParamValue::Vec2(m.inverse().unwrap_or(Affine::IDENTITY).apply(r.point)))).collect();
            put_keys(param_entry(e, param, ParamValue::Vec2(Vec2::ZERO)), pts);
        }
        AttachTarget::Mask { effect, mask, base, .. } => {
            let m = it.effects.get_mut(*effect).and_then(|e| e.masks.get_mut(*mask)).ok_or_else(|| bad(cmd, "the attached mask is gone"))?;
            let paths: Vec<(Tick, ParamValue)> = kept
                .iter()
                .zip(&mats)
                .map(|(r, (mt, mm))| {
                    let s = r.rel_scale;
                    let to_seq = Affine::translate(r.point.x, r.point.y).then_apply(&Affine::rotate_deg(r.rel_rot)).then_apply(&Affine::scale(s, s));
                    let to_clip = mm.inverse().unwrap_or(Affine::IDENTITY).then_apply(&to_seq);
                    (*mt, ParamValue::Path(base.transformed(&to_clip)))
                })
                .collect();
            put_keys(&mut m.path, paths);
        }
    }
    Ok(if n > 1 { n } else { 0 })
}

/// Remove the keyframes an attachment wrote, keeping each parameter's value at `ph`.
fn clear_keys(pr: &mut Project, seq: ItemId, a: &Attachment, ph: Tick) {
    let Some(q) = pr.sequence_mut(seq) else { return };
    let Some((_, it)) = q.find_item_mut(a.target.clip()) else { return };
    let mt = it.effect_time_at(ph.clamp(it.start, (it.end() - Tick(1)).max(it.start)));
    let flatten = |p: &mut Param| {
        p.value = p.value_at(mt);
        p.keyframes.clear();
    };
    match &a.target {
        AttachTarget::Clip { .. } => {
            if let Some(mo) = it.effect_mut("motion") {
                for id in ["position", "scale", "scale_width", "rotation"] {
                    if let Some(p) = mo.param_mut(id) {
                        flatten(p);
                    }
                }
            }
        }
        AttachTarget::GraphicLayer { layer, .. } => {
            if let Some(li) = layer_index(it, &json!(layer)).filter(|_| *layer != 0)
                && let Some(e) = it.effects.get_mut(li)
            {
                for id in ["position", "scale", "rotation"] {
                    if let Some(p) = e.param_mut(id) {
                        flatten(p);
                    }
                }
            }
        }
        AttachTarget::EffectParam { effect, param, .. } => {
            if let Some(p) = it.effects.get_mut(*effect).and_then(|e| e.param_mut(param)) {
                flatten(p);
            }
        }
        AttachTarget::Mask { effect, mask, .. } => {
            if let Some(m) = it.effects.get_mut(*effect).and_then(|e| e.masks.get_mut(*mask)) {
                flatten(&mut m.path);
            }
        }
    }
}

#[cfg(test)]
#[path = "motion_track_tests.rs"]
mod tests;
