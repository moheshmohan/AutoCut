//! Sequence ▸ Auto Reframe Sequence… and Clip ▸ Auto Reframe (roadmap E3: the vertical factory).
//!
//! | command | does |
//! |---|---|
//! | `sequence.autoReframe` | a copy of a sequence (`sequence`, default the active one) in another shape — `aspect` `"9:16"` (the default), `"1:1"`, `"4:5"` or `"16:9"` — whose clips follow their subject with Motion ▸ Position and Scale keyframes; with `range` only that part of it, starting at zero; one undo step ("Auto Reframe Sequence") |
//! | `clip.autoReframe` | the same on clips (`clips`, default the selection, else every video clip) where they are, in the frame they are in, one undo step ("Auto Reframe"); `keepScale` keeps each clip's own Scale (a push-in already keyframed) and moves only its position |
//! | `clip.cameraTrack` | a virtual camera on one clip (`clip`, default the selected one) in its own frame: a gentle push-in (`zoom` `subtle` 115 % / `medium` 130 % / `close` 160 % of the fill scale, less where the target would no longer fit) that keeps the `target` framed — `subject` (what stands out), `box` (`box` `[x, y, w, h]` of the picture drawn at `at`, tracked both ways from there) or `face` (the face detector every half second; the largest face) — with `smooth` 0 (close) … 1 (floaty) zero-phase smoothing of the crop path; one undo step ("Camera Track") |
//!
//! Both analyse in a background job (`jobs.list` shows its progress; `jobs.cancel` stops it and
//! nothing changes): every analysed frame of each clip is decoded small (at most
//! [`ANALYSIS_SIDE`] pixels a side) and followed by [`filmcraft_render::subject::SubjectTracker`]
//! (Median Flow on pyramidal Lucas–Kanade, a correlation template that re-detects the subject after
//! an occlusion or a cut); [`filmcraft_render::reframe::plan`] turns what it saw into a smooth path
//! (critically damped, zero-phase; a camera window in `motion: "smooth"`) and the fewest keyframes
//! that reproduce it. The edit is made when the job finishes — `wait: true` makes it before the
//! command answers — and the job's `jobs.list` entry then carries `reframe`: the new sequence and,
//! per clip, whether its subject was followed and how many keyframes it got.
//!
//! `subject` seeds the tracker: a box `{x, y, w, h}` (0..1 of the picture: the top-left corner and
//! the size) for the first frame of every clip, or a list of boxes, each for one `clip` and, with
//! `seconds` (media time), for that moment — the vision model's boxes re-seed the tracker where
//! they are. Without one it starts on the frame's salient region. A clip whose subject is seen in
//! fewer than 30 % of its analysed frames gets `fallback`: `"letterbox"` (fitted whole, the
//! default), `"center"` (a centred crop) or `"blur"`: fitted whole over a blurred, dimmed,
//! enlarged copy of itself — the look the showcase gives its portrait shots ([`BACKDROP_BLURRINESS`],
//! [`BACKDROP_OPACITY`], [`BACKDROP_ZOOM`]). `"blur"` also works per stretch: in a followed clip,
//! where the subject is lost for a second or more (or a shot has nobody in it) the crop eases out
//! to the whole picture and back in when the subject returns (see [`filmcraft_render::reframe`]).
//! The copies are trimmed to those stretches and go on a [`BLUR_FILL_TRACK`] video track directly
//! under the clip's own (one an earlier run made is reused, its old copies under the clip
//! replaced); clips on such a track are never reframed themselves. A copy at a clip's edge stays
//! under the transition there ([`crate::pillars`]): it reaches on under the transition out of
//! the clip and back under the one into it, as far as its media and the copies beside it allow;
//! two copies that meet at the cut dissolve into each other with their clips; and a dip to black
//! (or white) next to a blur-filled stretch goes down to the Blur Fill track, the clips fading out
//! and in over it with Opacity keyframes, so the fill dips with the picture (on the clips' track
//! a dip's colour would cover the frame at once).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, Interpolation, ItemId, ItemKind, Label, Param, ParamValue, Sequence, TrackId, TrackItem};
use filmcraft_render::reframe::{Fallback, Follow, Geometry, Plan, Sample, Tuning, Zoom};
use filmcraft_render::subject::{SubjectBox, SubjectTracker, downscale_rgba};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, has_seq, u64_p};
use crate::{EngineError, Result, Session};

/// Longest side of the frames the tracker sees.
pub const ANALYSIS_SIDE: usize = 384;
/// Frames analysed a second by default (every frame of a 30 fps sequence or slower).
const DEFAULT_ANALYSIS_FPS: f64 = 30.0;
/// Most frames analysed for one clip (a longer clip is analysed at a lower rate).
const MAX_FRAMES_PER_CLIP: i64 = 100_000;
/// Most frames one job analyses.
const MAX_FRAMES: usize = 1_000_000;
/// Most subject boxes a command takes.
pub const MAX_SUBJECTS: usize = 10_000;
/// Longest name of a new sequence.
const MAX_NAME_CHARS: usize = 400;
/// Finished jobs' outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 256;

const SEQ_CMD: &str = "sequence.autoReframe";
const CLIP_CMD: &str = "clip.autoReframe";
const CAMERA_CMD: &str = "clip.cameraTrack";
/// How often a camera track on a face looks for it again, seconds.
const FACE_EVERY_SECONDS: f64 = 0.5;

/// The look of the blurred copy behind a picture that does not fill the frame — Auto Reframe's
/// `blur` fallback and the showcase's whole portrait shots share it: Gaussian Blur at this
/// blurriness (edges repeated, so they blur from the picture rather than from black), this
/// opacity over black (dimmed), and this much larger than the frame it fills (its blurred edges
/// outside).
pub const BACKDROP_BLURRINESS: f64 = 60.0;
pub const BACKDROP_OPACITY: f64 = 45.0;
pub const BACKDROP_ZOOM: f64 = 1.08;
/// The video track the blurred copies go on, directly under the clips' own.
pub const BLUR_FILL_TRACK: &str = "Blur Fill";
/// Most blurred copies one clip gets.
const MAX_FILLS_PER_CLIP: usize = 1000;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: SEQ_CMD,
            label: "Auto Reframe Sequence…",
            menu: &[],
            shortcut: None,
            params: r#"{"sequence":id?,"aspect":"9:16"|"1:1"|"4:5"|"16:9"="9:16","subject":{"x":0..1,"y":0..1,"w":0..1,"h":0..1,"clip":id?,"seconds":f64?}|[box]?,"motion":"smooth"|"responsive"="smooth","fallback":"letterbox"|"center"|"blur"="letterbox","range":{"start":ticks,"end":ticks}|{"startSeconds":f64,"endSeconds":f64}?,"name":str?,"analysisFps":1..60=30,"wait":bool=false}"#,
            enabled: has_seq,
            run: auto_reframe_sequence,
            journal: true,
        },
        CommandSpec {
            id: CLIP_CMD,
            label: "Auto Reframe Clips",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"subject":{"x","y","w","h","clip"?,"seconds"?}|[box]?,"motion":"smooth"|"responsive"="smooth","fallback":"letterbox"|"center"|"blur"="letterbox","keepScale":bool=false,"analysisFps":1..60=30,"wait":bool=false}"#,
            enabled: has_seq,
            run: auto_reframe_clips,
            journal: true,
        },
        CommandSpec {
            id: CAMERA_CMD,
            label: "Camera Track",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?,"target":"subject"|"box"|"face"="subject","box":[x,y,w,h]?,"at":seconds?,"zoom":"subtle"|"medium"|"close"="medium","smooth":0..1=0.5,"analysisFps":1..60=30,"wait":bool=false}"#,
            enabled: has_seq,
            run: camera_track,
            journal: true,
        },
    ]
}

/// A subject box from the parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hint {
    clip: Option<ClipId>,
    /// Media time it is for (none: the clip's first analysed frame).
    media: Option<Tick>,
    bbox: SubjectBox,
}

/// One clip to analyse.
struct Work {
    clip: ClipId,
    name: String,
    src: filmcraft_media::SharedSource,
    scale: f32,
    /// (sequence time, media time) of each analysed frame.
    frames: Vec<(Tick, Tick)>,
    hints: HashMap<usize, SubjectBox>,
    geometry: Geometry,
    /// The clip's own scale (a share) at each analysed frame, with `keepScale`.
    keep: Option<Vec<f64>>,
    item: TrackItem,
    /// One sequence frame.
    fd: Tick,
    anchor: Vec2,
    /// A camera track's push-in ([`Zoom::Push`]).
    push: Option<f64>,
    /// Track both ways from this analysed frame (a box drawn there) instead of from the first.
    seed_at: Option<usize>,
    /// A camera track on a face: the detector, run every few analysed frames for hints.
    faces: Option<(crate::redact::DetectFn, usize)>,
}

/// What the analysis made of one clip, in media time.
#[derive(Clone, Debug)]
pub struct ClipReframe {
    pub clip: ClipId,
    pub name: String,
    /// (media time, position, scale share, hold).
    keys: Vec<(Tick, Vec2, f64, bool)>,
    animate_position: bool,
    animate_scale: bool,
    pub letterbox: bool,
    pub found: f64,
    pub shots: usize,
    pub frames: usize,
    anchor: Vec2,
    keep_scale: bool,
    pub fallback: Fallback,
    /// Where (sequence time) the clip gets the blurred copy under it.
    pub fill: Vec<(Tick, Tick)>,
    /// The copy's Motion: the frame's centre and its scale (a share).
    fill_position: Vec2,
    fill_scale: f64,
}

impl ClipReframe {
    fn to_json(&self) -> Value {
        let secs = |t: Tick| (t.seconds() * 1000.0).round() / 1000.0;
        json!({
            "clip": self.clip.0,
            "name": self.name,
            "followed": !self.letterbox,
            "letterbox": self.letterbox,
            "fallback": self.fallback.name(),
            "blurFill": self.fill.iter().map(|(a, b)| [secs(*a), secs(*b)]).collect::<Vec<_>>(),
            "found": (self.found * 1000.0).round() / 1000.0,
            "keyframes": if self.animate_position || self.animate_scale { self.keys.len() } else { 0 },
            "shots": self.shots,
            "frames": self.frames,
        })
    }
}

/// Where the edit goes.
enum Target {
    /// A new sequence: `seq` (prepared when the command ran) with the keyframes added.
    Copy { seq: Box<Sequence>, name: String, next_id: u64, label: Label, bin: Option<filmcraft_project::BinId>, aspect: String },
    /// The clips of sequence `seq`, where they are, as one undo step `undo`.
    InPlace { seq: ItemId, undo: &'static str },
}

type Results = Arc<Mutex<Option<std::result::Result<Vec<ClipReframe>, String>>>>;

/// A running analysis; applied by [`poll`] when it finishes.
pub struct PendingReframe {
    job: u64,
    cmd: &'static str,
    target: Target,
    results: Results,
}

/// The session's Auto Reframe jobs and what the finished ones made.
#[derive(Default)]
pub struct Reframes {
    pending: Vec<PendingReframe>,
    outcomes: HashMap<u64, Value>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------------------------

fn follow_p(cmd: &str, p: &Value) -> Result<Follow> {
    match p.get("motion").filter(|v| !v.is_null()) {
        None => Ok(Follow::Smooth),
        Some(Value::String(s)) => Follow::from_name(s).ok_or_else(|| bad(cmd, format!("`motion` is \"smooth\" or \"responsive\", not \"{s}\""))),
        Some(_) => Err(bad(cmd, "`motion` is \"smooth\" or \"responsive\"")),
    }
}

fn fallback_p(cmd: &str, p: &Value) -> Result<Fallback> {
    match p.get("fallback").filter(|v| !v.is_null()) {
        None => Ok(Fallback::Letterbox),
        Some(Value::String(s)) => Fallback::from_name(s).ok_or_else(|| bad(cmd, format!("`fallback` is \"letterbox\", \"center\" or \"blur\", not \"{s}\""))),
        Some(_) => Err(bad(cmd, "`fallback` is \"letterbox\", \"center\" or \"blur\"")),
    }
}

fn analysis_fps_p(cmd: &str, p: &Value) -> Result<f64> {
    match p.get("analysisFps").filter(|v| !v.is_null()) {
        None => Ok(DEFAULT_ANALYSIS_FPS),
        Some(v) => v
            .as_f64()
            .filter(|f| f.is_finite() && (1.0..=60.0).contains(f))
            .ok_or_else(|| bad(cmd, "`analysisFps` is a number of frames a second from 1 to 60")),
    }
}

/// `subject`: a box, a list of boxes, `"auto"` or nothing.
fn hints_p(cmd: &str, p: &Value) -> Result<Vec<Hint>> {
    let v = match p.get("subject") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("auto") => return Ok(Vec::new()),
        Some(v) => v,
    };
    let list: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => vec![v],
        _ => return Err(bad(cmd, "`subject` is a box {x, y, w, h} (0..1 of the picture) or a list of them")),
    };
    if list.len() > MAX_SUBJECTS {
        return Err(bad(cmd, format!("at most {MAX_SUBJECTS} subject boxes")));
    }
    list.into_iter()
        .map(|b| {
            let num = |k: &str| b.get(k).and_then(Value::as_f64).filter(|f| f.is_finite());
            let (Some(x), Some(y), Some(w), Some(h)) = (num("x"), num("y"), num("w"), num("h")) else {
                return Err(bad(cmd, "each subject box needs numbers x, y, w and h (0..1 of the picture)"));
            };
            if !(w > 0.0 && w <= 1.0 && h > 0.0 && h <= 1.0 && (-1.0..=2.0).contains(&x) && (-1.0..=2.0).contains(&y)) {
                return Err(bad(cmd, "a subject box's w and h are 0..1 of the picture, and x and y near it"));
            }
            let bbox = SubjectBox::from_corner(x, y, w, h).ok_or_else(|| bad(cmd, "a subject box is outside the picture"))?;
            let clip = match b.get("clip").filter(|v| !v.is_null()) {
                None => None,
                Some(c) => Some(ClipId(c.as_u64().ok_or_else(|| bad(cmd, "a subject box's `clip` is a clip id"))?)),
            };
            let media = match b.get("seconds").filter(|v| !v.is_null()) {
                None => None,
                Some(s) => {
                    let s = s
                        .as_f64()
                        .filter(|s| s.is_finite() && (0.0..=1e7).contains(s))
                        .ok_or_else(|| bad(cmd, "a subject box's `seconds` is a media time in seconds"))?;
                    Some(Tick::from_seconds_f64(s))
                }
            };
            Ok(Hint { clip, media, bbox })
        })
        .collect()
}

/// `range`: `{start, end}` in ticks or `{startSeconds, endSeconds}`.
fn range_p(cmd: &str, p: &Value) -> Result<Option<(Tick, Tick)>> {
    let Some(r) = p.get("range").filter(|v| !v.is_null()) else { return Ok(None) };
    let at = |ticks: &str, secs: &str| -> Option<Tick> {
        r.get(ticks)
            .and_then(Value::as_i64)
            .map(Tick)
            .or_else(|| r.get(secs).and_then(Value::as_f64).filter(|s| s.is_finite() && s.abs() < 1e9).map(Tick::from_seconds_f64))
    };
    match (at("start", "startSeconds"), at("end", "endSeconds")) {
        (Some(a), Some(b)) if a >= Tick::ZERO && b > a => Ok(Some((a, b))),
        (Some(_), Some(_)) => Err(bad(cmd, "`range` must end after it starts, at zero or later")),
        _ => Err(bad(cmd, "`range` is {start, end} in ticks or {startSeconds, endSeconds}")),
    }
}

// ---------------------------------------------------------------------------------------------
// preparing
// ---------------------------------------------------------------------------------------------

/// Trim a copy of a sequence to `[a, b)`, moved to start at zero: everything after `b` and before
/// `a` is extracted on every track (captions too); markers follow, In/Out go.
fn trim_to(q: &mut Sequence, a: Tick, b: Tick, project: &filmcraft_project::Project, media: &crate::media_pool::MediaPool, next_id: &mut u64) {
    let rate = q.settings.frame_rate;
    let (a, b) = (rate.snap(a), rate.snap_nearest(b).max(rate.snap(a) + rate.frame_duration()));
    // locks and sync locks would keep parts of the copy out of the trim
    let flags: Vec<(TrackId, bool, bool)> = q.all_tracks().map(|t| (t.id, t.locked, t.sync_lock)).collect();
    let cflags: Vec<(bool, bool)> = q.caption_tracks.iter().map(|t| (t.locked, t.sync_lock)).collect();
    for t in q.all_tracks_mut() {
        t.locked = false;
        t.sync_lock = true;
    }
    for t in q.caption_tracks.iter_mut() {
        t.locked = false;
        t.sync_lock = true;
    }
    let ids: Vec<TrackId> = q.all_tracks().map(|t| t.id).collect();
    let durations = |id: ItemId| crate::media_duration(project, media, id);
    let starts = |id: ItemId| crate::media_start(project, id);
    let mut ctx = filmcraft_edit::EditCtx { next_id, media_duration: &durations, media_start: &starts, min_duration: rate.frame_duration() };
    let end = q.duration();
    if end > b {
        filmcraft_edit::extract(q, &ids, TimeRange::from_bounds(b, end), &mut ctx);
    }
    if a > Tick::ZERO {
        filmcraft_edit::extract(q, &ids, TimeRange::from_bounds(Tick::ZERO, a), &mut ctx);
    }
    for t in q.all_tracks_mut() {
        if let Some((_, l, s)) = flags.iter().find(|f| f.0 == t.id) {
            t.locked = *l;
            t.sync_lock = *s;
        }
    }
    for (t, (l, s)) in q.caption_tracks.iter_mut().zip(cflags) {
        t.locked = l;
        t.sync_lock = s;
    }
    q.markers.retain(|m| m.start >= a && m.start < b);
    for m in &mut q.markers {
        m.start -= a;
    }
    q.mark_in = None;
    q.mark_out = None;
    q.work_area = None;
}

/// The clips of `q` to analyse (all video clips, or those in `only`).
fn build_work(s: &Session, q: &Sequence, only: Option<&[ClipId]>, hints: &[Hint], analysis_fps: f64, keep_scale: bool) -> Result<Vec<Work>> {
    let rate = q.settings.frame_rate.sane();
    let fd = rate.frame_duration().max(Tick(1));
    let step = ((rate.as_f64() / analysis_fps).round() as i64).max(1);
    let frame = (q.settings.width as f64, q.settings.height as f64);
    let mut work = Vec::new();
    let mut total = 0usize;
    // the blurred copies an earlier run made are not reframed themselves
    for t in q.video_tracks.iter().filter(|t| t.name != BLUR_FILL_TRACK) {
        for it in &t.items {
            if only.is_some_and(|o| !o.contains(&it.id)) || !it.enabled {
                continue;
            }
            let Some(pi) = s.project.item(it.item) else { continue };
            let is_media = match &pi.kind {
                ItemKind::Media(m) => m.info.has_video(),
                ItemKind::Subclip { .. } => s.project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.has_video()),
                _ => false,
            };
            if !is_media {
                continue;
            }
            let Some(src) = s.source(it.item) else { continue };
            let Some((w, h)) = filmcraft_render::source_size(&s.project, it.item) else { continue };
            let Some(display) = s.project.conformed_source_size(it.item, q.settings.par) else { continue };
            let n = (it.duration.0 / fd.0).max(1);
            let mut stride = step;
            while n / stride > MAX_FRAMES_PER_CLIP {
                stride = stride.saturating_mul(2);
            }
            let mut idx: Vec<i64> = (0..n).step_by(usize::try_from(stride).unwrap_or(usize::MAX).max(1)).collect();
            if idx.last().is_some_and(|l| *l != n - 1) {
                idx.push(n - 1);
            }
            let frames: Vec<(Tick, Tick)> = idx
                .iter()
                .map(|j| {
                    let st = it.start + Tick(fd.0.saturating_mul(*j));
                    (st, it.source_time_at(st))
                })
                .collect();
            total = total.saturating_add(frames.len());
            if total > MAX_FRAMES {
                return Err(EngineError::Other(format!("too many frames to analyse (more than {MAX_FRAMES}); lower analysisFps or reframe a shorter range")));
            }
            let mut mine: HashMap<usize, SubjectBox> = HashMap::new();
            for h in hints.iter().filter(|h| h.clip.is_none_or(|c| c == it.id)) {
                let k = match h.media {
                    None => 0,
                    Some(mt) => {
                        let (lo, hi) = frames.iter().fold((Tick(i64::MAX), Tick(i64::MIN)), |(lo, hi), f| (lo.min(f.1), hi.max(f.1)));
                        if mt < lo - fd || mt > hi + fd {
                            continue;
                        }
                        frames.iter().enumerate().min_by_key(|(_, f)| (f.1 - mt).0.unsigned_abs()).map_or(0, |(k, _)| k)
                    }
                };
                mine.insert(k, h.bbox);
            }
            let keep = keep_scale.then(|| {
                let fit = (frame.0 / display.0.max(1.0)).min(frame.1 / display.1.max(1.0));
                frames
                    .iter()
                    .map(|(st, _)| {
                        let mt = it.effect_time_at(*st);
                        let scale = it.effect("motion").map_or(100.0, |m| m.f64_at("scale", mt)) / 100.0;
                        if it.scale_to_frame { scale * fit } else { scale }
                    })
                    .collect()
            });
            let scale = (ANALYSIS_SIDE as f64 / w.max(h).max(1) as f64).min(1.0) as f32;
            work.push(Work {
                clip: it.id,
                name: it.name.clone(),
                src,
                scale,
                frames,
                hints: mine,
                geometry: Geometry { picture: display, frame },
                keep,
                item: it.clone(),
                fd,
                anchor: Vec2::new(w as f64 / 2.0, h as f64 / 2.0),
                push: None,
                seed_at: None,
                faces: None,
            });
        }
    }
    Ok(work)
}

/// Follow one clip's subject through its analysed frames `order` (indices into `w.frames`).
fn follow_frames(w: &Work, order: &[usize], progress: &mut dyn FnMut(usize) -> bool) -> Option<Vec<(usize, Sample)>> {
    let mut tr = SubjectTracker::new();
    let mut out = Vec::with_capacity(order.len());
    for &k in order {
        if !progress(k) {
            return None;
        }
        let Some((st, mt)) = w.frames.get(k) else { continue };
        let mut hint = w.hints.get(&k).copied();
        let t = st.seconds();
        let obs = match w.src.video_frame(filmcraft_media::FrameRequest { time: *mt, scale: w.scale }) {
            Ok(f) => {
                let rgba = f.to_rgba8();
                match downscale_rgba(&rgba, f.width as usize, f.height as usize, ANALYSIS_SIDE) {
                    Some((small, sw, sh)) => {
                        if let Some((detect, every)) = &w.faces
                            && k.is_multiple_of((*every).max(1))
                        {
                            hint = largest_face(detect, &small, sw, sh).or(hint);
                        }
                        tr.push(&small, sw, sh, t, hint)
                    }
                    None => tr.push(&[], 0, 0, t, hint),
                }
            }
            // a frame that does not decode is a frame where the subject is not seen
            Err(_) => tr.push(&[], 0, 0, t, hint),
        };
        out.push((k, Sample { t, bbox: obs.bbox, confidence: obs.confidence, cut: obs.cut }));
    }
    Some(out)
}

/// The largest face the detector finds in an RGBA frame, as a subject box (with a little room
/// around it).
fn largest_face(detect: &crate::redact::DetectFn, rgba: &[u8], w: usize, h: usize) -> Option<SubjectBox> {
    let faces = detect(rgba, w, h, w.max(h));
    let best = faces.iter().filter(|d| d.kind == filmcraft_vision::Kind::Face).max_by(|a, b| (a.bbox[2] * a.bbox[3]).total_cmp(&(b.bbox[2] * b.bbox[3])))?;
    let [x, y, bw, bh] = best.bbox;
    SubjectBox::new(f64::from(x + bw / 2.0), f64::from(y + bh / 2.0), f64::from(bw) * 1.4, f64::from(bh) * 1.4)
}

/// Follow one clip's subject and plan its keyframes.
fn analyse(w: &Work, tuning: Tuning, fallback: Fallback, progress: &mut dyn FnMut(usize) -> bool) -> Option<ClipReframe> {
    let n = w.frames.len();
    let samples: Vec<Sample> = match w.seed_at.filter(|k| *k < n) {
        // from the frame the box was drawn on, back to the start, then on to the end
        Some(k) => {
            let back: Vec<usize> = (0..=k).rev().collect();
            let fwd: Vec<usize> = (k..n).collect();
            let mut b = follow_frames(w, &back, progress)?;
            let f = follow_frames(w, &fwd, progress)?;
            b.reverse();
            // the backward run's cuts are on the frame after the change: one earlier going forward
            let mut cuts: Vec<bool> = b.iter().map(|(_, s)| s.cut).collect();
            let len = cuts.len();
            cuts.rotate_left(1.min(len));
            if let Some(last) = cuts.last_mut() {
                *last = false;
            }
            let mut all: Vec<Sample> = b.into_iter().zip(cuts).map(|((_, s), cut)| Sample { cut, ..s }).collect();
            all.pop();
            all.extend(f.into_iter().map(|(_, s)| s));
            all
        }
        None => follow_frames(w, &(0..n).collect::<Vec<_>>(), progress)?.into_iter().map(|(_, s)| s).collect(),
    };
    let zoom = match (&w.keep, w.push) {
        (Some(k), _) => Zoom::Keep(k),
        (None, Some(f)) => Zoom::Push(f),
        (None, None) => Zoom::Track,
    };
    let plan = filmcraft_render::reframe::plan_tuned(&samples, w.geometry, tuning, zoom, fallback);
    Some(result_of(w, &plan, fallback))
}

/// A plan's blurred-fill stretches (sample indices) in sequence time: from the first sample's
/// frame (the clip's start for the first) to the frame after the last's (the clip's end for the
/// last), merged, inside the clip.
fn fill_spans(w: &Work, spans: &[(usize, usize)]) -> Vec<(Tick, Tick)> {
    let (start, end) = (w.item.start, w.item.end());
    let mut out: Vec<(Tick, Tick)> = Vec::new();
    for &(a, b) in spans.iter().take(MAX_FILLS_PER_CLIP) {
        let from = if a == 0 { start } else { w.frames.get(a).map_or(start, |f| f.0) };
        let to = match b.checked_add(1).and_then(|k| w.frames.get(k)) {
            Some(f) => f.0,
            None => end,
        };
        let (from, to) = (from.max(start), to.min(end));
        if to <= from {
            continue;
        }
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// A plan in the clip's media time: a shot's last keyframe goes on the frame before the next
/// shot's first (the jump lands on the cut, whatever the analysis rate).
fn result_of(w: &Work, plan: &Plan, fallback: Fallback) -> ClipReframe {
    let frozen = w.item.frame_hold.is_some();
    let mut keys: Vec<(Tick, Vec2, f64, bool)> = plan
        .keys
        .iter()
        .filter_map(|k| {
            let (st, mt) = *w.frames.get(k.index)?;
            let mt = match w.frames.get(k.index + 1) {
                Some((next, _)) if k.hold && *next - w.fd > st => w.item.source_time_at(*next - w.fd),
                _ => mt,
            };
            Some((mt, Vec2::new(k.position[0], k.position[1]), k.scale, k.hold))
        })
        .collect();
    let animate = !frozen && keys.len() > 1;
    if !animate {
        keys.truncate(1);
    }
    // the blurred copy goes where the picture may not cover the frame: the plan's stretches, or
    // — a placement that does not move (a frozen frame) — the whole clip when it is fitted
    let blur = fallback == Fallback::Blur && w.keep.is_none();
    let fill = match keys.first() {
        _ if !blur => Vec::new(),
        _ if animate => fill_spans(w, &plan.fill),
        Some((_, p, s, _)) if !w.geometry.covers([p.x, p.y], *s) => vec![(w.item.start, w.item.end())],
        _ => Vec::new(),
    };
    let g = &w.geometry;
    ClipReframe {
        clip: w.clip,
        name: w.name.clone(),
        keys,
        animate_position: animate && plan.animate_position,
        animate_scale: animate && plan.animate_scale && w.keep.is_none(),
        letterbox: plan.letterbox,
        found: plan.found,
        shots: plan.shots,
        frames: w.frames.len(),
        anchor: w.anchor,
        keep_scale: w.keep.is_some(),
        fallback,
        fill,
        fill_position: Vec2::new(g.frame.0 / 2.0, g.frame.1 / 2.0),
        fill_scale: g.fill_scale() * BACKDROP_ZOOM,
    }
}

/// Make `b` — a copy of a clip — the blurred copy behind it ([`BACKDROP_BLURRINESS`] …): no link,
/// group or markers; Motion centred at `position` at `scale` (a share) about `anchor` (the
/// picture's centre); Opacity [`BACKDROP_OPACITY`] (no fades, blend modes or masks); Gaussian
/// Blur added before the intrinsic effects (its other effects stay, so its colour matches).
pub fn make_backdrop(b: &mut TrackItem, position: Vec2, scale: f64, anchor: Vec2) {
    b.link = None;
    b.group = None;
    b.markers.clear();
    b.scale_to_frame = false;
    let scale = if scale.is_finite() && scale > 0.0 { scale.min(100.0) } else { 1.0 };
    if let Some(def) = filmcraft_project::effect::find_effect("motion")
        && let Some(m) = b.effect_mut("motion")
    {
        *m = def.instance();
        set_static(m, "anchor", ParamValue::Vec2(anchor));
        set_static(m, "position", ParamValue::Vec2(position));
        set_static(m, "uniform_scale", ParamValue::Bool(true));
        set_static(m, "scale", ParamValue::Float(scale * 100.0));
    }
    if let Some(def) = filmcraft_project::effect::find_effect("opacity")
        && let Some(o) = b.effect_mut("opacity")
    {
        *o = def.instance();
        set_static(o, "opacity", ParamValue::Float(BACKDROP_OPACITY));
    }
    if let Some(def) = filmcraft_project::effect::find_effect("gaussian_blur") {
        let mut blur = def.instance();
        set_static(&mut blur, "blurriness", ParamValue::Float(BACKDROP_BLURRINESS));
        set_static(&mut blur, "repeat_edge", ParamValue::Bool(true));
        let at = b.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(b.effects.len());
        b.effects.insert(at, blur);
    }
}

/// A copy of `it` showing what it shows over `[a, e)` of the sequence (inside the clip), or None.
fn trimmed_copy(it: &TrackItem, a: Tick, e: Tick) -> Option<TrackItem> {
    let (a, e) = (a.max(it.start), e.min(it.end()));
    if e <= a {
        return None;
    }
    let mut b = it.clone();
    let speed = if it.speed.is_finite() { it.speed.abs() } else { 1.0 };
    let media = |d: Tick| Tick((d.0 as f64 * speed).round() as i64);
    b.start = a;
    b.duration = e - a;
    if it.frame_hold.is_none() {
        b.source_in = if it.reverse {
            // reversed: the piece's last frame is the clip's media at `a`
            it.source_in + media(it.duration) - media(a - it.start) - media(b.duration)
        } else {
            it.source_in + media(a - it.start)
        };
    }
    Some(b)
}

/// Where the media of each project item starts and ends (None: a still, without end), for
/// [`add_blur_fill`]'s copies reaching under transitions.
type MediaSpans = HashMap<ItemId, (Tick, Option<Tick>)>;

/// [`MediaSpans`] of the clips on `q`'s video tracks.
fn media_spans(project: &filmcraft_project::Project, media: &crate::media_pool::MediaPool, q: &Sequence) -> MediaSpans {
    q.video_tracks
        .iter()
        .flat_map(|t| t.items.iter().map(|i| i.item))
        .map(|id| (id, (crate::media_start(project, id), crate::media_duration(project, media, id))))
        .collect()
}

/// Put the blurred copies under the clips of `q` that got some ([`ClipReframe::fill`]): per
/// video track, a [`BLUR_FILL_TRACK`] directly under it — the one already there (an earlier run's,
/// its copies under these clips taken away) or a new one. The copies at a clip's edges then reach
/// under the transitions there, as far as `media` and the copies beside them allow, dissolving
/// into a neighbour's copy, and a dip there goes down to the copies, the clips fading over it
/// ([`crate::pillars`]). How many copies were made.
fn add_blur_fill(q: &mut Sequence, clips: &[ClipReframe], media: &MediaSpans, next_id: &mut u64) -> usize {
    let mut alloc = || {
        let v = *next_id;
        *next_id = next_id.saturating_add(1);
        v
    };
    let rate = q.settings.frame_rate.sane();
    let reach = |it: &TrackItem| crate::pillars::reach(it, media.get(&it.item).copied().unwrap_or((Tick::ZERO, None)), rate);
    let mut made = 0;
    // from the top down, so inserting a track leaves the indices still to do as they are
    for ti in (0..q.video_tracks.len()).rev() {
        let Some(track) = q.video_tracks.get(ti) else { continue };
        if track.name == BLUR_FILL_TRACK {
            continue;
        }
        let mut copies: Vec<TrackItem> = Vec::new();
        let mut owners: Vec<(ClipId, ClipId)> = Vec::new();
        let mut refilled: Vec<ClipId> = Vec::new();
        for r in clips.iter().filter(|r| !r.fill.is_empty() && !r.keep_scale) {
            let Some(it) = track.items.iter().find(|i| i.id == r.clip) else { continue };
            refilled.push(it.id);
            for &(a, e) in &r.fill {
                let Some(mut b) = trimmed_copy(it, a, e) else { continue };
                b.id = ClipId(alloc());
                b.name = format!("{} (blur fill)", it.name);
                make_backdrop(&mut b, r.fill_position, r.fill_scale, r.anchor);
                owners.push((b.id, it.id));
                copies.push(b);
            }
        }
        if copies.is_empty() {
            continue;
        }
        made += copies.len();
        let pictures = track.items.clone();
        let reuse = ti.checked_sub(1).filter(|b| q.video_tracks.get(*b).is_some_and(|t| t.name == BLUR_FILL_TRACK));
        let (fills, above) = match reuse {
            Some(b) => (b, ti),
            None => {
                q.video_tracks.insert(ti, filmcraft_project::Track::new(TrackId(alloc()), filmcraft_project::TrackKind::Video, BLUR_FILL_TRACK.into()));
                (ti, ti + 1)
            }
        };
        let Some(t) = q.video_tracks.get_mut(fills) else { continue };
        let cleared = crate::pillars::clear_under(t, &pictures, &refilled);
        t.items.extend(copies);
        t.sort();
        owners.extend(cleared.owners);
        crate::pillars::through_transitions(q, (above, fills), &owners, &cleared.moved, &reach, &mut alloc);
    }
    made
}

/// Set Motion parameter `id` to `v` (no keyframes), adding it when the instance lacks it.
fn set_static(m: &mut filmcraft_project::EffectInstance, id: &str, v: ParamValue) {
    match m.param_mut(id) {
        Some(prm) => {
            prm.keyframes.clear();
            prm.value = v;
        }
        None => {
            m.params.insert(id.to_string(), Param::new(v));
        }
    }
}

/// Keyframe Motion parameter `id` at `keys` (time, value, hold).
fn set_keys(m: &mut filmcraft_project::EffectInstance, id: &str, keys: &[(Tick, ParamValue, bool)]) {
    let Some(first) = keys.first() else { return };
    set_static(m, id, first.1.clone());
    let Some(prm) = m.param_mut(id) else { return };
    for (t, v, _) in keys {
        prm.put_keyframe(*t, v.clone());
    }
    for (t, _, hold) in keys {
        if *hold && let Some(k) = prm.keyframes.iter_mut().find(|k| k.time == *t) {
            k.interp = Interpolation::Hold;
        }
    }
}

/// Write a clip's reframing into its Motion.
fn apply_clip(it: &mut TrackItem, r: &ClipReframe) {
    let Some(&(_, pos0, scale0, _)) = r.keys.first() else { return };
    // a clip keeping its own scale whose subject was not found keeps its placement too
    if r.keep_scale && r.letterbox {
        return;
    }
    if !r.keep_scale {
        it.scale_to_frame = false;
    }
    let reversed = it.reverse || it.speed < 0.0;
    let Some(m) = it.effect_mut("motion") else { return };
    m.enabled = true;
    set_static(m, "anchor", ParamValue::Vec2(r.anchor));
    if r.animate_position {
        let keys: Vec<(Tick, ParamValue, bool)> = r.keys.iter().map(|k| (k.0, ParamValue::Vec2(k.1), k.3 && !reversed)).collect();
        set_keys(m, "position", &keys);
    } else {
        set_static(m, "position", ParamValue::Vec2(pos0));
    }
    if r.keep_scale {
        return;
    }
    set_static(m, "uniform_scale", ParamValue::Bool(true));
    if r.animate_scale {
        let keys: Vec<(Tick, ParamValue, bool)> = r.keys.iter().map(|k| (k.0, ParamValue::Float(k.2 * 100.0), k.3 && !reversed)).collect();
        set_keys(m, "scale", &keys);
    } else {
        set_static(m, "scale", ParamValue::Float(scale0 * 100.0));
    }
}

// ---------------------------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------------------------

fn auto_reframe_sequence(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = SEQ_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let seq_id = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => ItemId(v.as_u64().ok_or_else(|| bad(cmd, "`sequence` is a sequence id"))?),
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let item = s.project.item(seq_id).cloned().ok_or_else(|| bad(cmd, "no such sequence"))?;
    let ItemKind::Sequence(src) = &item.kind else { return Err(bad(cmd, "no such sequence")) };
    let aspect_name = match p.get("aspect").filter(|v| !v.is_null()) {
        None => "9:16".to_string(),
        Some(Value::String(a)) => a.trim().to_string(),
        Some(_) => return Err(bad(cmd, "`aspect` is \"9:16\", \"1:1\", \"4:5\" or \"16:9\"")),
    };
    let aspect = filmcraft_render::reframe::parse_aspect(&aspect_name)
        .ok_or_else(|| bad(cmd, format!("`aspect` is \"9:16\", \"1:1\", \"4:5\" or \"16:9\", not \"{aspect_name}\"")))?;
    let follow = follow_p(cmd, p)?;
    let fallback = fallback_p(cmd, p)?;
    let hints = hints_p(cmd, p)?;
    let fps = analysis_fps_p(cmd, p)?;
    let range = range_p(cmd, p)?;
    let aspect_label = format!("{}:{}", aspect.0, aspect.1);
    let name = match p.get("name").filter(|v| !v.is_null()) {
        None => format!("{} ({aspect_label})", item.name),
        Some(Value::String(n)) if !n.trim().is_empty() && n.chars().count() <= MAX_NAME_CHARS && !n.chars().any(char::is_control) => n.trim().to_string(),
        Some(_) => return Err(bad(cmd, format!("`name` is a sequence name on one line, at most {MAX_NAME_CHARS} characters"))),
    };
    let (fw, fh) =
        filmcraft_render::reframe::frame_for((src.settings.width, src.settings.height), aspect).ok_or_else(|| bad(cmd, "the sequence has no frame size"))?;
    let mut target = (**src).clone();
    target.settings.width = fw;
    target.settings.height = fh;
    let mut next_id = s.project.next_id;
    if let Some((a, b)) = range {
        if a >= target.duration() {
            return Err(bad(cmd, "`range` starts after the sequence ends"));
        }
        let media = s.media.clone();
        trim_to(&mut target, a, b, &s.project, &media, &mut next_id);
    }
    target.check().map_err(|e| bad(cmd, format!("the reframed copy is not valid: {e}")))?;
    let work = build_work(s, &target, None, &hints, fps, false)?;
    let bin = s.project.root.parent_of(seq_id).filter(|b| b.0 != 0);
    let label_text = format!("Auto Reframe {aspect_label}: {}", item.name);
    let target = Target::Copy { seq: Box::new(target), name, next_id, label: item.label, bin, aspect: aspect_label };
    start(s, cmd, label_text, target, work, follow.tuning(), fallback, p)
}

fn auto_reframe_clips(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = CLIP_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let follow = follow_p(cmd, p)?;
    let fallback = fallback_p(cmd, p)?;
    let hints = hints_p(cmd, p)?;
    let fps = analysis_fps_p(cmd, p)?;
    let keep_scale = bool_p(p, "keepScale").unwrap_or(false);
    let only: Option<Vec<ClipId>> = match (p.get("clips").filter(|v| !v.is_null()), u64_p(p, "clip")) {
        (Some(Value::Array(a)), _) => {
            if a.len() > 100_000 {
                return Err(bad(cmd, "too many clips"));
            }
            Some(a.iter().map(|v| v.as_u64().map(ClipId)).collect::<Option<Vec<_>>>().ok_or_else(|| bad(cmd, "`clips` is a list of clip ids"))?)
        }
        (Some(_), _) => return Err(bad(cmd, "`clips` is a list of clip ids")),
        (None, Some(c)) => Some(vec![ClipId(c)]),
        (None, None) if !s.state.selection.is_empty() => Some(s.state.selection.clone()),
        (None, None) => None,
    };
    let work = build_work(s, &q, only.as_deref(), &hints, fps, keep_scale)?;
    if work.is_empty() {
        return Err(bad(cmd, "no video clip to reframe (select video clips of media, or pass `clips`)"));
    }
    let n = work.len();
    let label = format!("Auto Reframe ({n} clip{})", if n == 1 { "" } else { "s" });
    let undo = if n == 1 { "Auto Reframe" } else { "Auto Reframe Clips" };
    start(s, cmd, label, Target::InPlace { seq: seq_id, undo }, work, follow.tuning(), fallback, p)
}

/// `clip.cameraTrack`: one clip's Motion keeps its target framed with a gentle push-in.
fn camera_track(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = CAMERA_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    let push = match p.get("zoom").filter(|v| !v.is_null()) {
        None => 1.3,
        Some(Value::String(z)) => match z.trim().to_ascii_lowercase().as_str() {
            "subtle" => 1.15,
            "medium" => 1.3,
            "close" => 1.6,
            _ => return Err(bad(cmd, format!("`zoom` is \"subtle\", \"medium\" or \"close\", not \"{z}\""))),
        },
        Some(_) => return Err(bad(cmd, "`zoom` is \"subtle\", \"medium\" or \"close\"")),
    };
    let smooth = match p.get("smooth").filter(|v| !v.is_null()) {
        None => 0.5,
        Some(v) => {
            v.as_f64().filter(|x| x.is_finite() && (0.0..=1.0).contains(x)).ok_or_else(|| bad(cmd, "`smooth` is a number from 0 (close) to 1 (floaty)"))?
        }
    };
    let fps = analysis_fps_p(cmd, p)?;
    let clip = match u64_p(p, "clip") {
        Some(c) => ClipId(c),
        None => s
            .state
            .selection
            .iter()
            .copied()
            .find(|c| q.video_tracks.iter().any(|t| t.item(*c).is_some()))
            .ok_or_else(|| bad(cmd, "select the video clip to camera-track (or pass `clip`)"))?,
    };
    let it = q.video_tracks.iter().find_map(|t| t.item(clip)).cloned().ok_or_else(|| bad(cmd, "no such video clip in the active sequence"))?;
    let target = match p.get("target").filter(|v| !v.is_null()) {
        None => "subject".to_string(),
        Some(Value::String(t)) => t.trim().to_ascii_lowercase(),
        Some(_) => return Err(bad(cmd, "`target` is \"subject\", \"box\" or \"face\"")),
    };
    let at = match p.get("at").filter(|v| !v.is_null()) {
        None => s.playhead().clamp(it.start, (it.end() - Tick(1)).max(it.start)),
        Some(v) => {
            let t = Tick::from_seconds_f64(
                v.as_f64().filter(|x| x.is_finite() && x.abs() <= 1e7).ok_or_else(|| bad(cmd, "`at` is a time in seconds on the sequence"))?,
            );
            if t < it.start || t >= it.end() {
                return Err(bad(cmd, "`at` is outside the clip"));
            }
            t
        }
    };
    let mut hints = Vec::new();
    let mut faces = None;
    match target.as_str() {
        "subject" => {}
        "box" => {
            let b = p
                .get("box")
                .and_then(Value::as_array)
                .filter(|a| a.len() == 4)
                .ok_or_else(|| bad(cmd, "`target: \"box\"` needs `box` [x, y, w, h] (0..1 of the picture)"))?;
            let v: Vec<f64> = b.iter().filter_map(Value::as_f64).filter(|x| x.is_finite()).collect();
            let [x, y, w, h] = v.as_slice() else { return Err(bad(cmd, "`box` is four finite numbers")) };
            if !(*w > 0.0 && *h > 0.0 && *w <= 1.0 && *h <= 1.0 && (-1.0..=2.0).contains(x) && (-1.0..=2.0).contains(y)) {
                return Err(bad(cmd, "`box` w and h are 0..1 of the picture, and x and y near it"));
            }
            let bbox = SubjectBox::from_corner(*x, *y, *w, *h).ok_or_else(|| bad(cmd, "`box` is outside the picture"))?;
            hints.push(Hint { clip: Some(clip), media: Some(it.source_time_at(at)), bbox });
        }
        "face" => {
            let detect = match s.redacts.detector.clone() {
                Some(d) => d,
                None => crate::privacy_check::face_detector(s, cmd)?,
            };
            faces = Some(detect);
        }
        other => return Err(bad(cmd, format!("`target` is \"subject\", \"box\" or \"face\", not \"{other}\""))),
    }
    let mut work = build_work(s, &q, Some(&[clip]), &hints, fps, false)?;
    let Some(w) = work.first_mut() else { return Err(bad(cmd, "the clip has no picture to track")) };
    w.push = Some(push);
    if target == "box" {
        w.seed_at = w.hints.keys().min().copied();
    }
    if let Some(d) = faces {
        let every = ((FACE_EVERY_SECONDS * fps).round() as usize).max(1);
        w.faces = Some((d, every));
    }
    let label = format!("Camera Track: {}", it.name);
    start(s, cmd, label, Target::InPlace { seq: seq_id, undo: "Camera Track" }, work, Tuning::smoothness(smooth), Fallback::Center, p)
}

#[allow(clippy::too_many_arguments)]
fn start(s: &mut Session, cmd: &'static str, label: String, target: Target, work: Vec<Work>, tuning: Tuning, fallback: Fallback, p: &Value) -> Result<Value> {
    let total: usize = work.iter().map(|w| w.frames.len()).sum();
    let clips = work.len();
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(total as u64, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let run = move || {
        let t0 = web_time::Instant::now();
        let mut done = 0u64;
        let mut made = Vec::with_capacity(work.len());
        let mut stopped = false;
        for (i, w) in work.iter().enumerate() {
            let n = w.frames.len();
            let mut tick = |k: usize| {
                if prog.cancel.load(Ordering::Relaxed) {
                    return false;
                }
                done += 1;
                prog.done.store(done, Ordering::Relaxed);
                if k.is_multiple_of(8) {
                    *lock(&prog.status) = format!("{}: frame {} of {n} (clip {} of {})", w.name, k + 1, i + 1, work.len());
                }
                true
            };
            match analyse(w, tuning, fallback, &mut tick) {
                Some(r) => made.push(r),
                None => {
                    stopped = true;
                    break;
                }
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let followed = made.iter().filter(|r| !r.letterbox).count();
        *lock(&prog.status) = if stopped {
            "Stopped: nothing was changed".into()
        } else {
            format!("Followed the subject in {followed} of {} clip(s), {done} frame(s) in {secs:.1}s", made.len())
        };
        // the results before `finished`: whoever sees the job finished can apply them
        *lock(&out) = Some(if stopped { Err("stopped".to_string()) } else { Ok(made) });
        *lock(&res) = Some(if stopped {
            Err("stopped".into())
        } else {
            Ok(filmcraft_export::Report {
                path: String::new(),
                frames: done,
                seconds: secs,
                bytes: 0,
                render_fps: done as f64 / secs.max(1e-6),
                extra_files: Vec::new(),
            })
        });
        prog.finished.store(true, Ordering::Relaxed);
    };
    s.reframes.pending.push(PendingReframe { job: id, cmd, target, results });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-auto-reframe", run)?;
    let mut answer = json!({"job": id, "clips": clips, "frames": total});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.reframes.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["reframe"] = o.clone();
        }
    }
    Ok(answer)
}

/// Apply finished jobs (one undo step each) and keep what they made for `jobs.list`. Called each
/// UI frame (through [`Session::poll_persistence`]), by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.reframes.pending.len() {
        let Some(pj) = s.reframes.pending.get(i) else { break };
        let ready = lock(&pj.results).is_some();
        let job = s.jobs.iter().find(|j| j.id == pj.job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !ready && !finished {
            i += 1;
            continue;
        }
        let pj = s.reframes.pending.remove(i);
        let results = lock(&pj.results).take();
        let outcome = match results {
            Some(Ok(clips)) if !cancelled => match apply(s, pj.target, &clips) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(pj.cmd, format!("Auto Reframe: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            _ => json!({"error": "stopped: nothing was changed"}),
        };
        s.reframes.outcomes.insert(pj.job, outcome);
        if s.reframes.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.reframes.outcomes.keys().min().copied()
        {
            s.reframes.outcomes.remove(&old);
        }
    }
}

fn apply(s: &mut Session, target: Target, clips: &[ClipReframe]) -> Result<Value> {
    let followed = clips.iter().filter(|r| !r.letterbox).count();
    let filled = clips.iter().filter(|r| !r.fill.is_empty() && !r.keep_scale).count();
    let summary: Vec<Value> = clips.iter().map(ClipReframe::to_json).collect();
    let pool = s.media.clone();
    match target {
        Target::Copy { seq, name, next_id, label, bin, aspect } => {
            let (w, h) = (seq.settings.width, seq.settings.height);
            let n = name.clone();
            let id = s.edit("Auto Reframe Sequence", move |pr, st| {
                let mut q = *seq;
                for r in clips {
                    if let Some((_, it)) = q.find_item_mut(r.clip) {
                        apply_clip(it, r);
                    }
                }
                pr.next_id = pr.next_id.max(next_id);
                let media = media_spans(pr, &pool, &q);
                add_blur_fill(&mut q, clips, &media, &mut pr.next_id);
                q.check().map_err(EngineError::Other)?;
                let id = pr.add_item(&n, label, ItemKind::Sequence(Arc::new(q)), bin);
                st.active_sequence = Some(id);
                if !st.open_sequences.contains(&id) {
                    st.open_sequences.push(id);
                }
                st.selection.clear();
                st.project_selection = vec![id];
                Ok(id)
            })?;
            s.events.push(crate::Event::OpenSequence(id));
            s.toast(format!("Auto Reframe: \"{name}\" ({aspect}), the subject followed in {followed} of {} clip(s)", clips.len()));
            Ok(json!({"sequence": id.0, "name": name, "aspect": aspect, "width": w, "height": h, "followed": followed, "blurFill": filled, "clips": summary}))
        }
        Target::InPlace { seq, undo } => {
            if s.project.sequence(seq).is_none() {
                return Err(EngineError::Other("the sequence is gone".into()));
            }
            s.edit(undo, |pr, _| {
                let mut next_id = pr.next_id;
                let media = media_spans(pr, &pool, pr.sequence(seq).ok_or(EngineError::NoSequence)?);
                let q = pr.sequence_mut(seq).ok_or(EngineError::NoSequence)?;
                for r in clips {
                    if let Some((_, it)) = q.find_item_mut(r.clip) {
                        apply_clip(it, r);
                    }
                }
                add_blur_fill(q, clips, &media, &mut next_id);
                q.check().map_err(EngineError::Other)?;
                pr.next_id = pr.next_id.max(next_id);
                Ok(())
            })?;
            Ok(json!({"sequence": seq.0, "followed": followed, "blurFill": filled, "clips": summary}))
        }
    }
}

/// What a finished Auto Reframe job made (`reframe` in its `jobs.list` entry).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.reframes.outcomes.get(&job).cloned()
}

#[cfg(test)]
#[path = "reframe_tests.rs"]
mod tests;
