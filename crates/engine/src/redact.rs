//! Clip ▸ Auto Redact Faces and Plates… — the automatic privacy blur.
//!
//! | command | does |
//! |---|---|
//! | `clip.autoRedact` | job: find clearly recognisable faces and legible number plates in clips (`clips`, default the selected video clips) and hide each behind a tracked mask on a blur (or mosaic) effect; one undo step ("Auto Redact") |
//! | `sequence.autoRedact` | the same on every video clip of `sequence` (default the active one) |
//! | `redact.downloadModels` | download the two detectors (4.4 MB, SHA-256 checked) as a background job, after the user agreed |
//! | `redact.status` | whether the detectors are installed (and their licences and sizes), and whether the vision check is available |
//!
//! **How.** In a background job (`jobs.list` progress; `jobs.cancel` stops it and nothing
//! changes) each clip's used part is decoded and analysed `analysisFps` times a second (10 by
//! default) with `filmcraft-vision`'s YuNet face and LPD-YuNet plate detectors (pure Rust on
//! candle, CPU). Detections are linked into tracks, gaps filled and the paths smoothed
//! (`filmcraft_vision::track`). A face is hidden when it is at least `minFaceHeight` of the
//! sequence's frame tall at some point as the sequence shows it (5 %: smaller faces in the crowd
//! are not recognisable; Motion ▸ Scale and Scale to Frame Size count, so a clip zoomed in or a
//! portrait clip filling a landscape frame has its faces counted larger, and the face detector
//! looks at it closer), a plate when it is at least `minPlateWidth` of the frame's width
//! (2.5 %); either must be seen twice, or once with a score of 0.9, and its centre must not be
//! inside an `exclude` box.
//!
//! **Not robots.** With `verify` (the default) every track still to be hidden is shown once,
//! cropped with its surroundings, to a local vision model (the host's [`FaceJudge`]: AutoCut's
//! Qwen2.5-VL through Ollama): a face is hidden only if it is a real person's (not a humanoid
//! robot, mannequin, statue, poster or screen), a plate only if it is a vehicle's (not a sign or
//! a robot's name badge); an unclear answer hides it. Without a judge the check is skipped and
//! noted: everything detected is hidden. Plates must also have a plate's shape (wider than tall,
//! not huge: `filmcraft_vision::post::plausible_plate`).
//!
//! **The edit.** Each clip with something to hide gets one Gaussian Blur (or Mosaic, `style`)
//! effect, its strength from the largest face or plate, with one mask per track — an ellipse for
//! a face, the plate's own quadrilateral for a plate — grown by `padding` (a fifth of the box
//! on each side), feathered outward, with a Mask Path keyframe at every analysed frame and
//! its Opacity switched on and off (hold keyframes) where the track starts and ends. Masks are in
//! clip pixels and the effect applies before Motion, so the blur follows the clip's own motion,
//! scale, crop and reframing. The masks are named `Redact face 1`, `Redact plate 2`…; running
//! it again replaces an earlier automatic redaction of the clip (`replace: false` keeps it).
//! `dryRun` only reports the tracks (the job's `redact` outcome), changing nothing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, EffectInstance, Interpolation, ItemId, ItemKind, Keyframe, Mask, MaskPath, Param, ParamValue, TrackKind};
use filmcraft_time::Tick;
use filmcraft_vision::Kind;
use filmcraft_vision::track::{Exclude, KeepParams, LinkParams, Skip, Track};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, has_seq};
use crate::{EngineError, Result, Session};

/// The undo step's name.
pub const LABEL: &str = "Auto Redact";
/// What the masks this command makes are called (`Redact face 1`).
pub const MASK_PREFIX: &str = "Redact ";
const CLIP_CMD: &str = "clip.autoRedact";
const SEQ_CMD: &str = "sequence.autoRedact";
const DOWNLOAD_CMD: &str = "redact.downloadModels";
/// Most frames one job analyses.
const MAX_FRAMES: usize = 1_000_000;
/// Most exclusion boxes a command takes.
const MAX_EXCLUDE: usize = 10_000;
/// Most clips a command takes.
const MAX_CLIPS: usize = 100_000;
/// Finished jobs' outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 256;
/// Crops the vision check is shown at once.
pub const JUDGE_BATCH: usize = 6;
/// Longest side of a crop for the vision check.
pub const CROP_SIDE: u32 = 224;
/// A crop shows this many times the face box's size, centred on it (the head and shoulders
/// around a face tell a robot from a person).
const CROP_CONTEXT: f64 = 2.2;

/// A picture for the vision check: RGBA, `width` × `height`.
#[derive(Clone, Debug, PartialEq)]
pub struct Crop {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// The vision check: whether a detected face is a real person's (and a detected plate a
/// vehicle's). Hosts install one ([`Redacts::judge`]); AutoCut's asks Qwen2.5-VL through a local
/// Ollama.
pub trait FaceJudge: Send + Sync {
    /// Which model answers ("ollama:qwen2.5vl:7b").
    fn name(&self) -> String;
    /// For each crop of a `kind` detection, in order. Faces: `Some(true)` a real person's face,
    /// `Some(false)` something that looks like one (a humanoid robot, mannequin, statue, poster,
    /// a face on a screen). Plates (the plate is outlined in red in the crop): `Some(true)` a
    /// vehicle's number plate, `Some(false)` something else (a sign, a label or name on a
    /// machine). `None`: cannot tell. An error means the check could not run at all.
    fn judge(&self, kind: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String>;
}

/// The effect the masks go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Blur,
    Mosaic,
}

/// What to look for and how to hide it.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub faces: bool,
    pub plates: bool,
    pub keep: KeepParams,
    pub link: LinkParams,
    /// Detector confidence a face (plate) needs.
    pub face_score: f32,
    pub plate_score: f32,
    pub style: Style,
    /// 1..100.
    pub strength: f64,
    /// Share of the box added on every side.
    pub padding: f64,
    pub verify: bool,
    pub dry_run: bool,
    /// Frames analysed a second.
    pub analysis_fps: f64,
    /// Longest side of the picture the face (plate) detector sees.
    pub face_side: usize,
    pub plate_side: usize,
    /// Remove an earlier automatic redaction of the clip first.
    pub replace: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            faces: true,
            plates: true,
            keep: KeepParams::default(),
            link: LinkParams::default(),
            face_score: 0.6,
            plate_score: 0.8,
            style: Style::Blur,
            strength: 60.0,
            padding: 0.2,
            verify: true,
            dry_run: false,
            analysis_fps: 10.0,
            face_side: 640,
            plate_side: 640,
            replace: true,
        }
    }
}

/// An exclusion box from the parameters, for one clip or all.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ExcludeP {
    clip: Option<ClipId>,
    ex: Exclude,
}

/// One clip to analyse.
struct Work {
    clip: ClipId,
    /// Other clips showing the same frames (a blurred background copy of a clip): they share
    /// this analysis and get the same masks.
    also: Vec<ClipId>,
    item: ItemId,
    name: String,
    src: filmcraft_media::SharedSource,
    /// Media time of each analysed frame, in order, without repeats.
    times: Vec<Tick>,
    /// Clip pixels (the mask space).
    size: (f64, f64),
    /// Scale of the frames asked for (the detectors see at most `max(face_side, plate_side)`).
    scale: f32,
    /// The size rules for this clip: the command's, measured on the picture as the sequence
    /// shows it (a zoomed-in clip's faces count larger).
    keep: KeepParams,
    /// Longest side of the picture the face detector sees for this clip (larger for a zoomed-in
    /// clip, so its smallest face to hide is still big enough to find).
    face_side: usize,
    exclude: Vec<Exclude>,
    /// The clip's media range (the masks' lead-in and lead-out stay inside it).
    media: (Tick, Tick),
}

/// What the analysis made of one clip.
#[derive(Clone, Debug)]
pub struct ClipRedact {
    pub clip: ClipId,
    /// Clips sharing this analysis ([`Work::also`]).
    pub also: Vec<ClipId>,
    pub name: String,
    /// Clip pixels.
    pub size: (f64, f64),
    /// Each track and why it is not hidden (`None`: hidden).
    pub tracks: Vec<(Track, Option<Skip>)>,
    pub frames: usize,
    /// Seconds between analysed frames.
    pub interval: f64,
    pub media: (Tick, Tick),
    /// Frames that did not decode.
    pub missed: usize,
}

impl ClipRedact {
    fn hidden(&self) -> impl Iterator<Item = &Track> {
        self.tracks.iter().filter(|(_, s)| s.is_none()).map(|(t, _)| t)
    }
}

/// How the vision check went.
#[derive(Clone, Debug, PartialEq)]
enum Verified {
    Off,
    Done { judge: String, asked: usize, not_people: usize, not_plates: usize, unsure: usize },
    Skipped(String),
    Failed(String),
}

impl Verified {
    fn to_json(&self) -> Value {
        match self {
            Verified::Off => json!({"state": "off"}),
            Verified::Done { judge, asked, not_people, not_plates, unsure } => {
                json!({"state": "done", "judge": judge, "asked": asked, "notPeople": not_people, "notPlates": not_plates, "unsure": unsure})
            }
            Verified::Skipped(why) => json!({"state": "skipped", "reason": why}),
            Verified::Failed(why) => json!({"state": "failed", "reason": why}),
        }
    }
}

/// What a job found, how the vision check went, and how long the analysis and the check took (seconds).
type Found = (Vec<ClipRedact>, Verified, [f64; 2]);
type Results = Arc<Mutex<Option<std::result::Result<Found, String>>>>;

/// A running analysis; applied by [`poll`] when it finishes.
pub struct PendingRedact {
    job: u64,
    cmd: &'static str,
    seq: ItemId,
    opts: Options,
    skipped: Vec<Value>,
    results: Results,
}

/// The session's redaction jobs, what the finished ones made, and the host's vision check.
#[derive(Default)]
pub struct Redacts {
    pending: Vec<PendingRedact>,
    outcomes: HashMap<u64, Value>,
    /// The vision check (None: not available; faces are hidden unchecked).
    pub judge: Option<Arc<dyn FaceJudge>>,
    /// A detector used instead of the downloaded models (tests; a host with its own).
    pub detector: Option<DetectFn>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether this build has the detectors (feature `privacy-models`).
pub fn available() -> bool {
    cfg!(feature = "privacy-models")
}

/// Whether the detectors are downloaded.
pub fn installed() -> bool {
    crate::transcript::models_dir().is_some_and(|d| filmcraft_vision::catalog::installed(&d))
}

// ---------------------------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------------------------

fn unit_p(cmd: &str, p: &Value, k: &str, lo: f64, hi: f64, d: f64) -> Result<f64> {
    match p.get(k).filter(|v| !v.is_null()) {
        None => Ok(d),
        Some(v) => v.as_f64().filter(|f| f.is_finite() && (lo..=hi).contains(f)).ok_or_else(|| bad(cmd, format!("`{k}` is a number from {lo} to {hi}"))),
    }
}

/// `kinds`: `["faces", "plates"]`, `"faces,plates"`, `"faces"`…
fn kinds_p(cmd: &str, p: &Value) -> Result<(bool, bool)> {
    let names: Vec<String> = match p.get("kinds").filter(|v| !v.is_null()) {
        None => return Ok((true, true)),
        Some(Value::String(s)) => s.split(',').map(str::to_string).collect(),
        Some(Value::Array(a)) if a.len() <= 8 => {
            a.iter().map(|v| v.as_str().map(str::to_string)).collect::<Option<_>>().ok_or_else(|| bad(cmd, "`kinds` lists \"faces\" and/or \"plates\""))?
        }
        Some(_) => return Err(bad(cmd, "`kinds` lists \"faces\" and/or \"plates\"")),
    };
    let (mut faces, mut plates) = (false, false);
    for n in names.iter().map(|n| n.trim()).filter(|n| !n.is_empty()) {
        match Kind::from_name(n) {
            Some(Kind::Face) => faces = true,
            Some(Kind::Plate) => plates = true,
            None => return Err(bad(cmd, format!("`kinds`: \"{}\" is neither \"faces\" nor \"plates\"", n.chars().take(40).collect::<String>()))),
        }
    }
    if !faces && !plates {
        return Err(bad(cmd, "`kinds` needs \"faces\", \"plates\" or both"));
    }
    Ok((faces, plates))
}

/// The options of a command's parameters.
pub fn options_p(cmd: &str, p: &Value) -> Result<Options> {
    let d = Options::default();
    let (faces, plates) = kinds_p(cmd, p)?;
    let style = match p.get("style").filter(|v| !v.is_null()) {
        None => d.style,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("blur") => Style::Blur,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("mosaic") || s.eq_ignore_ascii_case("pixelate") => Style::Mosaic,
        Some(_) => return Err(bad(cmd, "`style` is \"blur\" or \"mosaic\"")),
    };
    let flag = |k: &str, d: bool| -> Result<bool> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(d),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(bad(cmd, format!("`{k}` is true or false"))),
        }
    };
    let keep = KeepParams {
        min_face_height: unit_p(cmd, p, "minFaceHeight", 0.0, 1.0, d.keep.min_face_height)?,
        min_plate_width: unit_p(cmd, p, "minPlateWidth", 0.0, 1.0, d.keep.min_plate_width)?,
        ..d.keep
    };
    Ok(Options {
        faces,
        plates,
        keep,
        link: d.link,
        face_score: unit_p(cmd, p, "faceScore", 0.05, 1.0, f64::from(d.face_score))? as f32,
        plate_score: unit_p(cmd, p, "plateScore", 0.05, 1.0, f64::from(d.plate_score))? as f32,
        style,
        strength: unit_p(cmd, p, "strength", 1.0, 100.0, d.strength)?,
        padding: unit_p(cmd, p, "padding", 0.0, 1.0, d.padding)?,
        verify: flag("verify", d.verify)?,
        dry_run: flag("dryRun", d.dry_run)?,
        analysis_fps: unit_p(cmd, p, "analysisFps", 1.0, 60.0, d.analysis_fps)?,
        face_side: unit_p(cmd, p, "faceSide", 160.0, 1920.0, d.face_side as f64)? as usize,
        plate_side: unit_p(cmd, p, "plateSide", 160.0, 2560.0, d.plate_side as f64)? as usize,
        replace: flag("replace", d.replace)?,
    })
}

/// `exclude`: a box `{x, y, w, h}` (0..1 of the clip's picture; `clip` and `seconds` of media
/// time optional) or a list of them.
fn exclude_p(cmd: &str, p: &Value) -> Result<Vec<ExcludeP>> {
    let list: Vec<&Value> = match p.get("exclude").filter(|v| !v.is_null()) {
        None => return Ok(Vec::new()),
        Some(Value::Array(a)) => a.iter().collect(),
        Some(v @ Value::Object(_)) => vec![v],
        Some(_) => return Err(bad(cmd, "`exclude` is a box {x, y, w, h} (0..1 of the picture) or a list of them")),
    };
    if list.len() > MAX_EXCLUDE {
        return Err(bad(cmd, format!("at most {MAX_EXCLUDE} exclusion boxes")));
    }
    list.into_iter()
        .map(|b| {
            let num = |k: &str| b.get(k).and_then(Value::as_f64).filter(|f| f.is_finite());
            let (Some(x), Some(y), Some(w), Some(h)) = (num("x"), num("y"), num("w"), num("h")) else {
                return Err(bad(cmd, "each exclusion box needs numbers x, y, w and h (0..1 of the picture)"));
            };
            if !(w > 0.0 && w <= 2.0 && h > 0.0 && h <= 2.0 && (-1.0..=2.0).contains(&x) && (-1.0..=2.0).contains(&y)) {
                return Err(bad(cmd, "an exclusion box's w and h are 0..1 of the picture, and x and y near it"));
            }
            let clip = match b.get("clip").filter(|v| !v.is_null()) {
                None => None,
                Some(c) => Some(ClipId(c.as_u64().ok_or_else(|| bad(cmd, "an exclusion box's `clip` is a clip id"))?)),
            };
            let t = match b.get("seconds").filter(|v| !v.is_null()) {
                None => None,
                Some(s) => Some(
                    s.as_f64()
                        .filter(|s| s.is_finite() && (0.0..=1e7).contains(s))
                        .ok_or_else(|| bad(cmd, "an exclusion box's `seconds` is a media time in seconds"))?,
                ),
            };
            Ok(ExcludeP { clip, ex: Exclude { t, bbox: [x, y, w, h] } })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// preparing
// ---------------------------------------------------------------------------------------------

/// The clips of sequence `seq` to analyse: video clips of media (all, or those in `only`), and
/// why others in `only` are left out.
fn build_work(s: &Session, seq: ItemId, only: Option<&[ClipId]>, o: &Options, exclude: &[ExcludeP]) -> Result<(Vec<Work>, Vec<Value>)> {
    let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?;
    let rate = q.settings.frame_rate.sane();
    let fd = rate.frame_duration().max(Tick(1));
    let step = ((rate.as_f64() / o.analysis_fps).round() as i64).max(1);
    let (mut work, mut skipped) = (Vec::new(), Vec::new());
    let mut total = 0usize;
    for t in &q.video_tracks {
        for it in &t.items {
            if only.is_some_and(|o| !o.contains(&it.id)) {
                continue;
            }
            let skip = |why: &str| json!({"clip": it.id.0, "name": it.name, "reason": why});
            if !it.enabled {
                skipped.push(skip("disabled"));
                continue;
            }
            if t.locked {
                skipped.push(skip("on a locked track"));
                continue;
            }
            let Some(pi) = s.project.item(it.item) else { continue };
            let is_media = match &pi.kind {
                ItemKind::Media(m) => m.info.has_video(),
                ItemKind::Subclip { .. } => s.project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.has_video()),
                ItemKind::Sequence(_) => {
                    skipped.push(skip("a nested sequence: redact the clips inside it"));
                    continue;
                }
                _ => false,
            };
            if !is_media {
                if only.is_some() {
                    skipped.push(skip("not video footage"));
                }
                continue;
            }
            let Some(src) = s.source(it.item) else {
                skipped.push(skip("its media is offline"));
                continue;
            };
            let Some((w, h)) = filmcraft_render::source_size(&s.project, it.item).filter(|(w, h)| *w > 0 && *h > 0) else { continue };
            let n = (it.duration.0 / fd.0).max(1);
            let mut times: Vec<Tick> =
                (0..n).step_by(usize::try_from(step).unwrap_or(1).max(1)).map(|j| it.source_time_at(it.start + Tick(fd.0.saturating_mul(j)))).collect();
            times.push(it.source_time_at(it.start + Tick(fd.0.saturating_mul(n - 1))));
            times.sort();
            times.dedup();
            total = total.saturating_add(times.len());
            if total > MAX_FRAMES {
                return Err(EngineError::Other(format!("too many frames to analyse (more than {MAX_FRAMES}); lower analysisFps or redact fewer clips")));
            }
            let media = (times.first().copied().unwrap_or(it.source_in), times.last().copied().unwrap_or(it.source_in));
            let size = (f64::from(w), f64::from(h));
            // how large the picture is shown: the rules are about the sequence's frame
            let display = s.project.conformed_source_size(it.item, q.settings.par).filter(|d| d.0 > 0.0 && d.1 > 0.0).unwrap_or(size);
            let (seq_w, seq_h) = (f64::from(q.settings.width.max(1)), f64::from(q.settings.height.max(1)));
            let fit = if it.scale_to_frame { (seq_w / display.0).min(seq_h / display.1) } else { 1.0 };
            let zoom = (0..n)
                .step_by(usize::try_from(step).unwrap_or(1).max(1))
                .map(|j| it.effect("motion").map_or(100.0, |m| m.f64_at("scale", it.effect_time_at(it.start + Tick(fd.0.saturating_mul(j))))))
                .filter(|v| v.is_finite())
                .fold(0.0f64, f64::max)
                .max(1.0)
                / 100.0
                * fit;
            let (mag_w, mag_h) = ((zoom * display.0 / seq_w).clamp(0.05, 50.0), (zoom * display.1 / seq_h).clamp(0.05, 50.0));
            let keep = KeepParams {
                min_face_height: (o.keep.min_face_height / mag_h).clamp(0.0, 1.0),
                min_plate_width: (o.keep.min_plate_width / mag_w).clamp(0.0, 1.0),
                ..o.keep
            };
            // the smallest face to hide is at least 16 px where the detector looks
            let long = size.0.max(size.1);
            let needed = 16.0 / (keep.min_face_height.max(0.004) * size.1 / long);
            let face_side = (needed.ceil() as usize).clamp(o.face_side, 1920.max(o.face_side)).min(long as usize).max(160);
            let side = (if o.faces { face_side } else { 0 }).max(if o.plates { o.plate_side } else { 0 }).max(64) as f64;
            let scale = (side / long).min(1.0) as f32;
            let ex = exclude.iter().filter(|e| e.clip.is_none_or(|c| c == it.id)).map(|e| e.ex).collect();
            let w = Work { clip: it.id, also: Vec::new(), item: it.item, name: it.name.clone(), src, times, size, scale, keep, face_side, exclude: ex, media };
            // a copy of a clip already listed (same media, same frames, same exclusions) is
            // analysed once
            match work.iter_mut().find(|o: &&mut Work| o.item == w.item && o.times == w.times && o.exclude == w.exclude && o.keep == w.keep) {
                Some(o) => {
                    o.also.push(w.clip);
                    total = total.saturating_sub(w.times.len());
                }
                None => work.push(w),
            }
        }
    }
    Ok((work, skipped))
}

// ---------------------------------------------------------------------------------------------
// analysing (the job)
// ---------------------------------------------------------------------------------------------

/// Progress shared by the workers.
struct Shared<'a> {
    prog: &'a filmcraft_export::Progress,
    done: AtomicU64,
    clips_done: AtomicUsize,
    clips: usize,
    stop: AtomicBool,
}

impl Shared<'_> {
    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.prog.cancel.load(Ordering::Relaxed)
    }
    fn frame(&self) {
        let d = self.done.fetch_add(1, Ordering::Relaxed) + 1;
        self.prog.done.store(d, Ordering::Relaxed);
    }
}

/// A detector: the faces and plates in an RGBA frame (`rgba`, width, height; the longest side
/// the face detector looks at), normalised to it.
pub type DetectFn = Arc<dyn Fn(&[u8], usize, usize, usize) -> Vec<filmcraft_vision::Detection> + Send + Sync>;

/// The downloaded detectors, for what `o` looks for.
#[cfg(feature = "privacy-models")]
fn load_detectors(o: &Options) -> std::result::Result<DetectFn, String> {
    let dir = crate::transcript::models_dir().ok_or("no data directory for the detectors")?;
    let det = filmcraft_vision::Detectors::load(&dir, o.faces, o.plates).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => format!("the face and plate detectors are not downloaded yet ({DOWNLOAD_CMD})"),
        e => e.to_string(),
    })?;
    let o = o.clone();
    Ok(Arc::new(move |rgba: &[u8], w: usize, h: usize, face_side: usize| {
        let mut out = Vec::new();
        if let Some(f) = det.face.as_ref().filter(|_| o.faces) {
            out.extend(f.detect(rgba, w, h, face_side, o.face_score).unwrap_or_default());
        }
        if let Some(p) = det.plate.as_ref().filter(|_| o.plates) {
            out.extend(p.detect(rgba, w, h, o.plate_side, o.plate_score).unwrap_or_default());
        }
        out
    }))
}

#[cfg(not(feature = "privacy-models"))]
fn load_detectors(_: &Options) -> std::result::Result<DetectFn, String> {
    Err("automatic redaction is not available in this build (built without the `privacy-models` feature)".into())
}

/// Analyse one clip: detect on every analysed frame, link, decide. `None` when stopped.
fn analyse(w: &Work, det: &DetectFn, o: &Options, sh: &Shared<'_>) -> Option<ClipRedact> {
    let mut frames = Vec::with_capacity(w.times.len());
    let mut missed = 0usize;
    for (k, mt) in w.times.iter().enumerate() {
        if sh.cancelled() {
            return None;
        }
        let detections = match w.src.video_frame(filmcraft_media::FrameRequest { time: *mt, scale: w.scale }) {
            Ok(f) => {
                let rgba = f.to_rgba8();
                det(&rgba, f.width as usize, f.height as usize, w.face_side)
            }
            Err(_) => {
                // a frame that does not decode shows nothing; its gap is filled if short
                missed += 1;
                Vec::new()
            }
        };
        frames.push(filmcraft_vision::track::Frame { t: mt.seconds(), detections });
        sh.frame();
        if k % 16 == 0 {
            *lock(&sh.prog.status) =
                format!("{}: frame {} of {} (clip {} of {})", w.name, k + 1, w.times.len(), sh.clips_done.load(Ordering::Relaxed) + 1, sh.clips);
        }
    }
    let tracks = filmcraft_vision::track::link(&frames, &o.link);
    let tracks = tracks.into_iter().map(|t| {
        let v = filmcraft_vision::track::verdict(&t, &w.keep, &w.exclude);
        (t, v)
    });
    let interval = if w.times.len() > 1 { (w.media.1 - w.media.0).seconds().abs() / (w.times.len() - 1) as f64 } else { 0.0 };
    Some(ClipRedact {
        clip: w.clip,
        also: w.also.clone(),
        name: w.name.clone(),
        size: w.size,
        tracks: tracks.collect(),
        frames: w.times.len(),
        interval,
        media: w.media,
        missed,
    })
}

/// Analyse every clip, a few at a time (decoding one while detecting in another).
fn analyse_all(work: &[Work], det: &DetectFn, o: &Options, sh: &Shared<'_>) -> Option<Vec<ClipRedact>> {
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get() / 3).clamp(1, 4).min(work.len().max(1));
    let next = AtomicUsize::new(0);
    let made: Mutex<Vec<(usize, ClipRedact)>> = Mutex::new(Vec::with_capacity(work.len()));
    let body = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(w) = work.get(i) else { return };
            // a clip that panics in a decoder is a clip with nothing found, not a dead job
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| analyse(w, det, o, sh))) {
                Ok(Some(r)) => lock(&made).push((i, r)),
                Ok(None) => {
                    sh.stop.store(true, Ordering::Relaxed);
                    return;
                }
                Err(_) => log::error!("auto redact: analysing clip {} failed", w.name),
            }
            sh.clips_done.fetch_add(1, Ordering::Relaxed);
        }
    };
    if workers <= 1 || cfg!(target_arch = "wasm32") {
        body();
    } else {
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(body);
            }
        });
    }
    if sh.cancelled() {
        return None;
    }
    let mut made = made.into_inner().unwrap_or_else(PoisonError::into_inner);
    made.sort_by_key(|(i, _)| *i);
    Some(made.into_iter().map(|(_, r)| r).collect())
}

/// A crop around a box (normalised) of an RGBA frame: `CROP_CONTEXT` times the box's larger
/// side, square, inside the frame, at most [`CROP_SIDE`] a side; with `mark`, the box is
/// outlined in red.
pub fn crop_around(rgba: &[u8], w: usize, h: usize, bbox: &[f64; 4], mark: bool) -> Option<Crop> {
    if w == 0 || h == 0 || rgba.len() < w.checked_mul(h)?.checked_mul(4)? || !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (cx, cy) = ((bbox[0] + bbox[2] / 2.0) * w as f64, (bbox[1] + bbox[3] / 2.0) * h as f64);
    let side = (bbox[2] * w as f64).max(bbox[3] * h as f64) * CROP_CONTEXT;
    let clampx = |v: f64| v.clamp(0.0, w as f64) as usize;
    let clampy = |v: f64| v.clamp(0.0, h as f64) as usize;
    let (x0, x1, y0, y1) = (clampx(cx - side / 2.0), clampx(cx + side / 2.0), clampy(cy - side / 2.0), clampy(cy + side / 2.0));
    let (cw, ch) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
    if cw < 2 || ch < 2 {
        return None;
    }
    let k = (f64::from(CROP_SIDE) / cw.max(ch) as f64).min(1.0);
    let (ow, oh) = (((cw as f64 * k).round() as usize).max(1), ((ch as f64 * k).round() as usize).max(1));
    let mut out = vec![0u8; ow * oh * 4];
    for (oy, row) in out.chunks_exact_mut(ow * 4).enumerate() {
        let (sy0, sy1) = (y0 + oy * ch / oh, (y0 + (oy + 1) * ch / oh).max(y0 + oy * ch / oh + 1).min(y1));
        for (ox, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (sx0, sx1) = (x0 + ox * cw / ow, (x0 + (ox + 1) * cw / ow).max(x0 + ox * cw / ow + 1).min(x1));
            let (mut acc, mut n) = ([0u32; 4], 0u32);
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    if let Some(p) = rgba.get((sy * w + sx) * 4..(sy * w + sx) * 4 + 4) {
                        for (a, v) in acc.iter_mut().zip(p) {
                            *a += u32::from(*v);
                        }
                        n += 1;
                    }
                }
            }
            for (o, a) in px.iter_mut().zip(acc) {
                *o = (a / n.max(1)) as u8;
            }
        }
    }
    let mut crop = Crop { rgba: out, width: ow as u32, height: oh as u32 };
    if mark {
        // the box itself, outlined in red: what the question is about
        let to = |v: f64, o: usize| ((v - o as f64) * k).round();
        let (bx0, by0) = (to(bbox[0] * w as f64, x0), to(bbox[1] * h as f64, y0));
        let (bx1, by1) = (to((bbox[0] + bbox[2]) * w as f64, x0), to((bbox[1] + bbox[3]) * h as f64, y0));
        outline(&mut crop, [bx0 - 2.0, by0 - 2.0, bx1 + 2.0, by1 + 2.0]);
    }
    Some(crop)
}

/// A 2-pixel red rectangle `[x0, y0, x1, y1]` drawn on a crop (clipped to it).
fn outline(c: &mut Crop, r: [f64; 4]) {
    let (w, h) = (c.width as i64, c.height as i64);
    let [x0, y0, x1, y1] = r.map(|v| if v.is_finite() { v.clamp(-1.0, 1e6) as i64 } else { -1 });
    let mut put = |x: i64, y: i64| {
        if (0..w).contains(&x) && (0..h).contains(&y) {
            let i = ((y * w + x) * 4) as usize;
            if let Some(p) = c.rgba.get_mut(i..i + 4) {
                p.copy_from_slice(&[255, 0, 0, 255]);
            }
        }
    };
    for t in 0..2 {
        for x in x0.max(0)..=x1.min(w) {
            put(x, y0 + t);
            put(x, y1 - t);
        }
        for y in y0.max(0)..=y1.min(h) {
            put(x0 + t, y);
            put(x1 - t, y);
        }
    }
}

/// The vision check: one crop per track still to be hidden (a face at its largest sighting, a
/// plate at its widest), shown to the judge [`JUDGE_BATCH`] at a time; tracks it calls
/// something else (a robot's face, a sign) are left alone.
fn verify(made: &mut [ClipRedact], work: &[Work], judge: &dyn FaceJudge, sh: &Shared<'_>) -> Verified {
    // (kind, clip, track, crop)
    let mut asks: Vec<(Kind, usize, usize, Crop)> = Vec::new();
    for (ci, c) in made.iter().enumerate() {
        let Some(w) = work.iter().find(|w| w.clip == c.clip) else { continue };
        for (ti, (t, skip)) in c.tracks.iter().enumerate() {
            if skip.is_some() {
                continue;
            }
            if sh.cancelled() {
                return Verified::Skipped("stopped".into());
            }
            let size = |p: &&filmcraft_vision::track::TrackPoint| if t.kind == Kind::Face { p.bbox[3] } else { p.bbox[2] };
            let Some(best) = t.points.iter().filter(|p| p.seen).max_by(|a, b| size(a).total_cmp(&size(b))) else { continue };
            *lock(&sh.prog.status) = format!("Checking what was found in {}", c.name);
            let mt = Tick::from_seconds_f64(best.t);
            let scale = (1080.0 / w.size.0.max(w.size.1)).min(1.0) as f32;
            if let Ok(f) = w.src.video_frame(filmcraft_media::FrameRequest { time: mt, scale }) {
                let rgba = f.to_rgba8();
                if let Some(crop) = crop_around(&rgba, f.width as usize, f.height as usize, &best.bbox, t.kind == Kind::Plate) {
                    asks.push((t.kind, ci, ti, crop));
                }
            }
        }
    }
    let (mut not_people, mut not_plates, mut unsure) = (0usize, 0usize, 0usize);
    for kind in [Kind::Face, Kind::Plate] {
        let mine: Vec<&(Kind, usize, usize, Crop)> = asks.iter().filter(|a| a.0 == kind).collect();
        for batch in mine.chunks(JUDGE_BATCH) {
            if sh.cancelled() {
                return Verified::Skipped("stopped".into());
            }
            *lock(&sh.prog.status) = format!("Asking {} which of {} {}s are real", judge.name(), mine.len(), kind.name());
            let crops: Vec<Crop> = batch.iter().map(|a| a.3.clone()).collect();
            let answers = match judge.judge(kind, &crops) {
                Ok(a) => a,
                Err(e) => return Verified::Failed(format!("{e}; everything found is hidden")),
            };
            for ((_, ci, ti, _), a) in batch.iter().copied().zip(answers.into_iter().chain(std::iter::repeat(None))) {
                match a {
                    Some(false) => {
                        if let Some((_, skip)) = made.get_mut(*ci).and_then(|c| c.tracks.get_mut(*ti)) {
                            if kind == Kind::Face {
                                *skip = Some(Skip::NotAPerson);
                                not_people += 1;
                            } else {
                                *skip = Some(Skip::NotAPlate);
                                not_plates += 1;
                            }
                        }
                    }
                    None => unsure += 1,
                    Some(true) => {}
                }
            }
        }
    }
    Verified::Done { judge: judge.name(), asked: asks.len(), not_people, not_plates, unsure }
}

// ---------------------------------------------------------------------------------------------
// the edit
// ---------------------------------------------------------------------------------------------

fn px_box(b: &[f64; 4], size: (f64, f64)) -> [f64; 4] {
    [b[0] * size.0, b[1] * size.1, b[2] * size.0, b[3] * size.1]
}

/// A track's mask shape at one moment, in clip pixels.
fn shape(t: &Track, p: &filmcraft_vision::track::TrackPoint, size: (f64, f64), padding: f64) -> MaskPath {
    let b = px_box(&filmcraft_vision::track::padded(&p.bbox, padding), size);
    match t.kind {
        Kind::Plate if p.corners.len() == 4 => {
            let q = filmcraft_vision::track::padded_corners(&p.corners, &p.bbox, padding);
            MaskPath::polygon(&q.iter().map(|c| Vec2::new(c[0] * size.0, c[1] * size.1)).collect::<Vec<_>>())
        }
        Kind::Plate => MaskPath::rect(b[0], b[1], b[0] + b[2], b[1] + b[3]),
        Kind::Face => MaskPath::ellipse(Vec2::new(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0), Vec2::new(b[2] / 2.0, b[3] / 2.0)),
    }
}

fn hold(t: Tick, v: f64) -> Keyframe {
    let mut k = Keyframe::new(t, ParamValue::Float(v));
    k.interp = Interpolation::Hold;
    k
}

/// The mask hiding track `t` (number `n` of its kind): a Mask Path keyframe at every analysed
/// moment, one analysed interval earlier and later at the ends (the detector may first see a
/// face a few frames after it appears), the Opacity on only from then to then.
pub fn mask_for(t: &Track, n: usize, c: &ClipRedact, padding: f64) -> Option<Mask> {
    let first = t.points.first()?;
    let last = t.points.last()?;
    let lead = Tick::from_seconds_f64(c.interval.max(0.0));
    let (lo, hi) = (c.media.0.min(c.media.1), c.media.0.max(c.media.1));
    let start = (Tick::from_seconds_f64(first.t) - lead).max(lo);
    let end = (Tick::from_seconds_f64(last.t) + lead).min(hi);
    let mut keys: Vec<Keyframe> = Vec::with_capacity(t.points.len() + 2);
    let mut push = |time: Tick, path: MaskPath| {
        if keys.last().is_none_or(|k: &Keyframe| k.time < time) {
            keys.push(Keyframe::new(time, ParamValue::Path(path)));
        }
    };
    push(start, shape(t, first, c.size, padding));
    for p in &t.points {
        push(Tick::from_seconds_f64(p.t), shape(t, p, c.size, padding));
    }
    push(end, shape(t, last, c.size, padding));
    let first_shape = keys.first().map(|k| k.value.clone())?;
    let ParamValue::Path(path0) = first_shape else { return None };
    let mut m = Mask::new(format!("{MASK_PREFIX}{} {n}", t.kind.name()), path0);
    if keys.len() > 1 {
        m.path.keyframes = keys;
    }
    // a soft edge outside the shape: the whole padded shape is hidden, the blur fades out beyond
    let mut heights: Vec<f64> = t.points.iter().map(|p| p.bbox[3] * c.size.1).collect();
    heights.sort_by(f64::total_cmp);
    let median = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
    let feather = (median * 0.12).clamp(1.0, 200.0);
    m.feather = Param::new(ParamValue::Float(feather));
    m.expansion = Param::new(ParamValue::Float(feather / 2.0));
    let mut op = vec![hold(start, 100.0), hold(end, 0.0)];
    if start > lo {
        op.insert(0, hold(lo, 0.0));
    }
    if end >= hi {
        op.pop();
    }
    m.opacity = Param::new(ParamValue::Float(if start > lo { 0.0 } else { 100.0 }));
    if op.len() > 1 {
        m.opacity.keyframes = op;
    }
    Some(m)
}

/// The effect hiding a clip's tracks, or `None` when nothing in it is hidden.
pub fn effect_for(c: &ClipRedact, o: &Options) -> Option<EffectInstance> {
    let hidden: Vec<&Track> = c.hidden().collect();
    if hidden.is_empty() {
        return None;
    }
    // the size to hide: the largest face's height, a plate's width counting for less
    let size =
        hidden.iter().map(|t| if t.kind == Kind::Face { t.max_size() * c.size.1 } else { t.max_size() * c.size.0 * 0.6 }).fold(0.0f64, f64::max).max(8.0);
    let s = (o.strength / 100.0).clamp(0.01, 1.0);
    let mut e = match o.style {
        Style::Blur => {
            let mut e = filmcraft_project::effect::find_effect("gaussian_blur")?.instance();
            let blur = (size * 0.5 * s).clamp(6.0, 2000.0);
            *e.param_mut("blurriness")? = Param::new(ParamValue::Float(blur));
            if let Some(r) = e.param_mut("repeat_edge") {
                *r = Param::new(ParamValue::Bool(true));
            }
            e
        }
        Style::Mosaic => {
            let mut e = filmcraft_project::effect::find_effect("mosaic")?.instance();
            // 16 blocks down the largest face at the lowest strength, 5 at the highest
            let block = (size / (16.0 - 11.0 * s)).max(2.0);
            *e.param_mut("horizontal")? = Param::new(ParamValue::Float((c.size.0 / block).round().clamp(1.0, 4000.0)));
            *e.param_mut("vertical")? = Param::new(ParamValue::Float((c.size.1 / block).round().clamp(1.0, 4000.0)));
            if let Some(r) = e.param_mut("sharp") {
                *r = Param::new(ParamValue::Bool(true));
            }
            e
        }
    };
    let (mut faces, mut plates) = (0usize, 0usize);
    for t in hidden {
        let n = match t.kind {
            Kind::Face => {
                faces += 1;
                faces
            }
            Kind::Plate => {
                plates += 1;
                plates
            }
        };
        if let Some(m) = mask_for(t, n, c, o.padding) {
            e.masks.push(m);
        }
    }
    (!e.masks.is_empty()).then_some(e)
}

/// An effect this command made earlier: a blur or mosaic whose masks are all its own.
pub fn is_auto_redaction(e: &EffectInstance) -> bool {
    matches!(e.effect.as_str(), "gaussian_blur" | "mosaic") && !e.masks.is_empty() && e.masks.iter().all(|m| m.name.starts_with(MASK_PREFIX))
}

fn track_json(t: &Track, skip: Option<Skip>, with_path: bool) -> Value {
    let best = t.points.iter().filter(|p| p.seen).max_by(|a, b| a.bbox[3].total_cmp(&b.bbox[3])).or(t.points.first());
    let r = |v: f64| (v * 1000.0).round() / 1000.0;
    let mut v = json!({
        "kind": t.kind.name(),
        "start": r(t.start()),
        "end": r(t.end()),
        "hits": t.hits,
        "score": r(f64::from(t.best_score)),
        "size": r(t.max_size()),
        "hidden": skip.is_none(),
        "box": best.map(|p| json!({"seconds": r(p.t), "x": r(p.bbox[0]), "y": r(p.bbox[1]), "w": r(p.bbox[2]), "h": r(p.bbox[3])})),
    });
    if let Some(s) = skip {
        v["reason"] = serde_json::to_value(s).unwrap_or(Value::Null);
    }
    if with_path {
        v["path"] = Value::Array(t.points.iter().map(|p| json!([r(p.t), r(p.bbox[0]), r(p.bbox[1]), r(p.bbox[2]), r(p.bbox[3])])).collect());
    }
    v
}

fn clip_json(c: &ClipRedact, dry: bool, masks: usize) -> Value {
    let count = |k: Kind| c.hidden().filter(|t| t.kind == k).count();
    let skipped = |s: Skip| c.tracks.iter().filter(|(_, x)| *x == Some(s)).count();
    json!({
        "clip": c.clip.0,
        "also": c.also.iter().map(|a| a.0).collect::<Vec<_>>(),
        "name": c.name,
        "frames": c.frames,
        "missedFrames": c.missed,
        "faces": count(Kind::Face),
        "plates": count(Kind::Plate),
        "masks": masks,
        "skipped": {"tooSmall": skipped(Skip::TooSmall), "tooBrief": skipped(Skip::TooBrief), "excluded": skipped(Skip::Excluded), "notAPerson": skipped(Skip::NotAPerson), "notAPlate": skipped(Skip::NotAPlate)},
        "tracks": c.tracks.iter().map(|(t, s)| track_json(t, *s, dry)).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------------------------

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: CLIP_CMD,
            label: "Auto Redact Faces and Plates…",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clips":[id]?,"clip":id?,"kinds":["faces","plates"]|"faces,plates"=both,"minFaceHeight":0..1=0.05,"minPlateWidth":0..1=0.025,"faceScore":0.05..1=0.6,"plateScore":0.05..1=0.8,"style":"blur"|"mosaic"="blur","strength":1..100=60,"padding":0..1=0.2,"exclude":{"x":0..1,"y":0..1,"w":0..1,"h":0..1,"clip":id?,"seconds":f64?}|[box]?,"verify":bool=true,"dryRun":bool=false,"analysisFps":1..60=10,"replace":bool=true,"wait":bool=false}"#,
            enabled: has_video_selection,
            run: redact_clips,
            journal: true,
        },
        CommandSpec {
            id: SEQ_CMD,
            label: "Auto Redact Sequence…",
            menu: &[],
            shortcut: None,
            params: r#"{"sequence":id?,"kinds":["faces","plates"]|"faces,plates"=both,"minFaceHeight":0..1=0.05,"minPlateWidth":0..1=0.025,"faceScore":0.05..1=0.6,"plateScore":0.05..1=0.8,"style":"blur"|"mosaic"="blur","strength":1..100=60,"padding":0..1=0.2,"exclude":{"x":0..1,"y":0..1,"w":0..1,"h":0..1,"clip":id?,"seconds":f64?}|[box]?,"verify":bool=true,"dryRun":bool=false,"analysisFps":1..60=10,"replace":bool=true,"wait":bool=false}"#,
            enabled: has_seq,
            run: redact_sequence,
            journal: true,
        },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download Face and Plate Detectors",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool?}"#,
            enabled: always,
            run: download_models,
            journal: false,
        },
        CommandSpec { id: "redact.status", label: "Auto Redact Status", menu: &[], shortcut: None, params: "{}", enabled: always, run: status, journal: false },
    ]
}

fn has_video_selection(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    let q = s.active_sequence().ok_or("no sequence is open")?;
    let any = s.state.selection.iter().any(|c| q.find_item(*c).is_some_and(|(t, _)| q.track(t).is_some_and(|t| t.kind == TrackKind::Video)));
    if any { Ok(()) } else { Err("select the video clips to redact in the timeline".into()) }
}

/// The models must be there before a job starts.
fn ready(cmd: &str) -> Result<()> {
    if !available() {
        return Err(bad(cmd, "automatic redaction is not available in this build (built without the `privacy-models` feature)"));
    }
    if !installed() {
        return Err(bad(
            cmd,
            format!(
                "the face and plate detectors are not downloaded yet: run {DOWNLOAD_CMD} ({:.1} MB; {})",
                filmcraft_vision::catalog::total_bytes() as f64 / 1e6,
                filmcraft_vision::catalog::LICENSE
            ),
        ));
    }
    Ok(())
}

fn redact_clips(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = CLIP_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let o = options_p(cmd, p)?;
    let ex = exclude_p(cmd, p)?;
    let only: Vec<ClipId> = match (p.get("clips").filter(|v| !v.is_null()), p.get("clip").filter(|v| !v.is_null())) {
        (Some(Value::Array(a)), _) => {
            if a.len() > MAX_CLIPS {
                return Err(bad(cmd, "too many clips"));
            }
            a.iter().map(|v| v.as_u64().map(ClipId)).collect::<Option<Vec<_>>>().ok_or_else(|| bad(cmd, "`clips` is a list of clip ids"))?
        }
        (Some(_), _) => return Err(bad(cmd, "`clips` is a list of clip ids")),
        (None, Some(c)) => vec![ClipId(c.as_u64().ok_or_else(|| bad(cmd, "`clip` is a clip id"))?)],
        (None, None) => s.state.selection.clone(),
    };
    if only.is_empty() {
        return Err(bad(cmd, "select the video clips to redact (or pass `clips`)"));
    }
    if s.redacts.detector.is_none() {
        ready(cmd)?;
    }
    let (work, skipped) = build_work(s, seq, Some(&only), &o, &ex)?;
    if work.is_empty() {
        return Err(bad(cmd, "no video footage to redact among the clips (graphics, nested sequences and offline media are left alone)"));
    }
    start(s, cmd, seq, work, skipped, o, p)
}

fn redact_sequence(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = SEQ_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => ItemId(v.as_u64().ok_or_else(|| bad(cmd, "`sequence` is a sequence id"))?),
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    if s.project.sequence(seq).is_none() {
        return Err(bad(cmd, "no such sequence"));
    }
    let o = options_p(cmd, p)?;
    let ex = exclude_p(cmd, p)?;
    if s.redacts.detector.is_none() {
        ready(cmd)?;
    }
    let (work, skipped) = build_work(s, seq, None, &o, &ex)?;
    if work.is_empty() {
        return Err(bad(cmd, "the sequence has no video footage to redact"));
    }
    start(s, cmd, seq, work, skipped, o, p)
}

fn start(s: &mut Session, cmd: &'static str, seq: ItemId, work: Vec<Work>, skipped: Vec<Value>, o: Options, p: &Value) -> Result<Value> {
    let total: usize = work.iter().map(|w| w.times.len()).sum();
    let clips = work.len();
    let n = clips;
    let job = crate::narration::new_job(s, format!("{LABEL} ({n} clip{})", if n == 1 { "" } else { "s" }));
    job.progress.total.store(total as u64, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let judge = if o.verify { s.redacts.judge.clone() } else { None };
    let custom = s.redacts.detector.clone();
    let verify_wanted = o.verify;
    let opts = o.clone();
    let run = move || {
        let t0 = web_time::Instant::now();
        let finish = |r: std::result::Result<Found, String>, status: String, frames: u64| {
            let secs = t0.elapsed().as_secs_f64();
            *lock(&prog.status) = status;
            let report = match &r {
                Ok(_) => Ok(filmcraft_export::Report {
                    path: String::new(),
                    frames,
                    seconds: secs,
                    bytes: 0,
                    render_fps: frames as f64 / secs.max(1e-6),
                    extra_files: Vec::new(),
                }),
                Err(e) => {
                    *lock(&prog.error) = Some(e.clone());
                    Err(e.clone())
                }
            };
            *lock(&out) = Some(r);
            *lock(&res) = Some(report);
            prog.finished.store(true, Ordering::Relaxed);
        };
        *lock(&prog.status) = "Loading the face and plate detectors".into();
        let det = match custom.map_or_else(|| load_detectors(&opts), Ok) {
            Ok(d) => d,
            Err(e) => return finish(Err(e.clone()), e, 0),
        };
        let sh = Shared { prog: &prog, done: AtomicU64::new(0), clips_done: AtomicUsize::new(0), clips: work.len(), stop: AtomicBool::new(false) };
        let Some(mut made) = analyse_all(&work, &det, &opts, &sh) else {
            return finish(Err("stopped".into()), "Stopped: nothing was changed".into(), sh.done.load(Ordering::Relaxed));
        };
        let t_analysis = t0.elapsed().as_secs_f64();
        let verified = match (&judge, verify_wanted) {
            (_, false) => Verified::Off,
            (None, true) => Verified::Skipped("no local vision model to tell people from robots and plates from signs: everything found is hidden".into()),
            (Some(j), true) => verify(&mut made, &work, j.as_ref(), &sh),
        };
        if sh.cancelled() {
            return finish(Err("stopped".into()), "Stopped: nothing was changed".into(), sh.done.load(Ordering::Relaxed));
        }
        let hidden: usize = made.iter().map(|c| c.hidden().count()).sum();
        let frames = sh.done.load(Ordering::Relaxed);
        let status = format!("Found {hidden} face(s) and plate(s) to hide in {} clip(s), {frames} frame(s) in {:.1}s", made.len(), t0.elapsed().as_secs_f64());
        let t_verify = t0.elapsed().as_secs_f64() - t_analysis;
        finish(Ok((made, verified, [t_analysis, t_verify])), status, frames);
    };
    s.redacts.pending.push(PendingRedact { job: id, cmd, seq, opts: o, skipped, results });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-auto-redact", run)?;
    let mut answer = json!({"job": id, "clips": clips, "frames": total});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.redacts.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["redact"] = o.clone();
        }
    }
    Ok(answer)
}

/// Apply finished jobs (one undo step each) and keep what they made for `jobs.list`. Called each
/// UI frame (through [`Session::poll_persistence`]), by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.redacts.pending.len() {
        let Some(pj) = s.redacts.pending.get(i) else { break };
        let ready = lock(&pj.results).is_some();
        let job = s.jobs.iter().find(|j| j.id == pj.job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if !ready && !finished {
            i += 1;
            continue;
        }
        let pj = s.redacts.pending.remove(i);
        let results = lock(&pj.results).take();
        let outcome = match results {
            Some(Ok((clips, verified, secs))) => match apply(s, &pj, &clips, &verified, secs) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(pj.cmd, format!("{LABEL}: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "stopped: nothing was changed"}),
        };
        s.redacts.outcomes.insert(pj.job, outcome);
        if s.redacts.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.redacts.outcomes.keys().min().copied()
        {
            s.redacts.outcomes.remove(&old);
        }
    }
}

fn apply(s: &mut Session, pj: &PendingRedact, clips: &[ClipRedact], verified: &Verified, secs: [f64; 2]) -> Result<Value> {
    let o = &pj.opts;
    let made: Vec<Option<EffectInstance>> = clips.iter().map(|c| effect_for(c, o)).collect();
    let hidden: usize = made.iter().flatten().map(|e| e.masks.len()).sum();
    // every clip of an analysis, and its copies, with the effect hiding what it shows
    let effects: Vec<(ClipId, Option<EffectInstance>)> =
        clips.iter().zip(&made).flat_map(|(c, e)| std::iter::once(c.clip).chain(c.also.iter().copied()).map(move |id| (id, e.clone()))).collect();
    let masks = |id: ClipId| effects.iter().find(|(c, _)| *c == id).and_then(|(_, e)| e.as_ref()).map_or(0, |e| e.masks.len());
    let summary: Vec<Value> = clips.iter().map(|c| clip_json(c, o.dry_run, if o.dry_run { 0 } else { masks(c.clip) })).collect();
    let mut changed = 0usize;
    if !o.dry_run {
        if s.project.sequence(pj.seq).is_none() {
            return Err(EngineError::Other("the sequence is gone".into()));
        }
        let any_old = s
            .project
            .sequence(pj.seq)
            .is_some_and(|q| effects.iter().any(|(id, _)| q.find_item(*id).is_some_and(|(_, it)| o.replace && it.effects.iter().any(is_auto_redaction))));
        if hidden > 0 || any_old {
            changed = s.edit(LABEL, |pr, _| {
                let q = pr.sequence_mut(pj.seq).ok_or(EngineError::NoSequence)?;
                let mut n = 0usize;
                for (id, e) in &effects {
                    let Some((_, it)) = q.find_item_mut(*id) else { continue };
                    if o.replace {
                        it.effects.retain(|x| !is_auto_redaction(x));
                    }
                    if let Some(e) = e {
                        // ahead of the clip's other standard effects: the masks are in the
                        // source's own pixels, before anything moves them (Warp Stabilizer)
                        let at = it.effects.iter().position(|x| x.def().is_some_and(|d| !d.intrinsic)).unwrap_or(it.effects.len());
                        it.effects.insert(at, e.clone());
                        n += 1;
                    }
                }
                q.check().map_err(EngineError::Other)?;
                Ok(n)
            })?;
        }
        let faces: usize = clips.iter().map(|c| c.hidden().filter(|t| t.kind == Kind::Face).count()).sum();
        let plates: usize = clips.iter().map(|c| c.hidden().filter(|t| t.kind == Kind::Plate).count()).sum();
        let note = match verified {
            Verified::Skipped(why) | Verified::Failed(why) => format!(" (vision check: {why})"),
            _ => String::new(),
        };
        s.toast(format!("{LABEL}: {faces} face(s) and {plates} plate(s) hidden in {changed} clip(s){note}"));
    }
    Ok(json!({
        "sequence": pj.seq.0,
        "dryRun": o.dry_run,
        "applied": !o.dry_run,
        "clipsChanged": changed,
        "hidden": hidden,
        "verify": verified.to_json(),
        "seconds": {"analysis": (secs[0] * 10.0).round() / 10.0, "check": (secs[1] * 10.0).round() / 10.0},
        "clips": summary,
        "skippedClips": pj.skipped,
    }))
}

/// What a finished redaction job made (`redact` in its `jobs.list` entry).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.redacts.outcomes.get(&job).cloned()
}

fn status(s: &mut Session, _: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    Ok(json!({
        "available": available(),
        "installed": installed(),
        "missingBytes": dir.as_deref().map_or(filmcraft_vision::catalog::total_bytes(), filmcraft_vision::catalog::missing_bytes),
        "package": filmcraft_vision::catalog::PACKAGE_NAME,
        "license": filmcraft_vision::catalog::LICENSE,
        "source": filmcraft_vision::catalog::SOURCE,
        "files": filmcraft_vision::catalog::FILES,
        "judge": s.redacts.judge.as_ref().map(|j| j.name()),
    }))
}

/// Download the detectors in the background.
fn download_models(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = DOWNLOAD_CMD;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the detectors"))?;
    if filmcraft_vision::catalog::installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "automatic redaction is not available in this build (built without the `privacy-models` feature)"));
    }
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::PACKAGE_NAME));
    job.progress.total.store(filmcraft_vision::catalog::missing_bytes(&dir), Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-redact-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_package(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Face and plate detectors installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false}))
}

#[cfg(feature = "privacy-models")]
fn download_package(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::package_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_package(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("automatic redaction is not available in this build".into())
}

#[cfg(test)]
#[path = "redact_tests.rs"]
mod tests;
