//! Clip ▸ Auto Redact Faces and Plates… — the automatic privacy blur.
//!
//! | command | does |
//! |---|---|
//! | `clip.autoRedact` | job: find clearly recognisable faces and legible number plates in clips (`clips`, default the selected video clips) and hide each behind a tracked mask on a blur (or mosaic) effect; one undo step ("Auto Redact") |
//! | `sequence.autoRedact` | the same on every video clip of `sequence` (default the active one) |
//! | `redact.downloadModels` | download the two detectors (4.4 MB) and, unless `profiles: false`, the profile pass's person and pose models (35.8 MB), SHA-256 checked, as a background job, after the user agreed |
//! | `redact.status` | whether the detectors and the profile models are installed (and their licences and sizes), and whether the vision check is available |
//!
//! **How.** In a background job (`jobs.list` progress; `jobs.cancel` stops it and nothing
//! changes) each clip's used part is decoded and analysed `analysisFps` times a second (10 by
//! default) with `filmcraft-vision`'s YuNet face and LPD-YuNet plate detectors (pure Rust on
//! candle, CPU). Detections are linked into tracks, gaps filled and the paths smoothed
//! (`filmcraft_vision::track`). A face is hidden when it is at least `minFaceHeight` of the
//! sequence's frame tall at some point as the sequence shows it (5 %, measured within 15 %,
//! `filmcraft_vision::track::SIZE_TOLERANCE`: smaller faces in the crowd are not recognisable,
//! and one face measures some 10 % differently in the redaction's frames and the privacy check's;
//! Motion ▸ Scale and Scale to Frame Size count, so a clip zoomed in or a
//! portrait clip filling a landscape frame has its faces counted larger, and the face detector
//! looks at it closer; it looks at every frame twice, the second time twice as close where the
//! picture allows, [`close_side`]: at the first size a face of 5–8 % is 18–30 px, where YuNet
//! misses some), a plate when it is at least `minPlateWidth` of the frame's width
//! (2.5 %); either must be seen twice, or once with a score of 0.9. An `exclude` box is the
//! subject's (a robot's: its face is not hidden): see below.
//!
//! **Faces in profile.** YuNet finds frontal faces; a face in profile or half hidden it misses
//! (2026-10-22: two visitors in profile behind a robot arm stayed sharp in a reel). With
//! `profiles` (on by default once its models are downloaded, `redact.downloadModels`) every
//! analysed frame also goes through the profile pass (`filmcraft_vision::ProfileDetector`: the
//! D-FINE person detector, then RTMPose on each person, wholly on the GPU on macOS): each person whose
//! nose or an eye is found (the back of a head identifies nobody; an unsure face the person's own
//! body puts at the hips is a head outside the picture, `filmcraft_vision::head::below_the_hips`)
//! and whose head no YuNet face covers adds a face, the head box from the nose, eyes and ears. Those faces are followed
//! together with YuNet's (a visitor turning away stays one track), need two sightings, never
//! count as "sure" (so the vision check's "a robot" is believed: Ameca keeps its face), and are
//! reported (`fromPose`, `facesFromPose`). The job holds the machine's GPU lock while it runs
//! (`filmcraft_locks`; waited for up to ten minutes, then run anyway). Measured 2026-10-22 on an
//! M3 Max (14 cores) on a 1080p frame of 24 visitors: the people and their poses in 101 ms on an
//! idle machine and 105 ms at a load average of 33–52 (9.5 frames a second); before the decoder,
//! the pose head and the crops moved to the GPU, 118 ms and 973 ms (the CPU parts starved).
//!
//! **Not robots.** With `verify` (the default) every track still to be hidden is shown once,
//! cropped with its surroundings, to a local vision model (the host's [`VisionJudge`]: AutoCut's
//! Qwen2.5-VL through Ollama): a face is hidden only if it is a real person's (not a humanoid
//! robot, mannequin, statue, poster or screen), a plate only if it is a vehicle's (not a sign or
//! a robot's name badge); an unclear answer hides it. Faces whose centre is inside an `exclude`
//! box are shown too, and hidden if the model sees a person: a subject's box is coarse (the
//! robot's largest analysed box, taken square), and the visitors crowding round a robot stand in
//! it — on 2026-10-21 eight faces next to Ameca, two of them filling a sixth of the frame, were
//! left sharp by the box alone. A face the model calls not a person is looked at a second time,
//! alone and outlined in red ([`VisionJudge::confirm`]; its grid answer can carry a neighbour's
//! over), and left sharp only if both agree; a face the detector is sure of (`sureFace`, 0.85)
//! that is never inside a subject's box is hidden whatever the model says. Without a judge the
//! check is skipped and noted: everything detected is hidden but the faces inside an `exclude`
//! box (within a second of its `seconds`). Plates must also have a plate's shape (wider than
//! tall, not huge: `filmcraft_vision::post::plausible_plate`).
//!
//! **Named targets.** `targets` names other things to hide in words — a logo, a brand, a sign
//! ("DP World logo", "dpworld.com text"; a description, or `{text, clips}` for some clips only).
//! The host's vision model ([`VisionJudge::find`], grounding boxes) looks at the analysed frames
//! every `targetEvery` seconds (1) and at shot changes; the subject tracker carries each place
//! it finds between its looks, backwards to the previous look for a new place
//! (`redact_targets`). They are hidden at any size, inside an `exclude` box too (a robot's
//! box excludes its face, not the logo on its chest), under a rectangle with a wider margin.
//! Without a vision model the command refuses targets.
//!
//! **The edit.** Each clip with something to hide gets a Gaussian Blur (or Mosaic, `style`) effect
//! for each band of sizes ([`SIZE_BAND`]: within a factor of two), its strength from the band's
//! largest face or plate ([`blurriness`]: σ 0.15 of a face's height at the default strength, as
//! light as still hides who it is, proportional to each face so a small face is not smeared with a
//! large one's blur), with one mask per track — an ellipse for a face, the plate's own
//! quadrilateral for a plate — grown by `padding` (0.15 of the box on each side, a face [`CHIN`]
//! more below and at least [`MIN_FACE_MASK`] tall), with a soft edge outward ([`FEATHER`]: 0.16 of
//! the face's height; drawn in where another mask is near, so that two neighbours' blurs do not
//! meet unless their shapes nearly touch: a sharp line between two faces instead of one blob,
//! [`soft_edge`]), a Mask Path keyframe at every analysed frame (wider by half its move for a face
//! moving fast between them, `motion_margin`; carried past the frame's edge for a face it cuts,
//! [`EDGE_REACH`]: unseen, and where the detectors disagree) and its Opacity fully on from
//! [`LINGER`] before the track starts to as long after it ends (a face going behind something is
//! lost before it is gone; [`EDGE_LINGER`] at the frame's edge), fading in and out beyond
//! ([`FADE`]) rather than switching. Within a track the mask stays on: its gaps are filled, longer
//! ones after an unsure sighting (a head cut by the frame, a pose the network is unsure of:
//! `filmcraft_vision::track::LinkParams::unsure_gap`), so such a head neither starts a mask on one
//! sighting nor drops it on one miss; the tracks follow faces through camera pans
//! (`filmcraft_vision::track::LinkParams::pan`). (2026-10-22: the profile pass's head boxes were a
//! median 1.17 times YuNet's faces, one blur from the largest face smeared every face of a crowd
//! into one blob, and heads cut by the frame flickered; heads are now boxed like YuNet's faces,
//! `filmcraft_vision::head::FACE_BOX`.) Masks are in
//! clip pixels and the effect applies before Motion, so the blur follows the clip's own motion,
//! scale, crop and reframing. The masks are named `Redact face 1`, `Redact plate 2`…; running
//! it again replaces an earlier automatic redaction of the clip (`replace: false` keeps it).
//! `dryRun` only reports the tracks (the job's `redact` outcome), changing nothing.
//!
//! **The cache.** A clip is analysed at the media times of a grid (its first and last frame, and
//! every multiple of the analysis step in media time), so two windows of one clip share the
//! frames they overlap. With a redaction cache ([`crate::redact_cache`], `<data dir>/redact-cache`,
//! on in the app and the CLI) the detections at each of those times, the vision check's answer
//! for each track and the named targets' looks are kept by the file's content identity, the
//! settings and the models (the profile pass's models' ids are in the detections' key when it
//! runs, so entries made without it are analysed again once): the same footage redacted again —
//! a rebuilt showcase, the same show in another language — decodes and asks only what it has not
//! seen. The outcome's `cache` says
//! what came from it; `fresh: true` reads none of it. Measured 2026-10-20 on 11 clips of Robot
//! Showcase v5 (656 analysed frames, 64 vision checks; M-series Mac, load 5–10): 152 s the first
//! time, 0.13 s the second, 34 s with every clip's window moved half a second (576 frames and 52
//! answers from the cache, 80 frames decoded and 13 asked).
//!
//! **Coverage.** The outcome says, per clip, how much of the picture its masks cover (`coverage`:
//! the largest single mask at any keyframe, and the most hidden at once, with when in media and
//! sequence seconds): a mask over a large part of the frame hides the shot, not a face or a logo.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, EffectInstance, ItemId, ItemKind, Keyframe, Mask, MaskPath, MediaRef, Param, ParamValue, TrackKind};
use filmcraft_time::Tick;
use filmcraft_vision::track::{Exclude, KeepParams, LinkParams, Skip, Track};
pub use filmcraft_vision::{Detection, Kind};
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
pub trait VisionJudge: Send + Sync {
    /// Which model answers ("ollama:qwen2.5vl:7b").
    fn name(&self) -> String;
    /// For each crop of a `kind` detection, in order. Faces: `Some(true)` a real person's face,
    /// `Some(false)` something that looks like one (a humanoid robot, mannequin, statue, poster,
    /// a face on a screen). Plates (the plate is outlined in red in the crop): `Some(true)` a
    /// vehicle's number plate, `Some(false)` something else (a sign, a label or name on a
    /// machine). `None`: cannot tell. An error means the check could not run at all.
    fn judge(&self, kind: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String>;
    /// A second look at one detection the first look called something else: `crop` shows it
    /// alone, outlined in red (the first look's grid can carry a neighbour's answer over: a
    /// visitor beside a robot's head was called a robot). The same answers as
    /// [`VisionJudge::judge`]; by default, that with the one crop.
    fn confirm(&self, kind: Kind, crop: &Crop) -> std::result::Result<Option<bool>, String> {
        Ok(self.judge(kind, std::slice::from_ref(crop))?.into_iter().next().flatten())
    }
    /// Where the things the user named (`targets`: "DP World logo", "the company's website
    /// on the floor") are in a frame: `(index into targets, [x, y, w, h])` per place, the box
    /// normalised to the frame. Hosts without a grounding model say so.
    fn find(&self, frame: &Crop, targets: &[String]) -> std::result::Result<Vec<(usize, [f64; 4])>, String> {
        let _ = (frame, targets);
        Err("this vision model cannot find named things".into())
    }
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
    /// A face the detector is at least this sure of is hidden whatever the vision check says,
    /// unless it is on the subject (inside an `exclude` box at some moment).
    pub sure_face: f32,
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
    /// Things the user names ("DP World logo"), each for every clip or for some.
    pub targets: Vec<TargetP>,
    /// Seconds between the vision model's looks for the targets.
    pub target_every: f64,
    /// Read nothing from the redaction cache ([`crate::redact_cache`]): analyse and ask again
    /// (what is found still replaces what was kept).
    pub fresh: bool,
    /// The profile pass (people and their poses: the faces in profile or half hidden the face
    /// detector misses); `None`: on when its models are downloaded ([`profiles_on`]).
    pub profiles: Option<bool>,
}

/// [`Options::sure_face`] by default: real visitors' faces scored 0.86–0.92 on the GITEX footage
/// where the vision check called some of them robots (2026-10-21).
pub const SURE_FACE: f32 = 0.85;

/// A named target from the parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetP {
    pub text: String,
    /// Only in these clips (none: every clip).
    pub clips: Option<Vec<ClipId>>,
}

/// Most targets a command takes, and the longest description.
pub const MAX_TARGETS: usize = 16;
pub const MAX_TARGET_CHARS: usize = 120;

impl Default for Options {
    fn default() -> Self {
        Self {
            faces: true,
            plates: true,
            keep: KeepParams::default(),
            link: LinkParams::default(),
            face_score: 0.6,
            plate_score: 0.8,
            sure_face: SURE_FACE,
            style: Style::Blur,
            strength: 60.0,
            padding: PADDING,
            verify: true,
            dry_run: false,
            analysis_fps: 10.0,
            face_side: 640,
            plate_side: 640,
            replace: true,
            targets: Vec::new(),
            target_every: 1.0,
            fresh: false,
            profiles: None,
        }
    }
}

/// An exclusion box from the parameters, for one clip or all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExcludeP {
    pub(crate) clip: Option<ClipId>,
    pub(crate) ex: Exclude,
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
    /// The named targets looked for in this clip.
    targets: Vec<String>,
    exclude: Vec<Exclude>,
    /// The clip's media range (the masks' lead-in and lead-out stay inside it).
    media: (Tick, Tick),
    /// The file decoded (none, so not cached: a generator, or a proxy plays instead), and what
    /// else decides its frames (picture size, length, interpreted rate): the redaction cache's
    /// media key.
    file: Option<String>,
    desc: String,
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
    /// The named targets' looks: asked, places found, errors.
    pub looks: usize,
    pub found: usize,
    pub look_errors: Vec<String>,
    /// The subject's boxes in this clip (`exclude`).
    pub robots: Vec<Exclude>,
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
    /// `on_subject`: faces inside the subject's box shown to the judge; `confirmed` (and
    /// `confirmed_cached`): second looks at faces it called not a person; `overruled`: faces the
    /// detector was sure of, away from the subject, hidden although it called them not a person.
    Done {
        judge: String,
        asked: usize,
        cached: usize,
        not_people: usize,
        not_plates: usize,
        unsure: usize,
        on_subject: usize,
        confirmed: usize,
        confirmed_cached: usize,
        overruled: usize,
    },
    Skipped(String),
    Failed(String),
}

impl Verified {
    fn to_json(&self) -> Value {
        match self {
            Verified::Off => json!({"state": "off"}),
            Verified::Done { judge, asked, cached, not_people, not_plates, unsure, on_subject, confirmed, confirmed_cached, overruled } => json!({
                "state": "done", "judge": judge, "asked": asked, "fromCache": cached, "notPeople": not_people, "notPlates": not_plates, "unsure": unsure,
                "onSubject": on_subject, "lookedAgain": confirmed, "lookedAgainFromCache": confirmed_cached, "overruled": overruled,
            }),
            Verified::Skipped(why) => json!({"state": "skipped", "reason": why}),
            Verified::Failed(why) => json!({"state": "failed", "reason": why}),
        }
    }
}

/// What a job found, how the vision check went, how long the analysis and the check took
/// (seconds), the cache's statistics, and whether the profile pass ran.
type Found = (Vec<ClipRedact>, Verified, [f64; 2], Value, bool);
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
    pub judge: Option<Arc<dyn VisionJudge>>,
    /// A detector used instead of the downloaded models (tests; a host with its own).
    pub detector: Option<DetectFn>,
    /// What [`Redacts::detector`] is, for the cache's keys: a detector without an id is not
    /// cached.
    pub detector_id: Option<String>,
    /// The redaction cache ([`crate::redact_cache`]; native frontends turn it on with
    /// [`Session::use_redaction_cache`]).
    pub cache: Option<Arc<crate::redact_cache::RedactCache>>,
    /// The downloaded face detector alone, loaded once for `sequence.checkPrivacy`
    /// ([`crate::privacy_check`]), without and with the profile pass.
    pub(crate) faces_only: Vec<(bool, DetectFn)>,
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

/// Whether the profile pass's models (person detector, pose network) are downloaded.
pub fn profiles_installed() -> bool {
    crate::transcript::models_dir().is_some_and(|d| filmcraft_vision::catalog::profiles_installed(&d))
}

/// Whether the profile pass runs for `o`: faces are looked for, and it was asked for or (not
/// said) its models are downloaded.
pub fn profiles_on(o: &Options) -> bool {
    o.faces && available() && o.profiles.unwrap_or_else(profiles_installed)
}

// ---------------------------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------------------------

pub(crate) fn unit_p(cmd: &str, p: &Value, k: &str, lo: f64, hi: f64, d: f64) -> Result<f64> {
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
        Some(Value::Array(a)) if a.is_empty() => return Ok((false, false)),
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
            Some(Kind::Target) | None => {
                return Err(bad(cmd, format!("`kinds`: \"{}\" is neither \"faces\" nor \"plates\"", n.chars().take(40).collect::<String>())));
            }
        }
    }
    Ok((faces, plates))
}

/// `targets`: descriptions (`"DP World logo"`), or `{"text", "clips": [id]}` for some clips.
fn targets_p(cmd: &str, p: &Value) -> Result<Vec<TargetP>> {
    let list = match p.get("targets").filter(|v| !v.is_null()) {
        None => return Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= MAX_TARGETS => a,
        Some(Value::Array(_)) => return Err(bad(cmd, format!("at most {MAX_TARGETS} targets"))),
        Some(_) => return Err(bad(cmd, "`targets` is a list of descriptions (\"DP World logo\") or {text, clips}")),
    };
    list.iter()
        .map(|t| {
            let (text, clips) = match t {
                Value::String(s) => (s.as_str(), None),
                Value::Object(o) => {
                    let text = o.get("text").or_else(|| o.get("target")).and_then(Value::as_str).ok_or_else(|| bad(cmd, "a target object needs `text`"))?;
                    let clips = match (o.get("clips").filter(|v| !v.is_null()), o.get("clip").filter(|v| !v.is_null())) {
                        (Some(Value::Array(a)), _) if a.len() <= MAX_CLIPS => Some(
                            a.iter().map(|v| v.as_u64().map(ClipId)).collect::<Option<Vec<_>>>().ok_or_else(|| bad(cmd, "a target's `clips` are clip ids"))?,
                        ),
                        (Some(_), _) => return Err(bad(cmd, "a target's `clips` are clip ids")),
                        (None, Some(c)) => Some(vec![ClipId(c.as_u64().ok_or_else(|| bad(cmd, "a target's `clip` is a clip id"))?)]),
                        (None, None) => None,
                    };
                    (text, clips)
                }
                _ => return Err(bad(cmd, "a target is a description or {text, clips}")),
            };
            let text = text.trim();
            if text.is_empty() || text.chars().count() > MAX_TARGET_CHARS || text.chars().any(char::is_control) {
                return Err(bad(cmd, format!("a target is a description on one line, 1 to {MAX_TARGET_CHARS} characters")));
            }
            Ok(TargetP { text: text.to_string(), clips })
        })
        .collect()
}

/// `profiles`: true, false, or absent / null (on when the models are downloaded).
pub(crate) fn profiles_p(cmd: &str, p: &Value) -> Result<Option<bool>> {
    match p.get("profiles").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(bad(cmd, "`profiles` is true or false (the people-and-pose pass for faces in profile)")),
    }
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
        sure_face: unit_p(cmd, p, "sureFace", 0.05, 1.0, f64::from(d.sure_face))? as f32,
        style,
        strength: unit_p(cmd, p, "strength", 1.0, 100.0, d.strength)?,
        padding: unit_p(cmd, p, "padding", 0.0, 1.0, d.padding)?,
        verify: flag("verify", d.verify)?,
        dry_run: flag("dryRun", d.dry_run)?,
        analysis_fps: unit_p(cmd, p, "analysisFps", 1.0, 60.0, d.analysis_fps)?,
        face_side: unit_p(cmd, p, "faceSide", 160.0, 1920.0, d.face_side as f64)? as usize,
        plate_side: unit_p(cmd, p, "plateSide", 160.0, 2560.0, d.plate_side as f64)? as usize,
        replace: flag("replace", d.replace)?,
        targets: targets_p(cmd, p)?,
        target_every: unit_p(cmd, p, "targetEvery", 0.25, 10.0, d.target_every)?,
        fresh: flag("fresh", d.fresh)?,
        profiles: profiles_p(cmd, p)?,
    })
    .and_then(|o| {
        if o.faces || o.plates || !o.targets.is_empty() { Ok(o) } else { Err(bad(cmd, "nothing to hide: `kinds` faces and/or plates, or `targets`")) }
    })
}

/// `exclude`: a box `{x, y, w, h}` (0..1 of the clip's picture; `clip` and `seconds` of media
/// time optional) or a list of them.
pub(crate) fn exclude_p(cmd: &str, p: &Value) -> Result<Vec<ExcludeP>> {
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
            // (a reframed 360° clip is looked at through its virtual camera: what is shown)
            let Some(src) = crate::vr360::clip_source(s, it) else {
                skipped.push(skip("its media is offline"));
                continue;
            };
            let Some((w, h)) = filmcraft_render::clip_size(&s.project, it).filter(|(w, h)| *w > 0 && *h > 0) else { continue };
            let n = (it.duration.0 / fd.0).max(1);
            // a photo is one picture however long it is held: looked at once, and what is found
            // there is hidden for the whole clip (seen once is seen: there is no second look)
            let still = s.project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.kind == filmcraft_media::MediaKind::Still);
            let times = if still { vec![it.source_time_at(it.start)] } else { grid_times(it, fd, step, n) };
            total = total.saturating_add(times.len());
            if total > MAX_FRAMES {
                return Err(EngineError::Other(format!("too many frames to analyse (more than {MAX_FRAMES}); lower analysisFps or redact fewer clips")));
            }
            let media = (times.first().copied().unwrap_or(it.source_in), times.last().copied().unwrap_or(it.source_in));
            let size = (f64::from(w), f64::from(h));
            // how large the picture is shown: the rules are about the sequence's frame
            let display = s.project.conformed_clip_size(it, q.settings.par).filter(|d| d.0 > 0.0 && d.1 > 0.0).unwrap_or(size);
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
                min_hits: if still { 1 } else { o.keep.min_hits },
                ..o.keep
            };
            // the smallest face to hide is at least 16 px where the detector looks (in steps, so
            // two windows of a shot zoomed a little differently see the same picture, and share
            // the cache's detections)
            let long = size.0.max(size.1);
            let needed = 16.0 / (keep.min_face_height.max(0.004) * size.1 / long);
            let stepped = FACE_SIDES.iter().copied().find(|s| *s as f64 >= needed).unwrap_or(1920);
            let face_side = stepped.clamp(o.face_side, 1920.max(o.face_side)).min(long as usize).max(160);
            let targets: Vec<String> = o.targets.iter().filter(|t| t.clips.as_ref().is_none_or(|c| c.contains(&it.id))).map(|t| t.text.clone()).collect();
            let look = if targets.is_empty() { 0 } else { crate::redact_targets::LOOK_SIDE };
            // the frames are decoded large enough for the face detector's closer look
            let face_px = close_side(face_side, long as usize).unwrap_or(face_side);
            let side = (if o.faces { face_px } else { 0 }).max(if o.plates { o.plate_side } else { 0 }).max(look).max(64) as f64;
            let scale = (side / long).min(1.0) as f32;
            let ex = exclude.iter().filter(|e| e.clip.is_none_or(|c| c == it.id)).map(|e| e.ex).collect();
            // the file the frames come from, for the cache (not a proxy playing instead)
            let resolved = s.project.resolve_media(it.item);
            let file = resolved.and_then(|(_, m, _)| match &m.media {
                MediaRef::File { path } if !(s.media.use_proxies() && m.proxy.is_some()) => Some(path.clone()),
                _ => None,
            });
            let mut desc = resolved.map_or_else(String::new, |(_, m, _)| format!("{w}x{h}|{}|{:?}", m.duration().0, m.interpret.frame_rate));
            // a reframed 360° clip's frames are its camera's view: the camera decides them too
            if let Some(cam) = filmcraft_project::vr::reframe_of(it) {
                desc.push_str(&format!("|camera {}", crate::vr360::camera_key(cam)));
            }
            let w = Work {
                clip: it.id,
                also: Vec::new(),
                item: it.item,
                name: it.name.clone(),
                src,
                times,
                size,
                scale,
                keep,
                face_side,
                targets,
                exclude: ex,
                media,
                file,
                desc,
            };
            // a copy of a clip already listed (same media, same frames, same exclusions, the same
            // camera on 360° media) is analysed once
            match work.iter_mut().find(|o: &&mut Work| {
                o.item == w.item && o.times == w.times && o.exclude == w.exclude && o.keep == w.keep && o.targets == w.targets && o.desc == w.desc
            }) {
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

/// The steps of the picture size the face detector sees (its long side, pixels).
const FACE_SIDES: [usize; 6] = [640, 800, 960, 1280, 1600, 1920];

/// The face detector's second look at a frame, twice as close (its long side, pixels), when the
/// picture (`long`) allows one at least a quarter closer than `face_side`, at most 1920. At the
/// first look's size a face of 5 to 8 % of the frame is 18 to 30 px, where YuNet often misses
/// it (a visitor at 7 % next to Ameca was found at 1280 px and not at 640); a face filling the
/// frame is past what it was trained on at the closer size, so both are kept (one suppression
/// over the two).
pub fn close_side(face_side: usize, long: usize) -> Option<usize> {
    let close = face_side.saturating_mul(2).min(long).min(1920);
    (close > face_side && close.saturating_mul(4) >= face_side.saturating_mul(5)).then_some(close)
}

/// The media times clip `it` (`n` sequence frames of `fd`) is analysed at: its first and last
/// frame, and between them every multiple of the analysis step in media time (`step` sequence
/// frames at the clip's speed). A grid, so two windows of one clip share the frames they overlap
/// (and the redaction cache's entries). In order, without repeats.
fn grid_times(it: &filmcraft_project::TrackItem, fd: Tick, step: i64, n: i64) -> Vec<Tick> {
    let first = it.source_time_at(it.start);
    let last = it.source_time_at(it.start + Tick(fd.0.saturating_mul(n.saturating_sub(1))));
    let (lo, hi) = (first.min(last), first.max(last));
    let speed = if it.speed.is_finite() && it.speed != 0.0 { it.speed.abs().clamp(1e-3, 1e3) } else { 1.0 };
    let g = ((fd.0 as f64) * step.max(1) as f64 * speed).round().clamp(1.0, 1e15) as i64;
    let k0 = lo.0.div_euclid(g) + i64::from(lo.0.rem_euclid(g) != 0);
    let k1 = hi.0.div_euclid(g);
    let mut times = vec![first, last];
    // (a hostile clip gets one past the job's limit, which then refuses it)
    let count = usize::try_from(k1.saturating_sub(k0).saturating_add(1).max(0)).unwrap_or(0).min(MAX_FRAMES + 1);
    times.extend((0..count).map(|k| Tick((k0 + k as i64).saturating_mul(g))));
    times.sort();
    times.dedup();
    times
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

/// The downloaded detectors, for what `o` looks for (with the profile pass when
/// [`profiles_on`]: the heads of the people whose face shows and that the face detector missed).
#[cfg(feature = "privacy-models")]
pub(crate) fn load_detectors(o: &Options) -> std::result::Result<DetectFn, String> {
    let dir = crate::transcript::models_dir().ok_or("no data directory for the detectors")?;
    let det = filmcraft_vision::Detectors::load(&dir, o.faces, o.plates).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => format!("the face and plate detectors are not downloaded yet ({DOWNLOAD_CMD})"),
        e => e.to_string(),
    })?;
    let people = if profiles_on(o) {
        Some(filmcraft_vision::ProfileDetector::load_on(&dir, true).map_err(|e| match e {
            filmcraft_vision::VisionError::NotInstalled => {
                format!("the person and pose models for faces in profile are not downloaded yet ({DOWNLOAD_CMD}; or `profiles: false`)")
            }
            e => e.to_string(),
        })?)
    } else {
        None
    };
    let o = o.clone();
    Ok(Arc::new(move |rgba: &[u8], w: usize, h: usize, face_side: usize| {
        let mut out = Vec::new();
        if let Some(f) = det.face.as_ref().filter(|_| o.faces) {
            let mut faces = f.detect(rgba, w, h, face_side, o.face_score).unwrap_or_default();
            // and closer, where the picture allows ([`close_side`]): the faces of a crowd
            if let Some(close) = close_side(face_side, w.max(h)) {
                faces.extend(f.detect(rgba, w, h, close, o.face_score).unwrap_or_default());
                faces = filmcraft_vision::post::nms(faces, 0.3, 750);
            }
            // the heads the face detector missed: in profile, half hidden
            let heads = people.as_ref().map(|p| p.heads(rgba, w, h, &faces).unwrap_or_default()).unwrap_or_default();
            out.extend(faces);
            out.extend(heads);
        }
        if let Some(p) = det.plate.as_ref().filter(|_| o.plates) {
            out.extend(p.detect(rgba, w, h, o.plate_side, o.plate_score).unwrap_or_default());
        }
        out
    }))
}

#[cfg(not(feature = "privacy-models"))]
pub(crate) fn load_detectors(_: &Options) -> std::result::Result<DetectFn, String> {
    Err("automatic redaction is not available in this build (built without the `privacy-models` feature)".into())
}

/// The redaction cache as one job uses it ([`crate::redact_cache`]): each work's media key (none:
/// not cached), the detector's identity (none: not cached), and what came from it.
struct Cached<'a> {
    cache: &'a crate::redact_cache::RedactCache,
    /// Read nothing ([`Options::fresh`]); what is found is still kept.
    fresh: bool,
    media: Vec<Option<String>>,
    detector: Option<String>,
    /// Frames whose detections came from the cache, frames decoded, verdicts and looks reused.
    frames: AtomicUsize,
    decoded: AtomicUsize,
    verdicts: AtomicUsize,
    looks: AtomicUsize,
}

impl Cached<'_> {
    fn stats(&self) -> Value {
        json!({
            "framesFromCache": self.frames.load(Ordering::Relaxed),
            "framesDecoded": self.decoded.load(Ordering::Relaxed),
            "verdictsFromCache": self.verdicts.load(Ordering::Relaxed),
            "looksFromCache": self.looks.load(Ordering::Relaxed),
        })
    }

    /// The key of `w`'s detections (media key `media`): with the profile pass, its models' ids
    /// too (entries made without it, or with other models, are analysed again once).
    fn detections_key(&self, media: &str, w: &Work, o: &Options) -> Option<String> {
        let d = self.detector.as_ref()?;
        // the face detector's closer look (none, or its size): its finds are in the entry
        let close = close_side(w.face_side, (w.size.0.max(w.size.1) * f64::from(w.scale)).round() as usize).unwrap_or(0);
        Some(detections_key(d, media, w.face_side, w.scale, close, o))
    }
}

/// The redaction cache's key of a file's detections (`media`) by detector `d` with these
/// settings. With the profile pass, its models' ids and the rules that make faces of poses
/// (`filmcraft_vision::head::RULES`) too: entries made without it, with other models or under
/// other rules, are analysed again once; without it the key is what it always was.
fn detections_key(d: &str, media: &str, face_side: usize, scale: f32, close: usize, o: &Options) -> String {
    let profiles = if o.profiles == Some(true) {
        format!(" profiles={} heads=v{}", filmcraft_vision::catalog::profile_model_ids(), filmcraft_vision::head::RULES)
    } else {
        String::new()
    };
    format!(
        "det v{}|{media}|{d}{profiles}|faces={} plates={} fs={:.3} ps={:.3} side={} pside={} scale={:.5} close={}",
        crate::redact_cache::VERSION,
        o.faces,
        o.plates,
        o.face_score,
        o.plate_score,
        face_side,
        o.plate_side,
        scale,
        close
    )
}

/// The vision model's looks for named targets through the cache: a look at a media time asked
/// before (the same file, picture, model and targets) is answered from it.
struct CachedFinder<'a> {
    inner: &'a dyn VisionJudge,
    known: crate::redact_cache::Looks,
    fresh: Mutex<crate::redact_cache::Looks>,
    /// The media time (ticks) of the frame being looked at.
    now: std::sync::atomic::AtomicI64,
    hits: &'a AtomicUsize,
}

impl VisionJudge for CachedFinder<'_> {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn judge(&self, kind: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
        self.inner.judge(kind, crops)
    }
    fn find(&self, frame: &Crop, targets: &[String]) -> std::result::Result<Vec<(usize, [f64; 4])>, String> {
        let t = self.now.load(Ordering::Relaxed);
        // what can be the target, cached or fresh (a cache written before a rule tightened still
        // holds the boxes it dropped)
        let plausible = |found: &[(usize, [f64; 4])]| -> Vec<(usize, [f64; 4])> {
            found.iter().copied().filter(|(_, b)| crate::redact_targets::plausible_target(*b, frame.width, frame.height)).collect()
        };
        if let Some(found) = self.known.get(&t) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(plausible(found));
        }
        let found = self.inner.find(frame, targets)?;
        lock(&self.fresh).insert(t, found.clone());
        Ok(plausible(&found))
    }
}

/// Analyse one clip: detect on every analysed frame (or take what the cache keeps for it), link,
/// decide. `None` when stopped.
fn analyse(
    w: &Work,
    det: &DetectFn,
    o: &Options,
    vision: Option<&dyn VisionJudge>,
    sh: &Shared<'_>,
    cached: Option<(&Cached<'_>, &str)>,
) -> Option<ClipRedact> {
    let mut frames = Vec::with_capacity(w.times.len());
    let mut missed = 0usize;
    // what the cache knows of this file: detections, and the targets' looks
    let det_key = cached.and_then(|(c, media)| c.detections_key(media, w, o)).filter(|_| o.faces || o.plates);
    let known: crate::redact_cache::Detections = match (cached, &det_key) {
        (Some((c, _)), Some(k)) if !c.fresh => c.cache.load(k).unwrap_or_default(),
        _ => Default::default(),
    };
    let mut found: crate::redact_cache::Detections = Default::default();
    let look_key = cached.zip(vision).filter(|_| !w.targets.is_empty()).map(|((c, media), v)| {
        let key = format!(
            "find v{}|{media}|{}|scale={:.5} side={}|{}",
            crate::redact_cache::VERSION,
            v.name(),
            w.scale,
            crate::redact_targets::LOOK_SIDE,
            w.targets.join("\u{1f}")
        );
        let known = if c.fresh { Default::default() } else { c.cache.load(&key).unwrap_or_default() };
        (key, CachedFinder { inner: v, known, fresh: Mutex::default(), now: Default::default(), hits: &c.looks })
    });
    let finder: Option<&dyn VisionJudge> = match &look_key {
        Some((_, f)) => Some(f),
        None => vision,
    };
    let mut follower = finder.filter(|_| !w.targets.is_empty()).map(|v| crate::redact_targets::Follower::new(&w.targets, o.target_every, v));
    for (k, mt) in w.times.iter().enumerate() {
        if sh.cancelled() {
            return None;
        }
        let known_here = known.get(&mt.0);
        let detections = match known_here {
            // nothing to decode: the detections are known and no target is followed
            Some(d) if follower.is_none() => {
                if let Some((c, _)) = cached {
                    c.frames.fetch_add(1, Ordering::Relaxed);
                }
                d.clone()
            }
            _ => match w.src.video_frame(filmcraft_media::FrameRequest { time: *mt, scale: w.scale }) {
                Ok(f) => {
                    if let Some((c, _)) = cached {
                        c.decoded.fetch_add(1, Ordering::Relaxed);
                    }
                    let rgba = f.to_rgba8();
                    let (fw, fh) = (f.width as usize, f.height as usize);
                    if let Some(fo) = follower.as_mut() {
                        if let Some((_, f)) = &look_key {
                            f.now.store(mt.0, Ordering::Relaxed);
                        }
                        fo.push(&rgba, fw, fh, mt.seconds());
                    }
                    match known_here {
                        Some(d) => {
                            if let Some((c, _)) = cached {
                                c.frames.fetch_add(1, Ordering::Relaxed);
                            }
                            d.clone()
                        }
                        None if o.faces || o.plates => {
                            let d = det(&rgba, fw, fh, w.face_side);
                            if det_key.is_some() {
                                found.insert(mt.0, d.clone());
                            }
                            d
                        }
                        None => Vec::new(),
                    }
                }
                Err(_) => {
                    // a frame that does not decode shows nothing; its gap is filled if short
                    missed += 1;
                    known_here.cloned().unwrap_or_default()
                }
            },
        };
        frames.push(filmcraft_vision::track::Frame { t: mt.seconds(), detections });
        sh.frame();
        if k % 16 == 0 {
            *lock(&sh.prog.status) =
                format!("{}: frame {} of {} (clip {} of {})", w.name, k + 1, w.times.len(), sh.clips_done.load(Ordering::Relaxed) + 1, sh.clips);
        }
    }
    let tracks = filmcraft_vision::track::link(&frames, &o.link);
    let mut tracks: Vec<(Track, Option<Skip>)> = tracks
        .into_iter()
        .map(|t| {
            let v = filmcraft_vision::track::verdict(&t, &w.keep, &w.exclude);
            (t, v)
        })
        .collect();
    let followed = follower.map(|f| f.finish(o.link.smooth)).unwrap_or_default();
    tracks.extend(followed.tracks.into_iter().map(|t| (t, None)));
    // what was found for the first time, kept for the next run
    if let (Some((c, _)), Some(key)) = (cached, &det_key)
        && !found.is_empty()
    {
        c.cache.update::<crate::redact_cache::Detections>(key, |e| e.append(&mut found));
    }
    if let (Some((c, _)), Some((key, f))) = (cached, look_key) {
        let mut fresh = f.fresh.into_inner().unwrap_or_else(PoisonError::into_inner);
        if !fresh.is_empty() {
            c.cache.update::<crate::redact_cache::Looks>(&key, |e| e.append(&mut fresh));
        }
    }
    let interval = if w.times.len() > 1 { (w.media.1 - w.media.0).seconds().abs() / (w.times.len() - 1) as f64 } else { 0.0 };
    Some(ClipRedact {
        clip: w.clip,
        also: w.also.clone(),
        name: w.name.clone(),
        size: w.size,
        tracks,
        frames: w.times.len(),
        interval,
        media: w.media,
        missed,
        looks: followed.looks,
        found: followed.found,
        look_errors: followed.errors,
        robots: w.exclude.clone(),
    })
}

/// Analyse every clip, a few at a time (decoding one while detecting in another).
fn analyse_all(
    work: &[Work],
    det: &DetectFn,
    o: &Options,
    vision: Option<&dyn VisionJudge>,
    sh: &Shared<'_>,
    cached: Option<&Cached<'_>>,
) -> Option<Vec<ClipRedact>> {
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get() / 3).clamp(1, 4).min(work.len().max(1));
    let next = AtomicUsize::new(0);
    let made: Mutex<Vec<(usize, ClipRedact)>> = Mutex::new(Vec::with_capacity(work.len()));
    let body = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(w) = work.get(i) else { return };
            let media = cached.and_then(|c| Some((c, c.media.get(i)?.as_deref()?)));
            // a clip that panics in a decoder is a clip with nothing found, not a dead job
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| analyse(w, det, o, vision, sh, media))) {
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

/// A question for the vision check: (kind, clip, track, crop, where its answer is kept: the
/// cache entry and the moment's key and box).
type Ask = (Kind, usize, usize, Crop, Option<(usize, (String, [f64; 4]))>);

/// The vision check's answers kept for each media file: (cache key, kept, new).
type Kept = Vec<(String, crate::redact_cache::Verdicts, crate::redact_cache::Verdicts)>;

/// The cache entry (in `kept`) of the vision check's answers about work `wi`'s file under `what`
/// (`judge` for the first look, `confirm` for the second): its index, loaded the first time.
fn kept_slot(kept: &mut Kept, cached: Option<&Cached<'_>>, wi: usize, what: &str, name: &str) -> Option<usize> {
    let cc = cached?;
    let media = cc.media.get(wi)?.as_ref()?;
    let key = format!("{what} v{}|{media}|{name}", crate::redact_cache::VERSION);
    Some(match kept.iter().position(|k| k.0 == key) {
        Some(i) => i,
        None => {
            let old = if cc.fresh { Default::default() } else { cc.cache.load(&key).unwrap_or_default() };
            kept.push((key, old, Default::default()));
            kept.len() - 1
        }
    })
}

/// A track's moment shown to the vision check: a face at its largest sighting, a plate at its
/// widest.
fn best_point(t: &Track) -> Option<&filmcraft_vision::track::TrackPoint> {
    let size = |p: &&filmcraft_vision::track::TrackPoint| if t.kind == Kind::Plate { p.bbox[2] } else { p.bbox[3] };
    t.points.iter().filter(|p| p.seen).max_by(|a, b| size(a).total_cmp(&size(b)))
}

/// The crop of work `w`'s frame around track `t` at [`best_point`] (`mark`: the box outlined in
/// red), from the frame at most 1080 px a side.
fn crop_of(w: &Work, t: &Track, mark: bool) -> Option<Crop> {
    let best = best_point(t)?;
    let scale = (1080.0 / w.size.0.max(w.size.1)).min(1.0) as f32;
    let f = w.src.video_frame(filmcraft_media::FrameRequest { time: Tick::from_seconds_f64(best.t), scale }).ok()?;
    crop_around(&f.to_rgba8(), f.width as usize, f.height as usize, &best.bbox, mark)
}

/// The vision check: one crop per track still to be hidden (a face at its largest sighting, a
/// plate at its widest), shown to the judge [`JUDGE_BATCH`] at a time; tracks it calls
/// something else (a robot's face, a sign) are left alone. Answers kept in the cache from an
/// earlier run (the same track, at any of its moments) are not asked again.
///
/// Faces inside the subject's box ([`Skip::Excluded`]: the robot's, going by the box alone) are
/// shown too, and hidden unless the judge says they are not a person: the box is coarse, and
/// visitors crowd around a robot (two of them were left sharp next to Ameca, 2026-10-21). A
/// face the judge calls not a person is looked at again, alone and outlined
/// ([`VisionJudge::confirm`]), and left sharp only if that agrees; a face the detector is sure
/// of ([`Options::sure_face`]) is left sharp only on the subject (inside its box at some moment),
/// whatever the judge says of it elsewhere.
fn verify(made: &mut [ClipRedact], work: &[Work], judge: &dyn VisionJudge, o: &Options, sh: &Shared<'_>, cached: Option<&Cached<'_>>) -> Verified {
    let name = judge.name();
    let mut kept: Kept = Vec::new();
    let mut asks: Vec<Ask> = Vec::new();
    // (kind, clip, track, answer) from the cache
    let mut known: Vec<(Kind, usize, usize, Option<bool>)> = Vec::new();
    let mut on_subject = 0usize;
    for (ci, c) in made.iter().enumerate() {
        let Some(wi) = work.iter().position(|w| w.clip == c.clip) else { continue };
        let Some(w) = work.get(wi) else { continue };
        let slot = kept_slot(&mut kept, cached, wi, "judge", &name);
        for (ti, (t, skip)) in c.tracks.iter().enumerate() {
            let subject = *skip == Some(Skip::Excluded) && t.kind == Kind::Face;
            if (skip.is_some() && !subject) || t.kind == Kind::Target {
                continue;
            }
            if sh.cancelled() {
                return Verified::Skipped("stopped".into());
            }
            on_subject += usize::from(subject);
            let Some(best) = best_point(t) else { continue };
            let mt = Tick::from_seconds_f64(best.t);
            // the answer about this track kept from an earlier run, at any of its moments
            let moments = std::iter::once(best).chain(t.points.iter().filter(|p| p.seen)).map(|p| (Tick::from_seconds_f64(p.t), &p.bbox));
            if let Some(a) = slot.and_then(|i| kept.get(i)).and_then(|k| crate::redact_cache::verdict_for(&k.1, t.kind, moments)) {
                known.push((t.kind, ci, ti, a));
                continue;
            }
            let vkey = (crate::redact_cache::verdict_key(t.kind, mt), best.bbox);
            *lock(&sh.prog.status) = format!("Checking what was found in {}", c.name);
            if let Some(crop) = crop_of(w, t, t.kind == Kind::Plate) {
                asks.push((t.kind, ci, ti, crop, slot.map(|i| (i, vkey))));
            }
        }
    }
    if let Some(cc) = cached {
        cc.verdicts.fetch_add(known.len(), Ordering::Relaxed);
    }
    // what the model answered is kept, also when a later question fails
    let save = |kept: &Kept| {
        for (key, _, new) in kept.iter().filter(|k| !k.2.is_empty()) {
            if let Some(cc) = cached {
                cc.cache.update::<crate::redact_cache::Verdicts>(key, |e| crate::redact_cache::merge_verdicts(e, new));
            }
        }
    };
    // the first looks: the cache's answers, then the grids
    let mut answers = known.clone();
    let mut failed: Option<String> = None;
    'kinds: for kind in [Kind::Face, Kind::Plate] {
        let mine: Vec<&Ask> = asks.iter().filter(|a| a.0 == kind).collect();
        for batch in mine.chunks(JUDGE_BATCH) {
            if sh.cancelled() {
                save(&kept);
                return Verified::Skipped("stopped".into());
            }
            *lock(&sh.prog.status) = format!("Asking {name} which of {} {}s are real", mine.len(), kind.name());
            let crops: Vec<Crop> = batch.iter().map(|a| a.3.clone()).collect();
            let got = match judge.judge(kind, &crops) {
                Ok(a) => a,
                Err(e) => {
                    failed = Some(e);
                    break 'kinds;
                }
            };
            for ((_, ci, ti, _, slot), a) in batch.iter().copied().zip(got.into_iter().chain(std::iter::repeat(None))) {
                answers.push((kind, *ci, *ti, a));
                if let Some((i, (vkey, b))) = slot
                    && let Some(k) = kept.get_mut(*i)
                {
                    k.2.entry(vkey.clone()).or_default().push((*b, a));
                }
            }
        }
    }
    // the second looks at the faces called not a person (where that would count)
    let mut second: HashMap<(usize, usize), Option<bool>> = HashMap::new();
    let mut overrule: Vec<(usize, usize)> = Vec::new();
    let (mut confirmed, mut confirmed_cached) = (0usize, 0usize);
    for (kind, ci, ti, a) in answers.iter().filter(|_| failed.is_none()) {
        if *kind != Kind::Face || *a != Some(false) {
            continue;
        }
        let Some(c) = made.get(*ci) else { continue };
        let Some((t, _)) = c.tracks.get(*ti) else { continue };
        let Some(wi) = work.iter().position(|w| w.clip == c.clip) else { continue };
        let Some(w) = work.get(wi) else { continue };
        // a face the detector is sure of, away from the subject: hidden whatever the model says
        if t.best_score >= o.sure_face && !w.exclude.iter().any(|e| e.near(t)) {
            overrule.push((*ci, *ti));
            continue;
        }
        if sh.cancelled() {
            save(&kept);
            return Verified::Skipped("stopped".into());
        }
        let Some(best) = best_point(t) else { continue };
        let slot = kept_slot(&mut kept, cached, wi, "confirm", &name);
        let moments = std::iter::once(best).chain(t.points.iter().filter(|p| p.seen)).map(|p| (Tick::from_seconds_f64(p.t), &p.bbox));
        if let Some(a) = slot.and_then(|i| kept.get(i)).and_then(|k| crate::redact_cache::verdict_for(&k.1, Kind::Face, moments)) {
            confirmed_cached += 1;
            second.insert((*ci, *ti), a);
            continue;
        }
        *lock(&sh.prog.status) = format!("Looking again at a face {name} called not a person in {}", c.name);
        let Some(crop) = crop_of(w, t, true) else { continue };
        match judge.confirm(Kind::Face, &crop) {
            Ok(a) => {
                confirmed += 1;
                second.insert((*ci, *ti), a);
                if let Some(k) = slot.and_then(|i| kept.get_mut(i)) {
                    k.2.entry(crate::redact_cache::verdict_key(Kind::Face, Tick::from_seconds_f64(best.t))).or_default().push((best.bbox, a));
                }
            }
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    save(&kept);
    // the decisions
    let (mut not_people, mut not_plates, mut unsure, mut overruled) = (0usize, 0usize, 0usize, 0usize);
    for (kind, ci, ti, a) in answers {
        let Some((_, skip)) = made.get_mut(ci).and_then(|c| c.tracks.get_mut(ti)) else { continue };
        let subject = *skip == Some(Skip::Excluded);
        let a = match (kind, a) {
            (Kind::Face, Some(false)) if overrule.contains(&(ci, ti)) => {
                overruled += 1;
                Some(true)
            }
            (Kind::Face, Some(false)) => match second.get(&(ci, ti)) {
                Some(again) => *again,
                // not looked at again (the model failed): the subject's face keeps its box's
                // answer, any other face is hidden
                None if subject => Some(false),
                None => Some(true),
            },
            (_, a) => a,
        };
        match a {
            Some(false) if kind == Kind::Face => {
                *skip = Some(Skip::NotAPerson);
                not_people += 1;
            }
            Some(false) => {
                *skip = Some(Skip::NotAPlate);
                not_plates += 1;
            }
            None => {
                *skip = None;
                unsure += 1;
            }
            Some(true) => *skip = None,
        }
    }
    if let Some(e) = failed {
        return Verified::Failed(format!("{e}; everything found is hidden but the subject's face"));
    }
    Verified::Done { judge: name, asked: asks.len(), cached: known.len(), not_people, not_plates, unsure, on_subject, confirmed, confirmed_cached, overruled }
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
        Kind::Face => {
            let b = px_box(&shape_box(t, p, padding), size);
            MaskPath::ellipse(Vec2::new(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0), Vec2::new(b[2] / 2.0, b[3] / 2.0))
        }
        // a named target's box is the vision model's, looser than a detector's: a wider margin
        Kind::Target => {
            let b = px_box(&filmcraft_vision::track::padded(&p.bbox, padding + TARGET_MARGIN), size);
            MaskPath::rect(b[0], b[1], b[0] + b[2], b[1] + b[3])
        }
    }
}

/// What a named target's mask adds to `padding` on every side.
const TARGET_MARGIN: f64 = 0.1;

/// The box (normalised) the mask of track `t` fills at point `p`: a face's padded box reaching
/// [`CHIN`] further down, at least [`MIN_FACE_MASK`] tall, wider for its motion
/// (`motion_margin`) and carried past the frame's edges it reaches (its ellipse is inscribed in
/// it), a plate's or a named target's padded box.
fn shape_box(t: &Track, p: &filmcraft_vision::track::TrackPoint, padding: f64) -> [f64; 4] {
    let pad = if t.kind == Kind::Target { padding + TARGET_MARGIN } else { padding };
    let mut b = filmcraft_vision::track::padded(&p.bbox, pad);
    if t.kind != Kind::Face {
        return b;
    }
    let chin = p.bbox[3] * CHIN;
    if chin.is_finite() && chin > 0.0 {
        b[3] += chin;
    }
    // a face at the size limit: reaching further down, to MIN_FACE_MASK tall
    if b[3].is_finite() && b[3] < MIN_FACE_MASK {
        b[3] = MIN_FACE_MASK;
    }
    let [mx, my] = motion_margin(t, p);
    let b = [b[0] - mx, b[1] - my, b[2] + 2.0 * mx, b[3] + 2.0 * my];
    past_edges(&b, &p.bbox)
}

/// How much wider (and taller) a face's mask is on each side at point `p` of track `t` for its
/// motion: half the larger move of its centre to the analysed moment before or after, along
/// each axis (at most half the face's size). The mask's shape is keyframed at the analysed
/// moments and goes in a straight line between them, while a face in a whip pan moves in jerks:
/// between two moments it is at most half a move off that line (2026-10-22, OnRobot at 86.25 s: a
/// face moving 1.6 of its widths a tenth of a second beside its mask). A still face's mask stays
/// as it is.
fn motion_margin(t: &Track, p: &filmcraft_vision::track::TrackPoint) -> [f64; 2] {
    let i = t.points.partition_point(|q| q.t < p.t);
    let centre = |q: &filmcraft_vision::track::TrackPoint| [q.bbox[0] + q.bbox[2] / 2.0, q.bbox[1] + q.bbox[3] / 2.0];
    let c = centre(p);
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    for q in [i.checked_sub(1).and_then(|k| t.points.get(k)), t.points.get(i + 1)].into_iter().flatten() {
        let d = centre(q);
        let (dx, dy) = ((d[0] - c[0]).abs(), (d[1] - c[1]).abs());
        if dx.is_finite() && dy.is_finite() {
            (mx, my) = (mx.max(dx), my.max(dy));
        }
    }
    let cap = |v: f64, size: f64| if size.is_finite() { (v / 2.0).clamp(0.0, size.abs() / 2.0) } else { 0.0 };
    [cap(mx, p.bbox[2]), cap(my, p.bbox[3])]
}

/// How much further down than its padding a face's mask reaches, a share of the face's height:
/// a face box ends at the chin, and the jaw and beard below it, sharp under a blurred face, are
/// what a face detector finds again on the redacted picture (2026-10-22, OnRobot at 87 s: the
/// privacy check's face there was the blur, the chin and the neck, its centre under the mask's
/// lower edge).
pub const CHIN: f64 = 0.15;

/// The least height of a face's mask, a share of the picture's height, reached by extending it
/// further down: a face at the size limit (5 % of the picture, some 50 px in 1080p) has the least
/// sure box, and what a face detector finds again around its blur is the head with its scarf,
/// chin, neck or the hand at its mouth, a box twice the face's height reaching below it
/// (2026-10-22, OnRobot at 90.75 s: a face 4.5 % tall masked, the privacy check's face there 9.7 %
/// tall, its centre under the mask's lower edge). Larger faces' masks are taller anyway.
pub const MIN_FACE_MASK: f64 = 0.09;

/// Points along the outline of a mask's shape filling box `b` (pixels): an ellipse (a face's),
/// else the box's sides.
fn outline_points(b: &[f64; 4], ellipse: bool) -> Vec<(f64, f64)> {
    const N: u32 = 48;
    let (cx, cy, rx, ry) = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, b[2] / 2.0, b[3] / 2.0);
    (0..N)
        .map(|k| {
            let u = f64::from(k) / f64::from(N);
            if ellipse {
                let a = u * std::f64::consts::TAU;
                (cx + rx * a.cos(), cy + ry * a.sin())
            } else {
                let (side, f) = ((u * 4.0).floor(), (u * 4.0).fract());
                match side as u8 {
                    0 => (b[0] + f * b[2], b[1]),
                    1 => (b[0] + b[2], b[1] + f * b[3]),
                    2 => (b[0] + b[2] - f * b[2], b[1] + b[3]),
                    _ => (b[0], b[1] + b[3] - f * b[3]),
                }
            }
        })
        .collect()
}

/// Whether point `(x, y)` is inside the shape filling box `b` (an ellipse, else the box).
fn inside_shape(b: &[f64; 4], ellipse: bool, (x, y): (f64, f64)) -> bool {
    if !ellipse {
        return x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3];
    }
    let (rx, ry) = (b[2] / 2.0, b[3] / 2.0);
    if !(rx > 0.0 && ry > 0.0) {
        return false;
    }
    let (dx, dy) = ((x - b[0] - rx) / rx, (y - b[1] - ry) / ry);
    dx * dx + dy * dy <= 1.0
}

/// The distance between boxes `a` and `b` (0 where they meet).
fn box_gap(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dx = (b[0] - (a[0] + a[2])).max(a[0] - (b[0] + b[2])).max(0.0);
    let dy = (b[1] - (a[1] + a[3])).max(a[1] - (b[1] + b[3])).max(0.0);
    let d = dx.hypot(dy);
    if d.is_finite() { d } else { 0.0 }
}

/// The gap, pixels, between the shapes filling boxes `a` and `b` (ellipses or boxes): 0 where
/// they touch or overlap, else the nearest two points of their outlines.
fn shape_gap(a: &[f64; 4], ea: bool, b: &[f64; 4], eb: bool) -> f64 {
    if !a.iter().chain(b).all(|v| v.is_finite()) {
        return 0.0;
    }
    let (pa, pb) = (outline_points(a, ea), outline_points(b, eb));
    if pa.iter().any(|p| inside_shape(b, eb, *p)) || pb.iter().any(|p| inside_shape(a, ea, *p)) {
        return 0.0;
    }
    let best = pa.iter().flat_map(|p| pb.iter().map(move |q| (p.0 - q.0).hypot(p.1 - q.1))).fold(f64::INFINITY, f64::min);
    if best.is_finite() { best } else { 0.0 }
}

/// The point of `t` nearest time `s` (its points are in time order).
fn point_near(t: &Track, s: f64) -> Option<&filmcraft_vision::track::TrackPoint> {
    let i = t.points.partition_point(|q| q.t < s);
    let after = t.points.get(i);
    let before = i.checked_sub(1).and_then(|k| t.points.get(k));
    match (before, after) {
        (Some(a), Some(b)) => Some(if (s - a.t).abs() <= (b.t - s).abs() { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// How far (pixels) the blur of the mask of track `t` reaches beyond its shape at each of its
/// points: `feather` (its soft edge wholly outside the shape), less where another mask of the
/// clip is near, so that each reaches at most half way across the gap between the two shapes
/// ([`SOFT_EDGE_HOLD`] before and after) and neighbouring faces are blurred apart, a sharp line
/// between them, not smeared into one blob — unless their shapes nearly touch, where it keeps a
/// quarter of `feather` ([`soft_edge`] makes it a soft edge).
fn reaches(t: &Track, c: &ClipRedact, padding: f64, feather: f64) -> Vec<f64> {
    let feather = if feather.is_finite() { feather.max(1.0) } else { 1.0 };
    let least = feather / 4.0;
    let others: Vec<&Track> = c.hidden().filter(|o| !std::ptr::eq(*o, t) && *o != t).collect();
    let raw: Vec<f64> = t
        .points
        .iter()
        .map(|p| {
            let a = px_box(&shape_box(t, p, padding), c.size);
            let mut r = feather;
            for o in &others {
                let (Some(first), Some(last)) = (o.points.first(), o.points.last()) else { continue };
                // the neighbour's mask shows from its linger and fade before its first sighting
                // to as long after its last
                if !(p.t >= first.t - linger(o, first) - FADE - 1e-9 && p.t <= last.t + linger(o, last) + FADE + 1e-9) {
                    continue;
                }
                let Some(q) = point_near(o, p.t) else { continue };
                let b = px_box(&shape_box(o, q, padding), c.size);
                if box_gap(&a, &b) >= 2.0 * r {
                    continue;
                }
                r = r.min(shape_gap(&a, t.kind == Kind::Face, &b, o.kind == Kind::Face) / 2.0);
            }
            if r.is_finite() { r.clamp(least, feather) } else { least }
        })
        .collect();
    // each the least within SOFT_EDGE_HOLD: it draws in before a neighbour comes near and
    // reaches out again after it has gone, not frame by frame
    let mut out = Vec::with_capacity(raw.len());
    for (i, p) in t.points.iter().enumerate() {
        let mut r = raw.get(i).copied().unwrap_or(feather);
        for (q, x) in t.points.iter().zip(&raw).skip(i.saturating_sub(64)).take(129) {
            if (q.t - p.t).abs() <= SOFT_EDGE_HOLD {
                r = r.min(*x);
            }
        }
        out.push(r);
    }
    out
}

/// A mask's soft edge (feather, expansion; pixels) for its blur to reach `reach` beyond its shape
/// (at most `feather`, at least a quarter of it): the whole `feather` wide and wholly outside the
/// shape while it reaches `feather`; drawn in as it reaches less, down to centred on the shape's
/// outline; then narrower, down to half `feather`. The face inside the padded shape stays wholly
/// hidden (the padding is wider than half the narrowest soft edge's ramp inside the outline).
pub fn soft_edge(reach: f64, feather: f64) -> (f64, f64) {
    let feather = if feather.is_finite() { feather.max(1.0) } else { 1.0 };
    let r = if reach.is_finite() { reach.clamp(feather / 4.0, feather) } else { feather / 4.0 };
    if r >= feather / 2.0 { (feather, r - feather / 2.0) } else { (2.0 * r, 0.0) }
}

/// How long, seconds, a mask's soft edge stays drawn in around a moment a neighbour is near.
pub const SOFT_EDGE_HOLD: f64 = 0.2;

/// A face's padded box `b` (normalised) carried past each edge of the frame its box `face`
/// reaches ([`EDGE_MARGIN`]) by half the face's size: a head cut by the frame is where its
/// detectors disagree most (the check's head box ran past the edge while the redaction's stopped
/// short of it, 2026-10-22), and what lies beyond the frame is never seen, so the mask is made
/// generous there at almost no cost in the picture.
fn past_edges(b: &[f64; 4], face: &[f64; 4]) -> [f64; 4] {
    if !face.iter().chain(b).all(|v| v.is_finite()) {
        return *b;
    }
    let (ew, eh) = (face[2].max(0.0) * EDGE_REACH, face[3].max(0.0) * EDGE_REACH);
    let (mut x0, mut y0, mut x1, mut y1) = (b[0], b[1], b[0] + b[2], b[1] + b[3]);
    if face[0] < EDGE_MARGIN {
        x0 -= ew;
    }
    if face[0] + face[2] > 1.0 - EDGE_MARGIN {
        x1 += ew;
    }
    if face[1] < EDGE_MARGIN {
        y0 -= eh;
    }
    if face[1] + face[3] > 1.0 - EDGE_MARGIN {
        y1 += eh;
    }
    [x0, y0, x1 - x0, y1 - y0]
}

/// How far a face's mask is carried past an edge of the frame it reaches, a share of the face.
pub const EDGE_REACH: f64 = 0.5;

/// A mask's soft edge, a share of its track's median box height: the blur fades out over it,
/// beyond the padded shape (which stays wholly hidden).
pub const FEATHER: f64 = 0.16;

/// [`Options::padding`] by default: a share of the box added on every side.
pub const PADDING: f64 = 0.15;

/// How long, seconds, a mask stays fully on before a face is first found and after it is last
/// found, where it was then (at least an analysed interval): the detector loses a face going
/// behind something before it is gone (a visitor's smile under Ameca's head, 2026-10-21) and
/// finds one coming out a little after it shows. Longer at the frame's edge ([`EDGE_LINGER`]),
/// and the mask fades in and out beyond ([`FADE`]).
pub const LINGER: f64 = 0.4;

/// The mask hiding track `t` (number `n` of its kind): a Mask Path keyframe at every analysed
/// moment, the ends held, the Opacity fully on from [`LINGER`] (or [`EDGE_LINGER`]) before the
/// first sighting to as long after the last, fading in and out over [`FADE`] beyond.
pub fn mask_for(t: &Track, n: usize, c: &ClipRedact, padding: f64) -> Option<Mask> {
    let first = t.points.first()?;
    let last = t.points.last()?;
    let interval = if c.interval.is_finite() { c.interval.max(0.0) } else { 0.0 };
    let (lo, hi) = (c.media.0.min(c.media.1), c.media.0.max(c.media.1));
    // fully on from a linger before the first sighting to one after the last, fading in and
    // out beyond
    let on0 = Tick::from_seconds_f64(first.t - linger(t, first).max(interval));
    let on1 = Tick::from_seconds_f64(last.t + linger(t, last).max(interval));
    let fade = Tick::from_seconds_f64(FADE);
    let (f0, f1) = (on0 - fade, on1 + fade);
    let (start, end) = (f0.max(lo), f1.min(hi));
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
    let mut m = Mask::new(mask_name(t, n), path0);
    if keys.len() > 1 {
        m.path.keyframes = keys;
    }
    // a soft edge outside the shape: the whole padded shape is hidden, the blur fades out beyond
    let mut heights: Vec<f64> = t.points.iter().map(|p| p.bbox[3] * c.size.1).filter(|h| h.is_finite()).collect();
    heights.sort_by(f64::total_cmp);
    let median = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
    let feather = (median * FEATHER).clamp(1.0, 300.0);
    m.feather = Param::new(ParamValue::Float(feather));
    m.expansion = Param::new(ParamValue::Float(feather / 2.0));
    // drawn in where a neighbour's mask is near (keyframed at the analysed moments)
    let reach = reaches(t, c, padding, feather);
    if reach.iter().any(|r| *r < feather - 1e-6) {
        let (mut fk, mut ek): (Vec<Keyframe>, Vec<Keyframe>) = (Vec::new(), Vec::new());
        for (p, r) in t.points.iter().zip(&reach) {
            let time = Tick::from_seconds_f64(p.t);
            if fk.last().is_none_or(|k: &Keyframe| k.time < time) {
                let (f, e) = soft_edge(*r, feather);
                fk.push(Keyframe::new(time, ParamValue::Float(f)));
                ek.push(Keyframe::new(time, ParamValue::Float(e)));
            }
        }
        m.feather.keyframes = fk;
        m.expansion.keyframes = ek;
    }
    // the opacity: 0 until f0, up to 100 at on0, 100 until on1, down to 0 at f1 (linear), as the
    // clip's media range shows it
    let at = |time: Tick| -> f64 {
        if time <= f0 || time >= f1 {
            0.0
        } else if time < on0 {
            100.0 * (time - f0).0 as f64 / fade.0.max(1) as f64
        } else if time > on1 {
            100.0 * (f1 - time).0 as f64 / fade.0.max(1) as f64
        } else {
            100.0
        }
    };
    let mut times: Vec<Tick> = [lo, f0, on0, on1, f1, hi].into_iter().filter(|x| *x >= lo && *x <= hi).collect();
    times.sort();
    times.dedup();
    let op: Vec<Keyframe> = times.iter().map(|x| Keyframe::new(*x, ParamValue::Float(at(*x).clamp(0.0, 100.0)))).collect();
    m.opacity = Param::new(ParamValue::Float(at(lo).clamp(0.0, 100.0)));
    if op.iter().any(|k| k.value != ParamValue::Float(100.0)) && op.len() > 1 {
        m.opacity.keyframes = op;
    }
    Some(m)
}

/// How long a mask stays fully on before track `t` is first seen (at `p`, its first point) or
/// after it is last seen (`p`, its last): [`EDGE_LINGER`] where the face is at the frame's edge
/// (cut by it: the detectors are least sure there, and the person is likely still there, half
/// in the frame), else [`LINGER`].
fn linger(_t: &Track, p: &filmcraft_vision::track::TrackPoint) -> f64 {
    if filmcraft_vision::track::at_edge(&p.bbox, EDGE_MARGIN) { EDGE_LINGER } else { LINGER }
}

/// [`LINGER`] for a face at the frame's edge (within [`EDGE_MARGIN`] of it): 2026-10-22, most of
/// the faces the privacy check still saw on the Ameca clip were people cut by the frame whose
/// pose came and went.
pub const EDGE_LINGER: f64 = 1.0;
/// A box this close to the frame's edge (a share of the frame) is at the edge.
pub use filmcraft_vision::track::EDGE_MARGIN;
/// How long, seconds, a mask fades in before it is fully on and out after: a face's blur comes
/// and goes softly instead of switching (it is fully on whenever the face was seen, and
/// [`LINGER`] around that).
pub const FADE: f64 = 0.25;

/// The sizes of the tracks one blur (or mosaic) effect hides are within this factor of each other:
/// a clip gets one effect per band, so a small face is not smeared with a large face's blur.
pub const SIZE_BAND: f64 = 2.0;

/// The blur's blurriness (twice its σ, clip pixels) per pixel of a face's height at the default
/// strength (60): σ 0.15 of the face's height, the lightest blur that defeats the face detector
/// itself (the privacy check's proxy for a face that shows). Measured on the 687 faces of 174
/// frames of the OnRobot and Ameca clips, each blurred inside its mask (2026-10-22, ignored test
/// `blurred_faces_defeat_the_face_detector` in `filmcraft-vision`): at σ 0.10 h YuNet still found
/// 21 % of them and 4 with their centre outside the mask (a face the check would flag); at 0.15 h
/// 12 %, none outside; 0.20 h 10 %. (A face's identity lives in the details from about 4 to 16
/// cycles across it, which σ = 0.15 h removes entirely; what the detector still finds is the
/// blob of a head.) Proportional to each face (by its band), so the blur is as light as it can be
/// at any size and at any output resolution; before, every face of a clip had its largest face's
/// σ (0.15 of it), several times what a small one needed.
pub const BLUR_PER_HEIGHT: f64 = 0.3;

/// The size a track is hidden for, clip pixels: a face's (or named target's) largest height, a
/// plate's largest width counting for less.
fn hide_size(t: &Track, c: &ClipRedact) -> f64 {
    let s = if t.kind == Kind::Plate { t.max_size() * c.size.0 * 0.6 } else { t.max_size() * c.size.1 };
    if s.is_finite() { s.max(8.0) } else { 8.0 }
}

/// The blurriness for a face (or plate) `size` px tall at `strength` (1..100):
/// [`BLUR_PER_HEIGHT`] at 60, proportional to the strength, at least 6.
pub fn blurriness(size: f64, strength: f64) -> f64 {
    let s = if strength.is_finite() { strength.clamp(1.0, 100.0) } else { 60.0 };
    let b = size * BLUR_PER_HEIGHT * s / 60.0;
    if b.is_finite() { b.clamp(6.0, 2000.0) } else { 6.0 }
}

/// The effects hiding a clip's tracks: one per size band ([`SIZE_BAND`], smallest first), its
/// strength from the band's largest face or plate ([`blurriness`]), with one mask per track. None
/// when nothing in it is hidden.
pub fn effects_for(c: &ClipRedact, o: &Options) -> Vec<EffectInstance> {
    let mut bands: std::collections::BTreeMap<i64, Vec<(&Track, usize)>> = std::collections::BTreeMap::new();
    for (t, n) in numbered(c) {
        let band = ((hide_size(t, c) / 8.0).ln() / SIZE_BAND.ln()).floor();
        bands.entry(if band.is_finite() { band.clamp(0.0, 64.0) as i64 } else { 0 }).or_default().push((t, n));
    }
    bands.into_values().filter_map(|tracks| effect_of(c, o, &tracks)).collect()
}

/// One effect hiding `tracks` of `c` (each with its number).
fn effect_of(c: &ClipRedact, o: &Options, tracks: &[(&Track, usize)]) -> Option<EffectInstance> {
    let size = tracks.iter().map(|(t, _)| hide_size(t, c)).fold(8.0f64, f64::max);
    let s = (o.strength / 100.0).clamp(0.01, 1.0);
    let mut e = match o.style {
        Style::Blur => {
            let mut e = filmcraft_project::effect::find_effect("gaussian_blur")?.instance();
            *e.param_mut("blurriness")? = Param::new(ParamValue::Float(blurriness(size, o.strength)));
            if let Some(r) = e.param_mut("repeat_edge") {
                *r = Param::new(ParamValue::Bool(true));
            }
            e
        }
        Style::Mosaic => {
            let mut e = filmcraft_project::effect::find_effect("mosaic")?.instance();
            // 16 blocks down the band's largest face at the lowest strength, 5 at the highest
            let block = (size / (16.0 - 11.0 * s)).max(2.0);
            *e.param_mut("horizontal")? = Param::new(ParamValue::Float((c.size.0 / block).round().clamp(1.0, 4000.0)));
            *e.param_mut("vertical")? = Param::new(ParamValue::Float((c.size.1 / block).round().clamp(1.0, 4000.0)));
            if let Some(r) = e.param_mut("sharp") {
                *r = Param::new(ParamValue::Bool(true));
            }
            e
        }
    };
    for (t, n) in tracks {
        if let Some(m) = mask_for(t, *n, c, o.padding) {
            e.masks.push(m);
        }
    }
    (!e.masks.is_empty()).then_some(e)
}

/// The tracks of `c` to hide, each with its number among its kind (per named target).
fn numbered(c: &ClipRedact) -> Vec<(&Track, usize)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    c.hidden()
        .map(|t| {
            let n = counts.entry(format!("{}:{}", t.kind.name(), t.label.as_deref().unwrap_or(""))).or_insert(0);
            *n += 1;
            (t, *n)
        })
        .collect()
}

/// The name of the mask hiding track `t`, number `n` of its kind (`Redact face 2`, `Redact DP
/// World logo 1`).
fn mask_name(t: &Track, n: usize) -> String {
    let what = match (&t.kind, &t.label) {
        (Kind::Target, Some(l)) => l.chars().take(40).collect::<String>(),
        (k, _) => k.name().to_string(),
    };
    format!("{MASK_PREFIX}{what} {n}")
}

/// How much of a clip's picture its masks cover, as shares of the clip's frame: the largest
/// single mask at any of its keyframes (a face's ellipse as an ellipse, a plate or a named target
/// as its box, each padded as it is hidden) and the most hidden at once, with when (media
/// seconds). A mask over a large part of the picture hides the shot, not a face or a logo.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coverage {
    pub largest: f64,
    pub largest_name: String,
    pub largest_at: f64,
    pub most: f64,
    pub most_at: f64,
}

/// The share of the frame the mask of track `t` covers at point `p`.
fn share(t: &Track, p: &filmcraft_vision::track::TrackPoint, padding: f64) -> f64 {
    let b = shape_box(t, p, padding);
    let w = ((b[0] + b[2]).min(1.0) - b[0].max(0.0)).max(0.0);
    let h = ((b[1] + b[3]).min(1.0) - b[1].max(0.0)).max(0.0);
    let a = w * h * if t.kind == Kind::Face { std::f64::consts::FRAC_PI_4 } else { 1.0 };
    if a.is_finite() { a.clamp(0.0, 1.0) } else { 0.0 }
}

/// [`Coverage`] of the masks hiding `c`'s tracks.
pub fn coverage(c: &ClipRedact, padding: f64) -> Coverage {
    let mut cov = Coverage::default();
    // the share hidden at each analysed moment (milliseconds of media time)
    let mut at: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
    for (t, n) in numbered(c) {
        for p in t.points.iter().filter(|p| p.t.is_finite()) {
            let a = share(t, p, padding);
            if a > cov.largest {
                (cov.largest, cov.largest_name, cov.largest_at) = (a, mask_name(t, n), p.t);
            }
            *at.entry((p.t * 1000.0).round() as i64).or_insert(0.0) += a;
        }
    }
    if let Some((k, a)) = at.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
        (cov.most, cov.most_at) = (a.min(1.0), *k as f64 / 1000.0);
    }
    cov
}

/// Where media time `media` of clip `it` plays in its sequence (inside the clip; a frame hold:
/// its start).
fn timeline_at(it: &filmcraft_project::TrackItem, media: Tick) -> Tick {
    let speed = it.speed.abs();
    if it.frame_hold.is_some() || !(speed.is_finite() && speed > 0.0) {
        return it.start;
    }
    let rel = if it.reverse {
        (it.source_in.0 as f64 + it.duration.0 as f64 * speed - 1.0 - media.0 as f64) / speed
    } else {
        (media.0 as f64 - it.source_in.0 as f64) / speed
    };
    let last = Tick(it.end().0.saturating_sub(1)).max(it.start);
    if !rel.is_finite() {
        return it.start;
    }
    Tick(it.start.0.saturating_add(rel.round() as i64)).clamp(it.start, last)
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
        "label": t.label,
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
    if t.pose_hits > 0 {
        // seen (also) by the profile pass: in profile or half hidden
        v["fromPose"] = json!(t.pose_hits);
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
        // hidden faces the profile pass found: at some moment, and only it
        "facesFromPose": c.hidden().filter(|t| t.kind == Kind::Face && t.pose_hits > 0).count(),
        "facesOnlyFromPose": c.hidden().filter(|t| t.kind == Kind::Face && t.pose_hits > 0 && t.pose_hits == t.hits).count(),
        "plates": count(Kind::Plate),
        "targets": count(Kind::Target),
        "looks": {"asked": c.looks, "found": c.found, "errors": c.look_errors},
        "masks": masks,
        "skipped": {"tooSmall": skipped(Skip::TooSmall), "tooBrief": skipped(Skip::TooBrief), "excluded": skipped(Skip::Excluded), "notAPerson": skipped(Skip::NotAPerson), "notAPlate": skipped(Skip::NotAPlate)},
        "tracks": c.tracks.iter().map(|(t, s)| track_json(t, *s, dry)).collect::<Vec<_>>(),
        // the subject's boxes, for the check of the finished video ([`crate::privacy_check`])
        "robots": c.robots.iter().map(|e| json!({"x": e.bbox[0], "y": e.bbox[1], "w": e.bbox[2], "h": e.bbox[3], "seconds": e.t})).collect::<Vec<_>>(),
    })
}

/// A clip's [`Coverage`] for its outcome: shares to a thousandth, and when, in media and (as the
/// clip `it` plays) sequence seconds.
fn coverage_json(c: &ClipRedact, padding: f64, it: Option<&filmcraft_project::TrackItem>) -> Value {
    let cov = coverage(c, padding);
    let r = |v: f64| (v * 1000.0).round() / 1000.0;
    let when = |m: f64| json!({"media": r(m), "sequence": it.map(|it| r(timeline_at(it, Tick::from_seconds_f64(m)).seconds()))});
    json!({
        "largestMask": r(cov.largest),
        "largestMaskName": (!cov.largest_name.is_empty()).then_some(cov.largest_name),
        "largestAt": when(cov.largest_at),
        "mostHidden": r(cov.most),
        "mostHiddenAt": when(cov.most_at),
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
            params: r#"{"clips":[id]?,"clip":id?,"kinds":["faces","plates"]|"faces,plates"=both,"minFaceHeight":0..1=0.05,"minPlateWidth":0..1=0.025,"faceScore":0.05..1=0.6,"plateScore":0.05..1=0.8,"sureFace":0.05..1=0.85,"style":"blur"|"mosaic"="blur","strength":1..100=60,"padding":0..1=0.15,"exclude":{"x":0..1,"y":0..1,"w":0..1,"h":0..1,"clip":id?,"seconds":f64?}|[box]?,"verify":bool=true,"dryRun":bool=false,"analysisFps":1..60=10,"targets":[str|{"text":str,"clips":[id]}]?,"targetEvery":0.25..10=1,"replace":bool=true,"fresh":bool=false,"profiles":bool=installed,"wait":bool=false}"#,
            enabled: has_video_selection,
            run: redact_clips,
            journal: true,
        },
        CommandSpec {
            id: SEQ_CMD,
            label: "Auto Redact Sequence…",
            menu: &[],
            shortcut: None,
            params: r#"{"sequence":id?,"kinds":["faces","plates"]|"faces,plates"=both,"minFaceHeight":0..1=0.05,"minPlateWidth":0..1=0.025,"faceScore":0.05..1=0.6,"plateScore":0.05..1=0.8,"sureFace":0.05..1=0.85,"style":"blur"|"mosaic"="blur","strength":1..100=60,"padding":0..1=0.15,"exclude":{"x":0..1,"y":0..1,"w":0..1,"h":0..1,"clip":id?,"seconds":f64?}|[box]?,"verify":bool=true,"dryRun":bool=false,"analysisFps":1..60=10,"targets":[str|{"text":str,"clips":[id]}]?,"targetEvery":0.25..10=1,"replace":bool=true,"fresh":bool=false,"profiles":bool=installed,"wait":bool=false}"#,
            enabled: has_seq,
            run: redact_sequence,
            journal: true,
        },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download Face and Plate Detectors",
            menu: &[],
            shortcut: None,
            params: r#"{"profiles":bool=true,"wait":bool?}"#,
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

/// What a job needs before it starts: the detectors for faces and plates, the vision model for
/// named targets.
fn preflight(s: &Session, cmd: &str, o: &Options) -> Result<()> {
    if (o.faces || o.plates) && s.redacts.detector.is_none() {
        ready(cmd)?;
    }
    if !o.targets.is_empty() && s.redacts.judge.is_none() {
        return Err(bad(cmd, "finding named targets needs the local vision model (Ollama with qwen2.5vl:7b), which this app does not have"));
    }
    Ok(())
}

/// The models must be there before a job starts.
pub(crate) fn ready(cmd: &str) -> Result<()> {
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
    preflight(s, cmd, &o)?;
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
    preflight(s, cmd, &o)?;
    let (work, skipped) = build_work(s, seq, None, &o, &ex)?;
    if work.is_empty() {
        return Err(bad(cmd, "the sequence has no video footage to redact"));
    }
    start(s, cmd, seq, work, skipped, o, p)
}

fn start(s: &mut Session, cmd: &'static str, seq: ItemId, work: Vec<Work>, skipped: Vec<Value>, mut o: Options, p: &Value) -> Result<Value> {
    // the profile pass, decided now (the cache's keys and the outcome say which ran); a host's own
    // detector is the whole detector. Asked for, its models must load; not said, a model that
    // does not load leaves the face detector alone.
    let asked = o.profiles == Some(true);
    o.profiles = Some(s.redacts.detector.is_none() && profiles_on(&o));
    let total: usize = work.iter().map(|w| w.times.len()).sum();
    let clips = work.len();
    let n = clips;
    let job = crate::narration::new_job(s, format!("{LABEL} ({n} clip{})", if n == 1 { "" } else { "s" }));
    job.progress.total.store(total as u64, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let judge = if o.verify { s.redacts.judge.clone() } else { None };
    let vision = if o.targets.is_empty() { None } else { s.redacts.judge.clone() };
    let custom = s.redacts.detector.clone();
    // the cache: a host's own detector is cached only under its id
    let cache = s.redacts.cache.clone();
    let detector_id = match &custom {
        Some(_) => s.redacts.detector_id.clone().map(|id| format!("custom:{id}")),
        None => Some(format!("models:{}", crate::redact_cache::detector_pins())),
    };
    let services = s.services.clone();
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
        // the person and pose networks are heavy GPU work: one user of the GPU at a time (waited
        // for a while, then run anyway: the lock is advisory)
        let _gpu = (opts.profiles == Some(true)).then(|| gpu_lock(&prog)).flatten();
        *lock(&prog.status) = "Loading the face and plate detectors".into();
        let mut opts = opts;
        let det = match custom.map_or_else(|| load_detectors(&opts), Ok) {
            Ok(d) => d,
            Err(e) if opts.profiles == Some(true) && !asked => {
                log::warn!("auto redact: the profile pass's models did not load ({e}); the face detector looks alone");
                opts.profiles = Some(false);
                match load_detectors(&opts) {
                    Ok(d) => d,
                    Err(e) => return finish(Err(e.clone()), e, 0),
                }
            }
            Err(e) => return finish(Err(e.clone()), e, 0),
        };
        let sh = Shared { prog: &prog, done: AtomicU64::new(0), clips_done: AtomicUsize::new(0), clips: work.len(), stop: AtomicBool::new(false) };
        // each clip's file as the cache knows it (its content identity, read once per file)
        let cached = cache.as_deref().map(|c| {
            *lock(&prog.status) = "Reading the redaction cache".into();
            let mut ids: HashMap<String, Option<String>> = HashMap::new();
            let media = work
                .iter()
                .map(|w| {
                    let file = w.file.as_ref()?;
                    let id = ids
                        .entry(file.clone())
                        .or_insert_with(|| crate::relink::identity_of(&*services, file).ok().map(|i| format!("file:{}:{:016x}", i.size, i.fingerprint)))
                        .clone()?;
                    Some(format!("{id}|{}", w.desc))
                })
                .collect();
            Cached {
                cache: c,
                fresh: opts.fresh,
                media,
                detector: detector_id.clone(),
                frames: AtomicUsize::new(0),
                decoded: AtomicUsize::new(0),
                verdicts: AtomicUsize::new(0),
                looks: AtomicUsize::new(0),
            }
        });
        let Some(mut made) = analyse_all(&work, &det, &opts, vision.as_deref(), &sh, cached.as_ref()) else {
            return finish(Err("stopped".into()), "Stopped: nothing was changed".into(), sh.done.load(Ordering::Relaxed));
        };
        let t_analysis = t0.elapsed().as_secs_f64();
        let verified = match (&judge, verify_wanted) {
            (_, false) => Verified::Off,
            (None, true) => Verified::Skipped("no local vision model to tell people from robots and plates from signs: everything found is hidden".into()),
            (Some(j), true) => verify(&mut made, &work, j.as_ref(), &opts, &sh, cached.as_ref()),
        };
        if sh.cancelled() {
            return finish(Err("stopped".into()), "Stopped: nothing was changed".into(), sh.done.load(Ordering::Relaxed));
        }
        let hidden: usize = made.iter().map(|c| c.hidden().count()).sum();
        let frames = sh.done.load(Ordering::Relaxed);
        let from_cache = cached.as_ref().map_or(0, |c| c.frames.load(Ordering::Relaxed));
        let status = format!(
            "Found {hidden} face(s) and plate(s) to hide in {} clip(s), {frames} frame(s) ({from_cache} from the cache) in {:.1}s",
            made.len(),
            t0.elapsed().as_secs_f64()
        );
        let t_verify = t0.elapsed().as_secs_f64() - t_analysis;
        let stats = cached.as_ref().map_or(json!({"enabled": false}), |c| {
            let mut v = c.stats();
            v["enabled"] = json!(true);
            v
        });
        finish(Ok((made, verified, [t_analysis, t_verify], stats, opts.profiles == Some(true))), status, frames);
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

/// How long a redaction waits for another user of the GPU before running anyway.
const GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// The machine's GPU lock for a redaction job ([`filmcraft_locks`]), waiting up to [`GPU_WAIT`]
/// with the job's status saying who has it; `None` when it could not be taken (the job runs
/// anyway).
pub(crate) fn gpu_lock(prog: &filmcraft_export::Progress) -> Option<filmcraft_locks::Guard> {
    *lock(&prog.status) = "Taking the GPU".into();
    filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, "auto redaction (people and poses)", GPU_WAIT, &mut |e| {
        *lock(&prog.status) = format!("Waiting for the GPU: {e}");
    })
    .ok()
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
            Some(Ok((clips, verified, secs, cache, profiles))) => match apply(s, &pj, &clips, &verified, secs, cache, profiles) {
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

fn apply(s: &mut Session, pj: &PendingRedact, clips: &[ClipRedact], verified: &Verified, secs: [f64; 2], cache: Value, profiles: bool) -> Result<Value> {
    let o = &pj.opts;
    let made: Vec<Vec<EffectInstance>> = clips.iter().map(|c| effects_for(c, o)).collect();
    let hidden: usize = made.iter().flatten().map(|e| e.masks.len()).sum();
    // every clip of an analysis, and its copies, with the effects hiding what it shows
    let effects: Vec<(ClipId, Vec<EffectInstance>)> =
        clips.iter().zip(&made).flat_map(|(c, e)| std::iter::once(c.clip).chain(c.also.iter().copied()).map(move |id| (id, e.clone()))).collect();
    let masks = |id: ClipId| effects.iter().find(|(c, _)| *c == id).map_or(0, |(_, e)| e.iter().map(|e| e.masks.len()).sum());
    let placed = s.project.sequence(pj.seq);
    let summary: Vec<Value> = clips
        .iter()
        .map(|c| {
            let mut v = clip_json(c, o.dry_run, if o.dry_run { 0 } else { masks(c.clip) });
            v["coverage"] = coverage_json(c, o.padding, placed.and_then(|q| q.find_item(c.clip)).map(|(_, it)| it));
            v
        })
        .collect();
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
                    if !e.is_empty() {
                        // ahead of the clip's other standard effects: the masks are in the
                        // source's own pixels, before anything moves them (Warp Stabilizer)
                        let at = it.effects.iter().position(|x| x.def().is_some_and(|d| !d.intrinsic)).unwrap_or(it.effects.len());
                        for (k, fx) in e.iter().enumerate() {
                            it.effects.insert(at + k, fx.clone());
                        }
                        n += 1;
                    }
                }
                q.check().map_err(EngineError::Other)?;
                Ok(n)
            })?;
        }
        let faces: usize = clips.iter().map(|c| c.hidden().filter(|t| t.kind == Kind::Face).count()).sum();
        let plates: usize = clips.iter().map(|c| c.hidden().filter(|t| t.kind == Kind::Plate).count()).sum();
        let targets: usize = clips.iter().map(|c| c.hidden().filter(|t| t.kind == Kind::Target).count()).sum();
        let note = match verified {
            Verified::Skipped(why) | Verified::Failed(why) => format!(" (vision check: {why})"),
            _ => String::new(),
        };
        let named = if pj.opts.targets.is_empty() { String::new() } else { format!(", {targets} named place(s)") };
        s.toast(format!("{LABEL}: {faces} face(s) and {plates} plate(s){named} hidden in {changed} clip(s){note}"));
    }
    Ok(json!({
        "sequence": pj.seq.0,
        "dryRun": o.dry_run,
        "applied": !o.dry_run,
        "clipsChanged": changed,
        "hidden": hidden,
        "profiles": profiles,
        "verify": verified.to_json(),
        "seconds": {"analysis": (secs[0] * 10.0).round() / 10.0, "check": (secs[1] * 10.0).round() / 10.0},
        "cache": cache,
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
        "installed": installed() || s.redacts.detector.is_some(),
        "missingBytes": dir.as_deref().map_or(filmcraft_vision::catalog::total_bytes(), filmcraft_vision::catalog::missing_bytes),
        "package": filmcraft_vision::catalog::PACKAGE_NAME,
        "license": filmcraft_vision::catalog::LICENSE,
        "source": filmcraft_vision::catalog::SOURCE,
        "files": filmcraft_vision::catalog::FILES,
        // the profile pass (faces in profile and half hidden): on by default once downloaded
        "profiles": {
            "installed": profiles_installed(),
            "missingBytes": dir.as_deref().map_or(filmcraft_vision::catalog::profile_total_bytes(), filmcraft_vision::catalog::profile_missing_bytes),
            "package": filmcraft_vision::catalog::PROFILE_NAME,
            "license": filmcraft_vision::catalog::PROFILE_LICENSE,
            "source": filmcraft_vision::catalog::PROFILE_SOURCE,
            "files": filmcraft_vision::catalog::PROFILE_FILES,
        },
        "judge": s.redacts.judge.as_ref().map(|j| j.name()),
    }))
}

/// The files `redact.downloadModels` fetches: the detectors, and (`profiles`) the profile pass's
/// models.
fn wanted_files(profiles: bool) -> Vec<filmcraft_vision::catalog::ModelFile> {
    let mut v = filmcraft_vision::catalog::FILES.to_vec();
    if profiles {
        v.extend_from_slice(filmcraft_vision::catalog::PROFILE_FILES);
    }
    v
}

/// Download the detectors (and the profile pass's models) in the background.
fn download_models(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = DOWNLOAD_CMD;
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let profiles = profiles_p(cmd, p)?.unwrap_or(true);
    let wait = match p.get("wait").filter(|v| !v.is_null()) {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`wait` is true or false")),
    };
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the detectors"))?;
    let files = wanted_files(profiles);
    let pkg = filmcraft_vision::catalog::package_dir(&dir);
    let missing: u64 = files.iter().filter(|f| !std::fs::metadata(pkg.join(f.name)).is_ok_and(|m| m.len() == f.size)).map(|f| f.size).sum();
    if missing == 0 {
        return Ok(json!({"job": null, "installed": true, "profiles": profiles}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "automatic redaction is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::PACKAGE_NAME));
    job.progress.total.store(missing, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-redact-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_package(&dir, &files, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Privacy detectors installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": missing, "profiles": profiles}))
}

#[cfg(feature = "privacy-models")]
fn download_package(
    dir: &std::path::Path,
    wanted: &[filmcraft_vision::catalog::ModelFile],
    prog: &filmcraft_export::Progress,
) -> std::result::Result<(), String> {
    let files: Vec<filmcraft_speech::models::ModelFile> =
        wanted.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::package_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_package(_: &std::path::Path, _: &[filmcraft_vision::catalog::ModelFile], _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("automatic redaction is not available in this build".into())
}

#[cfg(test)]
#[path = "redact_tests.rs"]
mod tests;
