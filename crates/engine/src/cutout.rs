//! People cut out of their clips, and the effects made of the cut-outs (ideas of Insta360's Shot
//! Lab): each is a command, built from ordinary clips, masks and titles, so it renders on the GPU,
//! saves with the project and stays editable.
//!
//! | command | does |
//! |---|---|
//! | `cutout.status` | whether the segmenter (MobileSAM) and the person detector (D-FINE, RTMPose) are downloaded: sizes, licences, sources |
//! | `cutout.downloadModel` | job: downloads MobileSAM (after the user agreed to it), its size and SHA-256 checked |
//! | `clip.cutOut` | job: the person in a clip (`person`: `largest`, `centre` or `[x, y]` of the picture) outlined every `every` seconds (0.2) — found by the person detector, cut out by MobileSAM from points on their body, traced as a `points`-point outline (96) — kept as the clip's Opacity mask `Person`, switched off (the clip still shows whole): the shape the effects below cut with |
//! | `clip.cloneTrail` | `copies` (4) frozen copies of the person left along their path where they were, each from when they were there to the clip's end |
//! | `clip.shadowClone` | `copies` (2) see-through copies following the person `delay` (0.35 s) apart, the person in front |
//! | `clip.timeFreeze` | the person frozen `at` (seconds into the clip; the middle) while the world goes on: the moving person hidden by the background seen where they are not (a clean plate: each pixel's median over 24 frames of the clip in which the person is elsewhere, written to `plateFolder`, else the data directory's `cutout-plates`, and imported) |
//! | `clip.textBehind` | a title (`text`, `size`, `font`) behind the person: the title above the clip, the person cut out above the title |
//! | `clip.personGlow` | a glow round the person: `color` (white), `width` (12 px), `strength` (80 %) |
//! | `clip.personInFront` | the person cut out, in step with the clip, above everything over it: the titles and graphics there go behind them |
//!
//! **How the cut-out is made.** Each analysed frame (at most 1024 pixels on its long side) goes to
//! the person detector; the person is the one `person` names on the first frame and after that the
//! one overlapping where they were (else the nearest); MobileSAM is given their body's keypoints
//! inside and points beside them outside; the largest piece of what it cuts out is traced and
//! spaced into the outline, turned to start where the last one did, and written as a Mask Path
//! keyframe (in media time; masks interpolate point by point between keyframes). A person lost for
//! 1.5 s ends the outline there.
//!
//! **The effects.** Copies of the clip (its picture only) go on tracks above it with the `Person`
//! mask switched on: a frozen copy holds its frame and its mask (Frame Hold with Hold Filters), a
//! following copy starts later with the same media (its mask follows by itself: masks are in media
//! time). A copy in step with the clip (the glow, the person in front of a title) lines up whatever
//! the camera does; for frozen and following copies the camera must stand still: a 360° clip's
//! virtual camera may not move (its stabilisation, which holds the view still in the world, is
//! kept); `clip.stabilize360` with `mode: "lock"` makes a walk still.

use std::sync::{Arc, Mutex};

use filmcraft_project::{ClipId, ItemId, Mask, MaskMode, MaskPath, MaskVertex, ParamValue, Sequence, Track, TrackId, TrackItem, TrackKind};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, clips_p, f64_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

/// The name of the person's mask on a clip's Opacity effect.
pub const PERSON: &str = "Person";
const CUT_CMD: &str = "clip.cutOut";
const DOWNLOAD_CMD: &str = "cutout.downloadModel";
/// Seconds between outlines by default, and the range allowed.
const EVERY: f64 = 0.2;
const EVERY_RANGE: (f64, f64) = (1.0 / 30.0, 2.0);
/// Points of an outline by default, and the range allowed.
const POINTS: usize = 96;
const POINTS_RANGE: (usize, usize) = (16, 256);
/// Most outlines one clip gets.
const MAX_KEYS: usize = 3_000;
/// Longest side of the pictures analysed.
const SIDE: f64 = 1024.0;
/// A person lost this long ends the outline.
const LOST_SECONDS: f64 = 1.5;
/// Most copies an effect makes.
const MAX_COPIES: u64 = 12;
/// Frames the clean plate of `clip.timeFreeze` is made from.
const PLATE_FRAMES: usize = 24;
/// A virtual camera's parameters that may not move under the effects.
const CAMERA_PARAMS: [&str; 5] = ["yaw", "pitch", "roll", "fov", "distance"];

/// Who to cut out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pick {
    Largest,
    Centre,
    /// The person at this point of the picture (0..1).
    At(f32, f32),
}

/// A person cut out of a picture: their box `[x, y, w, h]` and outline, in the picture's pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    pub bbox: [f32; 4],
    pub outline: Vec<(f32, f32)>,
}

/// Cuts people out of pictures: the downloaded person detector and segmenter, or a stand-in.
pub trait PersonCutter: Send + Sync {
    /// The person `pick` names — or, with `last` (their box in the picture before), the one
    /// there now — cut out of an RGBA picture, outlined in `points` points; `None` when no one is.
    fn cut(&self, rgba: &[u8], w: usize, h: usize, pick: Pick, last: Option<[f32; 4]>, points: usize) -> std::result::Result<Option<Cut>, String>;
}

/// The session's cut-outs: a stand-in cutter (tests), else the downloaded models are loaded.
#[derive(Default)]
pub struct Cutouts {
    pub cutter: Option<Arc<dyn PersonCutter>>,
}

pub fn commands() -> Vec<CommandSpec> {
    let spec = |id: &'static str, label: &'static str, menu: &'static [&'static str], params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| {
        CommandSpec { id, label, menu, shortcut: None, params, enabled: has_seq, run, journal: true }
    };
    vec![
        CommandSpec { id: "cutout.status", label: "Cut-out Models", menu: &[], shortcut: None, params: "{}", enabled: |_| Ok(()), run: status, journal: false },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download the Cut-out Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: |_| Ok(()),
            run: download,
            journal: false,
        },
        spec(
            CUT_CMD,
            "Cut Out the Person",
            &["Clip", "Cut-outs"],
            r#"{"clip":id?,"person":"largest"|"centre"|[x,y]="largest","every":seconds=0.2,"points":16..256=96,"wait":bool=false}"#,
            cut_out,
        ),
        spec("clip.cloneTrail", "Clone Trail", &["Clip", "Cut-outs"], r#"{"clip":id?,"copies":1..12=4}"#, clone_trail),
        spec(
            "clip.shadowClone",
            "Shadow Clone",
            &["Clip", "Cut-outs"],
            r#"{"clip":id?,"copies":1..12=2,"delay":seconds=0.35,"opacity":0..100=60}"#,
            shadow_clone,
        ),
        spec("clip.timeFreeze", "Time Freeze", &["Clip", "Cut-outs"], r#"{"clip":id?,"at":seconds?,"plateFolder":str?}"#, time_freeze),
        spec(
            "clip.textBehind",
            "Text Behind the Person",
            &["Clip", "Cut-outs"],
            r#"{"clip":id?,"text":str,"size":px?,"font":str?,"position":[x,y]?}"#,
            text_behind,
        ),
        spec("clip.personInFront", "Person in Front", &["Clip", "Cut-outs"], r#"{"clip":id?}"#, person_in_front),
        spec(
            "clip.personGlow",
            "Glow Round the Person",
            &["Clip", "Cut-outs"],
            r##"{"clip":id?,"color":"#rrggbb"="ffffff","width":px=12,"strength":0..100=80}"##,
            person_glow,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// the models
// ---------------------------------------------------------------------------------------------

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    let installed = dir.as_deref().is_some_and(filmcraft_vision::catalog::segmenter_installed);
    let people = dir.as_deref().is_some_and(filmcraft_vision::catalog::profiles_installed);
    Ok(json!({
        "available": cfg!(feature = "privacy-models"),
        "segmenter": {"name": filmcraft_vision::catalog::SEGMENT_NAME, "installed": installed, "bytes": filmcraft_vision::catalog::SEGMENTER.size, "license": filmcraft_vision::catalog::SEGMENT_LICENSE, "source": filmcraft_vision::catalog::SEGMENT_SOURCE},
        "personDetector": {"name": filmcraft_vision::catalog::PROFILE_NAME, "installed": people, "bytes": filmcraft_vision::catalog::profile_total_bytes(), "license": filmcraft_vision::catalog::PROFILE_LICENSE, "download": "redact.downloadModels"},
    }))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = crate::commands::bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::segmenter_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "cutting people out is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::SEGMENT_NAME));
    job.progress.total.store(filmcraft_vision::catalog::SEGMENTER.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-cutout-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "MobileSAM installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::SEGMENTER.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::SEGMENT_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::segment_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("cutting people out is not available in this build".into())
}

/// The downloaded person detector and segmenter.
#[cfg(feature = "privacy-models")]
struct Models {
    people: filmcraft_vision::ProfileDetector,
    seg: filmcraft_vision::segment::Segmenter,
}

#[cfg(feature = "privacy-models")]
impl PersonCutter for Models {
    fn cut(&self, rgba: &[u8], w: usize, h: usize, pick: Pick, last: Option<[f32; 4]>, points: usize) -> std::result::Result<Option<Cut>, String> {
        let people = self.people.people(rgba, w, h).map_err(|e| e.to_string())?;
        let boxes: Vec<[f32; 4]> = people.iter().map(|p| p.bbox).collect();
        let Some(i) = choose(&boxes, w, h, pick, last) else { return Ok(None) };
        let Some(person) = people.get(i) else { return Ok(None) };
        let (inside, outside) = filmcraft_vision::matte::prompts(person, w, h);
        let e = self.seg.embed(rgba, w, h).map_err(|e| e.to_string())?;
        let m = self.seg.cut(&e, &inside, &outside).map_err(|e| e.to_string())?;
        let k = e.scale().max(1e-6);
        Ok(filmcraft_vision::matte::outline(&m, points).map(|o| Cut { bbox: person.bbox, outline: o.into_iter().map(|(x, y)| (x / k, y / k)).collect() }))
    }
}

/// The cutter a job uses: the session's stand-in, else the downloaded models.
fn cutter(s: &Session, cmd: &str) -> Result<Arc<dyn PersonCutter>> {
    if let Some(c) = &s.cutouts.cutter {
        return Ok(c.clone());
    }
    load_models(cmd)
}

#[cfg(feature = "privacy-models")]
fn load_models(cmd: &str) -> Result<Arc<dyn PersonCutter>> {
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    let not_yet = |what: &str, how: &str| bad(cmd, format!("{what} is not downloaded yet ({how})"));
    let people = filmcraft_vision::ProfileDetector::load_on(&dir, true).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => not_yet("the person detector", "redact.downloadModels"),
        e => bad(cmd, e.to_string()),
    })?;
    let seg = filmcraft_vision::segment::Segmenter::load(&dir, true).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => not_yet("the segmenter (MobileSAM)", DOWNLOAD_CMD),
        e => bad(cmd, e.to_string()),
    })?;
    Ok(Arc::new(Models { people, seg }))
}

#[cfg(not(feature = "privacy-models"))]
fn load_models(cmd: &str) -> Result<Arc<dyn PersonCutter>> {
    Err(bad(cmd, "cutting people out is not available in this build (built without the `privacy-models` feature)"))
}

/// Which of `boxes` (`[x, y, w, h]`, picture pixels of a `w` × `h` picture) is the person: the one
/// overlapping `last` most (else the nearest to it, within a third of the picture), or, without
/// `last`, the one `pick` names.
pub fn choose(boxes: &[[f32; 4]], w: usize, h: usize, pick: Pick, last: Option<[f32; 4]>) -> Option<usize> {
    let ok = |b: &[f32; 4]| b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0;
    let centre = |b: &[f32; 4]| (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
    let (fw, fh) = (w.max(1) as f32, h.max(1) as f32);
    let idx = boxes.iter().enumerate().filter(|(_, b)| ok(b));
    if let Some(l) = last.filter(ok) {
        let iou = |b: &[f32; 4]| {
            let (x0, y0) = (b[0].max(l[0]), b[1].max(l[1]));
            let (x1, y1) = ((b[0] + b[2]).min(l[0] + l[2]), (b[1] + b[3]).min(l[1] + l[3]));
            let i = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
            i / (b[2] * b[3] + l[2] * l[3] - i).max(1e-6)
        };
        if let Some((i, v)) = idx.clone().map(|(i, b)| (i, iou(b))).max_by(|a, b| a.1.total_cmp(&b.1)).filter(|(_, v)| *v >= 0.1) {
            let _ = v;
            return Some(i);
        }
        let (lx, ly) = centre(&l);
        let near = idx.map(|(i, b)| (i, ((centre(b).0 - lx) / fw).hypot((centre(b).1 - ly) / fh))).min_by(|a, b| a.1.total_cmp(&b.1));
        return near.filter(|(_, d)| *d < 1.0 / 3.0).map(|(i, _)| i);
    }
    match pick {
        Pick::Largest => idx.max_by(|a, b| (a.1[2] * a.1[3]).total_cmp(&(b.1[2] * b.1[3]))).map(|(i, _)| i),
        Pick::Centre => idx
            .min_by(|a, b| ((centre(a.1).0 / fw - 0.5).hypot(centre(a.1).1 / fh - 0.5)).total_cmp(&(centre(b.1).0 / fw - 0.5).hypot(centre(b.1).1 / fh - 0.5)))
            .map(|(i, _)| i),
        Pick::At(x, y) => {
            let (px, py) = (x * fw, y * fh);
            let inside = |b: &[f32; 4]| px >= b[0] && px <= b[0] + b[2] && py >= b[1] && py <= b[1] + b[3];
            // the smallest box holding the point (the person in front)
            idx.filter(|(_, b)| inside(b)).min_by(|a, b| (a.1[2] * a.1[3]).total_cmp(&(b.1[2] * b.1[3]))).map(|(i, _)| i)
        }
    }
}

/// `next` turned (its points shifted round) to start where `prev` does: the outline the nearest to
/// it point by point, so a mask between them does not twist.
pub fn aligned(prev: &[(f32, f32)], next: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    let n = next.len();
    if n == 0 || prev.len() != n {
        return next;
    }
    let cost = |k: usize| -> f32 { (0..n).map(|i| (next[(i + k) % n].0 - prev[i].0).powi(2) + (next[(i + k) % n].1 - prev[i].1).powi(2)).sum() };
    let best = (0..n).min_by(|a, b| cost(*a).total_cmp(&cost(*b))).unwrap_or(0);
    (0..n).map(|i| next[(i + best) % n]).collect()
}

// ---------------------------------------------------------------------------------------------
// clip.cutOut
// ---------------------------------------------------------------------------------------------

/// The video clip a command acts on: `clip`, else the first selected video clip.
pub(crate) fn video_clip(s: &Session, p: &Value, cmd: &str) -> Result<(usize, TrackItem)> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let ids = clips_p(s, p);
    for c in &ids {
        if let Some((ti, it)) = seq.video_tracks.iter().enumerate().find_map(|(i, t)| t.items.iter().find(|x| x.id == *c).map(|x| (i, x.clone()))) {
            return Ok((ti, it));
        }
    }
    Err(bad(cmd, if ids.is_empty() { "no clip given or selected".to_string() } else { "no video clip given or selected".to_string() }))
}

fn pick_p(p: &Value, cmd: &str) -> Result<Pick> {
    match p.get("person") {
        None | Some(Value::Null) => Ok(Pick::Largest),
        Some(Value::String(s)) if s == "largest" => Ok(Pick::Largest),
        Some(Value::String(s)) if s == "centre" || s == "center" => Ok(Pick::Centre),
        Some(Value::Array(a)) if a.len() == 2 => {
            let v: Vec<f64> = a.iter().filter_map(Value::as_f64).filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).collect();
            match v[..] {
                [x, y] => Ok(Pick::At(x as f32, y as f32)),
                _ => Err(bad(cmd, "`person` as a point is [x, y], each 0..1 of the picture")),
            }
        }
        Some(_) => Err(bad(cmd, "`person` is \"largest\", \"centre\" or [x, y] (0..1 of the picture)")),
    }
}

/// A closed polygon mask path through `pts` (clip pixels).
fn polygon(pts: &[(f32, f32)]) -> MaskPath {
    MaskPath { vertices: pts.iter().map(|(x, y)| MaskVertex::corner(filmcraft_geom::Vec2::new(f64::from(*x), f64::from(*y)))).collect(), closed: true }
}

/// Finds an outline in a picture (RGBA, `w` × `h`): `points` points in its pixels, `None` when the
/// thing is not there.
pub(crate) type FindOutline = Box<dyn FnMut(&[u8], usize, usize, usize) -> std::result::Result<Option<Vec<(f32, f32)>>, String> + Send>;

/// What an outline job is: its mask's name, its undo step, its job's label, what it says when it
/// finds nothing, and whether the thing lost for [`LOST_SECONDS`] ends it.
pub(crate) struct Outlining {
    pub mask: &'static str,
    pub undo: &'static str,
    pub what: &'static str,
    pub not_found: &'static str,
    pub lost_ends: bool,
}

/// `every` and `points` of an outline job's parameters.
pub(crate) fn outline_params(p: &Value, cmd: &str, points_default: usize) -> Result<(f64, usize)> {
    let every = match f64_p(p, "every") {
        None => EVERY,
        Some(e) if e.is_finite() && (EVERY_RANGE.0..=EVERY_RANGE.1).contains(&e) => e,
        Some(_) => return Err(bad(cmd, "`every` is 1/30 to 2 seconds")),
    };
    let points = match u64_p(p, "points") {
        None => points_default,
        Some(n) if (POINTS_RANGE.0 as u64..=POINTS_RANGE.1 as u64).contains(&n) => n as usize,
        Some(_) => return Err(bad(cmd, "`points` is 16 to 256")),
    };
    Ok((every, points))
}

/// A job outlining something in clip `it` every `every` seconds with `find` (`points` points),
/// written as its Opacity mask `o.mask` (switched off) as the outlines come: one undo step when
/// waited for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn outline_job(
    s: &mut Session,
    cmd: &'static str,
    it: &TrackItem,
    o: Outlining,
    every: f64,
    points: usize,
    mut find: FindOutline,
    wait: bool,
) -> Result<Value> {
    if s.mask_jobs.iter().any(|j| j.target.clip == it.id) {
        return Err(bad(cmd, "this clip's masks are being tracked already"));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let size = filmcraft_render::clip_size(&s.project, it).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    // (a reframed 360° clip's picture is its virtual camera's)
    let src = crate::vr360::clip_source(s, it).ok_or_else(|| bad(cmd, "the clip's media is offline"))?;
    // the media times looked at: every `every` seconds of the clip
    let step = Tick::from_seconds_f64(every).max(Tick(1));
    let mut times: Vec<Tick> = Vec::new();
    let mut t = it.start;
    while t < it.end() && times.len() < MAX_KEYS {
        times.push(it.source_time_at(t));
        t += step;
    }
    times.dedup();
    if times.is_empty() {
        return Err(bad(cmd, "the clip is too short"));
    }
    // the mask, switched off, its outlines written as they come
    let clip = it.id;
    let n0 = s.history.undo.len();
    let name = o.mask;
    let mask_index = s.edit_sequence(o.undo, |q, _, _| {
        let (_, item) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let op = item.effect_mut("opacity").ok_or_else(|| bad(cmd, "the clip has no Opacity"))?;
        op.masks.retain(|m| m.name != name);
        let mut m = Mask::new(
            name,
            MaskPath::ellipse(filmcraft_geom::Vec2::new(f64::from(size.0) / 2.0, f64::from(size.1) / 2.0), filmcraft_geom::Vec2::new(8.0, 8.0)),
        );
        m.mode = MaskMode::None;
        m.feather = filmcraft_project::Param::new(ParamValue::Float(2.0));
        op.masks.push(m);
        Ok(op.masks.len() - 1)
    })?;
    let effect = s.active_sequence().and_then(|q| q.find_item(clip)).and_then(|(_, i)| i.effects.iter().position(|e| e.effect == "opacity")).unwrap_or(0);
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let job = crate::Job { id, label: format!("Outline {} ({} frames)", o.what, times.len()), progress: Default::default(), result: Default::default() };
    job.progress.total.store(times.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let keys: Arc<Mutex<Vec<(Tick, MaskPath)>>> = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), keys.clone());
    let scale = (SIDE / f64::from(size.0.max(size.1).max(1))).min(1.0) as f32;
    let lost_limit = (LOST_SECONDS / every).ceil() as usize;
    let (what, not_found, lost_ends) = (o.what, o.not_found, o.lost_ends);
    let run = move || {
        use std::sync::atomic::Ordering;
        let t0 = web_time::Instant::now();
        let mut prev: Vec<(f32, f32)> = Vec::new();
        let (mut done, mut found, mut lost) = (0u64, 0usize, 0usize);
        let mut err: Option<String> = None;
        for t in &times {
            if prog.cancel.load(Ordering::Relaxed) {
                err = Some("stopped".into());
                break;
            }
            done += 1;
            prog.done.store(done, Ordering::Relaxed);
            let Ok(f) = src.video_frame(filmcraft_media::FrameRequest { time: *t, scale }) else { continue };
            let (rgba, w, h) = crate::scenes::small_rgba(&f, SIDE as u32);
            let (w, h) = (w as usize, h as usize);
            match find(&rgba, w, h, points) {
                Ok(Some(outline)) if outline.len() >= 3 => {
                    // picture pixels → clip pixels
                    let (kx, ky) = (size.0 as f32 / w.max(1) as f32, size.1 as f32 / h.max(1) as f32);
                    let pts: Vec<(f32, f32)> = outline.iter().map(|(x, y)| (x * kx, y * ky)).collect();
                    let pts = aligned(&prev, pts);
                    lock(&out).push((*t, polygon(&pts)));
                    prev = pts;
                    found += 1;
                    lost = 0;
                }
                Ok(_) => {
                    lost += 1;
                    if lost_ends && found > 0 && lost > lost_limit {
                        break;
                    }
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
            *lock(&prog.status) = format!("Frame {done} of {} · {what} in {found}", times.len());
        }
        let secs = t0.elapsed().as_secs_f64();
        let r = match &err {
            Some(e) if e != "stopped" => Err(e.clone()),
            _ if found == 0 => Err(not_found.to_string()),
            _ => Ok(filmcraft_export::Report {
                path: String::new(),
                frames: found as u64,
                seconds: secs,
                bytes: 0,
                render_fps: done as f64 / secs.max(1e-6),
                extra_files: Vec::new(),
            }),
        };
        *lock(&prog.status) = match &r {
            Ok(_) => format!("Outlined {what} in {found} of {done} frames in {secs:.1}s"),
            Err(e) => e.clone(),
        };
        prog.finished.store(true, Ordering::Relaxed);
        *lock(&res) = Some(r);
    };
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    s.mask_jobs.push(crate::masks::pending_track(id, crate::masks::MaskSel { clip, effect, mask: mask_index }, seq_id, keys));
    if wait || cfg!(target_arch = "wasm32") {
        run();
        crate::masks::poll(s);
        crate::clip_ops::collapse_history(s, n0, o.undo);
    } else {
        std::thread::Builder::new().name("filmcraft-outline".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    let st = s.jobs.iter().find(|j| j.id == id).map(crate::Job::to_json).unwrap_or(Value::Null);
    let outlines = s
        .active_sequence()
        .and_then(|q| q.find_item(clip))
        .and_then(|(_, i)| i.effect("opacity"))
        .and_then(|e| e.masks.iter().find(|m| m.name == name))
        .map_or(0, |m| m.path.keyframes.len());
    Ok(json!({"job": id, "clip": clip.0, "mask": mask_index, "outlines": outlines, "status": st}))
}

fn cut_out(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = CUT_CMD;
    let (_, it) = video_clip(s, p, cmd)?;
    let pick = pick_p(p, cmd)?;
    let (every, points) = outline_params(p, cmd, POINTS)?;
    let cutter = cutter(s, cmd)?;
    // the person picked on the first frame, then whoever is where they were
    let mut last: Option<[f32; 4]> = None;
    let find: FindOutline = Box::new(move |rgba, w, h, n| {
        Ok(cutter.cut(rgba, w, h, pick, last, n)?.map(|c| {
            last = Some(c.bbox);
            c.outline
        }))
    });
    let o = Outlining { mask: PERSON, undo: "Cut Out the Person", what: "the person", not_found: "no one to cut out was found in the clip", lost_ends: true };
    outline_job(s, cmd, &it, o, every, points, find, crate::commands::bool_p(p, "wait").unwrap_or(false))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A clip's person mask, when it has been cut out.
pub fn person_mask(it: &TrackItem) -> Option<&Mask> {
    it.effect("opacity")?.masks.iter().find(|m| m.name == PERSON && !m.path.keyframes.is_empty())
}

// ---------------------------------------------------------------------------------------------
// the effects
// ---------------------------------------------------------------------------------------------

/// A clip the effects build on: its track (index among the video tracks), the clip, its person.
struct Base {
    track: usize,
    it: TrackItem,
    mask: Mask,
}

/// `still`: the effect's copies show other moments than the clip (frozen, following), so a 360°
/// clip's virtual camera may not move.
fn base(s: &Session, p: &Value, cmd: &str, still: bool) -> Result<Base> {
    let (track, it) = video_clip(s, p, cmd)?;
    let mask = person_mask(&it).cloned().ok_or_else(|| bad(cmd, format!("cut the person out of clip {} first ({CUT_CMD})", it.id.0)))?;
    if still
        && let Some(cam) = it.effect(filmcraft_project::vr::REFRAME)
        && CAMERA_PARAMS.iter().any(|id| cam.param(id).is_some_and(|p| p.keyframes.len() > 1))
    {
        return Err(bad(
            cmd,
            "the clip's virtual camera moves: its copies would not line up (remove the camera's moves, or hold the view still with clip.stabilize360 `mode: \"lock\"`)",
        ));
    }
    Ok(Base { track, it, mask })
}

/// A copy of the clip `b` — its picture only — from `start` to `end` (timeline), the person's
/// mask switched on (`expand`, `feather` pixels), at `opacity` % with `blend` (Opacity's blend
/// mode index): frozen at media time `hold` when given (its mask held with it).
#[allow(clippy::too_many_arguments)]
fn copy(id: ClipId, b: &Base, start: Tick, end: Tick, hold: Option<Tick>, expand: f64, feather: f64, opacity: f64, blend: u32, role: &str) -> TrackItem {
    let mut c = b.it.clone();
    c.id = id;
    c.name = format!("{} ({role})", b.it.name);
    c.link = None;
    c.group = None;
    c.markers.clear();
    // a later start shows the same media later (the copy follows); a hold shows one moment
    c.start = start;
    c.duration = (end - start).max(Tick(1));
    if let Some(h) = hold {
        c.frame_hold = Some(h);
        c.hold_filters = true;
    }
    let mut m = b.mask.clone();
    m.mode = MaskMode::Add;
    m.expansion = filmcraft_project::Param::new(ParamValue::Float(expand));
    m.feather = filmcraft_project::Param::new(ParamValue::Float(feather.max(0.0)));
    if let Some(op) = c.effect_mut("opacity") {
        op.masks = vec![m];
        if let Some(o) = op.param_mut("opacity") {
            o.keyframes.clear();
            o.value = ParamValue::Float(opacity.clamp(0.0, 100.0));
        }
        if let Some(bl) = op.param_mut("blend") {
            bl.keyframes.clear();
            bl.value = ParamValue::Choice(blend);
        }
    }
    c
}

/// Puts `clips` on the video tracks above `from`, each above the one before it (a track added
/// when none is free there); the tracks they went on.
pub(crate) fn place_above(q: &mut Sequence, from: usize, clips: Vec<TrackItem>, alloc: &mut dyn FnMut() -> u64) -> Vec<usize> {
    let mut below = from;
    let mut used = Vec::new();
    for c in clips {
        let range = c.range();
        let free = |t: &Track| !t.locked && !t.items.iter().any(|i| i.range().overlaps(&range));
        let at = match (below + 1..q.video_tracks.len()).find(|&i| q.video_tracks.get(i).is_some_and(free)) {
            Some(i) => i,
            None => {
                let n = q.video_tracks.len() + 1;
                q.video_tracks.push(Track::new(TrackId(alloc()), TrackKind::Video, format!("Video {n}")));
                q.video_tracks.len() - 1
            }
        };
        if let Some(t) = q.video_tracks.get_mut(at) {
            t.items.push(c);
            t.sort();
        }
        used.push(at);
        below = at;
    }
    used
}

/// Adds `make`'s copies of the base clip above it, one undo step `label`; the copies' ids.
fn add_copies(s: &mut Session, label: &str, b: &Base, make: impl FnOnce(&mut dyn FnMut() -> ClipId) -> Vec<TrackItem>) -> Result<Vec<u64>> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let from = b.track;
    s.edit(label, |pr, st| {
        let mut ids = Vec::new();
        let clips = {
            let mut next = || {
                let id = ClipId(pr.alloc_id());
                ids.push(id);
                id
            };
            make(&mut next)
        };
        let mut track_ids: Vec<u64> = (0..clips.len()).map(|_| pr.alloc_id()).collect();
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        place_above(q, from, clips, &mut || track_ids.pop().unwrap_or(0));
        q.check().map_err(EngineError::Other)?;
        st.selection = ids.clone();
        Ok(ids.iter().map(|c| c.0).collect())
    })
}

fn copies_p(p: &Value, default: u64, cmd: &str) -> Result<u64> {
    match u64_p(p, "copies") {
        None => Ok(default),
        Some(n) if (1..=MAX_COPIES).contains(&n) => Ok(n),
        Some(_) => Err(bad(cmd, format!("`copies` is 1 to {MAX_COPIES}"))),
    }
}

/// The timeline time at which clip `it` shows media time `m` (its first such time).
fn timeline_at(it: &TrackItem, m: Tick, rate: filmcraft_time::FrameRate) -> Tick {
    let fd = rate.frame_duration().max(Tick(1));
    let n = (it.duration.0 / fd.0).max(1);
    let mut best = (i64::MAX, it.start);
    for k in 0..n.min(100_000) {
        let t = it.start + Tick(fd.0.saturating_mul(k));
        let d = (it.source_time_at(t) - m).0.abs();
        if d < best.0 {
            best = (d, t);
        }
    }
    best.1
}

/// A mask's box at media time `t` (clip pixels): `[x0, y0, x1, y1]`.
fn mask_box(m: &Mask, t: Tick) -> Option<[f64; 4]> {
    let path = m.path_at(t);
    let (mut b, mut any) = ([f64::MAX, f64::MAX, f64::MIN, f64::MIN], false);
    for v in &path.vertices {
        b = [b[0].min(v.p.x), b[1].min(v.p.y), b[2].max(v.p.x), b[3].max(v.p.y)];
        any = true;
    }
    any.then_some(b)
}

fn clone_trail(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.cloneTrail";
    let b = base(s, p, cmd, true)?;
    let n = copies_p(p, 4, cmd)? as usize;
    let rate = s.sequence_rate().sane();
    // the moments: where the person has moved on by their width since the copy before (the
    // outline's own keyframes, from the start; the last moment left to the clip itself)
    let keys: Vec<Tick> = b.mask.path.keyframes.iter().map(|k| k.time).collect();
    let mut moments: Vec<Tick> = Vec::new();
    let mut last: Option<[f64; 4]> = None;
    for t in &keys {
        let Some(bx) = mask_box(&b.mask, *t) else { continue };
        let apart = last.is_none_or(|l| {
            let w = (l[2] - l[0]).max(bx[2] - bx[0]).max(1.0);
            ((bx[0] + bx[2]) / 2.0 - (l[0] + l[2]) / 2.0).abs() >= w * 0.9
                || ((bx[1] + bx[3]) / 2.0 - (l[1] + l[3]) / 2.0).abs() >= (bx[3] - bx[1]).max(1.0) * 0.9
        });
        if apart {
            moments.push(*t);
            last = Some(bx);
        }
    }
    // too few apart: as many as asked, evenly through the outline
    if moments.len() < n && keys.len() >= n {
        moments = (0..n).filter_map(|k| keys.get(k * keys.len().saturating_sub(1) / n.max(1))).copied().collect();
    }
    moments.truncate(n);
    if moments.is_empty() {
        return Err(bad(cmd, "the person's outline has no moments to leave copies at"));
    }
    let end = b.it.end();
    let at: Vec<(Tick, Tick)> = moments.iter().map(|m| (timeline_at(&b.it, *m, rate), *m)).filter(|(t, _)| *t < end).collect();
    let ids = add_copies(s, "Clone Trail", &b, |next| at.iter().map(|(t, m)| copy(next(), &b, *t, end, Some(*m), 0.0, 1.5, 100.0, 0, "clone")).collect())?;
    Ok(json!({"clip": b.it.id.0, "copies": ids, "seconds": at.iter().map(|(t, _)| ((*t - b.it.start).seconds() * 100.0).round() / 100.0).collect::<Vec<_>>()}))
}

fn shadow_clone(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.shadowClone";
    let b = base(s, p, cmd, true)?;
    let n = copies_p(p, 2, cmd)?;
    let delay = match f64_p(p, "delay") {
        None => 0.35,
        Some(d) if d.is_finite() && (0.05..=5.0).contains(&d) => d,
        Some(_) => return Err(bad(cmd, "`delay` is 0.05 to 5 seconds")),
    };
    let opacity = match f64_p(p, "opacity") {
        None => 60.0,
        Some(o) if o.is_finite() && (0.0..=100.0).contains(&o) => o,
        Some(_) => return Err(bad(cmd, "`opacity` is 0 to 100")),
    };
    let (start, end) = (b.it.start, b.it.end());
    let d = Tick::from_seconds_f64(delay);
    let ids = add_copies(s, "Shadow Clone", &b, |next| {
        // the furthest behind lowest, the person in front of them all
        let mut out: Vec<TrackItem> = (1..=n)
            .rev()
            .filter_map(|k| {
                let t = start + Tick(d.0.saturating_mul(k as i64));
                (t < end).then(|| copy(next(), &b, t, end, None, 0.0, 2.0, opacity * (1.0 - 0.35 * (k - 1) as f64 / n as f64), 0, "shadow"))
            })
            .collect();
        out.push(copy(next(), &b, start, end, None, 0.0, 1.5, 100.0, 0, "person"));
        out
    })?;
    Ok(json!({"clip": b.it.id.0, "copies": ids}))
}

fn person_glow(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.personGlow";
    let b = base(s, p, cmd, false)?;
    let color = match str_p(p, "color") {
        None => [1.0, 1.0, 1.0, 1.0],
        Some(c) => hex_color(c).ok_or_else(|| bad(cmd, "`color` is #rrggbb"))?,
    };
    let width = f64_p(p, "width").filter(|w| w.is_finite() && (1.0..=200.0).contains(w)).unwrap_or(12.0);
    let strength = f64_p(p, "strength").filter(|w| w.is_finite() && (0.0..=100.0).contains(w)).unwrap_or(80.0);
    let (start, end) = (b.it.start, b.it.end());
    // Screen, in Opacity's blend modes
    let screen = filmcraft_project::effect::BLEND_MODES.iter().position(|m| *m == "Screen").unwrap_or(0) as u32;
    let ids = add_copies(s, "Glow Round the Person", &b, |next| {
        let mut glow = copy(next(), &b, start, end, None, width, width * 1.5, strength, screen, "glow");
        // the copy painted the glow's colour all over (Tint mapping black and white to it)
        if let Some(mut tint) = filmcraft_project::find_effect("tint").map(|d| d.instance()) {
            for id in ["black", "white"] {
                if let Some(p) = tint.param_mut(id) {
                    p.value = ParamValue::Color(color);
                }
            }
            glow.effects.push(tint);
        }
        vec![glow, copy(next(), &b, start, end, None, 0.0, 1.0, 100.0, 0, "person")]
    })?;
    Ok(json!({"clip": b.it.id.0, "copies": ids}))
}

/// Every keyframe of `it`'s effects and masks `by` earlier: for a copy whose own media time is
/// the original's less `by` (a still put where the clip's media was at `by`).
pub(crate) fn shift_keys(it: &mut TrackItem, by: Tick) {
    for e in &mut it.effects {
        for p in e.params.values_mut() {
            for k in &mut p.keyframes {
                k.time -= by;
            }
        }
        for m in &mut e.masks {
            for p in [&mut m.path, &mut m.feather, &mut m.opacity, &mut m.expansion] {
                for k in &mut p.keyframes {
                    k.time -= by;
                }
            }
        }
    }
}

/// `clip.personInFront`: the person cut out, in step with the clip, above everything over it.
fn person_in_front(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.personInFront";
    let b = base(s, p, cmd, false)?;
    let range = b.it.range();
    let top = s
        .active_sequence()
        .map(|q| q.video_tracks.iter().rposition(|t| t.items.iter().any(|i| i.range().overlaps(&range))).unwrap_or(b.track))
        .unwrap_or(b.track);
    let b = Base { track: top.max(b.track), ..b };
    let (start, end) = (b.it.start, b.it.end());
    let ids = add_copies(s, "Person in Front", &b, |next| vec![copy(next(), &b, start, end, None, 0.0, 1.5, 100.0, 0, "person")])?;
    Ok(json!({"clip": b.it.id.0, "copies": ids}))
}

fn hex_color(c: &str) -> Option<[f32; 4]> {
    let h = c.trim().trim_start_matches('#');
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let v = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok().map(|b| f32::from(b) / 255.0);
    Some([v(0)?, v(2)?, v(4)?, 1.0])
}

fn text_behind(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.textBehind";
    let b = base(s, p, cmd, false)?;
    let text = str_p(p, "text").map(str::trim).filter(|t| !t.is_empty()).ok_or_else(|| bad(cmd, "need `text`"))?.chars().take(200).collect::<String>();
    let (w, h) = s.active_sequence().map_or((1920, 1080), |q| (q.settings.width, q.settings.height));
    // (big, as wide as most of the frame at most: capitals are about 0.62 of the size wide)
    let fit = f64::from(w) * 0.85 / (0.62 * text.chars().count().max(1) as f64);
    let size = f64_p(p, "size").filter(|v| v.is_finite() && (8.0..=2000.0).contains(v)).unwrap_or((f64::from(h) * 0.22).min(fit));
    let n0 = s.history.undo.len();
    // the title on the first free track above the clip, as long as the clip
    let mut q = json!({"text": text, "size": size, "seconds": (b.it.duration.seconds()).max(0.1), "time": b.it.start.0, "track": b.track + 1, "newClip": true});
    if let Some(f) = str_p(p, "font") {
        q["font"] = json!(f);
    }
    q["position"] = p.get("position").cloned().unwrap_or_else(|| json!([f64::from(w) / 2.0, f64::from(h) * 0.45]));
    let title = match s.execute("graphics.newText", q.clone()) {
        Ok(v) => v,
        // V(n+1) taken: the first free one
        Err(_) => {
            q["track"] = json!("auto");
            s.execute("graphics.newText", q)?
        }
    };
    let title_clip = title["clip"].as_u64().map(ClipId);
    // centred on its position
    if let Some(tc) = title_clip {
        s.edit_sequence("Text Behind the Person", |q, _, _| {
            if let Some((_, t)) = q.find_item_mut(tc) {
                for e in t.effects.iter_mut().filter(|e| e.effect == filmcraft_project::graphic::TEXT_LAYER) {
                    if let Some(a) = e.param_mut("align") {
                        a.value = ParamValue::Choice(1);
                    }
                }
            }
            Ok(())
        })?;
    }
    let title_track = s.active_sequence().and_then(|q| q.video_tracks.iter().position(|t| t.items.iter().any(|i| Some(i.id) == title_clip))).unwrap_or(b.track);
    let b = Base { track: title_track, ..b };
    let (start, end) = (b.it.start, b.it.end());
    let ids = add_copies(s, "Text Behind the Person", &b, |next| vec![copy(next(), &b, start, end, None, 0.0, 1.5, 100.0, 0, "person")])?;
    crate::clip_ops::collapse_history(s, n0, "Text Behind the Person");
    Ok(json!({"clip": b.it.id.0, "title": title_clip.map(|c| c.0), "copies": ids}))
}

fn time_freeze(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.timeFreeze";
    let b = base(s, p, cmd, true)?;
    let rate = s.sequence_rate().sane();
    let at = match f64_p(p, "at") {
        None => b.it.start + Tick(b.it.duration.0 / 2),
        Some(x) if x.is_finite() && x >= 0.0 && x < b.it.duration.seconds() => rate.snap(b.it.start + Tick::from_seconds_f64(x)),
        Some(_) => return Err(bad(cmd, "`at` is seconds into the clip")),
    };
    let m = b.it.source_time_at(at);
    let end = b.it.end();
    // the clean plate: the background where the person is not, from frames across the clip
    let size = filmcraft_render::clip_size(&s.project, &b.it).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    let src = crate::vr360::clip_source(s, &b.it).ok_or_else(|| bad(cmd, "the clip's media is offline"))?;
    let times: Vec<Tick> =
        (0..PLATE_FRAMES).map(|k| b.it.source_time_at(b.it.start + Tick(b.it.duration.0.saturating_mul(k as i64) / PLATE_FRAMES as i64))).collect();
    if (b.it.speed - 1.0).abs() > 1e-6 || b.it.reverse || b.it.frame_hold.is_some() {
        return Err(bad(cmd, "the clip plays at another speed, backwards or held: its time can't be frozen"));
    }
    let plate = clean_plate(&src, &b.mask, size, &times).ok_or_else(|| bad(cmd, "the clip's frames do not decode"))?;
    if plate.behind < MIN_BEHIND || plate.unsteady > MAX_UNSTEADY {
        return Err(bad(
            cmd,
            format!(
                "the background behind the person is not seen clear and still (seen behind them: {:.0}%, background change {:.0}): a time freeze needs a still camera and a person who moves through the picture",
                plate.behind * 100.0,
                plate.unsteady.min(255.0)
            ),
        ));
    }
    let covered = plate.covered;
    // (at the clip's own size: the masks on it are in the clip's pixels)
    let (rgba, pw, ph) = resize_rgba(&plate.rgba, plate.w, plate.h, size.0 as usize, size.1 as usize);
    let dir = match str_p(p, "plateFolder") {
        Some(d) => std::path::PathBuf::from(crate::export_tools::expand_home(d)),
        None => crate::autosave::default_data_dir().map(|d| d.join("cutout-plates")).ok_or_else(|| bad(cmd, "no data directory for the clean plate"))?,
    };
    std::fs::create_dir_all(&dir).map_err(|e| bad(cmd, format!("{}: {e}", dir.display())))?;
    let path = dir.join(format!("plate-{}-{}.png", b.it.id.0, crate::autosave::unix_now()));
    let png = filmcraft_export::encode_png(rgba, pw as u32, ph as u32).map_err(|e| bad(cmd, e.to_string()))?;
    let path_s = path.to_string_lossy().into_owned();
    s.services.write_file(&path_s, &png).map_err(|e| bad(cmd, format!("{path_s}: {e}")))?;
    let n0 = s.history.undo.len();
    let r = s.execute("file.import", json!({"paths": [path_s]}))?;
    let plate_item = r["items"][0].as_u64().map(ItemId).ok_or_else(|| bad(cmd, "the clean plate did not import"))?;
    let ids = add_copies(s, "Time Freeze", &b, |next| {
        // the plate over the moving person (their mask, grown) from the freeze on, the frozen
        // person above it
        let mut plate_clip = copy(next(), &b, at, end, None, 12.0, 8.0, 100.0, 0, "clean plate");
        plate_clip.item = plate_item;
        plate_clip.name = "Clean plate".into();
        plate_clip.source_in = Tick::ZERO;
        plate_clip.frame_hold = None;
        plate_clip.effects.retain(|e| e.effect == "motion" || e.effect == "opacity");
        // the still's time starts at 0 where the clip's media is at `m`: its keyframes moved to
        // match
        shift_keys(&mut plate_clip, m);
        vec![plate_clip, copy(next(), &b, at, end, Some(m), 0.0, 1.5, 100.0, 0, "frozen")]
    })?;
    crate::clip_ops::collapse_history(s, n0, "Time Freeze");
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    Ok(
        json!({"clip": b.it.id.0, "copies": ids, "at": r3((at - b.it.start).seconds()), "plate": path_s, "plateCovered": r3(covered), "behindSeen": r3(plate.behind), "backgroundChange": (plate.unsteady * 10.0).round() / 10.0}),
    )
}

/// A clean plate, and how good it is.
struct Plate {
    rgba: Vec<u8>,
    w: usize,
    h: usize,
    /// The share of all pixels seen clear in some frame.
    covered: f64,
    /// The share of the pixels the person passes over that are seen clear in two frames or more:
    /// low when they stay where they are (the camera moving with them).
    behind: f64,
    /// How much the background changes from frame to frame where it is seen (the median absolute
    /// deviation of its luma, 0..255): high when the camera moves.
    unsteady: f64,
}

/// A time freeze needs the background behind the person seen clear this much…
const MIN_BEHIND: f64 = 0.85;
/// …and holding this still.
const MAX_UNSTEADY: f64 = 18.0;

/// The background of a clip where its person is not: for each pixel, the median of the frames at
/// `times` in which the person (`mask`, grown a little) does not cover it.
fn clean_plate(src: &filmcraft_media::SharedSource, mask: &Mask, size: (u32, u32), times: &[Tick]) -> Option<Plate> {
    let scale = (SIDE / f64::from(size.0.max(size.1).max(1))).min(1.0) as f32;
    let mut frames: Vec<(Vec<u8>, Vec<f32>)> = Vec::new();
    let (mut w, mut h) = (0usize, 0usize);
    for t in times {
        let Ok(f) = src.video_frame(filmcraft_media::FrameRequest { time: *t, scale }) else { continue };
        let (rgba, fw, fh) = crate::scenes::small_rgba(&f, SIDE as u32);
        let (fw, fh) = (fw as usize, fh as usize);
        if frames.is_empty() {
            (w, h) = (fw, fh);
        }
        if (fw, fh) != (w, h) {
            continue;
        }
        let mut m = mask.clone();
        m.mode = MaskMode::Add;
        m.expansion = filmcraft_project::Param::new(ParamValue::Float(8.0));
        let k = fw as f32 / size.0.max(1) as f32;
        let flat = filmcraft_render::mask::prepare(&[m], *t, k);
        let cover = filmcraft_render::mask::coverage(&flat, fw, fh).unwrap_or_else(|| vec![0.0; fw * fh]);
        frames.push((rgba, cover));
    }
    if frames.is_empty() || w == 0 || h == 0 {
        return None;
    }
    let mut out = vec![0u8; w * h * 4];
    let mut clear = 0usize;
    let (mut path, mut path_clear) = (0usize, 0usize);
    let (mut dev_sum, mut dev_n) = (0.0f64, 0usize);
    let mut vals: Vec<[u8; 3]> = Vec::with_capacity(frames.len());
    for i in 0..w * h {
        vals.clear();
        let mut passed = false;
        for (rgba, cover) in &frames {
            let c = cover.get(i).copied().unwrap_or(1.0);
            passed |= c >= 0.5;
            if c < 0.05
                && let Some(px) = rgba.get(i * 4..i * 4 + 3)
            {
                vals.push([px[0], px[1], px[2]]);
            }
        }
        if passed {
            path += 1;
            path_clear += usize::from(vals.len() >= 2);
        }
        // the background's steadiness, on a sparse grid
        if i % 7 == 0 && vals.len() >= 3 {
            let mut l: Vec<f64> = vals.iter().map(|v| 0.2126 * f64::from(v[0]) + 0.7152 * f64::from(v[1]) + 0.0722 * f64::from(v[2])).collect();
            l.sort_by(f64::total_cmp);
            let med = l[l.len() / 2];
            let mut d: Vec<f64> = l.iter().map(|v| (v - med).abs()).collect();
            d.sort_by(f64::total_cmp);
            dev_sum += d[d.len() / 2];
            dev_n += 1;
        }
        // nowhere clear: the first frame's pixel
        if vals.is_empty() {
            if let Some(px) = frames.first().and_then(|f| f.0.get(i * 4..i * 4 + 3)) {
                vals.push([px[0], px[1], px[2]]);
            }
        } else {
            clear += 1;
        }
        for c in 0..3 {
            let mut ch: Vec<u8> = vals.iter().map(|v| v[c]).collect();
            ch.sort_unstable();
            if let (Some(v), Some(o)) = (ch.get(ch.len() / 2), out.get_mut(i * 4 + c)) {
                *o = *v;
            }
        }
        if let Some(o) = out.get_mut(i * 4 + 3) {
            *o = 255;
        }
    }
    Some(Plate {
        rgba: out,
        w,
        h,
        covered: clear as f64 / (w * h) as f64,
        behind: if path == 0 { 0.0 } else { path_clear as f64 / path as f64 },
        unsteady: if dev_n == 0 { f64::MAX } else { dev_sum / dev_n as f64 },
    })
}

/// `rgba` (`w` × `h`) resized bilinearly to `tw` × `th` (at most 8192 a side).
pub(crate) fn resize_rgba(rgba: &[u8], w: usize, h: usize, tw: usize, th: usize) -> (Vec<u8>, usize, usize) {
    let (tw, th) = (tw.clamp(1, 8192), th.clamp(1, 8192));
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return (vec![0; tw * th * 4], tw, th);
    }
    let mut out = vec![0u8; tw * th * 4];
    for y in 0..th {
        let fy = ((y as f32 + 0.5) * h as f32 / th as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..tw {
            let fx = ((x as f32 + 0.5) * w as f32 / tw as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..4 {
                let at = |xx: usize, yy: usize| f32::from(rgba.get((yy * w + xx) * 4 + c).copied().unwrap_or(0));
                let v = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
                if let Some(o) = out.get_mut((y * tw + x) * 4 + c) {
                    *o = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (out, tw, th)
}

#[cfg(test)]
#[path = "cutout_tests.rs"]
mod tests;
