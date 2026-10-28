//! The scene index of media items: what is in each clip, shot by shot, so edits can be planned
//! from it and templates can fill themselves.
//!
//! | command | does |
//! |---|---|
//! | `media.analyzeScenes` | job: index `items` (default: the Project panel's selection, else every video and photo item) — shots, picture quality, camera moves, descriptions, embeddings; one undo step ("Analyze Scenes") |
//! | `media.scenes` | the index of `item`: its shots (seconds), quality, camera, description |
//! | `media.searchScenes` | shots ranked for a text `query`: what they look like (SigLIP 2), what their descriptions say, what is said in them (transcripts) |
//! | `media.bestShots` | the best `count` shots for a `purpose` (hook, b-roll, reveal, talking-head, establishing) |
//! | `scenes.status` | which scene models are installed, their sizes and licences |
//! | `scenes.downloadModel` | job: download a model (or the `fast` / `full` set), SHA-256 checked, after the user agreed |
//!
//! **How** ([`analyse`], a background job: `jobs.list` progress, `jobs.cancel` stops it and
//! nothing changes). Each clip is decoded at its own rate (at most 30 frames a second) and small:
//! the frames go to TransNetV2 (48×27) for the cuts when it is installed, else — and for a 360°
//! camera's own dual-fisheye recording, one take without cuts — to the luma-grid
//! and histogram detector of Scene Edit Detection (`filmcraft_render::scene`), and to the
//! global-motion measure (96 px) for each shot's camera move and shake. A shot's keyframes (its
//! middle; two or three with `detail: "full"`) are decoded larger, measured (sharpness, exposure),
//! scored by the LAION aesthetic head, embedded by SigLIP 2, and described by Qwen3-VL (2B, or 4B
//! with `detail: "full"`), sixteen shots at a time (four with `full`). A photo is one shot.
//! Every model is optional: what is not installed is left out (`scenes.status` says what is).
//!
//! **The cache.** With a scene cache (`<data dir>/scene-cache`, on in the app and the CLI) each
//! clip's index is kept by the file's content identity, its picture size, length and rate, the
//! detail and the models' pins: the same footage indexed again — in another project, after
//! re-importing — is not decoded or described again. `fresh: true` reads none of it.
//!
//! The index is stored with the project (`Project::scenes`, media time). The engine never trusts
//! one read from a file: [`filmcraft_project::scenes::SceneIndex::sanitized`].

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_project::scenes::{Embedding, SCENE_INDEX_VERSION, SceneIndex, Shot, ShotQuality, ShotUse};
use filmcraft_project::{ItemId, ItemKind, MediaRef};
use filmcraft_scenes::{best, boundaries, json as scene_json, quality, search};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, str_p, u64_p};
use crate::scene_models::{DescribeStats, Detail, Pic, SceneModels};
use crate::{EngineError, Result, Session};

/// The undo step's name.
pub const LABEL: &str = "Analyze Scenes";
const ANALYZE: &str = "media.analyzeScenes";
const DOWNLOAD: &str = "scenes.downloadModel";
/// Most items one job indexes.
const MAX_ITEMS: usize = 2_000;
/// Long side of a keyframe for the fast pass, and for `full`.
const KEY_SIDE_FAST: u32 = 640;
const KEY_SIDE_FULL: u32 = 960;
/// Finished jobs' outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 256;
/// The scene cache's directory under the data directory, and its cap.
pub const CACHE_DIR: &str = "scene-cache";
const CACHE_CAP: u64 = 256 * 1024 * 1024;
/// Bumped whenever what the analysis makes changes: older cache entries miss.
const CACHE_VERSION: u32 = 1;
/// Items a job holds keyframes for at once (the models are loaded once a group).
const GROUP: usize = 32;
/// The shortest shot kept (shorter ones join a neighbour).
const MIN_SHOT_SECONDS: f64 = 0.4;

/// The scene cache in `<data_dir>/scene-cache`.
pub fn cache_in(data_dir: &std::path::Path) -> crate::redact_cache::RedactCache {
    crate::redact_cache::RedactCache::new(data_dir.join(CACHE_DIR), CACHE_CAP)
}

/// The session's scene jobs, what the finished ones made, the models and the cache.
#[derive(Default)]
pub struct Scenes {
    pending: Vec<Pending>,
    outcomes: HashMap<u64, Value>,
    /// The networks (None: the downloaded models are used when the build has them). Hosts and
    /// tests install their own here.
    pub models: Option<Arc<dyn SceneModels>>,
    /// The downloaded models, made when first needed.
    local: Option<Arc<dyn SceneModels>>,
    /// The scene cache (native frontends turn it on with [`Session::use_scene_cache`]).
    pub cache: Option<Arc<crate::redact_cache::RedactCache>>,
    /// `media.checkText` jobs not yet collected ([`crate::scene_check`]).
    pub(crate) checks: Vec<crate::scene_check::PendingCheck>,
}

impl Scenes {
    /// The models to use: the installed ones, else the downloaded ones (made once).
    fn models(&mut self) -> Option<Arc<dyn SceneModels>> {
        if let Some(m) = &self.models {
            return Some(m.clone());
        }
        if self.local.is_none() {
            self.local = crate::transcript::models_dir().and_then(crate::scene_models::local);
        }
        self.local.clone()
    }

    /// The models `media.checkText` uses (the same as the index's).
    pub(crate) fn check_models(&mut self) -> Option<Arc<dyn SceneModels>> {
        self.models()
    }

    /// Keep what a finished job made for `jobs.list` (the oldest goes past [`MAX_OUTCOMES`]).
    pub(crate) fn put_outcome(&mut self, job: u64, outcome: Value) {
        if self.outcomes.len() >= MAX_OUTCOMES
            && let Some(old) = self.outcomes.keys().min().copied()
        {
            self.outcomes.remove(&old);
        }
        self.outcomes.insert(job, outcome);
    }
}

/// What a job made: each item's index and where it came from, and the report.
type Made = (Vec<(ItemId, SceneIndex, bool)>, Value);
type Results = Arc<Mutex<Option<std::result::Result<Made, String>>>>;

struct Pending {
    job: u64,
    results: Results,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: ANALYZE,
            label: "Analyze Scenes",
            menu: &[],
            shortcut: None,
            params: r#"{"items":[id]?,"detail":"fast"|"full"="fast","describe":bool=true,"embed":bool=true,"aesthetic":bool=true,"fresh":bool=false,"wait":bool=false}"#,
            enabled: always,
            run: analyze,
            journal: true,
        },
        CommandSpec {
            id: "media.scenes",
            label: "Media Scenes",
            menu: &[],
            shortcut: None,
            params: r#"{"item":id,"embeddings":bool=false}"#,
            enabled: always,
            run: show,
            journal: false,
        },
        CommandSpec {
            id: "media.searchScenes",
            label: "Search Scenes",
            menu: &[],
            shortcut: None,
            params: r#"{"query":str,"limit":u32=20,"items":[id]?}"#,
            enabled: always,
            run: search_scenes,
            journal: false,
        },
        CommandSpec {
            id: "media.bestShots",
            label: "Best Shots",
            menu: &[],
            shortcut: None,
            params: r#"{"items":[id]?,"purpose":"hook"|"broll"|"reveal"|"talking-head"|"establishing"="hook","count":u32=5}"#,
            enabled: always,
            run: best_shots,
            journal: false,
        },
        CommandSpec { id: "scenes.status", label: "Scene Models", menu: &[], shortcut: None, params: "{}", enabled: always, run: status, journal: false },
        CommandSpec {
            id: DOWNLOAD,
            label: "Download Scene Model",
            menu: &[],
            shortcut: None,
            params: r#"{"model":"fast"|"full"|"transnetv2"|"qwen3-vl-2b"|"qwen3-vl-4b"|"qwen3-vl-8b"|"siglip2-base-224"|"clip-vit-l14-aesthetic"="fast","wait":bool=false}"#,
            enabled: always,
            run: download,
            journal: false,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// what to analyse
// ---------------------------------------------------------------------------------------------

/// One media item to index.
struct Work {
    item: ItemId,
    name: String,
    src: filmcraft_media::SharedSource,
    still: bool,
    duration: Tick,
    /// The media's own frame duration, and every how many frames one is analysed.
    frame: Tick,
    step: usize,
    /// The cache's key of the file (content identity and what decides its frames), when it is
    /// a file.
    media_key: Option<String>,
    /// A camera's own recording of one take (dual-fisheye 360°, straight from the camera): it has
    /// no cuts to find, so the shot-boundary network is not run on it (the signature detector's
    /// look at sudden changes is enough).
    one_take: bool,
}

fn ids_p(cmd: &str, p: &Value) -> Result<Option<Vec<ItemId>>> {
    match p.get("items").filter(|v| !v.is_null()) {
        None => Ok(p.get("item").and_then(Value::as_u64).map(|i| vec![ItemId(i)])),
        Some(Value::Array(a)) if a.len() <= MAX_ITEMS => {
            a.iter().map(|v| v.as_u64().map(ItemId)).collect::<Option<Vec<_>>>().map(Some).ok_or_else(|| bad(cmd, "`items` is a list of item ids"))
        }
        Some(_) => Err(bad(cmd, format!("`items` is a list of at most {MAX_ITEMS} item ids"))),
    }
}

/// The media items `items` name (subclips resolve to their media), else the Project panel's
/// selection, else every video and photo item; in order, without repeats.
fn targets(s: &Session, cmd: &str, p: &Value) -> Result<Vec<ItemId>> {
    let named = ids_p(cmd, p)?;
    let raw: Vec<ItemId> = match named.clone() {
        Some(v) => v,
        None if !s.state.project_selection.is_empty() => s.state.project_selection.clone(),
        None => s.project.items.keys().copied().collect(),
    };
    let mut out = Vec::new();
    for i in raw {
        let Some((root, m, _)) = s.project.resolve_media(i) else {
            if named.is_some() && s.project.item(i).is_none() {
                return Err(bad(cmd, format!("no item {}", i.0)));
            }
            continue;
        };
        if m.info.has_video() && !out.contains(&root) {
            out.push(root);
        }
    }
    Ok(out)
}

fn work_for(s: &Session, item: ItemId) -> std::result::Result<Work, String> {
    let pi = s.project.item(item).ok_or("no such item")?;
    let ItemKind::Media(m) = &pi.kind else { return Err("not media".into()) };
    // (360° media is indexed as its stitched panorama)
    let src = crate::vr360::analysis_view(s, item).ok_or("its media is offline or unreadable")?;
    let still = matches!(m.info.kind, filmcraft_media::MediaKind::Still);
    let rate = m.frame_rate().sane();
    let frame = rate.frame_duration().max(Tick(1));
    let step = boundaries::frame_step(rate.as_f64());
    let (w, h) = m.info.video.as_ref().map_or((0, 0), |v| (v.width, v.height));
    let identity = match &m.media {
        MediaRef::File { path } if !(s.media.use_proxies() && m.proxy.is_some()) => {
            m.identity.map(|i| (i.size, i.fingerprint)).or_else(|| crate::relink::identity_of(&*s.services, path).ok().map(|i| (i.size, i.fingerprint)))
        }
        _ => None,
    };
    // (a 360° picture is indexed as its panorama: how it maps onto the sphere and its horizon
    // decide what the panorama shows)
    let sphere = s.project.vr_of(item).map(|v| format!("|sphere:{}", crate::vr360::sphere_key(v))).unwrap_or_default();
    let media_key = identity.map(|(size, fp)| format!("file:{size}:{fp:016x}|{w}x{h}|{}|{}/{}{sphere}", m.duration().0, rate.num, rate.den));
    let one_take = s.project.vr_of(item).is_some_and(|v| !v.is_equirect());
    Ok(Work { item, name: pi.name.clone(), src, still, duration: m.duration(), frame, step, media_key, one_take })
}

// ---------------------------------------------------------------------------------------------
// the job
// ---------------------------------------------------------------------------------------------

/// What the job is asked to do.
#[derive(Clone, Debug)]
struct Options {
    detail: Detail,
    describe: bool,
    embed: bool,
    aesthetic: bool,
    fresh: bool,
}

/// The models (with pins) a job uses, which key its cache.
#[derive(Clone, Debug, Default)]
struct Using {
    boundaries: Option<String>,
    describer: Option<String>,
    embedder: Option<String>,
    aesthetic: Option<String>,
}

impl Using {
    fn key(&self, o: &Options) -> String {
        let n = |v: &Option<String>| v.clone().unwrap_or_else(|| "-".into());
        format!("b={}|d={}|e={}|a={}|detail={}", n(&self.boundaries), n(&self.describer), n(&self.embedder), n(&self.aesthetic), o.detail.name())
    }
}

/// One item's analysis frames: 48×27 RGB for TransNetV2, the fallback detector's signature and
/// the motion luma, per analysed frame, and each frame's media time.
struct Frames {
    times: Vec<Tick>,
    small: Vec<Vec<u8>>,
    sigs: Vec<filmcraft_render::scene::Signature>,
    luma: Vec<quality::Luma>,
}

/// The frame TransNetV2 sees: 48×27 RGB, area-averaged.
fn rgb_48x27(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (tw, th) = (48usize, 27usize);
    let mut out = Vec::with_capacity(tw * th * 3);
    for y in 0..th {
        let (y0, y1) = (y * h / th, ((y + 1) * h / th).max(y * h / th + 1).min(h.max(1)));
        for x in 0..tw {
            let (x0, x1) = (x * w / tw, ((x + 1) * w / tw).max(x * w / tw + 1).min(w.max(1)));
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * w + sx) * 4;
                    if let Some(p) = rgba.get(i..i + 3) {
                        acc[0] += u32::from(p[0]);
                        acc[1] += u32::from(p[1]);
                        acc[2] += u32::from(p[2]);
                        n += 1;
                    }
                }
            }
            let n = n.max(1);
            out.extend_from_slice(&[(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]);
        }
    }
    out
}

/// A decoded frame as RGBA8, reduced first while it is still Y'CbCr (box-averaged by a power of
/// two, keeping its long side at least `long`): converting a whole 4K frame to RGBA to look at a
/// 96-pixel copy of it cost most of the analysis.
pub(crate) fn small_rgba(vf: &filmcraft_frame::VideoFrame, long: u32) -> (Vec<u8>, u32, u32) {
    let side = vf.width.max(vf.height);
    let mut n = 1u32;
    while n < 64 && side / (n * 2) >= long.max(1) {
        n *= 2;
    }
    if n >= 2
        && let Some(d) = vf.box_decimated(n as usize)
    {
        return (d.to_rgba8(), d.width, d.height);
    }
    (vf.to_rgba8(), vf.width, vf.height)
}

/// Decode `w`'s analysis frames (every `step`-th frame, small).
fn decode_frames(w: &Work, prog: &filmcraft_export::Progress) -> std::result::Result<Frames, String> {
    let info = w.src.info().video.clone().ok_or("no video")?;
    let scale = (quality::MOTION_SIDE as f32 / info.width.max(info.height).max(1) as f32).clamp(0.01, 1.0);
    let n = (w.duration.0 / w.frame.0.max(1)).max(1) as usize;
    let mut f = Frames { times: Vec::new(), small: Vec::new(), sigs: Vec::new(), luma: Vec::new() };
    let mut k = 0usize;
    while k < n && f.times.len() < boundaries::MAX_FRAMES {
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let t = Tick(w.frame.0.saturating_mul(k as i64));
        match w.src.video_frame(filmcraft_media::FrameRequest { time: t, scale }) {
            Ok(vf) => {
                let (rgba, fw, fh) = small_rgba(&vf, quality::MOTION_SIDE as u32);
                let (fw, fh) = (fw as usize, fh as usize);
                if !w.one_take {
                    f.small.push(rgb_48x27(&rgba, fw, fh));
                }
                f.sigs.push(filmcraft_render::scene::signature(&rgba, fw, fh));
                f.luma.push(quality::luma(&rgba, fw, fh, quality::MOTION_SIDE).unwrap_or(quality::Luma { w: 0, h: 0, v: Vec::new() }));
                f.times.push(t);
            }
            // a frame that does not decode ends the clip's analysis there
            Err(e) if f.times.is_empty() => return Err(format!("can't decode its first frame: {e}")),
            Err(_) => break,
        }
        prog.done.fetch_add(w.step as u64, Ordering::Relaxed);
        k += w.step;
    }
    Ok(f)
}

/// Shots `(first frame, end frame)` from the cut probabilities, or the fallback's cuts.
fn shots_of(f: &Frames, models: Option<&dyn SceneModels>, fps: f64, prog: &filmcraft_export::Progress) -> (Vec<(usize, usize)>, &'static str) {
    let n = f.times.len();
    let min = ((MIN_SHOT_SECONDS * fps).round() as usize).max(2);
    if let Some(m) = models.filter(|m| m.boundaries_id().is_some())
        && let Ok(p) = m.cuts(&f.small, &|| prog.cancel.load(Ordering::Relaxed))
    {
        let cuts = boundaries::cuts_from_probabilities(&p, 0.5);
        return (boundaries::shots_from_cuts(&cuts, n, min), "transnetv2");
    }
    let mut det = filmcraft_render::scene::SceneDetector::new();
    for s in &f.sigs {
        det.push(s.clone());
    }
    let opts = filmcraft_render::scene::SceneOptions { sensitivity: 50.0, min_shot_frames: min };
    (boundaries::shots_from_cuts(&det.cuts(&opts), n, min), "signature")
}

/// A keyframe of `w` at media time `t`, its long side at most `side`.
fn keyframe(w: &Work, t: Tick, side: u32) -> Option<Pic> {
    let info = w.src.info().video.clone()?;
    let scale = (side as f32 / info.width.max(info.height).max(1) as f32).clamp(0.01, 1.0);
    let vf = w.src.video_frame(filmcraft_media::FrameRequest { time: t, scale }).ok()?;
    let (rgba, fw, fh) = small_rgba(&vf, side);
    let (rgba, fw, fh) = crate::media_frame::downscale_rgba(&rgba, fw, fh, side);
    Some(Pic { rgba, w: fw as usize, h: fh as usize })
}

/// One item part-way: its shots and their keyframes.
struct Partial {
    item: ItemId,
    name: String,
    index: SceneIndex,
    keys: Vec<Vec<Pic>>,
    cache_key: Option<String>,
    timing: Timing,
}

fn now_unix() -> i64 {
    crate::autosave::unix_now()
}

/// Shots, quality and camera moves of one item (no models but TransNetV2).
/// Where one item's measuring went: frames analysed and seconds decoding them, finding the cuts,
/// decoding and measuring keyframes, and measuring motion.
#[derive(Clone, Copy, Debug, Default)]
struct Timing {
    frames: usize,
    decode: f64,
    cuts: f64,
    keyframes: f64,
    motion: f64,
}

impl Timing {
    fn json(&self) -> Value {
        let r = |v: f64| (v * 1000.0).round() / 1000.0;
        json!({"frames": self.frames, "decodeSeconds": r(self.decode), "cutsSeconds": r(self.cuts), "keyframeSeconds": r(self.keyframes), "motionSeconds": r(self.motion)})
    }
}

fn measure(
    w: &Work,
    o: &Options,
    models: Option<&dyn SceneModels>,
    prog: &filmcraft_export::Progress,
) -> std::result::Result<(SceneIndex, Vec<Vec<Pic>>, Timing), String> {
    let mut tm = Timing::default();
    let side = if o.detail == Detail::Full { KEY_SIDE_FULL } else { KEY_SIDE_FAST };
    let mut idx = SceneIndex { version: SCENE_INDEX_VERSION, duration: w.duration, created: now_unix(), detail: o.detail.name().into(), ..Default::default() };
    if w.still {
        idx.boundaries = "still".into();
        let pic = keyframe(w, Tick::ZERO, side).ok_or("the photo does not decode")?;
        let mut q = ShotQuality::default();
        quality::measure_keyframe(&mut q, &pic.rgba, pic.w, pic.h, 0);
        idx.shots.push(Shot { start: Tick::ZERO, end: w.duration.max(w.frame), keyframes: vec![Tick::ZERO], quality: q, ..Default::default() });
        prog.done.fetch_add(1, Ordering::Relaxed);
        return Ok((idx, vec![vec![pic]], tm));
    }
    let t = web_time::Instant::now();
    let f = decode_frames(w, prog)?;
    tm.frames = f.times.len();
    tm.decode = t.elapsed().as_secs_f64();
    let t = web_time::Instant::now();
    let fps = 1.0 / (w.frame.seconds() * w.step as f64).max(1e-6);
    let (shots, how) = shots_of(&f, models.filter(|_| !w.one_take), fps, prog);
    tm.cuts = t.elapsed().as_secs_f64();
    idx.boundaries = how.into();
    let full = o.detail == Detail::Full;
    let span = Tick(w.frame.0.saturating_mul(w.step as i64));
    let mut keys = Vec::with_capacity(shots.len());
    for (a, b) in shots {
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let start = f.times.get(a).copied().unwrap_or(Tick::ZERO);
        let end = if b >= f.times.len() { w.duration } else { f.times.get(b).copied().unwrap_or(w.duration) };
        let end = end.max(start + span).min(w.duration.max(start + span));
        let secs = (end - start).seconds();
        let picks = boundaries::keyframes(a, b, boundaries::keyframe_count(secs, full));
        let mut pics = Vec::new();
        let mut q = ShotQuality::default();
        let mut times = Vec::new();
        let t = web_time::Instant::now();
        for k in picks {
            let t = f.times.get(k).copied().unwrap_or(start);
            if let Some(pic) = keyframe(w, t, side) {
                quality::measure_keyframe(&mut q, &pic.rgba, pic.w, pic.h, pics.len());
                pics.push(pic);
                times.push(t);
            }
        }
        tm.keyframes += t.elapsed().as_secs_f64();
        let t = web_time::Instant::now();
        // the camera: shifts between consecutive analysis frames of the shot
        let shifts: Vec<(f32, f32)> =
            (a + 1..b).filter_map(|i| Some((f.luma.get(i - 1)?, f.luma.get(i)?))).filter_map(|(p, c)| quality::global_shift(p, c, 4)).collect();
        let width = f.luma.get(a).map_or(quality::MOTION_SIDE, |l| l.w.max(1));
        let (camera, motion, shake) = quality::camera_motion(&shifts, fps, width);
        tm.motion += t.elapsed().as_secs_f64();
        q.motion = motion;
        q.shake = shake;
        q.score = quality::overall(&q);
        idx.shots.push(Shot { start, end, keyframes: times, quality: q, camera, ..Default::default() });
        keys.push(pics);
    }
    Ok((idx, keys, tm))
}

/// Run the models over a group of items: aesthetic scores, embeddings, descriptions.
fn enrich(
    parts: &mut [Partial],
    o: &Options,
    models: Option<&dyn SceneModels>,
    using: &Using,
    prog: &filmcraft_export::Progress,
    stats: &mut DescribeStats,
) -> Vec<String> {
    let mut notes = Vec::new();
    let Some(m) = models else { return notes };
    // nothing new in this group (every item came from the cache): no model is loaded
    if parts.iter().all(|p| p.keys.iter().all(Vec::is_empty)) {
        return notes;
    }
    if o.aesthetic && using.aesthetic.is_some() {
        *lock(&prog.status) = "Scoring how the shots look".into();
        for p in parts.iter_mut() {
            for (shot, pics) in p.index.shots.iter_mut().zip(&p.keys) {
                if pics.is_empty() {
                    continue;
                }
                match m.aesthetic(pics) {
                    Ok(v) if !v.is_empty() => {
                        shot.quality.aesthetic = Some(v.iter().sum::<f32>() / v.len() as f32);
                        shot.quality.score = quality::overall(&shot.quality);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        notes.push(format!("aesthetic score: {e}"));
                        break;
                    }
                }
            }
            p.index.aesthetic = using.aesthetic.clone().unwrap_or_default();
        }
    }
    if o.embed && using.embedder.is_some() {
        *lock(&prog.status) = "Embedding the shots".into();
        for p in parts.iter_mut() {
            for (shot, pics) in p.index.shots.iter_mut().zip(&p.keys) {
                if pics.is_empty() {
                    continue;
                }
                match m.embed_images(pics) {
                    Ok(v) if !v.is_empty() => {
                        let d = v.first().map_or(0, Vec::len);
                        let mut mean = vec![0f32; d];
                        for e in v.iter().filter(|e| e.len() == d) {
                            mean.iter_mut().zip(e).for_each(|(a, b)| *a += b);
                        }
                        shot.embedding = Embedding(mean).normalized().quantized();
                    }
                    Ok(_) => {}
                    Err(e) => {
                        notes.push(format!("embeddings: {e}"));
                        break;
                    }
                }
            }
            p.index.embedder = using.embedder.clone().unwrap_or_default();
        }
    }
    if o.describe && using.describer.is_some() {
        // every shot of the group at once: the describer batches shots of one size
        let shots: Vec<(usize, usize)> = parts.iter().enumerate().flat_map(|(pi, p)| (0..p.keys.len()).map(move |si| (pi, si))).collect();
        let pics: Vec<Vec<Pic>> = shots.iter().map(|(pi, si)| parts.get(*pi).and_then(|p| p.keys.get(*si)).cloned().unwrap_or_default()).collect();
        let want: Vec<usize> = (0..pics.len()).filter(|i| pics.get(*i).is_some_and(|p| !p.is_empty())).collect();
        let batch: Vec<Vec<Pic>> = want.iter().filter_map(|i| pics.get(*i).cloned()).collect();
        let base = prog.done.load(Ordering::Relaxed);
        let mut progress = |done: usize, total: usize| {
            prog.done.store(base + done as u64, Ordering::Relaxed);
            *lock(&prog.status) = format!("Describing shots: {done} of {total}");
            !prog.cancel.load(Ordering::Relaxed)
        };
        match m.describe(&batch, o.detail, &mut progress) {
            Ok((answers, st)) => {
                stats.batches += st.batches;
                stats.prompt_tokens += st.prompt_tokens;
                stats.generated += st.generated;
                stats.vision_seconds += st.vision_seconds;
                stats.prefill_seconds += st.prefill_seconds;
                stats.decode_seconds += st.decode_seconds;
                stats.load_seconds += st.load_seconds;
                let mut unread = 0;
                for (k, a) in want.iter().zip(answers) {
                    let Some((pi, si)) = shots.get(*k).copied() else { continue };
                    match scene_json::parse_description(&a, o.detail == Detail::Full) {
                        Ok(d) => {
                            if let Some(shot) = parts.get_mut(pi).and_then(|p| p.index.shots.get_mut(si)) {
                                shot.description = Some(d);
                            }
                        }
                        Err(_) => unread += 1,
                    }
                }
                if unread > 0 {
                    notes.push(format!("{unread} description(s) could not be read"));
                }
                for p in parts.iter_mut() {
                    p.index.describer = using.describer.clone().unwrap_or_default();
                }
            }
            Err(e) => notes.push(format!("descriptions: {e}")),
        }
    }
    notes
}

/// The whole job: every item measured, then the models over each group of items.
fn analyse(
    work: Vec<Work>,
    o: Options,
    models: Option<Arc<dyn SceneModels>>,
    cache: Option<Arc<crate::redact_cache::RedactCache>>,
    prog: &filmcraft_export::Progress,
) -> std::result::Result<Made, String> {
    let t0 = web_time::Instant::now();
    let m = models.as_deref();
    let using = Using {
        boundaries: m.and_then(|m| m.boundaries_id()),
        describer: if o.describe { m.and_then(|m| m.describer_id(o.detail)) } else { None },
        embedder: if o.embed { m.and_then(|m| m.embedder_id()) } else { None },
        aesthetic: if o.aesthetic { m.and_then(|m| m.aesthetic_id()) } else { None },
    };
    let uses_gpu = using.boundaries.is_some() || using.describer.is_some() || using.embedder.is_some() || using.aesthetic.is_some();
    // the GPU, taken when the first item needs analysing (not for what the cache holds)
    let mut gpu: Option<Box<dyn std::any::Any + Send>> = None;
    let mut gpu_taken = false;
    let mut gpu_wait = 0f64;
    let mut gpu_wait_total = 0f64;
    let key_of = |w: &Work| w.media_key.as_ref().map(|k| format!("scenes|v{CACHE_VERSION}|{k}|{}", using.key(&o)));
    let mut made: Vec<(ItemId, SceneIndex, bool)> = Vec::new();
    let mut report = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut stats = DescribeStats::default();
    let mut measured_secs = 0f64;
    let n = work.len();
    for group in work.chunks(GROUP) {
        let mut parts: Vec<Partial> = Vec::new();
        let tm = web_time::Instant::now();
        for w in group {
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("stopped".into());
            }
            let key = key_of(w);
            if !o.fresh
                && let (Some(c), Some(k)) = (cache.as_deref(), key.as_deref())
                && let Some(idx) = c.load::<SceneIndex>(k)
            {
                let idx = idx.sanitized();
                report.push(json!({"item": w.item.0, "name": w.name, "shots": idx.shots.len(), "boundaries": idx.boundaries, "fromCache": true}));
                prog.done.fetch_add((w.duration.0 / w.frame.0.max(1)).max(1) as u64, Ordering::Relaxed);
                made.push((w.item, idx, true));
                continue;
            }
            if uses_gpu && !gpu_taken {
                *lock(&prog.status) = "Waiting for the GPU".into();
                let tw = web_time::Instant::now();
                gpu = m.map(|m| m.gpu()).transpose()?.flatten();
                gpu_wait = tw.elapsed().as_secs_f64();
                gpu_taken = true;
            }
            *lock(&prog.status) = format!("Finding the shots of {} ({} of {n})", w.name, made.len() + parts.len() + 1);
            match measure(w, &o, m, prog) {
                Ok((index, keys, timing)) => parts.push(Partial { item: w.item, name: w.name.clone(), index, keys, cache_key: key, timing }),
                Err(e) if e == "stopped" => return Err(e),
                Err(e) => {
                    notes.push(format!("{}: {e}", w.name));
                    report.push(json!({"item": w.item.0, "name": w.name, "error": e}));
                }
            }
        }
        measured_secs += tm.elapsed().as_secs_f64();
        if gpu_taken && measured_secs >= gpu_wait {
            // (once: the wait is in the first group's time)
            let w = std::mem::take(&mut gpu_wait);
            measured_secs -= w;
            gpu_wait_total += w;
        }
        notes.extend(enrich(&mut parts, &o, m, &using, prog, &mut stats));
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        for p in parts {
            if let (Some(c), Some(k)) = (cache.as_deref(), p.cache_key.as_deref()) {
                c.store(k, &p.index);
            }
            report.push(json!({"item": p.item.0, "name": p.name, "shots": p.index.shots.len(), "boundaries": p.index.boundaries, "described": p.index.described(), "fromCache": false, "timing": p.timing.json()}));
            made.push((p.item, p.index, false));
        }
    }
    if let Some(m) = m.filter(|_| gpu_taken) {
        m.release();
    }
    drop(gpu);
    notes.sort();
    notes.dedup();
    let secs = t0.elapsed().as_secs_f64();
    let shots: usize = made.iter().map(|(_, i, _)| i.shots.len()).sum();
    let shots_described: usize = made.iter().map(|(_, i, _)| i.shots.iter().filter(|s| s.description.is_some()).count()).sum();
    let fresh_shots: usize = made.iter().filter(|(_, _, cached)| !cached).map(|(_, i, _)| i.shots.len()).sum();
    let value = json!({
        "items": report,
        "shots": shots,
        "described": shots_described,
        "seconds": secs,
        "shotsPerSecond": if secs > 0.0 { fresh_shots as f64 / secs } else { 0.0 },
        "analysed": fresh_shots,
        "models": {"boundaries": using.boundaries, "describer": using.describer, "embedder": using.embedder, "aesthetic": using.aesthetic},
        "timings": {
            "gpuWaitSeconds": gpu_wait_total,
            "measureSeconds": measured_secs,
            "describeLoadSeconds": stats.load_seconds,
            "visionSeconds": stats.vision_seconds,
            "prefillSeconds": stats.prefill_seconds,
            "decodeSeconds": stats.decode_seconds,
            "batches": stats.batches,
            "generatedTokens": stats.generated,
        },
        "notes": notes,
    });
    Ok((made, value))
}

fn detail_p(cmd: &str, p: &Value) -> Result<Detail> {
    match str_p(p, "detail").map(|d| d.trim().to_ascii_lowercase()) {
        None => Ok(Detail::Fast),
        Some(d) if d == "fast" => Ok(Detail::Fast),
        Some(d) if d == "full" || d == "detail" || d == "detailed" => Ok(Detail::Full),
        Some(_) => Err(bad(cmd, "`detail` is \"fast\" or \"full\"")),
    }
}

fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = ANALYZE;
    let o = Options {
        detail: detail_p(cmd, p)?,
        describe: bool_p(p, "describe").unwrap_or(true),
        embed: bool_p(p, "embed").unwrap_or(true),
        aesthetic: bool_p(p, "aesthetic").unwrap_or(true),
        fresh: bool_p(p, "fresh").unwrap_or(false),
    };
    let items = targets(s, cmd, p)?;
    if items.is_empty() {
        return Err(bad(cmd, "no video or photo items to analyse (pass `items`, select clips in the Project panel, or import media)"));
    }
    if items.len() > MAX_ITEMS {
        return Err(bad(cmd, format!("at most {MAX_ITEMS} items a job")));
    }
    let mut work = Vec::new();
    let mut skipped = Vec::new();
    for i in items {
        match work_for(s, i) {
            Ok(w) => work.push(w),
            Err(e) => skipped.push(json!({"item": i.0, "reason": e})),
        }
    }
    if work.is_empty() {
        return Err(bad(cmd, format!("none of the items can be read: {}", Value::Array(skipped))));
    }
    let models = s.scenes.models();
    let cache = s.scenes.cache.clone();
    let total: u64 = work.iter().map(|w| (w.duration.0 / w.frame.0.max(1)).max(1) as u64).sum();
    let n = work.len();
    let job = crate::narration::new_job(s, format!("{LABEL} ({n} item{})", if n == 1 { "" } else { "s" }));
    job.progress.total.store(total, Ordering::Relaxed);
    let id = job.id;
    let results: Results = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let run = move || {
        let t0 = web_time::Instant::now();
        let r = analyse(work, o, models, cache, &prog);
        let status = match &r {
            Ok((made, v)) => format!("Indexed {} item(s), {} shot(s) in {:.1}s", made.len(), v["shots"], t0.elapsed().as_secs_f64()),
            Err(e) if e == "stopped" => "Stopped: nothing was changed".into(),
            Err(e) => e.clone(),
        };
        let r2 = r.as_ref().map(|_| status.clone()).map_err(Clone::clone);
        *lock(&out) = Some(r);
        crate::narration::finish(&prog, &res, r2, t0);
    };
    s.scenes.pending.push(Pending { job: id, results });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-scenes", run)?;
    let mut answer = json!({"job": id, "items": n, "skipped": skipped});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.scenes.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["scenes"] = o.clone();
        }
    } else {
        answer["poll"] = json!("jobs.list");
    }
    Ok(answer)
}

/// Apply finished jobs (one undo step each) and keep what they made for `jobs.list`. Called each
/// UI frame (through [`Session::poll_persistence`]), by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    crate::scene_check::poll(s);
    let mut i = 0;
    while i < s.scenes.pending.len() {
        let job = s.scenes.pending[i].job;
        let finished = s.jobs.iter().find(|j| j.id == job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.scenes.pending.remove(i);
        let cancelled = s.jobs.iter().find(|j| j.id == job).is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        let outcome = match lock(&pj.results).take() {
            Some(Ok((made, mut report))) if !cancelled => {
                let items: Vec<(ItemId, SceneIndex)> = made.into_iter().filter(|(i, _, _)| s.project.item(*i).is_some()).map(|(i, x, _)| (i, x)).collect();
                if !items.is_empty() {
                    let r = s.edit(LABEL, move |pr, _| {
                        for (i, x) in items {
                            pr.scenes.insert(i, Arc::new(x));
                        }
                        Ok(())
                    });
                    if let Err(e) = r {
                        report["error"] = json!(e.to_string());
                    }
                }
                report
            }
            Some(Ok(_)) => json!({"error": "stopped"}),
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "the job ended without a result"}),
        };
        s.scenes.put_outcome(job, outcome);
    }
}

/// What a finished scene job made (`scenes` in its `jobs.list` entry).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.scenes.outcomes.get(&job).cloned()
}

// ---------------------------------------------------------------------------------------------
// reading the index
// ---------------------------------------------------------------------------------------------

fn secs(t: Tick) -> f64 {
    (t.seconds() * 1000.0).round() / 1000.0
}

/// A shot as callers see it: times in seconds, the embedding left out unless asked for.
pub fn shot_json(i: usize, sh: &Shot, embeddings: bool) -> Value {
    let mut v = json!({
        "shot": i,
        "start": secs(sh.start),
        "end": secs(sh.end),
        "seconds": (sh.seconds() * 1000.0).round() / 1000.0,
        "keyframes": sh.keyframes.iter().map(|k| secs(*k)).collect::<Vec<_>>(),
        "camera": sh.camera,
        "quality": sh.quality,
    });
    if let Some(d) = &sh.description {
        v["description"] = serde_json::to_value(d).unwrap_or(Value::Null);
    }
    if embeddings && !sh.embedding.is_empty() {
        v["embedding"] = json!(sh.embedding.0);
    }
    v
}

fn media_of(s: &Session, cmd: &str, item: ItemId) -> Result<ItemId> {
    s.project.resolve_media(item).map(|(root, _, _)| root).ok_or_else(|| bad(cmd, format!("item {} is not media", item.0)))
}

fn show(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.scenes";
    let item = u64_p(p, "item").map(ItemId).ok_or_else(|| bad(cmd, "`item` is required"))?;
    let item = media_of(s, cmd, item)?;
    let name = s.project.item(item).map(|i| i.name.clone()).unwrap_or_default();
    let Some(idx) = s.project.scenes.get(&item) else {
        return Ok(json!({"item": item.0, "name": name, "analyzed": false, "shots": []}));
    };
    let idx = SceneIndex::clone(idx).sanitized();
    let emb = bool_p(p, "embeddings").unwrap_or(false);
    Ok(json!({
        "item": item.0,
        "name": name,
        "analyzed": true,
        "version": idx.version,
        "boundaries": idx.boundaries,
        "describer": idx.describer,
        "embedder": idx.embedder,
        "aesthetic": idx.aesthetic,
        "detail": idx.detail,
        "duration": secs(idx.duration),
        "created": idx.created,
        "shots": idx.shots.iter().enumerate().map(|(i, sh)| shot_json(i, sh, emb)).collect::<Vec<_>>(),
    }))
}

/// The indexed items to look in: `items` (media), else every indexed item.
fn indexed(s: &Session, cmd: &str, p: &Value) -> Result<Vec<(ItemId, SceneIndex)>> {
    let want: Option<Vec<ItemId>> = ids_p(cmd, p)?.map(|v| v.into_iter().filter_map(|i| s.project.resolve_media(i).map(|(r, _, _)| r)).collect());
    Ok(s.project
        .scenes
        .iter()
        .filter(|(i, _)| want.as_ref().is_none_or(|w| w.contains(i)) && s.project.item(**i).is_some())
        .map(|(i, x)| (*i, SceneIndex::clone(x).sanitized()))
        .collect())
}

/// The words spoken in `[a, b)` of `item`'s media, lower case.
fn spoken(s: &Session, item: ItemId, a: Tick, b: Tick) -> String {
    let Some(t) = s.project.transcripts.get(&item) else { return String::new() };
    t.words.iter().filter(|w| w.end > a && w.start < b).map(|w| w.normalized()).collect::<Vec<_>>().join(" ")
}

fn hit_json(s: &Session, idx: &HashMap<ItemId, SceneIndex>, item: u64, shot: usize, extra: Value) -> Value {
    let id = ItemId(item);
    let name = s.project.item(id).map(|i| i.name.clone()).unwrap_or_default();
    let Some(sh) = idx.get(&id).and_then(|x| x.shots.get(shot)) else { return json!({"item": item, "shot": shot}) };
    let mut v = json!({
        "item": item,
        "name": name,
        "shot": shot,
        "start": secs(sh.start),
        "end": secs(sh.end),
        "keyframe": secs(sh.keyframes.first().copied().unwrap_or_else(|| sh.middle())),
        "caption": sh.description.as_ref().map(|d| d.caption.clone()).unwrap_or_default(),
        "camera": sh.camera,
        "quality": sh.quality.score,
    });
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        for (k, x) in e {
            o.insert(k.clone(), x.clone());
        }
    }
    v
}

fn search_scenes(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.searchScenes";
    let query = str_p(p, "query").map(str::trim).filter(|q| !q.is_empty()).ok_or_else(|| bad(cmd, "`query` is required"))?;
    let query: String = query.chars().take(500).collect();
    let limit = u64_p(p, "limit").unwrap_or(20).clamp(1, 500) as usize;
    let list = indexed(s, cmd, p)?;
    if list.is_empty() {
        return Ok(json!({"query": query, "hits": [], "note": "no media is indexed yet: run media.analyzeScenes"}));
    }
    // the query's embedding, when the shots have embeddings and the model is here
    let any_emb = list.iter().any(|(_, x)| x.shots.iter().any(|sh| !sh.embedding.is_empty()));
    let mut note = None;
    let embedded = if any_emb {
        match s.scenes.models() {
            Some(m) => match m.embed_text(&query) {
                Ok(v) => Some(v),
                Err(e) => {
                    note = Some(format!("words only: {e}"));
                    None
                }
            },
            None => {
                note = Some("words only: this build has no embedding model".to_string());
                None
            }
        }
    } else {
        None
    };
    let speech: Vec<Vec<String>> = list.iter().map(|(i, x)| x.shots.iter().map(|sh| spoken(s, *i, sh.start, sh.end)).collect()).collect();
    let mut cands = Vec::new();
    for ((item, x), words) in list.iter().zip(&speech) {
        for (k, (sh, w)) in x.shots.iter().zip(words).enumerate() {
            cands.push(search::Candidate { item: item.0, shot: k, data: sh, speech: w.clone() });
        }
    }
    let (emb, cal) = match &embedded {
        Some((e, c)) => (Some(e.as_slice()), *c),
        None => (None, None),
    };
    let hits = search::rank(&query, emb, cal, &cands, limit);
    let map: HashMap<ItemId, SceneIndex> = list.into_iter().collect();
    let hits: Vec<Value> =
        hits.iter().map(|h| hit_json(s, &map, h.item, h.shot, json!({"score": h.score, "visual": h.visual, "text": h.text, "speech": h.speech}))).collect();
    let mut out = json!({"query": query, "hits": hits, "visual": embedded.is_some()});
    if let Some(n) = note {
        out["note"] = json!(n);
    }
    Ok(out)
}

fn best_shots(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.bestShots";
    let purpose = match str_p(p, "purpose") {
        None => ShotUse::Hook,
        Some(w) => ShotUse::from_words(w).ok_or_else(|| bad(cmd, "`purpose` is hook, broll, reveal, talking-head or establishing"))?,
    };
    let count = u64_p(p, "count").unwrap_or(5).clamp(1, 200) as usize;
    let list = indexed(s, cmd, p)?;
    if list.is_empty() {
        return Ok(json!({"purpose": purpose.name(), "shots": [], "note": "no media is indexed yet: run media.analyzeScenes"}));
    }
    let entries: Vec<best::Entry> =
        list.iter().flat_map(|(i, x)| x.shots.iter().enumerate().map(move |(k, sh)| best::Entry { item: i.0, shot: k, data: sh })).collect();
    let picks = best::best_shots(&entries, purpose, count);
    let map: HashMap<ItemId, SceneIndex> = list.iter().map(|(i, x)| (*i, x.clone())).collect();
    let shots: Vec<Value> = picks.iter().map(|pk| hit_json(s, &map, pk.item, pk.shot, json!({"score": pk.score, "why": pk.why}))).collect();
    Ok(json!({"purpose": purpose.name(), "shots": shots}))
}

// ---------------------------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------------------------

fn status(s: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_scenes::catalog;
    let dir = crate::transcript::models_dir();
    let models: Vec<Value> = catalog::catalogue()
        .iter()
        .map(|m| {
            json!({
                "id": m.id, "name": m.name, "role": m.role, "description": m.description,
                "license": m.license, "licenseUrl": m.license_url, "author": m.author, "source": m.source,
                "bytes": m.size(),
                "missingBytes": dir.as_deref().map_or(m.size(), |d| catalog::missing_bytes(d, m)),
                "installed": dir.as_deref().is_some_and(|d| catalog::installed(d, m)),
            })
        })
        .collect();
    let missing =
        |ids: &[&str]| -> u64 { ids.iter().filter_map(|i| catalog::find(i)).map(|m| dir.as_deref().map_or(m.size(), |d| catalog::missing_bytes(d, m))).sum() };
    Ok(json!({
        "available": filmcraft_scenes::available() || s.scenes.models.is_some(),
        "models": models,
        "fast": {"models": catalog::FAST, "missingBytes": missing(catalog::FAST)},
        "full": {"models": catalog::FULL, "missingBytes": missing(catalog::FULL)},
        "cache": s.scenes.cache.is_some(),
    }))
}

/// The model ids `model` names: one id, or the `fast` / `full` set.
fn model_ids(cmd: &str, p: &Value) -> Result<Vec<&'static str>> {
    use filmcraft_scenes::catalog;
    match str_p(p, "model").map(str::trim).unwrap_or("fast") {
        "fast" => Ok(catalog::FAST.to_vec()),
        "full" => Ok(catalog::FULL.to_vec()),
        id => catalog::find(id).map(|m| vec![m.id]).ok_or_else(|| {
            let ids: Vec<&str> = catalog::catalogue().iter().map(|m| m.id).collect();
            bad(cmd, format!("unknown model; use fast, full or one of {}", ids.join(", ")))
        }),
    }
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_scenes::catalog;
    let cmd = DOWNLOAD;
    let ids = model_ids(cmd, p)?;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    let todo: Vec<&'static catalog::ModelInfo> = ids.iter().filter_map(|i| catalog::find(i)).filter(|m| !catalog::installed(&dir, m)).collect();
    if todo.is_empty() {
        return Ok(json!({"job": null, "installed": true, "models": ids}));
    }
    if !cfg!(feature = "scene-models") {
        return Err(bad(cmd, "the scene models are not available in this build (built without the `scene-models` feature)"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let bytes: u64 = todo.iter().map(|m| catalog::missing_bytes(&dir, m)).sum();
    let names: Vec<&str> = todo.iter().map(|m| m.id).collect();
    let job = crate::narration::new_job(s, format!("Downloading {}", names.join(", ")));
    job.progress.total.store(bytes, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-scenes-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_models(&dir, &todo, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Scene models installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "models": names, "bytes": bytes}))
}

#[cfg(feature = "scene-models")]
fn download_models(
    dir: &std::path::Path,
    todo: &[&'static filmcraft_scenes::catalog::ModelInfo],
    prog: &filmcraft_export::Progress,
) -> std::result::Result<(), String> {
    let mut base = 0u64;
    for m in todo {
        let files: Vec<filmcraft_speech::models::ModelFile> =
            m.files.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
        let before = base;
        let mut progress = |done: u64, _total: u64, file: &str| {
            prog.done.store(before + done, Ordering::Relaxed);
            *lock(&prog.status) = format!("Downloading {} ({file})", m.id);
            !prog.cancel.load(Ordering::Relaxed)
        };
        let missing = filmcraft_scenes::catalog::missing_bytes(dir, m);
        filmcraft_speech::models::download_files(&filmcraft_scenes::catalog::model_dir(dir, m), &files, &mut progress).map_err(|e| format!("{}: {e}", m.id))?;
        base += missing;
    }
    Ok(())
}

#[cfg(not(feature = "scene-models"))]
fn download_models(
    _: &std::path::Path,
    _: &[&'static filmcraft_scenes::catalog::ModelInfo],
    _: &filmcraft_export::Progress,
) -> std::result::Result<(), String> {
    Err("the scene models are not available in this build".into())
}

#[cfg(test)]
#[path = "scenes_tests.rs"]
mod tests;
