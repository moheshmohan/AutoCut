//! Smooth Slow Motion: frames made between a video's frames by RIFE 4.26 (MIT, downloaded on the
//! user's consent; `filmcraft_vision::interp`, on the GPU), written once as a copy with two to
//! eight times as many frames and shown in the media's place, so a slowed clip moves smoothly
//! instead of repeating (or blending) frames.
//!
//! | command | does |
//! |---|---|
//! | `interpolate.status` | whether the network is downloaded (its name, size, licence and source) |
//! | `interpolate.downloadModel` | job: download it (22.9 MB), pinned and checked |
//! | `media.interpolate` | job: for the clips asked for (`clips`, default the selected ones: the part of their media they play, a few frames either side) or whole items (`items`), a copy with `factor` frames for each of the media's — 2 to 8, or `auto` (the default: as many as the slowest clip needs for the sequence's frame rate; clips that play at their own rate or faster are left out) — an H.264 (`format: "prores"`: ProRes 422 HQ) beside the media (or in `folder`); when written the item shows it in its range, one undo step; `off` shows the media as it is |
//!
//! **In its place.** The copy is the item's Interpretation (`interpolated`, with the media time
//! range it covers): the item keeps its frame rate; the media pool gives the copy's frames inside
//! the range ([`crate::media_pool::FinerSource`]) and the renderer times a clip's frames at the
//! copy's finer rate (`MediaSource::time_detail`): Frame Sampling shows the made frame nearest
//! each moment, Frame Blending and Optical Flow work between the made frames.
//!
//! **Limits.** A copy is at most 240 frames a second and [`MAX_MADE`] made frames (about an hour
//! of work); a whole long recording is refused for that — choose the clips. 360° media and image
//! sequences are left out, and so are items with a sharper copy (Upscale and Restore): one copy
//! of an item at a time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_media::{FrameRequest, MediaInfo, MediaKind, MediaSource, SharedSource};
use filmcraft_project::{ClipId, Interpolated, ItemId, ItemKind, MediaClip};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{EngineError, Result, Session};

pub const STATUS_CMD: &str = "interpolate.status";
pub const DOWNLOAD_CMD: &str = "interpolate.downloadModel";
pub const CMD: &str = "media.interpolate";
/// The most frames a copy has for each of the media's, and its highest rate.
pub const MAX_FACTOR: u32 = 8;
pub const MAX_RATE: f64 = 240.0;
/// The most frames one job makes (about an hour at 0.2 s a 1080p frame).
pub const MAX_MADE: u64 = 18_000;
/// Frames kept either side of the clips' range.
const MARGIN: i64 = 3;
/// Items in one job, at most.
const MAX_ITEMS: usize = 64;
const MAX_OUTCOMES: usize = 32;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: STATUS_CMD,
            label: "Frame Interpolation Model",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: status,
            journal: false,
        },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download the Frame Interpolation Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: always,
            run: download,
            journal: false,
        },
        CommandSpec {
            id: CMD,
            label: "Smooth Slow Motion",
            menu: &["Clip", "Modify"],
            shortcut: None,
            params: r#"{"clips":[id]?,"items":[id]?,"factor":"auto"|2..8="auto","format":"h264"|"prores"="h264","folder":str?,"off":bool=false,"wait":bool=false}"#,
            enabled: always,
            run: interpolate,
            journal: true,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// the network
// ---------------------------------------------------------------------------------------------

/// Makes the RGBA frame at `t` (0..1) between two (sRGB, `w` × `h`): the network, or a stand-in.
pub trait Tweener: Send + Sync {
    fn between(&self, a: &[u8], b: &[u8], w: usize, h: usize, t: f32) -> std::result::Result<Vec<u8>, String>;
}

static STAND_IN: Mutex<Option<Arc<dyn Tweener>>> = Mutex::new(None);

/// Use `t` instead of the downloaded network (tests and tools).
pub fn use_stand_in(t: Arc<dyn Tweener>) {
    *lock(&STAND_IN) = Some(t);
}

/// The network is downloaded (and this build can run it).
pub fn installed() -> bool {
    cfg!(feature = "privacy-models") && crate::transcript::models_dir().as_deref().is_some_and(filmcraft_vision::catalog::interpolator_installed)
}

#[cfg(feature = "privacy-models")]
struct Network(filmcraft_vision::interp::Interpolator);

#[cfg(feature = "privacy-models")]
impl Tweener for Network {
    fn between(&self, a: &[u8], b: &[u8], w: usize, h: usize, t: f32) -> std::result::Result<Vec<u8>, String> {
        self.0.between(a, b, w, h, t).map_err(|e| e.to_string())
    }
}

fn tweener(cmd: &str) -> Result<Arc<dyn Tweener>> {
    if let Some(t) = lock(&STAND_IN).clone() {
        return Ok(t);
    }
    network(cmd)
}

#[cfg(feature = "privacy-models")]
fn network(cmd: &str) -> Result<Arc<dyn Tweener>> {
    static NET: Mutex<Option<Arc<dyn Tweener>>> = Mutex::new(None);
    let mut slot = lock(&NET);
    if let Some(n) = slot.as_ref() {
        return Ok(n.clone());
    }
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    let net = filmcraft_vision::interp::Interpolator::load(&dir, true).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => bad(cmd, format!("RIFE is not downloaded yet ({DOWNLOAD_CMD}, 22.9 MB)")),
        e => bad(cmd, e.to_string()),
    })?;
    log::info!("RIFE loaded on the {}", net.device_name());
    let n: Arc<dyn Tweener> = Arc::new(Network(net));
    *slot = Some(n.clone());
    Ok(n)
}

#[cfg(not(feature = "privacy-models"))]
fn network(cmd: &str) -> Result<Arc<dyn Tweener>> {
    Err(bad(cmd, "frame interpolation is not available in this build (built without the `privacy-models` feature)"))
}

// ---------------------------------------------------------------------------------------------
// the copy
// ---------------------------------------------------------------------------------------------

/// The frames of a media range `factor` times as often, as they are asked for (for the
/// transcode that writes the copy): frame `j` of the copy is the media's frame `j / factor`, or
/// the frame `(j % factor) / factor` of the way to the next one. The last pair of media frames
/// decoded is kept for the frames between them. No lock is held while a frame decodes or the
/// network runs: the export renders frames on rayon's threads, and a decoder that works in
/// parallel lets its thread take another frame meanwhile (the work-stealing deadlock of #480);
/// the GPU's passes keep to one frame at a time themselves.
struct Between {
    inner: SharedSource,
    info: MediaInfo,
    net: Arc<dyn Tweener>,
    factor: u32,
    /// The media's rate and the range's first and last media frames.
    rate: FrameRate,
    first: i64,
    last: i64,
    pair: Mutex<Option<Pair>>,
    error: Arc<Mutex<Option<String>>>,
}

/// Media frame `i` and the next, decoded (RGBA, `w` × `h`).
#[derive(Clone)]
struct Pair {
    i: i64,
    a: Arc<Vec<u8>>,
    b: Arc<Vec<u8>>,
    w: usize,
    h: usize,
}

impl Between {
    fn decoded(&self, frame: i64) -> filmcraft_media::Result<(Vec<u8>, usize, usize)> {
        let f = self.inner.video_frame(FrameRequest { time: self.rate.tick_of(frame), scale: 1.0 })?;
        Ok((f.to_rgba8(), f.width as usize, f.height as usize))
    }

    /// Media frames `i` and `i + 1` (the kept pair, else decoded and kept).
    fn pair(&self, i: i64) -> filmcraft_media::Result<Pair> {
        if let Some(p) = lock(&self.pair).as_ref().filter(|p| p.i == i) {
            return Ok(p.clone());
        }
        let (a, w, h) = self.decoded(i)?;
        let b = if i < self.last { self.decoded(i + 1)?.0 } else { a.clone() };
        let p = Pair { i, a: Arc::new(a), b: Arc::new(b), w, h };
        *lock(&self.pair) = Some(p.clone());
        Ok(p)
    }
}

impl MediaSource for Between {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        let fine = self.info.frame_rate();
        let j = fine.frame_at(req.time).max(0);
        let k = i64::from(self.factor.max(1));
        let i = (self.first + j / k).min(self.last);
        let sub = j % k;
        let p = self.pair(i)?;
        let px = if sub == 0 || i >= self.last || p.b.len() != p.a.len() {
            p.a.as_ref().clone()
        } else {
            match self.net.between(&p.a, &p.b, p.w, p.h, sub as f32 / k as f32) {
                Ok(px) => px,
                Err(e) => {
                    *lock(&self.error) = Some(e.clone());
                    return Err(filmcraft_media::MediaError::Decode(e));
                }
            }
        };
        Ok(Arc::new(filmcraft_frame::VideoFrame::rgba8(p.w as u32, p.h as u32, px)))
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }
}

/// One copy to make: the item, its media, the range (media frames `first..=last`) and the factor.
struct Work {
    item: ItemId,
    clip: MediaClip,
    name: String,
    src: SharedSource,
    first: i64,
    last: i64,
    factor: u32,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct Made {
    item: ItemId,
    path: String,
    factor: u32,
    start: Tick,
    duration: Tick,
    seconds: f64,
}

/// Write the copy of `w`'s range through the export (no sound).
fn write_copy(w: &Work, net: Arc<dyn Tweener>, prores: bool, prog: &filmcraft_export::Progress) -> std::result::Result<(Tick, Tick), String> {
    use filmcraft_project::{Project, SequenceSettings, TrackKind};
    use filmcraft_time::TimeRange;
    let rate = w.clip.frame_rate();
    let k = i64::from(w.factor);
    let fine = FrameRate { num: rate.num.saturating_mul(k), den: rate.den };
    let frames = (w.last - w.first + 1).max(1);
    let (start, end) = (rate.tick_of(w.first), rate.tick_of(w.last + 1));
    let duration = end - start;
    let mut clip = w.clip.clone();
    clip.offline = false;
    clip.proxy = None;
    clip.interpret = Default::default();
    clip.info.duration = duration;
    if let Some(v) = clip.info.video.as_mut() {
        v.frame_rate = fine;
    }
    let (vw, vh) = clip.info.video.as_ref().map_or((1920, 1080), |v| (v.width, v.height));
    let st = SequenceSettings { width: vw, height: vh, par: clip.pixel_aspect(), frame_rate: fine, ..Default::default() };
    let mut p = Project::new("interpolate");
    let info = clip.info.clone();
    let item = p.add_item(&w.name, filmcraft_project::Label::Iris, ItemKind::Media(clip), None);
    let seq = p.new_sequence("interpolate", st, 1, 0, None);
    let range = TimeRange { start: Tick::ZERO, duration: fine.tick_of(frames * k) };
    let ti = p.make_track_item(item, TrackKind::Video, Tick::ZERO, range, fine).ok_or("the clip does not place")?;
    p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()).ok_or("no sequence")?.items.push(ti);
    let error = Arc::new(Mutex::new(None));
    let src: SharedSource = Arc::new(Between {
        inner: w.src.clone(),
        info,
        net,
        factor: w.factor,
        rate,
        first: w.first,
        last: w.last,
        pair: Mutex::new(None),
        error: error.clone(),
    });
    let pixels = f64::from(vw) * f64::from(vh) * fine.as_f64() / 30.0;
    let settings = filmcraft_export::ExportSettings {
        format: if prores { filmcraft_export::Format::ProRes } else { filmcraft_export::Format::H264 },
        path: w.out.to_string_lossy().into_owned(),
        scale: 1.0,
        include_audio: false,
        quality: 95,
        bitrate_kbps: (pixels / 40.0).clamp(8_000.0, 200_000.0) as u32,
        part_of_batch: true,
        prores_profile: "hq".into(),
        ..Default::default()
    };
    let project = Arc::new(p);
    let provider = move |id: ItemId| (id == item).then(|| src.clone());
    let r = filmcraft_export::export(&project, seq, &settings, &provider, prog).map_err(|e| e.to_string());
    if let Some(e) = lock(&error).take() {
        let _ = std::fs::remove_file(&w.out);
        return Err(format!("{}: {e}", w.name));
    }
    r.map(|_| (start, duration))
}

// ---------------------------------------------------------------------------------------------
// the commands
// ---------------------------------------------------------------------------------------------

/// Jobs whose copies are attached when they finish, and what finished ones made.
#[derive(Default)]
pub struct Interpolations {
    pending: Vec<Pending>,
    outcomes: BTreeMap<u64, Value>,
}

struct Pending {
    job: u64,
    out: Arc<Mutex<Option<std::result::Result<Vec<Made>, String>>>>,
}

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_vision::catalog as c;
    Ok(json!({
        "available": cfg!(feature = "privacy-models"),
        "installed": installed(),
        "name": c::INTERP_NAME,
        "bytes": c::INTERPOLATOR.size,
        "license": c::INTERP_LICENSE,
        "source": c::INTERP_SOURCE,
        "download": DOWNLOAD_CMD,
    }))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::interpolator_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "frame interpolation is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::INTERP_NAME));
    job.progress.total.store(filmcraft_vision::catalog::INTERPOLATOR.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-interpolate-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "RIFE installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::INTERPOLATOR.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::INTERP_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::interp_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("frame interpolation is not available in this build".into())
}

/// How many frames a clip needs for each of its media's: the sequence's rate over the media's,
/// over the clip's slowest speed (sampled along it, for time remapping).
fn need(it: &filmcraft_project::TrackItem, media: FrameRate, seq: FrameRate) -> f64 {
    let d = it.duration.0.max(1);
    let n = 32i64;
    let mut slowest = f64::INFINITY;
    for i in 0..n {
        let (a, b) = (it.start + Tick(d * i / n), it.start + Tick(d * (i + 1) / n));
        let dt = (b - a).0 as f64;
        let ds = (it.source_time_at(b) - it.source_time_at(a)).0.unsigned_abs() as f64;
        if dt > 0.0 {
            slowest = slowest.min(ds / dt);
        }
    }
    let speed = if slowest.is_finite() { slowest.max(1e-3) } else { 1.0 };
    seq.as_f64() / media.as_f64().max(1e-3) / speed
}

/// The factor for a need (`None`: it plays at its own rate or faster).
pub fn factor_for(need: f64, media_fps: f64) -> Option<u32> {
    if !(need.is_finite() && need > 1.05) {
        return None;
    }
    let most = (MAX_RATE / media_fps.max(1.0)).floor().clamp(2.0, f64::from(MAX_FACTOR)) as u32;
    Some((need - 0.05).ceil().clamp(2.0, f64::from(MAX_FACTOR)) as u32).map(|k| k.min(most))
}

fn interpolate(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let off = bool_p(p, "off").unwrap_or(false);
    // the clips asked for (or selected), else the items
    let clips: Vec<ClipId> = if p.get("items").is_some() { Vec::new() } else { crate::commands::clips_p(s, p) };
    let seq = s.active_sequence();
    let seq_rate = seq.map(|q| q.settings.frame_rate.sane());
    // item → (clips of it, their media range)
    let mut wanted: BTreeMap<ItemId, Vec<filmcraft_project::TrackItem>> = BTreeMap::new();
    if let Some(q) = seq {
        for c in &clips {
            if let Some((tid, it)) = q.find_item(*c)
                && q.track(tid).is_some_and(|t| t.kind == filmcraft_project::TrackKind::Video)
            {
                wanted.entry(it.item).or_default().push(it.clone());
            }
        }
    }
    let items: Vec<ItemId> = if wanted.is_empty() { crate::color::footage_targets(s, p) } else { wanted.keys().copied().collect() };
    let items: Vec<ItemId> = items.into_iter().take(MAX_ITEMS).collect();
    if items.is_empty() {
        return Err(bad(CMD, "select a video clip (or give `items`)"));
    }
    if off {
        let on: Vec<ItemId> =
            items.iter().copied().filter(|i| s.project.item(*i).and_then(|it| it.as_media()).is_some_and(|m| m.interpret.interpolated.is_some())).collect();
        if !on.is_empty() {
            s.edit("Smooth Slow Motion Off", |pr, _| {
                for i in &on {
                    if let Some(ItemKind::Media(m)) = pr.item_mut(*i).map(|it| &mut it.kind) {
                        m.interpret.interpolated = None;
                    }
                }
                Ok(())
            })?;
        }
        return Ok(json!({"off": on.iter().map(|i| i.0).collect::<Vec<_>>()}));
    }
    let asked = match p.get("factor") {
        None | Some(Value::Null) => None,
        Some(Value::String(a)) if a.trim().eq_ignore_ascii_case("auto") => None,
        Some(v) => match v.as_u64() {
            Some(k) if (2..=u64::from(MAX_FACTOR)).contains(&k) => Some(k as u32),
            _ => return Err(bad(CMD, format!("`factor` is auto or 2 to {MAX_FACTOR}"))),
        },
    };
    let prores = match str_p(p, "format") {
        None => false,
        Some(f) if f.eq_ignore_ascii_case("h264") => false,
        Some(f) if f.eq_ignore_ascii_case("prores") => true,
        Some(_) => return Err(bad(CMD, "`format` is h264 or prores")),
    };
    let folder = str_p(p, "folder").map(PathBuf::from);
    let wait = bool_p(p, "wait").unwrap_or(false);
    let (mut work, mut skipped) = (Vec::new(), Vec::new());
    for i in &items {
        let Some(pi) = s.project.item(*i) else { continue };
        let Some(m) = pi.as_media() else { continue };
        let name = pi.name.clone();
        let rate = m.frame_rate().sane();
        let why = if m.info.video.is_none() {
            Some("no picture".to_string())
        } else if crate::vr360::is_360(s, *i) {
            Some("360° media is seen through a virtual camera".to_string())
        } else if m.info.kind != MediaKind::Movie {
            Some("only videos".to_string())
        } else if m.interpret.upscaled.is_some() {
            Some("it has a sharper copy (Upscale and Restore): turn that off first".to_string())
        } else if rate.frame_duration().0 <= 0 {
            Some("no frame rate".to_string())
        } else {
            None
        };
        if let Some(why) = why {
            skipped.push(json!({"item": i.0, "why": why}));
            continue;
        }
        let total = rate.frame_at(m.info.duration).max(1);
        let its = wanted.get(i).cloned().unwrap_or_default();
        // the range: the clips' media, a few frames either side; else the whole media
        let (first, last) = if its.is_empty() {
            (0, total - 1)
        } else {
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            for it in &its {
                for k in 0..=32 {
                    let t = it.start + Tick(it.duration.0.max(1) * k / 32);
                    let f = rate.frame_at(it.source_time_at(t));
                    lo = lo.min(f);
                    hi = hi.max(f);
                }
            }
            ((lo - MARGIN).max(0), (hi + MARGIN).min(total - 1))
        };
        // how many frames for each: asked, else what the slowest clip needs (for the sequence's rate)
        let auto = || {
            let sr = seq_rate.unwrap_or(rate);
            let pool: Vec<filmcraft_project::TrackItem> = if its.is_empty() {
                seq.map(|q| q.video_tracks.iter().flat_map(|t| t.items.iter()).filter(|it| it.item == *i).cloned().collect()).unwrap_or_default()
            } else {
                its.clone()
            };
            pool.iter().map(|it| need(it, rate, sr)).fold(0.0, f64::max)
        };
        let k = match asked {
            Some(k) => Some(k.min((MAX_RATE / rate.as_f64().max(1.0)).floor().max(2.0) as u32)),
            None => factor_for(auto(), rate.as_f64()),
        };
        let Some(k) = k else {
            skipped.push(json!({"item": i.0, "why": "its clips play at its own rate or faster (give `factor` to make frames anyway)"}));
            continue;
        };
        let made = (last - first + 1).max(1) as u64 * u64::from(k - 1);
        if made > MAX_MADE {
            skipped.push(json!({"item": i.0, "why": format!("{made} frames to make, more than {MAX_MADE}: choose the clips that are slowed (a range of it)")}));
            continue;
        }
        let filmcraft_project::MediaRef::File { path } = &m.media else {
            skipped.push(json!({"item": i.0, "why": "not a file"}));
            continue;
        };
        let src = s.media.full_res_source(&s.project, *i, &*s.services).ok_or_else(|| bad(CMD, format!("{name}: the media does not open")))?;
        let media = Path::new(path);
        let dir = folder.clone().unwrap_or_else(|| media.parent().map(Path::to_path_buf).unwrap_or_default());
        let stem = media.file_stem().map_or_else(|| "video".to_string(), |s| s.to_string_lossy().into_owned());
        let out = crate::proxies::unique_path(&dir, &stem, &format!(" (×{k} frames)"), if prores { "mov" } else { "mp4" }, &[]);
        work.push(Work { item: *i, clip: m.clone(), name, src, first, last, factor: k, out });
    }
    if work.is_empty() {
        return Err(bad(CMD, format!("nothing to make: {}", skipped.iter().filter_map(|v| v["why"].as_str()).collect::<Vec<_>>().join("; "))));
    }
    let net = tweener(CMD)?;
    if let Some(dir) = work.iter().filter_map(|w| w.out.parent()).find(|d| std::fs::create_dir_all(d).is_err()) {
        return Err(bad(CMD, format!("{}: cannot write there", dir.display())));
    }
    let planned: Vec<Value> = work
        .iter()
        .map(|w| {
            let r = w.clip.frame_rate();
            json!({"item": w.item.0, "factor": w.factor, "path": w.out.to_string_lossy(), "from": r.tick_of(w.first).seconds(), "to": r.tick_of(w.last + 1).seconds(), "frames": (w.last - w.first + 1) * i64::from(w.factor)})
        })
        .collect();
    let job = crate::narration::new_job(s, format!("Smooth Slow Motion ({} item{})", work.len(), if work.len() == 1 { "" } else { "s" }));
    let frames: u64 = work.iter().map(|w| (w.last - w.first + 1).max(1) as u64 * u64::from(w.factor)).sum();
    job.progress.total.store(frames, Ordering::Relaxed);
    let out: Arc<Mutex<Option<std::result::Result<Vec<Made>, String>>>> = Arc::default();
    let (prog, res, made_out) = (job.progress.clone(), job.result.clone(), out.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-interpolate", move || {
        let t0 = web_time::Instant::now();
        let mut made = Vec::new();
        let mut failed = None;
        for w in &work {
            if prog.cancel.load(Ordering::Relaxed) {
                failed = Some("stopped".to_string());
                break;
            }
            let t = web_time::Instant::now();
            match write_copy(w, net.clone(), prores, &prog) {
                Ok((start, duration)) => made.push(Made {
                    item: w.item,
                    path: w.out.to_string_lossy().into_owned(),
                    factor: w.factor,
                    start,
                    duration,
                    seconds: t.elapsed().as_secs_f64(),
                }),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        let n = made.len();
        *lock(&made_out) = Some(match &failed {
            Some(e) if made.is_empty() => Err(e.clone()),
            _ => Ok(made),
        });
        let status = match failed {
            Some(e) if n == 0 => Err(e),
            Some(e) => Ok(format!("{n} made smooth; then {e}")),
            None => Ok(format!("{n} made smooth")),
        };
        crate::narration::finish(&prog, &res, status, t0);
    })?;
    s.interpolations.pending.push(Pending { job: id, out });
    if wait {
        poll(s);
        if let Some(o) = s.interpolations.outcomes.get(&id).filter(|o| o.get("error").is_some()) {
            return Err(EngineError::Other(format!("{CMD}: {}", o["error"].as_str().unwrap_or("failed"))));
        }
    }
    Ok(json!({"job": id, "items": planned, "skipped": skipped, "outcome": s.interpolations.outcomes.get(&id)}))
}

/// Attaches finished copies (one undo step a job). Called each UI frame, by `jobs.list` and after
/// a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.interpolations.pending.len() {
        let Some(pj) = s.interpolations.pending.get(i) else { break };
        let finished = s.jobs.iter().find(|j| j.id == pj.job).is_none_or(|j| j.progress.finished.load(std::sync::atomic::Ordering::Relaxed));
        if lock(&pj.out).is_none() && !finished {
            i += 1;
            continue;
        }
        let pj = s.interpolations.pending.remove(i);
        let outcome = match lock(&pj.out).take() {
            Some(Ok(made)) if !made.is_empty() => {
                let r = s.edit("Smooth Slow Motion", |pr, _| {
                    for m in &made {
                        if let Some(ItemKind::Media(c)) = pr.item_mut(m.item).map(|it| &mut it.kind) {
                            c.interpret.interpolated =
                                Some(Interpolated { path: m.path.clone(), factor: m.factor.min(255) as u8, start: m.start, duration: m.duration });
                        }
                    }
                    Ok(())
                });
                match r {
                    Ok(()) => {
                        json!({"made": made.iter().map(|m| json!({"item": m.item.0, "path": m.path, "factor": m.factor, "from": m.start.seconds(), "to": (m.start + m.duration).seconds(), "seconds": (m.seconds * 10.0).round() / 10.0})).collect::<Vec<_>>()})
                    }
                    Err(e) => json!({"error": e.to_string()}),
                }
            }
            Some(Ok(_)) => json!({"made": []}),
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "stopped: nothing was changed"}),
        };
        s.interpolations.outcomes.insert(pj.job, outcome);
        if s.interpolations.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.interpolations.outcomes.keys().min().copied()
        {
            s.interpolations.outcomes.remove(&old);
        }
    }
}

/// What a finished interpolation made (`interpolate` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.interpolations.outcomes.get(&job).cloned()
}

#[cfg(test)]
#[path = "interpolate_tests.rs"]
mod tests;
