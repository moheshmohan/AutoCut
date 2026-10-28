//! Upscale and Restore: a photo or a video made larger and cleaner by Real-ESRGAN (BSD-3-Clause,
//! downloaded on the user's consent; `filmcraft_vision::upscale`, on the GPU), written once
//! beside the media and shown in its place.
//!
//! | command | does |
//! |---|---|
//! | `upscale.status` | whether the network is downloaded (its name, size, licence and source) |
//! | `upscale.downloadModel` | job: download it (4.9 MB), pinned and checked |
//! | `media.upscale` | job: each item (`items`, default the selected clips' media) made `factor` times as large — 2, 3 or 4, 1 to restore it at its own size (blur, noise and compression's blocks taken out), or `auto` (the default: enough for the active sequence's frame, at most 4; items at least two thirds as large as it, or with a copy already, are left out) — with `strength` 0..1 of the network's answer (1); a photo's copy is a PNG, a video's an H.264 (`format: "prores"`: ProRes 422 HQ) beside the media (or in `folder`); when it is written the item shows it in its place, one undo step; `off` shows the media as it is again |
//!
//! **In its place.** The copy is the item's Interpretation (`upscaled`): the item keeps its size,
//! its clips their framing, masks and effects; the media pool gives its frames for the item's
//! ([`crate::media_pool::SharperSource`]) and the renderer asks it for as much detail as a clip
//! shows the picture with — up to the copy's size — so a 480p clip filling a 4K frame is drawn
//! from the copy at four times its size. The copy has no sound: the item's own plays.
//!
//! **Limits.** A photo's copy is at most 8192 pixels a side (the still decoder's cap) and the
//! network takes pictures of at most 4096 a side; a video's copy at most 7680 × 4320. 360° media
//! (seen through a virtual camera, already large) and image sequences are left out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_media::{FrameRequest, MediaInfo, MediaKind, MediaSource, SharedSource};
use filmcraft_project::{ItemId, ItemKind, MediaClip, Upscaled};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, str_p};
use crate::{EngineError, Result, Session};

pub const STATUS_CMD: &str = "upscale.status";
pub const DOWNLOAD_CMD: &str = "upscale.downloadModel";
pub const CMD: &str = "media.upscale";
/// The network's own factor.
pub const NET_FACTOR: u32 = 4;
/// The largest copy: a photo's long side, a video's frame.
pub const MAX_PHOTO_SIDE: u32 = filmcraft_media::still::MAX_STILL_SIDE;
pub const MAX_VIDEO: (u32, u32) = (7680, 4320);
/// The longest side of a picture the network takes.
pub const MAX_INPUT_SIDE: u32 = 4096;
/// The frame `auto` aims at without an active sequence (4K's long side).
const DEFAULT_TARGET: u32 = 3840;
/// Items in one job, at most.
const MAX_ITEMS: usize = 64;
/// Finished jobs' outcomes kept.
const MAX_OUTCOMES: usize = 32;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec { id: STATUS_CMD, label: "Upscale Model", menu: &[], shortcut: None, params: "{}", enabled: always, run: status, journal: false },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download the Upscale Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: always,
            run: download,
            journal: false,
        },
        CommandSpec {
            id: CMD,
            label: "Upscale and Restore",
            menu: &["Clip", "Modify"],
            shortcut: None,
            params: r#"{"items":[id]?,"factor":"auto"|1..4="auto","strength":0..1=1,"format":"h264"|"prores"="h264","folder":str?,"off":bool=false,"wait":bool=false}"#,
            enabled: always,
            run: upscale,
            journal: true,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// the network
// ---------------------------------------------------------------------------------------------

/// Makes RGBA pictures (sRGB, straight alpha) four times as large: the network, or a stand-in.
pub trait Enlarger: Send + Sync {
    /// `(rgba, 4w, 4h)`; `progress` hears tiles done and stops the work when it says false.
    fn enlarge(
        &self,
        rgba: &[u8],
        w: usize,
        h: usize,
        strength: f32,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> std::result::Result<(Vec<u8>, usize, usize), String>;
}

static STAND_IN: Mutex<Option<Arc<dyn Enlarger>>> = Mutex::new(None);

/// Use `e` instead of the downloaded network (tests and tools).
pub fn use_stand_in(e: Arc<dyn Enlarger>) {
    *lock(&STAND_IN) = Some(e);
}

/// The network is downloaded (and this build can run it).
pub fn installed() -> bool {
    cfg!(feature = "privacy-models") && crate::transcript::models_dir().as_deref().is_some_and(filmcraft_vision::catalog::upscaler_installed)
}

#[cfg(feature = "privacy-models")]
struct Network(filmcraft_vision::upscale::Upscaler);

#[cfg(feature = "privacy-models")]
impl Enlarger for Network {
    fn enlarge(
        &self,
        rgba: &[u8],
        w: usize,
        h: usize,
        strength: f32,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> std::result::Result<(Vec<u8>, usize, usize), String> {
        self.0.upscale(rgba, w, h, strength, progress).map_err(|e| e.to_string())
    }
}

/// The enlarger a job uses: the stand-in, else the network (loaded once).
fn enlarger(cmd: &str) -> Result<Arc<dyn Enlarger>> {
    if let Some(e) = lock(&STAND_IN).clone() {
        return Ok(e);
    }
    network(cmd)
}

#[cfg(feature = "privacy-models")]
fn network(cmd: &str) -> Result<Arc<dyn Enlarger>> {
    static NET: Mutex<Option<Arc<dyn Enlarger>>> = Mutex::new(None);
    let mut slot = lock(&NET);
    if let Some(n) = slot.as_ref() {
        return Ok(n.clone());
    }
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    let up = filmcraft_vision::upscale::Upscaler::load(&dir, true).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => bad(cmd, format!("Real-ESRGAN is not downloaded yet ({DOWNLOAD_CMD}, 4.9 MB)")),
        e => bad(cmd, e.to_string()),
    })?;
    log::info!("Real-ESRGAN loaded on the {}", up.device_name());
    let n: Arc<dyn Enlarger> = Arc::new(Network(up));
    *slot = Some(n.clone());
    Ok(n)
}

#[cfg(not(feature = "privacy-models"))]
fn network(cmd: &str) -> Result<Arc<dyn Enlarger>> {
    Err(bad(cmd, "upscaling is not available in this build (built without the `privacy-models` feature)"))
}

/// `rgba` (`w` × `h`) averaged down to `tw` × `th` (each target pixel the mean of the source
/// pixels whose centres fall in it).
pub fn area_resize(rgba: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    let (tw, th) = (tw.max(1), th.max(1));
    let mut out = vec![0u8; tw * th * 4];
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return out;
    }
    for y in 0..th {
        let (y0, y1) = (y * h / th, ((y + 1) * h / th).max(y * h / th + 1).min(h));
        for x in 0..tw {
            let (x0, x1) = (x * w / tw, ((x + 1) * w / tw).max(x * w / tw + 1).min(w));
            let mut acc = [0u32; 4];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += u32::from(rgba.get((yy * w + xx) * 4 + c).copied().unwrap_or(0));
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as u32;
            for c in 0..4 {
                out[(y * tw + x) * 4 + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

/// An RGBA picture `factor` times as large (1 to 4: the network's four times, averaged down).
pub fn enlarge_by(
    e: &dyn Enlarger,
    rgba: &[u8],
    w: usize,
    h: usize,
    factor: u32,
    strength: f32,
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    let (big, bw, bh) = e.enlarge(rgba, w, h, strength, progress)?;
    let k = factor.clamp(1, NET_FACTOR) as usize;
    if k == NET_FACTOR as usize {
        return Ok((big, bw, bh));
    }
    Ok((area_resize(&big, bw, bh, w * k, h * k), w * k, h * k))
}

// ---------------------------------------------------------------------------------------------
// the copies
// ---------------------------------------------------------------------------------------------

/// A video's frames made larger as they are asked for, for the transcode that writes its copy:
/// the original's frame at its own size, through the network (one at a time: the GPU is the
/// limit, and parallel frames would hold its memory many times over).
struct Enlarging {
    inner: SharedSource,
    info: MediaInfo,
    net: Arc<dyn Enlarger>,
    factor: u32,
    strength: f32,
    one: Mutex<()>,
    error: Arc<Mutex<Option<String>>>,
}

impl MediaSource for Enlarging {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        let f = self.inner.video_frame(FrameRequest { time: req.time, scale: 1.0 })?;
        let (w, h) = (f.width as usize, f.height as usize);
        let rgba = f.to_rgba8();
        let _one = lock(&self.one);
        match enlarge_by(&*self.net, &rgba, w, h, self.factor, self.strength, &mut |_, _| true) {
            Ok((px, ow, oh)) => Ok(Arc::new(filmcraft_frame::VideoFrame::rgba8(ow as u32, oh as u32, px).with_pts(f.pts))),
            Err(e) => {
                *lock(&self.error) = Some(e.clone());
                Err(filmcraft_media::MediaError::Decode(e))
            }
        }
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }
}

/// One item to make a copy of.
struct Work {
    item: ItemId,
    clip: MediaClip,
    name: String,
    src: SharedSource,
    size: (u32, u32),
    factor: u32,
    still: bool,
    out: PathBuf,
}

/// A copy made: the item, its file, how large.
#[derive(Clone, Debug)]
struct Made {
    item: ItemId,
    path: String,
    factor: u32,
    seconds: f64,
}

/// A photo's copy: decoded at its own size, enlarged, written as a PNG.
fn photo_copy(w: &Work, net: &dyn Enlarger, strength: f32, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let f = w.src.video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).map_err(|e| format!("{}: {e}", w.name))?;
    let (iw, ih) = (f.width as usize, f.height as usize);
    let rgba = f.to_rgba8();
    let mut progress = |done: usize, of: usize| {
        *lock(&prog.status) = format!("Upscaling {} ({done}/{of})", w.name);
        !prog.cancel.load(Ordering::Relaxed)
    };
    let (px, ow, oh) = enlarge_by(net, &rgba, iw, ih, w.factor, strength, &mut progress)?;
    let png = filmcraft_export::encode_png(px, ow as u32, oh as u32).map_err(|e| e.to_string())?;
    write_new(&w.out, &png)
}

/// Write `bytes` to `path` through a temporary name, so no half-written copy is ever shown.
fn write_new(path: &Path, bytes: &[u8]) -> std::result::Result<(), String> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// A video's copy: its frames through the network, encoded at `factor` times its size (no sound).
fn video_copy(w: &Work, net: Arc<dyn Enlarger>, strength: f32, prores: bool, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use filmcraft_project::{Project, SequenceSettings, TrackKind};
    use filmcraft_time::TimeRange;
    let (ow, oh) = (w.size.0 * w.factor, w.size.1 * w.factor);
    let mut clip = w.clip.clone();
    clip.offline = false;
    clip.proxy = None;
    clip.interpret.upscaled = None;
    clip.interpret.denoise = None;
    if let Some(v) = clip.info.video.as_mut() {
        v.width = ow;
        v.height = oh;
    }
    let rate = clip.frame_rate();
    let st = SequenceSettings { width: ow, height: oh, par: clip.pixel_aspect(), frame_rate: rate, ..Default::default() };
    let mut p = Project::new("upscale");
    let info = clip.info.clone();
    let item = p.add_item(&w.name, filmcraft_project::Label::Iris, ItemKind::Media(clip), None);
    let seq = p.new_sequence("upscale", st, 1, 0, None);
    let range = TimeRange { start: Tick::ZERO, duration: info.duration };
    let ti = p.make_track_item(item, TrackKind::Video, Tick::ZERO, range, rate).ok_or("the clip does not place")?;
    p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()).ok_or("no sequence")?.items.push(ti);
    let error = Arc::new(Mutex::new(None));
    let src: SharedSource = Arc::new(Enlarging { inner: w.src.clone(), info, net, factor: w.factor, strength, one: Mutex::new(()), error: error.clone() });
    let pixels = f64::from(ow) * f64::from(oh);
    let settings = filmcraft_export::ExportSettings {
        format: if prores { filmcraft_export::Format::ProRes } else { filmcraft_export::Format::H264 },
        path: w.out.to_string_lossy().into_owned(),
        scale: 1.0,
        include_audio: false,
        quality: 95,
        bitrate_kbps: (pixels / 40.0).clamp(8_000.0, 160_000.0) as u32,
        part_of_batch: true,
        prores_profile: "hq".into(),
        ..Default::default()
    };
    let project = Arc::new(p);
    let provider = move |id: ItemId| (id == item).then(|| src.clone());
    let r = filmcraft_export::export(&project, seq, &settings, &provider, prog).map(|_| ()).map_err(|e| e.to_string());
    match lock(&error).take() {
        Some(e) => {
            let _ = std::fs::remove_file(&w.out);
            Err(format!("{}: {e}", w.name))
        }
        None => r,
    }
}

// ---------------------------------------------------------------------------------------------
// the commands
// ---------------------------------------------------------------------------------------------

/// Jobs whose copies are attached when they finish, and what finished ones made.
#[derive(Default)]
pub struct Upscales {
    pending: Vec<Pending>,
    outcomes: BTreeMap<u64, Value>,
}

struct Pending {
    job: u64,
    strength: f32,
    out: Arc<Mutex<Option<std::result::Result<Vec<Made>, String>>>>,
}

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_vision::catalog as c;
    Ok(json!({
        "available": cfg!(feature = "privacy-models"),
        "installed": installed(),
        "name": c::UPSCALE_NAME,
        "bytes": c::UPSCALER.size,
        "license": c::UPSCALE_LICENSE,
        "source": c::UPSCALE_SOURCE,
        "download": DOWNLOAD_CMD,
    }))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::upscaler_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "upscaling is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::UPSCALE_NAME));
    job.progress.total.store(filmcraft_vision::catalog::UPSCALER.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-upscale-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Real-ESRGAN installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::UPSCALER.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::UPSCALE_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::upscale_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("upscaling is not available in this build".into())
}

/// The factor `auto` gives a `w` × `h` picture for a frame whose long side is `target`: enough to
/// reach it, at most the network's; `None` when the picture is at least two thirds as large (it
/// is not visibly soft there).
pub fn auto_factor(w: u32, h: u32, target: u32) -> Option<u32> {
    let long = w.max(h).max(1);
    (u64::from(long) * 3 < u64::from(target) * 2).then(|| target.div_ceil(long).clamp(2, NET_FACTOR))
}

/// The largest factor a `w` × `h` picture's copy may have (photo or video), at most `asked`.
fn capped(w: u32, h: u32, still: bool, asked: u32) -> u32 {
    let most = if still { MAX_PHOTO_SIDE / w.max(h).max(1) } else { (MAX_VIDEO.0 / w.max(1)).min(MAX_VIDEO.1 / h.max(1)) };
    asked.min(most)
}

fn upscale(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let items: Vec<ItemId> = crate::color::footage_targets(s, p).into_iter().take(MAX_ITEMS).collect();
    if items.is_empty() {
        return Err(bad(CMD, "select a photo or a video (`items`)"));
    }
    if bool_p(p, "off").unwrap_or(false) {
        let on: Vec<ItemId> =
            items.iter().copied().filter(|i| s.project.item(*i).and_then(|it| it.as_media()).is_some_and(|m| m.interpret.upscaled.is_some())).collect();
        if !on.is_empty() {
            s.edit("Upscale and Restore Off", |pr, _| {
                for i in &on {
                    if let Some(ItemKind::Media(m)) = pr.item_mut(*i).map(|it| &mut it.kind) {
                        m.interpret.upscaled = None;
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
            Some(k) if (1..=u64::from(NET_FACTOR)).contains(&k) => Some(k as u32),
            _ => return Err(bad(CMD, "`factor` is auto, or 1 (restore at its own size) to 4")),
        },
    };
    let strength = match f64_p(p, "strength") {
        None => 1.0,
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) => v as f32,
        Some(_) => return Err(bad(CMD, "`strength` is 0 to 1")),
    };
    let prores = match str_p(p, "format") {
        None => false,
        Some(f) if f.eq_ignore_ascii_case("h264") => false,
        Some(f) if f.eq_ignore_ascii_case("prores") => true,
        Some(_) => return Err(bad(CMD, "`format` is h264 or prores")),
    };
    let folder = str_p(p, "folder").map(PathBuf::from);
    let wait = bool_p(p, "wait").unwrap_or(false);
    let target = s.active_sequence().map_or(DEFAULT_TARGET, |q| q.settings.width.max(q.settings.height));
    let (mut work, mut skipped) = (Vec::new(), Vec::new());
    for i in &items {
        let Some(pi) = s.project.item(*i) else { continue };
        let Some(m) = pi.as_media() else { continue };
        let name = pi.name.clone();
        let Some(v) = m.info.video.as_ref() else {
            skipped.push(json!({"item": i.0, "why": "no picture"}));
            continue;
        };
        let (w, h) = (v.width, v.height);
        let still = m.info.kind == MediaKind::Still;
        let why = if crate::vr360::is_360(s, *i) {
            Some("360° media is seen through a virtual camera".to_string())
        } else if !matches!(m.info.kind, MediaKind::Still | MediaKind::Movie) {
            Some("only photos and videos".to_string())
        } else if w.max(h) > MAX_INPUT_SIDE {
            Some(format!("larger than {MAX_INPUT_SIDE} pixels a side"))
        } else {
            None
        };
        if let Some(why) = why {
            skipped.push(json!({"item": i.0, "why": why}));
            continue;
        }
        if asked.is_none() && m.interpret.upscaled.is_some() {
            skipped.push(json!({"item": i.0, "why": "it has a copy already"}));
            continue;
        }
        let Some(k) = asked.or_else(|| auto_factor(w, h, target)) else {
            skipped.push(json!({"item": i.0, "why": format!("already {w} × {h}, nearly as large as the frame")}));
            continue;
        };
        let k = capped(w, h, still, k);
        if k == 0 {
            skipped.push(json!({"item": i.0, "why": "its copy would be larger than the largest allowed"}));
            continue;
        }
        let filmcraft_project::MediaRef::File { path } = &m.media else {
            skipped.push(json!({"item": i.0, "why": "not a file"}));
            continue;
        };
        let src = s.media.full_res_source(&s.project, *i, &*s.services).ok_or_else(|| bad(CMD, format!("{name}: the media does not open")))?;
        let media = Path::new(path);
        let dir = folder.clone().unwrap_or_else(|| media.parent().map(Path::to_path_buf).unwrap_or_default());
        let stem = media.file_stem().map_or_else(|| "picture".to_string(), |s| s.to_string_lossy().into_owned());
        let suffix = if k == 1 { " (restored)".to_string() } else { format!(" (upscaled ×{k})") };
        let out = crate::proxies::unique_path(
            &dir,
            &stem,
            &suffix,
            if still {
                "png"
            } else if prores {
                "mov"
            } else {
                "mp4"
            },
            &[],
        );
        work.push(Work { item: *i, clip: m.clone(), name, src, size: (w, h), factor: k, still, out });
    }
    if work.is_empty() {
        return Err(bad(CMD, format!("nothing to upscale: {}", skipped.iter().filter_map(|v| v["why"].as_str()).collect::<Vec<_>>().join("; "))));
    }
    let net = enlarger(CMD)?;
    if let Some(dir) = work.iter().filter_map(|w| w.out.parent()).find(|d| std::fs::create_dir_all(d).is_err()) {
        return Err(bad(CMD, format!("{}: cannot write there", dir.display())));
    }
    let planned: Vec<Value> = work
        .iter()
        .map(|w| json!({"item": w.item.0, "factor": w.factor, "path": w.out.to_string_lossy(), "size": [w.size.0 * w.factor, w.size.1 * w.factor]}))
        .collect();
    let job = crate::narration::new_job(s, format!("Upscale and Restore ({} item{})", work.len(), if work.len() == 1 { "" } else { "s" }));
    let frames: u64 = work.iter().map(|w| if w.still { 1 } else { crate::proxies::frame_count(&w.clip, None).max(1) }).sum();
    job.progress.total.store(frames, Ordering::Relaxed);
    let out: Arc<Mutex<Option<std::result::Result<Vec<Made>, String>>>> = Arc::default();
    let (prog, res, made_out) = (job.progress.clone(), job.result.clone(), out.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-upscale", move || {
        let t0 = web_time::Instant::now();
        let mut made = Vec::new();
        let mut failed = None;
        for w in &work {
            if prog.cancel.load(Ordering::Relaxed) {
                failed = Some("stopped".to_string());
                break;
            }
            let t = web_time::Instant::now();
            let r = if w.still {
                let r = photo_copy(w, &*net, strength, &prog);
                prog.done.fetch_add(1, Ordering::Relaxed);
                r
            } else {
                video_copy(w, net.clone(), strength, prores, &prog)
            };
            match r {
                Ok(()) => made.push(Made { item: w.item, path: w.out.to_string_lossy().into_owned(), factor: w.factor, seconds: t.elapsed().as_secs_f64() }),
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
            Some(e) => Ok(format!("{n} upscaled; then {e}")),
            None => Ok(format!("{n} upscaled")),
        };
        crate::narration::finish(&prog, &res, status, t0);
    })?;
    s.upscales.pending.push(Pending { job: id, strength, out });
    if wait {
        poll(s);
        if let Some(o) = s.upscales.outcomes.get(&id).filter(|o| o.get("error").is_some()) {
            return Err(EngineError::Other(format!("{CMD}: {}", o["error"].as_str().unwrap_or("failed"))));
        }
    }
    Ok(json!({"job": id, "items": planned, "skipped": skipped, "outcome": s.upscales.outcomes.get(&id)}))
}

/// Attaches finished copies (one undo step a job). Called each UI frame, by `jobs.list` and after
/// a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.upscales.pending.len() {
        let Some(pj) = s.upscales.pending.get(i) else { break };
        let finished = s.jobs.iter().find(|j| j.id == pj.job).is_none_or(|j| j.progress.finished.load(std::sync::atomic::Ordering::Relaxed));
        if lock(&pj.out).is_none() && !finished {
            i += 1;
            continue;
        }
        let pj = s.upscales.pending.remove(i);
        let outcome = match lock(&pj.out).take() {
            Some(Ok(made)) if !made.is_empty() => {
                let r = s.edit("Upscale and Restore", |pr, _| {
                    for m in &made {
                        if let Some(ItemKind::Media(c)) = pr.item_mut(m.item).map(|it| &mut it.kind) {
                            c.interpret.upscaled = Some(Upscaled { path: m.path.clone(), factor: m.factor.min(255) as u8, strength: pj.strength });
                        }
                    }
                    Ok(())
                });
                match r {
                    Ok(()) => {
                        json!({"made": made.iter().map(|m| json!({"item": m.item.0, "path": m.path, "factor": m.factor, "seconds": (m.seconds * 10.0).round() / 10.0})).collect::<Vec<_>>()})
                    }
                    Err(e) => json!({"error": e.to_string()}),
                }
            }
            Some(Ok(_)) => json!({"made": []}),
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "stopped: nothing was changed"}),
        };
        s.upscales.outcomes.insert(pj.job, outcome);
        if s.upscales.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.upscales.outcomes.keys().min().copied()
        {
            s.upscales.outcomes.remove(&old);
        }
    }
}

/// What a finished upscale made (`upscale` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.upscales.outcomes.get(&job).cloned()
}

#[cfg(test)]
#[path = "upscale_tests.rs"]
mod tests;
