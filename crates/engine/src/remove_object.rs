//! Remove Object: the thing a clip's mask follows (Object Mask) taken out of the picture — the
//! hole filled with what the other frames show there, and where none does, with what MI-GAN (MIT,
//! downloaded on the user's consent; `filmcraft_vision::inpaint`, on the GPU) paints — written
//! once as a copy of the clip's range and shown in the media's place.
//!
//! | command | does |
//! |---|---|
//! | `removal.status` | whether MI-GAN is downloaded (its name, size, licence, training data and source) |
//! | `removal.downloadModel` | job: download it (29.5 MB), pinned and checked |
//! | `clip.removeObject` | job: the clip's mask `mask` (index or name on `effect`, Opacity; default the selected mask, else the clip's last Object mask) grown by `grow` (a hundredth of the picture's short side, at least 8 px) and softened by `feather` (6 px) is the hole; for the media range the clip plays where the mask has outlines, a copy with the hole filled (`fill`: `auto`, from the other frames then MI-GAN; `model`, MI-GAN in every frame, to compare), an H.264 (`format: "prores"`: ProRes 422 HQ) beside the media (or in `folder`); when written the item shows it in its range and the mask is switched off, one undo step; `off` shows the media as it is |
//!
//! **How the hole is filled.** The background's motion from each frame to the next is measured
//! once — feature points in a ring round the hole (the background next to the thing, which moves
//! like what is behind it even where the far background moves otherwise; the whole picture when
//! the ring has too little detail), a robust fit of position, scale and rotation — and chained. For each pixel of a
//! frame's hole, the frames up to 40 either side (nearest first) are looked at where the
//! background's motion takes that pixel: where one of them shows it outside its own hole, it is
//! a candidate; the median of the nearest five is the pixel. Real background, so it stays put
//! from frame to frame. What no frame shows is painted by MI-GAN — not in every frame (each
//! painting differs, and the fill would flicker) but in key frames half a second apart, each
//! painting carried to the frames near it by the same motion and the two nearest cross-faded.
//! The filled hole is blended in by the mask's soft edge.
//!
//! **In its place.** The copy is the item's Interpretation (`cleaned`, with the media time range
//! it covers); the media pool gives the copy's frames inside the range
//! ([`crate::media_pool::FinerSource`]). A second removal reads the first's copy, so removals
//! stack. One copy of an item at a time: not with Upscale and Restore or Smooth Slow Motion.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_geom::{Affine, Vec2};
use filmcraft_media::{FrameRequest, MediaInfo, MediaKind, MediaSource, SharedSource};
use filmcraft_project::{Cleaned, ItemId, ItemKind, Mask, MaskMode, MediaClip, TrackItem, TrackMethod};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, has_seq, str_p};
use crate::{EngineError, Result, Session};

pub const STATUS_CMD: &str = "removal.status";
pub const DOWNLOAD_CMD: &str = "removal.downloadModel";
pub const CMD: &str = "clip.removeObject";
/// The frames looked at either side of a frame for its hole (nearest first).
const OFFSETS: [i64; 10] = [1, 2, 3, 5, 8, 12, 17, 23, 31, 40];
/// Candidates a pixel takes the median of…
const MAX_CANDIDATES: usize = 5;
/// …when, three or more, they agree within this many levels (median absolute deviation).
const DISAGREE: f32 = 24.0;
/// A neighbour lines up when, round the hole, it is at most this many levels off (mean).
const LINES_UP: f64 = 10.0;
/// The other frames fill a hole when they show at least this share of it.
const SHOWN: f32 = 0.6;
/// A neighbour's pixel counts as background when its own hole covers it less than this.
const CLEAR: f32 = 0.02;
/// Key frames MI-GAN paints, this far apart (seconds).
const KEY_SECONDS: f64 = 0.5;
/// Longest side of the pictures the background's motion is measured on.
const MOTION_SIDE: f32 = 640.0;
/// The ring the background's motion is measured in reaches this many times the hole's size out.
const RING: f64 = 1.0;
/// A motion measured from fewer feature points is not trusted.
const MIN_INLIERS: usize = 8;
/// The mask's outlines are taken to last this long past the first and last (seconds): an Object
/// Mask outlines every tenth of a second, the clip's ends between.
const OUTLINE_REACH: f64 = 0.3;
/// Decoded frames kept, at most this many bytes of them.
const CACHE_BYTES: usize = 1_500_000_000;
/// The most frames one job writes (about an hour at 0.5 s a 1080p frame).
pub const MAX_FRAMES: i64 = 7_200;
const MAX_OUTCOMES: usize = 32;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec { id: STATUS_CMD, label: "Object Removal Model", menu: &[], shortcut: None, params: "{}", enabled: always, run: status, journal: false },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download the Object Removal Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: always,
            run: download,
            journal: false,
        },
        CommandSpec {
            id: CMD,
            label: "Remove Object",
            menu: &["Clip", "Cut-outs"],
            shortcut: None,
            params: r#"{"clip":id?,"mask":n|str?,"effect":index|id?="opacity","grow":px?,"feather":px=6,"fill":"auto"|"model"="auto","format":"h264"|"prores"="h264","folder":str?,"off":bool=false,"wait":bool=false}"#,
            enabled: has_seq,
            run: remove,
            journal: true,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

/// Fills holes in pictures: MI-GAN, or a stand-in.
pub trait Filler: Send + Sync {
    /// An RGBA picture (`w` × `h`) with its hole (`hole`: coverage 0 … 255 a pixel) filled; the
    /// rest as it is.
    fn fill(&self, rgba: &[u8], w: usize, h: usize, hole: &[u8]) -> std::result::Result<Vec<u8>, String>;
}

static STAND_IN: Mutex<Option<Arc<dyn Filler>>> = Mutex::new(None);

/// Use `f` instead of the downloaded network (tests and tools).
pub fn use_stand_in(f: Arc<dyn Filler>) {
    *lock(&STAND_IN) = Some(f);
}

/// MI-GAN is downloaded (and this build can run it).
pub fn installed() -> bool {
    cfg!(feature = "privacy-models") && crate::transcript::models_dir().as_deref().is_some_and(filmcraft_vision::catalog::inpainter_installed)
}

#[cfg(feature = "privacy-models")]
struct Network(filmcraft_vision::inpaint::Inpainter);

#[cfg(feature = "privacy-models")]
impl Filler for Network {
    fn fill(&self, rgba: &[u8], w: usize, h: usize, hole: &[u8]) -> std::result::Result<Vec<u8>, String> {
        self.0.inpaint(rgba, w, h, hole).map_err(|e| e.to_string())
    }
}

fn filler(cmd: &str) -> Result<Arc<dyn Filler>> {
    if let Some(f) = lock(&STAND_IN).clone() {
        return Ok(f);
    }
    network(cmd)
}

#[cfg(feature = "privacy-models")]
fn network(cmd: &str) -> Result<Arc<dyn Filler>> {
    static NET: Mutex<Option<Arc<dyn Filler>>> = Mutex::new(None);
    let mut slot = lock(&NET);
    if let Some(n) = slot.as_ref() {
        return Ok(n.clone());
    }
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    let net = filmcraft_vision::inpaint::Inpainter::load(&dir, true).map_err(|e| match e {
        filmcraft_vision::VisionError::NotInstalled => bad(cmd, format!("MI-GAN is not downloaded yet ({DOWNLOAD_CMD}, 29.5 MB)")),
        e => bad(cmd, e.to_string()),
    })?;
    log::info!("MI-GAN loaded on the {}", net.device_name());
    let n: Arc<dyn Filler> = Arc::new(Network(net));
    *slot = Some(n.clone());
    Ok(n)
}

#[cfg(not(feature = "privacy-models"))]
fn network(cmd: &str) -> Result<Arc<dyn Filler>> {
    Err(bad(cmd, "object removal is not available in this build (built without the `privacy-models` feature)"))
}

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_vision::catalog as c;
    Ok(json!({
        "available": cfg!(feature = "privacy-models"),
        "installed": installed(),
        "name": c::INPAINT_NAME,
        "bytes": c::INPAINTER.size,
        "license": c::INPAINT_LICENSE,
        "source": c::INPAINT_SOURCE,
        "training": c::INPAINT_TRAINING,
        "download": DOWNLOAD_CMD,
    }))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::inpainter_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "object removal is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::INPAINT_NAME));
    job.progress.total.store(filmcraft_vision::catalog::INPAINTER.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-migan-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "MI-GAN installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::INPAINTER.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::INPAINT_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::inpaint_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("object removal is not available in this build".into())
}

// ---------------------------------------------------------------------------------------------
// the hole filled
// ---------------------------------------------------------------------------------------------

/// A media frame decoded: its picture, its hole's coverage (0 … 1) and its picture for motion.
struct Frame {
    rgba: Vec<u8>,
    w: usize,
    h: usize,
    hole: Vec<f32>,
    /// The hole's box `[x0, y0, x1, y1]` (picture pixels), when it has one.
    bbox: Option<[f64; 4]>,
    gray: filmcraft_render::track::Prepared,
    /// Picture pixels to the motion picture's.
    k: f64,
}

impl Frame {
    fn empty(&self) -> bool {
        self.hole.iter().all(|v| *v <= 0.0)
    }
    /// Bilinear RGB at (x, y), `None` outside the picture.
    fn at(&self, x: f64, y: f64) -> Option<[f32; 3]> {
        if self.w == 0 || self.h == 0 || !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 || x > (self.w - 1) as f64 || y > (self.h - 1) as f64 {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let px = |xx: usize, yy: usize, c: usize| self.rgba.get((yy * self.w + xx) * 4 + c).copied().map_or(0.0, f32::from);
        let mut out = [0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = px(x0, y0, c) * (1.0 - fx) * (1.0 - fy) + px(x1, y0, c) * fx * (1.0 - fy) + px(x0, y1, c) * (1.0 - fx) * fy + px(x1, y1, c) * fx * fy;
        }
        Some(out)
    }
    /// The hole's coverage at the nearest pixel (1 outside the picture).
    fn hole_at(&self, x: f64, y: f64) -> f32 {
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return 1.0;
        }
        let (xi, yi) = (x.round() as usize, y.round() as usize);
        if xi >= self.w || yi >= self.h {
            return 1.0;
        }
        self.hole.get(yi * self.w + xi).copied().unwrap_or(1.0)
    }
}

/// What makes the copy's frames: the media, the hole (the mask, in clip pixels, `scale` to the
/// media's), the range (media frames `first..=last`; outside the mask's outlines, `keyed`, no
/// hole) and the caches. No lock is held while a frame decodes or a network runs (the export
/// renders frames on rayon's threads; see `interpolate::Between`).
struct Cleaning {
    inner: SharedSource,
    info: MediaInfo,
    rate: FrameRate,
    first: i64,
    last: i64,
    keyed: (i64, i64),
    mask: Mask,
    scale: f64,
    grow: f32,
    feather: f32,
    model_only: bool,
    key_every: i64,
    filler: Arc<dyn Filler>,
    frames: Mutex<(HashMap<i64, Arc<Frame>>, VecDeque<i64>)>,
    keep: usize,
    /// The background's motion from frame `i` to `i + 1` (picture pixels), once measured.
    steps: Mutex<HashMap<i64, Option<Affine>>>,
    /// Key frames' paintings (RGBA), once made.
    keys: Mutex<HashMap<i64, Arc<Vec<u8>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl Cleaning {
    /// Media frame `i` decoded (kept), with its hole.
    fn frame(&self, i: i64) -> filmcraft_media::Result<Arc<Frame>> {
        let i = i.clamp(self.first.min(self.keyed.0), self.last.max(self.keyed.1));
        if let Some(f) = lock(&self.frames).0.get(&i) {
            return Ok(f.clone());
        }
        let t = self.rate.tick_of(i);
        let vf = self.inner.video_frame(FrameRequest { time: t, scale: 1.0 })?;
        let (w, h) = (vf.width as usize, vf.height as usize);
        let rgba = vf.to_rgba8();
        let hole = if i >= self.keyed.0 && i <= self.keyed.1 { self.hole(t, w, h) } else { vec![0.0; w * h] };
        let k = (f64::from(MOTION_SIDE) / w.max(h).max(1) as f64).min(1.0);
        let (sw, sh) = (((w as f64 * k).round() as usize).max(8), ((h as f64 * k).round() as usize).max(8));
        let (small, sw, sh) = crate::cutout::resize_rgba(&rgba, w, h, sw, sh);
        let gray = filmcraft_render::track::Prepared::new(filmcraft_render::track::Gray::from_rgba8(sw, sh, &small));
        let bbox = hole_box(&hole, w);
        let f = Arc::new(Frame { rgba, w, h, hole, bbox, gray, k: sw as f64 / w.max(1) as f64 });
        let mut cache = lock(&self.frames);
        let (map, order) = &mut *cache;
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(i) {
            e.insert(f.clone());
            order.push_back(i);
            while order.len() > self.keep {
                if let Some(old) = order.pop_front() {
                    map.remove(&old);
                }
            }
        }
        Ok(f)
    }

    /// The hole at media time `t` in a `w` × `h` frame: the mask grown and softened.
    fn hole(&self, t: Tick, w: usize, h: usize) -> Vec<f32> {
        let s = self.scale;
        let pts: Vec<[f32; 2]> = self.mask.path_at(t).flatten(0.5).into_iter().map(|p| [(p.x * s) as f32, (p.y * s) as f32]).collect();
        let m = filmcraft_render::mask::FlatMask { pts, feather: self.feather, expansion: self.grow, opacity: 1.0, inverted: false, mode: MaskMode::Add };
        filmcraft_render::mask::coverage(&[m], w, h).unwrap_or_else(|| vec![0.0; w * h])
    }

    /// The background's motion from frame `i` to `i + 1`, measured once.
    fn step(&self, i: i64) -> Option<Affine> {
        if let Some(s) = lock(&self.steps).get(&i) {
            return *s;
        }
        let (a, b) = (self.frame(i).ok()?, self.frame(i + 1).ok()?);
        let g = a.gray.base();
        let (gw, gh) = (g.w as f64, g.h as f64);
        let whole = [Vec2::new(0.0, 0.0), Vec2::new(gw, 0.0), Vec2::new(gw, gh), Vec2::new(0.0, gh)];
        let fit = |region: &[Vec2]| {
            filmcraft_render::track::track_step(&a.gray, &b.gray, region, TrackMethod::PositionScaleRotation).filter(|s| s.inliers >= MIN_INLIERS)
        };
        // a ring round both frames' holes, else the whole picture
        let both = match (a.bbox, b.bbox) {
            (Some(p), Some(q)) => Some([p[0].min(q[0]), p[1].min(q[1]), p[2].max(q[2]), p[3].max(q[3])]),
            (p, q) => p.or(q),
        };
        let st = both.map(|bx| ring(bx.map(|v| v * a.k), gw, gh)).and_then(|r| fit(&r)).or_else(|| fit(&whole));
        // motion pixels → picture pixels
        let m = st.map(|s| Affine::scale(1.0 / b.k, 1.0 / b.k).then_apply(&s.transform).then_apply(&Affine::scale(a.k, a.k)));
        lock(&self.steps).insert(i, m);
        m
    }

    /// The background's motion from frame `i` to frame `n`, chained from frame to frame.
    fn motion(&self, i: i64, n: i64) -> Option<Affine> {
        let mut m = Affine::IDENTITY;
        if n >= i {
            for j in i..n {
                m = self.step(j)?.then_apply(&m);
            }
            Some(m)
        } else {
            self.motion(n, i)?.inverse()
        }
    }

    /// Frame `i`'s hole filled from the frames near it: the picture with what was found, and
    /// where nothing was (coverage 0 … 255).
    fn shown_elsewhere(&self, i: i64, f: &Frame) -> (Vec<u8>, Vec<u8>) {
        let mut out = f.rgba.clone();
        let mut missing = vec![0u8; f.w * f.h];
        let holes: Vec<usize> = f.hole.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(p, _)| p).collect();
        if holes.is_empty() {
            return (out, missing);
        }
        // the neighbours, nearest first, with the motion to them
        let mut near: Vec<(Arc<Frame>, Affine)> = Vec::new();
        for o in OFFSETS {
            for n in [i - o, i + o] {
                // (outside the mask's outlines the thing is not marked: no help)
                if n < self.keyed.0 || n > self.keyed.1 {
                    continue;
                }
                if let (Some(m), Ok(fr)) = (self.motion(i, n), self.frame(n))
                    && lines_up(f, &fr, &m)
                {
                    near.push((fr, m));
                }
            }
        }
        for p in holes {
            let (x, y) = ((p % f.w) as f64, (p / f.w) as f64);
            let mut found: Vec<[f32; 3]> = Vec::with_capacity(MAX_CANDIDATES);
            for (fr, m) in &near {
                let q = m.apply(Vec2::new(x, y));
                if fr.hole_at(q.x, q.y) < CLEAR
                    && let Some(c) = fr.at(q.x, q.y)
                {
                    found.push(c);
                    if found.len() >= MAX_CANDIDATES {
                        break;
                    }
                }
            }
            // the median of what the frames show, if they agree
            let mut med = [0f32; 3];
            let mut spread = 0f32;
            for (c, mc) in med.iter_mut().enumerate() {
                let mut v: Vec<f32> = found.iter().map(|q| q[c]).collect();
                v.sort_by(f32::total_cmp);
                *mc = v.get(v.len() / 2).copied().unwrap_or(0.0);
                let mut dev: Vec<f32> = v.iter().map(|x| (x - *mc).abs()).collect();
                dev.sort_by(f32::total_cmp);
                spread = spread.max(dev.get(dev.len() / 2).copied().unwrap_or(0.0));
            }
            if found.is_empty() || (found.len() >= 3 && spread > DISAGREE) {
                if let Some(slot) = missing.get_mut(p) {
                    *slot = 255;
                }
                continue;
            }
            for (c, mc) in med.iter().enumerate() {
                if let Some(slot) = out.get_mut(p * 4 + c) {
                    *slot = mc.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        // the other frames showed only slivers (the thing stays put on what is behind it): all of
        // it painted, not a patchwork of slivers and painting
        let (n, gone) = (f.hole.iter().filter(|v| **v > 0.0).count(), missing.iter().filter(|v| **v > 0).count());
        if n > 0 && (n - gone) as f32 / (n as f32) < SHOWN {
            for (p, v) in f.hole.iter().enumerate() {
                if *v > 0.0 {
                    if let Some(slot) = missing.get_mut(p) {
                        *slot = 255;
                    }
                    for c in 0..3 {
                        if let (Some(o), Some(src)) = (out.get_mut(p * 4 + c), f.rgba.get(p * 4 + c)) {
                            *o = *src;
                        }
                    }
                }
            }
        }
        (out, missing)
    }

    /// Key frame `k`'s painting: its hole filled from the frames near it, the rest by the model.
    fn key(&self, k: i64) -> std::result::Result<Arc<Vec<u8>>, String> {
        if let Some(p) = lock(&self.keys).get(&k) {
            return Ok(p.clone());
        }
        let f = self.frame(k).map_err(|e| e.to_string())?;
        let (mut pic, mut missing) =
            if self.model_only { (f.rgba.clone(), f.hole.iter().map(|v| if *v > 0.0 { 255 } else { 0 }).collect()) } else { self.shown_elsewhere(k, &f) };
        if missing.iter().any(|v| *v > 0) {
            grow_by(&mut missing, f.w, f.h, 2);
            pic = self.filler.fill(&pic, f.w, f.h, &missing)?;
        }
        let p = Arc::new(pic);
        let mut keys = lock(&self.keys);
        keys.insert(k, p.clone());
        if keys.len() > 64
            && let Some(old) = keys.keys().copied().max_by_key(|o| (o - k).abs())
        {
            keys.remove(&old);
        }
        Ok(p)
    }

    /// Media frame `i` with its hole filled.
    fn cleaned(&self, i: i64) -> std::result::Result<Vec<u8>, String> {
        let f = self.frame(i).map_err(|e| e.to_string())?;
        if f.empty() {
            return Ok(f.rgba.clone());
        }
        let (mut pic, missing) = if self.model_only {
            (f.rgba.clone(), f.hole.iter().map(|v| if *v > 0.0 { 255u8 } else { 0 }).collect::<Vec<u8>>())
        } else {
            self.shown_elsewhere(i, &f)
        };
        if missing.iter().any(|v| *v > 0) {
            if self.model_only {
                let mut m = missing;
                grow_by(&mut m, f.w, f.h, 2);
                pic = self.filler.fill(&pic, f.w, f.h, &m)?;
            } else {
                // the key frames' paintings carried here by the background's motion, cross-faded
                let ke = self.key_every.max(1);
                let k0 = (i - self.keyed.0).div_euclid(ke) * ke + self.keyed.0;
                let k1 = (k0 + ke).min(self.keyed.1);
                let wt = if k1 > k0 { (i - k0) as f32 / (k1 - k0) as f32 } else { 0.0 };
                let mut sources: Vec<(Arc<Vec<u8>>, Affine, f32)> = Vec::new();
                for (k, wk) in [(k0, 1.0 - wt), (k1, wt)] {
                    if wk <= 0.0 && k1 != k0 {
                        continue;
                    }
                    if let Some(m) = self.motion(i, k) {
                        sources.push((self.key(k)?, m, wk.max(1e-3)));
                    }
                }
                if sources.is_empty() {
                    let mut m = missing.clone();
                    grow_by(&mut m, f.w, f.h, 2);
                    pic = self.filler.fill(&pic, f.w, f.h, &m)?;
                } else {
                    for (p, m) in missing.iter().enumerate() {
                        if *m == 0 {
                            continue;
                        }
                        let (x, y) = ((p % f.w) as f64, (p / f.w) as f64);
                        let mut acc = [0f32; 3];
                        let mut got = 0f32;
                        for (px, mo, wk) in &sources {
                            let q = mo.apply(Vec2::new(x, y));
                            if let Some(c) = bilinear(px, f.w, f.h, q.x, q.y) {
                                for ch in 0..3 {
                                    acc[ch] += c[ch] * wk;
                                }
                                got += wk;
                            }
                        }
                        if got > 0.0 {
                            for ch in 0..3 {
                                if let Some(slot) = pic.get_mut(p * 4 + ch) {
                                    *slot = (acc[ch] / got).round().clamp(0.0, 255.0) as u8;
                                }
                            }
                        }
                    }
                }
            }
        }
        // blended in by the hole's soft edge
        let mut out = f.rgba.clone();
        for (p, a) in f.hole.iter().enumerate() {
            if *a <= 0.0 {
                continue;
            }
            for c in 0..3 {
                if let (Some(o), Some(v)) = (out.get_mut(p * 4 + c), pic.get(p * 4 + c)) {
                    *o = (f32::from(*o) * (1.0 - a) + f32::from(*v) * a).round() as u8;
                }
            }
        }
        Ok(out)
    }
}

/// Whether frame `n`, moved by `m` (from frame `f`'s pixels), shows what `f` shows round its
/// hole: the mean difference on a band of points round the hole (outside both holes) at most
/// [`LINES_UP`] levels. A neighbour that does not line up (the motion fits the band badly — the
/// camera moving in depth, a surface at a slant) would put smeared copies into the hole.
fn lines_up(f: &Frame, n: &Frame, m: &Affine) -> bool {
    let Some(b) = f.bbox else { return true };
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let reach = (bw.max(bh) * 0.25).max(8.0);
    let (x0, y0, x1, y1) = ((b[0] - reach).max(0.0), (b[1] - reach).max(0.0), (b[2] + reach).min(f.w as f64), (b[3] + reach).min(f.h as f64));
    let step = ((x1 - x0).max(y1 - y0) / 32.0).max(2.0);
    let (mut d, mut k) = (0f64, 0f64);
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            if f.hole_at(x, y) <= 0.0 {
                let q = m.apply(Vec2::new(x, y));
                if n.hole_at(q.x, q.y) < CLEAR
                    && let (Some(a), Some(c)) = (f.at(x, y), n.at(q.x, q.y))
                {
                    d += a.iter().zip(&c).map(|(u, v)| f64::from((u - v).abs())).sum::<f64>() / 3.0;
                    k += 1.0;
                }
            }
            x += step;
        }
        y += step;
    }
    k < 16.0 || d / k <= LINES_UP
}

/// The box `[x0, y0, x1, y1]` of a hole's pixels (coverage above nothing).
fn hole_box(hole: &[f32], w: usize) -> Option<[f64; 4]> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (p, v) in hole.iter().enumerate() {
        if *v > 0.0 {
            let (x, y) = (p % w.max(1), p / w.max(1));
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    (x1 > x0 && y1 > y0).then_some([x0 as f64, y0 as f64, x1 as f64, y1 as f64])
}

/// A ring round box `b` (a `w` × `h` picture's pixels) as one polygon: a box [`RING`] times as
/// large round it, a slit of no width in, and the box (a little grown) the other way round — so
/// the point-in-polygon test (non-zero winding) leaves the box out.
fn ring(b: [f64; 4], w: f64, h: f64) -> Vec<Vec2> {
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let reach = (bw.max(bh) * RING).max(24.0);
    let (ox0, oy0, ox1, oy1) = ((b[0] - reach).max(0.0), (b[1] - reach).max(0.0), (b[2] + reach).min(w), (b[3] + reach).min(h));
    let (ix0, iy0, ix1, iy1) = (b[0] - 3.0, b[1] - 3.0, b[2] + 3.0, b[3] + 3.0);
    [(ox0, oy0), (ox1, oy0), (ox1, oy1), (ox0, oy1), (ox0, oy0), (ix0, iy0), (ix0, iy1), (ix1, iy1), (ix1, iy0), (ix0, iy0)]
        .into_iter()
        .map(|(x, y)| Vec2::new(x, y))
        .collect()
}

/// Bilinear RGB of an RGBA picture at (x, y), `None` outside it.
fn bilinear(px: &[u8], w: usize, h: usize, x: f64, y: f64) -> Option<[f32; 3]> {
    if w == 0 || h == 0 || !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let at = |xx: usize, yy: usize, c: usize| px.get((yy * w + xx) * 4 + c).copied().map_or(0.0, f32::from);
    let mut out = [0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        *o = at(x0, y0, c) * (1.0 - fx) * (1.0 - fy) + at(x1, y0, c) * fx * (1.0 - fy) + at(x0, y1, c) * (1.0 - fx) * fy + at(x1, y1, c) * fx * fy;
    }
    Some(out)
}

/// A coverage map (0 … 255) grown by `r` pixels (the largest value within a square).
fn grow_by(m: &mut [u8], w: usize, h: usize, r: usize) {
    if r == 0 || w == 0 || h == 0 || m.len() < w * h {
        return;
    }
    let src = m.to_vec();
    let mut rows = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            rows[y * w + x] = src[y * w + a..=y * w + b].iter().copied().max().unwrap_or(0);
        }
    }
    for y in 0..h {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            m[y * w + x] = (a..=b).map(|yy| rows[yy * w + x]).max().unwrap_or(0);
        }
    }
}

impl MediaSource for Cleaning {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        let i = (self.first + self.rate.frame_at(req.time).max(0)).min(self.last);
        let f = self.frame(i)?;
        match self.cleaned(i) {
            Ok(px) => Ok(Arc::new(filmcraft_frame::VideoFrame::rgba8(f.w as u32, f.h as u32, px))),
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

// ---------------------------------------------------------------------------------------------
// the copy
// ---------------------------------------------------------------------------------------------

/// The copy to make.
struct Work {
    item: ItemId,
    clip: MediaClip,
    name: String,
    src: SharedSource,
    first: i64,
    last: i64,
    keyed: (i64, i64),
    mask: Mask,
    scale: f64,
    grow: f32,
    feather: f32,
    model_only: bool,
    out: PathBuf,
}

/// Write the copy of `w`'s range through the export (no sound): `(start, duration)`.
fn write_copy(w: Work, filler: Arc<dyn Filler>, prores: bool, prog: &filmcraft_export::Progress) -> std::result::Result<(Tick, Tick), String> {
    use filmcraft_project::{Project, SequenceSettings, TrackKind};
    use filmcraft_time::TimeRange;
    let rate = w.clip.frame_rate();
    let frames = (w.last - w.first + 1).max(1);
    let (start, end) = (rate.tick_of(w.first), rate.tick_of(w.last + 1));
    let duration = end - start;
    let mut clip = w.clip.clone();
    clip.offline = false;
    clip.proxy = None;
    clip.interpret = Default::default();
    clip.info.duration = duration;
    let (vw, vh) = clip.info.video.as_ref().map_or((1920, 1080), |v| (v.width, v.height));
    let st = SequenceSettings { width: vw, height: vh, par: clip.pixel_aspect(), frame_rate: rate, ..Default::default() };
    let mut p = Project::new("remove");
    let info = clip.info.clone();
    let item = p.add_item(&w.name, filmcraft_project::Label::Iris, ItemKind::Media(clip), None);
    let seq = p.new_sequence("remove", st, 1, 0, None);
    let range = TimeRange { start: Tick::ZERO, duration: rate.tick_of(frames) };
    let ti = p.make_track_item(item, TrackKind::Video, Tick::ZERO, range, rate).ok_or("the clip does not place")?;
    p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()).ok_or("no sequence")?.items.push(ti);
    let error = Arc::new(Mutex::new(None));
    let bytes = (vw as usize).saturating_mul(vh as usize).saturating_mul(9).max(1);
    let keep = (CACHE_BYTES / bytes).clamp(24, 4 * OFFSETS.len() + 96);
    let key_every = ((KEY_SECONDS * rate.as_f64()).round() as i64).max(1);
    let out = w.out.clone();
    let src: SharedSource = Arc::new(Cleaning {
        inner: w.src,
        info,
        rate,
        first: w.first,
        last: w.last,
        keyed: w.keyed,
        mask: w.mask,
        scale: w.scale,
        grow: w.grow,
        feather: w.feather,
        model_only: w.model_only,
        key_every,
        filler,
        frames: Mutex::new((HashMap::new(), VecDeque::new())),
        keep,
        steps: Mutex::new(HashMap::new()),
        keys: Mutex::new(HashMap::new()),
        error: error.clone(),
    });
    let pixels = f64::from(vw) * f64::from(vh) * rate.as_f64() / 30.0;
    let settings = filmcraft_export::ExportSettings {
        format: if prores { filmcraft_export::Format::ProRes } else { filmcraft_export::Format::H264 },
        path: out.to_string_lossy().into_owned(),
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
        let _ = std::fs::remove_file(&out);
        return Err(format!("{}: {e}", w.name));
    }
    r.map(|_| (start, duration))
}

// ---------------------------------------------------------------------------------------------
// the command
// ---------------------------------------------------------------------------------------------

/// Jobs whose copies are attached when they finish, and what finished ones made.
#[derive(Default)]
pub struct Removals {
    pending: Vec<Pending>,
    outcomes: BTreeMap<u64, Value>,
}

/// A copy made: the item, its file and range, the clip and mask to switch off.
#[derive(Clone, Debug)]
struct Made {
    item: ItemId,
    path: String,
    start: Tick,
    duration: Tick,
    seconds: f64,
}

struct Pending {
    job: u64,
    seq: ItemId,
    clip: filmcraft_project::ClipId,
    effect: usize,
    mask: usize,
    out: Arc<Mutex<Option<std::result::Result<Made, String>>>>,
}

/// The mask a removal takes out: `mask` (index or name on `effect`), else the selected mask on
/// the clip, else the clip's last mask tracked by object, else its last mask with outlines.
fn pick_mask(s: &Session, it: &TrackItem, p: &Value) -> Result<(usize, usize)> {
    let effect = match p.get("effect") {
        None | Some(Value::Null) => None,
        v => Some(crate::masks::effect_index(&it.effects, v, None, it.id).ok_or_else(|| bad(CMD, "no such effect on the clip"))?),
    };
    let in_effect = |ei: usize| it.effects.get(ei).map(|e| e.masks.as_slice()).unwrap_or_default();
    match p.get("mask") {
        Some(Value::Number(n)) => {
            let ei = effect.or_else(|| it.effects.iter().position(|e| e.effect == "opacity")).ok_or_else(|| bad(CMD, "the clip has no Opacity"))?;
            let mi = n.as_u64().and_then(|m| usize::try_from(m).ok()).filter(|m| *m < in_effect(ei).len()).ok_or_else(|| bad(CMD, "no such mask"))?;
            return Ok((ei, mi));
        }
        Some(Value::String(name)) => {
            for (ei, e) in it.effects.iter().enumerate().filter(|(ei, _)| effect.is_none_or(|x| x == *ei)) {
                if let Some(mi) = e.masks.iter().position(|m| &m.name == name) {
                    return Ok((ei, mi));
                }
            }
            return Err(bad(CMD, format!("no mask named `{name}` on the clip")));
        }
        None | Some(Value::Null) => {}
        Some(_) => return Err(bad(CMD, "`mask` is an index or a name")),
    }
    if let Some(sel) = s.state.selected_mask.filter(|m| m.clip == it.id && effect.is_none_or(|e| e == m.effect))
        && in_effect(sel.effect).get(sel.mask).is_some()
    {
        return Ok((sel.effect, sel.mask));
    }
    let all = || {
        it.effects
            .iter()
            .enumerate()
            .filter(|(ei, _)| effect.is_none_or(|x| x == *ei))
            .flat_map(|(ei, e)| e.masks.iter().enumerate().map(move |(mi, m)| (ei, mi, m)))
    };
    all()
        .rfind(|(_, _, m)| m.track_method == TrackMethod::Object)
        .or_else(|| all().rfind(|(_, _, m)| !m.path.keyframes.is_empty()))
        .map(|(ei, mi, _)| (ei, mi))
        .ok_or_else(|| bad(CMD, "mask the thing first (Object Mask: clip.objectMask)"))
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let (_, it) = crate::cutout::video_clip(s, p, CMD)?;
    if it.duration <= Tick::ZERO {
        return Err(bad(CMD, "the clip is empty"));
    }
    let item = it.item;
    let pi = s.project.item(item).ok_or_else(|| bad(CMD, "no such item"))?;
    let m = pi.as_media().ok_or_else(|| bad(CMD, "the clip is not of a media file"))?.clone();
    let name = pi.name.clone();
    if bool_p(p, "off").unwrap_or(false) {
        let on = m.interpret.cleaned.is_some();
        if on {
            s.edit("Remove Object Off", |pr, _| {
                if let Some(ItemKind::Media(c)) = pr.item_mut(item).map(|it| &mut it.kind) {
                    c.interpret.cleaned = None;
                }
                Ok(())
            })?;
        }
        return Ok(json!({"off": if on { vec![item.0] } else { Vec::new() }}));
    }
    let why = if m.info.video.is_none() {
        Some("the clip has no picture")
    } else if crate::vr360::is_360(s, item) {
        Some("360° media is seen through a virtual camera: remove things from its reframed copy")
    } else if m.info.kind != MediaKind::Movie {
        Some("only videos (a photo: use the model on the picture)")
    } else if m.interpret.upscaled.is_some() || m.interpret.interpolated.is_some() {
        Some("the media has a copy already (Upscale and Restore or Smooth Slow Motion): turn that off first")
    } else {
        None
    };
    if let Some(why) = why {
        return Err(bad(CMD, why));
    }
    let (ei, mi) = pick_mask(s, &it, p)?;
    let mask = it.effects.get(ei).and_then(|e| e.masks.get(mi)).cloned().ok_or_else(|| bad(CMD, "no such mask"))?;
    let num = |k: &str, d: f64, hi: f64| -> Result<f32> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d as f32),
            Some(v) => {
                v.as_f64().filter(|v| v.is_finite() && (0.0..=hi).contains(v)).map(|v| v as f32).ok_or_else(|| bad(CMD, format!("`{k}` is 0 to {hi} pixels")))
            }
        }
    };
    // by default a hundredth of the picture's short side (at least 8 px): edges and a little shadow
    let short = m.info.video.as_ref().map_or(1080, |v| v.width.min(v.height));
    let (grow, feather) = (num("grow", (f64::from(short) / 100.0).max(8.0), 200.0)?, num("feather", 6.0, 100.0)?);
    let model_only = match str_p(p, "fill") {
        None | Some("auto") => false,
        Some("model") => true,
        Some(f) => return Err(bad(CMD, format!("`fill` `{f}`: auto or model"))),
    };
    let prores = match str_p(p, "format") {
        None | Some("h264") => false,
        Some("prores") => true,
        Some(_) => return Err(bad(CMD, "`format` is h264 or prores")),
    };
    let rate = m.frame_rate().sane();
    if rate.frame_duration().0 <= 0 {
        return Err(bad(CMD, "the media has no frame rate"));
    }
    let total = rate.frame_at(m.info.duration).max(1);
    // the clip's media range, and where the mask has outlines
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for k in 0..=32 {
        let t = it.start + Tick(it.duration.0 / 32 * k).min(it.duration - Tick(1));
        let f = rate.frame_at(it.source_time_at(t));
        (lo, hi) = (lo.min(f), hi.max(f));
    }
    let (lo, hi) = (lo.clamp(0, total - 1), hi.clamp(0, total - 1));
    let keys: Vec<i64> = mask.path.keyframes.iter().map(|k| rate.frame_at(k.time)).collect();
    let reach = (OUTLINE_REACH * rate.as_f64()).round() as i64;
    let keyed = match (keys.iter().min(), keys.iter().max()) {
        (Some(a), Some(b)) => ((*a - reach).max(lo), (*b + reach).min(hi)),
        _ => (lo, hi),
    };
    if keyed.0 > keyed.1 {
        return Err(bad(CMD, "the mask has no outlines where the clip plays"));
    }
    // the copy covers the mask's range, and an earlier removal's (it is read through)
    let (mut first, mut last) = keyed;
    if let Some(c) = &m.interpret.cleaned {
        first = first.min(rate.frame_at(c.start).max(0));
        last = last.max((rate.frame_at(c.start + c.duration) - 1).min(total - 1));
    }
    if last - first + 1 > MAX_FRAMES {
        return Err(bad(CMD, format!("{} frames, more than {MAX_FRAMES}: trim the clip or the mask's outlines", last - first + 1)));
    }
    let filmcraft_project::MediaRef::File { path } = &m.media else { return Err(bad(CMD, "not a file")) };
    let size = filmcraft_render::clip_size(&s.project, &it).ok_or_else(|| bad(CMD, "the clip has no picture"))?;
    let vw = m.info.video.as_ref().map_or(size.0, |v| v.width);
    let scale = f64::from(vw) / f64::from(size.0.max(1));
    let src = s.media.full_res_source(&s.project, item, &*s.services).ok_or_else(|| bad(CMD, format!("{name}: the media does not open")))?;
    let media = Path::new(path);
    let dir = str_p(p, "folder").map(PathBuf::from).unwrap_or_else(|| media.parent().map(Path::to_path_buf).unwrap_or_default());
    let stem = media.file_stem().map_or_else(|| "video".to_string(), |s| s.to_string_lossy().into_owned());
    let out = crate::proxies::unique_path(&dir, &stem, " (object removed)", if prores { "mov" } else { "mp4" }, &[]);
    if std::fs::create_dir_all(&dir).is_err() {
        return Err(bad(CMD, format!("{}: cannot write there", dir.display())));
    }
    let filler = filler(CMD)?;
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let planned = json!({"item": item.0, "path": out.to_string_lossy(), "from": rate.tick_of(first).seconds(), "to": rate.tick_of(last + 1).seconds(), "frames": last - first + 1, "mask": mask.name});
    let work = Work { item, clip: m, name, src, first, last, keyed, mask, scale, grow, feather, model_only, out };
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Remove Object ({} frames)", last - first + 1));
    job.progress.total.store((last - first + 1) as u64, Ordering::Relaxed);
    let made: Arc<Mutex<Option<std::result::Result<Made, String>>>> = Arc::default();
    let (prog, res, made_out) = (job.progress.clone(), job.result.clone(), made.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-remove-object", move || {
        let t0 = web_time::Instant::now();
        let (item, path) = (work.item, work.out.to_string_lossy().into_owned());
        let r = write_copy(work, filler, prores, &prog).map(|(start, duration)| Made { item, path, start, duration, seconds: t0.elapsed().as_secs_f64() });
        let status = r.as_ref().map(|_| "the object removed".to_string()).map_err(Clone::clone);
        *lock(&made_out) = Some(r);
        crate::narration::finish(&prog, &res, status, t0);
    })?;
    s.removals.pending.push(Pending { job: id, seq, clip: it.id, effect: ei, mask: mi, out: made });
    if wait {
        poll(s);
        if let Some(o) = s.removals.outcomes.get(&id).filter(|o| o.get("error").is_some()) {
            return Err(EngineError::Other(format!("{CMD}: {}", o["error"].as_str().unwrap_or("failed"))));
        }
    }
    Ok(json!({"job": id, "copy": planned, "outcome": s.removals.outcomes.get(&id)}))
}

/// Attaches finished copies and switches their masks off (one undo step a job). Called each UI
/// frame, by `jobs.list` and after a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.removals.pending.len() {
        let Some(pj) = s.removals.pending.get(i) else { break };
        let finished = s.jobs.iter().find(|j| j.id == pj.job).is_none_or(|j| j.progress.finished.load(std::sync::atomic::Ordering::Relaxed));
        if lock(&pj.out).is_none() && !finished {
            i += 1;
            continue;
        }
        let pj = s.removals.pending.remove(i);
        let outcome = match lock(&pj.out).take() {
            Some(Ok(made)) => {
                let (seq, clip, ei, mi) = (pj.seq, pj.clip, pj.effect, pj.mask);
                let r = s.edit("Remove Object", |pr, _| {
                    if let Some(ItemKind::Media(c)) = pr.item_mut(made.item).map(|it| &mut it.kind) {
                        c.interpret.cleaned = Some(Cleaned { path: made.path.clone(), start: made.start, duration: made.duration });
                    }
                    // the mask did its work: the clip shows whole again
                    if let Some((_, it)) = pr.sequence_mut(seq).and_then(|q| q.find_item_mut(clip))
                        && let Some(m) = it.effects.get_mut(ei).and_then(|e| e.masks.get_mut(mi))
                    {
                        m.mode = MaskMode::None;
                    }
                    Ok(())
                });
                match r {
                    Ok(()) => {
                        json!({"made": {"item": made.item.0, "path": made.path, "from": made.start.seconds(), "to": (made.start + made.duration).seconds(), "seconds": (made.seconds * 10.0).round() / 10.0}})
                    }
                    Err(e) => json!({"error": e.to_string()}),
                }
            }
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "stopped: nothing was changed"}),
        };
        s.removals.outcomes.insert(pj.job, outcome);
        if s.removals.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.removals.outcomes.keys().min().copied()
        {
            s.removals.outcomes.remove(&old);
        }
    }
}

/// What a finished removal made (`removeObject` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.removals.outcomes.get(&job).cloned()
}

#[cfg(test)]
#[path = "remove_object_tests.rs"]
mod tests;
