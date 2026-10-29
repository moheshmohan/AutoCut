//! Depth for photos: how near each part of a picture is (Depth Anything V2 Small, downloaded on
//! the user's consent), and the two effects made from it — the 3D photo (`depth_parallax`: the
//! near parts move further than the far ones) and Depth Blur (`depth_blur`), both rendered by
//! `filmcraft_render::depth` on the GPU.
//!
//! | command | does |
//! |---|---|
//! | `depth.status` | whether the depth network is downloaded (its name, size, licence and source) |
//! | `depth.downloadModel` | job: download it (99.2 MB, Apache-2.0), pinned and checked |
//! | `clip.photo3d` | the 3D photo on photo clips (`clips`, default the selected): `move` `in` / `out` (the camera moving in or back, the farthest still), `left` / `right` / `up` / `down` (round the subject, which stays where it is), `circle` (a slow turn round it), or `auto` (the default: the clip's own Ken Burns move — in, out, or the way it pans — else in); `amount` 0..100 (50: the nearest moves 2.5 % of the width either way, or grows 10 %); `pivot` the depth that stays (`auto`, `far`, `subject` or 0..1); one undo step |
//! | `clip.depthBlur` | Depth Blur on photo clips: `focus` the depth kept sharp (`subject`, the default, `near`, `far` or 0..1), `blur` the farthest blur (% of the width, 1.5), `range` the depth of field (0.1), `pull` a focus pull from the other end to the focus over the first half of the clip; one undo step |
//!
//! **The subject's depth.** With a `subject` box (`[x, y, w, h]`, 0..1 of the picture, what a
//! vision model found) the median depth inside it; else the 80th percentile of the middle 60 % of
//! the picture (the main thing in front). `near` is the 95th percentile, `far` the 5th.
//!
//! **Kept depth.** The network's answer for a picture is kept under the data folder
//! (`cache/depth/<key>-<model>.depth`: 16 bits, deflated), so a photo is looked at once across sessions and
//! exports; the render keeps the last 16 in memory.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_project::effect::{DEPTH_BLUR, PHOTO_3D};
use filmcraft_project::{ClipId, EffectInstance, Interpolation, ItemKind, Keyframe, Param, ParamValue};
use filmcraft_render::depth::{DepthField, DepthModel};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, clips_p, f64_p, has_selection, str_p};
use crate::{EngineError, Result, Session};

pub const STATUS_CMD: &str = "depth.status";
pub const DOWNLOAD_CMD: &str = "depth.downloadModel";
pub const PHOTO_CMD: &str = "clip.photo3d";
pub const BLUR_CMD: &str = "clip.depthBlur";
/// The nearest part's travel either way at an amount of 50, % of the width.
pub const TRAVEL: f64 = 2.5;
/// The nearest part's push at an amount of 50, %.
pub const PUSH: f64 = 10.0;
/// Keyframes of a `circle` (a whole turn).
const CIRCLE_KEYS: usize = 16;
/// The long side of the picture the commands look at.
const LOOK_SIDE: u32 = filmcraft_render::depth::ANALYSIS_SIDE as u32;

pub fn commands() -> Vec<CommandSpec> {
    let spec = |id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| CommandSpec {
        id,
        label,
        menu: &["Clip", "Depth"],
        shortcut: None,
        params,
        enabled: has_selection,
        run,
        journal: true,
    };
    vec![
        CommandSpec { id: STATUS_CMD, label: "Depth Model", menu: &[], shortcut: None, params: "{}", enabled: |_| Ok(()), run: status, journal: false },
        CommandSpec {
            id: DOWNLOAD_CMD,
            label: "Download the Depth Model",
            menu: &[],
            shortcut: None,
            params: r#"{"wait":bool=false}"#,
            enabled: |_| Ok(()),
            run: download,
            journal: false,
        },
        spec(
            PHOTO_CMD,
            "3D Photo",
            r#"{"clips":[id]?,"move":"auto"|"in"|"out"|"left"|"right"|"up"|"down"|"circle"="auto","amount":0..100=50,"pivot":"auto"|"far"|"subject"|0..1="auto","subject":[x,y,w,h]?}"#,
            photo_3d,
        ),
        spec(
            BLUR_CMD,
            "Depth Blur",
            r#"{"clips":[id]?,"focus":"subject"|"near"|"far"|0..1="subject","blur":0..10=1.5,"range":0..1=0.1,"pull":bool=false,"subject":[x,y,w,h]?}"#,
            depth_blur,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

/// A stand-in model (tests), used before the downloaded network.
static STAND_IN: Mutex<Option<Arc<dyn DepthModel>>> = Mutex::new(None);

/// Use `m` instead of the downloaded network (tests and tools).
pub fn use_stand_in(m: Arc<dyn DepthModel>) {
    *STAND_IN.lock().unwrap_or_else(PoisonError::into_inner) = Some(m);
    install();
}

/// Whether depth can be made: a stand-in, or the network downloaded in a build that has it.
pub fn available() -> bool {
    STAND_IN.lock().unwrap_or_else(PoisonError::into_inner).is_some() || installed()
}

/// The network is downloaded (and this build can run it).
pub fn installed() -> bool {
    cfg!(feature = "privacy-models") && crate::transcript::models_dir().as_deref().is_some_and(filmcraft_vision::catalog::depth_installed)
}

/// The model the render asks: the stand-in, else the kept answers and the network.
struct Forward;

impl DepthModel for Forward {
    fn depth(&self, key: u64, rgba: &[u8], w: usize, h: usize) -> Option<DepthField> {
        let stand_in = STAND_IN.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(m) = stand_in {
            return m.depth(key, rgba, w, h);
        }
        if let Some(f) = kept(key) {
            return Some(f);
        }
        let f = network_depth(rgba, w, h)?;
        keep(key, &f);
        Some(f)
    }
}

/// Make the render's depth effects ask the engine's model (once; the network loads on first use).
pub fn install() {
    filmcraft_render::depth::set_model(Arc::new(Forward));
}

#[cfg(feature = "privacy-models")]
fn network_depth(rgba: &[u8], w: usize, h: usize) -> Option<DepthField> {
    use filmcraft_vision::depth::DepthEstimator;
    static NET: Mutex<Option<Arc<DepthEstimator>>> = Mutex::new(None);
    let net = {
        let mut slot = NET.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let dir = crate::transcript::models_dir()?;
            match DepthEstimator::load(&dir, true) {
                Ok(n) => {
                    log::info!("depth network loaded on the {}", n.device_name());
                    *slot = Some(Arc::new(n));
                }
                Err(filmcraft_vision::VisionError::NotInstalled) => return None,
                Err(e) => {
                    log::warn!("the depth network does not load: {e}");
                    return None;
                }
            }
        }
        slot.clone()?
    };
    // (the network is the last thing that could panic in a render: it must not take the frame)
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| net.estimate(rgba, w, h))) {
        Ok(Ok(m)) => DepthField::new(m.w, m.h, m.near),
        Ok(Err(e)) => {
            log::warn!("depth: {e}");
            None
        }
        Err(_) => {
            log::error!("depth: the network panicked; the picture is left flat");
            None
        }
    }
}

#[cfg(not(feature = "privacy-models"))]
fn network_depth(_: &[u8], _: usize, _: usize) -> Option<DepthField> {
    None
}

/// Where a picture's depth is kept.
fn kept_path(key: u64) -> Option<PathBuf> {
    let id = filmcraft_vision::catalog::depth_id();
    let model = id.split('=').nth(1).unwrap_or("model");
    crate::autosave::default_data_dir().map(|d| d.join("cache").join("depth").join(format!("{key:016x}-{model}.depth")))
}

/// The kept depth's header, then its width and height (u32 LE) and its rows of 16-bit values as
/// differences from the value before (wrapping), deflated (zlib).
const KEPT_MAGIC: &[u8; 8] = b"FCDEPTH1";
/// The largest kept depth read back, a side.
const KEPT_SIDE: usize = 4096;

/// A depth field as kept on disk.
pub fn encode_kept(f: &DepthField) -> Option<Vec<u8>> {
    let (w, h) = (u32::try_from(f.w).ok()?, u32::try_from(f.h).ok()?);
    let mut raw = Vec::with_capacity(f.near.len() * 2);
    for row in f.near.chunks(f.w.max(1)) {
        let mut last = 0u16;
        for v in row {
            let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            raw.extend(q.wrapping_sub(last).to_le_bytes());
            last = q;
        }
    }
    let mut out = KEPT_MAGIC.to_vec();
    out.extend(w.to_le_bytes());
    out.extend(h.to_le_bytes());
    out.extend(miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6));
    Some(out)
}

/// A kept depth field read back (`None` for anything that is not one, or too large).
pub fn decode_kept(bytes: &[u8]) -> Option<DepthField> {
    let rest = bytes.strip_prefix(KEPT_MAGIC.as_slice())?;
    let word = |i: usize| rest.get(i..i + 4).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(u32::from_le_bytes);
    let (w, h) = (usize::try_from(word(0)?).ok()?, usize::try_from(word(4)?).ok()?);
    if w == 0 || h == 0 || w > KEPT_SIDE || h > KEPT_SIDE {
        return None;
    }
    let n = w * h * 2;
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(rest.get(8..)?, n).ok()?;
    if raw.len() != n {
        return None;
    }
    let mut near = Vec::with_capacity(w * h);
    for row in raw.chunks_exact(w * 2) {
        let mut last = 0u16;
        for d in row.as_chunks::<2>().0 {
            last = last.wrapping_add(u16::from_le_bytes(*d));
            near.push(f32::from(last) / 65535.0);
        }
    }
    DepthField::new(w, h, near)
}

fn kept(key: u64) -> Option<DepthField> {
    let path = kept_path(key)?;
    // (a kept file is at most its header and a deflated 4096² map: far under 64 MiB)
    if std::fs::metadata(&path).ok()?.len() > 64 << 20 {
        return None;
    }
    decode_kept(&std::fs::read(&path).ok()?)
}

fn keep(key: u64, f: &DepthField) {
    let (Some(path), Some(bytes)) = (kept_path(key), encode_kept(f)) else { return };
    if let Some(dir) = path.parent()
        && std::fs::create_dir_all(dir).is_ok()
    {
        let tmp = path.with_extension("part");
        if std::fs::write(&tmp, bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

/// The depth of a project item's picture (its first frame), through the same model and keeping.
fn item_depth(s: &Session, item: filmcraft_project::ItemId, cmd: &str) -> Result<DepthField> {
    let src = s.source(item).ok_or_else(|| bad(cmd, format!("no picture for item {}", item.0)))?;
    let info = src.info().video.as_ref().map_or((0, 0), |v| (v.width, v.height));
    let scale = (f64::from(LOOK_SIDE) / f64::from(info.0.max(info.1).max(1))).min(1.0) as f32;
    let f = src.video_frame(filmcraft_media::FrameRequest { time: Tick::ZERO, scale }).map_err(|e| bad(cmd, format!("the photo does not decode: {e}")))?;
    let (rgba, w, h) = crate::scenes::small_rgba(&f, LOOK_SIDE);
    let (w, h) = (w as usize, h as usize);
    let key = filmcraft_render::depth::picture_key(&rgba, w, h);
    Forward.depth(key, &rgba, w, h).ok_or_else(|| bad(cmd, "the depth network cannot say how far this picture's parts are"))
}

// ---------------------------------------------------------------------------------------------
// the commands
// ---------------------------------------------------------------------------------------------

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_vision::catalog as c;
    Ok(json!({
        "available": cfg!(feature = "privacy-models"),
        "installed": installed(),
        "name": c::DEPTH_NAME,
        "bytes": c::DEPTH.size,
        "license": c::DEPTH_LICENSE,
        "source": c::DEPTH_SOURCE,
        "training": c::DEPTH_TRAINING,
        "download": DOWNLOAD_CMD,
        "gpu": filmcraft_render::depth::on_gpu(),
    }))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::depth_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "depth is not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::DEPTH_NAME));
    job.progress.total.store(filmcraft_vision::catalog::DEPTH.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-depth-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Depth Anything V2 Small installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::DEPTH.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::DEPTH_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::depth_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("depth is not available in this build".into())
}

/// The 3D photo's moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    In,
    Out,
    Left,
    Right,
    Up,
    Down,
    Circle,
}

impl Move {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "in" => Self::In,
            "out" => Self::Out,
            "left" => Self::Left,
            "right" => Self::Right,
            "up" => Self::Up,
            "down" => Self::Down,
            "circle" | "orbit" => Self::Circle,
            _ => return None,
        })
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Left => "left",
            Self::Right => "right",
            Self::Up => "up",
            Self::Down => "down",
            Self::Circle => "circle",
        }
    }
    /// Whether it turns round the subject (else it moves in depth with the farthest still).
    fn round_subject(self) -> bool {
        !matches!(self, Self::In | Self::Out)
    }
}

/// The move the clip's own Ken Burns makes: in or out when Motion ▸ Scale grows or shrinks, else
/// the way its Position pans (the picture moving left: the camera going right), else none.
pub fn ken_burns_move(it: &filmcraft_project::TrackItem) -> Option<Move> {
    let m = it.effect("motion")?;
    let ends = |id: &str| m.param(id).and_then(|p| Some((p.keyframes.first()?.value.clone(), p.keyframes.last()?.value.clone())));
    if let Some((ParamValue::Float(a), ParamValue::Float(b))) = ends("scale")
        && (b / a.max(1e-9) - 1.0).abs() > 0.01
    {
        return Some(if b > a { Move::In } else { Move::Out });
    }
    if let Some((ParamValue::Vec2(a), ParamValue::Vec2(b))) = ends("position") {
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        if dx.abs().max(dy.abs()) > 1.0 {
            return Some(if dx.abs() >= dy.abs() {
                if dx < 0.0 { Move::Right } else { Move::Left }
            } else if dy < 0.0 {
                Move::Down
            } else {
                Move::Up
            });
        }
    }
    None
}

/// `subject`: `[x, y, w, h]` or `{x, y, w, h}`, each 0..1, a box inside the picture.
fn subject_p(p: &Value, cmd: &str) -> Result<Option<[f64; 4]>> {
    let v = match p.get("subject") {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let nums: Option<Vec<f64>> = match v {
        Value::Array(a) => a.iter().map(Value::as_f64).collect(),
        Value::Object(o) => ["x", "y", "w", "h"].iter().map(|k| o.get(*k).and_then(Value::as_f64)).collect(),
        _ => None,
    };
    let why = "`subject` is the box [x, y, w, h], each 0..1 of the picture";
    let Some([x, y, w, h]) = nums.as_deref().and_then(|n| <[f64; 4]>::try_from(n).ok()) else { return Err(bad(cmd, why)) };
    let inside = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
    if !(inside(x) && inside(y) && inside(w) && inside(h)) || w <= 0.0 || h <= 0.0 || x + w > 1.0 + 1e-6 || y + h > 1.0 + 1e-6 {
        return Err(bad(cmd, why));
    }
    Ok(Some([x, y, w.min(1.0 - x), h.min(1.0 - y)]))
}

/// The `q` quantile (0..1) of the field's values inside `[x, y, w, h]` (shares of the picture).
fn quantile(f: &DepthField, area: [f64; 4], q: f64) -> f32 {
    let (fw, fh) = (f.w as f64, f.h as f64);
    let (x0, y0) = ((area[0] * fw).floor().max(0.0) as usize, (area[1] * fh).floor().max(0.0) as usize);
    let (x1, y1) = (((area[0] + area[2]) * fw).ceil().min(fw) as usize, ((area[1] + area[3]) * fh).ceil().min(fh) as usize);
    let mut v: Vec<f32> =
        (y0..y1.max(y0 + 1).min(f.h)).flat_map(|y| (x0..x1.max(x0 + 1).min(f.w)).filter_map(move |x| f.near.get(y * f.w + x).copied())).collect();
    if v.is_empty() {
        return 0.5;
    }
    let k = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

/// The subject's depth: the median in its box, else the 80th percentile of the middle 60 %.
pub fn subject_depth(f: &DepthField, subject: Option<[f64; 4]>) -> f32 {
    match subject {
        Some(b) => quantile(f, b, 0.5),
        None => quantile(f, [0.2, 0.2, 0.6, 0.6], 0.8),
    }
}

/// The photo clips asked for (`clips`, else the selection): `(clip, item, media times of the
/// first and last frame)`. 360° clips and clips that move are left out, with what was left out.
fn photo_clips(s: &Session, p: &Value, cmd: &str) -> Result<Vec<(ClipId, filmcraft_project::ItemId, (Tick, Tick), Option<Move>)>> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let fd = q.settings.frame_rate.sane().frame_duration().max(Tick(1));
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for c in clips_p(s, p) {
        let Some((_, it)) = q.find_item(c) else { continue };
        if !q.video_tracks.iter().any(|t| t.item(c).is_some()) {
            continue;
        }
        let still = matches!(s.project.item(it.item).map(|i| &i.kind), Some(ItemKind::Media(m)) if m.info.kind == filmcraft_media::MediaKind::Still);
        if !still || crate::vr360::is_360(s, it.item) {
            skipped.push(it.name.clone());
            continue;
        }
        let last = it.start + (it.duration - fd).max(Tick::ZERO);
        out.push((c, it.item, (it.effect_time_at(it.start), it.effect_time_at(last)), ken_burns_move(it)));
    }
    if out.is_empty() {
        let why = if skipped.is_empty() {
            "select a photo clip".to_string()
        } else {
            format!("depth is for photos (not videos, whose depth flickers from frame to frame, nor 360° clips): {}", skipped.join(", "))
        };
        return Err(bad(cmd, why));
    }
    Ok(out)
}

fn key(t: Tick, v: f64, interp: Interpolation) -> Keyframe {
    let mut k = Keyframe::new(t, ParamValue::Float(v));
    k.interp = interp;
    k
}

/// `id` of `e` from `a` at `t0` to `b` at `t1`, eased like a Ken Burns move (or `a` alone).
fn ease(e: &mut EffectInstance, id: &str, (t0, t1): (Tick, Tick), a: f64, b: f64) {
    let mut prm = Param::new(ParamValue::Float(a));
    if t1 > t0 && (a - b).abs() > 1e-12 {
        prm.keyframes = vec![key(t0, a, Interpolation::EaseOut), key(t1, b, Interpolation::EaseIn)];
    }
    e.params.insert(id.to_string(), prm);
}

/// The clip's effect `id`, a new one added when it has none (its parameters back to defaults).
fn fresh_effect<'a>(it: &'a mut filmcraft_project::TrackItem, id: &str) -> Option<&'a mut EffectInstance> {
    let def = filmcraft_project::find_effect(id)?;
    match it.effects.iter().position(|e| e.effect == id) {
        Some(i) => {
            let e = it.effects.get_mut(i)?;
            *e = def.instance();
            Some(e)
        }
        None => {
            it.effects.push(def.instance());
            it.effects.last_mut()
        }
    }
}

fn photo_3d(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = PHOTO_CMD;
    let asked = match str_p(p, "move") {
        None => None,
        Some(m) if m.trim().eq_ignore_ascii_case("auto") => None,
        Some(m) => Some(Move::parse(m).ok_or_else(|| bad(cmd, "`move` is auto, in, out, left, right, up, down or circle"))?),
    };
    if p.get("move").is_some_and(|v| !v.is_string()) {
        return Err(bad(cmd, "`move` is a word"));
    }
    let amount = match f64_p(p, "amount") {
        None => 50.0,
        Some(a) if a.is_finite() && (0.0..=100.0).contains(&a) => a,
        Some(_) => return Err(bad(cmd, "`amount` is 0 to 100")),
    };
    enum Pivot {
        Auto,
        Far,
        Subject,
        At(f64),
    }
    let pivot = match p.get("pivot") {
        None | Some(Value::Null) => Pivot::Auto,
        Some(Value::String(w)) => match w.trim().to_ascii_lowercase().as_str() {
            "auto" => Pivot::Auto,
            "far" => Pivot::Far,
            "subject" => Pivot::Subject,
            _ => return Err(bad(cmd, "`pivot` is auto, far, subject or a depth 0..1")),
        },
        Some(v) => match v.as_f64() {
            Some(d) if d.is_finite() && (0.0..=1.0).contains(&d) => Pivot::At(d),
            _ => return Err(bad(cmd, "`pivot` is auto, far, subject or a depth 0..1")),
        },
    };
    let subject = subject_p(p, cmd)?;
    if !available() {
        return Err(bad(cmd, format!("the depth network is not downloaded yet ({DOWNLOAD_CMD}, 99 MB)")));
    }
    let clips = photo_clips(s, p, cmd)?;
    // each clip's move and still depth (the subject's from its depth)
    let mut work = Vec::new();
    for (c, item, times, own) in clips {
        let mv = asked.or(own).unwrap_or(Move::In);
        let at = match pivot {
            Pivot::Far => 0.0,
            Pivot::At(d) => d,
            Pivot::Auto if !mv.round_subject() => 0.0,
            Pivot::Auto | Pivot::Subject => f64::from(subject_depth(&item_depth(s, item, cmd)?, subject)),
        };
        work.push((c, times, mv, at));
    }
    install();
    let k = amount / 50.0;
    let (travel, push) = (TRAVEL * k, PUSH * k);
    let out = s.edit_sequence("3D Photo", |q, _, _| {
        let mut out = Vec::new();
        for (c, (t0, t1), mv, pivot) in &work {
            let Some((_, it)) = q.find_item_mut(*c) else { continue };
            let Some(e) = fresh_effect(it, PHOTO_3D) else { continue };
            e.params.insert("pivot".into(), Param::new(ParamValue::Float(pivot * 100.0)));
            match mv {
                Move::In => ease(e, "push", (*t0, *t1), 0.0, push),
                Move::Out => ease(e, "push", (*t0, *t1), push, 0.0),
                // the camera going right: the near parts go left past the subject
                Move::Right => ease(e, "travel_x", (*t0, *t1), travel, -travel),
                Move::Left => ease(e, "travel_x", (*t0, *t1), -travel, travel),
                Move::Down => ease(e, "travel_y", (*t0, *t1), travel, -travel),
                Move::Up => ease(e, "travel_y", (*t0, *t1), -travel, travel),
                Move::Circle => {
                    let span = (*t1 - *t0).0.max(0);
                    for (id, phase, r) in [("travel_x", 0.0, travel), ("travel_y", std::f64::consts::FRAC_PI_2, travel * 0.6)] {
                        let mut prm = Param::new(ParamValue::Float(r * phase.cos()));
                        if span > 0 {
                            prm.keyframes = (0..=CIRCLE_KEYS)
                                .map(|i| {
                                    let f = i as f64 / CIRCLE_KEYS as f64;
                                    let a = std::f64::consts::TAU * f + phase;
                                    key(*t0 + Tick((span as f64 * f).round() as i64), r * a.cos(), Interpolation::Linear)
                                })
                                .collect();
                        }
                        e.params.insert(id.into(), prm);
                    }
                }
            }
            out.push(json!({"clip": c.0, "move": mv.id(), "pivot": (pivot * 1000.0).round() / 1000.0}));
        }
        Ok(out)
    })?;
    Ok(json!({"clips": out, "gpu": filmcraft_render::depth::on_gpu()}))
}

fn depth_blur(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = BLUR_CMD;
    enum Focus {
        Subject,
        Near,
        Far,
        At(f64),
    }
    let focus = match p.get("focus") {
        None | Some(Value::Null) => Focus::Subject,
        Some(Value::String(w)) => match w.trim().to_ascii_lowercase().as_str() {
            "subject" => Focus::Subject,
            "near" => Focus::Near,
            "far" => Focus::Far,
            _ => return Err(bad(cmd, "`focus` is subject, near, far or a depth 0..1")),
        },
        Some(v) => match v.as_f64() {
            Some(d) if d.is_finite() && (0.0..=1.0).contains(&d) => Focus::At(d),
            _ => return Err(bad(cmd, "`focus` is subject, near, far or a depth 0..1")),
        },
    };
    let blur = match f64_p(p, "blur") {
        None => 1.5,
        Some(b) if b.is_finite() && (0.0..=10.0).contains(&b) => b,
        Some(_) => return Err(bad(cmd, "`blur` is the farthest blur, 0 to 10 % of the width")),
    };
    let range = match f64_p(p, "range") {
        None => 0.1,
        Some(r) if r.is_finite() && (0.0..=1.0).contains(&r) => r,
        Some(_) => return Err(bad(cmd, "`range` is the depth of field, 0..1")),
    };
    let pull = bool_p(p, "pull").unwrap_or(false);
    let subject = subject_p(p, cmd)?;
    if !available() {
        return Err(bad(cmd, format!("the depth network is not downloaded yet ({DOWNLOAD_CMD}, 99 MB)")));
    }
    let clips = photo_clips(s, p, cmd)?;
    let mut work = Vec::new();
    for (c, item, times, _) in clips {
        let at = match focus {
            Focus::At(d) => d,
            Focus::Near | Focus::Far | Focus::Subject => {
                let f = item_depth(s, item, cmd)?;
                f64::from(match focus {
                    Focus::Near => quantile(&f, [0.0, 0.0, 1.0, 1.0], 0.95),
                    Focus::Far => quantile(&f, [0.0, 0.0, 1.0, 1.0], 0.05),
                    _ => subject_depth(&f, subject),
                })
            }
        };
        work.push((c, times, at));
    }
    install();
    let out = s.edit_sequence("Depth Blur", |q, _, _| {
        let mut out = Vec::new();
        for (c, (t0, t1), at) in &work {
            let Some((_, it)) = q.find_item_mut(*c) else { continue };
            let Some(e) = fresh_effect(it, DEPTH_BLUR) else { continue };
            e.params.insert("blur".into(), Param::new(ParamValue::Float(blur)));
            e.params.insert("range".into(), Param::new(ParamValue::Float(range * 100.0)));
            let focus = at * 100.0;
            if pull && *t1 > *t0 {
                // from the other end of the depth to the focus, over the first half
                let from = if *at >= 0.5 { 0.0 } else { 100.0 };
                ease(e, "focus", (*t0, *t0 + Tick((*t1 - *t0).0 / 2)), from, focus);
            } else {
                e.params.insert("focus".into(), Param::new(ParamValue::Float(focus)));
            }
            out.push(json!({"clip": c.0, "focus": (at * 1000.0).round() / 1000.0, "pull": pull}));
        }
        Ok(out)
    })?;
    Ok(json!({"clips": out, "gpu": filmcraft_render::depth::on_gpu()}))
}

#[cfg(test)]
#[path = "depth_tests.rs"]
mod tests;
