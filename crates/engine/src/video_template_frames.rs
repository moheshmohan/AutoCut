//! Frames of video templates, for the template browser and agents: thumbnails rendered from a
//! template's own placeholder sequence, previews of a template filled with the project's media
//! (made in a scratch copy, the project untouched), and the project media that can fill a slot.
//!
//! | command | does |
//! |---|---|
//! | `template.thumbnail` | template `template` applied unfilled in layout `variant` (default: its own), drawn at `seconds` (default: its poster frame), the long side at most `maxSide` (default 320): PNG as `pngBase64`, or written to `path` |
//! | `template.preview` | the template filled with `slots` (project item ids, text, cues — as `template.apply` takes them, files must be imported first), `frames` (1–24, default 6) frames spread over it as PNGs (`pngBase64` each, or files in `dir`); the project does not change |
//! | `template.media` | the project's media that can fill a slot of `kind`: id, name, kind (`video`, `photo`, `audio`), size, length, sound, whether it is a Google Photos pick (`googlePhotos: true` keeps only those) |
//!
//! Thumbnails are cached (by template, its manifest's hash, layout, time and size), so the browser
//! draws each once.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::{ItemId, ItemKind, MediaRef, Project};
use filmcraft_templates::apply::{self as tapply, ApplyOptions, SlotFill};
use filmcraft_templates::manifest::{Look, SlotKind};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, str_p, u64_p};
use crate::media_pool::MediaPool;
use crate::{EngineError, Result, Session};

/// Largest `maxSide`.
pub const MAX_SIDE: u32 = 1920;
/// The thumbnail size the browser asks for.
pub const THUMB_SIDE: u32 = 320;
/// Most preview frames.
pub const MAX_FRAMES: usize = 24;
/// Thumbnails kept.
const CACHE_CAP: usize = 96;

/// A rendered frame: straight sRGB RGBA8, opaque.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Its time in the applied template.
    pub at: Tick,
}

/// A project item that can fill a slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub item: ItemId,
    pub name: String,
    /// `video`, `photo` or `audio`.
    pub kind: &'static str,
    pub width: u32,
    pub height: u32,
    pub seconds: f64,
    pub has_audio: bool,
    /// Downloaded by the Google Photos picker (its file is in the picker's download cache).
    pub google_photos: bool,
}

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "template.thumbnail",
            "Video Template Thumbnail",
            r#"{"template":id|name|path,"variant":"16:9"|"9:16"|…?,"seconds":f64?,"maxSide":u32=320,"path":str?}"#,
            thumbnail_cmd,
        ),
        spec(
            "template.preview",
            "Preview Video Template",
            r#"{"template":id|name|path,"variant":str?,"slots":{"<slot>": itemId|text|{"item","in"?,"focus"?}|[…]}?,"frames":1..24=6,"maxSide":u32=320,"dir":str?}"#,
            preview_cmd,
        ),
        spec("template.media", "Media for Template Slots", r#"{"kind":"video"|"photo"|"logo"|"music"?,"googlePhotos":bool?}"#, media_cmd),
    ]
}

fn cache() -> &'static Mutex<Vec<(String, Arc<Frame>)>> {
    static C: OnceLock<Mutex<Vec<(String, Arc<Frame>)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Vec::new()))
}

/// The look as an effect for a scratch render (a Lumetri preset; effect presets are the
/// session's and are left out of previews).
fn look_effect(l: Option<&Look>) -> Option<filmcraft_project::EffectInstance> {
    match l {
        Some(Look::Lumetri(n)) => filmcraft_render::lumetri_presets::find(n).map(|p| p.instance()),
        _ => None,
    }
}

/// Ids of what a preview adds to its scratch copy of the project start this far past the
/// project's own, so the session's media pool (which caches sources by item) never confuses
/// them with items the project makes later.
const SCRATCH_IDS: u64 = 1 << 40;

/// Apply `pkg` into `scratch` with `opts` and draw it at the times `times` picks (from the
/// applied length) with `pool`'s sources, the long side at most `max_side`.
fn render_applied(
    s: &Session,
    mut scratch: Project,
    pool: Arc<MediaPool>,
    pkg: &filmcraft_templates::Package,
    opts: &ApplyOptions,
    times: impl Fn(Tick) -> Vec<Tick>,
    max_side: u32,
) -> Result<(Vec<Frame>, Vec<String>)> {
    scratch.next_id = scratch.next_id.saturating_add(SCRATCH_IDS);
    let a = tapply::apply(&mut scratch, pkg, opts).map_err(|e| EngineError::Other(e.to_string()))?;
    let seq = scratch.sequence(a.sequence).ok_or(EngineError::NoSequence)?;
    let (w, h, rate) = (seq.settings.width, seq.settings.height, seq.settings.frame_rate);
    let scale = (f64::from(max_side.clamp(16, MAX_SIDE)) / f64::from(w.max(h).max(1))).clamp(0.01, 1.0) as f32;
    let dur = a.duration;
    let last = (dur - rate.frame_duration()).max(Tick::ZERO);
    let project = Arc::new(scratch);
    let provider = pool.provider(project.clone(), s.services.clone());
    let opts = filmcraft_render::RenderOptions { scale, captions: true, ..Default::default() };
    let mut frames = Vec::new();
    for t in times(dur).into_iter().take(MAX_FRAMES) {
        let t = rate.snap(t.max(Tick::ZERO).min(last));
        let img = filmcraft_render::render_sequence(&project, a.sequence, t, opts, &provider);
        let (fw, fh) = (u32::try_from(img.w).unwrap_or(0), u32::try_from(img.h).unwrap_or(0));
        if fw == 0 || fh == 0 {
            return Err(EngineError::Other("the template rendered an empty frame".into()));
        }
        frames.push(Frame { rgba: img.over_black_rgba8(), width: fw, height: fh, at: t });
    }
    Ok((frames, a.warnings))
}

/// A thumbnail of template `key` (id, name or path): its placeholder sequence in layout
/// `variant`, at `at` (default: its poster frame), the long side at most `max_side`. Cached.
pub fn thumbnail(s: &Session, key: &str, variant: Option<&str>, at: Option<Tick>, max_side: u32) -> Result<Arc<Frame>> {
    const CMD: &str = "template.thumbnail";
    let e = crate::video_templates::find_template(s, key, CMD)?;
    let m = &e.package.manifest;
    let v = m.variant(variant).ok_or_else(|| {
        let ids: Vec<&str> = m.variants.iter().map(|v| v.id.as_str()).collect();
        bad(CMD, format!("the template has no variant `{}` (its variants: {})", variant.unwrap_or(""), ids.join(", ")))
    })?;
    let t = at.unwrap_or_else(|| m.poster_time()).max(Tick::ZERO).min(m.duration);
    let side = max_side.clamp(16, MAX_SIDE);
    let hash = filmcraft_templates::sha256_hex(m.to_json().as_bytes());
    let k = format!("{}|{}|{}|{}|{}", m.id, v.id, side, t.0, hash.get(..16).unwrap_or(""));
    if let Some(f) = cache().lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(x, _)| *x == k).map(|(_, f)| f.clone()) {
        return Ok(f);
    }
    let opts = ApplyOptions { variant: Some(v.id.clone()), look: look_effect(m.look.as_ref()), ..Default::default() };
    // a pool of its own: the template's generated placeholders only
    let pool = Arc::new(MediaPool::default());
    let (mut frames, _) = render_applied(s, Project::new("Template Thumbnail"), pool, &e.package, &opts, |_| vec![t], side)?;
    let f = Arc::new(frames.pop().ok_or_else(|| EngineError::Other("no frame".into()))?);
    let mut c = cache().lock().unwrap_or_else(PoisonError::into_inner);
    c.retain(|(x, _)| *x != k);
    c.push((k, f.clone()));
    if c.len() > CACHE_CAP {
        let extra = c.len() - CACHE_CAP;
        c.drain(..extra);
    }
    Ok(f)
}

/// `n` frames of template `key` filled with `fills` (the project's items, text, cues), made in a
/// scratch copy of the project. Returns the frames and the apply's warnings.
pub fn preview(
    s: &Session,
    key: &str,
    variant: Option<&str>,
    fills: std::collections::BTreeMap<String, SlotFill>,
    n: usize,
    max_side: u32,
) -> Result<(Vec<Frame>, Vec<String>)> {
    let e = crate::video_templates::find_template(s, key, "template.preview")?;
    let m = &e.package.manifest;
    let opts = ApplyOptions { variant: variant.map(str::to_string), fills, look: look_effect(m.look.as_ref()), ..Default::default() };
    let n = n.clamp(1, MAX_FRAMES);
    let spread = move |dur: Tick| -> Vec<Tick> { (0..n).map(|k| Tick(((dur.0 as f64) * (k as f64 + 0.5) / n as f64) as i64)).collect() };
    // the session's pool: the project's media decode as they do in the monitors
    render_applied(s, (*s.project).clone(), s.media.clone(), &e.package, &opts, spread, max_side)
}

fn png_json(s: &Session, f: &Frame, path: Option<&str>, cmd: &str) -> Result<Value> {
    let png = filmcraft_export::encode_png(f.rgba.clone(), f.width, f.height).map_err(|e| bad(cmd, e.to_string()))?;
    let mut out = json!({"width": f.width, "height": f.height, "time": f.at.0, "seconds": f.at.seconds()});
    match path {
        Some(p) => {
            s.services.write_file(p, &png).map_err(|e| bad(cmd, format!("{p}: {e}")))?;
            out["path"] = json!(p);
        }
        None => out["pngBase64"] = json!(filmcraft_project::gtemplate::base64_encode(&png)),
    }
    Ok(out)
}

fn side_p(p: &Value, cmd: &str) -> Result<u32> {
    match u64_p(p, "maxSide") {
        None if p.get("maxSide").is_some_and(|v| !v.is_null()) => Err(bad(cmd, "`maxSide` is a whole number of pixels")),
        None => Ok(THUMB_SIDE),
        Some(n) if (16..=u64::from(MAX_SIDE)).contains(&n) => Ok(n as u32),
        Some(n) => Err(bad(cmd, format!("`maxSide` {n} is out of range (16 to {MAX_SIDE})"))),
    }
}

fn thumbnail_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.thumbnail";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).unwrap_or("");
    let at = match (p.get("seconds").filter(|v| !v.is_null()), f64_p(p, "seconds")) {
        (None, _) => None,
        (Some(_), Some(x)) if x.is_finite() && (0.0..=86_400.0).contains(&x) => Some(Tick::from_seconds_f64(x)),
        _ => return Err(bad(CMD, "`seconds` is a time in the template, 0 or more")),
    };
    let side = side_p(p, CMD)?;
    let f = thumbnail(s, key, str_p(p, "variant"), at, side)?;
    let mut out = png_json(s, &f, str_p(p, "path"), CMD)?;
    out["template"] = json!(key);
    Ok(out)
}

fn preview_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.preview";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).unwrap_or("");
    let e = crate::video_templates::find_template(s, key, CMD)?;
    let wanted = crate::video_templates::read_slots(&e.package.manifest, p, CMD)?;
    let fills = crate::video_templates::fills_without_import(&e.package.manifest, wanted, CMD)?;
    let n = match u64_p(p, "frames") {
        None => 6,
        Some(n) if (1..=MAX_FRAMES as u64).contains(&n) => n as usize,
        Some(n) => return Err(bad(CMD, format!("`frames` {n} is out of range (1 to {MAX_FRAMES})"))),
    };
    let side = side_p(p, CMD)?;
    let variant = str_p(p, "variant");
    if let Some(v) = variant
        && e.package.manifest.variant(Some(v)).is_none()
    {
        return Err(bad(CMD, format!("the template has no variant `{v}`")));
    }
    let (frames, warnings) = preview(s, key, variant, fills, n, side).map_err(|e| bad(CMD, e.to_string()))?;
    let dir = str_p(p, "dir").map(crate::export_tools::expand_home);
    if let Some(d) = &dir
        && !cfg!(target_arch = "wasm32")
    {
        std::fs::create_dir_all(d).map_err(|e| bad(CMD, format!("{d}: {e}")))?;
    }
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let path = dir.as_ref().map(|d| std::path::Path::new(d).join(format!("preview-{:02}.png", i + 1)).to_string_lossy().into_owned());
        out.push(png_json(s, f, path.as_deref(), CMD)?);
    }
    Ok(json!({"template": e.package.manifest.id, "frames": out, "warnings": warnings}))
}

/// Whether `path` is in the Google Photos picker's download cache
/// (`<data dir>/google-photos/cache/<media item id>/<file>`).
pub fn is_google_photos_pick(path: &str) -> bool {
    let p = path.replace('\\', "/");
    p.contains("/google-photos/cache/")
}

/// The project's media that can fill a slot of `kind` (any media without one), in project order.
pub fn candidates(s: &Session, kind: Option<SlotKind>) -> Vec<Candidate> {
    let mut out = Vec::new();
    for it in s.project.items.values() {
        let ItemKind::Media(m) = &it.kind else { continue };
        let video = m.info.video.as_ref();
        let audio = m.info.has_audio();
        let still = matches!(m.info.kind, filmcraft_media::MediaKind::Still);
        let k = match (video, still) {
            (Some(_), true) => "photo",
            (Some(_), false) => "video",
            (None, _) if audio => "audio",
            (None, _) => continue,
        };
        let fits = match kind {
            None => true,
            Some(SlotKind::Music) => audio,
            Some(SlotKind::Video | SlotKind::Photo | SlotKind::Logo) => video.is_some(),
            Some(SlotKind::Text | SlotKind::Captions) => false,
        };
        if !fits {
            continue;
        }
        let google_photos = matches!(&m.media, MediaRef::File { path } if is_google_photos_pick(path));
        out.push(Candidate {
            item: it.id,
            name: it.name.clone(),
            kind: k,
            width: video.map_or(0, |v| v.width),
            height: video.map_or(0, |v| v.height),
            seconds: if still { 0.0 } else { m.info.duration.seconds() },
            has_audio: audio,
            google_photos,
        });
    }
    out
}

fn media_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.media";
    let kind = match str_p(p, "kind") {
        Some(k) => Some(SlotKind::from_id(k).ok_or_else(|| bad(CMD, format!("unknown slot kind `{k}`")))?),
        None => None,
    };
    let only_gp = bool_p(p, "googlePhotos").unwrap_or(false);
    let v: Vec<Value> = candidates(s, kind)
        .into_iter()
        .filter(|c| !only_gp || c.google_photos)
        .map(|c| {
            json!({"item": c.item.0, "name": c.name, "kind": c.kind, "width": c.width, "height": c.height, "seconds": c.seconds, "hasAudio": c.has_audio, "googlePhotos": c.google_photos})
        })
        .collect();
    Ok(json!({"media": v}))
}
