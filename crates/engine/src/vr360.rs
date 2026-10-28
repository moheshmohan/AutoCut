//! 360° video in the edit (`filmcraft_project::vr`, `filmcraft_render::vr`).
//!
//! | command | does |
//! |---|---|
//! | `clip.reframe360` | looks at 360° clips (`clips`, default the selection, else every clip of 360° media) through a virtual camera (the `vr_reframe` effect, added when missing, framing the sequence's frame): `yaw` (turn right), `pitch` (look up), `roll`, `fov` (degrees across) and `distance` (0 flat … 1 tiny planet) set, or keyframed `at` a clip time, or a whole camera move as `keys`; a `preset` (`natural`, `wide`, `tinyPlanet`, `rabbitHole`, `sphere`) first; one undo step |
//! | `media.vrProperties` | Interpret Footage ▸ VR Properties: `projection` `auto` (what the file says), `equirectangular`, `dualFisheye` or `none`; equirectangular `layout` and `capturedView`; a dual fisheye's `halfFov` (alone, they change the item's own); `level` levels the horizon by the camera's motion sensor and `steady` steadies its heading (both on by themselves for Insta360 recordings); `seamFlow` (on by itself) stitches near things on a dual fisheye's seam by the lenses' shifts measured on each frame (`filmcraft_render::vr_seam`) rather than only cross-fading them; `nadir` (`off`, `mirror`, `smooth`) patches the bottom of the sphere — the hand and stick — within `nadirSize` degrees (22) of straight down: the ground round it mirrored in, or its edge's colour drawn to the middle (tiny planets switch it on, mirrored, where it was never set); with nothing to change, what the items' VR Properties are |
//! | `media.seam360` | the seam of a dual-fisheye item at `seconds` (default: the middle), measured on the full-size frame (`filmcraft_render::vr_seam`): how much the lenses' pictures disagree across it with a plain cross-fade and stitched (`ghosting`, mean luma difference), the largest shift (degrees) and the share of the seam moved, the milliseconds measuring took (`shifts`: every column's, across and along, degrees); with `path`, a PNG of the band unrolled round the seam — lens 0, lens 1, the cross-fade, the stitch |
//! | `media.align360` | lines the two lenses of dual-fisheye media up along their seam (a few frames across the clip, [`filmcraft_render::vr::align_lenses`]) and keeps the result in its VR Properties |
//! | `clip.stabilize360` | holds the virtual camera steady: the camera's turn measured from the pictures frame to frame (`filmcraft_render::vr_motion`: points on cube faces, a robust rotation fit), then `mode` `smooth` / `light` / `strong` (the operator's own turns kept, the shake gone) or `lock` (the first frame's direction throughout), written as the effect's Stabilization keyframes; a job; `off` |
//! | `media.salience360` | where things happen in a 360° item: around `seconds` (default the middle of `from`..`to`, else of the item), a few frames of its panorama 0.2 s apart scored by motion (else detail) in each direction ([`filmcraft_render::vr_motion::salient_directions`]): up to `max` directions (yaw, pitch, score), strongest first, at least 45° apart |
//! | `clip.follow360` | turns the virtual camera to follow a subject (`target` `subject`, `box` in the clip's frame at `at`, or `face`): a wide tracking camera kept on it while the subject tracker follows it both ways from `at`, the path smoothed (`smooth` 0 close … 1 floaty) into yaw and pitch keyframes; a subject lost for a few frames (blocked, gone behind something, turned round) is looked for again by its look in the tracking view and the views either side (`refind`: `auto` — the colours find it, SigLIP 2 confirms it when the scene models are downloaded —, `colour`, `siglip` — every window by SigLIP, slower — or `off`), the report's `foundAgain` counting it; a job |
//!
//! A clip of 360° media shows its raw picture (an equirectangular panorama, or the two fisheye
//! circles of an Insta360 recording) until it is reframed; reframed, it is a flat clip of the
//! virtual camera's frame for everything else (Motion, effects, tracking, Auto Reframe, export).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_media::Spherical;
use filmcraft_media::spherical::{Projection, StereoLayout};
use filmcraft_project::vr::{MAX_SIDE, MIN_SIDE, REFRAME};
use filmcraft_project::{ClipId, EffectInstance, ItemId, ItemKind, Param, ParamValue, TrackItem, TrackKind};
use filmcraft_render::vr::M3;
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, bool_p, checked_u32_p, clips_p, f64_p, has_seq, item_p, str_p, u64_p};
use crate::{EngineError, Result, Session};

const REFRAME_CMD: &str = "clip.reframe360";
const PROPS_CMD: &str = "media.vrProperties";
const ALIGN_CMD: &str = "media.align360";
const SEAM_CMD: &str = "media.seam360";
/// The picture of the seam `media.seam360` writes: its width and each strip's height, pixels.
const SEAM_PICTURE: (usize, usize) = (1440, 96);
/// Most keyframes one `keys` list writes.
const MAX_KEYS: usize = 10_000;
/// Most frames `media.align360` looks at.
const MAX_ALIGN_FRAMES: u64 = 9;
pub(crate) const STAB_CMD: &str = "clip.stabilize360";
const SALIENCE_CMD: &str = "media.salience360";
/// The panorama width salience reads, pixels.
const SALIENCE_WIDTH: f64 = 480.0;
/// The frames salience compares, and how far apart, seconds.
const SALIENCE_FRAMES: u64 = 6;
const SALIENCE_STEP: f64 = 0.2;
const FOLLOW_CMD: &str = "clip.follow360";
/// The virtual camera's stabilisation parameters.
const STAB_PARAMS: [&str; 3] = ["stab_yaw", "stab_pitch", "stab_roll"];
/// Side of the cube faces the stabiliser tracks on, pixels.
const STAB_FACE: usize = 160;
/// How closely the stabilisation keyframes redraw the measured correction, degrees.
const STAB_TOLERANCE_DEGREES: f64 = 0.05;
/// The follow's tracking camera picture (16:9).
const FOLLOW_SIZE: (u32, u32) = (384, 216);
/// Frames a second the follow analyses by default.
const FOLLOW_FPS: f64 = 10.0;
/// How closely the follow's keyframes redraw the smoothed path, degrees.
const FOLLOW_TOLERANCE_DEGREES: f64 = 0.25;
/// Most frames one clip's analysis looks at (a longer clip is analysed at a lower rate).
const MAX_360_FRAMES: usize = 60_000;
/// Finished analyses' outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 256;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: REFRAME_CMD,
            label: "360 Reframe",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clips":[id]?,"preset":"natural"|"wide"|"tinyPlanet"|"rabbitHole"|"sphere"?,"yaw":deg?,"pitch":-90..90?,"roll":deg?,"fov":5..175?,"distance":0..1?,"output":"flat"|"sphere"?,"width":u32?,"height":u32?,"at":seconds?,"keys":[{"seconds":f64,"yaw"?,"pitch"?,"roll"?,"fov"?,"distance"?}]?,"off":bool=false}"#,
            enabled: has_seq,
            run: reframe,
            journal: true,
        },
        CommandSpec {
            id: PROPS_CMD,
            label: "VR Properties…",
            menu: &["Clip", "Modify"],
            shortcut: None,
            params: r#"{"items":[id]?,"projection":"auto"|"equirectangular"|"dualFisheye"|"none"?,"layout":"mono"|"overUnder"|"sideBySide"?,"capturedView":[h,v]?,"halfFov":30..135?,"level":bool?,"steady":bool?,"seamFlow":bool?,"nadir":"off"|"mirror"|"smooth"?,"nadirSize":3..60?}"#,
            enabled: |_| Ok(()),
            run: vr_properties,
            journal: true,
        },
        CommandSpec {
            id: STAB_CMD,
            label: "360 Stabilize",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clips":[id]?,"mode":"smooth"|"light"|"strong"|"lock"="smooth","off":bool=false,"wait":bool=false}"#,
            enabled: has_seq,
            run: stabilize,
            journal: true,
        },
        CommandSpec {
            id: FOLLOW_CMD,
            label: "360 Follow Subject",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clip":id?,"target":"subject"|"box"|"face"="subject","box":[x,y,w,h]?,"at":seconds?,"smooth":0..1=0.5,"analysisFps":1..30=10,"refind":"auto"|"siglip"|"colour"|"off"="auto","wait":bool=false}"#,
            enabled: has_seq,
            run: follow,
            journal: true,
        },
        CommandSpec {
            id: SALIENCE_CMD,
            label: "Find 360 Action",
            menu: &[],
            shortcut: None,
            params: r#"{"item":id,"seconds":f64?,"from":f64?,"to":f64?,"frames":2..24=6,"max":1..8=4}"#,
            enabled: |_| Ok(()),
            run: salience,
            journal: false,
        },
        CommandSpec {
            id: ALIGN_CMD,
            label: "Align 360 Lenses",
            menu: &[],
            shortcut: None,
            params: r#"{"items":[id]?,"frames":1..9=3}"#,
            enabled: |_| Ok(()),
            run: align,
            journal: true,
        },
        CommandSpec {
            id: SEAM_CMD,
            label: "360° Seam",
            menu: &[],
            shortcut: None,
            params: r#"{"item":id,"seconds":f64?,"path":str?,"shifts":bool=false}"#,
            enabled: |_| Ok(()),
            run: seam,
            journal: false,
        },
    ]
}

/// Whether an item's media is 360° (VR Properties set).
pub fn is_360(s: &Session, item: ItemId) -> bool {
    s.project.vr_of(item).is_some()
}

/// The source an analysis reads an item through: 360° media as its whole-sphere
/// equirectangular panorama (dual fisheye stitched, one eye of stereo, a partial view placed on
/// the sphere), so scene detection, descriptions and subject boxes see every direction and a
/// box's centre is a direction ([`filmcraft_project::vr::panorama_direction`]); anything else as it
/// is. A 360° recording with a proxy (an Insta360 `.lrv`, attached on import) is read from it:
/// the same lenses, a tenth of the pixels to decode ([`panorama_view`] with `full` reads the media).
pub fn analysis_view(s: &Session, item: ItemId) -> Option<filmcraft_media::SharedSource> {
    panorama_view(s, item, false)
}

/// [`analysis_view`], from the full-size media with `full` (a sharp still of the panorama).
pub fn panorama_view(s: &Session, item: ItemId, full: bool) -> Option<filmcraft_media::SharedSource> {
    let plain = |v: &Spherical| v.is_equirect() && v.layout == StereoLayout::Mono && v.captured == [360.0, 180.0];
    match s.project.vr_of(item).filter(|v| !plain(v)) {
        Some(v) => {
            let src = if full { s.media.full_res_source(&s.project, item, &*s.services) } else { analysis_source(s, item) }?;
            Some(filmcraft_render::vr_source::PanoramaSource::new(src.clone(), v).map_or(src, |p| Arc::new(p) as filmcraft_media::SharedSource))
        }
        None => s.source(item),
    }
}

/// A stable fingerprint of a virtual camera (its parameters and keyframes): caches of what a
/// reframed clip shows key on it, as on the file.
pub(crate) fn camera_key(e: &EffectInstance) -> String {
    fnv_key(&serde_json::to_vec(e).unwrap_or_default())
}

/// A stable fingerprint of how a 360° picture maps onto the sphere (VR Properties, its horizon
/// included): caches of what its panorama shows key on it.
pub(crate) fn sphere_key(v: &Spherical) -> String {
    // (the patch over the bottom of the sphere hides the hand and stick, not what is there to see)
    let v = Spherical { nadir: None, ..v.clone() };
    fnv_key(&serde_json::to_vec(&v).unwrap_or_default())
}

/// A tiny planet looks straight down, where the hand and stick are: the media of `clips` whose
/// bottom was never patched gets the patch ([`filmcraft_media::spherical::Nadir`], mirrored in),
/// folded into the undo step since `n0` (`label`). The items patched.
pub(crate) fn patch_planets(s: &mut Session, clips: &[ClipId], n0: usize, label: &str) -> Vec<u64> {
    let Some(seq) = s.active_sequence() else { return Vec::new() };
    let mut items: Vec<ItemId> =
        clips.iter().filter_map(|c| seq.find_item(*c).map(|(_, it)| it.item)).filter(|i| s.project.vr_of(*i).is_some_and(|v| v.nadir.is_none())).collect();
    items.sort();
    items.dedup();
    if items.is_empty() {
        return Vec::new();
    }
    let done = s.edit(label, |pr, _| {
        for id in &items {
            if let Some(ItemKind::Media(m)) = pr.item_mut(*id).map(|i| &mut i.kind)
                && let Some(v) = m.interpret.vr.as_mut()
            {
                v.nadir = Some(filmcraft_media::spherical::Nadir {
                    mode: filmcraft_media::spherical::NadirMode::Mirror,
                    size: filmcraft_media::spherical::NADIR_SIZE,
                });
            }
        }
        Ok(())
    });
    crate::clip_ops::collapse_history(s, n0, label);
    if done.is_ok() { items.iter().map(|i| i.0).collect() } else { Vec::new() }
}

fn fnv_key(bytes: &[u8]) -> String {
    // FNV-1a: the same on every machine and build, unlike std's hasher
    let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3));
    format!("{h:016x}")
}

/// The source of a clip's own picture, for analyses in its pixels (tracking, masks, redaction):
/// a reframed 360° clip's virtual camera view at its keyframes, anything else its media.
pub fn clip_source(s: &Session, it: &TrackItem) -> Option<filmcraft_media::SharedSource> {
    let src = s.source(it.item)?;
    match filmcraft_render::vr_source::ReframedSource::new(src.clone(), s.project.vr_of(it.item), it) {
        Some(r) => Some(Arc::new(r)),
        None => Some(src),
    }
}

/// A clip's virtual camera, added (framing `frame`, Motion back to its defaults so the camera's
/// frame fills the sequence's) when it has none; enabled.
pub fn ensure_reframe(it: &mut TrackItem, frame: (u32, u32)) -> Option<&mut EffectInstance> {
    if !it.effects.iter().any(|e| e.effect == REFRAME) {
        // the camera's picture is the clip's: Motion places it like any picture of that size
        if let (Some(m), Some(mdef)) = (it.effect_mut("motion"), filmcraft_project::find_effect("motion")) {
            for id in ["position", "anchor", "scale", "scale_width", "rotation"] {
                if let (Some(p), Some(d)) = (m.param_mut(id), mdef.param(id)) {
                    p.keyframes.clear();
                    p.value = d.default.clone();
                }
            }
            filmcraft_project::resolve_auto_points(m, frame, frame);
        }
        it.scale_to_frame = false;
    }
    filmcraft_project::vr::add_camera(it, frame)
}

/// The angle a flat view of field of view `fov` (degrees across) sees along the short side of
/// a `frame`.
fn short_side_fov(fov: f64, frame: (u32, u32)) -> f64 {
    let (w, h) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    if w <= h { fov } else { 2.0 * ((fov.to_radians() / 2.0).tan() * h / w).atan().to_degrees() }
}

/// The field of view across a `frame` that sees `short` degrees along its short side.
fn fov_across(short: f64, frame: (u32, u32)) -> f64 {
    let (w, h) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    let fov = if w <= h { short } else { 2.0 * ((short.to_radians() / 2.0).tan() * w / h).atan().to_degrees() };
    if fov.is_finite() { fov.clamp(5.0, 175.0) } else { 90.0 }
}

/// Re-shapes a reframed clip's virtual camera from the frame `from` to the frame `to` (Auto
/// Reframe's copy): the camera makes a `to` picture and sees as much along its short side as
/// before (a 16:9 view's height becomes a 9:16 view's width), a whole-sphere camera outside a 2:1
/// frame becomes a natural flat view, and Motion's position and anchor move with the frame.
/// `false` for a clip without a virtual camera.
pub fn reshape_camera(it: &mut TrackItem, from: (u32, u32), to: (u32, u32)) -> bool {
    let Some(e) = it.effects.iter_mut().find(|e| e.enabled && e.effect == REFRAME) else { return false };
    let old = filmcraft_project::vr::effect_size(e).unwrap_or(from);
    let side = |v: u32| ParamValue::Float(f64::from(v.clamp(MIN_SIDE, MAX_SIDE)));
    e.params.insert("width".into(), Param::new(side(to.0)));
    e.params.insert("height".into(), Param::new(side(to.1)));
    let sphere = matches!(e.param("output").map(|p| &p.value), Some(ParamValue::Choice(1)));
    if sphere && to.0 != to.1.saturating_mul(2) {
        e.params.insert("output".into(), Param::new(ParamValue::Choice(0)));
        set_static(e, "fov", fov_across(short_side_fov(90.0, (16, 9)), to));
    } else if !sphere && let Some(p) = e.param_mut("fov") {
        let turn = |v: &mut ParamValue| {
            if let ParamValue::Float(f) = v {
                *f = fov_across(short_side_fov(*f, old), to);
            }
        };
        turn(&mut p.value);
        for k in &mut p.keyframes {
            turn(&mut k.value);
        }
    }
    let (sx, sy) = (f64::from(to.0) / f64::from(from.0.max(1)), f64::from(to.1) / f64::from(from.1.max(1)));
    let (px, py) = (f64::from(to.0) / f64::from(old.0.max(1)), f64::from(to.1) / f64::from(old.1.max(1)));
    if let Some(m) = it.effect_mut("motion") {
        for (id, (fx, fy)) in [("position", (sx, sy)), ("anchor", (px, py))] {
            if let Some(p) = m.param_mut(id) {
                let mv = |v: &mut ParamValue| {
                    if let ParamValue::Vec2(v) = v {
                        v.x *= fx;
                        v.y *= fy;
                    }
                };
                mv(&mut p.value);
                for k in &mut p.keyframes {
                    mv(&mut k.value);
                }
            }
        }
    }
    true
}

/// A preset's values: (yaw, pitch, fov — none: the frame's natural view — distance, sphere).
fn preset(name: &str) -> Option<(Option<f64>, Option<f64>, Option<f64>, f64, bool)> {
    Some(match name {
        "natural" => (None, None, None, 0.0, false),
        "wide" => (None, None, Some(120.0), 0.3, false),
        "tinyPlanet" | "tiny_planet" | "planet" => (None, Some(-90.0), Some(120.0), 1.0, false),
        "rabbitHole" | "rabbit_hole" | "tunnel" => (None, Some(90.0), Some(120.0), 1.0, false),
        "sphere" | "equirectangular" => (Some(0.0), Some(0.0), Some(90.0), 0.0, true),
        _ => return None,
    })
}

/// The camera parameters `p` sets (finite, in range), by param id.
fn camera_values(p: &Value, cmd: &str) -> Result<Vec<(&'static str, f64)>> {
    let mut out = Vec::new();
    for (key, id, lo, hi) in [
        ("yaw", "yaw", -3600.0, 3600.0),
        ("pitch", "pitch", -90.0, 90.0),
        ("roll", "roll", -180.0, 180.0),
        ("fov", "fov", 5.0, 175.0),
        ("distance", "distance", 0.0, 1.0),
    ] {
        match p.get(key) {
            None | Some(Value::Null) => {}
            Some(v) => {
                let x = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number")))?;
                if !(lo..=hi).contains(&x) {
                    return Err(bad(cmd, format!("`{key}` must be within {lo}..{hi}, got {x}")));
                }
                out.push((id, x));
            }
        }
    }
    Ok(out)
}

pub(crate) fn set_static(e: &mut EffectInstance, id: &str, v: f64) {
    match e.param_mut(id) {
        Some(p) => {
            p.keyframes.clear();
            p.value = ParamValue::Float(v);
        }
        None => {
            e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
        }
    }
}

pub(crate) fn set_key(e: &mut EffectInstance, id: &str, t: Tick, v: f64) {
    if e.param(id).is_none() {
        e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
    }
    if let Some(p) = e.param_mut(id) {
        p.put_keyframe(t, ParamValue::Float(v));
    }
}

/// The media time of clip time `seconds` (from the clip's start on the timeline).
pub(crate) fn media_time(it: &TrackItem, seconds: f64) -> Tick {
    let st = it.start + Tick::from_seconds_f64(seconds.clamp(0.0, it.duration.seconds().max(0.0)));
    it.effect_time_at(st)
}

pub(crate) fn view_json(c: ClipId, it: &TrackItem) -> Value {
    let Some(e) = it.effect(REFRAME) else { return json!({"clip": c.0, "reframed": false}) };
    let t = it.effect_time_at(it.start);
    let animated: Vec<&str> =
        ["yaw", "pitch", "roll", "fov", "distance"].into_iter().filter(|id| e.param(id).is_some_and(|p| !p.keyframes.is_empty())).collect();
    let r = |v: f64| (v * 1000.0).round() / 1000.0;
    json!({
        "clip": c.0,
        "reframed": e.enabled,
        "yaw": r(e.f64_at("yaw", t)),
        "pitch": r(e.f64_at("pitch", t)),
        "roll": r(e.f64_at("roll", t)),
        "fov": r(e.f64_at("fov", t)),
        "distance": r(e.f64_at("distance", t)),
        "output": if matches!(e.param("output").map(|p| &p.value), Some(ParamValue::Choice(1))) { "sphere" } else { "flat" },
        "frame": filmcraft_project::vr::effect_size(e),
        "animated": animated,
    })
}

fn reframe(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = REFRAME_CMD;
    let preset = match str_p(p, "preset") {
        Some(n) => Some(preset(n).ok_or_else(|| bad(cmd, format!("unknown preset `{n}` (natural, wide, tinyPlanet, rabbitHole, sphere)")))?),
        None => None,
    };
    let values = camera_values(p, cmd)?;
    let output = match str_p(p, "output") {
        None => None,
        Some("flat") => Some(0u32),
        Some("sphere") => Some(1u32),
        Some(o) => return Err(bad(cmd, format!("`output` is flat or sphere, not `{o}`"))),
    };
    let width = checked_u32_p(p, "width", cmd)?;
    let height = checked_u32_p(p, "height", cmd)?;
    for (k, v) in [("width", width), ("height", height)] {
        if let Some(v) = v
            && !(MIN_SIDE..=MAX_SIDE).contains(&v)
        {
            return Err(bad(cmd, format!("`{k}` must be within {MIN_SIDE}..{MAX_SIDE}")));
        }
    }
    let at = match p.get("at") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| bad(cmd, "`at` is seconds from the clip's start"))?),
    };
    let mut keys: Vec<(f64, Vec<(&'static str, f64)>)> = Vec::new();
    if let Some(list) = p.get("keys").filter(|v| !v.is_null()) {
        let list = list.as_array().ok_or_else(|| bad(cmd, "`keys` is a list of {seconds, yaw, pitch, roll, fov, distance}"))?;
        if list.len() > MAX_KEYS {
            return Err(bad(cmd, format!("at most {MAX_KEYS} keys")));
        }
        for k in list {
            let t = f64_p(k, "seconds").filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| bad(cmd, "every key needs `seconds` (from the clip's start)"))?;
            keys.push((t, camera_values(k, cmd)?));
        }
    }
    let off = bool_p(p, "off").unwrap_or(false);
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    // the clips named, else the selection, else every clip of 360° media
    let mut clips = clips_p(s, p);
    if clips.is_empty() && p.get("clips").is_none_or(Value::is_null) {
        clips = seq.video_tracks.iter().flat_map(|t| &t.items).filter(|it| s.project.vr_of(it.item).is_some()).map(|it| it.id).collect();
    }
    let frame = (seq.settings.width, seq.settings.height);
    let video: Vec<ClipId> =
        clips.iter().copied().filter(|c| seq.find_item(*c).and_then(|(t, _)| seq.track(t)).is_some_and(|t| t.kind == TrackKind::Video && !t.locked)).collect();
    if video.is_empty() {
        return Err(bad(cmd, "no unlocked video clip given or selected"));
    }
    let not_360: Vec<u64> = video
        .iter()
        .filter(|c| seq.find_item(**c).is_some_and(|(_, it)| s.project.vr_of(it.item).is_none() && it.effect(REFRAME).is_none()))
        .map(|c| c.0)
        .collect();
    let label = if off { "360 Reframe Off" } else { "360 Reframe" };
    let n0 = s.history.undo.len();
    let report = s.edit_sequence(label, |q, _, _| {
        let mut out = Vec::new();
        for c in &video {
            let (_, it) = q.find_item_mut(*c).ok_or(filmcraft_edit::EditError::NoItem(*c))?;
            if off {
                if let Some(e) = it.effect_mut(REFRAME) {
                    e.enabled = false;
                }
                out.push(view_json(*c, it));
                continue;
            }
            let key_times: Vec<(Tick, Vec<(&'static str, f64)>)> = keys.iter().map(|(t, v)| (media_time(it, *t), v.clone())).collect();
            let at_time = at.map(|a| media_time(it, a));
            let Some(e) = ensure_reframe(it, frame) else { return Err(EngineError::Other("no 360 Reframe effect definition".into())) };
            if let (Some(w), Some(h)) = (width.or(height.map(|_| frame.0)), height.or(width.map(|_| frame.1))) {
                e.params.insert("width".into(), Param::new(ParamValue::Float(f64::from(w))));
                e.params.insert("height".into(), Param::new(ParamValue::Float(f64::from(h))));
            }
            if let Some((yaw, pitch, fov, distance, sphere)) = preset {
                if let Some(y) = yaw {
                    set_static(e, "yaw", y);
                }
                if let Some(pt) = pitch {
                    set_static(e, "pitch", pt);
                }
                set_static(e, "roll", 0.0);
                let (w, h) = filmcraft_project::vr::effect_size(e).unwrap_or(frame);
                set_static(e, "fov", fov.unwrap_or_else(|| filmcraft_project::vr::natural_fov(f64::from(w.max(1)) / f64::from(h.max(1))).round()));
                set_static(e, "distance", distance);
                e.params.insert("output".into(), Param::new(ParamValue::Choice(u32::from(sphere))));
            }
            if let Some(o) = output {
                e.params.insert("output".into(), Param::new(ParamValue::Choice(o)));
            }
            match at_time {
                Some(t) => values.iter().for_each(|(id, v)| set_key(e, id, t, *v)),
                None => values.iter().for_each(|(id, v)| set_static(e, id, *v)),
            }
            if !key_times.is_empty() {
                // a camera move: the params it names follow it, from scratch
                let named: std::collections::BTreeSet<&str> = key_times.iter().flat_map(|(_, v)| v.iter().map(|(id, _)| *id)).collect();
                for id in &named {
                    if let Some(prm) = e.param_mut(id) {
                        prm.keyframes.clear();
                    }
                }
                for (t, vals) in &key_times {
                    for (id, v) in vals {
                        set_key(e, id, *t, *v);
                    }
                }
            }
            out.push(view_json(*c, it));
        }
        Ok(out)
    })?;
    let mut res = json!({"clips": report});
    // a tiny planet: its hand and stick patched over
    if !off && matches!(str_p(p, "preset"), Some("tinyPlanet" | "tiny_planet" | "planet")) {
        let patched = patch_planets(s, &video, n0, label);
        if !patched.is_empty() {
            res["nadirPatched"] = json!(patched);
        }
    }
    if !not_360.is_empty() {
        res["note"] = json!(format!(
            "clips {not_360:?} have no VR Properties: their pictures are taken as whole equirectangular panoramas (media.vrProperties sets how they map onto the sphere)"
        ));
    }
    Ok(res)
}

fn items_p(s: &Session, p: &Value) -> Vec<ItemId> {
    match p.get("items").and_then(Value::as_array) {
        Some(a) => a.iter().filter_map(|v| v.as_u64().map(ItemId)).collect(),
        None => item_p(p, "item").map(|i| vec![i]).unwrap_or_else(|| s.state.project_selection.clone()),
    }
}

/// The media items (not sequences, not missing) among `items`.
fn media_items(s: &Session, items: &[ItemId], cmd: &str) -> Result<Vec<ItemId>> {
    let out: Vec<ItemId> = items.iter().copied().filter(|i| s.project.item(*i).is_some_and(|it| matches!(it.kind, ItemKind::Media(_)))).collect();
    if out.is_empty() {
        return Err(bad(cmd, "no media item given or selected (`items`)"));
    }
    Ok(out)
}

fn vr_properties(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = PROPS_CMD;
    let items = media_items(s, &items_p(s, p), cmd)?;
    let projection = str_p(p, "projection");
    let layout = match str_p(p, "layout") {
        None => None,
        Some("mono") => Some(StereoLayout::Mono),
        Some("overUnder") => Some(StereoLayout::OverUnder),
        Some("sideBySide") => Some(StereoLayout::SideBySide),
        Some(o) => return Err(bad(cmd, format!("`layout` is mono, overUnder or sideBySide, not `{o}`"))),
    };
    let captured = match p.get("capturedView").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => {
            let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(cmd, "`capturedView` is [degrees across, degrees down]"))?;
            let h = a.first().and_then(Value::as_f64).filter(|x| x.is_finite() && (1.0..=360.0).contains(x));
            let w = a.get(1).and_then(Value::as_f64).filter(|x| x.is_finite() && (1.0..=180.0).contains(x));
            Some([h.ok_or_else(|| bad(cmd, "`capturedView` across is 1..360°"))?, w.ok_or_else(|| bad(cmd, "`capturedView` down is 1..180°"))?])
        }
    };
    let half_fov = match f64_p(p, "halfFov") {
        Some(h) if h.is_finite() && (30.0..=135.0).contains(&h) => Some(h),
        Some(_) => return Err(bad(cmd, "`halfFov` must be within 30..135°")),
        None => None,
    };
    let level = bool_p(p, "level");
    let steady = bool_p(p, "steady");
    let seam_flow = bool_p(p, "seamFlow");
    let nadir = match str_p(p, "nadir") {
        None => None,
        Some("off") => Some(filmcraft_media::spherical::NadirMode::Off),
        Some("mirror") => Some(filmcraft_media::spherical::NadirMode::Mirror),
        Some("smooth") => Some(filmcraft_media::spherical::NadirMode::Smooth),
        Some(o) => return Err(bad(cmd, format!("`nadir` is off, mirror or smooth, not `{}`", o.chars().take(20).collect::<String>()))),
    };
    let nadir_size = match f64_p(p, "nadirSize") {
        Some(v) if v.is_finite() && (filmcraft_media::spherical::NADIR_RANGE.0..=filmcraft_media::spherical::NADIR_RANGE.1).contains(&v) => Some(v),
        Some(_) => return Err(bad(cmd, "`nadirSize` is 3 to 60 degrees")),
        None => None,
    };
    // nothing to change: what the items' VR Properties are
    if projection.is_none()
        && layout.is_none()
        && captured.is_none()
        && half_fov.is_none()
        && level.is_none()
        && steady.is_none()
        && seam_flow.is_none()
        && nadir.is_none()
        && nadir_size.is_none()
    {
        return Ok(json!({"items": items.iter().map(|id| json!({"item": id.0, "vr": vr_json(s.project.vr_of(*id))})).collect::<Vec<_>>()}));
    }
    // what each item gets (decided before the edit: `auto` opens the file)
    let mut new: Vec<(ItemId, Option<Spherical>)> = Vec::new();
    for id in &items {
        let from_file = || s.media.full_res_source(&s.project, *id, &*s.services).and_then(|src| src.spherical());
        let mut v = match projection {
            // only the layout, captured view or lens angle given: the item's own, changed
            None => s.project.vr_of(*id).cloned(),
            Some("auto") => from_file(),
            Some("none") => None,
            Some("equirectangular") => Some(Spherical::equirect()),
            Some("dualFisheye") => {
                Some(match from_file().filter(|f| !f.is_equirect()).or_else(|| s.project.vr_of(*id).filter(|v| !v.is_equirect()).cloned()) {
                    Some(f) => f,
                    None => default_dual(),
                })
            }
            Some(o) => return Err(bad(cmd, format!("`projection` is auto, equirectangular, dualFisheye or none, not `{o}`"))),
        };
        if let Some(v) = v.as_mut() {
            if let Some(l) = layout {
                v.layout = l;
            }
            if let Some(c) = captured {
                v.captured = c;
            }
            if let (Some(h), Projection::DualFisheye { lenses }) = (half_fov, &mut v.projection) {
                for l in lenses.iter_mut() {
                    l.half_fov = h;
                }
            }
            if let Some(f) = seam_flow {
                v.seam_flow = f;
            }
            if nadir.is_some() || nadir_size.is_some() {
                let was = v.nadir;
                v.nadir = Some(filmcraft_media::spherical::Nadir {
                    mode: nadir.or(was.map(|n| n.mode)).unwrap_or_default(),
                    size: nadir_size.or(was.map(|n| n.size)).unwrap_or(filmcraft_media::spherical::NADIR_SIZE),
                });
            }
            match level {
                // the recording's own horizon, levelling
                Some(true) => {
                    let h = v.horizon.clone().filter(|h| !h.up.is_empty()).or_else(|| from_file().and_then(|f| f.horizon));
                    let Some(h) = h else { return Err(bad(cmd, format!("item {} has no motion-sensor record to level its horizon by", id.0))) };
                    v.horizon = Some(filmcraft_media::spherical::Horizon { level: true, ..h });
                }
                Some(false) => {
                    if let Some(h) = v.horizon.as_mut() {
                        h.level = false;
                    }
                }
                None => {}
            }
            match steady {
                // the heading steadied by the camera's turning (a horizon kept from before the
                // steadying has none: the file's, levelled as it was)
                Some(true) => {
                    // (levelled as `level` left it)
                    let levelled = v.horizon.as_ref().is_none_or(|h| h.level);
                    let h =
                        v.horizon.clone().filter(|h| !h.heading.is_empty()).or_else(|| from_file().and_then(|f| f.horizon).filter(|h| !h.heading.is_empty()));
                    let Some(h) = h else { return Err(bad(cmd, format!("item {} has no motion-sensor record to steady its heading by", id.0))) };
                    v.horizon = Some(filmcraft_media::spherical::Horizon { steady: true, level: levelled, ..h });
                }
                Some(false) => {
                    if let Some(h) = v.horizon.as_mut() {
                        h.steady = false;
                    }
                }
                None => {}
            }
        } else if level == Some(true) || steady == Some(true) {
            return Err(bad(cmd, format!("item {} is not 360° media", id.0)));
        }
        new.push((*id, v.map(|v| v.sanitized())));
    }
    let report = s.edit("VR Properties", |pr, _| {
        let mut out = Vec::new();
        for (id, v) in &new {
            if let Some(ItemKind::Media(m)) = pr.item_mut(*id).map(|i| &mut i.kind) {
                m.interpret.vr = v.clone();
                out.push(json!({"item": id.0, "vr": vr_json(v.as_ref())}));
            }
        }
        Ok(out)
    })?;
    Ok(json!({"items": report}))
}

/// VR Properties as JSON, a horizon summed up (whether it levels, how long it runs, how many
/// samples, the largest tilt) rather than listed.
fn vr_json(v: Option<&Spherical>) -> Value {
    let Some(v) = v else { return Value::Null };
    let mut j = serde_json::to_value(v).unwrap_or(Value::Null);
    if let (false, Some(o)) = (v.is_equirect(), j.as_object_mut()) {
        o.insert("seamFlow".into(), json!(v.seam_flow));
    }
    if let (Some(h), Some(o)) = (&v.horizon, j.as_object_mut()) {
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        let mut sum = json!({"level": h.level, "seconds": r1(h.seconds()), "samples": h.up.len(), "maxTilt": r1(h.max_tilt())});
        if !h.heading.is_empty() {
            sum["steady"] = json!(h.steady);
            sum["maxSteady"] = json!(r1(h.max_steady()));
        }
        o.insert("horizon".into(), sum);
    }
    j
}

/// Two circular fisheye pictures side by side, the left looking back, filling their halves.
fn default_dual() -> Spherical {
    let lens = |k: usize| filmcraft_media::spherical::FisheyeLens {
        rect: [k as f64 * 0.5, 0.0, 0.5, 1.0],
        cx: 0.5,
        cy: 0.5,
        radius: 0.5,
        half_fov: 97.5,
        poly: Vec::new(),
        yaw: if k == 0 { 180.0 } else { 0.0 },
        pitch: 0.0,
        roll: 0.0,
    };
    Spherical::dual_fisheye(vec![lens(0), lens(1)], "")
}

/// `frames` linear pictures spread across a source (decoded at half size).
pub fn sample_frames(src: &filmcraft_media::SharedSource, frames: u64) -> Vec<filmcraft_render::Image> {
    let dur = src.info().duration;
    let n = frames.clamp(1, MAX_ALIGN_FRAMES);
    (0..n)
        .filter_map(|k| {
            let t = Tick((dur.0 as f64 * (k as f64 + 0.5) / n as f64) as i64);
            let f = src.video_frame(filmcraft_media::FrameRequest { time: t, scale: 0.5 }).ok()?;
            let dec = if f.width > 2048 { 2 } else { 1 };
            let (w, h, px) = f.to_linear_f32_decimated(dec);
            Some(filmcraft_render::Image { w, h, px })
        })
        .collect()
}

fn salience(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = SALIENCE_CMD;
    let item = item_p(p, "item").ok_or_else(|| bad(cmd, "need `item`"))?;
    if s.project.vr_of(item).is_none() {
        return Err(bad(cmd, "not 360° media (media.vrProperties says how a picture maps onto the sphere)"));
    }
    let src = analysis_view(s, item).ok_or_else(|| bad(cmd, "the item's media is offline"))?;
    let v = src.info().video.clone().ok_or_else(|| bad(cmd, "the item has no video"))?;
    let dur = src.info().duration.seconds().max(0.0);
    let secs = |k: &str| -> Result<Option<f64>> {
        match f64_p(p, k) {
            None => Ok(None),
            Some(x) if x.is_finite() && x >= 0.0 => Ok(Some(x.min(dur))),
            Some(_) => Err(bad(cmd, format!("`{k}` is seconds of the media"))),
        }
    };
    let (from, to) = (secs("from")?.unwrap_or(0.0), secs("to")?.unwrap_or(dur));
    let mid = secs("seconds")?.unwrap_or((from + to.max(from)) / 2.0);
    let n = u64_p(p, "frames").unwrap_or(SALIENCE_FRAMES).clamp(2, 24);
    let max = u64_p(p, "max").unwrap_or(4).clamp(1, 8) as usize;
    let scale = (SALIENCE_WIDTH / f64::from(v.width.max(1))).clamp(1.0 / 64.0, 1.0) as f32;
    let fd = v.frame_rate.sane().frame_duration().seconds().max(1e-3);
    let step = SALIENCE_STEP.max(fd);
    let first = (mid - step * (n as f64 - 1.0) / 2.0).clamp(0.0, (dur - fd).max(0.0));
    let mut frames: Vec<filmcraft_render::Image> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    for k in 0..n {
        let t = (first + step * k as f64).min((dur - fd).max(0.0));
        let Ok(f) = src.video_frame(filmcraft_media::FrameRequest { time: Tick::from_seconds_f64(t), scale }) else { continue };
        let dec = (f.width as usize / SALIENCE_WIDTH as usize).max(1);
        let (w, h, px) = f.to_linear_f32_decimated(dec);
        if frames.first().is_some_and(|a| a.w != w || a.h != h) {
            continue;
        }
        frames.push(filmcraft_render::Image { w, h, px });
        times.push(t);
    }
    let found = filmcraft_render::vr_motion::salient_directions(&frames, 45.0, max);
    let r = |x: f64| (x * 10.0).round() / 10.0;
    Ok(json!({
        "item": item.0,
        "seconds": times.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        "directions": found.iter().map(|d| json!({"yaw": r(d.yaw), "pitch": r(d.pitch), "score": d.score, "by": if d.motion { "motion" } else { "detail" }})).collect::<Vec<_>>(),
    }))
}

/// `media.seam360`: the seam of a dual-fisheye item at `seconds`, measured on the full-size frame.
fn seam(s: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_render::vr::SourceModel;
    use filmcraft_render::vr_seam::{frame_luma, seam_of};
    let cmd = SEAM_CMD;
    let item = item_p(p, "item").ok_or_else(|| bad(cmd, "need `item`"))?;
    let Some(vr) = s.project.vr_of(item).cloned().filter(|v| !v.is_equirect()) else { return Err(bad(cmd, "not dual-fisheye 360° media")) };
    let src = s.media.full_res_source(&s.project, item, &*s.services).ok_or_else(|| bad(cmd, "the item's media is offline"))?;
    let dur = src.info().duration.seconds().max(0.0);
    let fd = src.info().video.as_ref().map_or(1.0 / 30.0, |v| v.frame_rate.sane().frame_duration().seconds().max(1e-3));
    let secs = match f64_p(p, "seconds") {
        Some(x) if x.is_finite() && x >= 0.0 => x.min((dur - fd).max(0.0)),
        Some(_) => return Err(bad(cmd, "`seconds` is seconds of the media, zero or more")),
        None => dur / 2.0,
    };
    let t = Tick::from_seconds_f64(secs);
    let frame = src.video_frame(filmcraft_media::FrameRequest { time: t, scale: 1.0 }).map_err(|e| bad(cmd, format!("cannot decode the frame: {e}")))?;
    let model = SourceModel::at(&vr, (frame.width as usize, frame.height as usize), t);
    let frames = filmcraft_render::vr_seam::seam_frames(&src, t, 1.0, frame.clone());
    let clock = web_time::Instant::now();
    let flow = seam_of(&model, &frames.iter().map(|f| f.as_ref()).collect::<Vec<_>>()).ok_or_else(|| bad(cmd, "its lenses have no seam between them"))?;
    let ms = clock.elapsed().as_secs_f64() * 1000.0;
    let SourceModel::DualFisheye { lenses, .. } = &model else { return Err(bad(cmd, "not dual-fisheye 360° media")) };
    let luma = frame_luma(&frame);
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let across: Vec<f64> = flow.shift.iter().map(|s| f64::from(s[0]).to_degrees().abs()).collect();
    let mut out = json!({
        "item": item.0,
        "seconds": r3(secs),
        "measureMs": (ms * 10.0).round() / 10.0,
        "ghosting": {"crossFade": flow.ghosting(lenses, &luma, false).map(r3), "stitched": flow.ghosting(lenses, &luma, true).map(r3)},
        "largestShift": (across.iter().copied().fold(0.0, f64::max) * 100.0).round() / 100.0,
        "moved": r3(across.iter().filter(|a| **a > 0.25).count() as f64 / across.len().max(1) as f64),
        "seamFlow": vr.seam_flow,
    });
    if bool_p(p, "shifts").unwrap_or(false) {
        out["shifts"] = json!(
            flow.shift
                .iter()
                .map(|s| [(f64::from(s[0]).to_degrees() * 100.0).round() / 100.0, (f64::from(s[1]).to_degrees() * 100.0).round() / 100.0])
                .collect::<Vec<_>>()
        );
    }
    if let Some(path) = str_p(p, "path") {
        let (grey, w, h) = flow.band_picture(lenses, &luma, SEAM_PICTURE.0, SEAM_PICTURE.1).ok_or_else(|| bad(cmd, "its lenses have no seam between them"))?;
        let rgba: Vec<u8> = grey
            .iter()
            .flat_map(|v| {
                let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                [g, g, g, 255]
            })
            .collect();
        let png = filmcraft_export::encode_png(rgba, w as u32, h as u32).map_err(|e| bad(cmd, e.to_string()))?;
        s.services.write_file(path, &png).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
        out["path"] = json!(path);
    }
    Ok(out)
}

fn align(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = ALIGN_CMD;
    let items = media_items(s, &items_p(s, p), cmd)?;
    let frames = u64_p(p, "frames").unwrap_or(3);
    let mut found = Vec::new();
    let mut report = Vec::new();
    for id in items {
        let Some(vr) = s.project.vr_of(id).cloned().filter(|v| !v.is_equirect()) else {
            report.push(json!({"item": id.0, "aligned": false, "why": "not dual-fisheye 360° media"}));
            continue;
        };
        let Some(src) = s.media.full_res_source(&s.project, id, &*s.services) else {
            report.push(json!({"item": id.0, "aligned": false, "why": "the media is offline"}));
            continue;
        };
        let pics = sample_frames(&src, frames);
        match filmcraft_render::vr::align_lenses(&pics, &vr) {
            Some(a) => {
                report.push(json!({"item": id.0, "aligned": true, "seamBefore": a.before, "seamAfter": a.after}));
                found.push((id, a.spherical));
            }
            None => report.push(json!({"item": id.0, "aligned": false, "why": "the lenses share no picture to compare"})),
        }
    }
    if !found.is_empty() {
        s.edit("Align 360 Lenses", |pr, _| {
            for (id, v) in &found {
                if let Some(ItemKind::Media(m)) = pr.item_mut(*id).map(|i| &mut i.kind) {
                    m.interpret.vr = Some(v.clone());
                }
            }
            Ok(())
        })?;
    }
    Ok(json!({"items": report}))
}

// ---------------------------------------------------------------------------------------------
// analyses that end as virtual-camera keyframes: stabilising and following
// ---------------------------------------------------------------------------------------------

/// Keyframes an analysis made for one clip: per parameter of the virtual camera, (media time,
/// value) — written from scratch over the clip's old keys of that parameter.
#[derive(Clone, Debug)]
pub struct ClipKeys {
    pub clip: ClipId,
    pub keys: Vec<(&'static str, Vec<(Tick, f64)>)>,
    pub summary: Value,
}

type KeyResults = Arc<Mutex<Option<std::result::Result<Vec<ClipKeys>, String>>>>;

struct PendingKeys {
    job: u64,
    cmd: &'static str,
    undo: &'static str,
    seq: ItemId,
    frame: (u32, u32),
    results: KeyResults,
}

/// The session's 360° analyses and what the finished ones made (for `jobs.list`).
#[derive(Default)]
pub struct Vr360Jobs {
    pending: Vec<PendingKeys>,
    outcomes: HashMap<u64, Value>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `analyse` as a job (`jobs.list`; `jobs.cancel` stops it and nothing changes) and writes
/// its keyframes into the active sequence's clips when it finishes — before answering with `wait`.
fn start_keys(
    s: &mut Session,
    cmd: &'static str,
    undo: &'static str,
    label: String,
    frames: usize,
    wait: bool,
    analyse: impl FnOnce(&filmcraft_export::Progress) -> std::result::Result<Vec<ClipKeys>, String> + Send + 'static,
) -> Result<Value> {
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let frame = s.active_sequence().map_or((1920, 1080), |q| (q.settings.width, q.settings.height));
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(frames as u64, Ordering::Relaxed);
    let id = job.id;
    let results: KeyResults = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), results.clone());
    let run = move || {
        let t0 = web_time::Instant::now();
        let r = analyse(&prog);
        let secs = t0.elapsed().as_secs_f64();
        let done = prog.done.load(Ordering::Relaxed);
        let report = match &r {
            Ok(_) => Ok(filmcraft_export::Report {
                path: String::new(),
                frames: done,
                seconds: secs,
                bytes: 0,
                render_fps: done as f64 / secs.max(1e-6),
                extra_files: Vec::new(),
            }),
            Err(e) => {
                *lock(&prog.status) = e.clone();
                Err(e.clone())
            }
        };
        // the results before `finished`: whoever sees the job finished can apply them
        *lock(&out) = Some(r);
        *lock(&res) = Some(report);
        prog.finished.store(true, Ordering::Relaxed);
    };
    s.vr360.pending.push(PendingKeys { job: id, cmd, undo, seq, frame, results });
    crate::narration::spawn(s, job, wait, "filmcraft-360-analysis", run)?;
    let mut answer = json!({"job": id, "frames": frames});
    if wait || cfg!(target_arch = "wasm32") {
        poll(s);
        if let Some(o) = s.vr360.outcomes.get(&id) {
            if let Some(e) = o.get("error").and_then(Value::as_str) {
                return Err(EngineError::Other(format!("{cmd}: {e}")));
            }
            answer["result"] = o.clone();
        }
    }
    Ok(answer)
}

/// Applies finished analyses (one undo step each). Called each UI frame, by `jobs.list` and after
/// a `wait: true` run.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.vr360.pending.len() {
        let Some(pj) = s.vr360.pending.get(i) else { break };
        let ready = lock(&pj.results).is_some();
        let job = s.jobs.iter().find(|j| j.id == pj.job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !ready && !finished {
            i += 1;
            continue;
        }
        let pj = s.vr360.pending.remove(i);
        let results = lock(&pj.results).take();
        let outcome = match results {
            Some(Ok(clips)) if !cancelled => match apply_keys(s, &pj, &clips) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(pj.cmd, format!("360: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            _ => json!({"error": "stopped: nothing was changed"}),
        };
        s.vr360.outcomes.insert(pj.job, outcome);
        if s.vr360.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.vr360.outcomes.keys().min().copied()
        {
            s.vr360.outcomes.remove(&old);
        }
    }
}

/// What a finished 360° analysis made (`vr360` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.vr360.outcomes.get(&job).cloned()
}

fn apply_keys(s: &mut Session, pj: &PendingKeys, clips: &[ClipKeys]) -> Result<Value> {
    let (seq_id, frame) = (pj.seq, pj.frame);
    let summary: Vec<Value> = clips.iter().map(|c| c.summary.clone()).collect();
    s.edit(pj.undo, |pr, _| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        for c in clips {
            // (a clip deleted, or moved off the video tracks, while the job ran is skipped)
            if q.find_item(c.clip).and_then(|(t, _)| q.track(t)).is_none_or(|t| t.kind != TrackKind::Video) {
                continue;
            }
            let Some((_, it)) = q.find_item_mut(c.clip) else { continue };
            let Some(e) = ensure_reframe(it, frame) else { continue };
            for (id, keys) in &c.keys {
                if keys.is_empty() {
                    continue;
                }
                if let Some(p) = e.param_mut(id) {
                    p.keyframes.clear();
                }
                for (t, v) in keys {
                    set_key(e, id, *t, *v);
                }
            }
        }
        Ok(())
    })?;
    Ok(json!({"clips": summary}))
}

/// The clips a 360° command acts on: `clips` / `clip`, else the selection, else every clip of
/// 360° media in the active sequence; each must be 360°.
pub(crate) fn clips_360(s: &Session, p: &Value, cmd: &str) -> Result<Vec<TrackItem>> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let mut ids = clips_p(s, p);
    if ids.is_empty() && p.get("clips").is_none_or(Value::is_null) && p.get("clip").is_none_or(Value::is_null) {
        ids = seq.video_tracks.iter().flat_map(|t| &t.items).filter(|it| s.project.vr_of(it.item).is_some()).map(|it| it.id).collect();
    }
    let mut out = Vec::new();
    for c in ids {
        let Some((track, it)) = seq.find_item(c) else { return Err(bad(cmd, format!("no clip {} in the active sequence", c.0))) };
        // (a selection holds a clip's linked sound too: only pictures are looked at)
        if seq.track(track).is_none_or(|t| t.kind != TrackKind::Video) {
            continue;
        }
        if s.project.vr_of(it.item).is_none() {
            return Err(bad(cmd, format!("clip {} is not 360° media (media.vrProperties says how a picture maps onto the sphere)", c.0)));
        }
        out.push(it.clone());
    }
    if out.is_empty() {
        return Err(bad(cmd, "no 360° clip given, selected or in the sequence"));
    }
    Ok(out)
}

/// The analysis source of a 360° item: its proxy when one is attached (an `.lrv`: the same
/// lenses, small and quick to decode) and of the media's shape (the lenses where they are in the
/// media's picture), else the media.
fn analysis_source(s: &Session, item: ItemId) -> Option<filmcraft_media::SharedSource> {
    let media = s.project.resolve_media(item).map(|(_, m, _)| m);
    let proxy = media.and_then(|m| match &m.proxy {
        Some(filmcraft_project::MediaRef::File { path }) if !m.offline => Some(path.clone()),
        _ => None,
    });
    let aspect = |v: Option<&filmcraft_media::VideoStreamInfo>| v.map(|v| f64::from(v.width.max(1)) / f64::from(v.height.max(1)));
    let want = aspect(media.and_then(|m| m.info.video.as_ref()));
    if let Some(p) = proxy
        && let Ok(src) = s.media.open_file(&p, &*s.services)
        && let (Some(a), Some(b)) = (aspect(src.info().video.as_ref()), want)
        && (a / b - 1.0).abs() < 0.01
    {
        return Some(src);
    }
    s.media.full_res_source(&s.project, item, &*s.services)
}

/// The analysed frames of a clip: every `fps`-th of a second of it (at most [`MAX_360_FRAMES`]),
/// as (seconds from the clip's start, media time).
fn clip_frames(it: &TrackItem, rate: filmcraft_time::FrameRate, fps: f64) -> Vec<(f64, Tick)> {
    let fd = rate.frame_duration().max(Tick(1));
    let n = (it.duration.0 / fd.0).max(1);
    let step = ((rate.as_f64() / fps.clamp(1.0, 120.0)).round() as i64).max(1);
    let mut stride = usize::try_from(step).unwrap_or(1).max(1);
    while usize::try_from(n).unwrap_or(usize::MAX) / stride > MAX_360_FRAMES {
        stride = stride.saturating_mul(2);
    }
    (0..n)
        .step_by(stride)
        .map(|k| {
            let st = it.start + Tick(fd.0.saturating_mul(k));
            ((st - it.start).seconds(), it.source_time_at(st))
        })
        .collect()
}

/// A linear picture of `src` at media time `t`, small (analysis).
fn analysis_picture(src: &filmcraft_media::SharedSource, t: Tick) -> Option<filmcraft_render::Image> {
    let f = src.video_frame(filmcraft_media::FrameRequest { time: t, scale: 0.5 }).ok()?;
    let dec = if f.width > 2048 { 2 } else { 1 };
    let (w, h, px) = f.to_linear_f32_decimated(dec);
    Some(filmcraft_render::Image { w, h, px })
}

/// How the stabiliser holds the view: the Gaussian σ (seconds) of the camera motion kept, or
/// `None` to keep the first frame's direction throughout.
fn stab_mode(name: &str) -> Option<Option<f64>> {
    Some(match name {
        "smooth" => Some(1.0),
        "light" => Some(0.35),
        "strong" => Some(2.5),
        "lock" | "lockDirection" => None,
        _ => return None,
    })
}

/// The camera's orientation at every analysed frame (camera → the first frame's camera), chained
/// from frame-to-frame rotations; a pair that cannot be measured holds the orientation. `None`
/// when cancelled; else with the count of unmeasured pairs.
fn orientations(src: &filmcraft_media::SharedSource, spherical: &Spherical, times: &[Tick], prog: &filmcraft_export::Progress) -> Option<(Vec<M3>, usize)> {
    use filmcraft_render::vr::{mmul, transpose};
    use filmcraft_render::vr_motion::{Faces, rotation_between};
    let mut orient = Vec::with_capacity(times.len());
    let mut prev: Option<Faces> = None;
    let mut r = filmcraft_render::vr::camera_rotation(0.0, 0.0, 0.0);
    let mut lost = 0usize;
    for t in times {
        if prog.cancel.load(Ordering::Relaxed) {
            return None;
        }
        prog.done.fetch_add(1, Ordering::Relaxed);
        let faces = analysis_picture(src, *t).map(|img| {
            // (the picture levelled by the recording's horizon, as it is shown)
            let model = filmcraft_render::vr::SourceModel::at(spherical, (img.w, img.h), *t);
            Faces::new(&img, &model, STAB_FACE)
        });
        match (prev.as_ref(), faces.as_ref()) {
            // b ≈ ΔR·a for the camera's directions: Rₖ = Rₖ₋₁ · ΔRᵀ
            (Some(a), Some(b)) => match rotation_between(a, b) {
                Some((dr, _)) => r = mmul(&r, &transpose(&dr)),
                None => lost += 1,
            },
            (Some(_), None) => lost += 1,
            _ => {}
        }
        orient.push(r);
        if faces.is_some() {
            prev = faces;
        }
    }
    Some((orient, lost))
}

fn stabilize(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = STAB_CMD;
    let mode = match str_p(p, "mode") {
        None => Some(1.0),
        Some(m) => stab_mode(m).ok_or_else(|| bad(cmd, format!("unknown mode `{m}` (smooth, light, strong, lock)")))?,
    };
    let clips = clips_360(s, p, cmd)?;
    if bool_p(p, "off").unwrap_or(false) {
        let ids: Vec<ClipId> = clips.iter().map(|c| c.id).collect();
        s.edit_sequence("360 Stabilize Off", |q, _, _| {
            for c in &ids {
                if let Some((_, it)) = q.find_item_mut(*c)
                    && let Some(e) = it.effect_mut(REFRAME)
                {
                    for id in STAB_PARAMS {
                        set_static(e, id, 0.0);
                    }
                }
            }
            Ok(())
        })?;
        return Ok(json!({"clips": ids.iter().map(|c| c.0).collect::<Vec<_>>(), "stabilized": false}));
    }
    let rate = s.sequence_rate().sane();
    let mut work = Vec::new();
    for it in &clips {
        let Some(sph) = s.project.vr_of(it.item).cloned() else { continue };
        let Some(src) = analysis_source(s, it.item) else { return Err(bad(cmd, format!("clip {}'s media is offline", it.id.0))) };
        // every frame: shake lives between frames
        work.push((it.id, src, sph, clip_frames(it, rate, rate.as_f64())));
    }
    let total: usize = work.iter().map(|w| w.3.len()).sum();
    let wait = bool_p(p, "wait").unwrap_or(false);
    let label = format!("Stabilizing {} 360° clip(s)", work.len());
    start_keys(s, cmd, "360 Stabilize", label, total, wait, move |prog| {
        let mut made = Vec::new();
        for (clip, src, sph, frames) in &work {
            let times: Vec<Tick> = frames.iter().map(|f| f.1).collect();
            let Some((orient, lost)) = orientations(src, sph, &times, prog) else { return Err("stopped: nothing was changed".into()) };
            let secs: Vec<f64> = frames.iter().map(|f| f.0).collect();
            let target: Vec<M3> = match mode {
                Some(sigma) => filmcraft_render::vr_motion::smooth(&orient, &secs, sigma),
                None => vec![filmcraft_render::vr::camera_rotation(0.0, 0.0, 0.0); orient.len()],
            };
            let corr = filmcraft_render::vr_motion::corrections(&orient, &target);
            let angles: Vec<(f64, f64, f64)> = corr.iter().map(filmcraft_render::vr::rotation_angles).collect();
            let ys = filmcraft_render::vr_motion::unwrap_degrees(&angles.iter().map(|a| a.0).collect::<Vec<_>>());
            let ps: Vec<f64> = angles.iter().map(|a| a.1).collect();
            let rs = filmcraft_render::vr_motion::unwrap_degrees(&angles.iter().map(|a| a.2).collect::<Vec<_>>());
            let keep = filmcraft_render::vr_motion::reduce(&secs, &[&ys, &ps, &rs], STAB_TOLERANCE_DEGREES);
            let key = |v: &[f64]| keep.iter().filter_map(|&i| Some((frames.get(i)?.1, *v.get(i)?))).collect::<Vec<_>>();
            let biggest = angles.iter().map(|a| a.0.abs().max(a.1.abs()).max(a.2.abs())).fold(0.0, f64::max);
            made.push(ClipKeys {
                clip: *clip,
                keys: vec![("stab_yaw", key(&ys)), ("stab_pitch", key(&ps)), ("stab_roll", key(&rs))],
                summary: json!({"clip": clip.0, "frames": frames.len(), "unmeasured": lost, "keyframes": keep.len(), "largestCorrection": (biggest * 10.0).round() / 10.0}),
            });
        }
        Ok(made)
    })
}

/// What to follow.
#[derive(Clone, Copy, Debug)]
enum FollowTarget {
    /// A box (shares of the virtual camera's frame at the seed time).
    Box(filmcraft_render::subject::SubjectBox),
    /// What stands out where the camera looks at the seed time.
    Subject,
    /// The largest face, looked for every half second.
    Face,
}

/// Where the subject was seen at one analysed frame: the virtual camera's yaw and pitch that
/// centre it (NaN where it was not seen).
struct Seen {
    yaw: f64,
    pitch: f64,
}

/// Follows the subject through frames `order` (indices into `frames`), starting where the
/// clip's camera looks at the first of them; once it is lost for a few frames it is looked for
/// again by its look (`refind`, [`crate::follow_reid`]). What was seen, and when it was found
/// again (seconds) and how alike it was then; `None` when cancelled.
#[allow(clippy::too_many_arguments)]
fn follow_run(
    src: &filmcraft_media::SharedSource,
    spherical: &Spherical,
    frames: &[(f64, Tick)],
    views: &[(filmcraft_render::vr::View, M3)],
    order: &[usize],
    target: FollowTarget,
    faces: Option<&crate::redact::DetectFn>,
    refind: Option<&crate::follow_reid::Judge>,
    prog: &filmcraft_export::Progress,
) -> Option<(Vec<(usize, Seen)>, Vec<(f64, f32, Option<f32>)>)> {
    use crate::follow_reid::{Look, MAX_SEARCHES, SEARCH_AFTER};
    use filmcraft_render::subject::{SubjectBox, SubjectTracker, TrackState};
    use filmcraft_render::vr_motion::{TRACK_FOV, TrackingCamera};
    let first = *order.first()?;
    let (start_view, _) = views.get(first)?;
    let (lon, lat) = filmcraft_render::vr::lonlat(start_view.ray(0.5, 0.5));
    let mut cam = TrackingCamera::new(lon.to_degrees(), lat.to_degrees(), FOLLOW_SIZE);
    // a box drawn in the clip's own frame: the same directions, seen by the tracking camera
    let mut hint: Option<SubjectBox> = match target {
        FollowTarget::Box(b) => {
            cam = cam.recentred(start_view.ray(b.cx, b.cy));
            let scale = (view_fov(start_view) / TRACK_FOV).clamp(0.05, 20.0);
            SubjectBox::new(0.5, 0.5, b.w * scale, b.h * scale)
        }
        _ => None,
    };
    let (fw, fh) = (FOLLOW_SIZE.0 as usize, FOLLOW_SIZE.1 as usize);
    let every = (FOLLOW_FPS / 2.0).round().max(1.0) as usize;
    let mut tr = SubjectTracker::new();
    let mut out = Vec::with_capacity(order.len());
    // the subject's look, its size in the tracking camera's picture, and how long it is lost
    let mut look = refind.map(|j| Look::new(j.clone()));
    let mut size = (0.15, 0.3);
    let (mut lost, mut searches, mut refound) = (0usize, 0usize, Vec::new());
    for (n, &k) in order.iter().enumerate() {
        if prog.cancel.load(Ordering::Relaxed) {
            return None;
        }
        prog.done.fetch_add(1, Ordering::Relaxed);
        let Some(&(t, mt)) = frames.get(k) else { continue };
        let Some(img) = analysis_picture(src, mt) else {
            let _ = tr.push(&[], 0, 0, t, None);
            out.push((k, Seen { yaw: f64::NAN, pitch: f64::NAN }));
            continue;
        };
        let model = filmcraft_render::vr::SourceModel::at(spherical, (img.w, img.h), mt);
        let pic = filmcraft_render::vr::reframe(&img, &model, &cam.view, fw, fh, 1).over_black_rgba8();
        if let (FollowTarget::Face, Some(detect)) = (target, faces)
            && n.is_multiple_of(every)
        {
            hint = crate::reframe::largest_face(detect, &pic, fw, fh).or(hint);
        }
        let obs = tr.push(&pic, fw, fh, t, hint.take());
        // held where it was last seen, or gone: looked for again by its look every few frames
        let missing = matches!(obs.state, TrackState::Coasting | TrackState::Lost | TrackState::None);
        lost = if missing { lost + 1 } else { 0 };
        if let Some(lk) = look.as_ref().filter(|l| l.ready())
            && lost >= SEARCH_AFTER
            && lost.is_multiple_of(SEARCH_AFTER)
            && searches < MAX_SEARCHES
        {
            // in the tracking camera's picture, and every other time in the views either side too
            let sides = searches.is_multiple_of(2);
            searches += 1;
            if let Some((c, fb, sim, agree)) = find_again(lk, &img, &model, &cam, &pic, size, sides) {
                refound.push((t, sim, agree));
                lost = 0;
                let d = c.as_ref().unwrap_or(&cam).direction(fb.cx, fb.cy);
                let aim = views.get(k).map_or(d, |(_, m)| filmcraft_render::vr::mtvec(m, d));
                let (lon, lat) = filmcraft_render::vr::lonlat(aim);
                out.push((k, Seen { yaw: lon.to_degrees(), pitch: lat.to_degrees() }));
                match c {
                    // another view: the tracking camera turned onto it, the tracker started again
                    Some(c) => {
                        cam = c.recentred(d);
                        tr = SubjectTracker::new();
                        hint = SubjectBox::new(0.5, 0.5, fb.w, fb.h);
                    }
                    None => {
                        tr = SubjectTracker::new();
                        hint = Some(fb);
                    }
                }
                continue;
            }
        }
        let Some(b) = obs.bbox.filter(|_| obs.confidence >= filmcraft_render::reframe::MIN_CONFIDENCE) else {
            out.push((k, Seen { yaw: f64::NAN, pitch: f64::NAN }));
            continue;
        };
        if !missing {
            size = (b.w, b.h);
            if let Some(lk) = look.as_mut().filter(|l| l.learning()) {
                lk.learn(&pic, fw, fh, b);
            }
        }
        let d = cam.direction(b.cx, b.cy);
        // the clip's stabilisation turns its rays first (final = C · view): aim the view at Cᵀ·d
        let aim = views.get(k).map_or(d, |(_, c)| filmcraft_render::vr::mtvec(c, d));
        let (lon, lat) = filmcraft_render::vr::lonlat(aim);
        out.push((k, Seen { yaw: lon.to_degrees(), pitch: lat.to_degrees() }));
        if cam.should_recentre(b.cx, b.cy) {
            // the tracking camera turned onto the subject, the tracker started again on its box
            cam = cam.recentred(d);
            tr = SubjectTracker::new();
            hint = SubjectBox::new(0.5, 0.5, b.w, b.h);
        }
    }
    Some((out, refound))
}

/// How far either side of the tracking camera the subject is looked for, degrees.
const FIND_SIDE: f64 = 0.9 * filmcraft_render::vr_motion::TRACK_FOV;

/// Where a lost subject (of `size`, shares) is now and how alike it is: in the tracking camera's
/// picture `pic` (no new camera), else — with `sides` — in the view either side of it (that
/// camera).
fn find_again(
    look: &crate::follow_reid::Look,
    img: &filmcraft_render::Image,
    model: &filmcraft_render::vr::SourceModel,
    cam: &filmcraft_render::vr_motion::TrackingCamera,
    pic: &[u8],
    size: (f64, f64),
    sides: bool,
) -> Option<(Option<filmcraft_render::vr_motion::TrackingCamera>, filmcraft_render::subject::SubjectBox, f32, Option<f32>)> {
    let (fw, fh) = (FOLLOW_SIZE.0 as usize, FOLLOW_SIZE.1 as usize);
    if let Some((b, sim, agree)) = look.search(pic, fw, fh, size.0, size.1) {
        return Some((None, b, sim, agree));
    }
    if !sides {
        return None;
    }
    let (lon, lat) = filmcraft_render::vr::lonlat(cam.direction(0.5, 0.5));
    let mut best: Option<(filmcraft_render::vr_motion::TrackingCamera, filmcraft_render::subject::SubjectBox, f32, Option<f32>)> = None;
    for side in [FIND_SIDE, -FIND_SIDE] {
        let c = filmcraft_render::vr_motion::TrackingCamera::new(lon.to_degrees() + side, lat.to_degrees(), FOLLOW_SIZE);
        let p = filmcraft_render::vr::reframe(img, model, &c.view, fw, fh, 1).over_black_rgba8();
        if let Some((b, sim, agree)) = look.search(&p, fw, fh, size.0, size.1)
            && best.as_ref().is_none_or(|x| sim > x.2)
        {
            best = Some((c, b, sim, agree));
        }
    }
    best.map(|(c, b, sim, agree)| (Some(c), b, sim, agree))
}

/// A flat view's field of view, degrees across (90 for the sphere).
fn view_fov(v: &filmcraft_render::vr::View) -> f64 {
    match v.output {
        filmcraft_render::vr::Output::Flat { tan_x, .. } => (2.0 * tan_x.atan()).to_degrees(),
        filmcraft_render::vr::Output::Sphere => 90.0,
    }
}

fn follow(s: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_render::subject::SubjectBox;
    let cmd = FOLLOW_CMD;
    let clips = clips_360(s, p, cmd)?;
    let [it] = clips.as_slice() else { return Err(bad(cmd, "follow one clip at a time (`clip`)")) };
    let at = match p.get("at") {
        None | Some(Value::Null) => 0.0,
        Some(v) => v.as_f64().filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| bad(cmd, "`at` is seconds from the clip's start"))?,
    };
    let smooth = match f64_p(p, "smooth") {
        None => 0.5,
        Some(v) if (0.0..=1.0).contains(&v) => v,
        Some(_) => return Err(bad(cmd, "`smooth` is 0 (close) … 1 (floaty)")),
    };
    let fps = match f64_p(p, "analysisFps") {
        None => FOLLOW_FPS,
        Some(v) if (1.0..=30.0).contains(&v) => v,
        Some(_) => return Err(bad(cmd, "`analysisFps` is 1..30")),
    };
    // how a lost subject is found again: SigLIP 2 when the scene models are downloaded
    let refind = match str_p(p, "refind").unwrap_or("auto") {
        "off" => None,
        "colour" | "color" => Some(crate::follow_reid::Judge::Colour),
        "auto" => {
            Some(s.scenes.check_models().filter(|m| m.embedder_id().is_some()).map_or(crate::follow_reid::Judge::Colour, crate::follow_reid::Judge::Confirmed))
        }
        "siglip" => Some(crate::follow_reid::Judge::Embedder(
            s.scenes
                .check_models()
                .filter(|m| m.embedder_id().is_some())
                .ok_or_else(|| bad(cmd, "the scene models (SigLIP 2) are not downloaded: `refind: \"colour\"`"))?,
        )),
        o => return Err(bad(cmd, format!("`refind` is auto, siglip, colour or off, not `{o}`"))),
    };
    let mut faces = None;
    let target = match str_p(p, "target").unwrap_or("subject") {
        "subject" => FollowTarget::Subject,
        "box" => {
            let b =
                p.get("box").and_then(Value::as_array).filter(|a| a.len() == 4).ok_or_else(|| bad(cmd, "`box` is [x, y, w, h], shares of the clip's frame"))?;
            let v: Vec<f64> =
                b.iter().map(|x| x.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, "`box` values are numbers"))).collect::<Result<_>>()?;
            let bx = SubjectBox::from_corner(v[0], v[1], v[2], v[3])
                .filter(|_| (0.0..=1.0).contains(&v[2]) && (0.0..=1.0).contains(&v[3]))
                .ok_or_else(|| bad(cmd, "`box` is outside the frame"))?;
            FollowTarget::Box(bx)
        }
        "face" => {
            let detect = match s.redacts.detector.clone() {
                Some(d) => d,
                None => crate::privacy_check::face_detector(s, cmd, Some(false))?,
            };
            faces = Some(detect);
            FollowTarget::Face
        }
        o => return Err(bad(cmd, format!("`target` is subject, box or face, not `{o}`"))),
    };
    let rate = s.sequence_rate().sane();
    let frames = clip_frames(it, rate, fps);
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let frame = (seq.settings.width, seq.settings.height);
    // the clip's camera at each analysed frame (its own keyframes, its stabilisation), or the
    // natural view of a clip not reframed yet
    let views: Vec<(filmcraft_render::vr::View, M3)> = frames
        .iter()
        .map(|(_, mt)| match filmcraft_project::vr::reframe_of(it) {
            Some(e) => (
                filmcraft_render::vr::clip_view(e, *mt).map_or_else(|| filmcraft_render::vr::View::flat(0.0, 0.0, 0.0, 90.0, 0.0, frame.0, frame.1), |v| v.0),
                filmcraft_render::vr::stabilization(e, *mt),
            ),
            None => (filmcraft_render::vr::View::flat(0.0, 0.0, 0.0, 90.0, 0.0, frame.0, frame.1), filmcraft_render::vr::camera_rotation(0.0, 0.0, 0.0)),
        })
        .collect();
    let seed = frames.iter().position(|f| f.0 >= at - 1e-9).unwrap_or(0);
    let Some(sph) = s.project.vr_of(it.item).cloned() else { return Err(bad(cmd, "not 360° media")) };
    let Some(src) = analysis_source(s, it.item) else { return Err(bad(cmd, "the clip's media is offline")) };
    let clip = it.id;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let n = frames.len();
    // a calm operator at 1, a close one at 0 (the spring's angular frequency, rad/s)
    let omega = 6.0 * (1.0 - smooth) + 0.8 * smooth;
    start_keys(s, cmd, "360 Follow Subject", format!("Following the subject in {}", it.name), n, wait, move |prog| {
        // from the seed back to the start, then on to the end
        let back: Vec<usize> = (0..=seed.min(n.saturating_sub(1))).rev().collect();
        let fwd: Vec<usize> = (seed..n).collect();
        let mut seen: Vec<Option<Seen>> = (0..n).map(|_| None).collect();
        // the embedder on the GPU while the follow runs (without the GPU, the colours)
        let (refind, _gpu) = match refind {
            Some(j) => match j.models().map(|m| m.gpu()) {
                Some(Ok(g)) => (Some(j), g),
                Some(Err(_)) => (Some(crate::follow_reid::Judge::Colour), None),
                None => (Some(j), None),
            },
            None => (None, None),
        };
        let mut refound: Vec<(f64, f32, Option<f32>)> = Vec::new();
        for order in [back, fwd] {
            if order.is_empty() {
                continue;
            }
            let (run, again) = follow_run(&src, &sph, &frames, &views, &order, target, faces.as_ref(), refind.as_ref(), prog)
                .ok_or_else(|| "stopped: nothing was changed".to_string())?;
            refound.extend(again);
            for (k, sn) in run {
                if let Some(slot) = seen.get_mut(k)
                    && slot.as_ref().is_none_or(|old| old.yaw.is_nan())
                {
                    *slot = Some(sn);
                }
            }
        }
        let secs: Vec<f64> = frames.iter().map(|f| f.0).collect();
        let found = seen.iter().filter(|x| x.as_ref().is_some_and(|x| x.yaw.is_finite())).count();
        if found * 10 < n * 3 {
            return Err(format!("the subject was seen in only {found} of {n} analysed frames: try `target: \"box\"` around it"));
        }
        let yaws = filmcraft_render::vr_motion::unwrap_degrees(&seen.iter().map(|x| x.as_ref().map_or(f64::NAN, |x| x.yaw)).collect::<Vec<_>>());
        let pitches: Vec<f64> = seen.iter().map(|x| x.as_ref().map_or(f64::NAN, |x| x.pitch)).collect();
        let ys = filmcraft_render::vr_motion::smooth_path(&secs, &yaws, omega);
        let ps: Vec<f64> = filmcraft_render::vr_motion::smooth_path(&secs, &pitches, omega).into_iter().map(|v| v.clamp(-85.0, 85.0)).collect();
        let keep = filmcraft_render::vr_motion::reduce(&secs, &[&ys, &ps], FOLLOW_TOLERANCE_DEGREES);
        let key = |v: &[f64]| keep.iter().filter_map(|&i| Some((frames.get(i)?.1, *v.get(i)?))).collect::<Vec<_>>();
        Ok(vec![ClipKeys {
            clip,
            keys: vec![("yaw", key(&ys)), ("pitch", key(&ps))],
            summary: json!({
                "clip": clip.0,
                "frames": n,
                "found": (found as f64 / n.max(1) as f64 * 1000.0).round() / 1000.0,
                "keyframes": keep.len(),
                "foundAgain": refound.len(),
                // when (clip seconds) and how alike (cosine), the first few
                "foundAgainAt": refound
                    .iter()
                    .take(20)
                    .map(|(t, m, a)| {
                        let r3 = |x: f32| (f64::from(x) * 1000.0).round() / 1000.0;
                        json!({"seconds": (t * 100.0).round() / 100.0, "match": r3(*m), "siglip": a.map(r3)})
                    })
                    .collect::<Vec<_>>(),
                "refind": refind.as_ref().map(crate::follow_reid::Judge::name),
            }),
        }])
    })
}

/// Sessions with 360° media for the tests of the modules that build on this one.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// The demo project, its first clip's media taken as a whole equirectangular panorama.
    pub fn demo_360() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.state.selection.clear();
        let (_, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        s
    }

    pub fn first_clip(s: &Session) -> (ClipId, ItemId) {
        let it = &s.active_sequence().unwrap().video_tracks[0].items[0];
        (it.id, it.item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.state.selection.clear();
        s
    }

    fn first_clip(s: &Session) -> (ClipId, ItemId) {
        let it = &s.active_sequence().unwrap().video_tracks[0].items[0];
        (it.id, it.item)
    }

    fn clip(s: &Session, c: ClipId) -> TrackItem {
        s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
    }

    #[test]
    fn reframing_adds_a_virtual_camera_framing_the_sequence() {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        let r = s.execute(REFRAME_CMD, json!({"clips": [c.0], "yaw": 40.0, "fov": 100.0})).unwrap();
        assert!(r["note"].is_string(), "the demo clip has no VR Properties: {r}");
        let it = clip(&s, c);
        let e = it.effect(REFRAME).unwrap();
        let seq = s.active_sequence().unwrap();
        assert_eq!(filmcraft_project::vr::effect_size(e), Some((seq.settings.width, seq.settings.height)));
        assert_eq!(s.project.clip_source_size(&it), Some((seq.settings.width, seq.settings.height)));
        assert_eq!(e.f64_at("yaw", Tick::ZERO), 40.0);
        // the virtual camera's picture renders (the demo picture taken as a panorama; mid-clip,
        // past the demo's fade in)
        let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
        let opts = filmcraft_render::RenderOptions { scale: 0.25, ..Default::default() };
        let mid = it.start + Tick(it.duration.0 / 2);
        let img = filmcraft_render::render_sequence(&s.project, s.state.active_sequence.unwrap(), mid, opts, &provider);
        assert!(img.px.iter().any(|v| *v > 0.0));
        // a keyframed move, then off
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "keys": [{"seconds": 0.0, "yaw": 0.0}, {"seconds": 1.0, "yaw": 90.0, "pitch": 10.0}]})).unwrap();
        let it = clip(&s, c);
        let e = it.effect(REFRAME).unwrap();
        assert_eq!(e.param("yaw").unwrap().keyframes.len(), 2);
        assert_eq!(e.param("pitch").unwrap().keyframes.len(), 1);
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "off": true})).unwrap();
        let it = clip(&s, c);
        assert!(!it.effect(REFRAME).unwrap().enabled);
        assert_eq!(s.project.clip_source_size(&it), s.project.source_size(item));
        // one undo step each
        s.execute("edit.undo", json!({})).unwrap();
        assert!(clip(&s, c).effect(REFRAME).unwrap().enabled);
    }

    #[test]
    fn presets_and_vr_properties() {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular", "layout": "overUnder"})).unwrap();
        assert_eq!(s.project.vr_of(item).unwrap().layout, StereoLayout::OverUnder);
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "preset": "tinyPlanet"})).unwrap();
        let e = clip(&s, c).effect(REFRAME).cloned().unwrap();
        assert_eq!((e.f64_at("pitch", Tick::ZERO), e.f64_at("distance", Tick::ZERO)), (-90.0, 1.0));
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "dualFisheye", "halfFov": 99.0})).unwrap();
        let Projection::DualFisheye { lenses } = &s.project.vr_of(item).unwrap().projection else { panic!() };
        assert_eq!(lenses[1].half_fov, 99.0);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "none"})).unwrap();
        assert!(s.project.vr_of(item).is_none());
        // the demo footage says nothing about the sphere
        s.execute(PROPS_CMD, json!({"items": [item.0]})).unwrap();
        assert!(s.project.vr_of(item).is_none());
        let r = s.execute(ALIGN_CMD, json!({"items": [item.0]})).unwrap();
        assert_eq!(r["items"][0]["aligned"], false);
    }

    #[test]
    fn the_horizon_is_switched_and_summed_up() {
        let mut s = demo();
        let (_, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "dualFisheye"})).unwrap();
        // the demo footage has no motion sensor to level by
        let e = s.execute(PROPS_CMD, json!({"items": [item.0], "level": true})).unwrap_err().to_string();
        assert!(e.contains("motion-sensor"), "{e}");
        let before = s.project.vr_of(item).cloned().unwrap();
        let key = sphere_key(&before);
        // (as an Insta360 recording brings it)
        let h =
            filmcraft_media::spherical::Horizon { level: true, steady: true, step: 0.1, up: vec![[0.0, 1.0, 0.0], [0.5, 0.866, 0.0]], heading: vec![0.0, 3.0] };
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "none"})).unwrap();
        s.edit("Horizon", |pr, _| {
            if let Some(ItemKind::Media(m)) = pr.item_mut(item).map(|i| &mut i.kind) {
                m.interpret.vr = Some(Spherical { horizon: Some(h), ..before });
            }
            Ok(())
        })
        .unwrap();
        let q = s.execute(PROPS_CMD, json!({"items": [item.0]})).unwrap();
        assert_eq!(q["items"][0]["vr"]["horizon"], json!({"level": true, "seconds": 0.1, "samples": 2, "maxTilt": 30.0, "steady": true, "maxSteady": 3.0}));
        // the heading steadied, or not
        s.execute(PROPS_CMD, json!({"items": [item.0], "steady": false})).unwrap();
        assert!(s.project.vr_of(item).unwrap().heading_at(0.1).is_none() && s.project.vr_of(item).unwrap().up_at(0.1).is_some());
        s.execute(PROPS_CMD, json!({"items": [item.0], "steady": true})).unwrap();
        assert_eq!(s.project.vr_of(item).unwrap().heading_at(0.1), Some(3.0));
        let on = sphere_key(s.project.vr_of(item).unwrap());
        assert_ne!(on, key, "the panorama's caches change with the horizon");
        s.execute(PROPS_CMD, json!({"items": [item.0], "level": false})).unwrap();
        assert!(s.project.vr_of(item).unwrap().up_at(0.1).is_none());
        assert_ne!(sphere_key(s.project.vr_of(item).unwrap()), on);
        // back on: the samples it kept
        s.execute(PROPS_CMD, json!({"items": [item.0], "level": true})).unwrap();
        assert!(s.project.vr_of(item).unwrap().up_at(0.1).is_some());
        s.undo().unwrap();
        assert!(s.project.vr_of(item).unwrap().up_at(0.1).is_none());
    }

    /// The demo's first clip, its media taken as an equirectangular panorama, and its sequence's
    /// clips selected (the linked sound with the picture).
    fn demo_360() -> (Session, ClipId) {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        let linked: Vec<ClipId> = s.active_sequence().unwrap().all_tracks().flat_map(|t| &t.items).filter(|i| i.item == item).map(|i| i.id).collect();
        s.state.selection = linked;
        (s, c)
    }

    #[test]
    fn stabilising_writes_the_cameras_steadying_turn() {
        let (mut s, c) = demo_360();
        let r = s.execute(STAB_CMD, json!({"mode": "light", "wait": true})).unwrap();
        let clips = r["result"]["clips"].as_array().unwrap();
        // the picture only: the selection's linked sound is not looked at
        assert_eq!(clips.len(), 1, "{r}");
        assert_eq!(clips[0]["clip"], c.0);
        let it = clip(&s, c);
        let e = it.effect(REFRAME).expect("the virtual camera was added");
        assert!(STAB_PARAMS.iter().all(|id| e.param(id).is_some_and(|p| !p.keyframes.is_empty())));
        assert!(s.active_sequence().unwrap().audio_tracks.iter().flat_map(|t| &t.items).all(|a| a.effect(REFRAME).is_none()));
        // one undo step; off clears it
        s.execute(STAB_CMD, json!({"clips": [c.0], "off": true})).unwrap();
        assert!(clip(&s, c).effect(REFRAME).unwrap().param("stab_yaw").unwrap().keyframes.is_empty());
        s.execute("edit.undo", json!({})).unwrap();
        assert!(!clip(&s, c).effect(REFRAME).unwrap().param("stab_yaw").unwrap().keyframes.is_empty());
        let job = r["job"].as_u64().unwrap();
        assert!(outcome_json(&s, job).is_some());
    }

    #[test]
    fn following_turns_the_camera_or_says_why_not() {
        let (mut s, c) = demo_360();
        let r = s.execute(FOLLOW_CMD, json!({"clip": c.0, "target": "box", "box": [0.35, 0.3, 0.3, 0.4], "at": 1.0, "analysisFps": 5, "wait": true}));
        match r {
            Ok(v) => {
                assert_eq!(v["result"]["clips"][0]["clip"], c.0);
                assert!(clip(&s, c).effect(REFRAME).unwrap().param("yaw").is_some_and(|p| !p.keyframes.is_empty()));
            }
            Err(e) => assert!(e.to_string().contains("seen in only"), "{e}"),
        }
    }

    /// A 360° image sequence of a red-and-blue figure walking right, blocked by a passer-by for
    /// 12 frames and then 70° further on (out of the tracking camera's view): its frames'
    /// paths and the figure's yaw at each.
    fn blocked_walk(dir: &std::path::Path) -> (Vec<String>, Vec<f64>) {
        let (w, h) = (1024u32, 512u32);
        let mut paths = Vec::new();
        let mut yaws = Vec::new();
        for k in 0..80u32 {
            let yaw = if k < 30 { -30.0 + f64::from(k) } else { 60.0 + f64::from(k - 30) };
            let blocked = (18..30).contains(&k);
            let img = image::RgbaImage::from_fn(w, h, |x, y| {
                let (lon, lat) = (f64::from(x) / f64::from(w) * 360.0 - 180.0, 90.0 - f64::from(y) / f64::from(h) * 180.0);
                // a faint pattern on the walls, the passer-by, the figure
                let wall = 100 + ((lon / 9.0).floor() as i64 + (lat / 9.0).floor() as i64).rem_euclid(2) as u8 * 12;
                if blocked && (lon - (yaw + 3.0)).abs() < 14.0 && lat.abs() < 30.0 {
                    image::Rgba([60, 50, 40, 255])
                } else if (lon - yaw).abs() < 7.0 && lat.abs() < 18.0 {
                    if lat > 0.0 { image::Rgba([210, 30, 30, 255]) } else { image::Rgba([30, 40, 210, 255]) }
                } else {
                    image::Rgba([wall, wall, wall, 255])
                }
            });
            let p = dir.join(format!("walk{k:04}.png"));
            img.save(&p).unwrap();
            paths.push(p.to_string_lossy().into_owned());
            yaws.push(yaw);
        }
        (paths, yaws)
    }

    #[test]
    fn a_blocked_subject_is_found_again_and_followed_on() {
        let dir = crate::media_test_util::tmp_dir("follow-blocked");
        let (paths, yaws) = blocked_walk(&dir);
        let follow = |refind: &str| {
            let mut s = Session::default();
            let r = s.execute("file.import", json!({"paths": [paths[0]], "imageSequence": true})).unwrap();
            let item = r["items"][0].as_u64().unwrap();
            s.execute(PROPS_CMD, json!({"items": [item], "projection": "equirectangular"})).unwrap();
            s.execute("file.newSequence", json!({"name": "walk", "width": 320, "height": 180, "fps": 24})).unwrap();
            let r = s.execute("timeline.place", json!({"item": item, "track": "V1", "seconds": 0})).unwrap();
            let c = ClipId(r["reframed360"][0].as_u64().unwrap());
            // the figure in the natural view (100° across) at the start: 30° left
            let r = s
                .execute(
                    FOLLOW_CMD,
                    json!({"clip": c.0, "target": "box", "box": [0.2, 0.25, 0.12, 0.5], "analysisFps": 24, "smooth": 0, "refind": refind, "wait": true}),
                )
                .unwrap();
            let it = clip(&s, c);
            let e = it.effect(REFRAME).unwrap().clone();
            // the camera's yaw where the sequence shows source frame `k`
            let rate = s.source(ItemId(item)).unwrap().info().frame_rate();
            let yaw_at = |k: i64| e.f64_at("yaw", rate.tick_of(k));
            (r["result"]["clips"][0].clone(), yaw_at(3), yaw_at(76))
        };
        let (found, start, end) = follow("colour");
        assert!(found["foundAgain"].as_u64().unwrap() >= 1, "{found}");
        assert!((start - yaws[3]).abs() < 8.0, "followed from the start: {start} vs {}", yaws[3]);
        assert!((end - yaws[76]).abs() < 10.0, "found again and followed on: {end} vs {}: {found}", yaws[76]);
        // without looking for it again, the camera stays where it lost the figure
        let (lost, _, end_off) = follow("off");
        assert_eq!(lost["foundAgain"], 0);
        assert!((end_off - yaws[76]).abs() > 30.0, "{end_off}: {lost}");
    }

    #[test]
    fn hostile_parameters_are_refused() {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        for bad in [
            json!({"clips": [c.0], "yaw": "left"}),
            json!({"clips": [c.0], "pitch": 120.0}),
            json!({"clips": [c.0], "fov": 0.0}),
            json!({"clips": [c.0], "distance": f64::MAX}),
            json!({"clips": [c.0], "width": 3}),
            json!({"clips": [c.0], "width": -5}),
            json!({"clips": [c.0], "preset": "fisheye"}),
            json!({"clips": [c.0], "output": "cube"}),
            json!({"clips": [c.0], "at": -1.0}),
            json!({"clips": [c.0], "keys": [{"yaw": 1.0}]}),
            json!({"clips": [c.0], "keys": "now"}),
            json!({"clips": [999_999]}),
        ] {
            assert!(s.execute(REFRAME_CMD, bad.clone()).is_err(), "{bad}");
        }
        for bad in [
            json!({"items": [item.0], "projection": "cubemap"}),
            json!({"items": [item.0], "layout": "diagonal"}),
            json!({"items": [item.0], "capturedView": [400.0, 10.0]}),
            json!({"items": [item.0], "capturedView": "all"}),
            json!({"items": [item.0], "halfFov": 1e9}),
            json!({"items": []}),
        ] {
            assert!(s.execute(PROPS_CMD, bad.clone()).is_err(), "{bad}");
        }
        assert!(s.execute(ALIGN_CMD, json!({"items": [424_242]})).is_err());
        // the demo clip is not 360° until its VR Properties say so
        assert!(s.execute(STAB_CMD, json!({"clips": [c.0]})).is_err());
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        for bad in [json!({"clips": [c.0], "mode": "shaky"}), json!({"clips": [999_999]})] {
            assert!(s.execute(STAB_CMD, bad.clone()).is_err(), "{bad}");
        }
        for bad in [
            json!({"clip": c.0, "target": "dog"}),
            json!({"clip": c.0, "target": "box"}),
            json!({"clip": c.0, "target": "box", "box": [0.1, 0.1, 2.0, 0.2]}),
            json!({"clip": c.0, "target": "box", "box": ["a", 0, 0, 0]}),
            json!({"clip": c.0, "smooth": 3.0}),
            json!({"clip": c.0, "analysisFps": 0}),
            json!({"clip": c.0, "at": -2.0}),
            json!({"clip": c.0, "refind": "smell"}),
        ] {
            assert!(s.execute(FOLLOW_CMD, bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn placing_360_media_frames_it_through_a_camera() {
        let mut s = demo();
        let (_, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        let r = s.execute("timeline.place", json!({"item": item.0, "track": "V1", "seconds": 0.0, "audio": false})).unwrap();
        let c = ClipId(r["clips"][0].as_u64().unwrap());
        assert_eq!(r["reframed360"], json!([c.0]), "{r}");
        let it = clip(&s, c);
        let seq = s.active_sequence().unwrap();
        let frame = (seq.settings.width, seq.settings.height);
        assert_eq!(filmcraft_project::vr::reframe_size(&it), Some(frame));
        assert_eq!(it.effects.first().map(|e| e.effect.as_str()), Some(REFRAME), "first in the chain");
        assert_eq!(it.effect("motion").unwrap().f64_at("scale", Tick::ZERO), 100.0, "the camera's picture is the frame: no media scaling");
        // analyses in the clip's pixels see the camera's picture
        let v = clip_source(&s, &it).unwrap().info().video.clone().unwrap();
        assert_eq!((v.width, v.height), frame);
        // a dual-fisheye recording is analysed as its stitched panorama
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "dualFisheye"})).unwrap();
        let pano = analysis_view(&s, item).unwrap().info().video.clone().unwrap();
        assert_eq!(pano.width, 2 * pano.height);
        // a flat picture is placed as it always was
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "none"})).unwrap();
        let r = s.execute("timeline.place", json!({"item": item.0, "track": "V1", "seconds": 0.0, "audio": false})).unwrap();
        assert!(r.get("reframed360").is_none(), "{r}");
        assert!(clip(&s, ClipId(r["clips"][0].as_u64().unwrap())).effect(REFRAME).is_none());
    }

    #[test]
    fn the_export_says_video_is_vr_as_asked() {
        let s = demo();
        let vr = |p: Value| crate::export_tools::settings_from_params(&s, &p, "file.exportMedia").map(|(_, st)| st.vr);
        assert_eq!(vr(json!({"format": "h264"})).unwrap(), None, "unset: as the sequence is");
        assert_eq!(vr(json!({"format": "h264", "vr": true})).unwrap(), Some(true));
        assert_eq!(vr(json!({"format": "h264", "vr": false})).unwrap(), Some(false));
        assert_eq!(vr(json!({"format": "h264", "vr": "auto"})).unwrap(), None);
        assert_eq!(vr(json!({"format": "h264", "settings": {"vr": true}})).unwrap(), Some(true));
        assert!(vr(json!({"format": "h264", "vr": "yes"})).is_err());
        // the demo is not a 360° sequence
        assert!(!s.project.is_360_sequence(s.state.active_sequence.unwrap()));
    }

    #[test]
    fn a_360_export_imports_as_360_and_asking_changes_nothing() {
        let mut s = demo();
        let (_, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        s.execute("file.newSequence", json!({"name": "pano", "width": 320, "height": 160, "fps": 24.0})).unwrap();
        s.execute("timeline.place", json!({"item": item.0, "track": "V1", "seconds": 0.0, "audio": false})).unwrap();
        let dir = std::env::temp_dir().join(format!("fc-vr360-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pano.mp4").to_string_lossy().to_string();
        let r = s
            .execute(
                "file.exportMedia",
                json!({"path": path, "format": "h264", "audio": false, "range": "custom", "startSeconds": 0.0, "endSeconds": 0.25, "wait": true}),
            )
            .unwrap();
        let path = r["path"].as_str().unwrap().to_string();
        // a 360° sequence exports as 360° video by itself, and imports as 360° media
        let mut t = Session::default();
        let pano = ItemId(t.execute("file.import", json!({"paths": [path]})).unwrap()["items"][0].as_u64().unwrap());
        assert_eq!(t.project.vr_of(pano), Some(&Spherical { camera: format!("FilmCraft {}", env!("CARGO_PKG_VERSION")), ..Spherical::equirect() }));
        // asking is not changing: the properties stay, and nothing is undone
        let undo = t.history.undo.len();
        let r = t.execute(PROPS_CMD, json!({"items": [pano.0]})).unwrap();
        assert_eq!(r["items"][0]["vr"]["projection"]["kind"], json!("equirectangular"), "{r}");
        assert!(t.project.vr_of(pano).is_some());
        assert_eq!(t.history.undo.len(), undo);
        // `none` clears them; `auto` reads the file again (through the media pool)
        t.execute(PROPS_CMD, json!({"items": [pano.0], "projection": "none"})).unwrap();
        assert!(t.project.vr_of(pano).is_none());
        t.execute(PROPS_CMD, json!({"items": [pano.0], "projection": "auto"})).unwrap();
        assert!(t.project.vr_of(pano).is_some_and(Spherical::is_equirect));
        // a layout alone changes the item's own properties
        t.execute(PROPS_CMD, json!({"items": [pano.0], "layout": "overUnder"})).unwrap();
        assert!(t.project.vr_of(pano).is_some_and(|v| v.is_equirect() && v.layout == StereoLayout::OverUnder));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_hand_and_stick_are_patched_for_planets_and_on_request() {
        let (mut s, c) = demo_360();
        let item = clip(&s, c).item;
        let nadir = |s: &Session| s.project.vr_of(item).and_then(|v| v.nadir);
        assert!(nadir(&s).is_none());
        // a tiny planet patches it, in its own undo step
        let n0 = s.history.undo.len();
        let r = s.execute("clip.vrMove", json!({"clips": [c.0], "move": "planetIn"})).unwrap();
        assert_eq!(r["nadirPatched"], json!([item.0]), "{r}");
        assert_eq!(nadir(&s).map(|n| n.mode), Some(filmcraft_media::spherical::NadirMode::Mirror));
        assert_eq!(s.history.undo.len(), n0 + 1);
        // switched off on purpose: a planet leaves it off
        s.execute(PROPS_CMD, json!({"items": [item.0], "nadir": "off"})).unwrap();
        let r = s.execute("clip.vrMove", json!({"clips": [c.0], "move": "planetOut"})).unwrap();
        assert!(r.get("nadirPatched").is_none());
        assert_eq!(nadir(&s).map(|n| n.mode), Some(filmcraft_media::spherical::NadirMode::Off));
        // by hand, and its size; the analyses' key does not change with it
        let key = sphere_key(s.project.vr_of(item).unwrap());
        s.execute(PROPS_CMD, json!({"items": [item.0], "nadir": "smooth", "nadirSize": 30})).unwrap();
        assert_eq!(nadir(&s).map(|n| (n.mode, n.size)), Some((filmcraft_media::spherical::NadirMode::Smooth, 30.0)));
        assert_eq!(sphere_key(s.project.vr_of(item).unwrap()), key);
        for bad in [json!({"items": [item.0], "nadir": "patch"}), json!({"items": [item.0], "nadirSize": 90})] {
            assert!(s.execute(PROPS_CMD, bad.clone()).is_err(), "{bad}");
        }
        // a logo laid round the bottom, seen through the clip's camera
        let dir = crate::media_test_util::tmp_dir("nadir-logo");
        let png = dir.join("logo.png");
        let logo: Vec<u8> = (0..64 * 64).flat_map(|_| [200u8, 30, 30, 255]).collect();
        std::fs::write(&png, filmcraft_export::encode_png(logo, 64, 64).unwrap()).unwrap();
        let img = s.execute("file.import", json!({"paths": [png.to_string_lossy()]})).unwrap()["items"][0].as_u64().unwrap();
        let r = s.execute("clip.nadirLogo", json!({"clip": c.0, "image": img, "folder": dir.to_string_lossy()})).unwrap();
        let logo_clip = clip(&s, ClipId(r["logo"].as_u64().unwrap()));
        assert!(logo_clip.effect(REFRAME).is_some(), "through the clip's own camera");
        assert!(s.project.vr_of(logo_clip.item).is_some_and(Spherical::is_equirect));
        assert!(s.execute("clip.nadirLogo", json!({"clip": c.0})).is_err(), "needs an image");
    }

    #[test]
    fn salience_points_at_where_things_happen() {
        let mut s = demo();
        let (_, item) = first_clip(&s);
        assert!(s.execute(SALIENCE_CMD, json!({"item": item.0})).is_err(), "not 360° yet");
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        let r = s.execute(SALIENCE_CMD, json!({"item": item.0, "frames": 4, "max": 3})).unwrap();
        let dirs = r["directions"].as_array().unwrap();
        assert!(!dirs.is_empty() && dirs.len() <= 3, "{r}");
        assert!(dirs.iter().all(|d| d["yaw"].as_f64().is_some_and(|y| (-180.0..=180.0).contains(&y))), "{r}");
        assert_eq!(r["seconds"].as_array().unwrap().len(), 4);
        for bad in [json!({}), json!({"item": item.0, "from": -1}), json!({"item": 424_242})] {
            assert!(s.execute(SALIENCE_CMD, bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_cameras_key_changes_with_what_it_shows() {
        let def = filmcraft_project::find_effect(REFRAME).unwrap();
        let (a, mut b) = (def.instance(), def.instance());
        assert_eq!(camera_key(&a), camera_key(&b));
        assert_eq!(camera_key(&a).len(), 16);
        set_static(&mut b, "yaw", 10.0);
        assert_ne!(camera_key(&a), camera_key(&b));
    }

    #[test]
    fn a_camera_is_reshaped_for_another_frame() {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "yaw": 30.0, "fov": 90.0})).unwrap();
        let mut it = clip(&s, c);
        assert!(reshape_camera(&mut it, (1920, 1080), (1080, 1920)));
        let e = it.effect(REFRAME).unwrap();
        assert_eq!(filmcraft_project::vr::effect_size(e), Some((1080, 1920)));
        // a 16:9 view's height (58.7°) becomes the vertical view's width; the direction is kept
        assert!((e.f64_at("fov", Tick::ZERO) - 58.72).abs() < 0.05, "{}", e.f64_at("fov", Tick::ZERO));
        assert_eq!(e.f64_at("yaw", Tick::ZERO), 30.0);
        // and back
        assert!(reshape_camera(&mut it, (1080, 1920), (1920, 1080)));
        assert!((it.effect(REFRAME).unwrap().f64_at("fov", Tick::ZERO) - 90.0).abs() < 1e-6);
        // a whole-sphere camera outside a 2:1 frame becomes a natural flat view
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "preset": "sphere"})).unwrap();
        let mut it = clip(&s, c);
        assert!(reshape_camera(&mut it, (1920, 1080), (1080, 1920)));
        let e = it.effect(REFRAME).unwrap();
        assert!(matches!(e.param("output").map(|p| &p.value), Some(ParamValue::Choice(0))));
        assert!((e.f64_at("fov", Tick::ZERO) - 58.72).abs() < 0.05);
        let mut flat = it.clone();
        flat.effects.retain(|e| e.effect != REFRAME);
        assert!(!reshape_camera(&mut flat, (1920, 1080), (1080, 1920)));
    }

    #[test]
    fn auto_reframe_keeps_360_cameras_and_reshapes_them() {
        let mut s = demo();
        let (c, item) = first_clip(&s);
        s.execute(PROPS_CMD, json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        s.execute(REFRAME_CMD, json!({"clips": [c.0], "yaw": -40.0})).unwrap();
        let (src, before) = (s.state.active_sequence.unwrap(), filmcraft_project::vr::reframe_size(&clip(&s, c)));
        let r = s.execute("sequence.autoReframe", json!({"aspect": "9:16", "analysisFps": 1, "wait": true})).unwrap();
        assert_eq!(r["cameras360"], json!(1), "{r}");
        let out = &r["reframe"];
        let seq = ItemId(out["sequence"].as_u64().unwrap_or_else(|| panic!("{r}")));
        let q = s.project.sequence(seq).unwrap();
        let frame = (q.settings.width, q.settings.height);
        let it = q.find_item(c).map(|(_, it)| it.clone()).unwrap_or_else(|| panic!("the copy keeps clip ids: {r}"));
        let e = it.effect(REFRAME).unwrap();
        assert_eq!(filmcraft_project::vr::effect_size(e), Some(frame));
        assert_eq!(e.f64_at("yaw", Tick::ZERO), -40.0);
        // not followed by moving its picture: Motion stays centred on the new frame
        let m = it.effect("motion").unwrap();
        assert!(m.param("position").unwrap().keyframes.is_empty());
        let pos = m.param("position").unwrap().value.as_vec2().unwrap();
        assert!((pos.x - f64::from(frame.0) / 2.0).abs() < 1.0 && (pos.y - f64::from(frame.1) / 2.0).abs() < 1.0, "{pos:?} in {frame:?}");
        // the source sequence's camera is untouched
        let kept = s.project.sequence(src).and_then(|q| q.find_item(c)).map(|(_, it)| filmcraft_project::vr::reframe_size(it));
        assert_eq!(kept, Some(before));
    }
}
