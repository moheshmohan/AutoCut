//! 360° video in the edit (`filmcraft_project::vr`, `filmcraft_render::vr`).
//!
//! | command | does |
//! |---|---|
//! | `clip.reframe360` | looks at 360° clips (`clips`, default the selection, else every clip of 360° media) through a virtual camera (the `vr_reframe` effect, added when missing, framing the sequence's frame): `yaw` (turn right), `pitch` (look up), `roll`, `fov` (degrees across) and `distance` (0 flat … 1 tiny planet) set, or keyframed `at` a clip time, or a whole camera move as `keys`; a `preset` (`natural`, `wide`, `tinyPlanet`, `rabbitHole`, `sphere`) first; one undo step |
//! | `media.vrProperties` | Interpret Footage ▸ VR Properties: `projection` `auto` (what the file says), `equirectangular`, `dualFisheye` or `none`; equirectangular `layout` and `capturedView`; a dual fisheye's `halfFov` |
//! | `media.align360` | lines the two lenses of dual-fisheye media up along their seam (a few frames across the clip, [`filmcraft_render::vr::align_lenses`]) and keeps the result in its VR Properties |
//!
//! A clip of 360° media shows its raw picture (an equirectangular panorama, or the two fisheye
//! circles of an Insta360 recording) until it is reframed; reframed, it is a flat clip of the
//! virtual camera's frame for everything else (Motion, effects, tracking, Auto Reframe, export).

use serde_json::{Value, json};

use filmcraft_media::Spherical;
use filmcraft_media::spherical::{Projection, StereoLayout};
use filmcraft_project::vr::{MAX_SIDE, MIN_SIDE, REFRAME};
use filmcraft_project::{ClipId, EffectInstance, ItemId, ItemKind, Param, ParamValue, TrackItem, TrackKind};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, bool_p, checked_u32_p, clips_p, f64_p, has_seq, item_p, str_p, u64_p};
use crate::{EngineError, Result, Session};

const REFRAME_CMD: &str = "clip.reframe360";
const PROPS_CMD: &str = "media.vrProperties";
const ALIGN_CMD: &str = "media.align360";
/// Most keyframes one `keys` list writes.
const MAX_KEYS: usize = 10_000;
/// Most frames `media.align360` looks at.
const MAX_ALIGN_FRAMES: u64 = 9;

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
            params: r#"{"items":[id]?,"projection":"auto"|"equirectangular"|"dualFisheye"|"none"="auto","layout":"mono"|"overUnder"|"sideBySide"?,"capturedView":[h,v]?,"halfFov":30..135?}"#,
            enabled: |_| Ok(()),
            run: vr_properties,
            journal: true,
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
    ]
}

/// Whether an item's media is 360° (VR Properties set).
pub fn is_360(s: &Session, item: ItemId) -> bool {
    s.project.vr_of(item).is_some()
}

/// A clip's virtual camera, added (framing `frame`, Motion back to its defaults so the camera's
/// frame fills the sequence's) when it has none; enabled.
pub fn ensure_reframe(it: &mut TrackItem, frame: (u32, u32)) -> Option<&mut EffectInstance> {
    if !it.effects.iter().any(|e| e.effect == REFRAME) {
        let def = filmcraft_project::find_effect(REFRAME)?;
        let mut e = def.instance();
        let side = |v: u32| ParamValue::Float(f64::from(v.clamp(MIN_SIDE, MAX_SIDE)));
        e.params.insert("width".into(), Param::new(side(frame.0)));
        e.params.insert("height".into(), Param::new(side(frame.1)));
        // the camera's picture is the clip's: Motion places it like any picture of that size
        if let (Some(m), Some(mdef)) = (it.effect_mut("motion"), filmcraft_project::find_effect("motion")) {
            for id in ["position", "anchor", "scale", "scale_width", "rotation"] {
                if let (Some(p), Some(d)) = (m.param_mut(id), mdef.param(id)) {
                    p.keyframes.clear();
                    p.value = d.default.clone();
                }
            }
        }
        it.scale_to_frame = false;
        // first in the chain: the other effects work on what the camera sees
        it.effects.insert(it.effects.iter().position(|e| e.def().is_some_and(|d| !d.intrinsic)).unwrap_or(it.effects.len()), e);
    }
    let e = it.effects.iter_mut().find(|e| e.effect == REFRAME)?;
    e.enabled = true;
    Some(e)
}

/// A preset's values: (yaw, pitch, fov, distance, sphere).
fn preset(name: &str) -> Option<(Option<f64>, Option<f64>, f64, f64, bool)> {
    Some(match name {
        "natural" => (None, None, 90.0, 0.0, false),
        "wide" => (None, None, 120.0, 0.3, false),
        "tinyPlanet" | "tiny_planet" | "planet" => (None, Some(-90.0), 120.0, 1.0, false),
        "rabbitHole" | "rabbit_hole" | "tunnel" => (None, Some(90.0), 120.0, 1.0, false),
        "sphere" | "equirectangular" => (Some(0.0), Some(0.0), 90.0, 0.0, true),
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

fn set_static(e: &mut EffectInstance, id: &str, v: f64) {
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

fn set_key(e: &mut EffectInstance, id: &str, t: Tick, v: f64) {
    if e.param(id).is_none() {
        e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
    }
    if let Some(p) = e.param_mut(id) {
        p.put_keyframe(t, ParamValue::Float(v));
    }
}

/// The media time of clip time `seconds` (from the clip's start on the timeline).
fn media_time(it: &TrackItem, seconds: f64) -> Tick {
    let st = it.start + Tick::from_seconds_f64(seconds.clamp(0.0, it.duration.seconds().max(0.0)));
    it.effect_time_at(st)
}

fn view_json(c: ClipId, it: &TrackItem) -> Value {
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
                set_static(e, "fov", fov);
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
    let projection = str_p(p, "projection").unwrap_or("auto");
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
    // what each item gets (decided before the edit: `auto` opens the file)
    let mut new: Vec<(ItemId, Option<Spherical>)> = Vec::new();
    for id in &items {
        let from_file = || s.media.full_res_source(&s.project, *id, &*s.services).and_then(|src| src.spherical());
        let mut v = match projection {
            "auto" => from_file(),
            "none" => None,
            "equirectangular" => Some(Spherical::equirect()),
            "dualFisheye" => Some(match from_file().filter(|f| !f.is_equirect()).or_else(|| s.project.vr_of(*id).filter(|v| !v.is_equirect()).cloned()) {
                Some(f) => f,
                None => default_dual(),
            }),
            o => return Err(bad(cmd, format!("`projection` is auto, equirectangular, dualFisheye or none, not `{o}`"))),
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
        }
        new.push((*id, v.map(|v| v.sanitized())));
    }
    let report = s.edit("VR Properties", |pr, _| {
        let mut out = Vec::new();
        for (id, v) in &new {
            if let Some(ItemKind::Media(m)) = pr.item_mut(*id).map(|i| &mut i.kind) {
                m.interpret.vr = v.clone();
                out.push(json!({"item": id.0, "vr": v}));
            }
        }
        Ok(out)
    })?;
    Ok(json!({"items": report}))
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
    }
}
