//! Clip ▸ Video Options ▸ Ken Burns — the slow pan and zoom over a still photo, aimed at what it
//! shows.
//!
//! | command | does |
//! |---|---|
//! | `clip.kenBurns` | keyframe Motion ▸ Position and Scale of video clips (`clips`, default the selected ones) over each clip's length, eased out of the first keyframe and into the last: from wide in to the `subject` (`move: "in"`, the default), from the subject out (`"out"`), across it at a steady zoom (`"pan"`; `reverse` goes the other way), or held framed on it (`"none"`, no keyframes); one undo step |
//!
//! **The move.** `fit: "fill"` (the default) scales the picture to cover the frame, as Fill Frame
//! does: the wide end shows it as filled, centred halfway between the picture's middle and the
//! subject (so a portrait photo filling a landscape frame starts with its subject in view); the
//! close end is centred on the subject and as close as the box filling [`SUBJECT_SHARE`] of the
//! frame asks, but never closer than `zoom` (by default [`ZOOM_PER_SECOND`] a second of the clip,
//! between [`MIN_ZOOM`] and [`MAX_ZOOM`]) — so a long hold moves further, and every move stays
//! slow. Each centre is kept where the picture still covers the frame, so no edge shows.
//! `fit: "full"` keeps the whole picture in the frame (for a portrait photo over blurred pillars):
//! it only breathes, at most [`MAX_FULL_ZOOM`] closer, towards the subject. `"pan"` travels
//! [`PAN_PER_SECOND`] of the picture a second (as far as it has room) along the axis with the most
//! room, past the subject. Without a `subject` the picture's middle is the subject.
//!
//! `subject` is the box `[x, y, w, h]` (or `{x, y, w, h}`), 0..1 of the picture, x right and y
//! down — what a vision model found in it. The command keeps the clip's anchor in the picture's
//! middle and turns Scale to Frame Size off, like Fit/Fill Frame.

use serde_json::{Value, json};

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, Interpolation, Keyframe, Param, ParamValue};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, bool_p, clips_p, f64_p, has_selection, str_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.kenBurns";

/// How much closer the close end is a second of the clip (2.5 %), and the least and most.
pub const ZOOM_PER_SECOND: f64 = 0.025;
pub const MIN_ZOOM: f64 = 0.04;
pub const MAX_ZOOM: f64 = 0.25;
/// The most `zoom` a command may ask for.
const MAX_ASKED_ZOOM: f64 = 1.0;
/// A whole picture (`fit: "full"`) comes at most this much closer.
pub const MAX_FULL_ZOOM: f64 = 0.05;
/// The share of the frame the subject's box fills at the close end (on its tighter side).
pub const SUBJECT_SHARE: f64 = 0.75;
/// How far a pan travels a second, as a share of the picture.
pub const PAN_PER_SECOND: f64 = 0.02;

/// Which way the picture moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// From wide in to the subject.
    In,
    /// From the subject out.
    Out,
    /// Across the subject at a steady zoom.
    Pan,
    /// Held on the subject.
    None,
}

impl Move {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "in" => Some(Self::In),
            "out" => Some(Self::Out),
            "pan" => Some(Self::Pan),
            "none" | "still" => Some(Self::None),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Pan => "pan",
            Self::None => "none",
        }
    }
}

/// Where the picture is at one end of the move: Motion ▸ Position (sequence pixels) and Scale
/// (percent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: [f64; 2],
    pub scale: f64,
}

/// A move asked for: the subject's box (0..1 of the picture), the move, whether the picture
/// fills the frame (else it is kept whole), how much closer the close end is (`None`: from the
/// clip's length), and a pan's direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ask {
    pub subject: Option<[f64; 4]>,
    pub motion: Move,
    pub fill: bool,
    pub zoom: Option<f64>,
    pub reverse: bool,
}

/// The default `zoom` for a clip `seconds` long.
pub fn zoom_for(seconds: f64) -> f64 {
    if seconds.is_finite() { (ZOOM_PER_SECOND * seconds).clamp(MIN_ZOOM, MAX_ZOOM) } else { MIN_ZOOM }
}

/// The move's two ends for a picture `picture` big (sequence pixels at 100 %) in a `frame`,
/// over a clip `seconds` long: `(first, last)`. `None` for sizes that make no sense.
pub fn path(picture: (f64, f64), frame: (f64, f64), seconds: f64, ask: Ask) -> Option<(Pose, Pose)> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    let ((dw, dh), (fw, fh)) = (picture, frame);
    if !(ok(dw) && ok(dh) && ok(fw) && ok(fh)) {
        return None;
    }
    let (sx, sy) = (fw / dw, fh / dh);
    // the scale (a fraction) at which the picture covers the frame, or fits in it whole
    let base = if ask.fill { sx.max(sy) } else { sx.min(sy) };
    let amount = ask.zoom.filter(|z| z.is_finite() && *z >= 0.0).map_or_else(|| zoom_for(seconds), |z| z.min(MAX_ASKED_ZOOM));
    let amount = if ask.fill { amount } else { amount.min(MAX_FULL_ZOOM) };
    let [bx, by, bw, bh] = ask.subject.unwrap_or([0.25, 0.25, 0.5, 0.5]);
    let centre = [bx + bw / 2.0, by + bh / 2.0];
    // the picture's point `f` (0..1) at the frame's middle at scale `s`, kept where the picture
    // still covers the frame (on an axis it does not cover, its middle)
    let held = |f: [f64; 2], s: f64| {
        let vis = [fw / (dw * s), fh / (dh * s)];
        let mut c = [0.5; 2];
        for k in 0..2 {
            if vis[k] < 1.0 {
                c[k] = f[k].clamp(vis[k] / 2.0, 1.0 - vis[k] / 2.0);
            }
        }
        c
    };
    let pose = |f: [f64; 2], s: f64| {
        let c = held(f, s);
        Pose { position: [fw / 2.0 + (0.5 - c[0]) * dw * s, fh / 2.0 + (0.5 - c[1]) * dh * s], scale: s * 100.0 }
    };
    // close: the box filling its share of the frame, within the zoom asked for (a whole picture
    // only breathes)
    let fits = (SUBJECT_SHARE * fw / (bw.max(1e-3) * dw * base)).min(SUBJECT_SHARE * fh / (bh.max(1e-3) * dh * base));
    let z = if ask.fill { fits.min(1.0 + amount).max(1.0 + amount.min(MIN_ZOOM)) } else { 1.0 + amount };
    let close = pose(centre, base * z);
    let wide = if ask.fill { pose([(0.5 + centre[0]) / 2.0, (0.5 + centre[1]) / 2.0], base) } else { pose([0.5, 0.5], base) };
    Some(match ask.motion {
        Move::In => (wide, close),
        Move::Out => (close, wide),
        Move::None => (close, close),
        Move::Pan => {
            // at a steady zoom between the two, along the axis with the most room
            let s = base * (1.0 + amount.max(MIN_ZOOM) * 0.6);
            let vis = [fw / (dw * s), fh / (dh * s)];
            let room = [(1.0 - vis[0]).max(0.0), (1.0 - vis[1]).max(0.0)];
            let axis = usize::from(room[1] > room[0]);
            let d = (PAN_PER_SECOND * seconds.max(0.0) / 2.0).min(room[axis] / 2.0);
            if !(d.is_finite() && d > 1e-3) {
                return path(picture, frame, seconds, Ask { motion: Move::In, ..ask });
            }
            let lo = vis[axis] / 2.0 + d;
            let hi = 1.0 - vis[axis] / 2.0 - d;
            let c = if lo <= hi { centre[axis].clamp(lo, hi) } else { 0.5 };
            let (mut a, mut b) = (centre, centre);
            a[axis] = c - d;
            b[axis] = c + d;
            let other = 1 - axis;
            let o = held(centre, s)[other];
            a[other] = o;
            b[other] = o;
            if ask.reverse { (pose(b, s), pose(a, s)) } else { (pose(a, s), pose(b, s)) }
        }
    })
}

/// The share of the picture the frame shows at `pose`, `[x, y, w, h]` (0..1 of the picture): what
/// is on screen, for checking where a move ends.
pub fn shown(picture: (f64, f64), frame: (f64, f64), pose: Pose) -> [f64; 4] {
    let ((dw, dh), (fw, fh)) = (picture, frame);
    let s = pose.scale / 100.0;
    let (pw, ph) = (dw * s, dh * s);
    // the picture's top-left corner in the frame
    let (x0, y0) = (pose.position[0] - pw / 2.0, pose.position[1] - ph / 2.0);
    let (l, t) = ((-x0 / pw).max(0.0), (-y0 / ph).max(0.0));
    let (r, b) = (((fw - x0) / pw).min(1.0), ((fh - y0) / ph).min(1.0));
    [l, t, r - l, b - t]
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Ken Burns",
        menu: &["Clip", "Video Options"],
        shortcut: None,
        params: r#"{"clips":[id]?,"subject":[f64;4]?,"move":"in|out|pan|none"="in","fit":"fill|full"="fill","zoom":f64?,"reverse":bool=false}"#,
        enabled: has_selection,
        run,
        journal: true,
    }]
}

/// `subject`: `[x, y, w, h]` or `{x, y, w, h}`, each 0..1, a box inside the picture.
fn subject_p(p: &Value) -> Result<Option<[f64; 4]>> {
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
    let Some([x, y, w, h]) = nums.as_deref().and_then(|n| <[f64; 4]>::try_from(n).ok()) else { return Err(bad(CMD, why)) };
    let inside = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
    if !(inside(x) && inside(y) && inside(w) && inside(h)) || w <= 0.0 || h <= 0.0 || x + w > 1.0 + 1e-6 || y + h > 1.0 + 1e-6 {
        return Err(bad(CMD, why));
    }
    Ok(Some([x, y, w.min(1.0 - x), h.min(1.0 - y)]))
}

fn keyframe(t: Tick, v: ParamValue, interp: Interpolation) -> Keyframe {
    let mut k = Keyframe::new(t, v);
    k.interp = interp;
    k
}

/// Set `param` of `m` to `a` at `t0` and `b` at `t1` (eased), or to `a` alone when they are the
/// same or the clip has no length.
fn animate(m: &mut filmcraft_project::EffectInstance, param: &str, (t0, t1): (Tick, Tick), a: ParamValue, b: ParamValue) {
    let mut prm = Param::new(a.clone());
    if t1 > t0 && a != b {
        prm.keyframes = vec![keyframe(t0, a, Interpolation::EaseOut), keyframe(t1, b, Interpolation::EaseIn)];
    }
    m.params.insert(param.to_string(), prm);
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let motion = match str_p(p, "move") {
        None => Move::In,
        Some(m) => Move::parse(m).ok_or_else(|| bad(CMD, "`move` is in, out, pan or none"))?,
    };
    let fill = match str_p(p, "fit") {
        None => true,
        Some(f) if f.trim().eq_ignore_ascii_case("fill") => true,
        Some(f) if f.trim().eq_ignore_ascii_case("full") => false,
        Some(_) => return Err(bad(CMD, "`fit` is fill (the frame covered) or full (the whole picture)")),
    };
    let zoom = match f64_p(p, "zoom") {
        None => None,
        Some(z) if z.is_finite() && (0.0..=MAX_ASKED_ZOOM).contains(&z) => Some(z),
        Some(_) => return Err(bad(CMD, format!("`zoom` is how much closer the move comes, 0 to {MAX_ASKED_ZOOM} (0.1: 10 %)"))),
    };
    if p.get("move").is_some_and(|v| !v.is_string()) || p.get("fit").is_some_and(|v| !v.is_string()) {
        return Err(bad(CMD, "`move` and `fit` are words"));
    }
    let ask = Ask { subject: subject_p(p)?, motion, fill, zoom, reverse: bool_p(p, "reverse").unwrap_or(false) };
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (fw, fh) = (f64::from(q.settings.width), f64::from(q.settings.height));
    let frame_par = q.settings.par;
    let fd = q.settings.frame_rate.sane().frame_duration().max(Tick(1));
    let clips: Vec<ClipId> = clips_p(s, p).into_iter().filter(|c| q.video_tracks.iter().any(|t| t.item(*c).is_some())).collect();
    // (clip, storage size for the anchor, display size, media times of its first and last frame, its length)
    let mut work = Vec::new();
    for c in &clips {
        let Some((_, it)) = q.find_item(*c) else { continue };
        let (Some(storage), Some(display)) = (filmcraft_render::clip_size(&s.project, it), s.project.conformed_clip_size(it, frame_par)) else {
            continue;
        };
        let last = it.start + (it.duration - fd).max(Tick::ZERO);
        work.push((*c, storage, display, (it.effect_time_at(it.start), it.effect_time_at(last)), it.duration.seconds()));
    }
    if work.is_empty() {
        return Err(bad(CMD, "select a video clip (a photo, usually)"));
    }
    let out = s.edit_sequence("Ken Burns", |q, _, _| {
        let mut out = Vec::new();
        for (c, (w, h), display, times, seconds) in &work {
            let Some((a, b)) = path(*display, (fw, fh), *seconds, ask) else { continue };
            let Some((_, it)) = q.find_item_mut(*c) else { continue };
            it.scale_to_frame = false;
            let Some(m) = it.effect_mut("motion") else { continue };
            animate(m, "scale", *times, ParamValue::Float(a.scale), ParamValue::Float(b.scale));
            animate(
                m,
                "position",
                *times,
                ParamValue::Vec2(Vec2::new(a.position[0], a.position[1])),
                ParamValue::Vec2(Vec2::new(b.position[0], b.position[1])),
            );
            m.params.insert("uniform_scale".into(), Param::new(ParamValue::Bool(true)));
            m.params.insert("anchor".into(), Param::new(ParamValue::Vec2(Vec2::new(f64::from(*w) / 2.0, f64::from(*h) / 2.0))));
            let pose = |x: Pose| json!({"position": x.position, "scale": x.scale});
            out.push(json!({"clip": c.0, "move": motion.id(), "from": pose(a), "to": pose(b), "shownAtEnd": shown(*display, (fw, fh), b)}));
        }
        Ok(out)
    })?;
    if out.is_empty() {
        return Err(bad(CMD, "nothing to move: the clips have no picture size"));
    }
    Ok(json!({"clips": out}))
}

#[cfg(test)]
#[path = "ken_burns_tests.rs"]
mod tests;
