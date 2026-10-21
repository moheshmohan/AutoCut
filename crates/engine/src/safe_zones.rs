//! Platform safe zones of vertical video (roadmap E3, the vertical factory): the parts of a 9:16
//! frame YouTube Shorts, Instagram Reels and TikTok cover with their own interface
//! ([`filmcraft_render::safe_zones`], where the numbers and their sources are).
//!
//! | command | does |
//! |---|---|
//! | `view.safeZones` | `platform` `"youtube"`, `"reels"`, `"tiktok"`, `"vertical"` (all at once) or `null` / `"off"`: the Program monitor (or `monitor: "source"`) draws that platform's zones as translucent rectangles with its name; without `platform` it says which is on. Returns the zones in the active sequence's pixels |
//! | `sequence.checkSafeZones` | which captions (on visible caption tracks) and graphics (titles, lower thirds: every visible layer of a graphic clip, where its Motion puts it in its middle frame) of `sequence` (default the active one) reach into `platform`'s (default `"youtube"`) zones: each with its box, the zones and how much of it is in them |
//!
//! The state is [`crate::EditorState::safe_zones`] (the Program monitor's; the desktop app keeps
//! each monitor's in its view state, `ui.set {"program": {"safe_zones": "tiktok"}}`).

use filmcraft_project::{ItemKind, Project, Sequence};
use filmcraft_time::Tick;
use serde_json::{Value, json};

pub use filmcraft_render::safe_zones::{Area, MIN_OVERLAP, Margins, Overlap, Platform, Zone};

use crate::commands::{CommandSpec, always, bad, has_seq};
use crate::{EngineError, Result, Session};

/// Most captions and graphic clips one check looks at.
const MAX_CHECKED: usize = 20_000;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "view.safeZones",
            label: "Platform Safe Zones",
            menu: &[],
            shortcut: None,
            params: r#"{"platform":"youtube"|"reels"|"tiktok"|"vertical"|null|"off"? (absent: say which is on),"monitor":"program"|"source"="program"}"#,
            enabled: always,
            run: view_safe_zones,
            journal: false,
        },
        CommandSpec {
            id: "sequence.checkSafeZones",
            label: "Check Platform Safe Zones",
            menu: &[],
            shortcut: None,
            params: r#"{"platform":"youtube"|"reels"|"tiktok"|"vertical"="youtube","sequence":id?}"#,
            enabled: has_seq,
            run: check_safe_zones,
            journal: false,
        },
    ]
}

/// `key` as a platform: absent → `Ok(None)`; `null`, `"off"`, `"none"` or `false` →
/// `Ok(Some(None))` (turned off); a platform name → `Ok(Some(Some(p)))`.
pub fn platform_p(cmd: &str, p: &Value, key: &str) -> Result<Option<Option<Platform>>> {
    let names = "\"youtube\", \"reels\", \"tiktok\", \"vertical\" or null";
    match p.get(key) {
        None => Ok(None),
        Some(Value::Null) | Some(Value::Bool(false)) => Ok(Some(None)),
        Some(Value::String(s)) if matches!(s.trim().to_ascii_lowercase().as_str(), "off" | "none" | "") => Ok(Some(None)),
        Some(Value::String(s)) if s.chars().count() <= 64 => {
            Platform::from_name(s).map(|p| Some(Some(p))).ok_or_else(|| bad(cmd, format!("`{key}` is {names}, not \"{s}\"")))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` is {names}"))),
    }
}

/// A platform's zones and safe area in the pixels of a `w`×`h` frame.
pub fn zones_json(platform: Option<Platform>, (w, h): (u32, u32)) -> Value {
    let Some(p) = platform else { return json!({"platform": null}) };
    let frame = (f64::from(w.max(1)), f64::from(h.max(1)));
    let px = |a: &Area| a.in_pixels(frame).map(|v| v.round());
    json!({
        "platform": p.name(),
        "label": p.label(),
        "frame": [w, h],
        "safe": px(&p.safe_area()),
        "zones": p.zones().iter().map(|z| json!({"name": z.name, "what": z.what, "rect": px(&z.area)})).collect::<Vec<_>>(),
    })
}

fn view_safe_zones(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "view.safeZones";
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let source = match p.get("monitor").filter(|v| !v.is_null()) {
        None => false,
        Some(Value::String(m)) if m == "program" => false,
        Some(Value::String(m)) if m == "source" => true,
        Some(_) => return Err(bad(cmd, "`monitor` is \"program\" or \"source\"")),
    };
    let now = match platform_p(cmd, p, "platform")? {
        Some(next) if !source => {
            s.state.safe_zones = next;
            next
        }
        // the Source monitor's lives in the frontend's view state
        Some(next) => next,
        None => s.state.safe_zones,
    };
    let frame = s.active_sequence().map_or((1080, 1920), |q| (q.settings.width, q.settings.height));
    let mut v = zones_json(now, frame);
    v["monitor"] = json!(if source { "source" } else { "program" });
    Ok(v)
}

/// Something drawn over the picture that a platform's interface could cover.
#[derive(Clone, Debug, PartialEq)]
pub struct Overlay {
    /// `caption` or `graphic`.
    pub kind: &'static str,
    pub clip: u64,
    /// `C1`… (caption tracks, top first as in the timeline) or `V1`….
    pub track: String,
    /// The caption's text, the graphic clip's name.
    pub text: String,
    pub start: Tick,
    pub end: Tick,
    /// Its box, sequence pixels: `[x0, y0, x1, y1]`.
    pub rect: [f64; 4],
}

/// The captions of visible caption tracks (as burned in) and the graphic clips on visible video
/// tracks (their visible layers where Motion puts them in their middle frame) of `q`.
pub fn overlays(project: &Project, q: &Sequence) -> Vec<Overlay> {
    let (w, h) = (q.settings.width as usize, q.settings.height as usize);
    let mut out = Vec::new();
    for (ti, t) in q.caption_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        // a style from a file is hostile: sizes as the style panel allows them
        let mut style = t.style.clone();
        let sane = |v: f32, lo: f32, hi: f32, or: f32| if v.is_finite() { v.clamp(lo, hi) } else { or };
        style.size = sane(style.size, 1.0, 1000.0, 54.0);
        style.outline = sane(style.outline, 0.0, 100.0, 0.0);
        style.line_spacing = sane(style.line_spacing, 0.5, 5.0, 1.25);
        style.margin = sane(style.margin, 0.0, 0.45, 0.08);
        // without a background box, the side padding the box would have is empty
        let pad = if style.background { 0.0 } else { (f64::from(style.size) * w.min(h) as f64 / 1080.0 * 0.3).round() };
        for c in t.captions.iter().take(MAX_CHECKED) {
            if out.len() >= MAX_CHECKED {
                return out;
            }
            let Some(o) = filmcraft_captions::burn::render_caption(c, &style, w, h) else { continue };
            let (x0, x1) = (o.x as f64, o.x as f64 + o.w as f64);
            let (x0, x1) = if x1 - x0 > 2.0 * pad { (x0 + pad, x1 - pad) } else { (x0, x1) };
            let rect = [x0, o.y as f64, x1, o.y as f64 + o.h as f64];
            let text = c.plain_lines().join(" ");
            out.push(Overlay { kind: "caption", clip: c.id.0, track: format!("C{}", ti + 1), text, start: c.start, end: c.start + c.duration, rect });
        }
    }
    for (ti, t) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        for it in t.items.iter().filter(|i| i.enabled) {
            if out.len() >= MAX_CHECKED {
                return out;
            }
            let graphic = project.item(it.item).is_some_and(|p| matches!(p.kind, ItemKind::Graphic { .. }));
            if !graphic {
                continue;
            }
            let Some(src) = filmcraft_render::source_size(project, it.item) else { continue };
            let mid = it.start + Tick(it.duration.0 / 2);
            let mt = it.effect_time_at(mid);
            let mut b: Option<[f64; 4]> = None;
            for (_, spec) in filmcraft_render::graphic_clip::item_layer_specs(it, mt, src).iter().filter(|(_, s)| s.enabled) {
                let lb = filmcraft_render::graphic_clip::layer_bounds(spec);
                if !lb.iter().all(|v| v.is_finite()) || lb[2] <= lb[0] || lb[3] <= lb[1] {
                    continue;
                }
                b = Some(match b {
                    Some(a) => [a[0].min(lb[0]), a[1].min(lb[1]), a[2].max(lb[2]), a[3].max(lb[3])],
                    None => lb,
                });
            }
            let Some(b) = b else { continue };
            let m = filmcraft_render::motion_matrix(q, it, src, None, mt);
            let corners = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])].map(|(x, y)| m.apply(filmcraft_geom::Vec2::new(x, y)));
            let rect = [
                corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
                corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
                corners.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
                corners.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
            ];
            if rect.iter().all(|v| v.is_finite()) {
                out.push(Overlay {
                    kind: "graphic",
                    clip: it.id.0,
                    track: format!("V{}", ti + 1),
                    text: it.name.clone(),
                    start: it.start,
                    end: it.end(),
                    rect,
                });
            }
        }
    }
    out
}

/// Which of `q`'s overlays reach into `platform`'s zones (see the module docs).
pub fn check(project: &Project, q: &Sequence, platform: Platform) -> Value {
    let frame = (f64::from(q.settings.width.max(1)), f64::from(q.settings.height.max(1)));
    let all = overlays(project, q);
    let secs = |t: Tick| (t.seconds() * 1000.0).round() / 1000.0;
    let hits: Vec<Value> = all
        .iter()
        .filter_map(|o| {
            let found = platform.overlaps(&Area::from_pixels(o.rect, frame));
            (!found.is_empty()).then(|| {
                json!({
                    "kind": o.kind,
                    "clip": o.clip,
                    "track": o.track,
                    "text": o.text.chars().take(200).collect::<String>(),
                    "seconds": secs(o.start),
                    "until": secs(o.end),
                    "rect": o.rect.map(f64::round),
                    "zones": found.iter().map(|f| f.zone).collect::<Vec<_>>(),
                    "share": (found.iter().map(|f| f.share).fold(0.0, f64::max).min(1.0) * 1000.0).round() / 1000.0,
                })
            })
        })
        .collect();
    let mut v = zones_json(Some(platform), (q.settings.width, q.settings.height));
    v["vertical"] = json!(q.settings.height > q.settings.width);
    v["checked"] = json!(all.len());
    v["ok"] = json!(hits.is_empty());
    v["overlaps"] = Value::Array(hits);
    v
}

fn check_safe_zones(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "sequence.checkSafeZones";
    if !p.is_object() && !p.is_null() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let platform = match platform_p(cmd, p, "platform")? {
        None => Platform::Youtube,
        Some(Some(pl)) => pl,
        Some(None) => return Err(bad(cmd, "`platform` names the platform to check against")),
    };
    let q = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = filmcraft_project::ItemId(v.as_u64().ok_or_else(|| bad(cmd, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(cmd, "no such sequence"))?
        }
        None => s.active_sequence().ok_or(EngineError::NoSequence)?,
    };
    Ok(check(&s.project, q, platform))
}

#[cfg(test)]
#[path = "safe_zones_tests.rs"]
mod tests;
