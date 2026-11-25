//! `graphics.chapterBar` (PX7.2): a bar across an explainer that shows its chapters and which
//! one is playing, built from the chapter starts (the sequence's chapter markers, as
//! `script.timeline` places them, or the chapters given). The idea of a chapter bar on a
//! narrated explainer comes from anything2explainer (PolyForm Noncommercial; idea only).
//!
//! One graphic clip a chapter, from its start to the next chapter's (the last to the end of the
//! sequence), all on one video track (a new top track by default). Each clip draws the whole bar:
//! one rounded segment a chapter, as wide as the chapter is long (gaps between them), the chapters
//! before it filled, its own in the accent colour, the ones after it dim; and, with `labels`, the
//! chapter's title beside the bar on a dark plate. With `progress` the current segment fills as
//! its chapter plays (keyframes on a fill layer); a fill that moves all the time counts as change
//! for `sequence.checkCraft`'s still checks, so it is off by default. Every layer is an ordinary
//! shape or text layer, editable in the Graphics panel. One undo step.
//!
//! Where: `position` `bottom` (the default for a landscape frame) or `top` (the default for a
//! portrait one, clear of the platforms' captions), 5 % of the frame in from the edges; the bar
//! 1.2 % of the short side high (at least 6 px), the label 3.2 % of it (above
//! `sequence.checkCraft`'s small-text floor). At most 64 chapters.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use filmcraft_geom::Vec2;
use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
use filmcraft_project::{EffectInstance, Keyframe, MarkerKind, Param, ParamValue, Track, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::motion_energy::r3;
use crate::{EngineError, Result, Session};

pub const CMD: &str = "graphics.chapterBar";
/// Most chapters on a bar.
pub const MAX_CHAPTERS: usize = 64;
/// The bar's height and the label's size, shares of the frame's short side; the margin, a share
/// of each side.
pub const BAR_SHARE: f64 = 0.012;
pub const LABEL_SHARE: f64 = 0.032;
pub const MARGIN_SHARE: f64 = 0.05;
/// The gap between segments, a share of the width.
const GAP_SHARE: f64 = 0.005;
/// The default colours (sRGB, straight alpha).
const TRACK: [f32; 4] = [1.0, 1.0, 1.0, 0.3];
const DONE: [f32; 4] = [1.0, 1.0, 1.0, 0.85];
const CURRENT: [f32; 4] = [1.0, 0.69, 0.0, 1.0];
const TEXT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const KEYS: &[&str] = &["chapters", "end", "position", "height", "labels", "progress", "colors", "track", "dryRun"];

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Chapter Bar",
        menu: &[],
        shortcut: None,
        params: r##"{"chapters":[{"start":ticks|"seconds":f64,"title":str?}]? (default the active sequence's chapter markers, as script.timeline places them),"end":ticks? (default the sequence's end),"position":"bottom"|"top"? (default bottom, top for a portrait frame),"height":px? (default 1.2 % of the short side),"labels":bool=true (the chapter's title beside the bar),"progress":bool=false (the current segment fills as it plays),"colors":{"track":"#ffffff4d","done":"#ffffffd9","current":"#ffb000","text":"#ffffff"}?,"track":"V<n>"? (default a new top track; it must be free),"dryRun":bool=false}"##,
        enabled: has_seq,
        run,
        journal: true,
    }]
}

/// A chapter on the bar.
#[derive(Clone, Debug, PartialEq)]
struct Chapter {
    start: Tick,
    end: Tick,
    title: String,
}

/// `#rrggbb` or `#rrggbbaa` as straight sRGB 0..1.
fn color_p(v: &Value, key: &str) -> Result<[f32; 4]> {
    let s = v.as_str().ok_or_else(|| bad(CMD, format!("`colors.{key}` is \"#rrggbb\" or \"#rrggbbaa\"")))?;
    let h = s.trim().strip_prefix('#').unwrap_or(s.trim());
    if !(h.len() == 6 || h.len() == 8) || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad(CMD, format!("`colors.{key}` is \"#rrggbb\" or \"#rrggbbaa\", not `{}`", s.chars().take(20).collect::<String>())));
    }
    let byte = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map_or(255.0, f32::from) / 255.0;
    Ok([byte(0), byte(2), byte(4), if h.len() == 8 { byte(6) } else { 1.0 }])
}

fn hex(c: [f32; 4]) -> String {
    let b = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]), b(c[3]))
}

/// The chapters: given, or the active sequence's chapter markers, ended by the next one (the last
/// by `end`).
fn chapters_p(s: &Session, p: &Value, end: Tick) -> Result<Vec<Chapter>> {
    let mut starts: Vec<(Tick, String)> = match p.get("chapters").filter(|v| !v.is_null()) {
        Some(v) => {
            let a = v.as_array().ok_or_else(|| bad(CMD, "`chapters` is a list of {start, title}"))?;
            if a.len() > MAX_CHAPTERS {
                return Err(bad(CMD, format!("{} chapters; a bar shows at most {MAX_CHAPTERS}", a.len())));
            }
            a.iter()
                .enumerate()
                .map(|(i, c)| {
                    let o = c.as_object().ok_or_else(|| bad(CMD, format!("chapter {}: {{start or seconds, title?}}", i + 1)))?;
                    let start = match (o.get("start").filter(|v| !v.is_null()), o.get("seconds").filter(|v| !v.is_null())) {
                        (Some(t), None) => {
                            Tick(t.as_i64().filter(|t| *t >= 0).ok_or_else(|| bad(CMD, format!("chapter {}: `start` is ticks, 0 or more", i + 1)))?)
                        }
                        (None, Some(x)) => Tick::from_seconds_f64(
                            x.as_f64()
                                .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 86_400.0)
                                .ok_or_else(|| bad(CMD, format!("chapter {}: `seconds` is 0 to 86400", i + 1)))?,
                        ),
                        _ => return Err(bad(CMD, format!("chapter {}: give `start` (ticks) or `seconds`", i + 1))),
                    };
                    let title = match o.get("title") {
                        None | Some(Value::Null) => String::new(),
                        Some(Value::String(t)) if t.chars().count() <= 200 => t.trim().to_string(),
                        Some(_) => return Err(bad(CMD, format!("chapter {}: `title` is a text of at most 200 characters", i + 1))),
                    };
                    Ok((start, title))
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => {
            let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
            let mut m: Vec<(Tick, String)> = q.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).map(|m| (m.start, m.name.clone())).collect();
            m.sort_by_key(|x| x.0);
            if m.len() > MAX_CHAPTERS {
                return Err(bad(CMD, format!("the sequence has {} chapter markers; a bar shows at most {MAX_CHAPTERS}", m.len())));
            }
            m
        }
    };
    if starts.is_empty() {
        return Err(bad(CMD, "no chapters: the sequence has no chapter markers (script.timeline places them), and `chapters` was not given"));
    }
    if starts.windows(2).any(|w| matches!(w, [a, b] if b.0 <= a.0)) {
        return Err(bad(CMD, "the chapters must start in order, each after the one before"));
    }
    if starts.last().is_some_and(|(t, _)| *t >= end) {
        return Err(bad(CMD, "the last chapter starts at or after the end of the bar"));
    }
    let ends: Vec<Tick> = starts.iter().skip(1).map(|x| x.0).chain(std::iter::once(end)).collect();
    Ok(starts.drain(..).zip(ends).map(|((start, title), end)| Chapter { start, end, title }).collect())
}

fn shape(name: &str, centre: Vec2, size: Vec2, color: [f32; 4], radius: f64) -> EffectInstance {
    let mut e = new_shape_layer(0, centre, size, Vec::new());
    e.params.insert("fill_color".into(), Param::new(ParamValue::Color(color)));
    e.params.insert("corner_radius".into(), Param::new(ParamValue::Float(radius.max(0.0))));
    e.params.insert("name".into(), Param::new(ParamValue::Text(name.into())));
    e
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !KEYS.contains(&k.as_str())).cloned()) {
        return Err(bad(CMD, format!("unknown parameter `{}` ({})", k.chars().take(64).collect::<String>(), KEYS.join(", "))));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (w, h, rate) = (f64::from(q.settings.width.max(1)), f64::from(q.settings.height.max(1)), q.settings.frame_rate.sane());
    let end = match p.get("end").filter(|v| !v.is_null()) {
        Some(v) => Tick(v.as_i64().filter(|t| *t > 0).ok_or_else(|| bad(CMD, "`end` is ticks after 0"))?),
        None => q.duration(),
    };
    if end <= Tick::ZERO {
        return Err(bad(CMD, "the sequence is empty: give `end`, or place the explainer first"));
    }
    let chapters = chapters_p(s, p, end)?;
    let short = w.min(h);
    let flag = |k: &str, d: bool| -> Result<bool> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v.as_bool().ok_or_else(|| bad(CMD, format!("`{k}` is true or false"))),
        }
    };
    let (labels, progress, dry) = (flag("labels", true)?, flag("progress", false)?, flag("dryRun", false)?);
    let top = match p.get("position") {
        None | Some(Value::Null) => h > w,
        Some(Value::String(x)) if x == "top" => true,
        Some(Value::String(x)) if x == "bottom" => false,
        Some(_) => return Err(bad(CMD, "`position` is \"top\" or \"bottom\"")),
    };
    let bar_h = match p.get("height") {
        None | Some(Value::Null) => (BAR_SHARE * short).round().max(6.0),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 1.0 && *x <= 0.25 * h)
            .ok_or_else(|| bad(CMD, format!("`height` is 1 to {:.0} px (a quarter of the frame)", 0.25 * h)))?,
    };
    let mut colors = [TRACK, DONE, CURRENT, TEXT];
    if let Some(c) = p.get("colors").filter(|v| !v.is_null()) {
        let o = c.as_object().ok_or_else(|| bad(CMD, "`colors` is {track?, done?, current?, text?}"))?;
        for (k, v) in o {
            let i = ["track", "done", "current", "text"].iter().position(|x| x == k).ok_or_else(|| {
                bad(CMD, format!("`colors.{}` is not a colour of the bar (track, done, current, text)", k.chars().take(40).collect::<String>()))
            })?;
            if let Some(slot) = colors.get_mut(i) {
                *slot = color_p(v, k)?;
            }
        }
    }
    let [track_c, done_c, current_c, text_c] = colors;
    // ---- the bar's geometry
    let margin_x = (MARGIN_SHARE * w).round();
    let bar_w = (w - 2.0 * margin_x).max(1.0);
    let cy = if top { (MARGIN_SHARE * h).round() + bar_h / 2.0 } else { h - (MARGIN_SHARE * h).round() - bar_h / 2.0 };
    let n = chapters.len();
    let gap = if n > 1 { (GAP_SHARE * w).round().max(2.0).min(bar_w / (2.0 * n as f64)) } else { 0.0 };
    let avail = (bar_w - gap * (n.saturating_sub(1)) as f64).max(1.0);
    let span = chapters.iter().map(|c| (c.end - c.start).0.max(0) as f64).sum::<f64>().max(1.0);
    let mut x = margin_x;
    let mut segments: Vec<(f64, f64)> = Vec::with_capacity(n);
    for c in &chapters {
        let sw = (avail * (c.end - c.start).0.max(0) as f64 / span).max(1.0);
        segments.push((x, x + sw));
        x += sw + gap;
    }
    let label_size = (LABEL_SHARE * short).round().max(8.0);
    let label_y = if top { cy + bar_h / 2.0 + label_size * 1.25 } else { cy - bar_h / 2.0 - label_size * 0.55 };
    let plan: Vec<Value> = chapters
        .iter()
        .zip(&segments)
        .enumerate()
        .map(|(i, (c, (x0, x1)))| {
            json!({"index": i, "title": c.title, "start": c.start.0, "end": c.end.0, "seconds": [r3(c.start.seconds()), r3(c.end.seconds())], "segment": [r3(*x0), r3(*x1)]})
        })
        .collect();
    let bar = json!({"x": margin_x, "y": r3(cy - bar_h / 2.0), "width": bar_w, "height": bar_h, "gap": gap, "position": if top { "top" } else { "bottom" },
        "colors": {"track": hex(track_c), "done": hex(done_c), "current": hex(current_c), "text": hex(text_c)}, "labels": labels, "progress": progress});
    if dry {
        return Ok(json!({"dryRun": true, "chapters": plan, "bar": bar}));
    }
    // ---- the track
    let first = chapters.first().map_or(Tick::ZERO, |c| c.start);
    let range = TimeRange::new(first, end - first);
    let want = match p.get("track").filter(|v| !v.is_null()) {
        None => None,
        Some(Value::String(t)) => {
            let k = t.trim().strip_prefix(['V', 'v']).and_then(|n| n.parse::<usize>().ok()).filter(|n| (1..=q.video_tracks.len()).contains(n));
            let k = k.ok_or_else(|| bad(CMD, format!("`track` is a video track of the sequence, \"V1\" to \"V{}\"", q.video_tracks.len())))? - 1;
            let tr = q.video_tracks.get(k).ok_or_else(|| bad(CMD, "no such track"))?;
            if tr.locked || tr.items.iter().any(|i| i.range().overlaps(&range)) {
                return Err(bad(CMD, format!("V{} is locked or has clips under the bar; leave `track` out for a new track", k + 1)));
            }
            Some(k)
        }
        Some(_) => return Err(bad(CMD, "`track` is \"V<n>\"")),
    };
    // ---- one graphic clip a chapter
    let mut layers_of: Vec<Vec<EffectInstance>> = Vec::with_capacity(n);
    for (i, c) in chapters.iter().enumerate() {
        let dur = (c.end - c.start).max(rate.frame_duration());
        let mut layers = Vec::with_capacity(n + 2);
        for (j, (x0, x1)) in segments.iter().enumerate() {
            let color = match j.cmp(&i) {
                std::cmp::Ordering::Less => done_c,
                std::cmp::Ordering::Equal if !progress => current_c,
                _ => track_c,
            };
            let centre = Vec2::new((x0 + x1) / 2.0, cy);
            layers.push(shape(&format!("Chapter {} segment", j + 1), centre, Vec2::new(x1 - x0, bar_h), color, bar_h / 2.0));
        }
        if progress && let Some((x0, x1)) = segments.get(i) {
            // the fill grows from the segment's left edge to its right over the chapter
            // (a pixel wide at first: a shape of no width has no outline to fill)
            let (from_size, to_size) = (Vec2::new(1.0, bar_h), Vec2::new(x1 - x0, bar_h));
            let (from_pos, to_pos) = (Vec2::new(x0 + 0.5, cy), Vec2::new((x0 + x1) / 2.0, cy));
            let mut fill = shape(&format!("Chapter {} progress", i + 1), from_pos, from_size, current_c, bar_h / 2.0);
            let last = dur - rate.frame_duration().min(dur);
            let keyed = |a: Vec2, b: Vec2| Param {
                value: ParamValue::Vec2(a),
                keyframes: vec![Keyframe::new(Tick::ZERO, ParamValue::Vec2(a)), Keyframe::new(last, ParamValue::Vec2(b))],
            };
            fill.params.insert("size".into(), keyed(from_size, to_size));
            fill.params.insert("position".into(), keyed(from_pos, to_pos));
            layers.push(fill);
        }
        if labels && !c.title.is_empty() {
            let mut t = new_text_layer(&c.title, Vec2::new(margin_x, label_y), label_size);
            for (k, v) in [
                ("fill_color", ParamValue::Color(text_c)),
                ("font_style", ParamValue::Text("Bold".into())),
                ("background", ParamValue::Bool(true)),
                ("background_color", ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
                ("background_opacity", ParamValue::Float(55.0)),
                ("background_size", ParamValue::Float((label_size * 0.3).round())),
                ("background_radius", ParamValue::Float((label_size * 0.25).round())),
                ("name", ParamValue::Text("Chapter title".into())),
            ] {
                t.params.insert(k.into(), Param::new(v));
            }
            layers.push(t);
        }
        for e in &mut layers {
            filmcraft_project::resolve_auto_points(e, (w as u32, h as u32), (w as u32, h as u32));
        }
        layers_of.push(layers);
    }
    let (wu, hu) = (w as u32, h as u32);
    let made = s.edit("Chapter Bar", |pr, st| {
        let src = crate::graphics::graphic_source(pr, wu, hu, rate);
        let mut items = Vec::with_capacity(n);
        for ((i, c), layers) in chapters.iter().enumerate().zip(layers_of) {
            let dur = (c.end - c.start).max(rate.frame_duration());
            let mut it = pr
                .make_track_item(src, TrackKind::Video, c.start, TimeRange::new(Tick::ZERO, dur), rate)
                .ok_or_else(|| bad(CMD, "the graphic source is missing"))?;
            it.name = if c.title.is_empty() { format!("Chapter Bar {}/{n}", i + 1) } else { format!("Chapter Bar {}/{n}: {}", i + 1, c.title) };
            it.effects.extend(layers);
            items.push(it);
        }
        let new_track = TrackId(pr.alloc_id());
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let idx = match want {
            Some(k) => k,
            None => {
                q.video_tracks.push(Track::new(new_track, TrackKind::Video, "Chapter Bar".into()));
                q.video_tracks.len() - 1
            }
        };
        let ids: Vec<u64> = items.iter().map(|i| i.id.0).collect();
        let tr = q.video_tracks.get_mut(idx).ok_or_else(|| bad(CMD, "the bar's track is gone"))?;
        tr.items.extend(items);
        tr.sort();
        q.check().map_err(EngineError::Other)?;
        st.selection.clear();
        Ok((idx, ids))
    })?;
    let (idx, clips) = made;
    Ok(json!({"track": format!("V{}", idx + 1), "clips": clips, "chapters": plan, "bar": bar}))
}

#[cfg(test)]
#[path = "chapter_bar_tests.rs"]
mod tests;
