//! `script.shots` (PX7.4): a picture for every shot of an explainer that `script.timeline`
//! placed, read from the sequence's markers (each shot marker `shot-NN` spans its shot; its
//! comment is the shot's first `[shot: …]` hint; the chapter markers name the chapters).
//!
//! **Through RAEA** (`via: "motion"`, or `auto` with a `source` when RAEA is installed): one
//! composition a shot, named by its id (`shot-01`, `shot-02`…, written from the `shot-NN.json`
//! files of `script.timeline {out}`), rendered by `motion.render` one after the other into
//! `<output>/<shot id>` and placed on the picture track at the shot's start. A job: it runs the
//! renders through the command port, each a `motion.render` job of its own under the GPU lock,
//! and stops at the first that fails. With `exact` (the default) a render must last its shot's
//! frames, so the cut stays on the narration.
//!
//! **As cards** (`via: "cards"`, or `auto` without RAEA or a source): one graphic clip a shot on
//! the picture track (V1 by default; it must be free): a plain backdrop, the chapter's title small
//! at the top, and the shot's hint (else its chapter's title) as the hero, fitted as large as it
//! goes within the frame and at least a quarter of its height when it can be, centred. The hero
//! enters by line (fade, slide up and blur in by turns, so no two neighbours enter alike), then
//! pushes in slowly (scale 100 → 104 %) until a second before its shot ends, so the picture is
//! never still for long and settles before the cut: the explainer rules of `sequence.checkCraft`
//! and `checkMotion` (PX7.5). One undo step. A shot without a hint whose chapter has no title gets
//! the backdrop alone (and the still check says so).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::PoisonError;
use std::sync::atomic::Ordering;

use filmcraft_geom::Vec2;
use filmcraft_project::graphic::{eval_layer, new_shape_layer, new_text_layer};
use filmcraft_project::{EffectInstance, Keyframe, MarkerKind, Param, ParamValue, Sequence, TrackKind};
use filmcraft_render::graphic_clip::{layer_bounds, layer_local_bounds};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::explainer::{MAX_SHOTS, shot_index};
use crate::motion_energy::r3;
use crate::{EngineError, Result, Session};

pub const CMD: &str = "script.shots";
/// The hero's least share of the frame's height (as `sequence.checkCraft` asks), the largest
/// font size tried and the smallest, shares of the height; the room it may take.
const HERO_SHARE: f64 = crate::craft_check::HERO_SHARE;
const HERO_MAX_SIZE: f64 = 0.26;
const HERO_MIN_SIZE: f64 = 0.06;
const HERO_WIDTH: f64 = 0.84;
const HERO_HEIGHT: f64 = 0.55;
/// The longest hint a card shows, characters and words.
const MAX_HERO_CHARS: usize = 120;
const MAX_HERO_WORDS: usize = 24;
/// The chapter title's size, a share of the height, and the margins, shares of each side.
const KICKER_SHARE: f64 = 0.045;
const MARGIN_X: f64 = 0.08;
const MARGIN_Y: f64 = 0.14;
/// The entrance (seconds, by line) and the slow push after it (scale, percent), ending this long
/// before the shot leaves.
const ENTRANCE: f64 = 0.5;
const ENTRANCE_KINDS: &[&str] = &["fade", "slideUp", "blurIn"];
const PUSH_TO: f64 = 104.0;
const SETTLE: f64 = 1.0;
/// The default colours (sRGB, straight alpha).
const BACKGROUND: [f32; 4] = [0.08, 0.13, 0.24, 1.0];
const TEXT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ACCENT: [f32; 4] = [1.0, 0.69, 0.0, 1.0];
const KEYS: &[&str] = &["via", "source", "output", "exact", "track", "style", "dryRun", "wait"];

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Explainer Shots",
        menu: &[],
        shortcut: None,
        params: r##"{"via":"auto"|"motion"|"cards"="auto" (motion: one RAEA composition a shot; auto: motion with a source when RAEA is installed, else cards),"source":path? (an absolute .tsx or .html holding one composition a shot, named by its id: shot-01…),"output":path? (motion: a new or empty folder; each shot renders into <output>/<shot id>),"exact":bool=true (motion: a render lasts its shot's frames),"track":"V<n>"="V1","style":{"background":"#14213d","text":"#ffffff","accent":"#ffb000","font":str?}?,"dryRun":bool=false (the plan only),"wait":bool=false (motion: return when every render has landed)}"##,
        enabled: has_seq,
        run,
        journal: true,
    }]
}

/// A shot as its marker says.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ShotMark {
    pub id: String,
    pub start: Tick,
    pub end: Tick,
    pub chapter: String,
    pub hint: String,
}

/// The shots of `q`: its shot markers in time order, each to the end of its span (to the next
/// shot when it has none, the last to the sequence's end), with the title of the chapter it is in.
pub(crate) fn shots_of(q: &Sequence) -> std::result::Result<Vec<ShotMark>, String> {
    let mut marks: Vec<&filmcraft_project::Marker> = q.markers.iter().filter(|m| m.kind == MarkerKind::Comment && shot_index(&m.name).is_some()).collect();
    if marks.is_empty() {
        return Err("the sequence has no shot markers (shot-01…): lay the script out with script.timeline {place} first".into());
    }
    if marks.len() > MAX_SHOTS {
        return Err(format!("{} shot markers; at most {MAX_SHOTS}", marks.len()));
    }
    marks.sort_by_key(|m| m.start);
    let mut chapters: Vec<(Tick, String)> = q.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).map(|m| (m.start, m.name.clone())).collect();
    chapters.sort_by_key(|c| c.0);
    let end = q.duration();
    let mut out = Vec::with_capacity(marks.len());
    for (i, m) in marks.iter().enumerate() {
        let next = marks.get(i + 1).map(|n| n.start);
        let mut stop = if m.duration > Tick::ZERO { m.start + m.duration } else { next.unwrap_or(end) };
        if let Some(n) = next {
            stop = stop.min(n);
        }
        if stop <= m.start {
            return Err(format!("shot marker {} has no length (the next starts with it, or the sequence ends there)", m.name));
        }
        let chapter = chapters.iter().rev().find(|c| c.0 <= m.start).map(|c| c.1.clone()).unwrap_or_default();
        out.push(ShotMark { id: m.name.clone(), start: m.start, end: stop, chapter, hint: m.comment.trim().to_string() });
    }
    Ok(out)
}

/// `#rrggbb` (or `#rrggbbaa`) as straight sRGB 0..1.
fn color_p(v: &Value, key: &str) -> Result<[f32; 4]> {
    let s = v.as_str().map(str::trim).unwrap_or("");
    let h = s.strip_prefix('#').unwrap_or(s);
    if !(h.len() == 6 || h.len() == 8) || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad(CMD, format!("`style.{key}` is \"#rrggbb\"")));
    }
    let byte = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map_or(255.0, f32::from) / 255.0;
    Ok([byte(0), byte(2), byte(4), if h.len() == 8 { byte(6) } else { 1.0 }])
}

/// The cards' look.
#[derive(Clone, Debug)]
struct Style {
    background: [f32; 4],
    text: [f32; 4],
    accent: [f32; 4],
    font: Option<String>,
}

fn style_p(p: &Value) -> Result<Style> {
    let mut st = Style { background: BACKGROUND, text: TEXT, accent: ACCENT, font: None };
    let Some(v) = p.get("style").filter(|v| !v.is_null()) else { return Ok(st) };
    let o = v.as_object().ok_or_else(|| bad(CMD, "`style` is {background?, text?, accent?, font?}"))?;
    for (k, x) in o {
        match k.as_str() {
            "background" => st.background = color_p(x, k)?,
            "text" => st.text = color_p(x, k)?,
            "accent" => st.accent = color_p(x, k)?,
            "font" => {
                st.font = Some(
                    x.as_str()
                        .map(str::trim)
                        .filter(|f| !f.is_empty() && f.chars().count() <= 100)
                        .ok_or_else(|| bad(CMD, "`style.font` is a font family name"))?
                        .to_string(),
                )
            }
            other => {
                return Err(bad(CMD, format!("`style.{}` is not a card setting (background, text, accent, font)", other.chars().take(40).collect::<String>())));
            }
        }
    }
    Ok(st)
}

/// A text layer of `text` at `size`, with the look's font, bold, in `color`, aligned (0 left,
/// 1 centre).
fn text_layer(text: &str, size: f64, color: [f32; 4], align: u32, st: &Style, name: &str) -> EffectInstance {
    let mut e = new_text_layer(text, Vec2::new(0.0, 0.0), size);
    let mut set = |k: &str, v: ParamValue| {
        e.params.insert(k.into(), Param::new(v));
    };
    set("fill_color", ParamValue::Color(color));
    set("font_style", ParamValue::Text("Bold".into()));
    set("align", ParamValue::Choice(align));
    set("name", ParamValue::Text(name.into()));
    if let Some(f) = &st.font {
        set("font", ParamValue::Text(f.clone()));
    }
    e
}

/// The layer's box on the frame (`[x0, y0, x1, y1]`) and in its own space.
fn measure(e: &EffectInstance, frame: (u32, u32)) -> Option<([f64; 4], [f32; 4])> {
    let spec = eval_layer(e, Tick::ZERO, frame)?;
    let b = layer_bounds(&spec);
    let local = layer_local_bounds(&spec);
    (b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]).then_some((b, local))
}

fn set_vec(e: &mut EffectInstance, k: &str, v: Vec2) {
    e.params.insert(k.into(), Param::new(ParamValue::Vec2(v)));
}

/// `words` wrapped greedily at `size` into lines no wider than `max_w`.
fn wrap(words: &[&str], size: f64, max_w: f64, st: &Style, frame: (u32, u32)) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for w in words {
        let candidate = if cur.is_empty() { (*w).to_string() } else { format!("{cur} {w}") };
        let wide = measure(&text_layer(&candidate, size, TEXT, 1, st, ""), frame).is_some_and(|(b, _)| b[2] - b[0] > max_w);
        if wide && !cur.is_empty() {
            lines.push(std::mem::take(&mut cur));
            cur = (*w).to_string();
        } else {
            cur = candidate;
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// The hero: `text` as large as it fits (from 26 % of the height down), its visual centre at
/// `centre`, anchored there so a scale grows it in place; and its height on the frame.
fn hero(text: &str, centre: Vec2, st: &Style, frame: (u32, u32)) -> Option<(EffectInstance, f64)> {
    let (w, h) = (f64::from(frame.0), f64::from(frame.1));
    let mut short: String = text.chars().take(MAX_HERO_CHARS).collect();
    if text.chars().count() > MAX_HERO_CHARS {
        short.push('\u{2026}');
    }
    let words: Vec<&str> = short.split_whitespace().take(MAX_HERO_WORDS).collect();
    if words.is_empty() {
        return None;
    }
    let mut chosen: Option<(EffectInstance, [f64; 4], [f32; 4])> = None;
    let steps = ((HERO_MAX_SIZE - HERO_MIN_SIZE) / 0.012).ceil() as usize;
    for k in 0..=steps {
        let size = (h * (HERO_MAX_SIZE - 0.012 * k as f64)).max(h * HERO_MIN_SIZE).round();
        let lines = wrap(&words, size, HERO_WIDTH * w, st, frame);
        let e = text_layer(&lines.join("\n"), size, st.text, 1, st, "Hero");
        let Some((b, local)) = measure(&e, frame) else { continue };
        let fits = b[2] - b[0] <= HERO_WIDTH * w && b[3] - b[1] <= HERO_HEIGHT * h;
        chosen = Some((e, b, local));
        if fits {
            break;
        }
    }
    let (mut e, b, local) = chosen?;
    // anchor at the box's centre (layer space) and put that centre at `centre`
    let anchor = Vec2::new(f64::from(local[0] + local[2]) / 2.0, f64::from(local[1] + local[3]) / 2.0);
    set_vec(&mut e, "anchor", anchor);
    set_vec(&mut e, "position", centre);
    // whatever the transform's convention, measure again and shift the box onto the centre
    if let Some((nb, _)) = measure(&e, frame) {
        let shift = Vec2::new(centre.x - (nb[0] + nb[2]) / 2.0, centre.y - (nb[1] + nb[3]) / 2.0);
        set_vec(&mut e, "position", Vec2::new(centre.x + shift.x, centre.y + shift.y));
    }
    let height = measure(&e, frame).map_or(b[3] - b[1], |(nb, _)| nb[3] - nb[1]);
    Some((e, height))
}

/// One card: the layers, the hero's layer index (among the graphic layers) and its height.
fn card(shot: &ShotMark, st: &Style, frame: (u32, u32)) -> (Vec<EffectInstance>, Option<usize>, f64) {
    let (w, h) = (f64::from(frame.0), f64::from(frame.1));
    let mut layers = Vec::new();
    let mut back = new_shape_layer(0, Vec2::new(w / 2.0, h / 2.0), Vec2::new(w, h), Vec::new());
    back.params.insert("fill_color".into(), Param::new(ParamValue::Color(st.background)));
    back.params.insert("name".into(), Param::new(ParamValue::Text("Backdrop".into())));
    layers.push(back);
    let main = if shot.hint.is_empty() { shot.chapter.clone() } else { shot.hint.clone() };
    if !shot.chapter.is_empty() && shot.chapter != main {
        let size = (KICKER_SHARE * h).round().max(12.0);
        let mut k = text_layer(&shot.chapter, size, st.accent, 0, st, "Chapter");
        set_vec(&mut k, "position", Vec2::new((MARGIN_X * w).round(), (MARGIN_Y * h).round()));
        layers.push(k);
    }
    let mut hero_index = None;
    let mut height = 0.0;
    if let Some((e, hh)) = hero(&main, Vec2::new(w / 2.0, h * 0.53), st, frame) {
        hero_index = Some(layers.len());
        height = hh;
        layers.push(e);
    }
    for e in &mut layers {
        filmcraft_project::resolve_auto_points(e, frame, frame);
    }
    (layers, hero_index, height)
}

/// An absolute path, or none (absent, null or empty: a recipe's empty parameter).
fn opt_path(p: &Value, key: &str) -> Result<Option<std::path::PathBuf>> {
    match p.get(key) {
        Some(Value::String(t)) if t.trim().is_empty() => Ok(None),
        _ => crate::motion_graphics::path_p(CMD, p, key),
    }
}

fn flag(p: &Value, key: &str, default: bool) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, format!("`{key}` is true or false"))),
    }
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !KEYS.contains(&k.as_str())).cloned()) {
        return Err(bad(CMD, format!("unknown parameter `{}` ({})", k.chars().take(64).collect::<String>(), KEYS.join(", "))));
    }
    let via = match p.get("via") {
        None | Some(Value::Null) => "auto",
        Some(Value::String(v)) if matches!(v.as_str(), "auto" | "motion" | "cards") => v.as_str(),
        Some(_) => return Err(bad(CMD, "`via` is \"auto\", \"motion\" or \"cards\"")),
    };
    let source = opt_path(p, "source")?;
    let output = opt_path(p, "output")?;
    let exact = flag(p, "exact", true)?;
    let dry = flag(p, "dryRun", false)?;
    let wait = flag(p, "wait", false)?;
    let style = style_p(p)?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let track_n = match p.get("track") {
        None | Some(Value::Null) => 1,
        Some(Value::String(t)) => t
            .trim()
            .strip_prefix(['V', 'v'])
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|n| (1..=q.video_tracks.len().max(1)).contains(n))
            .ok_or_else(|| bad(CMD, format!("`track` is a video track of the sequence, \"V1\" to \"V{}\"", q.video_tracks.len())))?,
        Some(_) => return Err(bad(CMD, "`track` is \"V<n>\"")),
    };
    let shots = shots_of(q).map_err(|e| bad(CMD, e))?;
    let frame = (q.settings.width.max(1), q.settings.height.max(1));
    let rate = q.settings.frame_rate.sane();
    // ---- which way
    let raea = match via {
        "cards" => Err("cards were asked for".to_string()),
        _ => match &source {
            None => Err("no `source` composition".to_string()),
            Some(_) => crate::motion_graphics::tool_for(CMD).map(|_| ()).map_err(|e| e.to_string()),
        },
    };
    let motion = match (via, &raea) {
        ("motion", Err(why)) => return Err(bad(CMD, format!("`via: \"motion\"` cannot render: {why}"))),
        ("motion", Ok(())) => true,
        (_, Ok(())) => true,
        _ => false,
    };
    let plan: Vec<Value> = shots
        .iter()
        .map(|sh| {
            let mut v = json!({"id": sh.id, "start": sh.start.0, "end": sh.end.0, "seconds": r3((sh.end - sh.start).seconds()),
                "frames": rate.frame_at(sh.end).saturating_sub(rate.frame_at(sh.start)), "chapter": sh.chapter, "hint": sh.hint});
            if motion {
                v["composition"] = json!(sh.id);
            } else {
                v["hero"] = json!(if sh.hint.is_empty() { &sh.chapter } else { &sh.hint });
            }
            v
        })
        .collect();
    let mut out = json!({"via": if motion { "motion" } else { "cards" }, "shots": plan, "track": format!("V{track_n}")});
    if !motion && via == "auto" {
        out["why"] = json!(match &raea {
            Err(why) => format!("cards: {why}"),
            Ok(()) => String::new(),
        });
    }
    if dry {
        out["dryRun"] = json!(true);
        return Ok(out);
    }
    if motion {
        let (Some(source), Some(output)) = (source, output) else {
            return Err(bad(CMD, "rendering through RAEA needs `source` (the composition) and `output` (a new or empty folder)"));
        };
        let job = motion_job(s, &shots, &source, &output, exact, track_n, frame, rate)?;
        out["job"] = json!(job);
        out["output"] = json!(output.display().to_string());
        if wait {
            crate::recipes::wait_for(s, job);
            if let Some(j) = s.jobs.iter().find(|j| j.id == job) {
                let v = j.to_json();
                if let Some(e) = v["result"]["error"].as_str() {
                    return Err(EngineError::Other(format!("{CMD}: {e}")));
                }
                out["result"] = v["result"].clone();
            }
        }
        return Ok(out);
    }
    // ---- cards
    let ti = track_n - 1;
    let busy = q
        .video_tracks
        .get(ti)
        .is_some_and(|t| t.locked || shots.iter().any(|sh| t.items.iter().any(|i| i.range().overlaps(&TimeRange::from_bounds(sh.start, sh.end)))));
    if busy {
        return Err(bad(CMD, format!("V{track_n} is locked or has clips under the shots; give a free `track`")));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let built: Vec<(Vec<EffectInstance>, Option<usize>, f64)> = shots.iter().map(|sh| card(sh, &style, frame)).collect();
    let heights: Vec<f64> = built.iter().map(|b| b.2).collect();
    let r = s.edit_group("Explainer Shots", |s| {
        let mut made: Vec<(u64, Option<usize>)> = Vec::with_capacity(shots.len());
        s.edit("Explainer Cards", |pr, st| {
            let src = crate::graphics::graphic_source(pr, frame.0, frame.1, rate);
            let mut items = Vec::with_capacity(shots.len());
            for (sh, (layers, hero_i, _)) in shots.iter().zip(&built) {
                let mut it = pr
                    .make_track_item(src, TrackKind::Video, sh.start, TimeRange::new(Tick::ZERO, sh.end - sh.start), rate)
                    .ok_or_else(|| bad(CMD, "the graphic source is missing"))?;
                it.name = format!("{}: {}", sh.id, if sh.hint.is_empty() { &sh.chapter } else { &sh.hint }).chars().take(120).collect();
                let mut layers = layers.clone();
                // the slow push after the entrance, ending a second before the shot leaves
                let dur = (sh.end - sh.start).seconds();
                let (t0, t1) = (ENTRANCE + 0.1, dur - SETTLE);
                if let Some(e) = hero_i.and_then(|i| layers.get_mut(i))
                    && t1 - t0 >= 0.5
                {
                    let at = |x: f64| rate.snap_nearest(Tick::from_seconds_f64(x));
                    e.params.insert(
                        "scale".into(),
                        Param {
                            value: ParamValue::Float(100.0),
                            keyframes: vec![Keyframe::new(at(t0), ParamValue::Float(100.0)), Keyframe::new(at(t1), ParamValue::Float(PUSH_TO))],
                        },
                    );
                }
                it.effects.extend(layers);
                made.push((it.id.0, *hero_i));
                items.push(it);
            }
            let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            let tr = q.video_tracks.get_mut(ti).ok_or_else(|| bad(CMD, "the picture track is gone"))?;
            tr.items.extend(items);
            tr.sort();
            q.check().map_err(EngineError::Other)?;
            st.selection.clear();
            Ok(())
        })?;
        // the entrances: by line, the kinds taking turns
        for (n, (clip, hero_i)) in made.iter().enumerate() {
            let Some(layer) = hero_i else { continue };
            let kind = ENTRANCE_KINDS.get(n % ENTRANCE_KINDS.len()).copied().unwrap_or("fade");
            s.execute(
                "graphics.animateText",
                json!({"clip": clip, "layer": layer, "in": {"kind": kind, "unit": "line", "stagger": 0.1, "duration": ENTRANCE}}),
            )?;
        }
        Ok(made)
    })?;
    let h = f64::from(frame.1);
    let small: Vec<&str> = shots.iter().zip(&heights).filter(|(_, hh)| **hh > 0.0 && **hh < HERO_SHARE * h).map(|(sh, _)| sh.id.as_str()).collect();
    out["clips"] = json!(r.iter().map(|(c, _)| c).collect::<Vec<_>>());
    out["heroHeights"] = json!(heights.iter().map(|x| r3(x / h)).collect::<Vec<_>>());
    if !small.is_empty() {
        out["warnings"] =
            json!([format!("the hint of {} is too long to show at a quarter of the frame's height: shorten it to a name or a number", small.join(", "))]);
    }
    Ok(out)
}

/// The renders as a job: `motion.render` for each shot through the command port, one after the
/// other (each its own job under the GPU lock), placed at the shot's start.
#[allow(clippy::too_many_arguments)]
fn motion_job(
    s: &mut Session,
    shots: &[ShotMark],
    source: &std::path::Path,
    output: &std::path::Path,
    exact: bool,
    track_n: usize,
    frame: (u32, u32),
    rate: filmcraft_time::FrameRate,
) -> Result<u64> {
    if cfg!(target_arch = "wasm32") {
        return Err(bad(CMD, "RAEA runs on the desktop app, not in the browser"));
    }
    // a new or empty folder: no render replaces one that is there
    crate::motion_graphics::revision_dir(CMD, output)?;
    let calls: Vec<(String, Value)> = shots
        .iter()
        .map(|sh| {
            let mut p = json!({
                "source": source.display().to_string(),
                "composition": sh.id,
                "output": output.join(&sh.id).display().to_string(),
                "size": format!("{}x{}", frame.0, frame.1),
                "place": {"time": sh.start.0, "track": format!("V{track_n}")},
                "wait": false,
            });
            if exact {
                p["expected"] = json!({"duration": (sh.end - sh.start).0, "width": frame.0, "height": frame.1, "fps": rate.as_f64()});
            }
            (sh.id.clone(), p)
        })
        .collect();
    let job = crate::narration::new_job(s, format!("Explainer shots: {} compositions", calls.len()));
    let total = (calls.len() as u64).saturating_mul(100).max(1);
    job.progress.total.store(total, Ordering::Relaxed);
    let id = job.id;
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let port = s.autocut.port.handle(Some(prog.clone()));
    let body = crate::export_tools::guard_job(prog.clone(), res.clone(), move || {
        let t0 = web_time::Instant::now();
        let mut done = Vec::new();
        let mut failed: Option<String> = None;
        for (n, (shot, params)) in calls.iter().enumerate() {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("{}/{} {shot}", n + 1, calls.len());
            let base = (n as u64).saturating_mul(100);
            let r = port.call("motion.render", params.clone()).and_then(|v| match v.get("job").and_then(Value::as_u64) {
                Some(j) => crate::recipes::wait_job(j, &port, &prog, base),
                None => Ok(v),
            });
            match r {
                Ok(_) => done.push(shot.clone()),
                Err(e) => {
                    failed = Some(format!("{shot}: {e}"));
                    break;
                }
            }
        }
        prog.done.store(total, Ordering::Relaxed);
        let status = match failed {
            Some(e) => Err(format!("{e} ({} of {} shots placed)", done.len(), calls.len())),
            None => Ok(format!("{} shots rendered and placed", done.len())),
        };
        crate::narration::finish(&prog, &res, status, t0);
    });
    s.jobs.push(job);
    std::thread::Builder::new()
        .name("filmcraft-explainer-shots".into())
        .spawn(body)
        .map_err(|e| EngineError::Other(format!("{CMD}: could not start the job: {e}")))?;
    Ok(id)
}

#[cfg(test)]
#[path = "explainer_shots_tests.rs"]
mod tests;
