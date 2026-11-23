//! `sequence.checkMotion` (PX1.5): how a sequence's titles, graphics and keyframes move, read
//! from the timeline alone (nothing is rendered). Thresholds after the motion guidance of
//! impeccable (`reference/animate.md`, Apache-2.0), taste-skill (`references/motion.md`, MIT)
//! and hallmark (`references/craft.md`, MIT), read on GitHub on 23 November 2026; every rule and
//! message here is written fresh, and every number is a starting value to tune on real projects.
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `overshoot` | an ease passes its mark or bounces: a text animator or a transition easing with Back, Elastic or Bounce (or a Custom Bézier whose handles leave 0..1), Pop Motion's overshoot or Spring Motion's bounce, a spring keyframe under the damping ratio, a custom curve (as `keyframes.choreograph` writes for `back`, `elastic` and `bounce`) whose progress leaves 0..1 or turns back; unless a playful style is declared | damping ratio 0.9 (PX5); 0.5 % past the mark, 2 % back |
//! | `entranceDuration` | a title's or overlay's entrance (a text animator's unit, a transition from nothing, its first keyframed move) is shorter or longer than the band | 8–30 frames at 30 fps, scaled by the rate (0.27–1 s) |
//! | `exitDuration` | the same for an exit | 6–24 frames at 30 fps (0.2–0.8 s) |
//! | `exitOutlastsEntrance` | an exit takes longer than the entrance it mirrors (the same text layer's animators, the clip's two transitions or its edge keyframes) | exit at most the entrance; 60–75 % of it reads best |
//! | `longStagger` | a text animator's letters, words or lines take longer than this from the first to the last one starting | 0.8 s |
//! | `sameEntrance` | three or more titles, and every one of them enters the same way | 3 titles |
//! | `loopingText` | a looping text animator keeps a title moving for its whole clip | loops quicker than 8 s a cycle (an ambient drift is slower) |
//! | `edgePop` | a keyframed position, scale, rotation or opacity changes by half its move or more between a clip's first two frames or its last two | 50 % of the keyframed range; at least 10 points of opacity, 5 % of scale, 5° or 1 % of the frame's short side |
//!
//! Where the numbers come from: entrances of 500–800 ms for a focal element and exits faster
//! than entrances (impeccable); a reveal of 0.4–0.8 s in all, exits at 60–75 % of the entrance,
//! a total stagger of at most 500 ms, no infinite loops and ambient loops of 8 s or slower
//! (hallmark); bounce, elastic, back and overshoot banned, no curve passing its mark, and exit
//! durations 0.6–0.8 of their entrances (taste-skill). The bands are wider than the web's
//! because a video title is read from further away and for longer.
//!
//! **A playful style** (`style: "playful"`, or `autocut.taste` with `motion: "playful"`) lets eases
//! overshoot and bounce; the other checks still run. `style: "calm"` checks overshoot even when
//! the taste is playful.
//!
//! A title is a graphic clip with at least one enabled text layer; an overlay is a clip of a
//! picture with alpha (what `motion.render` and `composition.import` bring in). Text-animator
//! units are counted from the laid-out text (letters as shaped clusters, as the renderer
//! animates them). Clips inside nested sequences are not read. Every finding is a warning
//! (decision D7) and never blocks a delivery by itself.
use std::collections::BTreeMap;

use filmcraft_project::easing::Bezier;
use filmcraft_project::gltransition::EASING_OPTS;
use filmcraft_project::graphic::{LayerContent, TEXT_LAYER};
use filmcraft_project::keyframe::Interpolation;
use filmcraft_project::keyframe_ease::curve_progress;
use filmcraft_project::text_anim::{AnimKind, AnimUnit, Animator, word_units};
use filmcraft_project::{EffectInstance, ItemId, ItemKind, Param, ParamValue, Project, Sequence, TrackItem};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::motion_energy::{bounded, finding, r3};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.checkMotion";
/// The bands at 30 frames a second: entrance and exit, frames (scaled by the sequence's rate).
pub const ENTRANCE_FRAMES_30: (f64, f64) = (8.0, 30.0);
pub const EXIT_FRAMES_30: (f64, f64) = (6.0, 24.0);
/// Longest stagger from a text animator's first unit starting to its last, seconds.
pub const STAGGER_SECONDS: f64 = 0.8;
/// A spring under this damping ratio overshoots visibly (PX5: 0.9 passes its mark by 0.15 %).
pub const SPRING_DAMPING: f64 = 0.9;
/// This many titles or more all entering the same way is a pattern.
pub const SAME_ENTRANCE_TITLES: usize = 3;
/// A loop of this period or slower is an ambient drift, seconds (hallmark).
pub const AMBIENT_LOOP_SECONDS: f64 = 8.0;
/// A change in one frame of at least this share of a property's keyframed range is a pop.
pub const EDGE_POP_SHARE: f64 = 0.5;
/// Least change that is seen: opacity points, scale percent, degrees, share of the short side.
const POP_OPACITY: f64 = 10.0;
const POP_SCALE: f64 = 5.0;
const POP_DEGREES: f64 = 5.0;
const POP_POSITION_SHARE: f64 = 0.01;
/// A curve past its mark by more than this share of the change overshoots; one that falls back by
/// more than this after rising bounces.
const OVERSHOOT_EPS: f64 = 0.005;
const BOUNCE_BACK: f64 = 0.02;
/// Points a custom curve is sampled at.
const CURVE_SAMPLES: usize = 64;
/// The properties that make an entrance or an exit, and that a pop is looked for in.
const MOVE_PROPS: &[&str] = &["position", "scale", "scale_width", "scale_height", "rotation", "opacity"];
/// Most clips, keyframes of one parameter and text units read.
const MAX_CLIPS: usize = 20_000;
const MAX_KEYFRAMES: usize = 10_000;
const MAX_UNITS: usize = 4096;
/// Longest band (frames) and stagger (seconds) a caller may set.
const MAX_BAND_FRAMES: f64 = 100_000.0;
const MAX_STAGGER: f64 = 600.0;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Motion",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (default the active one),"style":"calm"|"playful"? (playful lets eases overshoot; default the taste profile's motion, else calm),"bands":{"entrance":[minFrames,maxFrames]? (default 8–30 at 30 fps, scaled to the sequence's rate),"exit":[minFrames,maxFrames]? (default 6–24 at 30 fps),"stagger":seconds? (default 0.8)}?}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

/// The bands, in frames at the sequence's rate, and the longest stagger.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bands {
    entrance: (f64, f64),
    exit: (f64, f64),
    stagger: f64,
}

fn band_p(v: &Value, key: &str) -> Result<(f64, f64)> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(CMD, format!("`bands.{key}` is [minFrames, maxFrames]")))?;
    let n = |i: usize| a.get(i).and_then(Value::as_f64).filter(|x| x.is_finite() && (0.0..=MAX_BAND_FRAMES).contains(x));
    match (n(0), n(1)) {
        (Some(lo), Some(hi)) if lo <= hi && hi > 0.0 => Ok((lo, hi)),
        _ => Err(bad(CMD, format!("`bands.{key}` is [minFrames, maxFrames]: 0 ≤ min ≤ max ≤ {MAX_BAND_FRAMES}, max above 0"))),
    }
}

fn bands_p(p: &Value, fps: f64) -> Result<Bands> {
    let k = fps / 30.0;
    let mut b =
        Bands { entrance: (ENTRANCE_FRAMES_30.0 * k, ENTRANCE_FRAMES_30.1 * k), exit: (EXIT_FRAMES_30.0 * k, EXIT_FRAMES_30.1 * k), stagger: STAGGER_SECONDS };
    let Some(v) = p.get("bands").filter(|v| !v.is_null()) else { return Ok(b) };
    let o = v.as_object().ok_or_else(|| bad(CMD, "`bands` is {entrance?, exit?, stagger?}"))?;
    for (key, x) in o {
        match key.as_str() {
            "entrance" => b.entrance = band_p(x, key)?,
            "exit" => b.exit = band_p(x, key)?,
            "stagger" => {
                b.stagger = x
                    .as_f64()
                    .filter(|s| s.is_finite() && *s > 0.0 && *s <= MAX_STAGGER)
                    .ok_or_else(|| bad(CMD, format!("`bands.stagger` is seconds, above 0 and at most {MAX_STAGGER}")))?
            }
            other => return Err(bad(CMD, format!("`bands` has no `{}`", other.chars().take(64).collect::<String>()))),
        }
    }
    Ok(b)
}

/// Whether overshoot is allowed, and who said so.
fn style_p(s: &Session, p: &Value) -> Result<(bool, &'static str)> {
    match p.get("style").filter(|v| !v.is_null()) {
        Some(v) => match v.as_str().map(|t| t.trim().to_ascii_lowercase()).as_deref() {
            Some("playful") => Ok((true, "param")),
            Some("calm") => Ok((false, "param")),
            _ => Err(bad(CMD, "`style` is \"calm\" or \"playful\"")),
        },
        None if crate::taste::resolved(s).playful_motion() => Ok((true, "taste")),
        None => Ok((false, "default")),
    }
}

/// An easing index that passes its mark or bounces: its family, or None.
fn overshooting(easing: u32, bezier: Option<[f64; 4]>) -> Option<&'static str> {
    match easing {
        22..=24 => Some("back"),
        25..=27 => Some("elastic"),
        28..=30 => Some("bounce"),
        31 => {
            let b = bezier.map_or(Bezier::EASE, |[x1, y1, x2, y2]| Bezier { x1, y1, x2, y2 }).sanitized();
            // a Bézier stays inside its handles' hull: it passes 0..1 only when a handle does
            (b.y1 < -1e-9 || b.y1 > 1.0 + 1e-9 || b.y2 < -1e-9 || b.y2 > 1.0 + 1e-9).then_some("bezier")
        }
        _ => None,
    }
}

fn easing_name(easing: u32) -> &'static str {
    EASING_OPTS.get(easing as usize).copied().unwrap_or("Linear")
}

/// How far a custom curve passes its mark (`over`, a share of the change beyond 0..1) and falls
/// back after rising (`back`).
fn curve_shape(points: &[[f64; 2]]) -> (f64, f64) {
    let (mut hi, mut lo, mut peak, mut back) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..=CURVE_SAMPLES {
        let v = curve_progress(points, i as f64 / CURVE_SAMPLES as f64);
        if !v.is_finite() {
            continue;
        }
        hi = hi.max(v);
        lo = lo.min(v);
        peak = peak.max(v);
        back = back.max(peak - v);
    }
    ((hi - 1.0).max(-lo).max(0.0), back)
}

/// The share of the change an under-damped spring passes its mark by, at most its cap.
fn spring_overshoot(s: filmcraft_project::keyframe_ease::Spring) -> f64 {
    let s = s.sanitized();
    let z = s.damping_ratio;
    if z >= 1.0 {
        return 0.0;
    }
    let free = (-z * std::f64::consts::PI / (1.0 - z * z).sqrt()).exp();
    if free.is_finite() { free.min(s.max_overshoot) } else { 0.0 }
}

/// What a clip is to these checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Title,
    Overlay,
    Other,
}

fn role(project: &Project, it: &TrackItem) -> Role {
    match project.item(it.item).map(|p| &p.kind) {
        Some(ItemKind::Graphic { .. }) if it.effects.iter().any(|e| e.effect == TEXT_LAYER && e.enabled) => Role::Title,
        Some(ItemKind::Graphic { .. }) => Role::Other,
        _ if project.resolve_media(it.item).and_then(|(_, m, _)| m.info.video.as_ref()).is_some_and(|v| v.has_alpha) => Role::Overlay,
        _ => Role::Other,
    }
}

/// How many units (letters as shaped clusters, words or lines) text layer `ei` of `it` animates,
/// as the renderer counts them (the text at the clip's middle).
fn unit_count(project: &Project, it: &TrackItem, ei: usize, unit: AnimUnit) -> usize {
    let Some(src) = project.source_size(it.item) else { return 1 };
    let mid = Tick(it.start.0.saturating_add(it.duration.0 / 2));
    let specs = filmcraft_render::graphic_clip::item_layer_specs(it, it.effect_time_at(mid), src);
    let Some((_, spec)) = specs.iter().find(|(i, _)| *i == ei) else { return 1 };
    let LayerContent::Text(tp) = &spec.content else { return 1 };
    let l = filmcraft_render::graphic_clip::text_layout(tp);
    let text = &tp.text;
    let space = |b: usize| text.get(b..).and_then(|s| s.chars().next()).is_none_or(char::is_whitespace);
    let n = match unit {
        AnimUnit::Letter => {
            let mut c: Vec<usize> = l.glyphs.iter().map(|g| g.cluster).filter(|b| !space(*b)).collect();
            c.sort_unstable();
            c.dedup();
            c.len()
        }
        AnimUnit::Word => {
            let (words, _) = word_units(text);
            let mut used: Vec<usize> = l.glyphs.iter().filter(|g| !space(g.cluster)).filter_map(|g| words.get(g.cluster).copied()).collect();
            used.sort_unstable();
            used.dedup();
            used.len()
        }
        AnimUnit::Line => l.lines.iter().filter(|line| line.glyphs.clone().any(|gi| l.glyphs.get(gi).is_some_and(|g| !space(g.cluster)))).count(),
    };
    n.clamp(1, MAX_UNITS)
}

/// An entrance or an exit.
#[derive(Clone, Debug)]
struct Move {
    clip: u64,
    track: String,
    layer: Option<usize>,
    /// `animator`, `transition` or `keyframes`.
    source: &'static str,
    /// What moves: an animator's kind, a transition's effect, the keyframed properties.
    kind: String,
    /// The move as the messages name it.
    what: String,
    entering: bool,
    frames: f64,
    start: Tick,
    end: Tick,
}

/// Timeline seconds a clip plays a second of its media (1 when it cannot be told).
fn media_per_second(it: &TrackItem) -> f64 {
    let span = it.source_out().0.saturating_sub(it.source_in.0) as f64;
    let d = it.duration.0 as f64;
    let k = span / d;
    if d > 0.0 && k.is_finite() && k > 1e-6 { k } else { 1.0 }
}

/// Timeline time of media time `mt` of clip `it` (inside the clip).
fn timeline_of(it: &TrackItem, mt: Tick) -> Tick {
    let from = it.effect_time_at(it.start);
    let off = (mt.0.saturating_sub(from.0) as f64 / media_per_second(it)).clamp(0.0, it.duration.0.max(0) as f64);
    Tick(it.start.0.saturating_add(off as i64))
}

/// The least change of property `id` that is seen.
fn visible(id: &str, short_side: f64) -> f64 {
    match id {
        "opacity" => POP_OPACITY,
        "scale" | "scale_width" | "scale_height" => POP_SCALE,
        "rotation" => POP_DEGREES,
        _ => POP_POSITION_SHARE * short_side,
    }
}

fn components(v: &ParamValue) -> Option<Vec<f64>> {
    match v {
        ParamValue::Float(x) => Some(vec![*x]),
        ParamValue::Vec2(p) => Some(vec![p.x, p.y]),
        _ => None,
    }
}

/// The largest one-frame change of `param` between media times `a` and `b`, as a share of its
/// keyframed range, with the change and the two values; None when nothing visible moves.
fn jump(param: &Param, a: Tick, b: Tick, least: f64) -> Option<(f64, f64, Value, Value)> {
    let (va, vb) = (param.value_at(a), param.value_at(b));
    let (ca, cb) = (components(&va)?, components(&vb)?);
    let keys: Vec<Vec<f64>> = param.keyframes.iter().take(MAX_KEYFRAMES).filter_map(|k| components(&k.value)).collect();
    let mut worst: Option<(f64, f64)> = None;
    for (i, (x, y)) in ca.iter().zip(&cb).enumerate() {
        let vals: Vec<f64> = keys.iter().filter_map(|k| k.get(i).copied()).filter(|v| v.is_finite()).collect();
        let range = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max) - vals.iter().copied().fold(f64::INFINITY, f64::min);
        let step = (y - x).abs();
        if !(range.is_finite() && step.is_finite()) || range <= 1e-9 || step < least {
            continue;
        }
        let share = (step / range).min(1.0);
        if worst.is_none_or(|(s, _)| share > s) {
            worst = Some((share, step));
        }
    }
    let (share, step) = worst?;
    Some((share, step, crate::commands::param_value_to_json(&va), crate::commands::param_value_to_json(&vb)))
}

/// What every check reads: the sequence's rate, whether overshoot is allowed, the bands and the
/// frame's short side.
struct Ctx {
    fps: f64,
    playful: bool,
    bands: Bands,
    short_side: f64,
}

/// One clip's findings and moves, and what was read.
#[derive(Default)]
struct Read {
    findings: Vec<Value>,
    moves: Vec<Move>,
    /// Titles with a looping text animator.
    loops: Vec<u64>,
    animators: usize,
    keyframed: usize,
    springs: usize,
    curves: usize,
}

fn title_layer_name(e: &EffectInstance, i: usize) -> String {
    match e.params.get("text").map(|p| &p.value) {
        Some(ParamValue::Text(t)) if !t.trim().is_empty() => t.trim().chars().take(80).collect(),
        _ => format!("layer {}", i + 1),
    }
}

/// The text animators of title `it`: overshoot, stagger, loops, and their entrances and exits.
fn read_animators(project: &Project, it: &TrackItem, track: &str, c: &Ctx, out: &mut Read) {
    let end = it.end();
    for (li, e) in it.effects.iter().enumerate().filter(|(_, e)| e.effect == TEXT_LAYER && e.enabled) {
        let Some(anim) = e.layer.as_ref().and_then(|x| x.anim.as_ref()).map(|a| a.sanitized()) else { continue };
        let name = title_layer_name(e, li);
        let base = json!({"clip": it.id.0, "track": track, "layer": li, "text": name});
        let with = |extra: Value| {
            let mut v = base.clone();
            if let (Some(o), Value::Object(x)) = (v.as_object_mut(), extra) {
                o.extend(x);
            }
            v
        };
        let mut units: Option<usize> = None;
        let mut count = |a: &Animator| *units.get_or_insert_with(|| unit_count(project, it, li, a.unit));
        for (slot, a) in [("in", anim.enter.as_ref()), ("out", anim.out.as_ref())] {
            let Some(a) = a else { continue };
            out.animators += 1;
            if a.kind == AnimKind::Typewriter {
                // its units appear at once, at its typing speed: no ease, duration or stagger to judge
                continue;
            }
            let entering = slot == "in";
            let span = a.span(count(a));
            let (at, until) = if entering {
                (it.start, Tick(it.start.0.saturating_add(Tick::from_seconds_f64(span).0)).min(end))
            } else {
                (Tick(end.0.saturating_sub(Tick::from_seconds_f64(span).0)).max(it.start), end)
            };
            if !c.playful
                && let Some(family) = overshooting(a.easing, a.bezier)
            {
                out.findings.push(finding(
                    "overshoot",
                    at,
                    Some(until),
                    format!(
                        "the {} {} of \"{name}\" eases with {}, which {}; use Quart Out or Cubic Out, or declare a playful style (style: \"playful\", or autocut.taste motion: playful) if it is meant",
                        a.kind.label(),
                        if entering { "entrance" } else { "exit" },
                        easing_name(a.easing),
                        if family == "bounce" { "bounces before it settles" } else { "passes its mark and comes back" },
                    ),
                    with(json!({"source": "animator", "slot": slot, "kind": a.kind.name(), "easing": easing_name(a.easing), "family": family})),
                ));
            }
            out.moves.push(Move {
                clip: it.id.0,
                track: track.to_string(),
                layer: Some(li),
                source: "animator",
                kind: a.kind.name().to_string(),
                what: format!("{} {} of \"{name}\"", a.kind.label(), if entering { "entrance" } else { "exit" }),
                entering,
                frames: a.duration * c.fps,
                start: at,
                end: until,
            });
            let n = count(a);
            let spread = a.stagger * n.saturating_sub(1) as f64;
            if spread > c.bands.stagger + 1e-9 {
                let fits = c.bands.stagger / n.saturating_sub(1).max(1) as f64;
                let unit = match a.unit {
                    AnimUnit::Letter => "letters",
                    AnimUnit::Word => "words",
                    AnimUnit::Line => "lines",
                };
                let by_word = if a.unit == AnimUnit::Letter { "animate by word, or " } else { "" };
                out.findings.push(finding(
                    "longStagger",
                    at,
                    Some(until),
                    format!(
                        "the {n} {unit} of \"{name}\" take {spread:.2} s to all {} ({:.3} s apart); keep the whole stagger within {:.1} s: {by_word}set the stagger to {fits:.3} s or less",
                        if entering { "start arriving" } else { "start leaving" },
                        a.stagger,
                        c.bands.stagger,
                    ),
                    with(json!({"slot": slot, "kind": a.kind.name(), "unit": a.unit.name(), "units": n, "stagger": r3(a.stagger), "total": r3(spread), "limit": c.bands.stagger, "fits": r3(fits)})),
                ));
            }
        }
        if let (Some(i), Some(o)) = (&anim.enter, &anim.out)
            && i.kind != AnimKind::Typewriter
            && o.kind != AnimKind::Typewriter
            && o.duration > i.duration + 1e-9
        {
            let at = Tick(end.0.saturating_sub(Tick::from_seconds_f64(o.span(count(o))).0)).max(it.start);
            out.findings.push(finding(
                "exitOutlastsEntrance",
                at,
                Some(end),
                format!(
                    "the exit of \"{name}\" ({:.2} s a unit) takes longer than its entrance ({:.2} s); exits read best quicker than entrances: make it {:.2}–{:.2} s (60–75 % of the entrance)",
                    o.duration,
                    i.duration,
                    0.6 * i.duration,
                    0.75 * i.duration
                ),
                with(json!({"source": "animator", "entrance": r3(i.duration), "exit": r3(o.duration)})),
            ));
        }
        if let Some(l) = &anim.repeat {
            out.animators += 1;
            out.loops.push(it.id.0);
            if l.duration < AMBIENT_LOOP_SECONDS {
                let secs = it.duration.seconds();
                out.findings.push(finding(
                    "loopingText",
                    it.start,
                    Some(end),
                    format!(
                        "the {} loop keeps \"{name}\" moving for its whole {secs:.1} s (a cycle every {:.2} s); take the loop off and let the title settle after its entrance, or slow it to an ambient drift of {AMBIENT_LOOP_SECONDS} s a cycle or more",
                        l.kind.label(),
                        l.duration
                    ),
                    with(json!({"kind": l.kind.name(), "period": r3(l.duration), "length": r3(secs)})),
                ));
            }
        }
    }
}

/// The keyframes of clip `it`: overshooting springs and curves, pops at its edges and (for titles
/// and overlays) the keyframed moves that start or end it.
fn read_keyframes(it: &TrackItem, track: &str, role: Role, c: &Ctx, out: &mut Read) {
    let frame = Tick::from_seconds_f64(1.0 / c.fps).0.max(1);
    let (start, end) = (it.start.0, it.end().0);
    let first = it.effect_time_at(it.start);
    let last = it.effect_time_at(Tick(end.saturating_sub(frame).max(start)));
    let k = media_per_second(it);
    let media_frame = ((frame as f64) * k).round().max(1.0) as i64;
    // the clip's keyframed entrance and exit: the longest edge move, and what moved
    let (mut enter, mut leave): (Option<(i64, Vec<String>)>, Option<(i64, Vec<String>)>) = (None, None);
    for (ei, e) in it.effects.iter().enumerate().filter(|(_, e)| e.enabled) {
        let layer = filmcraft_project::graphic::is_layer(e).then_some(ei);
        for (id, param) in e.params.iter().filter(|(_, p)| p.is_animated()) {
            out.keyframed += 1;
            let keys = &param.keyframes;
            let base = json!({"clip": it.id.0, "track": track, "effect": e.effect, "param": id, "layer": layer});
            let with = |extra: Value| {
                let mut v = base.clone();
                if let (Some(o), Value::Object(x)) = (v.as_object_mut(), extra) {
                    o.extend(x);
                }
                v
            };
            // ---- eases that pass their mark
            for pair in keys.windows(2).take(MAX_KEYFRAMES) {
                let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else { continue };
                if a.value == b.value || b.time <= a.time {
                    continue;
                }
                let (at, until) = (timeline_of(it, a.time), timeline_of(it, b.time));
                match a.interp {
                    Interpolation::Spring => {
                        out.springs += 1;
                        let spring = a.segment().spring.unwrap_or_default().sanitized();
                        let over = spring_overshoot(spring);
                        if !c.playful && spring.damping_ratio < SPRING_DAMPING && over > 0.001 {
                            out.findings.push(finding(
                                "overshoot",
                                at,
                                Some(until),
                                format!(
                                    "the spring on {id} (damping ratio {:.2}) passes its mark by {:.1} % of the move and rings back; raise the damping ratio to {SPRING_DAMPING} or more (1 for none at all), or declare a playful style if it is meant",
                                    spring.damping_ratio,
                                    over * 100.0
                                ),
                                with(json!({"source": "spring", "dampingRatio": r3(spring.damping_ratio), "overshoot": r3(over)})),
                            ));
                        }
                    }
                    Interpolation::Custom => {
                        out.curves += 1;
                        let (over, back) = curve_shape(&a.segment().curve);
                        if !c.playful && (over > OVERSHOOT_EPS || back > BOUNCE_BACK) {
                            let what = if over > OVERSHOOT_EPS {
                                format!("goes {:.0} % past its mark", over * 100.0)
                            } else {
                                format!("turns back by {:.0} % of the move (a bounce)", back * 100.0)
                            };
                            out.findings.push(finding(
                                "overshoot",
                                at,
                                Some(until),
                                format!(
                                    "the custom curve on {id} {what} before it settles; keep its progress between 0 and 1 and rising (power2.out or expo.out in keyframes.choreograph), or declare a playful style if it is meant"
                                ),
                                with(json!({"source": "curve", "overshoot": r3(over), "back": r3(back)})),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            if !MOVE_PROPS.contains(&id.as_str()) {
                continue;
            }
            let least = visible(id, c.short_side);
            // ---- pops at the clip's edges
            let edges = [
                (first, Tick(first.0.saturating_add(media_frame)), it.start, "first"),
                (Tick(last.0.saturating_sub(media_frame)), last, Tick(end.saturating_sub(frame)), "last"),
            ];
            if end.saturating_sub(start) > frame.saturating_mul(2) {
                for (a, b, at, edge) in edges {
                    let Some((share, step, va, vb)) = jump(param, a, b, least) else { continue };
                    if share >= EDGE_POP_SHARE {
                        out.findings.push(finding(
                            "edgePop",
                            at,
                            Some(Tick(at.0.saturating_add(frame))),
                            format!(
                                "{id} jumps {:.0} % of its move between the clip's {edge} two frames; {}",
                                share * 100.0,
                                if edge == "first" {
                                    "start the move a few frames before the clip, or ease it over 8 frames or more"
                                } else {
                                    "end the move a few frames before the clip ends, or ease it over 8 frames or more"
                                }
                            ),
                            with(json!({"edge": edge, "share": r3(share), "step": r3(step), "from": va, "to": vb})),
                        ));
                    }
                }
            }
            // ---- keyframed entrances and exits of titles and overlays
            if role == Role::Other || keys.len() < 2 {
                continue;
            }
            let near = |t: Tick, edge: Tick| t.0.saturating_sub(edge.0).abs() <= media_frame;
            if let (Some(a), Some(b)) = (keys.first(), keys.get(1))
                && near(a.time, first)
                && a.value != b.value
            {
                let span = b.time.0.saturating_sub(a.time.0);
                let e = enter.get_or_insert((0, Vec::new()));
                e.0 = e.0.max(span);
                e.1.push(id.clone());
            }
            if let (Some(b), Some(a)) = (keys.last(), keys.len().checked_sub(2).and_then(|i| keys.get(i)))
                && near(b.time, last)
                && a.value != b.value
            {
                let span = b.time.0.saturating_sub(a.time.0);
                let e = leave.get_or_insert((0, Vec::new()));
                e.0 = e.0.max(span);
                e.1.push(id.clone());
            }
        }
    }
    for (entering, m) in [(true, enter), (false, leave)] {
        let Some((span, mut props)) = m else { continue };
        props.sort();
        props.dedup();
        let secs = span as f64 / filmcraft_time::TICKS_PER_SECOND as f64 / k;
        let (at, until) = if entering {
            (it.start, Tick(it.start.0.saturating_add(Tick::from_seconds_f64(secs).0)))
        } else {
            (Tick(end.saturating_sub(Tick::from_seconds_f64(secs).0)), it.end())
        };
        out.moves.push(Move {
            clip: it.id.0,
            track: track.to_string(),
            layer: None,
            source: "keyframes",
            kind: format!("keyframes ({})", props.join(", ")),
            what: format!("keyframed {} ({})", if entering { "entrance" } else { "exit" }, props.join(", ")),
            entering,
            frames: secs * c.fps,
            start: at,
            end: until,
        });
    }
}

/// A transition's easing, read from its parameters.
fn transition_easing(e: &EffectInstance) -> (u32, Option<[f64; 4]>) {
    let num = |k: &str, d: f64| e.params.get(k).and_then(|p| p.value.as_f64()).filter(|v| v.is_finite()).unwrap_or(d);
    let easing = match e.params.get("easing").map(|p| &p.value) {
        Some(ParamValue::Choice(n)) => *n,
        Some(v) => v.as_f64().filter(|x| x.is_finite() && *x >= 0.0 && *x < 64.0).map_or(0, |x| x as u32),
        None => 0,
    };
    let b = Bezier::EASE;
    (easing, Some([num("bezier_x1", b.x1), num("bezier_y1", b.y1), num("bezier_x2", b.x2), num("bezier_y2", b.y2)]))
}

/// The transitions of one video track: overshoot, and the entrances and exits of its titles and
/// overlays.
fn read_transitions(q: &Sequence, ti: usize, roles: &BTreeMap<u64, Role>, c: &Ctx, out: &mut Read) -> usize {
    let Some(track) = q.video_tracks.get(ti) else { return 0 };
    let name = format!("V{}", ti + 1);
    let mut n = 0;
    for tr in track.transitions.iter().take(MAX_CLIPS) {
        n += 1;
        let id = filmcraft_project::creator_transitions::base_id(&tr.effect.effect).to_string();
        let label = filmcraft_project::find_effect(&tr.effect.effect).map_or_else(|| tr.effect.effect.clone(), |d| d.name.to_string());
        let base = json!({"transition": tr.id.0, "track": name, "effect": tr.effect.effect, "from": tr.from.map(|c| c.0), "to": tr.to.map(|c| c.0)});
        let with = |extra: Value| {
            let mut v = base.clone();
            if let (Some(o), Value::Object(x)) = (v.as_object_mut(), extra) {
                o.extend(x);
            }
            v
        };
        if !c.playful {
            let (easing, bezier) = transition_easing(&tr.effect);
            let param = |k: &str| tr.effect.params.get(k).and_then(|p| p.value.as_f64()).filter(|v| v.is_finite()).unwrap_or(0.0);
            let built_in = match id.as_str() {
                "pop_motion" if param("overshoot") > 0.0 => Some(("Overshoot", param("overshoot"))),
                "spring_motion" if param("bounce") > 0.0 => Some(("Bounce", param("bounce"))),
                _ => None,
            };
            if let Some(family) = overshooting(easing, bezier) {
                out.findings.push(finding(
                    "overshoot",
                    tr.start,
                    Some(tr.end()),
                    format!(
                        "the {label} transition eases with {}, which {}; use Quart Out, Cubic Out or Sine In-Out, or declare a playful style if it is meant",
                        easing_name(easing),
                        if family == "bounce" { "bounces at its end" } else { "passes its mark (clamped, it arrives early and stops dead)" }
                    ),
                    with(json!({"source": "transition", "easing": easing_name(easing), "family": family})),
                ));
            } else if let Some((what, amount)) = built_in {
                out.findings.push(finding(
                    "overshoot",
                    tr.start,
                    Some(tr.end()),
                    format!(
                        "the {label} transition overshoots by design ({what} {amount:.0} %); set its {what} to 0, choose a push or a dissolve, or declare a playful style if it is meant"
                    ),
                    with(json!({"source": "transition", "family": "builtIn", "amount": r3(amount)})),
                ));
            }
        }
        let frames = tr.duration.seconds() * c.fps;
        for (clip, entering) in [(tr.to, true), (tr.from, false)] {
            let Some(clip) = clip else { continue };
            if roles.get(&clip.0).is_none_or(|r| *r == Role::Other) {
                continue;
            }
            out.moves.push(Move {
                clip: clip.0,
                track: name.clone(),
                layer: None,
                source: "transition",
                kind: id.clone(),
                what: format!("{label} transition {}", if entering { "into the clip" } else { "out of the clip" }),
                entering,
                frames,
                start: tr.start,
                end: tr.end(),
            });
        }
    }
    n
}

/// A number of frames as it reads: whole when it is.
fn frames_text(f: f64) -> String {
    if (f - 1.0).abs() < 0.05 {
        "1 frame".into()
    } else if (f - f.round()).abs() < 0.05 {
        format!("{:.0} frames", f.round())
    } else {
        format!("{f:.1} frames")
    }
}

/// The bands, and an exit against the entrance it mirrors.
fn judge_moves(moves: &[Move], c: &Ctx, findings: &mut Vec<Value>) {
    let fps = c.fps;
    for m in moves {
        let (code, band) = if m.entering { ("entranceDuration", c.bands.entrance) } else { ("exitDuration", c.bands.exit) };
        let (lo, hi) = band;
        let secs = m.frames / fps;
        let advice = if m.frames < lo - 1e-6 {
            Some(format!("lengthen it to at least {lo:.0} frames ({:.2} s) so it reads as a move, not a cut", lo / fps))
        } else if m.frames > hi + 1e-6 {
            Some(format!("shorten it to {hi:.0} frames ({:.2} s) or less so it does not keep the viewer waiting", hi / fps))
        } else {
            None
        };
        if let Some(advice) = advice {
            findings.push(finding(
                code,
                m.start,
                Some(m.end),
                format!("the {} takes {} ({secs:.2} s); {advice}", m.what, frames_text(m.frames)),
                json!({"clip": m.clip, "track": m.track, "layer": m.layer, "source": m.source, "kind": m.kind, "frames": r3(m.frames), "seconds": r3(secs), "band": [r3(lo), r3(hi)], "short": m.frames < lo}),
            ));
        }
    }
    // the clip's two transitions, and its edge keyframes (the animators were compared per layer)
    let mut pairs: BTreeMap<(u64, &str), (Option<&Move>, Option<&Move>)> = BTreeMap::new();
    for m in moves.iter().filter(|m| m.source != "animator") {
        let e = pairs.entry((m.clip, m.source)).or_default();
        let slot = if m.entering { &mut e.0 } else { &mut e.1 };
        if slot.is_none_or(|x| m.frames > x.frames) {
            *slot = Some(m);
        }
    }
    for ((clip, source), (i, o)) in pairs {
        let (Some(i), Some(o)) = (i, o) else { continue };
        if o.frames > i.frames + 1e-6 {
            findings.push(finding(
                "exitOutlastsEntrance",
                o.start,
                Some(o.end),
                format!(
                    "the {} ({}) takes longer than the entrance ({}); exits read best quicker than entrances: make it {:.0}–{:.0} frames (60–75 % of the entrance)",
                    o.what,
                    frames_text(o.frames),
                    frames_text(i.frames),
                    0.6 * i.frames,
                    0.75 * i.frames
                ),
                json!({"clip": clip, "track": o.track, "source": source, "entrance": r3(i.frames), "exit": r3(o.frames)}),
            ));
        }
    }
}

/// Every clip of a sequence's visible video tracks read.
struct Sequenced {
    read: Read,
    roles: BTreeMap<u64, Role>,
    clips: usize,
    transitions: usize,
    truncated: bool,
}

fn read_sequence(project: &Project, q: &Sequence, c: &Ctx) -> Sequenced {
    let mut read = Read::default();
    let mut roles: BTreeMap<u64, Role> = BTreeMap::new();
    let (mut clips, mut transitions, mut truncated) = (0usize, 0usize, false);
    for (ti, track) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        let name = format!("V{}", ti + 1);
        for it in track.items.iter().filter(|i| i.enabled) {
            if clips >= MAX_CLIPS {
                truncated = true;
                break;
            }
            clips += 1;
            let r = role(project, it);
            roles.insert(it.id.0, r);
            if r == Role::Title {
                read_animators(project, it, &name, c, &mut read);
            }
            read_keyframes(it, &name, r, c, &mut read);
        }
        transitions += read_transitions(q, ti, &roles, c, &mut read);
    }
    Sequenced { read, roles, clips, transitions, truncated }
}

/// What moves in a sequence, for the motion filmstrips (`sequence.motionStrips`).
#[derive(Clone, Debug)]
pub(crate) struct Window {
    pub clip: u64,
    /// `title` or `overlay`.
    pub role: &'static str,
    /// `in` or `out` for a title's entrance or exit; None for a looping title's or an overlay's
    /// whole clip.
    pub edge: Option<&'static str>,
    pub start: Tick,
    pub end: Tick,
    pub what: String,
}

/// The windows of `q` that move: each title's entrance and exit (its text animators and edge
/// keyframes together; its transitions are items of their own), a looping title's whole clip and
/// each overlay's whole clip, in time order.
pub(crate) fn windows(project: &Project, q: &Sequence) -> Vec<Window> {
    let fps = q.settings.frame_rate.sane().as_f64();
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let k = fps / 30.0;
    let bands =
        Bands { entrance: (ENTRANCE_FRAMES_30.0 * k, ENTRANCE_FRAMES_30.1 * k), exit: (EXIT_FRAMES_30.0 * k, EXIT_FRAMES_30.1 * k), stagger: STAGGER_SECONDS };
    let c = Ctx { fps, playful: true, bands, short_side: f64::from(q.settings.width.min(q.settings.height).max(1)) };
    let seen = read_sequence(project, q, &c);
    let mut out = Vec::new();
    for (clip, r) in &seen.roles {
        let Some((_, it)) = q.find_item(filmcraft_project::ClipId(*clip)) else { continue };
        let name = || project.item(it.item).map_or_else(String::new, |p| p.name.chars().take(80).collect::<String>());
        match r {
            Role::Title => {
                let mut any = false;
                for (entering, edge) in [(true, "in"), (false, "out")] {
                    let mine: Vec<&Move> = seen.read.moves.iter().filter(|m| m.clip == *clip && m.entering == entering && m.source != "transition").collect();
                    let (Some(start), Some(end)) = (mine.iter().map(|m| m.start).min(), mine.iter().map(|m| m.end).max()) else { continue };
                    let what = mine.iter().map(|m| m.what.as_str()).next().unwrap_or("title").to_string();
                    out.push(Window { clip: *clip, role: "title", edge: Some(edge), start, end: end.max(start), what });
                    any = true;
                }
                if !any && seen.read.loops.contains(clip) {
                    out.push(Window { clip: *clip, role: "title", edge: None, start: it.start, end: it.end(), what: format!("looping title {}", name()) });
                }
            }
            Role::Overlay => out.push(Window { clip: *clip, role: "overlay", edge: None, start: it.start, end: it.end(), what: format!("overlay {}", name()) }),
            Role::Other => {}
        }
    }
    out.sort_by_key(|w| (w.start, w.end));
    out
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(o) = p.as_object()
        && let Some(k) = o.keys().find(|k| !matches!(k.as_str(), "sequence" | "style" | "bands"))
    {
        return Err(bad(CMD, format!("unknown parameter `{}`: sequence, style or bands", k.chars().take(64).collect::<String>())));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let (playful, said) = style_p(s, p)?;
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    if q.duration().0 <= 0 {
        return Err(bad(CMD, "the sequence is empty"));
    }
    let fps = q.settings.frame_rate.sane().as_f64();
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let bands = bands_p(p, fps)?;
    let c = Ctx { fps, playful, bands, short_side: f64::from(q.settings.width.min(q.settings.height).max(1)) };
    let Sequenced { mut read, roles, clips, transitions, truncated } = read_sequence(&project, q, &c);
    // the titles in time order, with how each enters
    let mut titles: Vec<(Tick, Tick, u64, String)> = Vec::new();
    let mut findings = std::mem::take(&mut read.findings);
    judge_moves(&read.moves, &c, &mut findings);
    // ---- every title entering the same way
    for (id, r) in &roles {
        if *r != Role::Title {
            continue;
        }
        let Some((_, it)) = q.find_item(filmcraft_project::ClipId(*id)) else { continue };
        let mine: Vec<&Move> = read.moves.iter().filter(|m| m.clip == *id && m.entering).collect();
        // what a viewer sees first: the text's own animator, else a transition, else keyframes
        let how = ["animator", "transition", "keyframes"]
            .iter()
            .find_map(|src| mine.iter().find(|m| m.source == *src))
            .map_or_else(|| "none".to_string(), |m| m.kind.clone());
        titles.push((it.start, it.end(), *id, how));
    }
    titles.sort();
    let kinds: std::collections::BTreeSet<&str> = titles.iter().map(|t| t.3.as_str()).collect();
    let same = titles.len() >= SAME_ENTRANCE_TITLES && kinds.len() == 1 && !kinds.contains("none");
    if same && let (Some(f), Some(l)) = (titles.first(), titles.last()) {
        findings.push(finding(
            "sameEntrance",
            f.0,
            Some(l.1),
            format!("all {} titles enter the same way ({}); vary it: give a title a different entrance, or let a quiet one simply appear", titles.len(), f.3),
            json!({"kind": f.3, "titles": titles.len(), "clips": titles.iter().map(|t| t.2).collect::<Vec<_>>()}),
        ));
    }
    let (findings, counts) = bounded(findings);
    let mut skipped = Vec::new();
    if playful {
        skipped.push(json!({"check": "overshoot", "reason": format!("a playful style is declared ({})", if said == "taste" { "autocut.taste motion: playful" } else { "style: playful" })}));
    }
    if truncated {
        skipped.push(json!({"check": "all", "reason": format!("only the first {MAX_CLIPS} clips were read")}));
    }
    let band = |b: (f64, f64)| json!({"frames": [r3(b.0), r3(b.1)], "seconds": [r3(b.0 / fps), r3(b.1 / fps)]});
    Ok(json!({
        "sequence": seq.0,
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "skipped": skipped,
        "style": {"playful": playful, "from": said},
        "checked": {
            "clips": clips,
            "titles": titles.len(),
            "overlays": roles.values().filter(|r| **r == Role::Overlay).count(),
            "animators": read.animators,
            "transitions": transitions,
            "keyframedParams": read.keyframed,
            "springSegments": read.springs,
            "curveSegments": read.curves,
            "entrances": read.moves.iter().filter(|m| m.entering).count(),
            "exits": read.moves.iter().filter(|m| !m.entering).count(),
            "titleEntrances": titles.iter().map(|t| json!({"clip": t.2, "kind": t.3})).collect::<Vec<_>>(),
        },
        "fps": r3(fps),
        "bands": {"entrance": band(bands.entrance), "exit": band(bands.exit), "staggerSeconds": bands.stagger},
        "thresholds": {
            "entranceFrames30": [ENTRANCE_FRAMES_30.0, ENTRANCE_FRAMES_30.1], "exitFrames30": [EXIT_FRAMES_30.0, EXIT_FRAMES_30.1],
            "staggerSeconds": STAGGER_SECONDS, "springDamping": SPRING_DAMPING, "sameEntranceTitles": SAME_ENTRANCE_TITLES,
            "ambientLoopSeconds": AMBIENT_LOOP_SECONDS, "edgePopShare": EDGE_POP_SHARE,
            "overshoot": OVERSHOOT_EPS, "bounceBack": BOUNCE_BACK,
            "note": "starting values, to be tuned on real projects",
        },
        "method": {
            "timeline": "read from the timeline alone, nothing rendered; clips on enabled video tracks, not inside nested sequences",
            "titles": "a graphic clip with an enabled text layer; its entrance is its text animator's, else a transition at its start, else keyframes from its first frame",
            "overlays": "a clip of a picture with alpha (motion.render, composition.import): its transitions and keyframes",
            "units": "letters as shaped clusters, words and lines of the laid-out text at the clip's middle, as the renderer animates them",
            "edgePop": "position, scale, rotation and opacity between the clip's first two and last two frames, against the property's keyframed range",
        },
    }))
}

#[cfg(test)]
#[path = "motion_check_tests.rs"]
mod tests;
