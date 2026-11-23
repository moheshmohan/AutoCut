//! `autocut.fingerprint` (PX10): a finished video's six axes, read from its timeline alone
//! (nothing is rendered), compared with the newest videos in the taste history
//! ([`crate::taste_history`]) so consecutive videos can differ.
//!
//! | axis | ids | read from |
//! |---|---|---|
//! | `opening` | `titleCard`, `talking`, `montage`, `still`, `coldOpen` | the first 5 s: a title on screen within the first second; else speech (a transcript's words) in the first 2 s; else three or more cuts in the first 5 s; else a still picture (or a frame hold) first; else footage from the first frame |
//! | `titleEntrance` | a `graphics.animateText` in-kind (`fade`, `pop`, `slideUp`, …) or `none` | each title's entrance: its text animator, else a transition into it (a dissolve is `fade`, another a family id), else its keyframes (opacity rising: `fade`; a position move: the slide it makes; scale rising: `scaleUp`); the most common one (the first title's on a tie) |
//! | `transitionFamily` | `dissolve`, `whip`, `zoom`, `light`, `slide`, `wipe`, `spatial`, `glitch`, `distortion`, `pattern`, `flow`, `other` or `cut` | the transitions between pictures (not into or out of nothing, not on titles), each by its catalog folder ([`family_of`]); the most common one |
//! | `colorLook` | our look (`punchy`, `warm`, …), a creator look (`tech`, `vhs`), a Lumetri preset or look (`steelBlue`, `bleachBypass`) or `none` | the pictures' Lumetri Color (the colour pass's own look, a preset matched by its parameters, a Creative look) and `Look: …` adjustment clips, by how long each is on screen |
//! | `captionStyle` | a caption preset (`bold`, `clean`, `twoLine`, …), `plain` or `none` | the caption tracks with captions, by how many captions each holds |
//! | `close` | `logo`, `endCard`, `title`, `fadeOut` or `cut` | the last 1.5 s: a logo (an overlay named so, or a still picture with alpha above the footage); a title over nothing, a graphic or blurred footage; a title over footage; a fade to nothing (a transition out of the last picture, or its opacity falling to nothing); else the last frame is cut |
//!
//! `matches` lists, for each of the three newest records, the axes they share with this video;
//! four or more matching one of them is a warning (decision D7: warnings, never a block).
//! `record: true` adds the fingerprint to the history — the project's file name only, the
//! `pipeline` given, and the deal choices used (`deals`, with those `autocut.deal` kept in the
//! session since the last record, which are then cleared). It writes the history file in the data directory; it is not
//! an undo step, and the result says so.

use std::collections::BTreeMap;

use filmcraft_project::graphic::{SHAPE_LAYER, TEXT_LAYER};
use filmcraft_project::{EffectInstance, ItemId, ItemKind, ParamValue, Project, Sequence, TrackItem, Transition};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::taste_history::{self, AXES, Axes, Record};
use crate::{EngineError, Result, Session};

pub const CMD: &str = "autocut.fingerprint";
/// How much of the start is the opening, and of the end the close, seconds.
const OPENING: f64 = 5.0;
const CLOSE: f64 = 1.5;
/// A title on screen this soon makes the opening a title card, seconds.
const TITLE_START: f64 = 1.0;
/// Speech this soon makes it a talking opening, seconds.
const TALK_START: f64 = 2.0;
/// This many cuts in the opening make it a montage.
const MONTAGE_CUTS: usize = 3;
/// Newest records compared, and how many axes matching one of them warn.
pub const RECENT: usize = 3;
pub const WARN_AXES: usize = 4;
/// Most clips and transcript words read.
const MAX_CLIPS: usize = 20_000;
const MAX_WORDS: usize = 200_000;
/// Opacity (%) under which a picture is gone.
const GONE: f64 = 10.0;
/// Longest pipeline name and most deal choices a record takes.
const MAX_PIPELINE: usize = 32;
const MAX_DEALS: usize = 16;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Video Fingerprint",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (default the active one),"record":bool=false (add it to the taste history in the data directory: not an undo step),"pipeline":str? (with record: what made it — direct, quick, showcase, …),"deals":{"look"|"titleStyle"|"transitionFamily"|"music"|"register"|"endCard":id}? (with record: the deal choices used; those autocut.deal kept since the last record are added)}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

fn secs(x: f64) -> Tick {
    Tick::from_seconds_f64(x)
}

fn add(a: Tick, b: Tick) -> Tick {
    Tick(a.0.saturating_add(b.0))
}

fn sub(a: Tick, b: Tick) -> Tick {
    Tick(a.0.saturating_sub(b.0))
}

fn r2(x: f64) -> f64 {
    if x.is_finite() { (x * 100.0).round() / 100.0 } else { 0.0 }
}

/// `"Teal & Orange"` → `tealOrange`.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for (i, w) in name.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).take(8).enumerate() {
        let mut cs = w.chars();
        if i == 0 {
            out.extend(cs.flat_map(char::to_lowercase));
        } else {
            out.extend(cs.next().into_iter().flat_map(char::to_uppercase));
            out.extend(cs.flat_map(char::to_lowercase));
        }
    }
    out.chars().take(48).collect()
}

/// The transition families, in catalog order (`cut` is a film without transitions).
pub const FAMILIES: &[&str] = &["dissolve", "whip", "zoom", "light", "slide", "wipe", "spatial", "glitch", "distortion", "pattern", "flow"];

/// The family of a transition effect id: by its own name where that says (whips, zooms), else
/// by its catalog folder (a Creator preset by the transition it is made from),
/// else by the words in its id; `other` when nothing says.
pub fn family_of(id: &str) -> &'static str {
    let base = filmcraft_project::creator_transitions::base_id(id);
    let has = |words: &[&str]| words.iter().any(|w| id.contains(w) || base.contains(w));
    if has(&["whip", "swipe"]) {
        return "whip";
    }
    // (the Cross Zoom and Zoom Blur sit in Lights & Blurs, the punch-ins in Zooms & Camera)
    if has(&["zoom", "punch", "pull_in", "pull_out"]) {
        return "zoom";
    }
    let folder = filmcraft_project::effect::find_effect(base).and_then(|d| d.category.get(1).copied()).unwrap_or_default();
    match folder {
        "Dissolve" | "Blend Dissolves" => return "dissolve",
        "Zooms & Camera" | "Zoom" => return if has(&["spin", "tilt", "flip"]) { "spatial" } else { "zoom" },
        "Light & Film" | "Lights & Blurs" => return "light",
        "Wipe" | "Luma Wipes" | "Iris" | "Page Peel" => return "wipe",
        "Slide" => return "slide",
        "3D & Spatial" | "3D Motion" | "Immersive Video" => return "spatial",
        "Glitch & Datamosh" | "Grunge & Distort" => return "glitch",
        "Distortion" => return "distortion",
        "Particles & Patterns" => return "pattern",
        "Flow & Morph" => return "flow",
        _ => {}
    }
    if has(&["dissolve", "dip", "fade", "mix", "crossfade"]) {
        "dissolve"
    } else if has(&["zoom", "punch", "pull"]) {
        "zoom"
    } else if has(&["leak", "flash", "flare", "burn", "light", "bokeh", "glow", "blur"]) {
        "light"
    } else if has(&["glitch", "rgb", "datamosh", "static", "pixel"]) {
        "glitch"
    } else if has(&["push", "slide"]) {
        "slide"
    } else if has(&["wipe", "iris", "clock", "slice"]) {
        "wipe"
    } else if has(&["cube", "flip", "spin", "door", "3d"]) {
        "spatial"
    } else if has(&["ripple", "swirl", "warp", "melt"]) {
        "distortion"
    } else if has(&["morph", "flow"]) {
        "flow"
    } else {
        "other"
    }
}

/// A clip on a video track as these reads see it.
#[derive(Clone, Debug)]
struct Clip<'a> {
    track: usize,
    it: &'a TrackItem,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Footage, a still or a nested sequence.
    Picture,
    /// A graphic with a text layer.
    Title,
    /// A graphic of shapes only.
    Shape,
    /// An adjustment layer.
    Adjustment,
}

fn kind_of(project: &Project, it: &TrackItem) -> Option<Kind> {
    match project.item(it.item).map(|p| &p.kind) {
        Some(ItemKind::Graphic { .. }) if it.effects.iter().any(|e| e.effect == TEXT_LAYER && e.enabled) => Some(Kind::Title),
        Some(ItemKind::Graphic { .. }) => it.effects.iter().any(|e| e.effect == SHAPE_LAYER && e.enabled).then_some(Kind::Shape),
        Some(ItemKind::AdjustmentLayer { .. }) => Some(Kind::Adjustment),
        Some(ItemKind::Sequence(_)) => Some(Kind::Picture),
        Some(_) => project.resolve_media(it.item).and_then(|(_, m, _)| m.info.video.as_ref()).map(|_| Kind::Picture),
        None => None,
    }
}

fn title_text(it: &TrackItem) -> String {
    let t = it
        .effects
        .iter()
        .filter(|e| e.effect == TEXT_LAYER && e.enabled)
        .find_map(|e| match e.params.get("text").map(|p| &p.value) {
            Some(ParamValue::Text(t)) if !t.trim().is_empty() => Some(t.trim().to_string()),
            _ => None,
        })
        .unwrap_or_else(|| it.name.clone());
    let one: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > 40 { format!("{}…", one.chars().take(40).collect::<String>()) } else { one }
}

/// What the axes read: every enabled clip on an enabled video track, and the transitions.
struct Timeline<'a> {
    project: &'a Project,
    q: &'a Sequence,
    clips: Vec<Clip<'a>>,
    transitions: Vec<(usize, &'a Transition)>,
    end: Tick,
}

impl<'a> Timeline<'a> {
    fn read(project: &'a Project, q: &'a Sequence) -> Timeline<'a> {
        let mut clips = Vec::new();
        let mut transitions = Vec::new();
        for (ti, t) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
            for it in t.items.iter().filter(|i| i.enabled && i.duration.0 > 0) {
                if clips.len() >= MAX_CLIPS {
                    break;
                }
                if let Some(kind) = kind_of(project, it) {
                    clips.push(Clip { track: ti, it, kind });
                }
            }
            transitions.extend(t.transitions.iter().take(MAX_CLIPS).map(|x| (ti, x)));
        }
        Timeline { project, q, clips, transitions, end: q.duration() }
    }

    fn at(&self, t: Tick) -> impl Iterator<Item = &Clip<'a>> {
        self.clips.iter().filter(move |c| c.it.start <= t && t < c.it.end())
    }

    fn overlapping(&self, from: Tick, to: Tick) -> impl Iterator<Item = &Clip<'a>> {
        self.clips.iter().filter(move |c| c.it.start < to && c.it.end() > from)
    }

    fn kind(&self, id: Option<filmcraft_project::ClipId>) -> Option<Kind> {
        let id = id?;
        self.clips.iter().find(|c| c.it.id == id).map(|c| c.kind)
    }

    /// The lowest picture at `t`.
    fn picture_at(&self, t: Tick) -> Option<&Clip<'a>> {
        self.at(t).filter(|c| c.kind == Kind::Picture).min_by_key(|c| c.track)
    }
}

/// An axis: its id and what it came from.
type Axis = (String, String);

fn axis(id: &str, evidence: impl Into<String>) -> Axis {
    (id.to_string(), evidence.into())
}

/// Whether clip `it`'s media (or its linked audio's) holds speech in timeline span `from..to`.
fn speaks(project: &Project, it: &TrackItem, from: Tick, to: Tick) -> bool {
    let from = from.max(it.start);
    let to = to.min(it.end());
    if from >= to {
        return false;
    }
    let media = project.resolve_media(it.item).map_or(it.item, |(id, _, _)| id);
    let Some(tr) = project.transcripts.get(&media).or_else(|| project.transcripts.get(&it.item)) else { return false };
    let (a, b) = (it.source_time_at(from), it.source_time_at(Tick(to.0.saturating_sub(1))));
    let (a, b) = (a.min(b), a.max(b));
    tr.words.iter().take(MAX_WORDS).any(|w| w.start < b && w.end > a && !w.text.trim().is_empty())
}

fn opening(tl: &Timeline) -> Axis {
    let open_end = secs(OPENING).min(tl.end);
    if let Some(t) = tl.clips.iter().filter(|c| c.kind == Kind::Title && c.it.start <= secs(TITLE_START)).min_by_key(|c| c.it.start) {
        let at = add(t.it.start, Tick(t.it.duration.0 / 2)).min(add(t.it.start, secs(0.5)));
        let over = match tl.at(at).filter(|c| c.kind == Kind::Picture && c.track < t.track).min_by_key(|c| c.track) {
            Some(_) => "over footage",
            None => "on its own",
        };
        return axis("titleCard", format!("the title \"{}\" is on screen from {} s, {over}", title_text(t.it), r2(t.it.start.seconds())));
    }
    let talk_end = secs(TALK_START).min(tl.end);
    let audio = tl.q.audio_tracks.iter().filter(|t| t.enabled && !t.muted).flat_map(|t| t.items.iter().filter(|i| i.enabled)).take(MAX_CLIPS);
    let pictures = tl.clips.iter().filter(|c| c.kind == Kind::Picture).map(|c| c.it);
    if let Some(it) = pictures.chain(audio).find(|it| it.start < talk_end && speaks(tl.project, it, Tick::ZERO, talk_end)) {
        return axis("talking", format!("\"{}\" speaks within the first {TALK_START} s", it.name.chars().take(40).collect::<String>()));
    }
    let mut starts: Vec<Tick> =
        tl.clips.iter().filter(|c| c.kind == Kind::Picture && c.it.start > Tick::ZERO && c.it.start < open_end).map(|c| c.it.start).collect();
    starts.sort_unstable();
    starts.dedup();
    if starts.len() >= MONTAGE_CUTS {
        return axis("montage", format!("{} cuts in the first {} s", starts.len(), r2(open_end.seconds())));
    }
    let Some(first) = tl.clips.iter().filter(|c| c.kind == Kind::Picture).min_by_key(|c| (c.it.start, c.track)) else {
        return axis("coldOpen", "no pictures: the film opens on nothing");
    };
    let still =
        first.it.frame_hold.is_some() || tl.project.resolve_media(first.it.item).is_some_and(|(_, m, _)| m.info.kind == filmcraft_media::MediaKind::Still);
    if still {
        return axis("still", format!("it opens on a still picture, \"{}\"", first.it.name.chars().take(40).collect::<String>()));
    }
    axis(
        "coldOpen",
        format!(
            "footage from {} s, \"{}\", with no title, speech or montage first",
            r2(first.it.start.seconds()),
            first.it.name.chars().take(40).collect::<String>()
        ),
    )
}

/// How title `c` enters: its kind and what it came from.
fn entrance(tl: &Timeline, c: &Clip) -> (String, &'static str) {
    for e in c.it.effects.iter().filter(|e| e.effect == TEXT_LAYER && e.enabled) {
        if let Some(a) = e.layer.as_ref().and_then(|x| x.anim.as_ref()).and_then(|a| a.enter.as_ref()) {
            return (a.kind.name().to_string(), "its text animator");
        }
    }
    let track = tl.q.video_tracks.get(c.track);
    if let Some(x) = track.into_iter().flat_map(|t| t.transitions.iter()).find(|x| x.to == Some(c.it.id) && x.from.is_none()) {
        let f = family_of(&x.effect.effect);
        return (if f == "dissolve" { "fade".to_string() } else { f.to_string() }, "a transition into it");
    }
    if let Some(k) = keyframed_entrance(c.it) {
        return (k.to_string(), "its keyframes");
    }
    ("none".to_string(), "nothing: it appears at once")
}

/// The entrance clip `it`'s own keyframes make at its start: opacity rising from nothing
/// (`fade`), a position move (the slide it makes), scale rising (`scaleUp`).
fn keyframed_entrance(it: &TrackItem) -> Option<&'static str> {
    let early = |e: &EffectInstance, id: &str| -> Option<(ParamValue, ParamValue)> {
        let p = e.params.get(id)?;
        let first = p.keyframes.first()?;
        let second = p.keyframes.get(1)?;
        let from = it.effect_time_at(it.start);
        // the move starts at the clip's start (within a fifth of a second)
        (first.time.0.saturating_sub(from.0).abs() <= TICKS_PER_SECOND / 5).then(|| (first.value.clone(), second.value.clone()))
    };
    if let Some(m) = it.effect("motion")
        && let Some((ParamValue::Vec2(a), ParamValue::Vec2(b))) = early(m, "position")
    {
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        if dx.is_finite() && dy.is_finite() && (dx.abs() > 1.0 || dy.abs() > 1.0) {
            // the text travels from `a` to `b`: from below it moves up
            return Some(if dy.abs() >= dx.abs() {
                if dy < 0.0 { "slideUp" } else { "slideDown" }
            } else if dx > 0.0 {
                "slideLeft"
            } else {
                "slideRight"
            });
        }
    }
    if let Some(o) = it.effect("opacity")
        && let Some((ParamValue::Float(a), ParamValue::Float(b))) = early(o, "opacity")
        && a < GONE
        && b > a
    {
        return Some("fade");
    }
    if let Some(m) = it.effect("motion")
        && let Some((ParamValue::Float(a), ParamValue::Float(b))) = early(m, "scale")
        && b > a + 1.0
    {
        return Some("scaleUp");
    }
    None
}

/// The most common id (the first seen on a tie), weighted.
fn dominant(votes: &[(String, f64)]) -> Option<(String, f64, f64)> {
    let mut sum: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    let mut total = 0.0;
    for (i, (id, w)) in votes.iter().enumerate() {
        let w = if w.is_finite() && *w > 0.0 { *w } else { 0.0 };
        total += w;
        let e = sum.entry(id.as_str()).or_insert((0.0, i));
        e.0 += w;
    }
    let best = sum.iter().max_by(|a, b| a.1.0.total_cmp(&b.1.0).then(b.1.1.cmp(&a.1.1)))?;
    Some((best.0.to_string(), best.1.0, total))
}

fn title_entrance(tl: &Timeline) -> Axis {
    let mut titles: Vec<&Clip> = tl.clips.iter().filter(|c| c.kind == Kind::Title).collect();
    titles.sort_by_key(|c| (c.it.start, c.track));
    if titles.is_empty() {
        return axis("none", "no titles");
    }
    let read: Vec<(String, &'static str)> = titles.iter().map(|c| entrance(tl, c)).collect();
    let votes: Vec<(String, f64)> = read.iter().map(|(k, _)| (k.clone(), 1.0)).collect();
    let Some((id, n, total)) = dominant(&votes) else { return axis("none", "no titles") };
    let from = read.iter().find(|(k, _)| *k == id).map_or("", |(_, f)| *f);
    axis(&id, format!("{} of {} titles enter so ({from})", n as usize, total as usize))
}

fn transition_family(tl: &Timeline) -> Axis {
    let mut votes = Vec::new();
    for (_, x) in &tl.transitions {
        let (Some(_), Some(_)) = (x.from, x.to) else { continue };
        let graphic = |k: Option<Kind>| matches!(k, Some(Kind::Title | Kind::Shape));
        if graphic(tl.kind(x.from)) || graphic(tl.kind(x.to)) {
            continue;
        }
        votes.push((family_of(&x.effect.effect).to_string(), 1.0));
    }
    match dominant(&votes) {
        Some((id, n, total)) => axis(&id, format!("{} of {} transitions between pictures", n as usize, total as usize)),
        None => axis("cut", "no transitions between pictures"),
    }
}

/// The look a Lumetri Color brings, if any.
fn lumetri_look(e: &EffectInstance, presets: &[filmcraft_render::lumetri_presets::LumetriPreset]) -> Option<(String, &'static str)> {
    if !e.enabled || e.effect != "lumetri" {
        return None;
    }
    let text = |k: &str| match e.params.get(k).map(|p| &p.value) {
        Some(ParamValue::Text(t)) if !t.trim().is_empty() => Some(t.trim().to_string()),
        _ => None,
    };
    if let Some(l) = text(crate::auto_color::MARK_LOOK) {
        return Some((slug(&l), "the colour pass's look"));
    }
    let pass = e.params.contains_key(crate::auto_color::MARK);
    if !pass
        && let Some(p) = presets
            .iter()
            .find(|p| !p.params.is_empty() && p.params.iter().all(|(k, v)| e.params.get(*k).is_some_and(|x| x.keyframes.is_empty() && x.value == *v)))
    {
        return Some((slug(p.name), "a Lumetri preset"));
    }
    match e.params.get("look").map(|p| &p.value) {
        Some(ParamValue::Choice(c)) if *c > 0 => {
            if let Some(ours) = crate::auto_color::look_of_choice(*c) {
                return Some((ours.to_string(), "a Lumetri Creative look"));
            }
            let label = filmcraft_project::effect::find_effect("lumetri")
                .and_then(|d| d.param("look"))
                .and_then(|p| match &p.kind {
                    filmcraft_project::effect::ParamKind::Choice(opts) => opts.get(*c as usize).copied(),
                    _ => None,
                })
                .unwrap_or("look");
            Some((slug(label), "a Lumetri Creative look"))
        }
        _ => None,
    }
}

fn color_look(tl: &Timeline) -> Axis {
    let presets = filmcraft_render::lumetri_presets::presets();
    let mut votes: Vec<(String, f64)> = Vec::new();
    let mut from: BTreeMap<String, &'static str> = BTreeMap::new();
    for c in &tl.clips {
        let w = c.it.duration.seconds();
        match c.kind {
            Kind::Adjustment => {
                if let Some(name) = c.it.name.strip_prefix("Look: ").map(str::trim).filter(|n| !n.is_empty()) {
                    let id = slug(name);
                    from.entry(id.clone()).or_insert("a look adjustment clip");
                    votes.push((id, w));
                }
                for e in &c.it.effects {
                    if let Some((id, f)) = lumetri_look(e, &presets) {
                        from.entry(id.clone()).or_insert(f);
                        votes.push((id, w));
                    }
                }
            }
            Kind::Picture => {
                for e in &c.it.effects {
                    if let Some((id, f)) = lumetri_look(e, &presets) {
                        from.entry(id.clone()).or_insert(f);
                        votes.push((id, w));
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    match dominant(&votes) {
        Some((id, w, _)) => {
            let f = from.get(&id).copied().unwrap_or("the pictures");
            axis(&id, format!("{f}, on screen {} s", r2(w)))
        }
        None => axis("none", "no look on the pictures"),
    }
}

fn caption_style(q: &Sequence) -> Axis {
    let mut votes = Vec::new();
    for t in q.caption_tracks.iter().filter(|t| t.enabled && !t.captions.is_empty()) {
        let id = t.style.animated.as_ref().map_or("plain", |a| a.preset.name());
        votes.push((id.to_string(), t.captions.len().min(MAX_CLIPS) as f64));
    }
    match dominant(&votes) {
        Some((id, n, _)) => axis(&id, format!("{} captions in that style", n as usize)),
        None => axis("none", "no captions"),
    }
}

fn blurred(it: &TrackItem) -> bool {
    it.effects.iter().any(|e| e.enabled && e.effect.contains("blur"))
}

fn named_logo(project: &Project, it: &TrackItem) -> bool {
    let item = project.item(it.item).map(|p| p.name.to_lowercase()).unwrap_or_default();
    it.name.to_lowercase().contains("logo") || item.contains("logo")
}

fn close(tl: &Timeline) -> Axis {
    let end = tl.end;
    let from = sub(end, secs(CLOSE)).max(Tick::ZERO);
    let last: Vec<&Clip> = tl.overlapping(from, end).collect();
    let lowest_picture = last.iter().filter(|c| c.kind == Kind::Picture).map(|c| c.track).min();
    let still_alpha = |c: &Clip| {
        tl.project
            .resolve_media(c.it.item)
            .is_some_and(|(_, m, _)| m.info.kind == filmcraft_media::MediaKind::Still && m.info.video.as_ref().is_some_and(|v| v.has_alpha))
    };
    if let Some(c) = last.iter().find(|c| c.kind == Kind::Picture && (named_logo(tl.project, c.it) || (Some(c.track) > lowest_picture && still_alpha(c)))) {
        return axis("logo", format!("a logo, \"{}\", in the last {CLOSE} s", c.it.name.chars().take(40).collect::<String>()));
    }
    let mut titles: Vec<&&Clip> = last.iter().filter(|c| c.kind == Kind::Title).collect();
    titles.sort_by_key(|c| std::cmp::Reverse(c.it.end()));
    if let Some(t) = titles.first() {
        let mid = add(t.it.start.max(from), Tick(sub(t.it.end().min(end), t.it.start.max(from)).0 / 2));
        let under: Vec<&Clip> = tl.at(mid).filter(|c| c.track < t.track).collect();
        let picture = under.iter().filter(|c| c.kind == Kind::Picture).min_by_key(|c| c.track);
        let card = match picture {
            None => Some("over nothing or a graphic"),
            Some(p) if blurred(p.it) => Some("over blurred footage"),
            Some(_) if under.iter().any(|c| c.kind == Kind::Shape && c.track > picture.map_or(0, |p| p.track)) => Some("over a graphic"),
            Some(_) => None,
        };
        return match card {
            Some(how) => axis("endCard", format!("the title \"{}\" closes the film {how}", title_text(t.it))),
            None => axis("title", format!("the title \"{}\" closes the film over footage", title_text(t.it))),
        };
    }
    let frame = Tick(tl.q.settings.frame_rate.sane().frame_duration().0.max(1));
    let edge_out = tl.transitions.iter().find(|(_, x)| x.to.is_none() && x.from.is_some() && add(x.start, x.duration) >= sub(end, frame));
    if let Some((_, x)) = edge_out {
        return axis("fadeOut", format!("the last picture fades out ({})", x.effect.effect));
    }
    let fading = |it: &TrackItem| {
        let Some(o) = it.effect("opacity").and_then(|e| e.params.get("opacity")) else { return false };
        if o.keyframes.len() < 2 {
            return false;
        }
        let at_end = o.value_at(it.effect_time_at(sub(it.end(), frame)));
        let before = o.value_at(it.effect_time_at(sub(it.end(), secs(CLOSE).min(it.duration))));
        matches!((at_end, before), (ParamValue::Float(a), ParamValue::Float(b)) if a < GONE && b > a)
    };
    if let Some(c) = last.iter().find(|c| c.kind == Kind::Picture && c.it.end() >= sub(end, frame) && fading(c.it)) {
        return axis("fadeOut", format!("\"{}\" fades to nothing", c.it.name.chars().take(40).collect::<String>()));
    }
    if let Some(c) = last.iter().find(|c| c.kind == Kind::Shape && c.it.end() >= sub(end, frame) && rising(c.it)) {
        return axis("fadeOut", format!("a shape, \"{}\", fades in over the end", c.it.name.chars().take(40).collect::<String>()));
    }
    match tl.picture_at(sub(end, frame)) {
        Some(c) => axis("cut", format!("it ends on \"{}\"", c.it.name.chars().take(40).collect::<String>())),
        None => axis("cut", "it ends without a picture"),
    }
}

/// Whether a clip's opacity rises from nothing to full over its end (a black fade drawn as a shape).
fn rising(it: &TrackItem) -> bool {
    let Some(o) = it.effect("opacity").and_then(|e| e.params.get("opacity")) else { return false };
    match (o.keyframes.first().map(|k| &k.value), o.keyframes.last().map(|k| &k.value)) {
        (Some(ParamValue::Float(a)), Some(ParamValue::Float(b))) => *a < GONE && *b > 100.0 - GONE,
        _ => false,
    }
}

/// The six axes of sequence `seq` and their evidence.
pub fn fingerprint(project: &Project, q: &Sequence) -> (Axes, BTreeMap<&'static str, String>) {
    let tl = Timeline::read(project, q);
    let read = [opening(&tl), title_entrance(&tl), transition_family(&tl), color_look(&tl), caption_style(q), close(&tl)];
    let mut evidence = BTreeMap::new();
    for (name, (_, ev)) in AXES.iter().zip(&read) {
        evidence.insert(*name, ev.clone());
    }
    let [o, t, f, c, cap, cl] = read;
    (Axes { opening: o.0, title_entrance: t.0, transition_family: f.0, color_look: c.0, caption_style: cap.0, close: cl.0 }, evidence)
}

/// How `axes` matches each of the newest records, and the warnings.
pub fn compare(axes: &Axes, recent: &[&Record]) -> (Vec<Value>, Vec<Value>) {
    let mut matches = Vec::new();
    let mut warnings = Vec::new();
    for (i, r) in recent.iter().enumerate() {
        let same = axes.matching(&r.axes);
        if same.is_empty() {
            continue;
        }
        let n = i.saturating_add(1);
        matches.push(json!({"record": n, "when": r.when, "pipeline": r.pipeline, "project": r.project, "axes": same, "count": same.len()}));
        if same.len() >= WARN_AXES {
            let which = if n == 1 { "the last video".to_string() } else { format!("video {n} back") };
            warnings.push(json!({
                "code": "sameAsRecent",
                "severity": "warn",
                "record": n,
                "axes": same,
                "message": format!("{} of the 6 axes match {which} ({}: {}); vary some of them (autocut.deal chooses against the history)", same.len(), r.when, same.join(", ")),
            }));
        }
    }
    (matches, warnings)
}

fn pipeline_p(p: &Value) -> Result<Option<String>> {
    match p.get("pipeline").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            let s = v.as_str().map(str::trim).unwrap_or_default();
            if s.is_empty() || s.chars().count() > MAX_PIPELINE || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(bad(CMD, format!("`pipeline` is a short name (letters, digits, - or _; at most {MAX_PIPELINE})")));
            }
            Ok(Some(s.to_string()))
        }
    }
}

fn deals_p(p: &Value) -> Result<BTreeMap<String, String>> {
    let Some(v) = p.get("deals").filter(|v| !v.is_null()) else { return Ok(BTreeMap::new()) };
    let o = v.as_object().filter(|o| o.len() <= MAX_DEALS).ok_or_else(|| bad(CMD, format!("`deals` is an object of at most {MAX_DEALS} kind → id")))?;
    let mut out = BTreeMap::new();
    for (k, v) in o {
        if !taste_history::DEAL_KINDS.contains(&k.as_str()) {
            return Err(bad(CMD, format!("`deals` has no kind `{}`: {}", k.chars().take(32).collect::<String>(), taste_history::DEAL_KINDS.join(", "))));
        }
        let id = v.as_str().map(str::trim).filter(|s| !s.is_empty() && s.chars().count() <= 64).ok_or_else(|| bad(CMD, format!("`deals.{k}` is an id")))?;
        out.insert(k.clone(), id.to_string());
    }
    Ok(out)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(o) = p.as_object()
        && let Some(k) = o.keys().find(|k| !matches!(k.as_str(), "sequence" | "record" | "pipeline" | "deals"))
    {
        return Err(bad(CMD, format!("unknown parameter `{}`: sequence, record, pipeline or deals", k.chars().take(64).collect::<String>())));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let record = match p.get("record").filter(|v| !v.is_null()) {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`record` is true or false"))?,
    };
    let pipeline = pipeline_p(p)?;
    let mut deals = deals_p(p)?;
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    let (axes, evidence) = fingerprint(&project, q);
    let history = taste_history::read(s);
    let recent = history.recent(RECENT);
    let (matches, warnings) = compare(&axes, &recent);
    let mut out = json!({
        "sequence": seq.0,
        "axes": axes,
        "evidence": evidence,
        "matches": matches,
        "warnings": warnings,
        "history": {"file": history.path, "records": history.records.len(), "notes": history.notes},
        "recorded": false,
    });
    if record {
        for (k, v) in std::mem::take(&mut s.taste_deals) {
            deals.entry(k).or_insert(v);
        }
        let r = Record { when: taste_history::now(), project: s.path.as_deref().map(taste_history::file_name), pipeline, axes, deals };
        let written = taste_history::append(s, &r).map_err(|e| bad(CMD, format!("the history could not be written: {e}")))?;
        out["recorded"] = json!(true);
        out["record"] = json!(r.sanitized());
        out["history"]["records"] = json!(written.records.len());
        out["note"] =
            json!(format!("added to the taste history ({}) in the data directory: a file outside the project, not an undo step", taste_history::FILE));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
