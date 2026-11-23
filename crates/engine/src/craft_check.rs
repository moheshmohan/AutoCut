//! `sequence.checkCraft`: craft checks of a sequence's text and timeline (PX1; reading time after
//! lemo-opuscar, the check list after hyperframes' check codes; written fresh).
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `readingTime` | a title (all the text layers of a graphic clip) or a caption is on screen for less time than it takes to read | 3 words a second plus 0.5 s |
//! | `smallText` | a text layer, or a burned-in caption track, is smaller than the floor | 2.6 % of the frame's short side (the font size) |
//! | `lowContrast` | the text against the picture behind it, at its worst sample, is under the WCAG ratio | 4.5:1, 3:1 for large text (3.9 % of the short side, 3.03 % bold) |
//! | `emptyFirstFrame` | the first frame has no picture, starts at opacity 0, or renders black or one flat colour | as `qa.motionEnergy`: mean grey < 0.04, spread < 0.01 |
//! | `staticTimeline` | more than 3 s with nothing changing on the timeline, outside the declared holds | 3 s |
//! | `silence` | more than 15 s with no clip on an audible audio track | 15 s |
//!
//! **What "behind the text" is.** For contrast the sequence is rendered at a few times across each
//! text's life (3 to 8, about one every 2 s) with every title's text layers turned off and no
//! captions: so the backdrop is the footage, shapes and other graphics under the text, but also
//! (an approximation) without other titles' text. Inside the text's box the 10th and 90th
//! percentile of the picture's luminance stand for its darkest and brightest parts; a text
//! colour between them reads as 1:1. A background box behind the text is laid over the backdrop
//! with its opacity, and the text's own opacity (layer and clip) is mixed in. A stroke or caption
//! outline at least 5 % of the font size counts as a backdrop of its own colour, and the better of
//! the two ratios is kept. Shadows, glows, per-character colours, animated-caption highlight
//! colours and titles inside nested sequences are not measured. At most 400 frames are rendered:
//! past that, each text gets one sample and then only some texts are measured (`checked` says how
//! many).
//!
//! The timeline checks read the timeline, not pixels: something changes when a clip, caption or
//! transition starts or ends, a moving picture plays, an animated parameter moves, a title's text
//! animates or a nested sequence plays (taken as always moving). It is sampled every 0.1 s and at
//! every edit point. A clip of silence counts as sound; levels are not measured.
//!
//! Every finding is a warning (decision D7) and never blocks a delivery.
use std::collections::BTreeMap;

use filmcraft_geom::Vec2;
use filmcraft_project::graphic::{LayerContent, LayerSpec, TEXT_LAYER};
use filmcraft_project::{ItemId, ItemKind, MediaRef, ParamValue, Project, Sequence, TrackItem};
use filmcraft_render::graphic_clip::{item_layer_specs, layer_bounds, layer_matrix};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::motion_energy::{bounded, finding, holds_p, r3, uncovered};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.checkCraft";
/// Reading speed and the floor added to every line.
pub const WORDS_PER_SECOND: f64 = 3.0;
pub const READING_FLOOR: f64 = 0.5;
/// Smallest text, a share of the frame's short side (its font size).
pub const MIN_TEXT_SHARE: f64 = 0.026;
/// Large text, for the lower contrast ratio: 1.5 × the floor (WCAG's 18 pt against 12 pt), or
/// 14/12 of it when bold (WCAG's 14 pt bold).
pub const LARGE_TEXT_SHARE: f64 = 0.039;
pub const LARGE_BOLD_SHARE: f64 = 0.0303;
/// WCAG contrast ratios for text and large text.
pub const CONTRAST: f64 = 4.5;
pub const CONTRAST_LARGE: f64 = 3.0;
/// A stroke or outline this share of the font size counts as the text's backdrop.
pub const STROKE_SHARE: f64 = 0.05;
/// Longest stretch with nothing changing, and with no sound, seconds.
pub const STATIC_SECONDS: f64 = 3.0;
pub const SILENCE_SECONDS: f64 = 15.0;
/// Most frames rendered for contrast, and their long side.
const MAX_RENDERS: usize = 400;
const BACKDROP_SIDE: f64 = 480.0;
/// The timeline's sampling step, seconds, and the most samples (the step grows past it).
const TIMELINE_STEP: f64 = 0.1;
const MAX_TIMELINE_SAMPLES: usize = 72_000;
/// Most edit points added to the samples.
const MAX_EDIT_POINTS: usize = 40_000;
/// Most text layers and captions looked at.
const MAX_TEXTS: usize = 20_000;
/// Most measured texts whose worst ratio is listed (findings or not).
const MAX_RATIOS: usize = 500;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Craft",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (default the active one),"holds":[{"start":ticks,"end":ticks,"reason":str}]? (intended static stretches)}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

fn linear(c: f64) -> f64 {
    let c = if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 };
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// WCAG relative luminance of an sRGB-encoded colour (0..1 channels).
pub(crate) fn luminance(rgb: [f64; 3]) -> f64 {
    0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2])
}

/// WCAG contrast ratio of two relative luminances.
pub(crate) fn ratio(a: f64, b: f64) -> f64 {
    let (hi, lo) = (a.max(b).clamp(0.0, 1.0), a.min(b).clamp(0.0, 1.0));
    (hi + 0.05) / (lo + 0.05)
}

/// How a text looks at one time.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Look {
    /// Its box, sequence pixels `[x0, y0, x1, y1]`.
    pub rect: [f64; 4],
    /// Font size on screen, sequence pixels.
    pub em: f64,
    /// Luminance of each colour the letters have (a gradient has two).
    pub text: Vec<f64>,
    /// The letters' opacity (fill, layer and clip).
    pub alpha: f64,
    /// A box behind the text: its luminance and opacity.
    pub plate: Option<(f64, f64)>,
    /// A stroke wide enough to count as the backdrop: its luminance.
    pub stroke: Option<f64>,
    pub bold: bool,
}

impl Look {
    fn large(&self, short: f64) -> bool {
        let share = self.em / short.max(1.0);
        share >= LARGE_TEXT_SHARE || (self.bold && share >= LARGE_BOLD_SHARE)
    }
}

/// The worst contrast of `look` against a backdrop whose darkest and brightest parts have
/// luminance `dark` and `bright`.
pub(crate) fn worst_contrast(look: &Look, dark: f64, bright: f64) -> f64 {
    let under = |l: f64| match look.plate {
        Some((p, a)) => a * p + (1.0 - a) * l,
        None => l,
    };
    let (lo, hi) = (under(dark).min(under(bright)), under(dark).max(under(bright)));
    let a = look.alpha.clamp(0.0, 1.0);
    let mut worst = f64::INFINITY;
    for &t in &look.text {
        for bg in [lo, hi] {
            worst = worst.min(ratio(a * t + (1.0 - a) * bg, bg));
        }
        // a backdrop as bright as the letters somewhere in the box
        if (lo..=hi).contains(&t) {
            worst = 1.0;
        }
    }
    if let Some(s) = look.stroke {
        let against_stroke = look.text.iter().map(|t| ratio(*t, s)).fold(f64::INFINITY, f64::min);
        worst = worst.max(against_stroke);
    }
    if worst.is_finite() { worst } else { 1.0 }
}

/// The 10th and 90th percentile luminance of `img` (premultiplied linear RGBA, as rendered) inside
/// `rect` (image pixels), over black.
pub(crate) fn backdrop(img: &filmcraft_render::Image, rect: [f64; 4]) -> Option<(f64, f64)> {
    const BINS: usize = 1024;
    if img.w == 0 || img.h == 0 || !rect.iter().all(|v| v.is_finite()) {
        return None;
    }
    let x0 = rect[0].floor().clamp(0.0, img.w as f64) as usize;
    let x1 = rect[2].ceil().clamp(0.0, img.w as f64) as usize;
    let y0 = rect[1].floor().clamp(0.0, img.h as f64) as usize;
    let y1 = rect[3].ceil().clamp(0.0, img.h as f64) as usize;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let px = img.px.as_chunks::<4>().0;
    let mut hist = [0u64; BINS];
    let mut n = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            let Some(p) = px.get(y * img.w + x) else { continue };
            let l = f64::from(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]);
            let l = if l.is_finite() { l.clamp(0.0, 1.0) } else { 0.0 };
            if let Some(b) = hist.get_mut(((l * (BINS - 1) as f64).round() as usize).min(BINS - 1)) {
                *b += 1;
            }
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let at = |share: f64| {
        let want = ((n as f64 * share).ceil() as u64).clamp(1, n);
        let mut seen = 0;
        for (i, c) in hist.iter().enumerate() {
            seen += c;
            if seen >= want {
                return i as f64 / (BINS - 1) as f64;
            }
        }
        1.0
    };
    Some((at(0.1), at(0.9)))
}

/// A text to measure: a text layer of a graphic clip, or a caption (an animated caption's page).
#[derive(Clone, Debug)]
enum Subject {
    Layer { track: usize, clip: u64, effect: usize },
    Caption { look: Look },
}

#[derive(Clone, Debug)]
struct Text {
    kind: &'static str,
    clip: u64,
    track: String,
    text: String,
    start: Tick,
    end: Tick,
    subject: Subject,
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

fn opacity(it: &TrackItem, mt: Tick) -> f64 {
    match it.effect("opacity") {
        Some(e) if e.enabled => (e.f64_at("opacity", mt) / 100.0).clamp(0.0, 1.0),
        _ => 1.0,
    }
}

fn rgb(c: [f32; 4]) -> [f64; 3] {
    [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])]
}

fn rgb8(c: [u8; 4]) -> [f64; 3] {
    [f64::from(c[0]) / 255.0, f64::from(c[1]) / 255.0, f64::from(c[2]) / 255.0]
}

/// How text layer `spec` of clip `it` looks at media time `mt` (`src` the graphic's size).
fn layer_look(q: &Sequence, it: &TrackItem, spec: &LayerSpec, src: (u32, u32), mt: Tick) -> Option<Look> {
    let LayerContent::Text(tp) = &spec.content else { return None };
    if !spec.enabled {
        return None;
    }
    let m = filmcraft_render::motion_matrix(q, it, src, None, mt);
    let b = layer_bounds(spec);
    if !b.iter().all(|v| v.is_finite()) || b[2] <= b[0] || b[3] <= b[1] {
        return None;
    }
    let corners = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])].map(|(x, y)| m.apply(Vec2::new(x, y)));
    let rect = [
        corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
        corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
        corners.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
        corners.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
    ];
    let lm = layer_matrix(spec);
    let size = f64::from(tp.size);
    let (o, up) = (m.apply(lm.apply(Vec2::new(0.0, 0.0))), m.apply(lm.apply(Vec2::new(0.0, size))));
    let em = (up.x - o.x).hypot(up.y - o.y);
    let look = &spec.appearance;
    let (text, alpha) = match (&look.gradient, look.fill) {
        (Some(g), _) => (vec![luminance(rgb(g.start)), luminance(rgb(g.end))], f64::from(g.start[3].min(g.end[3]))),
        (None, Some(f)) => (vec![luminance(rgb(f))], f64::from(f[3])),
        // letters drawn by their stroke alone
        (None, None) => {
            let (c, _, _) = look.strokes.first()?;
            (vec![luminance(rgb(*c))], f64::from(c[3]))
        }
    };
    let wide = |w: f32| f64::from(w) * em / size.max(1e-6) >= STROKE_SHARE * em;
    let stroke = if look.fill.is_some() || look.gradient.is_some() {
        look.strokes.iter().rev().find(|(c, w, _)| c[3] >= 0.5 && wide(*w)).map(|(c, _, _)| luminance(rgb(*c)))
    } else {
        None
    };
    let plate = look.background.map(|(c, _, _)| (luminance(rgb(c)), f64::from(c[3]).clamp(0.0, 1.0)));
    let alpha = (alpha * f64::from(spec.transform.opacity) * opacity(it, mt)).clamp(0.0, 1.0);
    let style = tp.style.to_ascii_lowercase();
    let bold = tp.faux_bold || ["bold", "black", "heavy"].iter().any(|w| style.contains(w));
    rect.iter().all(|v| v.is_finite()).then_some(Look { rect, em, text, alpha, plate, stroke, bold })
}

/// The text layer `effect` of clip `clip` on video track `track` at timeline time `t`.
fn subject_look(project: &Project, q: &Sequence, subject: &Subject, t: Tick) -> Option<Look> {
    match subject {
        Subject::Caption { look } => Some(look.clone()),
        Subject::Layer { track, clip, effect } => {
            let it = q.video_tracks.get(*track)?.items.iter().find(|i| i.id.0 == *clip)?;
            let src = project.source_size(it.item)?;
            let mt = it.effect_time_at(t);
            let specs = item_layer_specs(it, mt, src);
            let (_, spec) = specs.iter().find(|(i, _)| i == effect)?;
            layer_look(q, it, spec, src, mt)
        }
    }
}

/// The enabled text layers of the graphic clips on enabled video tracks of `q`, at most `max`:
/// (track index, clip, effect index, its text at the clip's middle). `copy.lint` reads titles
/// through this too.
pub(crate) fn title_layers<'a>(project: &Project, q: &'a Sequence, max: usize) -> Vec<(usize, &'a TrackItem, usize, String)> {
    let mut out = Vec::new();
    for (ti, t) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        for it in t.items.iter().filter(|i| i.enabled) {
            if !project.item(it.item).is_some_and(|p| matches!(p.kind, ItemKind::Graphic { .. })) {
                continue;
            }
            for (i, e) in it.effects.iter().enumerate() {
                if out.len() >= max {
                    return out;
                }
                if e.effect != TEXT_LAYER || !e.enabled {
                    continue;
                }
                let text = match e.params.get("text").map(|p| p.value_at(it.effect_time_at(it.start + Tick(it.duration.0 / 2)))) {
                    Some(ParamValue::Text(s)) => s,
                    _ => String::new(),
                };
                out.push((ti, it, i, text));
            }
        }
    }
    out
}

/// The texts of `q`: every text layer of the graphic clips on visible video tracks, and every
/// caption (each page of an animated one) on visible caption tracks.
fn texts(project: &Project, q: &Sequence) -> Vec<Text> {
    let mut out: Vec<Text> = title_layers(project, q, MAX_TEXTS)
        .into_iter()
        .map(|(ti, it, i, text)| Text {
            kind: "title",
            clip: it.id.0,
            track: format!("V{}", ti + 1),
            text,
            start: it.start,
            end: it.end(),
            subject: Subject::Layer { track: ti, clip: it.id.0, effect: i },
        })
        .collect();
    let (w, h) = (q.settings.width as usize, q.settings.height as usize);
    let scale = w.min(h) as f64 / 1080.0;
    for o in crate::safe_zones::overlays(project, q).into_iter().filter(|o| o.kind == "caption") {
        if out.len() >= MAX_TEXTS {
            break;
        }
        let Some(track) = o.track.strip_prefix('C').and_then(|n| n.parse::<usize>().ok()).and_then(|n| n.checked_sub(1)).and_then(|i| q.caption_tracks.get(i))
        else {
            continue;
        };
        let st = &track.style;
        let size = if st.size.is_finite() { f64::from(st.size).clamp(4.0, 256.0) } else { 54.0 };
        let em = size * scale;
        let outline = if st.outline.is_finite() { f64::from(st.outline).max(0.0) } else { 0.0 };
        let look = Look {
            rect: o.rect,
            em,
            text: vec![luminance(rgb8(st.color))],
            alpha: f64::from(st.color[3]) / 255.0,
            plate: st.background.then(|| (luminance(rgb8(st.background_color)), f64::from(st.background_color[3]) / 255.0)),
            stroke: (outline >= STROKE_SHARE * size && st.outline_color[3] >= 128).then(|| luminance(rgb8(st.outline_color))),
            bold: false,
        };
        out.push(Text { kind: "caption", clip: o.clip, track: o.track, text: o.text, start: o.start, end: o.end, subject: Subject::Caption { look } });
    }
    out
}

/// One thing the timeline shows at a time, for telling whether anything changed.
#[derive(Clone, Debug, PartialEq)]
enum Shown {
    Clip(usize, u64),
    Frame(u64, i64),
    Value(u64, ParamValue),
    Layers(u64, Vec<LayerSpec>),
    Transition(u64, i64),
    Caption(usize, u64),
    /// Something taken to change all the time (a nested sequence, an animated caption or mask).
    Moving(usize),
}

/// Whether clip `it` shows a picture that moves by itself (a movie playing, not a still, a frame
/// hold or a plain colour).
fn moving_picture(project: &Project, it: &TrackItem) -> bool {
    if it.frame_hold.is_some() {
        return false;
    }
    match project.item(it.item).map(|p| &p.kind) {
        Some(ItemKind::Sequence(_)) => true,
        Some(ItemKind::Graphic { .. } | ItemKind::AdjustmentLayer { .. }) | None => false,
        Some(_) => project.resolve_media(it.item).is_some_and(|(_, m, _)| {
            use filmcraft_media::{Generator, MediaKind};
            let plain = matches!(m.media, MediaRef::Generator(Generator::ColorMatte { .. } | Generator::BlackVideo | Generator::TransparentVideo));
            !plain && !matches!(m.info.kind, MediaKind::Still | MediaKind::AudioOnly)
        }),
    }
}

/// What the timeline shows at `t` (sample number `n`).
fn shown(project: &Project, q: &Sequence, t: Tick, n: usize) -> Vec<Shown> {
    let mut v = Vec::new();
    for (ti, track) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        for tr in track.transitions.iter().filter(|tr| tr.start <= t && t < tr.end()) {
            v.push(Shown::Transition(tr.id.0, (tr.progress(t) * 1000.0).round() as i64));
        }
        let Some(it) = track.item_at(t).filter(|i| i.enabled) else { continue };
        v.push(Shown::Clip(ti, it.id.0));
        let mt = it.effect_time_at(t);
        match project.item(it.item).map(|p| &p.kind) {
            Some(ItemKind::Sequence(_)) => v.push(Shown::Moving(n)),
            Some(ItemKind::Graphic { .. }) => {
                if let Some(src) = project.source_size(it.item) {
                    v.push(Shown::Layers(it.id.0, item_layer_specs(it, mt, src).into_iter().map(|(_, s)| s).collect()));
                }
            }
            _ if moving_picture(project, it) => {
                let rate = project.resolve_media(it.item).map(|(_, m, _)| m.info.frame_rate()).unwrap_or_default();
                v.push(Shown::Frame(it.id.0, rate.frame_at(it.source_time_at(t))));
            }
            _ => {}
        }
        for e in it.effects.iter().filter(|e| e.enabled && !filmcraft_project::graphic::is_layer(e)) {
            if e.masks.iter().any(|m| m.is_animated()) {
                v.push(Shown::Moving(n));
            }
            for p in e.params.values().filter(|p| p.is_animated()) {
                v.push(Shown::Value(it.id.0, p.value_at(mt)));
            }
        }
    }
    for (ci, track) in q.caption_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        if let Some(c) = track.caption_at(t) {
            v.push(Shown::Caption(ci, c.id.0));
            if track.style.animated.is_some() {
                v.push(Shown::Moving(n));
            }
        }
    }
    v
}

/// The timeline's sample times over `[0, end)`: every [`TIMELINE_STEP`] (wider for a long
/// sequence) and every edit point.
fn sample_times(q: &Sequence, end: Tick) -> Vec<i64> {
    let secs = end.seconds();
    let step = TIMELINE_STEP.max(secs / MAX_TIMELINE_SAMPLES as f64);
    let count = ((secs / step).ceil() as usize).min(MAX_TIMELINE_SAMPLES);
    let mut times: Vec<i64> = (0..count).map(|k| Tick::from_seconds_f64(k as f64 * step).0).filter(|t| *t < end.0).collect();
    let points = q
        .video_tracks
        .iter()
        .flat_map(|t| t.items.iter().flat_map(|c| [c.start.0, c.end().0]).chain(t.transitions.iter().flat_map(|tr| [tr.start.0, tr.end().0])))
        .chain(q.caption_tracks.iter().flat_map(|t| t.captions.iter().flat_map(|c| [c.start.0, c.end().0])))
        .filter(|t| *t > 0 && *t < end.0)
        .take(MAX_EDIT_POINTS);
    times.extend(points);
    times.push(0);
    times.sort_unstable();
    times.dedup();
    times
}

/// Stretches longer than [`STATIC_SECONDS`] with nothing changing, outside `holds`; and how many
/// samples were read.
fn static_stretches(project: &Project, q: &Sequence, end: Tick, holds: &[(i64, i64)]) -> (Vec<Value>, usize) {
    let times = sample_times(q, end);
    let mut findings = Vec::new();
    let longest = Tick::from_seconds_f64(STATIC_SECONDS).0;
    let mut run_start = 0i64;
    let mut prev: Option<Vec<Shown>> = None;
    let close = |from: i64, to: i64, findings: &mut Vec<Value>| {
        for (a, b) in uncovered(from, to, holds) {
            if b.saturating_sub(a) > longest {
                let secs = Tick(b - a).seconds();
                findings.push(finding(
                    "staticTimeline",
                    Tick(a),
                    Some(Tick(b)),
                    format!("nothing changes on the timeline for {secs:.1} s; declare a hold if it is meant"),
                    json!({"length": r3(secs)}),
                ));
            }
        }
    };
    for (n, &t) in times.iter().enumerate() {
        let now = shown(project, q, Tick(t), n);
        if prev.as_ref().is_some_and(|p| *p != now) {
            close(run_start, t, &mut findings);
            run_start = t;
        }
        prev = Some(now);
    }
    close(run_start, end.0, &mut findings);
    (findings, times.len())
}

/// Stretches longer than [`SILENCE_SECONDS`] with no clip on an audible audio track.
fn silences(q: &Sequence, end: Tick) -> Vec<Value> {
    let mut spans: Vec<(i64, i64)> = q
        .audio_tracks
        .iter()
        .filter(|t| t.enabled && !t.muted)
        .flat_map(|t| t.items.iter().filter(|c| c.enabled).map(|c| (c.start.0, c.end().0)))
        .filter(|(a, b)| b > a)
        .collect();
    spans.sort_unstable();
    let longest = Tick::from_seconds_f64(SILENCE_SECONDS).0;
    let mut findings = Vec::new();
    let mut gap = |a: i64, b: i64| {
        if b.saturating_sub(a) > longest {
            let secs = Tick(b - a).seconds();
            findings.push(finding("silence", Tick(a), Some(Tick(b)), format!("{secs:.1} s with no sound on the timeline"), json!({"length": r3(secs)})));
        }
    };
    let mut heard_to = 0i64;
    for (a, b) in spans {
        if a > heard_to {
            gap(heard_to, a.min(end.0));
        }
        heard_to = heard_to.max(b);
    }
    if heard_to < end.0 {
        gap(heard_to, end.0);
    }
    findings
}

/// Renders sequence `seq` of `project` on the GPU frame renderer where the host has one, else
/// on the CPU, counting which drew each frame.
struct Renderer {
    gpu: Option<Box<dyn filmcraft_export::FrameRenderer>>,
    on_gpu: usize,
    on_cpu: usize,
}

impl Renderer {
    fn render(
        &mut self,
        project: &Project,
        seq: ItemId,
        t: Tick,
        opts: filmcraft_render::RenderOptions,
        sources: &dyn filmcraft_render::SourceProvider,
    ) -> filmcraft_render::Image {
        if let Some(img) = self.gpu.as_mut().and_then(|r| r.render(project, seq, t, opts, sources)) {
            self.on_gpu += 1;
            return img;
        }
        self.on_cpu += 1;
        filmcraft_render::render_sequence(project, seq, t, opts, sources)
    }
}

/// `project` with every text layer of `seq`'s graphic clips turned off: the picture behind the
/// titles (captions are not drawn unless asked).
fn without_text(project: &Project, seq: ItemId) -> Project {
    let mut p = project.clone();
    if let Some(q) = p.sequence_mut(seq) {
        for t in q.video_tracks.iter_mut() {
            for it in t.items.iter_mut() {
                for e in it.effects.iter_mut().filter(|e| e.effect == TEXT_LAYER) {
                    e.enabled = false;
                }
            }
        }
    }
    p
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    let end = q.duration();
    if end.0 <= 0 {
        return Err(bad(CMD, "the sequence is empty"));
    }
    let holds = holds_p(CMD, p, "holds", end)?;
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let short = f64::from(w.min(h));
    let mut findings = Vec::new();
    let mut skipped = Vec::new();
    let all = texts(&project, q);
    // ---- reading time: a title is all its text layers together; every caption track counts
    // (hidden ones go out as sidecar subtitles)
    let mut titles: BTreeMap<u64, (Tick, Tick, usize, String, String)> = BTreeMap::new();
    for t in all.iter().filter(|t| t.kind == "title") {
        let e = titles.entry(t.clip).or_insert((t.start, t.end, 0, t.track.clone(), String::new()));
        e.2 += words(&t.text);
        if e.4.chars().count() < 200 {
            e.4 = format!("{} {}", e.4, t.text).trim().chars().take(200).collect();
        }
    }
    let mut read = |kind: &str, clip: u64, track: &str, text: &str, n: usize, start: Tick, stop: Tick| {
        if n == 0 {
            return;
        }
        let shown = Tick(stop.0.min(end.0).saturating_sub(start.0.max(0))).seconds();
        let needed = n as f64 / WORDS_PER_SECOND + READING_FLOOR;
        if shown < needed {
            findings.push(finding(
                "readingTime",
                start,
                Some(stop),
                format!("{n} words are on screen for {shown:.2} s; reading them takes about {needed:.2} s"),
                json!({"kind": kind, "clip": clip, "track": track, "text": text.chars().take(200).collect::<String>(), "words": n, "onScreen": r3(shown), "needed": r3(needed)}),
            ));
        }
    };
    for (clip, (start, stop, n, track, text)) in &titles {
        read("title", *clip, track, text, *n, *start, *stop);
    }
    let mut cues = 0usize;
    for (ci, track) in q.caption_tracks.iter().enumerate() {
        for c in track.captions.iter().take(MAX_TEXTS) {
            cues += 1;
            let text = c.plain_lines().join(" ");
            read("caption", c.id.0, &format!("C{}", ci + 1), &text, words(&text), c.start, c.end());
        }
    }
    // ---- text size: each text layer at the middle of its clip; a caption track by its style
    let mut caption_tracks_sized = std::collections::BTreeSet::new();
    for t in &all {
        let mid = Tick(t.start.0.saturating_add(t.end.0.saturating_sub(t.start.0) / 2));
        let Some(look) = subject_look(&project, q, &t.subject, mid) else { continue };
        if t.kind == "caption" && !caption_tracks_sized.insert(t.track.clone()) {
            continue;
        }
        let share = look.em / short;
        if share < MIN_TEXT_SHARE {
            let what = if t.kind == "caption" { format!("captions on {}", t.track) } else { "text".to_string() };
            findings.push(finding(
                "smallText",
                t.start,
                Some(t.end),
                format!("the {what} is {:.1} px, {:.1} % of the frame's short side (at least {:.1} % reads on a phone)", look.em, share * 100.0, MIN_TEXT_SHARE * 100.0),
                json!({"kind": t.kind, "clip": t.clip, "track": t.track, "text": t.text.chars().take(200).collect::<String>(), "size": r3(look.em), "share": r3(share), "minimum": r3(MIN_TEXT_SHARE * short)}),
            ));
        }
    }
    // ---- contrast against the picture behind the text
    let life = |t: &Text| Tick(t.end.0.min(end.0).saturating_sub(t.start.0.max(0)));
    let samples_for = |t: &Text, n: usize| -> Vec<i64> {
        let (a, d) = (t.start.0.max(0), life(t).0);
        (0..n).map(|k| a.saturating_add((d as f64 * (k as f64 + 0.5) / n as f64) as i64)).filter(|x| *x < end.0).collect()
    };
    let per_text = |t: &Text| ((life(t).seconds() / 2.0).round() as usize).clamp(3, 8);
    let measurable: Vec<usize> = (0..all.len()).filter(|i| all.get(*i).is_some_and(|t| life(t).0 > 0)).collect();
    let mut plan: Vec<(usize, Vec<i64>)> = measurable.iter().filter_map(|i| all.get(*i).map(|t| (*i, samples_for(t, per_text(t))))).collect();
    let distinct = |plan: &[(usize, Vec<i64>)]| plan.iter().flat_map(|(_, v)| v.iter().copied()).collect::<std::collections::BTreeSet<i64>>().len();
    let mut samples_per_text = "3 to 8";
    if distinct(&plan) > MAX_RENDERS {
        samples_per_text = "1";
        plan = measurable.iter().filter_map(|i| all.get(*i).map(|t| (*i, samples_for(t, 1)))).collect();
        if plan.len() > MAX_RENDERS {
            let keep = plan.len().div_ceil(MAX_RENDERS);
            plan = plan.into_iter().step_by(keep).collect();
        }
    }
    let checked = plan.len();
    let mut at: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, times) in &plan {
        for t in times {
            at.entry(*t).or_default().push(*i);
        }
    }
    let backdrop_project = std::sync::Arc::new(without_text(&project, seq));
    let provider = s.media.provider(backdrop_project.clone(), s.services.clone());
    let mut renderer = Renderer { gpu: filmcraft_export::frame_renderer(), on_gpu: 0, on_cpu: 0 };
    let scale = (BACKDROP_SIDE / f64::from(w.max(h))).min(1.0) as f32;
    let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
    // per text: its worst ratio, when and against what, and its samples
    let mut worst: BTreeMap<usize, (f64, i64, Vec<Value>, bool)> = BTreeMap::new();
    for (time, subjects) in &at {
        let img = renderer.render(&backdrop_project, seq, Tick(*time), opts, &provider);
        let (kx, ky) = (img.w as f64 / f64::from(w), img.h as f64 / f64::from(h));
        for i in subjects {
            let Some(t) = all.get(*i) else { continue };
            let Some(look) = subject_look(&project, q, &t.subject, Tick(*time)) else { continue };
            if look.alpha < 0.05 {
                continue;
            }
            let r = look.rect;
            let Some((dark, bright)) = backdrop(&img, [r[0] * kx, r[1] * ky, r[2] * kx, r[3] * ky]) else { continue };
            let c = worst_contrast(&look, dark, bright);
            let e = worst.entry(*i).or_insert((f64::INFINITY, *time, Vec::new(), look.large(short)));
            if c < e.0 {
                e.0 = c;
                e.1 = *time;
                e.3 = look.large(short);
            }
            e.2.push(json!({"time": time, "ratio": r3(c), "backdrop": [r3(dark), r3(bright)]}));
        }
    }
    let mut ratios = Vec::new();
    for (i, (c, time, samples, large)) in &worst {
        let Some(t) = all.get(*i) else { continue };
        let need = if *large { CONTRAST_LARGE } else { CONTRAST };
        if ratios.len() < MAX_RATIOS {
            ratios.push(json!({"kind": t.kind, "clip": t.clip, "track": t.track, "time": time, "ratio": r3(*c), "required": need}));
        }
        if *c < need {
            findings.push(finding(
                "lowContrast",
                Tick(*time),
                Some(t.end),
                format!("the {} reaches {c:.2}:1 against the picture behind it; {need}:1 is needed", if t.kind == "caption" { "caption" } else { "title text" }),
                json!({"kind": t.kind, "clip": t.clip, "track": t.track, "text": t.text.chars().take(200).collect::<String>(), "ratio": r3(*c), "required": need, "large": large, "samples": samples}),
            ));
        }
    }
    if checked < measurable.len() {
        skipped.push(
            json!({"check": "lowContrast", "reason": format!("{} of {} texts measured: at most {MAX_RENDERS} frames are rendered", checked, measurable.len())}),
        );
    }
    // ---- the first frame
    let first: Vec<&TrackItem> = q.video_tracks.iter().filter(|t| t.enabled).filter_map(|t| t.item_at(Tick::ZERO)).filter(|i| i.enabled).collect();
    let rendered = {
        let full = s.media.provider(project.clone(), s.services.clone());
        let opts = filmcraft_render::RenderOptions { scale, captions: true, ..Default::default() };
        let img = renderer.render(&project, seq, Tick::ZERO, opts, &full);
        crate::motion_energy::box_reduce(img.w as u32, img.h as u32, &img.over_black_rgba8(), 96, 54).ok().map(|small| crate::motion_energy::grey_of(&small))
    };
    let (mean, spread) = rendered.as_deref().map_or((0.0, 0.0), |g| {
        let n = g.len().max(1) as f64;
        let mu = g.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        (mu / 255.0, (g.iter().map(|v| (f64::from(*v) - mu).powi(2)).sum::<f64>() / n).sqrt() / 255.0)
    });
    let why = if first.is_empty() {
        Some("nothing is on the picture tracks at the first frame")
    } else if first.iter().all(|it| opacity(it, it.effect_time_at(Tick::ZERO)) <= 0.01) {
        Some("every clip at the first frame starts at opacity 0")
    } else if rendered.is_some() && mean < crate::motion_energy::BLACK_MEAN {
        Some("the first frame renders black")
    } else if rendered.is_some() && spread < crate::motion_energy::EMPTY_SPREAD {
        Some("the first frame renders as one flat colour")
    } else {
        None
    };
    if let Some(why) = why {
        findings.push(finding("emptyFirstFrame", Tick::ZERO, None, why.to_string(), json!({"mean": r3(mean), "spread": r3(spread)})));
    }
    // ---- the timeline
    let (stills, timeline_samples) = static_stretches(&project, q, end, &holds);
    findings.extend(stills);
    findings.extend(silences(q, end));
    let (findings, counts) = bounded(findings);
    Ok(json!({
        "sequence": seq.0,
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "skipped": skipped,
        "checked": {
            "titles": titles.len(),
            "textLayers": all.iter().filter(|t| t.kind == "title").count(),
            "captionCues": cues,
            "burnedCaptionPages": all.iter().filter(|t| t.kind == "caption").count(),
            "contrast": {"texts": measurable.len(), "checked": checked, "renders": at.len(), "samplesPerText": samples_per_text, "gpuFrames": renderer.on_gpu, "cpuFrames": renderer.on_cpu, "ratios": ratios},
            "timelineSamples": timeline_samples,
        },
        "holds": holds.iter().map(|(a, b)| json!({"start": a, "end": b})).collect::<Vec<_>>(),
        "thresholds": {
            "wordsPerSecond": WORDS_PER_SECOND, "readingFloor": READING_FLOOR,
            "minTextShare": MIN_TEXT_SHARE, "largeTextShare": LARGE_TEXT_SHARE, "largeBoldShare": LARGE_BOLD_SHARE,
            "contrast": CONTRAST, "contrastLarge": CONTRAST_LARGE, "strokeShare": STROKE_SHARE,
            "staticSeconds": STATIC_SECONDS, "silenceSeconds": SILENCE_SECONDS,
        },
        "method": {
            "contrast": "rendered with every title's text layers off and no captions; 10th and 90th percentile luminance inside the text's box, a background box and the text's opacity mixed in; a stroke or outline of 5 % of the font size or more counts as backdrop; shadows, glows, per-character colours and nested sequences not measured",
            "readingTime": "whitespace-separated words; a title is all text layers of its clip; every caption track",
            "timeline": "clip, caption and transition edges, moving pictures, animated parameters and title animation, sampled every 0.1 s and at every edit point",
            "silence": "clips on enabled, unmuted audio tracks; levels not measured",
        },
    }))
}

#[cfg(test)]
#[path = "craft_check_tests.rs"]
mod tests;
