//! The graphics of the bundled Elements library: original vector stickers and animated text
//! presets, each a [`GraphicsTemplate`] built here in code (shape and text layers, keyframes,
//! responsive pins) — no image, font or template files. The engine's `elements.*` commands list
//! and insert them next to the generated sound effects (`filmcraft_engine::elements`).
//!
//! - **Stickers** ([`stickers`]): arrows, callout bubbles, highlight strokes, simple shapes and a
//!   generic notification bell, drawn as polygons, ellipses and rounded rectangles on a
//!   1920×1080 canvas. They are static; move, scale or animate them with the clip's Motion.
//!   `assets/elements/stickers/*.svg` are previews of the same vectors, exported by the engine.
//! - **Animated text presets** ([`text_presets`]): a lower third that slides in, a title that
//!   pops, kinetic word-by-word text, a caption box, a 3-2-1 countdown and an end card, animated
//!   with keyframes; and titles animated letter, word or line at a time by text animators
//!   ([`crate::text_anim`], saved in the template): a typewriter, letters popping in, a wave, a
//!   flickering neon glow, words sliding up, and a title on an arc dropping in. Intros and outros
//!   are protected by Responsive Design – Time, so trimming or lengthening a clip keeps them whole.
//!
//! Every design is original work by FilmCraft contributors (MIT OR Apache-2.0), drawn from
//! generic conventions (an arrow, a speech bubble, a star) without reference to anyone's
//! artwork. Text uses only the bundled Inter font.

use filmcraft_geom::Vec2;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::effect::EffectInstance;
use crate::graphic::{new_shape_layer, new_text_layer};
use crate::graphic_design::{LayerExtra, Pin, PinTarget};
use crate::gtemplate::{ControlKind, GraphicsTemplate, TemplateControl};
use crate::keyframe::{Interpolation, Keyframe, ParamValue};
use crate::text_anim::{AnimKind, AnimUnit, Animator, TextAnimation};

/// Template ids of elements start with this (`element:sticker.arrow`).
pub const TEMPLATE_PREFIX: &str = "element:";
/// Category of the stickers.
pub const STICKERS: &str = "Stickers";
/// Category of the animated text presets.
pub const ANIMATED_TEXT: &str = "Animated Text";
/// Author and licence of every bundled element.
pub const AUTHOR: &str = "FilmCraft contributors";
pub const LICENSE: &str = "MIT OR Apache-2.0";

/// The element id of a template made here (`element:sticker.arrow` → `sticker.arrow`).
pub fn element_id(t: &GraphicsTemplate) -> &str {
    t.id.strip_prefix(TEMPLATE_PREFIX).unwrap_or(&t.id)
}

// ------------------------------------------------------------------------------------------------
// building blocks
// ------------------------------------------------------------------------------------------------

/// Canvas centre.
const CX: f64 = 960.0;
const CY: f64 = 540.0;

/// Milliseconds → ticks (exact: a tick count per millisecond is an integer).
fn ms(n: i64) -> Tick {
    Tick(n * (TICKS_PER_SECOND / 1000))
}

fn set(e: &mut EffectInstance, id: &str, v: ParamValue) {
    match e.params.get_mut(id) {
        Some(p) => {
            p.value = v;
            p.keyframes.clear();
        }
        None => {
            e.params.insert(id.to_string(), crate::keyframe::Param::new(v));
        }
    }
}

fn rgb(hex: u32) -> ParamValue {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    ParamValue::Color([c(16), c(8), c(0), 1.0])
}

fn uid(e: &mut EffectInstance, id: u64) {
    e.layer = Some(Box::new(LayerExtra { uid: id, ..Default::default() }));
}

/// A text layer. `centred` puts the anchor on the middle of the capitals, so the layer scales
/// about its visual centre and `pos` is that centre (otherwise `pos` is the first baseline's
/// alignment point).
#[allow(clippy::too_many_arguments)]
fn text(id: u64, name: &str, s: &str, pos: (f64, f64), size: f64, style: &str, align: u32, color: u32, centred: bool) -> EffectInstance {
    let mut e = new_text_layer(s, Vec2::new(pos.0, pos.1), size);
    set(&mut e, "name", ParamValue::Text(name.into()));
    set(&mut e, "font", ParamValue::Text("Inter".into()));
    set(&mut e, "font_style", ParamValue::Text(style.into()));
    set(&mut e, "align", ParamValue::Choice(align));
    set(&mut e, "fill_color", rgb(color));
    if centred {
        // Inter's capitals are ~0.73 em tall: their middle sits 0.36 em above the baseline
        set(&mut e, "anchor", ParamValue::Vec2(Vec2::new(0.0, (-0.36 * size).round())));
    }
    uid(&mut e, id);
    e
}

/// A shape layer: 0 rectangle, 1 ellipse, 2 polygon, 3 path (`points`, layer pixels around the
/// layer origin).
fn shape(id: u64, name: &str, kind: u32, pos: (f64, f64), size: (f64, f64), points: Vec<[f32; 2]>, color: u32) -> EffectInstance {
    let mut e = new_shape_layer(kind, Vec2::new(pos.0, pos.1), Vec2::new(size.0, size.1), points);
    set(&mut e, "name", ParamValue::Text(name.into()));
    set(&mut e, "fill_color", rgb(color));
    uid(&mut e, id);
    e
}

/// A path layer from a polygon.
fn path(id: u64, name: &str, pos: (f64, f64), points: Vec<[f32; 2]>, color: u32) -> EffectInstance {
    let (w, h) = extent(&points);
    shape(id, name, 3, pos, (w, h), points, color)
}

/// Width and height of a point set.
fn extent(points: &[[f32; 2]]) -> (f64, f64) {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for p in points {
        b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    }
    // whole pixels: the size is informational for a path, and integers survive JSON exactly
    if b[0] > b[2] { (0.0, 0.0) } else { (f64::from(b[2] - b[0]).round(), f64::from(b[3] - b[1]).round()) }
}

/// A centre-aligned outline (so the exported SVG, whose strokes are centred, matches).
fn outline(e: &mut EffectInstance, color: u32, width: f64) {
    set(e, "stroke", ParamValue::Bool(true));
    set(e, "stroke_color", rgb(color));
    set(e, "stroke_width", ParamValue::Float(width));
    set(e, "stroke_type", ParamValue::Choice(1));
}

fn shadow(e: &mut EffectInstance, distance: f64, blur: f64, opacity: f64) {
    set(e, "shadow", ParamValue::Bool(true));
    set(e, "shadow_distance", ParamValue::Float(distance));
    set(e, "shadow_blur", ParamValue::Float(blur));
    set(e, "shadow_opacity", ParamValue::Float(opacity));
}

/// Pin `e`'s `edges` (left, top, right, bottom) to layer `to` at `offsets` (this layer's edge
/// minus the target's).
fn pin(e: &mut EffectInstance, to: u64, edges: [bool; 4], offsets: [f64; 4]) {
    let p = Pin { to: PinTarget::Layer(to), left: edges[0], top: edges[1], right: edges[2], bottom: edges[3], offsets };
    e.layer.get_or_insert_with(Default::default).pin = Some(p);
}

/// How a keyframe leaves (and, for [`Arrive`], how the segment before it arrives).
#[derive(Clone, Copy)]
enum Ease {
    /// Constant speed.
    Linear,
    /// Hold the value until the next keyframe.
    Hold,
    /// Leave slowly (accelerate away).
    Leave,
    /// Arrive slowly (decelerate into this keyframe).
    Arrive,
    /// Both: a smooth peak or rest.
    Smooth,
}

/// Animate parameter `id` through `(milliseconds, value, ease)` keyframes.
fn animate(e: &mut EffectInstance, id: &str, keys: &[(i64, ParamValue, Ease)]) {
    let Some(p) = e.params.get_mut(id) else { return };
    p.keyframes.clear();
    for (t, v, ease) in keys {
        let mut k = Keyframe::new(ms(*t), v.clone());
        k.interp = match ease {
            Ease::Linear => Interpolation::Linear,
            Ease::Hold => Interpolation::Hold,
            Ease::Leave => Interpolation::EaseOut,
            Ease::Arrive => Interpolation::EaseIn,
            Ease::Smooth => Interpolation::Bezier,
        };
        // strong eases: the motion settles over most of the segment
        k.in_influence = 0.7;
        k.out_influence = 0.7;
        p.keyframes.push(k);
    }
    p.keyframes.sort_by_key(|k| k.time);
    if let Some(first) = p.keyframes.first() {
        p.value = first.value.clone();
    }
}

fn f(x: f64) -> ParamValue {
    ParamValue::Float(x)
}

fn v2(x: f64, y: f64) -> ParamValue {
    ParamValue::Vec2(Vec2::new(x, y))
}

fn ctl(id: &str, name: &str, layer: u64, param: &str) -> TemplateControl {
    TemplateControl { id: id.into(), name: name.into(), kind: ControlKind::for_param(param), layer, param: param.into(), min: None, max: None }
}

fn slider(id: &str, name: &str, layer: u64, param: &str, min: f64, max: f64) -> TemplateControl {
    TemplateControl { min: Some(min), max: Some(max), kind: ControlKind::Slider, ..ctl(id, name, layer, param) }
}

fn template(
    id: &str,
    name: &str,
    category: &str,
    description: &str,
    millis: i64,
    layers: Vec<EffectInstance>,
    controls: Vec<TemplateControl>,
) -> GraphicsTemplate {
    GraphicsTemplate {
        id: format!("{TEMPLATE_PREFIX}{id}"),
        name: name.into(),
        category: category.into(),
        description: description.into(),
        author: AUTHOR.into(),
        license: LICENSE.into(),
        tags: vec!["element".into(), category.to_ascii_lowercase()],
        duration: ms(millis),
        layers,
        controls,
        ..Default::default()
    }
}

/// Protect an intro and an outro (milliseconds) with Responsive Design – Time.
fn responsive(t: &mut GraphicsTemplate, intro: i64, outro: i64) {
    t.graphic.intro = ms(intro);
    t.graphic.outro = ms(outro);
    t.graphic.design_in = Tick::ZERO;
    t.graphic.design_duration = t.duration;
}

fn pt(x: f64, y: f64) -> [f32; 2] {
    [x as f32, y as f32]
}

/// Points of a circular arc around `c`, radius `r`, from `a0` to `a1` degrees (screen angles: 0°
/// right, 90° down), `n` segments, both ends included.
fn arc(c: (f64, f64), r: (f64, f64), a0: f64, a1: f64, n: usize) -> Vec<[f32; 2]> {
    let n = n.max(1);
    (0..=n)
        .map(|i| {
            let a = (a0 + (a1 - a0) * i as f64 / n as f64).to_radians();
            pt(c.0 + r.0 * a.cos(), c.1 + r.1 * a.sin())
        })
        .collect()
}

/// A band along samples `(centre, unit normal, half-width)`: one side forward, the other back — a
/// closed outline for a brush stroke.
fn band(samples: &[((f64, f64), (f64, f64), f64)]) -> Vec<[f32; 2]> {
    let mut out: Vec<[f32; 2]> = samples.iter().map(|(c, n, h)| pt(c.0 + n.0 * h, c.1 + n.1 * h)).collect();
    out.extend(samples.iter().rev().map(|(c, n, h)| pt(c.0 - n.0 * h, c.1 - n.1 * h)));
    out
}

/// A star of `points` tips around the origin: tips at radii `outer` (x, y), notches at `inner`;
/// `jitter` scales successive tips (a ragged burst), empty for a regular star.
fn star(points: usize, outer: (f64, f64), inner: (f64, f64), jitter: &[f64]) -> Vec<[f32; 2]> {
    let n = points.max(2) * 2;
    (0..n)
        .map(|i| {
            let a = (std::f64::consts::TAU * i as f64 / n as f64) - std::f64::consts::FRAC_PI_2;
            let k = if jitter.is_empty() { 1.0 } else { jitter.get((i / 2) % jitter.len()).copied().unwrap_or(1.0) };
            let (rx, ry) = if i % 2 == 0 { (outer.0 * k, outer.1 * k) } else { inner };
            pt(rx * a.cos(), ry * a.sin())
        })
        .collect()
}

/// A rounded rectangle outline (corner radius `r`, `seg` segments a corner) from `(x0, y0)` to
/// `(x1, y1)`, clockwise from the top edge; `bottom` is spliced into the bottom edge (right to
/// left) — the tail of a speech bubble.
fn rounded_rect(x0: f64, y0: f64, x1: f64, y1: f64, r: f64, seg: usize, bottom: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut v = Vec::new();
    v.extend(arc((x1 - r, y0 + r), (r, r), 270.0, 360.0, seg));
    v.extend(arc((x1 - r, y1 - r), (r, r), 0.0, 90.0, seg));
    v.extend_from_slice(bottom);
    v.extend(arc((x0 + r, y1 - r), (r, r), 90.0, 180.0, seg));
    v.extend(arc((x0 + r, y0 + r), (r, r), 180.0, 270.0, seg));
    v
}

// ------------------------------------------------------------------------------------------------
// stickers
// ------------------------------------------------------------------------------------------------

const YELLOW: u32 = 0xfacc15;
const RED: u32 = 0xef4444;
const INK: u32 = 0x111827;
const WHITE: u32 = 0xffffff;
const PINK: u32 = 0xec4899;
const GREEN: u32 = 0x22c55e;
const BLUE: u32 = 0x2563eb;
const ORANGE: u32 = 0xf97316;
/// How long a sticker clip is by default, milliseconds.
const STICKER_MS: i64 = 4000;

/// Controls of a one-layer sticker (layer 1): colour, outline colour, size, rotation, position.
fn single_controls(outlined: bool) -> Vec<TemplateControl> {
    let mut v = vec![ctl("color", "Color", 1, "fill_color")];
    if outlined {
        v.push(ctl("outline_color", "Outline Color", 1, "stroke_color"));
    }
    v.push(slider("size", "Size", 1, "scale", 10.0, 400.0));
    v.push(slider("rotation", "Rotation", 1, "rotation", -180.0, 180.0));
    v.push(ctl("position", "Position", 1, "position"));
    v
}

/// A bold arrow pointing right.
fn arrow() -> GraphicsTemplate {
    let pts = vec![pt(-210.0, -34.0), pt(50.0, -34.0), pt(50.0, -96.0), pt(210.0, 0.0), pt(50.0, 96.0), pt(50.0, 34.0), pt(-210.0, 34.0)];
    let mut a = path(1, "Arrow", (CX, CY), pts, YELLOW);
    outline(&mut a, INK, 10.0);
    template("sticker.arrow", "Arrow", STICKERS, "A bold arrow; turn it with Rotation to point anywhere.", STICKER_MS, vec![a], single_controls(true))
}

/// An arrow curving over the top from left to right.
fn curved_arrow() -> GraphicsTemplate {
    let c = (0.0, 120.0);
    let (outer, inner) = (250.0, 190.0);
    let (a0, a1) = (205.0, 330.0);
    let mut pts = arc(c, (outer, outer), a0, a1, 28);
    let head = |r: f64, a: f64| {
        let a = f64::to_radians(a);
        pt(c.0 + r * a.cos(), c.1 + r * a.sin())
    };
    pts.push(head(outer + 44.0, a1));
    pts.push(head((outer + inner) / 2.0, a1 + 24.0));
    pts.push(head(inner - 44.0, a1));
    pts.extend(arc(c, (inner, inner), a1, a0, 24));
    let mut a = path(1, "Arrow", (CX, CY + 20.0), pts, RED);
    outline(&mut a, WHITE, 8.0);
    template("sticker.arrow-curved", "Arrow – Curved", STICKERS, "An arrow that arcs over the top, left to right.", STICKER_MS, vec![a], single_controls(true))
}

/// A hand-drawn-looking oval loop around something.
fn circle_highlight() -> GraphicsTemplate {
    let n = 120;
    let (rx, ry) = (290.0, 165.0);
    let samples: Vec<_> = (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let a = (200.0 + 375.0 * u).to_radians();
            // a slight spiral, so the loop overshoots its start like a marker stroke
            let k = 1.0 + 0.1 * u;
            let c = (rx * k * a.cos(), ry * k * a.sin());
            let h = 3.0 + 9.0 * (std::f64::consts::PI * u).sin().max(0.0).powf(0.7);
            ((c.0, c.1), (a.cos(), a.sin()), h)
        })
        .collect();
    let ring = path(1, "Loop", (CX, CY), band(&samples), RED);
    template(
        "sticker.circle-highlight",
        "Circle Highlight",
        STICKERS,
        "A marker-style loop to circle something on screen.",
        STICKER_MS,
        vec![ring],
        single_controls(false),
    )
}

/// A brush-stroke underline.
fn underline() -> GraphicsTemplate {
    let n = 60;
    let samples: Vec<_> = (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let x = -300.0 + 600.0 * u;
            // rises gently towards its end; the normal follows the slope
            let y = 10.0 - 22.0 * u * u;
            let slope = -44.0 * u / 600.0;
            let len = (1.0 + slope * slope).sqrt();
            let h = 2.0 + 12.0 * (std::f64::consts::PI * (0.08 + 0.84 * u)).sin().max(0.0).powf(0.5);
            ((x, y), (-slope / len, 1.0 / len), h)
        })
        .collect();
    let u = path(1, "Stroke", (CX, CY), band(&samples), YELLOW);
    template(
        "sticker.underline",
        "Underline – Brush",
        STICKERS,
        "A brush-stroke underline that tapers at both ends.",
        STICKER_MS,
        vec![u],
        single_controls(false),
    )
}

/// A speech bubble that fits its text.
fn speech_bubble() -> GraphicsTemplate {
    let tail = [pt(-70.0, 100.0), pt(-190.0, 200.0), pt(-140.0, 100.0)];
    let pts = rounded_rect(-250.0, -140.0, 250.0, 100.0, 56.0, 8, &tail);
    let mut bubble = path(1, "Bubble", (CX, CY - 20.0), pts, WHITE);
    outline(&mut bubble, INK, 8.0);
    shadow(&mut bubble, 8.0, 24.0, 35.0);
    let t = text(2, "Text", "Wow!", (CX, CY - 40.0), 110.0, "Bold", 1, INK, true);
    // the bubble keeps its padding around the text (and the tail below it) as the text changes
    pin(&mut bubble, 2, [true; 4], BUBBLE_PAD);
    template(
        "sticker.speech-bubble",
        "Speech Bubble",
        STICKERS,
        "A rounded speech bubble with a tail; it grows to fit its text.",
        STICKER_MS,
        vec![bubble, t],
        vec![
            ctl("text", "Text", 2, "text"),
            ctl("text_color", "Text Color", 2, "fill_color"),
            ctl("bubble_color", "Bubble Color", 1, "fill_color"),
            ctl("outline_color", "Outline Color", 1, "stroke_color"),
            ctl("font", "Font", 2, "font"),
            ctl("position", "Position", 2, "position"),
        ],
    )
}

/// Padding of the speech bubble around its text `[left, top, right, bottom]` (the bottom
/// includes the tail).
const BUBBLE_PAD: [f64; 4] = [-97.0, -53.0, 97.0, 153.0];

/// A jagged burst with a shout in it.
fn burst_callout() -> GraphicsTemplate {
    let jitter = [1.0, 0.9, 0.97, 0.86, 1.0, 0.93, 0.88, 0.98, 0.91];
    let pts = star(18, (330.0, 210.0), (250.0, 150.0), &jitter);
    let mut burst = path(1, "Burst", (CX, CY), pts, RED);
    outline(&mut burst, YELLOW, 10.0);
    let t = text(2, "Text", "NEW!", (CX, CY), 110.0, "Bold", 1, WHITE, true);
    pin(&mut burst, 2, [true; 4], BURST_PAD);
    template(
        "sticker.burst-callout",
        "Burst Callout",
        STICKERS,
        "A jagged burst for a shout-out; it grows to fit its text.",
        STICKER_MS,
        vec![burst, t],
        vec![
            ctl("text", "Text", 2, "text"),
            ctl("text_color", "Text Color", 2, "fill_color"),
            ctl("burst_color", "Burst Color", 1, "fill_color"),
            ctl("outline_color", "Outline Color", 1, "stroke_color"),
            ctl("font", "Font", 2, "font"),
            ctl("position", "Position", 2, "position"),
        ],
    )
}

/// Padding of the burst around its text.
const BURST_PAD: [f64; 4] = [-179.0, -143.0, 179.0, 143.0];

/// A five-pointed star.
fn five_star() -> GraphicsTemplate {
    let mut s = path(1, "Star", (CX, CY + 10.0), star(5, (170.0, 170.0), (72.0, 72.0), &[]), YELLOW);
    outline(&mut s, INK, 10.0);
    template("sticker.star", "Star", STICKERS, "A five-pointed star.", STICKER_MS, vec![s], single_controls(true))
}

/// A heart (the classic parametric curve).
fn heart() -> GraphicsTemplate {
    let n = 96;
    let pts: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / n as f64;
            let x = 16.0 * t.sin().powi(3);
            let y = -(13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos());
            pt(x * 11.0, (y - 2.5) * 11.0)
        })
        .collect();
    let mut h = path(1, "Heart", (CX, CY), pts, PINK);
    outline(&mut h, WHITE, 8.0);
    template("sticker.heart", "Heart", STICKERS, "A heart.", STICKER_MS, vec![h], single_controls(true))
}

/// The outline of a generic bell: a knob on a domed body that flares to a rim, with the clapper
/// showing below (one closed polygon, so one fill and one outline).
fn bell_outline() -> Vec<[f32; 2]> {
    let mut right: Vec<[f32; 2]> = Vec::new();
    // dome: a quarter ellipse from the knob down to the shoulder
    for i in 0..=14 {
        let y = -138.0 + 98.0 * i as f64 / 14.0;
        let d = (y + 40.0) / 100.0;
        right.push(pt(100.0 * (1.0 - d * d).max(0.0).sqrt(), y));
    }
    // body: widening slightly
    for i in 1..=10 {
        let u = i as f64 / 10.0;
        right.push(pt(100.0 + 22.0 * u * u, -40.0 + 110.0 * u));
    }
    // rim: a quick flare
    for i in 1..=6 {
        let u = i as f64 / 6.0;
        right.push(pt(122.0 + 40.0 * u.powf(0.6), 70.0 + 26.0 * u));
    }
    let mut v = arc((0.0, -138.0), (20.0, 20.0), 180.0, 360.0, 10);
    v.extend(right.iter().copied());
    // the clapper: a half disc under the rim's middle
    v.extend(arc((0.0, 96.0), (34.0, 34.0), 0.0, 180.0, 12));
    v.extend(right.iter().rev().map(|p| [-p[0], p[1]]));
    v
}

/// A generic notification bell with a count badge (no platform's mark).
fn bell() -> GraphicsTemplate {
    let mut b = path(1, "Bell", (CX, CY), bell_outline(), YELLOW);
    outline(&mut b, INK, 8.0);
    let mut badge = text(2, "Badge", "3", (CX + 128.0, CY - 104.0), 64.0, "Bold", 1, WHITE, true);
    set(&mut badge, "background", ParamValue::Bool(true));
    set(&mut badge, "background_color", rgb(RED));
    set(&mut badge, "background_opacity", f(100.0));
    set(&mut badge, "background_size", f(18.0));
    set(&mut badge, "background_radius", f(48.0));
    // the badge rides on the bell's top-right corner
    pin(&mut badge, 1, [false, true, true, false], BADGE_OFFSETS);
    template(
        "sticker.bell",
        "Notification Bell",
        STICKERS,
        "A generic notification bell with a count badge.",
        STICKER_MS,
        vec![b, badge],
        vec![
            ctl("bell_color", "Bell Color", 1, "fill_color"),
            ctl("outline_color", "Outline Color", 1, "stroke_color"),
            ctl("badge", "Badge Text", 2, "text"),
            ctl("badge_color", "Badge Color", 2, "background_color"),
            ctl("show_badge", "Show Badge", 2, "enabled"),
            ctl("position", "Position", 1, "position"),
        ],
    )
}

/// Where the bell's badge sits: its top and right edges against the bell's.
const BADGE_OFFSETS: [f64; 4] = [0.0, 17.0, -14.0, 0.0];

/// A round badge with a check mark.
fn check_badge() -> GraphicsTemplate {
    let mut disc = shape(1, "Disc", 1, (CX, CY), (280.0, 280.0), vec![], GREEN);
    outline(&mut disc, WHITE, 8.0);
    shadow(&mut disc, 6.0, 20.0, 35.0);
    // a check of even thickness: the elbow, the short arm up-left, the long arm up-right
    let pts = vec![pt(-80.0, -6.0), pt(-56.0, -30.0), pt(-22.0, 4.0), pt(56.0, -74.0), pt(80.0, -50.0), pt(-22.0, 52.0)];
    let mut check = path(2, "Check", (CX - 4.0, CY + 11.0), pts, WHITE);
    pin(&mut check, 1, [true, true, false, false], CHECK_OFFSETS);
    template(
        "sticker.check-badge",
        "Check Badge",
        STICKERS,
        "A round badge with a check mark.",
        STICKER_MS,
        vec![disc, check],
        vec![
            ctl("disc_color", "Badge Color", 1, "fill_color"),
            ctl("check_color", "Check Color", 2, "fill_color"),
            ctl("outline_color", "Outline Color", 1, "stroke_color"),
            ctl("position", "Position", 1, "position"),
        ],
    )
}

/// The check's left and top edges against the disc's.
const CHECK_OFFSETS: [f64; 4] = [56.0, 77.0, 0.0, 0.0];

/// The stickers: original vector designs (see the module docs).
pub fn stickers() -> Vec<GraphicsTemplate> {
    vec![arrow(), curved_arrow(), speech_bubble(), burst_callout(), circle_highlight(), underline(), five_star(), heart(), bell(), check_badge()]
}

// ------------------------------------------------------------------------------------------------
// animated text presets
// ------------------------------------------------------------------------------------------------

/// Opacity 0 → 100 from `a` to `b` ms, then 100 → 0 from `c` to `d` ms.
fn fade(e: &mut EffectInstance, a: i64, b: i64, c: i64, d: i64) {
    animate(e, "opacity", &[(a, f(0.0), Ease::Linear), (b, f(100.0), Ease::Linear), (c, f(100.0), Ease::Linear), (d, f(0.0), Ease::Linear)]);
}

/// A lower third whose bar slides in from the left, the name and role following it.
fn lower_third_slide() -> GraphicsTemplate {
    let (y, off) = (900.0, -340.0);
    let mut bar = shape(1, "Bar", 0, (400.0, y), (640.0, 132.0), vec![], BLUE);
    set(&mut bar, "opacity", f(95.0));
    animate(
        &mut bar,
        "position",
        &[(0, v2(off, y), Ease::Linear), (500, v2(400.0, y), Ease::Arrive), (5400, v2(400.0, y), Ease::Leave), (6000, v2(off, y), Ease::Linear)],
    );
    let mut accent = shape(2, "Accent", 0, (86.0, y), (12.0, 132.0), vec![], YELLOW);
    let ax = off - 314.0;
    animate(
        &mut accent,
        "position",
        &[(0, v2(ax, y), Ease::Linear), (500, v2(86.0, y), Ease::Arrive), (5400, v2(86.0, y), Ease::Leave), (6000, v2(ax, y), Ease::Linear)],
    );
    let mut name = text(3, "Name", "Alex Morgan", (120.0, 893.0), 60.0, "SemiBold", 0, WHITE, false);
    fade(&mut name, 300, 750, 5350, 5600);
    animate(&mut name, "position", &[(300, v2(80.0, 893.0), Ease::Linear), (750, v2(120.0, 893.0), Ease::Arrive)]);
    let mut role = text(4, "Role", "Field Producer", (122.0, 948.0), 34.0, "Regular", 0, 0xdbeafe, false);
    fade(&mut role, 450, 900, 5300, 5550);
    animate(&mut role, "position", &[(450, v2(82.0, 948.0), Ease::Linear), (900, v2(122.0, 948.0), Ease::Arrive)]);
    let mut t = template(
        "text.lower-third-slide-in",
        "Lower Third – Slide In",
        ANIMATED_TEXT,
        "A bar slides in from the left with the name and role after it, and slides back out at the end.",
        6000,
        vec![bar, accent, name, role],
        vec![
            ctl("name", "Name", 3, "text"),
            ctl("role", "Role", 4, "text"),
            ctl("bar_color", "Bar Color", 1, "fill_color"),
            ctl("accent_color", "Accent Color", 2, "fill_color"),
            ctl("name_color", "Name Color", 3, "fill_color"),
            ctl("show_role", "Show Role", 4, "enabled"),
            ctl("font", "Font", 3, "font"),
        ],
    );
    responsive(&mut t, 900, 700);
    t
}

/// A title that pops in with an overshoot, an accent rule drawing itself underneath.
fn title_pop() -> GraphicsTemplate {
    let mut title = text(1, "Title", "BIG REVEAL", (CX, 520.0), 150.0, "Bold", 1, WHITE, true);
    set(&mut title, "tracking", f(20.0));
    shadow(&mut title, 6.0, 24.0, 50.0);
    animate(
        &mut title,
        "scale",
        &[(0, f(0.0), Ease::Linear), (220, f(115.0), Ease::Smooth), (380, f(100.0), Ease::Linear), (3600, f(100.0), Ease::Leave), (4000, f(0.0), Ease::Linear)],
    );
    animate(&mut title, "opacity", &[(0, f(0.0), Ease::Linear), (100, f(100.0), Ease::Linear), (3850, f(100.0), Ease::Linear), (4000, f(0.0), Ease::Linear)]);
    let mut rule = shape(2, "Rule", 0, (CX, 618.0), (260.0, 10.0), vec![], ORANGE);
    animate(
        &mut rule,
        "size",
        &[(250, v2(0.0, 10.0), Ease::Linear), (600, v2(260.0, 10.0), Ease::Arrive), (3400, v2(260.0, 10.0), Ease::Leave), (3700, v2(0.0, 10.0), Ease::Linear)],
    );
    let mut t = template(
        "text.title-pop",
        "Title – Pop",
        ANIMATED_TEXT,
        "A big title that pops in with a little overshoot while an accent rule draws itself underneath.",
        4000,
        vec![title, rule],
        vec![
            ctl("title", "Title", 1, "text"),
            ctl("title_color", "Title Color", 1, "fill_color"),
            ctl("accent", "Accent Color", 2, "fill_color"),
            ctl("font", "Font", 1, "font"),
        ],
    );
    responsive(&mut t, 600, 600);
    t
}

/// Three words, one at a time, each punching in.
fn kinetic_words() -> GraphicsTemplate {
    let words = [("MAKE", WHITE), ("IT", YELLOW), ("COUNT", WHITE)];
    let starts = [0i64, 600, 1200];
    let end = 3200;
    let mut layers = Vec::new();
    for (i, ((w, color), &t0)) in words.iter().zip(&starts).enumerate() {
        let mut l = text(i as u64 + 1, &format!("Word {}", i + 1), w, (CX, CY), 190.0, "Bold", 1, *color, true);
        shadow(&mut l, 6.0, 20.0, 55.0);
        let next = starts.get(i + 1).copied();
        let mut keys = vec![(0, f(0.0), Ease::Hold), (t0, f(100.0), Ease::Hold)];
        match next {
            Some(t1) => keys.push((t1, f(0.0), Ease::Hold)),
            None => {
                if let Some(k) = keys.last_mut() {
                    k.2 = Ease::Linear;
                }
                keys.push((end - 300, f(100.0), Ease::Linear));
                keys.push((end, f(0.0), Ease::Linear));
            }
        }
        if t0 == 0 {
            keys.remove(0);
        }
        animate(&mut l, "opacity", &keys);
        animate(&mut l, "scale", &[(t0, f(135.0), Ease::Linear), (t0 + 180, f(100.0), Ease::Arrive)]);
        layers.push(l);
    }
    let mut t = template(
        "text.kinetic-words",
        "Kinetic – Word by Word",
        ANIMATED_TEXT,
        "Three words, one at a time, each punching in; the last one stays.",
        end,
        layers,
        vec![
            ctl("word1", "Word 1", 1, "text"),
            ctl("word2", "Word 2", 2, "text"),
            ctl("word3", "Word 3", 3, "text"),
            ctl("accent", "Accent Color", 2, "fill_color"),
        ],
    );
    responsive(&mut t, 1400, 300);
    t
}

/// A caption on a rounded box that rises and fades in.
fn caption_box() -> GraphicsTemplate {
    let y = 930.0;
    let mut c = text(1, "Caption", "Add your caption here", (CX, y), 54.0, "Medium", 1, WHITE, false);
    set(&mut c, "background", ParamValue::Bool(true));
    set(&mut c, "background_color", rgb(INK));
    set(&mut c, "background_opacity", f(80.0));
    set(&mut c, "background_size", f(22.0));
    set(&mut c, "background_radius", f(14.0));
    fade(&mut c, 0, 300, 3700, 4000);
    animate(&mut c, "position", &[(0, v2(CX, y + 26.0), Ease::Linear), (350, v2(CX, y), Ease::Arrive)]);
    let mut t = template(
        "text.caption-box",
        "Caption Box",
        ANIMATED_TEXT,
        "A caption on a rounded box near the bottom of the frame; it rises and fades in, and fades out.",
        4000,
        vec![c],
        vec![
            ctl("caption", "Caption", 1, "text"),
            ctl("text_color", "Text Color", 1, "fill_color"),
            ctl("box_color", "Box Color", 1, "background_color"),
            slider("box_opacity", "Box Opacity", 1, "background_opacity", 0.0, 100.0),
            ctl("font", "Font", 1, "font"),
        ],
    );
    responsive(&mut t, 350, 300);
    t
}

/// 3, 2, 1, GO! with a pulsing ring.
fn countdown() -> GraphicsTemplate {
    let mut disc = shape(1, "Disc", 1, (CX, CY), (300.0, 300.0), vec![], INK);
    set(&mut disc, "opacity", f(70.0));
    animate(&mut disc, "opacity", &[(0, f(70.0), Ease::Linear), (3000, f(70.0), Ease::Linear), (3400, f(0.0), Ease::Linear)]);
    let mut ring = shape(2, "Ring", 1, (CX, CY), (300.0, 300.0), vec![], WHITE);
    set(&mut ring, "fill", ParamValue::Bool(false));
    outline(&mut ring, WHITE, 14.0);
    let mut pulse = Vec::new();
    for s in [0i64, 1000, 2000] {
        pulse.push((s, f(122.0), Ease::Linear));
        pulse.push((s + 250, f(100.0), Ease::Arrive));
    }
    pulse.push((3000, f(100.0), Ease::Leave));
    pulse.push((3500, f(170.0), Ease::Linear));
    animate(&mut ring, "scale", &pulse);
    animate(&mut ring, "opacity", &[(0, f(100.0), Ease::Linear), (3000, f(100.0), Ease::Linear), (3500, f(0.0), Ease::Linear)]);
    let mut number = text(3, "Number", "3", (CX, CY), 170.0, "Bold", 1, WHITE, true);
    animate(
        &mut number,
        "text",
        &[
            (0, ParamValue::Text("3".into()), Ease::Hold),
            (1000, ParamValue::Text("2".into()), Ease::Hold),
            (2000, ParamValue::Text("1".into()), Ease::Hold),
            (3000, ParamValue::Text("GO!".into()), Ease::Hold),
        ],
    );
    let mut pop = Vec::new();
    for s in [0i64, 1000, 2000, 3000] {
        pop.push((s, f(135.0), Ease::Linear));
        pop.push((s + 250, f(100.0), Ease::Arrive));
    }
    animate(&mut number, "scale", &pop);
    animate(&mut number, "opacity", &[(0, f(100.0), Ease::Linear), (3700, f(100.0), Ease::Linear), (4000, f(0.0), Ease::Linear)]);
    template(
        "text.countdown",
        "Countdown – 3, 2, 1",
        ANIMATED_TEXT,
        "3, 2, 1 and GO! in a pulsing ring, one count a second.",
        4000,
        vec![disc, ring, number],
        vec![
            ctl("number_color", "Number Color", 3, "fill_color"),
            ctl("ring_color", "Ring Color", 2, "stroke_color"),
            ctl("disc_color", "Disc Color", 1, "fill_color"),
            ctl("show_disc", "Show Disc", 1, "enabled"),
            ctl("font", "Font", 3, "font"),
        ],
    )
}

/// A closing card: a thank-you heading and two frames for what to watch next.
fn end_card() -> GraphicsTemplate {
    let end = 8000;
    let mut heading = text(1, "Heading", "Thanks for watching", (CX, 230.0), 96.0, "Bold", 1, WHITE, true);
    shadow(&mut heading, 4.0, 18.0, 45.0);
    fade(&mut heading, 0, 500, end - 500, end);
    animate(&mut heading, "position", &[(0, v2(CX, 260.0), Ease::Linear), (500, v2(CX, 230.0), Ease::Arrive)]);
    let mut sub = text(2, "Subheading", "See you in the next one", (CX, 330.0), 44.0, "Regular", 1, 0xe5e7eb, true);
    fade(&mut sub, 200, 700, end - 500, end);
    let mut layers = vec![heading, sub];
    for (k, (x, label, delay)) in [(590.0, "Watch next", 400i64), (1330.0, "More videos", 550)].into_iter().enumerate() {
        let id = 3 + 2 * k as u64;
        let mut frame = shape(id, &format!("Frame {}", k + 1), 0, (x, 640.0), (640.0, 360.0), vec![], WHITE);
        set(&mut frame, "fill", ParamValue::Bool(false));
        set(&mut frame, "corner_radius", f(18.0));
        outline(&mut frame, WHITE, 6.0);
        fade(&mut frame, delay, delay + 500, end - 500, end);
        animate(&mut frame, "scale", &[(delay, f(90.0), Ease::Linear), (delay + 500, f(100.0), Ease::Arrive)]);
        let mut l = text(id + 1, &format!("Label {}", k + 1), label, (x, 872.0), 38.0, "SemiBold", 1, WHITE, true);
        fade(&mut l, delay + 200, delay + 600, end - 500, end);
        layers.push(frame);
        layers.push(l);
    }
    let mut t = template(
        "text.end-card",
        "End Card",
        ANIMATED_TEXT,
        "A thank-you heading and two framed spaces for what to watch next; place clips or stills under the frames.",
        end,
        layers,
        vec![
            ctl("heading", "Heading", 1, "text"),
            ctl("subheading", "Subheading", 2, "text"),
            ctl("label1", "Left Label", 4, "text"),
            ctl("label2", "Right Label", 6, "text"),
            ctl("heading_color", "Heading Color", 1, "fill_color"),
            ctl("font", "Font", 1, "font"),
        ],
    );
    responsive(&mut t, 1200, 500);
    t
}

/// Give text layer `e` animators (in, out, loop) — drawn letter, word or line at a time by the
/// renderer ([`crate::text_anim`]), saved with the template.
fn animators(e: &mut EffectInstance, enter: Option<Animator>, out: Option<Animator>, repeat: Option<Animator>) {
    e.layer.get_or_insert_with(Default::default).anim = Some(TextAnimation { enter, out, repeat });
}

fn anim(kind: AnimKind) -> Animator {
    Animator::preset(kind)
}

/// Controls of a one-layer animated title (layer 1): text, colour, font, size.
fn title_controls() -> Vec<TemplateControl> {
    vec![
        ctl("title", "Title", 1, "text"),
        ctl("title_color", "Title Color", 1, "fill_color"),
        ctl("font", "Font", 1, "font"),
        slider("size", "Size", 1, "scale", 10.0, 400.0),
    ]
}

/// A title typed letter by letter, erased the same way at the end.
fn typewriter() -> GraphicsTemplate {
    let mut t = text(1, "Title", "Once upon a time…", (CX, CY), 110.0, "SemiBold", 1, WHITE, true);
    shadow(&mut t, 4.0, 16.0, 50.0);
    animators(&mut t, Some(Animator { stagger: 0.07, ..anim(AnimKind::Typewriter) }), Some(Animator { stagger: 0.03, ..anim(AnimKind::Typewriter) }), None);
    let mut g = template(
        "text.typewriter",
        "Title – Typewriter",
        ANIMATED_TEXT,
        "A line typed letter by letter, then erased the same way.",
        5000,
        vec![t],
        title_controls(),
    );
    responsive(&mut g, 1500, 700);
    g
}

/// Each letter pops in with an overshoot; they shrink away in turn at the end.
fn pop_letters() -> GraphicsTemplate {
    let mut t = text(1, "Title", "POP IT", (CX, CY), 180.0, "Bold", 1, YELLOW, true);
    outline(&mut t, INK, 10.0);
    shadow(&mut t, 8.0, 10.0, 60.0);
    // designed with the overshoot: Back Out by name (Pop's own easing is quiet since PX5.2)
    let back = Animator { easing: crate::text_anim::curve::BACK_OUT, ..anim(AnimKind::Pop) };
    animators(&mut t, Some(back.clone()), Some(Animator { duration: 0.3, ..back }), None);
    let mut g = template(
        "text.pop-letters",
        "Title – Pop by Letter",
        ANIMATED_TEXT,
        "Each letter pops in with a little overshoot, and shrinks away in turn at the end.",
        4000,
        vec![t],
        title_controls(),
    );
    responsive(&mut g, 900, 600);
    g
}

/// Letters rising and falling in a travelling wave.
fn wave() -> GraphicsTemplate {
    let mut t = text(1, "Title", "Good vibes", (CX, CY), 150.0, "Bold", 1, WHITE, true);
    outline(&mut t, PINK, 6.0);
    animators(
        &mut t,
        Some(Animator { unit: AnimUnit::Word, ..anim(AnimKind::Fade) }),
        Some(Animator { unit: AnimUnit::Word, ..anim(AnimKind::Fade) }),
        Some(anim(AnimKind::Wave)),
    );
    let mut g = template(
        "text.wave",
        "Title – Wave",
        ANIMATED_TEXT,
        "A title whose letters ride a wave the whole time it is on screen.",
        5000,
        vec![t],
        title_controls(),
    );
    responsive(&mut g, 500, 500);
    g
}

/// A neon sign: a glowing title that flickers on and keeps a faint flicker.
fn neon() -> GraphicsTemplate {
    let mut t = text(1, "Title", "OPEN LATE", (CX, CY), 160.0, "Bold", 1, 0xe0fbff, true);
    set(&mut t, "glow", ParamValue::Bool(true));
    set(&mut t, "glow_color", rgb(0x22d3ee));
    set(&mut t, "glow_radius", f(40.0));
    set(&mut t, "glow_intensity", f(150.0));
    animators(
        &mut t,
        Some(Animator { stagger: 0.04, ..anim(AnimKind::Glitch) }),
        Some(Animator { unit: AnimUnit::Line, ..anim(AnimKind::Fade) }),
        Some(Animator { amount: 40.0, ..anim(AnimKind::Flicker) }),
    );
    let mut g = template(
        "text.neon-glow",
        "Neon Glow Title",
        ANIMATED_TEXT,
        "A glowing neon title that flickers on, keeps a faint flicker and fades at the end.",
        5000,
        vec![t],
        vec![
            ctl("title", "Title", 1, "text"),
            ctl("glow_color", "Glow Color", 1, "glow_color"),
            ctl("title_color", "Title Color", 1, "fill_color"),
            slider("glow_radius", "Glow Radius", 1, "glow_radius", 0.0, 200.0),
            ctl("font", "Font", 1, "font"),
        ],
    );
    responsive(&mut g, 800, 500);
    g
}

/// Words sliding up into place one after another, and down out at the end.
fn slide_words() -> GraphicsTemplate {
    let mut t = text(1, "Title", "Here is the thing", (CX, CY), 130.0, "Bold", 1, WHITE, true);
    shadow(&mut t, 6.0, 18.0, 55.0);
    animators(&mut t, Some(anim(AnimKind::SlideUp)), Some(Animator { stagger: 0.05, ..anim(AnimKind::SlideDown) }), None);
    let mut g = template(
        "text.slide-words",
        "Words – Slide Up",
        ANIMATED_TEXT,
        "Words slide up into place one after another, and drop away at the end.",
        4000,
        vec![t],
        title_controls(),
    );
    responsive(&mut g, 900, 700);
    g
}

/// A title on an arc that drops in letter by letter and bounces gently.
fn arc_badge() -> GraphicsTemplate {
    let mut t = text(1, "Title", "SUBSCRIBE FOR MORE", (CX, CY + 60.0), 96.0, "Bold", 1, WHITE, true);
    set(&mut t, "curve", f(55.0));
    outline(&mut t, RED, 8.0);
    set(&mut t, "inner_shadow", ParamValue::Bool(true));
    animators(
        &mut t,
        Some(Animator { stagger: 0.03, ..anim(AnimKind::Drop) }),
        Some(anim(AnimKind::Fade)),
        Some(Animator { amount: 60.0, unit: AnimUnit::Line, ..anim(AnimKind::Bounce) }),
    );
    let mut g = template(
        "text.arc-badge",
        "Arc Title – Drop In",
        ANIMATED_TEXT,
        "A title set on an arc whose letters drop in and bounce gently.",
        4000,
        vec![t],
        vec![
            ctl("title", "Title", 1, "text"),
            ctl("title_color", "Title Color", 1, "fill_color"),
            ctl("outline_color", "Outline Color", 1, "stroke_color"),
            slider("curve", "Curve", 1, "curve", -100.0, 100.0),
            ctl("font", "Font", 1, "font"),
        ],
    );
    responsive(&mut g, 1100, 500);
    g
}

/// The animated text presets (see the module docs).
pub fn text_presets() -> Vec<GraphicsTemplate> {
    vec![
        lower_third_slide(),
        title_pop(),
        kinetic_words(),
        caption_box(),
        countdown(),
        end_card(),
        typewriter(),
        pop_letters(),
        wave(),
        neon(),
        slide_words(),
        arc_badge(),
    ]
}

/// Every graphic element: stickers, then text presets.
pub fn all() -> Vec<GraphicsTemplate> {
    let mut v = stickers();
    v.extend(text_presets());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_is_a_valid_template_that_round_trips() {
        let all = all();
        assert_eq!(stickers().len(), 10);
        assert_eq!(text_presets().len(), 12);
        let mut ids: Vec<&str> = all.iter().map(|t| t.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), all.len(), "unique ids");
        for t in &all {
            assert!(t.id.starts_with(TEMPLATE_PREFIX), "{}", t.id);
            assert!(element_id(t).starts_with("sticker.") == (t.category == STICKERS), "{}", t.id);
            t.validate().unwrap_or_else(|e| panic!("{}: {e}", t.name));
            let back = GraphicsTemplate::from_bytes(t.to_json().as_bytes()).unwrap();
            assert_eq!(&back, t, "{}", t.id);
            assert_eq!((t.author.as_str(), t.license.as_str()), (AUTHOR, LICENSE));
            // only the bundled font
            for l in &t.layers {
                if let Some(p) = l.params.get("font") {
                    assert!(l.effect != crate::graphic::TEXT_LAYER || p.value == ParamValue::Text("Inter".into()), "{}", t.id);
                }
            }
        }
    }

    #[test]
    fn text_presets_animate_and_protect_their_intro_and_outro() {
        for t in text_presets() {
            // keyframes, or text animators (letters, words or lines)
            let animators = t.layers.iter().any(|l| l.layer.as_ref().and_then(|x| x.anim.as_ref()).is_some_and(|a| !a.is_empty()));
            assert!(animators || t.layers.iter().any(|l| l.params.values().any(|p| p.is_animated())), "{} is animated", t.id);
            for l in &t.layers {
                for p in l.params.values() {
                    assert!(p.keyframes.windows(2).all(|w| w[0].time < w[1].time), "{}: keyframes in order", t.id);
                    assert!(p.keyframes.iter().all(|k| k.time >= Tick::ZERO && k.time <= t.duration), "{}: keyframes inside the clip", t.id);
                }
            }
            if t.graphic.has_responsive_time() {
                assert_eq!(t.graphic.design_duration, t.duration, "{}", t.id);
                assert!(t.graphic.intro + t.graphic.outro <= t.duration, "{}", t.id);
            }
        }
        // the countdown counts
        let c = text_presets().into_iter().find(|t| t.id == "element:text.countdown").unwrap();
        let number = c.layers.iter().find(|l| l.effect == crate::graphic::TEXT_LAYER).unwrap();
        let at = |s: i64| number.params["text"].value_at(ms(s));
        assert_eq!(
            (at(500), at(1500), at(2500), at(3500)),
            (ParamValue::Text("3".into()), ParamValue::Text("2".into()), ParamValue::Text("1".into()), ParamValue::Text("GO!".into()))
        );
        // one word at a time
        let k = text_presets().into_iter().find(|t| t.id == "element:text.kinetic-words").unwrap();
        for (s, want) in [(300, 0usize), (900, 1), (1500, 2), (3000, 2)] {
            let shown: Vec<usize> = (0..3).filter(|&i| k.layers[i].params["opacity"].f64_at(ms(s)) > 50.0).collect();
            assert_eq!(shown, vec![want], "at {s} ms");
        }
    }

    #[test]
    fn stickers_are_static_closed_shapes_on_the_canvas() {
        for t in stickers() {
            assert_eq!(t.duration, ms(STICKER_MS));
            for l in &t.layers {
                assert!(l.params.values().all(|p| !p.is_animated()), "{} is static", t.id);
                if let Some(ParamValue::Curve(pts)) = l.params.get("points").map(|p| &p.value)
                    && l.params.get("shape").map(|p| &p.value) == Some(&ParamValue::Choice(3))
                {
                    assert!(pts.len() >= 3, "{}", t.id);
                    let pos = l.params["position"].value.as_vec2().unwrap();
                    for p in pts {
                        let (x, y) = (pos.x + f64::from(p[0]), pos.y + f64::from(p[1]));
                        assert!((0.0..1920.0).contains(&x) && (0.0..1080.0).contains(&y), "{}: {x},{y} on the canvas", t.id);
                    }
                }
            }
        }
    }
}
