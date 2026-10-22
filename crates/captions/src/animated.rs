//! Animated captions: a few words on screen at a time (a *page*), the word being spoken
//! highlighted — the look of [`CaptionPreset`]s (Bold, Clean, Glow, Box, Karaoke, Two-Line
//! Subtitle) with the options of [`AnimatedCaptions`].
//!
//! - **Words and timing** come from [`Caption::timed_words`]: the caption's word timing (from
//!   the transcript, a narration or ASS karaoke) or, without it, the caption's time spread over its
//!   words by length. Any script the text engine shapes works word by word, Malayalam included.
//! - **Pages** ([`pages`]): words fill a page up to the preset's words per page (2–4 for Bold),
//!   at most two lines as wide as the space allows; a sentence end or a pause of [`PAGE_PAUSE`]
//!   starts a new page; a single word too wide for the space gets a page of its own in a smaller
//!   size. The two-line subtitle shows the whole caption.
//! - **Placement**: with a platform ([`AnimatedCaptions::platform`]) the block is centred on the
//!   middle of the platform's safe span at its rows (plus the style's `x_offset`), as wide as the
//!   span at most, and the anchor margin counts from the safe area's edge — so on YouTube Shorts
//!   captions use the full safe width, not a width centred on the frame. Without one, the frame.
//! - **Highlight**: the spoken word (the last one that started) is drawn in the active colour and
//!   pops to `pop` × its size with an overshoot over [`POP_SECONDS`] (Bold); Glow lights it with a
//!   neon glow; Box sets it on a rounded box of the active colour; Karaoke sweeps the active
//!   colour through each word as it is spoken. Keywords are drawn in the keyword colour; emoji
//!   follow their words. A Bold, Glow or Box page pops in over [`PAGE_IN_SECONDS`].
//!
//! Preview and export both draw through [`crate::burn::sequence_overlays`], so the two look the
//! same. Colours are composited in linear light; the overlay is premultiplied RGBA.

use std::sync::Arc;

use filmcraft_project::easing::{Bezier, ease_overshoot};
use filmcraft_project::{AnimatedCaptions, Caption, CaptionAlign, CaptionAnchor, CaptionPreset, CaptionStyle};
use filmcraft_text::{Layout, Mask, ParagraphStyle, StrokeKind, TextStyle, layout, mask, measure, render};
use filmcraft_time::Tick;

use crate::burn::{Overlay, lin};

/// A pause between words at least this long starts a new page, seconds.
pub const PAGE_PAUSE: f64 = 0.6;
/// How long the spoken word takes to pop to its size, seconds.
pub const POP_SECONDS: f64 = 0.12;
/// How long a page takes to pop in, seconds.
pub const PAGE_IN_SECONDS: f64 = 0.1;
/// "Back Out" in `EASING_OPTS`: an overshoot that settles.
const BACK_OUT: u32 = 23;
/// Most pages one caption is cut into.
const MAX_PAGES: usize = 500;
/// Largest overlay drawn, pixels.
const MAX_OVERLAY_PIXELS: usize = 32 << 20;

/// A word placed on a page.
#[derive(Clone, Debug)]
pub struct PageWord {
    /// As shown (capitals for Bold, its emoji after it).
    pub text: String,
    pub start: Tick,
    pub end: Tick,
    /// Left edge of the word's box (its ink or its advance, whichever reaches further), its
    /// baseline and its width, frame pixels; the text's origin is `dx` right of the left edge.
    pub x: f32,
    pub baseline: f32,
    pub width: f32,
    pub dx: f32,
    pub keyword: bool,
    layout: Arc<Layout>,
}

/// A page of a caption: the words on screen together.
#[derive(Clone, Debug)]
pub struct Page {
    pub start: Tick,
    pub end: Tick,
    pub words: Vec<PageWord>,
    /// Font size, pixels (smaller than the track's when a word is wider than the space).
    pub size: f32,
    /// The text block `[x0, y0, x1, y1]`, frame pixels.
    pub rect: [f32; 4],
}

/// How a preset draws (sizes in ems of the page's font size, unless said).
struct Look {
    weight: &'static str,
    /// Drop shadow: (down, blur, opacity).
    shadow: Option<(f32, f32, f32)>,
    glow: bool,
    boxed: bool,
    karaoke: bool,
    page_pop: bool,
}

fn look(p: CaptionPreset) -> Look {
    match p {
        CaptionPreset::Bold => Look { weight: "Black", shadow: Some((0.07, 0.03, 0.75)), glow: false, boxed: false, karaoke: false, page_pop: true },
        CaptionPreset::Clean => Look { weight: "SemiBold", shadow: Some((0.04, 0.16, 0.65)), glow: false, boxed: false, karaoke: false, page_pop: false },
        CaptionPreset::Glow => Look { weight: "Bold", shadow: Some((0.03, 0.1, 0.5)), glow: true, boxed: false, karaoke: false, page_pop: true },
        CaptionPreset::Box => Look { weight: "Bold", shadow: Some((0.04, 0.12, 0.55)), glow: false, boxed: true, karaoke: false, page_pop: true },
        CaptionPreset::Karaoke => Look { weight: "Bold", shadow: Some((0.04, 0.12, 0.6)), glow: false, boxed: false, karaoke: true, page_pop: false },
        CaptionPreset::TwoLine => Look { weight: "SemiBold", shadow: Some((0.03, 0.08, 0.7)), glow: false, boxed: false, karaoke: false, page_pop: false },
    }
}

/// The face of a preset at `px`: the track's family (Inter when empty) in the preset's weight
/// (Bold asks for Black: a family that has it — an installed one the user picked — gets it;
/// Inter gives its Bold).
fn text_style(style: &CaptionStyle, preset: CaptionPreset, px: f32) -> TextStyle {
    let family = if style.font.trim().is_empty() { filmcraft_text::fonts::DEFAULT_FAMILY } else { style.font.trim() };
    TextStyle { family: family.into(), style: look(preset).weight.into(), size: px, ..Default::default() }
}

fn is_rtl(c: char) -> bool {
    matches!(c as u32, 0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF)
}

/// Where the block goes: `(left, right)` of the space it may use, its centre, and the rows it
/// may use `(top, bottom)`, frame pixels.
struct Space {
    left: f32,
    right: f32,
    centre: f32,
    top: f32,
    bottom: f32,
}

fn space(style: &CaptionStyle, anim: &AnimatedCaptions, w: f32, h: f32, block_h: f32) -> Space {
    let (rows, span) = match anim.platform {
        Some(p) => {
            let a = p.safe_area();
            let (top, bottom) = (a.y0 as f32 * h, a.y1 as f32 * h);
            let margin = margin_of(style) * h;
            let (y0, y1) = block_rows(style.anchor, top, bottom, margin, block_h);
            let (l, r) = p.safe_span(f64::from(y0 / h), f64::from(y1 / h));
            ((top, bottom), (l as f32 * w, r as f32 * w))
        }
        None => ((0.0, h), (0.05 * w, 0.95 * w)),
    };
    let off = if style.x_offset.is_finite() { style.x_offset.clamp(-0.5, 0.5) * w } else { 0.0 };
    let centre = ((span.0 + span.1) / 2.0 + off).clamp(span.0, span.1);
    // the block stays inside the span around its (offset) centre
    let half = (centre - span.0).min(span.1 - centre).max(0.0);
    Space { left: centre - half, right: centre + half, centre, top: rows.0, bottom: rows.1 }
}

/// The block's top and bottom for an anchor inside rows `top..bottom`.
fn block_rows(anchor: CaptionAnchor, top: f32, bottom: f32, margin: f32, block_h: f32) -> (f32, f32) {
    let y0 = match anchor {
        CaptionAnchor::Top => top + margin,
        CaptionAnchor::Middle => (top + bottom - block_h) / 2.0,
        CaptionAnchor::Bottom => bottom - margin - block_h,
    };
    let y0 = y0.min(bottom - block_h).max(top.min(bottom - block_h));
    (y0, y0 + block_h)
}

/// Lines of a page: word indices, greedily filled to `max_w` (`breaks`: a written line ends
/// before this word).
/// How far a laid-out word reaches: from `x0` (≤ 0, ink left of the origin) to `x1` (its advance
/// or its ink, whichever is further), and its ink above and below the baseline.
#[derive(Clone, Copy, Debug, Default)]
struct Extent {
    x0: f32,
    x1: f32,
    ascent: f32,
    descent: f32,
}

impl Extent {
    fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }
}

fn extent(l: &Layout) -> Extent {
    let adv = l.lines.iter().map(|ln| ln.x + ln.width).fold(0.0f32, f32::max);
    let mut e = Extent { x0: 0.0, x1: adv, ascent: 0.0, descent: 0.0 };
    for g in &l.glyphs {
        let ink = render::glyph_mask(g, g.size, 0.0)
            .map(|m| (m.left as f32, m.top as f32, m.mask.w as f32, m.mask.h as f32))
            .or_else(|| render::color_glyph(g, g.size).map(|c| (c.left, c.top, c.w as f32, c.h as f32)));
        if let Some((left, top, w, h)) = ink {
            e.x0 = e.x0.min(g.x + left);
            e.x1 = e.x1.max(g.x + left + w);
            e.ascent = e.ascent.max(-(g.y + top));
            e.descent = e.descent.max(g.y + top + h);
        }
    }
    e
}

fn wrap_lines(widths: &[f32], breaks: &[bool], space_w: f32, max_w: f32) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cw = 0.0f32;
    for (i, &wd) in widths.iter().enumerate() {
        let next = if cur.is_empty() { wd } else { cw + space_w + wd };
        if !cur.is_empty() && (next > max_w || breaks.get(i).copied().unwrap_or(false)) {
            lines.push(std::mem::take(&mut cur));
            cw = wd;
        } else {
            cw = next;
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// The pages of caption `c` on a `w`×`h` frame (see the module docs).
pub fn pages(c: &Caption, style: &CaptionStyle, anim: &AnimatedCaptions, w: usize, h: usize) -> Vec<Page> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let (wf, hf) = (w as f32, h as f32);
    let scale = wf.min(hf) / 1080.0;
    let size = if style.size.is_finite() { style.size.clamp(4.0, 400.0) } else { 54.0 };
    let px = (size * scale).max(4.0);
    let words = c.timed_words();
    if words.is_empty() {
        return Vec::new();
    }
    let upper = anim.is_uppercase();
    let shown: Vec<String> = words
        .iter()
        .map(|tw| {
            let mut t = if upper { tw.text.to_uppercase() } else { tw.text.clone() };
            if let Some(e) = anim.emoji_for(&tw.text) {
                t.push(' ');
                t.push_str(e);
            }
            t
        })
        .collect();
    let preset = anim.preset;
    let ts = text_style(style, preset, px);
    let lh_k = if style.line_spacing.is_finite() { style.line_spacing.clamp(0.8, 4.0) } else { 1.2 };
    // the space at the rows of a two-line block
    let sp = space(style, anim, wf, hf, 2.0 * px * lh_k);
    let max_w = ((sp.right - sp.left) * 0.96).max(px);
    let widths: Vec<f32> = shown.iter().map(|s| extent(&layout(s, &ts, &ParagraphStyle::default())).width()).collect();
    // a line's width leaves out trailing spaces: the space is the difference
    let space_w = (measure("n n", &ts) - measure("nn", &ts)).max(px * 0.2);
    let per = anim.page_words();
    let max_lines = if per == 0 { usize::MAX } else { 2 };
    // cut into pages
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in 0..words.len() {
        if let (Some(prev), Some(now)) = (i.checked_sub(1).and_then(|p| words.get(p)), words.get(i))
            && !cur.is_empty()
        {
            let gap = secs(prev.end, now.start);
            let sentence = prev.text.trim_end_matches(['"', '\'', ')', '”', '’']).ends_with(['.', '!', '?', '…', '।']);
            let full = per > 0 && cur.len() >= per;
            let breaks = per > 0 && (sentence || gap >= PAGE_PAUSE);
            let mut cand: Vec<f32> = cur.iter().filter_map(|&k| widths.get(k).copied()).collect();
            cand.push(widths.get(i).copied().unwrap_or(0.0));
            let fits = wrap_lines(&cand, &[], space_w, max_w).len() <= max_lines;
            if full || breaks || !fits {
                groups.push(std::mem::take(&mut cur));
                if groups.len() >= MAX_PAGES {
                    break;
                }
            }
        }
        cur.push(i);
    }
    if !cur.is_empty() && groups.len() < MAX_PAGES {
        groups.push(cur);
    }
    let rtl = words.iter().any(|w| w.text.chars().any(is_rtl));
    let mut out: Vec<Page> = Vec::with_capacity(groups.len());
    for (gi, g) in groups.iter().enumerate() {
        // a word wider than the space: this page in a smaller size
        let widest = g.iter().filter_map(|&k| widths.get(k)).fold(0.0f32, |a, b| a.max(*b));
        let k = if widest > max_w { (max_w / widest).max(0.2) } else { 1.0 };
        let page_px = px * k;
        let pts = TextStyle { size: page_px, ..ts.clone() };
        let ws: Vec<f32> = g.iter().map(|&i| widths.get(i).copied().unwrap_or(0.0) * k).collect();
        let breaks: Vec<bool> = g
            .iter()
            .enumerate()
            .map(|(j, &i)| j > 0 && per == 0 && words.get(i).map(|w| w.line) != g.get(j - 1).and_then(|p| words.get(*p)).map(|w| w.line))
            .collect();
        let lines = wrap_lines(&ws, &breaks, space_w * k, max_w);
        // the words at the page's size, and how far their ink reaches (tall scripts, emoji)
        let lays: Vec<(Arc<Layout>, Extent)> = g
            .iter()
            .map(|&i| {
                let l = layout(shown.get(i).map_or("", String::as_str), &pts, &ParagraphStyle::default());
                let e = extent(&l);
                (l, e)
            })
            .collect();
        let vm = filmcraft_text::fonts::face(filmcraft_text::resolve(&pts.family, &pts.style).face).metrics(page_px);
        let asc = lays.iter().map(|(_, e)| e.ascent).fold(vm.cap_height, f32::max);
        let desc = lays.iter().map(|(_, e)| e.descent).fold(0.0, f32::max);
        // lines a line height apart, or further when the ink needs it
        let lh = (page_px * lh_k).max(asc + desc + 0.12 * page_px);
        let block_h = lh * lines.len() as f32;
        let sp = space(style, anim, wf, hf, block_h);
        let margin = margin_of(style) * hf;
        let (top, _) = block_rows(style.anchor, sp.top, sp.bottom, margin, block_h);
        let mut placed: Vec<PageWord> = Vec::with_capacity(g.len());
        let mut rect = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for (li, line) in lines.iter().enumerate() {
            let lw: f32 = line.iter().filter_map(|&j| lays.get(j)).map(|(_, e)| e.width()).sum::<f32>() + space_w * k * line.len().saturating_sub(1) as f32;
            let x0 = match style.align {
                CaptionAlign::Center => sp.centre - lw / 2.0,
                CaptionAlign::Left => sp.left,
                CaptionAlign::Right => sp.right - lw,
            };
            // the ink of the line centred in its line height
            let baseline = (top + lh * li as f32 + (lh - (asc + desc)) / 2.0 + asc).round();
            let order: Vec<usize> = if rtl { line.iter().rev().copied().collect() } else { line.clone() };
            let mut x = x0;
            for j in order {
                let (Some(&i), Some((l, e))) = (g.get(j), lays.get(j)) else { continue };
                let (Some(tw), Some(text)) = (words.get(i), shown.get(i)) else { continue };
                let wd = e.width();
                placed.push(PageWord {
                    text: text.clone(),
                    start: tw.start,
                    end: tw.end,
                    x,
                    dx: -e.x0,
                    baseline,
                    width: wd,
                    keyword: anim.is_keyword(&tw.text),
                    layout: l.clone(),
                });
                rect = [rect[0].min(x), rect[1].min(baseline - e.ascent.max(vm.cap_height)), rect[2].max(x + wd), rect[3].max(baseline + e.descent)];
                x += wd + space_w * k;
            }
        }
        // keep reading order for highlighting: placed words are in visual order on RTL lines
        placed.sort_by_key(|p| p.start);
        let start = if gi == 0 { c.start } else { placed.first().map_or(c.start, |p| p.start) };
        out.push(Page { start, end: Tick(c.start.0.saturating_add(c.duration.0)), words: placed, size: page_px, rect });
    }
    for i in 1..out.len() {
        let next = out[i].start;
        if let Some(prev) = out.get_mut(i - 1) {
            prev.end = next.max(prev.start);
        }
    }
    out
}

/// The text block of every page of `c`: `(start, end, [x0, y0, x1, y1])` frame pixels, grown by
/// the outline and the spoken word's pop (for safe-zone checks).
pub fn page_rects(c: &Caption, style: &CaptionStyle, anim: &AnimatedCaptions, w: usize, h: usize) -> Vec<(Tick, Tick, [f32; 4])> {
    let scale = w.min(h) as f32 / 1080.0;
    pages(c, style, anim, w, h)
        .into_iter()
        .map(|p| {
            let widest = p.words.iter().map(|w| w.width).fold(0.0f32, f32::max);
            let grow = outline_px(style, scale, p.size) + (anim.pop_scale() - 1.0) * (p.size * 0.6 + widest * 0.55);
            (p.start, p.end, [p.rect[0] - grow, p.rect[1] - grow, p.rect[2] + grow, p.rect[3] + grow])
        })
        .collect()
}

/// The outline width of a page set at `page_px` (a page set smaller keeps it in proportion).
fn outline_px(style: &CaptionStyle, scale: f32, page_px: f32) -> f32 {
    let base = if style.outline.is_finite() { style.outline.clamp(0.0, 40.0) } else { 0.0 } * scale;
    let full = (if style.size.is_finite() { style.size.clamp(4.0, 400.0) } else { 54.0 } * scale).max(4.0);
    base * (page_px / full).min(1.0)
}

/// Seconds from `a` to `b` (saturating: times from files are hostile).
fn secs(a: Tick, b: Tick) -> f64 {
    Tick(b.0.saturating_sub(a.0)).seconds()
}

/// The style's margin as a share of the frame height, 0..0.45.
fn margin_of(style: &CaptionStyle) -> f32 {
    if style.margin.is_finite() { style.margin.clamp(0.0, 0.45) } else { 0.08 }
}

fn over(p: &mut [f32], c: [f32; 4], a: f32) {
    if a > 0.0 {
        for k in 0..4 {
            p[k] = c[k] * a + p[k] * (1.0 - c[3] * a);
        }
    }
}

fn mix(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let f = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round().clamp(0.0, 255.0) as u8;
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

/// Animated caption `c` at timeline time `t` for a `w`×`h` frame (None when nothing shows).
pub fn render(c: &Caption, style: &CaptionStyle, anim: &AnimatedCaptions, w: usize, h: usize, t: Tick) -> Option<Overlay> {
    let pages = pages(c, style, anim, w, h);
    let page = pages.iter().rev().find(|p| p.start <= t).or(pages.first())?;
    let preset = anim.preset;
    let lk = look(preset);
    let px = page.size;
    let scale = w.min(h) as f32 / 1080.0;
    let outline = outline_px(style, scale, px);
    let pop = anim.pop_scale();
    // the spoken word: the last one that started (none before the first)
    let active = page.words.iter().rposition(|wd| wd.start <= t).filter(|_| preset != CaptionPreset::TwoLine);
    let eased = |p: f64| ease_overshoot(BACK_OUT, Bezier::EASE, p) as f32;
    let page_scale = if lk.page_pop { 0.82 + 0.18 * eased((secs(page.start, t) / PAGE_IN_SECONDS).clamp(0.0, 1.0)) } else { 1.0 };
    // the overlay: the block grown by everything drawn around it
    let (sh_dy, sh_blur) = lk.shadow.map_or((0.0, 0.0), |(dy, b, _)| (dy * px, b * px));
    let glow_r = if lk.glow { 0.6 * px } else { 0.0 };
    let widest = page.words.iter().map(|w| w.width).fold(0.0f32, f32::max);
    let pad = outline + sh_dy.abs() + sh_blur * 3.0 + glow_r * 2.0 + (pop - 1.0).max(0.0) * 1.15 * (px + widest * 0.6) + 0.3 * px + 2.0;
    let r = page.rect;
    let ox = (r[0] - pad).floor() as i32;
    let oy = (r[1] - pad).floor() as i32;
    let ow = ((r[2] - r[0]) + 2.0 * pad).ceil().max(1.0) as usize;
    let oh = ((r[3] - r[1]) + 2.0 * pad).ceil().max(1.0) as usize;
    if !r.iter().all(|v| v.is_finite()) || ow.saturating_mul(oh) > MAX_OVERLAY_PIXELS {
        return None;
    }
    let bc = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0);
    let base_c = style.color;
    let active_c = anim.active_color;
    // fill groups by colour (sRGB), and the spoken word's own mask (glow, box)
    let mut fills: Vec<([u8; 4], Mask)> = Vec::new();
    let add_to = |fills: &mut Vec<([u8; 4], Mask)>, col: [u8; 4], m: &Mask| match fills.iter_mut().find(|(c, _)| *c == col) {
        Some((_, f)) => f.max_with(m),
        None => fills.push((col, m.clone())),
    };
    let mut cover = Mask::new(ow, oh);
    let mut active_mask: Option<Mask> = None;
    let mut active_box: Option<[f32; 4]> = None;
    // emoji drawn from their pictures, on top, at their overlay positions
    let mut pictures: Vec<(Arc<render::ColorGlyph>, i32, i32)> = Vec::new();
    let pop_of = |wd: &PageWord| if pop > 1.0 { 1.0 + (pop - 1.0) * eased((secs(wd.start, t) / POP_SECONDS).clamp(0.0, 1.0)) } else { 1.0 };
    // the spoken word grows; the words beside it on its line move aside to keep their spacing
    let grow = active.and_then(|a| page.words.get(a)).map(|a| (a.baseline, a.x, (pop_of(a) - 1.0) * a.width / 2.0));
    for (i, wd) in page.words.iter().enumerate() {
        let is_active = active == Some(i);
        let ws = if is_active { pop_of(wd) } else { 1.0 };
        let s = page_scale * ws;
        let aside = match grow {
            Some((b, ax, half)) if !is_active && (b - wd.baseline).abs() < 0.5 => {
                if wd.x < ax {
                    -half
                } else {
                    half
                }
            }
            _ => 0.0,
        };
        let wc = (wd.x + aside + wd.width / 2.0, wd.baseline - 0.35 * px);
        let tx = bc.0 + page_scale * (wc.0 + ws * (wd.x + aside + wd.dx - wc.0) - bc.0) - ox as f32;
        // the box's left edge, where the text starts
        let left = tx - wd.dx * s;
        let ty = bc.1 + page_scale * (wc.1 + ws * (wd.baseline - wc.1) - bc.1) - oy as f32;
        let mut m = Mask::new(ow, oh);
        render::draw(&wd.layout, &[s, 0.0, 0.0, s, tx, ty], &mut m, (0, 0));
        for g in &wd.layout.glyphs {
            if let Some(cg) = render::color_glyph(g, g.size * s) {
                let (gx, gy) = ((tx + g.x * s + cg.left).round() as i32, (ty + g.y * s + cg.top).round() as i32);
                // the picture's outline and shadow come with the text's
                let mut pm = Mask::new(ow, oh);
                let alpha = Mask { w: cg.w, h: cg.h, a: cg.rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect() };
                pm.add(&alpha, gx, gy);
                cover.max_with(&pm);
                pictures.push((cg, gx, gy));
            }
        }
        cover.max_with(&m);
        if lk.karaoke {
            // sung words in the active colour; the word being sung swept left to right
            let f = if t >= wd.end {
                1.0
            } else if t < wd.start {
                0.0
            } else {
                (secs(wd.start, t) / secs(wd.start, wd.end).max(1e-6)).clamp(0.0, 1.0) as f32
            };
            let split = left + f * wd.width * s;
            let (mut sung, mut rest) = (Mask::new(ow, oh), Mask::new(ow, oh));
            for y in 0..oh {
                for x in 0..ow {
                    let j = y * ow + x;
                    let a = m.a.get(j).copied().unwrap_or(0.0);
                    let k = ((split - x as f32).clamp(0.0, 1.0)) * a;
                    if let (Some(sv), Some(rv)) = (sung.a.get_mut(j), rest.a.get_mut(j)) {
                        *sv = k;
                        *rv = a - k;
                    }
                }
            }
            add_to(&mut fills, active_c, &sung);
            add_to(&mut fills, if wd.keyword { anim.keyword_color } else { base_c }, &rest);
            continue;
        }
        let col = if is_active {
            match preset {
                CaptionPreset::Box => base_c,
                CaptionPreset::Glow => mix(active_c, [255, 255, 255, 255], 0.55),
                _ => active_c,
            }
        } else if wd.keyword {
            anim.keyword_color
        } else {
            base_c
        };
        if is_active {
            let bx = 0.14 * px * s;
            let by = 0.08 * px * s;
            active_box = Some([left - bx, ty - 0.8 * px * s - by, left + wd.width * s + bx, ty + 0.2 * px * s + by]);
            active_mask = Some(m.clone());
        }
        add_to(&mut fills, col, &m);
    }
    let stroke = (outline > 0.05).then(|| mask::stroke(&cover, outline, StrokeKind::Outer));
    let mut out = vec![0.0f32; ow * oh * 4];
    // background boxes behind each line (the track's box)
    if style.background && style.background_color[3] > 0 {
        let bgc = lin(style.background_color);
        let mut lines: Vec<(f32, f32, f32)> = Vec::new();
        for wd in &page.words {
            match lines.iter_mut().find(|(b, _, _)| (*b - wd.baseline).abs() < 0.5) {
                Some(l) => {
                    l.1 = l.1.min(wd.x);
                    l.2 = l.2.max(wd.x + wd.width);
                }
                None => lines.push((wd.baseline, wd.x, wd.x + wd.width)),
            }
        }
        let mut bm = Mask::new(ow, oh);
        for (b, x0, x1) in lines {
            let p = 0.25 * px;
            let path =
                filmcraft_text::Path::round_rect(x0 - p - ox as f32, b - 0.95 * px - oy as f32, x1 + p - ox as f32, b + 0.35 * px - oy as f32, 0.18 * px);
            filmcraft_text::raster::fill_into(&path, &mut bm, 0.0, 0.0);
        }
        for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&bm.a) {
            over(p, bgc, *a);
        }
    }
    // drop shadow of the text and its outline
    if let Some((_, _, op)) = lk.shadow {
        let mut u = cover.clone();
        if let Some(sm) = &stroke {
            u.max_with(sm);
        }
        let sm = mask::blur(&mask::offset(&u, 0, sh_dy.round() as i32), sh_blur / 1.5);
        let sc = lin([0, 0, 0, (op * 255.0).round() as u8]);
        for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&sm.a) {
            over(p, sc, *a);
        }
    }
    // the spoken word's box
    if lk.boxed
        && let Some(b) = active_box
    {
        let mut bm = Mask::new(ow, oh);
        filmcraft_text::raster::fill_into(&filmcraft_text::Path::round_rect(b[0], b[1], b[2], b[3], 0.2 * px), &mut bm, 0.0, 0.0);
        let bcol = lin(active_c);
        for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&bm.a) {
            over(p, bcol, *a);
        }
    }
    // the spoken word's neon glow
    if lk.glow
        && let Some(am) = &active_mask
    {
        // a wide soft halo and a tight bright one, like a neon tube
        let gc = lin(active_c);
        for (r, gain) in [(glow_r, 1.6f32), (glow_r * 0.35, 2.4)] {
            let mut g = mask::blur(am, (r / 2.5).max(0.5));
            g.a.iter_mut().for_each(|v| *v = (*v * gain).min(1.0));
            for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&g.a) {
                over(p, gc, *a);
            }
        }
    }
    if let Some(sm) = &stroke {
        let oc = lin(style.outline_color);
        for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&sm.a) {
            over(p, oc, *a);
        }
    }
    for (col, m) in &fills {
        let fc = lin(*col);
        for (p, a) in out.as_chunks_mut::<4>().0.iter_mut().zip(&m.a) {
            over(p, fc, *a);
        }
    }
    for (cg, gx, gy) in &pictures {
        for y in 0..cg.h {
            let oyy = *gy + y as i32;
            if oyy < 0 || oyy >= oh as i32 {
                continue;
            }
            for x in 0..cg.w {
                let oxx = *gx + x as i32;
                if oxx < 0 || oxx >= ow as i32 {
                    continue;
                }
                let (Some(s), Some(d)) = (
                    cg.rgba.get((y * cg.w + x) * 4..(y * cg.w + x) * 4 + 4),
                    out.get_mut((oyy as usize * ow + oxx as usize) * 4..(oyy as usize * ow + oxx as usize) * 4 + 4),
                ) else {
                    continue;
                };
                let k = 1.0 - s[3];
                for c in 0..4 {
                    d[c] = s[c] + d[c] * k;
                }
            }
        }
    }
    Some(Overlay { x: ox, y: oy, w: ow, h: oh, px: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::safe_zones::Platform;
    use filmcraft_project::{CaptionWord, ClipId};

    const S: i64 = filmcraft_time::TICKS_PER_SECOND;

    fn cap(text: &str, secs: f64) -> Caption {
        Caption {
            id: ClipId(1),
            start: Tick(0),
            duration: Tick::from_seconds_f64(secs),
            text: text.into(),
            speaker: None,
            cue_id: None,
            settings: String::new(),
            words: Vec::new(),
        }
    }

    /// Words one every 0.4 s, each 0.35 s long.
    fn timed(text: &str) -> Caption {
        let n = text.split_whitespace().count();
        let mut c = cap(text, 0.4 * n as f64 + 0.2);
        c.words = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| CaptionWord { text: w.into(), start: Tick(i as i64 * 4 * S / 10), end: Tick(i as i64 * 4 * S / 10 + 35 * S / 100) })
            .collect();
        c
    }

    fn bold() -> (CaptionStyle, AnimatedCaptions) {
        let a = AnimatedCaptions::preset(CaptionPreset::Bold);
        (CaptionStyle { animated: Some(a.clone()), ..CaptionPreset::Bold.base_style() }, a)
    }

    #[test]
    fn pages_of_two_to_four_words() {
        let (st, a) = bold();
        let c = timed("this is how you make your captions pop on every short");
        let p = pages(&c, &st, &a, 1080, 1920);
        assert!(p.iter().all(|p| (1..=3).contains(&p.words.len())), "{:?}", p.iter().map(|p| p.words.len()).collect::<Vec<_>>());
        assert_eq!(p.iter().map(|p| p.words.len()).sum::<usize>(), 11);
        assert!(p.iter().all(|p| p.words.iter().all(|w| w.text == w.text.to_uppercase())), "Bold is in capitals");
        // pages follow each other: each starts with its first word and ends when the next starts
        assert_eq!(p[0].start, Tick(0));
        assert_eq!(p[1].start, Tick(12 * S / 10));
        assert_eq!(p[0].end, p[1].start);
        assert_eq!(p.last().unwrap().end, c.end());
        // four words a page when asked, one line or two inside the space
        let four = AnimatedCaptions { words_per_page: 4, ..a.clone() };
        let p4 = pages(&c, &st, &four, 1080, 1920);
        assert!(p4.iter().all(|p| p.words.len() <= 4) && p4.iter().any(|p| p.words.len() == 4));
        // a sentence end and a pause start new pages
        let mut s = timed("one two. three four five");
        s.words[3].start = Tick(3 * S);
        s.words[3].end = Tick(3 * S + S / 3);
        s.words[4].start = Tick(3 * S + S / 2);
        s.words[4].end = Tick(4 * S);
        s.duration = Tick(5 * S);
        let ps = pages(&s, &st, &a, 1080, 1920);
        assert_eq!(ps.iter().map(|p| p.words.len()).collect::<Vec<_>>(), vec![2, 1, 2]);
    }

    #[test]
    fn the_active_word_at_given_times() {
        let (st, a) = bold();
        let c = timed("make it count");
        let p = &pages(&c, &st, &a, 1080, 1920)[0];
        let active = |t: f64| p.words.iter().rposition(|w| w.start <= Tick::from_seconds_f64(t));
        assert_eq!(active(0.0), Some(0));
        assert_eq!(active(0.39), Some(0));
        assert_eq!(active(0.41), Some(1));
        assert_eq!(active(0.9), Some(2));
        // and the drawn colours follow: the spoken word in yellow, the others white
        let yellow_left = |t: f64| {
            let o = render(&c, &st, &a, 1080, 1920, Tick::from_seconds_f64(t)).unwrap();
            let mut xs = Vec::new();
            for y in 0..o.h {
                for x in 0..o.w {
                    let q = &o.px[(y * o.w + x) * 4..(y * o.w + x) * 4 + 4];
                    // linear yellow: red and green high, blue low
                    if q[3] > 0.95 && q[0] > 0.8 && q[1] > 0.5 && q[2] < 0.1 {
                        xs.push(o.x + x as i32);
                    }
                }
            }
            xs.iter().sum::<i32>() as f64 / xs.len().max(1) as f64
        };
        let (x0, x1, x2) = (yellow_left(0.2), yellow_left(0.6), yellow_left(1.0));
        assert!(x0 < x1 && x1 < x2, "the highlight walks right: {x0} {x1} {x2}");
    }

    #[test]
    fn the_spoken_word_pops_with_an_overshoot() {
        let (st, a) = bold();
        let c = timed("pop goes");
        let ink = |t: f64| render(&c, &st, &a, 1080, 1920, Tick::from_seconds_f64(t)).map_or(0.0, |o| o.px.chunks(4).map(|p| p[3]).sum::<f32>());
        // page pop-in at 0, the word growing; at 0.06 s it overshoots past its settled size
        let (early, peak, settled) = (ink(0.02), ink(0.07), ink(0.3));
        assert!(early < settled, "{early} {settled}");
        assert!(peak > settled * 1.01, "overshoot: {peak} vs {settled}");
    }

    #[test]
    fn safe_zone_centring_on_shorts() {
        let (mut st, mut a) = bold();
        a.platform = Some(Platform::Youtube);
        st.animated = Some(a.clone());
        let c = timed("centred on the safe span");
        for p in pages(&c, &st, &a, 1080, 1920) {
            let (l, r) = Platform::Youtube.safe_span(f64::from(p.rect[1]) / 1920.0, f64::from(p.rect[3]) / 1920.0);
            let mid = ((l + r) / 2.0 * 1080.0) as f32;
            let centre = (p.rect[0] + p.rect[2]) / 2.0;
            // YouTube's right margin (18 %) is wider than its left (5 %): the middle sits left of the frame's
            assert!((centre - mid).abs() < 2.0, "centre {centre} vs safe middle {mid}");
            assert!(mid < 540.0 - 60.0);
            assert!(p.rect[0] >= (l * 1080.0) as f32 - 1.0 && p.rect[2] <= (r * 1080.0) as f32 + 1.0, "{:?} inside {l}..{r}", p.rect);
            // just above the bottom zone (35 %), the margin counted from the safe area
            assert!(p.rect[3] <= 1920.0 * 0.65 && p.rect[3] > 1920.0 * 0.45, "{:?}", p.rect);
        }
        // the whole safe width is available: a long word fits the span, not a centred band
        let wide = cap("SUPERCALIFRAGILISTIC", 1.0);
        let p = &pages(&wide, &st, &a, 1080, 1920)[0];
        let (l, r) = Platform::Youtube.safe_span(0.5, 0.6);
        assert!(p.rect[2] - p.rect[0] <= ((r - l) * 1080.0) as f32 + 1.0);
        // TikTok's button column narrows the span low in the frame
        let mut tk = a.clone();
        tk.platform = Some(Platform::TikTok);
        let pt = &pages(&c, &st, &tk, 1080, 1920)[0];
        assert!(pt.rect[2] <= 1080.0 * 0.8 + 1.0);
        // an x offset moves the block, still inside the span
        let st2 = CaptionStyle { x_offset: 0.05, ..st.clone() };
        let po = &pages(&c, &st2, &a, 1080, 1920)[0];
        let p0 = &pages(&c, &st, &a, 1080, 1920)[0];
        assert!((po.rect[0] + po.rect[2]) / 2.0 > (p0.rect[0] + p0.rect[2]) / 2.0 + 20.0);
    }

    #[test]
    fn every_preset_draws() {
        let c = timed("every preset draws these words");
        for p in CaptionPreset::ALL {
            let a = AnimatedCaptions::preset(p);
            let st = CaptionStyle { animated: Some(a.clone()), ..p.base_style() };
            let o = render(&c, &st, &a, 1080, 1920, Tick(S / 2)).unwrap_or_else(|| panic!("{p:?}"));
            assert!(o.px.chunks(4).filter(|q| q[3] > 0.9).count() > 300, "{p:?}");
            if p == CaptionPreset::TwoLine {
                assert_eq!(pages(&c, &st, &a, 1080, 1920).len(), 1, "the whole caption");
            }
        }
        // box: a purple box behind the spoken word
        let a = AnimatedCaptions::preset(CaptionPreset::Box);
        let st = CaptionStyle { animated: Some(a.clone()), ..CaptionPreset::Box.base_style() };
        let o = render(&c, &st, &a, 1080, 1920, Tick(S / 10)).unwrap();
        assert!(o.px.chunks(4).any(|q| q[3] > 0.95 && q[2] > q[1] * 2.0 && q[0] > q[1]), "purple box pixels");
        // glow: cyan light around the spoken word, outside the letters
        let a = AnimatedCaptions::preset(CaptionPreset::Glow);
        let st = CaptionStyle { animated: Some(a.clone()), ..CaptionPreset::Glow.base_style() };
        let o = render(&c, &st, &a, 1080, 1920, Tick(S / 10)).unwrap();
        assert!(o.px.chunks(4).any(|q| q[3] > 0.2 && q[3] < 0.9 && q[1] > q[0] * 3.0 && q[2] > q[0] * 3.0), "cyan glow pixels");
    }

    #[test]
    fn karaoke_sweeps_through_the_word() {
        let a = AnimatedCaptions::preset(CaptionPreset::Karaoke);
        let st = CaptionStyle { animated: Some(a.clone()), ..CaptionPreset::Karaoke.base_style() };
        let c = timed("sweeping");
        let yellow = |t: f64| {
            let o = render(&c, &st, &a, 1080, 1920, Tick::from_seconds_f64(t)).unwrap();
            o.px.chunks(4).filter(|q| q[3] > 0.95 && q[0] > 0.8 && q[1] > 0.5 && q[2] < 0.1).count()
        };
        let (a0, a1, a2) = (yellow(0.0), yellow(0.17), yellow(0.4));
        assert!(a0 < a1 && a1 < a2, "{a0} {a1} {a2}");
    }

    #[test]
    fn keywords_and_emoji() {
        let (st, mut a) = bold();
        a.keywords = vec!["money".into()];
        a.emoji = vec![filmcraft_project::WordEmoji { word: "money".into(), emoji: "💰".into() }];
        let c = timed("save money now");
        let p = &pages(&c, &st, &a, 1080, 1920)[0];
        assert!(p.words[1].keyword && p.words[1].text.starts_with("MONEY") && p.words[1].text.ends_with('💰'));
        let o = render(&c, &st, &a, 1080, 1920, Tick(0)).unwrap();
        // the keyword green while another word is spoken
        assert!(o.px.chunks(4).any(|q| q[3] > 0.95 && q[1] > 0.5 && q[0] < 0.2), "green keyword");
    }

    #[test]
    fn malayalam_words_highlight_one_by_one() {
        filmcraft_text::fonts::scan_system();
        if !filmcraft_text::fonts::all_faces().iter().any(|f| f.covers_text("മലയാളം")) {
            eprintln!("SKIPPED: no Malayalam font installed");
            return;
        }
        let (st, a) = bold();
        let c = timed("ഇത് ഒരു പരീക്ഷണം ആണ്");
        let p = pages(&c, &st, &a, 1080, 1920);
        assert_eq!(p.iter().map(|p| p.words.len()).sum::<usize>(), 4);
        for pg in &p {
            for w in &pg.words {
                assert!(w.layout.glyphs.iter().all(|g| g.id != 0), "{} has no missing glyphs", w.text);
                assert!(w.width > 10.0);
            }
        }
        let o0 = render(&c, &st, &a, 1080, 1920, Tick(S / 10)).unwrap();
        let o1 = render(&c, &st, &a, 1080, 1920, Tick(5 * S / 10)).unwrap();
        assert_ne!(o0.px, o1.px, "the highlight moves to the next word");
    }

    #[test]
    fn hostile_styles_and_times_never_panic() {
        let c = timed("a b c d e f g h");
        for (size, outline, margin, xo) in [(f32::NAN, f32::INFINITY, -1.0, f32::NAN), (1e9, -5.0, 9.0, 3.0), (0.0, 0.0, 0.45, -0.5)] {
            for p in CaptionPreset::ALL {
                let a = AnimatedCaptions { words_per_page: u32::MAX, pop: f32::INFINITY, platform: Some(Platform::TikTok), ..AnimatedCaptions::preset(p) };
                let st = CaptionStyle { size, outline, margin, x_offset: xo, line_spacing: f32::NAN, ..p.base_style() };
                for t in [Tick(i64::MIN), Tick(-5), Tick(0), Tick(S), Tick(i64::MAX)] {
                    for (w, h) in [(0, 0), (1, 1), (1080, 1920), (7, 3000)] {
                        let r = std::panic::catch_unwind(|| render(&c, &st, &a, w, h, t));
                        assert!(r.is_ok(), "{p:?} {size} {t:?} {w}x{h}");
                    }
                }
            }
        }
        let empty = cap("   ", 1.0);
        assert!(render(&empty, &bold().0, &bold().1, 1080, 1920, Tick(0)).is_none());
    }
}
