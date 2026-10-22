//! Text layers drawn glyph by glyph: animated text (letters, words or lines animating in, out and
//! in a loop — [`filmcraft_project::text_anim`]) and text set along an arc.
//!
//! Every glyph gets its own transform: on an arc, each glyph sits on the circle turned to its
//! tangent; an animator moves, scales and turns each unit (letter, word, line) about its centre,
//! fades and blurs it. Units are rasterised one by one with the layer's full appearance (fill or
//! gradient, per-character fills, up to three strokes, shadow, glow, inner shadow) and composed
//! pass by pass — every unit's shadow, then glow, outer strokes, fill, inner shadow and inner
//! strokes — so the strokes of one letter never cover the fill of its neighbour. With no
//! animation in motion and no arc the static path of [`crate::graphic_clip`] draws the layer,
//! and at rest this path draws what that one does.

use filmcraft_project::graphic::{LayerSpec, TextCurve, TextProps};
use filmcraft_project::text_anim::{AnimUnit, UnitState, unit_state, word_units};
use filmcraft_text::raster::{Xform, compose, fill_into, xform_scale};
use filmcraft_text::{Layout, Mask, StrokeKind, mask, render};

use crate::graphic_clip::{
    LayerRaster, appearance_reach, bbox_of, glow_mask, inner_shadow_mask, invert_xform, over_linear_gradient, stroke_kind, text_layout, union,
};
use crate::graphics::premul_linear;
use crate::image::Image;

/// Most units one layer animates (the rest move with the last).
const MAX_UNITS: usize = 4096;

/// Translate.
fn tr(x: f32, y: f32) -> Xform {
    [1.0, 0.0, 0.0, 1.0, x, y]
}

/// Rotate (radians, clockwise on screen) and scale about the origin.
fn rot_scale(a: f32, s: f32) -> Xform {
    let (sin, cos) = a.sin_cos();
    [cos * s, sin * s, -sin * s, cos * s, 0.0, 0.0]
}

/// Each glyph's advance (to the next glyph on its line, or the line's end).
fn advances(l: &Layout) -> Vec<f32> {
    let mut adv = vec![0.0f32; l.glyphs.len()];
    for line in &l.lines {
        let end = line.x + line.width;
        for gi in line.glyphs.clone() {
            let Some(g) = l.glyphs.get(gi) else { continue };
            let next = if gi + 1 < line.glyphs.end { l.glyphs.get(gi + 1).map_or(end, |n| n.x) } else { end };
            if let Some(a) = adv.get_mut(gi) {
                *a = (next - g.x).abs().min(g.size * 4.0);
            }
        }
    }
    adv
}

/// Glyph → layer transforms: on the arc of `curve`, or plain positions.
fn glyph_places(l: &Layout, curve: Option<TextCurve>, adv: &[f32]) -> Vec<Xform> {
    let mut out: Vec<Xform> = l.glyphs.iter().map(|g| tr(g.x, g.y)).collect();
    let Some(c) = curve.filter(|c| c.bend.is_finite() && c.bend.abs() > 1e-4 && !l.vertical) else { return out };
    let widest = l.lines.iter().map(|ln| ln.width).fold(0.0f32, f32::max);
    let r = if c.radius > 0.5 { c.radius } else { widest / (c.bend.abs() * std::f32::consts::PI) };
    if !r.is_finite() || r < 1.0 {
        return out;
    }
    let up = c.bend > 0.0;
    for line in &l.lines {
        let cx = line.x + line.width / 2.0;
        for gi in line.glyphs.clone() {
            let (Some(g), Some(a)) = (l.glyphs.get(gi), adv.get(gi)) else { continue };
            let theta = (g.x + a / 2.0 - cx) / r;
            let (px, py, turn) =
                if up { (cx + r * theta.sin(), g.y + r * (1.0 - theta.cos()), theta) } else { (cx + r * theta.sin(), g.y - r * (1.0 - theta.cos()), -theta) };
            if let Some(o) = out.get_mut(gi) {
                // the glyph turns about the middle of its advance on the baseline
                *o = compose(&compose(&tr(px, py), &rot_scale(turn, 1.0)), &tr(-a / 2.0, 0.0));
            }
        }
    }
    out
}

/// The unit of every glyph (None for spaces) and the number of units.
fn glyph_units(l: &Layout, text: &str, unit: AnimUnit) -> (Vec<Option<usize>>, usize) {
    let space = |b: usize| text.get(b..).and_then(|s| s.chars().next()).is_none_or(char::is_whitespace);
    let mut ids: Vec<Option<usize>> = vec![None; l.glyphs.len()];
    match unit {
        AnimUnit::Letter => {
            // a shaped cluster is a letter: a Malayalam conjunct with its vowel sign moves as one
            let mut clusters: Vec<usize> = l.glyphs.iter().map(|g| g.cluster).filter(|b| !space(*b)).collect();
            clusters.sort_unstable();
            clusters.dedup();
            for (g, id) in l.glyphs.iter().zip(ids.iter_mut()) {
                if !space(g.cluster) {
                    *id = clusters.binary_search(&g.cluster).ok().map(|i| i.min(MAX_UNITS - 1));
                }
            }
            (ids, clusters.len().min(MAX_UNITS))
        }
        AnimUnit::Word => {
            let (words, _) = word_units(text);
            let mut used: Vec<usize> = l.glyphs.iter().filter(|g| !space(g.cluster)).filter_map(|g| words.get(g.cluster).copied()).collect();
            used.sort_unstable();
            used.dedup();
            for (g, id) in l.glyphs.iter().zip(ids.iter_mut()) {
                if !space(g.cluster) {
                    *id = words.get(g.cluster).and_then(|w| used.binary_search(w).ok()).map(|i| i.min(MAX_UNITS - 1));
                }
            }
            (ids, used.len().min(MAX_UNITS))
        }
        AnimUnit::Line => {
            let mut n = 0;
            for line in &l.lines {
                let mut any = false;
                for gi in line.glyphs.clone() {
                    if let (Some(g), Some(id)) = (l.glyphs.get(gi), ids.get_mut(gi))
                        && !space(g.cluster)
                    {
                        *id = Some(n.min(MAX_UNITS - 1));
                        any = true;
                    }
                }
                if any {
                    n += 1;
                }
            }
            (ids, n.min(MAX_UNITS))
        }
    }
}

/// One unit, rasterised: its masks in a rectangle at `at` of the layer image.
struct UnitPaint {
    at: (i32, i32),
    /// Glyphs in the layer's fill (solid or gradient), and per-character solid fills.
    base: Mask,
    solids: Vec<([f32; 4], Mask)>,
    strokes: Vec<Mask>,
    shadow: Option<Mask>,
    glow: Option<Mask>,
    inner: Option<Mask>,
    opacity: f32,
}

/// Over-composite `c` (premultiplied) through `m` placed at `at` into `px` (`pw` wide).
fn over_mask_at(px: &mut [f32], pw: usize, ph: usize, m: &Mask, at: (i32, i32), c: [f32; 4]) {
    if c[3] <= 0.0 {
        return;
    }
    for y in 0..m.h {
        let py = at.1 + y as i32;
        if py < 0 || py >= ph as i32 {
            continue;
        }
        for x in 0..m.w {
            let pxx = at.0 + x as i32;
            if pxx < 0 || pxx >= pw as i32 {
                continue;
            }
            let a = m.a.get(y * m.w + x).copied().unwrap_or(0.0);
            if a <= 0.0 {
                continue;
            }
            let i = (py as usize * pw + pxx as usize) * 4;
            if let Some(p) = px.get_mut(i..i + 4) {
                let k = 1.0 - c[3] * a;
                for ch in 0..4 {
                    p[ch] = c[ch] * a + p[ch] * k;
                }
            }
        }
    }
}

/// Over-composite a premultiplied image (`m.w`×`m.h`, `src`) at `at`, scaled by `op`.
fn over_image_at(px: &mut [f32], pw: usize, ph: usize, src: &[f32], sw: usize, sh: usize, at: (i32, i32), op: f32) {
    for y in 0..sh {
        let py = at.1 + y as i32;
        if py < 0 || py >= ph as i32 {
            continue;
        }
        for x in 0..sw {
            let pxx = at.0 + x as i32;
            if pxx < 0 || pxx >= pw as i32 {
                continue;
            }
            let si = (y * sw + x) * 4;
            let Some(s) = src.get(si..si + 4) else { continue };
            if s[3] <= 0.0 {
                continue;
            }
            let i = (py as usize * pw + pxx as usize) * 4;
            if let Some(p) = px.get_mut(i..i + 4) {
                let k = 1.0 - s[3] * op;
                for ch in 0..4 {
                    p[ch] = s[ch] * op + p[ch] * k;
                }
            }
        }
    }
}

fn scaled(c: [f32; 4], op: f32) -> [f32; 4] {
    [c[0] * op, c[1] * op, c[2] * op, c[3] * op]
}

/// Rasterise a text layer glyph by glyph (see the module docs); `m` maps layer pixels to the
/// device, clipped to a `cw`×`ch` canvas.
pub fn raster_text_units(spec: &LayerSpec, t: &TextProps, m: &Xform, cw: usize, ch: usize) -> Option<LayerRaster> {
    let l = text_layout(t);
    let ap = &spec.appearance;
    if l.glyphs.is_empty() && ap.background.is_none() {
        return None;
    }
    let s = xform_scale(m).max(1e-6);
    let adv = advances(&l);
    let places = glyph_places(&l, t.curve, &adv);
    // units and their states
    let (anim, (u, d)) = match (&t.anim, t.clock) {
        (Some(a), Some(c)) => (Some(a), c),
        _ => (None, (0.0, f64::INFINITY)),
    };
    let unit = anim.map_or(AnimUnit::Letter, |a| a.unit());
    let (ids, n) = glyph_units(&l, &t.text, unit);
    let states: Vec<UnitState> = (0..n.max(1)).map(|i| anim.map_or(UnitState::REST, |a| unit_state(a, i, n, u, d))).collect();
    // each unit's pivot (layer space): the middle of its glyphs' boxes
    let mut sums = vec![(0.0f32, 0.0f32, 0.0f32, 0usize); n.max(1)];
    for (gi, g) in l.glyphs.iter().enumerate() {
        let (Some(Some(id)), Some(p), Some(a)) = (ids.get(gi), places.get(gi), adv.get(gi)) else { continue };
        let c = filmcraft_text::raster::apply(p, a / 2.0, -0.35 * g.size);
        if let Some(e) = sums.get_mut(*id) {
            *e = (e.0 + c.0, e.1 + c.1, e.2.max(g.size), e.3 + 1);
        }
    }
    let unit_xf: Vec<(Xform, f32, f32)> = states
        .iter()
        .zip(&sums)
        .map(|(st, (sx, sy, size, k))| {
            let (px, py) = if *k > 0 { (sx / *k as f32, sy / *k as f32) } else { (0.0, 0.0) };
            let em = *size;
            let a = compose(&tr(px + (st.dx as f32) * em, py + (st.dy as f32) * em), &rot_scale((st.rotation as f32).to_radians(), st.scale as f32));
            (compose(&a, &tr(-px, -py)), st.opacity as f32, (st.blur as f32) * em)
        })
        .collect();
    let reach = appearance_reach(ap, s);
    let shadow_dev = ap.shadow.as_ref().map(|sh| (m[0] * sh.offset.0 + m[2] * sh.offset.1, m[1] * sh.offset.0 + m[3] * sh.offset.1));
    let shadow_grow = ap.shadow.as_ref().map_or(0.0, |sh| (sh.size + sh.blur) * s + 2.0);
    // group the glyphs by unit; spaces and glyphs of hidden units are not drawn
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n.max(1)];
    for (gi, id) in ids.iter().enumerate() {
        if let Some(id) = id
            && let Some(v) = members.get_mut(*id)
        {
            v.push(gi);
        }
    }
    let run_fill = |g: &filmcraft_text::Glyph| -> Option<[f32; 4]> { g.run.checked_sub(1).and_then(|r| t.runs.get(r as usize)).and_then(|(_, c)| c.fill) };
    let mut units: Vec<UnitPaint> = Vec::new();
    let mut layer_box: Option<[f32; 4]> = None;
    let canvas_margin = 2.0 + shadow_grow + shadow_dev.map_or(0.0, |o| o.0.abs().max(o.1.abs())) + reach;
    for (ui, glyphs) in members.iter().enumerate() {
        let Some((uxf, op, blur)) = unit_xf.get(ui).copied() else { continue };
        if glyphs.is_empty() || op <= 0.0 {
            continue;
        }
        let dev: Vec<(usize, Xform)> = glyphs.iter().filter_map(|&gi| Some((gi, compose(&compose(m, &uxf), places.get(gi)?)))).collect();
        // device box of the unit's glyphs (generous: glyph boxes of 1.6 em by 1.6 em)
        let mut bb: Option<[f32; 4]> = None;
        for (gi, x) in &dev {
            let Some(g) = l.glyphs.get(*gi) else { continue };
            let b = bbox_of(x, [-0.3 * g.size, -1.15 * g.size, 1.4 * g.size, 0.45 * g.size], 0.0);
            bb = Some(bb.map_or(b, |a| union(a, b)));
        }
        let Some(mut bb) = bb else { continue };
        let pad = reach + blur * s * 3.0 + 2.0;
        bb = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad];
        if let Some(o) = shadow_dev {
            bb = union(bb, [bb[0] + o.0 - shadow_grow, bb[1] + o.1 - shadow_grow, bb[2] + o.0 + shadow_grow, bb[3] + o.1 + shadow_grow]);
        }
        let x0 = bb[0].floor().max(-canvas_margin) as i32;
        let y0 = bb[1].floor().max(-canvas_margin) as i32;
        let x1 = bb[2].ceil().min(cw as f32 + canvas_margin) as i32;
        let y1 = bb[3].ceil().min(ch as f32 + canvas_margin) as i32;
        if x1 <= x0 || y1 <= y0 || !bb.iter().all(|v| v.is_finite()) {
            continue;
        }
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        if w.saturating_mul(h) > 16 << 20 {
            continue;
        }
        let mut base = Mask::new(w, h);
        let mut solids: Vec<([f32; 4], Mask)> = Vec::new();
        for (gi, x) in &dev {
            let Some(g) = l.glyphs.get(*gi) else { continue };
            match run_fill(g) {
                Some(c) if ap.fill.is_some() || ap.gradient.is_some() => {
                    let k = match solids.iter().position(|(sc, _)| *sc == c) {
                        Some(k) => k,
                        None => {
                            solids.push((c, Mask::new(w, h)));
                            solids.len() - 1
                        }
                    };
                    if let Some((_, sm)) = solids.get_mut(k) {
                        render::draw_glyph(g, x, sm, (x0, y0));
                    }
                }
                _ => render::draw_glyph(g, x, &mut base, (x0, y0)),
            }
        }
        if blur > 0.01 {
            let sigma = blur * s;
            base = mask::blur(&base, sigma);
            for (_, sm) in &mut solids {
                *sm = mask::blur(sm, sigma);
            }
        }
        let mut cover = base.clone();
        for (_, sm) in &solids {
            cover.max_with(sm);
        }
        let mut strokes: Vec<Mask> = ap.strokes.iter().map(|(_, sw, k)| mask::stroke(&cover, sw * s, stroke_kind(*k))).collect();
        let outer = |k: u32| k != 1 && k != 2;
        let filled = ap.fill.is_some() || ap.gradient.is_some();
        if !filled {
            for (sm, (_, _, k)) in strokes.iter_mut().zip(&ap.strokes) {
                if outer(*k) {
                    sm.a.iter_mut().zip(&cover.a).for_each(|(v, c)| *v = v.min(1.0 - c));
                }
            }
        }
        let union_mask = || {
            let mut un = if filled { cover.clone() } else { Mask::new(w, h) };
            for sm in &strokes {
                un.max_with(sm);
            }
            un
        };
        let shadow = ap.shadow.as_ref().zip(shadow_dev).map(|(sh, o)| {
            let mut un = union_mask();
            if sh.size > 0.0 {
                un = mask::stroke(&un, sh.size * s, StrokeKind::Outer);
            }
            mask::blur(&mask::offset(&un, o.0.round() as i32, o.1.round() as i32), sh.blur * s / 3.0)
        });
        let glow = ap.glow.as_ref().map(|gl| glow_mask(&union_mask(), gl.radius * s, gl.intensity));
        let inner = ap.inner_shadow.as_ref().filter(|_| filled).map(|is| {
            let off = (m[0] * is.offset.0 + m[2] * is.offset.1, m[1] * is.offset.0 + m[3] * is.offset.1);
            inner_shadow_mask(&cover, off, is.blur * s)
        });
        layer_box = Some(layer_box.map_or([x0 as f32, y0 as f32, x1 as f32, y1 as f32], |a| union(a, [x0 as f32, y0 as f32, x1 as f32, y1 as f32])));
        units.push(UnitPaint { at: (x0, y0), base, solids, strokes, shadow, glow, inner, opacity: op });
    }
    // the background box and underlines stay where the text is laid out, at the units' mean opacity
    let mean_op = if units.is_empty() { 0.0 } else { units.iter().map(|u| u.opacity).sum::<f32>() / units.len() as f32 };
    let lb = l.bounds;
    let mut deco: Option<(i32, i32, Mask, Option<Mask>)> = None;
    if (ap.background.is_some() || !l.underlines.is_empty()) && mean_op > 0.0 {
        let pad = ap.background.map_or(0.0, |(_, p, _)| p) + 2.0 / s;
        let bb = bbox_of(m, lb, pad);
        let x0 = bb[0].floor().max(-canvas_margin) as i32;
        let y0 = bb[1].floor().max(-canvas_margin) as i32;
        let x1 = bb[2].ceil().min(cw as f32 + canvas_margin) as i32;
        let y1 = bb[3].ceil().min(ch as f32 + canvas_margin) as i32;
        if x1 > x0 && y1 > y0 && ((x1 - x0) as usize).saturating_mul((y1 - y0) as usize) <= 16 << 20 {
            let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
            let bg = ap.background.map(|(_, p, r)| {
                let mut bm = Mask::new(w, h);
                fill_into(&filmcraft_text::Path::round_rect(lb[0] - p, lb[1] - p, lb[2] + p, lb[3] + p, r).transformed(m), &mut bm, -x0 as f32, -y0 as f32);
                bm
            });
            let mut ul = Mask::new(w, h);
            for u in &l.underlines {
                fill_into(&filmcraft_text::Path::rect(u[0], u[1], u[2], u[3]).transformed(m), &mut ul, -x0 as f32, -y0 as f32);
            }
            layer_box = Some(layer_box.map_or([x0 as f32, y0 as f32, x1 as f32, y1 as f32], |a| union(a, [x0 as f32, y0 as f32, x1 as f32, y1 as f32])));
            deco = Some((x0, y0, ul, bg));
        }
    }
    let lbx = layer_box?;
    let (lx0, ly0) = (lbx[0] as i32, lbx[1] as i32);
    let (pw, ph) = ((lbx[2] - lbx[0]).max(1.0) as usize, (lbx[3] - lbx[1]).max(1.0) as usize);
    if pw.saturating_mul(ph) > 64 << 20 {
        return None;
    }
    let mut px = vec![0.0f32; pw * ph * 4];
    let rel = |at: (i32, i32)| (at.0 - lx0, at.1 - ly0);
    // pass 1: shadows
    if let Some(sh) = &ap.shadow {
        for u in &units {
            if let Some(sm) = &u.shadow {
                over_mask_at(&mut px, pw, ph, sm, rel(u.at), scaled(premul_linear(sh.color), u.opacity));
            }
        }
    }
    // pass 2: the background box
    if let (Some((bx, by, _, Some(bm))), Some((c, _, _))) = (&deco, ap.background) {
        over_mask_at(&mut px, pw, ph, bm, rel((*bx, *by)), scaled(premul_linear(c), mean_op));
    }
    // pass 3: glow
    if let Some(gl) = &ap.glow {
        for u in &units {
            if let Some(gm) = &u.glow {
                over_mask_at(&mut px, pw, ph, gm, rel(u.at), scaled(premul_linear(gl.color), u.opacity));
            }
        }
    }
    // pass 4: outer strokes (outermost first)
    for (k, (c, _, kind)) in ap.strokes.iter().enumerate().rev() {
        if *kind == 1 || *kind == 2 {
            continue;
        }
        for u in &units {
            if let Some(sm) = u.strokes.get(k) {
                over_mask_at(&mut px, pw, ph, sm, rel(u.at), scaled(premul_linear(*c), u.opacity));
            }
        }
    }
    // pass 5: fills (gradient in layer space, solid, per-character colours) and underlines
    let inv = invert_xform(m);
    for u in &units {
        if let (Some(grad), Some(inv)) = (&ap.gradient, &inv) {
            let mut tmp = vec![0.0f32; u.base.w * u.base.h * 4];
            over_linear_gradient(&mut tmp, &u.base, grad, lb, inv, u.at);
            over_image_at(&mut px, pw, ph, &tmp, u.base.w, u.base.h, rel(u.at), u.opacity);
        } else if let Some(c) = ap.fill {
            over_mask_at(&mut px, pw, ph, &u.base, rel(u.at), scaled(premul_linear(c), u.opacity));
        }
        for (c, sm) in &u.solids {
            over_mask_at(&mut px, pw, ph, sm, rel(u.at), scaled(premul_linear(*c), u.opacity));
        }
    }
    if let (Some((bx, by, ul, _)), Some(c)) = (&deco, ap.fill.or(ap.gradient.as_ref().map(|g| g.start))) {
        over_mask_at(&mut px, pw, ph, ul, rel((*bx, *by)), scaled(premul_linear(c), mean_op));
    }
    // pass 6: inner shadows, then centre and inner strokes
    if let Some(is) = &ap.inner_shadow {
        for u in &units {
            if let Some(im) = &u.inner {
                over_mask_at(&mut px, pw, ph, im, rel(u.at), scaled(premul_linear(is.color), u.opacity));
            }
        }
    }
    for (k, (c, _, kind)) in ap.strokes.iter().enumerate().rev() {
        if *kind != 1 && *kind != 2 {
            continue;
        }
        for u in &units {
            if let Some(sm) = u.strokes.get(k) {
                over_mask_at(&mut px, pw, ph, sm, rel(u.at), scaled(premul_linear(*c), u.opacity));
            }
        }
    }
    let op = spec.transform.opacity;
    if op < 1.0 {
        px.iter_mut().for_each(|v| *v *= op);
    }
    Some(LayerRaster { x: lx0, y: ly0, img: Image { w: pw, h: ph, px } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphic_clip::{graphic_specs, render_graphic_layers};
    use filmcraft_geom::{Affine, Vec2};
    use filmcraft_project::graphic::{LayerContent, new_text_layer};
    use filmcraft_project::graphic_design::LayerExtra;
    use filmcraft_project::text_anim::{AnimKind, Animator, TextAnimation};
    use filmcraft_project::{EffectInstance, ParamValue};
    use filmcraft_time::Tick;

    fn set(e: &mut EffectInstance, k: &str, v: ParamValue) {
        e.params.get_mut(k).unwrap_or_else(|| panic!("{k}")).value = v;
    }

    fn render_at(e: &EffectInstance, secs: f64, w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        render_graphic_layers(std::slice::from_ref(e), Tick::from_seconds_f64(secs), (w as u32, h as u32), &Affine::IDENTITY, &mut img);
        img
    }

    fn ink(img: &Image) -> f32 {
        img.px.chunks(4).map(|p| p[3]).sum()
    }

    fn animated(kind: AnimKind) -> EffectInstance {
        let mut e = new_text_layer("WAVE", Vec2::new(40.0, 120.0), 80.0);
        e.layer = Some(Box::new(LayerExtra { anim: Some(TextAnimation { enter: Some(Animator::preset(kind)), ..Default::default() }), ..Default::default() }));
        e
    }

    #[test]
    fn at_rest_an_animated_title_draws_like_a_still_one() {
        let still = new_text_layer("WAVE", Vec2::new(40.0, 120.0), 80.0);
        let mut e = animated(AnimKind::Pop);
        set(&mut e, "stroke", ParamValue::Bool(true));
        let mut still2 = still.clone();
        set(&mut still2, "stroke", ParamValue::Bool(true));
        // long after the pop: the clock is gone and the layer is drawn by the static path
        let spec = graphic_specs(std::slice::from_ref(&e), Tick::from_seconds_f64(5.0), (400, 200));
        let LayerContent::Text(t) = &spec[0].content else { panic!() };
        assert!(t.clock.is_none() && t.anim.is_some());
        let (a, b) = (render_at(&e, 5.0, 400, 200), render_at(&still2, 5.0, 400, 200));
        assert_eq!(a.px, b.px);
        // and the glyph-by-glyph path agrees with it when forced at rest (an empty in animator)
        let mut sp = spec[0].clone();
        if let LayerContent::Text(t) = &mut sp.content {
            t.clock = Some((5.0, f64::INFINITY));
        }
        let r = raster_text_units(
            &sp,
            match &sp.content {
                LayerContent::Text(t) => t,
                _ => unreachable!(),
            },
            &crate::graphic_clip::xform(&crate::graphic_clip::layer_matrix(&sp)),
            400,
            200,
        )
        .unwrap();
        let mut c = Image::new(400, 200);
        crate::graphic_clip::composite_raster(&mut c, &r, 0, 0);
        let diff: f32 = c.px.iter().zip(&b.px).map(|(x, y)| (x - y).abs()).sum::<f32>() / ink(&b).max(1.0);
        assert!(diff < 0.02, "per-unit and static paths differ by {diff}");
    }

    #[test]
    fn pop_grows_letters_one_after_another() {
        let e = animated(AnimKind::Pop);
        let start = render_at(&e, 0.0, 400, 200);
        assert!(ink(&start) < 1.0, "nothing at the start: {}", ink(&start));
        // part way: the first letters are drawn, the last not yet
        let mid = render_at(&e, 0.12, 400, 200);
        let col_ink = |img: &Image, x0: usize, x1: usize| (x0..x1).flat_map(|x| (0..200).map(move |y| (x, y))).map(|(x, y)| img.get(x, y)[3]).sum::<f32>();
        let end = render_at(&e, 2.0, 400, 200);
        let (l, r) = (col_ink(&mid, 0, 120), col_ink(&mid, 260, 400));
        assert!(l > 50.0 && r < col_ink(&end, 260, 400) * 0.5, "left drawn {l}, right still small {r}");
        assert!((ink(&end) - ink(&render_at(&new_text_layer("WAVE", Vec2::new(40.0, 120.0), 80.0), 0.0, 400, 200))).abs() < 1.0);
    }

    #[test]
    fn slide_moves_words_and_fade_fades() {
        let mut e = new_text_layer("ONE TWO", Vec2::new(40.0, 120.0), 60.0);
        let slide = Animator { unit: AnimUnit::Word, ..Animator::preset(AnimKind::SlideUp) };
        e.layer = Some(Box::new(LayerExtra { anim: Some(TextAnimation { enter: Some(slide), ..Default::default() }), ..Default::default() }));
        let early = render_at(&e, 0.15, 500, 260);
        let late = render_at(&e, 3.0, 500, 260);
        let bottom = |img: &Image| {
            let mut b = 0;
            for y in 0..img.h {
                if (0..img.w).any(|x| img.get(x, y)[3] > 0.3) {
                    b = y;
                }
            }
            b
        };
        assert!(bottom(&early) > bottom(&late) + 5, "rises into place: {} {}", bottom(&early), bottom(&late));
        let fade = animated(AnimKind::Fade);
        let half = render_at(&fade, 0.05, 400, 200);
        let full = render_at(&fade, 3.0, 400, 200);
        assert!(ink(&half) > 10.0 && ink(&half) < ink(&full) * 0.8);
    }

    #[test]
    fn curve_bends_the_baseline_into_an_arc() {
        let mut e = new_text_layer("ARCHED TITLE", Vec2::new(300.0, 200.0), 50.0);
        set(&mut e, "align", ParamValue::Choice(1));
        let flat = render_at(&e, 0.0, 600, 400);
        set(&mut e, "curve", ParamValue::Float(60.0));
        let arch = render_at(&e, 0.0, 600, 400);
        set(&mut e, "curve", ParamValue::Float(-60.0));
        let smile = render_at(&e, 0.0, 600, 400);
        // lowest ink in the outer columns versus the middle
        let low = |img: &Image, x0: usize, x1: usize| (0..img.h).rev().find(|&y| (x0..x1).any(|x| img.get(x, y)[3] > 0.5)).unwrap_or(0) as i32;
        let edge = |img: &Image| low(img, 0, 220).max(low(img, 380, 600));
        let mid = |img: &Image| low(img, 270, 330);
        assert!((edge(&flat) - mid(&flat)).abs() <= 2, "flat: {} {}", edge(&flat), mid(&flat));
        assert!(edge(&arch) > mid(&arch) + 15, "an arch: the ends drop below the middle ({} {})", edge(&arch), mid(&arch));
        assert!(mid(&smile) > edge(&smile) + 15, "a smile: the middle is lowest ({} {})", mid(&smile), edge(&smile));
        assert!((ink(&arch) - ink(&flat)).abs() / ink(&flat) < 0.05, "same letters, turned");
    }

    #[test]
    fn glow_spreads_light_around_text() {
        let mut e = new_text_layer("NEON", Vec2::new(60.0, 130.0), 80.0);
        set(&mut e, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
        set(&mut e, "glow", ParamValue::Bool(true));
        set(&mut e, "glow_color", ParamValue::Color([0.0, 1.0, 1.0, 1.0]));
        set(&mut e, "glow_radius", ParamValue::Float(20.0));
        let img = render_at(&e, 0.0, 400, 200);
        let plain = render_at(&new_text_layer("NEON", Vec2::new(60.0, 130.0), 80.0), 0.0, 400, 200);
        assert!(ink(&img) > ink(&plain) * 1.5, "glow adds light: {} vs {}", ink(&img), ink(&plain));
        // just above the letters: cyan light, no red
        let p = img.get(120, 60);
        assert!(p[3] > 0.05 && p[1] > p[0] * 3.0 && p[2] > p[0] * 3.0, "{p:?}");
        assert_eq!(plain.get(120, 40)[3], 0.0);
        // the letters stay white on top
        let inside = (0..400).flat_map(|x| (0..200).map(move |y| (x, y))).filter(|&(x, y)| plain.get(x, y)[3] > 0.99).collect::<Vec<_>>();
        assert!(inside.iter().all(|&(x, y)| img.get(x, y)[0] > 0.9));
        // and an animated, glowing title draws its glow too
        let mut a = animated(AnimKind::Fade);
        set(&mut a, "glow", ParamValue::Bool(true));
        assert!(ink(&render_at(&a, 0.1, 400, 200)) > 0.0);
    }

    #[test]
    fn inner_shadow_darkens_inside_the_edges() {
        let mut e = filmcraft_project::graphic::new_shape_layer(0, Vec2::new(100.0, 100.0), Vec2::new(100.0, 100.0), vec![]);
        set(&mut e, "fill_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0]));
        set(&mut e, "inner_shadow", ParamValue::Bool(true));
        set(&mut e, "inner_shadow_opacity", ParamValue::Float(100.0));
        set(&mut e, "inner_shadow_distance", ParamValue::Float(8.0));
        set(&mut e, "inner_shadow_blur", ParamValue::Float(0.0));
        let img = render_at(&e, 0.0, 200, 200);
        // 135°: the light comes from the top left, the shadow falls inside the top-left edges
        let tl = img.get(53, 100);
        let br = img.get(146, 100);
        let mid = img.get(100, 100);
        assert!(tl[0] < 0.2 && tl[3] > 0.99, "dark inside the left edge {tl:?}");
        assert!(br[0] > 0.9 && mid[0] > 0.9, "lit elsewhere {br:?} {mid:?}");
        assert_eq!(img.get(40, 100)[3], 0.0, "nothing outside");
    }

    #[test]
    fn malayalam_letters_animate_as_whole_clusters() {
        filmcraft_text::fonts::scan_system();
        if !filmcraft_text::fonts::all_faces().iter().any(|f| f.covers_text("ക്ക")) {
            eprintln!("SKIPPED: no Malayalam font installed");
            return;
        }
        let t = TextProps {
            text: "ക്കാ മല".into(),
            font: "Inter".into(),
            style: "Bold".into(),
            size: 60.0,
            align: 0,
            tracking: 0.0,
            kerning: true,
            ligatures: true,
            leading: 0.0,
            baseline_shift: 0.0,
            faux_bold: false,
            faux_italic: false,
            caps: 0,
            underline: false,
            box_width: 0.0,
            box_height: 0.0,
            vertical: false,
            runs: vec![],
            curve: None,
            anim: None,
            clock: None,
        };
        let l = text_layout(&t);
        let (ids, n) = glyph_units(&l, &t.text, AnimUnit::Letter);
        // ക്കാ is one shaped cluster (its glyphs share a unit); മ and ല are two more
        assert_eq!(n, 3, "{:?}", l.glyphs.iter().map(|g| g.cluster).collect::<Vec<_>>());
        let first: Vec<_> = l.glyphs.iter().zip(&ids).filter(|(g, _)| g.cluster == 0).map(|(_, i)| *i).collect();
        assert!(first.iter().all(|i| *i == Some(0)));
        let (_, words) = glyph_units(&l, &t.text, AnimUnit::Word);
        assert_eq!(words, 2);
    }

    #[test]
    fn hostile_animation_and_curve_never_panic() {
        for (bend, radius) in [(100.0, 0.0), (-100.0, 1e9), (0.001, f64::NAN), (f64::INFINITY, -5.0), (50.0, 0.4)] {
            let mut e = new_text_layer("Hostile\nText", Vec2::new(10.0, 50.0), 40.0);
            set(&mut e, "curve", ParamValue::Float(bend));
            set(&mut e, "curve_radius", ParamValue::Float(radius));
            let wild = Animator { amount: 1e300, stagger: f64::NAN, duration: -1.0, ..Animator::preset(AnimKind::Glitch) };
            e.layer = Some(Box::new(LayerExtra {
                anim: Some(TextAnimation { enter: Some(wild.clone()), out: Some(wild), repeat: Some(Animator::preset(AnimKind::Shake)) }),
                ..Default::default()
            }));
            for s in [0.0, 0.01, 1e9, -3.0] {
                let r = std::panic::catch_unwind(|| render_at(&e, s, 120, 80));
                assert!(r.is_ok(), "{bend} {radius} at {s}");
            }
        }
    }
}
