//! Keeping a 360° follow (`clip.follow360`) through blocking: once the subject tracker has lost its
//! subject for a few frames — someone walked in front of it, it went behind a pillar and came out
//! elsewhere, turned round — it is looked for again by how it looks. Windows of its size over the
//! tracking camera's picture, and over the views either side of it, are compared with the subject
//! as it was first seen by a colour signature, and the best match, clearly better than the next
//! one elsewhere, re-seeds the tracker there — once SigLIP 2's image embedding (when the scene
//! models are downloaded) agrees that it is the same subject, so someone else in the same colours
//! is not taken for it. SigLIP alone can search every window too (slower).

use std::sync::Arc;

use filmcraft_render::subject::SubjectBox;

use crate::scene_models::{Pic, SceneModels};

/// Confident frames the subject's look is learnt from.
pub const LEARN_FRAMES: usize = 3;
/// Frames lost before the subject is looked for, and between searches while it stays lost.
pub const SEARCH_AFTER: usize = 3;
/// Most searches in one follow.
pub const MAX_SEARCHES: usize = 80;
/// Most windows compared in one picture.
const MAX_WINDOWS: usize = 64;
/// How much wider than the box a window's picture is cropped (context helps the embedder).
const CONTEXT: f64 = 1.2;
/// The side the crops are made at for the embedder.
const CROP_SIDE: usize = 96;

/// How much lower the bar is for the grid's best window, before it is moved onto the subject.
const COARSE: f32 = 0.2;
/// A colour match is taken when SigLIP 2's embeddings of it and of the subject are at least this
/// alike.
const CONFIRM: f32 = 0.75;

/// How the subject's look is compared.
#[derive(Clone)]
pub enum Judge {
    /// SigLIP 2 image embeddings (the scene models' embedder) of every window.
    Embedder(Arc<dyn SceneModels>),
    /// A colour signature: hue, saturation and lightness binned, square-rooted (the cosine is the
    /// Bhattacharyya coefficient).
    Colour,
    /// The colour signature finds it, SigLIP 2 confirms it.
    Confirmed(Arc<dyn SceneModels>),
}

impl Judge {
    /// The matches a window must reach, and by how much it must beat the best one elsewhere.
    fn thresholds(&self) -> (f32, f32) {
        match self {
            Judge::Embedder(_) => (0.82, 0.03),
            Judge::Colour | Judge::Confirmed(_) => (0.90, 0.02),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Judge::Embedder(_) => "siglip",
            Judge::Colour => "colour",
            Judge::Confirmed(_) => "colour+siglip",
        }
    }

    /// The scene models whose embedder confirms or searches, if any.
    pub fn models(&self) -> Option<&Arc<dyn SceneModels>> {
        match self {
            Judge::Embedder(m) | Judge::Confirmed(m) => Some(m),
            Judge::Colour => None,
        }
    }

    /// Unit-length signatures of `crops` for the search (`None` when the embedder fails).
    fn sign(&self, crops: &[Pic]) -> Option<Vec<Vec<f32>>> {
        match self {
            Judge::Embedder(m) => embed(m, crops),
            Judge::Colour | Judge::Confirmed(_) => Some(crops.iter().map(colour_signature).collect()),
        }
    }
}

fn embed(m: &Arc<dyn SceneModels>, crops: &[Pic]) -> Option<Vec<Vec<f32>>> {
    m.embed_images(crops).ok().filter(|v| v.len() == crops.len())
}

/// `sum` scaled to unit length (`None` when it is nothing).
fn unit(sum: &[f32]) -> Option<Vec<f32>> {
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm.is_finite() && norm > 0.0).then(|| sum.iter().map(|x| x / norm).collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == b.len() { a.iter().zip(b).map(|(x, y)| x * y).sum() } else { 0.0 }
}

/// The subject's look, learnt from its first confident frames: the search's signature, and the
/// embedding that confirms a colour match.
pub struct Look {
    judge: Judge,
    sum: Vec<f32>,
    check: Vec<f32>,
    learnt: usize,
}

impl Look {
    pub fn new(judge: Judge) -> Self {
        Look { judge, sum: Vec::new(), check: Vec::new(), learnt: 0 }
    }

    pub fn judge(&self) -> &Judge {
        &self.judge
    }

    /// Whether it still learns (fewer than [`LEARN_FRAMES`] frames seen).
    pub fn learning(&self) -> bool {
        self.learnt < LEARN_FRAMES
    }

    /// Whether there is a look to search for.
    pub fn ready(&self) -> bool {
        self.learnt > 0
    }

    /// Learn from the subject at box `b` of picture `rgba` (`w` × `h`).
    pub fn learn(&mut self, rgba: &[u8], w: usize, h: usize, b: SubjectBox) {
        let Some(crop) = crop(rgba, w, h, b) else { return };
        let Some(v) = self.judge.sign(std::slice::from_ref(&crop)).and_then(|mut v| v.pop()) else { return };
        if self.sum.is_empty() {
            self.sum = vec![0.0; v.len()];
        }
        if v.len() != self.sum.len() {
            return;
        }
        for (s, x) in self.sum.iter_mut().zip(&v) {
            *s += x;
        }
        if let Judge::Confirmed(m) = &self.judge
            && let Some(e) = embed(m, std::slice::from_ref(&crop)).and_then(|mut v| v.pop())
        {
            if self.check.is_empty() {
                self.check = vec![0.0; e.len()];
            }
            if e.len() == self.check.len() {
                for (s, x) in self.check.iter_mut().zip(&e) {
                    *s += x;
                }
            }
        }
        self.learnt += 1;
    }

    /// The window of picture `rgba` (`w` × `h`) most like the subject, boxes `bw` × `bh` (shares),
    /// how alike it is (cosine) and how much SigLIP agrees (when it confirms), when it is alike
    /// enough and clearly the best.
    pub fn search(&self, rgba: &[u8], w: usize, h: usize, bw: f64, bh: f64) -> Option<(SubjectBox, f32, Option<f32>)> {
        if !self.ready() {
            return None;
        }
        let reference = unit(&self.sum)?;
        let score = |boxes: Vec<SubjectBox>| -> Option<Vec<(SubjectBox, f32)>> {
            let crops: Vec<(SubjectBox, Pic)> = boxes.into_iter().filter_map(|b| crop(rgba, w, h, b).map(|c| (b, c))).collect();
            let pics: Vec<Pic> = crops.iter().map(|(_, c)| c.clone()).collect();
            let sigs = if pics.is_empty() { Vec::new() } else { self.judge.sign(&pics)? };
            Some(crops.iter().zip(&sigs).map(|((b, _), s)| (*b, cosine(s, &reference))).filter(|(_, c)| c.is_finite()).collect())
        };
        let (boxes, step) = windows(bw, bh);
        let scored = score(boxes)?;
        let (mut best, mut sim) = scored.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1))?;
        // the best window elsewhere (not overlapping this one)
        let other = scored.iter().filter(|(b, _)| b.iou(&best) == 0.0).map(|(_, c)| *c).fold(f32::NEG_INFINITY, f32::max);
        let (min, margin) = self.judge.thresholds();
        log::debug!("follow360 refind ({}): best {sim:.3} at ({:.2}, {:.2}), elsewhere {other:.3}", self.judge.name(), best.cx, best.cy);
        // (the grid's windows can straddle the subject: the bar is lower until it is centred)
        if !(sim >= min - COARSE && (other.is_infinite() || sim - other >= margin)) {
            return None;
        }
        // the grid's best moved by half a step, a quarter, an eighth, while that matches better
        for f in [0.5, 0.25, 0.125] {
            let near: Vec<SubjectBox> = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                .iter()
                .filter_map(|(dx, dy)| {
                    SubjectBox::new(
                        (best.cx + dx * f * step.0).clamp(bw / 2.0, 1.0 - bw / 2.0),
                        (best.cy + dy * f * step.1).clamp(bh / 2.0, 1.0 - bh / 2.0),
                        bw,
                        bh,
                    )
                })
                .collect();
            if let Some((b, c)) = score(near)?.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))
                && c > sim
            {
                (best, sim) = (b, c);
            }
        }
        if sim < min {
            return None;
        }
        // the colours found it; SigLIP says whether it is the same subject
        let mut agreed = None;
        if let Judge::Confirmed(m) = &self.judge {
            let check = unit(&self.check)?;
            let e = embed(m, &[crop(rgba, w, h, best)?])?.pop()?;
            let agree = cosine(&e, &check);
            log::debug!("follow360 refind: SigLIP {agree:.3} for the colour match {sim:.3}");
            if agree < CONFIRM {
                return None;
            }
            agreed = Some(agree);
        }
        Some((best, sim, agreed))
    }
}

/// Windows of `bw` × `bh` (shares) over a picture, half a window apart (further apart when there
/// would be more than [`MAX_WINDOWS`]), and how far apart.
pub fn windows(bw: f64, bh: f64) -> (Vec<SubjectBox>, (f64, f64)) {
    let (bw, bh) = (bw.clamp(0.03, 1.0), bh.clamp(0.03, 1.0));
    if !(bw.is_finite() && bh.is_finite()) {
        return (Vec::new(), (0.0, 0.0));
    }
    let mut step = (bw / 2.0, bh / 2.0);
    loop {
        let nx = ((1.0 - bw) / step.0).floor() as usize + 1;
        let ny = ((1.0 - bh) / step.1).floor() as usize + 1;
        if nx * ny <= MAX_WINDOWS {
            let boxes = (0..ny)
                .flat_map(|j| (0..nx).map(move |i| (i, j)))
                .filter_map(|(i, j)| SubjectBox::new(bw / 2.0 + i as f64 * step.0, bh / 2.0 + j as f64 * step.1, bw, bh))
                .collect();
            return (boxes, step);
        }
        step = (step.0 * 1.25, step.1 * 1.25);
    }
}

/// The picture at box `b` (a little wider), scaled to [`CROP_SIDE`] on its longer side.
fn crop(rgba: &[u8], w: usize, h: usize, b: SubjectBox) -> Option<Pic> {
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return None;
    }
    let (cw, ch) = (b.w * CONTEXT * w as f64, b.h * CONTEXT * h as f64);
    let (x0, y0) = (b.cx * w as f64 - cw / 2.0, b.cy * h as f64 - ch / 2.0);
    if !(cw >= 2.0 && ch >= 2.0) {
        return None;
    }
    let k = CROP_SIDE as f64 / cw.max(ch);
    let (ow, oh) = (((cw * k).round() as usize).max(1), ((ch * k).round() as usize).max(1));
    let mut out = Vec::with_capacity(ow * oh * 4);
    for y in 0..oh {
        let sy = ((y0 + (y as f64 + 0.5) / k).floor().max(0.0) as usize).min(h - 1);
        for x in 0..ow {
            let sx = ((x0 + (x as f64 + 0.5) / k).floor().max(0.0) as usize).min(w - 1);
            let i = (sy * w + sx) * 4;
            out.extend_from_slice(rgba.get(i..i + 4)?);
        }
    }
    Some(Pic { rgba: out, w: ow, h: oh })
}

/// A crop's colour signature: hue (12), saturation (3) and lightness (3) binned, weighted towards
/// the middle (the subject more than what is round it), square-rooted and of unit length.
pub fn colour_signature(p: &Pic) -> Vec<f32> {
    const H: usize = 12;
    const S: usize = 3;
    const L: usize = 3;
    let mut bins = vec![0f32; H * S * L];
    for y in 0..p.h {
        for x in 0..p.w {
            let Some(px) = p.rgba.get((y * p.w + x) * 4..(y * p.w + x) * 4 + 3) else { continue };
            let (r, g, b) = (f32::from(px[0]) / 255.0, f32::from(px[1]) / 255.0, f32::from(px[2]) / 255.0);
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            let l = (mx + mn) / 2.0;
            let d = mx - mn;
            let s = if d < 1e-6 { 0.0 } else { d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-6) };
            let hue = if d < 1e-6 {
                0.0
            } else if mx == r {
                ((g - b) / d).rem_euclid(6.0)
            } else if mx == g {
                (b - r) / d + 2.0
            } else {
                (r - g) / d + 4.0
            } / 6.0;
            let hi = ((hue * H as f32) as usize).min(H - 1);
            let si = ((s.clamp(0.0, 1.0) * S as f32) as usize).min(S - 1);
            let li = ((l.clamp(0.0, 1.0) * L as f32) as usize).min(L - 1);
            // the middle counts most
            let (dx, dy) = ((x as f32 + 0.5) / p.w as f32 - 0.5, (y as f32 + 0.5) / p.h as f32 - 0.5);
            let wt = (1.0 - 2.0 * (dx * dx + dy * dy)).max(0.1);
            if let Some(v) = bins.get_mut((hi * S + si) * L + li) {
                *v += wt;
            }
        }
    }
    let mut v: Vec<f32> = bins.iter().map(|x| x.sqrt()).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey picture with a red-and-blue figure at (`cx`, `cy`) and a green one elsewhere.
    fn picture(w: usize, h: usize, cx: f64, cy: f64, green_at: f64) -> Vec<u8> {
        let mut px = vec![110u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                let i = (y * w + x) * 4;
                let c = if (u - cx).abs() < 0.04 && (v - cy).abs() < 0.12 {
                    if v < cy { [200, 30, 30] } else { [30, 40, 200] }
                } else if (u - green_at).abs() < 0.04 && (v - 0.5).abs() < 0.12 {
                    [40, 190, 50]
                } else {
                    [110, 110, 110]
                };
                px[i..i + 3].copy_from_slice(&c);
                px[i + 3] = 255;
            }
        }
        px
    }

    #[test]
    fn the_subject_is_found_again_by_its_colours() {
        let (w, h) = (384, 216);
        let mut look = Look::new(Judge::Colour);
        assert!(look.search(&picture(w, h, 0.5, 0.5, 0.2), w, h, 0.08, 0.24).is_none(), "nothing learnt yet");
        for k in 0..LEARN_FRAMES {
            let cx = 0.5 + k as f64 * 0.01;
            look.learn(&picture(w, h, cx, 0.5, 0.2), w, h, SubjectBox::new(cx, 0.5, 0.08, 0.24).unwrap());
        }
        assert!(!look.learning() && look.ready());
        // after the blocking it is somewhere else: found there, not at the green figure
        let (b, sim, _) = look.search(&picture(w, h, 0.8, 0.45, 0.3), w, h, 0.08, 0.24).expect("found");
        assert!((b.cx - 0.8).abs() < 0.05 && (b.cy - 0.45).abs() < 0.08, "{b:?} {sim}");
        // gone: nothing found
        let empty = picture(w, h, -1.0, -1.0, 0.3);
        assert!(look.search(&empty, w, h, 0.08, 0.24).is_none());
        // hostile pictures and sizes
        assert!(look.search(&[], w, h, 0.08, 0.24).is_none());
        assert!(look.search(&empty, w, h, f64::NAN, 0.2).is_none());
        assert!(windows(0.5, 0.5).0.len() <= MAX_WINDOWS && !windows(0.01, 0.01).0.is_empty());
        assert!(windows(0.01, 0.01).0.len() <= MAX_WINDOWS);
    }

    /// A stand-in embedder that sees layout: the mean colour of a crop's top and bottom halves.
    struct Halves;

    impl SceneModels for Halves {
        fn embedder_id(&self) -> Option<String> {
            Some("halves".into())
        }
        fn embed_images(&self, pics: &[Pic]) -> std::result::Result<Vec<Vec<f32>>, String> {
            Ok(pics
                .iter()
                .map(|p| {
                    let mut v = [0f32; 6];
                    for y in 0..p.h {
                        for x in 0..p.w {
                            let i = (y * p.w + x) * 4;
                            let half = if y < p.h / 2 { 0 } else { 3 };
                            for c in 0..3 {
                                v[half + c] += f32::from(p.rgba[i + c]);
                            }
                        }
                    }
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    v.iter().map(|x| x / n).collect()
                })
                .collect())
        }
    }

    #[test]
    fn a_colour_match_is_taken_only_once_the_embedder_agrees() {
        let (w, h) = (384, 216);
        let mut look = Look::new(Judge::Confirmed(Arc::new(Halves)));
        for _ in 0..LEARN_FRAMES {
            look.learn(&picture(w, h, 0.5, 0.5, 0.2), w, h, SubjectBox::new(0.5, 0.5, 0.08, 0.24).unwrap());
        }
        // the subject again: found and confirmed
        assert!(look.search(&picture(w, h, 0.7, 0.5, 0.2), w, h, 0.08, 0.24).is_some());
        // someone in the same colours the other way up: the colours alone would take them
        let mut impostor = picture(w, h, 0.7, 0.5, 0.2);
        for px in impostor.as_chunks_mut::<4>().0 {
            match [px[0], px[1], px[2]] {
                [200, 30, 30] => px[..3].copy_from_slice(&[30, 40, 200]),
                [30, 40, 200] => px[..3].copy_from_slice(&[200, 30, 30]),
                _ => {}
            }
        }
        let mut colours = Look::new(Judge::Colour);
        for _ in 0..LEARN_FRAMES {
            colours.learn(&picture(w, h, 0.5, 0.5, 0.2), w, h, SubjectBox::new(0.5, 0.5, 0.08, 0.24).unwrap());
        }
        assert!(colours.search(&impostor, w, h, 0.08, 0.24).is_some(), "the colours cannot tell");
        assert!(look.search(&impostor, w, h, 0.08, 0.24).is_none(), "the embedder can");
    }
}
