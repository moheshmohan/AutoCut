//! Telling faces apart: **AuraFace-v1** (fal, Apache-2.0), an ArcFace-trained IResNet-100, maps a
//! face aligned to 112 × 112 to a 512-number fingerprint; two faces of one person point the same
//! way (cosine similarity near 1), two people's apart. The faces come from YuNet
//! ([`crate::FaceDetector`]), whose five landmarks (eyes, nose tip, mouth corners) align them.
//!
//! The network is written out by hand over [`Backend`], checked against the ONNX file's layer
//! list: the stem (3×3 convolution, PReLU), 3 + 13 + 30 + 3 residual blocks (batch norm, 3×3
//! convolution, PReLU, 3×3 convolution — stride 2 and a 1×1 shortcut in each stage's first
//! block), a batch norm, the 25 088 → 512 projection and a last batch norm. A block's first batch
//! norm comes before a zero-padded convolution, so it stays its own pass (a 1×1 per-channel
//! convolution); the last two fold exactly into the projection. The convolutions run on the GPU
//! (`gpunn`) when there is one, on the CPU otherwise (the reference); the projection runs on the
//! CPU after the 7 × 7 maps come back.

use std::path::Path;

use rayon::prelude::*;

use crate::VisionError;
use crate::gpunn::{GpuUse, Nn};
use crate::layers::{Act, Backend, Conv, Cpu, Feat, err};
use crate::onnx::{self, Attr, Model, Node};

/// The side of an aligned face.
pub const INPUT: usize = 112;
/// The fingerprint's length.
pub const DIM: usize = 512;
/// Faces a batch sends through the network at once.
pub const BATCH: usize = 16;
/// Where the five landmarks of an aligned 112 × 112 face go (InsightFace's ArcFace template):
/// the eye on the picture's left, the other eye, the nose tip, the mouth's left and right corners.
pub const TEMPLATE: [[f32; 2]; 5] = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
/// The stages' widths and depths.
const STAGES: [(usize, usize); 4] = [(64, 3), (128, 13), (256, 30), (512, 3)];
/// The final maps: 512 planes of 7 × 7.
const FLAT: usize = DIM * 7 * 7;

/// One residual block: `c2(prelu(c1(bn(x)))) + shortcut(x)`.
struct Block {
    bn: Conv,
    c1: Conv,
    c2: Conv,
    down: Option<Conv>,
}

/// The face fingerprint network.
pub struct FaceEmbedder {
    stem: Conv,
    blocks: Vec<Block>,
    /// The projection with both batch norms folded in: `[512, 25 088]` (inputs in plane order).
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
    gpu: Option<GpuUse>,
}

fn tensor<'a>(m: &'a Model, name: Option<&String>, dims: &[usize]) -> Result<&'a [f32], VisionError> {
    let name = name.ok_or_else(|| err("a layer misses an input"))?;
    let t = m.initializers.get(name).ok_or_else(|| err(format!("no weights named {name}")))?;
    if t.dims != dims {
        return Err(err(format!("{name} is {:?}, expected {dims:?}", t.dims)));
    }
    t.f32s().ok_or_else(|| err(format!("{name} is not float32")))
}

/// A walk through the graph's nodes in order.
struct Walk<'a> {
    m: &'a Model,
    at: usize,
}

impl<'a> Walk<'a> {
    fn next(&mut self, op: &str) -> Result<&'a Node, VisionError> {
        let n = self.m.nodes.get(self.at).ok_or_else(|| err(format!("the graph ends before a {op}")))?;
        if n.op != op {
            return Err(err(format!("node {} is a {}, expected a {op}", self.at, n.op)));
        }
        self.at += 1;
        Ok(n)
    }

    fn peek(&self) -> Option<&'a str> {
        self.m.nodes.get(self.at).map(|n| n.op.as_str())
    }

    /// A convolution with its bias (the export folded the batch norms after it in).
    fn conv(&mut self, cin: usize, cout: usize, k: usize, stride: usize) -> Result<Conv, VisionError> {
        let n = self.next("Conv")?;
        let ints = |a: &str| n.ints(a).map(<[i64]>::to_vec).unwrap_or_default();
        let pad = (k / 2) as i64;
        if ints("kernel_shape") != [k as i64, k as i64]
            || ints("strides") != [stride as i64, stride as i64]
            || ints("pads") != [pad, pad, pad, pad]
            || n.int("group").unwrap_or(1) != 1
        {
            return Err(err(format!("a convolution is not the {k}×{k}/{stride} the network has")));
        }
        let w = tensor(self.m, n.inputs.get(1), &[cout, cin, k, k])?.to_vec();
        let b = tensor(self.m, n.inputs.get(2), &[cout])?.to_vec();
        Conv::new(w, b, cin, cout, k, stride, k / 2, 1, Act::None)
    }

    fn prelu(&mut self, c: Conv) -> Result<Conv, VisionError> {
        let n = self.next("PRelu")?;
        let slopes = tensor(self.m, n.inputs.get(1), &[c.cout, 1, 1])?.to_vec();
        c.with_prelu(slopes)
    }

    /// A batch norm: the scale and shift per channel.
    fn bn(&mut self, c: usize) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
        let n = self.next("BatchNormalization")?;
        let eps = match n.attr("epsilon") {
            Some(Attr::Float(e)) if e.is_finite() && *e >= 0.0 => *e,
            None => 1e-5,
            _ => return Err(err("a batch norm's epsilon")),
        };
        let p = |i: usize| tensor(self.m, n.inputs.get(i), &[c]);
        let (g, b, mean, var) = (p(1)?, p(2)?, p(3)?, p(4)?);
        let mut scale = Vec::with_capacity(c);
        let mut shift = Vec::with_capacity(c);
        for (((g, b), m), v) in g.iter().zip(b).zip(mean).zip(var) {
            let s = g / (v + eps).max(1e-12).sqrt();
            scale.push(s);
            shift.push(b - m * s);
        }
        if scale.iter().chain(&shift).any(|v| !v.is_finite()) {
            return Err(err("a batch norm is not finite"));
        }
        Ok((scale, shift))
    }
}

impl FaceEmbedder {
    pub fn from_model(m: &Model) -> Result<Self, VisionError> {
        let mut g = Walk { m, at: 0 };
        let stem = g.conv(3, 64, 3, 1)?;
        let stem = g.prelu(stem)?;
        let mut blocks = Vec::new();
        let mut cin = 64;
        for (width, depth) in STAGES {
            for i in 0..depth {
                let (scale, shift) = g.bn(cin)?;
                let bn = Conv::new(scale, shift, cin, cin, 1, 1, 0, cin, Act::None)?;
                let c1 = g.conv(cin, width, 3, 1)?;
                let c1 = g.prelu(c1)?;
                let c2 = g.conv(width, width, 3, if i == 0 { 2 } else { 1 })?;
                let down = if i == 0 { Some(g.conv(cin, width, 1, 2)?) } else { None };
                g.next("Add")?;
                blocks.push(Block { bn, c1, c2, down });
                cin = width;
            }
        }
        let (s2, t2) = g.bn(DIM)?;
        let flat = g.next("Flatten")?;
        if flat.int("axis").unwrap_or(1) != 1 {
            return Err(err("the flatten keeps the batch"));
        }
        let fc = g.next("Gemm")?;
        if fc.int("transB").unwrap_or(0) != 1 {
            return Err(err("the projection's weights are not [out, in]"));
        }
        let mut w = tensor(m, fc.inputs.get(1), &[DIM, FLAT])?.to_vec();
        let mut b = tensor(m, fc.inputs.get(2), &[DIM])?.to_vec();
        let (s3, t3) = g.bn(DIM)?;
        if g.peek().is_some() {
            return Err(err("the graph goes on after the last batch norm"));
        }
        // fold bn2 (per input plane: 49 inputs a channel) into the projection
        let plane = FLAT / DIM;
        w.par_chunks_mut(FLAT).zip(b.par_iter_mut()).for_each(|(row, bias)| {
            let mut add = 0f32;
            for (c, chunk) in row.chunks_mut(plane).enumerate() {
                let (s, t) = (s2.get(c).copied().unwrap_or(1.0), t2.get(c).copied().unwrap_or(0.0));
                for v in chunk {
                    add += *v * t;
                    *v *= s;
                }
            }
            *bias += add;
        });
        // and the features' batch norm on its outputs
        w.par_chunks_mut(FLAT).zip(b.par_iter_mut()).zip(s3.par_iter().zip(t3.par_iter())).for_each(|((row, bias), (s, t))| {
            for v in row {
                *v *= s;
            }
            *bias = *bias * s + t;
        });
        if w.iter().chain(&b).any(|v| !v.is_finite()) {
            return Err(err("the folded projection is not finite"));
        }
        Ok(Self { stem, blocks, fc_w: w, fc_b: b, gpu: None })
    }

    /// Load from the ONNX file, on the CPU.
    pub fn load(path: &Path) -> Result<Self, VisionError> {
        let lim = onnx::Limits { max_bytes: 300 << 20, ..onnx::Limits::default() };
        Self::from_model(&onnx::read(path, &lim)?)
    }

    /// Run the convolutions on this machine's GPU from now on (the CPU stays the fallback);
    /// `false` without one.
    pub fn use_gpu(&mut self) -> bool {
        let Some(nn) = Nn::shared() else { return false };
        self.stem.upload(&nn);
        for b in &mut self.blocks {
            b.bn.upload(&nn);
            b.c1.upload(&nn);
            b.c2.upload(&nn);
            if let Some(d) = b.down.as_mut() {
                d.upload(&nn);
            }
        }
        self.gpu = Some(GpuUse::new(nn));
        true
    }

    pub fn on_gpu(&self) -> bool {
        GpuUse::active(&self.gpu)
    }

    fn maps<B: Backend>(&self, be: &B, x: &B::T) -> Result<B::T, VisionError> {
        let mut y = be.conv(&self.stem, x, None)?;
        for b in &self.blocks {
            let h = be.conv(&b.c1, &be.conv(&b.bn, &y, None)?, None)?;
            y = match &b.down {
                Some(d) => {
                    let s = be.conv(d, &y, None)?;
                    be.conv(&b.c2, &h, Some(&s))?
                }
                None => be.conv(&b.c2, &h, Some(&y))?,
            };
        }
        Ok(y)
    }

    /// The fingerprints of `x.n` aligned faces `[n, 3, 112, 112]` (RGB, `(v − 127.5) / 127.5`),
    /// each of unit length. The convolutions on the GPU when there is one.
    pub(crate) fn embed_feat(&self, x: &Feat) -> Result<Vec<[f32; DIM]>, VisionError> {
        if x.c != 3 || x.h != INPUT || x.w != INPUT || x.n == 0 {
            return Err(err("the face network takes 112 × 112 RGB faces"));
        }
        let maps = match GpuUse::run(&self.gpu, |r| {
            let y = self.maps(r, &r.input(x)?)?;
            r.read(&[&y])?.into_iter().next().ok_or_else(|| err("the face maps did not come back"))
        }) {
            Some(f) => f,
            None => self.maps(&Cpu, x)?,
        };
        if maps.c != DIM || maps.h * maps.w * maps.c != FLAT || maps.n != x.n {
            return Err(err("the face maps are not 512 × 7 × 7"));
        }
        maps.v
            .par_chunks(FLAT)
            .map(|img| {
                let mut out = [0f32; DIM];
                for ((o, row), b) in out.iter_mut().zip(self.fc_w.as_chunks::<FLAT>().0).zip(&self.fc_b) {
                    *o = row.iter().zip(img).map(|(a, x)| a * x).sum::<f32>() + b;
                }
                let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
                if !norm.is_finite() || norm <= 1e-6 {
                    return Err(err("a face's fingerprint is zero"));
                }
                for v in &mut out {
                    *v /= norm;
                }
                Ok(out)
            })
            .collect()
    }

    /// The fingerprints of aligned faces (each `3 · 112 · 112` values, as [`align`] makes them),
    /// [`BATCH`] at a time.
    pub fn embed(&self, faces: &[Vec<f32>]) -> Result<Vec<[f32; DIM]>, VisionError> {
        let mut out = Vec::with_capacity(faces.len());
        for chunk in faces.chunks(BATCH) {
            let mut v = Vec::with_capacity(chunk.len() * 3 * INPUT * INPUT);
            for f in chunk {
                if f.len() != 3 * INPUT * INPUT {
                    return Err(err("an aligned face is not 3 × 112 × 112"));
                }
                v.extend_from_slice(f);
            }
            out.extend(self.embed_feat(&Feat::new(v, chunk.len(), 3, INPUT, INPUT)?)?);
        }
        Ok(out)
    }
}

/// The similarity transform (scale, rotation, shift) taking the five `landmarks` closest to
/// [`TEMPLATE`], as `[a, b, tx, ty]`: `x' = a·x − b·y + tx`, `y' = b·x + a·y + ty`. Least squares
/// (Umeyama's in 2-D, no reflection). `None` when the points are degenerate.
pub fn similarity(landmarks: &[[f32; 2]; 5]) -> Option<[f32; 4]> {
    let mean = |p: &[[f32; 2]; 5]| {
        let (x, y) = p.iter().fold((0f64, 0f64), |(x, y), q| (x + f64::from(q[0]), y + f64::from(q[1])));
        (x / 5.0, y / 5.0)
    };
    let (sx, sy) = mean(landmarks);
    let (dx, dy) = mean(&TEMPLATE);
    let (mut num_re, mut num_im, mut den) = (0f64, 0f64, 0f64);
    for (s, d) in landmarks.iter().zip(&TEMPLATE) {
        let (ux, uy) = (f64::from(s[0]) - sx, f64::from(s[1]) - sy);
        let (vx, vy) = (f64::from(d[0]) - dx, f64::from(d[1]) - dy);
        // (v)(conj u)
        num_re += vx * ux + vy * uy;
        num_im += vy * ux - vx * uy;
        den += ux * ux + uy * uy;
    }
    if !den.is_finite() || den < 1e-6 {
        return None;
    }
    let (a, b) = (num_re / den, num_im / den);
    let (tx, ty) = (dx - (a * sx - b * sy), dy - (b * sx + a * sy));
    let m = [a as f32, b as f32, tx as f32, ty as f32];
    m.iter().all(|v| v.is_finite()).then_some(m)
}

/// A face cut out of an RGBA picture (`w` × `h`) and aligned to 112 × 112 by its landmarks, as
/// the network takes it: RGB planes, `(v − 127.5) / 127.5`; outside the picture is black (as
/// InsightFace's `warpAffine` makes it). `None` when the landmarks are degenerate.
pub fn align(rgba: &[u8], w: usize, h: usize, landmarks: &[[f32; 2]; 5]) -> Option<Vec<f32>> {
    if rgba.len() != w.checked_mul(h)?.checked_mul(4)? || w == 0 || h == 0 {
        return None;
    }
    let [a, b, tx, ty] = similarity(landmarks)?;
    // the inverse: x = (a·(x'−tx) + b·(y'−ty)) / (a² + b²), y = (−b·(x'−tx) + a·(y'−ty)) / (a² + b²)
    let d = a * a + b * b;
    if d <= 1e-12 {
        return None;
    }
    let plane = INPUT * INPUT;
    let mut out = vec![-1f32; 3 * plane];
    let px = |x: i64, y: i64, c: usize| -> f32 {
        if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
            return 0.0;
        }
        rgba.get((y as usize * w + x as usize) * 4 + c).map_or(0.0, |v| f32::from(*v))
    };
    for oy in 0..INPUT {
        for ox in 0..INPUT {
            let (u, v) = (ox as f32 - tx, oy as f32 - ty);
            let sx = (a * u + b * v) / d;
            let sy = (-b * u + a * v) / d;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            for c in 0..3 {
                let top = px(x0, y0, c) * (1.0 - fx) + px(x0 + 1, y0, c) * fx;
                let bottom = px(x0, y0 + 1, c) * (1.0 - fx) + px(x0 + 1, y0 + 1, c) * fx;
                let val = top * (1.0 - fy) + bottom * fy;
                if let Some(o) = out.get_mut(c * plane + oy * INPUT + ox) {
                    *o = (val - 127.5) / 127.5;
                }
            }
        }
    }
    Some(out)
}

/// How alike two fingerprints are: their cosine (both have unit length), −1 to 1.
pub fn similarity_of(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Most fingerprints [`group`] takes (about 90 000 faces' worth of a whole library and more).
pub const MAX_GROUPED: usize = 400_000;

/// Fingerprints grouped into people: a label per fingerprint, `0` the largest group, then by
/// size; a face like no other is a group of its own.
///
/// Each fingerprint's `k` nearest others at a cosine of at least `threshold` are its neighbours
/// (blocked matrix products on the CPU); Chinese Whispers then lets every face take, over a fixed
/// number of rounds in a fixed order, the label its neighbours hold with the most similarity.
/// Unlike plain linking, one face that looks a little like two people does not join them.
pub fn group(v: &[[f32; DIM]], threshold: f32, k: usize) -> Result<Vec<usize>, VisionError> {
    use candle_core::{Device, Tensor};
    let n = v.len();
    if n > MAX_GROUPED {
        return Err(err(format!("at most {MAX_GROUPED} fingerprints are grouped at once, got {n}")));
    }
    if !(threshold.is_finite() && (-1.0..=1.0).contains(&threshold)) || k == 0 || k > 200 {
        return Err(err("group with a threshold of −1 to 1 and 1–200 neighbours"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let flat: Vec<f32> = v.iter().flatten().copied().collect();
    if flat.iter().any(|x| !x.is_finite()) {
        return Err(err("a fingerprint is not finite"));
    }
    let all = Tensor::from_slice(&flat, (n, DIM), &Device::Cpu).map_err(err)?;
    let all_t = all.t().map_err(err)?.contiguous().map_err(err)?;
    // neighbours, a block of rows at a time (a block's similarities are block × n floats)
    let block = (64 << 20) / (4 * n).max(1);
    let block = block.clamp(1, 1024);
    let mut nb: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let rows = block.min(n - start);
        let sims = all.narrow(0, start, rows).map_err(err)?.matmul(&all_t).map_err(err)?;
        let sims: Vec<f32> = sims.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        let part: Vec<Vec<(u32, f32)>> = sims
            .par_chunks(n)
            .enumerate()
            .map(|(r, row)| {
                let me = start + r;
                let mut best: Vec<(u32, f32)> = row.iter().enumerate().filter(|&(j, s)| j != me && *s >= threshold).map(|(j, s)| (j as u32, *s)).collect();
                if best.len() > k {
                    best.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
                    best.truncate(k);
                }
                best
            })
            .collect();
        nb.extend(part);
        start += rows;
    }
    // edges both ways (a neighbour of mine sees me too)
    let mut edges: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for (i, list) in nb.iter().enumerate() {
        for &(j, s) in list {
            if let Some(e) = edges.get_mut(i) {
                e.push((j, s));
            }
            if let Some(e) = edges.get_mut(j as usize) {
                e.push((i as u32, s));
            }
        }
    }
    for e in &mut edges {
        e.sort_unstable_by_key(|x| x.0);
        e.dedup_by_key(|x| x.0);
    }
    // Chinese Whispers, in a fixed shuffled order (a 64-bit LCG), 30 rounds or until still
    let mut order: Vec<usize> = (0..n).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..n).rev() {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        order.swap(i, (seed >> 33) as usize % (i + 1));
    }
    let mut label: Vec<usize> = (0..n).collect();
    let mut tally: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for _ in 0..30 {
        let mut changed = 0usize;
        for &i in &order {
            let Some(e) = edges.get(i) else { continue };
            if e.is_empty() {
                continue;
            }
            tally.clear();
            for &(j, s) in e {
                if let Some(&l) = label.get(j as usize) {
                    *tally.entry(l).or_insert(0.0) += s;
                }
            }
            let Some((&best, _)) = tally.iter().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0))) else { continue };
            if let Some(l) = label.get_mut(i)
                && *l != best
            {
                *l = best;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    // relabel: the largest group 0, ties by first face
    let mut size: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
    for (i, &l) in label.iter().enumerate() {
        let e = size.entry(l).or_insert((0, i));
        e.0 += 1;
    }
    let mut ranked: Vec<(usize, (usize, usize))> = size.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.0.cmp(&a.1.0).then(a.1.1.cmp(&b.1.1)));
    let new: std::collections::HashMap<usize, usize> = ranked.iter().enumerate().map(|(r, (l, _))| (*l, r)).collect();
    Ok(label.iter().map(|l| new.get(l).copied().unwrap_or(0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap_or_default().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    fn unit(seed: u32, near: Option<&[f32; DIM]>) -> [f32; DIM] {
        let mut x = seed.wrapping_mul(2_654_435_761);
        let mut v = [0f32; DIM];
        for (i, o) in v.iter_mut().enumerate() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let r = (x % 2000) as f32 / 1000.0 - 1.0;
            // a person's faces: their direction plus noise of about a third of its length
            *o = near.map_or(r, |c| c[i] + 0.03 * r);
        }
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.map(|a| a / n)
    }

    #[test]
    fn grouping_finds_people_and_keeps_strangers_apart() {
        let (a, b) = (unit(1, None), unit(2, None));
        let mut faces: Vec<[f32; DIM]> = (0..12).map(|i| unit(100 + i, Some(&a))).collect();
        faces.extend((0..5).map(|i| unit(200 + i, Some(&b))));
        faces.push(unit(999, None));
        let labels = group(&faces, 0.5, 20).expect("grouped");
        assert!(labels[..12].iter().all(|&l| l == 0), "{labels:?}");
        assert!(labels[12..17].iter().all(|&l| l == 1), "{labels:?}");
        assert_eq!(labels[17], 2);
        assert_eq!(group(&[], 0.5, 20).expect("empty"), Vec::<usize>::new());
        assert!(group(&faces, f32::NAN, 20).is_err() && group(&faces, 0.5, 0).is_err() && group(&faces, 2.0, 5).is_err());
        let mut bad = faces.clone();
        bad[3][0] = f32::INFINITY;
        assert!(group(&bad, 0.5, 20).is_err());
        // the same input, the same answer
        assert_eq!(group(&faces, 0.5, 20).expect("again"), labels);
    }

    #[test]
    fn the_template_maps_onto_itself_and_moves_are_recovered() {
        let m = similarity(&TEMPLATE).expect("template");
        assert!((m[0] - 1.0).abs() < 1e-5 && m[1].abs() < 1e-5 && m[2].abs() < 1e-3 && m[3].abs() < 1e-3, "{m:?}");
        // the template turned 30°, made twice as large and moved: the transform undoes it
        let (c, s) = (30f32.to_radians().cos() * 2.0, 30f32.to_radians().sin() * 2.0);
        let moved = TEMPLATE.map(|[x, y]| [c * x - s * y + 300.0, s * x + c * y - 40.0]);
        let [a, b, tx, ty] = similarity(&moved).expect("moved");
        for (p, q) in moved.iter().zip(&TEMPLATE) {
            let (x, y) = (a * p[0] - b * p[1] + tx, b * p[0] + a * p[1] + ty);
            assert!((x - q[0]).abs() < 1e-2 && (y - q[1]).abs() < 1e-2, "{x},{y} vs {q:?}");
        }
        assert_eq!(similarity(&[[5.0, 5.0]; 5]), None);
        assert_eq!(similarity(&[[f32::NAN, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), None);
    }

    #[test]
    fn alignment_refuses_bad_pictures_and_fills_outside_with_black() {
        assert!(align(&[0; 12], 2, 2, &TEMPLATE).is_none());
        assert!(align(&[], 0, 0, &TEMPLATE).is_none());
        assert!(align(&[0; 16], usize::MAX, 4, &TEMPLATE).is_none());
        let white = vec![255u8; 112 * 112 * 4];
        let a = align(&white, 112, 112, &TEMPLATE).expect("aligned");
        assert_eq!(a.len(), 3 * 112 * 112);
        assert!(a.iter().take(112 * 100).all(|v| (*v - 1.0).abs() < 1e-4));
        // a face near the picture's corner: what falls outside is black (−1)
        let corner = TEMPLATE.map(|[x, y]| [x - 60.0, y - 60.0]);
        let b = align(&white, 112, 112, &corner).expect("aligned");
        assert!((b[0] + 1.0).abs() < 1e-4);
    }

    /// Against onnxruntime's fingerprints (a test oracle, never shipped): set
    /// `FACE_ID_FIXTURE` to a folder made by the oracle script (aligned crops, expected
    /// fingerprints, pictures with their landmarks); skipped without it or without the model.
    #[test]
    fn fingerprints_match_onnxruntime() {
        let Some(dir) = std::env::var_os("FACE_ID_FIXTURE").map(std::path::PathBuf::from) else { return };
        let Some(models) = std::env::var_os("FILMCRAFT_MODELS").map(std::path::PathBuf::from) else { return };
        let model = crate::catalog::face_id_dir(&models).join(crate::catalog::FACE_ID.name);
        let mut net = FaceEmbedder::load(&model).expect("model");
        let crops = f32s(&dir.join("crops.f32"));
        let expected = f32s(&dir.join("expected.f32"));
        let n = expected.len() / DIM;
        assert!(n > 0 && crops.len() == n * 3 * INPUT * INPUT);
        let faces: Vec<Vec<f32>> = crops.as_chunks::<{ 3 * INPUT * INPUT }>().0.iter().map(|c| c.to_vec()).collect();
        let cos = |a: &[f32; DIM], i: usize| a.iter().zip(&expected[i * DIM..(i + 1) * DIM]).map(|(x, y)| x * y).sum::<f32>();
        let cpu = net.embed(&faces[..2]).expect("cpu");
        for (i, e) in cpu.iter().enumerate() {
            assert!(cos(e, i) > 0.9999, "cpu face {i}: {}", cos(e, i));
        }
        assert!(net.use_gpu());
        let t = std::time::Instant::now();
        let gpu = net.embed(&faces).expect("gpu");
        eprintln!("{n} faces on the GPU in {:?}", t.elapsed());
        for (i, e) in gpu.iter().enumerate() {
            assert!(cos(e, i) > 0.9995, "gpu face {i}: {}", cos(e, i));
        }
        // our alignment of the original pictures against OpenCV's
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("meta")).expect("json");
        for (i, m) in meta.as_array().expect("list").iter().enumerate() {
            let (w, h) = (m["w"].as_u64().expect("w") as usize, m["h"].as_u64().expect("h") as usize);
            let lm: Vec<f32> =
                m["landmarks"].as_array().expect("lm").iter().flat_map(|p| p.as_array().expect("pt").iter().map(|v| v.as_f64().expect("v") as f32)).collect();
            let lm: [[f32; 2]; 5] = std::array::from_fn(|k| [lm[2 * k], lm[2 * k + 1]]);
            let rgba = std::fs::read(dir.join(format!("image_{i}.rgba"))).expect("rgba");
            let ours = align(&rgba, w, h, &lm).expect("aligned");
            let diff = ours.iter().zip(&faces[i]).map(|(a, b)| (a - b).abs()).sum::<f32>() / ours.len() as f32;
            assert!(diff < 0.02, "face {i}: alignment differs by {diff}");
            let e = net.embed(&[ours]).expect("embed");
            assert!(cos(&e[0], i) > 0.995, "face {i} end to end: {}", cos(&e[0], i));
        }
    }
}
