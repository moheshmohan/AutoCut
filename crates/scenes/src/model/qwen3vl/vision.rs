//! Qwen3-VL's vision encoder: 16-px patches (two frames deep; a still is repeated) embedded by a
//! linear layer, a learned 48×48 position table interpolated to the picture's grid, 2-D rotary
//! position on height and width, pre-norm transformer blocks with full attention within each
//! picture, and a merger folding each 2×2 block of patches into one token of the text model's
//! width. "DeepStack": the outputs of some middle blocks, merged the same way, are added to the
//! text model's first layers.
//!
//! Patches come in merge order — for each 2×2 block in raster order, its four patches in raster
//! order — so the merger's 2×2 groups are four consecutive rows.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module};

use super::super::{R, err};
use super::config::VisionConfig;
use super::{apply, take};

const LN_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

struct Merger {
    norm: LayerNorm,
    /// Normalise after folding the 2×2 block (DeepStack mergers), or before (the last one).
    post: bool,
    fc1: Linear,
    fc2: Linear,
}

pub struct Vision {
    cfg: VisionConfig,
    /// `[hidden, 3 · temporal · patch²]`
    patch_w: Tensor,
    patch_b: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    dtype: DType,
    dev: Device,
}

fn layer_norm(w: &mut HashMap<String, Tensor>, name: &str, n: usize) -> R<LayerNorm> {
    Ok(LayerNorm::new(take(w, &format!("{name}.weight"), &[n])?, take(w, &format!("{name}.bias"), &[n])?, LN_EPS))
}

fn linear(w: &mut HashMap<String, Tensor>, name: &str, out: usize, inp: usize) -> R<Linear> {
    Ok(Linear::new(take(w, &format!("{name}.weight"), &[out, inp])?, Some(take(w, &format!("{name}.bias"), &[out])?)))
}

fn merger(w: &mut HashMap<String, Tensor>, name: &str, c: &VisionConfig, post: bool) -> R<Merger> {
    let wide = c.hidden * c.merge * c.merge;
    Ok(Merger {
        norm: layer_norm(w, &format!("{name}.norm"), if post { wide } else { c.hidden })?,
        post,
        fc1: linear(w, &format!("{name}.linear_fc1"), wide, wide)?,
        fc2: linear(w, &format!("{name}.linear_fc2"), c.out_hidden, wide)?,
    })
}

impl Merger {
    fn forward(&self, x: &Tensor, wide: usize) -> R<Tensor> {
        let n = x.dim(0).map_err(err)?;
        let g = n / (wide / x.dim(1).map_err(err)?).max(1);
        let x = if self.post {
            self.norm.forward(&x.reshape((g, wide)).map_err(err)?).map_err(err)?
        } else {
            self.norm.forward(x).and_then(|y| y.reshape((g, wide))).map_err(err)?
        };
        let h = self.fc1.forward(&x).and_then(|h| h.gelu_erf()).map_err(err)?;
        self.fc2.forward(&h).map_err(err)
    }
}

/// The encoder's output for a batch of pictures: tokens `[pictures · merged, out_hidden]` and the
/// DeepStack features, each the same shape.
pub struct Encoded {
    pub tokens: Tensor,
    pub deepstack: Vec<Tensor>,
}

impl Vision {
    /// Build from the weights named `model.visual.*` (taken out of `w`).
    pub fn load(w: &mut HashMap<String, Tensor>, cfg: &VisionConfig, dtype: DType, dev: &Device) -> R<Vision> {
        let p = "model.visual";
        let c = cfg;
        let k = 3 * c.temporal * c.patch * c.patch;
        let conv = take(w, &format!("{p}.patch_embed.proj.weight"), &[c.hidden, 3, c.temporal, c.patch, c.patch])?;
        let patch_w = conv.reshape((c.hidden, k)).map_err(err)?;
        let patch_b = take(w, &format!("{p}.patch_embed.proj.bias"), &[c.hidden])?;
        let pos = take(w, &format!("{p}.pos_embed.weight"), &[c.grid * c.grid, c.hidden])?;
        let mut blocks = Vec::with_capacity(c.depth);
        for i in 0..c.depth {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: layer_norm(w, &format!("{b}.norm1"), c.hidden)?,
                norm2: layer_norm(w, &format!("{b}.norm2"), c.hidden)?,
                qkv: linear(w, &format!("{b}.attn.qkv"), 3 * c.hidden, c.hidden)?,
                proj: linear(w, &format!("{b}.attn.proj"), c.hidden, c.hidden)?,
                fc1: linear(w, &format!("{b}.mlp.linear_fc1"), c.intermediate, c.hidden)?,
                fc2: linear(w, &format!("{b}.mlp.linear_fc2"), c.hidden, c.intermediate)?,
            });
        }
        let merger = merger(w, &format!("{p}.merger"), c, false)?;
        let deepstack = (0..c.deepstack.len()).map(|i| merger_at(w, p, i, c)).collect::<R<Vec<_>>>()?;
        Ok(Vision { cfg: c.clone(), patch_w, patch_b, pos, blocks, merger, deepstack, dtype, dev: dev.clone() })
    }

    /// Values in one patch row: channels × frames deep × patch².
    pub fn patch_len(&self) -> usize {
        3 * self.cfg.temporal * self.cfg.patch * self.cfg.patch
    }

    /// Encode `n` pictures of `gh` × `gw` patches each, `patches` `[n · gh · gw, patch_len]`
    /// in merge order (f32 on the device).
    pub fn forward(&self, patches: &Tensor, n: usize, gh: usize, gw: usize) -> R<Encoded> {
        let c = &self.cfg;
        let s = gh * gw;
        if n == 0 || !gh.is_multiple_of(c.merge) || !gw.is_multiple_of(c.merge) || patches.dims() != [n * s, self.patch_len()] {
            return Err(err("Qwen3-VL vision: patches do not match the grid"));
        }
        let x = patches.to_dtype(self.dtype).and_then(|x| x.matmul(&self.patch_w.t()?)).and_then(|x| x.broadcast_add(&self.patch_b)).map_err(err)?;
        let pos = self.positions(gh, gw)?;
        let x = x.reshape((n, s, c.hidden)).and_then(|x| x.broadcast_add(&pos)).map_err(err)?;
        let (cos, sin) = self.rotary(gh, gw)?;
        let head = c.hidden / c.heads;
        let mut x = x;
        let mut deep = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            // attention within each picture
            let h = b.norm1.forward(&x).map_err(err)?;
            let qkv = apply(&b.qkv, &h).and_then(|q| q.reshape((n, s, 3, c.heads, head))).and_then(|q| q.permute((2, 0, 3, 1, 4))).map_err(err)?;
            let q = qkv.get(0).and_then(|q| q.contiguous()).map_err(err)?;
            let k = qkv.get(1).and_then(|k| k.contiguous()).map_err(err)?;
            let v = qkv.get(2).and_then(|v| v.contiguous()).map_err(err)?;
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin).map_err(err)?;
            let k = candle_nn::rotary_emb::rope(&k, &cos, &sin).map_err(err)?;
            let scale = 1.0 / (head as f64).sqrt();
            let att = q
                .matmul(&k.t().map_err(err)?)
                .and_then(|a| a.to_dtype(DType::F32))
                .and_then(|a| a * scale)
                .and_then(|a| candle_nn::ops::softmax_last_dim(&a))
                .and_then(|a| a.to_dtype(self.dtype))
                .and_then(|a| a.matmul(&v))
                .and_then(|o| o.permute((0, 2, 1, 3)))
                .and_then(|o| o.reshape((n, s, c.hidden)))
                .map_err(err)?;
            x = (x + apply(&b.proj, &att).map_err(err)?).map_err(err)?;
            let h = b.norm2.forward(&x).and_then(|h| apply(&b.fc1, &h)).and_then(|h| h.gelu()).and_then(|h| apply(&b.fc2, &h)).map_err(err)?;
            x = (x + h).map_err(err)?;
            if let Some(j) = c.deepstack.iter().position(|d| *d == i)
                && let Some(m) = self.deepstack.get(j)
            {
                let flat = x.reshape((n * s, c.hidden)).map_err(err)?;
                deep.push(m.forward(&flat, c.hidden * c.merge * c.merge)?);
            }
        }
        let flat = x.reshape((n * s, c.hidden)).map_err(err)?;
        let tokens = self.merger.forward(&flat, c.hidden * c.merge * c.merge)?;
        Ok(Encoded { tokens, deepstack: deep })
    }

    /// The position table interpolated (bilinearly, corners aligned) to `gh` × `gw`, in merge
    /// order: `[gh · gw, hidden]`.
    fn positions(&self, gh: usize, gw: usize) -> R<Tensor> {
        let g = self.cfg.grid;
        let m = self.cfg.merge;
        let lin = |steps: usize| -> Vec<f32> {
            if steps <= 1 {
                return vec![0.0];
            }
            let step = (g - 1) as f32 / (steps - 1) as f32;
            (0..steps).map(|i| i as f32 * step).collect()
        };
        let (hs, ws) = (lin(gh), lin(gw));
        let mut idx: [Vec<u32>; 4] = Default::default();
        let mut wt: [Vec<f32>; 4] = Default::default();
        for (r, c) in merge_order(gh, gw, m) {
            let (fy, fx) = (hs.get(r).copied().unwrap_or(0.0), ws.get(c).copied().unwrap_or(0.0));
            let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
            let (y1, x1) = ((fy.ceil() as usize).min(g - 1), (fx.ceil() as usize).min(g - 1));
            let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
            for (k, (yy, xx, w)) in
                [(y0, x0, (1.0 - dy) * (1.0 - dx)), (y0, x1, (1.0 - dy) * dx), (y1, x0, dy * (1.0 - dx)), (y1, x1, dy * dx)].into_iter().enumerate()
            {
                idx[k].push((yy * g + xx) as u32);
                wt[k].push(w);
            }
        }
        let s = gh * gw;
        let mut acc: Option<Tensor> = None;
        for k in 0..4 {
            let i = Tensor::from_vec(std::mem::take(&mut idx[k]), s, &self.dev).map_err(err)?;
            let w = Tensor::from_vec(std::mem::take(&mut wt[k]), (s, 1), &self.dev).and_then(|w| w.to_dtype(self.dtype)).map_err(err)?;
            let rows = self.pos.index_select(&i, 0).and_then(|r| r.broadcast_mul(&w)).map_err(err)?;
            acc = Some(match acc {
                Some(a) => (a + rows).map_err(err)?,
                None => rows,
            });
        }
        acc.ok_or_else(|| err("Qwen3-VL vision: empty grid"))
    }

    /// 2-D rotary tables for `gh` × `gw` patches in merge order: `(cos, sin)`, `[gh · gw, head / 2]`.
    fn rotary(&self, gh: usize, gw: usize) -> R<(Tensor, Tensor)> {
        let head = self.cfg.hidden / self.cfg.heads;
        let dim = head / 2;
        let inv: Vec<f64> = (0..dim / 2).map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f64 / dim as f64)).collect();
        let order = merge_order(gh, gw, self.cfg.merge);
        let mut cos = Vec::with_capacity(order.len() * dim);
        let mut sin = Vec::with_capacity(order.len() * dim);
        for (r, c) in order {
            for p in [r, c] {
                for f in &inv {
                    let a = p as f64 * f;
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        let s = gh * gw;
        let t = |v: Vec<f32>| Tensor::from_vec(v, (s, dim), &self.dev).and_then(|t| t.to_dtype(self.dtype)).map_err(err);
        Ok((t(cos)?, t(sin)?))
    }
}

fn merger_at(w: &mut HashMap<String, Tensor>, p: &str, i: usize, c: &VisionConfig) -> R<Merger> {
    merger(w, &format!("{p}.deepstack_merger_list.{i}"), c, true)
}

/// `(row, column)` of every patch of a `gh` × `gw` grid in merge order.
pub fn merge_order(gh: usize, gw: usize, m: usize) -> Vec<(usize, usize)> {
    let m = m.max(1);
    let mut out = Vec::with_capacity(gh * gw);
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    out.push((br * m + ir, bc * m + ic));
                }
            }
        }
    }
    out
}

/// The patch rows of one picture (`img`, already resized to `gh · patch` × `gw · patch`) in merge
/// order, normalised to −1..1, a still repeated over the patch's frames. Appends to `out`.
pub fn patchify(img: &crate::model::image::Rgb, patch: usize, temporal: usize, merge: usize, out: &mut Vec<f32>) {
    let (gh, gw) = (img.h / patch.max(1), img.w / patch.max(1));
    for (r, c) in merge_order(gh, gw, merge) {
        for ch in 0..3 {
            for _ in 0..temporal {
                for py in 0..patch {
                    for px in 0..patch {
                        let (y, x) = (r * patch + py, c * patch + px);
                        let v = img.px.get(y * img.w + x).map_or(0.0, |p| p[ch]);
                        out.push(v / 127.5 - 1.0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_order_groups_two_by_two_blocks() {
        assert_eq!(merge_order(2, 4, 2), vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(merge_order(4, 2, 2).len(), 8);
        assert!(merge_order(1, 1, 2).is_empty(), "a grid smaller than a block has no blocks");
    }

    #[test]
    fn patches_are_laid_out_channel_frame_row_column() {
        let (patch, temporal) = (2, 2);
        // a 4×4 picture: red = column, green = row, blue = 255
        let px = (0..16).map(|i| [(i % 4) as f32 * 85.0, (i / 4) as f32 * 85.0, 255.0]).collect();
        let img = crate::model::image::Rgb { w: 4, h: 4, px };
        let mut out = Vec::new();
        patchify(&img, patch, temporal, 2, &mut out);
        let len = 3 * temporal * patch * patch;
        assert_eq!(out.len(), 4 * len);
        // the second patch in merge order is row 0, column 1: its first red value is column 2
        let second = &out[len..2 * len];
        assert!((second[0] - (170.0 / 127.5 - 1.0)).abs() < 1e-6);
        // the still is repeated: frame 0 and frame 1 of a channel match
        assert_eq!(&second[0..4], &second[4..8]);
        // blue is 1 everywhere
        assert!(second[2 * temporal * patch * patch..].iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }
}
