//! D-FINE (Peng et al., "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution
//! Refinement", arXiv 2410.13842; Apache-2.0), the person detector of the profile pass, written
//! out by hand on the CPU ([`crate::layers`]).
//!
//! The weights are the Hugging Face Transformers port's safetensors file and its `config.json`
//! (`ustc-community/dfine-*`); batch norms are folded into the convolutions when loading, and a
//! RepVGG bottleneck's 1×1 branch into its 3×3 kernel. Every weight's shape is checked against
//! the layer it fills, so a file of another architecture is refused, never misread.
//!
//! - **Backbone** HGNetv2 (B0): a stem of 3×3, 2×2 and 1×1 convolutions with a max-pool branch;
//!   four stages of HG blocks (3×3 convolutions, or 1×1 then depthwise 5×5 in the light stages),
//!   their outputs concatenated and squeezed by two 1×1 convolutions; ReLU then a learnable
//!   affine after each; a depthwise 3×3/2 between stages.
//! - **Hybrid encoder**: each level projected to `d` channels; one transformer encoder layer on
//!   the coarsest level (2-D sine positions); a top-down then bottom-up path of RepNCSPELAN4
//!   blocks (SiLU), nearest 2× upsampling and depthwise 3×3/2 downsampling.
//! - **Decoder**: the `num_queries` best encoder tokens (by their best class) become the queries,
//!   their boxes the anchors plus a regression; each decoder layer has self-attention, deformable
//!   cross-attention (bilinear samples of every level around the query's box, no value or output
//!   projection), a sigmoid gate in place of the residual, and a feed-forward block. The boxes
//!   are refined as distributions over 33 bins of the distances from the first layer's box to
//!   its four sides (FDR); the class scores get a localisation-quality term from the last
//!   distribution's sharpness (LQE).
//!
//! Positions on a non-square map: the reference builds its sine positions with a transposing
//! mesh grid, which on its square 640 × 640 input gives token (row y, column x) the embedding
//! of (y, x) in the order [sin y, cos y, sin x, cos x]. This code gives every token that same
//! embedding, so a picture of another shape (a 16:9 frame at 640 × 352) is seen the way the
//! square ones were trained.

use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;

use crate::VisionError;
use crate::gpunn::{GpuUse, Nn};
use crate::layers::{Act, Attention, Backend, Conv, Cpu, Feat, Linear, Mlp, Norm, err, fold_bn, sample_add, sigmoid, softmax_rows};
use crate::safetensors::SafeTensors;

/// The parts of `config.json` the network needs.
#[derive(Clone, Debug, Deserialize)]
struct Config {
    d_model: usize,
    decoder_attention_heads: usize,
    decoder_layers: usize,
    decoder_n_points: serde_json::Value,
    decoder_offset_scale: f32,
    encoder_attention_heads: usize,
    encode_proj_layers: Vec<usize>,
    num_feature_levels: usize,
    num_queries: usize,
    max_num_bins: usize,
    top_prob_values: usize,
    layer_norm_eps: f32,
    batch_norm_eps: f32,
    positional_encoding_temperature: f32,
    backbone_config: BackboneConfig,
    #[serde(default)]
    id2label: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
struct BackboneConfig {
    out_indices: Vec<usize>,
    stage_downsample: Vec<bool>,
    stage_light_block: Vec<bool>,
    stage_kernel_size: Vec<usize>,
    stage_num_blocks: Vec<usize>,
    stage_numb_of_layers: Vec<usize>,
}

/// The checked numbers of a configuration.
#[derive(Clone, Debug)]
struct Shape {
    d: usize,
    dec_heads: usize,
    layers: usize,
    points: Vec<usize>,
    offset_scale: f32,
    enc_heads: usize,
    aifi_level: usize,
    levels: usize,
    queries: usize,
    bins: usize,
    topk: usize,
    ln_eps: f32,
    bn_eps: f32,
    temperature: f32,
    out_stages: Vec<usize>,
    bb: BackboneConfig,
}

fn bad_cfg(m: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("D-FINE config: {m}"))
}

impl Shape {
    fn from_config(c: Config) -> Result<Self, VisionError> {
        let points: Vec<usize> = match &c.decoder_n_points {
            serde_json::Value::Number(n) => {
                vec![n.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| bad_cfg("decoder_n_points"))?; c.num_feature_levels]
            }
            serde_json::Value::Array(a) => {
                a.iter().map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok())).collect::<Option<_>>().ok_or_else(|| bad_cfg("decoder_n_points"))?
            }
            _ => return Err(bad_cfg("decoder_n_points")),
        };
        let ok = (16..=1024).contains(&c.d_model)
            && (1..=4).contains(&c.num_feature_levels)
            && points.len() == c.num_feature_levels
            && points.iter().all(|p| (1..=16).contains(p))
            && (1..=32).contains(&c.decoder_attention_heads)
            && c.d_model.is_multiple_of(c.decoder_attention_heads)
            && (1..=32).contains(&c.encoder_attention_heads)
            && c.d_model.is_multiple_of(c.encoder_attention_heads)
            && c.d_model.is_multiple_of(4)
            && (1..=12).contains(&c.decoder_layers)
            && (1..=1000).contains(&c.num_queries)
            && (2..=256).contains(&c.max_num_bins)
            && c.max_num_bins.is_multiple_of(2)
            && (1..=c.max_num_bins + 1).contains(&c.top_prob_values)
            && c.encode_proj_layers.len() == 1
            && c.encode_proj_layers.iter().all(|l| *l < c.num_feature_levels)
            && c.backbone_config.out_indices.len() == c.num_feature_levels
            && c.backbone_config.out_indices.iter().all(|i| (1..=4).contains(i))
            && c.backbone_config.out_indices.windows(2).all(|w| w[0] < w[1])
            && [&c.backbone_config.stage_downsample.len(), &c.backbone_config.stage_light_block.len()].iter().all(|n| **n == 4)
            && c.backbone_config.stage_kernel_size.len() == 4
            && c.backbone_config.stage_kernel_size.iter().all(|k| k % 2 == 1 && *k <= 7)
            && c.backbone_config.stage_num_blocks.len() == 4
            && c.backbone_config.stage_num_blocks.iter().all(|n| (1..=16).contains(n))
            && c.backbone_config.stage_numb_of_layers.len() == 4
            && c.backbone_config.stage_numb_of_layers.iter().all(|n| (1..=16).contains(n))
            && [c.decoder_offset_scale, c.layer_norm_eps, c.batch_norm_eps, c.positional_encoding_temperature].iter().all(|v| v.is_finite() && *v > 0.0);
        if !ok {
            return Err(bad_cfg("numbers outside what this code implements"));
        }
        if c.id2label.get("0").is_some_and(|l| l != "person") {
            return Err(bad_cfg("class 0 is not \"person\""));
        }
        Ok(Self {
            d: c.d_model,
            dec_heads: c.decoder_attention_heads,
            layers: c.decoder_layers,
            points,
            offset_scale: c.decoder_offset_scale,
            enc_heads: c.encoder_attention_heads,
            aifi_level: c.encode_proj_layers.first().copied().unwrap_or(0),
            levels: c.num_feature_levels,
            queries: c.num_queries,
            bins: c.max_num_bins,
            topk: c.top_prob_values,
            ln_eps: c.layer_norm_eps,
            bn_eps: c.batch_norm_eps,
            temperature: c.positional_encoding_temperature,
            out_stages: c.backbone_config.out_indices.iter().map(|i| i - 1).collect(),
            bb: c.backbone_config,
        })
    }
}

/// Reads layers out of the weights file.
struct Loader<'a> {
    st: &'a SafeTensors,
    bn_eps: f32,
}

impl Loader<'_> {
    fn dims(&self, name: &str) -> Result<Vec<usize>, VisionError> {
        Ok(self.st.get(name)?.shape.clone())
    }

    /// A convolution `{p}.{conv}.weight` with batch norm `{p}.{norm}.*`, `[cout, cin/g, k, k]`.
    fn conv_bn(&self, p: &str, conv: &str, norm: &str, stride: usize, groups: Groups, act: Act) -> Result<Conv, VisionError> {
        let wn = format!("{p}.{conv}.weight");
        let d = self.dims(&wn)?;
        let [cout, per, k, k2] = d[..] else { return Err(err(format!("`{wn}` is {d:?}, not a convolution"))) };
        if k != k2 {
            return Err(err(format!("`{wn}` is not square")));
        }
        let (cin, g) = match groups {
            Groups::One => (per, 1),
            Groups::Depthwise if per == 1 => (cout, cout),
            Groups::Depthwise => return Err(err(format!("`{wn}` is not depthwise"))),
        };
        let mut w = self.st.take(&wn, &d)?;
        let bias = if self.st.has(&format!("{p}.{conv}.bias")) { Some(self.st.take(&format!("{p}.{conv}.bias"), &[cout])?) } else { None };
        let t = |n: &str| self.st.take(&format!("{p}.{norm}.{n}"), &[cout]);
        let b = fold_bn(&mut w, bias.as_deref(), &t("weight")?, &t("bias")?, &t("running_mean")?, &t("running_var")?, self.bn_eps)?;
        Conv::new(w, b, cin, cout, k, stride, (k - 1) / 2, g, act)
    }

    /// HGNetv2's convolution: `{p}.convolution`, `{p}.normalization`, then (with `act`) ReLU and
    /// the learnable affine `{p}.lab`.
    fn hg(&self, p: &str, stride: usize, groups: Groups, act: bool) -> Result<Conv, VisionError> {
        let act = if act {
            if self.st.has(&format!("{p}.lab.scale")) {
                Act::ReluAffine(self.st.scalar(&format!("{p}.lab.scale"))?, self.st.scalar(&format!("{p}.lab.bias"))?)
            } else {
                Act::Relu
            }
        } else {
            Act::None
        };
        self.conv_bn(p, "convolution", "normalization", stride, groups, act)
    }

    /// The encoder's ConvNormLayer: `{p}.conv`, `{p}.norm`.
    fn cn(&self, p: &str, stride: usize, groups: Groups, act: Act) -> Result<Conv, VisionError> {
        self.conv_bn(p, "conv", "norm", stride, groups, act)
    }

    fn linear(&self, p: &str, cin: usize, cout: usize) -> Result<Linear, VisionError> {
        let w = self.st.take(&format!("{p}.weight"), &[cout, cin])?;
        let b = self.st.take(&format!("{p}.bias"), &[cout])?;
        Linear::new(&w, Some(b), cin, cout)
    }

    fn norm(&self, p: &str, c: usize, eps: f32) -> Result<Norm, VisionError> {
        Norm::new(self.st.take(&format!("{p}.weight"), &[c])?, self.st.take(&format!("{p}.bias"), &[c])?, eps)
    }

    fn mlp(&self, p: &str, dims: &[usize]) -> Result<Mlp, VisionError> {
        let layers = dims.windows(2).enumerate().map(|(i, w)| self.linear(&format!("{p}.layers.{i}"), w[0], w[1])).collect::<Result<_, _>>()?;
        Ok(Mlp { layers })
    }

    fn attention(&self, p: &str, d: usize, heads: usize) -> Result<Attention, VisionError> {
        Ok(Attention {
            q: self.linear(&format!("{p}.q_proj"), d, d)?,
            k: self.linear(&format!("{p}.k_proj"), d, d)?,
            v: self.linear(&format!("{p}.v_proj"), d, d)?,
            o: self.linear(&format!("{p}.out_proj"), d, d)?,
            heads,
        })
    }
}

#[derive(Clone, Copy)]
enum Groups {
    One,
    Depthwise,
}

/// One layer of an HG block: a 3×3 convolution, or (light) a 1×1 then a depthwise one.
struct HgLayer {
    light: Option<Conv>,
    conv: Conv,
}

struct HgBlock {
    layers: Vec<HgLayer>,
    squeeze: Conv,
    excite: Conv,
    residual: bool,
}

impl HgBlock {
    fn forward<B: Backend>(&self, b: &B, x: &B::T) -> Result<B::T, VisionError> {
        let mut parts: Vec<B::T> = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let prev = parts.last().unwrap_or(x);
            let h = match &l.light {
                Some(a) => b.conv(&l.conv, &b.conv(a, prev, None)?, None)?,
                None => b.conv(&l.conv, prev, None)?,
            };
            parts.push(h);
        }
        let refs: Vec<&B::T> = std::iter::once(x).chain(parts.iter()).collect();
        let sq = b.conv(&self.squeeze, &b.concat(&refs)?, None)?;
        b.conv(&self.excite, &sq, if self.residual { Some(x) } else { None })
    }

    fn upload(&mut self, nn: &Nn) {
        for l in &mut self.layers {
            if let Some(a) = l.light.as_mut() {
                a.upload(nn);
            }
            l.conv.upload(nn);
        }
        self.squeeze.upload(nn);
        self.excite.upload(nn);
    }
}

struct Stage {
    down: Option<Conv>,
    blocks: Vec<HgBlock>,
}

struct Backbone {
    stem: [Conv; 5],
    stages: Vec<Stage>,
    out: Vec<usize>,
}

impl Backbone {
    fn load(l: &Loader<'_>, s: &Shape) -> Result<Self, VisionError> {
        let p = "model.backbone.model";
        let e = |n: &str, stride| l.hg(&format!("{p}.embedder.{n}"), stride, Groups::One, true);
        let stem = [e("stem1", 2)?, e("stem2a", 1)?, e("stem2b", 1)?, e("stem3", 2)?, e("stem4", 1)?];
        if stem[1].k != 2 || stem[2].k != 2 {
            return Err(err("the stem's 2×2 convolutions are not 2×2"));
        }
        let mut stages = Vec::with_capacity(4);
        for i in 0..4 {
            let sp = format!("{p}.encoder.stages.{i}");
            let down = if s.bb.stage_downsample.get(i).copied().unwrap_or(false) {
                Some(l.hg(&format!("{sp}.downsample"), 2, Groups::Depthwise, false)?)
            } else {
                None
            };
            let light = s.bb.stage_light_block.get(i).copied().unwrap_or(false);
            let (nb, nl) = (s.bb.stage_num_blocks.get(i).copied().unwrap_or(1), s.bb.stage_numb_of_layers.get(i).copied().unwrap_or(3));
            let mut blocks = Vec::with_capacity(nb);
            for j in 0..nb {
                let bp = format!("{sp}.blocks.{j}");
                let layers = (0..nl)
                    .map(|k| -> Result<HgLayer, VisionError> {
                        let lp = format!("{bp}.layers.{k}");
                        if light {
                            Ok(HgLayer {
                                light: Some(l.hg(&format!("{lp}.conv1"), 1, Groups::One, false)?),
                                conv: l.hg(&format!("{lp}.conv2"), 1, Groups::Depthwise, true)?,
                            })
                        } else {
                            Ok(HgLayer { light: None, conv: l.hg(&lp, 1, Groups::One, true)? })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let squeeze = l.hg(&format!("{bp}.aggregation.0"), 1, Groups::One, true)?;
                let excite = l.hg(&format!("{bp}.aggregation.1"), 1, Groups::One, true)?;
                blocks.push(HgBlock { layers, squeeze, excite, residual: j > 0 });
            }
            stages.push(Stage { down, blocks });
        }
        Ok(Self { stem, stages, out: s.out_stages.clone() })
    }

    fn forward<B: Backend>(&self, be: &B, x: &B::T) -> Result<Vec<B::T>, VisionError> {
        let [s1, s2a, s2b, s3, s4] = &self.stem;
        let a = be.pad_rb(&be.conv(s1, x, None)?)?;
        let b = be.conv(s2b, &be.pad_rb(&be.conv(s2a, &a, None)?)?, None)?;
        let pool = be.max_pool(&a, 2, 1, 0)?;
        let mut y = be.conv(s4, &be.conv(s3, &be.concat(&[&pool, &b])?, None)?, None)?;
        let mut outs = Vec::with_capacity(self.out.len());
        let last = self.out.iter().copied().max().unwrap_or(0);
        for (i, st) in self.stages.iter().enumerate().take(last + 1) {
            if let Some(d) = &st.down {
                y = be.conv(d, &y, None)?;
            }
            for blk in &st.blocks {
                y = blk.forward(be, &y)?;
            }
            if self.out.contains(&i) {
                // the stage's output is both kept and the next stage's input
                let kept = be.channels(&y, 0, B::shape(&y).1)?;
                outs.push(kept);
            }
        }
        Ok(outs)
    }

    fn upload(&mut self, nn: &Nn) {
        for c in &mut self.stem {
            c.upload(nn);
        }
        for st in &mut self.stages {
            if let Some(d) = st.down.as_mut() {
                d.upload(nn);
            }
            for b in &mut st.blocks {
                b.upload(nn);
            }
        }
    }
}

/// A CSP layer of RepVGG bottlenecks (their 1×1 branches folded into the 3×3 kernels).
struct CspRep {
    conv1: Conv,
    conv2: Conv,
    bottlenecks: Vec<Conv>,
    conv3: Option<Conv>,
}

impl CspRep {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self, VisionError> {
        let conv1 = l.cn(&format!("{p}.conv1"), 1, Groups::One, Act::Silu)?;
        let conv2 = l.cn(&format!("{p}.conv2"), 1, Groups::One, Act::Silu)?;
        let mut bottlenecks = Vec::new();
        for j in 0..32 {
            let bp = format!("{p}.bottlenecks.{j}");
            if !l.st.has(&format!("{bp}.conv1.conv.weight")) {
                break;
            }
            bottlenecks.push(rep_vgg(l, &bp)?);
        }
        let conv3 = if l.st.has(&format!("{p}.conv3.conv.weight")) { Some(l.cn(&format!("{p}.conv3"), 1, Groups::One, Act::Silu)?) } else { None };
        Ok(Self { conv1, conv2, bottlenecks, conv3 })
    }

    fn forward<B: Backend>(&self, b: &B, x: &B::T) -> Result<B::T, VisionError> {
        let mut a = b.conv(&self.conv1, x, None)?;
        for k in &self.bottlenecks {
            a = b.conv(k, &a, None)?;
        }
        let y = b.conv(&self.conv2, x, Some(&a))?;
        match &self.conv3 {
            Some(c) => b.conv(c, &y, None),
            None => Ok(y),
        }
    }

    fn upload(&mut self, nn: &Nn) {
        self.conv1.upload(nn);
        self.conv2.upload(nn);
        for k in &mut self.bottlenecks {
            k.upload(nn);
        }
        if let Some(c) = self.conv3.as_mut() {
            c.upload(nn);
        }
    }
}

/// A RepVGG block (3×3 and 1×1 convolutions with their batch norms, summed, then SiLU) as one
/// 3×3 convolution: the 1×1 kernel added to the 3×3 kernel's centre.
fn rep_vgg(l: &Loader<'_>, p: &str) -> Result<Conv, VisionError> {
    let read = |name: &str, k: usize| -> Result<(Vec<f32>, Vec<f32>, usize, usize), VisionError> {
        let wn = format!("{p}.{name}.conv.weight");
        let d = l.dims(&wn)?;
        let [cout, cin, kk, kk2] = d[..] else { return Err(err(format!("`{wn}` is not a convolution"))) };
        if kk != k || kk2 != k {
            return Err(err(format!("`{wn}` is not {k}×{k}")));
        }
        let mut w = l.st.take(&wn, &d)?;
        let t = |n: &str| l.st.take(&format!("{p}.{name}.norm.{n}"), &[cout]);
        let b = fold_bn(&mut w, None, &t("weight")?, &t("bias")?, &t("running_mean")?, &t("running_var")?, l.bn_eps)?;
        Ok((w, b, cin, cout))
    };
    let (mut w3, b3, cin, cout) = read("conv1", 3)?;
    let (w1, b1, cin1, cout1) = read("conv2", 1)?;
    if (cin, cout) != (cin1, cout1) {
        return Err(err(format!("`{p}`'s branches differ in shape")));
    }
    for o in 0..cout {
        for i in 0..cin {
            if let (Some(c), Some(v)) = (w3.get_mut((o * cin + i) * 9 + 4), w1.get(o * cin + i)) {
                *c += v;
            }
        }
    }
    let b: Vec<f32> = b3.iter().zip(&b1).map(|(a, b)| a + b).collect();
    Conv::new(w3, b, cin, cout, 3, 1, 1, 1, Act::Silu)
}

/// RepNCSPELAN4: a 1×1 convolution split in two halves, the second through two CSP + 3×3
/// stages, all four parts joined by a 1×1 convolution.
struct Elan {
    conv1: Conv,
    csp1: CspRep,
    conv2: Conv,
    csp2: CspRep,
    conv3: Conv,
    conv4: Conv,
}

impl Elan {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self, VisionError> {
        Ok(Self {
            conv1: l.cn(&format!("{p}.conv1"), 1, Groups::One, Act::Silu)?,
            csp1: CspRep::load(l, &format!("{p}.csp_rep1"))?,
            conv2: l.cn(&format!("{p}.conv2"), 1, Groups::One, Act::Silu)?,
            csp2: CspRep::load(l, &format!("{p}.csp_rep2"))?,
            conv3: l.cn(&format!("{p}.conv3"), 1, Groups::One, Act::Silu)?,
            conv4: l.cn(&format!("{p}.conv4"), 1, Groups::One, Act::Silu)?,
        })
    }

    fn forward<B: Backend>(&self, be: &B, x: &B::T) -> Result<B::T, VisionError> {
        let y = be.conv(&self.conv1, x, None)?;
        let c = B::shape(&y).1;
        let half = c / 2;
        let b = be.channels(&y, half, c)?;
        let cc = be.conv(&self.conv2, &self.csp1.forward(be, &b)?, None)?;
        let d = be.conv(&self.conv3, &self.csp2.forward(be, &cc)?, None)?;
        // y holds both halves in order: [a, b] then c and d
        be.conv(&self.conv4, &be.concat(&[&y, &cc, &d])?, None)
    }

    fn upload(&mut self, nn: &Nn) {
        for c in [&mut self.conv1, &mut self.conv2, &mut self.conv3, &mut self.conv4] {
            c.upload(nn);
        }
        self.csp1.upload(nn);
        self.csp2.upload(nn);
    }
}

/// The transformer encoder layer on the coarsest level (post-norm, GELU).
struct EncLayer {
    attn: Attention,
    norm1: Norm,
    fc1: Linear,
    fc2: Linear,
    norm2: Norm,
}

/// The error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    s * y
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

impl EncLayer {
    fn forward(&self, x: &[f32], pos: &[f32]) -> Result<Vec<f32>, VisionError> {
        let qk: Vec<f32> = x.iter().zip(pos).map(|(a, b)| a + b).collect();
        let a = self.attn.forward(&qk, x)?;
        let mut h: Vec<f32> = x.iter().zip(&a).map(|(a, b)| a + b).collect();
        self.norm1.apply(&mut h);
        let mut f = self.fc1.forward(&h)?;
        f.par_iter_mut().for_each(|v| *v = gelu(*v));
        let f = self.fc2.forward(&f)?;
        let mut y: Vec<f32> = h.iter().zip(&f).map(|(a, b)| a + b).collect();
        self.norm2.apply(&mut y);
        Ok(y)
    }
}

/// The 2-D sine positions of an `h` × `w` map of `d` channels (see the module docs).
fn positions(h: usize, w: usize, d: usize, temperature: f32) -> Vec<f32> {
    let q = d / 4;
    let omega: Vec<f32> = (0..q).map(|i| 1.0 / temperature.powf(i as f32 / q as f32)).collect();
    let mut out = Vec::with_capacity(h * w * d);
    for y in 0..h {
        for x in 0..w {
            let (fy, fx) = (y as f32, x as f32);
            out.extend(omega.iter().map(|o| (fy * o).sin()));
            out.extend(omega.iter().map(|o| (fy * o).cos()));
            out.extend(omega.iter().map(|o| (fx * o).sin()));
            out.extend(omega.iter().map(|o| (fx * o).cos()));
        }
    }
    out
}

/// One decoder layer.
struct DecLayer {
    self_attn: Attention,
    norm1: Norm,
    offsets: Linear,
    weights: Linear,
    /// Per sampling point (all levels in order): one over its level's number of points.
    point_scale: Vec<f32>,
    gate: Linear,
    gate_norm: Norm,
    fc1: Linear,
    fc2: Linear,
    norm3: Norm,
}

/// The encoder's memory as the decoder samples it.
struct Memory<'a> {
    /// `[tokens, d]`, levels in order.
    tokens: &'a [f32],
    /// Per level: first token, rows, columns.
    levels: &'a [(usize, usize, usize)],
}

impl DecLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, x: &[f32], refs: &[[f32; 4]], pos: &[f32], mem: &Memory<'_>, s: &Shape) -> Result<Vec<f32>, VisionError> {
        let d = s.d;
        let qk: Vec<f32> = x.iter().zip(pos).map(|(a, b)| a + b).collect();
        let a = self.self_attn.forward(&qk, x)?;
        let mut h1: Vec<f32> = x.iter().zip(&a).map(|(a, b)| a + b).collect();
        self.norm1.apply(&mut h1);
        let q: Vec<f32> = h1.iter().zip(pos).map(|(a, b)| a + b).collect();
        let offsets = self.offsets.forward(&q)?;
        let mut weights = self.weights.forward(&q)?;
        let total: usize = s.points.iter().sum();
        let heads = s.dec_heads;
        softmax_rows(&mut weights, total);
        let hd = d / heads;
        // the level of each sampling point
        let level_of: Vec<usize> = s.points.iter().enumerate().flat_map(|(l, n)| std::iter::repeat_n(l, *n)).collect();
        let nq = refs.len();
        if offsets.len() != nq * heads * total * 2 || weights.len() != nq * heads * total || h1.len() != nq * d {
            return Err(err("decoder sampling shapes"));
        }
        let mut cross = vec![0f32; nq * d];
        cross.par_chunks_mut(d).enumerate().for_each(|(qi, out)| {
            let Some(r) = refs.get(qi) else { return };
            for hh in 0..heads {
                let Some(o) = out.get_mut(hh * hd..(hh + 1) * hd) else { continue };
                for p in 0..total {
                    let at = (qi * heads + hh) * total + p;
                    let (Some(&ox), Some(&oy), Some(&wgt), Some(&lvl), Some(&ps)) =
                        (offsets.get(at * 2), offsets.get(at * 2 + 1), weights.get(at), level_of.get(p), self.point_scale.get(p))
                    else {
                        continue;
                    };
                    let Some(&(start, lh, lw)) = mem.levels.get(lvl) else { continue };
                    let lx = r[0] + ox * ps * r[2] * s.offset_scale;
                    let ly = r[1] + oy * ps * r[3] * s.offset_scale;
                    // grid_sample, align_corners = false: pixel centres at (i + 0.5) / size
                    let (px, py) = (lx * lw as f32 - 0.5, ly * lh as f32 - 0.5);
                    let vals = mem.tokens.get(start * d..).unwrap_or(&[]);
                    sample_add(vals, d, hh * hd, hd, lh, lw, px, py, wgt, o);
                }
            }
        });
        // the gate in place of the residual
        let mut both = Vec::with_capacity(nq * 2 * d);
        for (a, b) in h1.chunks_exact(d).zip(cross.chunks_exact(d)) {
            both.extend_from_slice(a);
            both.extend_from_slice(b);
        }
        let g = self.gate.forward(&both)?;
        let mut h2 = vec![0f32; nq * d];
        for (((o, gr), a), b) in h2.chunks_exact_mut(d).zip(g.chunks_exact(2 * d)).zip(h1.chunks_exact(d)).zip(cross.chunks_exact(d)) {
            for i in 0..d {
                o[i] = sigmoid(gr[i]) * a[i] + sigmoid(gr[d + i]) * b[i];
            }
        }
        self.gate_norm.apply(&mut h2);
        let mut f = self.fc1.forward(&h2)?;
        for v in &mut f {
            *v = v.max(0.0);
        }
        let f = self.fc2.forward(&f)?;
        let mut y: Vec<f32> = h2.iter().zip(&f).map(|(a, b)| (a + b).clamp(-65504.0, 65504.0)).collect();
        self.norm3.apply(&mut y);
        Ok(y)
    }
}

/// A person (class 0) found by D-FINE: `[cx, cy, w, h]` shares of the picture and the score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Found {
    pub cxcywh: [f32; 4],
    pub score: f32,
}

/// D-FINE.
pub struct Dfine {
    s: Shape,
    backbone: Backbone,
    input_proj: Vec<Conv>,
    aifi: EncLayer,
    lateral: Vec<Conv>,
    fpn: Vec<Elan>,
    down: Vec<(Conv, Conv)>,
    pan: Vec<Elan>,
    enc_output: Linear,
    enc_norm: Norm,
    enc_score: Linear,
    enc_bbox: Mlp,
    query_pos: Mlp,
    layers: Vec<DecLayer>,
    pre_bbox: Mlp,
    bbox_embed: Vec<Mlp>,
    class_last: Linear,
    lqe_last: Mlp,
    project: Vec<f32>,
    reg_scale: f32,
    pub(crate) gpu: Option<GpuUse>,
}

/// Longest side of the picture D-FINE looks at by default (it was trained at 640 × 640).
pub const PERSON_SIDE: usize = 640;

impl Dfine {
    /// Load from the weights file and its `config.json`.
    pub fn load(weights: &Path, config: &Path) -> Result<Self, VisionError> {
        let cfg_bytes = std::fs::read(config).map_err(|e| err(format!("{}: {e}", config.display())))?;
        if cfg_bytes.len() > 1 << 20 {
            return Err(bad_cfg("too large"));
        }
        let cfg: Config = serde_json::from_slice(&cfg_bytes).map_err(bad_cfg)?;
        let st = SafeTensors::read(weights)?;
        Self::from_parts(cfg, &st)
    }

    fn from_parts(cfg: Config, st: &SafeTensors) -> Result<Self, VisionError> {
        let s = Shape::from_config(cfg)?;
        let l = Loader { st, bn_eps: s.bn_eps };
        let d = s.d;
        let backbone = Backbone::load(&l, &s)?;
        let input_proj = (0..s.levels)
            .map(|i| {
                let p = format!("model.encoder_input_proj.{i}");
                let wn = format!("{p}.0.weight");
                let dims = l.dims(&wn)?;
                let [cout, cin, 1, 1] = dims[..] else { return Err(err(format!("`{wn}` is not a 1×1 convolution"))) };
                if cout != d {
                    return Err(err(format!("`{wn}` makes {cout} channels, not {d}")));
                }
                let mut w = st.take(&wn, &dims)?;
                let t = |n: &str| st.take(&format!("{p}.1.{n}"), &[cout]);
                let b = fold_bn(&mut w, None, &t("weight")?, &t("bias")?, &t("running_mean")?, &t("running_var")?, s.bn_eps)?;
                Conv::new(w, b, cin, cout, 1, 1, 0, 1, Act::None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ep = "model.encoder.encoder.0.layers.0";
        let ffn = l.dims(&format!("{ep}.fc1.weight"))?.first().copied().unwrap_or(0);
        let aifi = EncLayer {
            attn: l.attention(&format!("{ep}.self_attn"), d, s.enc_heads)?,
            norm1: l.norm(&format!("{ep}.self_attn_layer_norm"), d, s.ln_eps)?,
            fc1: l.linear(&format!("{ep}.fc1"), d, ffn)?,
            fc2: l.linear(&format!("{ep}.fc2"), ffn, d)?,
            norm2: l.norm(&format!("{ep}.final_layer_norm"), d, s.ln_eps)?,
        };
        let n = s.levels - 1;
        let lateral = (0..n).map(|i| l.cn(&format!("model.encoder.lateral_convs.{i}"), 1, Groups::One, Act::None)).collect::<Result<Vec<_>, _>>()?;
        let fpn = (0..n).map(|i| Elan::load(&l, &format!("model.encoder.fpn_blocks.{i}"))).collect::<Result<Vec<_>, _>>()?;
        let down = (0..n)
            .map(|i| -> Result<(Conv, Conv), VisionError> {
                let p = format!("model.encoder.downsample_convs.{i}");
                Ok((l.cn(&format!("{p}.conv1"), 1, Groups::One, Act::None)?, l.cn(&format!("{p}.conv2"), 2, Groups::Depthwise, Act::None)?))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pan = (0..n).map(|i| Elan::load(&l, &format!("model.encoder.pan_blocks.{i}"))).collect::<Result<Vec<_>, _>>()?;
        let classes = l.dims("model.enc_score_head.weight")?.first().copied().unwrap_or(0);
        if classes == 0 {
            return Err(err("no classes"));
        }
        let total: usize = s.points.iter().sum();
        let heads = s.dec_heads;
        let dec_ffn = l.dims("model.decoder.layers.0.fc1.weight")?.first().copied().unwrap_or(0);
        let layers = (0..s.layers)
            .map(|i| -> Result<DecLayer, VisionError> {
                let p = format!("model.decoder.layers.{i}");
                let point_scale = st.take(&format!("{p}.encoder_attn.num_points_scale"), &[total])?;
                Ok(DecLayer {
                    self_attn: l.attention(&format!("{p}.self_attn"), d, heads)?,
                    norm1: l.norm(&format!("{p}.self_attn_layer_norm"), d, s.ln_eps)?,
                    offsets: l.linear(&format!("{p}.encoder_attn.sampling_offsets"), d, heads * total * 2)?,
                    weights: l.linear(&format!("{p}.encoder_attn.attention_weights"), d, heads * total)?,
                    point_scale,
                    gate: l.linear(&format!("{p}.gateway.gate"), 2 * d, 2 * d)?,
                    gate_norm: l.norm(&format!("{p}.gateway.norm"), d, s.ln_eps)?,
                    fc1: l.linear(&format!("{p}.fc1"), d, dec_ffn)?,
                    fc2: l.linear(&format!("{p}.fc2"), dec_ffn, d)?,
                    norm3: l.norm(&format!("{p}.final_layer_norm"), d, s.ln_eps)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side = 4 * (s.bins + 1);
        let last = s.layers - 1;
        let lqe_hidden = l.dims(&format!("model.decoder.lqe_layers.{last}.reg_conf.layers.0.weight"))?.first().copied().unwrap_or(0);
        let reg_scale = st.scalar("model.decoder.reg_scale")?.abs();
        let up = st.scalar("model.decoder.up")?.abs();
        let project = weighting(s.bins, up, reg_scale)?;
        Ok(Self {
            backbone,
            input_proj,
            aifi,
            lateral,
            fpn,
            down,
            pan,
            enc_output: l.linear("model.enc_output.0", d, d)?,
            enc_norm: l.norm("model.enc_output.1", d, s.ln_eps)?,
            enc_score: l.linear("model.enc_score_head", d, classes)?,
            enc_bbox: l.mlp("model.enc_bbox_head", &[d, d, d, 4])?,
            query_pos: l.mlp("model.decoder.query_pos_head", &[4, 2 * d, d])?,
            layers,
            pre_bbox: l.mlp("model.decoder.pre_bbox_head", &[d, d, d, 4])?,
            bbox_embed: (0..s.layers).map(|i| l.mlp(&format!("model.decoder.bbox_embed.{i}"), &[d, d, d, side])).collect::<Result<_, _>>()?,
            class_last: l.linear(&format!("model.decoder.class_embed.{last}"), d, classes)?,
            lqe_last: l.mlp(&format!("model.decoder.lqe_layers.{last}.reg_conf"), &[4 * (s.topk + 1), lqe_hidden, 1])?,
            project,
            reg_scale,
            gpu: None,
            s,
        })
    }

    /// The raw outputs for a `[1, 3, H, W]` RGB picture (0..1; sides multiples of 32): every
    /// query's class logits (`[queries, classes]`) and box (`[cx, cy, w, h]`, shares).
    /// Put the convolutional parts on this machine's GPU (the CPU stays the fallback); `false`
    /// without one.
    pub fn use_gpu(&mut self) -> bool {
        let Some(nn) = Nn::shared() else { return false };
        self.backbone.upload(&nn);
        for c in self.input_proj.iter_mut().chain(self.lateral.iter_mut()) {
            c.upload(&nn);
        }
        for (a, b) in &mut self.down {
            a.upload(&nn);
            b.upload(&nn);
        }
        for e in self.fpn.iter_mut().chain(self.pan.iter_mut()) {
            e.upload(&nn);
        }
        self.gpu = Some(GpuUse::new(nn));
        true
    }

    /// Whether the convolutional parts run on the GPU (it has one that has not failed repeatedly).
    pub fn on_gpu(&self) -> bool {
        GpuUse::active(&self.gpu)
    }

    /// The hybrid encoder's outputs (one map per level) for a `[1, 3, H, W]` picture: the
    /// convolutions on `b`, the transformer layer on the CPU.
    pub(crate) fn encode<B: Backend>(&self, b: &B, x: &Feat) -> Result<Vec<Feat>, VisionError> {
        let s = &self.s;
        let d = s.d;
        let xi = b.input(x)?;
        let feats = self.backbone.forward(b, &xi)?;
        let mut proj: Vec<B::T> = feats.iter().zip(&self.input_proj).map(|(f, p)| b.conv(p, f, None)).collect::<Result<_, _>>()?;
        // the transformer layer on the coarsest level, on the CPU
        let al = s.aifi_level;
        if al >= proj.len() {
            return Err(err("missing level"));
        }
        let top = b.read(vec![proj.remove(al)])?.pop().ok_or_else(|| err("missing level"))?;
        let (th, tw) = (top.h, top.w);
        let pos = positions(th, tw, d, s.temperature);
        let enc = self.aifi.forward(&top.tokens(0), &pos)?;
        proj.insert(al, b.input(&Feat::from_tokens(&enc, d, th, tw)?)?);
        // top-down
        let n = s.levels;
        let mut inner: Vec<B::T> = vec![proj.pop().ok_or_else(|| err("no levels"))?];
        for idx in (1..n).rev() {
            let k = n - 1 - idx;
            let (lat, fpn) = (self.lateral.get(k).ok_or_else(|| err("lateral"))?, self.fpn.get(k).ok_or_else(|| err("fpn"))?);
            let high = b.conv(lat, inner.first().ok_or_else(|| err("fpn"))?, None)?;
            let up = b.upsample2(&high)?;
            if let Some(first) = inner.first_mut() {
                *first = high;
            }
            let low = proj.get(idx - 1).ok_or_else(|| err("fpn"))?;
            let out = fpn.forward(b, &b.concat(&[&up, low])?)?;
            inner.insert(0, out);
        }
        // bottom-up
        if inner.is_empty() {
            return Err(err("pan"));
        }
        let mut outs: Vec<B::T> = vec![inner.remove(0)];
        for idx in 0..n - 1 {
            let ((c1, c2), pan) = (self.down.get(idx).ok_or_else(|| err("pan"))?, self.pan.get(idx).ok_or_else(|| err("pan"))?);
            let low = b.conv(c2, &b.conv(c1, outs.last().ok_or_else(|| err("pan"))?, None)?, None)?;
            let high = inner.get(idx).ok_or_else(|| err("pan"))?;
            outs.push(pan.forward(b, &b.concat(&[&low, high])?)?);
        }
        b.read(outs)
    }

    /// The raw outputs for a `[1, 3, H, W]` RGB picture (0..1; sides multiples of 32): every
    /// query's class logits (`[queries, classes]`) and box (`[cx, cy, w, h]`, shares).
    pub(crate) fn raw(&self, x: &Feat) -> Result<(Vec<f32>, Vec<[f32; 4]>), VisionError> {
        if x.n != 1 || x.c != 3 || x.h == 0 || x.w == 0 || !x.h.is_multiple_of(32) || !x.w.is_multiple_of(32) {
            return Err(err(format!("D-FINE takes one RGB picture with sides multiples of 32, got {}×{}", x.w, x.h)));
        }
        let outs = match GpuUse::run(&self.gpu, |r| self.encode(r, x)) {
            Some(o) => o,
            None => self.encode(&Cpu, x)?,
        };
        self.decode(&outs)
    }

    /// The decoder over the encoder's outputs (on the CPU).
    pub(crate) fn decode(&self, outs: &[Feat]) -> Result<(Vec<f32>, Vec<[f32; 4]>), VisionError> {
        let s = &self.s;
        let d = s.d;
        // the decoder's memory and anchors
        let mut tokens: Vec<f32> = Vec::new();
        let mut levels: Vec<(usize, usize, usize)> = Vec::new();
        let mut anchors: Vec<[f32; 4]> = Vec::new();
        let mut valid: Vec<bool> = Vec::new();
        for (l, f) in outs.iter().enumerate() {
            levels.push((tokens.len() / d, f.h, f.w));
            tokens.extend(f.tokens(0));
            let wh = 0.05 * 2f32.powi(l as i32);
            for y in 0..f.h {
                for x in 0..f.w {
                    let a = [(x as f32 + 0.5) / f.w as f32, (y as f32 + 0.5) / f.h as f32, wh, wh];
                    let ok = a.iter().all(|v| *v > 0.01 && *v < 0.99);
                    valid.push(ok);
                    anchors.push(if ok { a.map(|v| (v / (1.0 - v)).ln()) } else { [f32::MAX; 4] });
                }
            }
        }
        let len = anchors.len();
        let mut masked = tokens.clone();
        for (row, ok) in masked.chunks_exact_mut(d).zip(&valid) {
            if !ok {
                row.fill(0.0);
            }
        }
        let mut memory_out = self.enc_output.forward(&masked)?;
        self.enc_norm.apply(&mut memory_out);
        let classes = self.enc_score.cout;
        let scores = self.enc_score.forward(&memory_out)?;
        let best: Vec<f32> = scores.chunks_exact(classes).map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
        let mut order: Vec<usize> = (0..len).collect();
        order.sort_by(|a, b| best.get(*b).unwrap_or(&0.0).total_cmp(best.get(*a).unwrap_or(&0.0)));
        order.truncate(s.queries);
        let nq = order.len();
        let picked: Vec<f32> = order.iter().flat_map(|i| memory_out.get(i * d..(i + 1) * d).unwrap_or(&[]).to_vec()).collect();
        let reg = self.enc_bbox.forward(&picked)?;
        let mut refs: Vec<[f32; 4]> = Vec::with_capacity(nq);
        for (qi, i) in order.iter().enumerate() {
            let a = anchors.get(*i).copied().unwrap_or([f32::MAX; 4]);
            let r = reg.get(qi * 4..qi * 4 + 4).unwrap_or(&[0.0; 4]);
            refs.push([sigmoid(r[0] + a[0]), sigmoid(r[1] + a[1]), sigmoid(r[2] + a[2]), sigmoid(r[3] + a[3])]);
        }
        // the decoder
        let mem = Memory { tokens: &tokens, levels: &levels };
        let mut x = picked;
        let mut prev_out = vec![0f32; nq * d];
        let side = 4 * (s.bins + 1);
        let mut corners = vec![0f32; nq * side];
        let mut init = refs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let flat: Vec<f32> = refs.iter().flatten().copied().collect();
            let mut pos = self.query_pos.forward(&flat)?;
            for v in &mut pos {
                *v = v.clamp(-10.0, 10.0);
            }
            let out = layer.forward(&x, &refs, &pos, &mem, s)?;
            if i == 0 {
                let pre = self.pre_bbox.forward(&out)?;
                for (k, r) in init.iter_mut().enumerate() {
                    let p = pre.get(k * 4..k * 4 + 4).unwrap_or(&[0.0; 4]);
                    let prev = refs.get(k).copied().unwrap_or([0.5; 4]);
                    for j in 0..4 {
                        r[j] = sigmoid(p[j] + inverse_sigmoid(prev[j]));
                    }
                }
            }
            let sum: Vec<f32> = out.iter().zip(&prev_out).map(|(a, b)| a + b).collect();
            let delta = self.bbox_embed.get(i).ok_or_else(|| err("bbox head"))?.forward(&sum)?;
            for (c, dlt) in corners.iter_mut().zip(&delta) {
                *c += dlt;
            }
            refs = (0..nq).map(|k| self.to_box(init.get(k).copied().unwrap_or([0.5; 4]), corners.get(k * side..(k + 1) * side).unwrap_or(&[]))).collect();
            prev_out = out.clone();
            x = out;
        }
        let mut logits = self.class_last.forward(&x)?;
        // localisation quality from the last distributions' sharpness
        let bins = s.bins + 1;
        let mut stats = Vec::with_capacity(nq * 4 * (s.topk + 1));
        for row in corners.chunks_exact(side) {
            for side_bins in row.chunks_exact(bins) {
                let mut p = side_bins.to_vec();
                softmax_rows(&mut p, bins);
                p.sort_by(|a, b| b.total_cmp(a));
                let top = p.get(..s.topk).unwrap_or(&[]);
                stats.extend_from_slice(top);
                stats.push(top.iter().sum::<f32>() / s.topk as f32);
            }
        }
        let quality = self.lqe_last.forward(&stats)?;
        for (row, q) in logits.chunks_exact_mut(classes).zip(&quality) {
            for v in row {
                *v += q;
            }
        }
        Ok((logits, refs))
    }

    /// A query's box from its first layer's box and its distance distributions (FDR).
    fn to_box(&self, p: [f32; 4], corners: &[f32]) -> [f32; 4] {
        let bins = self.s.bins + 1;
        let mut dist = [0f32; 4];
        for (k, c) in corners.chunks_exact(bins).take(4).enumerate() {
            let mut pr = c.to_vec();
            softmax_rows(&mut pr, bins);
            dist[k] = pr.iter().zip(&self.project).map(|(a, b)| a * b).sum();
        }
        let r = self.reg_scale.max(1e-6);
        let x1 = p[0] - (0.5 * r + dist[0]) * (p[2] / r);
        let y1 = p[1] - (0.5 * r + dist[1]) * (p[3] / r);
        let x2 = p[0] + (0.5 * r + dist[2]) * (p[2] / r);
        let y2 = p[1] + (0.5 * r + dist[3]) * (p[3] / r);
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
    }

    /// The people (class 0) in an RGB picture prepared by [`prepare`], best first: scores of at
    /// least `threshold`.
    pub fn people(&self, x: &Feat, threshold: f32) -> Result<Vec<Found>, VisionError> {
        let (logits, boxes) = self.raw(x)?;
        let classes = self.enc_score.cout;
        let mut out: Vec<Found> = logits
            .chunks_exact(classes)
            .zip(&boxes)
            .filter_map(|(l, b)| {
                let score = sigmoid(*l.first()?);
                (score >= threshold && b.iter().all(|v| v.is_finite()) && b[2] > 0.0 && b[3] > 0.0).then_some(Found { cxcywh: *b, score })
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }
}

fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x.max(1e-5) / (1.0 - x).max(1e-5)).ln()
}

/// D-FINE's bin positions W(n) for `bins` (32) bins: 0 in the middle, growing geometrically
/// to ±`up · reg_scale` at the ends and ±twice that beyond.
fn weighting(bins: usize, up: f32, reg_scale: f32) -> Result<Vec<f32>, VisionError> {
    let ub1 = up * reg_scale;
    let ub2 = ub1 * 2.0;
    let half = bins / 2;
    if bins < 4 || !ub1.is_finite() {
        return Err(err("bin positions"));
    }
    let step = (ub1 + 1.0).powf(2.0 / (bins as f32 - 2.0));
    let mut v = Vec::with_capacity(bins + 1);
    v.push(-ub2);
    for i in (1..half).rev() {
        v.push(-step.powi(i as i32) + 1.0);
    }
    v.push(0.0);
    for i in 1..half {
        v.push(step.powi(i as i32) - 1.0);
    }
    v.push(ub2);
    if v.len() != bins + 1 || v.iter().any(|x| !x.is_finite()) {
        return Err(err("bin positions"));
    }
    Ok(v)
}

/// An RGBA picture as D-FINE takes it: area-averaged to fit inside `side` (never enlarged), each
/// side rounded to a multiple of 32, RGB in 0..1, `[1, 3, H, W]`.
pub fn prepare(rgba: &[u8], w: usize, h: usize, side: usize) -> Result<Feat, VisionError> {
    if w == 0 || h == 0 || w > 1 << 15 || h > 1 << 15 {
        return Err(VisionError::Invalid(format!("a {w} × {h} picture cannot be analysed")));
    }
    let k = (side.clamp(64, 4096) as f64 / w.max(h) as f64).min(1.0);
    let round = |v: f64| ((v / 32.0).round() as usize).max(1) * 32;
    prepare_exact(rgba, w, h, round(w as f64 * k), round(h as f64 * k))
}

/// An RGBA picture resampled to exactly `ow` × `oh` (area average when shrinking, nearest
/// when enlarging), RGB in 0..1, `[1, 3, oh, ow]`.
pub fn prepare_exact(rgba: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Result<Feat, VisionError> {
    let len = w.checked_mul(h).and_then(|n| n.checked_mul(4)).ok_or_else(|| VisionError::Invalid("picture too large".into()))?;
    let src = rgba.get(..len).ok_or_else(|| VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())))?;
    if ow == 0 || oh == 0 || ow > 8192 || oh > 8192 {
        return Err(VisionError::Invalid(format!("cannot resample to {ow} × {oh}")));
    }
    let plane = ow * oh;
    let rows: Vec<Vec<[f32; 3]>> = (0..oh)
        .into_par_iter()
        .map(|y| {
            let (y0, y1) = (y * h / oh, ((y + 1) * h / oh).max(y * h / oh + 1).min(h));
            (0..ow)
                .map(|x| {
                    let (x0, x1) = (x * w / ow, ((x + 1) * w / ow).max(x * w / ow + 1).min(w));
                    // u64: even a whole picture averaged into one pixel cannot overflow it
                    let mut sum = [0u64; 3];
                    let mut n = 0u64;
                    for yy in y0..y1 {
                        for xx in x0..x1 {
                            if let Some(p) = src.get((yy * w + xx) * 4..(yy * w + xx) * 4 + 3) {
                                sum[0] += u64::from(p[0]);
                                sum[1] += u64::from(p[1]);
                                sum[2] += u64::from(p[2]);
                                n += 1;
                            }
                        }
                    }
                    let n = n.max(1) as f32 * 255.0;
                    [sum[0] as f32 / n, sum[1] as f32 / n, sum[2] as f32 / n]
                })
                .collect()
        })
        .collect();
    let mut data = vec![0f32; plane * 3];
    for (y, row) in rows.iter().enumerate() {
        for (x, p) in row.iter().enumerate() {
            let i = y * ow + x;
            data[i] = p[0];
            data[plane + i] = p[1];
            data[2 * plane + i] = p[2];
        }
    }
    Feat::new(data, 1, 3, oh, ow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_positions_match_the_reference_shape() {
        let w = weighting(32, 0.5, 4.0).unwrap();
        assert_eq!(w.len(), 33);
        assert_eq!(w[16], 0.0);
        assert_eq!((w[0], w[32]), (-4.0, 4.0));
        // symmetric, increasing, and the last regular bin at the upper bound minus one step
        for i in 0..16 {
            assert!((w[i] + w[32 - i]).abs() < 1e-5);
        }
        assert!(w.windows(2).all(|p| p[0] < p[1]));
        assert!((w[31] - (3f32.powf(15.0 / 15.0) - 1.0)).abs() < 1e-4);
        assert!(weighting(2, 0.5, 4.0).is_err());
        assert!(weighting(32, f32::INFINITY, 4.0).is_err());
    }

    #[test]
    fn erf_and_gelu_are_accurate() {
        // erf(0.5), erf(1), erf(2) to 7 places
        for (x, want) in [(0.5f32, 0.520_499_9f32), (1.0, 0.842_700_8), (2.0, 0.995_322_3), (-1.0, -0.842_700_8), (0.0, 0.0)] {
            assert!((erf(x) - want).abs() < 3e-7, "{x}");
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
    }

    #[test]
    fn positions_put_rows_first() {
        let p = positions(2, 3, 8, 10000.0);
        assert_eq!(p.len(), 2 * 3 * 8);
        // token (row 1, column 2): sin(1·1), sin(1·ω1), cos…, then sin(2·1)…
        let t = &p[(3 + 2) * 8..(3 + 2) * 8 + 8];
        assert!((t[0] - 1f32.sin()).abs() < 1e-6 && (t[2] - 1f32.cos()).abs() < 1e-6);
        assert!((t[4] - 2f32.sin()).abs() < 1e-6 && (t[6] - 2f32.cos()).abs() < 1e-6);
        assert!((t[1] - (1.0 / 100f32).sin()).abs() < 1e-6);
    }

    #[test]
    fn pictures_are_prepared_in_multiples_of_32() {
        let (w, h) = (1920, 1080);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i % 251) as u8).collect();
        let x = prepare(&rgba, w, h, 640).unwrap();
        assert_eq!((x.c, x.w, x.h), (3, 640, 352));
        assert!(x.v.iter().all(|v| (0.0..=1.0).contains(v)));
        // a small picture is not enlarged beyond rounding
        let small = prepare(&rgba[..100 * 50 * 4], 100, 50, 640).unwrap();
        assert_eq!((small.w, small.h), (96, 64));
        assert!(prepare(&rgba, 0, 10, 640).is_err());
        assert!(prepare(&rgba[..10], 100, 100, 640).is_err());
    }

    #[test]
    fn hostile_configs_are_refused() {
        let good = r#"{"d_model":128,"decoder_attention_heads":8,"decoder_layers":3,"decoder_n_points":[6,6],"decoder_offset_scale":0.5,
            "encoder_attention_heads":8,"encode_proj_layers":[1],"num_feature_levels":2,"num_queries":300,"max_num_bins":32,"top_prob_values":4,
            "layer_norm_eps":1e-5,"batch_norm_eps":1e-5,"positional_encoding_temperature":10000,
            "backbone_config":{"out_indices":[3,4],"stage_downsample":[false,true,true,true],"stage_light_block":[false,false,true,true],
            "stage_kernel_size":[3,3,5,5],"stage_num_blocks":[1,1,2,1],"stage_numb_of_layers":[3,3,3,3]},"id2label":{"0":"person"}}"#;
        let c: Config = serde_json::from_str(good).unwrap();
        assert!(Shape::from_config(c.clone()).is_ok());
        for bad in [
            Config { d_model: 0, ..c.clone() },
            Config { decoder_attention_heads: 7, ..c.clone() },
            Config { decoder_n_points: serde_json::json!([6]), ..c.clone() },
            Config { num_queries: 1_000_000, ..c.clone() },
            Config { max_num_bins: 31, ..c.clone() },
            Config { encode_proj_layers: vec![5], ..c.clone() },
            Config { layer_norm_eps: f32::NAN, ..c.clone() },
            Config { id2label: [("0".to_string(), "cat".to_string())].into(), ..c.clone() },
        ] {
            assert!(Shape::from_config(bad).is_err());
        }
        // a weights file without the layers is refused, not misread
        assert!(Dfine::from_parts(c, &SafeTensors::default()).is_err());
    }
}
