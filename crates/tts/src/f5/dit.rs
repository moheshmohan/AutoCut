//! The F5-TTS diffusion transformer (DiT backbone) that predicts the flow.
//!
//! - **Text embedding**: the character embedding (token 0 is the filler), plus a sinusoidal
//!   position table (`precompute_freqs_cis(text_dim, 4096)`: cosines then sines), through four
//!   ConvNeXt-V2 blocks (depthwise 7-tap convolution, LayerNorm, GELU, global response
//!   normalisation over time, residual).
//! - **Input embedding**: the noisy mel, the masked reference mel and the text embedding
//!   concatenated and projected; a convolutional position embedding (two grouped 31-tap
//!   convolutions with Mish) added.
//! - **Time embedding**: a 256-wide sinusoid of `1000·t` (sines then cosines) through a SiLU MLP.
//! - **Blocks**: adaptive LayerNorm-Zero (shift, scale and gate for attention and MLP from the
//!   time embedding), attention whose rotary embedding (interleaved pairs) turns only the first
//!   64 channels of the queries and keys — the first head — exactly as the reference applies
//!   it before splitting heads, and a tanh-GELU MLP.
//! - **Output**: a final adaptive LayerNorm (scale, then shift) and a projection to the mel.
//!
//! Classifier-free guidance runs the conditioned and unconditioned passes as one batch of two.
//! The text embeddings depend on neither the noise nor the time, so they are made once per
//! utterance ([`Dit::context`]).

use candle_core::{D, Device, Tensor};

use super::config::Config;
use crate::voxcpm::{Linear, Loader, R, err};

/// Positions in the text embedding's sinusoid table (the reference's ~44 s).
pub(crate) const TEXT_POSITIONS: usize = 4096;
const LN_EPS: f32 = 1e-6;

/// LayerNorm over the last dimension with `w` and `b`.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
    candle_nn::ops::layer_norm(&x.contiguous().map_err(err)?, w, b, LN_EPS).map_err(err)
}

/// Depthwise convolution along time of `x` `[b, n, c]` with `w` `[c, k]` (odd `k`, same
/// padding) and bias `[c]`: a sum of shifted, scaled copies (no transposes, one kernel launch
/// per tap on any device).
pub(crate) fn depthwise(x: &Tensor, w: &Tensor, bias: &Tensor) -> R<Tensor> {
    let (_, n, _) = x.dims3().map_err(err)?;
    let k = w.dim(1).map_err(err)?;
    let pad = k / 2;
    let xp = x.pad_with_zeros(1, pad, pad).map_err(err)?;
    let mut y = bias.broadcast_as(x.shape()).map_err(err)?.contiguous().map_err(err)?;
    for j in 0..k {
        let tap = w.narrow(1, j, 1).map_err(err)?.squeeze(1).map_err(err)?;
        y = (y + xp.narrow(1, j, n).map_err(err)?.broadcast_mul(&tap).map_err(err)?).map_err(err)?;
    }
    Ok(y)
}

/// `x · tanh(softplus(x))`.
fn mish(x: &Tensor) -> R<Tensor> {
    let sp = (x.exp().map_err(err)? + 1.0).map_err(err)?.log().map_err(err)?;
    x.mul(&sp.tanh().map_err(err)?).map_err(err)
}

/// A ConvNeXt-V2 block of the text embedding.
struct ConvNext {
    dw_w: Tensor,
    dw_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Linear,
    grn_g: Tensor,
    grn_b: Tensor,
    pw2: Linear,
}

impl ConvNext {
    fn load(ld: &mut Loader, p: &str, dim: usize, inner: usize) -> R<ConvNext> {
        Ok(ConvNext {
            dw_w: ld.tensor(&format!("{p}.dwconv.weight"), &[dim, 1, 7])?.squeeze(1).map_err(err)?,
            dw_b: ld.tensor(&format!("{p}.dwconv.bias"), &[dim])?,
            ln_w: ld.tensor(&format!("{p}.norm.weight"), &[dim])?,
            ln_b: ld.tensor(&format!("{p}.norm.bias"), &[dim])?,
            pw1: ld.linear(&format!("{p}.pwconv1"), dim, inner, true)?,
            grn_g: ld.tensor(&format!("{p}.grn.gamma"), &[1, 1, inner])?,
            grn_b: ld.tensor(&format!("{p}.grn.beta"), &[1, 1, inner])?,
            pw2: ld.linear(&format!("{p}.pwconv2"), inner, dim, true)?,
        })
    }

    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let h = depthwise(x, &self.dw_w, &self.dw_b)?;
        let h = layer_norm(&h, &self.ln_w, &self.ln_b)?;
        let h = self.pw1.fwd(&h)?.gelu_erf().map_err(err)?;
        // global response normalisation: the L2 norm over time, relative to its channel mean
        let g = h.sqr().map_err(err)?.sum_keepdim(1).map_err(err)?.sqrt().map_err(err)?;
        let nx = g.broadcast_div(&(g.mean_keepdim(D::Minus1).map_err(err)? + 1e-6).map_err(err)?).map_err(err)?;
        let h = (self.grn_g.broadcast_mul(&h.broadcast_mul(&nx).map_err(err)?).map_err(err)?.broadcast_add(&self.grn_b).map_err(err)? + &h).map_err(err)?;
        x.add(&self.pw2.fwd(&h)?).map_err(err)
    }
}

struct Block {
    ada: Linear,
    /// v, q and k in one matrix (v first: see `attend`).
    qkv: Linear,
    o: Linear,
    ff1: Linear,
    ff2: Linear,
}

pub(crate) struct Dit {
    cfg: Config,
    dev: Device,
    text_table: Tensor,
    text_pos: Tensor,
    text_blocks: Vec<ConvNext>,
    time1: Linear,
    time2: Linear,
    proj: Linear,
    pos1_w: Tensor,
    pos1_b: Tensor,
    pos2_w: Tensor,
    pos2_b: Tensor,
    inv_freq: Vec<f32>,
    blocks: Vec<Block>,
    norm_out: Linear,
    proj_out: Linear,
}

/// Inputs that stay fixed through an utterance's sampling: both guidance branches' text
/// embeddings and masked reference mel, and the rotary tables.
pub(crate) struct Context {
    /// `[2, n, mel + text_dim]`: (cond mel, text) for the conditioned branch, zeros and the
    /// filler text for the unconditioned one.
    cond_text: Tensor,
    /// `[2 · heads, n, head_dim / 2]`: the rotary angles' cosines and sines for every (batch,
    /// head) — the real ones for head 0, the identity (1, 0) for the others — so one kernel
    /// turns exactly the reference's first 64 channels.
    cos: Tensor,
    sin: Tensor,
    n: usize,
}

impl Dit {
    pub(crate) fn load(ld: &mut Loader, cfg: &Config) -> R<Dit> {
        let p = |n: &str| format!("{}{n}", cfg.dit);
        let (d, td) = (cfg.dim, cfg.text_dim);
        let text_table = ld.tensor(&p("text_embed.text_embed.weight"), &[cfg.text_rows, td])?;
        let text_blocks =
            (0..cfg.text_blocks).map(|i| ConvNext::load(ld, &p(&format!("text_embed.text_blocks.{i}")), td, cfg.text_ff)).collect::<R<Vec<_>>>()?;
        let inv_freq = ld.tensor(&p("rotary_embed.inv_freq"), &[cfg.head_dim / 2])?.to_device(&Device::Cpu).map_err(err)?.to_vec1::<f32>().map_err(err)?;
        let gc = d / cfg.pos_groups.max(1);
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = |n: &str| p(&format!("transformer_blocks.{i}.{n}"));
            blocks.push(Block {
                ada: ld.linear(&b("attn_norm.linear"), d, 6 * d, true)?,
                qkv: Linear::stack(&[
                    ld.linear(&b("attn.to_v"), d, d, true)?,
                    ld.linear(&b("attn.to_q"), d, d, true)?,
                    ld.linear(&b("attn.to_k"), d, d, true)?,
                ])?,
                o: ld.linear(&b("attn.to_out.0"), d, d, true)?,
                ff1: ld.linear(&b("ff.ff.0.0"), d, cfg.ff, true)?,
                ff2: ld.linear(&b("ff.ff.2"), cfg.ff, d, true)?,
            });
        }
        let dev = ld.dev.clone();
        Ok(Dit {
            text_table,
            text_pos: Tensor::from_vec(text_positions(td), (TEXT_POSITIONS, td), &dev).map_err(err)?,
            text_blocks,
            time1: ld.linear(&p("time_embed.time_mlp.0"), 256, d, true)?,
            time2: ld.linear(&p("time_embed.time_mlp.2"), d, d, true)?,
            proj: ld.linear(&p("input_embed.proj"), 2 * super::MELS + td, d, true)?,
            pos1_w: ld.tensor(&p("input_embed.conv_pos_embed.conv1d.0.weight"), &[d, gc, cfg.pos_kernel])?,
            pos1_b: ld.tensor(&p("input_embed.conv_pos_embed.conv1d.0.bias"), &[d])?,
            pos2_w: ld.tensor(&p("input_embed.conv_pos_embed.conv1d.2.weight"), &[d, gc, cfg.pos_kernel])?,
            pos2_b: ld.tensor(&p("input_embed.conv_pos_embed.conv1d.2.bias"), &[d])?,
            inv_freq,
            blocks,
            norm_out: ld.linear(&p("norm_out.linear"), d, 2 * d, true)?,
            proj_out: ld.linear(&p("proj_out"), d, super::MELS, true)?,
            cfg: cfg.clone(),
            dev,
        })
    }

    /// The text embedding of `ids` (vocabulary ids + 1, the filler 0 after them) over `n`
    /// frames: `[1, n, text_dim]`.
    fn text(&self, ids: &[u32], n: usize) -> R<Tensor> {
        let ids = Tensor::from_vec(ids.to_vec(), ids.len(), &self.dev).map_err(err)?;
        let mut x = self.text_table.index_select(&ids, 0).map_err(err)?;
        // positions past the table repeat its last row, as the reference clamps them
        let have = n.min(TEXT_POSITIONS);
        let mut pos = self.text_pos.narrow(0, 0, have).map_err(err)?;
        if n > have {
            let last = self.text_pos.narrow(0, TEXT_POSITIONS - 1, 1).map_err(err)?.repeat((n - have, 1)).map_err(err)?;
            pos = Tensor::cat(&[&pos, &last], 0).map_err(err)?;
        }
        x = x.add(&pos).map_err(err)?.unsqueeze(0).map_err(err)?;
        for b in &self.text_blocks {
            x = b.fwd(&x)?;
        }
        Ok(x)
    }

    /// The fixed inputs for `n` frames: the reference mel `cond` (`[n · MELS]`, zeros past the
    /// reference) and the text tokens (vocabulary ids, cut or padded to `n`).
    pub(crate) fn context(&self, cond: &[f32], tokens: &[u32], n: usize) -> R<Context> {
        let ids: Vec<u32> = (0..n).map(|i| tokens.get(i).map_or(0, |t| t.saturating_add(1))).collect();
        let text = self.text(&ids, n)?;
        let null_text = self.text(&vec![0; n], n)?;
        let cond = Tensor::from_vec(cond.to_vec(), (1, n, super::MELS), &self.dev).map_err(err)?;
        let a = Tensor::cat(&[&cond, &text], 2).map_err(err)?;
        let b = Tensor::cat(&[&cond.zeros_like().map_err(err)?, &null_text], 2).map_err(err)?;
        let cond_text = Tensor::cat(&[&a, &b], 0).map_err(err)?;
        // rotary angles in f32, as the reference multiplies them; head 0 of each branch turns
        let half = self.inv_freq.len();
        let heads = self.cfg.heads;
        let table = n.checked_mul(half).and_then(|x| x.checked_mul(2 * heads)).ok_or_else(|| err("too many frames"))?;
        let (mut cos, mut sin) = (Vec::with_capacity(table), Vec::with_capacity(table));
        for bh in 0..2 * heads {
            for t in 0..n {
                for f in &self.inv_freq {
                    if bh % heads == 0 {
                        let a = f64::from(t as f32 * f);
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    } else {
                        cos.push(1.0);
                        sin.push(0.0);
                    }
                }
            }
        }
        Ok(Context {
            cond_text,
            cos: Tensor::from_vec(cos, (2 * heads, n, half), &self.dev).map_err(err)?,
            sin: Tensor::from_vec(sin, (2 * heads, n, half), &self.dev).map_err(err)?,
            n,
        })
    }

    /// The time embedding of `t`: `[1, dim]`.
    fn time(&self, t: f32) -> R<Tensor> {
        let half = 128usize;
        let scale = (10000f64.ln() / (half as f64 - 1.0)) as f32;
        let mut e = Vec::with_capacity(2 * half);
        let args: Vec<f32> = (0..half).map(|i| (1000.0 * t) * (f64::from(i as f32 * -scale).exp() as f32)).collect();
        e.extend(args.iter().map(|a| f64::from(*a).sin() as f32));
        e.extend(args.iter().map(|a| f64::from(*a).cos() as f32));
        let x = Tensor::from_vec(e, (1, 2 * half), &self.dev).map_err(err)?;
        self.time2.fwd(&self.time1.fwd(&x)?.silu().map_err(err)?)
    }

    /// The grouped position convolution of `x` `[b, n, d]`.
    fn pos_conv(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
        let pad = self.cfg.pos_kernel / 2;
        let y = x.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?.conv1d(w, pad, 1, 1, self.cfg.pos_groups).map_err(err)?;
        y.broadcast_add(&b.reshape((1, (), 1)).map_err(err)?).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err)
    }

    /// Attention of `x` `[2, n, d]` (the two guidance branches).
    fn attend(&self, blk: &Block, x: &Tensor, ctx: &Context) -> R<Tensor> {
        let (b, n, _) = x.dims3().map_err(err)?;
        let (h, hd) = (self.cfg.heads, self.cfg.head_dim);
        // [b, n, 3·d] → [3, b, h, n, hd] in one copy: v, q and k come out contiguous. v is
        // first, so it starts the buffer: candle's Metal attention reads its inputs from their
        // offset in elements as if it were bytes (q and k are new tensors after the rotation)
        let qkv = blk.qkv.fwd(x)?.reshape((b, n, 3, h, hd)).map_err(err)?.permute((2, 0, 3, 1, 4)).map_err(err)?.contiguous().map_err(err)?;
        let part = |i: usize| -> R<Tensor> { qkv.get(i).map_err(err) };
        // rotary on head 0 only (the reference rotates the first 64 channels of the projection)
        let rotate = |y: Tensor| -> R<Tensor> {
            let flat = y.reshape((b * h, 1, n, hd)).map_err(err)?;
            candle_nn::rotary_emb::rope_i(&flat, &ctx.cos, &ctx.sin).map_err(err)?.reshape((b, h, n, hd)).map_err(err)
        };
        let v = part(0)?;
        let q = rotate(part(1)?)?;
        let k = rotate(part(2)?)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let o = if x.device().is_metal() {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0).map_err(err)?
        } else {
            let s = (q.matmul(&k.t().map_err(err)?.contiguous().map_err(err)?).map_err(err)? * scale).map_err(err)?;
            candle_nn::ops::softmax_last_dim(&s).map_err(err)?.matmul(&v).map_err(err)?
        };
        let o = o.transpose(1, 2).map_err(err)?.reshape((b, n, h * hd)).map_err(err)?;
        blk.o.fwd(&o)
    }

    /// The flow predicted for the noisy mel `x` `[1, n, MELS]` at time `t`, for the
    /// conditioned and the unconditioned branch: `[2, n, MELS]`.
    pub(crate) fn forward(&self, x: &Tensor, t: f32, ctx: &Context) -> R<Tensor> {
        let d = self.cfg.dim;
        let st = self.time(t)?.silu().map_err(err)?;
        let x2 = x.broadcast_as((2, ctx.n, super::MELS)).map_err(err)?;
        let inp = Tensor::cat(&[&x2, &ctx.cond_text], 2).map_err(err)?;
        let mut h = self.proj.fwd(&inp)?;
        let c = mish(&self.pos_conv(&h, &self.pos1_w, &self.pos1_b)?)?;
        let c = mish(&self.pos_conv(&c, &self.pos2_w, &self.pos2_b)?)?;
        h = (c + h).map_err(err)?;
        for blk in &self.blocks {
            // the modulation is the same for every frame and branch: LayerNorm's own scale and
            // shift (`norm(x)·(1 + scale) + shift`)
            let e = blk.ada.fwd(&st)?.squeeze(0).map_err(err)?; // [6d]
            let part = |i: usize| -> R<Tensor> { e.narrow(0, i * d, d).map_err(err) };
            let (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
            let norm = layer_norm(&h, &(scale_a + 1.0).map_err(err)?, &shift_a)?;
            let a = self.attend(blk, &norm, ctx)?;
            h = h.add(&a.broadcast_mul(&gate_a).map_err(err)?).map_err(err)?;
            let norm = layer_norm(&h, &(scale_m + 1.0).map_err(err)?, &shift_m)?;
            let f = blk.ff2.fwd(&blk.ff1.fwd(&norm)?.gelu().map_err(err)?)?;
            h = h.add(&f.broadcast_mul(&gate_m).map_err(err)?).map_err(err)?;
        }
        let e = self.norm_out.fwd(&st)?.squeeze(0).map_err(err)?;
        let (scale, shift) = (e.narrow(0, 0, d).map_err(err)?, e.narrow(0, d, d).map_err(err)?);
        self.proj_out.fwd(&layer_norm(&h, &(scale + 1.0).map_err(err)?, &shift)?)
    }
}

/// `precompute_freqs_cis(dim, 4096)`: `[4096, dim]`, the cosines then the sines.
fn text_positions(dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half).map(|i| (1.0 / 10000f64.powf(f64::from((2 * i) as f32 / dim as f32))) as f32).collect();
    let mut out = Vec::with_capacity(TEXT_POSITIONS * dim);
    for t in 0..TEXT_POSITIONS {
        let a: Vec<f32> = freqs.iter().map(|f| t as f32 * f).collect();
        out.extend(a.iter().map(|x| f64::from(*x).cos() as f32));
        out.extend(a.iter().map(|x| f64::from(*x).sin() as f32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_matches_a_loop() {
        let dev = Device::Cpu;
        let (n, c, k) = (9usize, 3usize, 7usize);
        let x: Vec<f32> = (0..n * c).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.25).collect();
        let w: Vec<f32> = (0..c * k).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
        let b = vec![0.5f32, -0.25, 0.0];
        let y = depthwise(
            &Tensor::from_vec(x.clone(), (1, n, c), &dev).unwrap(),
            &Tensor::from_vec(w.clone(), (c, k), &dev).unwrap(),
            &Tensor::from_vec(b.clone(), c, &dev).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        for t in 0..n {
            for ch in 0..c {
                let mut s = b[ch];
                for j in 0..k {
                    let src = t as isize + j as isize - 3;
                    if (0..n as isize).contains(&src) {
                        s += w[ch * k + j] * x[src as usize * c + ch];
                    }
                }
                assert!((y[t * c + ch] - s).abs() < 1e-5, "{t} {ch}");
            }
        }
    }

    /// Batched rotary tables (head 0 turned, the rest identity) against plain loops, on the CPU
    /// and (where there is one) the GPU.
    #[test]
    fn rope_turns_only_the_first_head() {
        let (b, h, n, hd) = (2usize, 3usize, 5usize, 4usize);
        let x: Vec<f32> = (0..b * h * n * hd).map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.1).collect();
        let freqs = [1.0f32, 0.1];
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for bh in 0..b * h {
            for t in 0..n {
                for f in freqs {
                    let a = t as f32 * f;
                    cos.push(if bh % h == 0 { a.cos() } else { 1.0 });
                    sin.push(if bh % h == 0 { a.sin() } else { 0.0 });
                }
            }
        }
        let mut want = x.clone();
        for bi in 0..b {
            for t in 0..n {
                for (i, f) in freqs.iter().enumerate() {
                    let a = t as f32 * f;
                    let base = ((bi * h) * n + t) * hd + 2 * i;
                    let (x0, x1) = (x[base], x[base + 1]);
                    want[base] = x0 * a.cos() - x1 * a.sin();
                    want[base + 1] = x1 * a.cos() + x0 * a.sin();
                }
            }
        }
        let mut devs = vec![Device::Cpu];
        if let Ok(d) = Device::new_metal(0) {
            devs.push(d);
        }
        for dev in devs {
            let t = Tensor::from_vec(x.clone(), (b * h, 1, n, hd), &dev).unwrap();
            let c = Tensor::from_vec(cos.clone(), (b * h, n, hd / 2), &dev).unwrap();
            let s = Tensor::from_vec(sin.clone(), (b * h, n, hd / 2), &dev).unwrap();
            let got = candle_nn::rotary_emb::rope_i(&t, &c, &s).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-5, "{dev:?} {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn mish_is_mish() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-30f32, -1.0, 0.0, 1.0, 30.0, 100.0], 6, &dev).unwrap();
        let y = mish(&x).unwrap().to_vec1::<f32>().unwrap();
        let want = |v: f64| v * (1.0 + v.exp()).ln().tanh();
        for (a, v) in y.iter().zip([-30.0, -1.0, 0.0, 1.0, 30.0, 100.0]) {
            assert!((f64::from(*a) - want(v)).abs() < 1e-5, "{v}: {a}");
        }
    }

    #[test]
    fn position_table_is_cos_then_sin() {
        let p = text_positions(8);
        assert_eq!(p.len(), TEXT_POSITIONS * 8);
        assert_eq!(&p[0..8], &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((p[8] - 1f32.cos()).abs() < 1e-6 && (p[12] - 1f32.sin()).abs() < 1e-6);
    }
}
