//! The ACE-Step 1.5 DiT (`AceStepConditionGenerationModel` of the reference): the condition
//! encoder and the diffusion transformer that predicts the flow-matching velocity.
//!
//! **Condition encoder.** The text encoder's states go through a linear projection; the lyric
//! tokens' embeddings through the lyric encoder (a projection, 8 bidirectional pre-norm layers —
//! even layers attend within ±`sliding_window` positions, odd ones fully — and an RMSNorm); a
//! reference latent through the timbre encoder (4 such layers; the state of its first frame is
//! the timbre embedding). They are packed into one sequence: lyrics, timbre, text.
//!
//! **Diffusion transformer.** The input is the context (source latents and chunk mask) and the
//! noisy latents side by side, patched in pairs of frames (a stride-2 convolution, written as a
//! linear layer over the pair); the timestep `t` and `t − r` each go through a sinusoidal
//! embedding and an MLP, giving the output modulation and six per-layer AdaLN parameters
//! (shift, scale, gate for attention and for the MLP). Each of the 24 layers: AdaLN → rotary
//! self-attention (sliding window on even layers) → gated residual; RMSNorm → cross-attention to
//! the packed conditions → residual; AdaLN → SiLU MLP → gated residual. Then AdaLN and a
//! stride-2 transposed convolution back to frames.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::config::DitConfig;
use super::nn::{Attention, Block, Linear, Loader, Mask, Mlp, Rope, mask_tensor, rms};
use super::weights::SafeTensors;
use super::{R, err};

/// Width of the sinusoidal timestep embedding.
const TIME_DIM: usize = 256;
/// The reference multiplies timesteps by this before the sinusoid.
const TIME_SCALE: f32 = 1000.0;

struct TimeEmbed {
    l1: Linear,
    l2: Linear,
    proj: Linear,
}

impl TimeEmbed {
    fn load(ld: &mut Loader, p: &str, d: usize) -> R<TimeEmbed> {
        Ok(TimeEmbed {
            l1: ld.linear(&format!("{p}.linear_1"), TIME_DIM, d, true)?,
            l2: ld.linear(&format!("{p}.linear_2"), d, d, true)?,
            proj: ld.linear(&format!("{p}.time_proj"), d, 6 * d, true)?,
        })
    }

    /// `(temb [1, d], modulation [1, 6, d])` for timestep `t` (in f32, as the reference computes).
    fn fwd(&self, t: f32, d: usize, dev: &Device) -> R<(Tensor, Tensor)> {
        let half = TIME_DIM / 2;
        let ts = t * TIME_SCALE;
        let mut f = Vec::with_capacity(TIME_DIM);
        // -ln(10000) · i / half, in f32 like the reference
        let neg_ln = -(10_000f64.ln()) as f32;
        let freqs: Vec<f32> = (0..half).map(|i| (neg_ln * i as f32 / half as f32).exp()).collect();
        f.extend(freqs.iter().map(|w| (ts * w).cos()));
        f.extend(freqs.iter().map(|w| (ts * w).sin()));
        let x = Tensor::from_vec(f, (1, TIME_DIM), dev).map_err(err)?;
        let h = candle_nn::ops::silu(&self.l1.fwd(&x)?).map_err(err)?;
        let temb = self.l2.fwd(&h)?;
        let m = self.proj.fwd(&candle_nn::ops::silu(&temb).map_err(err)?)?.reshape((1, 6, d)).map_err(err)?;
        Ok((temb, m))
    }
}

struct Layer {
    self_norm: Tensor,
    self_attn: Attention,
    cross_norm: Tensor,
    cross_attn: Attention,
    mlp_norm: Tensor,
    mlp: Mlp,
    /// `[1, 6, d]`.
    table: Tensor,
    sliding: bool,
}

/// The packed conditions as each layer's cross-attention keys and values (computed once).
pub(crate) struct Cond {
    kv: Vec<(Tensor, Tensor)>,
}

/// Intermediate states of one velocity pass (oracle tests).
#[derive(Default)]
pub(crate) struct Trace {
    pub items: Vec<(&'static str, Tensor)>,
}

pub(crate) struct Dit {
    pub(crate) cfg: DitConfig,
    dev: Device,
    // condition encoder
    text_projector: Linear,
    lyric_embed: Linear,
    lyric_layers: Vec<Block>,
    lyric_norm: Tensor,
    timbre_embed: Linear,
    timbre_layers: Vec<Block>,
    timbre_norm: Tensor,
    // diffusion transformer
    proj_in: Linear,
    time: TimeEmbed,
    time_r: TimeEmbed,
    cond_embed: Linear,
    layers: Vec<Layer>,
    norm_out: Tensor,
    out_table: Tensor,
    proj_out: Linear,
    proj_out_bias: Tensor,
}

impl Dit {
    pub(crate) fn load(path: &Path, cfg: &DitConfig, dev: &Device, wdtype: DType) -> R<Dit> {
        let mut st = SafeTensors::open(path)?;
        let s = &cfg.shape;
        let d = s.hidden;
        let lat = cfg.latent;
        let p = cfg.patch;
        let mut ld = Loader { st: &mut st, dev, wdtype };
        let window = |i: usize| cfg.sliding.get(i).copied().unwrap_or(false);
        let mut lyric_layers = Vec::with_capacity(cfg.lyric_layers);
        for i in 0..cfg.lyric_layers {
            lyric_layers.push(Block::load(&mut ld, &format!("encoder.lyric_encoder.layers.{i}"), s, window(i))?);
        }
        let mut timbre_layers = Vec::with_capacity(cfg.timbre_layers);
        for i in 0..cfg.timbre_layers {
            timbre_layers.push(Block::load(&mut ld, &format!("encoder.timbre_encoder.layers.{i}"), s, window(i))?);
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let pre = format!("decoder.layers.{i}");
            layers.push(Layer {
                self_norm: ld.tensor(&format!("{pre}.self_attn_norm.weight"), &[d])?,
                self_attn: Attention::load(&mut ld, &format!("{pre}.self_attn"), s, false)?,
                cross_norm: ld.tensor(&format!("{pre}.cross_attn_norm.weight"), &[d])?,
                cross_attn: Attention::load(&mut ld, &format!("{pre}.cross_attn"), s, true)?,
                mlp_norm: ld.tensor(&format!("{pre}.mlp_norm.weight"), &[d])?,
                mlp: Mlp::load(&mut ld, &format!("{pre}.mlp"), d, s.ffn)?,
                table: ld.tensor(&format!("{pre}.scale_shift_table"), &[1, 6, d])?,
                sliding: window(i),
            });
        }
        // the patch convolution as a linear layer over a pair of frames: [d, c, p] → [d, p·c]
        let cin = 3 * lat;
        let w = ld.st.load("decoder.proj_in.1.weight", &[d, cin, p], &Device::Cpu)?;
        let w = w.permute((0, 2, 1)).map_err(err)?.reshape((d, p * cin)).map_err(err)?;
        let proj_in = Linear::new(to(&w, dev, wdtype)?, Some(ld.tensor("decoder.proj_in.1.bias", &[d])?))?;
        // the transposed convolution likewise: [d, lat, p] → rows (k · lat + o)
        let w = ld.st.load("decoder.proj_out.1.weight", &[d, lat, p], &Device::Cpu)?;
        let w = w.permute((2, 1, 0)).map_err(err)?.reshape((p * lat, d)).map_err(err)?;
        let proj_out = Linear::new(to(&w, dev, wdtype)?, None)?;
        Ok(Dit {
            cfg: cfg.clone(),
            dev: dev.clone(),
            text_projector: ld.linear("encoder.text_projector", cfg.text_dim, d, false)?,
            lyric_embed: ld.linear("encoder.lyric_encoder.embed_tokens", cfg.text_dim, d, true)?,
            lyric_layers,
            lyric_norm: ld.tensor("encoder.lyric_encoder.norm.weight", &[d])?,
            timbre_embed: ld.linear("encoder.timbre_encoder.embed_tokens", lat, d, true)?,
            timbre_layers,
            timbre_norm: ld.tensor("encoder.timbre_encoder.norm.weight", &[d])?,
            proj_in,
            time: TimeEmbed::load(&mut ld, "decoder.time_embed", d)?,
            time_r: TimeEmbed::load(&mut ld, "decoder.time_embed_r", d)?,
            cond_embed: ld.linear("decoder.condition_embedder", d, d, true)?,
            layers,
            norm_out: ld.tensor("decoder.norm_out.weight", &[d])?,
            out_table: ld.tensor("decoder.scale_shift_table", &[1, 2, d])?,
            proj_out,
            proj_out_bias: ld.tensor("decoder.proj_out.1.bias", &[lat])?,
        })
    }

    fn stack(&self, layers: &[Block], x: Tensor) -> R<Tensor> {
        let t = x.dim(1).map_err(err)?;
        let rope = Rope::new(self.cfg.shape.rope_theta, self.cfg.shape.head_dim, t, &self.dev)?;
        let window = mask_tensor(Mask::Window(self.cfg.window), t, &self.dev)?;
        let mut x = x;
        for l in layers {
            x = l.fwd(&x, &rope, if l.sliding { window.as_ref() } else { None })?;
        }
        Ok(x)
    }

    /// The lyric encoder over the lyric tokens' embeddings `[1, n, text_dim]` → `[1, n, d]`.
    pub(crate) fn lyrics(&self, embeds: &Tensor) -> R<Tensor> {
        let x = self.stack(&self.lyric_layers, self.lyric_embed.fwd(embeds)?)?;
        rms(&x, &self.lyric_norm, self.cfg.shape.eps)
    }

    /// The timbre embedding `[1, 1, d]` of a reference latent `[1, frames, latent]`.
    pub(crate) fn timbre(&self, latent: &Tensor) -> R<Tensor> {
        let x = self.stack(&self.timbre_layers, self.timbre_embed.fwd(latent)?)?;
        rms(&x, &self.timbre_norm, self.cfg.shape.eps)?.narrow(1, 0, 1).map_err(err)
    }

    /// The packed conditions `[1, lyrics + 1 + text, d]`: lyrics, timbre, projected text.
    pub(crate) fn encode(&self, text_hidden: &Tensor, lyric_embeds: &Tensor, timbre_latent: &Tensor) -> R<Tensor> {
        let text = self.text_projector.fwd(text_hidden)?;
        let lyr = self.lyrics(lyric_embeds)?;
        let tim = self.timbre(timbre_latent)?;
        Tensor::cat(&[&lyr, &tim, &text], 1).map_err(err)
    }

    /// Each layer's cross-attention keys and values over the packed conditions.
    pub(crate) fn condition(&self, enc: &Tensor) -> R<Cond> {
        let e = self.cond_embed.fwd(enc)?;
        let kv = self.layers.iter().map(|l| l.cross_attn.cross_kv(&e)).collect::<R<Vec<_>>>()?;
        Ok(Cond { kv })
    }

    /// The velocity `[1, frames, latent]` at timestep `t` for noisy latents `xt` and context `ctx`
    /// `[1, frames, 2 · latent]`.
    pub(crate) fn velocity(&self, xt: &Tensor, t: f32, ctx: &Tensor, cond: &Cond) -> R<Tensor> {
        self.velocity_traced(xt, t, ctx, cond, None)
    }

    pub(crate) fn velocity_traced(&self, xt: &Tensor, t: f32, ctx: &Tensor, cond: &Cond, mut trace: Option<&mut Trace>) -> R<Tensor> {
        let d = self.cfg.shape.hidden;
        let lat = self.cfg.latent;
        let p = self.cfg.patch;
        let frames = xt.dim(1).map_err(err)?;
        if ctx.dims() != [1, frames, 2 * lat] || xt.dims() != [1, frames, lat] {
            return Err(err("DiT input shapes do not match"));
        }
        if cond.kv.len() != self.layers.len() {
            return Err(err("conditions from another model"));
        }
        let (temb_t, mod_t) = self.time.fwd(t, d, &self.dev)?;
        // r = t: the second embedding is of t − r = 0
        let (temb_r, mod_r) = self.time_r.fwd(0.0, d, &self.dev)?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.items.push(("temb_t", temb_t.clone()));
            tr.items.push(("tproj_t", mod_t.clone()));
        }
        let temb = (temb_t + temb_r).map_err(err)?;
        let modulation = (mod_t + mod_r).map_err(err)?;
        // [ctx | xt], padded to whole patches, patched
        let mut h = Tensor::cat(&[ctx, xt], 2).map_err(err)?;
        let padded = frames.div_ceil(p) * p;
        if padded != frames {
            h = h.pad_with_zeros(1, 0, padded - frames).map_err(err)?;
        }
        let tokens = padded / p;
        let h = h.reshape((1, tokens, p * 3 * lat)).map_err(err)?;
        let mut h = self.proj_in.fwd(&h)?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.items.push(("dit_proj_in", h.clone()));
        }
        let rope = Rope::new(self.cfg.shape.rope_theta, self.cfg.shape.head_dim, tokens, &self.dev)?;
        let window = mask_tensor(Mask::Window(self.cfg.window), tokens, &self.dev)?;
        let eps = self.cfg.shape.eps;
        for (i, (l, (k, v))) in self.layers.iter().zip(&cond.kv).enumerate() {
            let m = l.table.broadcast_add(&modulation).map_err(err)?;
            let part = |j: usize| m.narrow(1, j, 1).map_err(err);
            let (shift, scale, gate, c_shift, c_scale, c_gate) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
            let n = rms(&h, &l.self_norm, eps)?.broadcast_mul(&(scale + 1.0).map_err(err)?).map_err(err)?.broadcast_add(&shift).map_err(err)?;
            let a = l.self_attn.self_fwd(&n, Some(&rope), if l.sliding { window.as_ref() } else { None })?;
            h = (h + a.broadcast_mul(&gate).map_err(err)?).map_err(err)?;
            let n = rms(&h, &l.cross_norm, eps)?;
            h = (&h + l.cross_attn.cross_fwd(&n, k, v)?).map_err(err)?;
            let n = rms(&h, &l.mlp_norm, eps)?.broadcast_mul(&(c_scale + 1.0).map_err(err)?).map_err(err)?.broadcast_add(&c_shift).map_err(err)?;
            h = (&h + l.mlp.fwd(&n)?.broadcast_mul(&c_gate).map_err(err)?).map_err(err)?;
            if let Some(tr) = trace.as_deref_mut() {
                if i == 0 {
                    tr.items.push(("dit_layer0", h.clone()));
                }
                if i == 1 {
                    tr.items.push(("dit_layer1", h.clone()));
                }
                if i + 1 == self.layers.len() {
                    tr.items.push(("dit_layer_last", h.clone()));
                }
            }
        }
        let m = self.out_table.broadcast_add(&temb.unsqueeze(1).map_err(err)?).map_err(err)?;
        let (shift, scale) = (m.narrow(1, 0, 1).map_err(err)?, m.narrow(1, 1, 1).map_err(err)?);
        let h = rms(&h, &self.norm_out, eps)?.broadcast_mul(&(scale + 1.0).map_err(err)?).map_err(err)?.broadcast_add(&shift).map_err(err)?;
        let y = self.proj_out.fwd(&h)?.reshape((1, padded, lat)).map_err(err)?.broadcast_add(&self.proj_out_bias).map_err(err)?;
        y.narrow(1, 0, frames).map_err(err)
    }

    /// The condition embedder's output (oracle tests).
    #[cfg(test)]
    pub(crate) fn embed_condition(&self, enc: &Tensor) -> R<Tensor> {
        self.cond_embed.fwd(enc)
    }

    #[cfg(test)]
    pub(crate) fn device(&self) -> &Device {
        &self.dev
    }
}

/// `w` (f32 on the CPU) in `dtype` on `dev`.
fn to(w: &Tensor, dev: &Device, dtype: DType) -> R<Tensor> {
    w.contiguous().map_err(err)?.to_dtype(dtype).map_err(err)?.to_device(dev).map_err(err)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::config::{parse_dit, tests::tiny_dit};
    use crate::model::qwen3::tests::fake_block;

    /// Random weights of a tiny DiT written to `dir/dit.safetensors`.
    pub(crate) fn fake(dir: &Path) -> (std::path::PathBuf, DitConfig) {
        let cfg = parse_dit(&tiny_dit()).unwrap();
        let s = cfg.shape.clone();
        let (d, lat, p, td) = (s.hidden, cfg.latent, cfg.patch, cfg.text_dim);
        let mut rng = crate::model::Noise::new(33);
        let mut ts: Vec<(String, Tensor)> = Vec::new();
        let dev = Device::Cpu;
        let put = |ts: &mut Vec<(String, Tensor)>, rng: &mut crate::model::Noise, name: &str, dims: &[usize], scale: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale).collect();
            ts.push((name.to_string(), Tensor::from_vec(v, dims, &dev).unwrap()));
        };
        let sc = 1.0 / (d as f32).sqrt();
        for i in 0..cfg.lyric_layers {
            fake_block(&format!("encoder.lyric_encoder.layers.{i}."), &s, &mut rng, &mut ts);
        }
        for i in 0..cfg.timbre_layers {
            fake_block(&format!("encoder.timbre_encoder.layers.{i}."), &s, &mut rng, &mut ts);
        }
        put(&mut ts, &mut rng, "encoder.text_projector.weight", &[d, td], sc);
        put(&mut ts, &mut rng, "encoder.lyric_encoder.embed_tokens.weight", &[d, td], sc);
        put(&mut ts, &mut rng, "encoder.lyric_encoder.embed_tokens.bias", &[d], 0.1);
        put(&mut ts, &mut rng, "encoder.lyric_encoder.norm.weight", &[d], 0.3);
        put(&mut ts, &mut rng, "encoder.timbre_encoder.embed_tokens.weight", &[d, lat], sc);
        put(&mut ts, &mut rng, "encoder.timbre_encoder.embed_tokens.bias", &[d], 0.1);
        put(&mut ts, &mut rng, "encoder.timbre_encoder.norm.weight", &[d], 0.3);
        for i in 0..cfg.layers {
            let pre = format!("decoder.layers.{i}");
            let mut sub = Vec::new();
            fake_block(&format!("{pre}."), &s, &mut rng, &mut sub);
            // the DiT layer names: self_attn.*, mlp.* from the block; its own norms and table
            for (n, t) in sub {
                if n.contains("layernorm") {
                    continue;
                }
                ts.push((n, t));
            }
            let mut cross = Vec::new();
            fake_block(&format!("{pre}.cross_x."), &s, &mut rng, &mut cross);
            for (n, t) in cross {
                if n.contains(".self_attn.") {
                    ts.push((n.replace(".cross_x.self_attn.", ".cross_attn."), t));
                }
            }
            put(&mut ts, &mut rng, &format!("{pre}.self_attn_norm.weight"), &[d], 0.3);
            put(&mut ts, &mut rng, &format!("{pre}.cross_attn_norm.weight"), &[d], 0.3);
            put(&mut ts, &mut rng, &format!("{pre}.mlp_norm.weight"), &[d], 0.3);
            put(&mut ts, &mut rng, &format!("{pre}.scale_shift_table"), &[1, 6, d], 0.1);
        }
        put(&mut ts, &mut rng, "decoder.proj_in.1.weight", &[d, 3 * lat, p], 0.3);
        put(&mut ts, &mut rng, "decoder.proj_in.1.bias", &[d], 0.1);
        put(&mut ts, &mut rng, "decoder.proj_out.1.weight", &[d, lat, p], sc);
        put(&mut ts, &mut rng, "decoder.proj_out.1.bias", &[lat], 0.1);
        for te in ["decoder.time_embed", "decoder.time_embed_r"] {
            put(&mut ts, &mut rng, &format!("{te}.linear_1.weight"), &[d, TIME_DIM], 0.06);
            put(&mut ts, &mut rng, &format!("{te}.linear_1.bias"), &[d], 0.1);
            put(&mut ts, &mut rng, &format!("{te}.linear_2.weight"), &[d, d], sc);
            put(&mut ts, &mut rng, &format!("{te}.linear_2.bias"), &[d], 0.1);
            put(&mut ts, &mut rng, &format!("{te}.time_proj.weight"), &[6 * d, d], sc * 0.3);
            put(&mut ts, &mut rng, &format!("{te}.time_proj.bias"), &[6 * d], 0.05);
        }
        put(&mut ts, &mut rng, "decoder.condition_embedder.weight", &[d, d], sc);
        put(&mut ts, &mut rng, "decoder.condition_embedder.bias", &[d], 0.1);
        put(&mut ts, &mut rng, "decoder.norm_out.weight", &[d], 0.3);
        put(&mut ts, &mut rng, "decoder.scale_shift_table", &[1, 2, d], 0.1);
        let path = dir.join("dit.safetensors");
        crate::model::weights::write(&path, &ts);
        (path, cfg)
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-dit-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_tiny_dit_runs_deterministically_on_odd_lengths() {
        let d = tmp("run");
        let (p, cfg) = fake(&d);
        let dit = Dit::load(&p, &cfg, &Device::Cpu, DType::F32).unwrap();
        let dev = dit.device().clone();
        let mut rng = crate::model::Noise::new(4);
        let text = Tensor::from_vec(rng.take(5 * 16), (1, 5, 16), &dev).unwrap();
        let lyr = Tensor::from_vec(rng.take(3 * 16), (1, 3, 16), &dev).unwrap();
        let tim = Tensor::from_vec(rng.take(9 * 4), (1, 9, 4), &dev).unwrap();
        let enc = dit.encode(&text, &lyr, &tim).unwrap();
        assert_eq!(enc.dims(), &[1, 3 + 1 + 5, 32]);
        let cond = dit.condition(&enc).unwrap();
        for frames in [7usize, 8, 15] {
            let xt = Tensor::from_vec(rng.take(frames * 4), (1, frames, 4), &dev).unwrap();
            let ctx = Tensor::from_vec(rng.take(frames * 8), (1, frames, 8), &dev).unwrap();
            let v = dit.velocity(&xt, 0.75, &ctx, &cond).unwrap();
            assert_eq!(v.dims(), &[1, frames, 4]);
            let again = dit.velocity(&xt, 0.75, &ctx, &cond).unwrap();
            assert_eq!(v.flatten_all().unwrap().to_vec1::<f32>().unwrap(), again.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            assert!(v.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()));
            // the timestep matters
            let other = dit.velocity(&xt, 0.3, &ctx, &cond).unwrap();
            assert!((other - &v).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() > 1e-4);
        }
        // mismatched shapes are errors
        let xt = Tensor::zeros((1, 6, 4), DType::F32, &dev).unwrap();
        let ctx = Tensor::zeros((1, 5, 8), DType::F32, &dev).unwrap();
        assert!(dit.velocity(&xt, 0.5, &ctx, &cond).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_patch_convolutions_match_their_definitions() {
        // proj_in / proj_out written as linear layers equal the strided convolutions
        let dev = Device::Cpu;
        let (dm, c, p, t) = (3usize, 2usize, 2usize, 4usize);
        let mut rng = crate::model::Noise::new(9);
        let w: Vec<f32> = rng.take(dm * c * p);
        let x: Vec<f32> = rng.take(t * c);
        // conv: out[t', o] = Σ_c Σ_k w[o, c, k] x[p t' + k, c]
        let mut want = vec![0f32; (t / p) * dm];
        for tp in 0..t / p {
            for o in 0..dm {
                for ci in 0..c {
                    for k in 0..p {
                        want[tp * dm + o] += w[(o * c + ci) * p + k] * x[(p * tp + k) * c + ci];
                    }
                }
            }
        }
        let wt = Tensor::from_vec(w.clone(), (dm, c, p), &dev).unwrap().permute((0, 2, 1)).unwrap().reshape((dm, p * c)).unwrap();
        let lin = Linear::new(wt, None).unwrap();
        let got = lin.fwd(&Tensor::from_vec(x.clone(), (1, t / p, p * c), &dev).unwrap()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5);
        }
        // transposed: out[p t' + k, o] = Σ_i w[i, o, k] h[t', i]
        let (din, cout) = (3usize, 2usize);
        let w: Vec<f32> = rng.take(din * cout * p);
        let h: Vec<f32> = rng.take(2 * din);
        let mut want = vec![0f32; 2 * p * cout];
        for tp in 0..2 {
            for k in 0..p {
                for o in 0..cout {
                    for i in 0..din {
                        want[(p * tp + k) * cout + o] += w[(i * cout + o) * p + k] * h[tp * din + i];
                    }
                }
            }
        }
        let wt = Tensor::from_vec(w, (din, cout, p), &dev).unwrap().permute((2, 1, 0)).unwrap().reshape((p * cout, din)).unwrap();
        let lin = Linear::new(wt, None).unwrap();
        let got = lin
            .fwd(&Tensor::from_vec(h, (1, 2, din), &dev).unwrap())
            .unwrap()
            .reshape((1, 2 * p, cout))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
