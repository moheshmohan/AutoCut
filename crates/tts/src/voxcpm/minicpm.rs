//! The MiniCPM-4 transformer stack. The text-semantic LM, the residual acoustic LM, LocEnc and
//! LocDiT are all instances, differing in size, depth, causality and (VoxCPM2's residual LM)
//! rotary embedding. Pre-norm decoder layers: RMSNorm → grouped-query attention with LongRoPE →
//! residual; RMSNorm → SiLU-gated MLP → residual (each residual branch scaled when the model
//! uses µP); a final RMSNorm.
//!
//! Two ways in: [`Stack::forward`] runs a whole sequence (and can fill a [`Cache`]);
//! [`Stack::step`] runs one new position against the cache, as autoregressive decoding does.

use candle_core::{D, Device, Tensor};

use super::config::{LmConfig, Stack as Shape};
use super::{Linear, Loader, R, err};

/// LongRoPE tables `[positions, head_dim / 2]`, already multiplied by the attention scaling
/// factor (as the reference does to both cos and sin).
pub(crate) struct Rope {
    cos: Tensor,
    sin: Tensor,
    len: usize,
}

impl Rope {
    pub(crate) fn new(lm: &LmConfig, head_dim: usize, len: usize, dev: &Device) -> R<Rope> {
        let rs = &lm.rope_scaling;
        let half = head_dim / 2;
        // the reference builds the table for max_position_embeddings positions and picks the
        // factors from that length; we only build as many rows as can be used
        let factors = if lm.max_position_embeddings > rs.original_max_position_embeddings { &rs.long_factor } else { &rs.short_factor };
        if factors.len() != half {
            return Err(err("rope factors do not match the head size"));
        }
        let orig = rs.original_max_position_embeddings as f64;
        let scale = lm.max_position_embeddings as f64 / orig;
        let attn = if orig > 1.0 { (1.0 + scale.ln() / orig.ln()).sqrt() } else { 1.0 };
        if !attn.is_finite() {
            return Err(err("rope scaling is not finite"));
        }
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for t in 0..len {
            for (i, f) in factors.iter().enumerate() {
                let inv = 1.0 / lm.rope_theta.powf((2 * i) as f64 / head_dim as f64);
                let a = t as f64 / f * inv;
                cos.push((a.cos() * attn) as f32);
                sin.push((a.sin() * attn) as f32);
            }
        }
        Ok(Rope { cos: Tensor::from_vec(cos, (len, half), dev).map_err(err)?, sin: Tensor::from_vec(sin, (len, half), dev).map_err(err)?, len })
    }

    /// Rotate `x` `[b, heads, t, head_dim]` for positions `start..start + t`.
    fn apply(&self, x: &Tensor, start: usize) -> R<Tensor> {
        let t = x.dim(2).map_err(err)?;
        if start.checked_add(t).is_none_or(|e| e > self.len) {
            return Err(err("the sequence is longer than the model allows"));
        }
        let c = self.cos.narrow(0, start, t).map_err(err)?;
        let s = self.sin.narrow(0, start, t).map_err(err)?;
        candle_nn::rotary_emb::rope(&x.contiguous().map_err(err)?, &c, &s).map_err(err)
    }
}

struct Layer {
    ln1: Tensor,
    ln2: Tensor,
    /// q, k and v projections in one matrix (one product instead of three).
    qkv: Linear,
    o: Linear,
    /// gate and up projections in one matrix.
    gate_up: Linear,
    down: Linear,
}

/// Keys and values of the positions run so far, per layer `[b, kv_heads, len, head_dim]`.
pub(crate) struct Cache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    len: usize,
}

impl Cache {
    pub(crate) fn new(layers: usize) -> Cache {
        Cache { k: vec![None; layers], v: vec![None; layers], len: 0 }
    }
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

pub(crate) struct Stack {
    shape: Shape,
    layers: Vec<Layer>,
    norm: Tensor,
    rope: Option<Rope>,
    /// Largest number of positions (the rope table's length).
    max_len: usize,
}

impl Stack {
    /// Load `<prefix>.layers.N.*` and `<prefix>.norm.weight`; `max_len` positions at most.
    pub(crate) fn load(ld: &mut Loader, prefix: &str, shape: &Shape, lm: &LmConfig, max_len: usize) -> R<Stack> {
        let (h, nh, kvh, hd, f) = (shape.hidden, shape.heads, shape.kv_heads, shape.head_dim, shape.ffn);
        let mut layers = Vec::with_capacity(shape.layers);
        for i in 0..shape.layers {
            let p = format!("{prefix}.layers.{i}");
            layers.push(Layer {
                ln1: ld.tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                ln2: ld.tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                qkv: Linear::stack(&[
                    ld.linear(&format!("{p}.self_attn.q_proj"), h, nh * hd, false)?,
                    ld.linear(&format!("{p}.self_attn.k_proj"), h, kvh * hd, false)?,
                    ld.linear(&format!("{p}.self_attn.v_proj"), h, kvh * hd, false)?,
                ])?,
                o: ld.linear(&format!("{p}.self_attn.o_proj"), nh * hd, h, false)?,
                gate_up: Linear::stack(&[ld.linear(&format!("{p}.mlp.gate_proj"), h, f, false)?, ld.linear(&format!("{p}.mlp.up_proj"), h, f, false)?])?,
                down: ld.linear(&format!("{p}.mlp.down_proj"), f, h, false)?,
            });
        }
        let norm = ld.tensor(&format!("{prefix}.norm.weight"), &[h])?;
        let rope = if shape.rope { Some(Rope::new(lm, hd, max_len, ld.dev)?) } else { None };
        Ok(Stack { shape: shape.clone(), layers, norm, rope, max_len })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.shape.hidden
    }

    fn rms(&self, x: &Tensor, w: &Tensor) -> R<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous().map_err(err)?, w, self.shape.eps as f32).map_err(err)
    }

    /// Attention of `x` `[b, t, hidden]` at positions `start..start + t` over the cached
    /// positions and itself; causal within `x` when `causal`.
    fn attend(&self, l: &Layer, x: &Tensor, start: usize, causal: bool, past: Option<(&Tensor, &Tensor)>) -> R<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3().map_err(err)?;
        let (nh, kvh, hd) = (self.shape.heads, self.shape.kv_heads, self.shape.head_dim);
        let g = nh / kvh;
        let heads = |y: Tensor, n: usize| -> R<Tensor> { y.reshape((b, t, n, hd)).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err) };
        let qkv = l.qkv.fwd(x)?;
        let part = |from: usize, n: usize| -> R<Tensor> { heads(qkv.narrow(D::Minus1, from * hd, n * hd).map_err(err)?, n) };
        let mut q = part(0, nh)?;
        let mut k = part(nh, kvh)?;
        let v = part(nh + kvh, kvh)?;
        if let Some(r) = &self.rope {
            q = r.apply(&q, start)?;
            k = r.apply(&k, start)?;
        }
        let (k_all, v_all) = match past {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2).map_err(err)?, Tensor::cat(&[pv, &v], 2).map_err(err)?),
            None => (k.clone(), v.clone()),
        };
        let total = k_all.dim(2).map_err(err)?;
        let scale = 1.0 / (hd as f64).sqrt();
        if x.device().is_metal() {
            // one fused kernel on the GPU (grouped heads and the causal mask included)
            let o = fused_attention(&q, &k_all, &v_all, causal && t > 1, scale as f32)?;
            let o = o.transpose(1, 2).map_err(err)?.reshape((b, t, nh * hd)).map_err(err)?;
            return Ok((l.o.fwd(&o)?, k_all, v_all));
        }
        // grouped queries: head h reads key/value head h / g
        let qg = q.reshape((b, kvh, g * t, hd)).map_err(err)?;
        let mut scores = (qg.matmul(&k_all.t().map_err(err)?.contiguous().map_err(err)?).map_err(err)? * scale).map_err(err)?;
        if causal && t > 1 {
            let past_len = total - t;
            let mut m = vec![0f32; g * t * total];
            for gi in 0..g {
                for i in 0..t {
                    for j in (past_len + i + 1)..total {
                        if let Some(x) = m.get_mut((gi * t + i) * total + j) {
                            *x = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let mask = Tensor::from_vec(m, (g * t, total), x.device()).map_err(err)?;
            scores = scores.broadcast_add(&mask).map_err(err)?;
        }
        let p = candle_nn::ops::softmax_last_dim(&scores).map_err(err)?;
        let o = p.matmul(&v_all).map_err(err)?; // [b, kvh, g·t, hd]
        let o = o.reshape((b, nh, t, hd)).map_err(err)?.transpose(1, 2).map_err(err)?.reshape((b, t, nh * hd)).map_err(err)?;
        Ok((l.o.fwd(&o)?, k_all, v_all))
    }

    fn mlp(&self, l: &Layer, x: &Tensor) -> R<Tensor> {
        let f = self.shape.ffn;
        let gu = l.gate_up.fwd(x)?;
        let a = candle_nn::ops::silu(&gu.narrow(D::Minus1, 0, f).map_err(err)?).map_err(err)?;
        l.down.fwd(&(a * gu.narrow(D::Minus1, f, f).map_err(err)?).map_err(err)?)
    }

    fn layer(&self, l: &Layer, x: &Tensor, start: usize, causal: bool, past: Option<(&Tensor, &Tensor)>) -> R<(Tensor, Tensor, Tensor)> {
        let rs = self.shape.residual_scale;
        let h = self.rms(x, &l.ln1)?;
        let (a, k, v) = self.attend(l, &h, start, causal, past)?;
        let x = if rs == 1.0 { (x + a).map_err(err)? } else { (x + (a * rs).map_err(err)?).map_err(err)? };
        let h = self.rms(&x, &l.ln2)?;
        let m = self.mlp(l, &h)?;
        let x = if rs == 1.0 { (x + m).map_err(err)? } else { (x + (m * rs).map_err(err)?).map_err(err)? };
        Ok((x, k, v))
    }

    /// Run `x` `[b, t, hidden]` from position 0. With `cache`, its keys and values replace the
    /// cache's contents (the prefill of autoregressive decoding).
    pub(crate) fn forward(&self, x: &Tensor, causal: bool, mut cache: Option<&mut Cache>) -> R<Tensor> {
        let t = x.dim(1).map_err(err)?;
        if t > self.max_len {
            return Err(err("the sequence is longer than the model allows"));
        }
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            let (y, k, v) = self.layer(l, &x, 0, causal, None)?;
            x = y;
            if let Some(c) = cache.as_deref_mut() {
                if let Some(s) = c.k.get_mut(i) {
                    *s = Some(k);
                }
                if let Some(s) = c.v.get_mut(i) {
                    *s = Some(v);
                }
            }
        }
        if let Some(c) = cache {
            c.len = t;
        }
        self.rms(&x, &self.norm)
    }

    /// Run one new position `x` `[b, hidden]` against `cache` and append it.
    pub(crate) fn step(&self, x: &Tensor, cache: &mut Cache) -> R<Tensor> {
        let start = cache.len;
        if start >= self.max_len {
            return Err(err("the utterance is longer than the model allows"));
        }
        let mut x = x.unsqueeze(1).map_err(err)?;
        for (i, l) in self.layers.iter().enumerate() {
            let past = match (cache.k.get(i).and_then(|k| k.as_ref()), cache.v.get(i).and_then(|v| v.as_ref())) {
                (Some(k), Some(v)) => Some((k, v)),
                _ => None,
            };
            let (y, k, v) = self.layer(l, &x, start, false, past)?;
            x = y;
            if let Some(s) = cache.k.get_mut(i) {
                *s = Some(k);
            }
            if let Some(s) = cache.v.get_mut(i) {
                *s = Some(v);
            }
        }
        cache.len = start + 1;
        self.rms(&x, &self.norm)?.squeeze(1).map_err(err)
    }

    /// A fresh cache for this stack.
    pub(crate) fn cache(&self) -> Cache {
        Cache::new(self.layers.len())
    }
}

/// Queries above this length use candle's full fused kernel; from 2 up to it, its vector kernel
/// (meant for one query) gives wrong results, so such sequences are padded past it and masked.
const SDPA_VECTOR_MAX: usize = 8;

/// `softmax(q kᵀ · scale) v` with candle's fused Metal kernel: `q` `[b, heads, t, d]`, `k`/`v`
/// `[b, kv_heads, total, d]` (the queries are the last `t` positions).
fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, scale: f32) -> R<Tensor> {
    let (b, nh, t, _) = q.dims4().map_err(err)?;
    let total = k.dim(2).map_err(err)?;
    let (k, v) = (k.contiguous().map_err(err)?, v.contiguous().map_err(err)?);
    if t == 1 || t > SDPA_VECTOR_MAX {
        return candle_nn::ops::sdpa(q, &k, &v, None, causal, scale, 1.0).map_err(err);
    }
    // pad queries, keys and values to a length the full kernel takes; padded keys are masked out
    // and padded queries' outputs dropped, so the result is exact
    let pad = SDPA_VECTOR_MAX + 1 - t;
    let qp = q.pad_with_zeros(2, 0, pad).map_err(err)?;
    let kp = k.pad_with_zeros(2, 0, pad).map_err(err)?;
    let vp = v.pad_with_zeros(2, 0, pad).map_err(err)?;
    let (tq, tk) = (t + pad, total + pad);
    let past = total - t;
    let mut m = vec![0f32; tq * tk];
    for i in 0..tq {
        for j in 0..tk {
            let hidden = j >= total || (causal && i < t && j > past + i);
            if hidden && let Some(x) = m.get_mut(i * tk + j) {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(m, (1, 1, tq, tk), q.device()).map_err(err)?.broadcast_as((b, nh, tq, tk)).map_err(err)?.contiguous().map_err(err)?;
    let o = candle_nn::ops::sdpa(&qp, &kp, &vp, Some(&mask), false, scale, 1.0).map_err(err)?;
    o.narrow(2, 0, t).map_err(err)
}

/// The same attention, written out with plain loops (tests: the reference for the batched one).
#[cfg(test)]
pub(crate) fn naive_attention(q: &[f32], k: &[f32], v: &[f32], t: usize, nh: usize, kvh: usize, hd: usize, causal: bool) -> Vec<f32> {
    // q [t, nh, hd], k/v [t, kvh, hd] → out [t, nh, hd]
    let g = nh / kvh;
    let mut out = vec![0f32; t * nh * hd];
    for h in 0..nh {
        let kh = h / g;
        for i in 0..t {
            let lim = if causal { i + 1 } else { t };
            let mut s: Vec<f64> = (0..lim)
                .map(|j| (0..hd).map(|d| f64::from(q[(i * nh + h) * hd + d]) * f64::from(k[(j * kvh + kh) * hd + d])).sum::<f64>() / (hd as f64).sqrt())
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s
                .iter_mut()
                .map(|x| {
                    *x = (*x - m).exp();
                    *x
                })
                .sum();
            for d in 0..hd {
                out[(i * nh + h) * hd + d] = (0..lim).map(|j| s[j] / z * f64::from(v[(j * kvh + kh) * hd + d])).sum::<f64>() as f32;
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::voxcpm::config;

    /// Random weights for a stack named `prefix` with `shape`, as safetensors entries.
    pub(crate) fn fake(prefix: &str, s: &Shape, rng: &mut crate::voxcpm::Noise, out: &mut Vec<(String, Tensor)>) {
        let dev = Device::Cpu;
        let mut t = |name: String, dims: &[usize], scale: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale).collect();
            out.push((name, Tensor::from_vec(v, dims, &dev).unwrap()));
        };
        let (h, nh, kvh, hd, f) = (s.hidden, s.heads, s.kv_heads, s.head_dim, s.ffn);
        let sc = 1.0 / (h as f32).sqrt();
        for i in 0..s.layers {
            let p = format!("{prefix}.layers.{i}");
            t(format!("{p}.input_layernorm.weight"), &[h], 0.1);
            t(format!("{p}.post_attention_layernorm.weight"), &[h], 0.1);
            t(format!("{p}.self_attn.q_proj.weight"), &[nh * hd, h], sc);
            t(format!("{p}.self_attn.k_proj.weight"), &[kvh * hd, h], sc);
            t(format!("{p}.self_attn.v_proj.weight"), &[kvh * hd, h], sc);
            t(format!("{p}.self_attn.o_proj.weight"), &[h, nh * hd], sc);
            t(format!("{p}.mlp.gate_proj.weight"), &[f, h], sc);
            t(format!("{p}.mlp.up_proj.weight"), &[f, h], sc);
            t(format!("{p}.mlp.down_proj.weight"), &[h, f], sc);
        }
        t(format!("{prefix}.norm.weight"), &[h], 0.1);
    }

    fn tiny_stack(causal_rope: bool, tag: &str) -> (Stack, config::Config) {
        let cfg = config::parse(config::tests::tiny(!causal_rope).as_bytes()).unwrap();
        let mut ts = Vec::new();
        let mut rng = crate::voxcpm::Noise::new(3);
        fake("m", &cfg.base, &mut rng, &mut ts);
        let d = std::env::temp_dir().join(format!("filmcraft-voxcpm-mc-{}-{causal_rope}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.safetensors");
        crate::voxcpm::weights::write(&p, &ts);
        let mut st = crate::voxcpm::weights::SafeTensors::open(&p).unwrap();
        let dev = Device::Cpu;
        let mut ld = Loader { st: &mut st, dev: &dev, wdtype: candle_core::DType::F32 };
        let s = Stack::load(&mut ld, "m", &cfg.base, &cfg.lm, 64).unwrap();
        let _ = std::fs::remove_dir_all(&d);
        (s, cfg)
    }

    #[test]
    fn grouped_attention_matches_the_loops() {
        let dev = Device::Cpu;
        let (t, nh, kvh, hd) = (5, 4, 2, 4);
        let mut rng = crate::voxcpm::Noise::new(11);
        let (q, k, v) = (rng.take(t * nh * hd), rng.take(t * kvh * hd), rng.take(t * kvh * hd));
        for causal in [false, true] {
            let want = naive_attention(&q, &k, &v, t, nh, kvh, hd, causal);
            // the batched path of `attend`, on the same projections
            let qt = Tensor::from_vec(q.clone(), (1, t, nh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let kt = Tensor::from_vec(k.clone(), (1, t, kvh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let vt = Tensor::from_vec(v.clone(), (1, t, kvh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let g = nh / kvh;
            let qg = qt.reshape((1, kvh, g * t, hd)).unwrap();
            let mut s = (qg.matmul(&kt.t().unwrap().contiguous().unwrap()).unwrap() * (1.0 / (hd as f64).sqrt())).unwrap();
            if causal {
                let mut m = vec![0f32; g * t * t];
                for gi in 0..g {
                    for i in 0..t {
                        for j in i + 1..t {
                            m[(gi * t + i) * t + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                s = s.broadcast_add(&Tensor::from_vec(m, (g * t, t), &dev).unwrap()).unwrap();
            }
            let p = candle_nn::ops::softmax_last_dim(&s).unwrap();
            let o = p.matmul(&vt).unwrap().reshape((1, nh, t, hd)).unwrap().transpose(1, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (a, b) in o.iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b} (causal {causal})");
            }
        }
    }

    /// The fused Metal kernel against the loops (runs only where there is a Metal device).
    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_matches_the_loops() {
        let Ok(dev) = Device::new_metal(0) else { return };
        for (t, nh, kvh, hd, causal) in [
            (1, 16, 2, 64, false),
            (2, 16, 2, 64, false),
            (5, 16, 2, 64, false),
            (5, 16, 2, 64, true),
            (8, 16, 2, 64, true),
            (9, 16, 2, 64, false),
            (5, 16, 2, 128, false),
            (17, 16, 2, 64, true),
            (40, 16, 2, 64, true),
            (11, 16, 2, 128, false),
            (300, 16, 2, 128, true),
        ] {
            let mut rng = crate::voxcpm::Noise::new(t as u64);
            let (q, k, v) = (rng.take(t * nh * hd), rng.take(t * kvh * hd), rng.take(t * kvh * hd));
            let want = naive_attention(&q, &k, &v, t, nh, kvh, hd, causal);
            let qt = Tensor::from_vec(q, (1, t, nh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let kt = Tensor::from_vec(k, (1, t, kvh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let vt = Tensor::from_vec(v, (1, t, kvh, hd), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let o = fused_attention(&qt, &kt, &vt, causal && t > 1, 1.0 / (hd as f32).sqrt()).unwrap();
            let o = o.transpose(1, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let d = o.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d < 1e-3, "t {t} heads {nh}/{kvh} dim {hd} causal {causal}: off by {d}");
        }
    }

    #[test]
    fn stepping_with_the_cache_matches_a_full_causal_pass() {
        let (s, _) = tiny_stack(true, "step");
        let dev = Device::Cpu;
        let t = 6;
        let x = Tensor::from_vec(crate::voxcpm::Noise::new(5).take(t * 16), (1, t, 16), &dev).unwrap();
        let full = s.forward(&x, true, None).unwrap();
        // prefill 4, then step 2
        let mut c = s.cache();
        let pre = s.forward(&x.narrow(1, 0, 4).unwrap(), true, Some(&mut c)).unwrap();
        assert_eq!(c.len(), 4);
        let a = s.step(&x.narrow(1, 4, 1).unwrap().squeeze(1).unwrap(), &mut c).unwrap();
        let b = s.step(&x.narrow(1, 5, 1).unwrap().squeeze(1).unwrap(), &mut c).unwrap();
        assert_eq!(c.len(), 6);
        let got = Tensor::cat(&[pre, a.unsqueeze(1).unwrap(), b.unsqueeze(1).unwrap()], 1).unwrap();
        let d = (got - &full).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-4, "{d}");
        // causal: the prefix does not see the future
        let pre_full = full.narrow(1, 0, 4).unwrap();
        let again = s.forward(&x.narrow(1, 0, 4).unwrap(), true, None).unwrap();
        assert!((pre_full - again).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
        // and the non-causal pass differs
        let nc = s.forward(&x, false, None).unwrap();
        assert!((nc - full).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() > 1e-3);
    }

    #[test]
    fn sequences_past_the_table_are_errors() {
        let (s, _) = tiny_stack(true, "table");
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 65, 16), candle_core::DType::F32, &dev).unwrap();
        assert!(s.forward(&x, true, None).is_err());
        let mut c = s.cache();
        let x = Tensor::zeros((1, 64, 16), candle_core::DType::F32, &dev).unwrap();
        s.forward(&x, true, Some(&mut c)).unwrap();
        assert!(s.step(&Tensor::zeros((1, 16), candle_core::DType::F32, &dev).unwrap(), &mut c).is_err());
    }

    #[test]
    fn rope_rotates_pairs_and_keeps_norms() {
        let cfg = config::parse(config::tests::tiny(false).as_bytes()).unwrap();
        let dev = Device::Cpu;
        let r = Rope::new(&cfg.lm, 4, 8, &dev).unwrap();
        let x = Tensor::from_vec(vec![1f32, 0., 0., 0., 0., 1., 0., 0.], (1, 1, 2, 4), &dev).unwrap();
        let y = r.apply(&x, 3).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // position 3, pair 0 (factor 1, inv freq 1): rotate (1, 0) by 3 rad
        assert!((y[0] - 3f32.cos()).abs() < 1e-6 && (y[2] - 3f32.sin()).abs() < 1e-6, "{y:?}");
        // position 4, pair 1 (factor 1.5, inv freq 1/100): (1, 0) by 4/1.5/100 rad
        let a = 4.0f32 / 1.5 / 100.0;
        assert!((y[5] - a.cos()).abs() < 1e-6 && (y[7] - a.sin()).abs() < 1e-6, "{y:?}");
        assert!(r.apply(&x, 7).is_err(), "past the table");
    }
}
