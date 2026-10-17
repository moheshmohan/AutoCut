//! Building blocks shared by the text encoder and the DiT: linear layers (bf16 or f32 weights),
//! RMSNorm, rotary embeddings, grouped-query attention with an optional additive mask, and the
//! pre-norm transformer block (Qwen3 style: per-head RMSNorm on queries and keys, SiLU-gated MLP).

use candle_core::{D, DType, Device, Tensor};

use super::config::Shape;
use super::weights::SafeTensors;
use super::{R, err};

/// Query positions attended at once (bounds the score matrix of long sequences).
const QUERY_CHUNK: usize = 1024;

/// Reads named weights with their expected shapes; linear weights in `wdtype`.
pub(crate) struct Loader<'a> {
    pub st: &'a mut SafeTensors,
    pub dev: &'a Device,
    /// Storage type of the linear layers' weights (activations stay f32).
    pub wdtype: DType,
}

impl Loader<'_> {
    pub(crate) fn tensor(&mut self, name: &str, shape: &[usize]) -> R<Tensor> {
        self.st.load(name, shape, self.dev)
    }

    pub(crate) fn linear(&mut self, name: &str, inp: usize, out: usize, bias: bool) -> R<Linear> {
        let w = self.st.load_as(&format!("{name}.weight"), &[out, inp], self.dev, self.wdtype)?;
        let b = if bias { Some(self.tensor(&format!("{name}.bias"), &[out])?) } else { None };
        Ok(Linear { w, b, out })
    }
}

/// A linear layer `y = x Wᵀ + b` over the last dimension.
pub(crate) struct Linear {
    w: Tensor,
    b: Option<Tensor>,
    out: usize,
}

impl Linear {
    /// From a weight `[out, in]` (and bias) already in memory.
    pub(crate) fn new(w: Tensor, b: Option<Tensor>) -> R<Linear> {
        let out = w.dim(0).map_err(err)?;
        Ok(Linear { w, b, out })
    }

    /// Several bias-free layers over the same input as one (outputs concatenated).
    pub(crate) fn stack(parts: &[Linear]) -> R<Linear> {
        if parts.iter().any(|p| p.b.is_some()) {
            return Err(err("only bias-free layers are stacked"));
        }
        let ws: Vec<&Tensor> = parts.iter().map(|p| &p.w).collect();
        Ok(Linear { w: Tensor::cat(&ws, 0).map_err(err)?, b: None, out: parts.iter().map(|p| p.out).sum() })
    }

    pub(crate) fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let dims = x.dims().to_vec();
        let (last, lead) = dims.split_last().ok_or_else(|| err("linear: scalar input"))?;
        let rows: usize = lead.iter().product();
        let mut x2 = x.reshape((rows, *last)).map_err(err)?;
        if x2.dtype() != DType::F32 {
            x2 = x2.to_dtype(DType::F32).map_err(err)?;
        }
        // bf16-stored weights are widened for the product (exactly: every bf16 is an f32), so
        // the arithmetic is the reference's f32 at half the resident memory
        let w = if self.w.dtype() == DType::F32 { self.w.clone() } else { self.w.to_dtype(DType::F32).map_err(err)? };
        let mut y = x2.matmul(&w.t().map_err(err)?).map_err(err)?;
        if let Some(b) = &self.b {
            y = y.broadcast_add(b).map_err(err)?;
        }
        let mut shape = lead.to_vec();
        shape.push(self.out);
        y.reshape(shape).map_err(err)
    }
}

pub(crate) fn rms(x: &Tensor, w: &Tensor, eps: f64) -> R<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous().map_err(err)?, w, eps as f32).map_err(err)
}

/// Rotary tables `[positions, head_dim / 2]`, computed in f32 as the reference does
/// (`inv_freq = 1 / θ^(2i/d)`, `freqs = position · inv_freq`).
pub(crate) struct Rope {
    cos: Tensor,
    sin: Tensor,
    len: usize,
}

impl Rope {
    pub(crate) fn new(theta: f64, head_dim: usize, len: usize, dev: &Device) -> R<Rope> {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32).collect();
        let n = len.checked_mul(half).ok_or_else(|| err("rotary table too large"))?;
        let mut cos = Vec::with_capacity(n);
        let mut sin = Vec::with_capacity(n);
        for t in 0..len {
            for f in &inv {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Rope { cos: Tensor::from_vec(cos, (len, half), dev).map_err(err)?, sin: Tensor::from_vec(sin, (len, half), dev).map_err(err)?, len })
    }

    /// Rotate `x` `[b, heads, t, head_dim]` for positions `0..t` (halves rotated, as `rotate_half`).
    pub(crate) fn apply(&self, x: &Tensor) -> R<Tensor> {
        let t = x.dim(2).map_err(err)?;
        if t > self.len {
            return Err(err("the sequence is longer than the rotary table"));
        }
        let c = self.cos.narrow(0, 0, t).map_err(err)?;
        let s = self.sin.narrow(0, 0, t).map_err(err)?;
        candle_nn::rotary_emb::rope(&x.contiguous().map_err(err)?, &c, &s).map_err(err)
    }
}

/// Which keys each query sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mask {
    /// Keys at or before the query.
    Causal,
    /// Keys within `w` positions either side (`|i − j| ≤ w`).
    Window(usize),
}

/// The additive mask for `t` queries over `t` keys (`None`: nothing hidden).
pub(crate) fn mask_tensor(m: Mask, t: usize, dev: &Device) -> R<Option<Tensor>> {
    let hide = |i: usize, j: usize| match m {
        Mask::Causal => j > i,
        Mask::Window(w) => i.abs_diff(j) > w,
    };
    let any = match m {
        Mask::Causal => t > 1,
        Mask::Window(w) => t > w.saturating_add(1),
    };
    if !any {
        return Ok(None);
    }
    let n = t.checked_mul(t).ok_or_else(|| err("attention mask too large"))?;
    let mut v = vec![0f32; n];
    for i in 0..t {
        for j in 0..t {
            if hide(i, j)
                && let Some(x) = v.get_mut(i * t + j)
            {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Some(Tensor::from_vec(v, (t, t), dev).map_err(err)?))
}

/// `softmax(q kᵀ · scale + mask) v`: `q` `[b, heads, tq, d]`, `k`/`v` `[b, kv_heads, tk, d]`,
/// `mask` `[tq, tk]`; returns `[b, tq, heads · d]`. Head `h` reads key/value head `h / g`.
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> R<Tensor> {
    let (b, nh, tq, d) = q.dims4().map_err(err)?;
    let kvh = k.dim(1).map_err(err)?;
    if kvh == 0 || nh % kvh != 0 {
        return Err(err("attention: heads do not group"));
    }
    let g = nh / kvh;
    let scale = 1.0 / (d as f64).sqrt();
    let kt = k.t().map_err(err)?.contiguous().map_err(err)?;
    let v = v.contiguous().map_err(err)?;
    let mut outs = Vec::new();
    let mut s = 0;
    while s < tq {
        let n = QUERY_CHUNK.min(tq - s);
        let qc = q.narrow(2, s, n).map_err(err)?.contiguous().map_err(err)?;
        // grouped queries: [b, kvh, g·n, d]
        let qg = qc.reshape((b, kvh, g * n, d)).map_err(err)?;
        let mut scores = (qg.matmul(&kt).map_err(err)? * scale).map_err(err)?;
        if let Some(m) = mask {
            let mc = m.narrow(0, s, n).map_err(err)?;
            let tk = scores.dim(3).map_err(err)?;
            scores = scores.reshape((b, kvh, g, n, tk)).map_err(err)?.broadcast_add(&mc).map_err(err)?.reshape((b, kvh, g * n, tk)).map_err(err)?;
        }
        let p = candle_nn::ops::softmax_last_dim(&scores).map_err(err)?;
        let o = p.matmul(&v).map_err(err)?; // [b, kvh, g·n, d]
        outs.push(o.reshape((b, nh, n, d)).map_err(err)?);
        s += n;
    }
    let o = if outs.len() == 1 { outs.pop().ok_or_else(|| err("attention: empty"))? } else { Tensor::cat(&outs, 2).map_err(err)? };
    o.transpose(1, 2).map_err(err)?.reshape((b, tq, nh * d)).map_err(err)
}

/// The SiLU-gated MLP `down(silu(gate x) · up x)`.
pub(crate) struct Mlp {
    gate_up: Linear,
    down: Linear,
    ffn: usize,
}

impl Mlp {
    pub(crate) fn load(ld: &mut Loader, p: &str, hidden: usize, ffn: usize) -> R<Mlp> {
        Ok(Mlp {
            gate_up: Linear::stack(&[ld.linear(&format!("{p}.gate_proj"), hidden, ffn, false)?, ld.linear(&format!("{p}.up_proj"), hidden, ffn, false)?])?,
            down: ld.linear(&format!("{p}.down_proj"), ffn, hidden, false)?,
            ffn,
        })
    }

    pub(crate) fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let gu = self.gate_up.fwd(x)?;
        let a = candle_nn::ops::silu(&gu.narrow(D::Minus1, 0, self.ffn).map_err(err)?).map_err(err)?;
        self.down.fwd(&(a * gu.narrow(D::Minus1, self.ffn, self.ffn).map_err(err)?).map_err(err)?)
    }
}

/// Self- or cross-attention with per-head RMSNorm on queries and keys (Qwen3 / ACE-Step).
pub(crate) struct Attention {
    /// Self-attention: q, k and v in one matrix; cross-attention: q alone.
    q_or_qkv: Linear,
    /// Cross-attention: k and v over the encoder states.
    kv: Option<Linear>,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    pub(crate) fn load(ld: &mut Loader, p: &str, s: &Shape, cross: bool) -> R<Attention> {
        let (h, nh, kvh, hd) = (s.hidden, s.heads, s.kv_heads, s.head_dim);
        let q = ld.linear(&format!("{p}.q_proj"), h, nh * hd, false)?;
        let k = ld.linear(&format!("{p}.k_proj"), h, kvh * hd, false)?;
        let v = ld.linear(&format!("{p}.v_proj"), h, kvh * hd, false)?;
        let (q_or_qkv, kv) = if cross { (q, Some(Linear::stack(&[k, v])?)) } else { (Linear::stack(&[q, k, v])?, None) };
        Ok(Attention {
            q_or_qkv,
            kv,
            o: ld.linear(&format!("{p}.o_proj"), nh * hd, h, false)?,
            q_norm: ld.tensor(&format!("{p}.q_norm.weight"), &[hd])?,
            k_norm: ld.tensor(&format!("{p}.k_norm.weight"), &[hd])?,
            heads: nh,
            kv_heads: kvh,
            head_dim: hd,
            eps: s.eps,
        })
    }

    /// `[b, t, n·d]` → `[b, n, t, d]`, normalised per head when `norm` is given.
    fn heads(&self, y: &Tensor, n: usize, norm: Option<&Tensor>) -> R<Tensor> {
        let (b, t, _) = y.dims3().map_err(err)?;
        let mut y = y.reshape((b, t, n, self.head_dim)).map_err(err)?;
        if let Some(w) = norm {
            y = rms(&y, w, self.eps)?;
        }
        y.transpose(1, 2).map_err(err)?.contiguous().map_err(err)
    }

    /// Self-attention of `x` `[b, t, hidden]` (rotary positions `0..t`).
    pub(crate) fn self_fwd(&self, x: &Tensor, rope: Option<&Rope>, mask: Option<&Tensor>) -> R<Tensor> {
        let (nh, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let qkv = self.q_or_qkv.fwd(x)?;
        let q = self.heads(&qkv.narrow(D::Minus1, 0, nh * hd).map_err(err)?, nh, Some(&self.q_norm))?;
        let k = self.heads(&qkv.narrow(D::Minus1, nh * hd, kvh * hd).map_err(err)?, kvh, Some(&self.k_norm))?;
        let v = self.heads(&qkv.narrow(D::Minus1, (nh + kvh) * hd, kvh * hd).map_err(err)?, kvh, None)?;
        let (q, k) = match rope {
            Some(r) => (r.apply(&q)?, r.apply(&k)?),
            None => (q, k),
        };
        self.o.fwd(&attention(&q, &k, &v, mask)?)
    }

    /// Keys and values of the encoder states `[b, s, hidden]` (computed once per generation).
    pub(crate) fn cross_kv(&self, enc: &Tensor) -> R<(Tensor, Tensor)> {
        let (kvh, hd) = (self.kv_heads, self.head_dim);
        let kvp = self.kv.as_ref().ok_or_else(|| err("not a cross-attention layer"))?;
        let kv = kvp.fwd(enc)?;
        let k = self.heads(&kv.narrow(D::Minus1, 0, kvh * hd).map_err(err)?, kvh, Some(&self.k_norm))?;
        let v = self.heads(&kv.narrow(D::Minus1, kvh * hd, kvh * hd).map_err(err)?, kvh, None)?;
        Ok((k, v))
    }

    /// Cross-attention of `x` over precomputed keys and values (no positions, no mask).
    pub(crate) fn cross_fwd(&self, x: &Tensor, k: &Tensor, v: &Tensor) -> R<Tensor> {
        let q = self.heads(&self.q_or_qkv.fwd(x)?, self.heads, Some(&self.q_norm))?;
        self.o.fwd(&attention(&q, k, v, None)?)
    }
}

/// A pre-norm transformer block: `x + attn(norm x)`, then `x + mlp(norm x)`.
pub(crate) struct Block {
    ln1: Tensor,
    ln2: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
    /// Sliding-window layer (else full attention).
    pub(crate) sliding: bool,
}

impl Block {
    pub(crate) fn load(ld: &mut Loader, p: &str, s: &Shape, sliding: bool) -> R<Block> {
        Ok(Block {
            ln1: ld.tensor(&format!("{p}.input_layernorm.weight"), &[s.hidden])?,
            ln2: ld.tensor(&format!("{p}.post_attention_layernorm.weight"), &[s.hidden])?,
            attn: Attention::load(ld, &format!("{p}.self_attn"), s, false)?,
            mlp: Mlp::load(ld, &format!("{p}.mlp"), s.hidden, s.ffn)?,
            eps: s.eps,
            sliding,
        })
    }

    pub(crate) fn fwd(&self, x: &Tensor, rope: &Rope, mask: Option<&Tensor>) -> R<Tensor> {
        let h = rms(x, &self.ln1, self.eps)?;
        let x = (x + self.attn.self_fwd(&h, Some(rope), mask)?).map_err(err)?;
        let h = rms(&x, &self.ln2, self.eps)?;
        (&x + self.mlp.fwd(&h)?).map_err(err)
    }
}

/// The same attention written out with plain loops (tests: the reference for the batched one).
/// `q` `[t, nh, d]`, `k`/`v` `[tk, kvh, d]` row-major; `hide(i, j)` masks key `j` for query `i`.
#[cfg(test)]
pub(crate) fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    tk: usize,
    nh: usize,
    kvh: usize,
    d: usize,
    hide: &dyn Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let g = nh / kvh;
    let mut out = vec![0f32; t * nh * d];
    for h in 0..nh {
        let kh = h / g;
        for i in 0..t {
            let keys: Vec<usize> = (0..tk).filter(|j| !hide(i, *j)).collect();
            let mut s: Vec<f64> = keys
                .iter()
                .map(|&j| (0..d).map(|x| f64::from(q[(i * nh + h) * d + x]) * f64::from(k[(j * kvh + kh) * d + x])).sum::<f64>() / (d as f64).sqrt())
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s
                .iter_mut()
                .map(|x| {
                    *x = (*x - m).exp();
                    *x
                })
                .sum();
            for x in 0..d {
                out[(i * nh + h) * d + x] = keys.iter().zip(&s).map(|(&j, p)| p / z * f64::from(v[(j * kvh + kh) * d + x])).sum::<f64>() as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut r = super::super::Noise::new(seed);
        r.take(n)
    }

    #[test]
    fn grouped_attention_matches_the_loops_with_every_mask() {
        let dev = Device::Cpu;
        let (t, nh, kvh, d) = (7, 4, 2, 8);
        let (q, k, v) = (rand(t * nh * d, 1), rand(t * kvh * d, 2), rand(t * kvh * d, 3));
        let to = |x: &[f32], h: usize| Tensor::from_vec(x.to_vec(), (1, t, h, d), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
        for m in [Mask::Causal, Mask::Window(2), Mask::Window(0), Mask::Window(100)] {
            let hide = move |i: usize, j: usize| match m {
                Mask::Causal => j > i,
                Mask::Window(w) => i.abs_diff(j) > w,
            };
            let want = naive_attention(&q, &k, &v, t, t, nh, kvh, d, &hide);
            let mt = mask_tensor(m, t, &dev).unwrap();
            let got = attention(&to(&q, nh), &to(&k, kvh), &to(&v, kvh), mt.as_ref()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(diff < 1e-5, "{m:?}: {diff}");
        }
    }

    #[test]
    fn long_sequences_are_attended_in_chunks_exactly() {
        let dev = Device::Cpu;
        let (t, nh, kvh, d) = (QUERY_CHUNK + 37, 2, 1, 4);
        let (q, k, v) = (rand(t * nh * d, 4), rand(t * kvh * d, 5), rand(t * kvh * d, 6));
        let to = |x: &[f32], h: usize| Tensor::from_vec(x.to_vec(), (1, t, h, d), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
        let w = 50;
        let want = naive_attention(&q, &k, &v, t, t, nh, kvh, d, &|i, j| i.abs_diff(j) > w);
        let mt = mask_tensor(Mask::Window(w), t, &dev).unwrap();
        let got = attention(&to(&q, nh), &to(&k, kvh), &to(&v, kvh), mt.as_ref()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(diff < 1e-5, "{diff}");
    }

    /// The batched attention on the GPU against the loops (runs where there is a Metal device).
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_attention_matches_the_loops() {
        let Ok(dev) = Device::new_metal(0) else { return };
        for (t, nh, kvh, d, m) in [(5, 16, 8, 128, Mask::Window(1000)), (300, 16, 8, 128, Mask::Window(128)), (9, 4, 2, 64, Mask::Causal)] {
            let (q, k, v) = (rand(t * nh * d, 7), rand(t * kvh * d, 8), rand(t * kvh * d, 9));
            let hide = move |i: usize, j: usize| match m {
                Mask::Causal => j > i,
                Mask::Window(w) => i.abs_diff(j) > w,
            };
            let want = naive_attention(&q, &k, &v, t, t, nh, kvh, d, &hide);
            let to = |x: &[f32], h: usize| Tensor::from_vec(x.to_vec(), (1, t, h, d), &dev).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let mt = mask_tensor(m, t, &dev).unwrap();
            let got = attention(&to(&q, nh), &to(&k, kvh), &to(&v, kvh), mt.as_ref()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(diff < 1e-3, "t {t} {m:?}: {diff}");
        }
    }

    #[test]
    fn masks_hide_the_right_keys() {
        let dev = Device::Cpu;
        assert!(mask_tensor(Mask::Window(8), 9, &dev).unwrap().is_none(), "a window wider than the sequence hides nothing");
        assert!(mask_tensor(Mask::Causal, 1, &dev).unwrap().is_none());
        let m = mask_tensor(Mask::Window(1), 4, &dev).unwrap().unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(m[0][1], 0.0);
        assert!(m[0][2].is_infinite() && m[3][1].is_infinite());
        assert_eq!(m[2][3], 0.0);
        let c = mask_tensor(Mask::Causal, 3, &dev).unwrap().unwrap().to_vec2::<f32>().unwrap();
        assert!(c[0][1].is_infinite() && c[2][1] == 0.0);
    }

    #[test]
    fn rope_rotates_halves_and_keeps_norms() {
        let dev = Device::Cpu;
        let r = Rope::new(10_000.0, 4, 8, &dev).unwrap();
        // x = (1, 0 | 0, 0): pair 0 is (x0, x2)
        let x = Tensor::from_vec(vec![1f32, 0., 0., 0., 1., 0., 0., 0.], (1, 1, 2, 4), &dev).unwrap();
        let y = r.apply(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // position 0 unchanged, position 1 rotated by 1 rad in pair 0
        assert_eq!(&y[..4], &[1., 0., 0., 0.]);
        assert!((y[4] - 1f32.cos()).abs() < 1e-6 && (y[6] - 1f32.sin()).abs() < 1e-6, "{y:?}");
        let long = Tensor::zeros((1, 1, 9, 4), DType::F32, &dev).unwrap();
        assert!(r.apply(&long).is_err());
    }

    #[test]
    fn linear_layers_stack_and_keep_leading_dims() {
        let dev = Device::Cpu;
        let a = Linear::new(Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &dev).unwrap(), None).unwrap();
        let b = Linear::new(Tensor::from_vec(vec![1f32, 1.], (1, 2), &dev).unwrap(), None).unwrap();
        let s = Linear::stack(&[a, b]).unwrap();
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &dev).unwrap();
        let y = s.fwd(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 3]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1., 2., 3., 3., 4., 7.]);
        let with_bias = Linear::new(Tensor::ones((1, 2), DType::F32, &dev).unwrap(), Some(Tensor::ones(1, DType::F32, &dev).unwrap())).unwrap();
        assert!(Linear::stack(&[with_bias]).is_err());
    }

    /// bf16-stored weights (the GPU's) are widened: f32 arithmetic, f32 output, on every device.
    #[test]
    fn bf16_weights_compute_in_f32() {
        let mut devs = vec![Device::Cpu];
        #[cfg(target_os = "macos")]
        if let Ok(d) = Device::new_metal(0) {
            devs.push(d);
        }
        for dev in devs {
            // 1 + 2⁻⁷ is exact in bf16; the input's low bits must survive (no bf16 activations)
            let w = Tensor::from_vec(vec![1.0078125f32, 0.25], (1, 2), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let l = Linear::new(w, None).unwrap();
            let y = l.fwd(&Tensor::from_vec(vec![1.0001f32, 4.], (1, 2), &dev).unwrap()).unwrap();
            assert_eq!(y.dtype(), DType::F32);
            let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            assert!((got - (1.0001f32 * 1.0078125 + 1.0)).abs() < 1e-6, "{got}");
        }
    }
}
