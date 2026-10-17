//! The text encoder: Qwen3-Embedding-0.6B (Qwen team, Apache-2.0) as the reference runs it
//! (`AutoModel` → `Qwen3Model`, `last_hidden_state`): token embeddings, 28 causal pre-norm
//! decoder layers (grouped-query attention with per-head RMSNorm on queries and keys, rotary
//! positions, SiLU MLP) and a final RMSNorm. The lyric encoder reads only the token embeddings.
//!
//! The 151 669 × 1024 embedding table is never loaded: the rows of the prompt's tokens are read
//! from the file on their own.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::config::TextConfig;
use super::nn::{Block, Loader, Mask, Rope, mask_tensor, rms};
use super::weights::SafeTensors;
use super::{R, err};

const EMBED: &str = "embed_tokens.weight";

pub(crate) struct TextEncoder {
    cfg: TextConfig,
    st: SafeTensors,
    layers: Vec<Block>,
    norm: Tensor,
    dev: Device,
}

impl TextEncoder {
    pub(crate) fn load(path: &Path, cfg: &TextConfig, dev: &Device, wdtype: DType) -> R<TextEncoder> {
        let mut st = SafeTensors::open(path)?;
        if st.shape(EMBED) != Some(&[cfg.vocab, cfg.shape.hidden][..]) {
            return Err(err(format!("{}: the embedding table does not match the config", path.display())));
        }
        let (layers, norm) = {
            let mut ld = Loader { st: &mut st, dev, wdtype };
            let mut layers = Vec::with_capacity(cfg.layers);
            for i in 0..cfg.layers {
                layers.push(Block::load(&mut ld, &format!("layers.{i}"), &cfg.shape, false)?);
            }
            let norm = ld.tensor("norm.weight", &[cfg.shape.hidden])?;
            (layers, norm)
        };
        Ok(TextEncoder { cfg: cfg.clone(), st, layers, norm, dev: dev.clone() })
    }

    /// Token embeddings `[1, ids, hidden]` (the lyric encoder's input).
    pub(crate) fn embed(&mut self, ids: &[u32]) -> R<Tensor> {
        if ids.is_empty() {
            return Err(err("nothing to encode"));
        }
        self.st.rows(EMBED, ids, &self.dev)?.unsqueeze(0).map_err(err)
    }

    /// The last hidden states `[1, ids, hidden]` of a causal pass over `ids`.
    pub(crate) fn encode(&mut self, ids: &[u32]) -> R<Tensor> {
        let mut x = self.embed(ids)?;
        let t = ids.len();
        let rope = Rope::new(self.cfg.shape.rope_theta, self.cfg.shape.head_dim, t, &self.dev)?;
        let mask = mask_tensor(Mask::Causal, t, &self.dev)?;
        for l in &self.layers {
            x = l.fwd(&x, &rope, mask.as_ref())?;
        }
        rms(&x, &self.norm, self.cfg.shape.eps)
    }

    /// The first layer's output (oracle tests).
    #[cfg(test)]
    pub(crate) fn first_layer(&mut self, ids: &[u32]) -> R<Tensor> {
        let x = self.embed(ids)?;
        let rope = Rope::new(self.cfg.shape.rope_theta, self.cfg.shape.head_dim, ids.len(), &self.dev)?;
        let mask = mask_tensor(Mask::Causal, ids.len(), &self.dev)?;
        self.layers.first().ok_or_else(|| err("no layers"))?.fwd(&x, &rope, mask.as_ref())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::config::{parse_text, tests::tiny_text};

    /// Random weights for a Qwen3 stack named `prefix` (`""` for the top level).
    pub(crate) fn fake_block(prefix: &str, s: &crate::model::config::Shape, rng: &mut crate::model::Noise, out: &mut Vec<(String, Tensor)>) {
        let dev = Device::Cpu;
        let mut t = |name: String, dims: &[usize], scale: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale).collect();
            out.push((name, Tensor::from_vec(v, dims, &dev).unwrap()));
        };
        let (h, nh, kvh, hd, f) = (s.hidden, s.heads, s.kv_heads, s.head_dim, s.ffn);
        let sc = 1.0 / (h as f32).sqrt();
        t(format!("{prefix}input_layernorm.weight"), &[h], 0.3);
        t(format!("{prefix}post_attention_layernorm.weight"), &[h], 0.3);
        t(format!("{prefix}self_attn.q_proj.weight"), &[nh * hd, h], sc);
        t(format!("{prefix}self_attn.k_proj.weight"), &[kvh * hd, h], sc);
        t(format!("{prefix}self_attn.v_proj.weight"), &[kvh * hd, h], sc);
        t(format!("{prefix}self_attn.o_proj.weight"), &[h, nh * hd], sc);
        t(format!("{prefix}self_attn.q_norm.weight"), &[hd], 0.3);
        t(format!("{prefix}self_attn.k_norm.weight"), &[hd], 0.3);
        t(format!("{prefix}mlp.gate_proj.weight"), &[f, h], sc);
        t(format!("{prefix}mlp.up_proj.weight"), &[f, h], sc);
        t(format!("{prefix}mlp.down_proj.weight"), &[h, f], sc);
    }

    /// A tiny random text encoder written to `dir/text.safetensors`.
    pub(crate) fn fake(dir: &Path) -> (std::path::PathBuf, TextConfig) {
        let cfg = parse_text(&tiny_text()).unwrap();
        let mut rng = crate::model::Noise::new(21);
        let mut ts = Vec::new();
        for i in 0..cfg.layers {
            fake_block(&format!("layers.{i}."), &cfg.shape, &mut rng, &mut ts);
        }
        let h = cfg.shape.hidden;
        let v: Vec<f32> = rng.take(cfg.vocab * h);
        ts.push((EMBED.into(), Tensor::from_vec(v, (cfg.vocab, h), &Device::Cpu).unwrap()));
        ts.push(("norm.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap()));
        let p = dir.join("text.safetensors");
        crate::model::weights::write(&p, &ts);
        (p, cfg)
    }

    #[test]
    fn a_tiny_encoder_is_causal_and_deterministic() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-qwen-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (p, cfg) = fake(&d);
        let mut e = TextEncoder::load(&p, &cfg, &Device::Cpu, DType::F32).unwrap();
        let a = e.encode(&[1, 2, 3, 4]).unwrap();
        assert_eq!(a.dims(), &[1, 4, 16]);
        let b = e.encode(&[1, 2, 3, 4]).unwrap();
        assert_eq!(a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        // causal: a prefix encodes the same whatever follows
        let p2 = e.encode(&[1, 2]).unwrap();
        let d2 = (p2 - a.narrow(1, 0, 2).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d2 < 1e-5, "{d2}");
        // out-of-vocabulary ids are errors
        assert!(e.encode(&[cfg.vocab as u32]).is_err());
        assert!(e.encode(&[]).is_err());
        assert_eq!(e.embed(&[5, 5]).unwrap().dims(), &[1, 2, 16]);
        // a file whose table does not match the config is refused
        let mut other = cfg.clone();
        other.vocab += 1;
        assert!(TextEncoder::load(&p, &other, &Device::Cpu, DType::F32).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
