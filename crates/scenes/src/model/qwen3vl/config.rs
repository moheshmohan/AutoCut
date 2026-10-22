//! Qwen3-VL's `config.json`, read and checked: every size bounded and consistent before a
//! tensor is allocated.

use serde::Deserialize;

use super::super::{R, err};

#[derive(Deserialize)]
struct RawRope {
    #[serde(default)]
    mrope_section: Vec<usize>,
}

#[derive(Deserialize)]
struct RawText {
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    vocab_size: usize,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    rope_scaling: Option<RawRope>,
    #[serde(default)]
    hidden_act: String,
}

#[derive(Deserialize)]
struct RawVision {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    num_position_embeddings: usize,
    #[serde(default)]
    deepstack_visual_indexes: Vec<usize>,
    #[serde(default)]
    in_channels: Option<usize>,
    #[serde(default)]
    hidden_act: String,
}

#[derive(Deserialize)]
struct Raw {
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    text_config: RawText,
    vision_config: RawVision,
}

/// The text decoder's sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f64,
    pub rope_theta: f64,
    /// Frequencies given to time, height and width (interleaved M-RoPE), summing to `head_dim / 2`.
    pub mrope: [usize; 3],
    pub tied: bool,
}

/// The vision encoder's sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub out_hidden: usize,
    pub patch: usize,
    pub merge: usize,
    pub temporal: usize,
    /// Side of the learned position-embedding grid (`num_position_embeddings` is its square).
    pub grid: usize,
    pub deepstack: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub image_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
}

fn bounded(name: &str, v: usize, lo: usize, hi: usize) -> R<usize> {
    if (lo..=hi).contains(&v) { Ok(v) } else { Err(err(format!("Qwen3-VL config: {name} = {v} is outside {lo}..={hi}"))) }
}

impl Config {
    pub fn parse(json: &str) -> R<Config> {
        let r: Raw = serde_json::from_str(json).map_err(|e| err(format!("Qwen3-VL config: {e}")))?;
        let t = &r.text_config;
        let v = &r.vision_config;
        let hidden = bounded("hidden_size", t.hidden_size, 64, 16_384)?;
        let heads = bounded("num_attention_heads", t.num_attention_heads, 1, 256)?;
        let kv_heads = bounded("num_key_value_heads", t.num_key_value_heads, 1, heads)?;
        let head_dim = bounded("head_dim", t.head_dim, 8, 512)?;
        if heads % kv_heads != 0 || head_dim % 2 != 0 {
            return Err(err("Qwen3-VL config: heads do not group evenly"));
        }
        let mrope = t.rope_scaling.as_ref().map(|r| r.mrope_section.clone()).unwrap_or_default();
        let [mt, mh, mw] = mrope.as_slice() else { return Err(err("Qwen3-VL config: mrope_section must have three parts")) };
        if mt + mh + mw != head_dim / 2 || mh.max(mw) * 3 > head_dim / 2 {
            return Err(err("Qwen3-VL config: mrope_section does not fit head_dim"));
        }
        if !(t.rms_norm_eps.is_finite() && t.rms_norm_eps > 0.0 && t.rms_norm_eps < 1.0) || !(t.rope_theta.is_finite() && t.rope_theta > 1.0) {
            return Err(err("Qwen3-VL config: bad epsilon or rope theta"));
        }
        if !t.hidden_act.is_empty() && t.hidden_act != "silu" {
            return Err(err(format!("Qwen3-VL config: text activation {} is not supported", t.hidden_act)));
        }
        let text = TextConfig {
            hidden,
            intermediate: bounded("intermediate_size", t.intermediate_size, 64, 65_536)?,
            heads,
            kv_heads,
            head_dim,
            layers: bounded("num_hidden_layers", t.num_hidden_layers, 1, 128)?,
            vocab: bounded("vocab_size", t.vocab_size, 256, 400_000)?,
            eps: t.rms_norm_eps,
            rope_theta: t.rope_theta,
            mrope: [*mt, *mh, *mw],
            tied: t.tie_word_embeddings.or(r.tie_word_embeddings).unwrap_or(false),
        };
        let vh = bounded("vision hidden_size", v.hidden_size, 64, 8_192)?;
        let vheads = bounded("vision num_heads", v.num_heads, 1, 128)?;
        if vh % vheads != 0 || (vh / vheads) % 4 != 0 {
            return Err(err("Qwen3-VL config: vision heads do not divide the width"));
        }
        let pos = bounded("num_position_embeddings", v.num_position_embeddings, 1, 1 << 20)?;
        let grid = (pos as f64).sqrt().round() as usize;
        if grid * grid != pos {
            return Err(err("Qwen3-VL config: num_position_embeddings is not a square"));
        }
        if v.in_channels.is_some_and(|c| c != 3) || (!v.hidden_act.is_empty() && v.hidden_act != "gelu_pytorch_tanh") {
            return Err(err("Qwen3-VL config: an unsupported vision encoder"));
        }
        let depth = bounded("vision depth", v.depth, 1, 128)?;
        if v.deepstack_visual_indexes.iter().any(|i| *i >= depth) || v.deepstack_visual_indexes.len() > text.layers {
            return Err(err("Qwen3-VL config: deepstack layers out of range"));
        }
        let vision = VisionConfig {
            depth,
            hidden: vh,
            intermediate: bounded("vision intermediate_size", v.intermediate_size, 64, 65_536)?,
            heads: vheads,
            out_hidden: bounded("out_hidden_size", v.out_hidden_size, 64, 16_384)?,
            patch: bounded("patch_size", v.patch_size, 2, 64)?,
            merge: bounded("spatial_merge_size", v.spatial_merge_size, 1, 4)?,
            temporal: bounded("temporal_patch_size", v.temporal_patch_size, 1, 4)?,
            grid,
            deepstack: v.deepstack_visual_indexes.clone(),
        };
        if vision.out_hidden != text.hidden {
            return Err(err("Qwen3-VL config: the vision encoder's output does not match the text width"));
        }
        Ok(Config { text, vision, image_token: r.image_token_id, vision_start: r.vision_start_token_id, vision_end: r.vision_end_token_id })
    }
}

#[cfg(test)]
pub(crate) const TEST_CONFIG: &str = r#"{
  "image_token_id": 151655, "vision_start_token_id": 151652, "vision_end_token_id": 151653, "tie_word_embeddings": true,
  "text_config": {"head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144, "num_attention_heads": 16, "num_hidden_layers": 28,
    "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_theta": 5000000, "vocab_size": 151936, "tie_word_embeddings": true,
    "hidden_act": "silu", "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"}},
  "vision_config": {"deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024,
    "in_channels": 3, "intermediate_size": 4096, "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2048,
    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_2b_config() {
        let c = Config::parse(TEST_CONFIG).unwrap();
        assert_eq!(c.text.mrope, [24, 20, 20]);
        assert_eq!(c.vision.grid, 48);
        assert!(c.text.tied);
        assert_eq!(c.vision.deepstack, vec![5, 11, 17]);
    }

    #[test]
    fn hostile_configs_are_refused() {
        let base: serde_json::Value = serde_json::from_str(TEST_CONFIG).unwrap();
        for (path, v) in [
            (vec!["text_config", "hidden_size"], serde_json::json!(1u64 << 40)),
            (vec!["text_config", "num_key_value_heads"], serde_json::json!(7)),
            (vec!["text_config", "head_dim"], serde_json::json!(127)),
            (vec!["text_config", "rope_scaling", "mrope_section"], serde_json::json!([24, 20])),
            (vec!["text_config", "rms_norm_eps"], serde_json::json!(-1.0)),
            (vec!["vision_config", "num_position_embeddings"], serde_json::json!(2305)),
            (vec!["vision_config", "deepstack_visual_indexes"], serde_json::json!([99])),
            (vec!["vision_config", "out_hidden_size"], serde_json::json!(1024)),
            (vec!["vision_config", "hidden_act"], serde_json::json!("relu")),
        ] {
            let mut doc = base.clone();
            let mut at = &mut doc;
            for k in &path[..path.len() - 1] {
                at = at.get_mut(*k).unwrap();
            }
            at[path[path.len() - 1]] = v;
            assert!(Config::parse(&doc.to_string()).is_err(), "{path:?}");
        }
        assert!(Config::parse("{}").is_err());
        assert!(Config::parse("nope").is_err());
    }
}
