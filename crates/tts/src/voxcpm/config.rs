//! `config.json` of a VoxCPM release, read and checked before any weight is touched. Every
//! number in it is hostile: sizes are capped, divisibility is checked, and an architecture this
//! code does not implement is refused with a message instead of being run.

use std::path::Path;

use serde::Deserialize;

use super::{R, err};

/// Which VoxCPM generation a checkpoint is (`architecture` in `config.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Variant {
    /// `voxcpm`: VoxCPM-0.5B and VoxCPM1.5 — summed DiT conditioning, AudioVAE V1 (one rate in
    /// and out), continuation prompts only.
    V1,
    /// `voxcpm2`: VoxCPM2 — concatenated DiT conditioning, a fusion layer into the residual LM,
    /// AudioVAE V2 (16 kHz in, 48 kHz out), isolated reference prompts as well.
    V2,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct RopeScaling {
    #[serde(rename = "type")]
    pub kind: String,
    pub long_factor: Vec<f64>,
    pub short_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub vocab_size: usize,
    #[serde(default = "yes")]
    pub use_mup: bool,
    pub scale_emb: f64,
    pub scale_depth: f64,
    #[serde(default)]
    pub kv_channels: Option<usize>,
    #[serde(default)]
    pub no_rope: bool,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct BlockConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub kv_channels: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct VaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    #[serde(default = "yes")]
    pub depthwise: bool,
    pub sample_rate: u32,
    #[serde(default)]
    pub use_noise_block: bool,
    /// V2: the decoder's output rate (it upsamples past the encoder's).
    #[serde(default)]
    pub out_sample_rate: Option<u32>,
    /// V2: the decoder is conditioned on the output rate, bucketed by these boundaries.
    #[serde(default)]
    pub sr_bin_boundaries: Option<Vec<u32>>,
    #[serde(default)]
    pub cond_type: Option<String>,
    #[serde(default)]
    pub cond_out_layer: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct Raw {
    #[serde(default)]
    architecture: Option<String>,
    lm_config: LmConfig,
    patch_size: usize,
    feat_dim: usize,
    residual_lm_num_layers: usize,
    #[serde(default)]
    residual_lm_no_rope: bool,
    scalar_quantization_latent_dim: usize,
    scalar_quantization_scale: f64,
    encoder_config: BlockConfig,
    dit_config: BlockConfig,
    audio_vae_config: VaeConfig,
    #[serde(default = "max_length")]
    max_length: usize,
}

fn max_length() -> usize {
    4096
}

/// A transformer stack's shape (the text-semantic LM, the residual LM, LocEnc and LocDiT are all
/// MiniCPM-4 stacks that differ only in these numbers).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Stack {
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub rope: bool,
    pub eps: f64,
    /// `scale_depth / sqrt(layers)` on every residual branch (µP), else 1.
    pub residual_scale: f64,
}

/// A checked configuration.
#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub variant: Variant,
    pub lm: LmConfig,
    pub base: Stack,
    pub residual: Stack,
    pub encoder: Stack,
    pub dit: Stack,
    pub patch: usize,
    pub feat: usize,
    pub fsq_dim: usize,
    pub fsq_scale: f64,
    pub vocab: usize,
    /// Text embeddings are multiplied by this (µP), else 1.
    pub emb_scale: f64,
    pub max_length: usize,
    pub vae: VaeConfig,
}

/// Generous bounds on every size: far above the released models, far below anything that could
/// exhaust memory through a doctored config.
const MAX_HIDDEN: usize = 8192;
const MAX_FFN: usize = 32_768;
const MAX_LAYERS: usize = 96;
const MAX_HEADS: usize = 128;
const MAX_VOCAB: usize = 1 << 20;
const MAX_PATCH: usize = 32;
const MAX_FEAT: usize = 512;
const MAX_SEQUENCE: usize = 32_768;
const MAX_VAE_DIM: usize = 4096;
const MAX_RATES: usize = 8;
const MAX_RATE: usize = 16;

pub(crate) fn read(path: &Path) -> R<Config> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    parse(&bytes)
}

pub(crate) fn parse(bytes: &[u8]) -> R<Config> {
    if bytes.len() > 1 << 20 {
        return Err(err("config.json is too large"));
    }
    let raw: Raw = serde_json::from_slice(bytes).map_err(|e| err(format!("config.json: {e}")))?;
    let variant = match raw.architecture.as_deref().unwrap_or("voxcpm").to_ascii_lowercase().as_str() {
        "voxcpm" => Variant::V1,
        "voxcpm2" => Variant::V2,
        other => return Err(err(format!("config.json: architecture `{other}` is not supported (voxcpm, voxcpm2)"))),
    };
    let lm = raw.lm_config;
    let stack = |hidden: usize, ffn: usize, heads: usize, layers: usize, kv: Option<usize>, rope: bool, what: &str| -> R<Stack> {
        if !(1..=MAX_HIDDEN).contains(&hidden) || !(1..=MAX_FFN).contains(&ffn) || !(1..=MAX_HEADS).contains(&heads) || !(1..=MAX_LAYERS).contains(&layers) {
            return Err(err(format!("config.json: the {what} sizes are out of range")));
        }
        let head_dim = match kv {
            Some(d) => d,
            None if hidden.is_multiple_of(heads) => hidden / heads,
            None => return Err(err(format!("config.json: the {what} hidden size is not a multiple of its heads"))),
        };
        if head_dim == 0 || head_dim % 2 != 0 || head_dim > 1024 {
            return Err(err(format!("config.json: the {what} head size {head_dim} is not supported")));
        }
        let kv_heads = lm.num_key_value_heads;
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(err(format!("config.json: the {what} heads are not a multiple of the key/value heads")));
        }
        if !(lm.rms_norm_eps.is_finite() && lm.rms_norm_eps > 0.0 && lm.rms_norm_eps < 1.0) {
            return Err(err("config.json: rms_norm_eps is out of range"));
        }
        let residual_scale = if lm.use_mup {
            let s = lm.scale_depth / (layers as f64).sqrt();
            if !s.is_finite() || s.abs() > 100.0 {
                return Err(err("config.json: scale_depth is out of range"));
            }
            s
        } else {
            1.0
        };
        Ok(Stack { hidden, ffn, heads, kv_heads, head_dim, layers, rope, eps: lm.rms_norm_eps, residual_scale })
    };
    let rope = !lm.no_rope;
    let base = stack(lm.hidden_size, lm.intermediate_size, lm.num_attention_heads, lm.num_hidden_layers, lm.kv_channels, rope, "language model")?;
    let residual_rope = rope && !raw.residual_lm_no_rope;
    let residual =
        stack(lm.hidden_size, lm.intermediate_size, lm.num_attention_heads, raw.residual_lm_num_layers, lm.kv_channels, residual_rope, "residual model")?;
    let e = &raw.encoder_config;
    let encoder = stack(e.hidden_dim, e.ffn_dim, e.num_heads, e.num_layers, e.kv_channels, rope, "local encoder")?;
    let d = &raw.dit_config;
    let dit = stack(d.hidden_dim, d.ffn_dim, d.num_heads, d.num_layers, d.kv_channels, rope, "local DiT")?;
    if rope {
        let rs = &lm.rope_scaling;
        if rs.kind != "longrope" {
            return Err(err(format!("config.json: rope scaling `{}` is not supported (longrope)", rs.kind)));
        }
        if !(lm.rope_theta.is_finite() && lm.rope_theta > 1.0) {
            return Err(err("config.json: rope_theta is out of range"));
        }
        for s in [&base, &encoder, &dit] {
            if rs.short_factor.len() != s.head_dim / 2 || rs.long_factor.len() != s.head_dim / 2 {
                return Err(err("config.json: the rope factors do not match the head size"));
            }
        }
        if rs.short_factor.iter().chain(&rs.long_factor).any(|f| !(f.is_finite() && *f > 0.0)) || rs.original_max_position_embeddings == 0 {
            return Err(err("config.json: the rope factors are out of range"));
        }
    }
    if !(1..=MAX_VOCAB).contains(&lm.vocab_size) {
        return Err(err("config.json: vocab_size is out of range"));
    }
    if !(1..=MAX_PATCH).contains(&raw.patch_size) || !(1..=MAX_FEAT).contains(&raw.feat_dim) {
        return Err(err("config.json: patch_size or feat_dim is out of range"));
    }
    if !(1..=MAX_HIDDEN).contains(&raw.scalar_quantization_latent_dim) || !(raw.scalar_quantization_scale.is_finite() && raw.scalar_quantization_scale >= 1.0) {
        return Err(err("config.json: the scalar quantizer is out of range"));
    }
    let emb_scale = if lm.use_mup { lm.scale_emb } else { 1.0 };
    if !(emb_scale.is_finite() && emb_scale.abs() <= 1000.0) {
        return Err(err("config.json: scale_emb is out of range"));
    }
    if !(16..=MAX_SEQUENCE).contains(&raw.max_length) {
        return Err(err("config.json: max_length is out of range"));
    }
    let vae = raw.audio_vae_config;
    check_vae(&vae, variant, raw.feat_dim)?;
    Ok(Config {
        variant,
        lm: lm.clone(),
        base,
        residual,
        encoder,
        dit,
        patch: raw.patch_size,
        feat: raw.feat_dim,
        fsq_dim: raw.scalar_quantization_latent_dim,
        fsq_scale: raw.scalar_quantization_scale,
        vocab: lm.vocab_size,
        emb_scale,
        max_length: raw.max_length,
        vae,
    })
}

fn check_vae(v: &VaeConfig, variant: Variant, feat: usize) -> R<()> {
    let rates_ok = |r: &[usize]| !r.is_empty() && r.len() <= MAX_RATES && r.iter().all(|x| (1..=MAX_RATE).contains(x));
    if !rates_ok(&v.encoder_rates) || !rates_ok(&v.decoder_rates) {
        return Err(err("config.json: the audio VAE rates are out of range"));
    }
    if v.latent_dim != feat {
        return Err(err("config.json: the audio VAE latent size differs from feat_dim"));
    }
    if !(1..=MAX_VAE_DIM).contains(&v.encoder_dim) || !(1..=MAX_VAE_DIM).contains(&v.decoder_dim) {
        return Err(err("config.json: the audio VAE sizes are out of range"));
    }
    // channels double per encoder block and halve per decoder block
    let enc_top = v.encoder_dim.checked_shl(v.encoder_rates.len() as u32).filter(|c| *c <= MAX_VAE_DIM * 16);
    if enc_top.is_none() || v.decoder_dim >> v.decoder_rates.len() == 0 || !v.decoder_dim.is_multiple_of(1 << v.decoder_rates.len()) {
        return Err(err("config.json: the audio VAE channel plan is not supported"));
    }
    if !v.depthwise || v.use_noise_block {
        return Err(err("config.json: only the depthwise audio VAE without noise blocks is supported"));
    }
    if !(8_000..=192_000).contains(&v.sample_rate) {
        return Err(err("config.json: the audio VAE sample rate is out of range"));
    }
    let enc_hop: usize = v.encoder_rates.iter().product();
    let dec_hop: usize = v.decoder_rates.iter().product();
    match variant {
        Variant::V1 => {
            if enc_hop != dec_hop || v.sr_bin_boundaries.is_some() {
                return Err(err("config.json: a VoxCPM 1 audio VAE decodes at its encoding rate"));
            }
        }
        Variant::V2 => {
            let out = v.out_sample_rate.unwrap_or(48_000);
            if !(8_000..=192_000).contains(&out) || (out as usize).checked_mul(enc_hop) != (v.sample_rate as usize).checked_mul(dec_hop) {
                return Err(err("config.json: the audio VAE output rate does not match its rates"));
            }
            if v.cond_type.as_deref().unwrap_or("scale_bias") != "scale_bias" || v.cond_out_layer {
                return Err(err("config.json: only scale-and-bias rate conditioning is supported"));
            }
            if v.sr_bin_boundaries.as_ref().is_some_and(|b| b.len() > 16) {
                return Err(err("config.json: too many sample-rate buckets"));
            }
        }
    }
    Ok(())
}

impl Config {
    /// Rate the audio VAE encodes at (prompt audio is resampled to it).
    pub fn encode_rate(&self) -> u32 {
        self.vae.sample_rate
    }

    /// Rate the generated audio comes out at.
    pub fn output_rate(&self) -> u32 {
        match self.variant {
            Variant::V1 => self.vae.sample_rate,
            Variant::V2 => self.vae.out_sample_rate.unwrap_or(48_000),
        }
    }

    /// Input samples per latent frame.
    pub fn encode_hop(&self) -> usize {
        self.vae.encoder_rates.iter().product()
    }

    /// Output samples per latent frame.
    pub fn decode_hop(&self) -> usize {
        self.vae.decoder_rates.iter().product()
    }

    /// The V2 decoder's rate bucket for the output rate (`torch.bucketize`, right = false).
    pub fn rate_bucket(&self) -> Option<usize> {
        if self.variant != Variant::V2 {
            return None;
        }
        let b = self.vae.sr_bin_boundaries.clone().unwrap_or_else(|| vec![20_000, 30_000, 40_000]);
        let out = self.output_rate();
        Some(b.iter().filter(|x| **x < out).count())
    }

    /// Number of rate buckets of the V2 decoder.
    pub fn rate_buckets(&self) -> usize {
        self.vae.sr_bin_boundaries.as_ref().map_or(4, |b| b.len() + 1)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny V1 or V2 configuration (also what the fake models in the tests are built from).
    pub(crate) fn tiny(v2: bool) -> String {
        let arch = if v2 { "voxcpm2" } else { "voxcpm" };
        let vae = if v2 {
            r#""audio_vae_config":{"encoder_dim":4,"encoder_rates":[8,8],"latent_dim":8,"decoder_dim":16,"decoder_rates":[8,12],"sample_rate":16000,"out_sample_rate":24000,"sr_bin_boundaries":[20000,30000,40000]}"#
        } else {
            r#""audio_vae_config":{"encoder_dim":4,"encoder_rates":[8,8],"latent_dim":8,"decoder_dim":16,"decoder_rates":[8,8],"sample_rate":16000}"#
        };
        format!(
            r#"{{"architecture":"{arch}","lm_config":{{"bos_token_id":1,"eos_token_id":2,"hidden_size":16,"intermediate_size":32,"max_position_embeddings":64,"num_attention_heads":4,"num_hidden_layers":2,"num_key_value_heads":2,"rms_norm_eps":1e-5,"rope_theta":10000,"rope_scaling":{{"type":"longrope","long_factor":[1.0,1.5],"short_factor":[1.0,1.5],"original_max_position_embeddings":64}},"vocab_size":300,"scale_emb":12,"dim_model_base":256,"scale_depth":1.4,"use_mup":false}},"patch_size":2,"feat_dim":8,"scalar_quantization_latent_dim":8,"scalar_quantization_scale":9,"residual_lm_num_layers":1,"residual_lm_no_rope":{v2},"encoder_config":{{"hidden_dim":16,"ffn_dim":32,"num_heads":4,"num_layers":1}},"dit_config":{{"hidden_dim":16,"ffn_dim":32,"num_heads":4,"num_layers":1,"cfm_config":{{"inference_cfg_rate":2.0}}}},{vae},"max_length":1024}}"#
        )
    }

    #[test]
    fn released_shapes_parse() {
        let c = parse(tiny(false).as_bytes()).unwrap();
        assert_eq!(c.variant, Variant::V1);
        assert_eq!(c.base.head_dim, 4);
        assert_eq!(c.residual.layers, 1);
        assert!(c.residual.rope);
        assert_eq!(c.encode_hop(), 64);
        assert_eq!(c.output_rate(), 16_000);
        assert_eq!(c.rate_bucket(), None);
        let c = parse(tiny(true).as_bytes()).unwrap();
        assert_eq!(c.variant, Variant::V2);
        assert!(!c.residual.rope, "VoxCPM2's residual LM has no rotary embedding");
        assert_eq!(c.output_rate(), 24_000);
        assert_eq!(c.decode_hop(), 96);
        assert_eq!(c.rate_bucket(), Some(1));
        assert_eq!(c.rate_buckets(), 4);
    }

    #[test]
    fn hostile_configs_are_refused() {
        let good = tiny(false);
        for (from, to) in [
            ("\"voxcpm\"", "\"voxcpm9\""),
            ("\"hidden_size\":16", "\"hidden_size\":0"),
            ("\"hidden_size\":16", "\"hidden_size\":15"),
            ("\"num_hidden_layers\":2", "\"num_hidden_layers\":100000"),
            ("\"num_key_value_heads\":2", "\"num_key_value_heads\":3"),
            ("\"num_key_value_heads\":2", "\"num_key_value_heads\":0"),
            ("\"long_factor\":[1.0,1.5]", "\"long_factor\":[1.0]"),
            ("\"short_factor\":[1.0,1.5]", "\"short_factor\":[1.0,-1.0]"),
            ("\"longrope\"", "\"yarn\""),
            ("\"patch_size\":2", "\"patch_size\":0"),
            ("\"feat_dim\":8", "\"feat_dim\":9"),
            ("\"encoder_rates\":[8,8]", "\"encoder_rates\":[]"),
            ("\"encoder_rates\":[8,8]", "\"encoder_rates\":[2,1000]"),
            ("\"decoder_rates\":[8,8]", "\"decoder_rates\":[2,3]"),
            ("\"sample_rate\":16000", "\"sample_rate\":0"),
            ("\"max_length\":1024", "\"max_length\":99999999"),
            ("\"rms_norm_eps\":1e-5", "\"rms_norm_eps\":-1"),
            ("\"vocab_size\":300", "\"vocab_size\":0"),
            ("\"scalar_quantization_scale\":9", "\"scalar_quantization_scale\":0"),
        ] {
            let bad = good.replacen(from, to, 1);
            assert_ne!(bad, good, "{from}");
            assert!(parse(bad.as_bytes()).is_err(), "{from} → {to}");
        }
        for junk in ["", "{", "[]", "null", "{\"lm_config\":5}"] {
            assert!(parse(junk.as_bytes()).is_err(), "{junk}");
        }
        assert!(parse(&vec![b' '; (1 << 20) + 1]).is_err(), "oversized");
        let v2 = tiny(true);
        assert!(parse(v2.replace("\"out_sample_rate\":24000", "\"out_sample_rate\":44100").as_bytes()).is_err(), "rates must agree");
    }
}
