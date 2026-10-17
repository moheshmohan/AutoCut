//! The three `config.json` files of a release (DiT, text encoder, VAE), checked against what this
//! code implements: an unknown architecture, an activation or option this port does not have, or
//! sizes past the caps are refused before any weight is read.

use std::path::Path;

use serde_json::Value;

use super::{R, err};

/// Largest config file read.
const MAX_FILE: u64 = 1 << 20;
/// Caps on sizes (far above the released models; they bound what a hostile file can allocate).
const MAX_HIDDEN: usize = 16_384;
const MAX_LAYERS: usize = 256;
const MAX_HEADS: usize = 256;
const MAX_VOCAB: usize = 1 << 20;

/// A transformer stack's shape (shared by the Qwen3 encoder and the DiT's stacks).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shape {
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub rope_theta: f64,
}

/// The ACE-Step DiT (`AceStepConfig`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DitConfig {
    pub shape: Shape,
    /// Diffusion transformer layers (`num_hidden_layers`).
    pub layers: usize,
    pub lyric_layers: usize,
    pub timbre_layers: usize,
    /// Per layer index: sliding-window attention (else full).
    pub sliding: Vec<bool>,
    /// Half-width of the sliding window (`|i − j| ≤ window`).
    pub window: usize,
    /// Width of the text encoder's states (`text_hidden_dim`).
    pub text_dim: usize,
    /// Latent channels (`audio_acoustic_hidden_dim`, 64).
    pub latent: usize,
    pub patch: usize,
    pub turbo: bool,
}

/// The Qwen3 text encoder.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextConfig {
    pub shape: Shape,
    pub layers: usize,
    pub vocab: usize,
}

/// The Oobleck VAE (`AutoencoderOobleck`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VaeConfig {
    pub channels: usize,
    pub multiples: Vec<usize>,
    /// Upsampling strides of the decoder, first block first (the encoder's, reversed).
    pub strides: Vec<usize>,
    pub latent: usize,
    pub audio_channels: usize,
    pub rate: u32,
}

fn read(path: &Path) -> R<Value> {
    let len = std::fs::metadata(path).map_err(|e| err(format!("{}: {e}", path.display())))?.len();
    if len > MAX_FILE {
        return Err(err(format!("{}: too large for a config", path.display())));
    }
    let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| err(format!("{}: {e}", path.display())))
}

fn num(v: &Value, key: &str, max: usize) -> R<usize> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| err(format!("config: `{key}` is missing or not a whole number")))?;
    let n = usize::try_from(n).map_err(|_| err(format!("config: `{key}` is too large")))?;
    if n == 0 || n > max {
        return Err(err(format!("config: `{key}` = {n} is out of range")));
    }
    Ok(n)
}

fn float(v: &Value, key: &str) -> R<f64> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| err(format!("config: `{key}` is missing or not a positive number")))
}

fn want_str(v: &Value, key: &str, want: &str) -> R<()> {
    match v.get(key).and_then(Value::as_str) {
        Some(s) if s == want => Ok(()),
        other => Err(err(format!("config: `{key}` is {other:?}; this build implements `{want}`"))),
    }
}

/// `key` absent, null or false.
fn off(v: &Value, key: &str) -> R<()> {
    match v.get(key) {
        None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(()),
        Some(x) => Err(err(format!("config: `{key}` = {x} is not supported"))),
    }
}

fn shape(v: &Value) -> R<Shape> {
    let hidden = num(v, "hidden_size", MAX_HIDDEN)?;
    let ffn = num(v, "intermediate_size", 4 * MAX_HIDDEN)?;
    let heads = num(v, "num_attention_heads", MAX_HEADS)?;
    let kv_heads = num(v, "num_key_value_heads", MAX_HEADS)?;
    let head_dim = num(v, "head_dim", 1024)?;
    if heads % kv_heads != 0 {
        return Err(err("config: the attention heads are not a multiple of the key/value heads"));
    }
    if head_dim % 2 != 0 {
        return Err(err("config: `head_dim` must be even (rotary embeddings)"));
    }
    if heads.checked_mul(head_dim).is_none_or(|n| n > 4 * MAX_HIDDEN) {
        return Err(err("config: attention too wide"));
    }
    let eps = float(v, "rms_norm_eps")?;
    let rope_theta = float(v, "rope_theta")?;
    want_str(v, "hidden_act", "silu")?;
    off(v, "attention_bias")?;
    off(v, "rope_scaling")?;
    Ok(Shape { hidden, ffn, heads, kv_heads, head_dim, eps, rope_theta })
}

pub(crate) fn parse_dit(v: &Value) -> R<DitConfig> {
    want_str(v, "model_type", "acestep")?;
    let shape = shape(v)?;
    let layers = num(v, "num_hidden_layers", MAX_LAYERS)?;
    let lyric_layers = num(v, "num_lyric_encoder_hidden_layers", MAX_LAYERS)?;
    let timbre_layers = num(v, "num_timbre_encoder_hidden_layers", MAX_LAYERS)?;
    let types = v.get("layer_types").and_then(Value::as_array).ok_or_else(|| err("config: no `layer_types`"))?;
    if types.len() > MAX_LAYERS {
        return Err(err("config: too many `layer_types`"));
    }
    let use_window = v.get("use_sliding_window").and_then(Value::as_bool).unwrap_or(false);
    let mut sliding = Vec::with_capacity(types.len());
    for t in types {
        sliding.push(match t.as_str() {
            Some("sliding_attention") => use_window,
            Some("full_attention") => false,
            other => return Err(err(format!("config: layer type {other:?} is not supported"))),
        });
    }
    if sliding.len() < layers.max(lyric_layers).max(timbre_layers) {
        return Err(err("config: `layer_types` is shorter than the stacks"));
    }
    let window = if use_window { num(v, "sliding_window", 1 << 20)? } else { usize::MAX };
    let text_dim = num(v, "text_hidden_dim", MAX_HIDDEN)?;
    let latent = num(v, "audio_acoustic_hidden_dim", 4096)?;
    if num(v, "timbre_hidden_dim", 4096)? != latent {
        return Err(err("config: the timbre encoder must read the latent channels"));
    }
    if num(v, "in_channels", 3 * 4096)? != 3 * latent {
        return Err(err("config: `in_channels` must be three times the latent channels (context, mask, noise)"));
    }
    let patch = num(v, "patch_size", 64)?;
    let turbo = v.get("is_turbo").and_then(Value::as_bool).unwrap_or(false);
    Ok(DitConfig { shape, layers, lyric_layers, timbre_layers, sliding, window, text_dim, latent, patch, turbo })
}

pub(crate) fn parse_text(v: &Value) -> R<TextConfig> {
    want_str(v, "model_type", "qwen3")?;
    let shape = shape(v)?;
    off(v, "use_sliding_window")?;
    let layers = num(v, "num_hidden_layers", MAX_LAYERS)?;
    let vocab = num(v, "vocab_size", MAX_VOCAB)?;
    Ok(TextConfig { shape, layers, vocab })
}

pub(crate) fn parse_vae(v: &Value) -> R<VaeConfig> {
    want_str(v, "_class_name", "AutoencoderOobleck")?;
    let channels = num(v, "decoder_channels", 4096)?;
    let latent = num(v, "decoder_input_channels", 4096)?;
    let audio_channels = num(v, "audio_channels", 2)?;
    if audio_channels != 2 {
        return Err(err("config: the VAE must be stereo"));
    }
    let list = |key: &str| -> R<Vec<usize>> {
        let a = v.get(key).and_then(Value::as_array).ok_or_else(|| err(format!("config: no `{key}`")))?;
        if a.is_empty() || a.len() > 16 {
            return Err(err(format!("config: `{key}` has {} entries", a.len())));
        }
        a.iter()
            .map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| (1..=64).contains(n)).ok_or_else(|| err(format!("config: bad `{key}`"))))
            .collect()
    };
    let multiples = list("channel_multiples")?;
    let down = list("downsampling_ratios")?;
    if multiples.len() != down.len() {
        return Err(err("config: one channel multiple per stride"));
    }
    if down.iter().any(|s| s % 2 != 0) {
        return Err(err("config: odd VAE strides are not supported"));
    }
    if down.iter().product::<usize>() != crate::request::HOP {
        return Err(err(format!("config: the VAE must upsample by {}", crate::request::HOP)));
    }
    if multiples.iter().any(|m| channels.checked_mul(*m).is_none_or(|c| c > 1 << 15)) {
        return Err(err("config: the VAE is too wide"));
    }
    let rate = v.get("sampling_rate").and_then(Value::as_u64).ok_or_else(|| err("config: no `sampling_rate`"))?;
    if rate != u64::from(crate::request::SAMPLE_RATE) {
        return Err(err(format!("config: the VAE must run at {} Hz", crate::request::SAMPLE_RATE)));
    }
    let strides = down.into_iter().rev().collect();
    Ok(VaeConfig { channels, multiples, strides, latent, audio_channels, rate: crate::request::SAMPLE_RATE })
}

pub(crate) fn load_dit(path: &Path) -> R<DitConfig> {
    parse_dit(&read(path)?)
}

pub(crate) fn load_text(path: &Path) -> R<TextConfig> {
    parse_text(&read(path)?)
}

pub(crate) fn load_vae(path: &Path) -> R<VaeConfig> {
    parse_vae(&read(path)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny DiT config of the released architecture (tests build random models from it).
    pub(crate) fn tiny_dit() -> Value {
        serde_json::json!({
            "model_type": "acestep", "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "num_hidden_layers": 3, "num_lyric_encoder_hidden_layers": 2,
            "num_timbre_encoder_hidden_layers": 2, "num_attention_pooler_hidden_layers": 1,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention"], "use_sliding_window": true,
            "sliding_window": 3, "text_hidden_dim": 16, "audio_acoustic_hidden_dim": 4, "timbre_hidden_dim": 4,
            "in_channels": 12, "patch_size": 2, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "hidden_act": "silu",
            "attention_bias": false, "rope_scaling": null, "is_turbo": true
        })
    }

    pub(crate) fn tiny_text() -> Value {
        serde_json::json!({
            "model_type": "qwen3", "hidden_size": 16, "intermediate_size": 24, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "num_hidden_layers": 2, "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0, "hidden_act": "silu", "vocab_size": 302, "attention_bias": false,
            "use_sliding_window": false, "rope_scaling": null
        })
    }

    pub(crate) fn tiny_vae() -> Value {
        serde_json::json!({
            "_class_name": "AutoencoderOobleck", "audio_channels": 2, "channel_multiples": [1, 2, 2, 2, 2],
            "decoder_channels": 4, "decoder_input_channels": 4, "downsampling_ratios": [2, 4, 4, 6, 10],
            "encoder_hidden_size": 4, "sampling_rate": 48000
        })
    }

    #[test]
    fn the_released_configs_parse() {
        // the shipped values (acestep-v15-turbo/config.json, Qwen3-Embedding-0.6B, vae/config.json)
        let mut dit = tiny_dit();
        for (k, x) in [
            ("hidden_size", 2048),
            ("intermediate_size", 6144),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("num_hidden_layers", 24),
            ("num_lyric_encoder_hidden_layers", 8),
            ("num_timbre_encoder_hidden_layers", 4),
            ("sliding_window", 128),
            ("text_hidden_dim", 1024),
            ("audio_acoustic_hidden_dim", 64),
            ("timbre_hidden_dim", 64),
            ("in_channels", 192),
        ] {
            dit[k] = serde_json::json!(x);
        }
        dit["layer_types"] = serde_json::json!((0..24).map(|i| if i % 2 == 0 { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>());
        let c = parse_dit(&dit).unwrap();
        assert_eq!((c.layers, c.lyric_layers, c.timbre_layers, c.window, c.latent, c.patch), (24, 8, 4, 128, 64, 2));
        assert!(c.sliding[0] && !c.sliding[1]);
        let mut v = tiny_vae();
        v["channel_multiples"] = serde_json::json!([1, 2, 4, 8, 16]);
        v["decoder_channels"] = serde_json::json!(128);
        v["decoder_input_channels"] = serde_json::json!(64);
        let c = parse_vae(&v).unwrap();
        assert_eq!(c.strides, vec![10, 6, 4, 4, 2]);
        assert!(parse_text(&tiny_text()).is_ok());
    }

    #[test]
    fn other_architectures_and_hostile_values_are_refused() {
        let cases: Vec<(&str, Value)> = vec![
            ("model_type", serde_json::json!("llama")),
            ("hidden_act", serde_json::json!("gelu")),
            ("attention_bias", serde_json::json!(true)),
            ("rope_scaling", serde_json::json!({"type": "yarn"})),
            ("num_key_value_heads", serde_json::json!(3)),
            ("head_dim", serde_json::json!(7)),
            ("hidden_size", serde_json::json!(0)),
            ("hidden_size", serde_json::json!(1u64 << 40)),
            ("num_hidden_layers", serde_json::json!(100_000)),
            ("num_hidden_layers", serde_json::json!(-1)),
            ("rms_norm_eps", serde_json::json!(-1.0)),
            ("rope_theta", serde_json::json!("big")),
            ("layer_types", serde_json::json!(["sliding_attention"])),
            ("layer_types", serde_json::json!(["chunked", "full_attention", "full_attention"])),
            ("in_channels", serde_json::json!(13)),
            ("timbre_hidden_dim", serde_json::json!(5)),
            ("sliding_window", serde_json::json!(0)),
        ];
        for (k, x) in cases {
            let mut v = tiny_dit();
            v[k] = x.clone();
            assert!(parse_dit(&v).is_err(), "dit {k} = {x}");
        }
        for (k, x) in [("model_type", serde_json::json!("qwen2")), ("use_sliding_window", serde_json::json!(true)), ("vocab_size", serde_json::json!(0))] {
            let mut v = tiny_text();
            v[k] = x.clone();
            assert!(parse_text(&v).is_err(), "text {k} = {x}");
        }
        for (k, x) in [
            ("_class_name", serde_json::json!("AutoencoderKL")),
            ("audio_channels", serde_json::json!(1)),
            ("downsampling_ratios", serde_json::json!([2, 4, 4, 8, 8])),
            ("downsampling_ratios", serde_json::json!([3, 640])),
            ("downsampling_ratios", serde_json::json!([])),
            ("channel_multiples", serde_json::json!([1, 2])),
            ("decoder_channels", serde_json::json!(1u64 << 20)),
            ("sampling_rate", serde_json::json!(44100)),
        ] {
            let mut v = tiny_vae();
            v[k] = x.clone();
            assert!(parse_vae(&v).is_err(), "vae {k} = {x}");
        }
        for junk in ["", "{", "[]", "null", "5"] {
            let v: Value = serde_json::from_str(junk).unwrap_or(Value::Null);
            assert!(parse_dit(&v).is_err() && parse_text(&v).is_err() && parse_vae(&v).is_err());
        }
    }

    #[test]
    fn files_are_read_with_a_cap() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("config.json");
        std::fs::write(&p, serde_json::to_vec(&tiny_dit()).unwrap()).unwrap();
        assert!(load_dit(&p).is_ok());
        std::fs::write(&p, vec![b' '; (MAX_FILE + 1) as usize]).unwrap();
        assert!(load_dit(&p).is_err());
        assert!(load_vae(&d.join("missing.json")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
