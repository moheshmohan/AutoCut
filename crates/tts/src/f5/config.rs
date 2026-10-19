//! IndicF5's architecture, read from the weights and checked.
//!
//! The release's `config.json` only names the Hugging Face wrapper (`INF5Model`); the network's
//! sizes are fixed in its code (`DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
//! conv_layers=4)`, Vocos `mel-24khz`). So the sizes are read from the tensor shapes in
//! `model.safetensors`, every one checked for consistency and against caps, and anything this
//! code does not implement (another mel size, FFT size, head size or layout) is refused before
//! a weight is read.

use super::{MELS, N_FFT};
use crate::voxcpm::weights::SafeTensors;
use crate::voxcpm::{R, err};

/// Most transformer blocks, ConvNeXt blocks and the largest width accepted.
const MAX_DEPTH: usize = 64;
const MAX_DIM: usize = 4096;
const MAX_TEXT_ROWS: usize = 100_000;

/// The sizes of an IndicF5 (F5-TTS v1 "Base" DiT) release.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Tensor name prefixes of the DiT and the vocoder in the file.
    pub dit: String,
    pub vocoder: String,
    /// Transformer width, depth, heads and head size (64: the rotary table's width).
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Feed-forward width.
    pub ff: usize,
    /// Text embedding: rows (vocabulary + the filler token 0), width, ConvNeXt blocks and
    /// their inner width.
    pub text_rows: usize,
    pub text_dim: usize,
    pub text_blocks: usize,
    pub text_ff: usize,
    /// The convolutional position embedding: kernel and groups.
    pub pos_kernel: usize,
    pub pos_groups: usize,
    /// Vocos: width, inner width and ConvNeXt blocks.
    pub voc_dim: usize,
    pub voc_ff: usize,
    pub voc_blocks: usize,
}

fn shape<'a>(st: &'a SafeTensors, name: &str) -> R<&'a [usize]> {
    st.shape(name).ok_or_else(|| err(format!("model.safetensors: missing `{name}` (not an IndicF5 release)")))
}

fn count(st: &SafeTensors, f: impl Fn(usize) -> String) -> usize {
    (0..=MAX_DEPTH).take_while(|i| st.shape(&f(*i)).is_some()).count()
}

impl Config {
    /// The architecture of the weights in `st`.
    pub(crate) fn read(st: &SafeTensors) -> R<Config> {
        // the Hugging Face wrapper saved the compiled modules (`_orig_mod`); a plain save has none
        let dit = ["ema_model._orig_mod.transformer.", "ema_model.transformer.", "transformer."]
            .into_iter()
            .find(|p| st.shape(&format!("{p}text_embed.text_embed.weight")).is_some())
            .ok_or_else(|| err("model.safetensors: no F5-TTS transformer (not an IndicF5 release)"))?
            .to_string();
        let vocoder = ["vocoder._orig_mod.", "vocoder."]
            .into_iter()
            .find(|p| st.shape(&format!("{p}head.out.weight")).is_some())
            .ok_or_else(|| err("model.safetensors: no Vocos vocoder (not an IndicF5 release)"))?
            .to_string();
        let d = |n: &str| format!("{dit}{n}");
        let v = |n: &str| format!("{vocoder}{n}");
        let dims2 = |name: &str| -> R<(usize, usize)> {
            match shape(st, name)? {
                [a, b] => Ok((*a, *b)),
                s => Err(err(format!("`{name}` is {s:?}, expected a matrix"))),
            }
        };
        let (text_rows, text_dim) = dims2(&d("text_embed.text_embed.weight"))?;
        let (dim, freq) = dims2(&d("time_embed.time_mlp.0.weight"))?;
        let rot = match shape(st, &d("rotary_embed.inv_freq"))? {
            [n] => *n,
            s => return Err(err(format!("rotary table is {s:?}"))),
        };
        let head_dim = 2 * rot;
        let depth = count(st, |i| d(&format!("transformer_blocks.{i}.attn.to_q.weight")));
        let (ff, _) = dims2(&d("transformer_blocks.0.ff.ff.0.0.weight"))?;
        let text_blocks = count(st, |i| d(&format!("text_embed.text_blocks.{i}.pwconv1.weight")));
        let (text_ff, _) = if text_blocks > 0 { dims2(&d("text_embed.text_blocks.0.pwconv1.weight"))? } else { (0, 0) };
        let (pos_out, pos_in, pos_kernel) = match shape(st, &d("input_embed.conv_pos_embed.conv1d.0.weight"))? {
            [a, b, c] => (*a, *b, *c),
            s => return Err(err(format!("position convolution is {s:?}"))),
        };
        let (mel_out, _) = dims2(&d("proj_out.weight"))?;
        let (proj_out, proj_in) = dims2(&d("input_embed.proj.weight"))?;
        let (voc_dim, voc_mels, _) = match shape(st, &v("backbone.embed.weight"))? {
            [a, b, c] => (*a, *b, *c),
            s => return Err(err(format!("vocoder input is {s:?}"))),
        };
        let voc_blocks = count(st, |i| v(&format!("backbone.convnext.{i}.pwconv1.weight")));
        let (voc_ff, _) = dims2(&v("backbone.convnext.0.pwconv1.weight"))?;
        let (head_out, _) = dims2(&v("head.out.weight"))?;
        let cfg = Config {
            dit,
            vocoder,
            dim,
            depth,
            heads: dim.checked_div(head_dim).unwrap_or(0),
            head_dim,
            ff,
            text_rows,
            text_dim,
            text_blocks,
            text_ff,
            pos_kernel,
            pos_groups: pos_out.checked_div(pos_in).unwrap_or(0),
            voc_dim,
            voc_ff,
            voc_blocks,
        };
        // what this code implements
        let checks: [(bool, &str); 14] = [
            (freq == 256, "the time embedding is not 256-wide"),
            (head_dim == 64, "the attention heads are not 64-wide"),
            (dim > 0 && dim <= MAX_DIM && dim.is_multiple_of(64), "bad transformer width"),
            ((1..=MAX_DEPTH).contains(&depth), "bad transformer depth"),
            (ff > 0 && ff <= 4 * MAX_DIM, "bad feed-forward width"),
            (text_rows > 1 && text_rows <= MAX_TEXT_ROWS, "bad text vocabulary"),
            (text_dim > 0 && text_dim <= MAX_DIM, "bad text width"),
            (text_blocks <= MAX_DEPTH && text_ff <= 4 * MAX_DIM, "bad text blocks"),
            (pos_out == dim && pos_in > 0 && dim.is_multiple_of(pos_in) && pos_kernel % 2 == 1 && pos_kernel <= 63, "bad position convolution"),
            (mel_out == MELS && voc_mels == MELS, "the mel spectrogram is not 100 bands"),
            (proj_out == dim && proj_in == 2 * MELS + text_dim, "the input projection does not match"),
            (head_out == N_FFT + 2, "the vocoder's FFT is not 1024 points"),
            (voc_dim > 0 && voc_dim <= MAX_DIM && voc_ff > 0 && voc_ff <= 4 * MAX_DIM, "bad vocoder width"),
            ((1..=MAX_DEPTH).contains(&voc_blocks), "bad vocoder depth"),
        ];
        if let Some((_, why)) = checks.iter().find(|(ok, _)| !ok) {
            return Err(err(format!("model.safetensors: {why} ({cfg:?})")));
        }
        Ok(cfg)
    }
}
