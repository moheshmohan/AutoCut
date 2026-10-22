//! SigLIP 2 base (Google; Apache-2.0) through `candle-transformers`' SigLIP: keyframes and text
//! queries embedded into one space, so a query finds the shots that look like it. The checkpoint
//! `google/siglip2-base-patch16-224` is the fixed-resolution SigLIP architecture with a 256 k
//! Gemma vocabulary; pictures are squashed to 224 × 224 and scaled to −1..1, text is lower-cased,
//! ends with `<eos>` and is padded to 64 tokens, as it was trained.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;

use super::image::{planar, resize};
use super::weights::SafeTensors;
use super::{R, err};
use crate::search::Calibration;

/// Picture side and text length the model was trained at.
pub const SIDE: usize = 224;
pub const TEXT_LEN: usize = 64;
/// Pictures embedded at once.
const BATCH: usize = 16;

pub struct Siglip {
    model: siglip::Model,
    tok: tokenizers::Tokenizer,
    pad: u32,
    eos: u32,
    calibration: Calibration,
    dev: Device,
}

impl Siglip {
    pub fn load(dir: &Path, dev: &Device) -> R<Siglip> {
        let mut st = SafeTensors::open(&dir.join("model.safetensors"))?;
        let mut w = HashMap::new();
        st.load_prefixed(&["vision_model.", "text_model.", "logit_scale", "logit_bias"], dev, DType::F32, &mut w)?;
        let scalar = |w: &HashMap<String, Tensor>, k: &str| -> R<f32> {
            w.get(k)
                .ok_or_else(|| err(format!("SigLIP: missing {k}")))?
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(err)?
                .first()
                .copied()
                .ok_or_else(|| err(format!("SigLIP: empty {k}")))
        };
        let (scale, bias) = (scalar(&w, "logit_scale")?, scalar(&w, "logit_bias")?);
        let vocab = w.get("text_model.embeddings.token_embedding.weight").map(|t| t.dims().first().copied().unwrap_or(0)).unwrap_or(0);
        let mut cfg = siglip::Config::base_patch16_224();
        cfg.text_config.vocab_size = vocab;
        let vb = VarBuilder::from_tensors(w, DType::F32, dev);
        let model = siglip::Model::new(&cfg, vb).map_err(|e| err(format!("SigLIP: {e}")))?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| err(format!("SigLIP tokenizer: {e}")))?;
        let pad = tok.token_to_id("<pad>").ok_or_else(|| err("SigLIP tokenizer: no <pad>"))?;
        let eos = tok.token_to_id("<eos>").ok_or_else(|| err("SigLIP tokenizer: no <eos>"))?;
        Ok(Siglip { model, tok, pad, eos, calibration: Calibration { scale: scale.exp(), bias }, dev: dev.clone() })
    }

    /// How a cosine becomes SigLIP's match probability.
    pub fn calibration(&self) -> Calibration {
        self.calibration
    }

    /// Unit-length embeddings of RGBA8 pictures `(rgba, w, h)`.
    pub fn embed_images(&self, pics: &[(&[u8], usize, usize)]) -> R<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(pics.len());
        for chunk in pics.chunks(BATCH) {
            let mut px = Vec::with_capacity(chunk.len() * 3 * SIDE * SIDE);
            for (rgba, w, h) in chunk {
                let img = resize(rgba, *w, *h, SIDE, SIDE).ok_or_else(|| crate::SceneError::Invalid("a keyframe is empty or damaged".into()))?;
                px.extend(planar(&img, [0.5; 3], [0.5; 3]));
            }
            let x = Tensor::from_vec(px, (chunk.len(), 3, SIDE, SIDE), &self.dev).map_err(err)?;
            let e = self.model.get_image_features(&x).map_err(err)?;
            out.extend(unit_rows(&e)?);
        }
        Ok(out)
    }

    /// The unit-length embedding of a text query.
    pub fn embed_text(&self, text: &str) -> R<Vec<f32>> {
        let text: String = text.chars().take(1000).collect::<String>().to_lowercase();
        let mut ids = self.tok.encode(text.as_str(), false).map_err(|e| err(format!("SigLIP tokenizer: {e}")))?.get_ids().to_vec();
        ids.truncate(TEXT_LEN - 1);
        ids.push(self.eos);
        ids.resize(TEXT_LEN, self.pad);
        let x = Tensor::from_vec(ids, (1, TEXT_LEN), &self.dev).map_err(err)?;
        let e = self.model.get_text_features(&x).map_err(err)?;
        unit_rows(&e)?.into_iter().next().ok_or_else(|| err("SigLIP: no text embedding"))
    }
}

/// The rows of `[n, d]` as unit-length vectors.
pub(crate) fn unit_rows(e: &Tensor) -> R<Vec<Vec<f32>>> {
    let rows: Vec<Vec<f32>> = e.to_dtype(DType::F32).and_then(|e| e.to_vec2()).map_err(err)?;
    Ok(rows.into_iter().map(|r| filmcraft_project::scenes::Embedding(r).normalized().0).collect())
}
