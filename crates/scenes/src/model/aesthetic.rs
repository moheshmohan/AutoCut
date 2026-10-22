//! The LAION aesthetic predictor (MIT): one linear layer over the unit-length CLIP ViT-L/14 image
//! embedding (OpenAI CLIP, MIT), trained on human ratings of how good a picture looks, about 1
//! (bad) to 10 (beautiful). The CLIP vision tower runs through `candle-transformers`; pictures are
//! scaled so their short side is 224 px, centre-cropped and normalised with CLIP's mean and
//! deviation. The head's weights are a PyTorch file read by candle's pickle reader (pure Rust,
//! data only: no code in it is run).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::clip::text_model::Activation;
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};

use super::image::{center_crop, crop, planar, resize};
use super::weights::SafeTensors;
use super::{R, err};

const SIDE: usize = 224;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
const BATCH: usize = 16;
/// The head's file in the model's directory.
pub const HEAD: &str = "sa_0_4_vit_l_14_linear.pth";

pub struct Aesthetic {
    vision: ClipVisionTransformer,
    projection: Linear,
    head_w: Tensor,
    head_b: Tensor,
    dev: Device,
}

fn vit_l14() -> ClipVisionConfig {
    ClipVisionConfig {
        embed_dim: 1024,
        activation: Activation::QuickGelu,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        projection_dim: 768,
        num_channels: 3,
        image_size: SIDE,
        patch_size: 14,
    }
}

impl Aesthetic {
    pub fn load(dir: &Path, dev: &Device) -> R<Aesthetic> {
        let mut st = SafeTensors::open(&dir.join("model.safetensors"))?;
        let mut w = HashMap::new();
        st.load_prefixed(&["vision_model.", "visual_projection."], dev, DType::F32, &mut w)?;
        let proj = w.remove("visual_projection.weight").ok_or_else(|| err("CLIP: missing visual_projection.weight"))?;
        if proj.dims() != [768, 1024] {
            return Err(err("CLIP: the projection is not ViT-L/14's"));
        }
        let vb = VarBuilder::from_tensors(w, DType::F32, dev);
        let vision = ClipVisionTransformer::new(vb.pp("vision_model"), &vit_l14()).map_err(|e| err(format!("CLIP: {e}")))?;
        let head = candle_core::pickle::read_all(dir.join(HEAD)).map_err(|e| err(format!("aesthetic head: {e}")))?;
        let get = |k: &str| head.iter().find(|(n, _)| n == k).map(|(_, t)| t.clone()).ok_or_else(|| err(format!("aesthetic head: missing {k}")));
        let head_w = get("weight")?.to_dtype(DType::F32).and_then(|t| t.to_device(dev)).map_err(err)?;
        let head_b = get("bias")?.to_dtype(DType::F32).and_then(|t| t.to_device(dev)).map_err(err)?;
        if head_w.dims() != [1, 768] || head_b.dims() != [1] {
            return Err(err("aesthetic head: not a 768 → 1 linear layer"));
        }
        Ok(Aesthetic { vision, projection: Linear::new(proj, None), head_w, head_b, dev: dev.clone() })
    }

    /// The aesthetic score of each RGBA8 picture `(rgba, w, h)`.
    pub fn score(&self, pics: &[(&[u8], usize, usize)]) -> R<Vec<f32>> {
        let mut out = Vec::with_capacity(pics.len());
        for chunk in pics.chunks(BATCH) {
            let mut px = Vec::with_capacity(chunk.len() * 3 * SIDE * SIDE);
            for (rgba, w, h) in chunk {
                let c = crop(rgba, *w, *h, center_crop(*w, *h, 1, 1)).ok_or_else(|| crate::SceneError::Invalid("a keyframe is empty or damaged".into()))?;
                let (_, _, cw, ch) = center_crop(*w, *h, 1, 1);
                let img = resize(&c, cw, ch, SIDE, SIDE).ok_or_else(|| crate::SceneError::Invalid("a keyframe is empty or damaged".into()))?;
                px.extend(planar(&img, MEAN, STD));
            }
            let x = Tensor::from_vec(px, (chunk.len(), 3, SIDE, SIDE), &self.dev).map_err(err)?;
            let e = self.vision.forward(&x).and_then(|e| self.projection.forward(&e)).map_err(err)?;
            let norm = e.sqr().and_then(|s| s.sum_keepdim(1)).and_then(|s| s.sqrt()).and_then(|s| s.clamp(1e-12, f64::MAX)).map_err(err)?;
            let e = e.broadcast_div(&norm).map_err(err)?;
            let s = e
                .matmul(&self.head_w.t().map_err(err)?)
                .and_then(|s| s.broadcast_add(&self.head_b))
                .and_then(|s| s.flatten_all())
                .and_then(|s| s.to_vec1::<f32>())
                .map_err(err)?;
            out.extend(s.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }));
        }
        Ok(out)
    }
}
