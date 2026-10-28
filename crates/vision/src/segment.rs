//! MobileSAM (Apache-2.0, [`crate::catalog::SEGMENTER`]): a person or a thing cut out of a picture
//! given points on it and round it, on candle's Segment Anything with the TinyViT encoder — on the
//! GPU (Metal) on macOS, the CPU elsewhere and as the fallback.
//!
//! A picture is embedded once ([`Segmenter::embed`]: its long side scaled to 1024, the encoder
//! run), then any number of prompts are answered from the embedding ([`Segmenter::cut`]: the mask
//! decoder's three answers, the surest kept, its 256 × 256 logits scaled up to the picture
//! bilinearly and thresholded softly at the edge).

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::segment_anything::sam::{IMAGE_SIZE, Sam};

use crate::matte::Matte;
use crate::{VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("segmenter: {e}"))
}

/// The segmenter, loaded.
pub struct Segmenter {
    sam: Sam,
    dev: Device,
}

/// A picture as the segmenter sees it: its encoder's embedding, and its size as scaled for it.
pub struct Embedded {
    emb: Tensor,
    w: usize,
    h: usize,
    /// Picture pixels to the embedded picture's.
    scale: f32,
}

impl Embedded {
    /// The size of the cut-outs [`Segmenter::cut`] makes of it.
    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    /// Picture pixels to cut-out pixels.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

/// The device the segmenter runs on: Metal on macOS when there is a GPU, else the CPU.
fn device(gpu: bool) -> Device {
    #[cfg(target_os = "macos")]
    if gpu && let Ok(d) = Device::new_metal(0) {
        return d;
    }
    let _ = gpu;
    Device::Cpu
}

impl Segmenter {
    /// Load MobileSAM from the models directory, on the GPU when `gpu` and there is one.
    pub fn load(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        if !catalog::segmenter_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let path = catalog::segment_dir(models_dir).join(catalog::SEGMENTER.name);
        let dev = device(gpu);
        let load = |dev: &Device| -> Result<Sam, VisionError> {
            let vb = candle_nn::VarBuilder::from_pth(&path, DType::F32, dev).map_err(err)?;
            Sam::new_tiny(vb).map_err(err)
        };
        match load(&dev) {
            Ok(sam) => Ok(Self { sam, dev }),
            // (the GPU refused it: the CPU)
            Err(_) if !matches!(dev, Device::Cpu) => Ok(Self { sam: load(&Device::Cpu)?, dev: Device::Cpu }),
            Err(e) => Err(e),
        }
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if matches!(self.dev, Device::Cpu) { "cpu" } else { "gpu" }
    }

    /// Embed an RGBA picture (`w` × `h`).
    pub fn embed(&self, rgba: &[u8], w: usize, h: usize) -> Result<Embedded, VisionError> {
        if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) {
            return Err(VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())));
        }
        let scale = IMAGE_SIZE as f32 / w.max(h) as f32;
        let (ew, eh) = (((w as f32 * scale).round() as usize).clamp(1, IMAGE_SIZE), ((h as f32 * scale).round() as usize).clamp(1, IMAGE_SIZE));
        // area-averaged down to the encoder's size, planar RGB 0..255
        let mut px = vec![0f32; 3 * ew * eh];
        for y in 0..eh {
            let (y0, y1) = (y * h / eh, ((y + 1) * h / eh).max(y * h / eh + 1).min(h));
            for x in 0..ew {
                let (x0, x1) = (x * w / ew, ((x + 1) * w / ew).max(x * w / ew + 1).min(w));
                let (mut acc, mut n) = ([0f32; 3], 0f32);
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        if let Some(p) = rgba.get((sy * w + sx) * 4..(sy * w + sx) * 4 + 3) {
                            acc[0] += f32::from(p[0]);
                            acc[1] += f32::from(p[1]);
                            acc[2] += f32::from(p[2]);
                            n += 1.0;
                        }
                    }
                }
                for c in 0..3 {
                    px[c * ew * eh + y * ew + x] = acc[c] / n.max(1.0);
                }
            }
        }
        let img = Tensor::from_vec(px, (3, eh, ew), &self.dev).map_err(err)?;
        let emb = self.sam.embeddings(&img).map_err(err)?;
        Ok(Embedded { emb, w: ew, h: eh, scale })
    }

    /// The cut-out of what is at `inside` and not at `outside` (the picture's pixels), at the
    /// embedded picture's size ([`Embedded::size`]; [`Embedded::scale`] maps the picture to it).
    pub fn cut(&self, e: &Embedded, inside: &[(f32, f32)], outside: &[(f32, f32)]) -> Result<Matte, VisionError> {
        let norm = |(x, y): &(f32, f32)| (f64::from(x * e.scale / e.w as f32).clamp(0.0, 1.0), f64::from(y * e.scale / e.h as f32).clamp(0.0, 1.0));
        let points: Vec<(f64, f64, bool)> = inside
            .iter()
            .filter(|p| p.0.is_finite() && p.1.is_finite())
            .map(|p| (norm(p).0, norm(p).1, true))
            .chain(outside.iter().filter(|p| p.0.is_finite() && p.1.is_finite()).map(|p| (norm(p).0, norm(p).1, false)))
            .take(32)
            .collect();
        if !points.iter().any(|p| p.2) {
            return Err(VisionError::Invalid("no point on what to cut out".into()));
        }
        let (masks, iou) = self.sam.forward_for_embeddings(&e.emb, e.h, e.w, &points, true).map_err(err)?;
        // the surest of the three answers
        let scores: Vec<f32> = iou.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1()).map_err(err)?;
        let (best, score) = scores.iter().copied().enumerate().filter(|(_, s)| s.is_finite()).fold((0usize, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
        let low = masks.i((0, best)).map_err(err)?;
        // the 256 × 256 logits cover the 1024 square the picture sits in, top-left
        let side = IMAGE_SIZE;
        let up = low.unsqueeze(0).and_then(|t| t.unsqueeze(0)).and_then(|t| t.upsample_bilinear2d(side, side, false)).map_err(err)?;
        let up = up.i((0, 0, ..e.h, ..e.w)).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(err)?;
        // logits to coverage: a soft edge a logit wide either side of 0
        let alpha = up.iter().map(|v| if v.is_finite() { ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 }).collect();
        Ok(Matte { w: e.w, h: e.h, alpha, score: score.clamp(0.0, 1.0) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The downloaded model on a drawn picture: a bright figure on a dark ground is cut out.
    /// (Skips when MobileSAM is not downloaded.)
    #[test]
    fn a_figure_is_cut_out_where_the_points_say() {
        let Some(dir) = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(seg) = Segmenter::load(&dir, std::env::var_os("SEG_CPU").is_none()) else {
            eprintln!("MobileSAM not downloaded; skipping");
            return;
        };
        let (w, h) = (640usize, 360usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                // a ground of soft noise, and a figure: a head and a body
                let g = (((x * 7 + y * 13) % 23) as u8).saturating_add(30);
                let head = ((x as f32 - 320.0).powi(2) + (y as f32 - 90.0).powi(2)).sqrt() < 30.0;
                let body = (270..370).contains(&x) && (120..330).contains(&y);
                let v = if head || body { [220, 180, 150] } else { [g, g + 10, g + 20] };
                rgba[i..i + 4].copy_from_slice(&[v[0], v[1], v[2], 255]);
            }
        }
        let t = std::time::Instant::now();
        let e = seg.embed(&rgba, w, h).unwrap();
        let m = seg.cut(&e, &[(320.0, 90.0), (320.0, 220.0)], &[(150.0, 200.0), (500.0, 200.0)]).unwrap();
        eprintln!(
            "MobileSAM on the {}: {:.0} ms, score {:.2}, {:.1}% covered",
            seg.device_name(),
            t.elapsed().as_secs_f64() * 1000.0,
            m.score,
            100.0 * m.area()
        );
        assert_eq!((m.w, m.h), e.size());
        // a video's frames after the first: the encoder warm
        let t = std::time::Instant::now();
        for _ in 0..3 {
            let e = seg.embed(&rgba, w, h).unwrap();
            let _ = seg.cut(&e, &[(320.0, 90.0)], &[]).unwrap();
        }
        eprintln!("then {:.0} ms a picture", t.elapsed().as_secs_f64() * 1000.0 / 3.0);
        let s = e.scale();
        let at = |x: f32, y: f32| m.at((x * s) as usize, (y * s) as usize);
        assert!(at(320.0, 220.0) > 200 && at(320.0, 90.0) > 200, "the figure is in");
        assert!(at(100.0, 100.0) < 50 && at(560.0, 300.0) < 50, "the ground is out");
        assert!(seg.cut(&e, &[], &[(1.0, 1.0)]).is_err());
        assert!(seg.embed(&rgba[..100], w, h).is_err());
    }
}
