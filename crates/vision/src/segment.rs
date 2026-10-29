//! MobileSAM (Apache-2.0, [`crate::catalog::SEGMENTER`]): a person or a thing cut out of a picture
//! given points on it and round it, on candle's Segment Anything with the TinyViT encoder — on the
//! GPU (Metal) on macOS, the CPU elsewhere and as the fallback — its image encoder as our own WGSL
//! passes on the GPU (`crate::sam_gpu`: candle runs its depthwise convolutions one channel at a
//! time), candle's after a GPU failure.
//!
//! A picture is embedded once ([`Segmenter::embed`]: its long side scaled to 1024, the encoder
//! run), then any number of prompts are answered from the embedding ([`Segmenter::cut`]: the mask
//! decoder's three answers, the surest kept, its 256 × 256 logits scaled up to the picture
//! bilinearly and thresholded softly at the edge). [`Segmenter::answers`] also takes a box round
//! the thing (how a tracker tells it where the thing went) and gives all four answers: the one
//! made for a prompt that is not ambiguous, then the three for one that is. (candle's `Sam` takes
//! points only and its prompt encoder drops a dimension embedding a box, so the prompt's points
//! and box corners are embedded here — each a random-Fourier code of where it is plus its kind's
//! learnt vector, as Segment Anything does — from the same weights, and its mask decoder is built
//! a second time.)

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::segment_anything::mask_decoder::MaskDecoder;
use candle_transformers::models::segment_anything::prompt_encoder::PromptEncoder;
use candle_transformers::models::segment_anything::sam::{IMAGE_SIZE, Sam};

use crate::matte::Matte;
use crate::{VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("segmenter: {e}"))
}

/// The width of the prompt embeddings and the side of the image embedding (as candle's `Sam`).
const PROMPT_DIM: usize = 256;
const EMBEDDING_SIDE: usize = IMAGE_SIZE / 16;
/// Most points one prompt takes.
const MAX_POINTS: usize = 32;

/// The segmenter, loaded.
pub struct Segmenter {
    sam: Sam,
    prompts: PromptEncoder,
    sparse: Sparse,
    decoder: MaskDecoder,
    dev: Device,
    /// The image encoder on the GPU (`sam_gpu`), when there is one; candle after it fails.
    encoder: Option<crate::sam_gpu::SamGpu>,
    encoder_failed: std::sync::atomic::AtomicBool,
}

/// What embeds a prompt's points and box corners: the position code's Gaussian matrix (2 × 128,
/// row-major) and the learnt vectors of a point off the thing, on it, a box's top-left and
/// bottom-right corners, and of no point (the padding after points given without a box).
struct Sparse {
    gauss: Vec<f32>,
    kinds: [Vec<f32>; 4],
    none: Vec<f32>,
}

impl Sparse {
    fn load(vb: &candle_nn::VarBuilder) -> Result<Self, VisionError> {
        let vec = |t: Tensor| t.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()).map_err(err);
        let gauss = vec(vb.pp("pe_layer").get((2, PROMPT_DIM / 2), "positional_encoding_gaussian_matrix").map_err(err)?)?;
        let kind = |i: usize| vec(vb.pp("point_embeddings").pp(i).get((1, PROMPT_DIM), "weight").map_err(err)?);
        let kinds = [kind(0)?, kind(1)?, kind(2)?, kind(3)?];
        let none = vec(vb.pp("not_a_point_embed").get((1, PROMPT_DIM), "weight").map_err(err)?)?;
        Ok(Self { gauss, kinds, none })
    }

    /// The code of a place in the 1024 square (its pixels): sines then cosines of its random
    /// projections.
    fn code(&self, x: f32, y: f32) -> Vec<f32> {
        let half = PROMPT_DIM / 2;
        let (cx, cy) = (2.0 * x / IMAGE_SIZE as f32 - 1.0, 2.0 * y / IMAGE_SIZE as f32 - 1.0);
        let proj: Vec<f32> = (0..half)
            .map(|j| std::f32::consts::TAU * (cx * self.gauss.get(j).copied().unwrap_or(0.0) + cy * self.gauss.get(half + j).copied().unwrap_or(0.0)))
            .collect();
        proj.iter().map(|v| v.sin()).chain(proj.iter().map(|v| v.cos())).collect()
    }

    /// The prompt's embeddings `(1, n, 256)`: each point (`(x, y)` in the 1024 square, on the
    /// thing or not), then the box's two corners — or, without a box, a padding "no point".
    fn embed(&self, points: &[((f32, f32), bool)], bbox: Option<[f32; 4]>, dev: &Device) -> Result<Tensor, VisionError> {
        let mut rows: Vec<f32> = Vec::with_capacity((points.len() + 2) * PROMPT_DIM);
        let mut put = |code: Vec<f32>, kind: &[f32]| rows.extend(code.iter().zip(kind).map(|(a, b)| a + b));
        for ((x, y), on) in points {
            put(self.code(x + 0.5, y + 0.5), &self.kinds[usize::from(*on)]);
        }
        match bbox {
            Some(b) => {
                put(self.code(b[0] + 0.5, b[1] + 0.5), &self.kinds[2]);
                put(self.code(b[2] + 0.5, b[3] + 0.5), &self.kinds[3]);
            }
            None => rows.extend_from_slice(&self.none),
        }
        let n = rows.len() / PROMPT_DIM;
        Tensor::from_vec(rows, (1, n, PROMPT_DIM), dev).map_err(err)
    }
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
        let load = |dev: &Device| -> Result<Self, VisionError> {
            let vb = candle_nn::VarBuilder::from_pth(&path, DType::F32, dev).map_err(err)?;
            let prompts =
                PromptEncoder::new(PROMPT_DIM, (EMBEDDING_SIDE, EMBEDDING_SIDE), (IMAGE_SIZE, IMAGE_SIZE), 16, vb.pp("prompt_encoder")).map_err(err)?;
            let sparse = Sparse::load(&vb.pp("prompt_encoder"))?;
            let decoder = MaskDecoder::new(PROMPT_DIM, 3, 3, 256, vb.pp("mask_decoder")).map_err(err)?;
            Ok(Self { sam: Sam::new_tiny(vb).map_err(err)?, prompts, sparse, decoder, dev: dev.clone(), encoder: None, encoder_failed: Default::default() })
        };
        let mut s = match load(&dev) {
            Ok(s) => s,
            // (the GPU refused it: the CPU)
            Err(_) if !matches!(dev, Device::Cpu) => load(&Device::Cpu)?,
            Err(e) => return Err(e),
        };
        // the image encoder as our own passes on the GPU (candle's runs its depthwise convolutions
        // one channel at a time)
        if gpu {
            let cpu = candle_nn::VarBuilder::from_pth(&path, DType::F32, &Device::Cpu).map_err(err)?;
            s.encoder = crate::sam_gpu::EncoderWeights::read(&cpu).ok().and_then(|w| crate::sam_gpu::SamGpu::new(&w));
        }
        Ok(s)
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if matches!(self.dev, Device::Cpu) && !self.encoder_on() { "cpu" } else { "gpu" }
    }

    fn encoder_on(&self) -> bool {
        self.encoder.is_some() && !self.encoder_failed.load(std::sync::atomic::Ordering::Relaxed)
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
        if let Some(enc) = self.encoder.as_ref().filter(|_| self.encoder_on()) {
            // normalised as candle's `Sam` does, padded with zeros to the 1024 square
            let n = crate::sam_gpu::INPUT;
            let (mean, std) = ([123.675f32, 116.28, 103.53], [58.395f32, 57.12, 57.375]);
            let mut x = vec![0f32; 3 * n * n];
            for c in 0..3 {
                for y in 0..eh {
                    for xx in 0..ew {
                        if let (Some(d), Some(v)) = (x.get_mut((c * n + y) * n + xx), px.get(c * ew * eh + y * ew + xx)) {
                            *d = (v - mean[c]) / std[c];
                        }
                    }
                }
            }
            match enc.run(&x) {
                Ok(v) => {
                    let (c, s) = (crate::sam_gpu::EMB_C, crate::sam_gpu::EMB_SIDE);
                    let emb = Tensor::from_vec(v, (1, c, s, s), &self.dev).map_err(err)?;
                    return Ok(Embedded { emb, w: ew, h: eh, scale });
                }
                Err(_) => self.encoder_failed.store(true, std::sync::atomic::Ordering::Relaxed),
            }
        }
        let img = Tensor::from_vec(px, (3, eh, ew), &self.dev).map_err(err)?;
        let emb = self.sam.embeddings(&img).map_err(err)?;
        Ok(Embedded { emb, w: ew, h: eh, scale })
    }

    /// The cut-out of what is at `inside` and not at `outside` (the picture's pixels), at the
    /// embedded picture's size ([`Embedded::size`]; [`Embedded::scale`] maps the picture to it):
    /// the surest of the three answers for an ambiguous prompt.
    pub fn cut(&self, e: &Embedded, inside: &[(f32, f32)], outside: &[(f32, f32)]) -> Result<Matte, VisionError> {
        if !inside.iter().any(|p| p.0.is_finite() && p.1.is_finite()) {
            return Err(VisionError::Invalid("no point on what to cut out".into()));
        }
        let (masks, scores) = self.decode(e, inside, outside, None, true)?;
        let best = scores.iter().copied().enumerate().filter(|(_, s)| s.is_finite()).fold((0usize, f32::MIN), |a, b| if b.1 > a.1 { b } else { a }).0;
        self.matte(e, &masks, best, scores.get(best).copied().unwrap_or(0.0))
    }

    /// Every answer to a prompt — points on the thing (`inside`), beside it (`outside`) and a box
    /// round it (`bbox`, `[x0, y0, x1, y1]`), all in the picture's pixels: the answer for a prompt
    /// that is not ambiguous first, then the three for one that is, each at the embedded
    /// picture's size with how sure the segmenter is of it.
    pub fn answers(&self, e: &Embedded, inside: &[(f32, f32)], outside: &[(f32, f32)], bbox: Option<[f32; 4]>) -> Result<Vec<Matte>, VisionError> {
        let bbox = bbox.filter(|b| b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]);
        if bbox.is_none() && !inside.iter().any(|p| p.0.is_finite() && p.1.is_finite()) {
            return Err(VisionError::Invalid("no point or box on what to cut out".into()));
        }
        let mut out = Vec::with_capacity(4);
        for multi in [false, true] {
            let (masks, scores) = self.decode(e, inside, outside, bbox, multi)?;
            for (i, s) in scores.iter().enumerate() {
                out.push(self.matte(e, &masks, i, *s)?);
            }
        }
        Ok(out)
    }

    /// The decoder's low-resolution logits `(1, n, 256, 256)` and how sure it is of each.
    fn decode(
        &self,
        e: &Embedded,
        inside: &[(f32, f32)],
        outside: &[(f32, f32)],
        bbox: Option<[f32; 4]>,
        multi: bool,
    ) -> Result<(Tensor, Vec<f32>), VisionError> {
        // picture pixels → the embedded picture's (the 1024 square's top-left)
        let (kw, kh) = (e.w as f32, e.h as f32);
        let to = |(x, y): &(f32, f32)| ((x * e.scale).clamp(0.0, kw), (y * e.scale).clamp(0.0, kh));
        let ok = |p: &&(f32, f32)| p.0.is_finite() && p.1.is_finite();
        let points: Vec<((f32, f32), bool)> =
            inside.iter().filter(ok).map(|p| (to(p), true)).chain(outside.iter().filter(ok).map(|p| (to(p), false))).take(MAX_POINTS).collect();
        let bbox = bbox.map(|b| {
            let ((x0, y0), (x1, y1)) = (to(&(b[0], b[1])), to(&(b[2], b[3])));
            [x0, y0, x1, y1]
        });
        let sparse = self.sparse.embed(&points, bbox, &self.dev)?;
        // (no mask given: the dense part is the "no mask" vector everywhere)
        let (_, dense) = self.prompts.forward(None, None, None).map_err(err)?;
        let pe = self.prompts.get_dense_pe().map_err(err)?;
        let (masks, iou) = self.decoder.forward(&e.emb, &pe, &sparse, &dense, multi).map_err(err)?;
        let scores: Vec<f32> = iou.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1()).map_err(err)?;
        Ok((masks, scores))
    }

    /// Answer `i` of the decoder's logits as a cut-out at the embedded picture's size.
    fn matte(&self, e: &Embedded, masks: &Tensor, i: usize, score: f32) -> Result<Matte, VisionError> {
        let low = masks.i((0, i)).map_err(err)?;
        // the 256 × 256 logits cover the 1024 square the picture sits in, top-left
        let side = IMAGE_SIZE;
        let up = low.unsqueeze(0).and_then(|t| t.unsqueeze(0)).and_then(|t| t.upsample_bilinear2d(side, side, false)).map_err(err)?;
        let up = up.i((0, 0, ..e.h, ..e.w)).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(err)?;
        // logits to coverage: a soft edge a logit wide either side of 0
        let alpha = up.iter().map(|v| if v.is_finite() { ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 }).collect();
        let score = if score.is_finite() { score.clamp(0.0, 1.0) } else { 0.0 };
        Ok(Matte { w: e.w, h: e.h, alpha, score })
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
        // a box round the figure alone: the answer for an unambiguous prompt is the figure
        let t = std::time::Instant::now();
        let a = seg.answers(&e, &[], &[], Some([255.0, 50.0, 385.0, 340.0])).unwrap();
        eprintln!("four answers to a box in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(a.len(), 4);
        let at = |m: &Matte, x: f32, y: f32| m.at((x * s) as usize, (y * s) as usize);
        assert!(at(&a[0], 320.0, 220.0) > 200 && at(&a[0], 320.0, 90.0) > 200, "the figure is in");
        assert!(at(&a[0], 100.0, 100.0) < 50 && at(&a[0], 560.0, 300.0) < 50, "the ground is out");
        assert!(seg.answers(&e, &[], &[], None).is_err());
        assert!(seg.answers(&e, &[], &[], Some([10.0, 10.0, 5.0, 5.0])).is_err(), "an empty box");
        assert!(seg.embed(&rgba[..100], w, h).is_err());
    }

    /// The image encoder's GPU passes give candle's embedding on the CPU. (Skips without
    /// MobileSAM or a GPU.)
    #[test]
    fn the_gpu_encoder_matches_candles() {
        let Some(dir) = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let (Ok(gpu), Ok(cpu)) = (Segmenter::load(&dir, true), Segmenter::load(&dir, false)) else { return };
        if gpu.encoder.is_none() {
            eprintln!("no GPU; skipping");
            return;
        }
        let (w, h) = (640usize, 360usize);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 3 + y * 5) % 251) as u8, ((x * 7) % 256) as u8, ((y * 11) % 256) as u8, 255]
            })
            .collect();
        let t = std::time::Instant::now();
        let c = cpu.embed(&rgba, w, h).unwrap();
        let tc = t.elapsed().as_secs_f64() * 1000.0;
        let _ = gpu.embed(&rgba, w, h).unwrap();
        let t = std::time::Instant::now();
        let g = gpu.embed(&rgba, w, h).unwrap();
        let tg = t.elapsed().as_secs_f64() * 1000.0;
        assert!(gpu.encoder_on(), "the GPU encoder ran");
        let a: Vec<f32> = c.emb.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = g.emb.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        if b.iter().any(|x| !x.is_finite()) {
            // where the values went bad
            let n = crate::sam_gpu::INPUT;
            let mut x = vec![0f32; 3 * n * n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 97) as f32 / 97.0 - 0.5) * 2.0;
            }
            panic!("the GPU's embedding is not finite; first bad pass: {:?}", gpu.encoder.as_ref().and_then(|e| e.first_bad(&x)));
        }
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let size = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        eprintln!("MobileSAM's encoder: candle on the CPU {tc:.0} ms, our GPU passes {tg:.0} ms; within {worst:.2e} (values up to {size:.2})");
        assert!(worst < 2e-3 * size.max(1.0), "{worst}");
    }
}
