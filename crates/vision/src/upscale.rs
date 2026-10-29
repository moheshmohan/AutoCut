//! Real-ESRGAN (BSD-3-Clause, [`crate::catalog::UPSCALER`]): a picture made four times as large
//! with its detail restored — blur, noise and compression's blocks taken out — by the compact
//! general model (`realesr-general-x4v3`, SRVGGNetCompact: 33 convolutions of 64 channels with
//! PReLUs, a pixel shuffle by 4 and the picture itself added back, enlarged by nearest
//! neighbour).
//!
//! On macOS the network runs as fused WGSL passes (`upscale_gpu`: candle's Metal convolution was
//! seven times slower), candle on the CPU being the reference and the fallback.
//!
//! A picture is worked on in tiles of [`TILE`] input pixels, each read with a margin of [`PAD`] —
//! as far as the 34 convolutions reach — so the tiles' seams are exactly what the whole picture
//! would give (each convolution pads with zeros at the picture's edges, as one tile there does).
//! `strength` below 1 mixes the network's answer with the picture enlarged smoothly (bilinear),
//! keeping some of its own grain.

use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, PReLU, VarBuilder};

use crate::{VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("upscaler: {e}"))
}

/// How many times as large the network makes a picture.
pub const FACTOR: usize = 4;
/// The tile's side, input pixels, and the margin read round it.
pub const TILE: usize = 512;
pub const PAD: usize = 34;
const FEAT: usize = 64;
const CONVS: usize = 32;
/// The largest picture taken, input pixels a side (its result is four times that).
pub const MAX_SIDE: usize = 4096;

struct Net {
    first: Conv2d,
    first_act: PReLU,
    body: Vec<(Conv2d, PReLU)>,
    last: Conv2d,
    dev: Device,
}

impl Net {
    fn load(path: &Path, dev: &Device) -> Result<Self, VisionError> {
        let vb = VarBuilder::from_pth_with_state(path, DType::F32, "params", dev).map_err(err)?;
        let build = || -> candle_core::Result<Self> {
            let b = vb.pp("body");
            let cfg = Conv2dConfig { padding: 1, ..Default::default() };
            let first = candle_nn::conv2d(3, FEAT, 3, cfg, b.pp("0"))?;
            let first_act = candle_nn::prelu(Some(FEAT), b.pp("1"))?;
            let body = (0..CONVS)
                .map(|i| {
                    Ok((candle_nn::conv2d(FEAT, FEAT, 3, cfg, b.pp((2 + 2 * i).to_string()))?, candle_nn::prelu(Some(FEAT), b.pp((3 + 2 * i).to_string()))?))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            let last = candle_nn::conv2d(FEAT, 3 * FACTOR * FACTOR, 3, cfg, b.pp((2 + 2 * CONVS).to_string()))?;
            Ok(Self { first, first_act, body, last, dev: dev.clone() })
        };
        build().map_err(err)
    }

    /// The network on a planar RGB tile (0..1), `(1, 3, h, w)` → `(1, 3, 4h, 4w)`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut y = self.first_act.forward(&self.first.forward(x)?)?;
        for (c, a) in &self.body {
            y = a.forward(&c.forward(&y)?)?;
        }
        let y = candle_nn::ops::pixel_shuffle(&self.last.forward(&y)?, FACTOR)?;
        let (_, _, h, w) = x.dims4()?;
        y + x.upsample_nearest2d(h * FACTOR, w * FACTOR)?
    }
}

impl Net {
    /// Each convolution's weights, bias and PReLU slopes, for the GPU.
    fn weights(&self) -> candle_core::Result<Vec<crate::upscale_gpu::LayerWeights>> {
        let v = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let layer = |c: &Conv2d, a: Option<&PReLU>, cin: usize, cout: usize| -> candle_core::Result<crate::upscale_gpu::LayerWeights> {
            Ok(crate::upscale_gpu::LayerWeights {
                cin,
                cout,
                weight: v(c.weight())?,
                bias: c.bias().map(v).transpose()?.unwrap_or_else(|| vec![0.0; cout]),
                slope: a.map(|a| v(a.weight())).transpose()?,
            })
        };
        let mut out = vec![layer(&self.first, Some(&self.first_act), 3, FEAT)?];
        for (c, a) in &self.body {
            out.push(layer(c, Some(a), FEAT, FEAT)?);
        }
        out.push(layer(&self.last, None, FEAT, 3 * FACTOR * FACTOR)?);
        Ok(out)
    }
}

/// The tiles of a `w` × `h` picture: each tile's place `(x, y, w, h)` and the part read for it
/// (the tile with up to [`PAD`] round it, inside the picture).
pub fn tiles(w: usize, h: usize, tile: usize, pad: usize) -> Vec<([usize; 4], [usize; 4])> {
    let tile = tile.max(1);
    let mut v = Vec::new();
    for y in (0..h).step_by(tile) {
        for x in (0..w).step_by(tile) {
            let (tw, th) = (tile.min(w - x), tile.min(h - y));
            let (rx, ry) = (x.saturating_sub(pad), y.saturating_sub(pad));
            let (rx1, ry1) = ((x + tw + pad).min(w), (y + th + pad).min(h));
            v.push(([x, y, tw, th], [rx, ry, rx1 - rx, ry1 - ry]));
        }
    }
    v
}

/// The upscaler, loaded.
pub struct Upscaler {
    /// candle on the CPU: the reference and the fallback.
    cpu: Net,
    gpu: Option<crate::upscale_gpu::GpuNet>,
    /// Set when the GPU failed a tile: the CPU from then on.
    gpu_failed: std::sync::atomic::AtomicBool,
    tile: usize,
}

impl Upscaler {
    /// Load the network from the models directory, on the GPU when `gpu` and there is one.
    pub fn load(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        if !catalog::upscaler_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let path = catalog::upscale_dir(models_dir).join(catalog::UPSCALER.name);
        let cpu = Net::load(&path, &Device::Cpu)?;
        let gpu = if gpu && cfg!(target_os = "macos") { cpu.weights().ok().and_then(|w| crate::upscale_gpu::GpuNet::new(&w)) } else { None };
        Ok(Self { cpu, gpu, gpu_failed: std::sync::atomic::AtomicBool::new(false), tile: TILE })
    }

    /// Work in tiles of `tile` input pixels (tests: the seams).
    pub fn with_tile(mut self, tile: usize) -> Self {
        self.tile = tile.clamp(16, 1024);
        self
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if self.gpu.is_some() && !self.gpu_failed.load(std::sync::atomic::Ordering::Relaxed) { "gpu" } else { "cpu" }
    }

    /// One tile through candle on the CPU (the reference).
    fn run_cpu(&self, px: &[f32], w: usize, h: usize) -> Result<Vec<f32>, VisionError> {
        let x = Tensor::from_vec(px.to_vec(), (1, 3, h, w), &self.cpu.dev).map_err(err)?;
        self.cpu.forward(&x).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(err)
    }

    /// One tile's RGB (planar, 0..1, `w` × `h`) through the network: `4w × 4h` planar RGB, on the
    /// GPU unless it failed (then on the CPU, from then on).
    fn run_tile(&self, px: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, VisionError> {
        use std::sync::atomic::Ordering;
        if let Some(g) = &self.gpu
            && !self.gpu_failed.load(Ordering::Relaxed)
        {
            match g.run(&px, w, h) {
                Ok(v) => return Ok(v),
                Err(_) => self.gpu_failed.store(true, Ordering::Relaxed),
            }
        }
        self.run_cpu(&px, w, h)
    }

    /// An RGBA picture (`w` × `h`, sRGB, straight alpha) four times as large: `(rgba, 4w, 4h)`.
    /// Alpha is enlarged smoothly; `strength` (0..1) mixes in the smooth enlargement of the colour
    /// too. `progress` hears each tile done (`done`, `of`) and stops the work when it says false.
    pub fn upscale(
        &self,
        rgba: &[u8],
        w: usize,
        h: usize,
        strength: f32,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<(Vec<u8>, usize, usize), VisionError> {
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) {
            return Err(VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture (at most {MAX_SIDE} a side)", rgba.len())));
        }
        let strength = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 1.0 };
        let (ow, oh) = (w * FACTOR, h * FACTOR);
        let mut out = smooth(rgba, w, h);
        let ts = tiles(w, h, self.tile, PAD);
        for (k, (t, r)) in ts.iter().enumerate() {
            let [rx, ry, rw, rh] = *r;
            let mut px = vec![0f32; 3 * rw * rh];
            for y in 0..rh {
                for x in 0..rw {
                    let i = ((ry + y) * w + rx + x) * 4;
                    for c in 0..3 {
                        px[c * rw * rh + y * rw + x] = rgba.get(i + c).map_or(0.0, |v| f32::from(*v) / 255.0);
                    }
                }
            }
            let up = self.run_tile(px, rw, rh)?;
            let (uw, uh) = (rw * FACTOR, rh * FACTOR);
            if up.len() != 3 * uw * uh {
                return Err(err(format!("{} values for a {uw} × {uh} tile", up.len())));
            }
            // the tile's own part, without its margin
            let [tx, ty, tw, th] = *t;
            let (ox, oy) = ((tx - rx) * FACTOR, (ty - ry) * FACTOR);
            for y in 0..th * FACTOR {
                for x in 0..tw * FACTOR {
                    let o = ((ty * FACTOR + y) * ow + tx * FACTOR + x) * 4;
                    for c in 0..3 {
                        let v = up.get(c * uw * uh + (oy + y) * uw + ox + x).copied().unwrap_or(0.0);
                        let v = if v.is_finite() { v.clamp(0.0, 1.0) * 255.0 } else { 0.0 };
                        if let Some(p) = out.get_mut(o + c) {
                            *p = (v * strength + f32::from(*p) * (1.0 - strength)).round() as u8;
                        }
                    }
                }
            }
            if !progress(k + 1, ts.len()) {
                return Err(VisionError::Invalid("stopped".into()));
            }
        }
        Ok((out, ow, oh))
    }
}

/// An RGBA picture enlarged four times bilinearly (centres aligned, edges held).
pub fn smooth(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (ow, oh) = (w * FACTOR, h * FACTOR);
    let mut out = vec![0u8; ow * oh * 4];
    let get = |x: usize, y: usize, c: usize| rgba.get((y.min(h - 1) * w + x.min(w - 1)) * 4 + c).map_or(0.0, |v| f32::from(*v));
    for y in 0..oh {
        let sy = ((y as f32 + 0.5) / FACTOR as f32 - 0.5).max(0.0);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        for x in 0..ow {
            let sx = ((x as f32 + 0.5) / FACTOR as f32 - 0.5).max(0.0);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            for c in 0..4 {
                let top = get(x0, y0, c) * (1.0 - fx) + get(x0 + 1, y0, c) * fx;
                let bottom = get(x0, y0 + 1, c) * (1.0 - fx) + get(x0 + 1, y0 + 1, c) * fx;
                out[(y * ow + x) * 4 + c] = (top * (1.0 - fy) + bottom * fy).round() as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_picture_once_with_margins_inside_it() {
        for (w, h, t) in [(1, 1, 256), (300, 200, 256), (513, 257, 128), (64, 1000, 16)] {
            let ts = tiles(w, h, t, PAD);
            let mut seen = vec![0u8; w * h];
            for ([x, y, tw, th], [rx, ry, rw, rh]) in &ts {
                assert!(rx <= x && ry <= y && rx + rw >= x + tw && ry + rh >= y + th && rx + rw <= w && ry + rh <= h);
                assert!(x - rx <= PAD && y - ry <= PAD);
                for yy in *y..y + th {
                    for xx in *x..x + tw {
                        seen[yy * w + xx] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|n| *n == 1), "{w} × {h}");
        }
        let s = smooth(&[0, 0, 0, 255, 255, 255, 255, 255], 2, 1);
        assert_eq!(s.len(), 8 * 4 * 4);
        assert_eq!(&s[..4], &[0, 0, 0, 255]);
        assert_eq!(&s[7 * 4..8 * 4], &[255, 255, 255, 255]);
    }

    /// A small drawn picture (soft gradients, a sharp edge, a blurred stripe pattern) as RGBA.
    fn picture(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let edge = if x > w / 2 { 200.0 } else { 40.0 };
                let stripes = 128.0 + 90.0 * ((x as f32 * 0.9).sin() * (y as f32 * 0.35).cos());
                let g = (x * 255 / w.max(1)) as f32;
                v[i..i + 4].copy_from_slice(&[edge as u8, stripes as u8, g as u8, 255]);
            }
        }
        v
    }

    /// The downloaded model: tiles give the whole picture's answer, an edge comes out sharper
    /// than enlarged smoothly, and strength 0 is the smooth enlargement. (Skips when it is not
    /// downloaded.)
    #[test]
    fn the_network_enlarges_sharply_and_its_tiles_join_without_seams() {
        let Some(dir) = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(up) = Upscaler::load(&dir, std::env::var_os("UPSCALE_CPU").is_none()) else {
            eprintln!("Real-ESRGAN not downloaded; skipping");
            return;
        };
        let (w, h) = (96usize, 80usize);
        let px = picture(w, h);
        let t = std::time::Instant::now();
        let (whole, ow, oh) = up.upscale(&px, w, h, 1.0, &mut |_, _| true).unwrap();
        eprintln!("Real-ESRGAN on the {}: {w} × {h} in {:.0} ms", up.device_name(), t.elapsed().as_secs_f64() * 1e3);
        assert_eq!((ow, oh), (384, 320));
        let up = up.with_tile(40);
        let (tiled, _, _) = up.upscale(&px, w, h, 1.0, &mut |_, _| true).unwrap();
        let worst = whole.iter().zip(&tiled).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(worst <= 1, "tiles differ from the whole by {worst} levels");
        // the edge at x = 48: steeper than the smooth enlargement's
        let soft = smooth(&px, w, h);
        let step = |img: &[u8]| (180..204).map(|x| i32::from(img[(160 * ow + x + 1) * 4]) - i32::from(img[(160 * ow + x) * 4])).max().unwrap();
        assert!(step(&whole) > step(&soft), "{} against {}", step(&whole), step(&soft));
        let (none, _, _) = up.upscale(&px, w, h, 0.0, &mut |_, _| true).unwrap();
        assert_eq!(none, soft);
        // stopped, hostile sizes
        assert!(up.upscale(&px, w, h, 1.0, &mut |_, _| false).is_err());
        assert!(up.upscale(&px[..10], w, h, 1.0, &mut |_, _| true).is_err());
        assert!(up.upscale(&px, 0, h, 1.0, &mut |_, _| true).is_err());
    }

    /// The GPU's passes against candle on the CPU, on a tile of the drawn picture.
    #[test]
    fn the_gpu_matches_the_cpu() {
        let Some(dir) = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(up) = Upscaler::load(&dir, true) else { return };
        let Some(g) = &up.gpu else {
            eprintln!("no GPU; skipping");
            return;
        };
        let (w, h) = (70usize, 45usize);
        let rgba = picture(w, h);
        let mut px = vec![0f32; 3 * w * h];
        for i in 0..w * h {
            for c in 0..3 {
                px[c * w * h + i] = f32::from(rgba[i * 4 + c]) / 255.0;
            }
        }
        let cpu = up.run_cpu(&px, w, h).unwrap();
        let gpu = g.run(&px, w, h).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("GPU against CPU: at most {worst:.2e}");
        assert!(worst < 1e-3, "{worst}");
        assert!(g.run(&px[..10], w, h).is_err());
    }

    /// The speed of a 640 × 480 picture (printed).
    #[test]
    #[ignore = "needs the model; prints timings"]
    fn upscale_speed() {
        let dir = std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join("Library/Application Support/FilmCraft/models");
        for gpu in [true, false] {
            let up = Upscaler::load(&dir, gpu).unwrap();
            let px = picture(640, 480);
            let _ = up.upscale(&px[..96 * 80 * 4], 96, 80, 1.0, &mut |_, _| true);
            let t = std::time::Instant::now();
            let _ = up.upscale(&px, 640, 480, 1.0, &mut |_, _| true).unwrap();
            eprintln!("640 × 480 on the {}: {:.0} ms", up.device_name(), t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
