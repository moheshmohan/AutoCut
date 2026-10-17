//! ACE-Step 1.5 turbo inference on candle (feature `model`).
//!
//! The text-to-music path of the official pipeline without the LM planner, for instrumental
//! music:
//!
//! 1. The description and metadata are laid out as [`crate::Request::text_prompt`] and tokenized
//!    ([`tokenizer`]); the Qwen3 text encoder (`qwen3`) gives their hidden states; the lyrics block
//!    (`[Instrumental]`) is tokenized and only embedded.
//! 2. The DiT's condition encoder (`dit`) packs lyrics, a timbre embedding of the silence latent
//!    (no reference audio) and the projected text into one sequence; the source latents are the
//!    silence latent, the chunk mask all ones.
//! 3. Eight Euler steps of the turbo flow-matching schedule (`shift` 3: t = 1, 0.95, 0.9, 0.83,
//!    0.75, 0.64, 0.5, 0.3, then straight to t = 0) from seeded Gaussian noise.
//! 4. The Oobleck VAE (`vae`) decodes the latents to 48 kHz stereo in windows.
//!
//! The three networks are loaded one after the other and dropped when done, so the peak is the
//! DiT alone. On macOS they run on the GPU through candle's Metal backend while holding the
//! machine-wide GPU lock (`filmcraft-locks`), their linear weights stored in bf16 as released
//! (≈ 4.5 GB) and widened to f32 for each product — exactly, so the arithmetic is the reference's
//! f32 (bf16 *activations* drifted 25 % from it in eight steps). On the CPU the weights are f32
//! (≈ 9 GB). `FILMCRAFT_ACESTEP_CPU` forces the CPU, `FILMCRAFT_ACESTEP_F32` keeps f32 weights on
//! the GPU (faster, twice the memory).

mod config;
mod dit;
mod nn;
mod qwen3;
mod silence;
pub mod tokenizer;
mod vae;
mod weights;

#[cfg(test)]
mod oracle_tests;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::catalog;
use crate::{Music, MusicError, Request};

pub(crate) type R<T> = Result<T, MusicError>;

pub(crate) fn err(e: impl std::fmt::Display) -> MusicError {
    MusicError::Model(e.to_string())
}

/// Longest text prompt and lyrics, tokens (the reference's truncation).
pub const MAX_TEXT_TOKENS: usize = 256;
pub const MAX_LYRIC_TOKENS: usize = 2048;

/// The turbo model's timestep schedules (`SHIFT_TIMESTEPS` of the reference, 8 steps; the last
/// step jumps to t = 0).
const SCHEDULES: [[f64; 8]; 3] = [
    [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
];

/// The timesteps for `shift` (1, 2 or 3), as the reference holds them (f32).
pub(crate) fn schedule(shift: u32) -> R<Vec<f32>> {
    let i = usize::try_from(shift)
        .ok()
        .and_then(|s| s.checked_sub(1))
        .filter(|i| *i < SCHEDULES.len())
        .ok_or_else(|| MusicError::Invalid("the shift must be 1, 2 or 3".into()))?;
    Ok(SCHEDULES.get(i).map(|s| s.iter().map(|t| *t as f32).collect()).unwrap_or_default())
}

/// The device inference runs on: Metal on macOS (when there is a GPU), else the CPU.
/// `FILMCRAFT_ACESTEP_CPU` forces the CPU.
pub fn device() -> Device {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("FILMCRAFT_ACESTEP_CPU").is_none()
            && let Ok(d) = Device::new_metal(0)
        {
            return d;
        }
    }
    Device::Cpu
}

/// A deterministic standard-normal source: SplitMix64 and the Box–Muller transform.
pub(crate) struct Noise {
    state: u64,
    spare: Option<f64>,
}

impl Noise {
    pub(crate) fn new(seed: u64) -> Noise {
        Noise { state: seed ^ 0x9e37_79b9_7f4a_7c15, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        // u1 in (0, 1], u2 in [0, 1)
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        self.spare = Some(r * a.sin());
        r * a.cos()
    }

    pub(crate) fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal() as f32).collect()
    }
}

/// How far a generation is: `stage` (`"text"`, `"load"`, `"diffusion"`, `"decode"`), `done` of
/// `total` steps within it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub stage: &'static str,
    pub done: usize,
    pub total: usize,
}

impl Progress {
    /// The whole generation's fraction done (the stages weighted by their usual share of time).
    pub fn fraction(&self) -> f64 {
        let (from, span) = match self.stage {
            "text" => (0.0, 0.05),
            "load" => (0.05, 0.10),
            "diffusion" => (0.15, 0.55),
            _ => (0.70, 0.30),
        };
        from + span * self.done as f64 / self.total.max(1) as f64
    }
}

/// A downloaded ACE-Step package: configs, tokenizer and silence latent read; the networks are
/// loaded for each generation and dropped after it.
pub struct AceStep {
    dir: PathBuf,
    dev: Device,
    wdtype: DType,
    dit_cfg: config::DitConfig,
    text_cfg: config::TextConfig,
    vae_cfg: config::VaeConfig,
    tokenizer: tokenizer::Tokenizer,
    silence: silence::Silence,
}

/// What a generation made besides the music (read by the oracle tests).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Detail {
    pub latents: Vec<f32>,
    pub frames: usize,
}

impl AceStep {
    /// Read a package directory (the files of [`catalog::Package`]) for the default device.
    pub fn load(dir: &Path) -> Result<AceStep, MusicError> {
        AceStep::load_on(dir, device())
    }

    pub fn load_on(dir: &Path, dev: Device) -> Result<AceStep, MusicError> {
        let need = [
            catalog::DIT_CONFIG,
            catalog::DIT_WEIGHTS,
            catalog::SILENCE_LATENT,
            catalog::VAE_CONFIG,
            catalog::VAE_WEIGHTS,
            catalog::TEXT_CONFIG,
            catalog::TEXT_TOKENIZER,
            catalog::TEXT_WEIGHTS,
        ];
        if need.iter().any(|f| !dir.join(f).is_file()) {
            return Err(MusicError::NotInstalled);
        }
        let dit_cfg = config::load_dit(&dir.join(catalog::DIT_CONFIG))?;
        let text_cfg = config::load_text(&dir.join(catalog::TEXT_CONFIG))?;
        let vae_cfg = config::load_vae(&dir.join(catalog::VAE_CONFIG))?;
        if text_cfg.shape.hidden != dit_cfg.text_dim {
            return Err(err("the text encoder does not match the DiT"));
        }
        if vae_cfg.latent != dit_cfg.latent {
            return Err(err("the VAE does not match the DiT"));
        }
        if !dit_cfg.turbo {
            return Err(err("only the turbo DiT is supported (8 steps, no guidance)"));
        }
        let tokenizer = tokenizer::Tokenizer::load(&dir.join(catalog::TEXT_TOKENIZER))?;
        let silence = silence::Silence::load(&dir.join(catalog::SILENCE_LATENT), dit_cfg.latent)?;
        let f32_weights = dev.is_cpu() || std::env::var_os("FILMCRAFT_ACESTEP_F32").is_some();
        let wdtype = if f32_weights { DType::F32 } else { DType::BF16 };
        Ok(AceStep { dir: dir.to_path_buf(), dev, wdtype, dit_cfg, text_cfg, vae_cfg, tokenizer, silence })
    }

    /// Whether inference runs on the GPU.
    pub fn on_gpu(&self) -> bool {
        !self.dev.is_cpu()
    }

    /// Generate the music `req` describes. `progress` is told how far it is and returns false to
    /// stop (the generation then fails with [`MusicError::Cancelled`]).
    pub fn generate(&self, req: &Request, progress: &mut dyn FnMut(Progress) -> bool) -> Result<Music, MusicError> {
        req.check()?;
        let _gpu = self.gpu_lock()?;
        let frames = req.frames();
        let noise = Noise::new(req.seed).take(frames.saturating_mul(self.dit_cfg.latent));
        self.generate_with(req, &noise, progress).map(|(m, _)| m)
    }

    /// The GPU, ours while the guard lives (no lock on the CPU, in tests, or without a place to
    /// keep locks).
    fn gpu_lock(&self) -> Result<Option<filmcraft_locks::Guard>, MusicError> {
        use filmcraft_locks::{LockError, Resource};
        const WAIT: std::time::Duration = std::time::Duration::from_secs(10 * 60);
        if cfg!(test) || self.dev.is_cpu() {
            return Ok(None);
        }
        match filmcraft_locks::acquire(Resource::Gpu, "ACE-Step music generation", WAIT, &mut |_| {}) {
            Ok(g) => Ok(Some(g)),
            Err(e @ LockError::Busy(..)) => Err(MusicError::Model(format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)"))),
            Err(LockError::Io(_)) => Ok(None),
        }
    }

    /// The token ids of the text prompt and the lyrics block.
    pub(crate) fn tokens(&self, req: &Request) -> R<(Vec<u32>, Vec<u32>)> {
        Ok((self.tokenizer.encode(&req.text_prompt(), MAX_TEXT_TOKENS)?, self.tokenizer.encode(req.lyrics_prompt(), MAX_LYRIC_TOKENS)?))
    }

    /// Text hidden states and lyric embeddings (the text encoder is loaded and dropped here).
    pub(crate) fn text(&self, text_ids: &[u32], lyric_ids: &[u32]) -> R<(Tensor, Tensor)> {
        let mut te = qwen3::TextEncoder::load(&self.dir.join(catalog::TEXT_WEIGHTS), &self.text_cfg, &self.dev, self.wdtype)?;
        let hidden = te.encode(text_ids)?;
        let lyrics = te.embed(lyric_ids)?;
        Ok((hidden, lyrics))
    }

    pub(crate) fn load_dit(&self) -> R<dit::Dit> {
        dit::Dit::load(&self.dir.join(catalog::DIT_WEIGHTS), &self.dit_cfg, &self.dev, self.wdtype)
    }

    pub(crate) fn load_vae(&self) -> R<vae::Vae> {
        vae::Vae::load(&self.dir.join(catalog::VAE_WEIGHTS), &self.vae_cfg, &self.dev)
    }

    /// The context `[1, frames, 2 · latent]`: silence latents and an all-ones chunk mask.
    pub(crate) fn context(&self, frames: usize) -> R<Tensor> {
        let c = self.dit_cfg.latent;
        let src = self.silence.frames(frames);
        let mut ctx = Vec::with_capacity(frames.saturating_mul(2 * c));
        for t in 0..frames {
            ctx.extend_from_slice(src.get(t * c..(t + 1) * c).unwrap_or(&[]));
            ctx.extend(std::iter::repeat_n(1f32, c));
        }
        Tensor::from_vec(ctx, (1, frames, 2 * c), &self.dev).map_err(err)
    }

    /// The timbre reference `[1, 750, latent]`: the start of the silence latent.
    pub(crate) fn timbre_reference(&self) -> R<Tensor> {
        let c = self.dit_cfg.latent;
        Tensor::from_vec(self.silence.frames(silence::TIMBRE_FRAMES), (1, silence::TIMBRE_FRAMES, c), &self.dev).map_err(err)
    }

    /// The whole generation from given initial noise (`frames × latent`).
    pub(crate) fn generate_with(&self, req: &Request, noise: &[f32], progress: &mut dyn FnMut(Progress) -> bool) -> R<(Music, Detail)> {
        req.check()?;
        let frames = req.frames();
        let c = self.dit_cfg.latent;
        if noise.len() != frames.saturating_mul(c) {
            return Err(err("the noise does not match the length"));
        }
        let go = |p: &mut dyn FnMut(Progress) -> bool, stage: &'static str, done: usize, total: usize| -> R<()> {
            if p(Progress { stage, done, total }) { Ok(()) } else { Err(MusicError::Cancelled) }
        };
        go(progress, "text", 0, 1)?;
        let (text_ids, lyric_ids) = self.tokens(req)?;
        let (text_hidden, lyric_embeds) = self.text(&text_ids, &lyric_ids)?;
        go(progress, "load", 0, 1)?;
        let dit = self.load_dit()?;
        let enc = dit.encode(&text_hidden, &lyric_embeds, &self.timbre_reference()?)?;
        let cond = dit.condition(&enc)?;
        drop((text_hidden, lyric_embeds, enc));
        let ctx = self.context(frames)?;
        let ts = schedule(req.shift)?;
        let mut xt = Tensor::from_slice(noise, (1, frames, c), &self.dev).map_err(err)?;
        for (i, &t) in ts.iter().enumerate() {
            go(progress, "diffusion", i, ts.len())?;
            let v = dit.velocity(&xt, t, &ctx, &cond)?;
            // Euler steps towards t = 0; the last one straight to x₀ = x_t − v·t
            let dt = match ts.get(i + 1) {
                Some(&next) => (f64::from(t) - f64::from(next)) as f32,
                None => t,
            };
            xt = (xt - (v * f64::from(dt)).map_err(err)?).map_err(err)?;
        }
        drop((dit, cond, ctx));
        let latents = xt.to_device(&Device::Cpu).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
        if latents.iter().any(|x| !x.is_finite()) {
            return Err(err("the diffusion produced non-finite latents"));
        }
        go(progress, "decode", 0, 1)?;
        let vae = self.load_vae()?;
        let mut cancelled = false;
        let (mut left, mut right) = vae
            .decode(&latents, frames, &mut |done, total| {
                let ok = progress(Progress { stage: "decode", done, total });
                cancelled |= !ok;
                ok
            })
            .map_err(|e| if cancelled { MusicError::Cancelled } else { e })?;
        let n = req.samples().min(left.len());
        left.truncate(n);
        right.truncate(n);
        // as the reference: scale down only when the decoder overshoots full scale
        let peak = left.iter().chain(&right).fold(0f32, |m, x| m.max(x.abs()));
        if !peak.is_finite() {
            return Err(err("the decoder produced non-finite audio"));
        }
        if peak > 1.0 {
            for x in left.iter_mut().chain(right.iter_mut()) {
                *x /= peak;
            }
        }
        Ok((Music { left, right, sample_rate: crate::SAMPLE_RATE }, Detail { latents, frames }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_seeded_and_normal() {
        let a = Noise::new(7).take(20_000);
        assert_eq!(a, Noise::new(7).take(20_000));
        assert_ne!(a[..64], Noise::new(8).take(64)[..]);
        let mean = a.iter().map(|x| f64::from(*x)).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "{mean} {var}");
        assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn schedules_are_the_reference_tables() {
        for s in 1..=3 {
            let t = schedule(s).unwrap();
            assert_eq!(t.len(), 8);
            assert_eq!(t[0], 1.0);
            assert!(t.windows(2).all(|w| w[1] < w[0]));
            // t = shift·u / (1 + (shift − 1)·u) on u = 1, 7/8, …, 1/8
            for (i, x) in t.iter().enumerate() {
                let u = 1.0 - i as f64 / 8.0;
                let want = f64::from(s) * u / (1.0 + f64::from(s - 1) * u);
                assert!((f64::from(*x) - want).abs() < 1e-6, "shift {s} step {i}");
            }
        }
        assert!(schedule(0).is_err() && schedule(4).is_err());
    }

    #[test]
    fn progress_fractions_rise() {
        let ps = [
            Progress { stage: "text", done: 0, total: 1 },
            Progress { stage: "load", done: 0, total: 1 },
            Progress { stage: "diffusion", done: 4, total: 8 },
            Progress { stage: "decode", done: 1, total: 2 },
            Progress { stage: "decode", done: 2, total: 2 },
        ];
        let f: Vec<f64> = ps.iter().map(Progress::fraction).collect();
        assert!(f.windows(2).all(|w| w[1] > w[0]), "{f:?}");
        assert!((f[4] - 1.0).abs() < 1e-12);
        assert_eq!(Progress { stage: "decode", done: 0, total: 0 }.fraction(), 0.7);
    }

    /// A tiny random package on disk: configs, weights, tokenizer and silence latent.
    pub(crate) fn fake_package(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let (dit, _) = dit::tests::fake(dir);
        std::fs::rename(dit, dir.join(catalog::DIT_WEIGHTS)).unwrap();
        let (te, _) = qwen3::tests::fake(dir);
        std::fs::rename(te, dir.join(catalog::TEXT_WEIGHTS)).unwrap();
        let (v, _) = vae::tests::fake(dir);
        std::fs::rename(v, dir.join(catalog::VAE_WEIGHTS)).unwrap();
        std::fs::write(dir.join(catalog::DIT_CONFIG), config::tests::tiny_dit().to_string()).unwrap();
        std::fs::write(dir.join(catalog::TEXT_CONFIG), config::tests::tiny_text().to_string()).unwrap();
        std::fs::write(dir.join(catalog::VAE_CONFIG), config::tests::tiny_vae().to_string()).unwrap();
        std::fs::write(dir.join(catalog::TEXT_TOKENIZER), tokenizer::tests::tiny_json()).unwrap();
        silence::tests::write(&dir.join(catalog::SILENCE_LATENT), 4, 800, "little");
    }

    #[test]
    fn a_tiny_package_generates_deterministic_music() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-pkg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        fake_package(&d);
        let m = AceStep::load_on(&d, Device::Cpu).unwrap();
        assert!(!m.on_gpu());
        let req = Request::new("calm piano", 6.0, 11);
        let mut stages = Vec::new();
        let a = m
            .generate(&req, &mut |p| {
                stages.push(p.stage);
                true
            })
            .unwrap();
        assert_eq!(a.sample_rate, 48_000);
        assert_eq!(a.left.len(), 6 * 48_000);
        assert_eq!(a.right.len(), a.left.len());
        assert!(a.left.iter().chain(&a.right).all(|x| x.is_finite() && x.abs() <= 1.0));
        for s in ["text", "load", "diffusion", "decode"] {
            assert!(stages.contains(&s), "{s}");
        }
        let b = m.generate(&req, &mut |_| true).unwrap();
        assert_eq!(a, b, "same request and seed, same music");
        let c = m.generate(&Request { seed: 12, ..req.clone() }, &mut |_| true).unwrap();
        assert_ne!(a, c, "another seed, other music");
        // cancelling at any stage stops with Cancelled
        for stop in ["text", "load", "diffusion", "decode"] {
            let r = m.generate(&req, &mut |p| p.stage != stop);
            assert_eq!(r.err(), Some(MusicError::Cancelled), "{stop}");
        }
        // hostile requests are refused before any work
        assert!(matches!(m.generate(&Request { seconds: f64::NAN, ..req.clone() }, &mut |_| true), Err(MusicError::Invalid(_))));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn incomplete_or_mismatched_packages_are_refused() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        assert_eq!(AceStep::load_on(&d, Device::Cpu).err(), Some(MusicError::NotInstalled));
        fake_package(&d);
        std::fs::remove_file(d.join(catalog::VAE_WEIGHTS)).unwrap();
        assert_eq!(AceStep::load_on(&d, Device::Cpu).err(), Some(MusicError::NotInstalled));
        fake_package(&d);
        let mut v = config::tests::tiny_dit();
        v["is_turbo"] = serde_json::json!(false);
        std::fs::write(d.join(catalog::DIT_CONFIG), v.to_string()).unwrap();
        assert!(AceStep::load_on(&d, Device::Cpu).is_err(), "base/SFT models need guidance this port lacks");
        let mut v = config::tests::tiny_dit();
        v["text_hidden_dim"] = serde_json::json!(24);
        std::fs::write(d.join(catalog::DIT_CONFIG), v.to_string()).unwrap();
        assert!(AceStep::load_on(&d, Device::Cpu).is_err());
        // damaged weights fail the generation, not the process
        fake_package(&d);
        let st = d.join(catalog::DIT_WEIGHTS);
        let mut b = std::fs::read(&st).unwrap();
        b.truncate(b.len() / 2);
        std::fs::write(&st, b).unwrap();
        let m = AceStep::load_on(&d, Device::Cpu).unwrap();
        assert!(m.generate(&Request::new("x", 5.0, 1), &mut |_| true).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
