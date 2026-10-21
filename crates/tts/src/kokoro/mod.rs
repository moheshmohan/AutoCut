//! Kokoro-82M neural voices (feature `kokoro`): pure-Rust inference on candle — Metal on macOS
//! ([`device`]), the CPU elsewhere and as the fallback.
//!
//! Implemented from the papers the model card cites (StyleTTS 2, iSTFTNet; with ALBERT, HiFi-GAN,
//! the neural source-filter model and the Snake activation they build on), the released
//! `config.json` and the tensor names and shapes in the checkpoint. No reference code was read.
//! Weights (Apache-2.0, hexgrad) are downloaded on first use and never bundled.
//!
//! [`Kokoro::speak`] takes Kokoro phonemes (from `filmcraft-tts-text`), at most
//! [`MAX_PHONEMES`] per call, and a voice pack.

pub(crate) mod dsp;
pub(crate) mod layers;
mod model;
pub(crate) mod weights;

use std::path::Path;

use crate::TtsError;

pub use model::SAMPLE_RATE as KOKORO_RATE;

/// Longest phoneme string one forward pass takes (the position table has 512 rows, two of them
/// for the boundary tokens).
pub const MAX_PHONEMES: usize = 510;

/// A loaded Kokoro model.
pub struct Kokoro {
    /// On [`device`] (the GPU where there is one).
    model: model::Model,
    vocab: std::collections::HashMap<char, u32>,
    dir: std::path::PathBuf,
    /// The GPU failed once (an error, a panic, invalid samples): the CPU from then on.
    gpu_failed: std::sync::atomic::AtomicBool,
    /// The model on the CPU, loaded when the GPU failed.
    cpu: std::sync::Mutex<Option<std::sync::Arc<model::Model>>>,
}

/// The device Kokoro runs on: Metal on macOS when there is a GPU (`FILMCRAFT_KOKORO_CPU` keeps
/// the CPU), else the CPU.
pub fn device() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("FILMCRAFT_KOKORO_CPU").is_none()
            && let Ok(d) = candle_core::Device::new_metal(0)
        {
            return d;
        }
    }
    candle_core::Device::Cpu
}

/// A loaded voice pack.
pub struct VoicePack(Vec<f32>);

impl VoicePack {
    pub fn load(path: &Path) -> Result<VoicePack, TtsError> {
        weights::read_voice(path).map(VoicePack)
    }
}

impl Kokoro {
    /// Load `config.json` and `kokoro-v1_0.pth` from `dir`, on [`device`] (the GPU where there
    /// is one; weights it cannot hold leave the model on the CPU).
    pub fn load(dir: &Path) -> Result<Kokoro, TtsError> {
        Self::load_on(dir, &device())
    }

    /// [`Kokoro::load`] on `dev`.
    pub fn load_on(dir: &Path, dev: &candle_core::Device) -> Result<Kokoro, TtsError> {
        let cfg = weights::read_config(&dir.join("config.json"))?;
        let w = weights::Weights::load(&dir.join("kokoro-v1_0.pth"))?;
        let model = match w.to_device(dev).and_then(|g| model::Model::load(&g)) {
            Ok(m) => m,
            Err(_) => model::Model::load(&w)?,
        };
        Ok(Kokoro { model, vocab: cfg.vocab, dir: dir.to_path_buf(), gpu_failed: std::sync::atomic::AtomicBool::new(false), cpu: std::sync::Mutex::new(None) })
    }

    /// Whether speech is synthesized on the GPU now.
    pub fn on_gpu(&self) -> bool {
        !self.model.device().is_cpu() && !self.gpu_failed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The model on the CPU (loaded on first use).
    fn cpu_model(&self) -> Result<std::sync::Arc<model::Model>, TtsError> {
        let mut g = self.cpu.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(m) = g.as_ref() {
            return Ok(m.clone());
        }
        let w = weights::Weights::load(&self.dir.join("kokoro-v1_0.pth"))?;
        let m = std::sync::Arc::new(model::Model::load(&w)?);
        *g = Some(m.clone());
        Ok(m)
    }

    /// One pass of `model`: samples, all finite.
    fn run(model: &model::Model, ids: &[u32], voice: &VoicePack, speed: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
        let style = weights::style_for(&voice.0, ids.len(), model.device())?;
        let out = model.forward(ids, &style, speed, seed)?;
        if out.iter().any(|s| !s.is_finite()) {
            return Err(TtsError::Model("the model produced invalid samples".into()));
        }
        Ok(out)
    }

    /// Phoneme ids of `phonemes`; symbols outside the vocabulary are dropped.
    pub fn ids(&self, phonemes: &str) -> Vec<u32> {
        phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()).collect()
    }

    /// Speak `phonemes` with `voice` at `speed` (0.5–2). Deterministic for a given `seed`.
    pub fn speak(&self, phonemes: &str, voice: &VoicePack, speed: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
        if !(0.5..=2.0).contains(&speed) {
            return Err(TtsError::Invalid("speed must be between 0.5 and 2".into()));
        }
        let ids = self.ids(phonemes);
        if ids.is_empty() {
            return Err(TtsError::Empty);
        }
        if ids.len() > MAX_PHONEMES {
            return Err(TtsError::Invalid(format!("at most {MAX_PHONEMES} phonemes per pass")));
        }
        if self.model.device().is_cpu() {
            return Self::run(&self.model, &ids, voice, speed, seed);
        }
        // the GPU first; a failure or a panic there is spoken again on the CPU, which keeps the
        // voice from then on
        if self.on_gpu() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::run(&self.model, &ids, voice, speed, seed))) {
                Ok(Ok(out)) => return Ok(out),
                _ => self.gpu_failed.store(true, std::sync::atomic::Ordering::Relaxed),
            }
        }
        Self::run(&*self.cpu_model()?, &ids, voice, speed, seed)
    }
}

/// Tests with the real model run when `FILMCRAFT_KOKORO_DIR` names a directory holding
/// `config.json`, `kokoro-v1_0.pth` and `af_heart.pt` (the downloaded package); they skip otherwise.
#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Option<(Kokoro, VoicePack)> {
        let dir = std::path::PathBuf::from(std::env::var_os("FILMCRAFT_KOKORO_DIR")?);
        let voice = [dir.join("af_heart.pt"), dir.join("voices/af_heart.pt")].into_iter().find(|p| p.exists())?;
        Some((Kokoro::load(&dir).unwrap(), VoicePack::load(&voice).unwrap()))
    }

    #[test]
    fn speaks_deterministically_and_pace_changes_the_length() {
        let Some((k, v)) = model() else { return };
        let ph = "həlˈO wˈɜɹld.";
        let a = k.speak(ph, &v, 1.0, 1).unwrap();
        assert_eq!(a, k.speak(ph, &v, 1.0, 1).unwrap(), "deterministic");
        assert_eq!(a.len() % 600, 0, "whole frames");
        let secs = a.len() as f64 / f64::from(KOKORO_RATE);
        assert!((0.5..3.0).contains(&secs), "{secs}");
        let peak = a.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak > 0.05 && peak < 1.0, "{peak}");
        let fast = k.speak(ph, &v, 2.0, 1).unwrap();
        let ratio = fast.len() as f64 / a.len() as f64;
        assert!((0.35..0.7).contains(&ratio), "{ratio}");
    }

    /// On the GPU Kokoro says the same thing as on the CPU: the same length, the waveform within
    /// float noise (SNR printed with both times). Skipped without the model or a GPU.
    #[test]
    fn gpu_speaks_like_the_cpu() {
        let Some((cpu, v)) = model() else { return };
        let gpu_dev = device();
        if gpu_dev.is_cpu() {
            eprintln!("SKIPPED (Kokoro on the GPU): no GPU");
            return;
        }
        let dir = std::path::PathBuf::from(std::env::var_os("FILMCRAFT_KOKORO_DIR").unwrap());
        let cpu = Kokoro { model: model::Model::load(&weights::Weights::load(&dir.join("kokoro-v1_0.pth")).unwrap()).unwrap(), ..cpu };
        let gpu = Kokoro::load_on(&dir, &gpu_dev).unwrap();
        assert!(gpu.on_gpu() && !cpu.on_gpu());
        let ph = "ðɪs ɪz ˈɛvɹi ɹˈOb\u{251}t aɪv mˈɛt fɹʌm twˈɛnti θˈɜɹtin tə twˈɛnti twˈɛnti fˈIv.";
        let _ = gpu.speak(ph, &v, 1.0, 1).unwrap();
        let t0 = std::time::Instant::now();
        let a = cpu.speak(ph, &v, 1.0, 1).unwrap();
        let t1 = std::time::Instant::now();
        let b = gpu.speak(ph, &v, 1.0, 1).unwrap();
        let t2 = std::time::Instant::now();
        assert!(gpu.on_gpu(), "the GPU did not fail");
        assert_eq!(a.len(), b.len());
        let signal: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
        let noise: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        let secs = a.len() as f64 / f64::from(KOKORO_RATE);
        eprintln!("Kokoro {secs:.2}s of speech: CPU {:.2}s, GPU {:.2}s; GPU vs CPU SNR {snr:.1} dB", (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64());
        assert!(snr > 30.0, "SNR {snr}");
    }

    #[test]
    fn refuses_bad_input() {
        let Some((k, v)) = model() else { return };
        assert_eq!(k.speak("", &v, 1.0, 1).err(), Some(TtsError::Empty));
        assert_eq!(k.speak("日本語👍", &v, 1.0, 1).err(), Some(TtsError::Empty), "nothing in the vocabulary");
        assert!(k.speak("a", &v, 3.0, 1).is_err());
        assert!(k.speak("a", &v, f64::NAN, 1).is_err());
        assert!(k.speak(&"ə".repeat(MAX_PHONEMES + 1), &v, 1.0, 1).is_err());
        assert_eq!(k.ids("a日b"), k.ids("ab"));
    }
}
