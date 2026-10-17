//! Neural text to music for FilmCraft: **ACE-Step 1.5** (ACE Studio and StepFun, MIT) as an
//! optional, downloadable backend next to the procedural music engine.
//!
//! - [`catalog`]: the downloadable models, every file pinned to a revision and SHA-256, with their
//!   licences and sizes. Nothing is bundled.
//! - [`request`]: what a generation asks for (description, length, seed, optional tempo, key and
//!   metre), checked, and the conditioning text laid out as the official pipeline does.
//! - `model` (feature `model`): the networks in pure Rust on candle — the Qwen3-Embedding text
//!   encoder and its byte-level BPE tokenizer, the turbo DiT (lyric and timbre encoders, a 24-layer
//!   diffusion transformer with AdaLN, sliding-window and cross attention), the 8-step
//!   flow-matching sampler and the Oobleck VAE decoder to 48 kHz stereo. On macOS the networks run
//!   on the GPU through candle's Metal backend (linear weights stored in bf16, computed in f32),
//!   elsewhere on the CPU.
//!
//! Everything is deterministic: the same request and seed give the same audio on a device.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod catalog;
pub mod request;

#[cfg(feature = "model")]
pub mod model;

pub use request::{Request, SAMPLE_RATE};

/// Why a generation failed.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MusicError {
    /// The request cannot be served as asked (the message says why).
    #[error("{0}")]
    Invalid(String),
    /// The model's files are not (all) downloaded.
    #[error("the music model is not downloaded yet")]
    NotInstalled,
    /// A model file is damaged or the computation failed.
    #[error("music model: {0}")]
    Model(String),
    /// The caller asked to stop.
    #[error("cancelled")]
    Cancelled,
}

/// Generated music: stereo at [`SAMPLE_RATE`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Music {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

impl Music {
    pub fn seconds(&self) -> f64 {
        self.left.len() as f64 / f64::from(self.sample_rate.max(1))
    }

    /// Scale so the louder channel peaks at `db` dBFS (the official pipeline's normalisation,
    /// −1 dBFS by default). Silence stays silence.
    pub fn normalize_peak(&mut self, db: f32) {
        let peak = self.left.iter().chain(&self.right).fold(0f32, |m, x| if x.is_finite() { m.max(x.abs()) } else { m });
        if peak <= 1e-9 || !db.is_finite() {
            return;
        }
        let g = 10f32.powf(db / 20.0) / peak;
        for x in self.left.iter_mut().chain(self.right.iter_mut()) {
            *x = if x.is_finite() { *x * g } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_normalisation() {
        let mut m = Music { left: vec![0.0, 0.5, -2.0], right: vec![0.25, f32::NAN, 1.0], sample_rate: 48_000 };
        m.normalize_peak(-6.0);
        let peak = m.left.iter().chain(&m.right).fold(0f32, |a, x| a.max(x.abs()));
        assert!((20.0 * peak.log10() + 6.0).abs() < 1e-4);
        assert!(m.right.iter().all(|x| x.is_finite()));
        let mut z = Music { left: vec![0.0; 4], right: vec![0.0; 4], sample_rate: 48_000 };
        z.normalize_peak(-1.0);
        assert!(z.left.iter().all(|x| *x == 0.0));
        assert!((Music { left: vec![0.0; 48_000], right: vec![0.0; 48_000], sample_rate: 48_000 }.seconds() - 1.0).abs() < 1e-12);
    }
}
