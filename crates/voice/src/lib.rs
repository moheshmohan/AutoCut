//! Voice isolation for FilmCraft: **DeepFilterNet 3** (Schröter et al.; MIT OR Apache-2.0) in
//! plain Rust, the neural model behind the Voice Isolation and Enhance Voice audio effects
//! (`filmcraft_audio_dsp::effects::voice`).
//!
//! - [`catalog`]: the model archive (8 MB), pinned to a commit of the DeepFilterNet repository
//!   and its SHA-256; downloaded after the user agrees (`audio.downloadVoiceModel`), never bundled.
//! - [`archive`]: a strict `.tar.gz` reader (gzip CRC and length, tar header checksums, caps).
//! - [`weights`]: the weights out of the archive's ONNX files (through `filmcraft-vision`'s strict
//!   ONNX reader), every layer checked against what [`net`] implements.
//! - [`net`]: the network, frame by frame. No ML framework: the layers are a few small matrix
//!   products a frame (≈3.3 M multiply-adds per 10 ms), written as plain loops the compiler
//!   vectorises; DeepFilterNet's own runtime (libDF on `tract`) compiles hand-written assembly with
//!   a C toolchain (`tract-linalg`), which FilmCraft's pure-Rust rule keeps out.
//! - [`Df3`]: the model (`VoiceModel`) the effects run, one state per stream.
//!
//! Hosts call [`install_loader`] once with the models directory: the first effect that needs the
//! model loads it (≈0.1 s) and every later one shares it; without the download, the effects run
//! their DSP fallback.
//!
//! DeepFilterNet is © 2021 Hendrik Schröter, dual-licensed MIT / Apache-2.0
//! (<https://github.com/Rikorose/DeepFilterNet>); [`net`] and [`df3`] follow its published model
//! and its streaming implementation (libDF's `tract.rs` and `lib.rs`).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod archive;
pub mod catalog;
pub mod df3;
pub mod net;
pub mod weights;

pub use df3::{Df3, Df3Stream};

/// Why the voice model is not available.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum VoiceError {
    /// The model is not downloaded.
    #[error("the voice isolation model is not downloaded (audio.downloadVoiceModel)")]
    NotInstalled,
    /// The file is damaged or not the network this code implements.
    #[error("voice model: {0}")]
    Model(String),
    #[error("{0}")]
    Io(String),
}

/// Load the model downloaded into `models_dir`.
pub fn load(models_dir: &Path) -> Result<Df3, VoiceError> {
    if !catalog::installed(models_dir) {
        return Err(VoiceError::NotInstalled);
    }
    let path = catalog::archive_path(models_dir);
    let bytes = std::fs::read(&path).map_err(|e| VoiceError::Io(format!("{}: {e}", path.display())))?;
    if bytes.len() as u64 != catalog::ARCHIVE.size {
        return Err(VoiceError::Model(format!("{} is not the pinned file", path.display())));
    }
    Df3::from_archive(&bytes)
}

/// Make the audio effects load the model from `models_dir` the first time one needs it.
pub fn install_loader(models_dir: PathBuf) {
    filmcraft_audio_dsp::effects::voice::set_voice_model_loader(move || {
        load(&models_dir).ok().map(|m| Arc::new(m) as Arc<dyn filmcraft_audio_dsp::effects::voice::VoiceModel>)
    });
}

/// Load the model now and install it for the effects (after a download); the error says why not.
pub fn install_now(models_dir: &Path) -> Result<(), VoiceError> {
    let m = load(models_dir)?;
    filmcraft_audio_dsp::effects::voice::install_voice_model(Some(Arc::new(m)));
    Ok(())
}

#[cfg(test)]
mod tests;
