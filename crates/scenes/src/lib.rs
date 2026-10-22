//! The scene index: what is in a clip, shot by shot, so editing can be automated and templates
//! can fill themselves. The engine's `media.analyzeScenes` drives it; the index itself
//! ([`filmcraft_project::scenes::SceneIndex`]) is stored with the project.
//!
//! - [`catalog`]: the downloadable models, pinned to a revision, SHA-256 and size, never bundled:
//!   TransNetV2 (shot boundaries, MIT), Qwen3-VL 2B / 4B / 8B Instruct (descriptions,
//!   Apache-2.0), SigLIP 2 base (embeddings and text search, Apache-2.0), CLIP ViT-L/14 with
//!   the LAION aesthetic predictor (aesthetic score, MIT).
//! - [`boundaries`]: shots from per-frame cut probabilities or detected cuts, and the keyframes
//!   to look at in each.
//! - [`quality`]: cheap picture measures — sharpness (variance of the Laplacian), exposure
//!   (mean and clipping), the camera's motion and shake (frame-to-frame global motion) — and
//!   the overall score.
//! - [`json`] and [`prompt`]: what the vision-language model is asked, and the defensive reading
//!   of its answer (balanced-JSON extraction, repair of a truncated answer, every field checked).
//! - [`search`]: ranking shots for a text query by image–text similarity, words in the
//!   descriptions and words spoken in the shot.
//! - [`best`]: the best shots for a purpose (hook, b-roll, reveal…).
//! - `model` (feature `model`): the networks on candle — TransNetV2 ported from its TensorFlow
//!   checkpoint, Qwen3-VL (vision encoder with DeepStack, interleaved M-RoPE text decoder,
//!   batched greedy decoding), SigLIP 2 and CLIP through `candle-transformers`.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod best;
pub mod boundaries;
pub mod catalog;
pub mod json;
pub mod prompt;
pub mod quality;
pub mod search;

#[cfg(feature = "model")]
pub mod model;

/// Why scene analysis failed.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SceneError {
    /// A model is not downloaded.
    #[error("{0} is not downloaded yet (scenes.downloadModel)")]
    NotInstalled(String),
    /// A model file is damaged or not the network this code implements, or computing failed.
    #[error("{0}")]
    Model(String),
    /// The input cannot be used.
    #[error("{0}")]
    Invalid(String),
    #[error("stopped")]
    Cancelled,
}

/// Whether this build has the networks (feature `model`).
pub fn available() -> bool {
    cfg!(feature = "model")
}
