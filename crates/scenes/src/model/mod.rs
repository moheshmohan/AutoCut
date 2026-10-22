//! The networks, on candle (feature `model`): TransNetV2 ([`transnet`]), Qwen3-VL
//! ([`qwen3vl`]), SigLIP 2 ([`siglip`]) and CLIP with the LAION aesthetic head ([`aesthetic`]).
//! On macOS they run on the GPU through candle's Metal backend (the CPU when there is none, or
//! with `FILMCRAFT_SCENES_CPU=1`), under the machine's `gpu` lock while a job runs
//! ([`gpu_lock`]).
//!
//! Weights are read without memory-mapping (which needs `unsafe`): safetensors files by a strict
//! header check and one seek per tensor ([`weights`]), TransNetV2's TensorFlow checkpoint by a
//! strict reader of its table and object graph ([`tfckpt`]).

pub mod aesthetic;
pub mod image;
pub mod qwen3vl;
pub mod siglip;
pub mod tfckpt;
pub mod transnet;
pub mod weights;

use candle_core::Device;

use crate::SceneError;

pub(crate) type R<T> = Result<T, SceneError>;

pub(crate) fn err(e: impl std::fmt::Display) -> SceneError {
    SceneError::Model(e.to_string())
}

/// The device the networks run on: Metal on macOS when there is a GPU (unless
/// `FILMCRAFT_SCENES_CPU` is set), else the CPU.
pub fn device() -> Device {
    #[cfg(target_os = "macos")]
    if std::env::var_os("FILMCRAFT_SCENES_CPU").is_none()
        && let Ok(d) = Device::new_metal(0)
    {
        return d;
    }
    Device::Cpu
}

/// How long a job waits for the GPU before giving up.
const GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// The GPU, ours while the guard lives: other processes' big models (narration, music) wait
/// for it and we for them. No lock on the CPU, in tests, or without a place to keep locks.
pub fn gpu_lock(dev: &Device, what: &str) -> R<Option<filmcraft_locks::Guard>> {
    use filmcraft_locks::{LockError, Resource};
    if cfg!(test) || dev.is_cpu() {
        return Ok(None);
    }
    match filmcraft_locks::acquire(Resource::Gpu, what, GPU_WAIT, &mut |_| {}) {
        Ok(g) => Ok(Some(g)),
        Err(e @ LockError::Busy(..)) => Err(SceneError::Model(format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)"))),
        Err(LockError::Io(_)) => Ok(None),
    }
}
