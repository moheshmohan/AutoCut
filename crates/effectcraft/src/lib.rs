//! EffectCraft compositions in process (PX4): `.ectemplate` packages (Essential Graphics
//! templates) read, filled with values and rendered by EffectCraft's own renderer, with footage
//! decoded by the host.
//!
//! EffectCraft (<https://github.com/storytold/effectcraft>, MIT OR Apache-2.0) is linked with
//! the cargo feature `effectcraft` (native builds only), pinned to [`PIN`]: its renderer,
//! effects, GPU compositor, expressions and text. Never its engine, media, host, export or
//! interchange crates: they bring upstream FilmCraft crates of the same names as ours, and
//! `cargo xtask layers` refuses them in `Cargo.lock`. Without the feature (the default, and every
//! wasm build) this crate is a stub: [`template_info`] and [`prepare`] answer [`Error::NotBuilt`],
//! while the package reader, the value checks and the colour helpers work the same either way.
//!
//! **Footage.** The engine (L4) decodes footage through [`HostFrames`], which this crate (L3)
//! turns into EffectCraft's `FootageSource`, converting colour both ways: AutoCut's frames are
//! premultiplied linear light, EffectCraft's premultiplied and sRGB-encoded ([`color`]). Media a
//! template carries (`media/<item>/<file>`) is written into a folder the caller names and decoded
//! the same way.
//!
//! **Templates.** A stored or deflated ZIP with `manifest.json` (format `effectcraft-template`,
//! version 1), `project.ecproj`, `media/<item>/<file>` and `poster.png`. Values go through each
//! control's own `coerce_json` and EffectCraft's `essential::with_overrides`, as an instance of the
//! template would show them. EffectCraft's engine also migrates old effects when it opens a
//! template; that step is not linked, so a template made by an old EffectCraft may render
//! differently here.
//!
//! **Never crash.** Everything in a template is hostile: the archive, the JSON (sizes capped,
//! nesting bounded by `serde_json`), the comp size (capped with the scale before anything is
//! allocated, as are the items' sizes and the frame count) and every value. Parsing, each frame
//! and each footage read run under `catch_unwind`; a panic inside EffectCraft becomes
//! [`Error::Internal`]. Out of memory cannot be caught, which is why the sizes are capped first.
//!
//! **Fonts.** EffectCraft's text crate embeds its own fonts (byte-identical to fonts AutoCut ships
//! and credits; an accepted duplicate). [`filmcraft_text::fonts::CRAFT_FONTS`] is added to its font
//! database once, without copying (empty is fine). A family it does not know makes it scan the
//! system's fonts on first use, so tests use its bundled Inter only.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use filmcraft_frame::VideoFrame;
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Map, Value};

pub mod color;
pub mod limits;
pub mod package;
pub mod params;

#[cfg(all(feature = "effectcraft", not(target_arch = "wasm32")))]
mod ec;
#[cfg(all(feature = "effectcraft", not(target_arch = "wasm32")))]
use ec as imp;

#[cfg(not(all(feature = "effectcraft", not(target_arch = "wasm32"))))]
mod stub;
#[cfg(not(all(feature = "effectcraft", not(target_arch = "wasm32"))))]
use stub as imp;

/// Builders of test templates (`.ectemplate` bytes made in code, never committed files).
#[cfg(all(feature = "effectcraft", not(target_arch = "wasm32"), any(test, feature = "test-support")))]
#[doc(hidden)]
pub mod testing;

#[cfg(test)]
mod tests;

/// The EffectCraft revision this crate is built against.
pub const PIN: &str = "1738360417403d8780aa3a4949bd6f6a49b6588d";
/// EffectCraft's workspace version at [`PIN`].
pub const EFFECTCRAFT_VERSION: &str = "0.24.1";

/// Why a template could not be read or rendered.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// This build does not link EffectCraft.
    #[error(
        "this build has no EffectCraft: it was built without the `effectcraft` feature (the packaged AutoCut app has it; `cargo build --features effectcraft`)"
    )]
    NotBuilt,
    /// The package, its manifest or its project.
    #[error("{0}")]
    Template(String),
    /// A value given for a control.
    #[error("{0}")]
    Values(String),
    /// A size, a count or a length over its cap.
    #[error("{0}")]
    TooLarge(String),
    /// Footage the template needs and the host cannot give.
    #[error("{0}")]
    Media(String),
    /// The render itself (writing a frame, a GPU failure the CPU could not stand in for).
    #[error("{0}")]
    Render(String),
    /// Cancelled.
    #[error("stopped")]
    Stopped,
    /// EffectCraft panicked (caught): what it was doing.
    #[error("EffectCraft failed while {0} (an internal error, caught; see the log)")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Whether this build links EffectCraft.
pub fn available() -> bool {
    cfg!(all(feature = "effectcraft", not(target_arch = "wasm32")))
}

/// What this build links: `{available, version, pin}`.
pub fn about() -> Value {
    serde_json::json!({"available": available(), "version": EFFECTCRAFT_VERSION, "pin": PIN})
}

/// What the host knows of a media file (for a Media Replacement value and the template's own
/// media).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostMedia {
    pub width: u32,
    pub height: u32,
    pub rate: FrameRate,
    pub duration: Tick,
    /// A still picture (one frame for any time).
    pub still: bool,
    pub has_alpha: bool,
    pub has_audio: bool,
}

/// The host's decoders, as the footage of a composition. Called from EffectCraft's worker
/// threads (and under `catch_unwind`): implementations must be thread-safe and must not hold a
/// lock the caller holds.
pub trait HostFrames: Send + Sync {
    /// What `path` holds, `None` when it cannot be read as a picture.
    fn probe(&self, path: &str) -> Option<HostMedia>;
    /// The picture of `path` at its own time `t` (any pixel format), `None` when there is none.
    fn frame(&self, path: &str, t: Tick) -> Option<Arc<VideoFrame>>;
}

/// No footage at all (a template without media, tests).
pub struct NoFrames;

impl HostFrames for NoFrames {
    fn probe(&self, _: &str) -> Option<HostMedia> {
        None
    }
    fn frame(&self, _: &str, _: Tick) -> Option<Arc<VideoFrame>> {
        None
    }
}

/// What a template is: its manifest, the comp (size, rate, duration), its controls (name, type,
/// default, range or options), the media it carries or needs, its poster and what reading its
/// project corrected. The template bytes are capped by the caller ([`limits::MAX_TEMPLATE_BYTES`]).
pub fn template_info(bytes: &[u8]) -> Result<Value> {
    imp::template_info(bytes)
}

/// A render asked of a template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderRequest {
    /// Control name (or id) → value; checked with [`params::values`] first.
    pub values: Map<String, Value>,
    /// The output size; `None` is the comp's. Its aspect ratio must be the comp's.
    pub size: Option<(u32, u32)>,
    /// The output rate; `None` is the comp's.
    pub fps: Option<FrameRate>,
    /// The part of the comp rendered (comp time): start and end, each `None` for the comp's own.
    pub range: (Option<Tick>, Option<Tick>),
    /// Composite on the GPU when there is one (the CPU stands in for a frame it declines or fails).
    pub gpu: bool,
}

/// A template checked and filled, ready to render: every check is done, nothing is rendered.
pub struct Prepared {
    inner: imp::Prepared,
}

/// Where [`Prepared::render`] hands each frame (its index, the frame); an error stops the render.
pub type Sink<'a> = dyn FnMut(u64, &Frame) -> std::result::Result<(), String> + Send + 'a;

/// One rendered frame: straight-alpha sRGB RGBA, 8 bits per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// How a render went.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderStats {
    pub frames: u64,
    pub gpu_frames: u64,
    pub cpu_frames: u64,
    /// The GPU adapter's name when the GPU was used.
    pub gpu: Option<String>,
    /// Why the GPU was not used, or stopped being used.
    pub gpu_note: Option<String>,
    pub seconds: f64,
}

/// Check `bytes` (an `.ectemplate`) against `req` and fill it: the package, the project, the
/// values, the sizes and the frame count, before anything is rendered. Media the template carries
/// is written into `media_dir` (made when needed) and probed through `host`, as is every file a
/// value names; a file that cannot be read is refused here.
pub fn prepare(bytes: &[u8], req: &RenderRequest, media_dir: &Path, host: &dyn HostFrames) -> Result<Prepared> {
    imp::prepare(bytes, req, media_dir, host).map(|inner| Prepared { inner })
}

impl Prepared {
    /// The output size.
    pub fn size(&self) -> (u32, u32) {
        self.inner.size()
    }
    /// The output rate.
    pub fn rate(&self) -> FrameRate {
        self.inner.rate()
    }
    /// How many frames [`Prepared::render`] makes.
    pub fn frames(&self) -> u64 {
        self.inner.frames()
    }
    /// What will be rendered: the template, the comp, the values as applied, the size, rate,
    /// range and frame count, and the notes (corrections, values that changed nothing).
    pub fn summary(&self) -> Value {
        self.inner.summary()
    }
    /// Render every frame in order, handing each to `sink` (frame index, frame); `sink` may stop
    /// the render with an error. `cancel` is read before each frame.
    pub fn render(&self, host: Arc<dyn HostFrames>, sink: &mut Sink<'_>, cancel: &AtomicBool) -> Result<RenderStats> {
        self.inner.render(host, sink, cancel)
    }
}
