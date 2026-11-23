//! The build without EffectCraft (the default, and every wasm build): the package can still be
//! read ([`crate::package`]), but nothing is filled or rendered.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use filmcraft_time::FrameRate;
use serde_json::Value;

use crate::{Error, HostFrames, RenderRequest, RenderStats, Result};

/// Never made: [`prepare`] always refuses.
pub(crate) enum Prepared {}

pub(crate) fn template_info(_: &[u8]) -> Result<Value> {
    Err(Error::NotBuilt)
}

pub(crate) fn prepare(_: &[u8], _: &RenderRequest, _: &Path, _: &dyn HostFrames) -> Result<Prepared> {
    Err(Error::NotBuilt)
}

impl Prepared {
    pub(crate) fn size(&self) -> (u32, u32) {
        match *self {}
    }
    pub(crate) fn rate(&self) -> FrameRate {
        match *self {}
    }
    pub(crate) fn frames(&self) -> u64 {
        match *self {}
    }
    pub(crate) fn summary(&self) -> Value {
        match *self {}
    }
    pub(crate) fn render(&self, _: Arc<dyn HostFrames>, _: &mut crate::Sink<'_>, _: &AtomicBool) -> Result<RenderStats> {
        match *self {}
    }
}
