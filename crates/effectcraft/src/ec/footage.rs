//! The host's frames as EffectCraft footage ([`effectcraft_render::FootageSource`]): decoded by
//! the host ([`HostFrames`]), converted from AutoCut's premultiplied linear light to EffectCraft's
//! premultiplied sRGB encoding ([`crate::color`]), and kept for a few frames (a still is read
//! once however many frames show it). Called from EffectCraft's workers: every read runs under
//! `catch_unwind`, and a read that fails gives no frame (EffectCraft then draws nothing there).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, PoisonError};

use effectcraft_project::{AlphaMode, Footage, FootageKind, ItemId};
use effectcraft_render::{FootageSource, Image};
use filmcraft_time::Tick;

use crate::HostFrames;
use crate::color;
use crate::limits::MAX_FOOTAGE_PIXELS;

/// Frames kept, and the most bytes they may hold.
const KEEP: usize = 16;
const KEEP_BYTES: usize = 1 << 30;

/// A host's frames as EffectCraft footage.
pub(crate) struct HostFootage {
    host: Arc<dyn HostFrames>,
    /// (file, frame) → pixels, the newest last.
    kept: Mutex<Vec<(String, i64, Arc<Image>)>>,
}

impl HostFootage {
    pub(crate) fn new(host: Arc<dyn HostFrames>) -> HostFootage {
        HostFootage { host, kept: Mutex::new(Vec::new()) }
    }

    /// The file and frame `f` shows at its time `t`, and the host time to read it at.
    fn locate(f: &Footage, t: effectcraft_time::Tick) -> Option<(String, i64, Tick)> {
        match f.kind {
            FootageKind::Still => Some((f.path.clone(), 0, Tick::ZERO)),
            FootageKind::Sequence => {
                let (n, _) = effectcraft_render::frame_position(t, f.frame_rate);
                let last = i64::try_from(f.sequence.len()).ok()?.checked_sub(1)?;
                let i = n.clamp(0, last);
                Some((f.sequence.get(usize::try_from(i).ok()?)?.clone(), i, Tick::ZERO))
            }
            FootageKind::Video => {
                let (n, _) = effectcraft_render::frame_position(t, f.frame_rate);
                Some((f.path.clone(), n, Tick(t.0.max(0))))
            }
            FootageKind::Audio | FootageKind::Data | FootageKind::Model => None,
        }
    }

    fn read(&self, f: &Footage, t: effectcraft_time::Tick) -> Option<Arc<Image>> {
        if !f.has_video || f.data.is_some() {
            return None;
        }
        let (path, n, at) = Self::locate(f, t)?;
        if path.is_empty() {
            return None;
        }
        {
            let kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((_, _, img)) = kept.iter().rev().find(|(p, i, _)| *p == path && *i == n) {
                return Some(img.clone());
            }
        }
        let frame = self.host.frame(&path, at)?;
        let (width, height, data) = color::host_to_encoded(&frame, f.alpha == AlphaMode::Ignore, MAX_FOOTAGE_PIXELS)?;
        let img = Arc::new(Image { width, height, data });
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        kept.push((path, n, img.clone()));
        let bytes = |k: &[(String, i64, Arc<Image>)]| k.iter().map(|(_, _, i)| i.data.len().saturating_mul(16)).fold(0usize, usize::saturating_add);
        while kept.len() > KEEP || (kept.len() > 1 && bytes(&kept) > KEEP_BYTES) {
            kept.remove(0);
        }
        Some(img)
    }
}

impl FootageSource for HostFootage {
    fn frame(&self, _item: ItemId, footage: &Footage, t: effectcraft_time::Tick) -> Option<Arc<Image>> {
        match catch_unwind(AssertUnwindSafe(|| self.read(footage, t))) {
            Ok(img) => img,
            Err(_) => {
                log::error!("effectcraft: reading a frame of {} failed inside (caught)", footage.path);
                None
            }
        }
    }
}
