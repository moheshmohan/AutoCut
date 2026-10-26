//! 360° clips. A media item is 360° when Interpret Footage ▸ VR Properties says so
//! ([`crate::Interpretation::vr`], filled from the file's metadata at import); a clip of it is
//! edited flat through a virtual camera, the `vr_reframe` effect: yaw, pitch, roll, field of view
//! and lens distance, keyframable, and the frame the camera makes (`width` × `height`, the
//! sequence frame by default).
//!
//! The virtual camera is the clip's picture: a reframed clip shows a `width` × `height` picture
//! of square pixels, which Motion, masks, effects, tracking and Auto Reframe see as its source
//! ([`crate::Project::clip_source_size`]). The pixels are made by `filmcraft_render::vr`.

use filmcraft_media::Spherical;

use crate::{EffectInstance, ItemId, ParamValue, Project, TrackItem};

/// The virtual camera's effect id.
pub const REFRAME: &str = "vr_reframe";
/// Smallest and largest side of the virtual camera's frame, pixels.
pub const MIN_SIDE: u32 = 16;
pub const MAX_SIDE: u32 = 16384;
/// The output choices of `vr_reframe`.
pub const OUTPUTS: &[&str] = &["Flat (Virtual Camera)", "Sphere (Equirectangular)"];

/// A clip's enabled virtual camera.
pub fn reframe_of(item: &TrackItem) -> Option<&EffectInstance> {
    item.effects.iter().find(|e| e.enabled && e.effect == REFRAME)
}

/// The frame a `vr_reframe` effect makes, `width` × `height` (each kept within
/// [`MIN_SIDE`]..=[`MAX_SIDE`]); `None` for another effect.
pub fn effect_size(e: &EffectInstance) -> Option<(u32, u32)> {
    if e.effect != REFRAME {
        return None;
    }
    let side = |id: &str, d: u32| -> u32 {
        match e.param(id).map(|p| &p.value) {
            Some(ParamValue::Float(v)) if v.is_finite() => v.round().clamp(f64::from(MIN_SIDE), f64::from(MAX_SIDE)) as u32,
            _ => d,
        }
    };
    Some((side("width", 1920), side("height", 1080)))
}

/// The frame a clip's virtual camera makes, if it has one.
pub fn reframe_size(item: &TrackItem) -> Option<(u32, u32)> {
    effect_size(reframe_of(item)?)
}

impl Project {
    /// The 360° description of an item's media (Interpret Footage ▸ VR Properties; subclips their
    /// parent's), if it is 360°.
    pub fn vr_of(&self, item: ItemId) -> Option<&Spherical> {
        self.resolve_media(item).and_then(|(_, m, _)| m.interpret.vr.as_ref())
    }

    /// The size of the picture a clip shows before Motion: its source's, or — a clip seen
    /// through a virtual camera — the camera's frame.
    pub fn clip_source_size(&self, item: &TrackItem) -> Option<(u32, u32)> {
        match reframe_size(item) {
            Some(s) if self.item(item.item).is_some() => Some(s),
            _ => self.source_size(item.item),
        }
    }

    /// The pixel aspect ratio of what a clip shows (square for a virtual camera's frame).
    pub fn clip_source_par(&self, item: &TrackItem) -> Option<(u32, u32)> {
        if reframe_of(item).is_some() { Some((1, 1)) } else { self.source_par(item.item) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_effect_is_defined_and_sizes_are_sane() {
        let def = crate::find_effect(REFRAME).expect("vr_reframe is in the catalogue");
        let mut e = def.instance();
        assert_eq!(effect_size(&e), Some((1920, 1080)));
        e.params.insert("width".into(), crate::Param::new(ParamValue::Float(1e12)));
        e.params.insert("height".into(), crate::Param::new(ParamValue::Float(f64::NAN)));
        assert_eq!(effect_size(&e), Some((MAX_SIDE, 1080)));
        e.params.insert("height".into(), crate::Param::new(ParamValue::Float(-5.0)));
        assert_eq!(effect_size(&e), Some((MAX_SIDE, MIN_SIDE)));
        assert!(effect_size(&crate::find_effect("gaussian_blur").unwrap().instance()).is_none());
        for id in ["yaw", "pitch", "roll", "fov", "distance"] {
            assert!(def.param(id).is_some_and(|p| p.animatable), "{id}");
        }
        assert!(def.param("width").is_some_and(|p| !p.animatable));
    }
}
