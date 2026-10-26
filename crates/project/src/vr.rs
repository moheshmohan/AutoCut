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

use crate::{EffectInstance, ItemId, Param, ParamValue, Project, TrackItem};

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

/// Whether a `vr_reframe` effect shows the whole sphere (its `output` is the equirectangular
/// sphere) rather than a flat view.
pub fn shows_sphere(e: &EffectInstance) -> bool {
    matches!(e.param("output").map(|p| &p.value), Some(ParamValue::Choice(1)))
}

/// The frame a clip's virtual camera makes, if it has one.
pub fn reframe_size(item: &TrackItem) -> Option<(u32, u32)> {
    effect_size(reframe_of(item)?)
}

/// A clip's virtual camera, added first in its chain (the other effects work on what it sees)
/// making a `frame` picture when it has none; enabled. Motion is left as it is.
pub fn add_camera(item: &mut TrackItem, frame: (u32, u32)) -> Option<&mut EffectInstance> {
    if !item.effects.iter().any(|e| e.effect == REFRAME) {
        let mut e = crate::find_effect(REFRAME)?.instance();
        let side = |v: u32| ParamValue::Float(f64::from(v.clamp(MIN_SIDE, MAX_SIDE)));
        e.params.insert("width".into(), Param::new(side(frame.0)));
        e.params.insert("height".into(), Param::new(side(frame.1)));
        item.effects.insert(0, e);
    }
    let e = item.effects.iter_mut().find(|e| e.effect == REFRAME)?;
    e.enabled = true;
    Some(e)
}

/// The direction — yaw and pitch, degrees right of and up from forward — of the point (`x`, `y`,
/// 0..1 across and down) of a whole-sphere equirectangular picture: where a virtual camera looks
/// to centre what an analysis found there.
pub fn panorama_direction(x: f64, y: f64) -> (f64, f64) {
    let at = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 };
    ((at(x) - 0.5) * 360.0, (0.5 - at(y)) * 180.0)
}

impl Project {
    /// The 360° description of an item's media (Interpret Footage ▸ VR Properties; subclips their
    /// parent's), if it is 360°.
    pub fn vr_of(&self, item: ItemId) -> Option<&Spherical> {
        self.resolve_media(item).and_then(|(_, m, _)| m.interpret.vr.as_ref())
    }

    /// Whether a sequence is 360° video — a 2:1 frame showing the whole sphere: clips seen through
    /// whole-sphere virtual cameras, or whole-sphere monoscopic equirectangular media not reframed
    /// — so its export says so (Export ▸ Video is VR, on unless set).
    pub fn is_360_sequence(&self, seq: ItemId) -> bool {
        let Some(q) = self.sequence(seq) else { return false };
        if q.settings.width != q.settings.height.saturating_mul(2) {
            return false;
        }
        q.video_tracks.iter().flat_map(|t| &t.items).filter(|it| it.enabled).any(|it| match reframe_of(it) {
            Some(e) => shows_sphere(e),
            None => self
                .vr_of(it.item)
                .is_some_and(|v| v.is_equirect() && v.layout == filmcraft_media::spherical::StereoLayout::Mono && v.captured == [360.0, 180.0]),
        })
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

    /// [`Project::conformed_source_size`] of what a clip shows: its picture's size at its display
    /// aspect in pixels of `frame_par`.
    pub fn conformed_clip_size(&self, item: &TrackItem, frame_par: (u32, u32)) -> Option<(f64, f64)> {
        let size = self.clip_source_size(item)?;
        Some(crate::conformed_size(size, self.clip_source_par(item).unwrap_or(frame_par), frame_par))
    }

    /// The size of the picture a motion track was made on: its clip's (a reframed clip's camera
    /// frame), else its media's.
    pub fn track_picture_size(&self, t: &crate::MotionTrack) -> Option<(u32, u32)> {
        self.sequence(t.sequence).and_then(|q| q.find_item(t.clip)).and_then(|(_, it)| self.clip_source_size(it)).or_else(|| self.source_size(t.item))
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

    #[test]
    fn panorama_points_are_directions() {
        assert_eq!(panorama_direction(0.5, 0.5), (0.0, 0.0));
        assert_eq!(panorama_direction(0.75, 0.25), (90.0, 45.0));
        assert_eq!(panorama_direction(0.0, 1.0), (-180.0, -90.0));
        assert_eq!(panorama_direction(f64::NAN, 9.0), (0.0, -90.0));
    }
}
