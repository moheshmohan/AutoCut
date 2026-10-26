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

/// The field of view across (degrees) a virtual camera making a frame `aspect` (width / height)
/// wide sees by default: the same coverage whatever the frame's shape — 100° across (67° down) in
/// 16:9, 68° across (100° down) in 9:16, 84° in a square — so a person near the camera fits a
/// vertical frame and a landscape one alike. (Coverage is kept as tan(across/2) · tan(down/2).)
pub fn natural_fov(aspect: f64) -> f64 {
    let a = if aspect.is_finite() && aspect > 0.0 { aspect.clamp(0.1, 10.0) } else { 16.0 / 9.0 };
    let wide = 50.0f64.to_radians().tan();
    let c = wide * wide / (16.0 / 9.0);
    (2.0 * (c * a).sqrt().atan().to_degrees()).clamp(40.0, 120.0)
}

/// The tallest subject framed whole, degrees: a taller box (someone right by the camera, a
/// selfie) is framed by its upper part — the face — instead ([`subject_aim`]).
pub const MAX_SUBJECT_DEGREES: f64 = 75.0;
/// The widest a view's long side gets, degrees: past it a little lens distance takes in the rest
/// instead of stretching the edges.
pub const MAX_VIEW_DEGREES: f64 = 110.0;

/// The framing of a subject `size` degrees across and down (its box on the sphere; a box taller
/// than [`MAX_SUBJECT_DEGREES`] counts as that tall) in a frame `aspect` wide: the field of view
/// across and the lens distance that show it — about 65 % of the frame's height and at most 80 %
/// of its width — never narrower than half the natural view (a far subject is not blown up), the
/// view's long side at most [`MAX_VIEW_DEGREES`]. Without a size, the natural view.
pub fn subject_framing(size: Option<(f64, f64)>, aspect: f64) -> (f64, f64) {
    let a = if aspect.is_finite() && aspect > 0.0 { aspect.clamp(0.1, 10.0) } else { 16.0 / 9.0 };
    let natural = natural_fov(a);
    let Some((w, h)) = size.filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0) else { return (natural, 0.0) };
    let tan_half = |deg: f64| (deg.clamp(1.0, 179.0).to_radians() / 2.0).tan();
    // across, in tangent terms, from the height wanted (down = across / aspect) and the width
    let need = (tan_half(h.min(MAX_SUBJECT_DEGREES) / 0.65) * a).max(tan_half(w.min(90.0) / 0.8));
    let need = need.max(tan_half(natural / 2.0));
    // the long side within its limit; what is cut, lens distance takes in
    let long = 2.0 * need.max(need / a).atan().to_degrees();
    if long <= MAX_VIEW_DEGREES {
        return (2.0 * need.atan().to_degrees(), 0.0);
    }
    let capped = if a >= 1.0 { tan_half(MAX_VIEW_DEGREES) } else { tan_half(MAX_VIEW_DEGREES) * a };
    (2.0 * capped.atan().to_degrees(), ((long - MAX_VIEW_DEGREES) / MAX_VIEW_DEGREES).clamp(0.0, 0.35))
}

/// Where a camera framing a subject looks up or down (degrees): the middle of its box, or for a
/// box taller than [`MAX_SUBJECT_DEGREES`] the middle of its top part that tall (a person by the
/// camera is framed by the face, not the knees). `pitch` the box's middle, `size` its extent.
pub fn subject_aim(pitch: f64, size: Option<(f64, f64)>) -> f64 {
    let lift = size.map_or(0.0, |(_, h)| if h.is_finite() { ((h - MAX_SUBJECT_DEGREES) / 2.0).max(0.0) } else { 0.0 });
    (pitch + lift).clamp(-85.0, 85.0)
}

/// A clip's virtual camera, added first in its chain (the other effects work on what it sees)
/// making a `frame` picture when it has none, at the frame's natural view ([`natural_fov`]);
/// enabled. Motion is left as it is.
pub fn add_camera(item: &mut TrackItem, frame: (u32, u32)) -> Option<&mut EffectInstance> {
    if !item.effects.iter().any(|e| e.effect == REFRAME) {
        let mut e = crate::find_effect(REFRAME)?.instance();
        let side = |v: u32| ParamValue::Float(f64::from(v.clamp(MIN_SIDE, MAX_SIDE)));
        e.params.insert("width".into(), Param::new(side(frame.0)));
        e.params.insert("height".into(), Param::new(side(frame.1)));
        e.params.insert("fov".into(), Param::new(ParamValue::Float(natural_fov(f64::from(frame.0.max(1)) / f64::from(frame.1.max(1))).round())));
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
    fn the_natural_view_covers_alike_in_every_shape() {
        assert!((natural_fov(16.0 / 9.0) - 100.0).abs() < 1e-9);
        let v = natural_fov(9.0 / 16.0);
        assert!((v - 67.6).abs() < 0.2, "{v}");
        // as much down in 9:16 as across in 16:9
        let down = 2.0 * ((v.to_radians() / 2.0).tan() * 16.0 / 9.0).atan().to_degrees();
        assert!((down - 100.0).abs() < 0.2, "{down}");
        assert!((natural_fov(1.0) - 83.6).abs() < 0.3);
        assert!((natural_fov(f64::NAN) - 100.0).abs() < 1e-9);
        let down_of = |across: f64, a: f64| 2.0 * ((across.to_radians() / 2.0).tan() / a).atan().to_degrees();
        // a person 45° tall near the camera fits a 16:9 frame: 45° is 65 % of the view's height
        let (across, d) = subject_framing(Some((20.0, 45.0)), 16.0 / 9.0);
        assert!(d == 0.0 && (down_of(across, 16.0 / 9.0) - 45.0 / 0.65).abs() < 0.5, "{across}");
        // closer still: the long side stays within its limit, lens distance takes in the rest
        for (size, a) in [((30.0, 70.0), 16.0 / 9.0), ((40.0, 74.0), 9.0 / 16.0), ((150.0, 150.0), 9.0 / 16.0)] {
            let (across, d) = subject_framing(Some(size), a);
            let long = across.max(down_of(across, a));
            assert!(long <= MAX_VIEW_DEGREES + 1e-6 && d > 0.0 && d <= 0.35, "{size:?} {a}: {across} {d}");
        }
        // 70° tall in 9:16: 108° top to bottom, within the limit, no lens distance
        let (across, d) = subject_framing(Some((40.0, 70.0)), 9.0 / 16.0);
        assert!(d == 0.0 && (down_of(across, 9.0 / 16.0) - 70.0 / 0.65).abs() < 0.5, "{across}");
        // a selfie-close subject is aimed by its top part
        assert_eq!(subject_aim(-10.0, Some((40.0, 115.0))), 10.0);
        assert_eq!(subject_aim(-10.0, Some((40.0, 50.0))), -10.0);
        assert_eq!(subject_aim(80.0, Some((40.0, 175.0))), 85.0);
        // a far subject: not narrower than half the natural view
        assert_eq!(subject_framing(None, 9.0 / 16.0).1, 0.0);
        assert_eq!(subject_framing(Some((f64::NAN, 1.0)), 1.0), (natural_fov(1.0), 0.0));
        let (far, _) = subject_framing(Some((2.0, 3.0)), 16.0 / 9.0);
        assert!((far - 50.0).abs() < 1e-9, "{far}");
    }

    #[test]
    fn panorama_points_are_directions() {
        assert_eq!(panorama_direction(0.5, 0.5), (0.0, 0.0));
        assert_eq!(panorama_direction(0.75, 0.25), (90.0, 45.0));
        assert_eq!(panorama_direction(0.0, 1.0), (-180.0, -90.0));
        assert_eq!(panorama_direction(f64::NAN, 9.0), (0.0, -90.0));
    }
}
