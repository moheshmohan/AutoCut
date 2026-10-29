//! FilmCraft's depth effects: a photo made to move in depth (**3D Photo**, in Perspective) and
//! blurred by how far each part of it is (**Depth Blur**, in Blur & Sharpen). Both read how near
//! each part of the clip's picture is from a depth network (Depth Anything V2 Small, downloaded on
//! the user's consent); without it they leave the picture as it is. They are FilmCraft's own, in
//! Premiere's folders ([`crate::effect::FILMCRAFT_OWN`]), and are rendered by
//! `filmcraft_render::depth` (on the GPU, the CPU as the fallback).

use super::*;

/// The 3D photo: the near parts of the picture move further than the far ones.
pub const PHOTO_3D: &str = "depth_parallax";
/// Blur by depth.
pub const DEPTH_BLUR: &str = "depth_blur";

pub(super) fn defs() -> Vec<EffectDef> {
    let still = |p: ParamDef| ParamDef { animatable: false, ..p };
    vec![
        video(
            PHOTO_3D,
            "3D Photo",
            PERSPECTIVE,
            vec![
                // how far the nearest part moves past the still depth, a share of the picture's width / height
                fs("travel_x", "Travel Across", 0.0, (-20.0, 20.0), (-8.0, 8.0), "%", 2),
                fs("travel_y", "Travel Down", 0.0, (-20.0, 20.0), (-8.0, 8.0), "%", 2),
                // how much the nearest part grows past the still depth (the camera moving in)
                fs("push", "Push In", 0.0, (-50.0, 50.0), (-25.0, 25.0), "%", 1),
                // the depth that stays where it is: 0 the farthest, 100 the nearest
                still(fs("pivot", "Still Depth", 0.0, (0.0, 100.0), (0.0, 100.0), "%", 0)),
                b("fill", "Zoom to Fill", true),
            ],
        ),
        video(
            DEPTH_BLUR,
            "Depth Blur",
            BLUR,
            vec![
                // the depth that is sharp: 0 the farthest, 100 the nearest
                fs("focus", "Focus Depth", 100.0, (0.0, 100.0), (0.0, 100.0), "%", 0),
                fs("range", "Depth of Field", 10.0, (0.0, 100.0), (0.0, 100.0), "%", 0),
                // the largest blur's radius, a share of the picture's width
                fs("blur", "Blur", 1.5, (0.0, 10.0), (0.0, 5.0), "%", 2),
                b("near", "Blur the Foreground", true),
            ],
        ),
    ]
}
