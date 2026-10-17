//! GL Transitions: transitions hand-ported from the MIT-licensed
//! [gl-transitions](https://github.com/gl-transitions/gl-transitions) collection to WGSL, with a
//! Rust reference of each (`filmcraft-render`, `transitions::shader`).
//!
//! They live in their own Effects-panel folder, `Video Transitions ▸ GL Transitions`, and are
//! named `GL <upstream name>` (the names users of the collection know, kept apart from the
//! Premiere-style transitions of the same name, such as Mosaic). Ids are `gl_<upstream name in
//! lower case>` and never change. Each transition keeps its upstream parameters (same units and
//! defaults), followed by the host's easing: progress is eased on the CPU at the exact frame time
//! before the shader sees it, so preview and export see the same value.

use crate::effect::{EffectDef, EffectKind, ParamDef, ang, b, ch, col, fs, pt};

/// Category of the GL Transitions folder.
pub const GL_TRANSITIONS: &[&str] = &["Video Transitions", "GL Transitions"];

/// Easing curves applied to the transition's progress (choice indices are stored in projects and
/// never change): linear, Robert Penner's ten families in / out / in-out, and a CSS-style
/// cubic Bézier given by the `bezier_*` parameters.
pub const EASING_OPTS: &[&str] = &[
    "Linear",
    "Sine In",
    "Sine Out",
    "Sine In-Out",
    "Quad In",
    "Quad Out",
    "Quad In-Out",
    "Cubic In",
    "Cubic Out",
    "Cubic In-Out",
    "Quart In",
    "Quart Out",
    "Quart In-Out",
    "Quint In",
    "Quint Out",
    "Quint In-Out",
    "Expo In",
    "Expo Out",
    "Expo In-Out",
    "Circ In",
    "Circ Out",
    "Circ In-Out",
    "Back In",
    "Back Out",
    "Back In-Out",
    "Elastic In",
    "Elastic Out",
    "Elastic In-Out",
    "Bounce In",
    "Bounce Out",
    "Bounce In-Out",
    "Custom Bézier",
];

/// Index of "Custom Bézier" in [`EASING_OPTS`].
pub const EASING_CUSTOM_BEZIER: u32 = 31;

/// Not animatable (transition parameters have no stopwatch).
fn fixed(mut p: ParamDef) -> ParamDef {
    p.animatable = false;
    p
}
fn num(id: &'static str, label: &'static str, def: f64, min: f64, max: f64, decimals: u8) -> ParamDef {
    fixed(fs(id, label, def, (min, max), (min, max), "", decimals))
}

/// The host's easing parameters, appended to every GL transition.
fn easing() -> [ParamDef; 5] {
    [
        ch("easing", "Easing", EASING_OPTS, 0),
        num("bezier_x1", "Bézier X1", 0.25, 0.0, 1.0, 2),
        num("bezier_y1", "Bézier Y1", 0.1, -1.0, 2.0, 2),
        num("bezier_x2", "Bézier X2", 0.25, 0.0, 1.0, 2),
        num("bezier_y2", "Bézier Y2", 1.0, -1.0, 2.0, 2),
    ]
}

fn gl(id: &'static str, name: &'static str, params: Vec<ParamDef>) -> EffectDef {
    let mut params = params;
    params.extend(easing());
    EffectDef { id, name, kind: EffectKind::VideoTransition, category: GL_TRANSITIONS, params, intrinsic: false, accelerated: true, float32: true, yuv: false }
}

/// The GL Transitions, sorted by name (the Effects panel lists them in this order).
pub(crate) fn defs() -> Vec<EffectDef> {
    vec![
        gl("gl_angular", "GL angular", vec![fixed(ang("starting_angle", "Starting Angle", 90.0))]),
        gl("gl_bookflip", "GL BookFlip", vec![]),
        gl(
            "gl_bounce",
            "GL Bounce",
            vec![
                // upstream's shadow_colour (0, 0, 0, 0.6): the colour, and its alpha as the strength
                fixed(col("shadow_colour", "Shadow Color", [0.0, 0.0, 0.0, 1.0])),
                num("shadow_alpha", "Shadow Opacity", 0.6, 0.0, 1.0, 2),
                num("shadow_height", "Shadow Height", 0.075, 0.01, 0.5, 3),
                num("bounces", "Bounces", 3.0, 1.0, 10.0, 1),
            ],
        ),
        gl("gl_burn", "GL burn", vec![fixed(col("color", "Color", [0.9, 0.4, 0.2, 1.0]))]),
        gl(
            "gl_butterflywavescrawler",
            "GL ButterflyWaveScrawler",
            vec![
                num("amplitude", "Amplitude", 1.0, 0.0, 10.0, 2),
                num("waves", "Waves", 30.0, 0.0, 200.0, 1),
                num("color_separation", "Color Separation", 0.3, 0.0, 2.0, 2),
            ],
        ),
        gl("gl_chessboard", "GL chessboard", vec![num("grid_num", "Squares per Row", 10.0, 1.0, 100.0, 1)]),
        gl("gl_circleopen", "GL circleopen", vec![num("smoothness", "Smoothness", 0.3, 0.0, 1.0, 2), b("opening", "Opening", true)]),
        gl(
            "gl_colorphase",
            "GL colorphase",
            vec![
                // upstream's fromStep / toStep for r, g, b; alpha keeps upstream's 0 → 1 (it only
                // shows on transparent clips)
                num("from_step_r", "From Step R", 0.0, 0.0, 1.0, 2),
                num("from_step_g", "From Step G", 0.2, 0.0, 1.0, 2),
                num("from_step_b", "From Step B", 0.4, 0.0, 1.0, 2),
                num("to_step_r", "To Step R", 0.6, 0.0, 1.0, 2),
                num("to_step_g", "To Step G", 0.8, 0.0, 1.0, 2),
                num("to_step_b", "To Step B", 1.0, 0.0, 1.0, 2),
            ],
        ),
        gl(
            "gl_crosshatch",
            "GL crosshatch",
            vec![
                fixed(pt("center", "Center", f64::NAN, f64::NAN)),
                num("threshold", "Threshold", 3.0, 0.01, 10.0, 2),
                num("fade_edge", "Fade Edge", 0.1, 0.0, 1.0, 2),
            ],
        ),
        gl("gl_crosswarp", "GL crosswarp", vec![]),
        gl("gl_crosszoom", "GL CrossZoom", vec![num("strength", "Strength", 0.4, 0.0, 1.0, 2)]),
        gl(
            "gl_cube",
            "GL cube",
            vec![
                num("persp", "Perspective", 0.7, 0.0, 1.0, 2),
                num("unzoom", "Unzoom", 0.3, 0.0, 1.0, 2),
                num("reflection", "Reflection", 0.4, 0.0, 1.0, 2),
                num("floating", "Floating", 3.0, 0.0, 100.0, 1),
            ],
        ),
        gl(
            "gl_directionalwarp",
            "GL directionalwarp",
            vec![
                num("smoothness", "Smoothness", 0.1, 0.0, 1.0, 2),
                num("direction_x", "Direction X", -1.0, -1.0, 1.0, 2),
                num("direction_y", "Direction Y", 1.0, -1.0, 1.0, 2),
            ],
        ),
        gl(
            "gl_doorway",
            "GL doorway",
            vec![
                num("reflection", "Reflection", 0.4, 0.0, 1.0, 2),
                num("perspective", "Perspective", 0.4, 0.0, 1.0, 2),
                num("depth", "Depth", 3.0, 1.0, 10.0, 2),
            ],
        ),
        gl("gl_dreamyzoom", "GL DreamyZoom", vec![num("rotation", "Rotation", 6.0, -360.0, 360.0, 1), num("scale", "Scale", 1.2, 0.1, 10.0, 2)]),
        gl("gl_filmburn", "GL FilmBurn", vec![num("seed", "Seed", 2.31, -1000.0, 1000.0, 2)]),
        gl("gl_glitchdisplace", "GL GlitchDisplace", vec![]),
        gl("gl_glitchmemories", "GL GlitchMemories", vec![]),
        gl(
            "gl_hexagonalize",
            "GL hexagonalize",
            vec![num("steps", "Steps", 50.0, 0.0, 200.0, 0), num("horizontal_hexagons", "Horizontal Hexagons", 20.0, 1.0, 400.0, 1)],
        ),
        gl(
            "gl_kaleidoscope",
            "GL kaleidoscope",
            vec![
                num("speed", "Speed", 1.0, -10.0, 10.0, 2),
                // radians added at each of the seven folds
                num("angle", "Angle", 1.0, -10.0, 10.0, 2),
                num("power", "Power", 1.5, 0.1, 10.0, 2),
            ],
        ),
        gl(
            "gl_luminance_melt",
            "GL luminance_melt",
            vec![
                // upstream's `direction` (true: down)
                b("melt_down", "Melt Downward", true),
                num("l_threshold", "Luminance Threshold", 0.8, 0.0, 1.0, 2),
                b("above", "Melt Above Threshold", false),
            ],
        ),
        gl("gl_mosaic", "GL Mosaic", vec![num("endx", "End X", 2.0, -10.0, 10.0, 0), num("endy", "End Y", -1.0, -10.0, 10.0, 0)]),
        gl(
            "gl_perlin",
            "GL perlin",
            vec![
                num("scale", "Scale", 4.0, 0.1, 100.0, 2),
                num("smoothness", "Smoothness", 0.01, 0.0, 1.0, 3),
                num("seed", "Seed", 12.9898, -1000.0, 1000.0, 4),
            ],
        ),
        gl("gl_pinwheel", "GL pinwheel", vec![num("speed", "Speed", 2.0, -20.0, 20.0, 2)]),
        gl(
            "gl_pixelize",
            "GL pixelize",
            vec![
                num("squares_min_x", "Minimum Squares X", 20.0, 1.0, 400.0, 0),
                num("squares_min_y", "Minimum Squares Y", 20.0, 1.0, 400.0, 0),
                num("steps", "Steps", 50.0, 0.0, 200.0, 0),
            ],
        ),
        gl("gl_polar_function", "GL polar_function", vec![num("segments", "Segments", 5.0, 0.0, 50.0, 0)]),
        gl(
            "gl_polkadotscurtain",
            "GL PolkaDotsCurtain",
            vec![
                num("dots", "Dots", 20.0, 1.0, 200.0, 1),
                // NaN: upstream's default, the bottom left corner
                fixed(pt("center", "Center", f64::NAN, f64::NAN)),
            ],
        ),
        gl(
            "gl_randomsquares",
            "GL randomsquares",
            vec![num("size_x", "Columns", 10.0, 1.0, 400.0, 0), num("size_y", "Rows", 10.0, 1.0, 400.0, 0), num("smoothness", "Smoothness", 0.5, 0.0, 1.0, 2)],
        ),
        gl("gl_ripple", "GL ripple", vec![num("amplitude", "Amplitude", 100.0, 0.0, 500.0, 1), num("speed", "Speed", 50.0, 0.0, 500.0, 1)]),
        gl("gl_simplezoom", "GL SimpleZoom", vec![num("zoom_quickness", "Zoom Quickness", 0.8, 0.2, 1.0, 2)]),
        gl("gl_squeeze", "GL squeeze", vec![num("color_separation", "Color Separation", 0.04, 0.0, 1.0, 3)]),
        gl(
            "gl_starwipe",
            "GL StarWipe",
            vec![
                num("border_thickness", "Border Thickness", 0.01, 0.0, 0.2, 3),
                num("star_rotation", "Star Rotation", 0.75, -5.0, 5.0, 2),
                fixed(col("border_color", "Border Color", [1.0, 1.0, 1.0, 1.0])),
                fixed(pt("center", "Center", f64::NAN, f64::NAN)),
            ],
        ),
        gl(
            "gl_swap",
            "GL swap",
            vec![
                num("reflection", "Reflection", 0.4, 0.0, 1.0, 2),
                num("perspective", "Perspective", 0.2, 0.0, 1.0, 2),
                num("depth", "Depth", 3.0, 1.0, 10.0, 2),
            ],
        ),
        gl("gl_swirl", "GL Swirl", vec![]),
        gl("gl_tvstatic", "GL TVStatic", vec![num("offset", "Offset", 0.05, 0.0, 0.5, 3)]),
        gl("gl_waterdrop", "GL WaterDrop", vec![num("amplitude", "Amplitude", 30.0, 0.0, 500.0, 1), num("speed", "Speed", 30.0, 0.0, 500.0, 1)]),
        gl("gl_wind", "GL wind", vec![num("size", "Size", 0.2, 0.0, 1.0, 2), b("reversed", "Reversed", false)]),
        gl("gl_windowslice", "GL windowslice", vec![num("count", "Count", 10.0, 1.0, 100.0, 1), num("smoothness", "Smoothness", 0.5, 0.0, 1.0, 2)]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_transitions_are_listed_in_their_folder() {
        let all = crate::effect::effect_defs();
        let gl: Vec<_> = all.iter().filter(|d| d.category == GL_TRANSITIONS).collect();
        assert_eq!(gl.len(), 38);
        let mut names: Vec<_> = gl.iter().map(|d| d.name.to_ascii_lowercase()).collect();
        let sorted = {
            let mut s = names.clone();
            s.sort();
            s
        };
        assert_eq!(names, sorted, "sorted by name");
        names.dedup();
        assert_eq!(names.len(), 38);
        for d in gl {
            assert!(d.id.starts_with("gl_") && d.name.starts_with("GL "), "{}", d.id);
            assert_eq!(d.kind, EffectKind::VideoTransition);
            assert!(d.params.iter().all(|p| !p.animatable), "{}", d.id);
            assert_eq!(d.param("easing").map(|p| &p.default), Some(&crate::ParamValue::Choice(0)), "{}: linear by default", d.id);
        }
        assert_eq!(EASING_OPTS.len(), EASING_CUSTOM_BEZIER as usize + 1);
        assert_eq!(EASING_OPTS[EASING_CUSTOM_BEZIER as usize], "Custom Bézier");
        // names users of the collection know resolve, and Mosaic stays the Premiere-style one
        let find = |n: &str| crate::vtransition::find_transition(n, EffectKind::VideoTransition).map(|d| d.id);
        assert_eq!(find("GL CrossZoom"), Some("gl_crosszoom"));
        assert_eq!(find("gl_waterdrop"), Some("gl_waterdrop"));
        assert_eq!(find("Mosaic"), Some("mosaic_transition"));
    }
}
