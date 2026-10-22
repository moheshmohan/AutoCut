//! The Creator effects: stylised looks short-form editors reach for (VHS, film grain, glitch,
//! kaleidoscope, zoom and motion blurs, a tech HUD, chromatic aberration, a light sweep, duotone
//! and halftone), in their own `Video Effects ▸ Creator` folder beside Premiere's.
//!
//! Every one is original maths written for FilmCraft (no presets, LUTs or artwork from another
//! product). Each runs on the GPU (`filmcraft_render::gpufx`, WGSL in `filmcraft-gpu`) with a CPU
//! reference of the same arithmetic, is deterministic for a frame time (noise comes from an
//! integer hash of the pixel, the frame and the effect's Random Seed), leaves the picture
//! untouched at an amount of zero, and keyframes its amounts.

use super::*;

/// The folder of the Creator effects.
pub const CREATOR: &[&str] = &["Video Effects", "Creator"];

/// Effect ids of the Creator folder, in the order they are defined.
pub const CREATOR_EFFECTS: &[&str] = &[
    "cr_vhs",
    "cr_film_grain",
    "cr_kaleidoscope",
    "cr_glitch",
    "cr_zoom_blur",
    "cr_motion_blur",
    "cr_tech_hud",
    "cr_chromatic_aberration",
    "cr_light_sweep",
    "cr_duotone",
    "cr_halftone",
];

/// The colour of the Tech HUD's lines and brackets by default (a cool cyan).
pub const HUD_COLOR: [f32; 4] = [0.25, 0.85, 1.0, 1.0];

fn pct(id: &'static str, label: &'static str, def: f64) -> ParamDef {
    f(id, label, def, 0.0, 100.0, "%")
}
fn px(id: &'static str, label: &'static str, def: f64, (min, max): (f64, f64), soft: f64) -> ParamDef {
    fs(id, label, def, (min, max), (min, soft), "px", 1)
}
fn seed() -> ParamDef {
    let mut p = fs("seed", "Random Seed", 0.0, (0.0, 10000.0), (0.0, 1000.0), "", 0);
    p.animatable = false;
    p
}

pub(super) fn defs() -> Vec<EffectDef> {
    let creator = |id, name, params| video(id, name, CREATOR, params);
    vec![
        creator(
            "cr_vhs",
            "VHS",
            vec![
                pct("amount", "Amount", 70.0),
                px("bleed", "Chroma Bleed", 6.0, (0.0, 200.0), 30.0),
                pct("tracking", "Tracking Lines", 40.0),
                pct("noise", "Noise", 25.0),
                px("wobble", "Tape Wobble", 3.0, (0.0, 100.0), 12.0),
                fs("saturation", "Saturation", 80.0, (0.0, 200.0), (0.0, 200.0), "%", 0),
                seed(),
            ],
        ),
        creator(
            "cr_film_grain",
            "Film Grain",
            vec![
                pct("amount", "Amount", 35.0),
                fs("size", "Grain Size", 1.5, (0.5, 20.0), (0.5, 6.0), "px", 1),
                b("color", "Color Grain", false),
                pct("response", "Midtone Response", 70.0),
                b("animate", "Animate", true),
                seed(),
            ],
        ),
        creator(
            "cr_kaleidoscope",
            "Kaleidoscope",
            vec![
                pct("amount", "Amount", 100.0),
                fs("segments", "Segments", 6.0, (2.0, 64.0), (2.0, 24.0), "", 0),
                ang("rotation", "Rotation", 0.0),
                pt("center", "Center", f64::NAN, f64::NAN),
                fs("zoom", "Zoom", 100.0, (10.0, 1000.0), (25.0, 400.0), "%", 0),
            ],
        ),
        creator(
            "cr_glitch",
            "Glitch",
            vec![
                pct("amount", "Amount", 50.0),
                px("block", "Block Size", 48.0, (2.0, 2000.0), 200.0),
                px("split", "RGB Split", 12.0, (0.0, 500.0), 60.0),
                pct("jitter", "Scanline Jitter", 40.0),
                fs("rate", "Rate", 12.0, (0.0, 60.0), (0.0, 30.0), "Hz", 0),
                ch("rhythm", "Rhythm", GLITCH_RHYTHMS, 1),
                seed(),
            ],
        ),
        creator(
            "cr_zoom_blur",
            "Zoom Blur",
            vec![pct("amount", "Amount", 25.0), pt("center", "Center", f64::NAN, f64::NAN), ch("kind", "Type", ZOOM_BLUR_KINDS, 0)],
        ),
        creator(
            "cr_motion_blur",
            "Directional Motion Blur",
            vec![
                ang("direction", "Direction", 90.0),
                px("length", "Length", 40.0, (0.0, 1000.0), 200.0),
                ch("shutter", "Shutter", MOTION_SHUTTERS, 0),
                pct("falloff", "Trail Falloff", 0.0),
            ],
        ),
        creator(
            "cr_tech_hud",
            "Tech HUD",
            vec![
                pct("amount", "Amount", 100.0),
                col("color", "HUD Color", HUD_COLOR),
                pct("scanlines", "Scanlines", 35.0),
                fs("spacing", "Line Spacing", 4.0, (1.0, 64.0), (2.0, 12.0), "px", 1),
                pct("grid", "Grid", 12.0),
                px("grid_size", "Grid Size", 96.0, (8.0, 4000.0), 300.0),
                pct("brackets", "Corner Brackets", 80.0),
                pct("vignette", "Vignette", 45.0),
                pct("tint", "Tint", 15.0),
                pct("flicker", "Flicker", 10.0),
            ],
        ),
        creator(
            "cr_chromatic_aberration",
            "Chromatic Aberration",
            vec![
                px("amount", "Amount", 8.0, (0.0, 400.0), 40.0),
                ch("kind", "Type", CHROMA_KINDS, 0),
                ang("angle", "Angle", 0.0),
                pt("center", "Center", f64::NAN, f64::NAN),
            ],
        ),
        creator(
            "cr_light_sweep",
            "Light Sweep",
            vec![
                fs("intensity", "Intensity", 80.0, (0.0, 400.0), (0.0, 200.0), "%", 0),
                col("color", "Light Color", [1.0, 0.97, 0.9, 1.0]),
                ang("angle", "Angle", 30.0),
                pct("width", "Width", 12.0),
                pct("softness", "Softness", 60.0),
                fs("position", "Position", 50.0, (-100.0, 200.0), (-20.0, 120.0), "%", 1),
                fs("cycle", "Auto Sweep (secs)", 0.0, (0.0, 60.0), (0.0, 5.0), "", 2),
            ],
        ),
        creator(
            "cr_duotone",
            "Duotone",
            vec![
                pct("amount", "Amount", 100.0),
                col("shadows", "Shadow Color", [0.10, 0.05, 0.35, 1.0]),
                col("highlights", "Highlight Color", [1.0, 0.75, 0.30, 1.0]),
                f("contrast", "Contrast", 0.0, -100.0, 100.0, ""),
            ],
        ),
        creator(
            "cr_halftone",
            "Halftone",
            vec![
                pct("amount", "Amount", 100.0),
                px("size", "Dot Size", 10.0, (2.0, 400.0), 60.0),
                ang("angle", "Angle", 45.0),
                ch("mode", "Mode", HALFTONE_MODES, 0),
                col("ink", "Ink Color", [0.05, 0.05, 0.08, 1.0]),
                col("paper", "Paper Color", [0.97, 0.95, 0.90, 1.0]),
            ],
        ),
    ]
}

/// Glitch ▸ Rhythm: the same strength every frame, or bursts (a hashed on / half / off pattern).
pub const GLITCH_RHYTHMS: &[&str] = &["Constant", "Bursts"];
/// Zoom Blur ▸ Type.
pub const ZOOM_BLUR_KINDS: &[&str] = &["Zoom", "Spin"];
/// Directional Motion Blur ▸ Shutter: taps on both sides of the pixel, or only behind it.
pub const MOTION_SHUTTERS: &[&str] = &["Centered", "Trailing"];
/// Chromatic Aberration ▸ Type: red and blue pushed out from the centre, or apart along an angle.
pub const CHROMA_KINDS: &[&str] = &["Radial", "Linear"];
/// Halftone ▸ Mode: ink dots on paper, or one dot per colour channel.
pub const HALFTONE_MODES: &[&str] = &["Mono", "Color"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creator_effects_are_in_their_folder_with_an_amount_and_keyframeable() {
        for id in CREATOR_EFFECTS {
            let d = find_effect(id).unwrap_or_else(|| panic!("{id}"));
            assert_eq!(d.category, CREATOR, "{id}");
            assert_eq!(d.kind, EffectKind::Video);
            assert!(d.accelerated, "{id}: GPU accelerated");
            // the amount-like parameter that turns the effect off at zero
            let amount = ["amount", "intensity", "length"].iter().find_map(|k| d.param(k)).unwrap_or_else(|| panic!("{id}: no amount"));
            assert!(amount.animatable, "{id}");
            assert!(matches!(amount.kind, ParamKind::Float { min, .. } if min == 0.0), "{id}");
        }
        let n = effect_defs().iter().filter(|d| d.category == CREATOR).count();
        assert_eq!(n, CREATOR_EFFECTS.len());
    }
}
