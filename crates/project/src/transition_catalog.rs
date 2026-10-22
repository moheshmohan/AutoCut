//! The transition catalogue's themed folders (roadmap C5) and its search keywords.
//!
//! After Premiere's ten `Video Transitions` folders come the folders users know from GL
//! Transitions, xfade and the mobile editors, each a mix of GL ports and FilmCraft's own engines:
//!
//! | folder | what is in it |
//! |---|---|
//! | Flow & Morph | the flow transitions and the Content-Aware Wipe ([`crate::flow_transitions`]) |
//! | Zooms & Camera | the GL zooms and warps, and the camera engine's zooms, spin, whip pan and shake |
//! | 3D & Spatial | the GL cube, doorway, swap, book flip and bounce, the camera engine's card flip and tilt slide |
//! | Light & Film | the GL burns and colour phase, the light engine's leaks, burn-through, flare, flash, bokeh and projector roll |
//! | Distortion | the GL ripples, swirls, kaleidoscope, displacement, squeeze and melt |
//! | Glitch & Datamosh | the GL glitches, static and pixelize, Datamosh, and the pattern engine's RGB split and scanline glitches |
//! | Particles & Patterns | the GL checkers, squares, dots, hexagons, crosshatch, noise and mosaic, the pattern engine's halftone, triangles, hearts and sparkles |
//! | Luma Wipes | the luma-wipe engine's presets and the GL shape wipes |
//! | Blend Dissolves | the blend-dissolve engine's presets |
//!
//! Ids never move with their folder: a project names a transition by id, so a saved project
//! loads whichever folder its transitions now live in. The GL ports keep their `GL …` names.
//!
//! **New procedural engines.** Light & Film, Particles & Patterns and the camera moves are
//! presets of three original shaders (`light_film.wgsl`, `patterns.wgsl`, `camera.wgsl` in
//! `filmcraft-render`, each with its Rust reference): every leak, burn, flare and bokeh is
//! generated from code, never from footage.
//!
//! **Keywords.** [`transition_keywords`] gives the words a search should also find a transition
//! by (its folder's, and its own: "zoom", "leak", "datamosh", "heart"…).

use crate::effect::{EffectDef, EffectKind, ParamDef, ang, ch, col, fs};
use crate::flow_transitions::{self, FLOW_MORPH};
use crate::transition_engines::{BLEND_DISSOLVES, LUMA_WIPES};
use crate::vtransition::DIR_OPTS;

pub const ZOOMS_CAMERA: &[&str] = &["Video Transitions", "Zooms & Camera"];
pub const SPATIAL_3D: &[&str] = &["Video Transitions", "3D & Spatial"];
pub const LIGHT_FILM: &[&str] = &["Video Transitions", "Light & Film"];
pub const DISTORTION: &[&str] = &["Video Transitions", "Distortion"];
pub const GLITCH_DATAMOSH: &[&str] = &["Video Transitions", "Glitch & Datamosh"];
pub const PARTICLES_PATTERNS: &[&str] = &["Video Transitions", "Particles & Patterns"];

/// The themed folders, in panel order (after Premiere's).
pub const THEMED_FOLDERS: &[&[&str]] =
    &[FLOW_MORPH, ZOOMS_CAMERA, SPATIAL_3D, LIGHT_FILM, DISTORTION, GLITCH_DATAMOSH, PARTICLES_PATTERNS, LUMA_WIPES, BLEND_DISSOLVES];

/// The folder of GL port `id` (None for other ids).
pub fn gl_folder(id: &str) -> Option<&'static [&'static str]> {
    Some(match id {
        "gl_crosszoom" | "gl_dreamyzoom" | "gl_simplezoom" | "gl_crosswarp" | "gl_directionalwarp" => ZOOMS_CAMERA,
        "gl_cube" | "gl_doorway" | "gl_swap" | "gl_bookflip" | "gl_bounce" => SPATIAL_3D,
        "gl_filmburn" | "gl_burn" | "gl_colorphase" => LIGHT_FILM,
        "gl_ripple" | "gl_waterdrop" | "gl_swirl" | "gl_butterflywavescrawler" | "gl_kaleidoscope" | "gl_displacement" | "gl_squeeze" | "gl_luminance_melt" => {
            DISTORTION
        }
        "gl_glitchdisplace" | "gl_glitchmemories" | "gl_tvstatic" | "gl_pixelize" => GLITCH_DATAMOSH,
        "gl_chessboard" | "gl_randomsquares" | "gl_polkadotscurtain" | "gl_hexagonalize" | "gl_crosshatch" | "gl_perlin" | "gl_mosaic" => PARTICLES_PATTERNS,
        "gl_luma" | "gl_angular" | "gl_circleopen" | "gl_starwipe" | "gl_pinwheel" | "gl_polar_function" | "gl_windowslice" | "gl_wind" => LUMA_WIPES,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------
// The procedural engines

/// A preset of one of the C5 engines: the engine, its mode and its folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StylePreset {
    pub id: &'static str,
    pub name: &'static str,
    pub engine: StyleEngine,
    /// The engine shader's mode.
    pub mode: u32,
    pub category: &'static [&'static str],
}

/// The C5 engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StyleEngine {
    /// Light leaks, burn-through, lens flare, exposure flash, bokeh, projector roll.
    Light,
    /// Halftone, triangles, hearts, sparkles; RGB split and scanline glitches.
    Pattern,
    /// Zooms, spin, whip pan, shake; card flip and tilt slide.
    Camera,
}

const fn style(id: &'static str, name: &'static str, engine: StyleEngine, mode: u32, category: &'static [&'static str]) -> StylePreset {
    StylePreset { id, name, engine, mode, category }
}

/// Every preset of the C5 engines, by engine and mode.
pub const STYLE_PRESETS: &[StylePreset] = {
    use StyleEngine::*;
    &[
        style("lf_light_leak", "Warm Light Leak", Light, 0, LIGHT_FILM),
        style("lf_prism_leak", "Prism Leak", Light, 1, LIGHT_FILM),
        style("lf_burn_through", "Burn Through", Light, 2, LIGHT_FILM),
        style("lf_flare_sweep", "Lens Flare Sweep", Light, 3, LIGHT_FILM),
        style("lf_exposure_flash", "Exposure Flash", Light, 4, LIGHT_FILM),
        style("lf_bokeh", "Bokeh Defocus", Light, 5, LIGHT_FILM),
        style("lf_projector_roll", "Projector Roll", Light, 6, LIGHT_FILM),
        style("pt_halftone", "Halftone Dots", Pattern, 0, PARTICLES_PATTERNS),
        style("pt_triangles", "Triangle Tiles", Pattern, 1, PARTICLES_PATTERNS),
        style("pt_hearts", "Heart Bubbles", Pattern, 2, PARTICLES_PATTERNS),
        style("pt_sparkle", "Sparkle Dissolve", Pattern, 3, PARTICLES_PATTERNS),
        style("glitch_rgb_split", "RGB Split Glitch", Pattern, 4, GLITCH_DATAMOSH),
        style("glitch_scanlines", "Scanline Glitch", Pattern, 5, GLITCH_DATAMOSH),
        style("cam_punch_in", "Punch Zoom In", Camera, 0, ZOOMS_CAMERA),
        style("cam_pull_out", "Pull Zoom Out", Camera, 1, ZOOMS_CAMERA),
        style("cam_spin_zoom", "Spin Zoom", Camera, 2, ZOOMS_CAMERA),
        style("cam_whip_pan", "Whip Pan", Camera, 3, ZOOMS_CAMERA),
        style("cam_shake", "Shake Cut", Camera, 4, ZOOMS_CAMERA),
        style("sp_card_flip", "Card Flip", Camera, 5, SPATIAL_3D),
        style("sp_tilt_slide", "Tilt Slide", Camera, 6, SPATIAL_3D),
    ]
};

/// The C5 engine preset with effect id `id`.
pub fn style_preset(id: &str) -> Option<&'static StylePreset> {
    STYLE_PRESETS.iter().find(|p| p.id == id)
}

/// Order choices of the cell patterns (stored in projects; never reordered).
pub const PATTERN_ORDER_OPTS: &[&str] = &["Random", "Left to Right", "Center Out"];

/// Not animatable (transition parameters have no stopwatch).
fn fixed(mut p: ParamDef) -> ParamDef {
    p.animatable = false;
    p
}
fn num(id: &'static str, label: &'static str, def: f64, min: f64, max: f64, unit: &'static str, decimals: u8) -> ParamDef {
    fixed(fs(id, label, def, (min, max), (min, max), unit, decimals))
}
fn seed() -> ParamDef {
    num("seed", "Random Seed", 1.0, 0.0, 9999.0, "", 0)
}
fn amount(def: f64) -> ParamDef {
    num("amount", "Amount", def, 0.0, 100.0, "%", 0)
}
fn colour(id: &'static str, label: &'static str, c: [f32; 4]) -> ParamDef {
    fixed(col(id, label, c))
}
fn motion_blur(def: f64) -> ParamDef {
    num("motion_blur", "Motion Blur", def, 0.0, 100.0, "%", 0)
}

/// The parameters of preset `p` (before the easing).
fn style_params(p: &StylePreset) -> Vec<ParamDef> {
    use StyleEngine::*;
    let cells = |def: f64| num("count", "Cells per Row", def, 2.0, 64.0, "", 0);
    match (p.engine, p.mode) {
        (Light, 0) => vec![colour("color", "Leak Color", [1.0, 0.55, 0.2, 1.0]), amount(80.0), seed()],
        (Light, 1) => vec![amount(80.0), num("split", "Color Split", 8.0, 0.0, 100.0, "px", 0), seed()],
        (Light, 2) => vec![colour("color", "Burn Color", [1.0, 0.45, 0.1, 1.0]), num("edge", "Edge Width", 8.0, 1.0, 40.0, "%", 0), seed()],
        (Light, 3) => vec![colour("color", "Flare Color", [1.0, 0.85, 0.6, 1.0]), amount(90.0), fixed(ang("angle", "Sweep Angle", 0.0))],
        (Light, 4) => vec![colour("color", "Flash Color", [1.0, 1.0, 1.0, 1.0]), num("intensity", "Intensity", 100.0, 0.0, 200.0, "%", 0)],
        (Light, 5) => vec![num("radius", "Blur Radius", 40.0, 0.0, 300.0, "px", 0), num("boost", "Highlight Boost", 60.0, 0.0, 200.0, "%", 0)],
        (Light, _) => vec![
            num("gap", "Frame Line", 60.0, 0.0, 400.0, "px", 0),
            num("flicker", "Flicker", 50.0, 0.0, 100.0, "%", 0),
            num("scratches", "Scratches", 50.0, 0.0, 100.0, "%", 0),
            seed(),
            motion_blur(50.0),
        ],
        (Pattern, 0) => vec![cells(16.0), ch("order", "Order", PATTERN_ORDER_OPTS, 2), seed()],
        (Pattern, 1) => vec![cells(10.0), ch("order", "Order", PATTERN_ORDER_OPTS, 1), seed()],
        (Pattern, 2) => vec![cells(6.0), ch("order", "Order", PATTERN_ORDER_OPTS, 0), seed()],
        (Pattern, 3) => vec![cells(24.0), colour("color", "Sparkle Color", [1.0, 0.95, 0.8, 1.0]), amount(80.0), seed()],
        (Pattern, 4) => vec![num("shift", "Shift", 24.0, 0.0, 400.0, "px", 0), num("bands", "Bands", 24.0, 2.0, 200.0, "", 0), seed()],
        (Pattern, _) => vec![num("shift", "Shift", 40.0, 0.0, 400.0, "px", 0), num("lines", "Line Height", 4.0, 1.0, 64.0, "px", 0), seed()],
        (Camera, 0) => vec![num("zoom", "Zoom", 60.0, 0.0, 300.0, "%", 0), motion_blur(60.0)],
        (Camera, 1) => vec![num("zoom", "Zoom", 60.0, 0.0, 300.0, "%", 0), motion_blur(60.0)],
        (Camera, 2) => vec![fixed(ang("rotation", "Rotation", 180.0)), num("zoom", "Zoom", 40.0, 0.0, 300.0, "%", 0), motion_blur(60.0)],
        (Camera, 3) => vec![ch("direction", "Direction", DIR_OPTS, 3), num("blur", "Blur", 80.0, 0.0, 100.0, "%", 0)],
        (Camera, 4) => vec![num("shake", "Shake", 30.0, 0.0, 400.0, "px", 0), num("speed", "Speed", 12.0, 1.0, 60.0, "", 0), seed()],
        (Camera, 5) => vec![ch("axis", "Axis", crate::vtransition::AXIS_OPTS, 0), num("perspective", "Perspective", 50.0, 0.0, 100.0, "%", 0)],
        (Camera, _) => vec![ch("direction", "Direction", DIR_OPTS, 3), fixed(ang("tilt", "Tilt", 35.0)), motion_blur(40.0)],
    }
}

fn def(id: &'static str, name: &'static str, category: &'static [&'static str], mut params: Vec<ParamDef>) -> EffectDef {
    params.extend(crate::gltransition::easing());
    EffectDef { id, name, kind: EffectKind::VideoTransition, category, params, intrinsic: false, accelerated: true, float32: true, yuv: false }
}

/// The transitions of the themed folders, folder by folder in panel order, each folder sorted by
/// name: the GL ports, the flow transitions and the C5 engines' presets in theirs, with the
/// Luma Wipes and Blend Dissolves.
pub(crate) fn defs() -> Vec<EffectDef> {
    let mut all: Vec<EffectDef> = crate::gltransition::defs();
    all.extend(flow_transitions::defs());
    all.extend(STYLE_PRESETS.iter().map(|p| def(p.id, p.name, p.category, style_params(p))));
    all.extend(crate::transition_engines::defs());
    let mut out = Vec::with_capacity(all.len());
    for folder in THEMED_FOLDERS {
        let mut v: Vec<EffectDef> = all.iter().filter(|d| d.category == *folder).cloned().collect();
        v.sort_by_key(|d| d.name.to_ascii_lowercase());
        out.extend(v);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Keywords

/// Words a whole folder is found by.
fn folder_keywords(cat: &[&str]) -> &'static [&'static str] {
    match cat.get(1).copied().unwrap_or("") {
        "Flow & Morph" => &["flow", "morph", "ai", "smart", "optical flow", "warp"],
        "Zooms & Camera" => &["zoom", "camera", "punch", "push in", "motion"],
        "3D & Spatial" => &["3d", "spatial", "perspective", "flip", "cube"],
        "Light & Film" => &["light", "film", "leak", "flash", "glow", "burn"],
        "Distortion" => &["distort", "warp", "wave", "liquid", "ripple"],
        "Glitch & Datamosh" => &["glitch", "datamosh", "digital", "error", "rgb"],
        "Particles & Patterns" => &["pattern", "particles", "shapes", "tiles", "grid"],
        "Luma Wipes" => &["wipe", "luma", "mask", "reveal", "shape"],
        "Blend Dissolves" => &["dissolve", "blend", "mix", "fade"],
        "Creator" => &["creator", "quick", "social", "reels", "shorts", "trending"],
        "Animation" => &["animation", "motion", "move"],
        "Dissolve" => &["dissolve", "fade", "mix", "crossfade"],
        "Grunge & Distort" => &["grunge", "distort", "glitch"],
        "Immersive Video" => &["vr", "360", "immersive"],
        "Lights & Blurs" => &["light", "blur", "glow", "flash"],
        "Slide" => &["slide", "push", "move"],
        "Smart Tools" => &["smart", "shape"],
        "Text" => &["text", "type"],
        "Transformers" => &["transform", "3d", "flip"],
        "Wipe" => &["wipe", "reveal"],
        _ => &[],
    }
}

/// Words transition `id` itself is found by, besides its name.
fn own_keywords(id: &str) -> &'static [&'static str] {
    if let Some(c) = crate::creator_transitions::creator_transition(id) {
        return c.keywords;
    }
    match id {
        "flow_morph" => &["ai transition", "morph cut", "shape match", "liquid"],
        "flow_doppler_warp" => &["streak", "radial blur", "speed", "zoom"],
        "flow_hype_ramp" => &["speed ramp", "velocity", "punch", "pulse", "beat"],
        "flow_glitch" => &["glitch", "datamosh", "blocks", "rgb split"],
        "flow_split_bands" => &["bands", "stripes", "slices", "slide"],
        "flow_datamosh" => &["pixel bleed", "melt", "compression", "smear"],
        "content_aware_wipe" => &["seam carving", "subject", "smart wipe", "object"],
        "lf_light_leak" => &["light leak", "warm", "vintage", "sun"],
        "lf_prism_leak" => &["rainbow", "prism", "chromatic", "light leak"],
        "lf_burn_through" => &["film burn", "fire", "melt", "projector"],
        "lf_flare_sweep" => &["lens flare", "anamorphic", "streak", "sun"],
        "lf_exposure_flash" => &["white flash", "overexposure", "blown out", "camera flash"],
        "lf_bokeh" => &["bokeh", "defocus", "out of focus", "blur"],
        "lf_projector_roll" => &["film frame", "vintage", "scratches", "flicker", "projector"],
        "pt_halftone" => &["dots", "halftone", "circles", "comic"],
        "pt_triangles" => &["triangles", "geometric", "mosaic", "low poly"],
        "pt_hearts" => &["hearts", "love", "valentine", "cute"],
        "pt_sparkle" => &["sparkles", "glitter", "stars", "magic", "particles"],
        "glitch_rgb_split" => &["chromatic aberration", "rgb shift", "vhs"],
        "glitch_scanlines" => &["scanlines", "crt", "vhs", "tear", "signal"],
        "cam_punch_in" => &["zoom in", "punch in", "crash zoom"],
        "cam_pull_out" => &["zoom out", "pull back"],
        "cam_spin_zoom" => &["spin", "rotate", "twist", "zoom"],
        "cam_whip_pan" => &["whip", "swish", "pan", "fast"],
        "cam_shake" => &["shake", "earthquake", "handheld", "impact"],
        "sp_card_flip" => &["flip", "card", "turn", "rotate"],
        "sp_tilt_slide" => &["tilt", "perspective", "slide", "push"],
        "gl_crosszoom" | "gl_dreamyzoom" | "gl_simplezoom" => &["zoom", "gl transitions"],
        "gl_crosswarp" | "gl_directionalwarp" => &["warp", "stretch", "gl transitions"],
        "gl_cube" => &["cube", "box", "rotate", "gl transitions"],
        "gl_doorway" => &["doors", "open", "gl transitions"],
        "gl_swap" => &["swap", "cards", "gl transitions"],
        "gl_bookflip" => &["page", "book", "turn", "gl transitions"],
        "gl_bounce" => &["drop", "bounce", "shadow", "gl transitions"],
        "gl_filmburn" | "gl_burn" => &["film burn", "fire", "gl transitions"],
        "gl_colorphase" => &["color", "channels", "gl transitions"],
        "gl_ripple" | "gl_waterdrop" => &["water", "ripple", "drop", "gl transitions"],
        "gl_swirl" => &["swirl", "twirl", "vortex", "gl transitions"],
        "gl_butterflywavescrawler" => &["butterfly", "wave", "gl transitions"],
        "gl_kaleidoscope" => &["kaleidoscope", "mirror", "gl transitions"],
        "gl_displacement" => &["displace", "clouds", "gl transitions"],
        "gl_squeeze" => &["squeeze", "squash", "gl transitions"],
        "gl_luminance_melt" => &["melt", "drip", "gl transitions"],
        "gl_glitchdisplace" | "gl_glitchmemories" => &["glitch", "gl transitions"],
        "gl_tvstatic" => &["static", "noise", "tv", "gl transitions"],
        "gl_pixelize" => &["pixel", "8-bit", "retro", "gl transitions"],
        "gl_chessboard" => &["checkerboard", "chess", "gl transitions"],
        "gl_randomsquares" => &["squares", "blocks", "random", "gl transitions"],
        "gl_polkadotscurtain" => &["dots", "polka", "gl transitions"],
        "gl_hexagonalize" => &["hexagon", "honeycomb", "gl transitions"],
        "gl_crosshatch" => &["crosshatch", "sketch", "gl transitions"],
        "gl_perlin" => &["noise", "perlin", "gl transitions"],
        "gl_mosaic" => &["mosaic", "tiles", "gl transitions"],
        "gl_luma" | "gl_angular" | "gl_circleopen" | "gl_starwipe" | "gl_pinwheel" | "gl_polar_function" | "gl_windowslice" | "gl_wind" => {
            &["wipe", "gl transitions"]
        }
        _ => &[],
    }
}

/// The words a search should also find transition `id` by (its folder's and its own; lower
/// case). Empty for ids that are not video transitions.
pub fn transition_keywords(id: &str) -> Vec<&'static str> {
    let Some(d) = crate::effect::find_effect(id).filter(|d| d.kind == EffectKind::VideoTransition) else { return Vec::new() };
    let mut v: Vec<&'static str> = folder_keywords(d.category).to_vec();
    for k in own_keywords(id) {
        if !v.contains(k) {
            v.push(k);
        }
    }
    v
}

/// Whether a search for `query` (lower case) finds transition `id` by one of its keywords.
pub fn keyword_match(id: &str, query: &str) -> bool {
    let q = query.trim();
    !q.is_empty() && transition_keywords(id).iter().any(|k| k.contains(q))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::effect_defs;

    #[test]
    fn every_gl_port_has_a_themed_folder() {
        let gl: Vec<&EffectDef> = effect_defs().iter().filter(|d| d.id.starts_with("gl_")).collect();
        assert_eq!(gl.len(), 40);
        for d in gl {
            assert_eq!(Some(d.category), gl_folder(d.id), "{}", d.id);
            assert!(d.name.starts_with("GL "), "{}", d.id);
        }
        assert!(gl_folder("cross_dissolve").is_none());
    }

    #[test]
    fn themed_folders_are_sorted_and_full() {
        let defs = effect_defs();
        for folder in THEMED_FOLDERS {
            let names: Vec<String> = defs.iter().filter(|d| d.category == *folder).map(|d| d.name.to_ascii_lowercase()).collect();
            assert!(names.len() >= 5, "{folder:?}: {}", names.len());
            let mut sorted = names.clone();
            sorted.sort();
            assert_eq!(names, sorted, "{folder:?} sorted by name");
        }
        // every C5 engine preset is a transition with the host's easing; modes are distinct
        for p in STYLE_PRESETS {
            let d = crate::find_effect(p.id).unwrap_or_else(|| panic!("{}", p.id));
            assert_eq!(d.category, p.category);
            assert!(d.param("easing").is_some() && d.params.iter().all(|q| !q.animatable), "{}", p.id);
            assert_eq!(STYLE_PRESETS.iter().filter(|q| q.engine == p.engine && q.mode == p.mode).count(), 1, "{}", p.id);
        }
        assert_eq!(crate::find_effect("flow_datamosh").map(|d| d.category), Some(GLITCH_DATAMOSH));
    }

    #[test]
    fn every_transition_has_keywords() {
        for d in effect_defs().iter().filter(|d| d.kind == EffectKind::VideoTransition && d.category.first() == Some(&"Video Transitions")) {
            let k = transition_keywords(d.id);
            assert!(!k.is_empty(), "{}", d.id);
            assert!(k.iter().all(|w| w.chars().all(|c| !c.is_uppercase())), "{}: lower case", d.id);
        }
        // the themed folders' transitions have words of their own or their folder's
        assert!(keyword_match("lf_light_leak", "leak"));
        assert!(keyword_match("flow_datamosh", "datamosh"));
        assert!(keyword_match("gl_cube", "3d"));
        assert!(keyword_match("pt_hearts", "love"));
        assert!(keyword_match("crt_swipe_left", "whip") && keyword_match("crt_pull_in", "reels"));
        assert!(!keyword_match("pt_hearts", "zoom"));
        assert!(!keyword_match("pt_hearts", "  "));
        assert!(transition_keywords("gaussian_blur").is_empty() && transition_keywords("nope").is_empty());
    }
}
