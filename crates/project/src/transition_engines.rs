//! Transition engines (roadmap C2): two shaders behind many named transitions.
//!
//! - **Luma Wipes** (`Video Transitions ▸ Luma Wipes`, ids `luma_*`): one wipe shader that follows a
//!   grayscale map, generated from scratch for each [`LumaShape`] (radial, linear at any angle,
//!   barn door, blinds, star, clock, iris shapes, noise, Worley cells…) with Softness (the width
//!   of the moving edge, in map values), Feather (a blur of the map, in pixels) and Invert. No
//!   map image is shipped or read: every map is computed (`filmcraft-render`,
//!   `transitions::shader::maps`).
//! - **Blend Dissolves** (`Video Transitions ▸ Blend Dissolves`, ids `blend_*`): a dissolve whose
//!   middle is the outgoing and incoming pictures combined with one of the compositor's blend
//!   modes ([`crate::effect::BLEND_MODES`]: Screen, Overlay, Difference…), at a Strength.
//!
//! Each named transition is a preset: the engine's parameters with its own defaults (a luma
//! preset also fixes its shape, and lists only the parameters that shape reads). Like the GL
//! Transitions, the progress is eased on the CPU at the exact frame time (the same easing
//! parameters) before either shader sees it. Ids never change.

use crate::effect::{BLEND_MODES, EffectDef, EffectKind, ParamDef, ang, b, ch, fs, pt};

/// Category of the Luma Wipes folder.
pub const LUMA_WIPES: &[&str] = &["Video Transitions", "Luma Wipes"];
/// Category of the Blend Dissolves folder.
pub const BLEND_DISSOLVES: &[&str] = &["Video Transitions", "Blend Dissolves"];

/// Spin choices of the clock and spiral maps.
pub const SPIN_OPTS: &[&str] = &["Clockwise", "Counter-clockwise"];

/// The shape of a luma wipe's map: which pixels the incoming picture reaches first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LumaShape {
    /// A straight edge sweeping across at `angle` (90° = left to right).
    Linear,
    /// A circle growing from `center`.
    Radial,
    /// An ellipse with the frame's proportions growing from `center`.
    Oval,
    /// A rectangle with the frame's proportions, turned by `angle`.
    Box,
    /// A diamond with the frame's proportions, turned by `angle`.
    Diamond,
    /// A plus sign whose arms widen, turned by `angle`.
    Cross,
    /// A star with `count` points and an `inner` radius, turned by `angle`.
    Star,
    /// Doors opening from a line through the centre (its normal at `angle`).
    BarnDoor,
    /// `count` slats, each wiped at `angle`.
    Blinds,
    /// A hand sweeping round `center` from `angle`.
    Clock,
    /// Two hands sweeping both ways round `center` from `angle`.
    Fan,
    /// `count` spiral arms turning round `center`.
    Spiral,
    /// A checkerboard of `count` squares per row: one colour of squares, then the other.
    Checker,
    /// `count` rings growing from `center`.
    Rings,
    /// A wavy edge (`count` waves) sweeping at `angle`.
    Wave,
    /// Blocks (`count` per row) appearing in random order.
    Blocks,
    /// Smooth value noise (`scale` pixels per cell).
    Noise,
    /// Worley cells (`count` per row) growing from their seeds.
    Worley,
    /// The cracks between Worley cells opening first.
    Crackle,
}

impl LumaShape {
    /// Parameter ids this shape reads (besides softness, feather, invert and the easing).
    pub fn params(self) -> &'static [&'static str] {
        use LumaShape::*;
        match self {
            Linear => &["angle"],
            Radial | Oval => &["center"],
            Box | Diamond | Cross => &["center", "angle"],
            Star => &["center", "count", "inner", "angle"],
            BarnDoor => &["angle"],
            Blinds | Wave => &["angle", "count"],
            Clock => &["center", "angle", "spin"],
            Fan => &["center", "angle"],
            Spiral => &["center", "count", "spin"],
            Checker => &["count"],
            Rings => &["center", "count"],
            Blocks | Worley | Crackle => &["count", "seed"],
            Noise => &["scale", "seed"],
        }
    }
}

/// One named luma wipe.
#[derive(Clone, Copy, Debug)]
pub struct LumaPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub shape: LumaShape,
    /// Default angle (degrees).
    pub angle: f64,
    /// Default count (slats, points, arms, squares, rings, waves, blocks or cells).
    pub count: f64,
    /// Default inner radius of a star (%).
    pub inner: f64,
    /// Default noise scale (sequence pixels).
    pub scale: f64,
    /// Default softness (% of the map's range).
    pub softness: f64,
    /// Default feather (sequence pixels).
    pub feather: f64,
    pub invert: bool,
    /// Default spin: counter-clockwise.
    pub ccw: bool,
}

const fn luma(id: &'static str, name: &'static str, shape: LumaShape) -> LumaPreset {
    LumaPreset { id, name, shape, angle: 0.0, count: 8.0, inner: 45.0, scale: 120.0, softness: 5.0, feather: 0.0, invert: false, ccw: false }
}

/// Every luma wipe, by id.
pub const LUMA_PRESETS: &[LumaPreset] = {
    use LumaShape::*;
    &[
        LumaPreset { angle: 45.0, ..luma("luma_barn_door_diagonal", "Luma Barn Door Diagonal", BarnDoor) },
        LumaPreset { angle: 0.0, ..luma("luma_barn_door_horizontal", "Luma Barn Door Horizontal", BarnDoor) },
        LumaPreset { angle: 90.0, ..luma("luma_barn_door_vertical", "Luma Barn Door Vertical", BarnDoor) },
        LumaPreset { angle: 135.0, count: 10.0, softness: 10.0, ..luma("luma_blinds_diagonal", "Luma Blinds Diagonal", Blinds) },
        LumaPreset { angle: 180.0, count: 8.0, softness: 10.0, ..luma("luma_blinds_horizontal", "Luma Blinds Horizontal", Blinds) },
        LumaPreset { angle: 90.0, count: 8.0, softness: 10.0, ..luma("luma_blinds_vertical", "Luma Blinds Vertical", Blinds) },
        LumaPreset { softness: 3.0, ..luma("luma_box", "Luma Box", Box) },
        LumaPreset { softness: 3.0, invert: true, ..luma("luma_box_in", "Luma Box In", Box) },
        LumaPreset { count: 12.0, softness: 10.0, ..luma("luma_cells", "Luma Cells", Worley) },
        LumaPreset { count: 8.0, softness: 2.0, ..luma("luma_checkerboard", "Luma Checkerboard", Checker) },
        LumaPreset { softness: 2.0, ..luma("luma_clock", "Luma Clock", Clock) },
        LumaPreset { softness: 2.0, ccw: true, ..luma("luma_clock_reverse", "Luma Clock Reverse", Clock) },
        LumaPreset { scale: 160.0, softness: 15.0, ..luma("luma_clouds", "Luma Clouds", Noise) },
        LumaPreset { count: 10.0, softness: 8.0, ..luma("luma_crackle", "Luma Crackle", Crackle) },
        LumaPreset { softness: 3.0, ..luma("luma_cross", "Luma Cross", Cross) },
        LumaPreset { angle: 135.0, ..luma("luma_diagonal", "Luma Diagonal", Linear) },
        LumaPreset { softness: 3.0, ..luma("luma_diamond", "Luma Diamond", Diamond) },
        LumaPreset { softness: 2.0, ..luma("luma_fan", "Luma Fan", Fan) },
        LumaPreset { angle: 90.0, ..luma("luma_linear", "Luma Linear", Linear) },
        LumaPreset { angle: 180.0, ..luma("luma_linear_vertical", "Luma Linear Vertical", Linear) },
        LumaPreset { scale: 6.0, softness: 10.0, ..luma("luma_noise", "Luma Noise", Noise) },
        LumaPreset { softness: 4.0, ..luma("luma_oval", "Luma Oval", Oval) },
        LumaPreset { softness: 4.0, ..luma("luma_radial", "Luma Radial", Radial) },
        LumaPreset { softness: 4.0, invert: true, ..luma("luma_radial_in", "Luma Radial In", Radial) },
        LumaPreset { count: 16.0, softness: 3.0, ..luma("luma_random_blocks", "Luma Random Blocks", Blocks) },
        LumaPreset { count: 6.0, softness: 6.0, ..luma("luma_rings", "Luma Rings", Rings) },
        LumaPreset { angle: 90.0, softness: 40.0, ..luma("luma_soft_linear", "Luma Soft Linear", Linear) },
        LumaPreset { count: 1.0, softness: 4.0, ..luma("luma_spiral", "Luma Spiral", Spiral) },
        LumaPreset { count: 4.0, softness: 4.0, ..luma("luma_spiral_arms", "Luma Spiral Arms", Spiral) },
        LumaPreset { count: 5.0, inner: 45.0, softness: 3.0, ..luma("luma_star", "Luma Star", Star) },
        LumaPreset { count: 12.0, inner: 35.0, softness: 3.0, ..luma("luma_starburst", "Luma Starburst", Star) },
        LumaPreset { angle: 90.0, count: 4.0, softness: 4.0, ..luma("luma_wave", "Luma Wave", Wave) },
    ]
};

/// The luma wipe with effect id `id`.
pub fn luma_preset(id: &str) -> Option<&'static LumaPreset> {
    LUMA_PRESETS.iter().find(|p| p.id == id)
}

/// One named blend dissolve: a blend mode (index into [`BLEND_MODES`]).
#[derive(Clone, Copy, Debug)]
pub struct BlendPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub mode: u32,
}

const fn blend(id: &'static str, name: &'static str, mode: u32) -> BlendPreset {
    BlendPreset { id, name, mode }
}

/// Every blend dissolve, by id: each blend mode but Normal (a plain cross dissolve). Dissolve's
/// own random pattern becomes a dither from one picture to the other.
pub const BLEND_PRESETS: &[BlendPreset] = &[
    blend("blend_color", "Color Dissolve", 25),
    blend("blend_color_burn", "Color Burn Dissolve", 4),
    blend("blend_color_dodge", "Color Dodge Dissolve", 9),
    blend("blend_darken", "Darken Dissolve", 2),
    blend("blend_darker_color", "Darker Color Dissolve", 6),
    blend("blend_difference", "Difference Dissolve", 19),
    blend("blend_dissolve", "Dither Dissolve", 1),
    blend("blend_divide", "Divide Dissolve", 22),
    blend("blend_exclusion", "Exclusion Dissolve", 20),
    blend("blend_hard_light", "Hard Light Dissolve", 14),
    blend("blend_hard_mix", "Hard Mix Dissolve", 18),
    blend("blend_hue", "Hue Dissolve", 23),
    blend("blend_lighten", "Lighten Dissolve", 7),
    blend("blend_lighter_color", "Lighter Color Dissolve", 11),
    blend("blend_linear_burn", "Linear Burn Dissolve", 5),
    blend("blend_linear_dodge", "Linear Dodge Dissolve", 10),
    blend("blend_linear_light", "Linear Light Dissolve", 16),
    blend("blend_luminosity", "Luminosity Dissolve", 26),
    blend("blend_multiply", "Multiply Dissolve", 3),
    blend("blend_overlay", "Overlay Dissolve", 12),
    blend("blend_pin_light", "Pin Light Dissolve", 17),
    blend("blend_saturation", "Saturation Dissolve", 24),
    blend("blend_screen", "Screen Dissolve", 8),
    blend("blend_soft_light", "Soft Light Dissolve", 13),
    blend("blend_subtract", "Subtract Dissolve", 21),
    blend("blend_vivid_light", "Vivid Light Dissolve", 15),
];

/// The blend dissolve with effect id `id`.
pub fn blend_preset(id: &str) -> Option<&'static BlendPreset> {
    BLEND_PRESETS.iter().find(|p| p.id == id)
}

/// Whether `id` is a transition of one of the engines.
pub fn is_engine_transition(id: &str) -> bool {
    luma_preset(id).is_some() || blend_preset(id).is_some()
}

/// Not animatable (transition parameters have no stopwatch).
fn fixed(mut p: ParamDef) -> ParamDef {
    p.animatable = false;
    p
}
fn num(id: &'static str, label: &'static str, def: f64, min: f64, max: f64, unit: &'static str, decimals: u8) -> ParamDef {
    fixed(fs(id, label, def, (min, max), (min, max), unit, decimals))
}

fn luma_def(p: &LumaPreset) -> EffectDef {
    use LumaShape::*;
    let mut params = Vec::new();
    for id in p.shape.params() {
        params.push(match *id {
            "center" => fixed(pt("center", "Center", f64::NAN, f64::NAN)),
            "angle" => {
                let label = match p.shape {
                    Linear | Blinds | Wave => "Wipe Angle",
                    Box | Diamond | Cross | Star => "Rotation",
                    Clock | Fan => "Start Angle",
                    _ => "Angle",
                };
                fixed(ang("angle", label, p.angle))
            }
            "count" => {
                let (label, min, max) = match p.shape {
                    Star => ("Points", 3.0, 24.0),
                    Blinds => ("Slats", 1.0, 64.0),
                    Spiral => ("Arms", 1.0, 12.0),
                    Checker => ("Squares per Row", 1.0, 64.0),
                    Rings => ("Rings", 1.0, 32.0),
                    Wave => ("Waves", 1.0, 32.0),
                    Blocks => ("Columns", 1.0, 200.0),
                    _ => ("Cells", 1.0, 100.0),
                };
                num("count", label, p.count, min, max, "", 0)
            }
            "inner" => num("inner", "Inner Radius", p.inner, 5.0, 95.0, "%", 0),
            "scale" => num("scale", "Scale", p.scale, 2.0, 2000.0, "px", 0),
            "seed" => num("seed", "Random Seed", 1.0, 0.0, 9999.0, "", 0),
            "spin" => ch("spin", "Spin Direction", SPIN_OPTS, u32::from(p.ccw)),
            _ => continue,
        });
    }
    params.push(num("softness", "Softness", p.softness, 0.0, 100.0, "%", 1));
    // (a blur leaves a straight ramp a straight ramp: a linear wipe's edge is Softness alone)
    if p.shape != Linear {
        params.push(num("feather", "Feather", p.feather, 0.0, 500.0, "px", 0));
    }
    params.push(b("invert", "Invert", p.invert));
    params.extend(crate::gltransition::easing());
    EffectDef {
        id: p.id,
        name: p.name,
        kind: EffectKind::VideoTransition,
        category: LUMA_WIPES,
        params,
        intrinsic: false,
        accelerated: true,
        float32: true,
        yuv: false,
    }
}

fn blend_def(p: &BlendPreset) -> EffectDef {
    let mut params = vec![ch("mode", "Blend Mode", BLEND_MODES, p.mode), num("strength", "Strength", 100.0, 0.0, 100.0, "%", 0)];
    params.extend(crate::gltransition::easing());
    EffectDef {
        id: p.id,
        name: p.name,
        kind: EffectKind::VideoTransition,
        category: BLEND_DISSOLVES,
        params,
        intrinsic: false,
        accelerated: true,
        float32: true,
        yuv: false,
    }
}

/// The engines' transitions: the luma wipes, then the blend dissolves (each sorted by name).
pub(crate) fn defs() -> Vec<EffectDef> {
    let by_name = |mut v: Vec<EffectDef>| {
        v.sort_by_key(|d| d.name.to_ascii_lowercase());
        v
    };
    let mut out = by_name(LUMA_PRESETS.iter().map(luma_def).collect());
    out.extend(by_name(BLEND_PRESETS.iter().map(blend_def).collect()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_listed_in_their_folders() {
        let all = crate::effect::effect_defs();
        for (cat, n) in [(LUMA_WIPES, LUMA_PRESETS.len()), (BLEND_DISSOLVES, BLEND_PRESETS.len())] {
            // (the GL shape wipes share the Luma Wipes folder)
            let names: Vec<String> = all.iter().filter(|d| d.category == cat).map(|d| d.name.to_ascii_lowercase()).collect();
            let mut sorted = names.clone();
            sorted.sort();
            assert_eq!(names, sorted, "{cat:?} sorted by name");
            let defs: Vec<_> = all.iter().filter(|d| d.category == cat && is_engine_transition(d.id)).collect();
            assert_eq!(defs.len(), n);
            for d in defs {
                assert!(is_engine_transition(d.id), "{}", d.id);
                assert_eq!(d.kind, EffectKind::VideoTransition);
                assert!(d.params.iter().all(|p| !p.animatable), "{}", d.id);
                assert_eq!(d.param("easing").map(|p| &p.default), Some(&crate::ParamValue::Choice(0)), "{}", d.id);
            }
        }
        assert!(LUMA_PRESETS.len() >= 30 && BLEND_PRESETS.len() == BLEND_MODES.len() - 1);
        // every shape has a preset, every blend mode but Normal has one
        use LumaShape::*;
        for s in [Linear, Radial, Oval, Box, Diamond, Cross, Star, BarnDoor, Blinds, Clock, Fan, Spiral, Checker, Rings, Wave, Blocks, Noise, Worley, Crackle] {
            assert!(LUMA_PRESETS.iter().any(|p| p.shape == s), "{s:?}");
        }
        for m in 1..BLEND_MODES.len() as u32 {
            assert!(BLEND_PRESETS.iter().any(|p| p.mode == m), "{}", BLEND_MODES[m as usize]);
        }
        let find = |n: &str| crate::vtransition::find_transition(n, EffectKind::VideoTransition).map(|d| d.id);
        assert_eq!(find("Luma Radial"), Some("luma_radial"));
        assert_eq!(find("screen dissolve"), Some("blend_screen"));
        assert_eq!(luma_preset("luma_barn_door_diagonal").map(|p| p.angle), Some(45.0));
        assert!(luma_preset("blend_screen").is_none() && blend_preset("luma_radial").is_none());
    }
}
