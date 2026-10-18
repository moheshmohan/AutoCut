//! Flow transitions (roadmap C3): transitions that look at the two clips they join.
//!
//! - **The flow engine** (`Video Transitions ▸ Flow & Morph`, ids `flow_*`): the dense optical flow
//!   between the outgoing clip at the cut and the incoming clip at the cut (Lucas–Kanade, both
//!   ways, smoothed) is computed once per transition and cached; every frame then warps each clip
//!   part of the way along it and dissolves between them, so shapes slide into each other instead
//!   of double-exposing (the "AI transition" morph). Its presets are styles on the same warp:
//!   Flow Morph (the plain morph), Doppler Warp (velocity-scaled radial streaks), Hype Ramp (a speed
//!   ramp with a motion blur, a punch-in and a brightness pulse peaking at the swap), Glitch Flow
//!   (datamosh blocks pushed along the flow, clustered at the cut) and Split Bands (bands sliding
//!   out alternately, the incoming clip reassembling behind them).
//! - **Content-Aware Wipe** (`content_aware_wipe`): a wipe whose edge is routed around what stands
//!   out in either picture, by seam carving (Avidan & Shamir, "Seam Carving for Content-Aware Image
//!   Resizing", SIGGRAPH 2007) on their gradient energy. A sequence of monotone seams across the
//!   frame sets when each pixel is reached; with no protection (or nothing salient) it is a
//!   straight wipe.
//!
//! The pictures, flow and seams are computed by `filmcraft-render` (`transitions::content`); like
//! the other engines, progress is eased on the CPU at the exact frame time first. Ids never change.

use crate::effect::{EffectDef, EffectKind, ParamDef, ch, fs, pt};
use crate::vtransition::{AXIS_OPTS, DIR_OPTS};

/// Category of the Flow & Morph folder.
pub const FLOW_MORPH: &[&str] = &["Video Transitions", "Flow & Morph"];

/// The style of a flow-engine transition (the shader's mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowStyle {
    /// The two clips warped towards each other along the flow, dissolving in the middle.
    Morph,
    /// The morph through a zoom with radial streaks as long as the motion is fast.
    Doppler,
    /// The morph on a speed ramp: motion blur, punch-in and a brightness pulse at the swap.
    HypeRamp,
    /// The morph broken into datamosh blocks pushed along the flow, clustered at the cut.
    Glitch,
    /// Bands sliding out alternately, the incoming clip reassembling behind them.
    SplitBands,
    /// The outgoing picture dragged along the flow until blocks refresh to the incoming one.
    Datamosh,
}

impl FlowStyle {
    /// The engine shader's mode number.
    pub fn mode(self) -> u32 {
        match self {
            FlowStyle::Morph => 0,
            FlowStyle::Doppler => 1,
            FlowStyle::HypeRamp => 2,
            FlowStyle::Glitch => 3,
            FlowStyle::SplitBands => 4,
            FlowStyle::Datamosh => 5,
        }
    }
}

/// One named flow transition.
#[derive(Clone, Copy, Debug)]
pub struct FlowPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub style: FlowStyle,
    /// Its Effects-panel folder.
    pub category: &'static [&'static str],
}

const fn flow(id: &'static str, name: &'static str, style: FlowStyle) -> FlowPreset {
    FlowPreset { id, name, style, category: FLOW_MORPH }
}

/// Every flow transition, by id.
pub const FLOW_PRESETS: &[FlowPreset] = &[
    flow("flow_datamosh", "Datamosh", FlowStyle::Datamosh),
    flow("flow_doppler_warp", "Doppler Warp", FlowStyle::Doppler),
    flow("flow_glitch", "Glitch Flow", FlowStyle::Glitch),
    flow("flow_hype_ramp", "Hype Ramp", FlowStyle::HypeRamp),
    flow("flow_morph", "Flow Morph", FlowStyle::Morph),
    flow("flow_split_bands", "Split Bands", FlowStyle::SplitBands),
];

/// The flow transition with effect id `id`.
pub fn flow_preset(id: &str) -> Option<&'static FlowPreset> {
    FLOW_PRESETS.iter().find(|p| p.id == id)
}

/// Id of the Content-Aware Wipe.
pub const CONTENT_AWARE_WIPE: &str = "content_aware_wipe";

/// Whether transition `id` reads the pictures it joins at the cut (the flow transitions and the
/// Content-Aware Wipe): the renderer computes their flow or seams once per transition.
pub fn reads_content(id: &str) -> bool {
    flow_preset(id).is_some() || id == CONTENT_AWARE_WIPE
}

/// Not animatable (transition parameters have no stopwatch).
fn fixed(mut p: ParamDef) -> ParamDef {
    p.animatable = false;
    p
}
fn num(id: &'static str, label: &'static str, def: f64, min: f64, max: f64, unit: &'static str, decimals: u8) -> ParamDef {
    fixed(fs(id, label, def, (min, max), (min, max), unit, decimals))
}

/// The parameters of a flow transition of `style` (before the easing).
fn flow_params(style: FlowStyle) -> Vec<ParamDef> {
    use FlowStyle::*;
    let strength = num("strength", "Flow Strength", 100.0, 0.0, 200.0, "%", 0);
    let smoothness = num("smoothness", "Flow Smoothness", 30.0, 0.0, 100.0, "%", 0);
    let width = num("width", "Dissolve Width", 60.0, 5.0, 100.0, "%", 0);
    let mut v = vec![strength, smoothness];
    match style {
        Morph => v.push(width),
        Doppler => v.extend([
            width,
            num("streaks", "Streak Length", 60.0, 0.0, 100.0, "%", 0),
            num("zoom", "Zoom", 30.0, 0.0, 100.0, "%", 0),
            fixed(pt("center", "Center", f64::NAN, f64::NAN)),
        ]),
        HypeRamp => v.extend([
            num("ramp", "Speed Ramp", 4.0, 0.5, 10.0, "", 1),
            num("blur", "Blur", 60.0, 0.0, 100.0, "%", 0),
            num("pulse", "Brightness Pulse", 80.0, 0.0, 200.0, "%", 0),
            num("zoom", "Punch Zoom", 25.0, 0.0, 100.0, "%", 0),
        ]),
        Glitch => v.extend([
            width,
            num("block_size", "Block Size", 48.0, 4.0, 400.0, "px", 0),
            num("density", "Glitch Amount", 60.0, 0.0, 100.0, "%", 0),
            num("rgb_split", "RGB Split", 12.0, 0.0, 200.0, "px", 0),
            num("smear", "Smear", 150.0, 0.0, 400.0, "%", 0),
            num("seed", "Random Seed", 1.0, 0.0, 9999.0, "", 0),
        ]),
        SplitBands => v.extend([
            num("bands", "Bands", 6.0, 2.0, 32.0, "", 0),
            ch("orientation", "Orientation", AXIS_OPTS, 0),
            num("stagger", "Stagger", 40.0, 0.0, 90.0, "%", 0),
            num("motion_blur", "Motion Blur", 50.0, 0.0, 100.0, "%", 0),
        ]),
        Datamosh => v.extend([
            num("block_size", "Block Size", 32.0, 4.0, 400.0, "px", 0),
            num("smear", "Smear", 200.0, 0.0, 400.0, "%", 0),
            num("spread", "Refresh Spread", 60.0, 0.0, 100.0, "%", 0),
            num("seed", "Random Seed", 1.0, 0.0, 9999.0, "", 0),
        ]),
    }
    v
}

fn def(id: &'static str, name: &'static str, category: &'static [&'static str], mut params: Vec<ParamDef>) -> EffectDef {
    params.extend(crate::gltransition::easing());
    EffectDef { id, name, kind: EffectKind::VideoTransition, category, params, intrinsic: false, accelerated: true, float32: true, yuv: false }
}

/// The flow-engine transition of preset `p`.
pub(crate) fn flow_def(p: &FlowPreset) -> EffectDef {
    def(p.id, p.name, p.category, flow_params(p.style))
}

/// The Content-Aware Wipe: Direction, Softness (the moving edge's width, % of the wipe), Subject
/// Protection (how much the seams avoid salient pixels; 0 is a straight wipe) and Seam Freedom
/// (how far an edge may bend away from the straight line).
pub(crate) fn content_aware_wipe_def() -> EffectDef {
    def(
        CONTENT_AWARE_WIPE,
        "Content-Aware Wipe",
        FLOW_MORPH,
        vec![
            ch("direction", "Direction", DIR_OPTS, 3),
            num("softness", "Softness", 4.0, 0.0, 100.0, "%", 1),
            num("protect", "Subject Protection", 80.0, 0.0, 100.0, "%", 0),
            num("freedom", "Seam Freedom", 60.0, 0.0, 100.0, "%", 0),
        ],
    )
}

/// The Flow & Morph folder (sorted by name).
pub(crate) fn defs() -> Vec<EffectDef> {
    let mut v: Vec<EffectDef> = FLOW_PRESETS.iter().filter(|p| p.category == FLOW_MORPH).map(flow_def).collect();
    v.push(content_aware_wipe_def());
    v.sort_by_key(|d| d.name.to_ascii_lowercase());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_transitions_are_listed_in_their_folder() {
        let all = crate::effect::effect_defs();
        let defs: Vec<_> = all.iter().filter(|d| d.category == FLOW_MORPH).collect();
        assert!(defs.len() >= 6, "{}", defs.len());
        let names: Vec<String> = defs.iter().map(|d| d.name.to_ascii_lowercase()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "sorted by name");
        for d in &defs {
            assert!(reads_content(d.id), "{}", d.id);
            assert_eq!(d.kind, EffectKind::VideoTransition);
            assert!(d.params.iter().all(|p| !p.animatable), "{}", d.id);
            assert_eq!(d.param("easing").map(|p| &p.default), Some(&crate::ParamValue::Choice(0)), "{}", d.id);
        }
        // every style has a preset, modes are distinct
        use FlowStyle::*;
        let styles = [Morph, Doppler, HypeRamp, Glitch, SplitBands, Datamosh];
        for s in styles {
            assert!(FLOW_PRESETS.iter().any(|p| p.style == s), "{s:?}");
        }
        let mut modes: Vec<u32> = [Morph, Doppler, HypeRamp, Glitch, SplitBands, Datamosh].iter().map(|s| s.mode()).collect();
        modes.dedup();
        assert_eq!(modes, vec![0, 1, 2, 3, 4, 5]);
        let find = |n: &str| crate::vtransition::find_transition(n, EffectKind::VideoTransition).map(|d| d.id);
        assert_eq!(find("Flow Morph"), Some("flow_morph"));
        assert_eq!(find("content-aware wipe"), Some(CONTENT_AWARE_WIPE));
        assert!(!reads_content("cross_dissolve") && !reads_content("morph_cut"));
    }
}
