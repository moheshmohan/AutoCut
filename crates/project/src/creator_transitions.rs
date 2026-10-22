//! The Creator transitions (`Video Transitions ▸ Creator`): the cuts short-form editors use most —
//! Pull In, Mix, White Flash, Swipe Left… — as tuned presets of FilmCraft's own transitions, each
//! with a short default length (0.3–0.5 s) and a paired whoosh or impact.
//!
//! A preset is a transition of its own (id `crt_*`, its own name in the Effects panel and Effect
//! Controls; where a Premiere transition already has the short name, the preset's says more:
//! Glitch Cut, Light Leak Fade, Pixelate, Slices) whose parameters are its base transition's with tuned defaults. Rendering resolves
//! it to the base ([`base_id`], [`resolve`]): the base's CPU reference and, where the base is a
//! shader transition, its WGSL on the GPU. Every base here is drawn by the GPU (the camera, light
//! and pattern engines, GL Transitions, cross dissolve, dip to black). No new shader was needed.
//!
//! The paired sound ([`paired_sound`]) is what `transitions.addSound` couples when it is not told
//! a kind, and what `transitions.applyToAll` / `transitions.randomOnBeats` add with `sound`.

use crate::effect::{EffectDef, EffectInstance, EffectKind};
use crate::keyframe::ParamValue;
use crate::transition_sound::SoundKind;

/// The Creator folder of the transitions.
pub const CREATOR_TRANSITIONS_FOLDER: &[&str] = &["Video Transitions", "Creator"];

/// A Creator preset: its base transition and the defaults it changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreatorTransition {
    pub id: &'static str,
    pub name: &'static str,
    /// The transition that draws it.
    pub base: &'static str,
    /// Parameter defaults over the base's (numbers; choices by index).
    pub params: &'static [(&'static str, f64)],
    /// Default length, seconds.
    pub seconds: f64,
    /// The whoosh, riser or impact that goes with it.
    pub sound: Option<SoundKind>,
    /// Words a search finds it by (lower case), besides its name.
    pub keywords: &'static [&'static str],
}

const fn ct(
    id: &'static str,
    name: &'static str,
    base: &'static str,
    params: &'static [(&'static str, f64)],
    seconds: f64,
    sound: Option<SoundKind>,
    keywords: &'static [&'static str],
) -> CreatorTransition {
    CreatorTransition { id, name, base, params, seconds, sound, keywords }
}

/// `easing` index of Cubic In-Out / Quad Out / Expo In-Out in [`crate::gltransition::EASING_OPTS`].
const CUBIC_IN_OUT: f64 = 9.0;
const QUAD_OUT: f64 = 5.0;
const EXPO_IN_OUT: f64 = 18.0;

/// Every Creator preset, in panel order.
pub const CREATOR_TRANSITIONS: &[CreatorTransition] = {
    use SoundKind::{Impact, Riser, Whoosh};
    &[
        ct(
            "crt_pull_in",
            "Pull In",
            "cam_punch_in",
            &[("zoom", 90.0), ("motion_blur", 75.0), ("easing", EXPO_IN_OUT)],
            0.4,
            Some(Whoosh),
            &["zoom in", "punch", "push"],
        ),
        ct(
            "crt_pull_out",
            "Pull Out",
            "cam_pull_out",
            &[("zoom", 90.0), ("motion_blur", 75.0), ("easing", EXPO_IN_OUT)],
            0.4,
            Some(Whoosh),
            &["zoom out", "pull back"],
        ),
        ct("crt_mix", "Mix", "cross_dissolve", &[], 0.5, None, &["dissolve", "crossfade", "fade"]),
        ct("crt_black_fade", "Black Fade", "dip_to_black", &[], 0.5, None, &["fade to black", "dip"]),
        ct("crt_white_flash", "White Flash", "lf_exposure_flash", &[("intensity", 150.0)], 0.3, Some(Impact), &["flash", "white", "blink"]),
        ct("crt_flash_zoom", "Flash Zoom", "gl_crosszoom", &[("strength", 0.7), ("easing", CUBIC_IN_OUT)], 0.4, Some(Impact), &["flash", "zoom", "blur"]),
        ct("crt_zoom_shake", "Zoom Shake", "cam_shake", &[("shake", 60.0), ("speed", 18.0)], 0.4, Some(Impact), &["shake", "impact", "earthquake", "zoom"]),
        ct(
            "crt_spin",
            "Spin",
            "cam_spin_zoom",
            &[("rotation", 360.0), ("zoom", 50.0), ("motion_blur", 80.0), ("easing", CUBIC_IN_OUT)],
            0.5,
            Some(Whoosh),
            &["spin", "rotate", "twist"],
        ),
        ct(
            "crt_swipe_left",
            "Swipe Left",
            "cam_whip_pan",
            &[("direction", 1.0), ("blur", 85.0), ("easing", CUBIC_IN_OUT)],
            0.35,
            Some(Whoosh),
            &["swipe", "slide", "whip", "left"],
        ),
        ct(
            "crt_swipe_right",
            "Swipe Right",
            "cam_whip_pan",
            &[("direction", 3.0), ("blur", 85.0), ("easing", CUBIC_IN_OUT)],
            0.35,
            Some(Whoosh),
            &["swipe", "slide", "whip", "right"],
        ),
        ct(
            "crt_swipe_up",
            "Swipe Up",
            "cam_whip_pan",
            &[("direction", 2.0), ("blur", 85.0), ("easing", CUBIC_IN_OUT)],
            0.35,
            Some(Whoosh),
            &["swipe", "slide", "whip", "up"],
        ),
        ct(
            "crt_swipe_down",
            "Swipe Down",
            "cam_whip_pan",
            &[("direction", 0.0), ("blur", 85.0), ("easing", CUBIC_IN_OUT)],
            0.35,
            Some(Whoosh),
            &["swipe", "slide", "whip", "down"],
        ),
        ct("crt_glitch", "Glitch Cut", "glitch_rgb_split", &[("shift", 40.0), ("bands", 32.0)], 0.3, Some(Impact), &["glitch", "rgb", "digital", "error"]),
        ct("crt_light_leak", "Light Leak Fade", "lf_light_leak", &[("amount", 90.0)], 0.5, Some(Riser), &["leak", "light", "warm", "film"]),
        ct("crt_burn", "Burn", "lf_burn_through", &[("edge", 10.0)], 0.5, Some(Riser), &["burn", "fire", "film"]),
        ct("crt_blur", "Blur", "lf_bokeh", &[("radius", 45.0), ("boost", 30.0), ("easing", QUAD_OUT)], 0.4, None, &["blur", "defocus", "soft"]),
        ct(
            "crt_mosaic",
            "Pixelate",
            "gl_pixelize",
            &[("squares_min_x", 24.0), ("squares_min_y", 14.0), ("steps", 18.0)],
            0.4,
            Some(Whoosh),
            &["mosaic", "pixel", "blocks"],
        ),
        ct("crt_ripple", "Ripple", "gl_ripple", &[("amplitude", 80.0), ("speed", 40.0)], 0.5, Some(Riser), &["ripple", "water", "wave"]),
        ct(
            "crt_cube",
            "Cube",
            "gl_cube",
            &[("persp", 0.6), ("unzoom", 0.2), ("reflection", 0.0), ("easing", CUBIC_IN_OUT)],
            0.5,
            Some(Whoosh),
            &["cube", "3d", "box", "rotate"],
        ),
        ct("crt_3d_flip", "3D Flip", "sp_card_flip", &[("perspective", 60.0), ("easing", CUBIC_IN_OUT)], 0.5, Some(Whoosh), &["flip", "3d", "card", "turn"]),
        ct("crt_slice", "Slices", "gl_windowslice", &[("count", 8.0), ("smoothness", 0.3)], 0.4, Some(Whoosh), &["slice", "stripes", "blinds"]),
    ]
};

/// The Creator preset with effect id `id`.
pub fn creator_transition(id: &str) -> Option<&'static CreatorTransition> {
    CREATOR_TRANSITIONS.iter().find(|c| c.id == id)
}

/// The preset by id or name (case-insensitive; `pull in`, `Pull In`, `crt_pull_in`, `pullIn`).
pub fn find_creator(name: &str) -> Option<&'static CreatorTransition> {
    let squash = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_ascii_lowercase();
    let n = squash(name);
    CREATOR_TRANSITIONS.iter().find(|c| c.id == name || squash(c.name) == n || squash(c.id.trim_start_matches("crt_")) == n)
}

/// The transition that draws `id`: its base for a Creator preset, else `id` itself.
pub fn base_id(id: &str) -> &str {
    creator_transition(id).map_or(id, |c| c.base)
}

/// The instance the renderers draw: a Creator preset becomes its base (same parameters).
pub fn resolve(e: &EffectInstance) -> std::borrow::Cow<'_, EffectInstance> {
    match creator_transition(&e.effect) {
        Some(c) => {
            let mut b = e.clone();
            b.effect = c.base.to_string();
            std::borrow::Cow::Owned(b)
        }
        None => std::borrow::Cow::Borrowed(e),
    }
}

/// The paired sound of transition `id` (Creator presets only).
pub fn paired_sound(id: &str) -> Option<SoundKind> {
    creator_transition(id).and_then(|c| c.sound)
}

/// The definitions: each base definition's parameters with the preset's defaults.
pub(crate) fn defs(all: &[EffectDef]) -> Vec<EffectDef> {
    let mut out = Vec::with_capacity(CREATOR_TRANSITIONS.len());
    for c in CREATOR_TRANSITIONS {
        let Some(base) = all.iter().find(|d| d.id == c.base && d.kind == EffectKind::VideoTransition) else { continue };
        let mut d = base.clone();
        d.id = c.id;
        d.name = c.name;
        d.category = CREATOR_TRANSITIONS_FOLDER;
        for (k, v) in c.params {
            if let Some(p) = d.params.iter_mut().find(|p| p.id == *k) {
                p.default = match p.default {
                    ParamValue::Choice(_) => ParamValue::Choice(v.max(0.0) as u32),
                    ParamValue::Bool(_) => ParamValue::Bool(*v != 0.0),
                    _ => ParamValue::Float(*v),
                };
            }
        }
        out.push(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::find_effect;

    #[test]
    fn every_preset_is_a_transition_over_an_existing_base_with_its_tuned_defaults() {
        assert_eq!(CREATOR_TRANSITIONS.len(), 21);
        for c in CREATOR_TRANSITIONS {
            let d = find_effect(c.id).unwrap_or_else(|| panic!("{}", c.id));
            let base = find_effect(c.base).unwrap_or_else(|| panic!("{} base {}", c.id, c.base));
            assert_eq!(d.kind, EffectKind::VideoTransition);
            assert_eq!(base.kind, EffectKind::VideoTransition);
            assert_eq!(d.category, CREATOR_TRANSITIONS_FOLDER);
            assert!(creator_transition(c.base).is_none(), "{}: a base is not itself a preset", c.id);
            assert!((0.3..=0.5).contains(&c.seconds), "{}: {} s", c.id, c.seconds);
            assert_eq!(d.params.len(), base.params.len(), "{}", c.id);
            for (k, v) in c.params {
                let p = d.param(k).unwrap_or_else(|| panic!("{}: no param {k} on {}", c.id, c.base));
                assert_eq!(p.default.as_f64(), Some(*v), "{}.{k}", c.id);
            }
            assert!(c.keywords.iter().all(|k| k.chars().all(|ch| !ch.is_uppercase())));
            assert_eq!(base_id(c.id), c.base);
            let inst = d.instance();
            assert_eq!(resolve(&inst).effect, c.base);
            assert_eq!(find_creator(c.name).map(|x| x.id), Some(c.id));
        }
        assert_eq!(base_id("cross_dissolve"), "cross_dissolve");
        assert_eq!(find_creator("swipeLeft").map(|c| c.id), Some("crt_swipe_left"));
        assert_eq!(find_creator("3d flip").map(|c| c.id), Some("crt_3d_flip"));
        assert!(find_creator("nope").is_none());
        assert_eq!(paired_sound("crt_white_flash"), Some(SoundKind::Impact));
        assert_eq!(paired_sound("crt_mix"), None);
        assert_eq!(paired_sound("cross_dissolve"), None);
    }
}
