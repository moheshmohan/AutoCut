//! Opt-in creator looks for the trailer and the shorts: `look` (`--look tech|vhs|clean`) and
//! `beatPulse` (`--beat-pulse`). Off unless asked: the showcase, the trailer and the shorts build
//! as before without them.
//!
//! **Looks** go on an adjustment clip over the stretch they cover (`timeline.newAdjustmentClip`),
//! so they sit in frame space whatever each clip's crop and scale (the HUD's corner brackets in
//! the frame's corners, grain the same size on every shot):
//! - `tech` — the robot channel: Tech HUD in the theme's accent colour (scanlines, a faint grid,
//!   corner brackets, vignette), a touch of chromatic aberration and fine grain;
//! - `vhs` — tape: VHS (chroma bleed, tracking, wobble, noise) and grain;
//! - `clean` — nothing (the default).
//!
//! **Beat pulse** keyframes a short zoom pulse on the given clips at every other beat
//! (`sequence.beatPulse`, Motion ▸ Scale +7 %).

use serde_json::{Value, json};

use crate::showcase::{run, ticks};
use crate::{AutoCutError, Executor};

/// A creator look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreatorLook {
    Clean,
    Tech,
    Vhs,
}

impl CreatorLook {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "clean" | "none" => CreatorLook::Clean,
            "tech" | "hud" => CreatorLook::Tech,
            "vhs" | "tape" => CreatorLook::Vhs,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            CreatorLook::Clean => "clean",
            CreatorLook::Tech => "tech",
            CreatorLook::Vhs => "vhs",
        }
    }

    /// The effects of the look and their settings, in order.
    pub fn effects(self, accent: [f32; 4]) -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
        match self {
            CreatorLook::Clean => Vec::new(),
            CreatorLook::Tech => vec![
                ("cr_chromatic_aberration", vec![("amount", json!(3.0))]),
                (
                    "cr_tech_hud",
                    vec![
                        ("color", json!(accent)),
                        ("scanlines", json!(28.0)),
                        ("grid", json!(8.0)),
                        ("brackets", json!(85.0)),
                        ("vignette", json!(40.0)),
                        ("tint", json!(10.0)),
                        ("flicker", json!(6.0)),
                    ],
                ),
                ("cr_film_grain", vec![("amount", json!(14.0)), ("size", json!(1.2))]),
            ],
            CreatorLook::Vhs => vec![
                ("cr_vhs", vec![("amount", json!(60.0)), ("tracking", json!(30.0)), ("noise", json!(20.0))]),
                ("cr_film_grain", vec![("amount", json!(22.0)), ("size", json!(2.0))]),
            ],
        }
    }
}

/// `look` from checked parameters (absent: none).
pub fn look_of(p: &Value) -> Result<Option<CreatorLook>, AutoCutError> {
    match p.get("look").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v.as_str().and_then(CreatorLook::parse).map(Some).ok_or_else(|| AutoCutError::Invalid("look: tech, vhs or clean".into())),
    }
}

/// A `#RRGGBB` colour as linear-free RGBA 0..1 (the effect's colour parameter); a cool cyan when
/// it does not parse.
pub fn hex_color(hex: &str) -> [f32; 4] {
    let h = hex.trim().trim_start_matches('#');
    let byte = |i: usize| h.get(i..i + 2).and_then(|s| u8::from_str_radix(s, 16).ok()).map(|b| b as f32 / 255.0);
    match (h.len(), byte(0), byte(2), byte(4)) {
        (6, Some(r), Some(g), Some(b)) => [r, g, b, 1.0],
        _ => filmcraft_engine::project::effect::HUD_COLOR,
    }
}

/// Put `look` over `[from, to)` seconds of the active sequence on an adjustment clip. Returns
/// the clip, or None for Clean.
pub async fn apply_look(ex: &dyn Executor, look: CreatorLook, from: f64, to: f64, accent: [f32; 4]) -> Result<Option<u64>, AutoCutError> {
    let effects = look.effects(accent);
    if effects.is_empty() || !(to.is_finite() && from.is_finite()) || to <= from {
        return Ok(None);
    }
    let ids: Vec<&str> = effects.iter().map(|(id, _)| *id).collect();
    let r = run(
        ex,
        "timeline.newAdjustmentClip",
        json!({"startTime": ticks(from), "duration": ticks(to - from), "effects": ids, "name": format!("Look: {}", look.name())}),
    )
    .await?;
    let clip = r["clip"].as_u64().ok_or_else(|| AutoCutError::Executor(format!("timeline.newAdjustmentClip made no clip: {r}")))?;
    for (effect, params) in effects {
        for (param, value) in params {
            run(ex, "effects.setParam", json!({"clip": clip, "effect": effect, "param": param, "value": value})).await?;
        }
    }
    Ok(Some(clip))
}

/// Zoom pulses on `clips` at every other beat of `beats` (seconds; None: the sequence's music).
pub async fn beat_pulse(ex: &dyn Executor, clips: &[u64], beats: Option<&[f64]>) -> Result<Value, AutoCutError> {
    let mut p = json!({"clips": clips, "effect": "zoomPulse", "strength": 0.35, "everyNthBeat": 2, "decay": 0.22});
    if let Some(b) = beats {
        p["beats"] = json!(b);
    }
    run(ex, "sequence.beatPulse", p).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_parse_and_name_their_effects() {
        assert_eq!(CreatorLook::parse("Tech"), Some(CreatorLook::Tech));
        assert_eq!(CreatorLook::parse(" vhs "), Some(CreatorLook::Vhs));
        assert_eq!(CreatorLook::parse("clean"), Some(CreatorLook::Clean));
        assert_eq!(CreatorLook::parse("sparkly"), None);
        assert!(CreatorLook::Clean.effects([1.0; 4]).is_empty());
        for look in [CreatorLook::Tech, CreatorLook::Vhs] {
            for (id, params) in look.effects([0.2, 0.8, 1.0, 1.0]) {
                let d = filmcraft_engine::project::find_effect(id).unwrap_or_else(|| panic!("{id}"));
                for (k, _) in params {
                    assert!(d.param(k).is_some(), "{id}.{k}");
                }
            }
        }
        assert_eq!(hex_color("#FF8000"), [1.0, 128.0 / 255.0, 0.0, 1.0]);
        assert_eq!(hex_color("nope"), filmcraft_engine::project::effect::HUD_COLOR);
        assert!(look_of(&json!({"look": "x"})).is_err());
        assert_eq!(look_of(&json!({})).unwrap(), None);
    }
}
