//! Speed ramps (`clip.speedRamp`, Clip ▸ Speed Ramp…): one of the
//! [`filmcraft_project::time_remap::RAMP_PRESETS`] written as the clip's Time Remapping speed
//! curve.
//!
//! | param | meaning |
//! |---|---|
//! | `clips` | the clips (default: the selection); linked partners (the clip's sound) ramp with them |
//! | `preset` | `montage`, `hero`, `bullet`, `jumpCut`, `flashIn`, `flashOut`, `smoothIn`, `smoothOut` (ids or names), or `none` to take a ramp off |
//! | `intensity` | 0 … 2 (default 1): each speed's distance from the clip's speed, in log space |
//! | `ripple` | default true: later clips move with the new length (false: the ramp is cut at the next clip) |
//! | `opticalFlow` | default true: a ramp with slow parts interpolates them with Optical Flow |
//! | `maintainPitch` | absent leaves each clip's Maintain Audio Pitch as it is |
//!
//! The clip keeps the media it plays (In to Out); its length becomes that media over the curve's
//! mean speed, on whole frames (as Speed/Duration), and its speed the average one. Speed
//! keyframes are Bézier (Jump Cut: held), relative to the clip's speed before the ramp. One undo
//! step for every clip. A reversed or frame-held clip is refused: its media does not run forward.

use serde_json::{Value, json};

use filmcraft_edit as edit;
use filmcraft_project::time_remap::{self, MAX_INTENSITY, RAMP_PRESETS};
use filmcraft_project::{ClipId, ParamValue, Sequence, TimeInterpolation, TrackKind};

use crate::commands::{CommandSpec, bad, bool_p, clips_p, f64_p, has_selection, str_p, with_links};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.speedRamp";

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Speed Ramp…",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"clips":[id]?,"preset":"montage|hero|bullet|jumpCut|flashIn|flashOut|smoothIn|smoothOut|none","intensity":0..2=1,"ripple":bool=true,"opticalFlow":bool=true,"maintainPitch":bool?}"#,
        enabled: has_selection,
        run: speed_ramp,
        journal: true,
    }]
}

/// The clips of `sel` grouped with their linked partners on other tracks (one group ramps as one
/// Speed/Duration edit, so a ripple moves the later clips once).
fn groups(q: &Sequence, sel: &[ClipId]) -> Vec<Vec<ClipId>> {
    let place = |c: ClipId| q.find_item(c).map(|(t, i)| (t, i.link));
    let mut groups: Vec<Vec<ClipId>> = Vec::new();
    for c in sel {
        let Some((track, link)) = place(*c) else {
            groups.push(vec![*c]);
            continue;
        };
        let joins = |g: &Vec<ClipId>| link.is_some() && g.iter().all(|o| place(*o).is_some_and(|(t, l)| l == link && t != track));
        match groups.iter_mut().find(|g| joins(g)) {
            Some(g) => g.push(*c),
            None => groups.push(vec![*c]),
        }
    }
    groups
}

/// Put `keys` on the clip's Time Remapping Speed (adding the effect to a clip without one, such
/// as a sound), the static value the clip's speed.
fn set_curve(it: &mut filmcraft_project::TrackItem, keys: Vec<filmcraft_project::Keyframe>) -> Result<()> {
    let pct = it.speed.abs() * 100.0;
    match it.effects.iter_mut().find(|e| e.effect == "time_remap") {
        Some(e) => {
            e.enabled = true;
            let p = e.params.entry("speed".into()).or_insert_with(|| filmcraft_project::Param::new(ParamValue::Float(pct)));
            p.value = ParamValue::Float(pct);
            p.keyframes = keys;
        }
        None => {
            let e = time_remap::remap_effect(keys, pct).ok_or_else(|| bad(CMD, "no Time Remapping effect definition"))?;
            it.effects.push(e);
        }
    }
    Ok(())
}

fn speed_ramp(s: &mut Session, p: &Value) -> Result<Value> {
    let name = str_p(p, "preset").ok_or_else(|| bad(CMD, "`preset` is one of montage, hero, bullet, jumpCut, flashIn, flashOut, smoothIn, smoothOut, none"))?;
    let preset = if name.eq_ignore_ascii_case("none") {
        None
    } else {
        let ids: Vec<&str> = RAMP_PRESETS.iter().map(|r| r.id).collect();
        Some(time_remap::ramp_preset(name).ok_or_else(|| bad(CMD, format!("unknown preset `{name}` ({}, none)", ids.join(", "))))?)
    };
    let intensity = match p.get("intensity") {
        None | Some(Value::Null) => 1.0,
        Some(_) => f64_p(p, "intensity")
            .filter(|i| i.is_finite() && (0.0..=MAX_INTENSITY).contains(i))
            .ok_or_else(|| bad(CMD, format!("`intensity` is from 0 to {MAX_INTENSITY}")))?,
    };
    let ripple = bool_p(p, "ripple").unwrap_or(true);
    let flow = bool_p(p, "opticalFlow").unwrap_or(true);
    let keep_pitch = match p.get("maintainPitch") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_bool().ok_or_else(|| bad(CMD, "`maintainPitch` must be true or false"))?),
    };
    let sel = with_links(s, &clips_p(s, p));
    if sel.is_empty() {
        return Err(bad(CMD, "no clips given or selected"));
    }
    let label = match preset {
        Some(r) => format!("Speed Ramp: {}", r.name),
        None => "Remove Speed Ramp".to_string(),
    };
    let report = s.edit_sequence(&label, |q, ctx, _| {
        let mut report = Vec::new();
        for group in groups(q, &sel) {
            let mut members: Vec<ClipId> = Vec::new();
            for c in &group {
                let (t, it) = q.find_item(*c).ok_or(edit::EditError::NoItem(*c))?;
                if q.track(t).is_some_and(|tr| tr.locked) {
                    continue;
                }
                if it.reverse || it.frame_hold.is_some() {
                    return Err(bad(CMD, format!("clip {} plays in reverse or is frame-held: a speed ramp needs media running forward", c.0)));
                }
                members.push(*c);
            }
            if members.is_empty() {
                return Err(edit::EditError::Locked.into());
            }
            // the picture's clip leads (its speed and media span), else the first member
            let first = members.first().copied().ok_or(edit::EditError::Locked)?;
            let lead =
                members.iter().copied().find(|c| q.find_item(*c).and_then(|(t, _)| q.track(t)).is_some_and(|t| t.kind == TrackKind::Video)).unwrap_or(first);
            let (_, it) = q.find_item(lead).ok_or(edit::EditError::NoItem(lead))?;
            let base_pct = it.speed.abs() * 100.0;
            let Some(preset) = preset else {
                // take the ramp off: the constant (average) speed, as it plays now on average
                for c in &members {
                    if let Some((_, it)) = q.find_item_mut(*c)
                        && let Some(sp) = it.effects.iter_mut().find(|e| e.effect == "time_remap").and_then(|e| e.params.get_mut("speed"))
                    {
                        sp.keyframes.clear();
                        sp.value = ParamValue::Float(base_pct);
                    }
                }
                report.extend(members.iter().map(|c| json!({"clip": c.0, "removed": true})));
                continue;
            };
            let span = it.source_out() - it.source_in;
            let keys = time_remap::ramp_keyframes(preset, intensity, it.source_in, span, base_pct);
            let mean =
                time_remap::ramp_mean(&keys, it.source_in, span, base_pct).ok_or_else(|| bad(CMD, format!("clip {} is too short for a speed ramp", lead.0)))?;
            let new_speed = it.speed.abs() * mean;
            edit::set_speed_group(q, &members, new_speed, false, ripple, ctx)?;
            let slow = preset.points.iter().any(|(_, sp)| sp.max(1e-3).powf(intensity) < 0.999);
            for c in &members {
                let Some((_, it)) = q.find_item_mut(*c) else { continue };
                // the curve over the media this member now plays, relative to the old speed
                let keys = time_remap::ramp_keyframes(preset, intensity, it.source_in, it.source_out() - it.source_in, base_pct);
                set_curve(it, keys)?;
                if let Some(k) = keep_pitch {
                    it.maintain_pitch = k;
                }
                report.push(json!({
                    "clip": c.0, "start": it.start.0, "duration": it.duration.0, "end": it.end().0,
                    "speed": it.speed, "meanSpeed": mean, "keyframes": time_remap::curve_of(it).map_or(0, |p| p.keyframes.len()),
                }));
            }
            // slow parts look smoother interpolated (the picture only)
            if flow && slow {
                for c in &members {
                    let is_video = q.find_item(*c).and_then(|(t, _)| q.track(t)).is_some_and(|t| t.kind == TrackKind::Video);
                    if is_video && let Some((_, it)) = q.find_item_mut(*c) {
                        it.time_interpolation = TimeInterpolation::OpticalFlow;
                    }
                }
            }
        }
        q.check().map_err(EngineError::Other)?;
        Ok(report)
    })?;
    Ok(json!({
        "preset": preset.map(|r| r.id),
        "intensity": intensity,
        "clips": report,
    }))
}

#[cfg(test)]
#[path = "speed_ramp_tests.rs"]
mod tests;
