//! Speed ramps (`clip.speedRamp`, Clip ▸ Speed Ramp…): one of the
//! [`filmcraft_project::time_remap::RAMP_PRESETS`] written as the clip's Time Remapping speed
//! curve.
//!
//! | param | meaning |
//! |---|---|
//! | `clips` | the clips (default: the selection); linked partners (the clip's sound) ramp with them |
//! | `preset` | `montage`, `hero`, `bullet`, `jumpCut`, `flashIn`, `flashOut`, `smoothIn`, `smoothOut` (ids or names), or `none` to take a ramp off |
//! | `intensity` | 0 … 2 (default 1): each speed's distance from the clip's speed, in log space |
//! | `fit` | `media` (default): the clip keeps the media it plays and its length changes; `length`: it keeps its length and plays the media the curve needs |
//! | `ripple` | default true: later clips move with the new length (false: the ramp is cut at the next clip); `fit: length` moves nothing |
//! | `opticalFlow` | default true: a ramp with slow parts interpolates them with Optical Flow |
//! | `maintainPitch` | absent leaves each clip's Maintain Audio Pitch as it is |
//!
//! With `fit: media` the clip keeps the media it plays (In to Out); its length becomes that media
//! over the curve's mean speed, on whole frames (as Speed/Duration), and its speed the average
//! one. With `fit: length` it keeps its place and length: a ramp that is faster on average plays
//! on past its Out into the source's handles, a slower one stops short of it; where the source
//! has not that much after In, the curve is eased (a lower intensity, said in `eased`) until it
//! fits. Speed keyframes are Bézier (Jump Cut: held), relative to the clip's speed before the
//! ramp, which the ramp remembers ([`RAMP_BASE`]): a ramp on a ramped clip replaces it, and
//! `none` puts that speed back (100 % for a clip at normal speed) — with `fit: media` the clip
//! plays its media at it again (its length back, the later clips moved with `ripple`), with
//! `fit: length` it keeps its length. One undo step for every clip. A reversed or frame-held clip
//! is refused: its media does not run forward.

use serde_json::{Value, json};

use filmcraft_edit as edit;
use filmcraft_project::time_remap::{self, MAX_INTENSITY, RAMP_PRESETS};
use filmcraft_project::{ClipId, ParamValue, Sequence, TimeInterpolation, TrackKind};

use crate::commands::{CommandSpec, bad, bool_p, clips_p, f64_p, has_selection, str_p, with_links};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.speedRamp";

/// The Time Remapping parameter that remembers a ramped clip's speed before the ramp (percent).
/// Not one of the effect's own parameters: Effect Controls do not show it.
pub const RAMP_BASE: &str = "ramp_base";

/// What a ramp keeps: the media the clip plays, or its length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fit {
    Media,
    Length,
}

/// The speed (percent) a ramped clip played at before its ramp; `None`: it has no ramp, or one
/// made before ramps remembered it.
fn ramp_base(it: &filmcraft_project::TrackItem) -> Option<f64> {
    let e = it.effects.iter().find(|e| e.effect == "time_remap")?;
    e.params.get(RAMP_BASE)?.value.as_f64().filter(|v| v.is_finite() && *v > 0.0 && *v <= time_remap::MAX_SPEED_PCT)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Speed Ramp…",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"clips":[id]?,"preset":"montage|hero|bullet|jumpCut|flashIn|flashOut|smoothIn|smoothOut|none","intensity":0..2=1,"fit":"media|length"="media","ripple":bool=true,"opticalFlow":bool=true,"maintainPitch":bool?}"#,
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
/// as a sound), the static value the clip's speed, and remember `base_pct`, its speed before the
/// ramp ([`RAMP_BASE`]).
fn set_curve(it: &mut filmcraft_project::TrackItem, keys: Vec<filmcraft_project::Keyframe>, base_pct: f64) -> Result<()> {
    let pct = it.speed.abs() * 100.0;
    let e = match it.effects.iter().position(|e| e.effect == "time_remap") {
        Some(i) => it.effects.get_mut(i).ok_or_else(|| bad(CMD, "no Time Remapping effect"))?,
        None => {
            let e = time_remap::remap_effect(Vec::new(), pct).ok_or_else(|| bad(CMD, "no Time Remapping effect definition"))?;
            it.effects.push(e);
            it.effects.last_mut().ok_or_else(|| bad(CMD, "no Time Remapping effect"))?
        }
    };
    e.enabled = true;
    let p = e.params.entry("speed".into()).or_insert_with(|| filmcraft_project::Param::new(ParamValue::Float(pct)));
    p.value = ParamValue::Float(pct);
    p.keyframes = keys;
    e.params.insert(RAMP_BASE.into(), filmcraft_project::Param::new(ParamValue::Float(base_pct)));
    Ok(())
}

/// Take a clip's ramp off: no Speed keyframes, the static speed `base_pct`, nothing remembered.
fn clear_curve(it: &mut filmcraft_project::TrackItem, base_pct: f64) {
    if let Some(e) = it.effects.iter_mut().find(|e| e.effect == "time_remap") {
        if let Some(sp) = e.params.get_mut("speed") {
            sp.keyframes.clear();
            sp.value = ParamValue::Float(base_pct);
        }
        e.params.remove(RAMP_BASE);
    }
}

/// The media after In the clips `members` can play, the shortest of theirs (ticks); `None`: as much
/// as they like (stills, generated media).
fn room_after_in(q: &Sequence, members: &[ClipId], ctx: &edit::EditCtx) -> Option<f64> {
    members
        .iter()
        .filter_map(|c| {
            let (_, it) = q.find_item(*c)?;
            let end = (ctx.media_duration)(it.item)?;
            Some((end.0.saturating_sub(it.source_in.0)).max(0) as f64)
        })
        .reduce(f64::min)
}

/// Bisections when easing a ramp until its media fits.
const EASE_STEPS: u32 = 24;

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
    let fit = match p.get("fit") {
        None | Some(Value::Null) => Fit::Media,
        Some(Value::String(f)) if f.eq_ignore_ascii_case("media") => Fit::Media,
        Some(Value::String(f)) if f.eq_ignore_ascii_case("length") => Fit::Length,
        Some(_) => return Err(bad(CMD, "`fit` is media (the clip keeps its media) or length (it keeps its length)")),
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
            // the speed before a ramp already on the clip (a new ramp replaces it), else its speed
            let base_pct = ramp_base(it).unwrap_or(it.speed.abs() * 100.0);
            let base = base_pct / 100.0;
            let Some(preset) = preset else {
                // take the ramp off: the speed from before it (a ramp that did not remember it,
                // made before ramps did: 100 %)
                let ramped: Vec<ClipId> = members
                    .iter()
                    .copied()
                    .filter(|c| q.find_item(*c).is_some_and(|(_, i)| time_remap::curve_of(i).is_some() || ramp_base(i).is_some()))
                    .collect();
                let back_pct = ramp_base(it).unwrap_or(100.0);
                if !ramped.is_empty() {
                    for c in &ramped {
                        if let Some((_, it)) = q.find_item_mut(*c) {
                            clear_curve(it, back_pct);
                        }
                    }
                    let back = back_pct / 100.0;
                    match fit {
                        Fit::Media => edit::set_speed_group(q, &ramped, back, false, ripple, ctx)?,
                        Fit::Length => {
                            // its length kept, unless the source ends first
                            let room = room_after_in(q, &ramped, ctx);
                            for c in &ramped {
                                if let Some((_, it)) = q.find_item_mut(*c) {
                                    it.speed = back;
                                    if let Some(r) = room.filter(|r| it.duration.0 as f64 * back > *r) {
                                        it.duration = filmcraft_time::Tick(((r / back).floor() as i64).max(ctx.min_duration.0));
                                    }
                                }
                            }
                        }
                    }
                }
                for c in &members {
                    let removed = ramped.contains(c);
                    let Some((_, it)) = q.find_item(*c) else { continue };
                    report.push(json!({"clip": c.0, "removed": removed, "speed": it.speed, "duration": it.duration.0, "end": it.end().0}));
                }
                continue;
            };
            let source_in = it.source_in;
            let span = it.source_out() - it.source_in;
            let mean_at = |k: f64| time_remap::ramp_mean(&time_remap::ramp_keyframes(preset, k, source_in, span, base_pct), source_in, span, base_pct);
            let mut mean = mean_at(intensity).ok_or_else(|| bad(CMD, format!("clip {} is too short for a speed ramp", lead.0)))?;
            let mut used = intensity;
            match fit {
                Fit::Media => {
                    // the media as it plays now, at the speed before the ramp times the curve's mean
                    edit::set_speed_group(q, &members, base * mean, false, ripple, ctx)?;
                }
                Fit::Length => {
                    // the clip's length kept: the media the curve needs from In, eased to the source
                    let dur = it.duration.0 as f64;
                    let need = |m: f64| dur * base * m;
                    let room = room_after_in(q, &members, ctx);
                    if let Some(room) = room.filter(|r| need(mean) > *r) {
                        let (mut lo, mut hi, mut best) = (0.0f64, intensity, (0.0f64, mean_at(0.0).unwrap_or(1.0)));
                        for _ in 0..EASE_STEPS {
                            let mid = (lo + hi) / 2.0;
                            match mean_at(mid) {
                                Some(m) if need(m) <= room => {
                                    best = (mid, m);
                                    lo = mid;
                                }
                                _ => hi = mid,
                            }
                        }
                        (used, mean) = best;
                    }
                    let most = room.filter(|_| dur > 0.0).map_or(f64::INFINITY, |r| r / dur);
                    let speed = (base * mean).min(most).max(1e-6);
                    for c in &members {
                        if let Some((_, it)) = q.find_item_mut(*c) {
                            it.speed = speed;
                        }
                    }
                }
            }
            let slow = preset.points.iter().any(|(_, sp)| sp.max(1e-3).powf(used) < 0.999);
            for c in &members {
                let Some((_, it)) = q.find_item_mut(*c) else { continue };
                // the curve over the media this member now plays, relative to the speed before
                let keys = time_remap::ramp_keyframes(preset, used, it.source_in, it.source_out() - it.source_in, base_pct);
                set_curve(it, keys, base_pct)?;
                if let Some(k) = keep_pitch {
                    it.maintain_pitch = k;
                }
                let mut line = json!({
                    "clip": c.0, "start": it.start.0, "duration": it.duration.0, "end": it.end().0,
                    "speed": it.speed, "meanSpeed": mean, "keyframes": time_remap::curve_of(it).map_or(0, |p| p.keyframes.len()),
                });
                if used < intensity {
                    line["eased"] = json!(used);
                }
                report.push(line);
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
        "fit": if fit == Fit::Length { "length" } else { "media" },
        "clips": report,
    }))
}

#[cfg(test)]
#[path = "speed_ramp_tests.rs"]
mod tests;
