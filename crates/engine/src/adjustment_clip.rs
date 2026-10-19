//! New Adjustment Clip (`timeline.newAdjustmentClip`): an adjustment layer made and placed on the
//! timeline in one step, Shotcut-style. Effects on it apply to the composite of every track below
//! for its duration (the renderer's adjustment-layer path), so one grade, blur or look covers a
//! whole passage of clips.
//!
//! | command | does |
//! |---|---|
//! | `timeline.newAdjustmentClip {track?, start…?, duration…?, effects?, name?}` | a new Adjustment Layer item, placed over the selected clips' span (else In to Out, else 5 s from the playhead) on the first video track above every clip there (a new top track when there is none); `effects` are applied to it at once. One undo step |
//!
//! Times take the usual forms: `startTime` (ticks) / `startFrame` / `startSeconds` /
//! `startTimecode`, and `duration` (ticks) or `seconds`.

use serde_json::{Value, json};

use filmcraft_project::{ItemKind, Label};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, bad, has_seq, sequence_time_p, str_p, track_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "timeline.newAdjustmentClip";
/// Length of a new adjustment clip with nothing selected and no In/Out (seconds).
const DEFAULT_SECONDS: f64 = 5.0;
/// Longest adjustment clip a call makes (24 hours).
const MAX_SECONDS: f64 = 24.0 * 3600.0;
/// Effects one call applies at most.
const MAX_EFFECTS: usize = 32;

/// The span the clip covers: explicit start / duration, else the selected clips, else In to
/// Out, else the default length from the playhead.
fn span(s: &Session, p: &Value) -> Result<TimeRange> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let rate = q.settings.frame_rate;
    let start = sequence_time_p(s, p, "start", CMD)?;
    let duration = match (p.get("duration").and_then(Value::as_i64), p.get("seconds").and_then(Value::as_f64)) {
        (Some(d), _) => Some(Tick(d)),
        (None, Some(sec)) if sec.is_finite() && sec.abs() <= MAX_SECONDS => Some(Tick::from_seconds_f64(sec)),
        (None, Some(_)) => return Err(bad(CMD, "`seconds`: a length of at most 24 hours")),
        _ => None,
    };
    let selected: Vec<TimeRange> = s.state.selection.iter().filter_map(|c| q.find_item(*c)).map(|(_, it)| it.range()).collect();
    let from_selection = selected.iter().map(|r| r.start).min().zip(selected.iter().map(|r| r.end()).max());
    let marks = q.mark_in.zip(q.mark_out).filter(|(a, b)| b > a);
    let (a, b) = match (start, duration) {
        (Some(a), Some(d)) => (a, a.0.checked_add(d.0).map(Tick).ok_or_else(|| bad(CMD, "the end overflows"))?),
        (Some(a), None) => (a, a + Tick::from_seconds_f64(DEFAULT_SECONDS)),
        (None, Some(d)) => {
            let a = from_selection.map(|x| x.0).unwrap_or_else(|| s.playhead());
            (a, a.0.checked_add(d.0).map(Tick).ok_or_else(|| bad(CMD, "the end overflows"))?)
        }
        // the Out point is the last frame shown, so the clip runs to the end of it
        (None, None) => from_selection.or(marks.map(|(a, b)| (a, b + rate.frame_duration()))).unwrap_or_else(|| {
            let a = s.playhead();
            (a, a + Tick::from_seconds_f64(DEFAULT_SECONDS))
        }),
    };
    let (a, b) = (rate.snap_nearest(a.max(Tick::ZERO)), rate.snap_nearest(b));
    if b <= a {
        return Err(bad(CMD, "the adjustment clip needs a positive duration"));
    }
    if (b - a).seconds() > MAX_SECONDS {
        return Err(bad(CMD, "an adjustment clip is at most 24 hours long"));
    }
    Ok(TimeRange::from_bounds(a, b))
}

/// The video track the clip goes on: `track` when free over `range`, else the first unlocked
/// track above every clip in `range`; `None` = a new track on top.
fn pick_track(s: &Session, p: &Value, range: TimeRange) -> Result<Option<filmcraft_project::TrackId>> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    if let Some(id) = track_p(s, p, "track", CMD)? {
        let t = q.video_tracks.iter().find(|t| t.id == id).ok_or_else(|| bad(CMD, "`track` must be a video track"))?;
        if t.locked {
            return Err(bad(CMD, "that track is locked"));
        }
        if t.items.iter().any(|i| i.range().overlaps(&range)) {
            return Err(bad(CMD, "that track has clips there: pick a free track, or leave `track` out to use the first free one above them"));
        }
        return Ok(Some(id));
    }
    let top = q.video_tracks.iter().rposition(|t| t.items.iter().any(|i| i.range().overlaps(&range)));
    let from = top.map_or(0, |i| i + 1);
    Ok(q.video_tracks.iter().skip(from).find(|t| !t.locked).map(|t| t.id))
}

fn new_adjustment_clip(s: &mut Session, p: &Value) -> Result<Value> {
    let range = span(s, p)?;
    let track = pick_track(s, p, range)?;
    let effects: Vec<String> = match p.get("effects") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.len() <= MAX_EFFECTS => {
            a.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad(CMD, "`effects`: a list of effect names"))).collect::<Result<_>>()?
        }
        Some(Value::String(one)) => vec![one.clone()],
        Some(_) => return Err(bad(CMD, format!("`effects`: a list of at most {MAX_EFFECTS} effect names"))),
    };
    let name = str_p(p, "name").map(|n| n.trim().chars().take(200).collect::<String>()).filter(|n| !n.is_empty()).unwrap_or_else(|| "Adjustment Clip".into());
    let st = s.active_sequence().map(|q| q.settings.clone()).ok_or(EngineError::NoSequence)?;
    s.edit_group("New Adjustment Clip", |s| {
        let item = s.edit("New Adjustment Clip", |pr, _| {
            Ok(pr.add_item(
                &name,
                Label::Lavender,
                ItemKind::AdjustmentLayer { width: st.width, height: st.height, rate: st.frame_rate, duration: range.duration },
                None,
            ))
        })?;
        let placed = s.execute(
            "timeline.place",
            json!({"item": item.0, "track": track.map_or(json!("new"), |t| json!(t.0)), "time": range.start.0, "video": true, "audio": false}),
        )?;
        let clip = placed["clips"].get(0).and_then(Value::as_u64).ok_or_else(|| EngineError::Other("the adjustment clip was not placed".into()))?;
        for e in &effects {
            s.execute("effects.apply", json!({"clips": [clip], "effect": e}))?;
        }
        s.execute("timeline.select", json!({"clips": [clip]}))?;
        let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (tid, it) = q.find_item(filmcraft_project::ClipId(clip)).ok_or_else(|| EngineError::Other("the adjustment clip was not placed".into()))?;
        let index = q.video_tracks.iter().position(|t| t.id == tid);
        Ok(json!({
            "clip": clip,
            "item": item.0,
            "track": tid.0,
            "trackName": index.map(|i| format!("V{}", i + 1)),
            "start": it.start.0,
            "duration": it.duration.0,
            "seconds": it.duration.seconds(),
            "effects": it.effects.iter().map(|e| e.effect.clone()).collect::<Vec<_>>(),
        }))
    })
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "New Adjustment Clip",
        menu: &[],
        shortcut: None,
        params: r#"{"track":"id|V1…"?,"startTime":ticks?,"startSeconds":f64?,"duration":ticks?,"seconds":f64?,"effects":[str]?,"name":str?}"#,
        enabled: has_seq,
        run: new_adjustment_clip,
        journal: true,
    }]
}
