//! Per-track auto-ducking (`track.setAutoDuck`): an audio track (music, ambience) whose clips'
//! Volume goes down while the tracks it is under (dialogue, voice-over) have sound, through the
//! same analysis as Essential Sound's Generate Ducking Keyframes (`essentialSound.generateDucking`:
//! the summed level of the trigger clips, activity detection with blip removal and gap bridging,
//! Volume keyframes on each ducked clip), and kept up to date: after every command that changes
//! the project, the clips whose own placement, trigger clips or settings changed are ducked again
//! ([`refresh`]), as part of that command's undo step.
//!
//! | command | does |
//! |---|---|
//! | `track.setAutoDuck {track, under?, amount?, attack?, release?, sensitivity?, enabled?}` | duck `track` (an audio track: id or `A1`, `A2`…) under the `under` tracks (default: the other audio tracks) by `amount` dB (12) with `attack` / `release` ramps (0.25 / 0.5 s); `enabled: false` turns it off and puts the clips' levels back |
//!
//! Auto-duck owns the Volume level of the ducked track's clips while it is on: the level a clip
//! had when ducking took it over is remembered and restored when it is turned off.

use std::sync::Arc;

use serde_json::{Value, json};

use filmcraft_audio_dsp::ducking;
use filmcraft_project::auto_duck::{AutoDuck, DuckedClip, MAX_CLIPS};
use filmcraft_project::{ClipId, ItemId, ParamValue, Project, Sequence, Track, TrackId, TrackItem, TrackKind};
use filmcraft_render::SourceProvider;
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, bool_p, has_seq, track_p};
use crate::essential_sound::{DUCK_BRIDGE_S, DUCK_HOP_S, DUCK_MIN_ON_S, trigger_envelope};
use crate::{EngineError, Result, Session};

const CMD: &str = "track.setAutoDuck";

/// New Volume keyframes of one ducked clip.
struct ClipPlan {
    clip: ClipId,
    base_db: f64,
    /// (timeline time, level dB).
    keyframes: Vec<(Tick, f64)>,
}

/// FNV-1a, stable across builds (the signatures are saved in projects).
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for x in b {
            self.0 ^= u64::from(*x);
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

fn level_param(it: &TrackItem) -> Option<&filmcraft_project::Param> {
    it.effect("volume").and_then(|e| e.param("level"))
}

/// The clip's Volume level as it is now: its value, or the loudest keyframe when animated.
fn current_level(it: &TrackItem) -> f64 {
    match level_param(it) {
        Some(p) if p.is_animated() => p.keyframes.iter().filter_map(|k| k.value.as_f64()).fold(f64::NEG_INFINITY, f64::max),
        Some(p) => p.value.as_f64().unwrap_or(0.0),
        None => 0.0,
    }
    .clamp(-96.0, 24.0)
}

/// What a clip's duck keyframes depend on: the clip itself (its Volume level aside: ducking
/// writes it), the trigger clips over it with their tracks' mute, and the settings.
fn signature(it: &TrackItem, triggers: &[(&Track, &TrackItem)], ad: &AutoDuck, sr: u32) -> u64 {
    let mut h = Fnv::new();
    for v in [ad.amount_db, ad.attack_s, ad.release_s, ad.sensitivity] {
        h.u64(v.to_bits());
    }
    h.u64(u64::from(sr));
    let mut own = it.clone();
    if let Some(p) = own.effect_mut("volume").and_then(|e| e.param_mut("level")) {
        p.keyframes.clear();
        p.value = ParamValue::Float(0.0);
    }
    h.bytes(&serde_json::to_vec(&own).unwrap_or_default());
    for (t, o) in triggers {
        h.u64(t.id.0);
        h.u64(u64::from(t.muted));
        h.bytes(&serde_json::to_vec(o).unwrap_or_default());
    }
    h.0
}

/// The duck plan of `track` (in `seq`) with settings `ad`: new keyframes for every clip whose
/// signature changed (all of them with `force`), and the clip list to remember.
fn plan_track(seq: &Sequence, track: &Track, ad: &AutoDuck, sources: &dyn SourceProvider, force: bool) -> (Vec<ClipPlan>, Vec<DuckedClip>) {
    let sr = seq.settings.sample_rate.clamp(8_000, 384_000);
    let ad = ad.sanitized();
    let triggers: Vec<(&Track, &TrackItem)> = seq
        .audio_tracks
        .iter()
        .filter(|t| t.id != track.id && ad.under.contains(&t.id) && !t.muted)
        .flat_map(|t| t.items.iter().filter(|i| i.enabled).map(move |i| (t, i)))
        .collect();
    let thr = ducking::sensitivity_threshold_db(ad.sensitivity as f32);
    let mut plans = Vec::new();
    let mut entries = Vec::new();
    for it in track.items.iter().take(MAX_CLIPS) {
        let over: Vec<(&Track, &TrackItem)> = triggers.iter().copied().filter(|(_, o)| o.range().overlaps(&it.range())).collect();
        let sig = signature(it, &over, &ad, sr);
        // the level before ducking: remembered, else that of the clip it was split from, else now
        let base = ad
            .clip(it.id)
            .map(|c| c.base_db)
            .or_else(|| ad.clips.iter().find(|c| c.clip != it.id && track.items.iter().any(|o| o.id == c.clip && o.item == it.item)).map(|c| c.base_db))
            .unwrap_or_else(|| current_level(it));
        entries.push(DuckedClip { clip: it.id, base_db: base, signature: sig });
        if !force && ad.clip(it.id).is_some_and(|c| c.signature == sig) {
            continue;
        }
        let (a0, a1) = (it.start.to_units_floor(i64::from(sr)), it.end().to_units_floor(i64::from(sr)));
        let regions = if over.is_empty() || a1 <= a0 {
            Vec::new()
        } else {
            let items: Vec<&TrackItem> = over.iter().map(|(_, o)| *o).collect();
            let env = trigger_envelope(&items, a0, a1, sr, sources);
            ducking::activity(&env, DUCK_HOP_S as f32, thr, DUCK_MIN_ON_S, DUCK_BRIDGE_S)
        };
        let start_s = a0 as f64 / f64::from(sr);
        let end_s = a1 as f64 / f64::from(sr);
        let abs: Vec<(f64, f64)> = regions.iter().map(|(x, y)| (start_s + x, start_s + y)).collect();
        let kfs = ducking::duck_keyframes_ar(&abs, start_s, end_s, base, ad.amount_db, ad.attack_s, ad.release_s);
        let keyframes = kfs.iter().map(|(t, v)| (Tick::from_seconds_f64(*t).clamp(it.start, it.end()), *v)).collect();
        plans.push(ClipPlan { clip: it.id, base_db: base, keyframes });
    }
    (plans, entries)
}

fn media_time(it: &TrackItem, t: Tick) -> Tick {
    it.source_time_at(t.clamp(it.start, (it.end() - Tick(1)).max(it.start)))
}

/// Write `plans` into `track` of `seq` and remember `entries` (or turn auto-duck off with `None`).
fn apply(seq: &mut Sequence, track: TrackId, ad: Option<AutoDuck>, plans: &[ClipPlan]) {
    let Some(t) = seq.audio_tracks.iter_mut().find(|t| t.id == track) else { return };
    for plan in plans {
        let Some(it) = t.items.iter_mut().find(|i| i.id == plan.clip) else { continue };
        let at: Vec<(Tick, f64)> = plan.keyframes.iter().map(|(tt, v)| (media_time(it, *tt), *v)).collect();
        let Some(par) = it.effect_mut("volume").and_then(|e| e.param_mut("level")) else { continue };
        par.keyframes.clear();
        par.value = ParamValue::Float(plan.base_db);
        for (mt, v) in at {
            par.put_keyframe(mt, ParamValue::Float(v));
        }
    }
    t.auto_duck = ad;
}

/// The audio tracks a parameter lists (ids or names), each an audio track of the sequence.
fn tracks_p(s: &Session, v: &Value) -> Result<Vec<TrackId>> {
    let list: Vec<Value> = match v {
        Value::Array(a) => a.clone(),
        Value::Null => Vec::new(),
        one => vec![one.clone()],
    };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let mut out = Vec::new();
    for x in list.iter().take(64) {
        let id = track_p(s, &json!({"t": x}), "t", CMD)?.ok_or_else(|| bad(CMD, "`under`: a track id or name"))?;
        if !seq.audio_tracks.iter().any(|t| t.id == id) {
            return Err(bad(CMD, format!("`under`: {x} is not an audio track")));
        }
        if !out.contains(&id) {
            out.push(id);
        }
    }
    Ok(out)
}

fn number(p: &Value, keys: &[&str], (lo, hi): (f64, f64), default: f64, what: &str) -> Result<f64> {
    match keys.iter().find_map(|k| p.get(*k).filter(|v| !v.is_null())) {
        None => Ok(default),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && (lo..=hi).contains(&x.abs()) => Ok(x.abs()),
            _ => Err(bad(CMD, format!("`{}`: {what} between {lo} and {hi}", keys[0]))),
        },
    }
}

fn state_json(track: TrackId, ad: Option<&AutoDuck>) -> Value {
    match ad {
        None => json!({"track": track.0, "enabled": false}),
        Some(a) => json!({
            "track": track.0,
            "enabled": true,
            "under": a.under.iter().map(|t| t.0).collect::<Vec<_>>(),
            "amountDb": a.amount_db,
            "attackSeconds": a.attack_s,
            "releaseSeconds": a.release_s,
            "sensitivity": a.sensitivity,
            "clips": a.clips.len(),
        }),
    }
}

fn set_auto_duck(s: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_project::auto_duck as ad;
    let track = track_p(s, p, "track", CMD)?.ok_or_else(|| bad(CMD, "need `track`: the audio track to duck (its id, or A1, A2…)"))?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let tr = seq.audio_tracks.iter().find(|t| t.id == track).ok_or_else(|| bad(CMD, "`track` must be an audio track"))?;
    if !bool_p(p, "enabled").unwrap_or(true) {
        // off: every remembered clip gets its level back, without keyframes
        let restore: Vec<ClipPlan> =
            tr.auto_duck.iter().flat_map(|a| a.clips.iter()).map(|c| ClipPlan { clip: c.clip, base_db: c.base_db, keyframes: Vec::new() }).collect();
        s.edit("Auto-Duck Off", |pr, _| {
            let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            apply(q, track, None, &restore);
            Ok(())
        })?;
        return Ok(state_json(track, None));
    }
    let current = tr.auto_duck.clone().unwrap_or_default();
    let under = match p.get("under") {
        Some(v) if !v.is_null() => tracks_p(s, v)?,
        _ if !current.under.is_empty() => current.under.clone(),
        _ => seq.audio_tracks.iter().filter(|t| t.id != track && t.auto_duck.is_none()).map(|t| t.id).collect(),
    };
    if under.contains(&track) {
        return Err(bad(CMD, "a track cannot duck under itself"));
    }
    let settings = AutoDuck {
        under,
        amount_db: number(p, &["amount", "amountDb"], ad::AMOUNT_DB, current.amount_db, "the duck amount in dB")?,
        attack_s: number(p, &["attack", "attackSeconds"], ad::ATTACK_S, current.attack_s, "the attack in seconds")?,
        release_s: number(p, &["release", "releaseSeconds"], ad::RELEASE_S, current.release_s, "the release in seconds")?,
        sensitivity: number(p, &["sensitivity"], (0.0, 10.0), current.sensitivity, "the sensitivity")?,
        clips: current.clips,
    };
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let (plans, entries) = plan_track(seq, tr, &settings, &provider, true);
    let report: Vec<Value> = plans
        .iter()
        .map(|c| json!({"clip": c.clip.0, "baseDb": c.base_db, "keyframes": c.keyframes.iter().map(|(t, v)| json!({"time": t.0, "seconds": t.seconds(), "levelDb": v})).collect::<Vec<_>>()}))
        .collect();
    let settings = AutoDuck { clips: entries, ..settings };
    let state = state_json(track, Some(&settings));
    s.edit("Auto-Duck Track", |pr, _| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        apply(q, track, Some(settings), &plans);
        Ok(())
    })?;
    let mut out = state;
    out["ducked"] = Value::Array(report);
    Ok(out)
}

/// Re-duck what changed: every auto-ducked track of every sequence whose clips, trigger clips
/// or settings no longer match the signatures its keyframes were made from. The project is
/// updated in place, inside the undo step of the command that changed it.
pub(crate) fn refresh(s: &mut Session) {
    let targets: Vec<(ItemId, TrackId)> = s
        .project
        .items
        .iter()
        .filter_map(|(id, it)| it.as_sequence().map(|q| (*id, q)))
        .flat_map(|(id, q)| q.audio_tracks.iter().filter(|t| t.auto_duck.is_some()).map(move |t| (id, t.id)))
        .collect();
    if targets.is_empty() {
        return;
    }
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let mut work: Vec<(ItemId, TrackId, AutoDuck, Vec<ClipPlan>)> = Vec::new();
    for (seq_id, track) in targets {
        let Some(q) = s.project.sequence(seq_id) else { continue };
        let Some(t) = q.audio_tracks.iter().find(|t| t.id == track) else { continue };
        let Some(ad) = t.auto_duck.as_ref() else { continue };
        let (plans, entries) = plan_track(q, t, ad, &provider, false);
        if plans.is_empty() && entries == ad.clips {
            continue;
        }
        work.push((seq_id, track, AutoDuck { clips: entries, ..ad.sanitized() }, plans));
    }
    if work.is_empty() {
        return;
    }
    let mut p: Project = (*s.project).clone();
    for (seq_id, track, ad, plans) in &work {
        if let Some(q) = p.sequence_mut(*seq_id) {
            apply(q, *track, Some(ad.clone()), plans);
        }
    }
    s.project = Arc::new(p);
    s.bump();
}

/// The auto-duck settings of an audio track, for agents and the UI.
pub fn state(s: &Session, track: TrackId) -> Value {
    let ad = s.active_sequence().and_then(|q| q.audio_tracks.iter().find(|t| t.id == track)).and_then(|t| t.auto_duck.as_ref());
    state_json(track, ad)
}

/// Whether `track` of the active sequence is auto-ducked.
pub fn is_on(s: &Session, track: TrackId) -> bool {
    s.active_sequence().and_then(|q| q.track(track)).is_some_and(|t| t.kind == TrackKind::Audio && t.auto_duck.is_some())
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Auto-Duck Track",
        menu: &[],
        shortcut: None,
        params: r#"{"track":"id|A1…","under":["id|A1…"]?,"amount":dB=12,"attack":seconds=0.25,"release":seconds=0.5,"sensitivity":0..10=5,"enabled":bool=true}"#,
        enabled: has_seq,
        run: set_auto_duck,
        journal: true,
    }]
}
