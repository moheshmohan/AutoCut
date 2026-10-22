//! Transition tools (roadmap C4): transitions on the music's beat, and sounds that belong to a
//! transition.
//!
//! | command | does |
//! |---|---|
//! | `transitions.snapToBeats` | moves each transition's midpoint to the nearest beat of the music and sets its duration to the nearest musical length (¼, ½, 1, 2 or 4 beats), on whole frames, as one undo step |
//! | `transitions.addSound` | couples a whoosh, riser or impact to video transitions: synthesised when the sequence is mixed, placed to the sample around the transition's midpoint, and gone with the transition |
//! | `transitions.removeSound` | removes the coupled sounds |
//! | `transitions.applyToAll` | puts one transition (a Creator preset or any video transition) on every cut of the video tracks, as one undo step |
//! | `transitions.randomOnBeats` | puts a transition from a folder (default Creator), picked by a seeded draw, on each cut that lands on a strong beat |
//!
//! **Beats.** `music` names an audio clip of the sequence (default: its longest audio clip); its
//! beats come from the same analysis as `media.beats` (the Remix beat tracker), mapped through the
//! clip's In point and speed onto the timeline. `beats` (seconds of the sequence) and `bpm` give
//! a grid instead, as an agent or a template might. The beat length is `60 / bpm`, else the
//! median gap between beats. Each transition keeps covering its cut and stays inside the clips it
//! joins and clear of its neighbours ([`filmcraft_edit::transitions::snap_to_beat`]); one that no
//! beat in reach fits is left as it was and reported. A fade to or from nothing keeps its clip
//! edge and only takes a musical length. The Showcase cuts its pictures on the beat the same way;
//! this brings it to any timeline.
//!
//! **All cuts, cuts on beats.** A cut is where a clip ends and the next on its track starts. A
//! transition goes centred on it, as long as `duration` (seconds; default: the preset's own,
//! 0.3–0.5 s for the Creator set, else Settings ▸ Timeline's default), on whole frames, never
//! longer than either clip; one already on a cut is replaced (`replace`, default true) or kept.
//! `randomOnBeats` takes the beat grid as `snapToBeats` does; the strong beats are every
//! `everyNthBeat`-th (default 2: the first and third of a 4/4 bar) from the first beat, and a cut
//! within `tolerance` seconds (default 0.1) of one takes a transition, never longer than 90 % of
//! a beat. The draw is a SplitMix64 hash of `seed` and the cut's place: the same seed gives the
//! same transitions, and a transition is never the same as the one before it on the track (with
//! two or more to draw from). `sound: true` couples each Creator preset's paired whoosh, riser or
//! impact in the same step.
//!
//! **Sounds.** The sound is part of the transition (`Transition::sound`), so it moves, resizes,
//! snaps to the beat and is removed with it; the mixer synthesises it with
//! `filmcraft_audio_dsp::synth` into the sequence's Mix (`filmcraft_render::mixer`). Where it
//! plays is in [`filmcraft_project::transition_sound`]; the commands report it in exact samples.

use serde_json::{Value, json};

use filmcraft_edit::transitions::{BEAT_FRACTIONS, snap_to_beat};
use filmcraft_project::creator_transitions;
use filmcraft_project::{ClipId, EffectDef, EffectKind, ItemId, Sequence, SoundKind, Track, TrackId, TrackKind, Transition, TransitionId, TransitionSound};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, str_p};
use crate::{EngineError, Result, Session};

/// Most beats a `beats` list may hold.
pub const MAX_BEATS: usize = 100_000;
/// Latest beat a `beats` list may name (seconds): a day.
const MAX_BEAT_SECONDS: f64 = 86_400.0;
/// Tempo range of `bpm`.
const BPM_RANGE: (f64, f64) = (20.0, 400.0);

/// A whole, non-negative number (an id or a seed): `7` or `7.0`, never `-4` or `2.5`.
fn whole_p(p: &Value, k: &str) -> Option<u64> {
    let v = p.get(k)?;
    v.as_u64().or_else(|| v.as_f64().filter(|f| f.is_finite() && f.fract() == 0.0 && (0.0..=9_007_199_254_740_992.0).contains(f)).map(|f| f as u64))
}

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal: true }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "transitions.snapToBeats",
            "Snap Transitions to Beats",
            r#"{"sequence":id?,"track":"V1"|id?,"transitions":[id]?,"music":clip id?,"beats":[seconds]?,"bpm":20..400?}"#,
            snap_to_beats,
        ),
        spec(
            "transitions.addSound",
            "Add Transition Sound",
            r#"{"transitions":[id]?,"kind":"whoosh|riser|impact"=whoosh,"gain":dB=-6,"intensity":0..1=0.7,"seed":u64?,"seconds":f64?}"#,
            add_sound,
        ),
        spec("transitions.removeSound", "Remove Transition Sound", r#"{"transitions":[id]?}"#, remove_sound),
        spec(
            "transitions.applyToAll",
            "Apply Transition to All Cuts",
            r#"{"preset":id|name,"duration":seconds?,"sequence":id?,"track":"V1"|id?,"replace":bool=true,"sound":bool=false}"#,
            apply_to_all,
        ),
        spec(
            "transitions.randomOnBeats",
            "Transitions on Strong Beats",
            r#"{"folder":"Creator"|folder=Creator,"seed":u64=1,"everyNthBeat":1..64=2,"tolerance":seconds=0.1,"duration":seconds?,"sequence":id?,"track":"V1"|id?,"sound":bool=false,"music":clip id?,"beats":[seconds]?,"bpm":20..400?}"#,
            random_on_beats,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// Snap to beats

/// The sequence a command works on: `sequence`, else the active one.
fn sequence_id(s: &Session, p: &Value, cmd: &str) -> Result<ItemId> {
    crate::export_tools::sequence_param(s, p, cmd)
}

/// A track of `seq` by id or by name (`V1`, `A2`).
fn track_ref(seq: &Sequence, v: &Value) -> Option<TrackId> {
    if let Some(id) = v.as_u64() {
        return seq.track(TrackId(id)).map(|t| t.id);
    }
    let name = v.as_str()?.trim();
    let mut chars = name.chars();
    let tracks = match chars.next()? {
        'v' | 'V' => &seq.video_tracks,
        'a' | 'A' => &seq.audio_tracks,
        _ => return None,
    };
    let i: usize = chars.as_str().parse().ok()?;
    tracks.get(i.checked_sub(1)?).map(|t| t.id)
}

/// The transitions a call names (`transitions` or `transition`), or None when it names none.
fn transition_ids(p: &Value, cmd: &str) -> Result<Option<Vec<TransitionId>>> {
    let Some(v) = p.get("transitions").or_else(|| p.get("transition")) else { return Ok(None) };
    let ids = match v {
        Value::Array(a) => a.iter().map(|x| x.as_u64().map(TransitionId)).collect::<Option<Vec<_>>>(),
        x => x.as_u64().map(|i| vec![TransitionId(i)]),
    };
    ids.map(Some).ok_or_else(|| bad(cmd, "`transitions` is a list of transition ids"))
}

/// A beat grid from the call: (beats in timeline ticks, sorted and distinct; beat length).
pub(crate) fn beat_grid(s: &Session, p: &Value, seq: &Sequence, cmd: &str) -> Result<(Vec<Tick>, Tick, Value)> {
    let bpm = match p.get("bpm") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let b = v.as_f64().filter(|b| b.is_finite() && (BPM_RANGE.0..=BPM_RANGE.1).contains(b));
            Some(b.ok_or_else(|| bad(cmd, format!("`bpm` is a tempo from {} to {} beats per minute", BPM_RANGE.0, BPM_RANGE.1)))?)
        }
    };
    let (mut beats, source) = match p.get("beats") {
        Some(Value::Array(list)) => {
            if list.len() > MAX_BEATS {
                return Err(bad(cmd, format!("at most {MAX_BEATS} beats")));
            }
            let mut out = Vec::with_capacity(list.len());
            for v in list {
                let t = v.as_f64().filter(|t| t.is_finite() && (0.0..=MAX_BEAT_SECONDS).contains(t));
                out.push(Tick::from_seconds_f64(t.ok_or_else(|| bad(cmd, format!("`beats` are seconds from 0 to {MAX_BEAT_SECONDS}, not {v}")))?));
            }
            (out, json!("beats"))
        }
        Some(Value::Null) | None => music_beats(s, p, seq, cmd)?,
        Some(other) => return Err(bad(cmd, format!("`beats` is a list of seconds, not {other}"))),
    };
    beats.sort_unstable();
    beats.dedup();
    let period = match bpm {
        Some(b) => Tick::from_seconds_f64(60.0 / b),
        None => {
            let mut gaps: Vec<Tick> = beats.windows(2).filter_map(|w| w.get(1).zip(w.first()).map(|(b, a)| *b - *a)).filter(|g| *g > Tick::ZERO).collect();
            gaps.sort_unstable();
            gaps.get(gaps.len() / 2).copied().ok_or_else(|| bad(cmd, "need at least two beats (or `bpm`) to know the beat length"))?
        }
    };
    if beats.is_empty() {
        return Err(bad(cmd, "no beats to snap to"));
    }
    Ok((beats, period, source))
}

/// The beats of the music clip (`music`, else the sequence's longest audio clip) on the timeline.
fn music_beats(s: &Session, p: &Value, seq: &Sequence, cmd: &str) -> Result<(Vec<Tick>, Value)> {
    let clip = match whole_p(p, "music") {
        Some(id) => {
            let (t, it) = seq.find_item(filmcraft_project::ClipId(id)).ok_or_else(|| bad(cmd, format!("no clip {id} in the sequence (`music`)")))?;
            if seq.track(t).is_none_or(|t| t.kind != TrackKind::Audio) {
                return Err(bad(cmd, "`music` is a clip on an audio track"));
            }
            it
        }
        None if p.get("music").is_some_and(|v| !v.is_null()) => return Err(bad(cmd, "`music` is the id of an audio clip")),
        None => seq
            .audio_tracks
            .iter()
            .flat_map(|t| t.items.iter())
            .filter(|i| i.enabled && s.project.item(i.item).is_some_and(|m| matches!(m.kind, filmcraft_project::ItemKind::Media(_)) && m.has_audio()))
            .max_by_key(|i| (i.duration, std::cmp::Reverse(i.id)))
            .ok_or_else(|| bad(cmd, "no music: give `music` (an audio clip) or `beats`"))?,
    };
    if clip.reverse {
        return Err(bad(cmd, "the music clip plays in reverse: its beats do not follow the timeline"));
    }
    if filmcraft_render::remix::Remix::of(clip).is_some() {
        return Err(bad(cmd, "the music clip is remixed: snap to its source clip, or give `beats`"));
    }
    let speed = if clip.speed.is_finite() && clip.speed > 0.0 { clip.speed } else { 1.0 };
    // the media the clip plays, in media ticks
    let media_len = if speed == 1.0 { clip.duration } else { Tick((clip.duration.0 as f64 * speed).round() as i64) };
    let (bpm, media_beats) = crate::remix::beat_ticks(s, clip.item, clip.source_in, media_len)?;
    let end = clip.start + clip.duration;
    let beats = media_beats
        .into_iter()
        .map(|b| {
            let offset = b - clip.source_in;
            clip.start + if speed == 1.0 { offset } else { Tick((offset.0 as f64 / speed).round() as i64) }
        })
        .filter(|t| clip.start <= *t && *t < end)
        .collect();
    Ok((beats, json!({"clip": clip.id.0, "bpm": bpm})))
}

fn fraction_label((n, d): (i64, i64)) -> String {
    if d == 1 { format!("{n}") } else { format!("{n}/{d}") }
}

/// `transitions.snapToBeats`.
fn snap_to_beats(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "transitions.snapToBeats";
    let seq_id = sequence_id(s, p, CMD)?;
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate.sane();
    let frame = rate.frame_duration();
    // which tracks and transitions
    let tracks: Vec<TrackId> = match p.get("track") {
        None | Some(Value::Null) => seq.video_tracks.iter().map(|t| t.id).collect(),
        Some(v) => vec![track_ref(seq, v).ok_or_else(|| bad(CMD, format!("no track {v} in this sequence (`track`)")))?],
    };
    let wanted = transition_ids(p, CMD)?;
    if let Some(ids) = &wanted
        && let Some(missing) = ids.iter().find(|id| !seq.all_tracks().any(|t| t.transitions.iter().any(|x| x.id == **id)))
    {
        return Err(bad(CMD, format!("no transition {}", missing.0)));
    }
    let (beats, period, source) = beat_grid(s, p, seq, CMD)?;
    // plan every transition, track by track in time order, each clear of the ones already placed
    let mut plan: Vec<(TransitionId, filmcraft_edit::transitions::BeatSnap)> = Vec::new();
    let mut skipped = Vec::new();
    for tid in &tracks {
        let Some(track) = seq.track(*tid) else { continue };
        let mut todo: Vec<_> = track.transitions.iter().filter(|x| wanted.as_ref().is_none_or(|w| w.contains(&x.id))).collect();
        todo.sort_by_key(|x| (x.start, x.id));
        if track.locked {
            skipped.extend(todo.iter().map(|x| json!({"transition": x.id.0, "reason": "its track is locked"})));
            continue;
        }
        let mut spans: Vec<(TransitionId, (Tick, Tick))> = track.transitions.iter().map(|x| (x.id, (x.start, x.end()))).collect();
        for tr in todo {
            let others: Vec<(Tick, Tick)> = spans.iter().filter(|(id, _)| *id != tr.id).map(|(_, s)| *s).collect();
            match snap_to_beat(track, tr, &beats, period, frame, |t| rate.snap_nearest(t), &others) {
                Some(snap) => {
                    if let Some(sp) = spans.iter_mut().find(|(id, _)| *id == tr.id) {
                        sp.1 = (snap.start, snap.start + snap.duration);
                    }
                    plan.push((tr.id, snap));
                }
                None => skipped.push(json!({"transition": tr.id.0, "reason": "no beat in reach fits between its clips"})),
            }
        }
    }
    if !plan.is_empty() {
        s.edit("Snap Transitions to Beats", |pr, _| {
            let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            for (id, snap) in &plan {
                filmcraft_edit::transitions::set_span(q, *id, snap.start, snap.duration, frame, frame).map_err(|e| bad(CMD, e.to_string()))?;
            }
            q.check().map_err(EngineError::Other)?;
            Ok(())
        })?;
    }
    let snapped: Vec<Value> = plan
        .iter()
        .map(|(id, sn)| {
            let mid = sn.start + sn.duration.mul_ratio(1, 2);
            json!({
                "transition": id.0,
                "start": sn.start.0,
                "duration": sn.duration.0,
                "beat": sn.beat.map(|b| b.0),
                "offBeat": sn.beat.map(|b| (mid - b).0),
                "beats": fraction_label(sn.beats),
            })
        })
        .collect();
    Ok(json!({
        "snapped": snapped,
        "skipped": skipped,
        "beatCount": beats.len(),
        "period": period.0,
        "bpm": TICKS_PER_SECOND as f64 * 60.0 / period.0.max(1) as f64,
        "source": source,
        "fractions": BEAT_FRACTIONS.iter().map(|f| fraction_label(*f)).collect::<Vec<_>>(),
    }))
}

// ---------------------------------------------------------------------------------------------
// Sounds

/// The video transitions a sound command works on: `transitions`, else the selected ones. Each
/// must exist, be on a video track and not be locked.
fn sound_targets(s: &Session, p: &Value, cmd: &str) -> Result<Vec<TransitionId>> {
    let ids = transition_ids(p, cmd)?.unwrap_or_else(|| s.state.transition_selection.clone());
    if ids.is_empty() {
        return Err(bad(cmd, "no transition given or selected"));
    }
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    for id in &ids {
        let track: &Track = seq.all_tracks().find(|t| t.transitions.iter().any(|x| x.id == *id)).ok_or_else(|| bad(cmd, format!("no transition {}", id.0)))?;
        if track.kind != TrackKind::Video {
            return Err(bad(cmd, format!("transition {} is an audio transition: sounds go with video transitions", id.0)));
        }
        if track.locked {
            return Err(bad(cmd, format!("transition {} is on a locked track", id.0)));
        }
    }
    Ok(ids)
}

/// Where each of `ids`' sounds plays: ticks and samples at the sequence's rate.
fn sound_report(s: &Session, ids: &[TransitionId]) -> Vec<Value> {
    let Some(seq) = s.active_sequence() else { return Vec::new() };
    let sr = seq.settings.sample_rate;
    ids.iter()
        .filter_map(|id| seq.all_tracks().flat_map(|t| t.transitions.iter()).find(|x| x.id == *id))
        .map(|tr| match &tr.sound {
            Some(snd) => {
                let (at, len) = snd.span(tr.start, tr.duration);
                let samples = snd.samples(tr.start, tr.duration, sr);
                json!({
                    "transition": tr.id.0, "kind": snd.kind.id(), "gain": snd.gain_db, "intensity": snd.intensity, "seed": snd.seed,
                    "start": at.0, "duration": len.0, "sampleRate": sr,
                    "startSample": samples.map(|s| s.0), "samples": samples.map(|s| s.1),
                })
            }
            None => json!({"transition": tr.id.0, "kind": Value::Null}),
        })
        .collect()
}

/// `transitions.addSound`.
fn add_sound(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "transitions.addSound";
    // (no kind: each transition's own pairing, a whoosh for those without one)
    let kind = match str_p(p, "kind") {
        Some(k) => Some(SoundKind::from_id(k).ok_or_else(|| bad(CMD, format!("unknown kind `{k}` (whoosh, riser, impact)")))?),
        None if p.get("kind").is_some_and(|v| !v.is_null()) => return Err(bad(CMD, "`kind` is whoosh, riser or impact")),
        None => None,
    };
    let gain = match p.get("gain") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|g| {
                    g.is_finite() && (filmcraft_project::transition_sound::GAIN_RANGE.0..=filmcraft_project::transition_sound::GAIN_RANGE.1).contains(g)
                })
                .ok_or_else(|| bad(CMD, "`gain` is dB from -60 to 12"))?,
        ),
    };
    let intensity = match p.get("intensity") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().filter(|i| (0.0..=1.0).contains(i)).ok_or_else(|| bad(CMD, "`intensity` is from 0 to 1"))?),
    };
    let seconds = match f64_p(p, "seconds") {
        None if p.get("seconds").is_some_and(|v| !v.is_null()) => return Err(bad(CMD, "`seconds` is a length in seconds")),
        None => None,
        Some(sec) if sec.is_finite() && sec > 0.0 && sec <= filmcraft_project::transition_sound::MAX_SECONDS => Some(Tick::from_seconds_f64(sec)),
        Some(_) => return Err(bad(CMD, format!("`seconds` is more than 0 and at most {}", filmcraft_project::transition_sound::MAX_SECONDS))),
    };
    let seed = whole_p(p, "seed");
    if p.get("seed").is_some_and(|v| !v.is_null()) && seed.is_none() {
        return Err(bad(CMD, "`seed` is a whole number"));
    }
    let ids = sound_targets(s, p, CMD)?;
    let label = kind.map_or_else(|| "Add Transition Sound".to_string(), |k| format!("Add {} Sound", k.label()));
    s.edit_sequence(&label, |q, _, _| {
        for t in q.video_tracks.iter_mut() {
            for x in t.transitions.iter_mut().filter(|x| ids.contains(&x.id)) {
                let kind = kind.or_else(|| creator_transitions::paired_sound(&x.effect.effect)).unwrap_or(SoundKind::Whoosh);
                // each transition its own sound unless a seed is given: the id is stable
                let mut snd = TransitionSound::new(kind, seed.unwrap_or(x.id.0));
                if let Some(g) = gain {
                    snd.gain_db = g;
                }
                if let Some(i) = intensity {
                    snd.intensity = i;
                }
                snd.length = seconds;
                x.sound = Some(snd);
            }
        }
        Ok(())
    })?;
    Ok(json!({"sounds": sound_report(s, &ids)}))
}

/// `transitions.removeSound`.
fn remove_sound(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "transitions.removeSound";
    let ids = sound_targets(s, p, CMD)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    if !seq.video_tracks.iter().flat_map(|t| t.transitions.iter()).any(|x| ids.contains(&x.id) && x.sound.is_some()) {
        return Err(bad(CMD, "the transitions have no sound"));
    }
    s.edit_sequence("Remove Transition Sound", |q, _, _| {
        for x in q.video_tracks.iter_mut().flat_map(|t| t.transitions.iter_mut()).filter(|x| ids.contains(&x.id)) {
            x.sound = None;
        }
        Ok(())
    })?;
    Ok(json!({"removed": ids.iter().map(|i| i.0).collect::<Vec<_>>()}))
}

// ---------------------------------------------------------------------------------------------
// Transitions on every cut / on strong beats

/// Longest `duration`, seconds.
const MAX_TRANSITION_SECONDS: f64 = 60.0;
/// Widest `tolerance`, seconds.
const MAX_TOLERANCE: f64 = 0.5;

/// The video transition a call names: an exact id, a Creator preset by name, any transition by name.
fn transition_def(name: &str) -> Option<&'static EffectDef> {
    let exact = filmcraft_project::find_effect(name.trim()).filter(|d| d.kind == EffectKind::VideoTransition);
    exact
        .or_else(|| creator_transitions::find_creator(name).and_then(|c| filmcraft_project::find_effect(c.id)))
        .or_else(|| filmcraft_project::vtransition::find_transition(name, EffectKind::VideoTransition))
}

/// A transition's own length: a Creator preset's, else Settings ▸ Timeline's default.
fn default_length(s: &Session, def: &EffectDef, rate: filmcraft_time::FrameRate) -> Tick {
    match creator_transitions::creator_transition(def.id) {
        Some(c) => Tick::from_seconds_f64(c.seconds),
        None => s.prefs.timeline.video_transition_duration(rate),
    }
}

/// `duration` (seconds) of a call, if given.
fn duration_p(p: &Value, cmd: &str) -> Result<Option<Tick>> {
    match p.get("duration") {
        None | Some(Value::Null) => Ok(None),
        Some(_) => f64_p(p, "duration")
            .filter(|d| d.is_finite() && *d > 0.0 && *d <= MAX_TRANSITION_SECONDS)
            .map(|d| Some(Tick::from_seconds_f64(d)))
            .ok_or_else(|| bad(cmd, format!("`duration` is seconds, more than 0 and at most {MAX_TRANSITION_SECONDS}"))),
    }
}

/// The cuts of the video tracks a call works on (`track`, else every video track): (track,
/// outgoing clip, incoming clip, cut), in time order per track. Locked tracks are left out.
fn cuts(seq: &Sequence, p: &Value, cmd: &str) -> Result<Vec<(TrackId, ClipId, ClipId, Tick)>> {
    let tracks: Vec<TrackId> = match p.get("track") {
        None | Some(Value::Null) => seq.video_tracks.iter().map(|t| t.id).collect(),
        Some(v) => vec![track_ref(seq, v).ok_or_else(|| bad(cmd, format!("no track {v} in this sequence (`track`)")))?],
    };
    let mut out = Vec::new();
    for tid in tracks {
        let Some(tr) = seq.track(tid).filter(|t| t.kind == TrackKind::Video && !t.locked) else { continue };
        let mut items: Vec<_> = tr.items.iter().collect();
        items.sort_by_key(|i| i.start);
        for w in items.windows(2) {
            if let [a, b] = w
                && a.end() == b.start
            {
                out.push((tid, a.id, b.id, b.start));
            }
        }
    }
    Ok(out)
}

/// A transition `def` centred on the cut between `from` and `to`, as long as `want` on whole
/// frames, no longer than either clip. None when a clip is shorter than a frame.
fn centred(seq: &Sequence, def: &EffectDef, from: ClipId, to: ClipId, cut: Tick, want: Tick) -> Option<Transition> {
    let rate = seq.settings.frame_rate.sane();
    let frame = rate.frame_duration().max(Tick(1));
    let (_, a) = seq.find_item(from)?;
    let (_, b) = seq.find_item(to)?;
    let room = a.duration.min(b.duration);
    let frames = (want.0 as f64 / frame.0 as f64).round().clamp(1.0, 1e9) as i64;
    let dur = Tick(frame.0.saturating_mul(frames)).min(room);
    // a whole number of frames
    let dur = Tick(dur.0 / frame.0 * frame.0);
    if dur < frame {
        return None;
    }
    let start = rate.snap(cut - dur.mul_ratio(1, 2)).max(a.start);
    let start = if start + dur > b.end() { b.end() - dur } else { start };
    Some(Transition {
        id: TransitionId(0),
        effect: def.instance(),
        start,
        duration: dur,
        from: Some(from),
        to: Some(to),
        align: Default::default(),
        reverse: false,
        sound: None,
    })
}

/// Whether a transition already joins `from` and `to`.
fn joined(seq: &Sequence, track: TrackId, from: ClipId, to: ClipId) -> bool {
    seq.track(track).is_some_and(|t| t.transitions.iter().any(|x| x.from == Some(from) && x.to == Some(to)))
}

/// Add `plan` (track, transition) as one undo step `label`, with paired sounds when `sound`.
fn place_all(s: &mut Session, label: &str, plan: Vec<(TrackId, Transition)>, sound: bool) -> Result<Vec<Value>> {
    s.edit_sequence(label, |q, ctx, _| {
        let mut placed = Vec::new();
        for (track, tr) in plan {
            let (effect, start, duration, from, to) = (tr.effect.effect.clone(), tr.start, tr.duration, tr.from, tr.to);
            let id = filmcraft_edit::add_transition(q, track, tr, ctx)?;
            if sound
                && let Some(kind) = creator_transitions::paired_sound(&effect)
                && let Some(x) = q.track_mut(track).and_then(|t| t.transitions.iter_mut().find(|x| x.id == id))
            {
                // (each its own sound: the seed is its id, as `transitions.addSound` does)
                x.sound = Some(TransitionSound::new(kind, id.0));
            }
            placed.push(
                json!({"transition": id.0, "effect": effect, "start": start.0, "duration": duration.0, "from": from.map(|c| c.0), "to": to.map(|c| c.0)}),
            );
        }
        q.check().map_err(EngineError::Other)?;
        Ok(placed)
    })
}

/// `transitions.applyToAll`.
fn apply_to_all(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "transitions.applyToAll";
    let name = str_p(p, "preset").ok_or_else(|| bad(CMD, "`preset` names a transition: a Creator preset (Pull In, Swipe Left…) or any video transition"))?;
    let def = transition_def(name).ok_or_else(|| bad(CMD, format!("unknown transition `{name}`")))?;
    let replace = p.get("replace").and_then(Value::as_bool).unwrap_or(true);
    let sound = p.get("sound").and_then(Value::as_bool).unwrap_or(false);
    let seq_id = sequence_id(s, p, CMD)?;
    if s.state.active_sequence != Some(seq_id) {
        return Err(bad(CMD, "`sequence` must be the active sequence"));
    }
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate.sane();
    let want = duration_p(p, CMD)?.unwrap_or_else(|| default_length(s, def, rate));
    let mut plan = Vec::new();
    let mut skipped = Vec::new();
    for (track, from, to, cut) in cuts(seq, p, CMD)? {
        if !replace && joined(seq, track, from, to) {
            skipped.push(json!({"from": from.0, "to": to.0, "reason": "a transition is already on this cut"}));
            continue;
        }
        match centred(seq, def, from, to, cut, want) {
            Some(tr) => plan.push((track, tr)),
            None => skipped.push(json!({"from": from.0, "to": to.0, "reason": "a clip is shorter than a frame"})),
        }
    }
    if plan.is_empty() {
        return Err(bad(CMD, "no cut to put a transition on (a cut is where a clip ends and the next on its track starts)"));
    }
    let placed = place_all(s, &format!("Apply {} to All Cuts", def.name), plan, sound)?;
    Ok(json!({"preset": def.id, "transitions": placed, "skipped": skipped}))
}

/// The video transitions of `folder` (`Creator`, `Light & Film`, or a path such as
/// `Video Transitions/Creator`), by id.
fn folder_transitions(folder: &str) -> Vec<&'static EffectDef> {
    let f = folder.trim();
    let f = f.strip_prefix("Video Transitions/").unwrap_or(f);
    let mut v: Vec<&'static EffectDef> = filmcraft_project::effect_defs()
        .iter()
        .filter(|d| d.kind == EffectKind::VideoTransition && d.category.first() == Some(&"Video Transitions"))
        .filter(|d| d.category.get(1).is_some_and(|c| c.eq_ignore_ascii_case(f)))
        .collect();
    v.sort_by_key(|d| d.id);
    v
}

/// SplitMix64.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `transitions.randomOnBeats`.
fn random_on_beats(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "transitions.randomOnBeats";
    let folder = str_p(p, "folder").unwrap_or("Creator");
    let pool = folder_transitions(folder);
    if pool.is_empty() {
        return Err(bad(CMD, format!("no video transitions in folder `{folder}`")));
    }
    let seed = match p.get("seed") {
        None | Some(Value::Null) => 1,
        Some(_) => whole_p(p, "seed").ok_or_else(|| bad(CMD, "`seed` is a whole number"))?,
    };
    let every = match p.get("everyNthBeat") {
        None | Some(Value::Null) => 2,
        Some(v) => v.as_u64().filter(|n| (1..=64).contains(n)).ok_or_else(|| bad(CMD, "`everyNthBeat` is a whole number from 1 to 64"))? as usize,
    };
    let tolerance = match p.get("tolerance") {
        None | Some(Value::Null) => 0.1,
        Some(_) => f64_p(p, "tolerance")
            .filter(|t| t.is_finite() && (0.0..=MAX_TOLERANCE).contains(t))
            .ok_or_else(|| bad(CMD, format!("`tolerance` is seconds from 0 to {MAX_TOLERANCE}")))?,
    };
    let sound = p.get("sound").and_then(Value::as_bool).unwrap_or(false);
    let fixed = duration_p(p, CMD)?;
    let seq_id = sequence_id(s, p, CMD)?;
    if s.state.active_sequence != Some(seq_id) {
        return Err(bad(CMD, "`sequence` must be the active sequence"));
    }
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate.sane();
    let (beats, period, source) = beat_grid(s, p, seq, CMD)?;
    let strong: Vec<Tick> = beats.iter().copied().step_by(every).collect();
    let tol = Tick::from_seconds_f64(tolerance);
    // 90 % of a beat, in whole frames (at least one)
    let frame = rate.frame_duration().max(Tick(1));
    let longest = Tick((period.mul_ratio(9, 10).0 / frame.0).max(1).saturating_mul(frame.0));
    let mut plan = Vec::new();
    let mut on_beats = Vec::new();
    let mut off = 0usize;
    let mut last: Option<(TrackId, &str)> = None;
    for (k, (track, from, to, cut)) in cuts(seq, p, CMD)?.into_iter().enumerate() {
        let i = strong.partition_point(|b| *b < cut);
        let near = [i.checked_sub(1), Some(i)].into_iter().flatten().filter_map(|j| strong.get(j)).min_by_key(|b| (**b - cut).abs()).copied();
        let Some(beat) = near.filter(|b| (*b - cut).abs() <= tol) else {
            off += 1;
            continue;
        };
        let n = pool.len();
        let mut pick = (mix64(seed ^ mix64(k as u64 + 1)) % n as u64) as usize;
        if n > 1 && last.is_some_and(|(t, id)| t == track && pool.get(pick).is_some_and(|d| d.id == id)) {
            pick = (pick + 1) % n;
        }
        let Some(def) = pool.get(pick).copied() else { continue };
        let want = fixed.unwrap_or_else(|| default_length(s, def, rate)).min(longest);
        if let Some(tr) = centred(seq, def, from, to, cut, want) {
            last = Some((track, def.id));
            on_beats.push((cut, beat));
            plan.push((track, tr));
        }
    }
    if plan.is_empty() {
        return Err(bad(CMD, format!("no cut lands within {tolerance} s of a strong beat ({off} cuts off the beat)")));
    }
    let placed = place_all(s, "Transitions on Beats", plan, sound)?;
    let placed: Vec<Value> = placed
        .into_iter()
        .zip(on_beats)
        .map(|(mut v, (cut, beat))| {
            v["cut"] = json!(cut.0);
            v["beat"] = json!(beat.0);
            v
        })
        .collect();
    Ok(json!({"folder": folder, "seed": seed, "transitions": placed, "offBeat": off, "strongBeats": strong.len(), "source": source}))
}

#[cfg(test)]
#[path = "transition_tools_creator_tests.rs"]
mod creator_tests;
