//! Revision-safe editable stems, deterministic loop crossfades, and timestamped audio joins.
use crate::commands::{CommandSpec, bad, has_seq, track_p};
use crate::{EngineError, Result, Session};
use filmcraft_project::{AudioType, ClipId, EssentialSound, ItemId, ParamValue, TrackId};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// The step between the samples either side of a boundary (full scale is 1) that counts as a click.
const CLICK_STEP: f64 = 0.35;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, params, run, journal, enabled: has_seq, menu: &[], shortcut: None }
}
pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "audio.configureStems",
            "Configure editable stems",
            r#"{"roles":{"A1":"source","A2":"narration","A3":"music","A4":"effects"},"muteSourceClips":[id]?}"#,
            configure,
            true,
        ),
        spec(
            "sequence.continuousMusic",
            "Build continuous music",
            r#"{"item":id,"track":"A3","replaceTracks":["A3"]?,"start":ticks=0,"end":ticks?,"sourceIn":ticks=0,"crossfade":ticks?,"gainDb":-60..0=-25}"#,
            bed,
            true,
        ),
        spec(
            "sequence.checkContinuity",
            "Check audio continuity",
            r#"{"times":[ticks]?,"maxJumpDb":1..30=6,"exemptions":[{"start":ticks,"end":ticks,"reason":str}]?}"#,
            check,
            false,
        ),
    ]
}
fn configure(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "audio.configureStems";
    let roles = p["roles"].as_object().filter(|m| m.len() <= 256).ok_or_else(|| bad(cmd, "roles must map audio tracks to roles"))?;
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for (track, role) in roles {
        let id = track_p(s, &json!({"track":track}), "track", cmd)?.ok_or_else(|| bad(cmd, "no such track"))?;
        if !seen.insert(id) {
            return Err(bad(cmd, "each track may have only one role"));
        }
        let (name, kind) = match role.as_str() {
            Some("narration") => ("Narration", AudioType::Dialogue),
            Some("music") => ("Music", AudioType::Music),
            Some("source") => ("Source", AudioType::Ambience),
            Some("effects") => ("Effects", AudioType::Sfx),
            _ => return Err(bad(cmd, "role is narration, music, source or effects")),
        };
        if !s.active_sequence().is_some_and(|q| q.audio_tracks.iter().any(|t| t.id == id)) {
            return Err(bad(cmd, "stem must be an audio track"));
        }
        targets.push((id, name, kind));
    }
    let mute = match p.get("muteSourceClips") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .filter(|v| v.len() <= 10000)
            .ok_or_else(|| bad(cmd, "muteSourceClips needs up to 10000 identities"))?
            .iter()
            .map(|v| v.as_u64().map(ClipId).ok_or_else(|| bad(cmd, "clip identity must be an unsigned integer")))
            .collect::<Result<Vec<_>>>()?,
    };
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let mut links = BTreeSet::new();
    for id in &mute {
        let (_, it) = q.find_item(*id).ok_or_else(|| bad(cmd, "source clip is absent"))?;
        if let Some(link) = it.link {
            links.insert(link);
        }
    }
    s.edit_sequence("Configure audio stems", |q, _, _| {
        for t in &mut q.audio_tracks {
            if let Some((_, name, kind)) = targets.iter().find(|(id, _, _)| *id == t.id) {
                t.name = (*name).into();
                for c in &mut t.items {
                    // Role assignment must not reset repair, gain, ducking or a privacy mute.
                    c.essential.get_or_insert_with(|| EssentialSound::new(*kind)).kind = *kind;
                }
            }
            for c in &mut t.items {
                if mute.contains(&c.id) || c.link.is_some_and(|l| links.contains(&l)) {
                    c.enabled = false;
                }
            }
        }
        Ok(())
    })?;
    Ok(json!({"roles":roles,"sourceMuted":mute.iter().map(|c|c.0).collect::<Vec<_>>(),"editable":true}))
}
fn ticks(p: &Value, key: &str, default: i64, cmd: &str) -> Result<i64> {
    match p.get(key) {
        None => Ok(default),
        Some(v) => {
            v.as_i64().filter(|n| *n >= 0 && *n <= Tick::from_seconds_f64(12.0 * 3600.0).0).ok_or_else(|| bad(cmd, format!("{key}: ticks in 0..12 hours")))
        }
    }
}
fn bed(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "sequence.continuousMusic";
    let item = ItemId(p["item"].as_u64().ok_or_else(|| bad(cmd, "provide music item"))?);
    let pi = s.project.item(item).ok_or_else(|| bad(cmd, "no music source"))?;
    if !pi.has_audio() {
        return Err(bad(cmd, "music source has no sound"));
    }
    let start = ticks(p, "start", 0, cmd)?;
    let end = ticks(p, "end", s.active_sequence().ok_or(EngineError::NoSequence)?.duration().0, cmd)?;
    let source = ticks(p, "sourceIn", 0, cmd)?;
    let available = pi.duration().0.checked_sub(source).filter(|v| *v > 0).ok_or_else(|| bad(cmd, "sourceIn must precede the source end"))?;
    let fade = ticks(p, "crossfade", Tick::from_seconds_f64(1.0).0.min(available / 4), cmd)?;
    if fade <= 0 || fade > available / 2 || end <= start {
        return Err(bad(cmd, "crossfade must be positive and at most half the music source; end must follow start"));
    }
    let step = available - fade;
    let span = end - start;
    let count = if span <= available { 1 } else { 1 + (span - available + step - 1) / step };
    if count > 2000 {
        return Err(bad(cmd, "music bed would need more than 2000 loops"));
    }
    let gain = p
        .get("gainDb")
        .map_or(Some(-25.0), Value::as_f64)
        .filter(|v| v.is_finite() && (-60.0..=0.0).contains(v))
        .ok_or_else(|| bad(cmd, "gainDb is -60..0"))?;
    let target = track_p(s, p, "track", cmd)?.ok_or_else(|| bad(cmd, "choose an existing audio track"))?;
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    if !q.audio_tracks.iter().any(|t| t.id == target) {
        return Err(bad(cmd, "music destination must be audio"));
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let partner_key = format!("autocut.musicPartner.{}", target.0);
    let existing_partner = s
        .project
        .item(seq_id)
        .and_then(|i| i.metadata.get(&partner_key))
        .map(|v| v.parse::<u64>().map(TrackId).map_err(|_| bad(cmd, "invalid music partner identity")))
        .transpose()?
        // A partner track deleted since the last bed is gone, not an error: a new one is made
        // (the saved identity blocked every later rebuild).
        .filter(|partner| q.audio_tracks.iter().any(|t| t.id == *partner));
    if let Some(partner) = existing_partner {
        let other_content =
            q.audio_tracks.iter().filter(|t| t.id == partner).flat_map(|t| &t.items).any(|c| c.essential.as_ref().is_none_or(|e| e.kind != AudioType::Music));
        if partner == target || other_content {
            return Err(bad(cmd, "music partner contains other content; move it before rebuilding the bed"));
        }
    } else if count > 1 && q.audio_tracks.len() >= 256 {
        return Err(bad(cmd, "crossfaded music needs one free audio track"));
    }
    let mut replace = BTreeSet::from([target]);
    if let Some(partner) = existing_partner {
        replace.insert(partner);
    }
    if let Some(v) = p.get("replaceTracks") {
        for tr in v.as_array().filter(|a| a.len() <= 256).ok_or_else(|| bad(cmd, "replaceTracks is an array"))? {
            let id = track_p(s, &json!({"track":tr}), "track", cmd)?.ok_or_else(|| bad(cmd, "unknown replace track"))?;
            if !q.audio_tracks.iter().any(|t| t.id == id) {
                return Err(bad(cmd, "replace track must be audio"));
            }
            replace.insert(id);
        }
    }
    if q.audio_tracks.iter().any(|t| replace.contains(&t.id) && t.locked) {
        return Err(bad(cmd, "a replacement music track is locked"));
    }
    s.edit_group("Continuous music bed", |s| {
        let partner = if count > 1 && existing_partner.is_none() {
            Some(s.edit("Create music crossfade partner", |p, _| {
                let mut track =
                    p.sequence(seq_id).and_then(|q| q.audio_tracks.iter().find(|t| t.id == target)).cloned().ok_or_else(|| bad(cmd, "music track vanished"))?;
                let id = TrackId(p.alloc_id());
                track.id = id;
                track.items.clear();
                track.transitions.clear();
                track.locked = false;
                track.solo = false;
                track.name = "Continuous Music B".into();
                track.mixer.record_arm = false;
                p.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?.audio_tracks.push(track);
                if let Some(i) = p.item_mut(seq_id) {
                    i.metadata.insert(partner_key.clone(), id.0.to_string());
                }
                Ok(id)
            })?)
        } else {
            existing_partner
        };
        s.edit_sequence("Replace music stems", |q, _, _| {
            for t in &mut q.audio_tracks {
                if replace.contains(&t.id) {
                    t.items.clear();
                    t.transitions.clear();
                }
            }
            Ok(())
        })?;
        let duration = available.min(span);
        let ids = crate::commands::place_item(
            s,
            item,
            TimeRange::new(Tick(source), Tick(duration)),
            Tick(start),
            None,
            Some(target),
            false,
            "Place continuous music",
            None,
        )?;
        let template = s
            .active_sequence()
            .and_then(|q| ids.first().and_then(|id| q.find_item(*id).map(|(_, c)| c.clone())))
            .ok_or_else(|| bad(cmd, "no audio clip placed"))?;
        s.edit_sequence("Crossfade music loops", |q, ctx, _| {
            for t in &mut q.audio_tracks {
                if t.id == target || Some(t.id) == partner {
                    t.items.clear();
                    t.name = if t.id == target { "Continuous Music A" } else { "Continuous Music B" }.into();
                }
            }
            for n in 0..count {
                let at = start + n * step;
                if at >= end {
                    break;
                }
                let len = available.min(end - at);
                let mut c = template.clone();
                c.id = ClipId(ctx.alloc());
                c.start = Tick(at);
                c.duration = Tick(len);
                c.gain_db = gain;
                c.link = None;
                c.essential = Some(EssentialSound::new(AudioType::Music));
                if let Some(effect) = c.effects.iter_mut().find(|e| e.effect == "volume")
                    && let Some(level) = effect.params.get_mut("level")
                {
                    level.keyframes.clear();
                    level.value = ParamValue::Float(0.0);
                    // Equal-power envelopes, represented by editable dB keys in source time.
                    for k in 0..=16 {
                        let u = k as f64 / 16.0;
                        let db = |v: f64| 20.0 * v.max(0.00001585).log10();
                        let f = fade.min(len / 2);
                        level.put_keyframe(Tick(source + ((f as f64) * u).round() as i64), ParamValue::Float(db((u * std::f64::consts::FRAC_PI_2).sin())));
                        level.put_keyframe(
                            Tick(source + len - f + ((f as f64) * u).round() as i64),
                            ParamValue::Float(db((u * std::f64::consts::FRAC_PI_2).cos())),
                        );
                    }
                }
                let destination = if n % 2 == 0 { target } else { partner.ok_or_else(|| bad(cmd, "music crossfade partner is missing"))? };
                q.audio_tracks.iter_mut().find(|t| t.id == destination).ok_or_else(|| bad(cmd, "music track vanished"))?.items.push(c);
            }
            Ok(())
        })?;
        Ok(json!({"track":target.0,"partnerTrack":partner.map(|id|id.0),"loops":count,"start":start,"end":end,"crossfade":fade,"editable":true,"seed":0}))
    })
}
fn check(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "sequence.checkContinuity";
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let sr = q.settings.sample_rate.clamp(8000, 384000);
    let limit = p
        .get("maxJumpDb")
        .map_or(Some(6.0), Value::as_f64)
        .filter(|v| v.is_finite() && (1.0..=30.0).contains(v))
        .ok_or_else(|| bad(cmd, "maxJumpDb must be 1..30"))?;
    let mut times = BTreeSet::new();
    if let Some(v) = p.get("times") {
        for t in v.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(cmd, "times needs up to 2000 tick values"))? {
            times.insert(t.as_i64().filter(|n| *n >= 0 && *n <= q.duration().0).ok_or_else(|| bad(cmd, "time outside sequence"))?);
        }
    } else {
        for t in q.video_tracks.iter().chain(q.audio_tracks.iter()) {
            for c in &t.items {
                times.insert(c.start.0);
                times.insert(c.end().0);
            }
        }
    }
    if times.len() > 2000 {
        return Err(bad(cmd, "more than 2000 boundaries; provide a bounded times selection"));
    }
    let exemptions = match p.get("exemptions") {
        None => Vec::new(),
        Some(v) => v.as_array().cloned().ok_or_else(|| bad(cmd, "exemptions must be an array"))?,
    };
    if exemptions.len() > 2000 {
        return Err(bad(cmd, "too many exemptions"));
    }
    for e in &exemptions {
        if e["start"].as_i64().is_none()
            || e["end"].as_i64().is_none()
            || e["end"].as_i64() < e["start"].as_i64()
            || e["reason"].as_str().is_none_or(|s| s.trim().is_empty() || s.len() > 512)
        {
            return Err(bad(cmd, "exemption needs ordered tick start/end and a short reason"));
        }
    }
    let project = s.project.clone();
    let provider = s.media.full_res_provider(project.clone(), s.services.clone());
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let radius = (sr / 10) as usize;
    let mut findings = Vec::new();
    let mut measured = Vec::new();
    for t in times.into_iter().filter(|t| *t > 0 && *t < q.duration().0) {
        let sample = Tick(t).to_units_floor(i64::from(sr));
        let first = sample.saturating_sub(radius as i64);
        if first < 0 {
            continue;
        }
        let buffer = filmcraft_render::audio::mix_sequence(&project, q, first, radius * 2, &provider, &filmcraft_render::MixStream::new());
        let mono: Vec<f64> = (0..radius * 2)
            .map(|i| buffer.channels.iter().filter_map(|c| c.get(i)).map(|v| f64::from(*v)).sum::<f64>() / buffer.channels.len().max(1) as f64)
            .collect();
        let Some((left, right)) = mono.split_at_checked(radius) else { continue };
        let db = |v: &[f64]| 10.0 * (v.iter().map(|v| v * v).sum::<f64>() / v.len().max(1) as f64).max(1e-12).log10();
        let (before, after) = (db(left), db(right));
        let step = (left.last().copied().unwrap_or(0.0) - right.first().copied().unwrap_or(0.0)).abs();
        let exempt = exemptions.iter().find(|e| e["start"].as_i64().is_some_and(|a| a <= t) && e["end"].as_i64().is_some_and(|b| b >= t));
        // A step between neighbouring samples is a click, whatever plays; a level change over
        // 100 ms is also what speech, music and sound effects do at every onset.
        let click = step > CLICK_STEP;
        let issue = (before > -60.0 && after > -60.0 && (after - before).abs() > limit) || click;
        let row = json!({"time":t,"seconds":Tick(t).seconds(),"rmsBeforeDb":before,"rmsAfterDb":after,"jumpDb":after-before,"sampleStep":step,"click":click,"exemption":exempt});
        if issue && exempt.is_none() {
            findings.push(row.clone());
        }
        measured.push(row);
    }
    Ok(
        json!({"sequence":seq.0,"passed":findings.is_empty(),"findings":findings,"boundaries":measured,"method":"100 ms RMS each side plus sample discontinuity; not a LUFS measurement"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session() -> (Session, ItemId) {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().audio_tracks.iter().flat_map(|t| &t.items).next().unwrap().item;
        (s, item)
    }
    #[test]
    fn music_is_separate_editable_deterministic_and_undoable() {
        let (mut s, item) = session();
        let before = s.project.clone();
        let p = json!({"item":item.0,"track":"A3","end":Tick::from_seconds_f64(7.0).0,"crossfade":Tick::from_seconds_f64(0.3).0});
        let first = s.execute("sequence.continuousMusic", p.clone()).unwrap();
        let count = s.active_sequence().unwrap().audio_tracks[2].items.len();
        assert!(count > 0);
        assert_eq!(first["loops"].as_u64().unwrap() as usize, count);
        let music = &s.active_sequence().unwrap().audio_tracks[2];
        assert!(music.items.iter().all(|c| c.essential.as_ref().unwrap().kind == AudioType::Music));
        assert!(music.items.iter().all(|c| c.effects.iter().any(|e| e.effect == "volume" && e.params["level"].keyframes.len() >= 16)));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(&*s.project, &*before);
        assert_eq!(s.execute("sequence.continuousMusic", p).unwrap()["loops"], first["loops"]);
    }
    #[test]
    fn stem_mute_persists_and_bad_requests_are_atomic() {
        let (mut s, _) = session();
        let q = s.active_sequence().unwrap();
        let id = q.audio_tracks[0].items[0].id;
        s.execute("audio.configureStems", json!({"roles":{"A1":"source","A2":"narration","A3":"music"},"muteSourceClips":[id.0]})).unwrap();
        assert!(!s.active_sequence().unwrap().audio_tracks[0].items[0].enabled);
        let before = s.project.clone();
        assert!(s.execute("sequence.continuousMusic", json!({"item":1,"crossfade":0,"track":"A3"})).is_err());
        assert_eq!(&*s.project, &*before);
    }
    #[test]
    fn boundary_findings_have_times_and_deliberate_silence_exemptions() {
        let (mut s, _) = session();
        let r = s.execute("sequence.checkContinuity", json!({"times":[Tick::from_seconds_f64(1.0).0]})).unwrap();
        assert!(r["boundaries"].is_array());
        assert!(s.execute("sequence.checkContinuity", json!({"times":[-1]})).is_err());
        assert!(s.execute("sequence.checkContinuity", json!({"exemptions":[{"start":0,"end":1,"reason":""}]})).is_err());
    }
}

#[cfg(test)]
mod loop_regression {
    use super::*;
    #[test]
    fn repeated_music_uses_two_nonoverlapping_tracks_and_one_undo() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().audio_tracks[0].items[0].item;
        s.edit("Short music source", |p, _| {
            p.item_mut(item).unwrap().as_media_mut().unwrap().info.duration = Tick::from_seconds_f64(2.0);
            Ok(())
        })
        .unwrap();
        let before = s.project.clone();
        let p = json!({"item":item.0,"track":"A3","end":Tick::from_seconds_f64(7.0).0,"crossfade":Tick::from_seconds_f64(0.3).0});
        let first = s.execute("sequence.continuousMusic", p.clone()).unwrap();
        assert!(first["loops"].as_u64().unwrap() > 2);
        let q = s.active_sequence().unwrap();
        let count = q.audio_tracks.len();
        assert!(q.check_bounds().is_ok());
        let tracks = [first["track"].as_u64().unwrap(), first["partnerTrack"].as_u64().unwrap()];
        let clips: Vec<_> = q.audio_tracks.iter().filter(|t| tracks.contains(&t.id.0)).flat_map(|t| &t.items).collect();
        assert_eq!(clips.len() as u64, first["loops"].as_u64().unwrap());
        assert_eq!(clips.iter().map(|c| c.id).collect::<BTreeSet<_>>().len(), clips.len());
        let again = s.execute("sequence.continuousMusic", p).unwrap();
        assert_eq!(again["partnerTrack"], first["partnerTrack"]);
        assert_eq!(s.active_sequence().unwrap().audio_tracks.len(), count);
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(&*s.project, &*before);
    }

    /// The partner track's identity is saved with the sequence. Deleting that track by hand left
    /// the saved identity behind, and every later bed failed with "partner track is missing".
    #[test]
    fn a_deleted_partner_track_is_made_again() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().audio_tracks[0].items[0].item;
        s.edit("Short music source", |p, _| {
            p.item_mut(item).unwrap().as_media_mut().unwrap().info.duration = Tick::from_seconds_f64(2.0);
            Ok(())
        })
        .unwrap();
        let p = json!({"item":item.0,"track":"A3","end":Tick::from_seconds_f64(7.0).0,"crossfade":Tick::from_seconds_f64(0.3).0});
        let first = s.execute("sequence.continuousMusic", p.clone()).unwrap();
        let partner = TrackId(first["partnerTrack"].as_u64().unwrap());
        s.edit_sequence("Delete the partner track", |q, _, _| {
            q.audio_tracks.retain(|t| t.id != partner);
            Ok(())
        })
        .unwrap();
        let again = s.execute("sequence.continuousMusic", p).unwrap();
        let rebuilt = TrackId(again["partnerTrack"].as_u64().unwrap());
        assert_ne!(rebuilt, partner);
        let q = s.active_sequence().unwrap();
        assert!(q.audio_tracks.iter().any(|t| t.id == rebuilt && !t.items.is_empty()));
        assert!(q.check_bounds().is_ok());
    }
}
