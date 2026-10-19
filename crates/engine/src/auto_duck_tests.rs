//! Per-track auto-duck (`track.setAutoDuck`): the music track goes down by the amount under the
//! dialogue track's sound with the attack / release ramps, follows clip changes (inside their
//! undo step), turns off cleanly, survives save / load, and refuses hostile parameters.

use std::sync::Arc;

use filmcraft_project::{ClipId, TrackItem};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::Session;

const SR: u32 = 48_000;

/// Dialogue-like bursts (a buzzy tone) where `active(t)`, else silence.
fn bursts(secs: f64, active: impl Fn(f64) -> bool) -> Vec<f32> {
    (0..(secs * SR as f64) as usize)
        .map(|i| {
            let t = i as f64 / SR as f64;
            if !active(t) {
                return 0.0;
            }
            let v: f64 = (1..8).map(|k| (2.0 * std::f64::consts::PI * 140.0 * k as f64 * t).sin() / k as f64).sum();
            (0.2 * v) as f32
        })
        .collect()
}

fn tone(secs: f64, f: f64, amp: f32) -> Vec<f32> {
    (0..(secs * SR as f64) as usize).map(|i| (amp as f64 * (2.0 * std::f64::consts::PI * f * i as f64 / SR as f64).sin()) as f32).collect()
}

fn place(s: &mut Session, sig: &[f32], name: &str, track: &str, at: f64) -> ClipId {
    let inter: Vec<f32> = sig.iter().flat_map(|v| [*v, *v]).collect();
    let bytes: Arc<[u8]> = crate::previews::write_wav_f32(&inter, SR).into();
    let item = crate::commands::import_bytes(s, &format!("/{name}.wav"), bytes, None).unwrap();
    let r = s.execute("timeline.place", json!({"item": item.0, "audioTrack": track, "seconds": at})).unwrap();
    ClipId(r["clips"][0].as_u64().unwrap())
}

/// Dialogue on A1 in [2, 4] and [6, 7.5] s, music on A2 for 10 s.
fn session() -> (Session, ClipId, ClipId) {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "duck", "audio": 3, "video": 1})).unwrap();
    let dlg = place(&mut s, &bursts(10.0, |t| (2.0..4.0).contains(&t) || (6.0..7.5).contains(&t)), "dlg", "A1", 0.0);
    let music = place(&mut s, &tone(10.0, 220.0, 0.2), "music", "A2", 0.0);
    s.execute("edit.deselectAll", json!({})).unwrap();
    (s, dlg, music)
}

fn item(s: &Session, c: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
}

/// The music clip's Volume level (dB) at timeline second `t`.
fn level(s: &Session, c: ClipId, t: f64) -> f64 {
    let it = item(s, c);
    it.effect("volume").unwrap().param("level").unwrap().f64_at(it.source_time_at(Tick::from_seconds_f64(t)))
}

/// Amplitude of the 220 Hz tone on the track of clip `c` (its mixer input: clip effects, Volume,
/// Panner) over `[a, b)` seconds.
fn heard(s: &Session, c: ClipId, a: f64, b: f64) -> f64 {
    let seq = s.active_sequence().unwrap();
    let (tid, _) = seq.find_item(c).unwrap();
    let track = seq.track(tid).unwrap();
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let a0 = (a * SR as f64) as i64;
    let n = ((b - a) * SR as f64) as usize;
    let x = filmcraft_render::audio::track_input(&s.project, track, a0, n, SR, &provider);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, v) in x.channels[0].iter().enumerate() {
        let w = 2.0 * std::f64::consts::PI * 220.0 * (a0 + i as i64) as f64 / SR as f64;
        re += *v as f64 * w.cos();
        im += *v as f64 * w.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n as f64
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-12).log10()
}

#[test]
fn ducks_the_track_under_the_other_by_the_amount() {
    let (mut s, _, music) = session();
    let r = s.execute("track.setAutoDuck", json!({"track": "A2", "under": ["A1"], "amount": 15, "attack": 0.3, "release": 0.8})).unwrap();
    assert_eq!(r["enabled"], true, "{r}");
    let kf: Vec<(f64, f64)> =
        r["ducked"][0]["keyframes"].as_array().unwrap().iter().map(|k| (k["seconds"].as_f64().unwrap(), k["levelDb"].as_f64().unwrap())).collect();
    let want = [(1.7, 0.0), (2.0, -15.0), (4.0, -15.0), (4.8, 0.0), (5.7, 0.0), (6.0, -15.0), (7.5, -15.0), (8.3, 0.0)];
    assert_eq!(kf.len(), want.len(), "{kf:?}");
    for ((t, v), (wt, wv)) in kf.iter().zip(want) {
        assert!((t - wt).abs() <= 0.06 && (v - wv).abs() < 1e-9, "keyframe ({t}, {v}), expected ({wt}, {wv})");
    }
    assert_eq!(level(&s, music, 3.0), -15.0);
    assert_eq!(level(&s, music, 5.2), 0.0);
    // what plays: 15 dB down under the dialogue
    let d = db(heard(&s, music, 2.5, 3.5) / heard(&s, music, 0.2, 1.2));
    assert!((d + 15.0).abs() < 0.3, "ducked by {d:.2} dB");
    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(level(&s, music, 3.0), 0.0);
    assert!(s.active_sequence().unwrap().audio_tracks[1].auto_duck.is_none());
}

#[test]
fn reruns_when_the_clips_change_and_undo_takes_both_back() {
    let (mut s, dlg, music) = session();
    s.execute("track.setAutoDuck", json!({"track": "A2", "under": ["A1"], "amount": 12})).unwrap();
    assert_eq!(level(&s, music, 2.5), -12.0);
    // the dialogue moves 1 s later: the duck follows, in the same undo step as the move
    let a1 = s.active_sequence().unwrap().audio_tracks[0].id.0;
    s.execute("timeline.move", json!({"moves": [{"clip": dlg.0, "track": a1, "time": Tick::from_seconds_f64(1.0).0}]})).unwrap();
    assert_eq!(item(&s, dlg).start, Tick::from_seconds_f64(1.0));
    assert_eq!(level(&s, music, 2.5), 0.0, "no dialogue at 2.5 s any more");
    assert_eq!(level(&s, music, 3.5), -12.0);
    assert_eq!(level(&s, music, 8.0), -12.0);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(item(&s, dlg).start, Tick::ZERO);
    assert_eq!(level(&s, music, 2.5), -12.0, "undo brings back the duck of the old position");
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(level(&s, music, 2.5), 0.0);
    // muting the dialogue track lifts the duck; unmuting brings it back
    s.execute("timeline.setTrack", json!({"track": "A1", "muted": true})).unwrap();
    assert_eq!(level(&s, music, 3.5), 0.0);
    s.execute("timeline.setTrack", json!({"track": "A1", "muted": false})).unwrap();
    assert_eq!(level(&s, music, 3.5), -12.0);
    // a clip added to the ducked track is ducked too
    let more = place(&mut s, &tone(4.0, 220.0, 0.2), "more", "A2", 11.0);
    let dlg2 = place(&mut s, &bursts(2.0, |_| true), "dlg2", "A1", 12.0);
    assert_eq!(level(&s, more, 12.5), -12.0, "the new music clip is ducked under the new dialogue");
    assert_eq!(level(&s, more, 14.8), 0.0);
    // a change elsewhere does not touch the keyframes
    let before = item(&s, music);
    s.execute("timeline.setTrack", json!({"track": "A3", "muted": true})).unwrap();
    assert_eq!(item(&s, music), before);
    // new settings re-plan every clip
    s.execute("track.setAutoDuck", json!({"track": "A2", "amount": 6})).unwrap();
    assert_eq!(level(&s, music, 3.5), -6.0);
    assert_eq!(level(&s, more, 12.5), -6.0);
    // off: the levels come back, no keyframes left
    let r = s.execute("track.setAutoDuck", json!({"track": "A2", "enabled": false})).unwrap();
    assert_eq!(r["enabled"], false);
    for c in [music, more] {
        let it = item(&s, c);
        let p = it.effect("volume").unwrap().param("level").unwrap();
        assert!(p.keyframes.is_empty() && p.value.as_f64() == Some(0.0), "{c:?}");
    }
    // and the dialogue changing no longer touches the music
    s.execute("timeline.move", json!({"moves": [{"clip": dlg2.0, "track": a1, "time": Tick::from_seconds_f64(11.0).0}]})).unwrap();
    assert!(item(&s, more).effect("volume").unwrap().param("level").unwrap().keyframes.is_empty());
}

#[test]
fn a_clip_level_set_before_ducking_is_kept_and_restored() {
    let (mut s, _, music) = session();
    s.execute("effects.setParam", json!({"clip": music.0, "effect": "volume", "param": "level", "value": -4.0})).unwrap();
    s.execute("track.setAutoDuck", json!({"track": "A2", "amount": 10})).unwrap();
    assert_eq!(level(&s, music, 1.0), -4.0);
    assert_eq!(level(&s, music, 3.0), -14.0);
    // re-running does not lower the base: a whole-clip duck would otherwise drift down
    s.execute("track.setAutoDuck", json!({"track": "A2", "amount": 10})).unwrap();
    assert_eq!(level(&s, music, 1.0), -4.0);
    s.execute("track.setAutoDuck", json!({"track": "A2", "enabled": false})).unwrap();
    assert_eq!(level(&s, music, 3.0), -4.0);
}

#[test]
fn settings_survive_save_and_load() {
    let (mut s, _, _) = session();
    s.execute("track.setAutoDuck", json!({"track": "A2", "under": ["A1"], "amount": 9, "attack": 0.1, "release": 2.0, "sensitivity": 7})).unwrap();
    let back = filmcraft_format::decode(&filmcraft_format::encode(&s.project, false)).unwrap().project;
    assert_eq!(back, *s.project);
    let seq = s.active_sequence().unwrap();
    let ad = seq.audio_tracks[1].auto_duck.clone().unwrap();
    assert_eq!((ad.amount_db, ad.attack_s, ad.release_s, ad.sensitivity), (9.0, 0.1, 2.0, 7.0));
    assert_eq!(ad.under, vec![seq.audio_tracks[0].id]);
    assert!(seq.audio_tracks[0].auto_duck.is_none());
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, _, _) = session();
    let undo = s.history.undo.len();
    for p in [
        json!({}),
        json!({"track": "V1"}),
        json!({"track": "A9"}),
        json!({"track": "A2", "under": ["A2"]}),
        json!({"track": "A2", "under": ["V1"]}),
        json!({"track": "A2", "under": [{"x": 1}]}),
        json!({"track": "A2", "amount": 1e9}),
        json!({"track": "A2", "amount": "loud"}),
        json!({"track": "A2", "attack": 1e300}),
        json!({"track": "A2", "release": 0.0}),
        json!({"track": "A2", "sensitivity": 11}),
    ] {
        let r: Result<Value, _> = s.execute("track.setAutoDuck", p.clone());
        assert!(r.is_err(), "{p} should be refused: {r:?}");
    }
    assert_eq!(s.history.undo.len(), undo, "nothing changed");
    // a hostile project file: numbers out of range are clamped when used, never a panic
    let mut p = (*s.project).clone();
    let seq = s.state.active_sequence.unwrap();
    let q = p.sequence_mut(seq).unwrap();
    let a1 = q.audio_tracks[0].id;
    q.audio_tracks[1].auto_duck = Some(filmcraft_project::auto_duck::AutoDuck {
        under: vec![a1, filmcraft_project::TrackId(999_999)],
        amount_db: f64::NAN,
        attack_s: -1.0,
        release_s: f64::INFINITY,
        sensitivity: 1e300,
        clips: vec![filmcraft_project::auto_duck::DuckedClip { clip: ClipId(424_242), base_db: f64::NAN, signature: 1 }],
    });
    s.project = Arc::new(p);
    s.execute("timeline.setTrack", json!({"track": "A3", "muted": true})).unwrap();
    let it = s.active_sequence().unwrap().audio_tracks[1].items[0].clone();
    let lvl = it.effect("volume").unwrap().param("level").unwrap();
    assert!(lvl.keyframes.iter().all(|k| k.value.as_f64().is_some_and(f64::is_finite)));
}
