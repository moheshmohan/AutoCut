//! `audio.normalizeLoudness`: the mix lands within ±0.5 LU of each destination's target with
//! true peaks under the ceiling (measured independently through the mixer), one undo step,
//! idempotent, and hostile parameters are refused.

use std::sync::Arc;

use filmcraft_audio_dsp::LoudnessMeter;
use filmcraft_render::MixStream;
use serde_json::{Value, json};

use crate::Session;

const SR: u32 = 48_000;

/// A sequence with `signal` (mono, played on both sides) on A1.
fn session(signal: &[f32]) -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "loud", "audio": 1, "video": 1})).unwrap();
    let inter: Vec<f32> = signal.iter().flat_map(|v| [*v, *v]).collect();
    let bytes: Arc<[u8]> = crate::previews::write_wav_f32(&inter, SR).into();
    let item = crate::commands::import_bytes(&mut s, "/mix.wav", bytes, None).unwrap();
    s.execute("timeline.place", json!({"item": item.0, "audioTrack": "A1", "seconds": 0.0})).unwrap();
    s
}

/// Generated drums, bass and pads at `gain`.
fn music(gain: f32) -> Vec<f32> {
    let (m, _) = filmcraft_audio_dsp::remix::test_music(SR, 120.0, 8, 4);
    m[0].iter().map(|v| v * gain).collect()
}

/// Speech-like syllables (a buzzy voice under a 4.5 Hz envelope) with pauses.
fn voice(gain: f32) -> Vec<f32> {
    (0..SR as usize * 12)
        .map(|i| {
            let t = i as f64 / SR as f64;
            if (t % 3.0) > 2.2 {
                return 0.0;
            }
            let f0 = 130.0 + 20.0 * (2.0 * std::f64::consts::PI * 0.6 * t).sin();
            let v: f64 = (1..12).map(|k| (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin() / k as f64).sum();
            let syl = 0.6 - 0.4 * (2.0 * std::f64::consts::PI * 4.5 * t).cos();
            (v * syl * 0.4) as f32 * gain
        })
        .collect()
}

/// The mix's integrated loudness and true peak, measured here (not by the command).
fn measured(s: &Session) -> (f64, f64) {
    let q = s.active_sequence().unwrap();
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let total = q.duration().to_units_floor(SR as i64);
    let mut m = LoudnessMeter::new(SR as f64, 2);
    let stream = filmcraft_render::MixStream::new();
    let mut pos = 0;
    while pos < total {
        let n = (total - pos).min(4800) as usize;
        let b = filmcraft_render::audio::mix_sequence(&s.project, q, pos, n, &provider, &stream);
        m.process(&[&b.channels[0], &b.channels[1]]);
        pos += n as i64;
    }
    (m.integrated(), m.true_peak_dbtp())
}

#[test]
fn destinations_reach_their_targets_under_the_ceiling() {
    for (target, want, signal) in
        [(json!("social"), -14.0, music(0.1)), (json!("voice"), -16.0, voice(0.05)), (json!("broadcast"), -23.0, music(0.6)), (json!(-9.0), -9.0, music(0.3))]
    {
        let mut s = session(&signal);
        let (before, _) = measured(&s);
        let r = s.execute("audio.normalizeLoudness", json!({"target": target})).unwrap();
        let (after, tp) = measured(&s);
        println!("{target}: {before:.2} → {after:.2} LUFS, true peak {tp:.2} dBTP, fader {:.2} dB", r["masterVolumeDb"].as_f64().unwrap());
        assert!((after - want).abs() <= 0.5, "{target}: {after:.2} LUFS, want {want} ± 0.5 ({r})");
        assert!(tp <= -1.0, "{target}: true peak {tp:.2} dBTP over the −1 dBTP ceiling");
        assert!((r["loudnessLufs"].as_f64().unwrap() - after).abs() < 0.05, "the command reports what plays: {r}");
        assert_eq!(r["targetLufs"].as_f64(), Some(want));
        // the Mix fader and one post-fader Hard Limiter at the ceiling
        let q = s.active_sequence().unwrap();
        let lim: Vec<_> = q.master_effects.iter().filter(|e| e.effect == "hard_limiter").collect();
        assert_eq!(lim.len(), 1);
        assert!(lim[0].post_fader && lim[0].enabled);
        assert!(lim[0].param("max").unwrap().value.as_f64().is_some_and(|m| m <= -1.0), "the limiter holds the ceiling");
    }
}

/// Chunk `k` (4800 samples) of the mix, left then right, rendered in `stream`.
fn chunk(s: &Session, k: usize, stream: &MixStream) -> Vec<f32> {
    let q = s.active_sequence().unwrap();
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    filmcraft_render::audio::mix_sequence(&s.project, q, k as i64 * 4800, 4800, &provider, stream).channels.concat()
}

/// Two sessions with the same structure (one clip, the Mix's limiter) but different signals,
/// measured chunk by chunk in turns (A B, then B A, …): each hears exactly its own mix, as alone.
#[test]
fn sessions_measured_in_turns_hear_their_own_mix() {
    let mut a = session(&music(0.5));
    let mut b = session(&voice(0.3));
    for s in [&mut a, &mut b] {
        s.execute("audio.normalizeLoudness", json!({"target": "social"})).unwrap();
    }
    const N: usize = 12;
    let alone = |s: &Session| {
        let stream = MixStream::new();
        (0..N).flat_map(|k| chunk(s, k, &stream)).collect::<Vec<f32>>()
    };
    let (alone_a, alone_b) = (alone(&a), alone(&b));
    // each measurement in its own stream, as every caller of the mixer does
    let (sa, sb) = (MixStream::new(), MixStream::new());
    let (mut turn_a, mut turn_b) = (Vec::new(), Vec::new());
    for k in 0..N {
        if k % 2 == 0 {
            turn_a.extend(chunk(&a, k, &sa));
            turn_b.extend(chunk(&b, k, &sb));
        } else {
            turn_b.extend(chunk(&b, k, &sb));
            turn_a.extend(chunk(&a, k, &sa));
        }
    }
    for (what, got, want) in [("music", &turn_a, &alone_a), ("voice", &turn_b, &alone_b)] {
        let diff = got.iter().zip(want.iter()).position(|(x, y)| x.to_bits() != y.to_bits());
        assert_eq!(diff, None, "{what}: measured in turns with another session, the mix differs from the mix alone");
    }
}

/// Sessions with the same structure (one clip, the Mix's limiter) but different signals,
/// normalised and measured at the same time on several threads: each command reports exactly
/// what its own mix plays. The mixer's stateful DSP (the limiter's envelope) continues only the
/// render it came from, never another session's.
#[test]
fn parallel_sessions_report_what_they_play() {
    let jobs = [(json!("social"), music(0.1)), (json!("social"), music(0.5)), (json!("voice"), voice(0.05)), (json!(-9.0), music(0.3))];
    std::thread::scope(|sc| {
        for (target, signal) in &jobs {
            sc.spawn(move || {
                for _ in 0..2 {
                    let mut s = session(signal);
                    let r = s.execute("audio.normalizeLoudness", json!({"target": target})).unwrap();
                    let (after, _) = measured(&s);
                    let reported = r["loudnessLufs"].as_f64().unwrap();
                    assert!((reported - after).abs() < 1e-3, "{target}: the command reports {reported:.4} LUFS, the mix plays {after:.4}: {r}");
                }
            });
        }
    });
}

#[test]
fn a_custom_ceiling_one_undo_step_and_idempotent() {
    let mut s = session(&music(0.4));
    let fader0 = s.active_sequence().unwrap().master_volume_db;
    let r = s.execute("audio.normalizeLoudness", json!({"target": "social", "truePeak": -2.0})).unwrap();
    let (lufs, tp) = measured(&s);
    assert!((lufs + 14.0).abs() <= 0.5 && tp <= -2.0, "{lufs:.2} LUFS, {tp:.2} dBTP: {r}");
    // again: nothing moves (much) and no second limiter
    let r2 = s.execute("audio.normalizeLoudness", json!({"target": "social", "truePeak": -2.0})).unwrap();
    assert!(r2["gainDb"].as_f64().unwrap().abs() < 0.3, "{r2}");
    assert_eq!(s.active_sequence().unwrap().master_effects.iter().filter(|e| e.effect == "hard_limiter").count(), 1);
    // each run is one undo step
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    assert_eq!(q.master_volume_db, fader0);
    assert!(q.master_effects.is_empty());
}

#[test]
fn hostile_parameters_and_silence_are_refused() {
    let mut s = session(&music(0.2));
    let undo = s.history.undo.len();
    for p in [
        json!({"target": "loud"}),
        json!({"target": 5.0}),
        json!({"target": -100.0}),
        json!({"target": [1]}),
        json!({"target": "social", "truePeak": 3.0}),
        json!({"target": "social", "truePeak": "high"}),
        json!({"target": "social", "truePeak": -1e300}),
        json!({"sequence": 999_999}),
    ] {
        let r: Result<Value, _> = s.execute("audio.normalizeLoudness", p.clone());
        assert!(r.is_err(), "{p} should be refused: {r:?}");
    }
    assert_eq!(s.history.undo.len(), undo, "nothing changed");
    let mut quiet = session(&vec![0.0; SR as usize * 3]);
    let e = quiet.execute("audio.normalizeLoudness", json!({})).unwrap_err().to_string();
    assert!(e.contains("silent"), "{e}");
    // the Mix full of inserts: no room for the limiter, said so
    let mut full = session(&music(0.2));
    let seq = full.state.active_sequence.unwrap();
    let mut p = (*full.project).clone();
    let q = p.sequence_mut(seq).unwrap();
    for _ in 0..filmcraft_project::mixer::MAX_INSERTS {
        q.master_effects.push(filmcraft_project::find_effect("amplify").unwrap().instance());
    }
    full.project = Arc::new(p);
    let e = full.execute("audio.normalizeLoudness", json!({})).unwrap_err().to_string();
    assert!(e.contains("insert"), "{e}");
}
