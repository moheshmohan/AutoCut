//! `audio.analyzeMix`: per-stem loudness and bands, speech against the music while it plays, the
//! limiter's gain reduction and clipping in the narration, on synthetic stems (a buzzy "voice" in
//! syllables with pauses, a bass-heavy "music" bed); hostile parameters are refused.

use std::sync::Arc;

use filmcraft_project::ClipId;
use serde_json::{Value, json};

use crate::Session;

const SR: u32 = 48_000;

/// Speech-like syllables (a voice of harmonics up to 4 kHz under a 4.5 Hz envelope, noise for
/// the consonants), talking 2.2 s of every 3 s.
fn voice(secs: f64, amp: f32) -> Vec<f32> {
    let mut seed = 0x1234_5678u64;
    (0..(secs * SR as f64) as usize)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let t = i as f64 / SR as f64;
            if (t % 3.0) > 2.2 {
                return 0.0;
            }
            let f0 = 130.0 + 20.0 * (2.0 * std::f64::consts::PI * 0.6 * t).sin();
            let v: f64 = (1..30).filter(|k| *k as f64 * f0 < 4000.0).map(|k| (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin() / k as f64).sum();
            let noise = ((seed >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) * 0.1;
            let syl = 0.6 - 0.4 * (2.0 * std::f64::consts::PI * 4.5 * t).cos();
            ((v * 0.4 + noise) * syl) as f32 * amp
        })
        .collect()
}

/// A bass-heavy bed: 55 and 110 Hz under a little 1 kHz.
fn bed(secs: f64, amp: f32) -> Vec<f32> {
    (0..(secs * SR as f64) as usize)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let w = |f: f64| (2.0 * std::f64::consts::PI * f * t).sin();
            ((0.6 * w(55.0) + 0.3 * w(110.0) + 0.05 * w(1000.0)) * amp as f64) as f32
        })
        .collect()
}

/// A sequence with each `(signal, track, audio type)` placed at 0 on its track; the clips.
fn session(stems: &[(Vec<f32>, &str, &str)]) -> (Session, Vec<ClipId>) {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "mix", "audio": 3, "video": 1})).unwrap();
    let mut clips = Vec::new();
    for (k, (x, track, kind)) in stems.iter().enumerate() {
        let inter: Vec<f32> = x.iter().flat_map(|v| [*v, *v]).collect();
        let bytes: Arc<[u8]> = crate::previews::write_wav_f32(&inter, SR).into();
        let item = crate::commands::import_bytes(&mut s, &format!("/stem{k}.wav"), bytes, None).unwrap();
        let r = s.execute("timeline.place", json!({"item": item.0, "audioTrack": track, "seconds": 0.0})).unwrap();
        let c = r["clips"][0].as_u64().unwrap();
        s.execute("essentialSound.setType", json!({"clips": [c], "type": kind})).unwrap();
        clips.push(ClipId(c));
    }
    (s, clips)
}

fn stem<'a>(r: &'a Value, track: &str) -> &'a Value {
    r["stems"].as_array().unwrap().iter().find(|s| s["track"] == track).unwrap_or_else(|| panic!("no stem {track}: {r}"))
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[test]
fn stems_bands_and_speech_over_the_music() {
    let (mut s, _) = session(&[(voice(12.0, 0.3), "A2", "dialogue"), (bed(12.0, 0.1), "A3", "music")]);
    let r = s.execute("audio.analyzeMix", json!({})).unwrap();
    println!("{r:#}");
    // roles from the clips' audio types; the bed's energy is below 120 Hz, the voice's is not
    assert_eq!(stem(&r, "A2")["role"], "speech");
    assert_eq!(stem(&r, "A3")["role"], "music");
    assert!(f(&stem(&r, "A3")["bands"]["sub"]) > 0.95, "{r}");
    let v = &stem(&r, "A2")["bands"];
    assert!(f(&v["sub"]) < 0.25 && f(&v["lowMid"]) > 0.3 && f(&v["presence"]) > 0.1, "{v}");
    assert_eq!(r["speech"]["tracks"], json!(["A2"]));
    assert_eq!(r["speech"]["music"], json!(["A3"]));
    // speech is active about 2.2 s of every 3 s (the momentary window blurs the edges)
    let active = f(&r["speech"]["toMusic"]["activeSeconds"]);
    assert!((7.0..=10.5).contains(&active), "{active}");
    let base = f(&r["speech"]["toMusic"]["medianLu"]);
    // the bed alone is the rest: the two ratios agree
    assert!((f(&r["speech"]["toRest"]["medianLu"]) - base).abs() < 0.2, "{r}");
    // 6 dB more bed: 6 LU less speech over it
    let bed_clip = s.active_sequence().unwrap().audio_tracks[2].items[0].id;
    s.execute("clip.audioGain", json!({"clips": [bed_clip.0], "mode": "adjust", "db": 6.0})).unwrap();
    let r2 = s.execute("audio.analyzeMix", json!({})).unwrap();
    let louder = f(&r2["speech"]["toMusic"]["medianLu"]);
    assert!((base - louder - 6.0).abs() < 0.3, "{base} → {louder}");
    // no limiter on the Mix: none reported; nothing clipped
    assert_eq!(r["limiter"]["present"], false);
    assert_eq!(r["clippedNarrationClips"], 0);
    assert_eq!(r["narrationClips"].as_array().unwrap().len(), 1);
    // the mix as it plays holds both
    assert!(f(&r["mix"]["bands"]["sub"]) > 0.3 && f(&r["mix"]["integratedLufs"]) < 0.0);
}

#[test]
fn the_limiter_gain_reduction_and_clipped_narration() {
    // a loud bed into a limiter at −12 dBFS: about 9 dB of reduction on its peaks
    let mut clipped = voice(6.0, 1.0);
    for v in clipped.iter_mut() {
        *v = v.clamp(-0.3, 0.3);
    }
    let (mut s, _) = session(&[(clipped, "A2", "dialogue"), (bed(6.0, 0.9), "A3", "music")]);
    s.execute("mixer.addInsert", json!({"strip": "Mix", "effect": "hard_limiter", "postFader": true})).unwrap();
    s.execute("mixer.setInsert", json!({"strip": "Mix", "slot": 0, "params": {"max": -12.0}})).unwrap();
    let r = s.execute("audio.analyzeMix", json!({"from": 1.0, "to": 5.0})).unwrap();
    println!("{r:#}");
    assert_eq!(r["limiter"]["present"], true);
    let gr = f(&r["limiter"]["maxGainReductionDb"]);
    assert!(gr > 6.0, "{r}");
    assert!(f(&r["limiter"]["secondsOver2Db"]) > 1.0);
    assert!(f(&r["preLimiter"]["truePeakDbtp"]) > f(&r["mix"]["truePeakDbtp"]) + 6.0);
    assert!(f(&r["mix"]["truePeakDbtp"]) <= -11.0, "the limiter holds its ceiling: {r}");
    // the voice's flat tops are found
    assert_eq!(r["clippedNarrationClips"], 1);
    let c = &r["narrationClips"][0];
    assert!(c["flatRuns"].as_u64().unwrap() > 50 && c["clipped"] == true, "{c}");
    assert_eq!((f(&r["from"]), f(&r["to"])), (1.0, 5.0));
    // tracks named instead of read from the clips' types
    let r = s.execute("audio.analyzeMix", json!({"speech": "A3", "music": [2]})).unwrap();
    assert_eq!((&r["speech"]["tracks"], &r["speech"]["music"]), (&json!(["A3"]), &json!(["A2"])));
    // a limiter on a track: its own gain reduction, on its stem
    assert!(stem(&r, "A2").get("limiter").is_none());
    s.execute("mixer.addInsert", json!({"strip": "A2", "effect": "hard_limiter"})).unwrap();
    s.execute("mixer.setInsert", json!({"strip": "A2", "slot": 0, "params": {"max": -20.0}})).unwrap();
    let r = s.execute("audio.analyzeMix", json!({"from": 1.0, "to": 5.0})).unwrap();
    let tl = &stem(&r, "A2")["limiter"];
    assert!(f(&tl["maxGainReductionDb"]) > 5.0 && f(&stem(&r, "A2")["truePeakDbtp"]) <= -19.0, "{r}");
}

#[test]
fn no_speech_and_hostile_parameters() {
    let (mut s, _) = session(&[(bed(3.0, 0.2), "A1", "music")]);
    let steps = s.history.undo.len();
    let r = s.execute("audio.analyzeMix", json!({})).unwrap();
    assert!(r["speech"].is_null());
    assert_eq!(r["narrationClips"], json!([]));
    for p in [
        json!({"from": 2.0, "to": 1.0}),
        json!({"from": -1.0}),
        json!({"to": "soon"}),
        json!({"from": 1e300, "to": 1e301}),
        json!({"speech": "A9"}),
        json!({"speech": ["A0"]}),
        json!({"music": [{"x": 1}]}),
        json!({"sequence": 999_999}),
    ] {
        assert!(s.execute("audio.analyzeMix", p.clone()).is_err(), "{p}");
    }
    // nothing changed: no undo step
    assert_eq!(s.history.undo.len(), steps);
}
