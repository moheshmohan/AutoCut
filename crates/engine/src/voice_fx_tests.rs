//! `audio.isolateVoice`, `audio.enhanceVoice`, `audio.enhanceFile`, `audio.voiceModelStatus`:
//! the effects land first in the chain, one undo step, loudness matched through the effects,
//! files processed with their loudness and noise floor reported, hostile parameters refused.

use std::sync::Arc;

use filmcraft_project::{ClipId, TrackItem};
use serde_json::json;

use super::*;

const SR: u32 = 48_000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Speech-like sound (a gliding voiced source under syllables, three to a word, a pause after
/// each word) in hiss and rumble.
fn noisy_speech(secs: f64) -> Vec<f32> {
    let n = (secs * f64::from(SR)) as usize;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut phase, mut rumble) = (0.0f64, 0.0f32);
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(SR);
            let f0 = 130.0 + 25.0 * (2.0 * std::f64::consts::PI * 0.7 * t).sin();
            phase += f0 / f64::from(SR);
            let mut v = 0.0;
            let mut k = 1;
            while k as f64 * f0 < 4000.0 {
                v += (2.0 * std::f64::consts::PI * phase * k as f64).sin() / k as f64;
                k += 1;
            }
            let word = t % 1.1 < 0.75;
            let syl = if word { (0.5 - 0.5 * (2.0 * std::f64::consts::PI * 4.0 * (t % 1.1)).cos()).powi(2) } else { 0.0 };
            rumble += 0.02 * (rng.next() - rumble);
            (v * syl * 0.15) as f32 + 0.01 * rng.next() + 0.4 * rumble
        })
        .collect()
}

fn session_with(x: &[f32]) -> (Session, ClipId) {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "voice", "audio": 2, "video": 1})).unwrap();
    let inter: Vec<f32> = x.iter().flat_map(|v| [*v, *v]).collect();
    let bytes: Arc<[u8]> = crate::previews::write_wav_f32(&inter, SR).into();
    let item = crate::commands::import_bytes(&mut s, "/voice.wav", bytes, None).unwrap();
    let r = s.execute("timeline.place", json!({"item": item.0, "audioTrack": "A1", "seconds": 0.0})).unwrap();
    let c = ClipId(r["clips"][0].as_u64().unwrap());
    s.execute("edit.deselectAll", json!({})).unwrap();
    (s, c)
}

fn item(s: &Session, c: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
}

fn standard(s: &Session, c: ClipId) -> Vec<String> {
    item(s, c).effects.iter().filter(|e| !e.def().is_some_and(|d| d.intrinsic)).map(|e| e.effect.clone()).collect()
}

#[test]
fn isolate_voice_goes_first_and_updates_in_place() {
    let (mut s, c) = session_with(&noisy_speech(2.0));
    s.execute("effects.apply", json!({"clips": [c.0], "effect": "parametric_eq"})).unwrap();
    let undo0 = s.execute("history.list", json!({})).unwrap()["undo"].as_array().unwrap().len();
    let r = s.execute("audio.isolateVoice", json!({"clips": [c.0], "amount": 55})).unwrap();
    assert_eq!(r["clips"], 1);
    assert!(r["model"].is_string() && r["neural"].is_boolean());
    assert_eq!(standard(&s, c), ["voice_isolation", "parametric_eq"]);
    let it = item(&s, c);
    assert_eq!(it.effects.iter().find(|e| e.effect == "voice_isolation").unwrap().f64_at("intensity", filmcraft_time::Tick::ZERO), 55.0);
    // again: the same effect, a new amount, no second copy
    s.execute("audio.isolateVoice", json!({"clips": [c.0], "amount": 90})).unwrap();
    assert_eq!(standard(&s, c), ["voice_isolation", "parametric_eq"]);
    assert_eq!(s.execute("history.list", json!({})).unwrap()["undo"].as_array().unwrap().len(), undo0 + 2);
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(standard(&s, c), ["parametric_eq"]);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, c) = session_with(&noisy_speech(1.0));
    for amount in [json!(-1), json!(100.5), json!("loud"), json!(1e300)] {
        assert!(s.execute("audio.isolateVoice", json!({"clips": [c.0], "amount": amount})).is_err(), "{amount}");
        assert!(s.execute("audio.enhanceVoice", json!({"clips": [c.0], "amount": amount})).is_err(), "{amount}");
    }
    for lufs in [json!(5), json!(-100), json!("loudest"), json!(true)] {
        assert!(s.execute("audio.enhanceVoice", json!({"clips": [c.0], "lufs": lufs})).is_err(), "{lufs}");
    }
    assert!(s.execute("audio.isolateVoice", json!({"clips": [987_654]})).is_err(), "no such clip");
    assert!(s.execute("audio.isolateVoice", json!({})).is_err(), "nothing selected");
    for p in [
        json!({}),
        json!({"path": "/nope/missing.wav", "out": "/tmp/x.wav"}),
        json!({"path": "/nope/missing.wav", "out": "/tmp/x.mp3"}),
        json!({"path": "/nope/missing.wav", "out": "/tmp/x.wav", "mode": "robot"}),
        json!({"path": "/nope/missing.wav", "out": "/tmp/x.m4a", "bitrate": 5}),
    ] {
        assert!(s.execute("audio.enhanceFile", p.clone()).is_err(), "{p}");
    }
    assert!(item(&s, c).effects.iter().all(|e| e.effect != "voice_isolation" && e.effect != "enhance_voice"));
}

#[test]
fn enhance_voice_matches_loudness_through_the_effects() {
    let (mut s, c) = session_with(&noisy_speech(4.0));
    let r = s.execute("audio.enhanceVoice", json!({"clips": [c.0], "amount": 80})).unwrap();
    assert_eq!(r["clips"], 1);
    assert_eq!(standard(&s, c), ["enhance_voice"]);
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let l = crate::essential_sound::clip_loudness(&item(&s, c), SR, &provider).unwrap();
    assert!((l - DEFAULT_LUFS).abs() < 0.6, "the clip measures {l:.2} LUFS, not −16: {r}");
    // "none" leaves the level alone; "mix" goes to the Dialogue target
    let g = item(&s, c).gain_db;
    s.execute("audio.enhanceVoice", json!({"clips": [c.0], "amount": 40, "lufs": "none"})).unwrap();
    assert_eq!(item(&s, c).gain_db, g);
    s.execute("audio.enhanceVoice", json!({"clips": [c.0], "lufs": "mix"})).unwrap();
    let target = s.prefs.audio.loudness_target(filmcraft_project::AudioType::Dialogue);
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let l = crate::essential_sound::clip_loudness(&item(&s, c), SR, &provider).unwrap();
    assert!((l - target).abs() < 0.6, "{l} vs {target}");
}

#[test]
fn enhanced_clips_are_more_intelligible() {
    let x = noisy_speech(5.0);
    let (mut s, c) = session_with(&x);
    s.execute("audio.enhanceVoice", json!({"clips": [c.0], "amount": 100, "lufs": "none"})).unwrap();
    let it = item(&s, c);
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let a1 = it.end().to_units_floor(i64::from(SR));
    let stream = filmcraft_render::MixStream::new();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < a1 {
        let n = (a1 - pos).min(i64::from(SR)) as usize;
        let [l, _] = filmcraft_render::audio::clip_signal(&it, pos, n, SR, &provider, &stream).unwrap();
        out.extend(l);
        pos += n as i64;
    }
    let skip = 2 * SR as usize;
    let (s0, s1) = (band_share(&x[skip..], SR, 1000.0, 4000.0), band_share(&out[skip..], SR, 1000.0, 4000.0));
    let (f0, f1) = (noise_floor_db(&x[skip..], SR), noise_floor_db(&out[skip..], SR));
    eprintln!("enhanced clip: 1–4 kHz share {s0:.3} → {s1:.3}, noise floor {f0:.1} → {f1:.1} dBFS");
    assert!(s1 > 1.5 * s0, "speech band {s0} → {s1}");
    assert!(f1 < f0 - 6.0, "noise floor {f0} → {f1}");
}

#[test]
fn a_file_is_enhanced_to_wav_and_m4a() {
    let dir = std::env::temp_dir().join(format!("filmcraft-voice-fx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let x = noisy_speech(4.0);
    let inter: Vec<f32> = x.iter().flat_map(|v| [*v, *v]).collect();
    let src = dir.join("noisy.wav");
    std::fs::write(&src, crate::previews::write_wav_f32(&inter, SR)).unwrap();
    let mut s = Session::default();
    for (out, mode) in [("enhanced.wav", "enhance"), ("isolated.m4a", "isolate")] {
        let out = dir.join(out);
        let r = s.execute("audio.enhanceFile", json!({"path": src.to_string_lossy(), "out": out.to_string_lossy(), "mode": mode})).unwrap();
        assert!(std::fs::metadata(&out).unwrap().len() > 1000, "{r}");
        assert!((r["after"]["lufs"].as_f64().unwrap() - DEFAULT_LUFS).abs() < 1.0, "{r}");
        assert!(r["after"]["truePeakDbtp"].as_f64().unwrap() <= -0.5, "{r}");
        assert!(r["after"]["noiseFloorDbfs"].as_f64().unwrap() < r["before"]["noiseFloorDbfs"].as_f64().unwrap(), "{r}");
        assert!((r["seconds"].as_f64().unwrap() - 4.0).abs() < 0.05);
        // the file reads back with the length it had
        let m = filmcraft_media::open_path(&out, &filmcraft_codecs::openers()).unwrap();
        let secs = m.info().duration.0 as f64 / filmcraft_time::TICKS_PER_SECOND as f64;
        assert!((secs - 4.0).abs() < 0.1, "{secs}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_status_names_the_model_and_its_licence() {
    let mut s = Session::default();
    let v = s.execute("audio.voiceModelStatus", json!({})).unwrap();
    assert!(v["license"].as_str().unwrap().contains("MIT"));
    assert!(v["files"][0]["sha256"].as_str().unwrap().len() == 64);
    assert!(v["model"].is_string() && v["neural"].is_boolean());
    if v["neural"] == false {
        assert!(v["hint"].as_str().unwrap().contains("audio.downloadVoiceModel"));
    }
}

/// Regression: in the mix a clip's chain starts on pre-roll silence, and the DSP fallback's
/// noise floor used to start from the frame straddling the clip's start (far too low) and creep
/// up at 0.9 dB a second, so the noise between words passed untouched (−33 → −32.5 dBFS). It now
/// tracks the floor by minimum statistics and ignores silence: the mix cleans like a file does.
#[test]
fn the_fallback_cleans_from_a_clips_start_in_the_mix() {
    use filmcraft_audio_dsp::AudioEffect;
    let x = noisy_speech(5.0);
    let skip = 2 * SR as usize;
    let floor = |y: &[f32]| noise_floor_db(&y[skip..], SR);
    let mut iso = filmcraft_audio_dsp::effects::VoiceIsolation::with_model(SR as f32, 1, None);
    iso.set_param("intensity", 100.0);
    iso.reset();
    let mut ch = vec![x.clone()];
    filmcraft_audio_dsp::effects::voice::process_offline(&mut iso, &mut ch);
    let (mut s, c) = session_with(&x);
    s.execute("audio.isolateVoice", json!({"clips": [c.0], "amount": 100})).unwrap();
    let it = item(&s, c);
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let stream = filmcraft_render::MixStream::new();
    let [l, _] = filmcraft_render::audio::clip_signal(&it, 0, x.len(), SR, &provider, &stream).unwrap();
    let (f_in, f_file, f_mix) = (floor(&x), floor(&ch[0]), floor(&l));
    eprintln!("noise floor {f_in:.1} dBFS: {f_file:.1} offline, {f_mix:.1} in the mix");
    if s.execute("audio.voiceModelStatus", json!({})).unwrap()["neural"] == false {
        assert!(f_file < f_in - 12.0 && f_mix < f_in - 12.0, "{f_in} → {f_file} offline, {f_mix} in the mix");
    }
}
