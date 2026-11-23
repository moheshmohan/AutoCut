//! Tests of [`crate::remix`]: Clip ▸ Remix on generated rhythmic music (120 BPM, 16 bars, 32 s).

use std::sync::Arc;

use filmcraft_audio_dsp::remix as dsp;
use filmcraft_project::{ClipId, TrackItem};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;

const SR: u32 = 48_000;

/// Generated music (stereo planar) and its analysis.
fn music() -> (Vec<Vec<f32>>, dsp::Analysis) {
    let (m, _) = dsp::test_music(SR, 120.0, 16, 3);
    let refs: Vec<&[f32]> = m.iter().map(Vec::as_slice).collect();
    let a = dsp::analyze(&refs, SR).unwrap();
    (m, a)
}

/// A session with `music` on A1 at 0 s (and optionally a second clip at `next` s), the music
/// clip selected.
fn session(m: &[Vec<f32>], next: Option<f64>) -> (Session, ClipId) {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "remix", "audio": 2, "video": 1})).unwrap();
    let inter: Vec<f32> = (0..m[0].len()).flat_map(|i| [m[0][i], m[1][i]]).collect();
    let bytes: Arc<[u8]> = crate::previews::write_wav_f32(&inter, SR).into();
    let item = crate::commands::import_bytes(&mut s, "/music.wav", bytes, None).unwrap();
    let r = s.execute("timeline.place", json!({"item": item.0, "audioTrack": "A1", "seconds": 0.0})).unwrap();
    let c = ClipId(r["clips"][0].as_u64().unwrap());
    if let Some(at) = next {
        s.execute("timeline.place", json!({"item": item.0, "audioTrack": "A1", "seconds": at})).unwrap();
    }
    s.execute("timeline.select", json!({"clips": [c.0]})).unwrap();
    (s, c)
}

fn item(s: &Session, c: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
}

fn beat() -> Tick {
    Tick::from_seconds_f64(0.5)
}

/// Every cut of the report lies on a detected beat of the music.
fn assert_cuts_on_beats(r: &Value, a: &dsp::Analysis) {
    for cut in r["cuts"].as_array().unwrap() {
        for k in ["out", "in"] {
            let t = Tick(cut[k].as_i64().unwrap());
            let smp = t.to_units_floor(SR as i64);
            assert!(a.beats.contains(&smp), "cut {k} at sample {smp} is not a detected beat");
        }
    }
}

#[test]
fn menu_commands_and_disabled_reasons() {
    let (m, _) = music();
    let (mut s, c) = session(&m, None);
    let pos = |id: &str| crate::commands::command_specs().iter().position(|c| c.id == id).unwrap();
    assert!(pos("clip.remix.enable") < pos("clip.remix.properties") && pos("clip.remix.properties") < pos("clip.remix.revert"));
    let spec = crate::commands::find("clip.remix.enable").unwrap();
    assert_eq!(spec.menu, &["Clip", "Remix"]);
    // not remixed yet: properties / revert are disabled
    assert!(s.execute("clip.remix.revert", json!({})).is_err());
    assert!(s.execute("clip.remix.properties", json!({})).is_err());
    // nothing selected: enable is disabled with a reason
    s.execute("edit.deselectAll", json!({})).unwrap();
    let why = (crate::commands::find("clip.remix.enable").unwrap().enabled)(&s).unwrap_err();
    assert!(why.contains("audio clip"), "{why}");
    s.execute("timeline.select", json!({"clips": [c.0]})).unwrap();
    assert!((spec.enabled)(&s).is_ok());
    // agents must say what duration they want
    assert!(s.execute("clip.remix", json!({"clip": c.0})).is_err());
}

#[test]
fn remix_shorter_and_longer_hits_target_at_beats_with_undo() {
    let (m, a) = music();
    let (mut s, c) = session(&m, None);
    let before = item(&s, c);
    let r = s.execute("clip.remix.enable", json!({})).unwrap();
    assert_eq!(r["remixed"], true);
    let enabled = item(&s, c);
    assert!((before.duration - enabled.duration).abs() < Tick::from_units(1, SR as i64), "enable keeps the duration");
    let n0 = s.history.undo.len();
    for secs in [12.0, 20.0, 26.5, 45.0, 70.0] {
        let r = s.execute("clip.remix", json!({"seconds": secs})).unwrap();
        let d = item(&s, c).duration;
        let err = (d - Tick::from_seconds_f64(secs)).abs();
        assert!(err <= beat(), "{secs} s: off by {:.3} s", err.seconds());
        assert_eq!(Tick(r["duration"].as_i64().unwrap()), d);
        assert_cuts_on_beats(&r, &a);
        assert!(!r["cuts"].as_array().unwrap().is_empty());
        eprintln!("remix to {secs} s: {:.4} s, {} cuts, {:.1} BPM", d.seconds(), r["cuts"].as_array().unwrap().len(), r["bpm"].as_f64().unwrap());
    }
    assert_eq!(s.history.undo.len(), n0 + 5, "one undo step per remix");
    // properties: read and change the sliders (keeps the target)
    let p = s.execute("clip.remix.properties", json!({})).unwrap();
    assert_eq!(p["targetSeconds"].as_f64().unwrap(), 70.0);
    let few = s.execute("clip.remix.properties", json!({"segments": 0.0})).unwrap();
    let many = s.execute("clip.remix.properties", json!({"segments": 100.0})).unwrap();
    assert!(few["pieces"].as_array().unwrap().len() < many["pieces"].as_array().unwrap().len());
    // determinism
    let again = s.execute("clip.remix.properties", json!({"segments": 100.0})).unwrap();
    assert_eq!(again["pieces"], many["pieces"]);
    // undo / redo
    let cur = item(&s, c);
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(item(&s, c), cur);
    // revert restores the clip exactly
    s.execute("clip.remix.revert", json!({})).unwrap();
    assert_eq!(item(&s, c), before);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(item(&s, c), cur);
}

#[test]
fn mixed_audio_is_the_source_pieces() {
    let (m, a) = music();
    let (mut s, c) = session(&m, None);
    let r = s.execute("clip.remix", json!({"seconds": 18.0})).unwrap();
    assert_cuts_on_beats(&r, &a);
    let it = item(&s, c);
    let rm = filmcraft_render::remix::Remix::of(&it).unwrap();
    let plan = rm.plan_at(SR);
    let total = it.duration.to_units_floor(SR as i64) as usize;
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let q = s.active_sequence().unwrap();
    let mix = filmcraft_render::audio::mix_sequence(&s.project, q, 0, total, &provider, &filmcraft_render::MixStream::new());
    let mut d = 0usize;
    let mut checked = 0;
    let mut worst = 0f32;
    for p in &plan.pieces {
        for k in (plan.xfade as usize..p.len as usize - plan.xfade as usize).step_by(31) {
            let want = m[0][p.src as usize + k];
            worst = worst.max((mix.channels[0][d + k] - want).abs());
            checked += 1;
        }
        d += p.len as usize;
    }
    assert!(worst < 1e-5, "mix differs from the pieces by {worst}");
    assert!(checked > 10_000);
}

#[test]
fn media_beats_gives_the_tempo_and_beat_times_of_an_item() {
    let (m, truth) = {
        let (m, beats) = dsp::test_music(SR, 120.0, 16, 3);
        (m, beats.iter().map(|b| *b as f64 / SR as f64).collect::<Vec<f64>>())
    };
    let (mut s, c) = session(&m, None);
    let item = item(&s, c).item;
    let r = s.execute("media.beats", json!({"item": item.0})).unwrap();
    assert!((r["bpm"].as_f64().unwrap() - 120.0).abs() < 1.0, "{r}");
    assert!((r["period"].as_f64().unwrap() - 0.5).abs() < 0.01, "{r}");
    let beats: Vec<f64> = r["beats"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
    assert!(beats.len() >= 56, "most of the 64 beats: {}", beats.len());
    // every detected beat is a real one (the attack, within 25 ms), in order
    for b in &beats {
        assert!(truth.iter().any(|t| (t - b).abs() < 0.025), "beat at {b} is not one of the music's");
    }
    assert!(beats.windows(2).all(|w| w[0] < w[1]));
    assert!((r["seconds"].as_f64().unwrap() - 32.0).abs() < 0.05, "{r}");
    // a range: beats in seconds of the media, inside it
    let r = s.execute("media.beats", json!({"item": item.0, "sourceIn": Tick::from_seconds_f64(8.0).0, "duration": Tick::from_seconds_f64(16.0).0})).unwrap();
    let ranged: Vec<f64> = r["beats"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
    assert!(!ranged.is_empty() && ranged.iter().all(|b| (8.0..24.0).contains(b)), "{ranged:?}");
    assert!(ranged.iter().all(|b| truth.iter().any(|t| (t - b).abs() < 0.025)), "{ranged:?}");
    // hostile parameters: errors, never a panic
    assert!(s.execute("media.beats", json!({})).is_err());
    assert!(s.execute("media.beats", json!({"item": 999_999})).is_err());
    let seq = s.state.active_sequence.unwrap();
    assert!(s.execute("media.beats", json!({"item": seq.0})).is_err(), "a sequence is not audio media");
    assert!(s.execute("media.beats", json!({"item": item.0, "sourceIn": i64::MAX})).is_err(), "nothing past the end");
    assert!(s.execute("media.beats", json!({"item": item.0, "duration": 0})).is_err());
    assert!(s.execute("media.beats", json!({"item": item.0, "sourceIn": -5, "duration": i64::MAX})).is_ok());
    assert!(s.execute("media.beats", json!({"item": item.0, "sourceIn": Tick::from_seconds_f64(31.0).0})).is_err(), "a second has too few beats");
}

/// `media.pauses` finds where a "narration" pauses between its "sentences" (bursts of tone with
/// silences between them), in seconds of the media, inside a range; hostile parameters are errors.
#[test]
fn media_pauses_finds_the_silences_between_sentences() {
    // 0.3 s quiet, 2 s of sound, 0.6 s pause, 1.5 s of sound with a 0.1 s gap in it, 0.5 s pause,
    // 1 s of sound, quiet to 6 s
    let sound = [(0.3, 2.3), (2.9, 3.6), (3.7, 4.4), (4.9, 5.9)];
    let n = (6.0 * SR as f64) as usize;
    let tone: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            if sound.iter().any(|(a, b)| t >= *a && t < *b) { (0.3 * (t * 2.0 * std::f64::consts::PI * 220.0).sin()) as f32 } else { 0.0 }
        })
        .collect();
    let (mut s, c) = session(&[tone.clone(), tone], None);
    let item = item(&s, c).item;
    let r = s.execute("media.pauses", json!({"item": item.0})).unwrap();
    let span = |v: &Value| (v["start"].as_f64().unwrap(), v["end"].as_f64().unwrap());
    let pauses: Vec<(f64, f64)> = r["pauses"].as_array().unwrap().iter().map(span).collect();
    assert_eq!(pauses.len(), 2, "the 0.1 s gap is not a pause: {r}");
    for ((a, b), (wa, wb)) in pauses.iter().zip([(2.3, 2.9), (4.4, 4.9)]) {
        assert!((a - wa).abs() < 0.06 && (b - wb).abs() < 0.06, "{a}–{b} for {wa}–{wb}");
    }
    let sound_spans = r["sound"].as_array().unwrap();
    assert_eq!(sound_spans.len(), 3);
    assert!((span(&sound_spans[0]).0 - 0.3).abs() < 0.06, "{r}");
    // a range: times stay in the media's seconds
    let r = s.execute("media.pauses", json!({"item": item.0, "sourceIn": Tick::from_seconds_f64(2.0).0, "duration": Tick::from_seconds_f64(3.0).0})).unwrap();
    let pauses: Vec<(f64, f64)> = r["pauses"].as_array().unwrap().iter().map(span).collect();
    assert_eq!(pauses.len(), 2, "{r}");
    assert!((pauses[0].0 - 2.3).abs() < 0.06, "{pauses:?}");
    // longer pauses only
    let r = s.execute("media.pauses", json!({"item": item.0, "minSeconds": 0.5})).unwrap();
    assert_eq!(r["pauses"].as_array().map(Vec::len), Some(1), "{r}");
    // hostile parameters: errors, never a panic
    for bad in [
        json!({}),
        json!({"item": 999_999}),
        json!({"item": item.0, "sourceIn": i64::MAX}),
        json!({"item": item.0, "duration": 0}),
        json!({"item": item.0, "minSeconds": f64::MAX}),
        json!({"item": item.0, "minSeconds": "long"}),
        json!({"item": item.0, "belowDb": -5}),
    ] {
        assert!(s.execute("media.pauses", bad.clone()).is_err(), "{bad}");
    }
    assert!(s.execute("media.pauses", json!({"item": item.0, "sourceIn": -5, "duration": i64::MAX})).is_ok());
    let seq = s.state.active_sequence.unwrap();
    assert!(s.execute("media.pauses", json!({"item": seq.0})).is_err(), "a sequence is not audio media");
}

/// `media.beats` on imported music whose tempo drifts (112 → 128 BPM over 30 s): every beat is
/// found on its attack, within 10 ms, and the reported tempo is the mean one.
#[test]
fn media_beats_follows_tempo_drift() {
    let (m, truth) = filmcraft_audio_dsp::beats::drift_music(SR, 112.0, 128.0, 30.0, 9);
    let truth: Vec<f64> = truth.iter().map(|b| *b as f64 / SR as f64).collect();
    let (mut s, c) = session(&m, None);
    let item = item(&s, c).item;
    let r = s.execute("media.beats", json!({"item": item.0})).unwrap();
    assert!((r["bpm"].as_f64().unwrap() - 120.0).abs() < 3.0, "{r}");
    let beats: Vec<f64> = r["beats"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
    let hit = truth.iter().filter(|t| beats.iter().any(|b| (*t - b).abs() < 0.010)).count();
    assert!(hit * 100 >= truth.len() * 95, "{hit} of {} beats found within 10 ms", truth.len());
    assert!(beats.iter().all(|b| truth.iter().any(|t| (t - b).abs() < 0.010)), "no beat between the beats");
}

#[test]
fn overlaps_are_refused_and_noise_cannot_be_remixed() {
    let (m, _) = music();
    let (mut s, c) = session(&m, Some(40.0));
    let err = s.execute("clip.remix", json!({"seconds": 50.0})).unwrap_err().to_string();
    assert!(err.contains("room"), "{err}");
    assert!(s.execute("clip.remix", json!({"seconds": 39.0})).is_ok());
    let _ = c;
    // a steady tone has no beat
    let tone: Vec<f32> = (0..SR as usize * 20).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let (mut s, _) = session(&[tone.clone(), tone], None);
    let err = s.execute("clip.remix.enable", json!({})).unwrap_err().to_string();
    assert!(err.contains("beat") || err.contains("short"), "{err}");
}

/// `media.beats` reports the structure of imported music with known answers: downbeats on the
/// bar starts, sections at the parts (a quiet intro, a verse, a build, a drop, a breakdown and
/// a second drop) with the drops the loudest, the two drops, the music's end and its cadence.
#[test]
fn media_beats_reports_downbeats_sections_drops_and_the_ending() {
    use filmcraft_audio_dsp::structure::{FixtureEnd, Part, fixture};
    let part = |bars, db, drums, build| Part { bars, db, drums, build };
    let parts = [
        part(4, -16.0, false, false),
        part(8, -7.0, true, false),
        part(4, -9.0, false, true),
        part(8, 0.0, true, false),
        part(4, -14.0, false, false),
        part(8, 0.0, true, false),
    ];
    let f = fixture(SR, 120.0, &parts, FixtureEnd::Cadence, 3);
    let secs = |x: i64| x as f64 / SR as f64;
    let (mut s, c) = session(&f.audio, None);
    let item = item(&s, c).item;
    let r = s.execute("media.beats", json!({"item": item.0})).unwrap();
    let nums = |k: &str| -> Vec<f64> { r[k].as_array().unwrap().iter().filter_map(Value::as_f64).collect() };
    assert_eq!(r["meter"], 4, "{r}");
    let downbeats = nums("downbeats");
    let bars: Vec<f64> = f.downbeats.iter().map(|d| secs(*d)).collect();
    assert!(downbeats.len() >= bars.len() * 8 / 10, "{} of {} bars", downbeats.len(), bars.len());
    assert!(downbeats.iter().all(|d| bars.iter().any(|b| (b - d).abs() < 0.03)), "downbeats off the bars: {downbeats:?}");
    let sections = r["sections"].as_array().unwrap();
    let starts: Vec<f64> = sections.iter().map(|x| x["start"].as_f64().unwrap()).collect();
    let parts_at: Vec<f64> = f.parts.iter().map(|p| secs(*p)).collect();
    assert_eq!(starts.len(), parts_at.len(), "{r}");
    for (a, b) in starts.iter().zip(&parts_at).skip(1) {
        assert!((a - b).abs() < 0.05, "section at {a}, part at {b}");
    }
    let energy: Vec<f64> = sections.iter().map(|x| x["energy"].as_f64().unwrap()).collect();
    assert!(energy[3] > 0.85 && energy[5] > 0.85 && energy[0] < 0.4 && energy[4] < 0.5, "{energy:?}");
    let drops = nums("drops");
    assert_eq!(drops.len(), 2, "{r}");
    assert!((drops[0] - parts_at[3]).abs() < 0.03 && (drops[1] - parts_at[5]).abs() < 0.03, "{drops:?}");
    assert_eq!(r["ending"]["kind"], "cadence", "{r}");
    assert!((r["ending"]["at"].as_f64().unwrap() - secs(f.ending_at)).abs() < 0.05, "{r}");
    let end = r["musicEnd"].as_f64().unwrap();
    assert!(end > secs(f.ending_at) + 1.0 && end < r["seconds"].as_f64().unwrap(), "{r}");
    let conf = r["confidence"].as_f64().unwrap();
    assert!((0.6..=1.0).contains(&conf), "{r}");
    // the earlier fields are as they were
    assert!((r["bpm"].as_f64().unwrap() - 120.0).abs() < 1.0);
    // a range: everything in seconds of the media, inside it
    let r = s.execute("media.beats", json!({"item": item.0, "sourceIn": Tick::from_seconds_f64(20.0).0, "duration": Tick::from_seconds_f64(30.0).0})).unwrap();
    let drops: Vec<f64> = r["drops"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
    assert!(drops.iter().any(|d| (d - parts_at[3]).abs() < 0.03), "{r}");
    assert!(r["sections"].as_array().unwrap().iter().all(|x| (20.0..=50.01).contains(&x["start"].as_f64().unwrap())), "{r}");
}
