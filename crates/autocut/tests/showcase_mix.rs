//! The showcase's mix against a real headless session: by default speech first for small
//! speakers (the narration's chain on its track, the stems high-passed, the music dipped under
//! the voice), measured by `audio.analyzeMix`; `cinematic` keeps the earlier balance; a remix
//! turns one into the other and can run again without stacking effects.

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::mix::{self, MixProfile};
use filmcraft_autocut::showcase::{self, Built, Options, Script};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::{Value, json};

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

fn fact(item: u64, label: &str, unix: i64, seconds: f64) -> ClipFacts {
    ClipFacts {
        item,
        name: format!("{item}.mp4"),
        seconds,
        created: None,
        unix: Some(unix),
        present: true,
        coverage: 1.0,
        label: label.into(),
        kind: "humanoid".into(),
        description: format!("a {label}"),
        best: seconds / 2.0,
        frames: vec![FrameFact { seconds: seconds / 2.0, present: true, area: 0.4, center: [0.5, 0.5] }],
        ..ClipFacts::default()
    }
}

/// A narrated three-robot show mixed with `profile`.
async fn narrated(profile: MixProfile) -> (SessionExecutor, Built) {
    let ex = demo();
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 12.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 12.0),
        fact(7, "red toy robot", 1_665_841_668, 12.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    let script = Script {
        intro: "Hello and welcome, this is every robot I have met.".into(),
        lines: vec![
            "A friendly humanoid that waved at me in a shop.".into(),
            "Spot walked right up to me in the middle of the hall.".into(),
            "A small red toy robot from a market stall.".into(),
        ],
        outro: "Bye for now, see you soon at the next one.".into(),
        ..Script::default()
    };
    let mut opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), width: 640, height: 360, ..Options::default() };
    opts.look.mix = profile;
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    (ex, built)
}

async fn strip(ex: &SessionExecutor, label: &str) -> Value {
    let m = ex.execute("mixer.inspect", json!({})).await.unwrap();
    m["strips"].as_array().unwrap().iter().find(|s| s["ref"] == label).cloned().unwrap_or_else(|| panic!("no strip {label}"))
}

fn effects(strip: &Value) -> Vec<String> {
    strip["inserts"].as_array().unwrap().iter().map(|i| i["effect"].as_str().unwrap().to_string()).collect()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn the_default_mix_is_speech_first_for_small_speakers() {
    let (ex, built) = narrated(MixProfile::default()).await;
    let mix = built.mix.as_ref().expect("the mix is reported");
    assert_eq!(mix["profile"], "speech", "{mix}");
    assert!(built.notes.iter().all(|n| !n.starts_with("mix:")), "{:?}", built.notes);
    // the narration's chain, the stems' EQs, the music's presence dip following the ducking
    assert_eq!(effects(&strip(&ex, "A2").await), ["parametric_eq", "deesser", "single_band_compressor", "hard_limiter"]);
    for t in ["A1", "A3"] {
        assert_eq!(effects(&strip(&ex, t).await), ["parametric_eq"], "{t}");
    }
    let music = strip(&ex, "A3").await;
    let dip = music["lanes"]["fx.0.mid_gain"].as_array().unwrap_or_else(|| panic!("the dip's automation: {music}"));
    assert!(dip.len() >= 3 && dip.iter().any(|k| f(&k[1]) <= mix::CARVE_DB + 0.01) && dip.iter().any(|k| f(&k[1]) == 0.0), "{dip:?}");
    // measured: the voice well above the music while it speaks, no sub-bass, the limiter idle
    let a = ex.execute("audio.analyzeMix", json!({"speech": ["A2"], "music": ["A3"]})).await.unwrap();
    println!("{a:#}");
    assert!(f(&a["speech"]["toMusic"]["medianLu"]) >= 15.0, "{}", a["speech"]);
    assert!(f(&a["mix"]["bands"]["sub"]) < 0.2, "{}", a["mix"]["bands"]);
    for s in a["stems"].as_array().unwrap().iter().filter(|s| s["track"] != "transitions") {
        assert!(f(&s["bands"]["sub"]) < 0.25, "{}: {}", s["track"], s["bands"]);
    }
    // the synthetic voice is bright already: its EQ only takes the rumble out
    let voice = &mix["voice"];
    assert!(voice["tiltDb"].as_f64().is_some_and(|t| t > mix::VOICE_TILT) && voice["eqAmount"] == 0.0, "{voice}");
    // the chain changes its tone, not its level: the fader made up what it took
    let a2 = a["stems"].as_array().unwrap().iter().find(|s| s["track"] == "A2").unwrap();
    assert!((f(&a2["integratedLufs"]) - f(&voice["loudnessLufs"])).abs() < 0.3, "{a2} vs {voice}");
    assert!(f(&voice["faderDb"]) != 0.0 && (f(&strip(&ex, "A2").await["volumeDb"]) - f(&voice["faderDb"])).abs() < 1e-9, "{voice}");
    assert!(f(&a["limiter"]["maxGainReductionDb"]) <= 1.5, "{}", a["limiter"]);
    assert!(f(&a["mix"]["truePeakDbtp"]) <= -0.95, "{}", a["mix"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn cinematic_keeps_the_earlier_balance_and_a_remix_switches_between_them() {
    let (ex, built) = narrated(MixProfile::Cinematic).await;
    assert_eq!(built.mix.as_ref().unwrap()["profile"], "cinematic");
    for t in ["A1", "A2", "A3"] {
        assert!(effects(&strip(&ex, t).await).is_empty(), "{t}: no track effects in a cinematic mix");
    }
    // a dark voice, like the cloned narration (the synthetic one is bright): each line through a
    // low-pass at 1.2 kHz
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let lines: Vec<u64> = seq["audio"][1]["items"].as_array().unwrap().iter().map(|c| c["clip"].as_u64().unwrap()).collect();
    assert!(!lines.is_empty());
    ex.execute("effects.apply", json!({"clips": lines, "effect": "lowpass"})).await.unwrap();
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    for c in seq["audio"][1]["items"].as_array().unwrap() {
        let index = c["effects"].as_array().unwrap().iter().position(|e| e["effect"] == "lowpass").unwrap();
        ex.execute("effects.setParam", json!({"clip": c["clip"], "effect": index, "param": "cutoff", "value": 1200.0})).await.unwrap();
    }
    let before = ex.execute("audio.analyzeMix", json!({"speech": ["A2"], "music": ["A3"]})).await.unwrap();
    // to speech, twice: the same effects once, the same measurements
    let mut notes = Vec::new();
    let r = mix::remix(&ex, None, MixProfile::Speech, None, &mut notes).await.expect("the remix runs");
    assert_eq!(r["profile"], "speech");
    assert!(notes.is_empty(), "{notes:?}");
    // measured dark (about 4 dB under the target): the share of the EQ that tilt calls for
    let t = f(&r["voice"]["tiltDb"]);
    assert!(t < mix::VOICE_TILT - 3.0 && (f(&r["voice"]["eqAmount"]) - mix::voice_eq_amount(Some(t))).abs() < 0.02 && f(&r["voice"]["eqAmount"]) > 0.4, "{r}");
    let once = ex.execute("audio.analyzeMix", json!({"speech": ["A2"], "music": ["A3"]})).await.unwrap();
    let r = mix::remix(&ex, None, MixProfile::Speech, None, &mut notes).await.unwrap();
    // four on the narration, an EQ on each of the others (the shots' sound, music, two effects)
    assert_eq!(r["removedEffects"], 8, "{r}");
    assert_eq!(effects(&strip(&ex, "A2").await), ["parametric_eq", "deesser", "single_band_compressor", "hard_limiter"]);
    assert_eq!(effects(&strip(&ex, "A3").await), ["parametric_eq"]);
    let twice = ex.execute("audio.analyzeMix", json!({"speech": ["A2"], "music": ["A3"]})).await.unwrap();
    for k in ["integratedLufs", "truePeakDbtp"] {
        assert!((f(&once["mix"][k]) - f(&twice["mix"][k])).abs() < 0.05, "{k}: {} vs {}", once["mix"][k], twice["mix"][k]);
    }
    // what speech changed, measured: less below 120 Hz, more presence in the voice
    let sub = |a: &Value| f(&a["mix"]["bands"]["sub"]);
    assert!(sub(&once) < sub(&before), "{} → {}", sub(&before), sub(&once));
    let voice = |a: &Value| a["stems"].as_array().unwrap().iter().find(|s| s["track"] == "A2").map(|s| f(&s["bands"]["presence"])).unwrap();
    assert!(voice(&once) > voice(&before), "{} → {}", voice(&before), voice(&once));
    // gentle: the same stems treated, the narration's EQ voice-preserving (no low-mid cut)
    let r = mix::remix(&ex, None, MixProfile::Gentle, None, &mut notes).await.unwrap();
    assert_eq!(r["profile"], "gentle");
    assert_eq!(effects(&strip(&ex, "A2").await), ["parametric_eq", "deesser", "single_band_compressor", "hard_limiter"]);
    let gentle = ex.execute("audio.analyzeMix", json!({"speech": ["A2"], "music": ["A3"]})).await.unwrap();
    assert!(sub(&gentle) < sub(&before) && voice(&gentle) > voice(&before), "{gentle}");
    // and back to cinematic: the track effects are gone
    let r = mix::remix(&ex, None, MixProfile::Cinematic, None, &mut notes).await.unwrap();
    assert_eq!(r["profile"], "cinematic");
    for t in ["A1", "A2", "A3"] {
        assert!(effects(&strip(&ex, t).await).is_empty(), "{t}");
    }
    assert!(strip(&ex, "A3").await["lanes"].as_object().is_none_or(|l| l.keys().all(|k| !k.starts_with("fx."))));
    // a sequence that is not a showcase is refused, not mixed
    ex.execute("file.newSequence", json!({"name": "empty", "audio": 2, "video": 1})).await.unwrap();
    assert!(mix::remix(&ex, None, MixProfile::Speech, None, &mut notes).await.is_err());
}
