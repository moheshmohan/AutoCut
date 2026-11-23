//! `sequence.checkMotion` on sequences of titles built here (no media is committed, nothing is
//! rendered).
use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

fn secs(s: f64) -> i64 {
    Tick::from_seconds_f64(s).0
}

fn with_code<'a>(r: &'a Value, code: &str) -> Vec<&'a Value> {
    r["findings"].as_array().unwrap().iter().filter(|f| f["code"] == code).collect()
}

/// A session with an empty 640×360 sequence at `fps` (two video tracks).
fn session(fps: u32) -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": fps, "video": 2, "audio": 1})).unwrap();
    s
}

/// A title of `text` on V2 from `at` for `seconds`: its clip id.
fn title(s: &mut Session, text: &str, at: f64, seconds: f64) -> u64 {
    s.execute("graphics.newText", json!({"text": text, "size": 40, "seconds": seconds, "time": secs(at), "track": "V2"})).unwrap()["clip"].as_u64().unwrap()
}

fn animate(s: &mut Session, p: Value) {
    s.execute("graphics.animateText", p).unwrap();
}

fn check(s: &mut Session, p: Value) -> Value {
    s.execute(CMD, p).unwrap()
}

#[test]
fn overshooting_text_eases_warn_unless_the_style_is_playful() {
    let mut s = session(30);
    let c = title(&mut s, "Hello", 0.0, 4.0);
    // the quiet default (Quart Out) passes
    animate(&mut s, json!({"clip": c, "in": "pop"}));
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "overshoot").is_empty(), "{r:#}");
    assert_eq!(r["style"], json!({"playful": false, "from": "default"}));
    // Back Out passes its mark; Bounce Out (the drop's own) bounces
    for (easing, family) in [("Back Out", "back"), ("Elastic Out", "elastic"), ("Bounce Out", "bounce")] {
        animate(&mut s, json!({"clip": c, "in": {"kind": "pop", "easing": easing}}));
        let r = check(&mut s, json!({}));
        let f = with_code(&r, "overshoot");
        assert_eq!(f.len(), 1, "{easing}: {r:#}");
        assert_eq!((f[0]["family"].as_str(), f[0]["clip"].as_u64(), f[0]["severity"].as_str()), (Some(family), Some(c), Some("warn")));
        assert!(f[0]["message"].as_str().unwrap().contains("Quart Out"), "says what to change: {}", f[0]["message"]);
    }
    // a Custom Bézier overshoots only when a handle leaves 0..1
    animate(&mut s, json!({"clip": c, "in": {"kind": "pop", "easing": 31, "bezier": [0.3, 1.6, 0.5, 1.0]}}));
    assert_eq!(with_code(&check(&mut s, json!({})), "overshoot").len(), 1);
    animate(&mut s, json!({"clip": c, "in": {"kind": "pop", "easing": 31, "bezier": [0.3, 0.9, 0.5, 1.0]}}));
    assert!(with_code(&check(&mut s, json!({})), "overshoot").is_empty());
    // a playful style, by the call or by the taste profile; calm overrules the profile
    animate(&mut s, json!({"clip": c, "in": {"kind": "pop", "easing": "Back Out"}}));
    let r = check(&mut s, json!({"style": "playful"}));
    assert!(with_code(&r, "overshoot").is_empty(), "{r:#}");
    assert_eq!(r["skipped"][0]["check"], "overshoot");
    s.execute("autocut.taste", json!({"set": {"motion": "playful"}})).unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "overshoot").is_empty());
    assert_eq!(r["style"]["from"], "taste");
    assert_eq!(with_code(&check(&mut s, json!({"style": "Calm"})), "overshoot").len(), 1);
}

#[test]
fn springs_and_choreographed_curves_that_overshoot_warn() {
    let mut s = session(30);
    let c = title(&mut s, "Spring", 0.0, 4.0);
    let tween = |ease: Value| json!({"target": {"clip": c, "effect": "motion"}, "tweens": [{"prop": "scale", "from": 50, "to": 100, "duration": 0.6, "ease": ease, "position": 0.5}]});
    let quiet = ["spring", "power2.out", "expo.out"];
    for ease in quiet {
        s.execute("keyframes.choreograph", tween(json!(ease))).unwrap();
        let r = check(&mut s, json!({}));
        assert!(with_code(&r, "overshoot").is_empty(), "{ease}: {r:#}");
    }
    for (ease, source) in [("spring(0.5,150,1)", "spring"), ("back.out", "curve"), ("elastic.out", "curve"), ("bounce.out", "curve")] {
        s.execute("keyframes.choreograph", tween(json!(ease))).unwrap();
        let r = check(&mut s, json!({}));
        let f = with_code(&r, "overshoot");
        assert_eq!(f.len(), 1, "{ease}: {r:#}");
        assert_eq!((f[0]["source"].as_str(), f[0]["param"].as_str()), (Some(source), Some("scale")), "{ease}");
        // at the keyframe, on the timeline
        assert_eq!(f[0]["time"].as_i64(), Some(secs(0.5)), "{ease}: {}", f[0]);
    }
    assert_eq!(check(&mut s, json!({}))["checked"]["curveSegments"], 1);
    s.execute("keyframes.choreograph", tween(json!("spring(0.6,150,1)"))).unwrap();
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "overshoot");
    assert!((f[0]["overshoot"].as_f64().unwrap() - 0.095).abs() < 0.01, "ζ 0.6 passes by about 9.5 %: {}", f[0]);
    assert!(with_code(&check(&mut s, json!({"style": "playful"})), "overshoot").is_empty());
}

#[test]
fn transitions_that_bounce_or_overshoot_by_design_warn() {
    let mut s = session(30);
    let c = title(&mut s, "Card", 1.0, 3.0);
    // a transition with the host's easing (the GL and engine transitions)
    s.execute("sequence.applyVideoTransition", json!({"clip": c, "edge": "in", "effect": "blend_dissolve", "frames": 12})).unwrap();
    assert!(with_code(&check(&mut s, json!({})), "overshoot").is_empty());
    let tr = s.active_sequence().unwrap().video_tracks[1].transitions[0].id.0;
    s.execute("sequence.setTransition", json!({"transition": tr, "params": {"easing": 29}})).unwrap();
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "overshoot");
    assert_eq!((f.len(), f[0]["family"].as_str(), f[0]["transition"].as_u64()), (1, Some("bounce"), Some(tr)), "{r:#}");
    s.execute("sequence.removeTransition", json!({"transitions": [tr]})).unwrap();
    s.execute("sequence.applyVideoTransition", json!({"clip": c, "edge": "in", "effect": "pop_motion", "frames": 12})).unwrap();
    let r = check(&mut s, json!({}));
    assert_eq!(with_code(&r, "overshoot")[0]["family"], "builtIn", "{r:#}");
    let tr = s.active_sequence().unwrap().video_tracks[1].transitions[0].id.0;
    s.execute("sequence.setTransition", json!({"transition": tr, "params": {"overshoot": 0}})).unwrap();
    assert!(with_code(&check(&mut s, json!({})), "overshoot").is_empty());
}

#[test]
fn entrances_and_exits_are_held_to_the_bands_at_the_sequences_rate() {
    let mut s = session(30);
    let c = title(&mut s, "Hi", 0.0, 4.0);
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "duration": 0.5}, "out": {"kind": "fade", "duration": 0.3}}));
    let r = check(&mut s, json!({}));
    for code in ["entranceDuration", "exitDuration", "exitOutlastsEntrance"] {
        assert!(with_code(&r, code).is_empty(), "{code}: {r:#}");
    }
    assert_eq!(r["bands"]["entrance"]["frames"], json!([8.0, 30.0]));
    // 3 frames in snaps; 1.5 s out drags, and outlasts the entrance
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "duration": 0.1}, "out": {"kind": "fade", "duration": 1.5}}));
    let r = check(&mut s, json!({}));
    let e = with_code(&r, "entranceDuration");
    assert_eq!((e.len(), e[0]["short"].as_bool(), e[0]["frames"].as_f64()), (1, Some(true), Some(3.0)), "{r:#}");
    assert!(e[0]["message"].as_str().unwrap().contains("at least 8 frames"));
    let x = with_code(&r, "exitDuration");
    assert_eq!((x.len(), x[0]["short"].as_bool()), (1, Some(false)), "{r:#}");
    assert!(x[0]["message"].as_str().unwrap().contains("24 frames"));
    assert_eq!(with_code(&r, "exitOutlastsEntrance").len(), 1);
    // a frame or a tenth longer is not worth a note (0.30 s against 0.28 s on a real project)
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "duration": 0.28}, "out": {"kind": "fade", "duration": 0.30}}));
    assert!(with_code(&check(&mut s, json!({})), "exitOutlastsEntrance").is_empty());
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "duration": 0.3}, "out": {"kind": "fade", "duration": 0.4}}));
    assert_eq!(with_code(&check(&mut s, json!({})), "exitOutlastsEntrance").len(), 1, "3 frames over 9 is");
    // at 60 fps the bands are in 60ths: 0.5 s is 30 frames, inside 16–60
    let mut s = session(60);
    let c = title(&mut s, "Hi", 0.0, 4.0);
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "duration": 0.5}}));
    let r = check(&mut s, json!({}));
    assert_eq!(r["bands"]["entrance"]["frames"], json!([16.0, 60.0]), "{r:#}");
    assert!(with_code(&r, "entranceDuration").is_empty());
    // bands of the caller's own, in frames at the sequence's rate
    let r = check(&mut s, json!({"bands": {"entrance": [40, 80]}}));
    assert_eq!(with_code(&r, "entranceDuration").len(), 1, "{r:#}");
}

#[test]
fn transition_and_keyframed_entrances_of_titles_are_measured() {
    let mut s = session(30);
    let c = title(&mut s, "Card", 1.0, 3.0);
    // a 2-frame fade from nothing, and a 1 s fade out: too quick, then outlasting it
    s.execute("sequence.applyVideoTransition", json!({"clip": c, "edge": "in", "effect": "cross_dissolve", "frames": 2})).unwrap();
    s.execute("sequence.applyVideoTransition", json!({"clip": c, "edge": "out", "effect": "cross_dissolve", "frames": 30})).unwrap();
    let r = check(&mut s, json!({}));
    let e = with_code(&r, "entranceDuration");
    assert_eq!((e.len(), e[0]["source"].as_str()), (1, Some("transition")), "{r:#}");
    assert_eq!(with_code(&r, "exitDuration").len(), 1);
    assert_eq!(with_code(&r, "exitOutlastsEntrance")[0]["source"], "transition");
    // keyframes: opacity up over 12 frames from the clip's first frame is an entrance in the band
    let mut s = session(30);
    let c = title(&mut s, "Card", 1.0, 3.0);
    let kf = |from: f64, to: f64, seconds: f64, at: f64| json!({"target": {"clip": c, "effect": "opacity"}, "tweens": [{"prop": "opacity", "from": from, "to": to, "duration": seconds, "ease": "power2.out", "position": at}]});
    s.execute("keyframes.choreograph", kf(0.0, 100.0, 0.4, 0.0)).unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "entranceDuration").is_empty() && with_code(&r, "edgePop").is_empty(), "{r:#}");
    assert_eq!(r["checked"]["titleEntrances"][0]["kind"], "keyframes (opacity)", "{r:#}");
}

#[test]
fn a_long_stagger_warns_and_says_what_fits() {
    let mut s = session(30);
    let c = title(&mut s, "Short line", 0.0, 4.0);
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "unit": "letter", "stagger": 0.03}}));
    assert!(with_code(&check(&mut s, json!({})), "longStagger").is_empty());
    // 37 letters 0.03 s apart: 1.08 s from the first to start to the last
    let c = title(&mut s, "Every letter of this title arrives one by one", 5.0, 4.0);
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "unit": "letter", "stagger": 0.03}}));
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "longStagger");
    assert_eq!((f.len(), f[0]["units"].as_u64(), f[0]["clip"].as_u64()), (1, Some(37), Some(c)), "{r:#}");
    assert!((f[0]["total"].as_f64().unwrap() - 1.08).abs() < 1e-9);
    assert!(f[0]["message"].as_str().unwrap().contains("by word"), "{}", f[0]["message"]);
    // by word it fits
    animate(&mut s, json!({"clip": c, "in": {"kind": "fade", "unit": "word", "stagger": 0.08}}));
    assert!(with_code(&check(&mut s, json!({})), "longStagger").is_empty());
    // the caller's own limit
    assert_eq!(with_code(&check(&mut s, json!({"bands": {"stagger": 0.3}})), "longStagger").len(), 1);
    // a typewriter types at its own speed
    animate(&mut s, json!({"clip": c, "clear": true, "preset": "typewriter"}));
    assert!(with_code(&check(&mut s, json!({})), "longStagger").is_empty());
}

#[test]
fn every_title_entering_alike_warns() {
    let mut s = session(30);
    let mut clips = Vec::new();
    for i in 0..3 {
        let c = title(&mut s, "Chapter", i as f64 * 3.0, 2.0);
        animate(&mut s, json!({"clip": c, "in": "pop"}));
        clips.push(c);
    }
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "sameEntrance");
    assert_eq!((f.len(), f[0]["kind"].as_str(), f[0]["titles"].as_u64()), (1, Some("pop"), Some(3)), "{r:#}");
    assert_eq!(f[0]["clips"], json!(clips));
    // one different entrance breaks the pattern; two titles are no pattern
    animate(&mut s, json!({"clip": clips[1], "in": "slideUp"}));
    assert!(with_code(&check(&mut s, json!({})), "sameEntrance").is_empty());
    let mut s = session(30);
    for i in 0..2 {
        let c = title(&mut s, "Chapter", i as f64 * 3.0, 2.0);
        animate(&mut s, json!({"clip": c, "in": "pop"}));
    }
    assert!(with_code(&check(&mut s, json!({})), "sameEntrance").is_empty());
}

#[test]
fn looping_text_warns_unless_it_drifts_slowly() {
    let mut s = session(30);
    let c = title(&mut s, "Wavy", 0.0, 5.0);
    animate(&mut s, json!({"clip": c, "in": "fade"}));
    assert!(with_code(&check(&mut s, json!({})), "loopingText").is_empty());
    animate(&mut s, json!({"clip": c, "loop": "wave"}));
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "loopingText");
    assert_eq!((f.len(), f[0]["kind"].as_str(), f[0]["time"].as_i64(), f[0]["end"].as_i64()), (1, Some("wave"), Some(0), Some(secs(5.0))), "{r:#}");
    animate(&mut s, json!({"clip": c, "loop": {"kind": "wave", "duration": 9}}));
    assert!(with_code(&check(&mut s, json!({})), "loopingText").is_empty());
}

#[test]
fn a_keyframe_jump_at_a_clips_edge_is_a_pop() {
    let mut s = session(30);
    let c = title(&mut s, "Pop", 1.0, 3.0);
    let frame = 1.0 / 30.0;
    // opacity 0 → 100 in one frame at the clip's first frame
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": c, "effect": "opacity"}, "tweens": [{"prop": "opacity", "from": 0, "to": 100, "duration": frame, "ease": "none", "position": 0}]}),
    )
    .unwrap();
    let r = check(&mut s, json!({}));
    let f = with_code(&r, "edgePop");
    assert_eq!((f.len(), f[0]["edge"].as_str(), f[0]["param"].as_str(), f[0]["time"].as_i64()), (1, Some("first"), Some("opacity"), Some(secs(1.0))), "{r:#}");
    // scale snapping at the last frame
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": c, "effect": "motion"}, "tweens": [{"prop": "scale", "from": 100, "to": 0, "duration": frame, "ease": "none", "position": 3.0 - 2.0 * frame}]}),
    )
    .unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "edgePop").iter().any(|f| f["edge"] == "last" && f["param"] == "scale"), "{r:#}");
    // eased over 10 frames: no pop
    let mut s = session(30);
    let c = title(&mut s, "Pop", 1.0, 3.0);
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": c, "effect": "opacity"}, "tweens": [{"prop": "opacity", "from": 0, "to": 100, "duration": 10.0 * frame, "ease": "power1.out", "position": 0}]}),
    )
    .unwrap();
    assert!(with_code(&check(&mut s, json!({})), "edgePop").is_empty());
}

#[test]
fn hostile_parameters_are_refused() {
    let mut s = session(30);
    title(&mut s, "Hi", 0.0, 2.0);
    for p in [
        json!(3),
        json!("loud"),
        json!({"sequence": "main"}),
        json!({"sequence": 999_999}),
        json!({"style": "bouncy"}),
        json!({"style": 1}),
        json!({"bands": "wide"}),
        json!({"bands": {"entrance": [5]}}),
        json!({"bands": {"entrance": [30, 8]}}),
        json!({"bands": {"entrance": [-1, 8]}}),
        json!({"bands": {"entrance": [0, 0]}}),
        json!({"bands": {"exit": [1, 1e12]}}),
        json!({"bands": {"exit": ["a", "b"]}}),
        json!({"bands": {"stagger": 0}}),
        json!({"bands": {"stagger": -1}}),
        json!({"bands": {"stagger": 1e9}}),
        json!({"bands": {"speed": 3}}),
        json!({"holds": []}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // an empty sequence says so
    let mut s = session(30);
    assert!(s.execute(CMD, json!({})).is_err());
    // hostile numbers inside the project are made safe, never a panic
    let mut s = session(30);
    let c = title(&mut s, "Hostile", 0.0, 2.0);
    animate(
        &mut s,
        json!({"clip": c, "in": {"kind": "drop", "amount": 1e300, "stagger": 60, "duration": 60, "easing": 3}, "loop": {"kind": "shake", "duration": 0}}),
    );
    let r = check(&mut s, json!({"bands": {"entrance": [0, 100000], "exit": [0, 0.5], "stagger": 600}}));
    assert_eq!(r["blocking"], false, "{r:#}");
}

#[test]
fn a_quiet_sequence_passes() {
    let mut s = session(30);
    for (i, kind) in ["fade", "slideUp", "pop"].iter().enumerate() {
        let c = title(&mut s, "Chapter one", i as f64 * 3.0, 2.5);
        animate(&mut s, json!({"clip": c, "in": kind, "out": {"kind": "fade", "duration": 0.3}}));
    }
    let r = check(&mut s, json!({}));
    assert_eq!(r["ok"], true, "{r:#}");
    assert_eq!((r["checked"]["titles"].as_u64(), r["checked"]["animators"].as_u64()), (Some(3), Some(6)));
}

#[test]
fn an_overlay_with_alpha_has_its_entrance_measured_and_footage_does_not() {
    let dir = crate::media_test_util::tmp_dir("motion-check-overlay");
    let mut s = session(30);
    let place = |s: &mut Session, name: &str, alpha: bool, track: &str| {
        let path = dir.join(name);
        if alpha {
            image::save_buffer(&path, &[255, 255, 255, 128].repeat(64 * 36), 64, 36, image::ColorType::Rgba8).unwrap();
        } else {
            image::save_buffer(&path, &[90; 64 * 36 * 3], 64, 36, image::ColorType::Rgb8).unwrap();
        }
        let item = crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap();
        let r = s.execute("timeline.place", json!({"item": item.0, "track": track, "time": secs(1.0), "duration": secs(2.0), "audio": false})).unwrap();
        let clip = r["clips"][0].as_u64().or_else(|| r["clip"].as_u64()).unwrap();
        s.execute("sequence.applyVideoTransition", json!({"clip": clip, "edge": "in", "effect": "cross_dissolve", "frames": 2})).unwrap();
        clip
    };
    let footage = place(&mut s, "plate.png", false, "V1");
    let overlay = place(&mut s, "overlay.png", true, "V2");
    let r = check(&mut s, json!({}));
    assert_eq!(r["checked"]["overlays"], 1, "{r:#}");
    let f = with_code(&r, "entranceDuration");
    assert_eq!(
        f.iter().map(|f| f["clip"].as_u64().unwrap()).collect::<Vec<_>>(),
        [overlay],
        "only the overlay's entrance, not the footage's ({footage}): {r:#}"
    );
}

#[test]
fn an_exit_is_not_held_to_an_entrance_that_is_a_cut() {
    assert!(!outlasts(1.0, 3.0), "a one-frame entrance is a cut");
    assert!(!outlasts(8.4, 9.0) && !outlasts(20.0, 22.0), "within a frame or a tenth");
    assert!(outlasts(8.0, 10.0) && outlasts(20.0, 22.5));
    assert!(!outlasts(f64::NAN, 10.0) && !outlasts(8.0, f64::NAN));
}
