use serde_json::{Value, json};

use filmcraft_project::keyframe::Interpolation;
use filmcraft_project::{ClipId, Param};
use filmcraft_time::Tick;

use super::MAX_TWEENS;
use crate::Session;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

/// The first three V1 clips (5, 4.5 and 4 s long).
fn clips(s: &Session) -> Vec<u64> {
    s.active_sequence().unwrap().video_tracks[0].items.iter().take(3).map(|i| i.id.0).collect()
}

fn param(s: &Session, clip: u64, effect: &str, id: &str) -> Param {
    s.active_sequence().unwrap().find_item(ClipId(clip)).unwrap().1.effect(effect).unwrap().params[id].clone()
}

/// Media time of `seconds` into `clip`.
fn media(s: &Session, clip: u64, seconds: f64) -> Tick {
    let it = s.active_sequence().unwrap().find_item(ClipId(clip)).unwrap().1.clone();
    it.effect_time_at(it.start + Tick::from_seconds_f64(seconds))
}

fn starts(r: &Value) -> Vec<f64> {
    r["tweens"].as_array().unwrap().iter().map(|t| t["start"].as_f64().unwrap()).collect()
}

#[test]
fn every_position_form_places_its_tween() {
    let mut s = demo();
    let c = clips(&s)[0];
    let before = s.project.clone();
    let undo = s.history.undo.len();
    let tw = |position: Value, duration: f64| json!({"prop": "rotation", "to": 10, "duration": duration, "position": position});
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({
                "target": {"clip": c},
                "labels": {"intro": 1.0, "beat": 2.5},
                "dryRun": true,
                "tweens": [
                    tw(Value::Null, 0.5),         // 0: the start of the timeline → 0
                    tw(Value::Null, 0.5),         // 1: its end → 0.5
                    tw(json!("<"), 0.2),          // 2: with the previous start → 0.5
                    tw(json!(">"), 0.3),          // 3: at the previous end → 0.7
                    tw(json!("+=0.2"), 0.2),      // 4: timeline end 1.0 + 0.2 → 1.2
                    tw(json!("-=0.1"), 0.2),      // 5: timeline end 1.4 - 0.1 → 1.3
                    tw(json!("<0.2"), 0.2),       // 6: previous start 1.3 + 0.2 → 1.5
                    tw(json!("<-0.1"), 0.2),      // 7: 1.5 - 0.1 → 1.4
                    tw(json!(">0.1"), 0.2),       // 8: previous end 1.6 + 0.1 → 1.7
                    tw(json!(">-=0.1"), 0.2),     // 9: 1.9 - 0.1 → 1.8
                    tw(json!("intro"), 0.2),      // 10: → 1.0
                    tw(json!("beat+=0.5"), 0.2),  // 11: → 3.0
                    tw(json!("beat-=0.25"), 0.2), // 12: → 2.25
                    tw(json!(0.4), 0.2),          // 13: absolute → 0.4
                    tw(json!("3.5"), 0.2),        // 14: a numeric string → 3.5
                    tw(json!("<25%"), 0.4),       // 15: 3.5 + 25 % of 0.2 → 3.55
                    tw(json!("+=50%"), 0.4),      // 16: timeline end 3.95 + 50 % of 0.4 → 4.15
                    tw(json!("<+=50%"), 0.2),     // 17: previous start 4.15 + 50 % of 0.2 → 4.25
                    tw(json!(">-25%"), 0.2),      // 18: previous end 4.45 - 25 % of 0.2 → 4.4
                    tw(json!("intro+=50%"), 0.2), // 19: 1.0 + 50 % of 0.2 → 1.1
                ]
            }),
        )
        .unwrap();
    let want = [0.0, 0.5, 0.5, 0.7, 1.2, 1.3, 1.5, 1.4, 1.7, 1.8, 1.0, 3.0, 2.25, 0.4, 3.5, 3.55, 4.15, 4.25, 4.4, 1.1];
    let got = starts(&r);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "tweens[{i}] starts at {g}, not {w}");
    }
    assert_eq!(r["dryRun"], true);
    assert!(r["written"].as_u64().unwrap() > 0);
    // a dry run reports and changes nothing
    assert!(std::sync::Arc::ptr_eq(&s.project, &before));
    assert_eq!(s.history.undo.len(), undo);
}

#[test]
fn tweens_become_ordinary_keyframes_in_one_undo_step() {
    let mut s = demo();
    let c = clips(&s)[0];
    let before = s.project.clone();
    let undo = s.history.undo.len();
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({
                "target": {"clip": c},
                "tweens": [
                    {"prop": "scale", "from": 100, "to": 400, "duration": 1.0, "ease": "none"},
                    {"prop": "scale", "to": 200, "duration": 1.0, "ease": "spring"},
                    {"prop": "opacity", "effect": "opacity", "from": 0, "to": 100, "duration": 0.5, "position": 0, "ease": "steps(4)"},
                    {"prop": "rotation", "to": 90, "duration": 1.0, "position": "<", "ease": "power2.out", "perceptual": false},
                    {"prop": "position", "to": [800, 400], "duration": 1.0, "position": 0, "ease": {"spring": {"dampingRatio": 1.0}}},
                ]
            }),
        )
        .unwrap();
    assert_eq!(s.history.undo.len(), undo + 1, "one undo step");
    assert_eq!(s.history.undo.last().unwrap().0, "Choreograph Keyframes");
    // scale: 100 → 400 linear in log space (a scale is perceptual by default), then a spring to 200
    let scale = param(&s, c, "motion", "scale");
    assert_eq!(scale.keyframes.len(), 3);
    assert_eq!(scale.keyframes[0].interp, Interpolation::Linear);
    assert!(scale.keyframes[0].segment().perceptual);
    assert!((scale.f64_at(media(&s, c, 0.5)) - 200.0).abs() < 1e-3, "an even zoom: {}", scale.f64_at(media(&s, c, 0.5)));
    assert_eq!(scale.keyframes[1].interp, Interpolation::Spring);
    assert_eq!(scale.f64_at(media(&s, c, 1.0)), 400.0);
    assert_eq!(scale.f64_at(media(&s, c, 2.0)), 200.0);
    // opacity in four steps over half a second
    let op = param(&s, c, "opacity", "opacity");
    assert_eq!(op.keyframes[0].interp, Interpolation::Steps);
    assert_eq!(op.f64_at(media(&s, c, 0.2)), 25.0);
    // rotation with the previous tween (opacity at 0): power2.out is a cubic ease out
    let rot = param(&s, c, "motion", "rotation");
    assert_eq!(rot.keyframes[0].interp, Interpolation::Custom);
    assert!(!rot.keyframes[0].segment().perceptual);
    let mid = rot.f64_at(media(&s, c, 0.5));
    assert!((mid - 90.0 * 0.875).abs() < 1.0, "{mid}");
    // position: a critically damped spring, no overshoot
    let pos = param(&s, c, "motion", "position");
    assert_eq!(pos.keyframes[0].segment().spring.map(|k| k.damping_ratio), Some(1.0));
    assert_eq!(pos.vec2_at(media(&s, c, 1.0)), filmcraft_geom::Vec2::new(800.0, 400.0));
    // the report lists every keyframe written, with its mode
    let ks = r["keyframes"].as_array().unwrap();
    assert_eq!(r["written"].as_u64().unwrap() as usize, ks.len());
    assert_eq!(ks.len(), 3 + 2 + 2 + 2);
    assert!(ks.iter().any(|k| k["param"] == "scale" && k["interpolation"] == "spring" && k["seconds"] == 1.0));
    assert!(ks.iter().any(|k| k["param"] == "position" && k["value"] == json!([800.0, 400.0])));
    // undo takes all of it back
    s.undo();
    assert!(std::sync::Arc::ptr_eq(&s.project, &before));
}

#[test]
fn stagger_delays_each_target_in_order() {
    let mut s = demo();
    let cs = clips(&s);
    for (from, delays) in [(json!("start"), [0.0, 0.1, 0.2]), (json!("end"), [0.2, 0.1, 0.0]), (json!("center"), [0.1, 0.0, 0.1]), (json!(0), [0.0, 0.1, 0.2])]
    {
        let r = s
            .execute(
                "keyframes.choreograph",
                json!({"target": {"clips": cs}, "tweens": [{"prop": "scale", "to": 150, "duration": 0.5}], "stagger": {"each": 0.1, "from": from}, "dryRun": true}),
            )
            .unwrap();
        let got: Vec<f64> = r["targets"].as_array().unwrap().iter().map(|t| t["delay"].as_f64().unwrap()).collect();
        for (g, w) in got.iter().zip(delays) {
            assert!((g - w).abs() < 1e-9, "{from}: {got:?}");
        }
        // the last tween ends at 0.5 s; the latest target is delayed by the largest delay
        let last = delays.iter().copied().fold(0.0, f64::max);
        assert!((r["duration"].as_f64().unwrap() - (0.5 + last)).abs() < 1e-9, "{from}: {}", r["duration"]);
    }
    // written for real: each clip's keyframes start at its delay
    s.execute("keyframes.choreograph", json!({"target": {"clips": cs}, "tweens": [{"prop": "scale", "to": 150, "duration": 0.5}], "stagger": 0.25})).unwrap();
    for (i, c) in cs.iter().enumerate() {
        let k = &param(&s, *c, "motion", "scale").keyframes;
        assert_eq!(k[0].time, media(&s, *c, 0.25 * i as f64), "clip {i}");
        assert_eq!(k[1].time, media(&s, *c, 0.25 * i as f64 + 0.5));
    }
    // several effects of one clip are several targets
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({"target": {"clip": cs[0], "effects": ["motion", "opacity"]}, "tweens": [{"prop": "scale", "to": 120}], "dryRun": true}),
        )
        .unwrap_err();
    assert!(r.to_string().contains("`opacity` has no `scale`"), "{r}");
}

#[test]
fn eases_map_to_the_keyframe_modes() {
    let mut s = demo();
    let c = clips(&s)[0];
    for (ease, mode) in [
        (json!("none"), Interpolation::Linear),
        (json!("linear"), Interpolation::Linear),
        (json!("power0.inOut"), Interpolation::Linear),
        (json!("hold"), Interpolation::Hold),
        (json!("bezier"), Interpolation::Bezier),
        (json!("easeOut"), Interpolation::EaseOut),
        (json!("spring"), Interpolation::Spring),
        (json!("spring(0.6, 300, 2)"), Interpolation::Spring),
        (json!("steps(5)"), Interpolation::Steps),
        (json!("wiggle(4, 3, 9)"), Interpolation::Wiggle),
        (json!("expo.inOut"), Interpolation::Custom),
        (json!("back.out"), Interpolation::Custom),
        (json!("elastic"), Interpolation::Custom),
        (json!("sine.in"), Interpolation::Custom),
        (json!({"curve": [[0.5, 0.8]]}), Interpolation::Custom),
        (json!({"steps": 3}), Interpolation::Steps),
        (json!({"wiggle": {"amplitude": 2}}), Interpolation::Wiggle),
    ] {
        let r = s
            .execute("keyframes.choreograph", json!({"target": {"clip": c}, "tweens": [{"prop": "rotation", "to": 45, "ease": ease}], "dryRun": true}))
            .unwrap();
        assert_eq!(r["keyframes"][0]["interpolation"], mode.name(), "{ease}");
    }
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({"target": {"clip": c}, "tweens": [{"prop": "rotation", "to": 45, "ease": "spring(0.6, 300, 2)"}], "dryRun": true}),
        )
        .unwrap();
    assert_eq!(r["keyframes"][0]["spring"], json!({"dampingRatio": 0.6, "stiffness": 300.0, "mass": 2.0, "maxOvershoot": 0.25}));
    // back.out overshoots (asked for by name), its argument noted as unused
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({"target": {"clip": c}, "tweens": [{"prop": "rotation", "from": 0, "to": 100, "duration": 1, "ease": "back.out(1.7)"}]}),
        )
        .unwrap();
    assert_eq!(r["notes"].as_array().unwrap().len(), 1, "{r}");
    let rot = param(&s, c, "motion", "rotation");
    let peak = (1..100).map(|i| rot.f64_at(media(&s, c, i as f64 / 100.0))).fold(0.0, f64::max);
    assert!(peak > 105.0 && peak < 115.0, "{peak}");
    // the default ease is GSAP's power1.out: quad out, 0.75 at half time
    s.execute("keyframes.choreograph", json!({"target": {"clip": c}, "tweens": [{"prop": "rotation", "from": 0, "to": 100, "duration": 1}]})).unwrap();
    let mid = param(&s, c, "motion", "rotation").f64_at(media(&s, c, 0.5));
    assert!((mid - 75.0).abs() < 0.5, "{mid}");
    // defaults apply to every tween that does not say
    let r = s
        .execute("keyframes.choreograph", json!({"target": {"clip": c}, "defaults": {"duration": 0.25, "ease": "steps(2)"}, "tweens": [{"prop": "rotation", "to": 1}, {"prop": "rotation", "to": 2}], "dryRun": true}))
        .unwrap();
    assert_eq!(starts(&r), [0.0, 0.25]);
    assert_eq!(r["tweens"][1]["interpolation"], "steps");
}

#[test]
fn hostile_params_are_refused_and_change_nothing() {
    let mut s = demo();
    let c = clips(&s)[0];
    let ok_tween = json!({"prop": "scale", "to": 150});
    let many: Vec<Value> = (0..=MAX_TWEENS).map(|_| json!({"prop": "rotation", "to": 1, "duration": 0.001})).collect();
    let target = json!({"clip": c});
    let before = s.project.clone();
    let undo = s.history.undo.len();
    for p in [
        json!(null),
        json!([1, 2]),
        json!({"tweens": [ok_tween]}),
        json!({"target": {"clip": c}, "tweens": []}),
        json!({"target": {"clip": c}}),
        json!({"target": target, "tweens": [ok_tween], "extra": 1}),
        json!({"target": {"clip": c, "track": 1}, "tweens": [ok_tween]}),
        json!({"target": {"clip": 987_654_321}, "tweens": [ok_tween]}),
        json!({"target": {"clip": "x"}, "tweens": [ok_tween]}),
        json!({"target": {}, "tweens": [ok_tween]}),
        json!({"target": {"clip": c, "effect": "nonesuch"}, "tweens": [ok_tween]}),
        json!({"target": {"clip": c, "effect": 99}, "tweens": [ok_tween]}),
        json!({"target": {"clip": c, "mask": 3}, "tweens": [ok_tween]}),
        json!({"target": {"clip": c, "effect": "motion", "layer": 0}, "tweens": [ok_tween]}),
        json!({"target": target, "tweens": [{"prop": "wobble", "to": 1}]}),
        json!({"target": target, "tweens": [{"to": 1}]}),
        json!({"target": target, "tweens": [{"prop": "scale"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": "big"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 1e9}]}),
        json!({"target": target, "tweens": [{"prop": "position", "to": 5}]}),
        json!({"target": target, "tweens": [{"prop": "opacity", "effect": "opacity", "to": 500}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "speed": 2}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "duration": -1}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "duration": 0}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "duration": "long"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "duration": 1e12}]}),
        json!({"target": target, "tweens": many}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": "nowhere"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": "intro+=1"}], "labels": {"outro": 1}}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": "+=x"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": "<abc"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": [1]}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": "-=1"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": -2}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": 4.8}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "position": 1e300}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "wobbly"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "power2.sideways"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "steps(0)"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "steps(x)"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "spring(1,2,3,4)"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "spring(0.9"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "custom"}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": {"curve": [[0.6, 0.5], [0.3, 0.5]]}}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": {"spring": {"damping": 1}}}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": {"spring": {}, "steps": 2}}]}),
        json!({"target": target, "tweens": [{"prop": "scale", "to": 150, "ease": "easeOut(2)"}]}),
        json!({"target": target, "tweens": [ok_tween], "labels": {"1st": 1}}),
        json!({"target": target, "tweens": [ok_tween], "labels": {"<x": 1}}),
        json!({"target": target, "tweens": [ok_tween], "labels": {"a+=b": 1}}),
        json!({"target": target, "tweens": [ok_tween], "labels": {"late": -1}}),
        json!({"target": target, "tweens": [ok_tween], "labels": {"late": "soon"}}),
        json!({"target": target, "tweens": [ok_tween], "labels": (0..=64).map(|i| (format!("l{i}"), json!(1))).collect::<serde_json::Map<String, Value>>()}),
        json!({"target": target, "tweens": [ok_tween], "stagger": -0.1}),
        json!({"target": target, "tweens": [ok_tween], "stagger": {"each": 0.1, "from": "edges"}}),
        json!({"target": target, "tweens": [ok_tween], "stagger": {"each": 0.1, "from": 5}}),
        json!({"target": target, "tweens": [ok_tween], "stagger": {"from": "start"}}),
        json!({"target": {"clips": clips(&s)}, "tweens": [{"prop": "scale", "to": 150, "duration": 3.9}], "stagger": 0.1}),
        json!({"target": target, "tweens": [ok_tween], "defaults": {"duration": 0}}),
        json!({"target": target, "tweens": [ok_tween], "defaults": {"speed": 1}}),
        json!({"target": target, "tweens": [ok_tween], "dryRun": "yes"}),
        json!({"target": {"clips": (0..65).map(|_| c).collect::<Vec<_>>()}, "tweens": [ok_tween]}),
    ] {
        assert!(s.execute("keyframes.choreograph", p.clone()).is_err(), "{p}");
    }
    assert!(std::sync::Arc::ptr_eq(&s.project, &before));
    assert_eq!(s.history.undo.len(), undo);
}

/// A later tween on the same property replaces the keyframes inside its window; its start value is
/// what the parameter shows there.
#[test]
fn overlapping_tweens_on_one_property_take_over() {
    let mut s = demo();
    let c = clips(&s)[0];
    let r = s
        .execute(
            "keyframes.choreograph",
            json!({"target": {"clip": c}, "tweens": [
                {"prop": "rotation", "from": 0, "to": 100, "duration": 1, "ease": "none"},
                {"prop": "rotation", "to": -50, "duration": 1, "position": "<0.5", "ease": "none"}
            ]}),
        )
        .unwrap();
    let rot = param(&s, c, "motion", "rotation");
    let times: Vec<Tick> = rot.keyframes.iter().map(|k| k.time).collect();
    assert_eq!(times, [media(&s, c, 0.0), media(&s, c, 0.5), media(&s, c, 1.5)]);
    assert!((rot.f64_at(media(&s, c, 0.5)) - 50.0).abs() < 1e-6);
    assert_eq!(rot.f64_at(media(&s, c, 1.5)), -50.0);
    // the report has only the keyframes that are still there
    assert_eq!(r["written"], 3);
}

/// A reversed clip's keyframes run backwards: refused with a clear message.
#[test]
fn reversed_clips_are_refused() {
    let mut s = demo();
    let c = clips(&s)[0];
    s.execute("timeline.select", json!({"clips": [c]})).unwrap();
    s.execute("clip.speedDuration", json!({"speed": 100.0, "reverse": true, "ripple": false})).unwrap();
    assert!(s.active_sequence().unwrap().find_item(ClipId(c)).unwrap().1.reverse);
    let e = s.execute("keyframes.choreograph", json!({"target": {"clip": c}, "tweens": [{"prop": "scale", "to": 150}]})).unwrap_err();
    assert!(e.to_string().contains("reversed"), "{e}");
}
