//! `transitions.applyToAll`, `transitions.randomOnBeats` and the Creator presets' paired sounds,
//! on the demo project.

use filmcraft_project::{SoundKind, Transition};
use serde_json::{Value, json};

use crate::Session;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.state.selection.clear();
    s
}

fn v1_transitions(s: &Session) -> Vec<Transition> {
    s.active_sequence().unwrap().video_tracks[0].transitions.clone()
}

/// The transitions of V1 that join two clips (the demo also fades from and to nothing).
fn v1_cut_transitions(s: &Session) -> Vec<Transition> {
    v1_transitions(s).into_iter().filter(|x| x.from.is_some() && x.to.is_some()).collect()
}

/// The cuts of V1 (seconds).
fn v1_cuts(s: &Session) -> Vec<f64> {
    let items = &s.active_sequence().unwrap().video_tracks[0].items;
    items.windows(2).filter(|w| w[0].end() == w[1].start).map(|w| w[1].start.seconds()).collect()
}

#[test]
fn apply_to_all_puts_the_preset_on_every_cut_as_one_step() {
    let mut s = demo();
    let cuts = v1_cuts(&s);
    assert!(cuts.len() >= 2, "{cuts:?}");
    let before = v1_transitions(&s);
    let r = s.execute("transitions.applyToAll", json!({"preset": "Pull In", "track": "V1", "sound": true})).unwrap();
    let placed = r["transitions"].as_array().unwrap();
    assert_eq!(placed.len(), cuts.len(), "{r}");
    let rate = s.active_sequence().unwrap().settings.frame_rate;
    let frame = rate.frame_duration();
    for cut in &cuts {
        let t = filmcraft_time::Tick::from_seconds_f64(*cut);
        let x = v1_transitions(&s).into_iter().find(|x| x.range().contains(t)).expect("a transition on the cut");
        assert_eq!(x.effect.effect, "crt_pull_in");
        // centred, 0.4 s on whole frames
        assert_eq!(x.duration.0 % frame.0, 0);
        assert!((x.duration.seconds() - 0.4).abs() <= frame.seconds(), "{:?}", x.duration);
        assert!(((x.start + x.duration.mul_ratio(1, 2)).seconds() - cut).abs() <= frame.seconds());
        assert_eq!(x.sound.as_ref().map(|snd| snd.kind), Some(SoundKind::Whoosh), "the paired whoosh");
        assert_eq!(x.effect.param("zoom").and_then(|p| p.value.as_f64()), Some(90.0), "the tuned defaults");
    }
    // one undo step back to what was there
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(v1_transitions(&s), before);
    // replace: false keeps the cuts that have one
    s.execute("transitions.applyToAll", json!({"preset": "crt_white_flash", "duration": 0.3})).unwrap();
    let r = s.execute("transitions.applyToAll", json!({"preset": "Mix", "replace": false}));
    assert!(r.is_err(), "every cut already has one: nothing to do");
    assert!(v1_cut_transitions(&s).iter().all(|x| x.effect.effect == "crt_white_flash"));
    // bad calls
    for p in [
        json!({}),
        json!({"preset": "Warp Drive"}),
        json!({"preset": "Mix", "duration": -1}),
        json!({"preset": "Mix", "duration": 1e9}),
        json!({"preset": "Mix", "track": "V99"}),
    ] {
        assert!(s.execute("transitions.applyToAll", p.clone()).is_err(), "{p}");
    }
}

#[test]
fn random_on_beats_is_seeded_and_lands_on_strong_beats() {
    let mut s = demo();
    let cuts = v1_cuts(&s);
    // a beat on every cut, and beats between them
    let mut beats: Vec<f64> = cuts.iter().flat_map(|c| [*c, c + 0.5]).collect();
    beats.sort_by(f64::total_cmp);
    let call = |seed: u64| json!({"folder": "Creator", "seed": seed, "everyNthBeat": 1, "beats": beats, "bpm": 120, "track": "V1"});
    let first = s.execute("transitions.randomOnBeats", call(7)).unwrap();
    let ids = |r: &Value| r["transitions"].as_array().unwrap().iter().map(|t| t["effect"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    let a = ids(&first);
    assert_eq!(a.len(), cuts.len(), "{first}");
    assert!(a.iter().all(|id| id.starts_with("crt_")), "{a:?}");
    assert!(a.windows(2).all(|w| w[0] != w[1]), "never the same twice in a row: {a:?}");
    for t in first["transitions"].as_array().unwrap() {
        assert_eq!(t["cut"], t["beat"], "on the beat");
        // never longer than 90 % of a beat (0.5 s)
        assert!(t["duration"].as_i64().unwrap() as f64 / filmcraft_time::TICKS_PER_SECOND as f64 <= 0.45 + 1e-9);
    }
    s.execute("edit.undo", json!({})).unwrap();
    let again = s.execute("transitions.randomOnBeats", call(7)).unwrap();
    assert_eq!(ids(&again), a, "the same seed, the same transitions");
    s.execute("edit.undo", json!({})).unwrap();
    let other: Vec<Vec<String>> = (8..12)
        .map(|seed| {
            let r = s.execute("transitions.randomOnBeats", call(seed)).unwrap();
            s.execute("edit.undo", json!({})).unwrap();
            ids(&r)
        })
        .collect();
    assert!(other.iter().any(|o| *o != a), "other seeds draw other transitions");
    // strong beats only: with every other beat, the in-between beats do not count
    let offset: Vec<f64> = cuts.iter().flat_map(|c| [c - 0.5, *c]).collect();
    let r = s.execute("transitions.randomOnBeats", json!({"seed": 1, "everyNthBeat": 2, "beats": offset, "track": "V1"}));
    assert!(r.is_err(), "the cuts sit on the weak beats: {r:?}");
    // another folder, and refusals
    s.execute("transitions.randomOnBeats", json!({"folder": "Light & Film", "everyNthBeat": 1, "beats": beats, "track": "V1"})).unwrap();
    assert!(
        v1_cut_transitions(&s)
            .iter()
            .all(|x| filmcraft_project::find_effect(&x.effect.effect).is_some_and(|d| d.category == ["Video Transitions", "Light & Film"]))
    );
    for p in [
        json!({"folder": "Nope", "beats": beats}),
        json!({"seed": -3, "beats": beats}),
        json!({"everyNthBeat": 0, "beats": beats}),
        json!({"tolerance": 9, "beats": beats}),
        json!({"beats": [1.0]}),
    ] {
        assert!(s.execute("transitions.randomOnBeats", p.clone()).is_err(), "{p}");
    }
}

#[test]
fn add_sound_pairs_creator_presets_by_default() {
    let mut s = demo();
    s.execute("transitions.applyToAll", json!({"preset": "White Flash", "track": "V1"})).unwrap();
    let ids: Vec<u64> = v1_cut_transitions(&s).iter().map(|x| x.id.0).collect();
    assert!(v1_cut_transitions(&s).iter().all(|x| x.sound.is_none()));
    s.execute("transitions.addSound", json!({"transitions": ids})).unwrap();
    assert!(v1_cut_transitions(&s).iter().all(|x| x.sound.as_ref().map(|snd| snd.kind) == Some(SoundKind::Impact)), "White Flash pairs with an impact");
    // any other transition takes a whoosh
    let fades: Vec<u64> = v1_transitions(&s).iter().filter(|x| x.from.is_none() || x.to.is_none()).map(|x| x.id.0).collect();
    if !fades.is_empty() {
        s.execute("transitions.addSound", json!({"transitions": fades})).unwrap();
        assert!(v1_transitions(&s).iter().filter(|x| fades.contains(&x.id.0)).all(|x| x.sound.as_ref().map(|snd| snd.kind) == Some(SoundKind::Whoosh)));
    }
    s.execute("transitions.addSound", json!({"transitions": ids, "kind": "riser"})).unwrap();
    assert!(v1_cut_transitions(&s).iter().all(|x| x.sound.as_ref().map(|snd| snd.kind) == Some(SoundKind::Riser)), "an explicit kind wins");
}
