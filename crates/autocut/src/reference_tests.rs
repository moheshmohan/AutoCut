//! A reference's fingerprint read into targets (hostile ones refused), the distance between two
//! rhythms, and Quick Short's template choice cut like a fast or a slow reference.

use super::*;
use crate::quick::{Facts, Place, SlotInfo, TemplateInfo, choose_template_like};

/// A fingerprint of `seconds` with shots of `median` seconds (its quarter points at ±30 %),
/// `cpm` cuts a minute, a flat pace and a burst at a third of the way in.
fn fingerprint(seconds: f64, median: f64, cpm: f64) -> Value {
    json!({
        "version": filmcraft_engine::style_measure::FINGERPRINT_VERSION,
        "seconds": seconds,
        "cutsPerMinute": cpm,
        "shot": {"median": median, "p25": median * 0.7, "p75": median * 1.3, "cv": 0.4},
        "bursts": {"perMinute": 1.0, "share": 0.1, "at": [[0.3, 0.4, 4]]},
        "pace": vec![median; 16],
        "energy": vec![0.4; 16],
        "loudness": {"integrated": -14.0, "range": 5.0, "curve": vec![0.0; 16]},
        "stillness": 0.0,
        "key": "0000000000000000",
    })
}

#[test]
fn targets_come_from_a_fingerprint() {
    let t = Targets::from_fingerprint(&fingerprint(30.0, 2.0, 30.0)).unwrap();
    assert_eq!((t.seconds, t.median, t.cuts_per_minute), (30.0, 2.0, 30.0));
    assert_eq!(t.pace.len(), POINTS);
    assert_eq!(t.bursts, vec![(0.3, 0.4, 4)]);
    // inside the burst a shot is the burst's own length (3 s over 4 shots), else the pace
    assert!((t.shot_at(0.35) - 0.75).abs() < 1e-9, "{}", t.shot_at(0.35));
    assert!((t.shot_at(0.8) - 2.0).abs() < 1e-9);
    assert!((t.shot_at(f64::NAN) - 2.0).abs() < 1e-9, "a hostile position reads as the start");
    // a tempo whose beats fit the quicker shots, inside the director's range
    for (median, lo, hi) in [(0.3, 140.0, 150.0), (2.0, 70.0, 150.0), (30.0, 70.0, 150.0)] {
        let b = Targets::from_fingerprint(&fingerprint(30.0, median, 10.0)).unwrap().bpm();
        assert!((lo..=hi).contains(&b), "{median} s shots: {b} bpm");
    }
    // a pace with holes is filled; shots outside the range are clamped
    let mut fp = fingerprint(20.0, 1.0, 60.0);
    fp["pace"] = json!([null, 1.0, null, 100_000.0]);
    let t = Targets::from_fingerprint(&fp).unwrap();
    assert!(t.pace.iter().all(|v| v.is_finite()));
    assert!(t.shot_at(0.9) <= SHOT_RANGE.1);
    assert!(t.json()["steers"].is_array());
}

#[test]
fn hostile_fingerprints_are_refused() {
    let good = fingerprint(30.0, 2.0, 30.0);
    let with = |k: &str, v: Value| {
        let mut f = good.clone();
        f[k] = v;
        f
    };
    let mut no_median = good.clone();
    no_median["shot"]["median"] = Value::Null;
    let mut bad_bursts = good.clone();
    bad_bursts["bursts"]["at"] = json!("everywhere");
    for fp in [
        json!(null),
        json!(5),
        json!("fingerprint"),
        json!([1, 2]),
        with("version", json!(99)),
        with("version", Value::Null),
        with("seconds", json!(-1)),
        with("seconds", json!(1e12)),
        with("cutsPerMinute", json!("fast")),
        with("pace", json!(vec![1.0; 65])),
        with("pace", json!({"a": 1})),
        with("energy", json!("high")),
        with("loudness", json!(3)),
        with("padding", json!("x".repeat(MAX_BYTES))),
        no_median,
        bad_bursts,
    ] {
        assert!(Targets::from_fingerprint(&fp).is_err(), "{}", fp.to_string().chars().take(120).collect::<String>());
    }
    // bursts that make no sense are left out, not trusted
    let mut odd = good.clone();
    odd["bursts"]["at"] = json!([[0.5, 0.2, 3], [0.1, 0.2, 0], [0.1, 9.0, 3], ["a", 1, 2], [0.1, 0.2, 3]]);
    assert_eq!(Targets::from_fingerprint(&odd).unwrap().bursts, vec![(0.1, 0.2, 3)]);
    // what a reference may be
    assert!(measure_params(&json!(3)).unwrap().is_some());
    assert!(measure_params(&json!("liked.mp4")).unwrap().is_some());
    assert!(measure_params(&json!({"fingerprint": good})).unwrap().is_none());
    for v in [json!(-3), json!(1.5), json!(""), json!(true), json!([1])] {
        assert!(measure_params(&v).is_err(), "{v}");
    }
}

#[test]
fn the_distance_is_zero_for_the_same_rhythm_and_grows_apart() {
    let target = fingerprint(30.0, 2.0, 30.0);
    let same = distance(&target, &target);
    assert_eq!(same["overall"], 0.0, "{same}");
    let near = distance(&target, &fingerprint(30.0, 1.8, 33.0));
    let far = distance(&target, &fingerprint(30.0, 0.5, 120.0));
    let o = |v: &Value| v["overall"].as_f64().unwrap();
    assert!(o(&near) > 0.0 && o(&near) < o(&far), "{near} {far}");
    assert_eq!(far["parts"]["shotLength"], 1.0, "four times as fast is as far as it goes");
    // a rhythm with no curves still compares its lengths and rates
    let rhythm = filmcraft_engine::style_measure::rhythm_of(&[(0.0, 2.0), (2.0, 2.0), (4.0, 2.0)], 6.0);
    let d = distance(&target, &rhythm);
    assert!(d["parts"]["shotLength"].as_f64().unwrap() < 0.01 && d["parts"]["energy"].is_null(), "{d}");
    // nothing to compare
    assert!(distance(&json!({}), &json!({}))["overall"].is_null());
}

/// A template of `n` picture placements each `secs` long, cut on a beat of 120.
fn template(id: &str, n: usize, secs: f64) -> TemplateInfo {
    TemplateInfo {
        id: id.into(),
        name: id.into(),
        description: String::new(),
        category: String::new(),
        tags: Vec::new(),
        seconds: n as f64 * secs,
        aspects: vec!["9:16".into()],
        ai_ready: true,
        captions: false,
        bpm: Some(120.0),
        slots: vec![SlotInfo {
            id: "shots".into(),
            kind: "video".into(),
            name: "Shots".into(),
            prompt: String::new(),
            count: n,
            min_count: 1,
            max_chars: None,
            ai: None,
            placements: (0..n).map(|m| Place { media: m, start: Some(m as f64 * secs), seconds: secs }).collect(),
        }],
    }
}

#[test]
fn quick_short_picks_the_template_that_cuts_like_the_reference() {
    let templates = [template("builtin:quick", 24, 0.6), template("builtin:slow", 6, 3.0)];
    let facts = Facts { items: 8, videos: 8, shots: 30, moments: 30, ..Facts::default() };
    let fast = Targets::from_fingerprint(&fingerprint(15.0, 0.6, 100.0)).unwrap();
    let slow = Targets::from_fingerprint(&fingerprint(18.0, 3.0, 20.0)).unwrap();
    let pick = |t: &Targets, secs: f64| choose_template_like(&templates, &facts, "9:16", secs, Some(t)).unwrap();
    let a = pick(&fast, 15.0);
    let b = pick(&slow, 18.0);
    assert_eq!(a.chosen.id, "builtin:quick", "{:?}", a);
    assert_eq!(b.chosen.id, "builtin:slow", "{:?}", b);
    assert!(a.chosen.reasons.iter().any(|r| r.starts_with("cuts like the reference")), "{:?}", a.chosen.reasons);
    // without a reference the choice is the one of before
    let plain = crate::quick::choose_template(&templates, &facts, "9:16", 15.0).unwrap();
    let none = choose_template_like(&templates, &facts, "9:16", 15.0, None).unwrap();
    assert_eq!(plain, none);
    assert!(plain.chosen.reasons.iter().all(|r| !r.contains("reference")));
}
