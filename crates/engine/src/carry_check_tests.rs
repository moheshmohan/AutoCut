//! `sequence.checkCarry` on stills made in code: a matched cut (the subject where it was), a cut
//! where the subject jumps across the frame, a cut into other colours, a dissolve, and hostile
//! parameters.
use filmcraft_time::FrameRate;
use serde_json::json;

use super::*;
use crate::Session;

/// A 192×108 still: a quiet room (a dark floor under a grey wall, `tint` added to its blue) with a
/// warm, bright subject at `x` of the width.
fn still(dir: &std::path::Path, name: &str, x: f64, tint: u8) -> String {
    let path = dir.join(name);
    let img = image::RgbImage::from_fn(192, 108, |px, py| {
        let (fx, fy) = (f64::from(px) / 192.0, f64::from(py) / 108.0);
        if (fx - x).abs() < 0.07 && (0.3..0.7).contains(&fy) {
            return image::Rgb([235, 190, 160]);
        }
        let stripe = if (px / 16) % 2 == 0 { 8 } else { 0 };
        let base: u8 = if fy > 0.65 { 50 } else { 110 };
        image::Rgb([base + stripe - tint / 3, base + stripe, (base + stripe).saturating_add(tint)])
    });
    img.save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

/// A 192×108 30 fps sequence of the stills, two seconds each: its session and id.
fn sequence(stills: &[(f64, u8)]) -> (Session, ItemId) {
    let dir = crate::media_test_util::tmp_dir("carry");
    let mut s = Session::default();
    let items: Vec<ItemId> = stills
        .iter()
        .enumerate()
        .map(|(k, (x, tint))| crate::commands::import_streamed(&mut s, &still(&dir, &format!("s{k}.png"), *x, *tint), None).unwrap())
        .collect();
    s.execute("file.newSequence", json!({"width": 192, "height": 108, "fps": 30, "video": 2, "audio": 1})).unwrap();
    let rate = FrameRate::FPS_30;
    for (k, item) in items.iter().enumerate() {
        s.execute(
            "timeline.place",
            json!({"item": item.0, "track": "V1", "time": rate.tick_of(60 * k as i64).0, "duration": rate.tick_of(60).0, "audio": false}),
        )
        .unwrap();
    }
    let seq = s.state.active_sequence.unwrap();
    (s, seq)
}

fn codes_at(r: &serde_json::Value, seconds: f64) -> Vec<String> {
    r["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| (f["seconds"].as_f64().unwrap() - seconds).abs() < 0.05)
        .map(|f| f["code"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_jumping_subject_is_flagged_and_a_matched_cut_is_not() {
    // the subject stays left (a matched cut), jumps right, then the room turns blue
    let (mut s, seq) = sequence(&[(0.25, 0), (0.27, 0), (0.75, 0), (0.75, 150)]);
    let r = s.execute(CMD, json!({"sequence": seq.0})).unwrap();
    assert_eq!(r["summary"]["cuts"], 3, "{r:#}");
    assert_eq!(r["blocking"], false);
    assert!(codes_at(&r, 2.0).is_empty(), "the matched cut carries: {:#}", r["findings"]);
    assert_eq!(codes_at(&r, 4.0), ["subjectJump"], "{:#}", r["findings"]);
    assert_eq!(codes_at(&r, 6.0), ["paletteJump"], "{:#}", r["findings"]);
    let cuts = r["cuts"].as_array().unwrap();
    let carry = |k: usize| cuts[k]["carry"].as_f64().unwrap();
    assert!(carry(0) > 0.8 && carry(1) < carry(0) && carry(2) < carry(0), "{cuts:#?}");
    let jump = &cuts[1]["subject"];
    assert!(jump["from"]["cx"].as_f64().unwrap() < 0.4 && jump["to"]["cx"].as_f64().unwrap() > 0.6, "{jump}");
    let lowest: Vec<f64> = r["summary"]["lowest"].as_array().unwrap().iter().map(|c| c["seconds"].as_f64().unwrap()).collect();
    assert_eq!(lowest.last(), Some(&2.0), "the matched cut carries best: {lowest:?}");
    let rendered = r["renderer"]["gpuFrames"].as_u64().unwrap() + r["renderer"]["cpuFrames"].as_u64().unwrap();
    assert_eq!(rendered, 6, "two frames a cut");
}

#[test]
fn a_cut_under_a_transition_is_scored_not_flagged() {
    let (mut s, seq) = sequence(&[(0.25, 0), (0.75, 0)]);
    let clip = s.project.sequence(seq).unwrap().video_tracks[0].items[1].id;
    s.execute("sequence.applyVideoTransition", json!({"clip": clip.0, "edge": "in", "effect": "cross_dissolve"})).unwrap();
    let r = s.execute(CMD, json!({"sequence": seq.0})).unwrap();
    assert_eq!(r["summary"]["cuts"], 1, "{r:#}");
    assert!(r["findings"].as_array().unwrap().is_empty(), "{:#}", r["findings"]);
    assert!(r["cuts"][0]["transition"].is_string(), "{:#}", r["cuts"]);
}

#[test]
fn hostile_parameters_are_refused_and_bounds_kept() {
    let (mut s, seq) = sequence(&[(0.25, 0), (0.75, 0), (0.25, 0)]);
    for p in [
        json!([1]),
        json!("V1"),
        json!({"sequence": "main"}),
        json!({"sequence": -1}),
        json!({"sequence": 987_654}),
        json!({"maxCuts": 0}),
        json!({"maxCuts": 401}),
        json!({"maxCuts": "all"}),
        json!({"cuts": [1]}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        assert!(matches!(r, Ok(Err(_))), "{p}: {r:?}");
    }
    // one cut read of two: the rest is said to be skipped
    let r = s.execute(CMD, json!({"sequence": seq.0, "maxCuts": 1})).unwrap();
    assert_eq!((r["summary"]["cuts"].as_u64(), r["summary"]["read"].as_u64()), (Some(2), Some(1)), "{r:#}");
    assert_eq!(r["skipped"].as_array().unwrap().len(), 1);
    // an empty sequence has no cuts to read; a sequence of one clip has none
    s.execute("file.newSequence", json!({"width": 64, "height": 36, "fps": 30})).unwrap();
    let empty = s.state.active_sequence.unwrap();
    assert!(s.execute(CMD, json!({"sequence": empty.0})).is_err());
    let (mut one, single) = sequence(&[(0.5, 0)]);
    let r = one.execute(CMD, json!({"sequence": single.0})).unwrap();
    assert_eq!((r["summary"]["cuts"].as_u64(), r["ok"].as_bool()), (Some(0), Some(true)));
    // the measures stay in range on odd input
    assert_eq!(palette_distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
    assert_eq!(palette_distance(&[], &[]), 0.0);
    assert!((0.0..=1.0).contains(&carry(Some(f64::INFINITY), f64::INFINITY)));
    assert_eq!(carry(None, 0.0), 1.0);
    let side = side_of(1, 1, &[0, 0, 0]);
    assert!(side.subject.is_none());
}
