//! The same moment filmed twice: a phone and a 360° camera (or any two) recording one scene, found
//! by when they were filmed and lined up by their sound, so an edit can cut between the two angles
//! of one moment.
//!
//! | command | does |
//! |---|---|
//! | `media.sameMoments` | pairs of `items` (default: every video with sound) filmed at the same time — their capture dates within `slack` seconds (600) of overlapping; without dates (or with `dates: false`, the clocks being wrong), every pair of a dozen items or fewer — lined up by their sound (GCC-PHAT, `filmcraft_audio_dsp::sync`, the search kept within the dates' difference and the slack): for each pair `a`, `b` and the `offset` (seconds) at which `b`'s media time `t` is `a`'s `t + offset`, how long they overlap, and how sure the match is (`confidence`, `distinct`); only matches clear enough to trust |
//!
//! Nothing is changed: an edit uses what it says (the director's two-angle cuts).

use std::collections::HashMap;

use filmcraft_audio_dsp::sync::{SyncOptions, find_offset};
use filmcraft_project::{ItemId, MediaRef};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, f64_p};
use crate::{Result, Session};

const CMD: &str = "media.sameMoments";
/// How far apart two cameras' clocks may be, seconds, by default.
const SLACK: f64 = 600.0;
/// Without capture dates, every pair is tried when there are at most this many items.
const MAX_UNDATED: usize = 12;
/// Most items one call looks at.
const MAX_ITEMS: usize = 200;
/// Most pairs one call lines up (each is a correlation of two recordings' sound).
const MAX_PAIRS: usize = 400;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Find the Same Moments",
        menu: &[],
        shortcut: None,
        params: r#"{"items":[id]?,"slack":seconds=600,"dates":bool=true}"#,
        enabled: always,
        run: same_moments,
        journal: false,
    }]
}

/// One video with sound: its item, length (seconds) and when it started being filmed (Unix
/// seconds), if known.
struct Take {
    item: ItemId,
    seconds: f64,
    filmed: Option<f64>,
}

fn takes(s: &Session, p: &Value) -> Result<Vec<Take>> {
    let ids: Vec<ItemId> = match p.get("items").and_then(Value::as_array) {
        Some(a) => a.iter().filter_map(Value::as_u64).map(ItemId).take(MAX_ITEMS).collect(),
        None => s.project.items.keys().copied().take(MAX_ITEMS * 4).collect(),
    };
    let mut out = Vec::new();
    for id in ids {
        let Some((root, m, _)) = s.project.resolve_media(id) else { continue };
        if !m.info.has_video() || !m.info.has_audio() || out.iter().any(|t: &Take| t.item == root) {
            continue;
        }
        let filmed = match &m.media {
            MediaRef::File { path } => crate::media_date::file_capture_date(path).map(|(t, _)| t as f64),
            _ => None,
        };
        out.push(Take { item: root, seconds: m.duration().seconds(), filmed });
        if out.len() >= MAX_ITEMS {
            break;
        }
    }
    Ok(out)
}

fn same_moments(s: &mut Session, p: &Value) -> Result<Value> {
    let slack = match f64_p(p, "slack") {
        None => SLACK,
        Some(v) if v.is_finite() && (0.0..=86_400.0).contains(&v) => v,
        Some(_) => return Err(bad(CMD, "`slack` is 0 to 86400 seconds")),
    };
    let takes = takes(s, p)?;
    let few = takes.len() <= MAX_UNDATED;
    // (`dates: false`: the cameras' clocks are wrong; a few items are tried pair by pair)
    let dated = crate::commands::bool_p(p, "dates").unwrap_or(true);
    // the pairs that may overlap: by their dates, else (a few items) all
    let mut pairs: Vec<(usize, usize, Option<f64>)> = Vec::new();
    for i in 0..takes.len() {
        for j in i + 1..takes.len() {
            let (a, b) = (&takes[i], &takes[j]);
            match (a.filmed, b.filmed) {
                (Some(fa), Some(fb)) if dated => {
                    if fa - slack < fb + b.seconds && fb - slack < fa + a.seconds {
                        pairs.push((i, j, Some(fb - fa)));
                    }
                }
                _ if few => pairs.push((i, j, None)),
                _ => {}
            }
        }
    }
    pairs.truncate(MAX_PAIRS);
    let mut audio: HashMap<ItemId, Option<Vec<f32>>> = HashMap::new();
    let rate = crate::sync::ANALYSIS_RATE;
    let mut found = Vec::new();
    for (i, j, apart) in pairs {
        let (a, b) = (&takes[i], &takes[j]);
        for t in [a, b] {
            audio.entry(t.item).or_insert_with(|| crate::sync::project_clip(&s.project, t.item).and_then(|c| crate::sync::clip_audio(s, &c)));
        }
        let (Some(Some(sa)), Some(Some(sb))) = (audio.get(&a.item), audio.get(&b.item)) else { continue };
        // b started `apart` seconds after a by their clocks: the search kept within the slack of it
        let reach = apart.map_or(a.seconds.max(b.seconds), |d| d.abs() + slack);
        let opts = SyncOptions { max_lag: Some((reach * f64::from(rate)).min(i64::MAX as f64 / 2.0) as i64), ..SyncOptions::default() };
        let Some(r) = find_offset(sa, sb, rate, &opts) else { continue };
        let offset = r.lag_fine / f64::from(rate);
        if !r.reliable() || apart.is_some_and(|d| (offset - d).abs() > slack) {
            continue;
        }
        // b's media time t is a's t + offset: where they overlap
        let (from, to) = (offset.max(0.0), (offset + b.seconds).min(a.seconds));
        if to - from < 1.0 {
            continue;
        }
        let r2 = |x: f64| (x * 1000.0).round() / 1000.0;
        found.push(json!({
            "a": a.item.0,
            "b": b.item.0,
            "offset": r2(offset),
            "overlap": r2(to - from),
            "confidence": (f64::from(r.confidence) * 10.0).round() / 10.0,
            "distinct": (f64::from(r.distinct) * 100.0).round() / 100.0,
            "inverted": r.inverted,
        }));
    }
    Ok(json!({"items": takes.len(), "pairs": found}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_test_util::{make_movie, tmp_dir};
    use filmcraft_media::DemoScene;

    #[test]
    fn two_takes_of_one_moment_are_lined_up_by_their_sound() {
        let dir = tmp_dir("same-moments");
        let (a, b, c) = (dir.join("phone.mp4"), dir.join("cam.mp4"), dir.join("other.mp4"));
        make_movie(&a, DemoScene::Aurora, 160, 90, 96);
        make_movie(&b, DemoScene::Aurora, 160, 90, 96);
        make_movie(&c, DemoScene::Plasma, 160, 90, 96);
        let mut s = Session::default();
        let r = s.execute("file.import", json!({"paths": [a.to_string_lossy(), b.to_string_lossy(), c.to_string_lossy()]})).unwrap();
        let ids: Vec<u64> = r["items"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
        let r = s.execute(CMD, json!({"items": [ids[0], ids[1], ids[2]]})).unwrap();
        assert_eq!(r["items"], 3, "{r}");
        let pairs = r["pairs"].as_array().unwrap();
        let ab = pairs.iter().find(|p| p["a"] == ids[0] && p["b"] == ids[1]).unwrap_or_else(|| panic!("{r}"));
        assert!(ab["offset"].as_f64().unwrap().abs() < 0.01, "{ab}");
        assert!(ab["overlap"].as_f64().unwrap() > 3.5);
        for bad in [json!({"slack": -1}), json!({"slack": 1e9})] {
            assert!(s.execute(CMD, bad.clone()).is_err(), "{bad}");
        }
        // nothing with sound: no pairs
        assert_eq!(s.execute(CMD, json!({"items": []})).unwrap()["pairs"], json!([]));
    }
}
