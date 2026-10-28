//! The time between an Insta360 recording's motion sensor and its pictures, measured for each
//! recording: the camera levels and steadies its pictures by its sensor (VR Properties ▸ Level
//! Horizon, Steady), and its clock and the pictures' were found to differ by 0 to 160 ms from one
//! recording to the next — the exposure times the camera writes do not say by how much.
//!
//! | command | does |
//! |---|---|
//! | `media.syncGyro360` | for each Insta360 recording (`items`, default the selected 360° media), how far its motion sensor runs behind its pictures: the levelled pictures' own sway — their turning about the vertical a tenth of a second at a time (the proxy's pictures; the `seconds` of the recording, 20, where the sensor's steadying moves most), less its half-second smoothing — against the sensor's steadying turn (the horizon's `heading`, the sway's opposite) read at each lag within ±0.3 s (5 ms steps, refined to a fraction); a clear one (`match` ≥ 0.5) becomes the horizon's `lag`, one undo step (`apply: false` reports it only). `check: true` also measures the sway the steadied pictures keep, at no lag and at the found one |
//!
//! The turning is compared about the vertical only, where the sensor's sign was matched to the
//! pictures by measurement (`filmcraft_codecs`' horizon).

use filmcraft_media::Spherical;
use filmcraft_media::spherical::Horizon;
use filmcraft_project::{ItemId, ItemKind};
use filmcraft_render::vr_motion::{Faces, rotation_between};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p};
use crate::vr360::{STAB_FACE, analysis_picture, analysis_source, items_p, media_items};
use crate::{Result, Session};

const CMD: &str = "media.syncGyro360";
/// The lags looked at, seconds either way.
pub const SEARCH: f64 = 0.3;
/// Their step, seconds.
const STEP: f64 = 0.005;
/// The stretch measured by default, seconds.
const WINDOW: f64 = 20.0;
/// The agreement a lag must reach to be kept.
pub const MIN_MATCH: f64 = 0.5;
/// Most pictures measured.
const MAX_FRAMES: usize = 900;
/// Seconds between the pictures measured.
const MEASURE_STEP: f64 = 0.1;
/// The smoothing the sway is measured against, seconds (a Gaussian's σ; the horizon's own).
const SWAY_SECONDS: f64 = 0.5;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Sync 360° Motion Sensor",
        menu: &[],
        shortcut: None,
        params: r#"{"items":[id]?,"seconds":2..60=12,"apply":bool=true,"check":bool=false}"#,
        enabled: always,
        run: sync,
        journal: true,
    }]
}

/// `v` (samples at `times`) less its Gaussian smoothing of σ `sigma` seconds: what sways.
pub fn sway(times: &[f64], v: &[f64], sigma: f64) -> Vec<f64> {
    let reach = 3.0 * sigma;
    times
        .iter()
        .zip(v)
        .map(|(t, x)| {
            let (mut sum, mut wsum) = (0.0, 0.0);
            let lo = times.partition_point(|u| *u < t - reach);
            for (u, y) in times.iter().zip(v).skip(lo).take_while(|(u, _)| **u <= t + reach) {
                let w = (-((u - t) / sigma).powi(2) / 2.0).exp();
                sum += w * y;
                wsum += w;
            }
            if wsum > 0.0 { x - sum / wsum } else { 0.0 }
        })
        .collect()
}

/// The lag (seconds) at which the sensor's steadying `turn` (read at picture time + lag) best
/// opposes the pictures' `swaying` at `times`, and how well (the normalised cross-correlation's
/// opposite: 1 when it turns exactly back); `None` with too little to go by.
pub fn best_lag(times: &[f64], swaying: &[f64], turn: impl Fn(f64) -> Option<f64>) -> Option<(f64, f64)> {
    let ncc = |lag: f64| -> Option<f64> {
        let pairs: Vec<(f64, f64)> =
            times.iter().zip(swaying).filter_map(|(t, v)| Some((*v, turn(t + lag)?))).filter(|(v, g)| v.is_finite() && g.is_finite()).collect();
        if pairs.len() < 20 {
            return None;
        }
        let n = pairs.len() as f64;
        let (mx, my) = (pairs.iter().map(|p| p.0).sum::<f64>() / n, pairs.iter().map(|p| p.1).sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (x, y) in &pairs {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx) * (x - mx);
            syy += (y - my) * (y - my);
        }
        (sxx > 1e-12 && syy > 1e-12).then(|| -sxy / (sxx * syy).sqrt())
    };
    let steps = (SEARCH / STEP).round() as i64;
    let scores: Vec<(f64, f64)> = (-steps..=steps).filter_map(|k| ncc(k as f64 * STEP).map(|c| (k as f64 * STEP, c))).collect();
    let (i, &(lag, c)) = scores.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1))?;
    // the parabola through the peak and its neighbours
    let refined = match (i.checked_sub(1).and_then(|j| scores.get(j)), scores.get(i + 1)) {
        (Some(&(_, lo)), Some(&(_, hi))) => {
            let den = lo - 2.0 * c + hi;
            if den < -1e-12 { lag + (0.5 * (lo - hi) / den).clamp(-0.5, 0.5) * STEP } else { lag }
        }
        _ => lag,
    };
    Some((refined, c))
}

/// The `seconds`-long stretch (from 0 to `end`) where the sensor's steadying turn moves most,
/// leaving out turns it clips (quick turns let through).
fn busiest(h: &Horizon, seconds: f64, end: f64) -> (f64, f64) {
    let end = end.min(h.seconds());
    if !(end.is_finite() && end > seconds) {
        return (0.0, end.max(0.0));
    }
    let clip = f64::from(filmcraft_media::spherical::MAX_STEADY) - 0.5;
    let moves: Vec<f64> =
        h.heading.windows(2).map(|w| if w.iter().any(|v| f64::from(v.abs()) >= clip) { 0.0 } else { f64::from((w[1] - w[0]).abs()) }).collect();
    let per = |a: f64| -> f64 {
        let (i, j) = ((a / h.step) as usize, ((a + seconds) / h.step) as usize);
        moves.get(i..j.min(moves.len())).map_or(0.0, |m| m.iter().sum())
    };
    let mut best = (0.0, f64::NEG_INFINITY);
    let mut a = 0.0;
    while a + seconds <= end + 1e-9 {
        let m = per(a);
        if m > best.1 {
            best = (a, m);
        }
        a += 1.0;
    }
    (best.0, best.0 + seconds)
}

/// The pictures' turning about the vertical between each pair of frames at `times` (picture
/// seconds), as model `sph` maps them, degrees: (the later frame's time, its turn).
fn picture_turns(src: &filmcraft_media::SharedSource, sph: &Spherical, times: &[f64]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut prev: Option<Faces> = None;
    for &t in times {
        let mt = Tick::from_seconds_f64(t);
        let Some(img) = analysis_picture(src, mt) else {
            prev = None;
            continue;
        };
        let model = filmcraft_render::vr::SourceModel::at(sph, (img.w, img.h), mt);
        let faces = Faces::new(&img, &model, STAB_FACE);
        if let Some(a) = prev.as_ref()
            && let Some((dr, _)) = rotation_between(a, &faces)
        {
            let yaw = filmcraft_render::vr::rotation_angles(&dr).0;
            if yaw.is_finite() {
                out.push((t, yaw));
            }
        }
        prev = Some(faces);
    }
    out
}

/// The root mean square of `v`.
fn rms(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt() }
}

fn sync(s: &mut Session, p: &Value) -> Result<Value> {
    let items = media_items(s, &items_p(s, p), CMD)?;
    let seconds = match f64_p(p, "seconds") {
        None => WINDOW,
        Some(v) if v.is_finite() && (2.0..=60.0).contains(&v) => v,
        Some(_) => return Err(bad(CMD, "`seconds` is 2 to 60")),
    };
    let apply = bool_p(p, "apply").unwrap_or(true);
    let check = bool_p(p, "check").unwrap_or(false);
    let mut report = Vec::new();
    let mut found: Vec<(ItemId, f64)> = Vec::new();
    for id in items {
        let Some(sph) = s.project.vr_of(id).cloned() else {
            report.push(json!({"item": id.0, "synced": false, "why": "not 360° media"}));
            continue;
        };
        let Some(h) = sph.horizon.clone() else {
            report.push(json!({"item": id.0, "synced": false, "why": "no motion-sensor horizon (not an Insta360 recording)"}));
            continue;
        };
        let Some(src) = analysis_source(s, id) else {
            report.push(json!({"item": id.0, "synced": false, "why": "the media is offline"}));
            continue;
        };
        let info = src.info();
        let (a, b) = busiest(&h, seconds, info.duration.seconds());
        // a picture a step of the horizon (a tenth of a second): turns large enough to measure
        let n = (((b - a) / MEASURE_STEP).floor() as usize).min(MAX_FRAMES);
        let times: Vec<f64> = (0..n).map(|k| a + k as f64 * MEASURE_STEP).collect();
        // the pictures levelled by the sensor but not steadied: their heading, added up
        let levelled = Spherical { horizon: Some(Horizon { steady: false, lag: 0.0, ..h.clone() }), ..sph.clone() };
        let turns = picture_turns(&src, &levelled, &times);
        let at: Vec<f64> = turns.iter().map(|x| x.0).collect();
        let heading: Vec<f64> = turns
            .iter()
            .scan(0.0, |acc, x| {
                *acc += x.1;
                Some(*acc)
            })
            .collect();
        let swaying = sway(&at, &heading, SWAY_SECONDS);
        let unlagged = Horizon { lag: 0.0, ..h.clone() };
        let Some((lag, agree)) = best_lag(&at, &swaying, |t| unlagged.heading_at(t)) else {
            report.push(json!({"item": id.0, "synced": false, "why": "too few frames measured (a still camera, or a dark one)"}));
            continue;
        };
        let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
        let mut row = json!({"item": id.0, "lagMs": (lag * 1e4).round() / 10.0, "match": r3(agree), "frames": turns.len(), "from": r3(a), "to": r3(b), "before": (h.lag * 1e4).round() / 10.0});
        let clear = agree >= MIN_MATCH;
        row["synced"] = json!(clear && apply);
        if !clear {
            row["why"] = json!(format!("the pictures and the sensor agree too little ({agree:.2} < {MIN_MATCH})"));
        }
        if check {
            // the sway the steadied pictures keep (rms of their turning about the vertical a
            // frame, degrees), at no lag and at the found one
            let kept = |lag: f64| {
                let with = Spherical { horizon: Some(Horizon { lag, ..h.clone() }), ..sph.clone() };
                let t = picture_turns(&src, &with, &times);
                rms(&t.iter().map(|x| x.1).collect::<Vec<_>>())
            };
            row["swayNoLag"] = json!(r3(kept(0.0)));
            row["swayLagged"] = json!(r3(kept(lag)));
        }
        if clear && apply {
            found.push((id, lag));
        }
        report.push(row);
    }
    if !found.is_empty() {
        s.edit("Sync 360° Motion Sensor", |pr, _| {
            for (id, lag) in &found {
                if let Some(ItemKind::Media(m)) = pr.item_mut(*id).map(|i| &mut i.kind)
                    && let Some(h) = m.interpret.vr.as_mut().and_then(|v| v.horizon.as_mut())
                {
                    h.lag = *lag;
                }
            }
            Ok(())
        })?;
    }
    Ok(json!({"items": report}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lag_is_found_where_the_sensor_turns_back_the_sway() {
        // a walk's sway (1.7 a second, a slower drift under it); the sensor's steadying turn is
        // its opposite, its clock 0.12 s behind the pictures
        let lag = 0.12;
        let sway_at = |t: f64| 6.0 * (std::f64::consts::TAU * 1.7 * t).sin() + 2.0 * (std::f64::consts::TAU * 0.37 * t).sin();
        let turn = |s: f64| Some(-sway_at(s - lag));
        let times: Vec<f64> = (0..360).map(|k| 2.0 + k as f64 / 30.0).collect();
        // the pictures' sway with some measuring noise
        let seen: Vec<f64> = times.iter().enumerate().map(|(k, t)| sway_at(*t) + if k % 3 == 0 { 0.4 } else { -0.2 }).collect();
        let (found, agree) = best_lag(&times, &seen, turn).unwrap();
        assert!((found - lag).abs() < 0.006, "{found}");
        assert!(agree > 0.9, "{agree}");
        // what sways is what the smoothing leaves: a steady turn sways not at all
        let steady: Vec<f64> = times.iter().map(|t| 10.0 * t).collect();
        assert!(sway(&times, &steady, 0.5)[100..260].iter().all(|v| v.abs() < 0.01));
        // too little to go by
        assert!(best_lag(&times[..5], &seen[..5], turn).is_none());
        assert!(best_lag(&times, &seen, |_| None).is_none());
    }

    #[test]
    fn hostile_parameters_and_media_are_refused_or_reported() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().video_tracks[0].items[0].item;
        assert!(s.execute(CMD, json!({"items": [item.0], "seconds": 0.5})).is_err());
        assert!(s.execute(CMD, json!({"items": []})).is_err());
        // not 360°, then 360° without a sensor
        let r = s.execute(CMD, json!({"items": [item.0]})).unwrap();
        assert_eq!(r["items"][0]["synced"], false, "{r}");
        s.execute("media.vrProperties", json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
        let r = s.execute(CMD, json!({"items": [item.0]})).unwrap();
        assert!(r["items"][0]["why"].as_str().unwrap().contains("horizon"), "{r}");
    }
}
