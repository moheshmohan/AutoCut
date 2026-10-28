//! Aiming a 360° clip's camera by hand: a drag in the Program monitor turns the view, and while the
//! sequence plays every turn is recorded as the camera's yaw and pitch keyframes, smoothed when the
//! drag ends (the monitor's side: `filmcraft_ui_egui`, automation id `program.aim360`).
//!
//! | command | does |
//! |---|---|
//! | `clip.aim360` | the camera of a 360° clip (`clip`, default the selected one, else the topmost one under the first sample) aimed along `samples` — `[seconds, yaw, pitch]`, sequence seconds as the playhead had them — the yaw unwrapped (no spin the long way round past ±180°), smoothed over `smooth` seconds (0..2, 0.5; 0 while the drag goes on), then kept to the keyframes needed to stay within 0.25° of it: the yaw and pitch keyframes inside the samples' span replaced, those outside kept; one sample sets the view there (a keyframe where the camera is animated, else its value); `merge`: one undo step with the call before (a drag), `begin` starts a new one |

use serde_json::{Value, json};

use filmcraft_project::keyframe::Interpolation;
use filmcraft_project::{EffectInstance, Param, ParamValue};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq};
use crate::vr360::{clips_360, ensure_reframe, media_time};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.aim360";
/// Most samples one call takes (an hour at 30 a second).
const MAX_SAMPLES: usize = 108_000;
/// How far the keyframes may stray from the smoothed path, degrees.
pub const TOLERANCE: f64 = 0.25;
/// The smoothing a finished drag gets by default, seconds.
pub const SMOOTH: f64 = 0.5;
/// Most smoothing, seconds.
const MAX_SMOOTH: f64 = 2.0;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Aim 360° Camera",
        menu: &[],
        shortcut: None,
        params: r#"{"clip":id?,"samples":[[seconds,yaw,pitch]],"smooth":0..2=0.5,"merge":bool?,"begin":bool?}"#,
        enabled: has_seq,
        run: aim,
        journal: true,
    }]
}

/// One sample of the aim: sequence seconds, yaw, pitch (degrees).
pub type Sample = [f64; 3];

/// The `samples` parameter: finite, in time order (a later sample at the same time wins).
fn samples_p(p: &Value) -> Result<Vec<Sample>> {
    let a = p.get("samples").and_then(Value::as_array).ok_or_else(|| bad(CMD, "need `samples`: [[seconds, yaw, pitch], …]"))?;
    if a.is_empty() || a.len() > MAX_SAMPLES {
        return Err(bad(CMD, format!("`samples` holds 1 to {MAX_SAMPLES} samples")));
    }
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        let n: Vec<f64> = v.as_array().map(|x| x.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        match n.as_slice() {
            [t, y, pi] if t.is_finite() && *t >= 0.0 && y.is_finite() && y.abs() <= 1e6 && pi.is_finite() => out.push([*t, *y, pi.clamp(-90.0, 90.0)]),
            _ => return Err(bad(CMD, "each sample is [seconds ≥ 0, yaw, pitch], finite numbers")),
        }
    }
    // (stable: of samples at one time the last stays)
    out.sort_by(|a, b| a[0].total_cmp(&b[0]));
    out.dedup_by(|later, earlier| {
        let same = later[0] == earlier[0];
        if same {
            *earlier = *later;
        }
        same
    });
    Ok(out)
}

/// Each yaw moved by whole turns to the one nearest the yaw before it.
pub fn unwrapped(s: &mut [Sample]) {
    for k in 1..s.len() {
        let prev = s[k - 1][1];
        let y = s[k][1];
        s[k][1] = y - ((y - prev) / 360.0).round() * 360.0;
    }
}

/// Each sample's yaw and pitch the mean of those within `window` / 2 seconds of it (the window
/// shrinking evenly at the ends, so the path still starts and ends where the drag did).
pub fn smoothed(s: &[Sample], window: f64) -> Vec<Sample> {
    let half = window / 2.0;
    if half.is_nan() || half <= 0.0 || s.len() < 3 {
        return s.to_vec();
    }
    let (first, last) = (s[0][0], s[s.len() - 1][0]);
    let mut out = Vec::with_capacity(s.len());
    let (mut lo, mut hi) = (0usize, 0usize);
    let (mut sy, mut sp) = (0.0f64, 0.0f64);
    for x in s {
        let h = half.min(x[0] - first).min(last - x[0]).max(0.0);
        // the samples from t − h to t + h: [lo, hi)
        while hi < s.len() && s[hi][0] <= x[0] + h + 1e-9 {
            sy += s[hi][1];
            sp += s[hi][2];
            hi += 1;
        }
        while lo < hi && s[lo][0] < x[0] - h - 1e-9 {
            sy -= s[lo][1];
            sp -= s[lo][2];
            lo += 1;
        }
        let n = (hi - lo).max(1) as f64;
        out.push([x[0], sy / n, sp / n]);
    }
    out
}

/// The samples to keep so the straight lines between them stay within `tol` degrees of every
/// sample (Ramer–Douglas–Peucker, without recursion): their indices, in order, first and last in.
pub fn kept(s: &[Sample], tol: f64) -> Vec<usize> {
    if s.len() <= 2 {
        return (0..s.len()).collect();
    }
    let mut keep = vec![false; s.len()];
    keep[0] = true;
    keep[s.len() - 1] = true;
    let mut stack = vec![(0usize, s.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ta, tb) = (s[a][0], s[b][0]);
        let mut worst = (0.0f64, a);
        for k in a + 1..b {
            let f = if tb > ta { (s[k][0] - ta) / (tb - ta) } else { 0.5 };
            let d = (1..3).map(|c| (s[k][c] - (s[a][c] + (s[b][c] - s[a][c]) * f)).abs()).fold(0.0, f64::max);
            if d > worst.0 {
                worst = (d, k);
            }
        }
        if worst.0 > tol {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..s.len()).filter(|k| keep[*k]).collect()
}

/// Set `id` to `v` at `t` as a linear keyframe.
fn put(e: &mut EffectInstance, id: &str, t: Tick, v: f64) {
    if e.param(id).is_none() {
        e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
    }
    let Some(p) = e.param_mut(id) else { return };
    p.put_keyframe(t, ParamValue::Float(v));
    if let Some(k) = p.keyframes.iter_mut().find(|k| k.time == t) {
        k.interp = Interpolation::Linear;
    }
}

fn aim(s: &mut Session, p: &Value) -> Result<Value> {
    let mut samples = samples_p(p)?;
    let smooth = match f64_p(p, "smooth") {
        None => SMOOTH,
        Some(v) if v.is_finite() && (0.0..=MAX_SMOOTH).contains(&v) => v,
        Some(_) => return Err(bad(CMD, "`smooth` is 0 to 2 seconds")),
    };
    let first = Tick::from_seconds_f64(samples[0][0]);
    let named = p.get("clip").is_some_and(|v| !v.is_null()) || !s.state.selection.is_empty();
    let mut clips = clips_360(s, p, CMD)?;
    if !named || clips.len() > 1 {
        // the topmost under the first sample
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let track_of = |c| seq.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == c)).unwrap_or(0);
        clips.retain(|it| it.start <= first && first < it.end());
        clips.sort_by_key(|it| std::cmp::Reverse(track_of(it.id)));
    }
    let it = clips.first().cloned().ok_or_else(|| bad(CMD, "no 360° clip under the samples (or given)"))?;
    samples.retain(|x| {
        let t = Tick::from_seconds_f64(x[0]);
        it.start <= t && t < it.end()
    });
    if samples.is_empty() {
        return Err(bad(CMD, "the samples are all outside the clip"));
    }
    unwrapped(&mut samples);
    let path = smoothed(&samples, smooth);
    let keys: Vec<Sample> = kept(&path, TOLERANCE).into_iter().filter_map(|k| path.get(k).copied()).collect();
    let frame = s.active_sequence().map(|q| (q.settings.width, q.settings.height)).ok_or(EngineError::NoSequence)?;
    let merge = bool_p(p, "merge").unwrap_or(false).then(|| format!("aim360:{}", it.id.0));
    if bool_p(p, "begin").unwrap_or(false) {
        s.history.merge_key = None;
    }
    let clip = it.id;
    let n = keys.len();
    s.edit_sequence_as("Aim 360° Camera", merge.as_deref(), |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let at = |sec: f64, it: &filmcraft_project::TrackItem| media_time(it, (sec - it.start.seconds()).max(0.0));
        let times: Vec<Tick> = keys.iter().map(|k| at(k[0], it)).collect();
        let e = ensure_reframe(it, frame).ok_or_else(|| bad(CMD, "the clip has no virtual camera"))?;
        match (keys.as_slice(), times.as_slice()) {
            ([k], [t]) => {
                for (id, v) in [("yaw", k[1]), ("pitch", k[2])] {
                    match e.param_mut(id) {
                        Some(pp) if !pp.keyframes.is_empty() => put(e, id, *t, v),
                        Some(pp) => pp.value = ParamValue::Float(v),
                        None => {
                            e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
                        }
                    }
                }
            }
            _ => {
                let (a, b) = (times.first().copied().unwrap_or_default(), times.last().copied().unwrap_or_default());
                for id in ["yaw", "pitch"] {
                    if let Some(pp) = e.param_mut(id) {
                        pp.keyframes.retain(|k| k.time < a || k.time > b);
                    }
                }
                for (k, t) in keys.iter().zip(&times) {
                    put(e, "yaw", *t, k[1]);
                    put(e, "pitch", *t, k[2]);
                }
            }
        }
        Ok(())
    })?;
    let r = |v: f64| (v * 1000.0).round() / 1000.0;
    Ok(json!({
        "clip": clip.0,
        "samples": samples.len(),
        "keys": n,
        "from": r(samples[0][0]),
        "to": r(samples[samples.len() - 1][0]),
        "end": {"yaw": r(keys.last().map_or(0.0, |k| k[1])), "pitch": r(keys.last().map_or(0.0, |k| k[2]))},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_test_util::{make_movie, tmp_dir};
    use filmcraft_media::DemoScene;
    use filmcraft_project::vr::REFRAME;

    #[test]
    fn a_path_is_unwrapped_smoothed_and_kept_to_the_keys_it_needs() {
        // a turn right across ±180°: no spin the long way round
        let mut s: Vec<Sample> = (0..60).map(|k| [k as f64 / 30.0, (170.0 + k as f64) % 360.0, 0.0]).collect();
        for x in s.iter_mut() {
            if x[1] > 180.0 {
                x[1] -= 360.0;
            }
        }
        unwrapped(&mut s);
        assert!(s.windows(2).all(|w| (w[1][1] - w[0][1] - 1.0).abs() < 1e-9), "{:?}", &s[8..12]);
        // a straight turn needs its two ends only
        assert_eq!(kept(&s, TOLERANCE), vec![0, 59]);
        // a shaky hand is smoothed: the wiggle shrinks
        let shaky: Vec<Sample> = (0..90).map(|k| [k as f64 / 30.0, if k % 2 == 0 { 2.0 } else { -2.0 }, 0.0]).collect();
        let smooth = smoothed(&shaky, 0.5);
        assert!(smooth[30..60].iter().all(|x| x[1].abs() < 0.3), "{:?}", &smooth[30..34]);
        // the ends stay where the drag was
        assert_eq!(smooth[0][1], 2.0);
        assert_eq!(smoothed(&shaky, 0.0), shaky);
        // a corner is kept
        let corner: Vec<Sample> = (0..61).map(|k| [k as f64 / 30.0, if k <= 30 { k as f64 } else { 60.0 - k as f64 }, 0.0]).collect();
        assert_eq!(kept(&corner, TOLERANCE), vec![0, 30, 60]);
    }

    #[test]
    fn a_drag_becomes_the_cameras_keyframes_in_one_undo_step() {
        let dir = tmp_dir("aim360");
        let path = dir.join("pano.mp4");
        make_movie(&path, DemoScene::Aurora, 256, 128, 96);
        let mut s = Session::default();
        let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
        let item = r["items"][0].as_u64().unwrap();
        s.execute("media.vrProperties", json!({"items": [item], "projection": "equirectangular"})).unwrap();
        s.execute("file.newSequence", json!({"name": "aim", "width": 320, "height": 180, "fps": 24})).unwrap();
        let r = s.execute("timeline.place", json!({"item": item, "track": "V1", "seconds": 0})).unwrap();
        let clip = r["reframed360"][0].as_u64().unwrap();
        s.state.selection.clear();
        let n0 = s.history.undo.len();
        // a drag while playing: the samples grow, each call one step of the same drag
        let mut samples = Vec::new();
        for k in 0..48 {
            samples.push(json!([k as f64 / 24.0, k as f64 * 2.0, -(k as f64) * 0.5]));
            s.execute(CMD, json!({"samples": samples, "smooth": 0, "merge": true, "begin": k == 0})).unwrap();
        }
        let r = s.execute(CMD, json!({"samples": samples, "merge": true})).unwrap();
        assert_eq!(r["clip"], clip, "{r}");
        assert_eq!(s.history.undo.len(), n0 + 1, "one undo step");
        let seq = s.active_sequence().unwrap();
        let (_, it) = seq.find_item(filmcraft_project::ClipId(clip)).unwrap();
        let e = it.effect(REFRAME).unwrap();
        let keys = e.param("yaw").unwrap().keyframes.len();
        assert!((2..10).contains(&keys), "a steady turn needs few keys: {keys}");
        let mid = it.effect_time_at(Tick::from_seconds_f64(1.0));
        assert!((e.f64_at("yaw", mid) - 48.0).abs() < 3.0, "{}", e.f64_at("yaw", mid));
        assert!((e.f64_at("pitch", mid) + 12.0).abs() < 2.0, "{}", e.f64_at("pitch", mid));
        // paused: one sample sets the view there
        s.execute(CMD, json!({"samples": [[3.5, 10.0, 5.0]]})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        let seq = s.active_sequence().unwrap();
        let (_, it) = seq.find_item(filmcraft_project::ClipId(clip)).unwrap();
        assert!(it.effect(REFRAME).unwrap().param("yaw").unwrap().keyframes.is_empty());
        for bad in [
            json!({"samples": []}),
            json!({"samples": [[0, "x", 0]]}),
            json!({"samples": [[f64::MAX, 0, 0]]}),
            json!({"samples": [[0, 0, 0]], "smooth": 9}),
            json!({}),
        ] {
            assert!(s.execute(CMD, bad.clone()).is_err(), "{bad}");
        }
    }
}
