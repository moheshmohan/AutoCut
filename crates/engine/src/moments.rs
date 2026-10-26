//! How each moment of a clip looks, measured from its own frames (`media.measureMoments`): the
//! director ranks the moments of a long take, which the scene index scores as one shot.
//!
//! For each `[from, to]` (media seconds): sharpness and exposure averaged over `frames` pictures
//! spread across it (`filmcraft_scenes::quality`, as the scene index measures a shot); the
//! camera's move and shake from a burst of six small frames a tenth of a second apart at its
//! middle; and — when the face detector is downloaded — the faces in its middle picture (how many,
//! the largest's height). 360° media is measured as its stitched, levelled panorama, as it is
//! indexed.

use filmcraft_project::scenes::ShotQuality;
use filmcraft_scenes::quality;
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, item_p, u64_p};
use crate::{Result, Session};

pub const MEASURE_CMD: &str = "media.measureMoments";
/// Most moments measured in one call.
pub const MAX_MOMENTS: usize = 64;
/// Frames of the burst the camera's move is measured on, and the seconds between them.
const BURST: usize = 6;
const BURST_STEP: f64 = 0.1;
/// A face counts from this detector score.
const FACE_SCORE: f32 = 0.6;
/// Longest side the face detector looks at.
const FACE_SIDE: usize = 640;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: MEASURE_CMD,
        label: "Measure Moments",
        menu: &[],
        shortcut: None,
        params: r#"{"item":id,"moments":[[from,to]] (media seconds, at most 64),"frames":1..8=3,"faces":bool=true}"#,
        enabled: |_| Ok(()),
        run: measure,
        journal: false,
    }]
}

/// One moment's `[from, to]`, checked against the media's length `dur`.
fn moment(v: &Value, dur: f64) -> Option<(f64, f64)> {
    let a = v.as_array().filter(|a| a.len() == 2)?;
    let (x, y) = (a.first()?.as_f64()?, a.get(1)?.as_f64()?);
    (x.is_finite() && y.is_finite() && x >= 0.0 && y >= x).then(|| (x.min(dur), y.min(dur)))
}

fn measure(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = MEASURE_CMD;
    let item = item_p(p, "item").ok_or_else(|| bad(cmd, "need `item`"))?;
    let src = crate::vr360::analysis_view(s, item).ok_or_else(|| bad(cmd, "the item's media is offline or unreadable"))?;
    let v = src.info().video.clone().ok_or_else(|| bad(cmd, "the item has no video"))?;
    let dur = src.info().duration.seconds().max(0.0);
    let list = p.get("moments").and_then(Value::as_array).ok_or_else(|| bad(cmd, "`moments` is a list of [from, to] media seconds"))?;
    if list.is_empty() || list.len() > MAX_MOMENTS {
        return Err(bad(cmd, format!("`moments` holds 1 to {MAX_MOMENTS} moments")));
    }
    let moments: Vec<(f64, f64)> =
        list.iter().map(|m| moment(m, dur).ok_or_else(|| bad(cmd, "each moment is [from, to], media seconds, from ≤ to"))).collect::<Result<_>>()?;
    let frames = u64_p(p, "frames").unwrap_or(3).clamp(1, 8) as usize;
    // (no face detector downloaded: the faces are not counted, and the answer says so)
    let faces = if bool_p(p, "faces").unwrap_or(true) { crate::privacy_check::face_detector(s, cmd, Some(false)).ok() } else { None };
    let fd = v.frame_rate.sane().frame_duration().seconds().max(1e-3);
    let last = (dur - fd).max(0.0);
    let long = f64::from(v.width.max(v.height).max(1));
    let scale = |side: usize| (side as f64 / long).clamp(1.0 / 64.0, 1.0) as f32;
    let picture = |t: f64, side: usize| -> Option<(Vec<u8>, u32, u32)> {
        let f = src.video_frame(filmcraft_media::FrameRequest { time: Tick::from_seconds_f64(t.clamp(0.0, last)), scale: scale(side) }).ok()?;
        Some(crate::scenes::small_rgba(&f, side as u32))
    };
    let mut out = Vec::with_capacity(moments.len());
    for (a, b) in moments {
        let mut q = ShotQuality::default();
        let mut n = 0usize;
        let mut middle: Option<(Vec<u8>, u32, u32)> = None;
        for k in 0..frames {
            let t = a + (b - a) * (k as f64 + 0.5) / frames as f64;
            let Some((rgba, w, h)) = picture(t, quality::MEASURE_SIDE) else { continue };
            quality::measure_keyframe(&mut q, &rgba, w as usize, h as usize, n);
            n += 1;
            if k == frames / 2 {
                middle = Some((rgba, w, h));
            }
        }
        if n == 0 {
            out.push(json!({"from": a, "to": b, "measured": false}));
            continue;
        }
        // the camera: a burst at the middle
        let first = ((a + b) / 2.0 - BURST_STEP * (BURST as f64 - 1.0) / 2.0).max(0.0);
        let lumas: Vec<quality::Luma> = (0..BURST)
            .filter_map(|k| {
                let (rgba, w, h) = picture(first + BURST_STEP * k as f64, quality::MOTION_SIDE)?;
                quality::luma(&rgba, w as usize, h as usize, quality::MOTION_SIDE)
            })
            .collect();
        let shifts: Vec<(f32, f32)> = lumas.windows(2).filter_map(|w| quality::global_shift(w.first()?, w.get(1)?, 4)).collect();
        let width = lumas.first().map_or(quality::MOTION_SIDE, |l| l.w.max(1));
        let (camera, motion, shake) = quality::camera_motion(&shifts, 1.0 / BURST_STEP, width);
        let mut m = json!({
            "from": a, "to": b, "measured": true,
            "sharpness": q.sharpness, "exposure": q.exposure, "brightness": q.brightness, "dark": q.dark, "blown": q.blown,
            "motion": motion, "shake": shake, "camera": camera,
        });
        if let (Some(detect), Some((rgba, w, h))) = (&faces, &middle) {
            let found: Vec<f32> = detect(rgba, *w as usize, *h as usize, FACE_SIDE)
                .iter()
                .filter(|d| d.kind == filmcraft_vision::Kind::Face && d.score >= FACE_SCORE)
                .map(|d| d.bbox[3])
                .filter(|h| h.is_finite())
                .collect();
            m["faces"] = json!(found.len());
            m["largestFace"] = json!(found.iter().copied().fold(0.0f32, f32::max));
        }
        out.push(m);
    }
    Ok(json!({"item": item.0, "moments": out, "facesCounted": faces.is_some()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> (Session, filmcraft_project::ItemId) {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().video_tracks[0].items[0].item;
        (s, item)
    }

    #[test]
    fn moments_are_measured_from_their_own_frames() {
        let (mut s, item) = demo();
        let r = s.execute(MEASURE_CMD, json!({"item": item.0, "moments": [[0.0, 1.0], [1.0, 2.5]], "frames": 2, "faces": false})).unwrap();
        let ms = r["moments"].as_array().unwrap();
        assert_eq!(ms.len(), 2);
        for m in ms {
            assert_eq!(m["measured"], true, "{m}");
            for k in ["sharpness", "exposure", "brightness", "shake", "motion"] {
                let v = m[k].as_f64().unwrap();
                assert!((0.0..=100.0).contains(&v), "{k} {v}");
            }
            assert!(m.get("faces").is_none());
        }
        assert_eq!(r["facesCounted"], false);
    }

    #[test]
    fn hostile_moments_are_refused() {
        let (mut s, item) = demo();
        for bad in [
            json!({"item": item.0}),
            json!({"item": item.0, "moments": []}),
            json!({"item": item.0, "moments": [[2.0, 1.0]]}),
            json!({"item": item.0, "moments": [[-1.0, 1.0]]}),
            json!({"item": item.0, "moments": [["a", 1.0]]}),
            json!({"item": item.0, "moments": [[0.0, 1.0, 2.0]]}),
            json!({"item": item.0, "moments": vec![json!([0.0, 1.0]); MAX_MOMENTS + 1]}),
            json!({"item": 999_999, "moments": [[0.0, 1.0]]}),
        ] {
            assert!(s.execute(MEASURE_CMD, bad.clone()).is_err(), "{bad}");
        }
        // past the end: measured up to it
        let r = s.execute(MEASURE_CMD, json!({"item": item.0, "moments": [[1e9, 1e12]], "frames": 1, "faces": false})).unwrap();
        assert_eq!(r["moments"].as_array().unwrap().len(), 1);
    }
}
