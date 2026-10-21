use serde_json::json;

use super::*;
use crate::media_test_util::tmp_dir;

const FRAME: (f64, f64) = (1920.0, 1080.0);

fn ask(subject: Option<[f64; 4]>, motion: Move) -> Ask {
    Ask { subject, motion, fill: true, zoom: None, reverse: false }
}

/// The picture covers the frame at `pose`: no edge of it shows.
fn covers(picture: (f64, f64), pose: Pose) -> bool {
    let s = pose.scale / 100.0;
    let (pw, ph) = (picture.0 * s, picture.1 * s);
    let (x0, y0) = (pose.position[0] - pw / 2.0, pose.position[1] - ph / 2.0);
    x0 <= 1e-6 && y0 <= 1e-6 && x0 + pw >= FRAME.0 - 1e-6 && y0 + ph >= FRAME.1 - 1e-6
}

fn contains(outer: [f64; 4], inner: [f64; 4]) -> bool {
    let e = 1e-6;
    inner[0] >= outer[0] - e && inner[1] >= outer[1] - e && inner[0] + inner[2] <= outer[0] + outer[2] + e && inner[1] + inner[3] <= outer[1] + outer[3] + e
}

#[test]
fn a_move_in_starts_wide_and_ends_framed_on_the_subject() {
    // a 4:3 photo, a robot right of the middle
    let photo = (1440.0, 1080.0);
    let robot = [0.55, 0.3, 0.2, 0.4];
    let (a, b) = path(photo, FRAME, 5.0, ask(Some(robot), Move::In)).unwrap();
    assert!(b.scale > a.scale, "closer at the end: {a:?} → {b:?}");
    assert!((a.scale - 100.0 * 1920.0 / 1440.0).abs() < 1e-9, "the wide end fills the frame, no closer");
    assert!(covers(photo, a) && covers(photo, b), "no edge shows at either end");
    let end = shown(photo, FRAME, b);
    assert!(contains(end, robot), "the robot is whole in the last frame: {end:?}");
    // …as near its middle as the picture allows: centred up and down, and sideways as far
    // towards it as the picture reaches (its right edge on the frame's)
    let mid = [end[0] + end[2] / 2.0, end[1] + end[3] / 2.0];
    assert!((mid[1] - 0.5).abs() < 1e-6 && mid[0] > 0.5 && (end[0] + end[2] - 1.0).abs() < 1e-6, "{end:?}");
    let centred = [0.4, 0.35, 0.2, 0.3];
    let (_, b) = path(photo, FRAME, 5.0, ask(Some(centred), Move::In)).unwrap();
    let end = shown(photo, FRAME, b);
    assert!((end[0] + end[2] / 2.0 - 0.5).abs() < 1e-6 && (end[1] + end[3] / 2.0 - 0.5).abs() < 1e-6, "a subject with room around it is centred: {end:?}");
    // slow and subtle: 2.5 % closer a second of the hold, at most a quarter
    assert!((b.scale / a.scale - 1.125).abs() < 1e-9, "{}", b.scale / a.scale);
    let (a, b) = path(photo, FRAME, 60.0, ask(Some([0.45, 0.45, 0.02, 0.02]), Move::In)).unwrap();
    assert!((b.scale / a.scale - 1.25).abs() < 1e-9);
    // a big subject is framed whole rather than zoomed past
    let big = [0.1, 0.0, 0.8, 1.0];
    let (_, b) = path(photo, FRAME, 8.0, ask(Some(big), Move::In)).unwrap();
    assert!((b.scale / (100.0 * 1920.0 / 1440.0) - (1.0 + MIN_ZOOM)).abs() < 1e-9, "the least zoom, {}", b.scale);
    assert_eq!(zoom_for(4.0), 0.1);
    assert_eq!((zoom_for(0.5), zoom_for(1e6), zoom_for(f64::NAN)), (MIN_ZOOM, MAX_ZOOM, MIN_ZOOM));
}

#[test]
fn out_is_in_backwards_and_none_holds() {
    let photo = (3000.0, 2000.0);
    let s = Some([0.2, 0.2, 0.2, 0.2]);
    let (a, b) = path(photo, FRAME, 4.0, ask(s, Move::In)).unwrap();
    assert_eq!(path(photo, FRAME, 4.0, ask(s, Move::Out)), Some((b, a)));
    assert_eq!(path(photo, FRAME, 4.0, ask(s, Move::None)), Some((b, b)));
}

#[test]
fn a_pan_crosses_the_subject_at_a_steady_zoom_either_way() {
    // a portrait photo filling a landscape frame: the room is up and down
    let photo = (810.0, 1080.0);
    let s = Some([0.3, 0.3, 0.3, 0.2]);
    let (a, b) = path(photo, FRAME, 6.0, ask(s, Move::Pan)).unwrap();
    assert_eq!(a.scale, b.scale);
    assert_eq!(a.position[0], b.position[0], "a tilt, not a sideways move");
    assert!(a.position[1] != b.position[1]);
    assert!(covers(photo, a) && covers(photo, b));
    // 2 % of the picture a second: 12 % over six seconds
    let (ta, tb) = (shown(photo, FRAME, a), shown(photo, FRAME, b));
    assert!(((tb[1] - ta[1]).abs() - 0.12).abs() < 1e-6, "{ta:?} {tb:?}");
    let (ra, rb) = path(photo, FRAME, 6.0, Ask { reverse: true, ..ask(s, Move::Pan) }).unwrap();
    assert_eq!((ra, rb), (b, a));
    // a landscape photo pans sideways
    let (a, b) = path((3000.0, 1600.0), FRAME, 6.0, ask(s, Move::Pan)).unwrap();
    assert!(a.position[0] != b.position[0] && a.position[1] == b.position[1]);
}

#[test]
fn no_edge_shows_wherever_the_subject_is() {
    for photo in [(4032.0, 3024.0), (3024.0, 4032.0), (1920.0, 1080.0), (100.0, 4000.0), (5000.0, 90.0)] {
        for subject in [None, Some([0.0, 0.0, 0.05, 0.05]), Some([0.95, 0.9, 0.05, 0.1]), Some([0.0, 0.0, 1.0, 1.0]), Some([0.5, 0.5, 0.001, 0.001])] {
            for motion in [Move::In, Move::Out, Move::Pan, Move::None] {
                for seconds in [0.0, 0.5, 4.0, 30.0] {
                    let (a, b) = path(photo, FRAME, seconds, ask(subject, motion)).unwrap();
                    assert!(covers(photo, a) && covers(photo, b), "{photo:?} {subject:?} {motion:?} {seconds}: {a:?} {b:?}");
                }
            }
        }
    }
    // the whole picture asked for: it only breathes, at most 5 % closer than fitted
    let photo = (3024.0, 4032.0);
    let (a, b) = path(photo, FRAME, 10.0, Ask { fill: false, ..ask(Some([0.1, 0.1, 0.2, 0.2]), Move::In) }).unwrap();
    let fitted = 100.0 * 1080.0 / 4032.0;
    assert!((a.scale - fitted).abs() < 1e-9 && (b.scale / fitted - 1.0 - MAX_FULL_ZOOM).abs() < 1e-9, "{a:?} {b:?}");
    // sizes that make no sense
    assert_eq!(path((0.0, 10.0), FRAME, 4.0, ask(None, Move::In)), None);
    assert_eq!(path((f64::NAN, 10.0), FRAME, 4.0, ask(None, Move::In)), None);
    assert_eq!(path((10.0, 10.0), (0.0, 0.0), 4.0, ask(None, Move::In)), None);
}

/// A session with a 1200 × 900 photo held 5 s on V1 of a 1920 × 1080 sequence, selected.
fn photo_session() -> (Session, u64, std::path::PathBuf) {
    let dir = tmp_dir("ken-burns");
    let path = dir.join("robot.png");
    image::RgbaImage::from_pixel(1200, 900, image::Rgba([90, 120, 200, 255])).save(&path).unwrap();
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = r["items"][0].as_u64().unwrap();
    s.execute("file.newSequence", json!({"name": "Photos", "width": 1920, "height": 1080, "fps": 30})).unwrap();
    let five = Tick::from_seconds_f64(5.0).0;
    let r = s.execute("timeline.place", json!({"item": item, "track": "V1", "time": 0, "sourceIn": 0, "duration": five, "audio": false})).unwrap();
    let clip = r["clips"][0].as_u64().unwrap();
    s.state.selection = vec![ClipId(clip)];
    (s, clip, dir)
}

#[test]
fn the_command_keyframes_position_and_scale_eased_and_undoes_in_one_step() {
    let (mut s, clip, dir) = photo_session();
    let seq = s.state.active_sequence.unwrap();
    let it = s.project.sequence(seq).unwrap().find_item(ClipId(clip)).unwrap().1.clone();
    assert_eq!(it.duration, s.sequence_rate().snap_nearest(Tick::from_seconds_f64(5.0)), "a photo held as long as asked");
    let r = s.execute(CMD, json!({"subject": [0.6, 0.25, 0.2, 0.3], "move": "in"})).unwrap();
    let c = &r["clips"][0];
    assert_eq!(c["move"], json!("in"));
    let end: Vec<f64> = c["shownAtEnd"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
    assert!(contains([end[0], end[1], end[2], end[3]], [0.6, 0.25, 0.2, 0.3]), "{c}");
    let it = s.project.sequence(seq).unwrap().find_item(ClipId(clip)).unwrap().1.clone();
    let m = it.effect("motion").unwrap();
    for param in ["scale", "position"] {
        let k = &m.params[param].keyframes;
        assert_eq!(k.len(), 2, "{param}");
        assert_eq!((k[0].interp, k[1].interp), (Interpolation::EaseOut, Interpolation::EaseIn), "{param}: eased out of the first, into the last");
        assert_eq!(k[0].time, it.effect_time_at(it.start));
        assert_eq!(k[1].time, it.effect_time_at(it.end() - s.sequence_rate().frame_duration()), "the last keyframe on the last frame");
    }
    let to = c["to"]["scale"].as_f64().unwrap();
    assert_eq!(m.params["scale"].keyframes[1].value, ParamValue::Float(to));
    assert!(!it.scale_to_frame);
    // the middle of the move is between its ends
    let mid = m.f64_at("scale", it.effect_time_at(it.start + Tick::from_seconds_f64(2.5)));
    assert!(mid > c["from"]["scale"].as_f64().unwrap() && mid < to, "{mid}");
    s.execute("edit.undo", json!({})).unwrap();
    let it = s.project.sequence(seq).unwrap().find_item(ClipId(clip)).unwrap().1.clone();
    assert!(it.effect("motion").unwrap().params.get("scale").is_none_or(|p| p.keyframes.is_empty()), "one undo step");
    // "none" holds it framed, without keyframes
    s.execute(CMD, json!({"clips": [clip], "move": "none", "subject": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}})).unwrap();
    let it = s.project.sequence(seq).unwrap().find_item(ClipId(clip)).unwrap().1.clone();
    assert!(it.effect("motion").unwrap().params["scale"].keyframes.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, clip, dir) = photo_session();
    for p in [
        json!({"move": "sideways"}),
        json!({"move": 3}),
        json!({"fit": "stretch"}),
        json!({"zoom": -1}),
        json!({"zoom": 50}),
        json!({"zoom": f64::MAX}),
        json!({"subject": [0.5, 0.5, 0.6, 0.1]}),
        json!({"subject": [0.1, 0.1, 0.0, 0.1]}),
        json!({"subject": [0.1, 0.1, 0.1]}),
        json!({"subject": "face"}),
        json!({"subject": {"x": -0.1, "y": 0, "w": 0.5, "h": 0.5}}),
        json!({"clips": [999_999]}),
        json!({"clips": []}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        assert!(r.expect("no panic").is_err(), "{p}");
    }
    assert!(s.execute(CMD, json!({"clips": [clip], "zoom": 0.0})).is_ok(), "no zoom is a pan of nothing, not an error");
    let _ = std::fs::remove_dir_all(&dir);
}
