use serde_json::json;

use super::*;
use crate::media_test_util::tmp_dir;

/// The stand-in depth: a picture's red is its nearness.
struct RedIsNear;

impl DepthModel for RedIsNear {
    fn depth(&self, _: u64, rgba: &[u8], w: usize, h: usize) -> Option<DepthField> {
        DepthField::new(w, h, rgba.as_chunks::<4>().0.iter().map(|p| f32::from(p[0]) / 255.0).collect())
    }
}

/// A session with a 1200 × 900 photo — its left half red (near), its right half dark blue
/// stripes (far) — held 5 s on V1 of a 1920 × 1080 sequence, selected.
fn photo_session(name: &str) -> (Session, u64, std::path::PathBuf) {
    use_stand_in(Arc::new(RedIsNear));
    let dir = tmp_dir(name);
    let path = dir.join("photo.png");
    image::RgbaImage::from_fn(1200, 900, |x, _| if x < 600 { image::Rgba([255, 40, 40, 255]) } else { image::Rgba([0, 20 + (x % 12) as u8 * 10, 120, 255]) })
        .save(&path)
        .unwrap();
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

fn item(s: &Session, clip: u64) -> filmcraft_project::TrackItem {
    s.active_sequence().unwrap().find_item(ClipId(clip)).unwrap().1.clone()
}

fn values(e: &EffectInstance, id: &str) -> Vec<f64> {
    e.params[id].keyframes.iter().filter_map(|k| k.value.as_f64()).collect()
}

#[test]
fn a_3d_photo_follows_the_ken_burns_move_and_undoes_in_one_step() {
    let (mut s, clip, dir) = photo_session("depth-auto");
    s.execute("clip.kenBurns", json!({"move": "in"})).unwrap();
    let r = s.execute(PHOTO_CMD, json!({})).unwrap();
    assert_eq!(r["clips"][0]["move"], json!("in"), "{r}");
    let it = item(&s, clip);
    let e = it.effect(PHOTO_3D).unwrap();
    assert_eq!(values(e, "push"), vec![0.0, PUSH], "pushed in over the clip");
    let k = &e.params["push"].keyframes;
    assert_eq!((k[0].interp, k[1].interp), (Interpolation::EaseOut, Interpolation::EaseIn));
    assert_eq!(k[0].time, it.effect_time_at(it.start));
    assert_eq!(e.params["pivot"].value, ParamValue::Float(0.0), "in: the farthest stays");
    s.execute("edit.undo", json!({})).unwrap();
    assert!(item(&s, clip).effect(PHOTO_3D).is_none(), "one undo step");
    // run again: still one 3D photo on the clip
    s.execute(PHOTO_CMD, json!({"move": "out", "amount": 100})).unwrap();
    s.execute(PHOTO_CMD, json!({"move": "out", "amount": 100})).unwrap();
    let it = item(&s, clip);
    assert_eq!(it.effects.iter().filter(|e| e.effect == PHOTO_3D).count(), 1);
    assert_eq!(values(it.effect(PHOTO_3D).unwrap(), "push"), vec![2.0 * PUSH, 0.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn moves_round_the_subject_keep_its_depth_still() {
    let (mut s, clip, dir) = photo_session("depth-round");
    // the subject on the red half: the nearest
    let r = s.execute(PHOTO_CMD, json!({"move": "right", "subject": [0.1, 0.2, 0.3, 0.5]})).unwrap();
    assert!((r["clips"][0]["pivot"].as_f64().unwrap() - 1.0).abs() < 0.01, "{r}");
    let e = item(&s, clip).effect(PHOTO_3D).unwrap().clone();
    assert_eq!(values(&e, "travel_x"), vec![TRAVEL, -TRAVEL], "the camera right: the near parts go left");
    assert!((e.params["pivot"].value.as_f64().unwrap() - 100.0).abs() < 1.0);
    // on the stripes: the farthest
    let r = s.execute(PHOTO_CMD, json!({"move": "up", "subject": [0.6, 0.2, 0.3, 0.5]})).unwrap();
    assert!(r["clips"][0]["pivot"].as_f64().unwrap() < 0.05, "{r}");
    assert_eq!(values(item(&s, clip).effect(PHOTO_3D).unwrap(), "travel_y"), vec![-TRAVEL, TRAVEL]);
    // a turn: a keyframe every sixteenth, back where it began (up first: y down is positive)
    s.execute(PHOTO_CMD, json!({"move": "circle", "pivot": 0.4})).unwrap();
    let e = item(&s, clip).effect(PHOTO_3D).unwrap().clone();
    let (x, y) = (values(&e, "travel_x"), values(&e, "travel_y"));
    assert_eq!((x.len(), y.len()), (CIRCLE_KEYS + 1, CIRCLE_KEYS + 1));
    assert!((x[0] - TRAVEL).abs() < 1e-9 && (x[CIRCLE_KEYS] - TRAVEL).abs() < 1e-9 && y[0].abs() < 1e-9);
    assert!((x[CIRCLE_KEYS / 2] + TRAVEL).abs() < 1e-9 && (y[CIRCLE_KEYS / 4] + 0.6 * TRAVEL).abs() < 1e-9);
    assert_eq!(e.params["pivot"].value, ParamValue::Float(40.0));
    // a Ken Burns pan says which way
    s.execute("clip.kenBurns", json!({"move": "pan"})).unwrap();
    let mv = ken_burns_move(&item(&s, clip)).unwrap();
    assert!(matches!(mv, Move::Left | Move::Right | Move::Up | Move::Down), "{mv:?}");
    let r = s.execute(PHOTO_CMD, json!({})).unwrap();
    assert_eq!(r["clips"][0]["move"], json!(mv.id()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn depth_blur_focuses_on_the_subject_and_pulls_focus() {
    let (mut s, clip, dir) = photo_session("depth-blur");
    let r = s.execute(BLUR_CMD, json!({"subject": [0.1, 0.1, 0.3, 0.3], "blur": 2.0})).unwrap();
    assert!((r["clips"][0]["focus"].as_f64().unwrap() - 1.0).abs() < 0.01, "{r}");
    let e = item(&s, clip).effect(DEPTH_BLUR).unwrap().clone();
    assert_eq!(e.params["blur"].value, ParamValue::Float(2.0));
    assert!(e.params["focus"].keyframes.is_empty());
    // a pull from the far end to the near subject over the first half
    s.execute(BLUR_CMD, json!({"focus": "near", "pull": true})).unwrap();
    let it = item(&s, clip);
    let e = it.effect(DEPTH_BLUR).unwrap();
    let v = values(e, "focus");
    assert!(v[0] == 0.0 && (v[1] - 100.0).abs() < 1.0, "{v:?}");
    assert!(e.params["focus"].keyframes[1].time < it.effect_time_at(it.start + Tick::from_seconds_f64(2.6)));
    s.execute(BLUR_CMD, json!({"focus": "far"})).unwrap();
    assert!(item(&s, clip).effect(DEPTH_BLUR).unwrap().params["focus"].value.as_f64().unwrap() < 5.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_render_moves_the_near_half_and_leaves_the_far_half() {
    let (mut s, clip, dir) = photo_session("depth-render");
    let frame = |s: &Session| {
        let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
        let opts = filmcraft_render::RenderOptions { scale: 0.25, ..Default::default() };
        filmcraft_render::render_sequence(&s.project, s.state.active_sequence.unwrap(), Tick::ZERO, opts, &provider)
    };
    let before = frame(&s);
    // the near (red) half moves 5 % of the width right, the far half stays (no zoom)
    s.execute(PHOTO_CMD, json!({"move": "left", "pivot": "far", "amount": 100})).unwrap();
    s.edit_sequence("fixed", |q, _, _| {
        let it = q.find_item_mut(ClipId(clip)).unwrap().1;
        let e = it.effects.iter_mut().find(|e| e.effect == PHOTO_3D).unwrap();
        e.params.insert("travel_x".into(), Param::new(ParamValue::Float(5.0)));
        e.params.insert("fill".into(), Param::new(ParamValue::Bool(false)));
        Ok(())
    })
    .unwrap();
    let after = frame(&s);
    assert_eq!((before.w, before.h), (after.w, after.h));
    // the boundary between red and blue: where red ends along the middle row (the photo is
    // pillarboxed)
    let edge = |img: &filmcraft_render::Image| (0..img.w).filter(|x| img.get(*x, img.h / 2)[0] > 0.5).max().unwrap_or(0);
    let (a, b) = (edge(&before), edge(&after));
    assert!(b > a + 5, "the red half reaches further right: {a} → {b}");
    assert!(a > after.w / 3, "{a}");
    let x = after.w * 3 / 4;
    assert_eq!(after.get(x, after.h / 2), before.get(x, after.h / 2), "the far right stays");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn videos_360_clips_and_hostile_parameters_are_refused() {
    let (mut s, clip, dir) = photo_session("depth-hostile");
    for p in [
        json!({"move": "sideways"}),
        json!({"move": 3}),
        json!({"amount": -1}),
        json!({"amount": f64::MAX}),
        json!({"pivot": "middle"}),
        json!({"pivot": 2}),
        json!({"subject": [0.5, 0.5, 0.6, 0.1]}),
        json!({"subject": "face"}),
        json!({"clips": [999_999]}),
    ] {
        assert!(s.execute(PHOTO_CMD, p.clone()).is_err(), "{p}");
    }
    for p in [json!({"focus": "middle"}), json!({"focus": -0.5}), json!({"blur": 11}), json!({"range": 2}), json!({"subject": [1, 1, 1, 1]})] {
        assert!(s.execute(BLUR_CMD, p.clone()).is_err(), "{p}");
    }
    assert!(item(&s, clip).effect(PHOTO_3D).is_none() && item(&s, clip).effect(DEPTH_BLUR).is_none(), "nothing written");
    // a video's depth would flicker: refused, saying why
    let mut v = Session::default();
    v.execute("file.openDemoProject", json!({})).unwrap();
    let c = v.active_sequence().unwrap().video_tracks[0].items[0].id;
    v.state.selection = vec![c];
    let e = v.execute(PHOTO_CMD, json!({})).unwrap_err().to_string();
    assert!(e.contains("depth is for photos"), "{e}");
    assert!(v.execute(BLUR_CMD, json!({"clips": [c.0]})).is_err());
    let st = s.execute(STATUS_CMD, json!({})).unwrap();
    assert_eq!(st["bytes"], json!(99_218_434));
    assert!(st["license"].as_str().unwrap().starts_with("Apache-2.0"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kept_depth_reads_back_and_refuses_damage() {
    let f = DepthField::new(7, 3, (0..21).map(|i| (i as f32 * 0.37).sin().abs()).collect()).unwrap();
    let bytes = encode_kept(&f).unwrap();
    let back = decode_kept(&bytes).unwrap();
    assert_eq!((back.w, back.h), (7, 3));
    assert!(back.near.iter().zip(&f.near).all(|(a, b)| (a - b).abs() < 1e-4));
    // truncated, flipped, the sizes lied about, not one at all: nothing (and no panic)
    for cut in 0..bytes.len() {
        let _ = decode_kept(&bytes[..cut]);
    }
    for i in 0..bytes.len() {
        let mut b = bytes.clone();
        b[i] ^= 0x5a;
        let _ = decode_kept(&b);
    }
    let mut big = bytes.clone();
    big[8..12].copy_from_slice(&100_000u32.to_le_bytes());
    assert!(decode_kept(&big).is_none());
    assert!(decode_kept(b"not a depth map").is_none());
}
