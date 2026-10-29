use serde_json::json;

use super::*;
use crate::media_test_util::{make_movie, tmp_dir};

/// The stand-in: each pixel a 4 × 4 block, with a fine checker on its red (detail the picture
/// does not have, which a 2 × 2 average takes out again).
struct Checker;

impl Enlarger for Checker {
    fn enlarge(
        &self,
        rgba: &[u8],
        w: usize,
        h: usize,
        _: f32,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> std::result::Result<(Vec<u8>, usize, usize), String> {
        if !progress(1, 1) {
            return Err("stopped".into());
        }
        let (ow, oh) = (w * 4, h * 4);
        let mut out = vec![0u8; ow * oh * 4];
        for y in 0..oh {
            for x in 0..ow {
                let i = ((y / 4) * w + x / 4) * 4;
                let o = (y * ow + x) * 4;
                out[o..o + 4].copy_from_slice(&rgba[i..i + 4]);
                out[o] = if (x + y) % 2 == 0 { out[o].saturating_add(40) } else { out[o].saturating_sub(40) };
            }
        }
        Ok((out, ow, oh))
    }
}

/// A session with a 320 × 180 photo (a soft gradient, red in the middle) held 2 s on V1 of a
/// 1920 × 1080 sequence: `(session, item, clip, folder)`.
fn photo_session(name: &str) -> (Session, ItemId, u64, std::path::PathBuf) {
    use_stand_in(Arc::new(Checker));
    let dir = tmp_dir(name);
    let path = dir.join("small.png");
    image::RgbaImage::from_fn(320, 180, |x, y| image::Rgba([120, (x * 255 / 320) as u8, (y * 255 / 180) as u8, 255])).save(&path).unwrap();
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap());
    s.execute("file.newSequence", json!({"name": "Photos", "width": 1920, "height": 1080, "fps": 30})).unwrap();
    let two = Tick::from_seconds_f64(2.0).0;
    let r = s.execute("timeline.place", json!({"item": item.0, "track": "V1", "time": 0, "sourceIn": 0, "duration": two, "audio": false})).unwrap();
    let clip = r["clips"][0].as_u64().unwrap();
    s.execute("clip.fillFrame", json!({"clips": [clip]})).unwrap();
    s.state.project_selection = vec![item];
    (s, item, clip, dir)
}

fn frame(s: &Session) -> filmcraft_render::Image {
    let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
    filmcraft_render::render_sequence(&s.project, s.state.active_sequence.unwrap(), Tick::ZERO, filmcraft_render::RenderOptions::default(), &provider)
}

/// The red channel's mean step between neighbours along the middle row (fine detail).
fn detail(img: &filmcraft_render::Image) -> f32 {
    let y = img.h / 2;
    (600..1300).map(|x| (img.get(x + 1, y)[0] - img.get(x, y)[0]).abs()).sum::<f32>() / 700.0
}

/// The frame averaged over 16 × 16 blocks (its picture's place, without fine detail).
fn coarse(img: &filmcraft_render::Image) -> Vec<f32> {
    let mut v = Vec::new();
    for by in 0..img.h / 16 {
        for bx in 0..img.w / 16 {
            let mut a = 0.0;
            for y in 0..16 {
                for x in 0..16 {
                    a += img.get(bx * 16 + x, by * 16 + y)[1];
                }
            }
            v.push(a / 256.0);
        }
    }
    v
}

fn upscaled(s: &Session, item: ItemId) -> Option<Upscaled> {
    s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.interpret.upscaled.clone())
}

#[test]
fn a_photo_is_shown_from_its_sharper_copy_with_its_framing_kept() {
    let (mut s, item, _, dir) = photo_session("upscale-photo");
    let before = frame(&s);
    let r = s.execute(CMD, json!({"wait": true})).unwrap();
    assert_eq!(r["items"][0]["factor"], json!(4), "{r}");
    assert_eq!(r["items"][0]["size"], json!([1280, 720]));
    let u = upscaled(&s, item).expect("the copy is attached");
    assert_eq!((u.factor, u.strength), (4, 1.0));
    assert!(u.path.ends_with("small (upscaled ×4).png"), "{}", u.path);
    let png = image::open(&u.path).unwrap();
    assert_eq!((png.width(), png.height()), (1280, 720));
    // the clip is drawn from the copy (the checker shows), where it was (the coarse picture holds)
    let after = frame(&s);
    assert!(detail(&after) > 10.0 * detail(&before).max(1e-4), "{} against {}", detail(&after), detail(&before));
    let (a, b) = (coarse(&before), coarse(&after));
    let (bw, bh) = (before.w / 16, before.h / 16);
    // (inside: at the frame's edges the two pictures' borders are held a little differently)
    let inner = (0..a.len()).filter(|i| (1..bw - 1).contains(&(i % bw)) && (1..bh - 1).contains(&(i / bw)));
    let off = inner.map(|i| (a[i] - b[i]).abs()).fold(0f32, f32::max);
    let mean = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    assert!(off < 0.01 && mean < 0.002, "the picture moved: at most {off}, mean {mean}");
    // its own size kept: the item is still 320 × 180
    let src = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    assert_eq!(src.info().video.as_ref().map(|v| (v.width, v.height)), Some((320, 180)));
    assert!((src.max_scale() - 4.0).abs() < 1e-6);
    // (a decoder may give more than asked for: the renderer shrinks it)
    assert!(src.video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).unwrap().width >= 320);
    assert_eq!(src.video_frame(FrameRequest { time: Tick::ZERO, scale: 4.0 }).unwrap().width, 1280);
    // one undo step; off takes it away again
    s.execute("edit.undo", json!({})).unwrap();
    assert!(upscaled(&s, item).is_none());
    assert_eq!(frame(&s).px, before.px, "back as it was");
    s.execute("edit.redo", json!({})).unwrap();
    assert!(upscaled(&s, item).is_some());
    let r = s.execute(CMD, json!({"off": true})).unwrap();
    assert_eq!(r["off"], json!([item.0]));
    assert!(upscaled(&s, item).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn factors_follow_the_frame_and_stay_within_the_limits() {
    assert_eq!(auto_factor(320, 180, 1920), Some(4));
    assert_eq!(auto_factor(1000, 800, 1920), Some(2));
    assert_eq!(auto_factor(1300, 700, 3840), Some(3));
    assert_eq!(auto_factor(1920, 1080, 1920), None);
    assert_eq!(auto_factor(1400, 800, 1920), None, "two thirds of the frame or more: not soft");
    assert_eq!(auto_factor(0, 0, 1920), Some(4));
    assert_eq!(capped(3000, 2000, true, 4), 2, "a photo's copy at most 8192 a side");
    assert_eq!(capped(2000, 1000, false, 4), 3, "a video's at most 7680 × 4320");
    assert_eq!(capped(9000, 10, true, 4), 0);
    // the network's four times averaged down to each factor
    // (values off the limits: the stand-in's checker would clip at 0 and 255)
    let px: Vec<u8> = (0..6 * 4 * 4).map(|i| (60 + i * 7 % 130) as u8).collect();
    for k in 1..=4u32 {
        let (out, w, h) = enlarge_by(&Checker, &px, 6, 4, k, 1.0, &mut |_, _| true).unwrap();
        assert_eq!((w, h, out.len()), (6 * k as usize, 4 * k as usize, 6 * 4 * (k * k) as usize * 4));
    }
    // restored at its own size: the checker averages out, the picture comes back
    let (one, _, _) = enlarge_by(&Checker, &px, 6, 4, 1, 1.0, &mut |_, _| true).unwrap();
    assert!(one.iter().zip(&px).all(|(a, b)| a.abs_diff(*b) <= 1), "{one:?}");
    assert_eq!(area_resize(&[], 0, 0, 3, 2), vec![0; 24]);
}

#[test]
fn a_video_gets_a_copy_at_twice_its_size_and_keeps_its_sound() {
    use_stand_in(Arc::new(Checker));
    let dir = tmp_dir("upscale-video");
    let path = dir.join("clip.mov");
    make_movie(&path, filmcraft_media::DemoScene::Aurora, 160, 96, 6);
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap());
    let r = s.execute(CMD, json!({"items": [item.0], "factor": 2, "wait": true})).unwrap();
    assert_eq!(r["items"][0]["size"], json!([320, 192]), "{r}");
    let u = upscaled(&s, item).expect("attached");
    assert!(u.path.ends_with("clip (upscaled ×2).mp4"), "{}", u.path);
    let src = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    assert!((src.max_scale() - 2.0).abs() < 1e-6);
    let f = src.video_frame(FrameRequest { time: Tick::ZERO, scale: 2.0 }).unwrap();
    assert_eq!((f.width, f.height), (320, 192));
    // (a video decoder gives its whole picture when asked for less: the renderer shrinks it)
    assert!(src.video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).unwrap().width >= 160);
    // the sound is the original's (the copy has none)
    let a = src.audio(0, 4800, 48_000).unwrap();
    assert!(a.frames() > 0, "the item still plays its sound");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_and_unfit_media_are_refused() {
    let (mut s, item, _, dir) = photo_session("upscale-hostile");
    for p in [
        json!({"factor": 5}),
        json!({"factor": 0}),
        json!({"factor": "huge"}),
        json!({"strength": 2}),
        json!({"strength": -0.5}),
        json!({"format": "avi"}),
        json!({"items": [999_999]}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    assert!(upscaled(&s, item).is_none(), "nothing attached");
    // already as large as the frame: left out, saying why
    let e = s.execute(CMD, json!({"items": [item.0], "wait": true}));
    assert!(e.is_ok());
    s.execute("file.newSequence", json!({"name": "Small", "width": 320, "height": 180, "fps": 30})).unwrap();
    s.execute(CMD, json!({"off": true})).unwrap();
    let e = s.execute(CMD, json!({"items": [item.0]})).unwrap_err().to_string();
    assert!(e.contains("nearly as large as the frame"), "{e}");
    // 360° media is seen through its camera: left out
    s.execute("media.vrProperties", json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
    let e = s.execute(CMD, json!({"items": [item.0], "factor": 2})).unwrap_err().to_string();
    assert!(e.contains("360"), "{e}");
    let st = s.execute(STATUS_CMD, json!({})).unwrap();
    assert_eq!(st["bytes"], json!(4_885_111));
    assert!(st["license"].as_str().unwrap().starts_with("BSD-3-Clause"));
    let _ = std::fs::remove_dir_all(&dir);
}
