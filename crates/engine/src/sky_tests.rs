//! Finding the sky (the first guess on drawn pictures, a stand-in finder through the job) and
//! swapping it (skies made here, the layer above the clip, the clip relit).

use std::sync::Arc;

use filmcraft_media::DemoScene;
use filmcraft_project::{ClipId, MaskMode, TrackItem};
use serde_json::json;

use super::*;
use crate::Session;
use crate::media_test_util::{make_movie, session_with, tmp_dir};

/// A picture: a smooth blue sky over the top `sky` share, busy grey-brown ground below.
fn picture(w: usize, h: usize, sky: f32) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let v = if (y as f32) < h as f32 * sky {
                let t = y as f32 / h as f32;
                [(90.0 + 80.0 * t) as u8, (150.0 + 60.0 * t) as u8, 235, 255]
            } else {
                let n = ((x * 37 + y * 91) % 53) as u8 * 3;
                [70 + n, 60 + n, 50 + n / 2, 255]
            };
            px[i..i + 4].copy_from_slice(&v);
        }
    }
    px
}

#[test]
fn the_sky_is_guessed_where_it_is_bright_plain_and_joins_the_top() {
    let (w, h) = (512, 288);
    let g = guess(&picture(w, h, 0.45), w, h).unwrap();
    assert!((g.share() - 0.45).abs() < 0.05, "{}", g.share());
    let o = sky_outline(&picture(w, h, 0.45), w, h, 64, None).unwrap();
    assert!(o.iter().all(|(_, y)| *y <= h as f32 * 0.5), "the outline stays in the sky");
    // all ground: no sky
    let ground = picture(w, h, 0.0);
    assert!(guess(&ground, w, h).unwrap().share() < MIN_SKY && sky_outline(&ground, w, h, 64, None).is_none());
    // a bright plain wall that does not reach the top is not sky
    let mut wall = picture(w, h, 0.0);
    for y in h / 3..2 * h / 3 {
        for x in 0..w {
            wall[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&[200, 200, 205]);
        }
    }
    assert!(guess(&wall, w, h).unwrap().share() < 0.01);
    // points inside the sky and in the ground below it
    let g = guess(&picture(w, h, 0.45), w, h).unwrap();
    let (inside, outside) = prompts(&g, |_, _| true, 6);
    assert!(!inside.is_empty() && inside.iter().all(|(_, y)| *y < h as f32 * 0.45));
    assert!(!outside.is_empty() && outside.iter().all(|(_, y)| *y > h as f32 * 0.4));
    // hostile pictures
    assert!(guess(&[], 0, 0).is_none() && guess(&[1, 2, 3], 100, 100).is_none());
}

#[test]
fn skies_are_made_of_their_kind() {
    let mean = |px: &[u8], w: usize, h: usize| mean_color(px, w, h, true).unwrap();
    let (w, h) = (256, 144);
    let clear = mean(&made_sky("clear", w, h, false, 1), w, h);
    let sunset = mean(&made_sky("sunset", w, h, false, 1), w, h);
    let night = mean(&made_sky("night", w, h, false, 1), w, h);
    let clouds = mean(&made_sky("clouds", w, h, false, 1), w, h);
    assert!(clear[2] > clear[0], "clear is blue");
    assert!(sunset[0] - sunset[2] > clear[0] - clear[2], "a sunset is warmer");
    assert!(night.iter().all(|v| *v < 0.05), "night is dark: {night:?}");
    assert!(clouds[0] > clear[0], "clouds are lighter");
    let sphere = made_sky("dusk", 512, 0, true, 3);
    assert_eq!(sphere.len(), 512 * 256 * 4, "a whole sphere is 2:1");
    assert_eq!(made_sky("clear", 1, 1, false, 0).len(), 16 * 8 * 4, "sizes clamped");
}

#[test]
fn the_scene_is_relit_towards_the_new_sky() {
    let blue = [0.2, 0.35, 0.8];
    let warm = [0.8, 0.4, 0.2];
    let (t, e) = relit(blue, warm, 0.3);
    assert!(t > 5.0, "warmer: {t}");
    let (_, e2) = relit(blue, [0.05, 0.06, 0.1], 0.3);
    assert!(e2 < 0.0 && e.abs() < 1.0, "darker: {e2}");
    assert_eq!(relit(blue, warm, 0.0), (0.0, 0.0));
    let (t, e) = relit([0.0; 3], [f64::NAN; 3], 1.0);
    assert!(t.is_finite() && e.is_finite());
}

/// The top 40% of any picture as the sky.
struct TopBand;

impl SkyFinder for TopBand {
    fn find(&self, _: &[u8], w: usize, h: usize, points: usize) -> std::result::Result<Option<Vec<(f32, f32)>>, String> {
        let (w, h) = (w as f32, h as f32 * 0.4);
        let q = points / 4;
        let mut o = Vec::new();
        for i in 0..q {
            o.push((w * i as f32 / q as f32, 0.0));
        }
        for i in 0..q {
            o.push((w, h * i as f32 / q as f32));
        }
        for i in 0..q {
            o.push((w - w * i as f32 / q as f32, h));
        }
        for i in 0..q {
            o.push((0.0, h - h * i as f32 / q as f32));
        }
        Ok(Some(o))
    }
}

fn session() -> (Session, ClipId) {
    let dir = tmp_dir("sky");
    let movie = dir.join("street.mp4");
    make_movie(&movie, DemoScene::Aurora, 320, 180, 36);
    let (mut s, _, _) = session_with(&[&movie]);
    s.cutouts.sky = Some(Arc::new(TopBand));
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    (s, clip)
}

fn clip(s: &Session, c: ClipId) -> TrackItem {
    s.active_sequence().unwrap().find_item(c).unwrap().1.clone()
}

#[test]
fn a_found_sky_is_swapped_above_the_clip_and_the_clip_relit() {
    let (mut s, c) = session();
    let e = s.execute("clip.skySwap", json!({"clip": c.0})).unwrap_err().to_string();
    assert!(e.contains("clip.findSky"), "{e}");
    let r = s.execute("clip.findSky", json!({"clip": c.0, "wait": true})).unwrap();
    assert!(r["outlines"].as_u64().unwrap() >= 5, "{r}");
    let m = sky_mask(&clip(&s, c)).unwrap().clone();
    assert_eq!(m.mode, MaskMode::None);
    let dir = tmp_dir("sky-made");
    let n0 = s.history.undo.len();
    let r = s.execute("clip.skySwap", json!({"clip": c.0, "sky": "sunset", "folder": dir.to_string_lossy()})).unwrap();
    assert_eq!(s.history.undo.len(), n0 + 1, "one undo step");
    let sky = clip(&s, ClipId(r["sky"].as_u64().unwrap()));
    let base = clip(&s, c);
    assert_ne!(sky.item, base.item);
    assert_eq!((sky.start, sky.duration), (base.start, base.duration));
    let sm = &sky.effect("opacity").unwrap().masks;
    assert_eq!((sm.len(), sm[0].mode), (1, MaskMode::Add));
    // the sky's picture made at the clip's size, the clip a little warmer under a sunset
    assert!(std::path::Path::new(r["picture"].as_str().unwrap()).exists());
    assert!(r["relight"]["temperature"].as_f64().unwrap() > 0.0, "{r}");
    assert!(base.effects.iter().any(|e| e.effect == "lumetri"));
    s.execute("playhead.set", json!({"seconds": 0.7})).unwrap();
    assert!(s.render_program(0.5).is_some());
    // a photo of the project as the sky
    s.execute("edit.undo", json!({})).unwrap();
    let photo = dir.join("sky.png");
    std::fs::write(&photo, filmcraft_export::encode_png(made_sky("clouds", 400, 300, false, 2), 400, 300).unwrap()).unwrap();
    let item = s.execute("file.import", json!({"paths": [photo.to_string_lossy()]})).unwrap()["items"][0].as_u64().unwrap();
    let r = s.execute("clip.skySwap", json!({"clip": c.0, "sky": item, "folder": dir.to_string_lossy(), "relight": 0})).unwrap();
    assert_eq!(r["relight"]["temperature"], 0.0);
    // refused
    for p in [json!({"clip": c.0, "sky": "rainbow"}), json!({"clip": c.0, "sky": [1]}), json!({"clip": c.0, "sky": 424_242})] {
        assert!(s.execute("clip.skySwap", p.clone()).is_err(), "{p}");
    }
}
