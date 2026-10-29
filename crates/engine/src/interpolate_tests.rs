use serde_json::json;

use super::*;
use crate::media_test_util::{make_movie, session_with, tmp_dir};

/// The stand-in: a cross-fade by `t` (predictable in-between frames).
struct Fade;

impl Tweener for Fade {
    fn between(&self, a: &[u8], b: &[u8], _: usize, _: usize, t: f32) -> std::result::Result<Vec<u8>, String> {
        Ok(a.iter().zip(b).map(|(x, y)| (f32::from(*x) * (1.0 - t) + f32::from(*y) * t).round() as u8).collect())
    }
}

/// A 12-frame 24 fps movie on V1 of a sequence like it, its clip selected: `(session, item,
/// clip, folder)`.
fn movie_session(name: &str) -> (Session, ItemId, ClipId, std::path::PathBuf) {
    use_stand_in(Arc::new(Fade));
    let dir = tmp_dir(name);
    let path = dir.join("clip.mov");
    make_movie(&path, filmcraft_media::DemoScene::Plasma, 160, 96, 12);
    let (mut s, items, _) = session_with(&[&path]);
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.state.selection = vec![clip];
    (s, items[0], clip, dir)
}

fn interpolated(s: &Session, item: ItemId) -> Option<Interpolated> {
    s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.interpret.interpolated.clone())
}

fn rgba(src: &SharedSource, t: Tick) -> Vec<u8> {
    src.video_frame(FrameRequest { time: t, scale: 1.0 }).unwrap().to_rgba8()
}

#[test]
fn a_slowed_clip_gets_the_frames_between_its_frames() {
    let (mut s, item, clip, dir) = movie_session("interp-slow");
    // at normal speed there is nothing to make
    let e = s.execute(CMD, json!({"wait": true})).unwrap_err().to_string();
    assert!(e.contains("own rate or faster"), "{e}");
    // a quarter speed: four frames for each
    s.execute("clip.speedDuration", json!({"speed": 25.0, "interpolation": "frameSampling"})).unwrap();
    let r = s.execute(CMD, json!({"wait": true})).unwrap();
    assert_eq!(r["items"][0]["factor"], json!(4), "{r}");
    let u = interpolated(&s, item).expect("attached");
    assert_eq!(u.factor, 4);
    assert!(u.path.ends_with("clip (×4 frames).mp4"), "{}", u.path);
    let rate = FrameRate::FPS_24;
    assert_eq!((u.start, u.duration), (Tick::ZERO, rate.tick_of(12)), "the whole clip, a few frames either side within the media");
    // the item gives the made frame between its frames 4 and 5, its own frames at theirs
    let src = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    assert_eq!(src.time_detail(), 4);
    assert_eq!(src.info().frame_rate(), rate, "the item keeps its rate");
    let path = match &s.project.item(item).and_then(|i| i.as_media()).unwrap().media {
        filmcraft_project::MediaRef::File { path } => path.clone(),
        _ => unreachable!(),
    };
    let original = s.media.open_file(&path, &*s.services).unwrap();
    let (f4, f5) = (rgba(&original, rate.tick_of(4)), rgba(&original, rate.tick_of(5)));
    let mid = rgba(&src, rate.tick_of(4) + Tick(rate.frame_duration().0 / 2));
    let want: Vec<u8> = f4.iter().zip(&f5).map(|(a, b)| (u16::from(*a) + u16::from(*b)).div_ceil(2) as u8).collect();
    let off = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| f64::from(x.abs_diff(*y))).sum::<f64>() / a.len() as f64;
    assert!(off(&mid, &want) < 4.0, "the frame halfway is the halfway mix: {}", off(&mid, &want));
    let apart = off(&f4, &f5);
    assert!(apart > 0.5, "the scene moves between frames 4 and 5: {apart}");
    assert!(
        off(&mid, &want) < off(&mid, &f4) && off(&mid, &want) < off(&mid, &f5),
        "not frame 4 or 5 held: {} {} {} ({apart} apart)",
        off(&mid, &want),
        off(&mid, &f4),
        off(&mid, &f5)
    );
    assert!(off(&rgba(&src, rate.tick_of(4)), &f4) < 4.0, "its own frames where they are");
    // one undo step; off
    s.execute("edit.undo", json!({})).unwrap();
    assert!(interpolated(&s, item).is_none());
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(s.execute(CMD, json!({"clips": [clip.0], "off": true})).unwrap()["off"], json!([item.0]));
    assert!(interpolated(&s, item).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn factors_follow_the_slowest_clip_and_the_sequence_rate() {
    assert_eq!(factor_for(1.0, 24.0), None);
    assert_eq!(factor_for(1.04, 24.0), None);
    assert_eq!(factor_for(2.0, 24.0), Some(2));
    assert_eq!(factor_for(2.5, 24.0), Some(3));
    assert_eq!(factor_for(4.0, 30.0), Some(4));
    assert_eq!(factor_for(20.0, 24.0), Some(8), "at most eight");
    assert_eq!(factor_for(20.0, 60.0), Some(4), "at most 240 frames a second");
    assert_eq!(factor_for(f64::NAN, 24.0), None);
    let (mut s, _, _, dir) = movie_session("interp-factor");
    s.execute("clip.speedDuration", json!({"speed": 40.0, "interpolation": "frameSampling"})).unwrap();
    let r = s.execute(CMD, json!({"wait": true})).unwrap();
    assert_eq!(r["items"][0]["factor"], json!(3), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_and_unfit_media_are_refused() {
    let (mut s, item, _, dir) = movie_session("interp-hostile");
    for p in [json!({"factor": 1}), json!({"factor": 9}), json!({"factor": "lots"}), json!({"format": "gif"}), json!({"items": [999_999]})] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    assert!(interpolated(&s, item).is_none());
    // one copy of an item at a time
    crate::upscale::use_stand_in(Arc::new(Noop));
    s.edit("sharper", |pr, _| {
        if let Some(ItemKind::Media(m)) = pr.item_mut(item).map(|it| &mut it.kind) {
            m.interpret.upscaled = Some(filmcraft_project::Upscaled { path: "/nowhere.mp4".into(), factor: 2, strength: 1.0 });
        }
        Ok(())
    })
    .unwrap();
    let e = s.execute(CMD, json!({"items": [item.0], "factor": 2})).unwrap_err().to_string();
    assert!(e.contains("sharper copy"), "{e}");
    let st = s.execute(STATUS_CMD, json!({})).unwrap();
    assert_eq!(st["bytes"], json!(22_869_906));
    assert!(st["license"].as_str().unwrap().starts_with("MIT"));
    let _ = std::fs::remove_dir_all(&dir);
}

struct Noop;

impl crate::upscale::Enlarger for Noop {
    fn enlarge(
        &self,
        rgba: &[u8],
        w: usize,
        h: usize,
        _: f32,
        _: &mut dyn FnMut(usize, usize) -> bool,
    ) -> std::result::Result<(Vec<u8>, usize, usize), String> {
        Ok((rgba.to_vec(), w, h))
    }
}
