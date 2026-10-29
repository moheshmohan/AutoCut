//! Remove Object on a movie whose background is known: a textured picture panning 2 px a frame
//! with a disc crossing it, masked frame by frame. Filled from the other frames, the hole is
//! the background; the stand-in painter (the hole's ring colour) shows what painting alone gives.

use std::sync::Arc;

use filmcraft_frame::VideoFrame;
use filmcraft_media::{FrameRequest, MediaInfo, MediaSource};
use filmcraft_project::{ClipId, ItemId, Mask, MaskMode, MaskPath, ParamValue};
use filmcraft_time::{FrameRate, Tick};
use serde_json::json;

use super::*;
use crate::media_test_util::{make_movie_from, session_with, tmp_dir};

const W: usize = 320;
const H: usize = 180;
const FRAMES: i64 = 48;
const R: f64 = 16.0;
/// The camera's pan, pixels a frame.
const PAN: f64 = 2.0;

/// The background at a frame: a texture that pans left.
fn background(x: f64, y: f64, n: i64) -> [u8; 3] {
    background_at(x, y, n, PAN)
}

fn background_at(x: f64, y: f64, n: i64, pan: f64) -> [u8; 3] {
    let u = x + pan * n as f64;
    let a = ((u * 0.11).sin() * (y * 0.07).cos() * 60.0 + 120.0) as u8;
    let b = (((u * 0.031 + y * 0.05).sin() * 0.5 + 0.5) * 140.0 + 40.0) as u8;
    let c = if ((u / 14.0).floor() as i64 + (y / 14.0).floor() as i64) % 2 == 0 { 170 } else { 90 };
    [a, b, c]
}

fn disc_at(n: i64) -> (f64, f64) {
    (60.0 + 4.0 * n as f64, 90.0 + 0.5 * n as f64)
}

/// The movie: the background with the disc (`still`: a still camera and a disc that stays).
struct Scene {
    info: MediaInfo,
    still: bool,
}

impl Scene {
    fn disc(&self, n: i64) -> (f64, f64) {
        if self.still { (160.0, 100.0) } else { disc_at(n) }
    }
    fn pan(&self) -> f64 {
        if self.still { 0.0 } else { PAN }
    }
}

impl MediaSource for Scene {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let n = FrameRate::FPS_24.frame_at(req.time);
        let (cx, cy) = self.disc(n);
        let mut px = vec![0u8; W * H * 4];
        for y in 0..H {
            for x in 0..W {
                let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                let mut c = if (fx - cx).hypot(fy - cy) < R { [230, 30, 40] } else { background_at(fx, fy, n, self.pan()) };
                if self.still {
                    // a camera's grain: ±8 levels, new each frame
                    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                        ^ (n as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
                    h ^= h >> 29;
                    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let g = (h >> 59) as i16 - 16;
                    c = c.map(|v| (i16::from(v) + g / 2).clamp(0, 255) as u8);
                }
                px[(y * W + x) * 4..(y * W + x) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Ok(Arc::new(VideoFrame::rgba8(W as u32, H as u32, px).with_pts(req.time)))
    }
    fn audio(&self, _: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        Ok(filmcraft_frame::AudioBuffer::silence(sample_rate, 2, frames))
    }
}

/// The stand-in painter: the hole the mean colour of what is round it.
struct Ring;

impl Filler for Ring {
    fn fill(&self, rgba: &[u8], w: usize, h: usize, hole: &[u8]) -> std::result::Result<Vec<u8>, String> {
        let mut grown = hole.to_vec();
        grow_by(&mut grown, w, h, 4);
        let (mut acc, mut n) = ([0f64; 3], 0f64);
        for p in 0..w * h {
            if grown[p] > 0 && hole[p] == 0 {
                for c in 0..3 {
                    acc[c] += f64::from(rgba[p * 4 + c]);
                }
                n += 1.0;
            }
        }
        let mut out = rgba.to_vec();
        for p in 0..w * h {
            if hole[p] > 0 {
                for c in 0..3 {
                    out[p * 4 + c] = (acc[c] / n.max(1.0)) as u8;
                }
            }
        }
        Ok(out)
    }
}

/// A session with the movie on V1, its clip selected and masked frame by frame round the disc.
fn session(name: &str) -> (Session, ClipId, ItemId, std::path::PathBuf) {
    use_stand_in(Arc::new(Ring));
    session_of(name, false)
}

fn session_of(name: &str, still: bool) -> (Session, ClipId, ItemId, std::path::PathBuf) {
    let dir = tmp_dir(name);
    let path = dir.join("pan.mp4");
    let rate = FrameRate::FPS_24;
    let demo = filmcraft_media::generators::GeneratorSource::new(
        filmcraft_media::Generator::Demo(filmcraft_media::DemoScene::Plasma),
        W as u32,
        H as u32,
        rate,
        rate.tick_of(FRAMES),
    );
    let scene = Scene { info: demo.info().clone(), still };
    let disc: Vec<(f64, f64)> = (0..FRAMES).map(|n| scene.disc(n)).collect();
    make_movie_from(&path, Arc::new(scene));
    let (mut s, items, _) = session_with(&[&path]);
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.state.selection = vec![clip];
    s.edit_sequence("mask", |q, _, _| {
        let (_, it) = q.find_item_mut(clip).unwrap();
        let op = it.effect_mut("opacity").unwrap();
        let mut m = Mask::new("Object (1)", MaskPath::ellipse(filmcraft_geom::Vec2::new(1.0, 1.0), filmcraft_geom::Vec2::new(1.0, 1.0)));
        for n in 0..FRAMES {
            let (cx, cy) = disc[n as usize];
            m.path.put_keyframe(
                rate.tick_of(n),
                ParamValue::Path(MaskPath::ellipse(filmcraft_geom::Vec2::new(cx, cy), filmcraft_geom::Vec2::new(R + 1.0, R + 1.0))),
            );
        }
        m.mode = MaskMode::Add;
        m.track_method = TrackMethod::Object;
        op.masks.push(m);
        Ok(())
    })
    .unwrap();
    (s, clip, items[0], dir)
}

/// The item's frame `n` (its copy where it has one).
fn shown(s: &Session, item: ItemId, n: i64) -> Vec<u8> {
    let src = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    src.video_frame(FrameRequest { time: FrameRate::FPS_24.tick_of(n), scale: 1.0 }).unwrap().to_rgba8()
}

/// How far a frame is from the background inside the disc's place (levels, mean).
fn off_background(px: &[u8], n: i64) -> f64 {
    let (cx, cy) = disc_at(n);
    let (mut d, mut k) = (0f64, 0f64);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
            if (fx - cx).hypot(fy - cy) < R {
                let b = background(fx, fy, n);
                for c in 0..3 {
                    d += f64::from(px[(y * W + x) * 4 + c].abs_diff(b[c]));
                }
                k += 3.0;
            }
        }
    }
    d / k.max(1.0)
}

#[test]
fn the_thing_is_taken_out_and_the_background_shows() {
    let (mut s, clip, item, dir) = session("remove-pan");
    let before: Vec<f64> = [8, 24, 40].iter().map(|n| off_background(&shown(&s, item, *n), *n)).collect();
    let undo0 = s.history.undo.len();
    let r = s.execute(CMD, json!({"wait": true})).unwrap();
    assert!(r["outcome"]["made"]["path"].as_str().is_some_and(|p| p.ends_with("pan (object removed).mp4")), "{r}");
    let c = s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.interpret.cleaned.clone()).expect("the copy attached");
    assert_eq!((c.start, c.duration), (Tick::ZERO, FrameRate::FPS_24.tick_of(FRAMES)));
    let (_, it) = s.active_sequence().unwrap().find_item(clip).unwrap();
    assert_eq!(it.effect("opacity").unwrap().masks[0].mode, MaskMode::None, "the mask switched off");
    let after: Vec<f64> = [8, 24, 40].iter().map(|n| off_background(&shown(&s, item, *n), *n)).collect();
    eprintln!("inside the disc's place, levels from the background: before {before:?}, after {after:?}");
    assert!(before.iter().all(|v| *v > 60.0));
    assert!(after.iter().all(|v| *v < 6.0), "the background, from the other frames: {after:?}");
    // steady: frame to frame inside the hole it moves as the background does
    let steps: Vec<f64> = (20..28)
        .map(|n| {
            let (a, b) = (shown(&s, item, n), shown(&s, item, n + 1));
            let (cx, cy) = disc_at(n);
            let (mut d, mut k) = (0f64, 0f64);
            for y in 0..H {
                for x in 0..W {
                    let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                    if (fx - cx).hypot(fy - cy) < R && x + 2 < W {
                        // the background moved 2 px left: compare with the next frame 2 px left
                        for ch in 0..3 {
                            d += f64::from(a[(y * W + x) * 4 + ch].abs_diff(b[(y * W + x - 2) * 4 + ch]));
                        }
                        k += 3.0;
                    }
                }
            }
            d / k.max(1.0)
        })
        .collect();
    eprintln!("the hole against the next frame moved with the background: {steps:?}");
    assert!(steps.iter().all(|v| *v < 8.0), "{steps:?}");
    // one undo step; off
    assert_eq!(s.history.undo.len(), undo0 + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.item(item).and_then(|i| i.as_media()).is_some_and(|m| m.interpret.cleaned.is_none()));
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(s.execute(CMD, json!({"off": true})).unwrap()["off"], json!([item.0]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn painting_alone_is_worse_than_the_other_frames() {
    let (mut s, _, item, dir) = session("remove-model");
    s.execute(CMD, json!({"wait": true, "fill": "model"})).unwrap();
    let painted: Vec<f64> = [8, 24, 40].iter().map(|n| off_background(&shown(&s, item, *n), *n)).collect();
    eprintln!("painted in every frame: {painted:?} levels from the background");
    assert!(painted.iter().all(|v| *v > 12.0), "{painted:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A thing that never moves on a still shot: no frame shows what is behind it, so all of it is
/// painted — in key frames, carried between them — not left as it was.
#[test]
fn a_thing_that_stays_is_painted_steadily() {
    use_stand_in(Arc::new(Ring));
    let mut flick = Vec::new();
    for fill in ["auto", "model"] {
        let (mut s, _, item, dir) = session_of(&format!("remove-stays-{fill}"), true);
        s.execute(CMD, json!({"wait": true, "fill": fill})).unwrap();
        let px = shown(&s, item, 24);
        let p = (100 * W + 160) * 4;
        assert!(px[p] < 200 || px[p + 1] > 60, "the disc's red is painted over: {:?}", &px[p..p + 3]);
        flick.push(flicker(&s, item, (160.0, 100.0), 1, FRAMES - 2));
        let _ = std::fs::remove_dir_all(&dir);
    }
    eprintln!("frame to frame in the hole: auto {:.2}, painted every frame {:.2}", flick[0], flick[1]);
    assert!(flick[0] <= flick[1] + 0.5, "{flick:?}");
}

#[test]
fn hostile_parameters_and_unfit_media_are_refused() {
    let (mut s, clip, item, dir) = session("remove-hostile");
    for p in [
        json!({"grow": -1}),
        json!({"grow": 1e9}),
        json!({"feather": "soft"}),
        json!({"fill": "magic"}),
        json!({"format": "gif"}),
        json!({"mask": 9}),
        json!({"mask": "Nope"}),
        json!({"mask": [1]}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // no mask: told to mask first
    s.edit_sequence("unmask", |q, _, _| {
        let (_, it) = q.find_item_mut(clip).unwrap();
        it.effect_mut("opacity").unwrap().masks.clear();
        Ok(())
    })
    .unwrap();
    s.state.selected_mask = None;
    let e = s.execute(CMD, json!({})).unwrap_err().to_string();
    assert!(e.contains("Object Mask"), "{e}");
    // one copy at a time
    s.edit("sharper", |pr, _| {
        if let Some(ItemKind::Media(m)) = pr.item_mut(item).map(|it| &mut it.kind) {
            m.interpret.upscaled = Some(filmcraft_project::Upscaled { path: "/nowhere.mp4".into(), factor: 2, strength: 1.0 });
        }
        Ok(())
    })
    .unwrap();
    assert!(s.execute(CMD, json!({})).unwrap_err().to_string().contains("copy already"));
    let st = s.execute(STATUS_CMD, json!({})).unwrap();
    assert_eq!(st["bytes"], json!(29_546_882));
    assert!(st["license"].as_str().unwrap().starts_with("MIT"));
    assert!(st["training"].as_str().unwrap().contains("Places2"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn coverage_grows_by_a_square() {
    let mut m = vec![0u8; 25];
    m[12] = 255;
    grow_by(&mut m, 5, 5, 1);
    assert_eq!(m.iter().filter(|v| **v == 255).count(), 9);
    grow_by(&mut m, 5, 5, 0);
    assert_eq!(m.iter().filter(|v| **v == 255).count(), 9);
    grow_by(&mut m[..3], 5, 5, 2);
    assert!(bilinear(&[], 0, 0, 0.0, 0.0).is_none());
    assert!(bilinear(&[1, 2, 3, 4], 1, 1, f64::NAN, 0.0).is_none());
    assert_eq!(bilinear(&[10, 20, 30, 255], 1, 1, 0.0, 0.0), Some([10.0, 20.0, 30.0]));
}

// ---------------------------------------------------------------------------------------------
// with MI-GAN (downloaded): measured, not run by default
// ---------------------------------------------------------------------------------------------

/// Mean change inside a circle between consecutive frames `from..to` of the item.
fn flicker(s: &Session, item: ItemId, (cx, cy): (f64, f64), from: i64, to: i64) -> f64 {
    let mut total = 0f64;
    for n in from..to {
        let (a, b) = (shown(s, item, n), shown(s, item, n + 1));
        let (mut d, mut k) = (0f64, 0f64);
        for y in 0..H {
            for x in 0..W {
                if (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy) < R {
                    for c in 0..3 {
                        d += f64::from(a[(y * W + x) * 4 + c].abs_diff(b[(y * W + x) * 4 + c]));
                    }
                    k += 3.0;
                }
            }
        }
        total += d / k.max(1.0);
    }
    total / (to - from).max(1) as f64
}

/// A still shot: the disc never moves, so no frame shows what is behind it and MI-GAN paints all
/// of it — in key frames carried to the frames between (`auto`) or in every frame (`model`).
#[test]
#[ignore = "needs MI-GAN downloaded (FILMCRAFT_DATA_DIR naming the data directory); measures"]
fn mi_gan_paints_a_still_shot_steadily() {
    *lock(&STAND_IN) = None;
    for fill in ["auto", "model"] {
        let (mut s, _, item, dir) = session_of(&format!("remove-still-{fill}"), true);
        let t0 = std::time::Instant::now();
        s.execute(CMD, json!({"wait": true, "fill": fill})).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let off: f64 = [8i64, 24, 40]
            .iter()
            .map(|n| {
                let px = shown(&s, item, *n);
                let (mut d, mut k) = (0f64, 0f64);
                for y in 0..H {
                    for x in 0..W {
                        let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                        if (fx - 160.0).hypot(fy - 100.0) < R {
                            let b = background_at(fx, fy, *n, 0.0);
                            for c in 0..3 {
                                d += f64::from(px[(y * W + x) * 4 + c].abs_diff(b[c]));
                            }
                            k += 3.0;
                        }
                    }
                }
                d / k.max(1.0)
            })
            .sum::<f64>()
            / 3.0;
        let f = flicker(&s, item, (160.0, 100.0), 1, FRAMES - 2);
        eprintln!("{fill}: {secs:.1} s for {FRAMES} frames; in the hole {off:.1} levels from the background, {f:.2} levels of change frame to frame");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Object Mask then Remove Object on a real clip: `REMOVE_CLIP` (a file), `REMOVE_POINT` (`x,y`,
/// 0..1, on the thing at `REMOVE_AT` seconds), the clip cut to `REMOVE_SECONDS` (3) from
/// `REMOVE_FROM` (0), `REMOVE_OUT` a folder for the strip: four frames as they were (the mask
/// drawn) over the same four with the thing removed.
#[test]
#[ignore = "needs MobileSAM and MI-GAN downloaded (FILMCRAFT_DATA_DIR naming the data directory) and a clip; measures"]
fn remove_object_on_a_real_clip() {
    *lock(&STAND_IN) = None;
    let Some(clip) = std::env::var_os("REMOVE_CLIP").map(std::path::PathBuf::from) else { return };
    let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let point: Vec<f64> = std::env::var("REMOVE_POINT").unwrap_or("0.5,0.5".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let (from, long, at) = (num("REMOVE_FROM", 0.0), num("REMOVE_SECONDS", 3.0), num("REMOVE_AT", 0.0));
    let out = std::env::var_os("REMOVE_OUT").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let (mut s, items, _) = session_with(&[&clip]);
    let item = items[0];
    let c = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.edit_sequence("cut", |q, _, _| {
        let (_, it) = q.find_item_mut(c).unwrap();
        it.source_in += Tick::from_seconds_f64(from);
        it.duration = it.duration.min(Tick::from_seconds_f64(long));
        Ok(())
    })
    .unwrap();
    s.state.selection = vec![c];
    let t0 = std::time::Instant::now();
    s.execute("clip.objectMask", json!({"point": point, "at": at, "wait": true})).unwrap();
    let t_mask = t0.elapsed().as_secs_f64();
    let it = s.active_sequence().unwrap().find_item(c).unwrap().1.clone();
    let mask = it.effect("opacity").unwrap().masks.last().unwrap().clone();
    let original = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    let t0 = std::time::Instant::now();
    let mut params = json!({"wait": true, "folder": out.join("copies")});
    if let Ok(g) = std::env::var("REMOVE_GROW") {
        params["grow"] = json!(g.parse::<f64>().unwrap_or(8.0));
    }
    if let Ok(f) = std::env::var("REMOVE_FILL") {
        params["fill"] = json!(f);
    }
    let r = s.execute(CMD, params).unwrap();
    let t_remove = t0.elapsed().as_secs_f64();
    let frames = r["copy"]["frames"].as_i64().unwrap_or(0);
    eprintln!(
        "Object Mask {t_mask:.1} s ({} outlines); Remove Object {t_remove:.1} s for {frames} frames ({:.2} s a frame)",
        mask.path.keyframes.len(),
        t_remove / frames.max(1) as f64
    );
    let cleaned = s.media.full_res_source(&s.project, item, &*s.services).unwrap();
    // how much the hole changes from frame to frame (the camera's motion in it too)
    let rate = s.project.item(item).unwrap().frame_rate();
    let (fa, fb) = (rate.frame_at(it.source_time_at(it.start)), rate.frame_at(it.source_time_at(it.end() - Tick(1))));
    let size0 = filmcraft_render::clip_size(&s.project, &it).unwrap();
    let mut change = Vec::new();
    for n in (fa + 1..fb - 1).step_by(3) {
        let (a, b) = (
            cleaned.video_frame(FrameRequest { time: rate.tick_of(n), scale: 1.0 }).unwrap(),
            cleaned.video_frame(FrameRequest { time: rate.tick_of(n + 1), scale: 1.0 }).unwrap(),
        );
        let (w, h) = (a.width as usize, a.height as usize);
        let (pa, pb) = (a.to_rgba8(), b.to_rgba8());
        let k = w as f64 / f64::from(size0.0);
        let pts: Vec<[f32; 2]> = mask.path_at(rate.tick_of(n)).flatten(0.5).iter().map(|v| [(v.x * k) as f32, (v.y * k) as f32]).collect();
        let fm = filmcraft_render::mask::FlatMask { pts, feather: 0.0, expansion: 0.0, opacity: 1.0, inverted: false, mode: MaskMode::Add };
        let cov = filmcraft_render::mask::coverage(&[fm], w, h).unwrap_or_default();
        let (mut d, mut c) = (0f64, 0f64);
        for (p, v) in cov.iter().enumerate() {
            if *v > 0.5 {
                for ch in 0..3 {
                    d += f64::from(pa[p * 4 + ch].abs_diff(pb[p * 4 + ch]));
                }
                c += 3.0;
            }
        }
        change.push(d / c.max(1.0));
    }
    eprintln!("inside the hole, frame to frame: mean {:.2} levels over {} pairs", change.iter().sum::<f64>() / change.len().max(1) as f64, change.len());
    // the strip: four moments, as they were (the outline drawn) over the thing removed
    let (a, b) = (it.source_time_at(it.start), it.source_time_at(it.end() - Tick(1)));
    let size = filmcraft_render::clip_size(&s.project, &it).unwrap();
    let tw = 480usize;
    let mut tiles: Vec<(Vec<u8>, usize, usize)> = Vec::new();
    for row in 0..2 {
        for i in 0..4 {
            let t = Tick(a.0 + (b.0 - a.0) * i / 3);
            let src = if row == 0 { &original } else { &cleaned };
            let f = src.video_frame(FrameRequest { time: t, scale: 1.0 }).unwrap();
            let (fw, fh) = (f.width as usize, f.height as usize);
            let th = (fh * tw / fw).max(1);
            let (mut px, _, _) = crate::cutout::resize_rgba(&f.to_rgba8(), fw, fh, tw, th);
            if row == 0 {
                let k = tw as f64 / f64::from(size.0);
                for v in mask.path_at(t).flatten(0.5) {
                    let (x, y) = ((v.x * k) as usize, (v.y * k) as usize);
                    if x < tw && y < th {
                        px[(y * tw + x) * 4..(y * tw + x) * 4 + 3].copy_from_slice(&[255, 220, 0]);
                    }
                }
            }
            tiles.push((px, tw, th));
        }
    }
    let th = tiles.iter().map(|t| t.2).max().unwrap_or(1);
    let mut sheet = vec![0u8; tw * 4 * th * 2 * 4];
    for (i, (px, w, h)) in tiles.iter().enumerate() {
        let (ox, oy) = ((i % 4) * tw, (i / 4) * th);
        for y in 0..*h {
            let at = ((oy + y) * tw * 4 + ox) * 4;
            sheet[at..at + w * 4].copy_from_slice(&px[y * w * 4..(y + 1) * w * 4]);
        }
    }
    let name = clip.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let file = out.join(format!("removed-{name}{}.png", std::env::var("REMOVE_TAG").unwrap_or_default()));
    std::fs::write(&file, filmcraft_export::encode_png(sheet, (tw * 4) as u32, (th * 2) as u32).unwrap()).unwrap();
    if let Some(p) = r["outcome"]["made"]["path"].as_str() {
        eprintln!("copy: {p}");
        if std::env::var_os("REMOVE_KEEP").is_none() {
            let _ = std::fs::remove_file(p);
        }
    }
    eprintln!("strip: {}", file.display());
}
