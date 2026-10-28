//! A sharper stand-in (an upscaled copy: [`filmcraft_media::MediaSource::max_scale`] above 1) is
//! asked for as much detail as a clip shows its picture with, up to its own, by the CPU renderer
//! and by the frame plan the GPU compositor draws — at the same place as the plain picture.

use std::sync::atomic::{AtomicU32, Ordering};

use super::*;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{Generator, MediaInfo, MediaSource};
use filmcraft_project::{Label, MediaClip, MediaRef, Param, ParamValue, SequenceSettings, TrackKind};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, TimeRange};

const W: u32 = 64;
const H: u32 = 36;

/// A 64 × 36 picture that has fine detail (a one-pixel checker) only when asked for at more than
/// its size, up to `most` times; it notes the largest scale asked for.
struct Sharp {
    info: MediaInfo,
    most: f32,
    asked: AtomicU32,
}

impl MediaSource for Sharp {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let s = req.scale.clamp(1.0 / 64.0, self.most);
        let prev = f32::from_bits(self.asked.load(Ordering::Relaxed));
        if s > prev {
            self.asked.store(s.to_bits(), Ordering::Relaxed);
        }
        let (w, h) = (((W as f32 * s).round() as u32).max(1), ((H as f32 * s).round() as u32).max(1));
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let g = (x * 200 / w) as u8;
                let fine = if s > 1.0 && (x + y) % 2 == 0 { 60 } else { 0 };
                px[i..i + 4].copy_from_slice(&[100 + fine, g, 80, 255]);
            }
        }
        Ok(Arc::new(VideoFrame::rgba8(w, h, px)))
    }
    fn audio(&self, _: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        Ok(filmcraft_frame::AudioBuffer::silence(sample_rate, 2, frames))
    }
    fn max_scale(&self) -> f32 {
        self.most
    }
}

/// A 640 × 360 sequence showing the 64 × 36 picture ten times as large, from a source that can
/// give `most` times its size.
fn rig(most: f32) -> (Project, ItemId, SourceMap, Arc<Sharp>) {
    let g = GeneratorSource::new(Generator::ColorMatte { color: [0.0, 0.0, 0.0, 1.0] }, W, H, FrameRate::FPS_24, Tick(10 * TICKS_PER_SECOND));
    let info = g.info().clone();
    let mut p = Project::new("sharper");
    let item = p.add_item(
        "small",
        Label::Iris,
        ItemKind::Media(MediaClip {
            media: MediaRef::Generator(g.generator.clone()),
            info: info.clone(),
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        }),
        None,
    );
    let rate = FrameRate::FPS_24;
    let seq = p.new_sequence("s", SequenceSettings { width: 640, height: 360, frame_rate: rate, ..Default::default() }, 1, 0, None);
    let mut ti = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, rate.tick_of(10)), rate).unwrap();
    if let Some(m) = ti.effect_mut("motion") {
        m.params.insert("scale".into(), Param::new(ParamValue::Float(1000.0)));
    }
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(ti);
    let src = Arc::new(Sharp { info, most, asked: AtomicU32::new(0) });
    let mut map = SourceMap::default();
    map.0.insert(item, src.clone() as SharedSource);
    (p, seq, map, src)
}

/// The red channel's mean step between neighbours along the middle row (the checker).
fn detail(img: &Image) -> f32 {
    let y = img.h / 2;
    (100..540).map(|x| (img.get(x + 1, y)[0] - img.get(x, y)[0]).abs()).sum::<f32>() / 440.0
}

#[test]
fn a_sharper_source_is_drawn_with_its_detail_where_the_plain_picture_is() {
    let (p, seq, map, src) = rig(4.0);
    let cpu = render_sequence(&p, seq, Tick::ZERO, RenderOptions::default(), &map);
    assert!((f32::from_bits(src.asked.load(Ordering::Relaxed)) - 4.0).abs() < 1e-6, "asked for as much as it has (ten times shown, four at most)");
    let planned = plan::execute_cpu(&plan::plan_frame(&p, seq, Tick::ZERO, RenderOptions::default(), &map));
    assert_eq!((cpu.w, cpu.h), (planned.w, planned.h));
    let worst = cpu.px.iter().zip(&planned.px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst < 0.02, "the plan draws what the CPU draws: {worst}");
    // the plain picture: no detail, at the same place (its green ramp the same)
    let (p1, seq1, map1, src1) = rig(1.0);
    let plain = render_sequence(&p1, seq1, Tick::ZERO, RenderOptions::default(), &map1);
    assert!((f32::from_bits(src1.asked.load(Ordering::Relaxed)) - 1.0).abs() < 1e-6, "a decoder is asked for its own size at most");
    // (the copy's one-pixel checker shown 2.5 times as large, smoothed between its pixels)
    assert!(detail(&cpu) > 0.015 && detail(&plain) < 1e-4, "{} against {}", detail(&cpu), detail(&plain));
    let ramp = |img: &Image| (0..640).step_by(40).map(|x| img.get(x, 180)[1]).collect::<Vec<f32>>();
    let (a, b) = (ramp(&cpu), ramp(&plain));
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 0.03), "{a:?} against {b:?}");
    // hostile scales stay within the source's
    assert_eq!(frame_scale(f64::NAN, None), 1.0);
    assert_eq!(frame_scale(100.0, None), 1.0);
    assert_eq!(frame_scale(1e-9, None), 1.0 / 64.0);
    let s: SharedSource = src;
    assert_eq!(frame_scale(100.0, Some(&s)), 4.0);
}
