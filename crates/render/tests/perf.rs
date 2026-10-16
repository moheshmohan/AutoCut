//! Per-frame cost of the frames of a phone-footage showcase (AutoCut's Robot Showcase) on the CPU
//! compositor at 1920×1080: a portrait clip fitted over a filled, blurred copy of itself, and the
//! 10-bit 4:4:4 HLG BT.2020 pictures that Google Photos hands out for iPhone HDR video, alone and
//! as a portrait pair. Synthetic in-memory sources, so decoding is not part of it. Ignored by
//! default; run with
//!
//!     cargo test --release -p filmcraft-render --test perf -- --ignored --nocapture
//!
//! Prints, for `render_sequence` plus the 8-bit conversion an export does, the median wall time of
//! 20 frames (after 3 warm-up frames) on the global rayon pool, and the median CPU time of a frame
//! rendered on one thread (the work itself: steadier than wall time on a busy machine).

use std::sync::Arc;
use std::time::Instant;

use filmcraft_color::{ColorInfo, Matrix, Primaries, Range, Transfer};
use filmcraft_frame::{AudioBuffer, Chroma, PixelData, VideoFrame};
use filmcraft_media::{FrameRequest, MediaInfo, MediaKind, MediaSource, VideoStreamInfo};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, ParamValue, Project, SequenceSettings, TrackKind, find_effect};
use filmcraft_render::{RenderOptions, SourceMap, render_sequence};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

const FPS: FrameRate = FrameRate::FPS_30;

struct Still {
    info: MediaInfo,
    frame: Arc<VideoFrame>,
}

impl MediaSource for Still {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, _req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        Ok(self.frame.clone())
    }
    fn audio(&self, _start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        Ok(AudioBuffer::silence(sample_rate, 2, frames))
    }
}

/// An 8-bit 4:2:0 BT.709 phone picture with texture (`w`×`h`).
fn camera(w: usize, h: usize) -> Arc<VideoFrame> {
    let y: Vec<u8> = (0..w * h).map(|i| (16 + ((i % w) * 200 / w + (i / w) * 30 / h + (i * 7919) % 23) % 219) as u8).collect();
    let (cw, ch) = (w / 2, h / 2);
    let u: Vec<u8> = (0..cw * ch).map(|i| (100 + (i % cw) * 40 / cw) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| (150 - (i / cw) * 40 / ch) as u8).collect();
    Arc::new(VideoFrame {
        width: w as u32,
        height: h as u32,
        data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, alpha: None },
        color: ColorInfo::REC709,
        par: (1, 1),
        pts: Tick::ZERO,
    })
}

/// A 10-bit 4:4:4 HLG BT.2020 picture (`w`×`h`), as Google Photos serves iPhone HDR video.
fn hlg(w: usize, h: usize) -> Arc<VideoFrame> {
    let n = w * h;
    let y: Vec<u16> = (0..n).map(|i| (64 + ((i % w) * 800 / w + (i * 7919) % 97) % 876) as u16).collect();
    let u: Vec<u16> = (0..n).map(|i| (400 + (i % w) * 200 / w) as u16).collect();
    let v: Vec<u16> = (0..n).map(|i| (600 - (i / w) * 200 / h) as u16).collect();
    Arc::new(VideoFrame {
        width: w as u32,
        height: h as u32,
        data: PixelData::Yuv16 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C444, bits: 10, alpha: None },
        color: ColorInfo { matrix: Matrix::Bt2020Ncl, transfer: Transfer::Hlg, primaries: Primaries::Bt2020, range: Range::Limited },
        par: (1, 1),
        pts: Tick::ZERO,
    })
}

/// One sequence: each entry is a clip of `frame` on its own track (V1 up), with a Motion scale (%)
/// and an optional Gaussian blurriness and opacity, as AutoCut builds its shots.
fn project(clips: &[(Arc<VideoFrame>, f64, Option<f64>, f64)]) -> (Project, filmcraft_project::ItemId, SourceMap) {
    let mut p = Project::new("bench");
    let mut map = SourceMap::default();
    let seq = p.new_sequence("s", SequenceSettings { width: 1920, height: 1080, frame_rate: FPS, ..Default::default() }, clips.len(), 1, None);
    for (track, (frame, scale, blur, opacity)) in clips.iter().enumerate() {
        let (w, h) = (frame.width, frame.height);
        let info = MediaInfo {
            name: format!("clip{track}.mov"),
            kind: MediaKind::Movie,
            duration: Tick(600 * TICKS_PER_SECOND),
            video: Some(VideoStreamInfo {
                width: w,
                height: h,
                frame_rate: FPS,
                par: (1, 1),
                codec: "bench".into(),
                pixel_format: String::new(),
                color: frame.color,
                has_alpha: false,
                bitrate: None,
                hdr: None,
            }),
            audio_streams: Vec::new(),
            container: "bench".into(),
            start_timecode: None,
            file_size: None,
        };
        let item = p.add_item(
            &info.name.clone(),
            Label::Iris,
            ItemKind::Media(MediaClip {
                media: MediaRef::File { path: info.name.clone() },
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
        let mut clip = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, FPS.tick_of(2000)), FPS).expect("clip");
        for e in clip.effects.iter_mut() {
            filmcraft_project::resolve_auto_points(e, (w, h), (1920, 1080));
        }
        if let Some(m) = clip.effect_mut("motion").and_then(|m| m.param_mut("scale")) {
            m.value = ParamValue::Float(*scale);
        }
        if let Some(o) = clip.effect_mut("opacity").and_then(|m| m.param_mut("opacity")) {
            o.value = ParamValue::Float(*opacity);
        }
        if let Some(b) = blur {
            let mut fx = find_effect("gaussian_blur").expect("gaussian blur").instance();
            if let Some(v) = fx.param_mut("blurriness") {
                v.value = ParamValue::Float(*b);
            }
            clip.effects.insert(0, fx);
        }
        if let Some(t) = p.sequence_mut(seq).and_then(|s| s.video_tracks.get_mut(track)) {
            t.items.push(clip);
        }
        map.0.insert(item, Arc::new(Still { info, frame: frame.clone() }));
    }
    (p, seq, map)
}

/// CPU time of the calling thread, seconds.
fn thread_cpu_secs() -> Option<f64> {
    #[cfg(unix)]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Some(t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Median wall time per call of `f` (ms) on the global rayon pool, after 3 warm-up calls.
fn wall_ms(mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut v: Vec<f64> = (0..20)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Median CPU time per call of `f` (ms) on a pool of one thread, where all of the work runs on the
/// timed thread: the work itself, which other programs competing for the cores barely change
/// (wall time on a busy machine does).
fn cpu_ms(f: impl Fn() + Send + Sync) -> Option<f64> {
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().ok()?;
    one.install(|| {
        f();
        let mut v = Vec::new();
        for _ in 0..10 {
            let t0 = thread_cpu_secs()?;
            f();
            v.push((thread_cpu_secs()? - t0) * 1000.0);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        Some(v[v.len() / 2])
    })
}

#[test]
#[ignore]
fn perf_showcase_frames() {
    println!("rayon threads: {}", rayon::current_num_threads());
    let fill = 1920.0 / 1080.0 * 100.0;
    let fit = 1080.0 / 1920.0 * 100.0;
    let (portrait, landscape) = (camera(1080, 1920), camera(1920, 1080));
    let (hlg_portrait, hlg_landscape) = (hlg(1080, 1920), hlg(1920, 1080));
    let scenarios: Vec<(&str, Vec<(Arc<VideoFrame>, f64, Option<f64>, f64)>)> = vec![
        ("H.264 landscape clip", vec![(landscape, 100.0, None, 100.0)]),
        ("portrait over its blurred fill", vec![(portrait.clone(), fill, Some(60.0), 45.0), (portrait, fit, None, 100.0)]),
        ("HLG 4:4:4 landscape clip", vec![(hlg_landscape, 100.0, None, 100.0)]),
        ("HLG portrait over its fill", vec![(hlg_portrait.clone(), fill, Some(60.0), 45.0), (hlg_portrait, fit, None, 100.0)]),
    ];
    for (name, clips) in scenarios {
        let (p, seq, map) = project(&clips);
        let opts = RenderOptions::default();
        let frame = || {
            let img = render_sequence(&p, seq, FPS.tick_of(10), opts, &map);
            std::hint::black_box(img.over_black_rgba8());
            filmcraft_frame::pool::recycle_f32(img.px);
        };
        // FILMCRAFT_PERF_LOOP=<part of a name>: render that scenario for 20 s first (to attach a
        // profiler such as `sample`)
        if std::env::var("FILMCRAFT_PERF_LOOP").is_ok_and(|n| name.contains(n.as_str())) {
            let t = Instant::now();
            while t.elapsed().as_secs() < 20 {
                frame();
            }
        }
        let ms = wall_ms(frame);
        let cpu = cpu_ms(frame).map_or_else(|| "-".to_string(), |c| format!("{c:7.2}"));
        println!("{name:32}: wall {ms:7.2} ms/frame ({:5.1} fps), one thread {cpu} ms CPU/frame", 1000.0 / ms);
    }
}
