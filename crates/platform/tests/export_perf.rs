//! Export speed of a phone-footage showcase (AutoCut's Robot Showcase) with the YouTube 1080p
//! preset, through the whole pipeline: a portrait shot fitted over its filled, blurred copy, a cross
//! dissolve into a 10-bit 4:4:4 HLG BT.2020 shot (what Google Photos serves for iPhone HDR video),
//! a tone under it all, encoded by VideoToolbox (the preset's hardware encoder) and, for
//! comparison, by the built-in encoder. In-memory sources, so decoding is not part of it. Ignored
//! by default; run with
//!
//!     cargo test --release -p filmcraft-platform --test export_perf -- --ignored --nocapture
//!
//! Prints frames per second, the CPU time of the process per frame and the export stages
//! (`perf.stats` `export.stages`). Before this benchmark's fixes the 40 s range of the real
//! showcase it models exported at 12 fps on a busy 14-core M-series Mac.
#![cfg(target_os = "macos")]

use std::sync::Arc;
use std::time::Instant;

use filmcraft_color::{ColorInfo, Matrix, Primaries, Range, Transfer};
use filmcraft_export::{ExportSettings, HardwareEncoding, Progress, Stage, export, hw_encode_stats, stage_times};
use filmcraft_frame::{AudioBuffer, Chroma, PixelData, VideoFrame};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{FrameRequest, Generator, MediaInfo, MediaKind, MediaSource, VideoStreamInfo};
use filmcraft_project::{
    ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, ParamValue, Project, SequenceSettings, TrackKind, Transition, TransitionId, find_effect,
};
use filmcraft_render::SourceMap;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

const FPS: FrameRate = FrameRate::FPS_30;
const SECONDS: i64 = 8;

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

/// An 8-bit 4:2:0 BT.709 phone picture with texture.
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

/// A 10-bit 4:4:4 HLG BT.2020 picture.
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

fn add_media(p: &mut Project, map: &mut SourceMap, name: &str, frame: &Arc<VideoFrame>) -> ItemId {
    let info = MediaInfo {
        name: name.into(),
        kind: MediaKind::Movie,
        duration: Tick(600 * TICKS_PER_SECOND),
        video: Some(VideoStreamInfo {
            width: frame.width,
            height: frame.height,
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
    let clip = MediaClip {
        media: MediaRef::File { path: name.into() },
        info: info.clone(),
        interpret: Default::default(),
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    let item = p.add_item(name, Label::Iris, ItemKind::Media(clip), None);
    map.0.insert(item, Arc::new(Still { info, frame: frame.clone() }));
    item
}

/// Place `item` on video track `track` from `start` for `dur` seconds, with a Motion scale (%) and
/// optionally a Gaussian blurriness and an opacity, as AutoCut builds its shots.
#[allow(clippy::too_many_arguments)]
fn place(
    p: &mut Project,
    seq: ItemId,
    track: usize,
    item: ItemId,
    size: (u32, u32),
    start: i64,
    dur: i64,
    scale: f64,
    blur: Option<f64>,
    opacity: f64,
) -> ClipId {
    let mut clip = p.make_track_item(item, TrackKind::Video, FPS.tick_of(start * 30), TimeRange::new(Tick::ZERO, FPS.tick_of(dur * 30)), FPS).expect("clip");
    for e in clip.effects.iter_mut() {
        filmcraft_project::resolve_auto_points(e, size, (1920, 1080));
    }
    if let Some(s) = clip.effect_mut("motion").and_then(|m| m.param_mut("scale")) {
        s.value = ParamValue::Float(scale);
    }
    if let Some(o) = clip.effect_mut("opacity").and_then(|m| m.param_mut("opacity")) {
        o.value = ParamValue::Float(opacity);
    }
    if let Some(b) = blur {
        let mut fx = find_effect("gaussian_blur").expect("gaussian blur").instance();
        if let Some(v) = fx.param_mut("blurriness") {
            v.value = ParamValue::Float(b);
        }
        clip.effects.insert(0, fx);
    }
    let id = clip.id;
    if let Some(t) = p.sequence_mut(seq).and_then(|s| s.video_tracks.get_mut(track)) {
        t.items.push(clip);
        t.sort();
    }
    id
}

fn showcase() -> (Arc<Project>, ItemId, SourceMap) {
    let mut p = Project::new("showcase bench");
    let mut map = SourceMap::default();
    let seq = p.new_sequence("s", SequenceSettings { width: 1920, height: 1080, frame_rate: FPS, ..Default::default() }, 2, 1, None);
    let (portrait, landscape) = (camera(1080, 1920), hlg(1920, 1080));
    let shot = add_media(&mut p, &mut map, "portrait.mp4", &portrait);
    let hdr = add_media(&mut p, &mut map, "hdr.mov", &landscape);
    let half = SECONDS / 2;
    place(&mut p, seq, 0, shot, (1080, 1920), 0, half, 1920.0 / 1080.0 * 100.0, Some(60.0), 45.0);
    let a = place(&mut p, seq, 1, shot, (1080, 1920), 0, half, 1080.0 / 1920.0 * 100.0, None, 100.0);
    let b = place(&mut p, seq, 1, hdr, (1920, 1080), half, half, 100.0, None, 100.0);
    let tr = Transition {
        id: TransitionId(1),
        effect: find_effect("cross_dissolve").expect("cross dissolve").instance(),
        start: FPS.tick_of(half * 30 - 6),
        duration: FPS.tick_of(12),
        from: Some(a),
        to: Some(b),
        align: Default::default(),
        reverse: false,
        sound: None,
    };
    if let Some(s) = p.sequence_mut(seq) {
        s.video_tracks[1].transitions.push(tr);
    }
    // a tone under the whole show
    let g = GeneratorSource::new(Generator::Tone { hz: 440.0, db: -12.0 }, 1920, 1080, FPS, Tick(600 * TICKS_PER_SECOND));
    let info = g.info().clone();
    let tone = p.add_item(
        "tone",
        Label::Iris,
        ItemKind::Media(MediaClip {
            media: MediaRef::Generator(g.generator.clone()),
            info,
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
    map.0.insert(tone, Arc::new(g));
    let clip = p.make_track_item(tone, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, FPS.tick_of(SECONDS * 30)), FPS).expect("tone clip");
    if let Some(s) = p.sequence_mut(seq) {
        s.audio_tracks[0].items.push(clip);
    }
    (Arc::new(p), seq, map)
}

fn process_cpu_secs() -> f64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

fn run(name: &str, settings: &ExportSettings, p: &Arc<Project>, seq: ItemId, map: &SourceMap) {
    let before = stage_times();
    let hw = hw_encode_stats().frames;
    let (t0, c0) = (Instant::now(), process_cpu_secs());
    let report = export(p, seq, settings, map, &Progress::default()).expect("export");
    let (secs, cpu) = (t0.elapsed().as_secs_f64(), process_cpu_secs() - c0);
    let after = stage_times();
    let stages: Vec<String> = after
        .iter()
        .zip(&before)
        .filter(|((s, _), _)| !matches!(s, Stage::Setup | Stage::Loudness | Stage::Mux | Stage::Finish))
        .map(|((s, a), (_, b))| format!("{} {:.0}", s.name(), a.saturating_sub(*b) as f64 / 1e6))
        .collect();
    let frames = report.frames.max(1) as f64;
    println!(
        "{name:28}: {:5.1} fps, {:6.1} ms CPU/frame, {} hardware frames | stages ms: {}",
        frames / secs,
        cpu * 1000.0 / frames,
        hw_encode_stats().frames - hw,
        stages.join(", ")
    );
}

#[test]
#[ignore]
fn perf_showcase_export() {
    filmcraft_platform::register();
    println!("cores: {}", std::thread::available_parallelism().map_or(1, |n| n.get()));
    let (p, seq, map) = showcase();
    let dir = std::env::temp_dir().join(format!("fc-export-perf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let preset = filmcraft_export::presets::find_builtin("YouTube 1080p Full HD").expect("the YouTube preset").settings;
    let path = |name: &str| dir.join(name).to_string_lossy().to_string();
    // warm up (thread pools, the encoder), then the two encoders
    let warm = ExportSettings { path: path("warm.mp4"), range: Some(TimeRange::new(Tick::ZERO, FPS.tick_of(16))), ..preset.clone() };
    export(&p, seq, &warm, &map, &Progress::default()).expect("warm-up export");
    run("YouTube preset (hardware)", &ExportSettings { path: path("hw.mp4"), ..preset.clone() }, &p, seq, &map);
    run("built-in encoder", &ExportSettings { path: path("sw.mp4"), hardware_encoding: HardwareEncoding::Off, ..preset }, &p, seq, &map);
    let _ = std::fs::remove_dir_all(&dir);
}
