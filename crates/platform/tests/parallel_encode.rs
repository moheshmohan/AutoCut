//! Parallel hardware encoding on VideoToolbox (macOS): two sessions encode alternate segments of
//! one stream (`ParallelEncoder`), joined into one. The joined stream decodes with our own H.264 and
//! HEVC decoders with every frame, in order, at its time; its pictures match a single session's
//! within the encoder's tolerance, with no seam at the joins; an export through the pipeline writes
//! one valid MP4; hostile input gives errors, never a crash. Tests that need a hardware encoder skip
//! without one. (Whether two sessions run faster is measured by `vt_encode_scaling`, not tested.)
#![cfg(target_os = "macos")]

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{
    EncodeSessions, EncodedPacket, EncoderFrame, ExportSettings, Format, HardwareEncoding, Progress, VideoEncoder, export, hw_encode_stats,
};
use filmcraft_frame::PixelData;
use filmcraft_isobmff::{CodecConfig, SampleEntry};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, MediaSource};
use filmcraft_platform::hardware_encode::{ParallelEncoder, videotoolbox_encoder_factory};
use filmcraft_platform::videotoolbox_encode::{VtConfig, VtEncoder, VtProfile, VtRate};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, Project, SequenceSettings, TrackKind};
use filmcraft_render::SourceMap;
use filmcraft_time::{FrameRate, Tick, TimeRange};

const W: u32 = 1280;
const H: u32 = 720;
const KEYINT: u32 = 30;
const FPS: FrameRate = FrameRate::FPS_30;

/// The process-wide session switch is changed by some tests: they take turns.
static SWITCH: Mutex<()> = Mutex::new(());

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Picture `i` of a slow pan over a detailed texture with a moving disc: continuous across the
/// joins, so a seam there would show against a single session's encode.
fn picture(i: u64) -> Planes {
    let (w, h) = (W as usize, H as usize);
    let t = i as f32;
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let (x, yy) = (c as f32 + t * 3.0, r as f32 + t * 0.5);
            let tex = (x * 0.05).sin() * 40.0 + (yy * 0.07).cos() * 35.0 + ((x + yy) * 0.013).sin() * 30.0 + ((x * 0.31).sin() * (yy * 0.27).cos()) * 12.0;
            let (dx, dy) = (c as f32 - (200.0 + t * 6.0) % w as f32, r as f32 - 360.0);
            let disc = if dx * dx + dy * dy < 90.0 * 90.0 { 60.0 } else { 0.0 };
            y[r * w + c] = (120.0 + tex + disc).clamp(16.0, 235.0) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let u: Vec<u8> = (0..cw * ch).map(|k| (128.0 + (((k % cw) as f32 + t) * 0.03).sin() * 30.0) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|k| (128.0 + (((k / cw) as f32 - t) * 0.04).cos() * 30.0) as u8).collect();
    (y, u, v)
}

fn config(profile: VtProfile) -> VtConfig {
    VtConfig {
        width: W,
        height: H,
        timescale: 30,
        frame_duration: 1,
        keyframe_interval: KEYINT,
        profile,
        rate: VtRate::Vbr { target_kbps: 6_000, max_kbps: 9_000 },
        realtime: false,
    }
}

fn frame(i: u64) -> EncoderFrame<'static> {
    EncoderFrame { width: W, height: H, rgba: &[], hdr: None, index: i }
}

/// Encode `n` pictures; the packets in decoding order.
fn encode_all(enc: &mut dyn VideoEncoder, pics: &[Planes]) -> Vec<EncodedPacket> {
    let mut out = Vec::new();
    for (i, (y, u, v)) in pics.iter().enumerate() {
        out.extend(enc.encode_yuv420(&frame(i as u64), y, u, v).expect("encode"));
    }
    out.extend(enc.flush().expect("flush"));
    out
}

/// One offline session (the single-session path), the same pictures: the reference.
fn encode_single(profile: VtProfile, pics: &[Planes]) -> Option<(Vec<EncodedPacket>, SampleEntry)> {
    let s = match profile {
        VtProfile::HevcMain => ExportSettings { format: Format::Hevc, ..Default::default() },
        _ => ExportSettings { format: Format::H264, hardware_encoding: HardwareEncoding::Auto, ..Default::default() },
    };
    let s = ExportSettings { bitrate_kbps: 6_000, max_bitrate_kbps: Some(9_000), keyframe_distance: Some(KEYINT), ..s };
    let _g = SWITCH.lock().unwrap_or_else(PoisonError::into_inner);
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::One);
    let made = videotoolbox_encoder_factory(s.format, W, H, FPS, &s);
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::Auto);
    let mut enc = made?.ok()?;
    assert_eq!(enc.hardware(), Some("VideoToolbox"));
    let packets = encode_all(enc.as_mut(), pics);
    Some((packets, enc.sample_entry()))
}

/// Decode `packets` with our software decoder; the luma planes in presentation order with their times.
fn decode(entry: &SampleEntry, packets: &[EncodedPacket]) -> Vec<(i64, Vec<u8>)> {
    let mut dec = filmcraft_codecs::software_video_decoder(entry).expect("software decoder");
    let mut out = Vec::new();
    let mut dts = 0i64;
    let take = |frames: Vec<filmcraft_codecs::DecodedFrame>, out: &mut Vec<(i64, Vec<u8>)>| {
        for f in frames {
            let PixelData::Yuv8 { planes, .. } = &f.frame.data else { panic!("expected 8-bit planes") };
            out.push((f.pts, planes[0].to_vec()));
        }
    };
    for p in packets {
        take(dec.decode(&p.data, dts + i64::from(p.composition_offset)).expect("decode"), &mut out);
        dts += i64::from(p.duration);
    }
    take(dec.flush(), &mut out);
    out.sort_by_key(|(pts, _)| *pts);
    out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// The joined stream of two sessions against one session's stream of the same pictures.
fn joined_stream_matches_one_session(profile: VtProfile, name: &str) {
    let n = 5 * u64::from(KEYINT) + 7;
    let pics: Vec<Planes> = (0..n).map(picture).collect();
    let mut par = match ParallelEncoder::new(config(profile), 2) {
        Ok(p) => p,
        Err(why) => {
            eprintln!("SKIPPED {name}: no hardware encoder ({why})");
            return;
        }
    };
    assert_eq!(par.lanes(), 2);
    // the export's report names the encoder from this: several sessions are still VideoToolbox
    assert_eq!(par.hardware(), Some("VideoToolbox"), "{name}");
    let packets = encode_all(&mut par, &pics);
    let entry = par.sample_entry();
    match (&entry.codec, profile) {
        (CodecConfig::Avc(avc), VtProfile::High) => assert!(!avc.sps.is_empty() && !avc.pps.is_empty() && avc.length_size == 4),
        (CodecConfig::Hevc(_), VtProfile::HevcMain) => {}
        (c, _) => panic!("{name}: unexpected sample entry {}", c.name()),
    }
    assert_eq!(par.lanes(), 2, "{name}: both sessions made the same parameter sets");

    // one sample per picture, a frame long, no reordering, a keyframe at every join
    assert_eq!(packets.len() as u64, n, "{name}");
    assert!(packets.iter().all(|p| p.duration == 1 && p.composition_offset == 0 && !p.data.is_empty()), "{name}");
    for (i, p) in packets.iter().enumerate() {
        if (i as u64).is_multiple_of(u64::from(KEYINT)) {
            assert!(p.key, "{name}: frame {i} starts a segment and must be a keyframe");
        }
    }
    let keys: Vec<usize> = packets.iter().enumerate().filter(|(_, p)| p.key).map(|(i, _)| i).collect();
    assert!(keys.windows(2).all(|w| w[1] - w[0] <= KEYINT as usize), "{name}: keyframes at {keys:?}");

    // our decoder gives every picture, in order, at its time
    let decoded = decode(&entry, &packets);
    assert_eq!(decoded.iter().map(|(pts, _)| *pts).collect::<Vec<_>>(), (0..n as i64).collect::<Vec<_>>(), "{name}");
    let ps: Vec<f64> = decoded.iter().zip(&pics).map(|((_, luma), (y, _, _))| psnr(luma, y)).collect();

    // ... at the quality of one session, with no seam at the joins
    let Some((single, single_entry)) = encode_single(profile, &pics) else {
        eprintln!("SKIPPED {name}: no single-session reference");
        return;
    };
    let reference: Vec<f64> = decode(&single_entry, &single).iter().zip(&pics).map(|((_, luma), (y, _, _))| psnr(luma, y)).collect();
    let (m, m1) = (mean(&ps), mean(&reference));
    let bytes = |p: &[EncodedPacket]| p.iter().map(|p| p.data.len()).sum::<usize>();
    eprintln!("{name}: two sessions {m:.2} dB mean luma PSNR, {} bytes; one session {m1:.2} dB, {} bytes", bytes(&packets), bytes(&single));
    assert!(m > 32.0, "{name}: mean PSNR {m:.2} dB");
    assert!(m > m1 - 1.0, "{name}: two sessions {m:.2} dB against one session's {m1:.2} dB");
    let size = bytes(&packets) as f64 / bytes(&single) as f64;
    assert!((0.8..1.25).contains(&size), "{name}: the same bitrate ({size:.2}× one session's bytes)");
    let mut seams = Vec::new();
    for join in (u64::from(KEYINT)..n).step_by(KEYINT as usize) {
        let j = join as usize;
        let around = |v: &[f64]| mean(&v[j..(j + 3).min(v.len())]);
        let (at, at1, before) = (around(&ps), around(&reference), mean(&ps[j - 3..j]));
        eprintln!("{name}: join at {join}: {before:.2} dB before, {at:.2} dB after (one session: {at1:.2} dB)");
        if at < at1 - 3.0 || at < m - 4.0 {
            seams.push(format!("join at {join}: {at:.2} dB after it, one session {at1:.2} dB there, {m:.2} dB on average"));
        }
    }
    assert!(seams.is_empty(), "{name}: seams {seams:?}");
}

#[test]
fn h264_joined_stream_matches_one_session() {
    joined_stream_matches_one_session(VtProfile::High, "H.264");
}

#[test]
fn hevc_joined_stream_matches_one_session() {
    joined_stream_matches_one_session(VtProfile::HevcMain, "HEVC");
}

/// An animated demo clip as the whole sequence.
fn project(frames: i64) -> (Arc<Project>, filmcraft_project::ItemId, SourceMap) {
    let mut p = Project::new("parallel");
    let g = GeneratorSource::demo(DemoScene::Plasma);
    let info = g.info().clone();
    let item = p.add_item(
        &info.name.clone(),
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
    let seq = p.new_sequence("s", SequenceSettings { width: W, height: H, frame_rate: FPS, ..Default::default() }, 1, 1, None);
    let mut clip = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, FPS.tick_of(frames)), FPS).unwrap();
    clip.scale_to_frame = true;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(clip);
    let mut m = SourceMap::default();
    m.0.insert(item, Arc::new(g));
    (Arc::new(p), seq, m)
}

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("fc-parallel-encode-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

/// The video track of an MP4 checked as a player reads it: one sample per frame, each a frame
/// long and presented at its decode time, keyframes (sync samples) at every segment start, the
/// sample entry the stream's, and every sample decoded by our decoder in order.
fn check_file(path: &str, frames: usize, name: &str) {
    let bytes = std::fs::read(path).unwrap();
    let file = filmcraft_isobmff::open(bytes.as_slice()).unwrap_or_else(|e| panic!("{name}: the container does not parse: {e}"));
    let vt = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
    let track = &file.tracks[vt];
    assert_eq!(track.entries.len(), 1, "{name}: one sample description");
    assert_eq!(track.samples.len(), frames, "{name}");
    for (i, s) in track.samples.iter().enumerate() {
        assert_eq!((s.dts, s.pts, s.duration, s.description_index), (i as i64, i as i64, 1, 1), "{name}: sample {i}");
        if i.is_multiple_of(KEYINT as usize) {
            assert!(s.is_sync, "{name}: sample {i} starts a segment");
        }
    }
    let entry = &track.entries[0];
    let mut dec = filmcraft_codecs::software_video_decoder(entry).expect("software decoder");
    let mut pts = Vec::new();
    for (i, s) in track.samples.iter().enumerate() {
        let data = file.read_sample(bytes.as_slice(), vt, i).unwrap();
        pts.extend(dec.decode(&data, s.pts).expect("decode").into_iter().map(|f| f.pts));
    }
    pts.extend(dec.flush().into_iter().map(|f| f.pts));
    assert_eq!(pts, (0..frames as i64).collect::<Vec<_>>(), "{name}: every picture decodes, in order");
}

#[test]
fn an_export_with_two_sessions_writes_one_valid_file() {
    filmcraft_platform::register();
    if VtEncoder::new(config(VtProfile::High)).is_err() {
        eprintln!("SKIPPED: no hardware H.264 encoder");
        return;
    }
    let _g = SWITCH.lock().unwrap_or_else(PoisonError::into_inner);
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::Two);
    let frames = 4 * i64::from(KEYINT) + 11;
    let (p, seq, m) = project(frames);
    let base = ExportSettings {
        format: Format::H264,
        hardware_encoding: HardwareEncoding::Auto,
        bitrate_kbps: 6_000,
        keyframe_distance: Some(KEYINT),
        ..Default::default()
    };
    // with and without sound (its chunks go to the file while the video waits for the other session)
    let cases = [
        ("H.264", ExportSettings { include_audio: false, ..base.clone() }),
        ("H.264 with AAC", base.clone()),
        ("HEVC", ExportSettings { format: Format::Hevc, include_audio: false, ..base.clone() }),
    ];
    let mut outcome = Ok(());
    for (name, s) in cases {
        let path = tmp(&format!("{}.mp4", name.replace(' ', "-")));
        let s = ExportSettings { path: path.clone(), ..s };
        let before = hw_encode_stats();
        let run = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let report = export(&p, seq, &s, &m, &Progress::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(report.frames, frames as u64, "{name}");
            assert!(hw_encode_stats().parallel > before.parallel, "{name}: two sessions encoded the export");
            check_file(&path, frames as usize, name);
        }));
        if run.is_err() {
            outcome = Err(name);
        }
    }
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::Auto);
    assert_eq!(outcome, Ok(()), "a case failed (see above)");
}

/// With `One`, and where two sessions do not pay (`Auto`, a picture this small), an export is
/// encoded by one session as before.
#[test]
fn one_session_unless_asked_or_it_pays() {
    let s = ExportSettings { format: Format::H264, hardware_encoding: HardwareEncoding::Auto, keyframe_distance: Some(KEYINT), ..Default::default() };
    if VtEncoder::new(config(VtProfile::High)).is_err() {
        eprintln!("SKIPPED: no hardware H.264 encoder");
        return;
    }
    let _g = SWITCH.lock().unwrap_or_else(PoisonError::into_inner);
    for (sessions, size, parallel) in [
        (EncodeSessions::One, (3840, 2160), false),
        (EncodeSessions::Auto, (W, H), false),
        (EncodeSessions::Two, (W, H), true),
        (EncodeSessions::Two, (640, 360), true),
    ] {
        filmcraft_export::set_hw_encode_sessions(sessions);
        let before = hw_encode_stats().parallel;
        let enc = videotoolbox_encoder_factory(Format::H264, size.0, size.1, FPS, &s);
        assert!(matches!(enc, Some(Ok(_))), "{sessions:?} {size:?}: a hardware encoder");
        assert_eq!(hw_encode_stats().parallel > before, parallel, "{sessions:?} {size:?}");
    }
    // a keyframe on every picture leaves nothing to cut into segments
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::Two);
    let before = hw_encode_stats().parallel;
    let _ = videotoolbox_encoder_factory(Format::H264, W, H, FPS, &ExportSettings { keyframe_distance: Some(1), ..s });
    assert_eq!(hw_encode_stats().parallel, before);
    filmcraft_export::set_hw_encode_sessions(EncodeSessions::Auto);
}

/// Run `f` and require that it does not panic.
fn calm<R>(what: &str, f: impl FnOnce() -> R) -> R {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| panic!("{what} panicked"))
}

#[test]
fn hostile_input_gives_errors_not_crashes() {
    if VtEncoder::new(config(VtProfile::High)).is_err() {
        eprintln!("SKIPPED: no hardware H.264 encoder");
        return;
    }
    let base = config(VtProfile::High);
    for (what, c, lanes) in [
        ("zero size", VtConfig { width: 0, height: 0, ..base.clone() }, 2),
        ("odd size", VtConfig { width: 321, ..base.clone() }, 2),
        ("huge size", VtConfig { width: u32::MAX, height: u32::MAX, ..base.clone() }, 2),
        ("zero frame duration", VtConfig { frame_duration: 0, ..base.clone() }, 2),
        ("zero keyframe interval", VtConfig { keyframe_interval: 0, ..base.clone() }, 2),
        ("keyframe on every picture", VtConfig { keyframe_interval: 1, ..base.clone() }, 2),
        ("no sessions", base.clone(), 0),
    ] {
        let made = calm(what, || ParallelEncoder::new(c, lanes));
        if what == "no sessions" {
            // asked for none: one session
            assert_eq!(made.map(|p| p.lanes()).ok(), Some(1), "{what}");
        } else {
            assert!(made.is_err(), "{what} must be refused");
        }
    }
    let ok = huge_keyint_ok(&base);
    assert!(ok, "a keyframe interval far longer than the stream is fine");

    let (y, u, v) = picture(0);
    let mut enc = ParallelEncoder::new(base.clone(), 2).unwrap();
    // planes of the wrong size, a picture of another size, an HDR picture
    assert!(calm("short planes", || enc.encode_yuv420(&frame(0), &y[..100], &u, &v)).is_err());
    assert!(calm("short chroma", || enc.encode_yuv420(&frame(0), &y, &u[..10], &v)).is_err());
    let other = EncoderFrame { width: 640, height: 360, rgba: &[], hdr: None, index: 0 };
    assert!(calm("other size", || enc.encode_yuv420(&other, &y, &u, &v)).is_err());
    let rgba = vec![0u8; 16];
    assert!(calm("short RGBA", || enc.encode(&EncoderFrame { width: W, height: H, rgba: &rgba, hdr: None, index: 0 })).is_err());
    // pictures out of order: an error, and the encoder stays failed
    let mut enc = ParallelEncoder::new(base.clone(), 2).unwrap();
    assert!(calm("in order", || enc.encode_yuv420(&frame(0), &y, &u, &v)).is_ok());
    assert!(calm("skipped picture", || enc.encode_yuv420(&frame(5), &y, &u, &v)).is_err());
    assert!(calm("after the error", || enc.encode_yuv420(&frame(1), &y, &u, &v)).is_err());
    assert!(calm("flush after the error", || enc.flush()).is_err());
    // flushing twice, encoding after the end, dropping mid-stream
    let mut enc = ParallelEncoder::new(base.clone(), 2).unwrap();
    let mut all = 0;
    for i in 0..(u64::from(KEYINT) + 3) {
        all += enc.encode_yuv420(&frame(i), &y, &u, &v).unwrap().len();
    }
    all += enc.flush().unwrap().len();
    assert_eq!(all as u32, KEYINT + 3);
    assert!(calm("second flush", || enc.flush()).unwrap().is_empty());
    assert!(calm("after the end", || enc.encode_yuv420(&frame(u64::from(KEYINT) + 3), &y, &u, &v)).is_err());
    let mut enc = ParallelEncoder::new(base, 2).unwrap();
    for i in 0..(u64::from(KEYINT) + 3) {
        enc.encode_yuv420(&frame(i), &y, &u, &v).unwrap();
    }
    calm("drop mid-stream", || drop(enc));
}

/// A keyframe interval of `u32::MAX`: one segment, one session busy, every frame out.
fn huge_keyint_ok(base: &VtConfig) -> bool {
    let Ok(mut enc) = ParallelEncoder::new(VtConfig { keyframe_interval: u32::MAX, ..base.clone() }, 2) else { return false };
    let (y, u, v) = picture(0);
    let mut n = 0;
    for i in 0..5 {
        n += enc.encode_yuv420(&frame(i), &y, &u, &v).map(|p| p.len()).unwrap_or(0);
    }
    n += enc.flush().map(|p| p.len()).unwrap_or(0);
    n == 5
}
