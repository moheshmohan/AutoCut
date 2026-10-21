//! VideoToolbox's hardware encoders (macOS) with one session against several at once: whether a
//! second compression session runs on a second media engine (the M-series Max and Ultra chips have
//! two or more) and what that costs in quality, the measurements behind
//! `hardware_encode::ParallelEncoder` (docs/performance.md). Ignored (benchmarks); run with
//! `cargo test --release -p filmcraft-platform --test vt_encode_scaling -- --ignored --nocapture <name>`:
//!
//! - `perf_vt_sessions_scale`: frames per second of 1–4 sessions encoding at once (`VT_RT=1` real-time
//!   sessions, `VT_RATE` frame rate, `VT_SIZES=3840x2160,…`, `VT_SESSIONS=1,2`, `VT_ONLY`, `VT_FRAMES`,
//!   `VT_ROUNDS`);
//! - `perf_vt_quality`: bitrate and luma PSNR of a real clip (`VT_CLIP`) encoded by one session or by
//!   two taking turns, cut and joined in several ways (`VT_VARIANTS`);
//! - `perf_export_encoder_one_vs_two_sessions` and `perf_export_encoder_quality`: the export's own
//!   encoders, one session against `ParallelEncoder`.
#![cfg(target_os = "macos")]

use std::sync::{Arc, Barrier};
use std::time::Instant;

use filmcraft_platform::videotoolbox_encode::{VtConfig, VtEncoder, VtProfile, VtRate};

/// Planar 4:2:0 pictures of a pan across a detailed texture (so motion search and the entropy
/// coder have real work), `count` of them, each `w`×`h`.
fn pictures(w: usize, h: usize, count: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let tw = w + 8 * count;
    let mut tex = vec![0u8; tw * h];
    let mut s = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..tw {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let smooth = ((x as f32 * 0.013).sin() * 60.0 + (y as f32 * 0.021).cos() * 50.0 + ((x + y) as f32 * 0.004).sin() * 40.0) as i32;
            tex[y * tw + x] = (128 + smooth + (s % 24) as i32 - 12).clamp(16, 235) as u8;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    (0..count)
        .map(|i| {
            let off = 8 * i;
            let mut y = vec![0u8; w * h];
            for r in 0..h {
                y[r * w..(r + 1) * w].copy_from_slice(&tex[r * tw + off..r * tw + off + w]);
            }
            let u: Vec<u8> = (0..cw * ch).map(|k| (128 + ((k % cw + off / 2) as f32 * 0.02).sin() as i32 * 30) as u8).collect();
            let v: Vec<u8> = (0..cw * ch).map(|k| (128 + ((k / cw) as f32 * 0.03).cos() as i32 * 30) as u8).collect();
            (y, u, v)
        })
        .collect()
}

fn config(w: u32, h: u32, profile: VtProfile, kbps: u32) -> VtConfig {
    VtConfig {
        width: w,
        height: h,
        timescale: 30,
        frame_duration: 1,
        keyframe_interval: 60,
        profile,
        rate: VtRate::Vbr { target_kbps: kbps, max_kbps: kbps * 6 / 5 },
        realtime: false,
    }
}

/// `sessions` encoders on threads of their own, each encoding `frames` pictures at once; returns
/// the aggregate frames per second and the mean time a submission (fill + encode call) took.
fn run(sessions: usize, frames: usize, cfg: &VtConfig, pics: &Arc<Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>>) -> Option<(f64, f64)> {
    let mut encoders = Vec::new();
    for _ in 0..sessions {
        match VtEncoder::new(cfg.clone()) {
            Ok(e) => encoders.push(e),
            Err(why) => {
                eprintln!("SKIPPED: {sessions} sessions: {why}");
                return None;
            }
        }
    }
    let barrier = Arc::new(Barrier::new(sessions + 1));
    let handles: Vec<_> = encoders
        .into_iter()
        .map(|mut enc| {
            let (barrier, pics) = (barrier.clone(), pics.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let mut submit = 0f64;
                let mut out = 0usize;
                for i in 0..frames {
                    let (y, u, v) = &pics[i % pics.len()];
                    let t = Instant::now();
                    out += enc.encode(i as u64, y, u, v).expect("encode").len();
                    submit += t.elapsed().as_secs_f64();
                }
                out += enc.flush().expect("flush").len();
                assert_eq!(out, frames);
                submit / frames as f64
            })
        })
        .collect();
    barrier.wait();
    let t = Instant::now();
    let submits: Vec<f64> = handles.into_iter().map(|h| h.join().expect("encoder thread")).collect();
    let wall = t.elapsed().as_secs_f64();
    Some(((sessions * frames) as f64 / wall, submits.iter().sum::<f64>() / submits.len() as f64 * 1000.0))
}

/// Aggregate frames per second of 1, 2, 3 and 4 sessions encoding at once, each fed by a thread of
/// its own (see the module docs for the knobs), and the level VideoToolbox picks.
#[test]
#[ignore]
fn perf_vt_sessions_scale() {
    let frames: usize = std::env::var("VT_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let rounds: usize = std::env::var("VT_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let only = std::env::var("VT_ONLY").unwrap_or_default();
    let mut cases = vec![
        ("4K H.264".to_string(), 3840u32, 2160u32, VtProfile::High, 45_000u32),
        ("4K HEVC".to_string(), 3840, 2160, VtProfile::HevcMain, 45_000),
        ("1080p H.264".to_string(), 1920, 1080, VtProfile::High, 16_000),
        ("1080p HEVC".to_string(), 1920, 1080, VtProfile::HevcMain, 16_000),
    ];
    if let Ok(sizes) = std::env::var("VT_SIZES") {
        cases.clear();
        for size in sizes.split(',') {
            let Some((w, h)) = size.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))) else { continue };
            let kbps = (45_000.0 * f64::from(w * h) / (3840.0 * 2160.0)) as u32;
            cases.push((format!("{size} H.264"), w, h, VtProfile::High, kbps));
            cases.push((format!("{size} HEVC"), w, h, VtProfile::HevcMain, kbps));
        }
    }
    let rate: u32 = std::env::var("VT_RATE").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let realtime = std::env::var("VT_RT").is_ok();
    for (name, w, h, profile, kbps) in cases {
        if !only.is_empty() && !name.contains(&only) {
            continue;
        }
        let pics = Arc::new(pictures(w as usize, h as usize, 30));
        let cfg = VtConfig { timescale: rate, realtime, ..config(w, h, profile, kbps) };
        if profile == VtProfile::High {
            let mut e = VtEncoder::new(cfg.clone()).expect("session");
            let (y, u, v) = &pics[0];
            e.encode(0, y, u, v).expect("encode");
            let sps = e.parameter_sets().expect("params").sps[0].clone();
            eprintln!("{name}: {rate} fps, real time {realtime}: SPS profile {} level {}", sps[1], sps[3]);
        }
        let counts: Vec<usize> =
            std::env::var("VT_SESSIONS").ok().map(|s| s.split(',').filter_map(|n| n.parse().ok()).collect()).unwrap_or_else(|| vec![1, 2, 3, 4]);
        for n in counts {
            let mut fps = Vec::new();
            let mut sub = Vec::new();
            for _ in 0..rounds {
                let Some((f, s)) = run(n, frames, &cfg, &pics) else { break };
                fps.push(f);
                sub.push(s);
            }
            if fps.is_empty() {
                continue;
            }
            fps.sort_by(f64::total_cmp);
            sub.sort_by(f64::total_cmp);
            eprintln!(
                "{name:12} {n} session(s): {:6.1} fps aggregate (median of {rounds}; {:.1}..{:.1}), submit {:.2} ms/frame",
                fps[fps.len() / 2],
                fps[0],
                fps[fps.len() - 1],
                sub[sub.len() / 2]
            );
        }
    }
}

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Pictures of a real clip, decoded by our decoders (VideoToolbox for H.264 / HEVC) as they are
/// asked for: `count` from each of `starts` (seconds).
struct Clip {
    src: filmcraft_media::SharedSource,
    rate: filmcraft_time::FrameRate,
    firsts: Vec<i64>,
    count: usize,
    w: u32,
    h: u32,
}

impl Clip {
    fn open(path: &str, starts: &[f64], count: usize) -> Option<Clip> {
        let _ = filmcraft_platform::register();
        let reader = filmcraft_media::reader::FileReader::open(std::path::Path::new(path)).ok()?;
        let src = filmcraft_media::reader::open_reader("clip", Arc::new(reader), &filmcraft_codecs::reader_openers(), &filmcraft_codecs::openers()).ok()?;
        let v = src.info().video.clone()?;
        let fps = v.frame_rate.num as f64 / v.frame_rate.den.max(1) as f64;
        Some(Clip { src, rate: v.frame_rate, firsts: starts.iter().map(|s| (s * fps).round() as i64).collect(), count, w: v.width, h: v.height })
    }

    fn len(&self) -> usize {
        self.firsts.len() * self.count
    }

    fn frame(&self, i: usize) -> Planes {
        let f = self.firsts[i / self.count] + (i % self.count) as i64;
        let frame = self.src.video_frame(filmcraft_media::FrameRequest { time: self.rate.tick_of(f), scale: 1.0 }).expect("source frame");
        let filmcraft_frame::PixelData::Yuv8 { planes, chroma: filmcraft_frame::Chroma::C420, .. } = &frame.data else { panic!("8-bit 4:2:0 source") };
        (planes[0].to_vec(), planes[1].to_vec(), planes[2].to_vec())
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>() / a.len().max(1) as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

/// Encode the clip with `sessions` sessions taking segments of `segment` pictures in turn (with a
/// forced keyframe at the start of each when `force`), joined in presentation order; returns the
/// packets (bytes, key flag) and the sample entry, after checking the sessions' parameter sets match.
fn encode_split(
    cfg: &VtConfig,
    sessions: usize,
    segment: usize,
    force: bool,
    warm: usize,
    prime: usize,
    clip: &Clip,
) -> Option<(Vec<(Vec<u8>, bool)>, filmcraft_isobmff::SampleEntry)> {
    let mut encs: Vec<VtEncoder> = (0..sessions).map(|_| VtEncoder::new(cfg.clone())).collect::<Result<_, _>>().ok()?;
    let mut local = vec![0u64; sessions];
    let mut pending: Vec<std::collections::VecDeque<filmcraft_platform::videotoolbox_encode::VtPacket>> = vec![Default::default(); sessions];
    let mut discard: Vec<std::collections::VecDeque<bool>> = vec![Default::default(); sessions];
    let mut started = vec![false; sessions];
    let mut owner = Vec::new();
    let submit = |encs: &mut Vec<VtEncoder>,
                  s: usize,
                  i: usize,
                  key: bool,
                  drop_it: bool,
                  local: &mut Vec<u64>,
                  pending: &mut Vec<std::collections::VecDeque<_>>,
                  discard: &mut Vec<std::collections::VecDeque<bool>>| {
        let (y, u, v) = clip.frame(i);
        let p = encs[s].encode_frame(local[s], &y, &u, &v, key).expect("encode");
        local[s] += 1;
        discard[s].push_back(drop_it);
        for pk in p {
            if !discard[s].pop_front().unwrap() {
                pending[s].push_back(pk);
            }
        }
    };
    for i in 0..clip.len() {
        let s = (i / segment) % sessions;
        if i.is_multiple_of(segment) {
            owner.push(s);
            let g = i / segment;
            if !started[s] && g > 0 && warm > 0 {
                for k in (g * segment).saturating_sub(warm)..g * segment {
                    submit(&mut encs, s, k, false, true, &mut local, &mut pending, &mut discard);
                }
            } else if started[s] && prime > 0 {
                for k in i.saturating_sub(prime)..i {
                    submit(&mut encs, s, k, false, true, &mut local, &mut pending, &mut discard);
                }
            }
            started[s] = true;
        }
        submit(&mut encs, s, i, force && i.is_multiple_of(segment), false, &mut local, &mut pending, &mut discard);
    }
    for (s, e) in encs.iter_mut().enumerate() {
        for pk in e.flush().expect("flush") {
            if !discard[s].pop_front().unwrap() {
                pending[s].push_back(pk);
            }
        }
    }
    let params: Vec<_> = encs.iter().filter_map(|e| e.parameter_sets()).collect();
    assert!(params.windows(2).all(|w| w[0] == w[1]), "the sessions' parameter sets differ: {params:?}");
    let mut packets = Vec::new();
    for (g, &s) in owner.iter().enumerate() {
        let n = segment.min(clip.len() - g * segment);
        packets.extend((0..n).map(|_| pending[s].pop_front().map(|p| (p.data, p.key)).expect("packet")));
    }
    let p = &params[0];
    let (w, h) = (cfg.width as u16, cfg.height as u16);
    let entry = match cfg.profile {
        VtProfile::HevcMain => filmcraft_isobmff::SampleEntry::hevc(filmcraft_isobmff::HevcConfig::parse(p.hvcc.as_deref()?).ok()?, w, h),
        _ => filmcraft_isobmff::SampleEntry::avc(filmcraft_isobmff::AvcConfig::new(p.sps.clone(), p.pps.clone(), 4), w, h),
    };
    Some((packets, entry))
}

/// Quality and size of a real clip's pictures encoded in one session (offline or real time) and in
/// two real-time sessions taking turns by segments. Set `VT_CLIP` to a 4K H.264 / HEVC file, and
/// optionally `VT_STARTS` (comma-separated seconds), `VT_COUNT` (frames from each), `VT_GOP`,
/// `VT_CODECS`, `VT_GROUPS=1` (per-group bytes and PSNR), `VT_DETAIL=5,7` (frames around joins).
/// `VT_VARIANTS` picks among: `offline` / `rt` (one session, offline / real time), `rt-forced`
/// (keyframes forced where segments would start), `alt1` / `alt2` (two sessions, segments of one /
/// two keyframe intervals), `alt1-warmN` (the second session first warms up on the last N pictures
/// before its first segment; `alt1-warm`: the whole segment), `alt1-warm-primeN` (and every session
/// re-primes on the N pictures before each of its segments), `halves` (the first and second half),
/// `rt-idle2` / `offline-idle2` (a second session exists but stays idle).
#[test]
#[ignore]
fn perf_vt_quality() {
    let Ok(path) = std::env::var("VT_CLIP") else {
        eprintln!("SKIPPED: set VT_CLIP");
        return;
    };
    let starts: Vec<f64> = std::env::var("VT_STARTS").unwrap_or_else(|_| "0,102,244.8".into()).split(',').filter_map(|s| s.parse().ok()).collect();
    let count: usize = std::env::var("VT_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let gop: usize = std::env::var("VT_GOP").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let variants = std::env::var("VT_VARIANTS").unwrap_or_else(|_| "offline,rt,rt-forced,alt1,alt2,halves".into());
    let codecs = std::env::var("VT_CODECS").unwrap_or_else(|_| "H.264,HEVC".into());
    let Some(clip) = Clip::open(&path, &starts, count) else { panic!("cannot read {path}") };
    let n = clip.len();
    eprintln!("{n} pictures of {}x{}", clip.w, clip.h);
    let halves = n.div_ceil(2).div_ceil(gop) * gop;
    for (name, profile, kbps) in [("H.264", VtProfile::High, 45_000u32), ("HEVC", VtProfile::HevcMain, 45_000)] {
        if !codecs.split(',').any(|c| c == name) {
            continue;
        }
        for (variant, realtime, sessions, segment, force, warm, prime) in [
            ("offline", false, 1usize, n, false, 0usize, 0usize),
            ("rt", true, 1, n, false, 0, 0),
            ("rt-forced", true, 1, gop, true, 0, 0),
            ("alt1", true, 2, gop, true, 0, 0),
            ("alt1-warm", true, 2, gop, true, gop, 0),
            ("alt1-warm5", true, 2, gop, true, 5, 0),
            ("alt1-warm10", true, 2, gop, true, 10, 0),
            ("alt1-warm20", true, 2, gop, true, 20, 0),
            ("alt1-warm30", true, 2, gop, true, 30, 0),
            ("alt1-warm-prime2", true, 2, gop, true, gop, 2),
            ("alt1-warm-prime5", true, 2, gop, true, gop, 5),
            ("alt2", true, 2, 2 * gop, true, 0, 0),
            ("halves", true, 2, halves, true, 0, 0),
            ("rt-idle2", true, 2, n, true, 0, 0),
            ("offline-idle2", false, 2, n, true, 0, 0),
        ] {
            if !variants.split(',').any(|v| v == variant) {
                continue;
            }
            let cfg = VtConfig { realtime, keyframe_interval: gop as u32, ..config(clip.w, clip.h, profile, kbps) };
            let Some((packets, entry)) = encode_split(&cfg, sessions, segment, force, warm, prime, &clip) else {
                eprintln!("SKIPPED {name} {variant}");
                continue;
            };
            let mut dec = filmcraft_codecs::make_video_decoder(&entry).expect("decoder");
            let mut ps = Vec::with_capacity(n);
            let mut next = 0usize;
            let mut check = |frames: Vec<filmcraft_codecs::DecodedFrame>, ps: &mut Vec<f64>| {
                for f in frames {
                    let filmcraft_frame::PixelData::Yuv8 { planes, .. } = &f.frame.data else { panic!("8-bit") };
                    ps.push(psnr(&planes[0], &clip.frame(next).0));
                    next += 1;
                }
            };
            for (k, (data, _)) in packets.iter().enumerate() {
                check(dec.decode(data, k as i64).expect("decode"), &mut ps);
            }
            check(dec.flush(), &mut ps);
            assert_eq!(ps.len(), n);
            let bytes: usize = packets.iter().map(|(d, _)| d.len()).sum();
            let kbps_out = bytes as f64 * 8.0 / (n as f64 / 30.0) / 1000.0;
            let mean = ps.iter().sum::<f64>() / ps.len() as f64;
            let min = ps.iter().copied().fold(f64::MAX, f64::min);
            let keys = packets.iter().filter(|(_, k)| *k).count();
            // the pictures right after each join against the same pictures elsewhere
            let firsts: Vec<f64> = ps.chunks(gop).map(|c| c.iter().take(5).sum::<f64>() / c.len().min(5) as f64).collect();
            eprintln!(
                "{name:6} {variant:10} {kbps_out:8.0} kb/s  luma PSNR mean {mean:.2} dB, min {min:.2} dB, {keys} keyframes; first 5 of each group {:.1?}",
                firsts
            );
            if std::env::var("VT_GROUPS").is_ok() {
                let kb: Vec<usize> = packets.chunks(gop).map(|c| c.iter().map(|(d, _)| d.len()).sum::<usize>() / 1000).collect();
                let pm: Vec<f64> = ps.chunks(gop).map(|c| c.iter().sum::<f64>() / c.len() as f64).collect();
                eprintln!("    group kB {kb:?}\n    group PSNR {pm:.1?}");
            }
            if let Ok(groups) = std::env::var("VT_DETAIL") {
                for g in groups.split(',').filter_map(|g| g.parse::<usize>().ok()) {
                    let a = (g * gop).saturating_sub(3);
                    let b = (g * gop + 8).min(n);
                    let kb: Vec<usize> = packets[a..b].iter().map(|(d, _)| d.len() / 1000).collect();
                    eprintln!("    group {g}: PSNR {:.1?} kB {kb:?}", &ps[a..b]);
                }
            }
        }
    }
}

/// The export's encoder, one session (`EncodeSessions::One`) or two (`ParallelEncoder`), for 4K at
/// the YouTube 2160p preset's bitrate.
fn export_encoder(profile: VtProfile, w: u32, h: u32, two: bool) -> Option<Box<dyn filmcraft_export::VideoEncoder>> {
    use filmcraft_export::{ExportSettings, Format, HardwareEncoding};
    let cfg = VtConfig { keyframe_interval: 60, ..config(w, h, profile, 45_000) };
    if two {
        return filmcraft_platform::hardware_encode::ParallelEncoder::new(cfg, 2).ok().map(|p| Box::new(p) as Box<dyn filmcraft_export::VideoEncoder>);
    }
    let format = if profile == VtProfile::HevcMain { Format::Hevc } else { Format::H264 };
    let s = ExportSettings {
        format,
        hardware_encoding: HardwareEncoding::Auto,
        bitrate_kbps: 45_000,
        max_bitrate_kbps: Some(54_000),
        keyframe_distance: Some(60),
        ..Default::default()
    };
    filmcraft_export::set_hw_encode_sessions(filmcraft_export::EncodeSessions::One);
    let e = filmcraft_platform::hardware_encode::videotoolbox_encoder_factory(format, w, h, filmcraft_time::FrameRate::FPS_30, &s)?.ok();
    filmcraft_export::set_hw_encode_sessions(filmcraft_export::EncodeSessions::Auto);
    e
}

/// Frames per second of the export's hardware encoder, fed from one thread as the export feeds it
/// (`encode_yuv420`, one picture after the other): one session against two (`ParallelEncoder`),
/// alternated, medians. `VT_FRAMES` pictures (default 600), `VT_ROUNDS` (default 3), `VT_ONLY`.
#[test]
#[ignore]
fn perf_export_encoder_one_vs_two_sessions() {
    let frames: u64 = std::env::var("VT_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    let rounds: usize = std::env::var("VT_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let only = std::env::var("VT_ONLY").unwrap_or_default();
    for (name, w, h, profile) in
        [("4K H.264", 3840u32, 2160u32, VtProfile::High), ("4K HEVC", 3840, 2160, VtProfile::HevcMain), ("DCI 4K H.264", 4096, 2160, VtProfile::High)]
    {
        if !only.is_empty() && !name.contains(&only) {
            continue;
        }
        let pics = pictures(w as usize, h as usize, 30);
        let mut fps = [Vec::new(), Vec::new()];
        for _ in 0..rounds {
            for (k, two) in [false, true].into_iter().enumerate() {
                let Some(mut enc) = export_encoder(profile, w, h, two) else {
                    eprintln!("SKIPPED {name}");
                    return;
                };
                let t = Instant::now();
                let mut out = 0usize;
                for i in 0..frames {
                    let (y, u, v) = &pics[(i % pics.len() as u64) as usize];
                    let f = filmcraft_export::EncoderFrame { width: w, height: h, rgba: &[], hdr: None, index: i };
                    out += enc.encode_yuv420(&f, y, u, v).expect("encode").len();
                }
                out += enc.flush().expect("flush").len();
                assert_eq!(out as u64, frames);
                fps[k].push(frames as f64 / t.elapsed().as_secs_f64());
            }
        }
        for f in &mut fps {
            f.sort_by(f64::total_cmp);
        }
        let med = |v: &Vec<f64>| v[v.len() / 2];
        eprintln!(
            "{name:14} one session {:6.1} fps ({:.1?}), two sessions {:6.1} fps ({:.1?}): {:.2}x",
            med(&fps[0]),
            fps[0],
            med(&fps[1]),
            fps[1],
            med(&fps[1]) / med(&fps[0])
        );
    }
}

/// The export's encoder with one session against two on a real clip's pictures (`VT_CLIP`,
/// `VT_STARTS`, `VT_COUNT` as for `perf_vt_quality`): bitrate, luma PSNR overall, per group of
/// pictures and right after each join.
#[test]
#[ignore]
fn perf_export_encoder_quality() {
    let Ok(path) = std::env::var("VT_CLIP") else {
        eprintln!("SKIPPED: set VT_CLIP");
        return;
    };
    let starts: Vec<f64> = std::env::var("VT_STARTS").unwrap_or_else(|_| "90".into()).split(',').filter_map(|s| s.parse().ok()).collect();
    let count: usize = std::env::var("VT_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    let codecs = std::env::var("VT_CODECS").unwrap_or_else(|_| "H.264".into());
    let Some(clip) = Clip::open(&path, &starts, count) else { panic!("cannot read {path}") };
    let n = clip.len();
    for (name, profile) in [("H.264", VtProfile::High), ("HEVC", VtProfile::HevcMain)] {
        if !codecs.split(',').any(|c| c == name) {
            continue;
        }
        for two in [false, true] {
            let Some(mut enc) = export_encoder(profile, clip.w, clip.h, two) else {
                eprintln!("SKIPPED {name}");
                return;
            };
            let mut packets = Vec::new();
            for i in 0..n {
                let (y, u, v) = clip.frame(i);
                let f = filmcraft_export::EncoderFrame { width: clip.w, height: clip.h, rgba: &[], hdr: None, index: i as u64 };
                packets.extend(enc.encode_yuv420(&f, &y, &u, &v).expect("encode"));
            }
            packets.extend(enc.flush().expect("flush"));
            assert_eq!(packets.len(), n);
            let mut dec = filmcraft_codecs::make_video_decoder(&enc.sample_entry()).expect("decoder");
            let mut ps = Vec::with_capacity(n);
            let mut next = 0usize;
            let mut check = |frames: Vec<filmcraft_codecs::DecodedFrame>, ps: &mut Vec<f64>| {
                for f in frames {
                    let filmcraft_frame::PixelData::Yuv8 { planes, .. } = &f.frame.data else { panic!("8-bit") };
                    ps.push(psnr(&planes[0], &clip.frame(next).0));
                    next += 1;
                }
            };
            for (k, p) in packets.iter().enumerate() {
                check(dec.decode(&p.data, k as i64).expect("decode"), &mut ps);
            }
            check(dec.flush(), &mut ps);
            assert_eq!(ps.len(), n);
            let bytes: usize = packets.iter().map(|p| p.data.len()).sum();
            let mean = ps.iter().sum::<f64>() / n as f64;
            let groups: Vec<f64> = ps.chunks(60).map(|c| c.iter().sum::<f64>() / c.len() as f64).collect();
            let joins: Vec<f64> = ps.chunks(60).map(|c| c.iter().take(5).sum::<f64>() / c.len().min(5) as f64).collect();
            eprintln!(
                "{name:6} {} {:6.0} kb/s, luma PSNR {mean:.2} dB\n    per group {groups:.1?}\n    first 5 of each group {joins:.1?}",
                if two { "two sessions" } else { "one session " },
                bytes as f64 * 8.0 / (n as f64 / 30.0) / 1000.0
            );
        }
    }
}
