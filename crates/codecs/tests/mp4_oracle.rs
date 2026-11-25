//! MP4 audio against ffmpeg: ffmpeg writes an AAC track (with the edit list that skips the encoder
//! priming), and reads at another sample rate must land on the same source samples as the native
//! decode, however the requests are cut.

mod common;

use common::*;
use filmcraft_time::FrameRate;

/// A source whose rate differs from the sequence's is resampled per request. An AAC track in MP4
/// carries an edit list that skips the encoder priming (ffmpeg: 1024 samples), and every request
/// must still land on the same source samples: reading at 44.1 kHz in video-frame-sized pieces (as
/// an export does) equals the 48 kHz decode interpolated at the same times.
#[test]
fn aac_mp4_resampled_reads_follow_the_edit_list() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(f) = fixture(&ff, "tone_aac_48k.mp4", &["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=3", "-c:a", "aac", "-b:a", "128k"]) else {
        panic!("could not write the AAC fixture")
    };
    let src = filmcraft_codecs::open_bytes("tone_aac_48k.mp4", bytes(&f)).unwrap();
    let native = src.audio(0, 2 * 48_000, 48_000).unwrap().channels[0].clone();
    let ratio = 48_000.0 / 44_100.0;
    // 30 fps at 44.1 kHz, starting mid-file and off the 48 kHz grid
    let (mut start, piece) = (22_051i64, 1470usize);
    let (mut worst, mut peak) = (0f32, 0f32);
    for _ in 0..20 {
        let got = src.audio(start, piece, 44_100).unwrap();
        for (k, &v) in got.channels[0].iter().enumerate() {
            let pos = (start + k as i64) as f64 * ratio;
            let i0 = pos.floor() as usize;
            let want = native[i0] + (native[i0 + 1] - native[i0]) * (pos - i0 as f64) as f32;
            worst = worst.max((v - want).abs());
            peak = peak.max(v.abs());
        }
        start += piece as i64;
    }
    let level = native.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak <= level * 1.01, "resampled peak {peak} above the source's {level}");
    assert!(worst < 1e-4, "resampled read differs from the source by {worst}");
}

/// ffmpeg's MP4 muxer skips the B-frame reorder delay with an edit list, so the media (`mdhd`) runs
/// one frame past what plays. The clip must be as long as ffmpeg decodes it: 60 frames, not 61.
#[test]
fn bframe_mp4_duration_is_the_decoded_frame_count() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let args = ["-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30:duration=2", "-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p"];
    let Some(f) = fixture(&ff, "bframes_x264.mp4", &args) else {
        eprintln!("SKIPPED: ffmpeg without libx264");
        return;
    };
    let n = ffmpeg_frames(&ff, &f, "yuv420p").len() / (160 * 96 * 3 / 2);
    assert_eq!(n, 60);
    let src = filmcraft_codecs::open_bytes("bframes_x264.mp4", bytes(&f)).unwrap();
    assert_eq!(src.info().duration, filmcraft_time::FrameRate::FPS_30.tick_of(n as i64));
}

/// `mp4` (moov after mdat, as ffmpeg writes it) with a clean aperture (`clap`) of `w`×`h` moved by
/// (`dx`, `dy`) from the centre, inserted at the end of the first video sample entry.
fn with_clap(mp4: &[u8], (w, h, dx, dy): (u32, u32, i32, i32)) -> Vec<u8> {
    let be = |b: &[u8], at: usize| u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
    // the chain moov > trak > mdia > minf > stbl > stsd > avc1 (stsd has 8 bytes before its entries)
    let mut chain = Vec::new();
    let (mut off, mut end) = (0, mp4.len());
    for (want, skip) in [(b"moov", 8), (b"trak", 8), (b"mdia", 8), (b"minf", 8), (b"stbl", 8), (b"stsd", 16), (b"avc1", 0)] {
        loop {
            assert!(off + 8 <= end, "no {}", String::from_utf8_lossy(want));
            let size = be(mp4, off);
            if &mp4[off + 4..off + 8] == want {
                chain.push(off);
                end = off + size;
                off += skip;
                break;
            }
            off += size;
        }
    }
    let entry = *chain.last().unwrap();
    assert!(mp4.windows(4).position(|x| x == b"mdat").unwrap() < chain[0], "moov must follow mdat");
    let mut clap = Vec::new();
    for v in [40u32, u32::from_be_bytes(*b"clap"), w, 1, h, 1, dx as u32, 1, dy as u32, 1] {
        clap.extend_from_slice(&v.to_be_bytes());
    }
    let mut out = mp4.to_vec();
    let at = entry + be(mp4, entry);
    out.splice(at..at, clap);
    for &b in &chain {
        let size = (be(&out, b) + 40) as u32;
        out[b..b + 4].copy_from_slice(&size.to_be_bytes());
    }
    out
}

/// Cropped pictures are the size and the samples ffmpeg decodes (#288): H.264 frame cropping (SPS)
/// and an HEVC conformance window while the sample entry gives the uncropped size, and a clean
/// aperture (`clap`, as iPhones write), centred, off-centre, and with a 90° display rotation
/// applied after it.
#[test]
fn cropped_pictures_match_ffmpeg() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let src = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1,noise=alls=10:allf=t", "-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p"];
    let Some(plain) = fixture(&ff, "crop_plain_x264.mp4", &src) else {
        eprintln!("SKIPPED: ffmpeg without libx264");
        return;
    };
    let sps =
        fixture(&ff, "crop_sps_x264.mp4", &[&src[..], &["-bsf:v", "h264_metadata=crop_left=16:crop_right=16:crop_top=8:crop_bottom=8"]].concat()).unwrap();
    // the display matrix added without re-encoding (an encode would turn the pixels instead)
    let rotated = fixture(&ff, "crop_rotated_x264.mp4", &["-display_rotation", "90", "-i", plain.to_str().unwrap(), "-c", "copy"]).unwrap();
    let dir = dir();
    let mut cases = vec![("SPS crop", sps, (288, 224))];
    // HEVC: the conformance window
    let hevc = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=1,noise=alls=10:allf=t",
        "-c:v",
        "libx265",
        "-x265-params",
        "log-level=error:bframes=3",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p",
    ];
    match fixture(&ff, "crop_conformance_x265.mp4", &[&hevc[..], &["-bsf:v", "hevc_metadata=crop_left=16:crop_right=16:crop_top=8:crop_bottom=8"]].concat()) {
        Some(p) => cases.push(("HEVC conformance window", p, (288, 224))),
        None => eprintln!("SKIPPED (HEVC case): ffmpeg without libx265"),
    }
    for (name, base, clap, size) in [
        ("centred clap", &plain, (280, 200, 0, 0), (280, 200)),
        ("off-centre clap", &plain, (200, 150, -30, 22), (200, 150)),
        ("clap then rotation", &rotated, (280, 200, 0, 0), (200, 280)),
    ] {
        let path = dir.join(format!("crop_{}.mp4", name.replace(' ', "_")));
        std::fs::write(&path, with_clap(&bytes(base), clap)).unwrap();
        cases.push((name, path, size));
    }
    for (name, path, (w, h)) in cases {
        let src = filmcraft_codecs::open_bytes(path.file_name().unwrap().to_str().unwrap(), bytes(&path)).unwrap();
        let v = src.info().video.clone().unwrap();
        assert_eq!((v.width, v.height), (w, h), "{name}: reported size");
        let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        // `-flags unaligned`: ffmpeg otherwise skips a left crop that would misalign its frame
        // pointers (on AVX-512 machines it needs 64-byte alignment) and returns wider frames
        let raw = ffmpeg_out(
            &ff,
            &["-flags", "unaligned", "-i", path.to_str().unwrap(), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        );
        let frame_bytes = (w * h) as usize + 2 * cw * ch;
        if name.contains("clap") && !raw.len().is_multiple_of(frame_bytes) {
            // ffmpeg before 7.1 ignores `clap` and decodes the whole picture
            eprintln!("SKIPPED ({name}): this ffmpeg does not apply the clean aperture");
            continue;
        }
        assert_eq!(raw.len() % frame_bytes, 0, "{name}: ffmpeg's frames are {w}x{h}");
        let n = raw.len() / frame_bytes;
        assert_eq!(n, 30, "{name}");
        for (i, want) in raw.chunks(frame_bytes).enumerate() {
            let f = src.video_frame(filmcraft_media::FrameRequest::full(v.frame_rate.tick_of(i as i64))).unwrap();
            assert_eq!((f.width, f.height), (w, h), "{name}: frame {i} size");
            assert!(planes(&f) == raw_planes(want, w as usize, h as usize, cw, ch, 1), "{name}: frame {i} differs from ffmpeg's");
        }
    }
}

/// An open GOP (x265 `open-gop`, like iPhone HEVC): each CRA after the first is followed in the
/// file by RASL pictures shown before it, predicted from the GOP before. Asked for first (a clip
/// trimmed to start on one, the playhead parked on one), such a picture came out black: decoding
/// started at the CRA after it, which leaves its RASL pictures out. Every frame, asked of a fresh
/// source and in an order that visits each CRA before its leading pictures, is ffmpeg's, from MP4
/// and from Matroska.
#[test]
fn open_gop_leading_pictures_decode_alone() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x96:rate=30:duration=2,noise=alls=10:allf=t",
        "-c:v",
        "libx265",
        "-x265-params",
        "log-level=error:keyint=12:min-keyint=12:scenecut=0:open-gop=1:bframes=3",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p",
    ];
    let Some(mp4) = fixture(&ff, "open_gop_x265.mp4", &args) else {
        eprintln!("SKIPPED: ffmpeg without libx265");
        return;
    };
    let mkv = fixture(&ff, "open_gop_x265.mkv", &["-i", mp4.to_str().unwrap(), "-c", "copy"]).unwrap();
    let raw = ffmpeg_frames(&ff, &mp4, "yuv420p");
    let frame_bytes = 160 * 96 * 3 / 2;
    let want: Vec<_> = raw.chunks(frame_bytes).map(|f| raw_planes(f, 160, 96, 80, 48, 1)).collect();
    assert_eq!(want.len(), 60);
    for path in [&mp4, &mkv] {
        let name = path.file_name().unwrap().to_str().unwrap();
        let open = || filmcraft_codecs::open_bytes(name, bytes(path)).unwrap();
        let rate = open().info().frame_rate();
        let frame = |src: &filmcraft_media::SharedSource, i: usize| {
            let f = src.video_frame(filmcraft_media::FrameRequest::full(rate.tick_of(i as i64))).unwrap_or_else(|e| panic!("{name}: frame {i}: {e}"));
            assert!(planes(&f) == want[i], "{name}: frame {i} differs from ffmpeg's");
        };
        for i in 0..want.len() {
            frame(&open(), i);
        }
        let src = open();
        for cra in [12, 24, 36, 48] {
            for i in [cra, cra - 1, cra - 2, cra - 3] {
                frame(&src, i);
            }
        }
    }
}

/// #800: MP2 in MP4 (`mp4a` + object type 0x6B, which names no layer) and in MOV (QuickTime
/// `.mp2`) decodes sample-exactly as ffmpeg does: same length, the encoder delay (481 samples)
/// trimmed by the edit list, and the same samples from the first one on, also after a seek. It
/// used to go to the layer III decoder (MP4: silence) or not decode at all (MOV).
#[test]
fn mp2_in_mp4_and_mov_matches_ffmpeg() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, rate) in [("mp2_stereo_224k_48k.mp4", 48_000u32), ("mp2_stereo_224k_44k.mov", 44_100)] {
        let input = format!("anoisesrc=d=1:c=pink:r={rate}:a=0.3");
        let Some(f) = fixture(&ff, name, &["-f", "lavfi", "-i", &input, "-ac", "2", "-c:a", "mp2", "-b:a", "224k"]) else {
            eprintln!("SKIPPED ({name}): this ffmpeg can't encode it");
            continue;
        };
        let src = filmcraft_codecs::open_bytes(name, bytes(&f)).unwrap();
        let a = src.info().audio().cloned().unwrap();
        assert_eq!((a.sample_rate, a.channels, a.codec.as_str()), (rate, 2, "MPEG Audio"), "{name}");
        let want = ffmpeg_audio_f32(&ff, &f, &[]);
        let n = want.len() / 2;
        assert!(n > rate as usize * 9 / 10, "{name}: ffmpeg decoded {n} samples");
        let worst = |from: usize, len: usize| {
            let got = src.audio(from as i64, len, rate).unwrap();
            (0..len).map(|k| (0..2).map(|c| (got.channels[c][k] - want[2 * (from + k) + c]).abs()).fold(0f32, f32::max)).fold(0f32, f32::max)
        };
        let level = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(level > 0.1, "{name}: the source is silent ({level})");
        // a seek into the middle first, then the whole track from the start
        let (mid, all) = (worst(n / 2 + 77, 3000), worst(0, n));
        println!("{name}: {n} samples, max error mid {mid:.2e}, whole {all:.2e}");
        assert!(mid <= 1e-4 && all <= 1e-4, "{name}: differs from ffmpeg by {mid} (mid) / {all} (whole)");
    }
}

/// Phone and messaging-app clips stamp frame times rounded to whole milliseconds inside a 90 kHz
/// track (issue #1700), as this command writes them: a 30 fps clip's samples then last 2970 and
/// 3060 units in turn (ffmpeg: 2880, 2970, 3060 and 3150), a 60 fps clip's 1350 to 1620. The median
/// duration read the clip as 30.303 / 58.824 fps, so New Sequence From Clip made a sequence at that
/// rate and dropped frames; the rate is the one the rounded durations average to (3000 / 1500).
#[test]
fn millisecond_rounded_frame_times_read_as_their_rate() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (rate, fps, n) in [(FrameRate::FPS_30, "30", 120u64), (FrameRate::FPS_60, "60", 240u64)] {
        let name = format!("ms_stamps_{fps}.mp4");
        let size = format!("testsrc2=size=320x240:rate={fps}");
        let pts = format!("settb=1/1000,setpts='floor(N*1000/{fps}+0.5)/1000/TB'");
        let args = [
            "-f",
            "lavfi",
            "-i",
            &size,
            "-t",
            "4",
            "-vf",
            &pts,
            "-enc_time_base:v",
            "1/1000",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "90000",
        ];
        let Some(f) = fixture(&ff, &name, &args) else {
            eprintln!("SKIPPED: ffmpeg without libx264");
            return;
        };
        let src = filmcraft_codecs::open_bytes(&name, bytes(&f)).unwrap();
        let info = src.info();
        assert_eq!(info.video.as_ref().map(|v| v.frame_rate), Some(rate), "{name} reads as its own rate");
        assert_eq!(info.duration, rate.tick_of(n as i64), "{name} lasts {n} frames");
    }
}

/// HEVC 4:2:2 camera footage (Main 4:2:2 10 profile, 8- and 10-bit; what Sony XAVC HS, Canon and
/// Fujifilm bodies record) in MP4 and MOV, at sizes that are not multiples of 8 or 16 (an odd
/// height included), opens and decodes with our own decoder: as many frames as ffmpeg decodes,
/// the first and last ones (asked for on a fresh source, the last one first) sample-exact in
/// every plane, as 4:2:2 frames whose chroma has every line. No hardware decoder is registered
/// in these tests (`filmcraft_platform` registers them in the app), and the track's decoder is
/// checked to be ours. Before, the decoder refused chroma_format_idc 2 as unsupported.
#[test]
fn hevc_422_in_mp4_and_mov_decodes_with_our_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, pix, bps, (w, h)) in [("hevc_422_8bit.mp4", "yuv422p", 1, (318usize, 182usize)), ("hevc_422_10bit.mov", "yuv422p10le", 2, (330, 187))] {
        // (the format first: testsrc2 drops an odd last line in its own 4:2:0 output)
        let input = format!("testsrc2=size={w}x{h}:rate=30:duration=1,format={pix},noise=alls=10:allf=t");
        let args = ["-f", "lavfi", "-i", &input, "-c:v", "libx265", "-x265-params", "log-level=error:bframes=3:keyint=12", "-tag:v", "hvc1", "-pix_fmt", pix];
        let Some(path) = fixture(&ff, name, &args) else {
            eprintln!("SKIPPED ({name}): ffmpeg without libx265");
            continue;
        };
        // our decoder, not a hardware one
        let file = filmcraft_isobmff::open(std::fs::read(&path).unwrap()).unwrap();
        let track = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
        let entry = &file.tracks[track].entries[0];
        assert_eq!(filmcraft_codecs::make_video_decoder(entry).unwrap().name(), "FilmCraft HEVC", "{name}");
        let open = || filmcraft_codecs::open_bytes(name, bytes(&path)).unwrap();
        let v = open().info().video.clone().unwrap();
        assert_eq!((v.width as usize, v.height as usize), (w, h), "{name}: reported size");
        let raw = ffmpeg_frames(&ff, &path, pix);
        let (cw, ch) = (w.div_ceil(2), h);
        let frame_bytes = (w * h + 2 * cw * ch) * bps;
        assert_eq!(raw.len() % frame_bytes, 0, "{name}: ffmpeg's frames are {w}x{h} {pix}");
        let n = raw.len() / frame_bytes;
        assert_eq!(n, 30, "{name}: ffmpeg's frame count");
        assert_eq!(open().info().duration, v.frame_rate.tick_of(n as i64), "{name}: as many frames as ffmpeg decodes");
        for i in [n - 1, 0] {
            let f =
                open().video_frame(filmcraft_media::FrameRequest::full(v.frame_rate.tick_of(i as i64))).unwrap_or_else(|e| panic!("{name}: frame {i}: {e}"));
            assert_eq!((f.width as usize, f.height as usize), (w, h), "{name}: frame {i} size");
            match &f.data {
                filmcraft_frame::PixelData::Yuv8 { chroma, .. } => assert_eq!((*chroma, bps), (filmcraft_frame::Chroma::C422, 1), "{name}"),
                filmcraft_frame::PixelData::Yuv16 { chroma, bits, .. } => assert_eq!((*chroma, *bits, bps), (filmcraft_frame::Chroma::C422, 10, 2), "{name}"),
                _ => panic!("{name}: frame {i} is not planar YUV"),
            }
            let want = raw_planes(&raw[i * frame_bytes..(i + 1) * frame_bytes], w, h, cw, ch, bps);
            assert!(planes(&f) == want, "{name}: frame {i} differs from ffmpeg's (max difference {})", max_diff(&planes(&f), &want));
        }
    }
}

/// HEVC 4:4:4 (Main 4:4:4 and Main 4:4:4 10, what screen recorders, graphics exports and some
/// cinema cameras write) and monochrome (Main 4:0:0 10) in MP4 and MOV, at odd sizes (4:4:4 and
/// monochrome allow any width and height), open and decode with our own decoder: as many frames as
/// ffmpeg decodes, the first and last ones (asked for on a fresh source, the last one first)
/// sample-exact in every plane, as 4:4:4 frames, and monochrome as grey 4:2:0 frames (the luma
/// sample-exact, neutral chroma at the luma depth). The track's decoder is checked to be ours (no
/// hardware decoder is registered in these tests). Before, the decoder refused them as unsupported
/// ("HEVC: 4:4:4 (only 4:2:0 and 4:2:2)", "HEVC: monochrome (only 4:2:0 and 4:2:2)").
#[test]
fn hevc_444_and_monochrome_in_mp4_and_mov_decode_with_our_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let cases = [
        ("hevc_444_10bit.mov", "yuv444p10le", 2, (331usize, 187usize)),
        ("hevc_444_8bit.mp4", "yuv444p", 1, (318, 181)),
        ("hevc_400_10bit.mp4", "gray10le", 2, (321, 179)),
    ];
    for (name, pix, bps, (w, h)) in cases {
        let input = format!("testsrc2=size={w}x{h}:rate=30:duration=1,format={pix},noise=alls=10:allf=t");
        let args = ["-f", "lavfi", "-i", &input, "-c:v", "libx265", "-x265-params", "log-level=error:bframes=3:keyint=12", "-tag:v", "hvc1", "-pix_fmt", pix];
        let Some(path) = fixture(&ff, name, &args) else {
            eprintln!("SKIPPED ({name}): ffmpeg without libx265");
            continue;
        };
        let file = filmcraft_isobmff::open(std::fs::read(&path).unwrap()).unwrap();
        let track = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
        let entry = &file.tracks[track].entries[0];
        assert_eq!(filmcraft_codecs::make_video_decoder(entry).unwrap().name(), "FilmCraft HEVC", "{name}");
        let open = || filmcraft_codecs::open_bytes(name, bytes(&path)).unwrap();
        let v = open().info().video.clone().unwrap();
        assert_eq!((v.width as usize, v.height as usize), (w, h), "{name}: reported size");
        let mono = pix.starts_with("gray");
        let raw = ffmpeg_frames(&ff, &path, pix);
        let frame_bytes = if mono { w * h * bps } else { 3 * w * h * bps };
        assert_eq!(raw.len() % frame_bytes, 0, "{name}: ffmpeg's frames are {w}x{h} {pix}");
        let n = raw.len() / frame_bytes;
        assert_eq!(n, 30, "{name}: ffmpeg's frame count");
        assert_eq!(open().info().duration, v.frame_rate.tick_of(n as i64), "{name}: as many frames as ffmpeg decodes");
        let bits = if bps == 2 { 10 } else { 8 };
        let want_chroma = if mono { filmcraft_frame::Chroma::C420 } else { filmcraft_frame::Chroma::C444 };
        for i in [n - 1, 0] {
            let f =
                open().video_frame(filmcraft_media::FrameRequest::full(v.frame_rate.tick_of(i as i64))).unwrap_or_else(|e| panic!("{name}: frame {i}: {e}"));
            assert_eq!((f.width as usize, f.height as usize), (w, h), "{name}: frame {i} size");
            match &f.data {
                filmcraft_frame::PixelData::Yuv8 { chroma, .. } => assert_eq!((*chroma, bps), (want_chroma, 1), "{name}"),
                filmcraft_frame::PixelData::Yuv16 { chroma, bits: b, .. } => assert_eq!((*chroma, *b, bps), (want_chroma, bits, 2), "{name}"),
                _ => panic!("{name}: frame {i} is not planar YUV"),
            }
            let raw = &raw[i * frame_bytes..(i + 1) * frame_bytes];
            let got = planes(&f);
            if mono {
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                let luma = raw_planes(raw, w, h, 0, 0, bps).swap_remove(0);
                assert!(got[0] == luma, "{name}: frame {i} luma differs from ffmpeg's");
                let neutral = vec![1u16 << (bits - 1); cw * ch];
                assert!(got[1] == neutral && got[2] == neutral, "{name}: frame {i} chroma is not neutral grey");
            } else {
                let want = raw_planes(raw, w, h, w, h, bps);
                assert!(got == want, "{name}: frame {i} differs from ffmpeg's (max difference {})", max_diff(&got, &want));
            }
        }
    }
}

/// H.264 High 10 and High 4:2:2 camera footage (what mirrorless and cinema cameras record:
/// 10-bit 4:2:0, 8- and 10-bit 4:2:2; B-frames, and all-intra like XAVC S-I and AVC-Intra) in MP4
/// and MOV, at sizes that are not multiples of 16 (an odd height in 4:2:2), opens and decodes with
/// our own decoder: as many frames as ffmpeg decodes, the first and last ones (asked for on a fresh
/// source, the last one first) sample-exact in every plane, as 10-bit or 4:2:2 frames. No hardware
/// decoder is registered in these tests (`filmcraft_platform` registers them in the app), and the
/// track's decoder is checked to be ours. Before, the decoder refused them as unsupported
/// ("H.264: chroma_format_idc 2 (only 4:2:0 is implemented)", "bit depth luma 10 chroma 10").
#[test]
fn h264_camera_footage_decodes_with_our_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let cases = [
        ("h264_422_10bit.mov", "yuv422p10le", "high422", "bframes=3:keyint=12", (330usize, 187usize)),
        ("h264_422_10bit_intra.mov", "yuv422p10le", "high422", "keyint=1", (320, 180)),
        ("h264_422_8bit.mp4", "yuv422p", "high422", "bframes=2:keyint=12:cabac=0", (318, 182)),
        ("h264_high10.mp4", "yuv420p10le", "high10", "bframes=3:keyint=12:weightp=2", (322, 180)),
    ];
    for (name, pix, profile, params, (w, h)) in cases {
        // (the format first: testsrc2 drops an odd last line in its own 4:2:0 output)
        let input = format!("testsrc2=size={w}x{h}:rate=30:duration=1,format={pix},noise=alls=10:allf=t");
        let x264 = format!("log-level=error:{params}");
        let args = ["-f", "lavfi", "-i", &input, "-c:v", "libx264", "-profile:v", profile, "-x264-params", &x264, "-pix_fmt", pix];
        let Some(path) = fixture(&ff, name, &args) else {
            eprintln!("SKIPPED ({name}): ffmpeg without libx264");
            continue;
        };
        // our decoder, not a hardware one
        let file = filmcraft_isobmff::open(std::fs::read(&path).unwrap()).unwrap();
        let track = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
        let entry = &file.tracks[track].entries[0];
        assert_eq!(filmcraft_codecs::make_video_decoder(entry).unwrap().name(), "FilmCraft H.264", "{name}");
        let open = || filmcraft_codecs::open_bytes(name, bytes(&path)).unwrap();
        let v = open().info().video.clone().unwrap();
        assert_eq!((v.width as usize, v.height as usize), (w, h), "{name}: reported size");
        let raw = ffmpeg_frames(&ff, &path, pix);
        let (c422, bps) = (pix.starts_with("yuv422"), if pix.ends_with("le") { 2 } else { 1 });
        let (cw, ch) = (w.div_ceil(2), if c422 { h } else { h.div_ceil(2) });
        let frame_bytes = (w * h + 2 * cw * ch) * bps;
        assert_eq!(raw.len() % frame_bytes, 0, "{name}: ffmpeg's frames are {w}x{h} {pix}");
        let n = raw.len() / frame_bytes;
        assert_eq!(n, 30, "{name}: ffmpeg's frame count");
        assert_eq!(open().info().duration, v.frame_rate.tick_of(n as i64), "{name}: as many frames as ffmpeg decodes");
        let chroma = if c422 { filmcraft_frame::Chroma::C422 } else { filmcraft_frame::Chroma::C420 };
        for i in [n - 1, 0] {
            let f =
                open().video_frame(filmcraft_media::FrameRequest::full(v.frame_rate.tick_of(i as i64))).unwrap_or_else(|e| panic!("{name}: frame {i}: {e}"));
            assert_eq!((f.width as usize, f.height as usize), (w, h), "{name}: frame {i} size");
            match &f.data {
                filmcraft_frame::PixelData::Yuv8 { chroma: c, .. } => assert_eq!((*c, bps), (chroma, 1), "{name}"),
                filmcraft_frame::PixelData::Yuv16 { chroma: c, bits, .. } => assert_eq!((*c, *bits, bps), (chroma, 10, 2), "{name}"),
                _ => panic!("{name}: frame {i} is not planar YUV"),
            }
            let want = raw_planes(&raw[i * frame_bytes..(i + 1) * frame_bytes], w, h, cw, ch, bps);
            assert!(planes(&f) == want, "{name}: frame {i} differs from ffmpeg's (max difference {})", max_diff(&planes(&f), &want));
        }
    }
}
