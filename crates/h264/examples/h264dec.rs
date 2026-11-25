//! Decode an Annex-B H.264 file to raw planar YUV (ffmpeg's yuv420p / yuv422p, or their 16-bit
//! little-endian forms above 8 bits): `h264dec in.h264 [out.yuv]`.
//! Prints timing information to stderr. `H264_BENCH_ITERS=n` (with `H264_THREADS=t`, and
//! `H264_DRAFT=1` for draft mode) turns it into a benchmark.

use filmcraft_h264::Decoder;
use std::io::Write;

/// Split an Annex-B stream into access units: a new one starts at an AUD, SPS, PPS or SEI after a
/// slice, or at a slice with first_mb_in_slice 0 (the decoder takes at most so many pictures per
/// call, so a whole file is fed one access unit at a time).
fn access_units(data: &[u8]) -> Vec<&[u8]> {
    let mut aus = Vec::new();
    let (mut start, mut seen_slice, mut i) = (0usize, false, 0usize);
    while i + 3 < data.len() {
        if data[i] != 0 || data[i + 1] != 0 || data[i + 2] != 1 {
            i += 1;
            continue;
        }
        let sc = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
        let t = data[i + 3] & 0x1f;
        let slice = t == 1 || t == 5;
        let first_mb_zero = slice && data.get(i + 4).is_some_and(|b| b & 0x80 != 0);
        if seen_slice && (matches!(t, 6..=9) || first_mb_zero) {
            aus.push(&data[start..sc]);
            start = sc;
            seen_slice = false;
        }
        seen_slice |= slice;
        i += 3;
    }
    if start < data.len() {
        aus.push(&data[start..]);
    }
    aus
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: h264dec in.h264 [out.yuv]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    let aus = access_units(&data);
    // H264_BENCH_ITERS=n: decode n times without output and report the best run
    if let Some(n) = std::env::var("H264_BENCH_ITERS").ok().and_then(|v| v.parse::<usize>().ok()) {
        let mut best = f64::MAX;
        let mut frames = 0;
        let threads = std::env::var("H264_THREADS").ok().and_then(|v| v.parse::<usize>().ok());
        for _ in 0..n {
            let mut dec = threads.map(Decoder::with_threads).unwrap_or_default();
            dec.set_draft(std::env::var_os("H264_DRAFT").is_some());
            let t0 = std::time::Instant::now();
            frames = 0;
            for (i, au) in aus.iter().enumerate() {
                frames += dec.decode(au, i as i64).expect("decode").len();
            }
            frames += dec.flush().len();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!("best of {n}: {frames} frames in {best:.3}s ({:.1} fps)", frames as f64 / best);
        return;
    }
    let mut out = args.get(2).map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create output")));
    let mut dec = std::env::var("H264_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).map(Decoder::with_threads).unwrap_or_default();
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    let mut write = |pics: Vec<filmcraft_h264::Picture>| {
        for p in pics {
            n += 1;
            if let Some(o) = out.as_mut() {
                for plane in [&p.y, &p.u, &p.v] {
                    o.write_all(&plane.to_le_bytes()).unwrap();
                }
            }
        }
    };
    for (i, au) in aus.iter().enumerate() {
        match dec.decode(au, i as i64) {
            Ok(p) => write(p),
            Err(e) => eprintln!("decode error at access unit {i}: {e}"),
        }
    }
    write(dec.flush());
    let dt = t0.elapsed().as_secs_f64();
    if let Some(e) = dec.take_error() {
        eprintln!("decode error: {e}");
    }
    if std::env::var("H264_STATS").is_ok() {
        eprintln!("{:#?}", dec.stats());
    }
    eprintln!("{n} frames in {dt:.3}s ({:.1} fps)", n as f64 / dt);
}
