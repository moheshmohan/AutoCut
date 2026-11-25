//! Damaged, truncated and random input must never panic or hang (errors are fine).

mod common;

use filmcraft_vp9::Decoder;
use std::time::{Duration, Instant};

fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn decode_all(frames: &[Vec<u8>], threads: usize) {
    let mut dec = Decoder::with_threads(threads);
    for (i, f) in frames.iter().enumerate() {
        let _ = dec.decode(f, i as i64);
    }
    let _ = dec.flush();
}

#[test]
fn corrupted_streams_do_not_panic() {
    for name in ["default_good", "altref_hidden", "profile2_10bit", "tiles_cols4", "aq_segmentation", "lossless", "profile1_444", "tiles_rows4"] {
        let f = common::fixture(name);
        let Some((ivf, _)) = common::ensure(f) else { return };
        let data = std::fs::read(ivf).unwrap();
        let frames: Vec<Vec<u8>> = common::ivf_frames(&data).into_iter().map(|f| f.to_vec()).collect();
        let mut seed = 0x1234_5678_9abc_def0u64 ^ name.len() as u64;
        for round in 0..10 {
            let t0 = Instant::now();
            let mut d = frames.clone();
            for _ in 0..(1 + round * 4) {
                let fi = (xorshift(&mut seed) as usize) % d.len();
                let fr = &mut d[fi];
                if fr.is_empty() {
                    continue;
                }
                let pos = (xorshift(&mut seed) as usize) % fr.len();
                match round % 3 {
                    0 => fr[pos] ^= 1 << (xorshift(&mut seed) % 8),
                    1 => fr[pos] = xorshift(&mut seed) as u8,
                    _ => fr.truncate(pos),
                }
            }
            for threads in [1, 4] {
                decode_all(&d, threads);
            }
            assert!(t0.elapsed() < Duration::from_secs(60), "{name} round {round} too slow");
        }
    }
}

#[test]
fn truncated_frames_do_not_panic() {
    let f = common::fixture("default_good");
    let Some((ivf, _)) = common::ensure(f) else { return };
    let data = std::fs::read(ivf).unwrap();
    let frames: Vec<Vec<u8>> = common::ivf_frames(&data).into_iter().map(|f| f.to_vec()).collect();
    for cut in [1usize, 2, 3, 5, 8, 13, 21, 40, 100, 400] {
        let d: Vec<Vec<u8>> = frames.iter().map(|f| f[..cut.min(f.len())].to_vec()).collect();
        decode_all(&d, 1);
        decode_all(&d, 3);
    }
}

#[test]
fn garbage_input_is_rejected_gracefully() {
    let mut seed = 42u64;
    for len in [0usize, 1, 2, 5, 30, 100, 5000] {
        for k in 0..40 {
            let mut g: Vec<u8> = (0..len).map(|_| xorshift(&mut seed) as u8).collect();
            if len > 10 && k % 2 == 0 {
                // A plausible key frame start: frame marker, profile 0, key frame, sync code.
                g[0] = 0x82;
                g[1] = 0x49;
                g[2] = 0x83;
                g[3] = 0x42;
            }
            let mut dec = Decoder::with_threads(1 + k % 3);
            let _ = dec.decode(&g, 0);
            let _ = dec.decode(&g, 1);
        }
    }
    // Inter frame / show_existing without references.
    let mut dec = Decoder::new();
    assert!(dec.decode(&[0x88], 0).is_err());
    assert!(dec.decode(&[0x86, 0, 0, 0, 0], 0).is_err());
}

/// Huge declared frame sizes must not allocate absurd amounts or panic.
#[test]
fn oversized_frame_is_rejected() {
    // Key frame header: marker 2, profile 0, show_existing 0, key, show, no error res,
    // sync, color (4 bits), width-1 = 0xffff, height-1 = 0xffff.
    let bits = "10".to_string() + "00" + "0" + "0" + "1" + "0" + "010010011000001101000010" + "0000" + &"1".repeat(32);
    let mut v = vec![0u8; bits.len().div_ceil(8) + 8];
    for (i, c) in bits.chars().enumerate() {
        if c == '1' {
            v[i / 8] |= 0x80 >> (i % 8);
        }
    }
    let mut dec = Decoder::new();
    assert!(dec.decode(&v, 0).is_err());
}

/// A key frame header of `w` x `h` (profile 0, 8-bit 4:2:0), then the rest of the uncompressed
/// header as zeros (28 bits) and header_size_in_bytes = 1.
fn key_frame_header(w: u16, h: u16) -> Vec<u8> {
    let size = |n: u16| format!("{:016b}", n - 1);
    let bits =
        "10".to_string() + "00" + "0" + "0" + "1" + "0" + "010010011000001101000010" + "0000" + &size(w) + &size(h) + &"0".repeat(28) + "0000000000000001";
    let mut v = vec![0u8; bits.len().div_ceil(8) + 8];
    for (i, c) in bits.chars().enumerate() {
        if c == '1' {
            v[i / 8] |= 0x80 >> (i % 8);
        }
    }
    v
}

fn over_budget(r: &filmcraft_vp9::Result<Vec<filmcraft_vp9::Picture>>) -> bool {
    r.as_ref().is_err_and(|e| e.to_string().contains("memory budget"))
}

/// A key frame whose picture store would not fit the decoder's memory budget is refused before
/// anything is allocated for it. A 64x64 4:2:0 frame is 14336 bytes of sample bands and its
/// 8-bit output picture 6144; one thread holds 9 frames (eight references, the new one) and 3
/// pictures.
#[test]
fn frame_over_the_memory_budget_is_rejected() {
    let v = key_frame_header(64, 64);
    let need = 9 * 14336 + 3 * 6144;
    let mut dec = Decoder::with_threads(1);
    assert_eq!(dec.memory_budget(), filmcraft_vp9::DECODE_MEMORY_BUDGET);
    dec.set_memory_budget(need - 1);
    assert!(over_budget(&dec.decode(&v, 0)));
    let mut dec = Decoder::with_threads(1);
    dec.set_memory_budget(need);
    assert!(!over_budget(&dec.decode(&v, 0)));
    // a reset keeps the budget
    let mut dec = Decoder::with_threads(1);
    dec.set_memory_budget(need - 1);
    dec.reset();
    assert!(over_budget(&dec.decode(&v, 0)));
    // frame threads hold more frames in their post stage (with the `threads` feature)
    let mut dec = Decoder::with_threads(4);
    dec.set_memory_budget(need);
    assert_eq!(over_budget(&dec.decode(&v, 0)), dec.threads() > 1);
}

/// The frame-size limit is the caller's: 8192x8200 is over the default 2^26 luma samples and in
/// at a raised limit, where the memory budget still bounds the picture store (16384x16384, the
/// largest frame accepted, needs 9.7 GB on one thread: over a 4 GiB budget).
#[test]
fn a_raised_frame_limit_is_still_bounded_by_the_budget() {
    let over_limit = |r: &filmcraft_vp9::Result<Vec<filmcraft_vp9::Picture>>| r.as_ref().is_err_and(|e| e.to_string().contains("memory limit"));
    assert_eq!(Decoder::new().max_picture_luma_samples(), filmcraft_vp9::MAX_LUMA_SAMPLES);
    let tall = key_frame_header(8192, 8200);
    assert!(over_limit(&Decoder::with_threads(1).decode(&tall, 0)));
    let mut dec = Decoder::with_threads(1);
    dec.set_max_picture_luma_samples(1 << 27);
    assert_eq!(dec.max_picture_luma_samples(), 1 << 27);
    dec.set_memory_budget(0);
    let r = dec.decode(&tall, 0);
    assert!(!over_limit(&r) && over_budget(&r), "{r:?}");
    let widest = key_frame_header(16384, 16384);
    let mut dec = Decoder::with_threads(1);
    dec.set_max_picture_luma_samples(u64::MAX);
    dec.set_memory_budget(4 << 30);
    assert!(over_budget(&dec.decode(&widest, 0)));
    // and the small frame that fits the default limit is not refused by either
    let mut dec = Decoder::with_threads(1);
    dec.set_max_picture_luma_samples(64 * 64);
    assert!(!over_limit(&dec.decode(&key_frame_header(64, 64), 0)));
    let mut dec = Decoder::with_threads(1);
    dec.set_max_picture_luma_samples(64 * 64 - 1);
    assert!(over_limit(&dec.decode(&key_frame_header(64, 64), 0)));
}

/// A real stream (352x288, 20 frames with references) decodes exactly as by default under a
/// budget that just fits its picture store (one thread: 9 frames of 430080 bytes of bands, 384
/// samples wide on the superblock grid with scratch rows, and 3 output pictures of 152064
/// bytes); one byte less and its key frame is refused, and every frame after it is an error (no
/// references), never a panic or an allocation for the refused size.
#[test]
fn a_real_stream_at_its_budget_decodes_as_by_default() {
    let f = common::fixture("default_good");
    let Some((ivf, _)) = common::ensure(f) else { return };
    let data = std::fs::read(ivf).unwrap();
    let frames = common::ivf_frames(&data);
    let need = 9 * 430_080 + 3 * 152_064;
    let decode = |budget: Option<u64>| {
        let mut dec = Decoder::with_threads(1);
        if let Some(b) = budget {
            dec.set_memory_budget(b);
        }
        let (mut pics, mut errors) = (Vec::new(), Vec::new());
        for (i, fr) in frames.iter().enumerate() {
            match dec.decode(fr, i as i64) {
                Ok(p) => pics.extend(p),
                Err(e) => errors.push(e),
            }
        }
        pics.extend(dec.flush());
        (pics, errors)
    };
    let (want, none) = decode(None);
    assert!(none.is_empty() && want.len() == frames.len());
    let (got, none) = decode(Some(need));
    assert!(none.is_empty());
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(g.y == w.y && g.u == w.u && g.v == w.v && g.pts == w.pts);
    }
    let (got, errors) = decode(Some(need - 1));
    assert!(got.is_empty());
    assert_eq!(errors.len(), frames.len());
    assert!(errors[0].to_string().contains("memory budget"), "{}", errors[0]);
}
