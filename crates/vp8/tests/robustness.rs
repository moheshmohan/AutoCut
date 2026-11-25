//! Damaged, truncated, reordered and random input must never panic (errors are fine).

mod common;

use filmcraft_vp8::{Decoder, Error};

fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn decode_all(frames: &[Vec<u8>]) {
    let mut dec = Decoder::new();
    for (i, f) in frames.iter().enumerate() {
        let _ = dec.is_disposable(f);
        let _ = dec.decode(f, i as i64);
        let _ = filmcraft_vp8::keyframe_info(f);
    }
}

#[test]
fn corrupted_streams_do_not_panic() {
    for name in ["default", "altref_2pass", "partitions8", "realtime_segments", "version3_fullpixel", "odd_size"] {
        let f = common::fixture(name);
        let Some((ivf, _)) = common::ensure(f) else { return };
        let data = std::fs::read(ivf).unwrap();
        let frames: Vec<Vec<u8>> = common::ivf_frames(&data).into_iter().map(|f| f.to_vec()).collect();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ name.len() as u64;
        for round in 0..12 {
            let mut d = frames.clone();
            for fr in d.iter_mut() {
                if fr.is_empty() {
                    continue;
                }
                match round % 4 {
                    0 => {
                        for _ in 0..1 + round {
                            let i = (xorshift(&mut seed) as usize) % fr.len();
                            fr[i] ^= 1 << (xorshift(&mut seed) % 8);
                        }
                    }
                    1 => {
                        let n = (xorshift(&mut seed) as usize) % (fr.len() + 1);
                        fr.truncate(n);
                    }
                    2 => {
                        let i = (xorshift(&mut seed) as usize) % fr.len().min(24);
                        fr[i] = xorshift(&mut seed) as u8;
                    }
                    // frames left intact; their order is shuffled below
                    _ => {}
                }
            }
            if round % 4 == 3 {
                for i in (1..d.len()).rev() {
                    let j = (xorshift(&mut seed) as usize) % (i + 1);
                    d.swap(i, j);
                }
            }
            decode_all(&d);
        }
    }
}

#[test]
fn random_bytes_and_tiny_frames_do_not_panic() {
    let mut seed = 42u64;
    let mut dec = Decoder::new();
    for len in 0..200 {
        let f: Vec<u8> = (0..len).map(|_| xorshift(&mut seed) as u8).collect();
        let _ = dec.decode(&f, len as i64);
    }
    // a key frame header claiming a large size followed by garbage
    let mut key = vec![0u8; 64];
    key[0] = 0x10; // key frame, version 0, shown, first partition 0 bytes
    key[3..6].copy_from_slice(&[0x9d, 0x01, 0x2a]);
    key[6..8].copy_from_slice(&1920u16.to_le_bytes());
    key[8..10].copy_from_slice(&1080u16.to_le_bytes());
    assert!(dec.decode(&key, 0).is_err(), "empty first partition must not reconstruct a frame");

    // A header can declare nearly 16K square dimensions with just ten bytes. It must be rejected
    // before the macroblock map and reference planes are allocated.
    key[0] = 0x30; // one-byte first partition
    key[10] = 0;
    key[6..8].copy_from_slice(&16_383u16.to_le_bytes());
    key[8..10].copy_from_slice(&16_383u16.to_le_bytes());
    assert!(dec.decode(&key, 0).is_err(), "oversized key frame must be rejected");
}

/// A key frame header of `w` x `h` with a one-byte first partition and garbage after it.
fn key_header(w: u16, h: u16) -> Vec<u8> {
    let mut key = vec![0u8; 64];
    key[0] = 0x30; // key frame, version 0, shown, one-byte first partition
    key[3..6].copy_from_slice(&[0x9d, 0x01, 0x2a]);
    key[6..8].copy_from_slice(&w.to_le_bytes());
    key[8..10].copy_from_slice(&h.to_le_bytes());
    key
}

fn over_limit(r: filmcraft_vp8::Result<Option<filmcraft_vp8::Picture>>) -> bool {
    matches!(r, Err(Error::Invalid(m)) if m.contains("memory limit"))
}

/// The frame limit is the caller's (`Decoder::with_max_luma_samples`), counted on the
/// macroblock grid: 70x40 pads to 80x48 (3840 samples), refused one sample below and decoded
/// at it. The default stays 2^26: 8192x8192 is within it, one macroblock row more is not, and
/// a reset keeps the limit.
#[test]
fn the_frame_limit_is_the_callers() {
    assert_eq!(Decoder::new().max_luma_samples(), 1 << 26);
    assert_eq!(filmcraft_vp8::MAX_LUMA_SAMPLES, 1 << 26);
    let small = key_header(70, 40);
    assert!(over_limit(Decoder::with_max_luma_samples(80 * 48 - 1).decode(&small, 0)));
    assert!(!over_limit(Decoder::with_max_luma_samples(80 * 48).decode(&small, 0)));
    assert!(!over_limit(Decoder::new().decode(&small, 0)));
    let tall = key_header(8192, 8193);
    assert!(over_limit(Decoder::new().decode(&tall, 0)));
    let mut dec = Decoder::with_max_luma_samples(80 * 48 - 1);
    dec.reset();
    assert_eq!(dec.max_luma_samples(), 80 * 48 - 1);
    assert!(over_limit(dec.decode(&small, 0)));
    assert!(over_limit(Decoder::with_max_luma_samples(0).decode(&key_header(1, 1), 0)));
}

#[test]
fn inter_frame_without_key_frame_is_an_error() {
    let f = common::fixture("default");
    let Some((ivf, _)) = common::ensure(f) else { return };
    let data = std::fs::read(ivf).unwrap();
    let frames = common::ivf_frames(&data);
    let mut dec = Decoder::new();
    assert!(matches!(dec.decode(frames[1], 1), Err(Error::MissingReference(_))));
    // and decoding works from the next key frame on; after a reset again only from a key frame
    assert!(dec.decode(frames[0], 0).unwrap().is_some());
    assert!(dec.decode(frames[1], 1).unwrap().is_some());
    dec.reset();
    assert!(matches!(dec.decode(frames[2], 2), Err(Error::MissingReference(_))));
    assert!(filmcraft_vp8::is_keyframe(frames[0]) && !filmcraft_vp8::is_keyframe(frames[1]));
}
