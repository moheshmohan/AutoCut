//! Ogg open speed (ignored by default; run with
//! `cargo test --release -p filmcraft-ogg --test perf -- --ignored --nocapture`).
//!
//! A synthetic Opus stream of 16,384 pages of 4 KiB (64 MiB, each page CRC checked at open), best
//! of seven runs, and the CRC alone over the same bytes.

use std::hint::black_box;
use std::time::Instant;

fn page(out: &mut Vec<u8>, flags: u8, granule: i64, seq: u32, lacing: &[u8], body: &[u8]) {
    let start = out.len();
    out.extend_from_slice(b"OggS");
    out.push(0);
    out.push(flags);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&7u32.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.push(lacing.len() as u8);
    out.extend_from_slice(lacing);
    out.extend_from_slice(body);
    let crc = filmcraft_ogg::crc32(&out[start..]);
    out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
}

fn best(runs: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn perf_open() {
    let mut f = Vec::new();
    let mut head = b"OpusHead".to_vec();
    head.extend([1, 2, 0, 0]);
    head.extend(48_000u32.to_le_bytes());
    head.extend([0, 0, 0]);
    page(&mut f, 0x02, 0, 0, &[head.len() as u8], &head);
    let tags = b"OpusTags\0\0\0\0\0\0\0\0";
    page(&mut f, 0, 0, 1, &[tags.len() as u8], tags);
    // pages of 16 packets of 254 bytes (a lacing value below 255 ends a packet), each a CELT
    // 20 ms TOC and noise
    let mut x = 0x2545_F491u32;
    for k in 0..16_384u32 {
        let body: Vec<u8> = (0..16 * 254)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                if i % 254 == 0 { 0xF8 } else { x as u8 }
            })
            .collect();
        page(&mut f, 0, i64::from(k + 1) * 16 * 960, k + 2, &[254; 16], &body);
    }
    let o = filmcraft_ogg::open(f.as_slice()).unwrap();
    assert_eq!(o.streams[0].packets.len(), 16 * 16_384);
    assert!(o.warnings.is_empty(), "{:?}", o.warnings);
    let t = best(7, || filmcraft_ogg::open(f.as_slice()).map_or(0, |o| o.streams[0].packets.len() as u64));
    println!("open: {:.1} ms for {} MiB ({:.2} GB/s)", t * 1e3, f.len() >> 20, f.len() as f64 / t / 1e9);
    let t = best(7, || u64::from(filmcraft_ogg::crc32(&f)));
    println!("crc32: {:.2} GB/s", f.len() as f64 / t / 1e9);
}
