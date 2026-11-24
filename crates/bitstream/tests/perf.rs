//! Bit reader speed (ignored by default; run with
//! `cargo test --release -p filmcraft-bitstream --test perf -- --ignored --nocapture`).
//!
//! Synthetic inputs, best of seven runs: fixed-width reads, single bits, Exp-Golomb codes as
//! slice headers and CAVLC / CABAC set-up read them, and emulation-prevention removal over
//! NAL-like payloads.

use filmcraft_bitstream::{BitReader, BitWriter, unescape_rbsp};
use std::hint::black_box;
use std::time::Instant;

/// Fixed-seed xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Best wall time of `runs` calls of `f`, in seconds.
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
fn perf_bit_reader() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let data: Vec<u8> = (0..1 << 20).map(|_| rng.next() as u8).collect();
    let bits = data.len() * 8;

    // fixed-width reads of 1..=24 bits
    let widths: Vec<u32> = (0..64).map(|i| 1 + (i * 7 % 24) as u32).collect();
    let mut reads = 0u64;
    let t = best(7, || {
        let mut r = BitReader::new(&data);
        let (mut sum, mut k) = (0u64, 0usize);
        while let Ok(v) = r.read_bits(widths[k % widths.len()]) {
            sum = sum.wrapping_add(u64::from(v));
            k += 1;
        }
        reads = k as u64;
        sum
    });
    println!("read_bits(1..=24): {:.2} ns/read ({reads} reads)", t * 1e9 / reads as f64);

    let t = best(7, || {
        let mut r = BitReader::new(&data);
        let mut ones = 0u64;
        while let Ok(b) = r.read_bit() {
            ones += u64::from(b);
        }
        ones
    });
    println!("read_bit: {:.2} ns/bit ({bits} bits)", t * 1e9 / bits as f64);

    let t = best(7, || {
        let r = BitReader::new(&data);
        let mut sum = 0u64;
        let mut r2 = r.clone();
        for p in (0..bits).step_by(13) {
            r2.seek_bits(p);
            sum = sum.wrapping_add(u64::from(r2.peek(32)));
        }
        sum
    });
    println!("peek(32): {:.2} ns/peek", t * 1e9 / (bits / 13) as f64);

    // Exp-Golomb codes: mostly small values, as headers and macroblock layers code them
    let mut w = BitWriter::new();
    let mut codes = 0u64;
    while w.bit_len() < bits {
        let v = match rng.next() % 16 {
            0..=9 => rng.next() % 4,
            10..=13 => rng.next() % 64,
            14 => rng.next() % 4096,
            _ => rng.next() % (1 << 20),
        };
        if codes.is_multiple_of(2) {
            w.write_ue(v as u32);
        } else {
            w.write_se(v as i32 - 2);
        }
        codes += 1;
    }
    w.rbsp_trailing();
    let eg = w.finish();
    let t = best(7, || {
        let mut r = BitReader::new(&eg);
        let mut sum = 0i64;
        for k in 0..codes {
            let v = if k.is_multiple_of(2) { r.read_ue().map(i64::from) } else { r.read_se().map(i64::from) };
            sum = sum.wrapping_add(v.unwrap_or(0));
        }
        sum as u64
    });
    println!("read_ue / read_se: {:.2} ns/code ({codes} codes)", t * 1e9 / codes as f64);

    let values: Vec<u32> = (0..1 << 20).map(|_| (rng.next() % 4096) as u32).collect();
    let t = best(7, || {
        let mut w = BitWriter::new();
        for &v in &values {
            w.write_ue(v);
        }
        w.bit_len() as u64
    });
    println!("write_ue: {:.2} ns/code", t * 1e9 / values.len() as f64);

    // NAL-like payloads: random bytes with an emulation-prevention byte every few hundred bytes
    let mut nal = Vec::with_capacity(4 << 20);
    while nal.len() < 4 << 20 {
        let run = 64 + (rng.next() % 512) as usize;
        nal.extend((0..run).map(|_| (rng.next() as u8) | 1));
        nal.extend([0, 0, 3]);
    }
    let t = best(7, || unescape_rbsp(&nal).len() as u64);
    println!("unescape_rbsp: {:.3} GB/s ({} MiB)", nal.len() as f64 / t / 1e9, nal.len() >> 20);
}
