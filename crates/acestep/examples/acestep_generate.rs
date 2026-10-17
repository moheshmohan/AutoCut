//! Generate music from a description with a downloaded ACE-Step package and write it as a WAV.
//!
//! `cargo run --release -p filmcraft-acestep --features model --example acestep_generate --
//! <package dir> <out.wav> <seconds> <seed> "<description>" [bpm] [key]`
//!
//! Prints the time of each stage and the real-time factor (generation time / music length).

use std::io::Write;
use std::time::Instant;

use filmcraft_acestep::Request;
use filmcraft_acestep::model::AceStep;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: acestep_generate <package dir> <out.wav> <seconds> <seed> \"<description>\" [bpm] [key]");
        std::process::exit(2);
    }
    let (dir, out) = (std::path::Path::new(&a[1]), &a[2]);
    let seconds: f64 = a[3].parse().unwrap_or(20.0);
    let seed: u64 = a[4].parse().unwrap_or(1);
    let mut req = Request::new(&a[5], seconds, seed);
    req.bpm = a.get(6).and_then(|b| b.parse().ok());
    req.key = a.get(7).cloned();
    let t0 = Instant::now();
    let m = match AceStep::load(dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    eprintln!("package read in {:.2?} (GPU: {})", t0.elapsed(), m.on_gpu());
    let t0 = Instant::now();
    let mut last = ("", Instant::now());
    let music = m.generate(&req, &mut |p| {
        if p.stage != last.0 {
            if !last.0.is_empty() {
                eprintln!("  {:<9} {:.2?}", last.0, last.1.elapsed());
            }
            last = (p.stage, Instant::now());
        }
        true
    });
    eprintln!("  {:<9} {:.2?}", last.0, last.1.elapsed());
    let music = match music {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let took = t0.elapsed().as_secs_f64();
    eprintln!("{:.1} s of music in {took:.1} s: real-time factor {:.2}", music.seconds(), took / music.seconds().max(1e-9));
    // 32-bit float stereo WAV
    let n = music.left.len();
    let data = n * 8;
    let mut v = Vec::with_capacity(44 + data);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&music.sample_rate.to_le_bytes());
    v.extend_from_slice(&(music.sample_rate * 8).to_le_bytes());
    v.extend_from_slice(&8u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(data as u32).to_le_bytes());
    for (l, r) in music.left.iter().zip(&music.right) {
        v.extend_from_slice(&l.to_le_bytes());
        v.extend_from_slice(&r.to_le_bytes());
    }
    match std::fs::File::create(out).and_then(|mut f| f.write_all(&v)) {
        Ok(()) => eprintln!("wrote {out}"),
        Err(e) => {
            eprintln!("{out}: {e}");
            std::process::exit(1);
        }
    }
}
