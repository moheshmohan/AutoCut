//! How fast DeepFilterNet 3 runs here: the real-time factor of the Voice Isolation effect (mono
//! and stereo, 48 kHz and 44.1 kHz) and of the network alone.
//!
//! `cargo run -p filmcraft-voice --release --example voice_bench -- <models dir> [seconds] [--dump <dir>]`
//!
//! `--dump` also writes the noisy test signal (`noisy.f32`, mono 48 kHz little-endian f32) and the
//! model's frame-by-frame output (`ours.f32`) for comparing with DeepFilterNet's own runtime.

use std::time::Instant;

use filmcraft_audio_dsp::AudioEffect;
use filmcraft_audio_dsp::effects::voice::{HOP, VoiceIsolation, VoiceModel, VoiceStream};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: voice_bench <models dir> [seconds] [--dump <dir>]");
        std::process::exit(2);
    };
    let secs: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let dump = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1)).map(std::path::PathBuf::from);
    let t0 = Instant::now();
    let model = match filmcraft_voice::load(std::path::Path::new(dir)) {
        Ok(m) => std::sync::Arc::new(m),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("model loaded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
    // a busy test signal: tones in bursts under noise
    let n = (48_000.0 * secs) as usize;
    let mut seed = 0x1234_5678u64;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let t = i as f32 / 48_000.0;
            let on = if (i / 9_000) % 3 == 0 { 0.0 } else { 0.3 };
            on * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
                + on * 0.5 * (2.0 * std::f32::consts::PI * 2400.0 * t).sin()
                + ((seed >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1
        })
        .collect();
    if let Some(d) = &dump {
        let _ = std::fs::create_dir_all(d);
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(d.join("noisy.f32"), bytes);
        let mut s = model.stream(1);
        let mut out = Vec::with_capacity(n);
        for h in x.as_chunks::<HOP>().0 {
            let mut a = [*h];
            s.process_hop(&mut a);
            out.extend_from_slice(&a[0]);
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(d.join("ours.f32"), bytes);
        println!("wrote {}", d.display());
    }
    // the network alone
    let mut s = model.stream(1);
    let t0 = Instant::now();
    for h in x.as_chunks::<HOP>().0 {
        let mut a = [*h];
        s.process_hop(&mut a);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("model stream, mono: {:.3} ms per 10 ms hop, real-time factor {:.4}", dt * 1000.0 / (n / HOP) as f64, dt / f64::from(secs));
    for (sr, ch) in [(48_000.0f32, 1usize), (48_000.0, 2), (44_100.0, 2)] {
        let m: std::sync::Arc<dyn VoiceModel> = model.clone();
        let mut fx = VoiceIsolation::with_model(sr, ch, Some(m));
        let len = (sr * secs) as usize;
        let mut chans: Vec<Vec<f32>> = (0..ch).map(|_| x.iter().copied().cycle().take(len).collect()).collect();
        let t0 = Instant::now();
        for k in (0..len).step_by(512) {
            let end = (k + 512).min(len);
            let mut refs: Vec<&mut [f32]> = chans.iter_mut().map(|c| &mut c[k..end]).collect();
            fx.process(&mut refs);
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "Voice Isolation effect, {sr} Hz, {ch} channel(s): real-time factor {:.4} ({:.0}× real time), latency {} samples",
            dt / f64::from(secs),
            f64::from(secs) / dt,
            fx.latency()
        );
    }
}
