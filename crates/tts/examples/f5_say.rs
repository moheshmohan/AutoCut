//! Speak a line with an IndicF5 release in the voice of a WAV clip and its transcript, and
//! report the time taken (real-time factor).
//!
//! `cargo run --release -p filmcraft-tts --features indicf5 --example f5_say -- <release dir>
//! <out.wav> "<text>" <reference.wav> "<reference transcript>" [--steps N] [--cfg X]
//! [--speed X] [--seed S]`

use filmcraft_tts::f5::{F5, RATE, Settings};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut s = Settings::default();
    let mut i = 0;
    while i < args.len() {
        let take = |args: &mut Vec<String>, i: usize| -> Option<String> {
            let v = args.get(i + 1).cloned();
            args.drain(i..(i + 2).min(args.len()));
            v
        };
        match args[i].as_str() {
            "--steps" => s.steps = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.steps),
            "--cfg" => s.cfg = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.cfg),
            "--speed" => s.speed = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.speed),
            "--seed" => s.seed = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.seed),
            _ => i += 1,
        }
    }
    let (Some(dir), Some(out), Some(line), Some(wav), Some(transcript)) = (args.first(), args.get(1), args.get(2), args.get(3), args.get(4)) else {
        eprintln!("usage: f5_say <release dir> <out.wav> \"<text>\" <reference.wav> \"<transcript>\" [--steps N] [--cfg X] [--speed X] [--seed S]");
        std::process::exit(2);
    };
    let t0 = std::time::Instant::now();
    let m = F5::load(std::path::Path::new(dir))?;
    eprintln!("loaded on {:?} in {:.1} s", m.device().location(), t0.elapsed().as_secs_f64());
    let (x, rate) = filmcraft_tts::wav::read(std::path::Path::new(wav))?;
    let x = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, RATE);
    let r = m.reference(&x, transcript)?;
    let t1 = std::time::Instant::now();
    let audio = m.speak(&r, line, &s, &mut |_, _| true)?;
    let secs = t1.elapsed().as_secs_f64();
    let len = audio.len() as f64 / f64::from(RATE);
    eprintln!("{len:.2} s of speech in {secs:.2} s (real-time factor {:.2}) after a {:.1} s reference", secs / len.max(1e-9), r.seconds());
    filmcraft_tts::wav::write(std::path::Path::new(out), &audio, RATE)?;
    Ok(())
}
