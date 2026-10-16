//! Speaker similarity of WAV files to a reference recording: the cosine of their OpenVoice V2
//! tone colours (1 = the same timbre). The first WAV is the reference.
//!
//! `cargo run --release -p filmcraft-tts --features kokoro --example voice_similarity -- <OpenVoice
//! converter dir> <reference.wav> <candidate.wav>…`

use filmcraft_tts::openvoice::{Converter, RATE};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    dot / (n(a) * n(b)).max(1e-12)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(dir), Some(reference)) = (args.first(), args.get(1)) else {
        eprintln!("usage: voice_similarity <converter dir> <reference.wav> <candidate.wav>…");
        std::process::exit(2);
    };
    let conv = Converter::load(std::path::Path::new(dir))?;
    let colour = |path: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (x, rate) = filmcraft_tts::wav::read(std::path::Path::new(path))?;
        let speech = filmcraft_tts::clone::speech_only(&x, rate);
        Ok(conv.tone_colour(&filmcraft_audio_dsp::resample::resample_sinc(&speech, rate, RATE))?)
    };
    let r = colour(reference)?;
    for c in args.iter().skip(2) {
        println!("{:.3}  {c}", cosine(&r, &colour(c)?));
    }
    Ok(())
}
