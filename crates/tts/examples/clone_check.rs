//! Measure a cloned voice: speak a line with the clone and with its base voice, and print how
//! close each result's tone colour is to the speaker's (cosine; 1 = same timbre). Writes both
//! as WAVs for listening.
//!
//! `cargo run -p filmcraft-tts --features kokoro --example clone_check -- <models dir> clone:<name> [out dir]`

use filmcraft_tts::openvoice::{Converter, RATE};
use filmcraft_tts::{Params, Voice, catalog, clone};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    dot / (n(a) * n(b)).max(1e-12)
}

fn wav(path: &std::path::Path, x: &[f32], rate: u32) -> std::io::Result<()> {
    let mut v = Vec::with_capacity(44 + x.len() * 2);
    let data = (x.len() * 2) as u32;
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data.to_le_bytes());
    for s in x {
        v.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, v)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(models), Some(id)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: clone_check <models dir> clone:<name> [out dir]");
        std::process::exit(2);
    };
    let models = std::path::Path::new(models);
    let out = args.get(3).map(std::path::PathBuf::from);
    let text = "Number seven: a four legged robot dog that walked right up to me at the expo. It could climb stairs, and it even danced.";
    let cv = clone::load(id, models)?;
    let p = cv.profile().clone();
    let conv = Converter::load(&catalog::clone_package_dir(models))?;
    let base = filmcraft_tts::neural::load(&p.base_voice, models)?;
    let t0 = std::time::Instant::now();
    let a = base.synthesize(text, &Params::default())?;
    let t1 = t0.elapsed();
    let b = cv.synthesize(text, &Params::default())?;
    let t2 = t0.elapsed() - t1;
    let colour = |x: &[f32], rate: u32| conv.tone_colour(&filmcraft_audio_dsp::resample::resample_sinc(x, rate, RATE));
    let ca = colour(&a.samples, a.sample_rate)?;
    let cb = colour(&b.samples, b.sample_rate)?;
    println!("speech: {:.1} s  (base synth {:.1?}, clone synth {:.1?})", b.seconds(), t1, t2);
    println!("cosine(base voice, speaker)  = {:.3}", cosine(&ca, &p.target_colour));
    println!("cosine(cloned voice, speaker) = {:.3}", cosine(&cb, &p.target_colour));
    println!("cosine(cloned voice, base)    = {:.3}", cosine(&cb, &ca));
    if let Some(dir) = out {
        std::fs::create_dir_all(&dir)?;
        wav(&dir.join("base.wav"), &a.samples, a.sample_rate)?;
        wav(&dir.join("clone.wav"), &b.samples, b.sample_rate)?;
        println!("wrote {}", dir.display());
    }
    Ok(())
}
