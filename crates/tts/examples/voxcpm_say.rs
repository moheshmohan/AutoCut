//! Speak a line with a VoxCPM release, optionally in the voice of a WAV clip and its transcript,
//! and report the time taken (real-time factor).
//!
//! `cargo run --release -p filmcraft-tts --features voxcpm --example voxcpm_say -- <release dir>
//! <out.wav> "<text>" [<prompt.wav> ["<prompt transcript>"]] [--steps N] [--cfg X] [--seed S]`
//! (VoxCPM2 without a transcript: a reference-only prompt.)

use filmcraft_tts::voxcpm::{Settings, VoxCpm, text};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut s = Settings::default();
    let mut denoise: Option<f32> = None;
    let mut i = 0;
    while i < args.len() {
        let take = |args: &mut Vec<String>, i: usize| -> Option<String> {
            let v = args.get(i + 1).cloned();
            args.drain(i..(i + 2).min(args.len()));
            v
        };
        match args[i].as_str() {
            "--steps" => s.timesteps = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.timesteps),
            "--cfg" => s.cfg = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.cfg),
            "--seed" => s.seed = take(&mut args, i).and_then(|v| v.parse().ok()).unwrap_or(s.seed),
            "--denoise" => denoise = take(&mut args, i).and_then(|v| v.parse().ok()),
            _ => i += 1,
        }
    }
    let (Some(dir), Some(out), Some(line)) = (args.first(), args.get(1), args.get(2)) else {
        eprintln!("usage: voxcpm_say <release dir> <out.wav> \"<text>\" [<prompt.wav> \"<transcript>\"] [--steps N] [--cfg X] [--seed S]");
        std::process::exit(2);
    };
    let t0 = std::time::Instant::now();
    let m = VoxCpm::load(std::path::Path::new(dir))?;
    eprintln!("loaded {:?} in {:.1?} on {:?}", m.variant(), t0.elapsed(), filmcraft_tts::voxcpm::device().location());
    let prompt = match args.get(3) {
        Some(wav) => {
            let (x, rate) = filmcraft_tts::wav::read(std::path::Path::new(wav))?;
            let mut x = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, m.encode_rate());
            if let Some(db) = denoise {
                use filmcraft_audio_dsp::AudioEffect;
                let mut d = filmcraft_audio_dsp::effects::restoration::DeNoise::new(m.encode_rate() as f32, 1);
                d.set_param("reduction", db);
                let lat = d.latency();
                let n = x.len();
                x.resize(n + lat, 0.0);
                for b in x.chunks_mut(4096) {
                    d.process(&mut [b]);
                }
                x.drain(..lat.min(x.len()));
                x.truncate(n);
            }
            let t = std::time::Instant::now();
            let normalised = match args.get(4) {
                Some(tr) => Some(text::sentences(tr)?.join(" ")),
                None => None,
            };
            let p = m.prompt(&x, normalised.as_deref())?;
            eprintln!("prompt: {:.1} s, {} patches, encoded in {:.1?}: {normalised:?}", p.seconds, p.patches(), t.elapsed());
            Some(p)
        }
        None => None,
    };
    let mut audio = Vec::new();
    let t1 = std::time::Instant::now();
    for piece in text::pieces(line)? {
        match piece {
            text::Piece::Say(u) => {
                let t = std::time::Instant::now();
                let a = m.speak(&u, prompt.as_ref(), &s, &mut |_, _| true)?;
                let secs = a.len() as f64 / f64::from(m.output_rate());
                eprintln!("{u:?}: {secs:.2} s in {:.2?} (RTF {:.2})", t.elapsed(), t.elapsed().as_secs_f64() / secs.max(1e-6));
                audio.extend(a);
            }
            text::Piece::Pause(ms) => audio.resize(audio.len() + (u64::from(ms) * u64::from(m.output_rate()) / 1000) as usize, 0.0),
        }
    }
    let secs = audio.len() as f64 / f64::from(m.output_rate());
    eprintln!("total {secs:.2} s of audio in {:.2?} (RTF {:.2})", t1.elapsed(), t1.elapsed().as_secs_f64() / secs.max(1e-6));
    filmcraft_tts::wav::write(std::path::Path::new(out), &audio, m.output_rate())?;
    eprintln!("wrote {out}");
    Ok(())
}
