//! Render the same pieces on a SoundFont and on the oscillators, for listening side by side.
//!
//! `cargo run --release -p filmcraft-soundfont --example render_samples -- <font.sf2> <out dir>`
//!
//! Writes `<piece> - soundfont.wav` and `<piece> - oscillators.wav` (48 kHz stereo float) for each
//! piece, and prints how long each took.

use std::path::{Path, PathBuf};

use filmcraft_audio_dsp::music::{self, Energy, Genre, Mood, Request};
use filmcraft_audio_dsp::synth::Stereo;
use filmcraft_soundfont::{SoundFont, SoundFontBank, generate};
use filmcraft_time::Tick;

fn wav(path: &Path, s: &Stereo) -> std::io::Result<()> {
    let frames = s.left.len().min(s.right.len());
    let data = frames * 8;
    let mut v = Vec::with_capacity(44 + data);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data as u32).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&s.sample_rate.to_le_bytes());
    v.extend_from_slice(&(s.sample_rate * 8).to_le_bytes());
    v.extend_from_slice(&8u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(data as u32).to_le_bytes());
    for (l, r) in s.left.iter().zip(&s.right) {
        v.extend_from_slice(&l.to_le_bytes());
        v.extend_from_slice(&r.to_le_bytes());
    }
    std::fs::write(path, v)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(font), Some(out)) = (args.get(1), args.get(2)) else {
        return Err("usage: render_samples <font.sf2> <out dir>".into());
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    let t = std::time::Instant::now();
    let font = SoundFont::open(Path::new(font))?;
    println!("loaded {} ({} presets, {:.1} MB) in {:.2} s", font.name(), font.presets(), font.bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
    let bank = SoundFontBank::new(font);
    let pieces = [
        ("epic cinematic", Mood::Epic, Genre::Cinematic, 50.0, 7, Some(Energy::Curve(vec![0.3, 0.5, 0.8, 1.0, 0.6]))),
        ("calm lofi", Mood::Calm, Genre::Lofi, 40.0, 3, None),
    ];
    for (name, mood, genre, seconds, seed, energy) in pieces {
        let req = Request { genre: Some(genre), seed, energy, stems: true, ..Request::new(mood, Tick::from_seconds_f64(seconds)) };
        let t = std::time::Instant::now();
        let (sf, note) = generate(&req, 48_000, &bank)?;
        let t_sf = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let osc = music::generate(&req, 48_000)?;
        let t_osc = t.elapsed().as_secs_f64();
        assert_eq!(sf.plan, osc.plan, "the same arrangement");
        wav(&out.join(format!("{name} - soundfont.wav")), &sf.mix)?;
        wav(&out.join(format!("{name} - oscillators.wav")), &osc.mix)?;
        println!(
            "{name}: {} at {} BPM, {} bars; {} in {t_sf:.2} s, oscillators in {t_osc:.2} s{}",
            sf.plan.key_name(),
            sf.plan.tempo.bpm(),
            sf.plan.arrangement.bars,
            sf.instruments,
            note.map(|n| format!(" ({n})")).unwrap_or_default()
        );
        // each stem's level, dB RMS: the font's next to the oscillators'
        let db = |s: &Stereo| {
            let e: f64 = s.left.iter().chain(&s.right).map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / (s.left.len() * 2).max(1) as f64;
            10.0 * e.max(1e-12).log10()
        };
        for ((stem, a), (_, b)) in sf.stems.iter().zip(&osc.stems) {
            println!("  {:7} soundfont {:6.1} dB, oscillators {:6.1} dB", stem.id(), db(a), db(b));
        }
    }
    Ok(())
}
