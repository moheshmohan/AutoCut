//! The Malayalam narration lab: measure speech recognisers on Malayalam, and IndicF5 readings
//! with the strict heard-back check ([`filmcraft_tts::heard`]). Measurement only: the numbers
//! in the crate README's "Sentence starts" come from here.
//!
//! ```sh
//! # a recogniser on clips with their words (`<clip>.txt` beside each WAV)
//! ml_lab asr <model dir> <clip.wav>…
//! # what IndicF5 is given as the reference's words, both ways
//! ml_lab ref <models dir> <voice>
//! # read test lines (one per row) in readings ("base", "raw+room300+sent", …), hear and score
//! ml_lab gen <models dir> <voice> <recogniser dir> <lines.txt> <out dir> <reading>[,<reading>…] [takes [first take]]
//! ```
//!
//! `<models dir>` is a FilmCraft data directory's `models` (its `voices` beside it). Readings
//! are `+`-joined changes to how the voice read before the fix: `raw` (the reference's words as
//! given), `tail<ms>` (silence after the clip), `room<ms>` (room tone after it), `sent` (each
//! sentence its own utterance), `nfe<n>`, `cfg<x>`, `sway<x>`, `lead` (a lead-in cut away),
//! `new` (the defaults). The lab does not take the GPU lock: hold it (`filmcraft-cli locks hold
//! gpu`) while it runs.

use std::io::Write;
use std::path::{Path, PathBuf};

use filmcraft_speech::Transcriber;
use filmcraft_tts::clone::{MalayalamVoice, Reading};
use filmcraft_tts::heard;
use filmcraft_tts::malayalam::Piece;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn load_asr(dir: &Path) -> Res<filmcraft_speech::whisper::Whisper> {
    let id = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    Ok(filmcraft_speech::whisper::Whisper::load_on(dir, &id, std::env::var_os("LAB_CPU").is_none())?)
}

fn to16k(x: &[f32], rate: u32) -> Vec<f32> {
    if rate == 16_000 { x.to_vec() } else { filmcraft_audio_dsp::resample::resample_sinc(x, rate, 16_000) }
}

/// What `w` hears in `x` (any rate), in stretches it can decode.
fn hear(w: &filmcraft_speech::whisper::Whisper, x: &[f32], rate: u32) -> Res<String> {
    let mut h = |piece: &[f32]| w.hear(&to16k(piece, rate), "ml").map_err(|e| filmcraft_tts::TtsError::Model(e.to_string()));
    Ok(heard::hear_in_pieces(x, rate, heard::HEARD_SECONDS, &mut h)?)
}

fn asr(args: &[String]) -> Res<()> {
    let (Some(dir), clips) = (args.first(), args.get(1..).unwrap_or(&[])) else {
        return Err("usage: asr <model dir> <clip.wav>…".into());
    };
    let t0 = std::time::Instant::now();
    let w = load_asr(Path::new(dir))?;
    eprintln!("loaded {dir} in {:.1} s", t0.elapsed().as_secs_f64());
    let (mut chars, mut edits, mut starts, mut secs, mut audio_s) = (0.0, 0.0, Vec::new(), 0.0, 0.0);
    for c in clips {
        let p = PathBuf::from(c);
        let said = std::fs::read_to_string(p.with_extension("txt"))?;
        let (x, rate) = filmcraft_tts::wav::read(&p)?;
        let t = std::time::Instant::now();
        let got = hear(&w, &x, rate)?;
        secs += t.elapsed().as_secs_f64();
        audio_s += x.len() as f64 / f64::from(rate);
        let s = heard::score(&said, &got);
        let n = heard::comparable(&said).len() as f64;
        chars += n;
        edits += s.cer * n;
        starts.extend(s.starts.iter().copied());
        println!("{}\tCER {:.3}\tstart {:.3}\t{}", p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), s.cer, s.start_cer, got);
    }
    let mean_start = starts.iter().sum::<f64>() / starts.len().max(1) as f64;
    println!("TOTAL\tCER {:.4}\tmean start-CER {:.4}\t{} clips, {audio_s:.0} s of audio in {secs:.0} s", edits / chars.max(1.0), mean_start, clips.len());
    Ok(())
}

/// A reading by name (see the module docs).
fn reading(name: &str) -> Res<Reading> {
    let mut r = Reading::before_fix();
    for part in name.split('+') {
        let num = |p: &str| part.strip_prefix(p).and_then(|v| v.parse::<f64>().ok());
        match part {
            "base" => {}
            "new" => r = Reading::default(),
            "raw" => r.english_transcript = false,
            "sent" => r.per_sentence = true,
            "packed" => r.per_sentence = false,
            "lead" => r.lead_in = Some("ശരി.".into()),
            _ if part.starts_with("tail") => r.tail_ms = num("tail").ok_or("tail<ms>")? as u32,
            _ if part.starts_with("room") => {
                r.tail_ms = num("room").ok_or("room<ms>")? as u32;
                r.room_tone = true;
            }
            _ if part.starts_with("nfe") => r.steps = num("nfe").ok_or("nfe<n>")? as usize,
            _ if part.starts_with("cfg") => r.cfg = num("cfg").ok_or("cfg<x>")? as f32,
            _ if part.starts_with("sway") => r.sway = num("sway").ok_or("sway<x>")? as f32,
            other => return Err(format!("unknown reading part `{other}`").into()),
        }
    }
    Ok(r)
}

fn reference(args: &[String]) -> Res<()> {
    let (Some(models), Some(voice)) = (args.first(), args.get(1)) else { return Err("usage: ref <models dir> <voice>".into()) };
    for (name, r) in [("english", Reading::before_fix()), ("raw", reading("raw")?)] {
        let v = filmcraft_tts::clone::malayalam_voice(voice, Path::new(models), r)?;
        println!(
            "{name}: {:?} ({:.2} s, {} characters, {} at most per utterance)",
            v.reference_text(),
            v.reference_seconds(),
            v.reference_text().chars().count(),
            v.max_chars()
        );
    }
    Ok(())
}

/// One line read the way the voice reads it (utterances, 250 ms between them), take `take`;
/// the text said (after the front end) and the seconds the synthesis took.
fn read_line(v: &MalayalamVoice, line: &str, take: u32) -> Res<(Vec<f32>, String, f64)> {
    let t = std::time::Instant::now();
    let mut out: Vec<f32> = Vec::new();
    let mut said = Vec::new();
    for (i, p) in v.pieces(line)?.into_iter().enumerate() {
        match p {
            Piece::Pause(ms) => out.resize(out.len() + (ms as usize) * 24, 0.0),
            Piece::Say(u) => {
                if !said.is_empty() {
                    out.resize(out.len() + 6000, 0.0);
                }
                out.extend(v.read(&u, 1.0, MalayalamVoice::seed(take, i, 0))?);
                said.push(u);
            }
        }
    }
    Ok((out, said.join(" "), t.elapsed().as_secs_f64()))
}

fn generate(args: &[String]) -> Res<()> {
    let [models, voice, asr_dir, lines, out, readings, rest @ ..] = args else {
        return Err("usage: gen <models dir> <voice> <recogniser dir> <lines.txt> <out dir> <reading>[,<reading>…] [takes [first take]]".into());
    };
    let takes: u32 = rest.first().and_then(|t| t.parse().ok()).unwrap_or(3);
    // the first take (fresh seeds for a confirmation run)
    let first: u32 = rest.get(1).and_then(|t| t.parse().ok()).unwrap_or(0);
    let lines: Vec<String> =
        std::fs::read_to_string(lines)?.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string).collect();
    let w = load_asr(Path::new(asr_dir))?;
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(out.join("results.jsonl"))?;
    for name in readings.split(',') {
        let r = reading(name)?;
        let v = filmcraft_tts::clone::malayalam_voice(voice, Path::new(models), r)?;
        let dir = out.join(name);
        std::fs::create_dir_all(&dir)?;
        let (mut cer, mut n, mut starts, mut bad_starts, mut clean, mut synth, mut audio) = (0.0, 0usize, Vec::new(), 0usize, 0usize, 0.0, 0.0);
        for (li, line) in lines.iter().enumerate() {
            for take in first..first + takes {
                let (x, said, secs) = read_line(&v, line, take)?;
                let got = hear(&w, &x, 24_000)?;
                let s = heard::score(&said, &got);
                let sig = heard::signal(&x, 24_000);
                filmcraft_tts::wav::write(&dir.join(format!("line{li:02}-take{take}.wav")), &x, 24_000)?;
                let row = serde_json::json!({"reading": name, "line": li, "take": take, "said": said, "heard": got, "cer": s.cer, "startCer": s.start_cer, "starts": s.starts, "signal": sig, "seconds": x.len() as f64 / 24_000.0, "synthSeconds": secs});
                writeln!(log, "{row}")?;
                eprintln!("{name} line {li} take {take}: CER {:.3} start {:.3} {:?} {:?}", s.cer, s.start_cer, s.starts, sig);
                cer += s.cer;
                n += 1;
                bad_starts += s.starts.iter().filter(|v| **v > 0.34).count();
                starts.extend(s.starts.iter().copied());
                clean += usize::from(s.cer <= 0.15 && s.start_cer <= 0.34);
                synth += secs;
                audio += x.len() as f64 / 24_000.0;
            }
        }
        let summary = serde_json::json!({"reading": name, "summary": true, "readings": n, "meanCer": cer / n.max(1) as f64, "meanStartCer": starts.iter().sum::<f64>() / starts.len().max(1) as f64, "garbledStarts": bad_starts, "starts": starts.len(), "cleanLines": clean, "synthSeconds": synth, "audioSeconds": audio});
        writeln!(log, "{summary}")?;
        println!("{summary}");
    }
    Ok(())
}

/// Hear the readings of `gen` again with another recogniser: `rescore <recogniser dir> <out dir>
/// <reading>[,<reading>…]` (the WAVs and `results.jsonl` of a `gen` run).
fn rescore(args: &[String]) -> Res<()> {
    let [asr_dir, out, readings, ..] = args else { return Err("usage: rescore <recogniser dir> <out dir> <reading>[,<reading>…]".into()) };
    let w = load_asr(Path::new(asr_dir))?;
    let out = PathBuf::from(out);
    let wanted: Vec<&str> = readings.split(',').collect();
    let (mut cer, mut n, mut starts, mut bad) = (0.0, 0usize, Vec::new(), 0usize);
    for l in std::fs::read_to_string(out.join("results.jsonl"))?.lines() {
        let r: serde_json::Value = serde_json::from_str(l)?;
        let name = r["reading"].as_str().unwrap_or("");
        if r["summary"] == true || !wanted.contains(&name) {
            continue;
        }
        let (li, take) = (r["line"].as_u64().unwrap_or(0), r["take"].as_u64().unwrap_or(0));
        let (x, rate) = filmcraft_tts::wav::read(&out.join(name).join(format!("line{li:02}-take{take}.wav")))?;
        let said = r["said"].as_str().unwrap_or("");
        let got = hear(&w, &x, rate)?;
        let s = heard::score(said, &got);
        println!("{name}\t{li}\t{take}\tCER {:.3}\tstart {:.3}\t{got}", s.cer, s.start_cer);
        cer += s.cer;
        n += 1;
        bad += s.starts.iter().filter(|v| **v > 0.34).count();
        starts.extend(s.starts);
    }
    println!(
        "TOTAL\t{asr_dir}\tmean CER {:.4}\tmean start-CER {:.4}\tstarts over 0.34: {bad} of {}\t{n} readings",
        cer / n.max(1) as f64,
        starts.iter().sum::<f64>() / starts.len().max(1) as f64,
        starts.len()
    );
    Ok(())
}

/// Score the stored hearings of `gen` runs again with the current scorer, and count what
/// policies would read again: `score <results.jsonl>…`.
fn score_again(args: &[String]) -> Res<()> {
    let mut by: std::collections::BTreeMap<String, Vec<heard::Score>> = Default::default();
    for f in args {
        for l in std::fs::read_to_string(f)?.lines() {
            let r: serde_json::Value = serde_json::from_str(l)?;
            if r["summary"] == true {
                continue;
            }
            let s = heard::score(r["said"].as_str().unwrap_or(""), r["heard"].as_str().unwrap_or(""));
            by.entry(format!("{f}:{}", r["reading"].as_str().unwrap_or(""))).or_default().push(s);
        }
    }
    for (name, scores) in &by {
        let first: Vec<f64> = scores.iter().filter_map(|s| s.starts.first().copied()).collect();
        let later: Vec<f64> = scores.iter().flat_map(|s| s.starts.iter().skip(1).copied()).collect();
        let over = |v: &[f64], t: f64| v.iter().filter(|x| **x > t).count();
        let cers: Vec<f64> = scores.iter().map(|s| s.cer).collect();
        println!(
            "{name}: {} readings; mean CER {:.3}; first starts mean {:.3} (>0.2 {} >0.3 {} >0.4 {}); later starts mean {:.3} (>0.2 {} >0.3 {} >0.4 {}); CER >0.2 {} >0.25 {} >0.3 {}",
            scores.len(),
            cers.iter().sum::<f64>() / cers.len().max(1) as f64,
            first.iter().sum::<f64>() / first.len().max(1) as f64,
            over(&first, 0.2),
            over(&first, 0.3),
            over(&first, 0.4),
            later.iter().sum::<f64>() / later.len().max(1) as f64,
            over(&later, 0.2),
            over(&later, 0.3),
            over(&later, 0.4),
            over(&cers, 0.2),
            over(&cers, 0.25),
            over(&cers, 0.3)
        );
    }
    Ok(())
}

/// Clips as mono 16 kHz little-endian f32 (`<out dir>/<name>.f32`, with `<name>.txt` copied),
/// for the speech crate's oracle test.
fn to_f32(args: &[String]) -> Res<()> {
    let (Some(out), clips) = (args.first(), args.get(1..).unwrap_or(&[])) else { return Err("usage: f32 <out dir> <clip.wav>…".into()) };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    for c in clips {
        let p = PathBuf::from(c);
        let (x, rate) = filmcraft_tts::wav::read(&p)?;
        let name = p.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let bytes: Vec<u8> = to16k(&x, rate).iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.f32")), bytes)?;
        if let Ok(t) = std::fs::read_to_string(p.with_extension("txt")) {
            std::fs::write(out.join(format!("{name}.txt")), t)?;
        }
    }
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("asr") => asr(rest),
        Some("ref") => reference(rest),
        Some("gen") => generate(rest),
        Some("f32") => to_f32(rest),
        Some("rescore") => rescore(rest),
        Some("score") => score_again(rest),
        _ => Err("usage: ml_lab asr|ref|gen|f32 … (see the source)".into()),
    }
}
