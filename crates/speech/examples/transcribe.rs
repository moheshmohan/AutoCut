//! Transcribe speech files and report the word error rate against `<file>.txt`, on the CPU, the
//! GPU, or both (`--both`: the two transcripts and their word times compared, and the times).
//!
//! Inputs: raw mono 16 kHz f32 (`.f32`) or WAV (16-bit PCM or 32-bit float, any rate and channel
//! count; mixed down and resampled to 16 kHz here). `--concat` transcribes the inputs as one clip.
//!
//! ```sh
//! cargo run --release -p filmcraft-speech --features whisper --example transcribe -- <model dir> [--gpu|--both] [--words] [--diarize] [--concat] speech.wav…
//! ```

use filmcraft_speech::{Options, Transcriber, word_error_rate};

/// Mono 16 kHz samples of a `.f32` or `.wav` file.
fn read_audio(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("read");
    if !p.ends_with(".wav") {
        return b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    }
    let (mut fmt, mut data) = (None, None);
    let mut i = 12;
    while i + 8 <= b.len() {
        let n = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + n).min(b.len())];
        match &b[i..i + 4] {
            b"fmt " => {
                fmt = Some((
                    u16::from_le_bytes([body[0], body[1]]),
                    u16::from_le_bytes([body[2], body[3]]),
                    u32::from_le_bytes([body[4], body[5], body[6], body[7]]),
                    u16::from_le_bytes([body[14], body[15]]),
                ))
            }
            b"data" => data = Some(body),
            _ => {}
        }
        i += 8 + n + (n & 1);
    }
    let ((tag, channels, rate, bits), data) = (fmt.expect("fmt chunk"), data.expect("data chunk"));
    let samples: Vec<f32> = match (tag, bits) {
        (3, 32) => data.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        (1, 16) => data.as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect(),
        other => panic!("{p}: unsupported WAV format {other:?}"),
    };
    let ch = usize::from(channels.max(1));
    let mono: Vec<f32> = samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    resample(&mono, rate, 16_000)
}

/// Windowed-sinc resampling (Hann, 32 taps a side) from `from` Hz to `to` Hz.
fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let cutoff = ratio.min(1.0) * 0.95;
    let n = (x.len() as f64 * ratio).floor() as usize;
    let half = 32i64;
    (0..n)
        .map(|j| {
            let centre = j as f64 / ratio;
            let k0 = centre.floor() as i64;
            let (mut acc, mut norm) = (0.0f64, 0.0f64);
            for k in k0 - half + 1..=k0 + half {
                let Some(&v) = usize::try_from(k).ok().and_then(|k| x.get(k)) else { continue };
                let t = k as f64 - centre;
                let sinc = if t.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * cutoff * t) };
                let w = 0.5 + 0.5 * (std::f64::consts::PI * t / half as f64).cos();
                acc += f64::from(v) * sinc * w;
                norm += sinc * w;
            }
            (acc / norm.max(1e-9)) as f32
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let id = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let mut show_words = false;
    let mut diarize = false;
    let mut concat = false;
    let mut devices = vec![false];
    let mut language = None;
    let mut files = Vec::new();
    for a in args {
        match a.as_str() {
            "--words" => show_words = true,
            "--diarize" => diarize = true,
            "--concat" => concat = true,
            "--gpu" => devices = vec![true],
            "--both" => devices = vec![false, true],
            l if l.starts_with("--lang=") => language = Some(l[7..].to_string()),
            p => files.push(p.to_string()),
        }
    }
    let mut clips: Vec<(String, Vec<f32>)> = files.iter().map(|p| (p.clone(), read_audio(p))).collect();
    if concat {
        let all: Vec<f32> = clips.iter().flat_map(|(_, a)| a.iter().copied().chain(std::iter::repeat_n(0.0, 8000))).collect();
        clips = vec![(format!("{} files", files.len()), all)];
    }
    let mut results: Vec<Vec<filmcraft_project::Transcript>> = Vec::new();
    for gpu in devices.iter().copied() {
        let name = if gpu { "gpu" } else { "cpu" };
        let t0 = std::time::Instant::now();
        let w = filmcraft_speech::whisper::Whisper::load_on(&dir, &id, gpu).expect("load");
        eprintln!("[{name}] loaded {id} in {:.2}s", t0.elapsed().as_secs_f64());
        // the first window compiles the GPU's kernels: not counted
        if gpu && let Some((_, a)) = clips.first() {
            let warm = &a[..a.len().min(16_000 * 5)];
            let _ = w.transcribe(warm, &Options { language: language.clone(), diarize: false, ..Default::default() }, &mut |_, _| true);
        }
        let (mut errs, mut words, mut audio_s, mut cpu_s) = (0.0, 0usize, 0.0, 0.0);
        let mut out = Vec::new();
        for (p, audio) in &clips {
            let t1 = std::time::Instant::now();
            let opts = Options { language: language.clone(), diarize, ..Default::default() };
            let t = w.transcribe(audio, &opts, &mut |_, _| true).expect("transcribe");
            let secs = t1.elapsed().as_secs_f64();
            audio_s += audio.len() as f64 / 16_000.0;
            cpu_s += secs;
            let hyp = t.text();
            print!("[{name}] {p} [{} {:.2}s]: {hyp}", t.language, secs);
            if let Ok(r) = std::fs::read_to_string(std::path::Path::new(p).with_extension("txt")) {
                let n = r.split_whitespace().count();
                let e = word_error_rate(&r, &hyp);
                errs += e * n as f64;
                words += n;
                print!("  (WER {:.1}%)", e * 100.0);
            }
            println!();
            if show_words {
                for x in &t.words {
                    println!(
                        "  {:7.2} {:7.2} {:>3} {}",
                        x.start.0 as f64 / 254_016_000_000.0,
                        x.end.0 as f64 / 254_016_000_000.0,
                        x.speaker.map(|s| s.to_string()).unwrap_or_default(),
                        x.text
                    );
                }
            }
            out.push(t);
        }
        println!("[{name}] {audio_s:.1}s of audio in {cpu_s:.2}s ({:.1}× realtime)", audio_s / cpu_s.max(1e-9));
        if words > 0 {
            println!("[{name}] total WER {:.2}% over {words} words", errs / words as f64 * 100.0);
        }
        results.push(out);
    }
    if let [cpu, gpu] = results.as_slice() {
        let mut same = 0;
        let mut worst = 0f64;
        for (a, b) in cpu.iter().zip(gpu) {
            if a.text() == b.text() {
                same += 1;
                for (x, y) in a.words.iter().zip(&b.words) {
                    worst = worst.max(((x.start.0 - y.start.0).abs().max((x.end.0 - y.end.0).abs())) as f64 / 254_016_000_000.0);
                }
            } else {
                println!("DIFFERENT TEXT:\n  cpu: {}\n  gpu: {}", a.text(), b.text());
            }
        }
        println!("cpu vs gpu: {same} of {} transcripts the same text; word times differ by at most {:.3}s", cpu.len(), worst);
    }
}
