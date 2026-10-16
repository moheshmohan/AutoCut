//! Evaluate cloned voices on a real recording: make natural clones (VoxCPM1.5, VoxCPM2) from a
//! stretch of a video through the engine's commands, speak test sentences with them and with
//! any other cloned voices given (e.g. an existing timbre-only clone), and report per voice:
//!
//! - **similarity** to held-out real speech of the speaker (speech outside the clone's voice
//!   sample): the cosine of OpenVoice V2 tone colours (1 = the same timbre);
//! - **WER**: Whisper's transcript against the sentence (the speech model in the data dir:
//!   `--asr whisper-small` is recommended);
//! - **RTF**: synthesis time over audio length (after a warm-up utterance).
//!
//! WAVs of every utterance and of the held-out speech are written to the output directory, and
//! `results.json` with every number. Runs in its own data directory (never the user's):
//!
//! ```sh
//! FILMCRAFT_DATA_DIR=/tmp/eval cargo run --release -p filmcraft-engine --features neural-voices,whisper \
//!   --example clone_eval -- <video> <out dir> [--from 0 --to 120] [--asr whisper-small] [--voices clone:a,clone:b]
//!   [--models voxcpm-1.5,voxcpm-2] [--steps N]
//! ```
//!
//! `<data dir>/models` must hold the packages used (VoxCPM releases, `openvoice-v2`, the
//! Whisper model, `kokoro-82m` for timbre-only clones), e.g. as symbolic links.

use filmcraft_engine::Session;
use serde_json::{Value, json};

const SENTENCES: [&str; 3] = [
    "In October 2022 I met Spot, the yellow robot dog from Boston Dynamics, walking through the crowd at GITEX in Dubai.",
    "Number seven: a four legged robot dog that walked right up to me at the expo. It could climb stairs, and it even danced.",
    "Honestly, the most surprising part was how quickly everyone stopped staring at the robots and started talking to them.",
];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    dot / (n(a) * n(b)).max(1e-12)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(video), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("usage: clone_eval <video> <out dir> [--from S --to S] [--asr whisper-…] [--voices clone:a,…] [--models voxcpm-1.5,…] [--steps N]");
        std::process::exit(2);
    };
    if std::env::var_os("FILMCRAFT_DATA_DIR").is_none() {
        eprintln!("set FILMCRAFT_DATA_DIR to a scratch data directory: the evaluation saves voices there");
        std::process::exit(2);
    }
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    let from: f64 = arg(&args, "--from").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let to: f64 = arg(&args, "--to").and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let asr = arg(&args, "--asr").unwrap_or_else(|| "whisper-base".into());
    let steps: Option<u64> = arg(&args, "--steps").and_then(|v| v.parse().ok());
    let extra: Vec<String> = arg(&args, "--voices").map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default();
    let only: Option<Vec<String>> = arg(&args, "--models").map(|v| v.split(',').map(str::to_string).collect());
    let models = filmcraft_engine::transcript::models_dir().ok_or("no data directory")?;

    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [video]})).map_err(|e| e.to_string())?;
    let item = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or("the video was not imported")?;
    eprintln!("imported item {item}");

    // the clones under test
    let mut voices: Vec<(String, String)> = Vec::new();
    for (name, model) in [("eval-voxcpm15", "voxcpm-1.5"), ("eval-voxcpm2", "voxcpm-2")] {
        if only.as_ref().is_some_and(|o| !o.iter().any(|m| m == model)) {
            continue;
        }
        let installed = filmcraft_tts::catalog::voxcpm(model).is_some_and(|m| filmcraft_tts::catalog::voxcpm_installed(&models, m));
        if !installed {
            eprintln!("{model} is not installed: skipped");
            continue;
        }
        let mut p = json!({"name": name, "consent": true, "segments": [{"item": item, "start": from, "end": to}], "engine": "voxcpm", "model": model, "speechModel": asr, "wait": true});
        if let Some(n) = steps {
            p["steps"] = json!(n);
        }
        let t0 = std::time::Instant::now();
        let r = s.execute("tts.cloneVoice", p).map_err(|e| e.to_string())?;
        let job = r["job"].as_u64().ok_or("no job")?;
        let j = s.jobs.iter().find(|j| j.id == job).ok_or("the job disappeared")?;
        let status = j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        if let Some(Err(e)) = &*j.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
            return Err(format!("cloning {name} failed: {e}").into());
        }
        eprintln!("cloned {name} with {model} in {:.1?}: {status}", t0.elapsed());
        voices.push((format!("clone:{name}"), model.to_string()));
    }
    for v in extra {
        voices.push((v.clone(), "openvoice-v2".into()));
    }
    let profiles = filmcraft_tts::clone::list(&filmcraft_tts::clone::voices_dir(&models));

    // held-out real speech: the stretch minus every natural clone's voice sample
    let rate = 16_000u32;
    let src = s.source(filmcraft_project::ItemId(item)).ok_or("no source")?;
    let n = ((to - from) * f64::from(rate)) as usize;
    let buf = src.audio((from * f64::from(rate)) as i64, n, rate).map_err(|e| e.to_string())?;
    let mut real = filmcraft_speech::downmix(&buf.channels);
    for p in &profiles {
        if let Some(nat) = &p.natural {
            let (a, b) = (((nat.from) * f64::from(rate)) as usize, ((nat.to) * f64::from(rate)) as usize);
            for v in real.get_mut(a.min(n)..b.min(n)).into_iter().flatten() {
                *v = 0.0;
            }
            eprintln!("{}: voice sample {:.1}–{:.1} s, transcript {:?}", p.id, nat.from + from, nat.to + from, nat.transcript);
        }
    }
    let held_out = filmcraft_tts::clone::speech_only(&real, rate);
    eprintln!("held-out speech: {:.1} s", held_out.len() as f64 / f64::from(rate));
    filmcraft_tts::wav::write(&out.join("held_out_real.wav"), &held_out, rate)?;
    let conv = filmcraft_tts::openvoice::Converter::load(&filmcraft_tts::catalog::clone_package_dir(&models))?;
    let colour = |x: &[f32], r: u32| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let speech = filmcraft_tts::clone::speech_only(x, r);
        Ok(conv.tone_colour(&filmcraft_audio_dsp::resample::resample_sinc(&speech, r, filmcraft_tts::openvoice::RATE))?)
    };
    let real_colour = colour(&held_out, rate)?;
    let whisper = filmcraft_speech::load(&models, &asr)?;
    let opts = filmcraft_speech::Options { language: Some("en".into()), diarize: false, ..Default::default() };

    let mut results = Vec::new();
    for (id, model) in &voices {
        let voice = filmcraft_tts::voice_in(id, Some(&models))?;
        // warm-up (model load, kernels)
        let _ = voice.synthesize("Hello.", &filmcraft_tts::Params::default())?;
        let (mut sims, mut wers, mut secs, mut time) = (Vec::new(), Vec::new(), 0.0, 0.0);
        for (k, sentence) in SENTENCES.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let a = voice.synthesize(sentence, &filmcraft_tts::Params::default())?;
            let dt = t0.elapsed().as_secs_f64();
            let file = out.join(format!("{}_{}.wav", id.trim_start_matches("clone:"), k + 1));
            filmcraft_tts::wav::write(&file, &a.samples, a.sample_rate)?;
            let sim = cosine(&colour(&a.samples, a.sample_rate)?, &real_colour);
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(&a.samples, a.sample_rate, rate);
            let hyp = whisper.transcribe(&x16, &opts, &mut |_, _| true)?.text();
            let wer = filmcraft_speech::word_error_rate(sentence, &hyp);
            eprintln!(
                "{id} #{}: {:.1} s in {dt:.1} s (RTF {:.2}), similarity {sim:.3}, WER {:.1} %: {hyp}",
                k + 1,
                a.seconds(),
                dt / a.seconds().max(1e-6),
                wer * 100.0
            );
            sims.push(sim);
            wers.push(wer);
            secs += a.seconds();
            time += dt;
            results.push(json!({"voice": id, "model": model, "sentence": k + 1, "file": file, "seconds": a.seconds(), "synthSeconds": dt, "similarity": sim, "wer": wer, "heard": hyp}));
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!("| {id} ({model}) | {:.3} | {:.1} % | {:.2} |", mean(&sims), mean(&wers) * 100.0, time / secs.max(1e-6));
    }
    // the ceiling: the speaker's own voice sample against the held-out speech
    for p in &profiles {
        if let Some(path) = filmcraft_tts::clone::sample_path(&filmcraft_tts::clone::voices_dir(&models), p) {
            let (x, r) = filmcraft_tts::wav::read(&path)?;
            println!("| real speech ({} voice sample) | {:.3} | | |", p.id, cosine(&colour(&x, r)?, &real_colour));
        }
    }
    std::fs::write(out.join("results.json"), serde_json::to_vec_pretty(&json!({"results": results, "asr": asr}))?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
