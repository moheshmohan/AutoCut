//! The diarization lab: a mix of several voices with known turns, labelled by the neural
//! speaker labels (Silero + ECAPA) and the MFCC fallback, scored by diarization error rate.
//!
//! ```text
//! cargo run -p filmcraft-speech --features whisper --release --example diarize_lab -- \
//!     <models dir> <out.json> <name>=<a.wav>,<b.wav>,… <name>=…
//! ```
//!
//! Each speaker's clips (16-bit or float WAV, any rate, resampled to 16 kHz) are cut into
//! utterances at their pauses (Silero); the utterances are dealt into one mix in a fixed
//! pseudo-random order of speakers with 0.25–0.9 s of silence between turns. The truth is where
//! each utterance went. Written: the mix (`<out>.wav`) and a JSON report with the error rate (no
//! collar and a 0.25 s collar), the speakers found, and a sweep of the clustering threshold.

use std::path::Path;

use filmcraft_speech::speakers::{self, Turn};
use filmcraft_speech::{SAMPLE_RATE, silero};

fn read_wav(path: &str) -> Result<(Vec<f32>, u32), String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut i = 12;
    let (mut fmt, mut ch, mut bits, mut rate) = (1u16, 1u16, 16u16, 16_000u32);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = b.get(i + 8..(i + 8 + n).min(b.len())).unwrap_or(&[]);
        if id == b"fmt " && body.len() >= 16 {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]).max(1);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        }
        if id == b"data" {
            let x: Vec<f32> = match (fmt, bits) {
                (3, 32) => body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                (_, 16) => body.as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect(),
                _ => return Err(format!("{path}: 16-bit or float WAV only")),
            };
            let ch = ch as usize;
            return Ok((x.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), rate));
        }
        i += 8 + n + (n & 1);
    }
    Err(format!("{path}: no data"))
}

/// Windowed-sinc resampling to 16 kHz (lab quality).
fn to_16k(x: &[f32], rate: u32) -> Vec<f32> {
    if rate == SAMPLE_RATE || rate == 0 {
        return x.to_vec();
    }
    let ratio = f64::from(SAMPLE_RATE) / f64::from(rate);
    let cutoff = ratio.min(1.0) * 0.95;
    let n = (x.len() as f64 * ratio) as usize;
    let taps = 24isize;
    (0..n)
        .map(|j| {
            let t = j as f64 / ratio;
            let c = t.floor() as isize;
            let mut acc = 0.0;
            for k in c - taps..=c + taps {
                let Some(v) = usize::try_from(k).ok().and_then(|k| x.get(k)) else { continue };
                let d = t - k as f64;
                let sinc = if d.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * d * cutoff).sin() / (std::f64::consts::PI * d) };
                let w = 0.5 + 0.5 * (std::f64::consts::PI * d / (taps as f64 + 1.0)).cos();
                acc += f64::from(*v) * sinc * w;
            }
            acc as f32
        })
        .collect()
}

fn write_wav(path: &Path, x: &[f32]) -> std::io::Result<()> {
    let mut w = Vec::with_capacity(44 + x.len() * 2);
    let data = (x.len() * 2) as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    w.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for v in x {
        w.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, w)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let (Some(models), Some(out)) = (a.get(1), a.get(2)) else {
        return Err("usage: diarize_lab <models dir> <out.json> <name>=<a.wav>,… …".into());
    };
    let models = Path::new(models);
    let vad = filmcraft_speech::vad::load_silero(models).ok_or("silero-vad is not downloaded")?;
    let ecapa = filmcraft_speech::ecapa::load_cached(models).ok_or("ecapa-voxceleb is not downloaded")?;
    // each speaker's utterances
    let mut pools: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for spec in &a[3..] {
        let (name, files) = spec.split_once('=').ok_or("speakers are <name>=<a.wav>,…")?;
        let mut utts = Vec::new();
        for f in files.split(',') {
            let (x, rate) = read_wav(f)?;
            let x = to_16k(&x, rate);
            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let x: Vec<f32> = x.iter().map(|v| v / peak * 0.5).collect();
            let segs = silero::segments(&vad.probabilities(&x), x.len(), &silero::SegmentParams { min_silence: 0.3, ..Default::default() });
            for s in segs {
                // utterances of 1–8 s
                let mut a0 = s.start;
                while a0 < s.end {
                    let b0 = (a0 + 8 * SAMPLE_RATE as usize).min(s.end);
                    if b0 - a0 >= SAMPLE_RATE as usize {
                        utts.push(x[a0..b0].to_vec());
                    }
                    a0 = b0;
                }
            }
        }
        eprintln!("{name}: {} utterances", utts.len());
        pools.push((name.to_string(), utts));
    }
    if pools.is_empty() {
        return Err("give at least one speaker".into());
    }
    // how alike windows of one voice are, and of different voices (what the threshold separates)
    let mut embs: Vec<(usize, Vec<f32>)> = Vec::new();
    for (k, (_, utts)) in pools.iter().enumerate() {
        let wins: Vec<&[f32]> = utts.iter().flat_map(|u| u.as_chunks::<24_000>().0.iter().map(|w| w.as_slice())).collect();
        for e in ecapa.embed_many(&wins).map_err(|e| e.to_string())? {
            embs.push((k, e));
        }
    }
    let (mut same, mut diff) = (Vec::new(), Vec::new());
    for i in 0..embs.len() {
        for j in i + 1..embs.len() {
            let c = filmcraft_speech::ecapa::cosine(&embs[i].1, &embs[j].1);
            if embs[i].0 == embs[j].0 { same.push(c) } else { diff.push(c) }
        }
    }
    let stats = |v: &mut Vec<f32>| -> serde_json::Value {
        v.sort_by(f32::total_cmp);
        let q = |p: f64| v.get(((v.len().max(1) - 1) as f64 * p) as usize).copied().unwrap_or(f32::NAN);
        serde_json::json!({"pairs": v.len(), "p05": q(0.05), "median": q(0.5), "p95": q(0.95), "min": q(0.0), "max": q(1.0)})
    };
    let (same, diff) = (stats(&mut same), stats(&mut diff));
    eprintln!("1.5 s windows, cosine: same voice {same}, different voices {diff}");
    // deal the utterances into a mix
    let mut rng = 12345u32;
    let mut next = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        rng >> 8
    };
    let mut mix: Vec<f32> = vec![0.0; SAMPLE_RATE as usize / 2];
    let mut truth: Vec<Turn> = Vec::new();
    let mut used = vec![0usize; pools.len()];
    let mut last = usize::MAX;
    loop {
        let open: Vec<usize> = (0..pools.len()).filter(|&k| used[k] < pools[k].1.len()).collect();
        if open.is_empty() {
            break;
        }
        // mostly a change of speaker, sometimes the same one goes on
        let mut k = open[(next() as usize) % open.len()];
        if k == last && open.len() > 1 && next() % 4 != 0 {
            k = open[(open.iter().position(|&o| o == k).unwrap_or(0) + 1) % open.len()];
        }
        let u = &pools[k].1[used[k]];
        used[k] += 1;
        let start = mix.len();
        mix.extend_from_slice(u);
        truth.push(Turn { start, end: mix.len(), speaker: k });
        let gap = (0.25 + (next() % 650) as f64 / 1000.0) * f64::from(SAMPLE_RATE);
        mix.resize(mix.len() + gap as usize, 0.0);
        last = k;
    }
    let out = Path::new(out);
    write_wav(&out.with_extension("wav"), &mix).map_err(|e| e.to_string())?;
    // the truth beside it (`tests/speech_models.rs` labels the mix again when both are in
    // target/fixtures as diarize-mix.wav and diarize-mix.truth.json)
    let truth_json = serde_json::to_vec(&truth).map_err(|e| e.to_string())?;
    std::fs::write(out.with_extension("truth.json"), truth_json).map_err(|e| e.to_string())?;
    let secs = mix.len() as f64 / f64::from(SAMPLE_RATE);
    eprintln!("mix: {secs:.1} s, {} turns", truth.len());
    // label it
    let t0 = std::time::Instant::now();
    let (segs, _) = filmcraft_speech::vad::speech_segments(&mix, Some(&vad), &silero::SegmentParams::default());
    let vad_secs = t0.elapsed().as_secs_f64();
    let collar = SAMPLE_RATE as usize / 4;
    let mut sweep = Vec::new();
    let mut best = None;
    let d = speakers::Params::default();
    let windows: Vec<f64> = std::env::var("LAB_WINDOWS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|| vec![d.window]);
    for (window, th) in windows.iter().flat_map(|w| [0.25f32, 0.3, d.threshold, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6].map(|t| (*w, t))) {
        let t1 = std::time::Instant::now();
        let p = speakers::Params { threshold: th, window, hop: window / 2.0, ..Default::default() };
        let hyp = speakers::turns(&mix, &segs, ecapa.as_ref(), &p).map_err(|e| e.to_string())?;
        let found = hyp.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
        let der = speakers::error_rate(&truth, &hyp, mix.len(), 0);
        let der_c = speakers::error_rate(&truth, &hyp, mix.len(), collar);
        eprintln!("window {window} s, threshold {th}: {found} speakers, DER {der:.3} ({der_c:.3} with a 0.25 s collar) in {:.1} s", t1.elapsed().as_secs_f64());
        if (th - d.threshold).abs() < 1e-6 && (window - d.window).abs() < 1e-9 {
            best = Some(serde_json::json!({"window": window, "threshold": th, "speakers": found, "der": der, "derCollar": der_c, "turns": hyp.len()}));
        }
        sweep.push(serde_json::json!({"window": window, "threshold": th, "speakers": found, "der": der, "derCollar": der_c}));
    }
    // the MFCC fallback on 0.4 s pseudo-words of the speech
    let mut words = Vec::new();
    for s in &segs {
        let mut a0 = s.start;
        while a0 + 1600 <= s.end {
            let b0 = (a0 + 6400).min(s.end);
            words.push(filmcraft_project::Word::new("w", filmcraft_speech::sample_tick(a0 as i64), filmcraft_speech::sample_tick(b0 as i64)));
            a0 = b0;
        }
    }
    let mut tr = filmcraft_project::Transcript { words, ..Default::default() };
    let n_mfcc = filmcraft_speech::diarize::diarize(&mix, &mut tr, &filmcraft_speech::diarize::Params::default());
    let mfcc_turns: Vec<Turn> = tr
        .words
        .iter()
        .map(|w| Turn {
            start: (w.start.0 / filmcraft_speech::TICKS_PER_SAMPLE) as usize,
            end: (w.end.0 / filmcraft_speech::TICKS_PER_SAMPLE) as usize,
            speaker: w.speaker.unwrap_or(0) as usize,
        })
        .collect();
    let mfcc = speakers::error_rate(&truth, &mfcc_turns, mix.len(), 0);
    let mfcc_c = speakers::error_rate(&truth, &mfcc_turns, mix.len(), collar);
    eprintln!("MFCC fallback: {n_mfcc} speakers, DER {mfcc:.3} ({mfcc_c:.3} with a collar)");
    let report = serde_json::json!({
        "seconds": secs,
        "speakers": pools.iter().map(|p| &p.0).collect::<Vec<_>>(),
        "turns": truth.len(),
        "vadSeconds": vad_secs,
        "cosineSameVoice": same,
        "cosineDifferentVoices": diff,
        "ecapa": best,
        "sweep": sweep,
        "mfcc": {"speakers": n_mfcc, "der": mfcc, "derCollar": mfcc_c},
    });
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(())
}
