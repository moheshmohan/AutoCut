//! Re-check the strict check's verdicts on readings already in a narration cache: a second
//! Malayalam recogniser's opinion (IndicWhisper) beside the one the cache kept (Whisper medium
//! ml), and the voice gate (ECAPA similarity of every utterance to the voice's own sample).
//!
//! ```text
//! cargo run -p filmcraft-engine --features whisper --example narration_recheck -- \
//!     <copy of narration-cache> <models dir> <voice sample.wav> <out.json>
//! ```
//!
//! Work on a copy of the cache: reading an entry refreshes its last use. Every entry with a
//! strict-check note is a checked Malayalam reading; its audio is cut back into its utterances
//! at the silences the voice put between them (runs of exact zeros), and each utterance is heard
//! by the second recogniser and scored against its text. Lines are counted as flagged by the
//! first model (the note's verdict), the second, both (what `agreement: both` retakes), either,
//! and the voice gate at a sweep of thresholds.

use std::path::Path;

use filmcraft_tts as tts;
use serde_json::{Value, json};

/// Utterances of a checked reading: cut at runs of exact zeros of at least `min` samples.
fn utterances(x: &[f32], min: usize) -> Vec<&[f32]> {
    let mut out = Vec::new();
    let (mut start, mut run) = (None::<usize>, 0usize);
    for (i, v) in x.iter().enumerate() {
        if *v == 0.0 {
            run += 1;
            if run == min
                && let Some(s) = start.take()
            {
                out.push(&x[s..i + 1 - min]);
            }
        } else {
            if start.is_none() {
                start = Some(i);
            }
            run = 0;
        }
    }
    if let Some(s) = start {
        out.push(&x[s..]);
    }
    out
}

/// `--rescore <recheck.json>`: the second opinions of an earlier run scored again with
/// IndicWhisper's digits spelled the way the narration's script spells numbers (what the strict
/// check does), no model needed.
fn rescore(path: &str) -> Result<(), String> {
    let d: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let policy = tts::heard::Policy::default();
    let (mut second, mut both, mut first) = (0, 0, 0);
    let lines = d["lines"].as_array().cloned().unwrap_or_default();
    for l in &lines {
        let text = l["text"].as_str().unwrap_or("");
        let heard = tts::malayalam::spell_numbers(l["second"]["heard"].as_str().unwrap_or(""), tts::malayalam::Numbers::English);
        let ok2 = policy.passes(&tts::heard::score(text, &heard));
        let ok1 = l["first"]["passed"].as_bool() == Some(true);
        second += usize::from(!ok2);
        first += usize::from(!ok1);
        both += usize::from(!ok1 && !ok2);
        if !ok2 {
            eprintln!("still flagged: {text} | {heard}");
        }
    }
    eprintln!("{} lines: first {first}, second (digits spelled) {second}, both {both}", lines.len());
    Ok(())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("--rescore") {
        return rescore(a.get(2).ok_or("--rescore <recheck.json>")?);
    }
    let (Some(cache), Some(models), Some(sample), Some(out)) = (a.get(1), a.get(2), a.get(3), a.get(4)) else {
        return Err("usage: narration_recheck <cache copy> <models dir> <voice sample.wav> <out.json>".into());
    };
    let models = Path::new(models);
    let second = filmcraft_speech::load_on(models, filmcraft_speech::models::INDICWHISPER_ML, true).map_err(|e| e.to_string())?;
    let ecapa = filmcraft_speech::ecapa::load_cached(models).ok_or("ecapa-voxceleb is not downloaded")?;
    let (ref_x, ref_rate) = tts::wav::read(Path::new(sample)).map_err(|e| e.to_string())?;
    let ref16 = filmcraft_audio_dsp::resample::resample_sinc(&ref_x, ref_rate, filmcraft_speech::SAMPLE_RATE);
    let reference = ecapa.embed(&ref16).map_err(|e| e.to_string())?;
    let disk = tts::cache::DiskCache::new(cache, tts::cache::DEFAULT_CAP_BYTES);
    let policy = tts::heard::Policy::default();
    let mut lines = Vec::new();
    let mut entries = 0;
    let mut whole = 0;
    let mut names: Vec<String> = std::fs::read_dir(cache)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    let t0 = std::time::Instant::now();
    for name in names {
        let key = name.trim_end_matches(".json");
        let Some(note) = disk.note(key) else { continue };
        let Ok(report) = serde_json::from_slice::<Value>(&note) else { continue };
        let Some(utts) = report.as_array() else { continue };
        let Some(audio) = disk.get(key) else { continue };
        entries += 1;
        let gap = (0.2 * f64::from(audio.sample_rate)) as usize;
        let mut pieces = utterances(&audio.samples, gap);
        // the cut must find every utterance; otherwise the reading is heard whole
        let joined;
        let texts: Vec<(String, bool, f64, f64)> = if pieces.len() == utts.len() {
            utts.iter()
                .map(|u| {
                    (
                        u["text"].as_str().unwrap_or("").to_string(),
                        u["passed"].as_bool().unwrap_or(false),
                        u["cer"].as_f64().unwrap_or(1.0),
                        u["startCer"].as_f64().unwrap_or(1.0),
                    )
                })
                .collect()
        } else {
            whole += 1;
            pieces = vec![&audio.samples[..]];
            joined = utts.iter().filter_map(|u| u["text"].as_str()).collect::<Vec<_>>().join(" ");
            let passed = utts.iter().all(|u| u["passed"].as_bool() == Some(true));
            let cer = utts.iter().filter_map(|u| u["cer"].as_f64()).fold(0.0, f64::max);
            vec![(joined.clone(), passed, cer, 0.0)]
        };
        for (piece, (text, first_ok, cer1, start1)) in pieces.iter().zip(texts) {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(piece, audio.sample_rate, filmcraft_speech::SAMPLE_RATE);
            let mut hear = |p: &[f32]| second.hear(p, "ml").map_err(|e| tts::TtsError::Model(e.to_string()));
            let heard = tts::heard::hear_in_pieces(&x16, filmcraft_speech::SAMPLE_RATE, tts::heard::HEARD_SECONDS, &mut hear).map_err(|e| e.to_string())?;
            let s2 = tts::heard::score(&text, &heard);
            let second_ok = policy.passes(&s2);
            let sim = ecapa.embed(&x16).map(|e| filmcraft_speech::ecapa::cosine(&e, &reference)).unwrap_or(f32::NAN);
            lines.push(json!({
                "entry": key, "text": text, "seconds": piece.len() as f64 / f64::from(audio.sample_rate),
                "first": {"passed": first_ok, "cer": cer1, "startCer": start1},
                "second": {"passed": second_ok, "heard": heard, "cer": s2.cer, "startCer": s2.start_cer},
                "similarity": sim,
            }));
        }
        eprintln!("{entries} entries, {} lines, {:.0} s", lines.len(), t0.elapsed().as_secs_f64());
    }
    let n = lines.len();
    let count = |f: &dyn Fn(&Value) -> bool| lines.iter().filter(|l| f(l)).count();
    let first = count(&|l| l["first"]["passed"] == json!(false));
    let second_f = count(&|l| l["second"]["passed"] == json!(false));
    let both = count(&|l| l["first"]["passed"] == json!(false) && l["second"]["passed"] == json!(false));
    let either = count(&|l| l["first"]["passed"] == json!(false) || l["second"]["passed"] == json!(false));
    let mut sims: Vec<f64> = lines.iter().filter_map(|l| l["similarity"].as_f64()).filter(|s| s.is_finite()).collect();
    sims.sort_by(f64::total_cmp);
    let q = |p: f64| sims.get(((sims.len().max(1) - 1) as f64 * p) as usize).copied().unwrap_or(f64::NAN);
    let gate: Vec<Value> =
        [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5].iter().map(|t| json!({"min": t, "flagged": sims.iter().filter(|s| **s < *t).count()})).collect();
    let summary = json!({
        "entries": entries, "heardWhole": whole, "lines": n,
        "flagged": {"first (whisper-medium-ml)": first, "second (indicwhisper-ml)": second_f, "both": both, "either": either},
        "similarity": {"min": q(0.0), "p01": q(0.01), "p05": q(0.05), "median": q(0.5), "max": q(1.0)},
        "voiceGate": gate,
    });
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
    std::fs::write(out, serde_json::to_vec_pretty(&json!({"summary": summary, "lines": lines})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(())
}
