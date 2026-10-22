//! The Silero VAD and ECAPA-TDNN ports against their reference implementations
//! (`tools/speech_oracle.py`: the authors' recipe for the Silero weights in PyTorch, SpeechBrain's
//! own classes).
//! They run when the models and the oracle's output are given, and pass silently otherwise:
//!
//! ```text
//! FILMCRAFT_MODELS_DIR=<models> SILERO_ORACLE=silero.json ECAPA_ORACLE=ecapa.json \
//!     cargo test -p filmcraft-speech --features whisper --test speech_models
//! ```

use std::path::PathBuf;

fn models() -> Option<PathBuf> {
    std::env::var_os("FILMCRAFT_MODELS_DIR").map(PathBuf::from)
}

fn oracle(var: &str) -> Option<serde_json::Value> {
    let p = std::env::var_os(var)?;
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

/// Mono 16 kHz samples of a 16-bit or float WAV.
fn wav(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let mut i = 12;
    let (mut fmt, mut ch, mut bits) = (1u16, 1u16, 16u16);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + n).min(b.len())];
        if id == b"fmt " {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        }
        if id == b"data" {
            let x: Vec<f32> = match (fmt, bits) {
                (3, 32) => body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                _ => body.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c) as f32 / 32768.0).collect(),
            };
            let ch = ch.max(1) as usize;
            return x.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
        }
        i += 8 + n + (n & 1);
    }
    panic!("no data in {path}");
}

#[test]
fn silero_matches_the_reference() {
    let (Some(dir), Some(o)) = (models(), oracle("SILERO_ORACLE")) else { return };
    let m = filmcraft_speech::vad::load_silero(&dir).expect("silero-vad is not in FILMCRAFT_MODELS_DIR");
    let audio = wav(o["clip"].as_str().unwrap());
    let want: Vec<f64> = o["probs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let got = m.probabilities(&audio);
    assert_eq!(got.len(), want.len());
    let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    let agree = got.iter().zip(&want).filter(|(g, w)| (**g >= 0.5) == (**w >= 0.5)).count();
    eprintln!("silero: {} chunks, largest difference {worst:.2e}, {agree} agree on speech", got.len());
    assert!(worst < 1e-3, "largest difference {worst}");
    assert_eq!(agree, got.len());
}

#[cfg(feature = "whisper")]
#[test]
fn ecapa_matches_speechbrain() {
    let (Some(dir), Some(o)) = (models(), oracle("ECAPA_ORACLE")) else { return };
    let path = filmcraft_speech::models::support_file(&dir, filmcraft_speech::models::ECAPA_MODEL).expect("ecapa-voxceleb is not in FILMCRAFT_MODELS_DIR");
    let m = filmcraft_speech::ecapa::Ecapa::load(&path, false).unwrap();
    let mut embs = Vec::new();
    for e in o.as_array().unwrap() {
        let audio = wav(e["clip"].as_str().unwrap());
        // the front end
        let (frames, f) = filmcraft_speech::fbank::fbank(&audio);
        assert_eq!(frames as u64, e["frames"].as_u64().unwrap());
        for (t, row) in e["fbank_first"].as_array().unwrap().iter().enumerate() {
            for (k, v) in row.as_array().unwrap().iter().enumerate() {
                let d = (f[t * 80 + k] as f64 - v.as_f64().unwrap()).abs();
                assert!(d < 1e-2, "fbank frame {t} band {k}: {} vs {}", f[t * 80 + k], v);
            }
        }
        let want: Vec<f32> = e["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let got = m.embed(&audio).unwrap();
        let cos = filmcraft_speech::ecapa::cosine(&got, &want);
        eprintln!("ecapa {}: cosine to SpeechBrain's {cos:.6}", e["clip"]);
        assert!(cos > 0.999, "cosine {cos}");
        embs.push(got);
        // and on the GPU when there is one
        if let Ok(g) = filmcraft_speech::ecapa::Ecapa::load(&path, true) {
            let gpu = g.embed(&audio).unwrap();
            let c = filmcraft_speech::ecapa::cosine(&gpu, &want);
            eprintln!("  on the GPU: {c:.6}");
            assert!(c > 0.999, "GPU cosine {c}");
        }
    }
}
