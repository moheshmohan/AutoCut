//! Frame descriptions: a small local vision model captions sampled frames, giving the
//! supervisor (and visual intents) eyes on the sequence.

use serde_json::{Value, json};

use crate::Executor;
use crate::ollama::Ollama;

/// Evenly spaced sample times across `duration` seconds (never empty, capped at `max`).
pub fn sample_times(duration: f64, max: usize) -> Vec<f64> {
    if duration <= 0.0 {
        return Vec::new();
    }
    let n = duration.min(max as f64).max(1.0).floor() as usize;
    (0..n).map(|i| duration * (i as f64 + 0.5) / n as f64).collect()
}

/// One caption from a vision model's prose reply: trimmed, first sentence, capped.
fn one_sentence(reply: &str) -> Option<String> {
    let trimmed = reply.trim().trim_start_matches('\n').trim();
    if trimmed.is_empty() {
        return None;
    }
    let end = trimmed.find(['.', '!', '?']).map(|i| i + 1).unwrap_or(trimmed.len());
    let mut s = trimmed[..end].trim().to_string();
    if s.len() > 200 {
        s.truncate(200);
    }
    (s.len() >= 8).then_some(s)
}

/// Render and caption `times` (≤8 recommended); frames that cannot render are skipped, and a
/// model failure returns an error the caller can degrade on. Returns `(seconds, description)`.
///
/// Plain prose, not forced JSON: small vision models (moondream, SmolVLM) caption reliably in
/// text, while `format: json` pushes some into hallucinated metadata.
pub async fn describe_frames(vlm: &Ollama, ex: &dyn Executor, times: &[f64], max_side: u32) -> Result<Vec<(f64, String)>, String> {
    let mut out = Vec::new();
    for &t in times {
        let Some(png) = ex.render(t, max_side).await else {
            continue;
        };
        // the HTTP call is blocking; keep it off the async threads
        let vlm = vlm.clone();
        let reply = tokio::task::spawn_blocking(move || {
            vlm.chat_text(
                "You caption video frames for an automated editor. Answer with one short sentence naming the main subject and the setting. No speculation beyond what is visible.",
                "Describe this image.",
                &[png],
            )
        })
        .await
        .map_err(|e| format!("internal error: {e}"))?
        .map_err(|e| e.to_string())?;
        if let Some(d) = one_sentence(&reply) {
            out.push((t, d));
        }
    }
    Ok(out)
}

/// Descriptions as the digest shape: `[{"seconds": f, "description": str}]`.
pub fn to_json(frames: &[(f64, String)]) -> Value {
    Value::Array(frames.iter().map(|(t, d)| json!({"seconds": (t * 10.0).round() / 10.0, "description": d})).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_times_are_evenly_spaced_and_capped() {
        assert_eq!(sample_times(0.0, 8), Vec::<f64>::new());
        assert_eq!(sample_times(2.0, 8), vec![0.5, 1.5], "one sample per second");
        let t = sample_times(10.0, 4);
        assert_eq!(t.len(), 4);
        assert!((t[0] - 1.25).abs() < 1e-9 && (t[3] - 8.75).abs() < 1e-9, "{t:?}");
        assert_eq!(sample_times(100.0, 8).len(), 8);
    }

    #[test]
    fn first_sentence_of_the_reply() {
        assert_eq!(one_sentence("\nThe image presents a serene sunset. More text follows.").as_deref(), Some("The image presents a serene sunset."));
        assert!(one_sentence("   ").is_none());
        assert!(one_sentence("ok").is_none(), "too short to be a caption");
        let long = one_sentence(&format!("{}.", "word ".repeat(100))).unwrap();
        assert!(long.len() <= 200);
    }

    #[test]
    fn json_shape() {
        let v = to_json(&[(1.234, "a rocket".to_string())]);
        assert_eq!(v, json!([{"seconds": 1.2, "description": "a rocket"}]));
    }
}
