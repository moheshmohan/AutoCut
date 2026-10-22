//! What AutoCut tells its models about the project, and which commands they may use.

use serde::Serialize;
use serde_json::{Value, json};

use crate::Executor;

/// Command families AutoCut (and its supervisor) may run — the allowlist every planned step is
/// checked against. Playback, prefs and UI stay out: AutoCut edits media, not the app.
pub const ALLOWED_FAMILIES: &[&str] = &[
    "file.",
    "sequence.",
    "timeline.",
    "source.",
    "markers.",
    "playhead.",
    "edit.",
    "transcript.",
    "captions.",
    "clip.",
    "effects.",
    "essentialSound.",
    "tts.",
    "graphics.",
    "media.",
    // single commands from families that otherwise hold interactive or device commands
    "trim.applyDefaultTransition",
    "audio.generate",
    // finding music by feel (a query: mood, genre, use case, energy → styles and their params)
    "music.suggest",
    "music.taxonomy",
    // the showcase's mix: track effects (a limiter on the mix bus, the stems' EQ and dynamics),
    // the narration's fader, the music's presence dip as track automation, a remix taking an
    // earlier mix's effects off, and the mix measured per stem (QA)
    "mixer.addInsert",
    "mixer.setInsert",
    "mixer.removeInsert",
    "mixer.setStrip",
    "mixer.inspect",
    "mixer.writeAutomation",
    "mixer.clearLane",
    "audio.analyzeMix",
    // "make a video of every robot since 2020": the showcase job, which plans and waits for review
    "autocut.makeShowcase",
    // shot-to-shot colour matching (the showcase's `match_color`)
    "lumetri.applyMatch",
    // the showcase's look: transitions on the beat with their sounds, the mix at a loudness
    // target, and the installed fonts its theme falls back from
    "transitions.snapToBeats",
    "transitions.addSound",
    // the creator tools (opt-in looks and pulses; transitions on every cut or on strong beats)
    "transitions.applyToAll",
    "transitions.randomOnBeats",
    "audio.normalizeLoudness",
    "fonts.list",
    // spec chips that follow the robot (`follow_labels`): a motion track and its callout
    "track.label",
];

/// Whether a command id is in an allowed family.
pub fn allowed(id: &str) -> bool {
    ALLOWED_FAMILIES.iter().any(|f| id.starts_with(f))
}

/// The project state the planners work from, read with ordinary query commands.
#[derive(Debug, Clone, Serialize)]
pub struct Digest {
    pub project: Value,
    pub sequence: Value,
    /// The active sequence's stitched transcript (`transcript.inspect`); `None` when there is none.
    pub transcript: Option<Value>,
}

impl Digest {
    /// Read the document through `ex`. Fails with [`crate::AutoCutError::NoSequence`] when no
    /// sequence is open.
    pub async fn read(ex: &dyn Executor) -> Result<Self, crate::AutoCutError> {
        let project = ex.execute("project.inspect", json!({})).await.map_err(crate::AutoCutError::Executor)?;
        let sequence = ex.execute("sequence.inspect", json!({})).await.map_err(|_| crate::AutoCutError::NoSequence)?;
        let transcript = ex.execute("transcript.inspect", json!({})).await.ok();
        Ok(Self { project, sequence, transcript })
    }

    /// Sequence duration in seconds (0 when unknown).
    pub fn duration_seconds(&self) -> f64 {
        let ticks = self.sequence["duration"].as_i64().unwrap_or(0);
        ticks as f64 / filmcraft_time::TICKS_PER_SECOND as f64
    }

    /// The active sequence's frame size in pixels, when the document says.
    pub fn frame_size(&self) -> Option<(u64, u64)> {
        let st = &self.sequence["settings"];
        match (st["width"].as_u64(), st["height"].as_u64()) {
            (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h)),
            _ => None,
        }
    }

    /// A compact, size-capped rendering of the document for model prompts: sequence shape,
    /// transcript text, no per-effect noise.
    pub fn compact(&self) -> Value {
        let seq = &self.sequence;
        let tracks = |key: &str| {
            seq[key]
                .as_array()
                .map(|ts| json!(ts.iter().map(|t| json!({"id": t["id"], "clips": t["items"].as_array().map(Vec::len).unwrap_or(0)})).collect::<Vec<_>>()))
        };
        let mut transcript = json!({"words": 0, "text": ""});
        if let Some(t) = &self.transcript {
            transcript["words"] = json!(t["words"].as_array().map(Vec::len).unwrap_or(0));
            transcript["speakers"] = t.get("speakers").cloned().unwrap_or(json!([]));
            // the paragraph texts, capped, are what a model can actually reason about
            let mut text = t["paragraphs"].as_array().into_iter().flatten().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n");
            text.truncate(6000);
            transcript["text"] = json!(text);
        }
        json!({
            "sequence": {
                "width": seq["settings"]["width"], "height": seq["settings"]["height"],
                "frameRate": frame_rate(seq).map(|(n, d)| json!(format!("{n}/{d}"))),
                "durationSeconds": self.duration_seconds(),
                "videoTracks": tracks("video"), "audioTracks": tracks("audio"),
                "markIn": seq["in"], "markOut": seq["out"],
            },
            "transcript": transcript,
        })
    }
}

fn frame_rate(seq: &Value) -> Option<(i64, i64)> {
    let r = &seq["settings"]["frameRate"];
    Some((r["numerator"].as_i64()?, r["denominator"].as_i64()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SessionExecutor;
    use filmcraft_engine::Session;

    fn demo() -> SessionExecutor {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        SessionExecutor::from(s)
    }

    #[tokio::test]
    async fn reads_the_demo_document() {
        let ex = demo();
        let d = Digest::read(&ex).await.unwrap();
        assert!(d.duration_seconds() > 0.0, "{:?}", d.duration_seconds());
        let c = d.compact();
        assert!(c["sequence"]["durationSeconds"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn no_sequence_is_a_clear_error() {
        let ex = SessionExecutor::from(Session::default());
        assert!(matches!(Digest::read(&ex).await, Err(crate::AutoCutError::NoSequence)));
    }

    #[test]
    fn families_allowlist() {
        assert!(allowed("timeline.razor"));
        assert!(allowed("essentialSound.autoMatch"));
        assert!(allowed("tts.create"));
        assert!(allowed("lumetri.applyMatch") && !allowed("lumetri.setInputLut"));
        assert!(!allowed("app.quit"));
        assert!(!allowed("prefs.set"));
    }
}
