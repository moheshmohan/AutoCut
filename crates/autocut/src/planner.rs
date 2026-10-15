//! Deterministic planning: a classified intent plus the document becomes engine commands.
//!
//! Every handler emits ordinary, undoable commands — the same ones a human's menu clicks run —
//! and never invents ids: [`crate::validate`] checks each step against the registry before
//! anything executes.

use serde_json::{Value, json};

use crate::digest::Digest;
use crate::highlight;
use crate::{AutoCutError, Executor, PlannedStep};

/// What planning produced.
pub enum Plan {
    /// A batch ready to run (or show, in a dry run).
    Steps { steps: Vec<PlannedStep>, notes: Vec<String> },
    /// The intent needs a transcript first: run `generate`, then plan again.
    GenerateThen { generate: PlannedStep, params: Value },
}

/// Plan a classified intent (never `custom`).
pub async fn plan(intent: &str, params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    match intent {
        "remove_silence" => remove_silence(params, digest, ex).await,
        "remove_fillers" => remove_fillers(params, digest, ex).await,
        "add_captions" => add_captions(params, digest, ex).await,
        "cut_range" => cut_range(params, digest),
        "keep_range" => keep_range(params, digest),
        "highlight_reel" => highlight_reel(params, digest, ex).await,
        "enhance_voice" => enhance_voice(params, digest),
        "add_voice_over" => add_voice_over(params, digest, ex).await,
        "export_video" => export_video(params, digest),
        "slow_motion" => slow_motion(params, digest),
        other => Err(AutoCutError::Invalid(format!("`{other}` is not a locally planned intent"))),
    }
}

/// Whether the sequence already has stitched transcript words.
fn has_words(digest: &Digest) -> bool {
    digest.transcript.as_ref().and_then(|t| t["words"].as_array()).is_some_and(|w| !w.is_empty())
}

/// A transcript-driven intent: prepend `transcript.generate` when there are no words yet, and
/// refuse (with a fix) when this build cannot transcribe at all.
async fn with_transcript(digest: &Digest, ex: &dyn Executor, steps: Vec<PlannedStep>) -> Result<Plan, AutoCutError> {
    if has_words(digest) {
        return Ok(Plan::Steps { steps, notes: Vec::new() });
    }
    ensure_can_transcribe(ex).await?;
    let mut out = vec![PlannedStep::new("transcript.generate", json!({}), "transcribe the audio first")];
    out.extend(steps);
    Ok(Plan::Steps { steps: out, notes: Vec::new() })
}

async fn ensure_can_transcribe(ex: &dyn Executor) -> Result<(), AutoCutError> {
    let enabled = ex
        .execute("command.list", json!({"filter": "transcript.generate"}))
        .await
        .ok()
        .and_then(|v| v.as_array().and_then(|a| a.first().cloned()))
        .and_then(|c| c["enabled"].as_bool())
        .unwrap_or(false);
    if enabled {
        return Ok(());
    }
    Err(AutoCutError::Transcript(
        "this sequence has no transcript, and this build cannot transcribe (build with the `whisper` feature, or import one with transcript.set)".into(),
    ))
}

async fn remove_silence(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    // an explicit threshold: the engine's own default (1.0s) lets half-second dead air through,
    // which is not what "remove the silences" means to a viewer
    let mut p = json!({"minSeconds": params["minSeconds"].as_f64().unwrap_or(0.5)});
    if let Some(keep) = params["keepSeconds"].as_f64() {
        p["keepSeconds"] = json!(keep);
    }
    with_transcript(digest, ex, vec![PlannedStep::new("transcript.removePauses", p, "cut the pauses between speech")]).await
}

async fn remove_fillers(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let mut p = json!({});
    if let Some(f) = params["fillers"].as_array().filter(|a| !a.is_empty()) {
        p["fillers"] = json!(f);
    }
    with_transcript(digest, ex, vec![PlannedStep::new("transcript.removeFillers", p, "cut the filler words")]).await
}

async fn add_captions(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let style = params["style"].as_str().unwrap_or("default");
    // the new track lands at index 0, i.e. C1; sizes are 1080-line units of the frame's short
    // side, so a style reads the same on landscape and portrait sequences
    let styling = match style {
        "tiktok" => json!({"track": "C1", "size": TIKTOK_SIZE, "anchor": "middle", "align": "center", "outline": 6, "margin": 0.06}),
        "bold" => json!({"track": "C1", "size": 72, "outline": 8}),
        "clean" => json!({"track": "C1", "background": false, "outline": 4}),
        "default" => json!(null),
        other => return Err(AutoCutError::Invalid(format!("caption style `{other}` is not default, tiktok, bold or clean"))),
    };
    // Chunking. A caption line longer than one rendered line wraps, so `lines: 2` became four
    // lines of 84 px text on a portrait frame. Cap characters per line at what fits (the local
    // classifier echoes the documented 42 / 2 defaults, so a stated number cannot be trusted
    // to be the user's). TikTok style is one short, punchy line.
    let size = styling["size"].as_f64().unwrap_or(DEFAULT_CAPTION_SIZE);
    let mut cap = chars_per_line(size, digest);
    if style == "tiktok" {
        cap = cap.min(TIKTOK_MAX_CHARS);
    }
    let mut p = json!({"maxChars": params["maxChars"].as_u64().unwrap_or(DEFAULT_MAX_CHARS).clamp(1, cap)});
    if style == "tiktok" {
        p["lines"] = json!(1);
    } else if let Some(l) = params["lines"].as_u64() {
        p["lines"] = json!(l.clamp(1, 2));
    }
    let mut steps = vec![PlannedStep::new("transcript.createCaptions", p, "caption track from the transcript")];
    if !styling.is_null() {
        steps.push(PlannedStep::new("captions.setStyle", styling, format!("the `{style}` caption style")));
    }
    with_transcript(digest, ex, steps).await
}

/// The caption style's default size (`CaptionStyle::default()`), in 1080-line units.
const DEFAULT_CAPTION_SIZE: f64 = 54.0;
/// `transcript.createCaptions`' default characters per line.
const DEFAULT_MAX_CHARS: u64 = 42;
/// The TikTok style's size and its longest line.
const TIKTOK_SIZE: f64 = 84.0;
const TIKTOK_MAX_CHARS: u64 = 24;

/// How many characters one rendered caption line holds at `size` on the active sequence's
/// frame without wrapping: the burn-in wraps at 90% of the width, sizes are 1080-line units
/// of the short side, and the caption face averages under 0.52 em per character (measured
/// ~0.48 on real captions).
fn chars_per_line(size: f64, digest: &Digest) -> u64 {
    let (w, h) = digest.frame_size().unwrap_or((1920, 1080));
    let short = w.min(h).max(1) as f64;
    let width_units = 1080.0 * w as f64 / short;
    let fit = (0.9 * width_units / (0.52 * size.max(1.0))).floor();
    if fit.is_finite() { fit.clamp(8.0, 120.0) as u64 } else { DEFAULT_MAX_CHARS }
}

/// `markers.markIn` + `markers.markOut` + `sequence.extract`/`lift` + `markers.clearInOut`.
fn mark_and_remove(start: f64, end: f64, mode: &str, why: &str) -> Vec<PlannedStep> {
    vec![
        PlannedStep::new("markers.markIn", json!({"seconds": start}), "mark the range start"),
        PlannedStep::new("markers.markOut", json!({"seconds": end}), "mark the range end"),
        PlannedStep::new(if mode == "lift" { "sequence.lift" } else { "sequence.extract" }, json!({}), why),
        PlannedStep::new("markers.clearInOut", json!({}), "clear the marks"),
    ]
}

fn range_params(params: &Value) -> Result<(f64, f64), AutoCutError> {
    let (start, end) = (params["start"].as_f64(), params["end"].as_f64());
    let (Some(start), Some(end)) = (start, end) else {
        return Err(AutoCutError::Invalid("this intent needs `start` and `end` seconds".into()));
    };
    if !(start.is_finite() && end.is_finite() && start >= 0.0 && end > start) {
        return Err(AutoCutError::Invalid(format!("the range {start}..{end}s makes no sense")));
    }
    Ok((start, end))
}

fn check_within(end: f64, digest: &Digest) -> Result<(), AutoCutError> {
    let duration = digest.duration_seconds();
    if duration > 0.0 && end > duration + 0.001 {
        return Err(AutoCutError::Invalid(format!("the range ends at {end}s but the sequence is only {duration:.1}s")));
    }
    Ok(())
}

fn cut_range(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let (start, end) = range_params(params)?;
    let mode = params["mode"].as_str().unwrap_or("extract");
    if mode != "extract" && mode != "lift" {
        return Err(AutoCutError::Invalid(format!("mode `{mode}` is not extract or lift")));
    }
    check_within(end, digest)?;
    Ok(Plan::Steps { steps: mark_and_remove(start, end, mode, "remove the marked range"), notes: Vec::new() })
}

/// Ripple-extract the head and tail around the range to keep. The tail goes first: removing it
/// does not shift the head's absolute time.
fn keep_range(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let (start, end) = range_params(params)?;
    let duration = digest.duration_seconds();
    if duration <= 0.0 {
        return Err(AutoCutError::Invalid("the sequence duration is unknown".into()));
    }
    if end > duration + 0.001 {
        return Err(AutoCutError::Invalid(format!("the range ends at {end}s but the sequence is only {duration:.1}s")));
    }
    let mut steps = Vec::new();
    if end < duration - 0.001 {
        steps.extend(mark_and_remove(end, duration, "extract", "cut the tail after the kept range"));
    }
    if start > 0.001 {
        steps.extend(mark_and_remove(0.0, start, "extract", "cut the head before the kept range"));
    }
    if steps.is_empty() {
        return Ok(Plan::Steps { steps: Vec::new(), notes: vec!["the range covers the whole sequence; nothing to cut".into()] });
    }
    Ok(Plan::Steps { steps, notes: Vec::new() })
}

async fn highlight_reel(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let Some(words) = digest.transcript.as_ref().and_then(highlight::parse_words) else {
        ensure_can_transcribe(ex).await?;
        return Ok(Plan::GenerateThen { generate: PlannedStep::new("transcript.generate", json!({}), "transcribe the audio first"), params: params.clone() });
    };
    let target = params["targetSeconds"].as_f64().unwrap_or(60.0).clamp(1.0, 3600.0);
    let topic = params["topic"].as_str().map(str::to_string);
    let (keep, removals, notes) = highlight::plan(&words, target, topic.as_deref());
    if removals.is_empty() {
        return Ok(Plan::Steps {
            steps: Vec::new(),
            notes: notes.into_iter().chain(std::iter::once("nothing to cut: everything is worth keeping".into())).collect(),
        });
    }
    let n = removals.len();
    let steps = removals
        .into_iter()
        .map(|(from, to)| PlannedStep::new("transcript.extract", json!({"from": from, "to": to}), "cut a stretch that is not part of the highlight reel"))
        .collect();
    let mut out = vec![format!("kept {keep} stretch(es) totalling about {target:.0}s of speech; cutting everything else in {n} cut(s)")];
    out.extend(notes);
    Ok(Plan::Steps { steps, notes: out })
}

// ---------------------------------------------------------------------------------------------
// Voice, narration, export and speed intents

/// Clip ids on a track kind ("audio" / "video") from the digest.
fn track_clips(digest: &Digest, kind: &str) -> Vec<u64> {
    digest.sequence[kind].as_array().into_iter().flatten().filter_map(|t| t["items"].as_array()).flatten().filter_map(|c| c["clip"].as_u64()).collect()
}

/// `enhance_voice`: treat every audio clip as dialogue, apply an Essential Sound preset,
/// optionally tune repair amounts, and loudness-match (BS.1770) to the dialogue target.
fn enhance_voice(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let clips = track_clips(digest, "audio");
    if clips.is_empty() {
        return Err(AutoCutError::Invalid("this sequence has no audio clips to enhance".into()));
    }
    let preset = match params["preset"].as_str().unwrap_or("clean") {
        "balanced" => "Balanced Voice",
        "podcast" => "Podcast Voice",
        "radio" => "Radio Announcer",
        "hall" => "Voice in a Hall",
        _ => "Clean Up Noisy Dialogue",
    };
    let mut steps = vec![
        PlannedStep::new("essentialSound.setType", json!({"clips": clips, "type": "dialogue"}), "treat the audio as dialogue"),
        PlannedStep::new("essentialSound.applyPreset", json!({"clips": clips, "preset": preset}), format!("the `{preset}` Essential Sound preset")),
    ];
    for (key, param) in [("repair.noise.amount", "noise"), ("repair.deess.amount", "deess"), ("repair.reverb.amount", "dereverb")] {
        if let Some(v) = params[param].as_f64() {
            steps.push(PlannedStep::new("essentialSound.set", json!({"clips": clips, "key": key, "value": v}), format!("{param} = {v}")));
        }
    }
    let mut notes = Vec::new();
    if params["loudness"].as_bool().unwrap_or(true) {
        steps.push(PlannedStep::new("essentialSound.autoMatch", json!({"clips": clips}), "match integrated loudness to the dialogue target (−23 LUFS)"));
    }
    if params["duckMusic"].as_bool().unwrap_or(false) {
        steps.push(PlannedStep::new("essentialSound.generateDucking", json!({"clips": clips}), "duck the music under the voice"));
    }
    if preset == "Clean Up Noisy Dialogue" {
        notes.push("the preset removes noise and hum; pass noise/deess amounts to tune, or preset: balanced|podcast|radio for another character".into());
    }
    Ok(Plan::Steps { steps, notes })
}

/// `add_voice_over`: synthesize narration onto the timeline. `fit` stretches the delivery
/// (pace 0.5..2) to cover the whole sequence — the preview run measures the natural length.
async fn add_voice_over(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let Some(text) = params["text"].as_str().map(str::trim).filter(|t| !t.is_empty()) else {
        return Err(AutoCutError::Invalid("this intent needs `text` to speak".into()));
    };
    let mut p = json!({"text": text, "time": 0});
    if let Some(v) = params["voice"].as_str() {
        p["voice"] = json!(v);
    }
    let mut notes = vec!["natural voices (Kokoro) need the `neural-voices` build and a one-time download; the built-in voice always works".to_string()];
    if params["fit"].as_bool().unwrap_or(false) {
        let target = digest.duration_seconds();
        if target <= 0.0 {
            return Err(AutoCutError::Invalid("cannot fit narration: the sequence duration is unknown".into()));
        }
        // measure the natural delivery, then pick the pace that covers the sequence
        let preview = ex.execute("tts.preview", json!({"text": text})).await.map_err(AutoCutError::Executor)?;
        let spoken = preview["seconds"].as_f64().unwrap_or(0.0);
        if spoken > 0.0 {
            let pace = (spoken / target).clamp(0.5, 2.0);
            p["pace"] = json!(pace);
            if (pace - 0.5).abs() < 1e-9 || (pace - 2.0).abs() < 1e-9 {
                notes.push(format!("the speech is {spoken:.0}s against a {target:.0}s sequence; pace clamped to {pace}"));
            }
        }
    }
    Ok(Plan::Steps { steps: vec![PlannedStep::new("tts.create", p, "synthesize the narration onto the timeline")], notes })
}

/// `export_video`: destination words map to the engine's built-in export presets.
fn export_video(params: &Value, _digest: &Digest) -> Result<Plan, AutoCutError> {
    let dest = params["destination"].as_str().unwrap_or("youtube");
    let (preset, ext) = match dest {
        "youtube" => ("YouTube 1080p Full HD", "mp4"),
        "youtube4k" | "youtube-4k" => ("YouTube 2160p 4K Ultra HD", "mp4"),
        "tiktok" | "reels" | "shorts" | "vertical" => ("Social Vertical 1080×1920", "mp4"),
        "vimeo" => ("Vimeo 1080p Full HD", "mp4"),
        "prores" | "master" => ("Apple ProRes 422 HQ", "mov"),
        "gif" => ("Animated GIF 640×360", "gif"),
        "audio" | "wav" => ("Waveform Audio 48 kHz 24-bit", "wav"),
        "aac" | "mp3" => ("AAC Audio 48 kHz 320 kbps", "m4a"),
        other => {
            return Err(AutoCutError::Invalid(format!(
                "unknown destination `{other}`; try youtube, youtube4k, tiktok, reels, vertical, vimeo, prores, gif, audio or aac"
            )));
        }
    };
    let path = params["path"].as_str().map(str::to_string).unwrap_or_else(|| format!("autocut-{dest}.{ext}"));
    let mut p = json!({"path": path, "preset": preset, "wait": true});
    if params["burnCaptions"].as_bool().unwrap_or(false) {
        p["burnCaptions"] = json!(true);
    }
    if let Some(s) = params["scale"].as_f64() {
        p["scale"] = json!(s);
    }
    let steps = vec![PlannedStep::new("file.exportMedia", p, format!("export with the `{preset}` preset"))];
    Ok(Plan::Steps { steps, notes: vec![format!("writing {path}; long exports report progress over MCP")] })
}

/// `slow_motion`: retime clips with motion-compensated interpolation — `opticalFlow` (the
/// default) is the CapCut-style smooth slow motion, on the engine's own Lucas-Kanade flow.
fn slow_motion(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let factor = params["factor"].as_f64().unwrap_or(0.5).clamp(0.1, 10.0);
    let interp = params["interpolation"].as_str().unwrap_or("opticalFlow");
    if !matches!(interp, "opticalFlow" | "frameBlending" | "frameSampling") {
        return Err(AutoCutError::Invalid(format!("interpolation `{interp}` is not opticalFlow, frameBlending or frameSampling")));
    }
    let clips = params["clips"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<u64>>())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| track_clips(digest, "video"));
    if clips.is_empty() {
        return Err(AutoCutError::Invalid("this sequence has no video clips to retime".into()));
    }
    // rippling makes room: a slowed clip grows, pushing later clips right instead of being
    // trimmed to its old slot (Premiere's Speed/Duration with "Ripple Edit" checked)
    let ripple = params["ripple"].as_bool().unwrap_or(true);
    let steps = vec![PlannedStep::new(
        "clip.speedDuration",
        json!({"clips": clips, "speed": factor * 100.0, "ripple": ripple, "interpolation": interp}),
        if interp == "opticalFlow" {
            format!("{:.0}% speed with optical-flow interpolation (motion-compensated new frames)", factor * 100.0)
        } else {
            format!("{:.0}% speed with {interp}", factor * 100.0)
        },
    )];
    Ok(Plan::Steps { steps, notes: Vec::new() })
}
