//! Voice cloning, part 2: cloned narration voices (feature `kokoro`; the natural engine needs
//! feature `voxcpm`).
//!
//! A cloned voice is a small [`Profile`] saved as JSON in the voices directory, made by one of
//! two engines ([`Engine`]):
//!
//! - **Natural (VoxCPM, profile version 2).** VoxCPM ([`crate::voxcpm`]) speaks directly in the
//!   speaker's voice from a few seconds of it: timbre, accent, rhythm and recording colour. The
//!   profile keeps a cleaned voice sample next to it (`<name>.wav`: the best continuous 10–25 s of
//!   speech in the reference, long pauses shortened, loudness-normalised, at the model's encoder
//!   rate) and that sample's transcript. VoxCPM2 clones from the sample alone (a reference
//!   prompt); VoxCPM1.5 continues the sample, which needs its exact transcript, and gets a
//!   denoised sample because a noisy one derails the continuation.
//! - **Timbre only (OpenVoice, profile version 1).** The speaker's tone colour, the Kokoro voice
//!   it is built on, and that voice's tone colour. To speak, the base voice reads the script and
//!   the OpenVoice converter ([`crate::openvoice`]) re-voices it from the base colour to the
//!   speaker's — words, timing and intonation from Kokoro, timbre from the speaker. Unless one is
//!   named, every installed Kokoro voice reads a calibration passage and the one whose tone colour
//!   is closest to the speaker's (cosine) is used.
//!
//! `clone:<name>` voices dispatch on the profile's engine ([`load`]); profiles made before the
//! natural engine existed keep working unchanged.
//!
//! **Consent.** A profile records that whoever made it confirmed they may clone that voice
//! (their own, or with the speaker's permission); [`create`] and [`create_natural`] refuse
//! without it and [`load`] refuses a profile without it.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use serde::{Deserialize, Serialize};

use crate::kokoro::KOKORO_RATE;
use crate::openvoice::{COLOUR, Converter, DEFAULT_TAU, RATE as OV_RATE};
use crate::{Audio, CLONE_PREFIX, Gender, Params, TtsError, Voice, VoiceInfo, catalog, neural};

/// Profile version of an OpenVoice (timbre-only) clone.
pub const PROFILE_VERSION: u32 = 1;
/// Profile version of a VoxCPM (natural) clone.
pub const NATURAL_PROFILE_VERSION: u32 = 2;
/// Least speech (after trimming silence) a clone is made from.
pub const MIN_SPEECH_SECONDS: f64 = 3.0;
/// Most speech used (the colour is an average; more adds time, not quality).
pub const MAX_SPEECH_SECONDS: f64 = 120.0;
/// The natural engine's voice sample: the best continuous stretch of speech this long.
pub const SAMPLE_SECONDS: (f64, f64) = (10.0, 25.0);
/// A sample VoxCPM continues is shorter: its transcript must be exact, and every word more is
/// one more chance for a transcription error the model would then "finish saying".
pub const CONTINUATION_SECONDS: (f64, f64) = (5.0, 10.0);
/// Loudness the voice sample is normalised to.
pub const SAMPLE_LUFS: f64 = -20.0;
/// Longest transcript kept with a profile.
const MAX_TRANSCRIPT_CHARS: usize = 4000;
/// Most profiles listed (each file at most 64 KB).
const MAX_PROFILES: usize = 200;
const MAX_PROFILE_BYTES: u64 = 64 * 1024;

/// What the base voices read to measure their tone colour: varied vowels and consonants, about
/// fifteen seconds.
const CALIBRATION: &str = "The quick brown fox jumps over the lazy dog. Every robot we met had its own personality: some were tiny and playful, others were enormous industrial machines. Would you believe how far they have come in only a few years?";

/// How a cloned voice speaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// Kokoro speech re-voiced by the OpenVoice tone-colour converter (timbre only).
    #[default]
    Openvoice,
    /// VoxCPM zero-shot cloning (timbre, accent and rhythm).
    Voxcpm,
}

impl Engine {
    pub fn id(self) -> &'static str {
        match self {
            Engine::Openvoice => "openvoice",
            Engine::Voxcpm => "voxcpm",
        }
    }

    pub fn from_id(s: &str) -> Option<Engine> {
        match s {
            "openvoice" => Some(Engine::Openvoice),
            "voxcpm" => Some(Engine::Voxcpm),
            _ => None,
        }
    }
}

/// How VoxCPM uses the voice sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    /// The sample as an isolated reference (VoxCPM2): no transcript needed, robust to noise and
    /// to transcription errors.
    #[default]
    Reference,
    /// The model continues the sample ("ultimate cloning"): needs its exact transcript.
    Continuation,
}

/// The natural engine's settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Natural {
    /// The VoxCPM release (`catalog::VOXCPM` id).
    pub model: String,
    /// The voice sample's file name next to the profile (`<name>.wav`).
    pub clip: String,
    /// The sample's transcript (empty when not known).
    #[serde(default)]
    pub transcript: String,
    #[serde(default)]
    pub mode: PromptMode,
    /// Classifier-free guidance.
    pub cfg: f32,
    /// Flow-matching steps per patch.
    pub timesteps: usize,
    /// Noise reduction applied to the sample when the prompt is built (dB, 0 = none).
    #[serde(default)]
    pub denoise_db: f32,
    /// Where in the reference the sample was taken from (seconds).
    #[serde(default)]
    pub from: f64,
    #[serde(default)]
    pub to: f64,
    /// Estimated from the sample's pitch.
    #[serde(default)]
    pub gender: Option<Gender>,
}

/// A cloned voice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub version: u32,
    /// `clone:<slug>`.
    pub id: String,
    /// What the user called it.
    pub name: String,
    /// Profiles from before the natural engine have none: OpenVoice.
    #[serde(default)]
    pub engine: Engine,
    /// OpenVoice: the Kokoro voice it speaks through.
    #[serde(default)]
    pub base_voice: String,
    /// OpenVoice: the base voice's tone colour (256).
    #[serde(default)]
    pub source_colour: Vec<f32>,
    /// OpenVoice: the speaker's tone colour (256).
    #[serde(default)]
    pub target_colour: Vec<f32>,
    /// OpenVoice: cosine similarity of the two colours before conversion.
    #[serde(default)]
    pub base_similarity: f32,
    /// VoxCPM: the voice sample and how it is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural: Option<Natural>,
    /// Seconds of speech the voice was taken from.
    pub reference_seconds: f64,
    /// Where the reference came from (file names, clip ids), for the user's records.
    pub source: String,
    /// Unix seconds.
    pub created: u64,
    /// The maker confirmed they may clone this voice.
    pub consent: bool,
    /// How often the voice's reference was replaced ([`promote`], [`rollback`]); 0 as made.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub revision: u32,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

impl Profile {
    fn check(&self) -> Result<(), TtsError> {
        let ok = |v: &[f32]| v.len() == COLOUR && v.iter().all(|x| x.is_finite());
        let slug_part = self.id.strip_prefix(CLONE_PREFIX).unwrap_or("");
        if slug_part.is_empty() || slug(slug_part) != slug_part {
            return Err(TtsError::Invalid(format!("bad cloned voice id `{}`", self.id)));
        }
        if !self.consent {
            return Err(TtsError::Invalid("this cloned voice was saved without consent".into()));
        }
        match (self.version, self.engine) {
            (PROFILE_VERSION, Engine::Openvoice) => {
                if !ok(&self.source_colour) || !ok(&self.target_colour) {
                    return Err(TtsError::Invalid("the voice profile's tone colours are damaged".into()));
                }
                if catalog::find(&self.base_voice).is_none() {
                    return Err(TtsError::UnknownVoice(self.base_voice.clone()));
                }
            }
            (NATURAL_PROFILE_VERSION, Engine::Voxcpm) => {
                let n = self.natural.as_ref().ok_or_else(|| TtsError::Invalid("the voice profile has no voice sample".into()))?;
                if catalog::voxcpm(&n.model).is_none() {
                    return Err(TtsError::Invalid(format!("unknown cloning model `{}`", n.model)));
                }
                if n.clip != format!("{slug_part}.wav") {
                    return Err(TtsError::Invalid("the voice profile's sample file name is damaged".into()));
                }
                if n.transcript.chars().count() > MAX_TRANSCRIPT_CHARS {
                    return Err(TtsError::Invalid("the voice sample's transcript is too long".into()));
                }
                if n.mode == PromptMode::Continuation && !n.transcript.chars().any(char::is_alphanumeric) {
                    return Err(TtsError::Invalid("continuing the voice sample needs its transcript".into()));
                }
                if n.model == "voxcpm-1.5" && n.mode != PromptMode::Continuation {
                    return Err(TtsError::Invalid("VoxCPM1.5 only continues the voice sample".into()));
                }
                if !(n.cfg.is_finite() && (1.0..=5.0).contains(&n.cfg)) || !(2..=50).contains(&n.timesteps) {
                    return Err(TtsError::Invalid("the voice profile's quality settings are out of range".into()));
                }
                if !(n.denoise_db.is_finite() && (0.0..=40.0).contains(&n.denoise_db)) {
                    return Err(TtsError::Invalid("the voice profile's noise reduction is out of range".into()));
                }
            }
            (v, e) => return Err(TtsError::Invalid(format!("voice profile version {v} ({}) is not supported", e.id()))),
        }
        Ok(())
    }

    /// The natural engine's model, if this is a natural clone.
    pub fn model(&self) -> Option<&str> {
        self.natural.as_ref().map(|n| n.model.as_str())
    }
}

/// The files of a natural voice's references next to its profile, by extension: the voice
/// sample, its full-band copy, the Malayalam reference, the clip it had before a trim, and the
/// Malayalam reference of each voice style ([`crate::styles::STYLES`], [`set_style`]).
pub const REFERENCE_FILES: [&str; 20] = [
    "wav",
    "full.wav",
    "indicf5.json",
    "indicf5.wav",
    "indicf5.untrimmed.wav",
    "indicf5.untrimmed.json",
    "indicf5.conversational.json",
    "indicf5.conversational.wav",
    "indicf5.warm.json",
    "indicf5.warm.wav",
    "indicf5.excited.json",
    "indicf5.excited.wav",
    "indicf5.amazed.json",
    "indicf5.amazed.wav",
    "indicf5.curious.json",
    "indicf5.curious.wav",
    "indicf5.calm.json",
    "indicf5.calm.wav",
    "indicf5.dramatic.json",
    "indicf5.dramatic.wav",
];
/// The folder of a voice's earlier references (`<name>.versions`).
const VERSIONS_EXT: &str = "versions";

#[path = "clone_versions.rs"]
mod versioning;
pub use versioning::{MAX_VERSIONS, NewReference, Part, Version, promote, rollback, sample_from, versions};

#[path = "clone_styles.rs"]
mod style_refs;
pub use style_refs::{NewStyle, STYLE_SECONDS, StyleSet, remove_style, set_style, style_reference, style_references, styles_on_disk};

/// `name` → a file-safe id part: lowercase ASCII letters, digits and dashes, at most 40.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 40 {
            break;
        }
    }
    out.trim_matches('-').to_string()
}

/// `<data>/voices` for `<data>/models`.
pub fn voices_dir(models_dir: &Path) -> PathBuf {
    models_dir.parent().unwrap_or(models_dir).join("voices")
}

fn profile_path(voices_dir: &Path, id: &str) -> Result<PathBuf, TtsError> {
    let s = slug(id.trim_start_matches(CLONE_PREFIX));
    if s.is_empty() {
        return Err(TtsError::Invalid(format!("bad cloned voice id `{id}`")));
    }
    Ok(voices_dir.join(format!("{s}.json")))
}

/// The voice sample of a natural clone.
pub fn sample_path(voices_dir: &Path, p: &Profile) -> Option<PathBuf> {
    p.natural.as_ref().map(|n| voices_dir.join(&n.clip))
}

/// Read and check the profile `id` (`clone:<slug>`).
pub fn load_profile(voices_dir: &Path, id: &str) -> Result<Profile, TtsError> {
    let path = profile_path(voices_dir, id)?;
    let meta = std::fs::metadata(&path).map_err(|_| TtsError::UnknownVoice(id.to_string()))?;
    if meta.len() > MAX_PROFILE_BYTES {
        return Err(TtsError::Invalid(format!("{}: too large for a voice profile", path.display())));
    }
    let bytes = std::fs::read(&path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    let p: Profile = serde_json::from_slice(&bytes).map_err(|e| TtsError::Invalid(format!("{}: {e}", path.display())))?;
    p.check()?;
    // the file name is the id: a renamed file must not pose as another voice
    if profile_path(voices_dir, &p.id)? != path {
        return Err(TtsError::Invalid(format!("{}: the profile is for `{}`", path.display(), p.id)));
    }
    Ok(p)
}

/// Every readable profile, by name.
pub fn list(voices_dir: &Path) -> Vec<Profile> {
    let Ok(rd) = std::fs::read_dir(voices_dir) else { return Vec::new() };
    let mut out: Vec<Profile> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        // a profile is `<slug>.json`: the references beside it (`<slug>.indicf5.json`, …) are not
        .filter(|e| e.path().file_stem().is_some_and(|s| !s.to_string_lossy().contains('.')))
        .take(MAX_PROFILES)
        .filter_map(|e| {
            let stem = e.path().file_stem()?.to_string_lossy().into_owned();
            load_profile(voices_dir, &format!("{CLONE_PREFIX}{stem}")).ok()
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Remove the profile `id` (and its voice sample).
pub fn delete(voices_dir: &Path, id: &str) -> Result<(), TtsError> {
    let path = profile_path(voices_dir, id)?;
    std::fs::remove_file(&path).map_err(|_| TtsError::UnknownVoice(id.to_string()))?;
    // the sample has the same stem; it may not exist (OpenVoice clones have none)
    for ext in REFERENCE_FILES {
        let _ = std::fs::remove_file(path.with_extension(ext));
    }
    // and the references it had before ([`promote`])
    let _ = std::fs::remove_dir_all(path.with_extension(VERSIONS_EXT));
    Ok(())
}

fn save(voices_dir: &Path, p: &Profile) -> Result<(), TtsError> {
    std::fs::create_dir_all(voices_dir).map_err(|e| TtsError::Model(format!("{}: {e}", voices_dir.display())))?;
    let path = profile_path(voices_dir, &p.id)?;
    let bytes = serde_json::to_vec_pretty(p).map_err(|e| TtsError::Model(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| TtsError::Model(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))
}

/// The converter in `models_dir`, loaded once and shared.
fn converter(models_dir: &Path) -> Result<Arc<Converter>, TtsError> {
    static C: OnceLock<Mutex<Option<(PathBuf, Arc<Converter>)>>> = OnceLock::new();
    if !catalog::clone_installed(models_dir) {
        return Err(TtsError::NotInstalled);
    }
    let dir = catalog::clone_package_dir(models_dir);
    let mut g = C.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((d, c)) = g.as_ref()
        && *d == dir
    {
        return Ok(c.clone());
    }
    let c = Arc::new(Converter::load(&dir)?);
    *g = Some((dir, c.clone()));
    Ok(c)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { (dot / (na * nb)) as f32 }
}

/// 20 ms frames of `x` (mono at `rate`) that hold speech: above an adaptive level, with 200 ms
/// of hang-over so words are not clipped and 60 ms before each onset. Returns the frame length.
fn speech_mask(x: &[f32], rate: u32) -> (usize, Vec<bool>) {
    let frame = (rate as usize / 50).max(1);
    let levels: Vec<f32> = x.chunks(frame).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len().max(1) as f32).sqrt()).collect();
    if levels.is_empty() {
        return (frame, Vec::new());
    }
    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let floor = sorted.get(sorted.len() / 10).copied().unwrap_or(0.0);
    let loud = sorted.get(sorted.len() * 9 / 10).copied().unwrap_or(0.0);
    // between the noise floor and the speech level, nearer the floor; never below −50 dBFS
    let threshold = (floor + 0.2 * (loud - floor)).max(0.003);
    let hang = 10usize;
    let mut keep = vec![false; levels.len()];
    let mut since = usize::MAX;
    for (i, l) in levels.iter().enumerate() {
        since = if *l >= threshold { 0 } else { since.saturating_add(1) };
        if since <= hang {
            if let Some(k) = keep.get_mut(i) {
                *k = true;
            }
            // and a little before the onset
            if since == 0 {
                for j in i.saturating_sub(3)..i {
                    if let Some(k) = keep.get_mut(j) {
                        *k = true;
                    }
                }
            }
        }
    }
    (frame, keep)
}

/// Keep the speech of `x` (mono at `rate`): 20 ms frames above an adaptive level, with 200 ms
/// of hang-over so words are not clipped; everything else (silence, room tone) is dropped.
pub fn speech_only(x: &[f32], rate: u32) -> Vec<f32> {
    let (frame, keep) = speech_mask(x, rate);
    x.chunks(frame).zip(&keep).filter(|(_, k)| **k).flat_map(|(c, _)| c.iter().copied()).collect()
}

/// A voice sample cut from a reference recording.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Mono at the reference's rate: pauses longer than half a second shortened, edges trimmed,
    /// loudness-normalised.
    pub audio: Vec<f32>,
    /// Where it was taken from, seconds.
    pub from: f64,
    pub to: f64,
    /// Seconds of speech in it.
    pub speech_seconds: f64,
}

/// The best continuous stretch of speech in `x` (mono at `rate`) for a voice sample: between
/// pauses (so no word is cut), [`SAMPLE_SECONDS`] long where the recording allows, as much speech
/// and as few long pauses as possible. Long pauses are shortened, the edges trimmed and the
/// loudness normalised to [`SAMPLE_LUFS`] (peaks kept below −0.5 dBFS).
pub fn pick_sample(x: &[f32], rate: u32) -> Result<Sample, TtsError> {
    pick_sample_within(x, rate, SAMPLE_SECONDS)
}

/// [`pick_sample`] with other length bounds (seconds).
pub fn pick_sample_within(x: &[f32], rate: u32, bounds: (f64, f64)) -> Result<Sample, TtsError> {
    if rate == 0 {
        return Err(TtsError::Invalid("the reference audio has no sample rate".into()));
    }
    let (frame, mask) = speech_mask(x, rate);
    let n = mask.len();
    let fps = f64::from(rate) / frame as f64;
    let speech_total = mask.iter().filter(|m| **m).count() as f64 / fps;
    if speech_total < MIN_SPEECH_SECONDS {
        return Err(TtsError::Invalid(format!("the reference has {speech_total:.1} s of speech; at least {MIN_SPEECH_SECONDS} s is needed")));
    }
    // silence runs (start, len) in frames
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if mask.get(i) == Some(&false) {
            let s = i;
            while mask.get(i) == Some(&false) {
                i += 1;
            }
            runs.push((s, i - s));
        } else {
            i += 1;
        }
    }
    // cut points: the recording's edges and the middle of every pause of 100 ms or more, each
    // with the excess (beyond 0.7 s) of the pauses before it
    let mut speech_before = Vec::with_capacity(n + 1);
    let mut count = 0usize;
    speech_before.push(0);
    for m in &mask {
        count += usize::from(*m);
        speech_before.push(count);
    }
    let (lo, hi) = bounds;
    if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo) {
        return Err(TtsError::Invalid("bad voice sample length bounds".into()));
    }
    // a short recording: everything between its first and last speech
    let lo = lo.min(speech_total * 0.9);
    // cut points: the recording's edges and the middle of every pause of `min_pause` frames or
    // more, each with the excess (beyond 0.7 s) of the pauses before it
    let cuts_at = |min_pause: usize| -> Vec<(usize, f64)> {
        let mut cuts: Vec<(usize, f64)> = vec![(0, 0.0)];
        let mut excess = 0.0;
        for (s, len) in runs.iter().take(20_000) {
            if *len >= min_pause {
                cuts.push((s + len / 2, excess));
                excess += (*len as f64 / fps - 0.7).max(0.0);
            }
        }
        cuts.push((n, excess));
        cuts
    };
    let best_window = |cuts: &[(usize, f64)]| -> Option<(f64, usize, usize)> {
        let mut best: Option<(f64, usize, usize)> = None;
        for (ai, (a, ex_a)) in cuts.iter().enumerate() {
            for (b, ex_b) in cuts.iter().skip(ai + 1) {
                let len = (b - a) as f64 / fps;
                if len > hi {
                    break;
                }
                if len < lo {
                    continue;
                }
                let speech = (speech_before.get(*b).copied().unwrap_or(0) - speech_before.get(*a).copied().unwrap_or(0)) as f64 / fps;
                // more speech is better; long pauses inside and lengths past 20 s cost a little
                let score = speech - 2.0 * (ex_b - ex_a) - 0.5 * (len - 20.0).max(0.0);
                if best.is_none_or(|(s, _, _)| score > s) {
                    best = Some((score, *a, *b));
                }
            }
        }
        best
    };
    // between pauses of 100 ms or more; else of any length; else (speech without a break as long
    // as the sample) every half second
    let grid: Vec<(usize, f64)> = (0..=n).step_by(((0.5 * fps) as usize).max(1)).chain(std::iter::once(n)).map(|f| (f, 0.0)).collect();
    let best = best_window(&cuts_at((0.1 * fps).ceil() as usize)).or_else(|| best_window(&cuts_at(1))).or_else(|| best_window(&grid));
    let (_, a, b) = best.ok_or_else(|| TtsError::Invalid("no stretch of continuous speech was found in the reference".into()))?;
    // the stretch, with pauses over 0.5 s cut down to 0.5 s and the edges trimmed to 0.15 s
    let keep_pause = (0.25 * fps) as usize;
    let edge = (0.15 * fps) as usize;
    let first = (a..b).find(|k| mask.get(*k) == Some(&true)).unwrap_or(a);
    let last = (a..b).rev().find(|k| mask.get(*k) == Some(&true)).unwrap_or(b);
    let (from_f, to_f) = (first.saturating_sub(edge).max(a), (last + 1 + edge).min(b));
    let mut audio: Vec<f32> = Vec::with_capacity((to_f - from_f) * frame);
    let mut k = from_f;
    while k < to_f {
        if mask.get(k) == Some(&false) {
            let s = k;
            while k < to_f && mask.get(k) == Some(&false) {
                k += 1;
            }
            let len = k - s;
            let ranges = if len > 2 * keep_pause && s > from_f && k < to_f { vec![(s, s + keep_pause), (k - keep_pause, k)] } else { vec![(s, k)] };
            for (p, q) in ranges {
                audio.extend_from_slice(x.get(p * frame..(q * frame).min(x.len())).unwrap_or(&[]));
            }
        } else {
            audio.extend_from_slice(x.get(k * frame..((k + 1) * frame).min(x.len())).unwrap_or(&[]));
            k += 1;
        }
    }
    let speech_seconds = (speech_before.get(to_f).copied().unwrap_or(0) - speech_before.get(from_f).copied().unwrap_or(0)) as f64 / fps;
    normalize_loudness(&mut audio, rate);
    Ok(Sample { audio, from: from_f as f64 / fps, to: to_f as f64 / fps, speech_seconds })
}

/// Bring speech to [`SAMPLE_LUFS`] (ITU-R BS.1770 integrated loudness), at most +30 dB, peaks
/// below −0.5 dBFS.
fn normalize_loudness(x: &mut [f32], rate: u32) {
    let l = filmcraft_audio_dsp::loudness::measure(&[x], f64::from(rate)).integrated_lufs;
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if !l.is_finite() || peak <= 0.0 {
        return;
    }
    let mut g = 10f64.powf((SAMPLE_LUFS - l) / 20.0).min(10f64.powf(30.0 / 20.0)) as f32;
    if peak * g > 0.944 {
        g = 0.944 / peak;
    }
    x.iter_mut().for_each(|v| *v *= g);
}

/// Speaker gender from the median pitch ([`median_pitch`]; 165 Hz splits the usual ranges).
/// `None` with too little voiced speech.
pub fn estimate_gender(x: &[f32], rate: u32) -> Option<Gender> {
    let median = median_pitch(x, rate)?;
    Some(if median > 165.0 { Gender::Female } else { Gender::Male })
}

/// The median pitch (Hz) of the voiced 40 ms frames of speech `x` at `rate` (autocorrelation
/// at 8 kHz, 60–400 Hz). `None` with too little voiced speech.
pub fn median_pitch(x: &[f32], rate: u32) -> Option<f64> {
    let mut f0s = pitch_frames(x, rate);
    if f0s.len() < 20 {
        return None;
    }
    f0s.sort_by(f64::total_cmp);
    f0s.get(f0s.len() / 2).copied()
}

/// The pitch (Hz) of each voiced 40 ms frame of `x` at `rate`, every 20 ms (autocorrelation at
/// 8 kHz, 60–400 Hz; at most 5000 frames): what [`median_pitch`] and the voice-style
/// measurements (pitch range) read.
pub fn pitch_frames(x: &[f32], rate: u32) -> Vec<f64> {
    let y = filmcraft_audio_dsp::resample::resample_sinc(x, rate, 8_000);
    let (win, hop) = (320usize, 160usize);
    let (min_lag, max_lag) = (20usize, 134usize); // 400 Hz … 60 Hz
    let mut f0s: Vec<f64> = Vec::new();
    let mut s = 0;
    while s + win + max_lag <= y.len() && f0s.len() < 5000 {
        let w = y.get(s..s + win + max_lag).unwrap_or(&[]);
        let e0: f64 = w.iter().take(win).map(|v| f64::from(*v) * f64::from(*v)).sum();
        if e0 > 1e-4 {
            let rs: Vec<f64> = (min_lag..max_lag)
                .map(|lag| {
                    let (mut num, mut el) = (0f64, 0f64);
                    for i in 0..win {
                        let (a, b) = (w.get(i).copied().unwrap_or(0.0), w.get(i + lag).copied().unwrap_or(0.0));
                        num += f64::from(a) * f64::from(b);
                        el += f64::from(b) * f64::from(b);
                    }
                    num / (e0 * el).sqrt().max(1e-12)
                })
                .collect();
            let top = rs.iter().copied().fold(0f64, f64::max);
            // the shortest period that correlates almost as well as the best (whole multiples of
            // the period correlate too: taking the best alone halves high voices)
            if top > 0.6
                && let Some(i) = rs.iter().position(|r| *r >= 0.9 * top)
            {
                f0s.push(8_000.0 / (min_lag + i) as f64);
            }
        }
        s += hop;
    }
    f0s
}

/// Make and save a timbre-only (OpenVoice) cloned voice from `reference` (mono speech at
/// `rate`). `base`: a Kokoro voice id, or `None` to pick the closest. `progress` gets one line
/// per step.
#[allow(clippy::too_many_arguments)]
pub fn create(
    name: &str,
    reference: &[f32],
    rate: u32,
    base: Option<&str>,
    source: &str,
    consent: bool,
    models_dir: &Path,
    progress: &mut dyn FnMut(&str),
) -> Result<Profile, TtsError> {
    let id_part = check_request(name, consent, rate)?;
    if let Some(b) = base
        && catalog::find(b).is_none()
    {
        return Err(TtsError::UnknownVoice(b.to_string()));
    }
    if !catalog::installed(models_dir) {
        return Err(TtsError::NotInstalled);
    }
    let conv = converter(models_dir)?;
    progress("finding the speech in the reference");
    let speech = speech_only(reference, rate);
    let secs = speech.len() as f64 / f64::from(rate);
    if secs < MIN_SPEECH_SECONDS {
        return Err(TtsError::Invalid(format!("the reference has {secs:.1} s of speech; at least {MIN_SPEECH_SECONDS} s is needed")));
    }
    let take = ((MAX_SPEECH_SECONDS * f64::from(rate)) as usize).min(speech.len());
    let speech = filmcraft_audio_dsp::resample::resample_sinc(speech.get(..take).unwrap_or(&speech), rate, OV_RATE);
    progress("measuring the speaker's tone colour");
    let target = conv.tone_colour(&speech)?;
    let candidates: Vec<&str> = match base {
        Some(b) => vec![b],
        None => catalog::VOICES.iter().map(|v| v.info.id).collect(),
    };
    let mut best: Option<(f32, &str, Vec<f32>)> = None;
    for id in candidates {
        progress(&format!("measuring {id}"));
        let colour = base_colour(id, models_dir, &conv)?;
        let sim = cosine(&colour, &target);
        if best.as_ref().is_none_or(|(s, _, _)| sim > *s) {
            best = Some((sim, id, colour));
        }
    }
    let (sim, base_voice, source_colour) = best.ok_or_else(|| TtsError::Model("no base voice to clone onto".into()))?;
    progress(&format!("speaking through {base_voice} (similarity {sim:.2})"));
    let profile = Profile {
        version: PROFILE_VERSION,
        id: format!("{CLONE_PREFIX}{id_part}"),
        name: name.trim().chars().take(80).collect(),
        engine: Engine::Openvoice,
        base_voice: base_voice.to_string(),
        source_colour,
        target_colour: target,
        base_similarity: sim,
        natural: None,
        reference_seconds: take as f64 / f64::from(rate),
        source: source.chars().take(500).collect(),
        created: now(),
        consent: true,
        revision: 0,
    };
    save(&voices_dir(models_dir), &profile)?;
    Ok(profile)
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Consent, a usable name and a sample rate: the id part.
fn check_request(name: &str, consent: bool, rate: u32) -> Result<String, TtsError> {
    if !consent {
        return Err(TtsError::Invalid("cloning a voice needs consent: confirm it is your own voice or that the speaker agreed (consent: true)".into()));
    }
    let id_part = slug(name);
    if id_part.is_empty() {
        return Err(TtsError::Invalid("give the cloned voice a name with letters or digits".into()));
    }
    if rate == 0 {
        return Err(TtsError::Invalid("the reference audio has no sample rate".into()));
    }
    Ok(id_part)
}

/// The tone colour of a Kokoro voice reading the calibration passage.
fn base_colour(id: &str, models_dir: &Path, conv: &Converter) -> Result<Vec<f32>, TtsError> {
    let v = neural::load(id, models_dir)?;
    let a = v.synthesize(CALIBRATION, &Params::default())?;
    let x = filmcraft_audio_dsp::resample::resample_sinc(&a.samples, a.sample_rate, OV_RATE);
    conv.tone_colour(&x)
}

/// What a natural clone is made with.
#[derive(Clone, Debug, PartialEq)]
pub struct NaturalRequest {
    /// `catalog::VOXCPM` id.
    pub model: String,
    /// `None`: the model's best (VoxCPM2 reference, VoxCPM1.5 continuation).
    pub mode: Option<PromptMode>,
    /// The sample's transcript when the caller has one; else `transcribe` is asked.
    pub transcript: Option<String>,
    pub cfg: f32,
    pub timesteps: usize,
}

impl Default for NaturalRequest {
    fn default() -> Self {
        NaturalRequest { model: catalog::DEFAULT_VOXCPM.into(), mode: None, transcript: None, cfg: 2.0, timesteps: 10 }
    }
}

/// Noise reduction for a continued sample: VoxCPM continues the recording, room noise and all,
/// and a noisy one (a phone video) derails it into babble; an isolated reference is used as is.
pub const CONTINUATION_DENOISE_DB: f32 = 18.0;

/// Make and save a natural (VoxCPM) cloned voice from `reference` (mono at `rate`, any length):
/// pick the voice sample ([`pick_sample`]), transcribe it when the mode needs a transcript and
/// none was given (`transcribe` takes mono 16 kHz), and save the profile and the sample.
#[allow(clippy::too_many_arguments)]
pub fn create_natural(
    name: &str,
    reference: &[f32],
    rate: u32,
    req: &NaturalRequest,
    source: &str,
    consent: bool,
    models_dir: &Path,
    transcribe: &mut dyn FnMut(&[f32]) -> Result<String, String>,
    progress: &mut dyn FnMut(&str),
) -> Result<Profile, TtsError> {
    let id_part = check_request(name, consent, rate)?;
    let pkg = catalog::voxcpm(&req.model).ok_or_else(|| TtsError::Invalid(format!("unknown cloning model `{}`", req.model)))?;
    let v1 = pkg.id == "voxcpm-1.5";
    let mode = match (req.mode, v1) {
        (_, true) => PromptMode::Continuation,
        (Some(m), false) => m,
        (None, false) => PromptMode::Reference,
    };
    if !(req.cfg.is_finite() && (1.0..=5.0).contains(&req.cfg)) || !(2..=50).contains(&req.timesteps) {
        return Err(TtsError::Invalid("guidance must be 1–5 and steps 2–50".into()));
    }
    if !catalog::voxcpm_installed(models_dir, pkg) {
        return Err(TtsError::NotInstalled);
    }
    save_natural(&id_part, name, reference, rate, pkg.id, mode, req, source, &voices_dir(models_dir), transcribe, progress)
}

/// The part of [`create_natural`] after the checks: pick, transcribe and save the sample.
#[allow(clippy::too_many_arguments)]
fn save_natural(
    id_part: &str,
    name: &str,
    reference: &[f32],
    rate: u32,
    model: &str,
    mode: PromptMode,
    req: &NaturalRequest,
    source: &str,
    voices: &Path,
    transcribe: &mut dyn FnMut(&[f32]) -> Result<String, String>,
    progress: &mut dyn FnMut(&str),
) -> Result<Profile, TtsError> {
    let encode_rate = natural_encode_rate(model);
    progress("finding the best stretch of speech");
    let bounds = if mode == PromptMode::Continuation { CONTINUATION_SECONDS } else { SAMPLE_SECONDS };
    let sample = pick_sample_within(reference, rate, bounds)?;
    let clip = filmcraft_audio_dsp::resample::resample_sinc(&sample.audio, rate, encode_rate);
    let gender = estimate_gender(&sample.audio, rate);
    let transcript = match req.transcript.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => t.chars().take(MAX_TRANSCRIPT_CHARS).collect(),
        None if mode == PromptMode::Continuation => {
            progress("transcribing the voice sample");
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(&sample.audio, rate, 16_000);
            let t = transcribe(&x16).map_err(|e| TtsError::Invalid(format!("the voice sample could not be transcribed ({e}); pass its transcript")))?;
            let t: String = t.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TRANSCRIPT_CHARS).collect();
            if !t.chars().any(char::is_alphanumeric) {
                return Err(TtsError::Invalid("no words were heard in the voice sample; pass its transcript".into()));
            }
            t
        }
        None => String::new(),
    };
    std::fs::create_dir_all(voices).map_err(|e| TtsError::Model(format!("{}: {e}", voices.display())))?;
    let clip_name = format!("{id_part}.wav");
    crate::wav::write(&voices.join(&clip_name), &clip, encode_rate)?;
    // and at the recording's own rate when that is higher: IndicF5 (Malayalam) speaks from a
    // full-band clip much more like the speaker than from VoxCPM2's 16 kHz sample
    let full = voices.join(format!("{id_part}.full.wav"));
    if rate > encode_rate {
        crate::wav::write(&full, &sample.audio, rate)?;
    } else {
        let _ = std::fs::remove_file(&full);
    }
    let profile = Profile {
        version: NATURAL_PROFILE_VERSION,
        id: format!("{CLONE_PREFIX}{id_part}"),
        name: name.trim().chars().take(80).collect(),
        engine: Engine::Voxcpm,
        base_voice: String::new(),
        source_colour: Vec::new(),
        target_colour: Vec::new(),
        base_similarity: 0.0,
        natural: Some(Natural {
            model: model.to_string(),
            clip: clip_name,
            transcript,
            mode,
            cfg: req.cfg,
            timesteps: req.timesteps,
            denoise_db: if mode == PromptMode::Continuation { CONTINUATION_DENOISE_DB } else { 0.0 },
            from: sample.from,
            to: sample.to,
            gender,
        }),
        reference_seconds: clip.len() as f64 / f64::from(encode_rate),
        source: source.chars().take(500).collect(),
        created: now(),
        consent: true,
        revision: 0,
    };
    profile.check()?;
    save(voices, &profile)?;
    progress(&format!("saved {:.1} s of speech from {:.1}–{:.1} s", profile.reference_seconds, sample.from, sample.to));
    Ok(profile)
}

/// The rate a release's audio encoder takes (from the pinned configs).
fn natural_encode_rate(model: &str) -> u32 {
    if model == "voxcpm-2" { 16_000 } else { 44_100 }
}

/// A timbre-only cloned voice ready to speak.
pub struct CloneVoice {
    profile: Profile,
    base: neural::NeuralVoice,
    converter: Arc<Converter>,
}

/// Load the cloned voice `id` (`clone:<slug>`), whichever engine made it.
pub fn load(id: &str, models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
    let profile = load_profile(&voices_dir(models_dir), id)?;
    match profile.engine {
        Engine::Openvoice => openvoice_voice(profile, models_dir).map(|v| Box::new(v) as Box<dyn Voice>),
        Engine::Voxcpm => natural::load(profile, models_dir),
    }
}

/// Load the timbre-only cloned voice `id`.
pub fn load_openvoice(id: &str, models_dir: &Path) -> Result<CloneVoice, TtsError> {
    let profile = load_profile(&voices_dir(models_dir), id)?;
    if profile.engine != Engine::Openvoice {
        return Err(TtsError::Invalid(format!("{id} is not a timbre-only clone")));
    }
    openvoice_voice(profile, models_dir)
}

fn openvoice_voice(profile: Profile, models_dir: &Path) -> Result<CloneVoice, TtsError> {
    let base = neural::load(&profile.base_voice, models_dir)?;
    let converter = converter(models_dir)?;
    Ok(CloneVoice { profile, base, converter })
}

impl CloneVoice {
    pub fn profile(&self) -> &Profile {
        &self.profile
    }
}

impl Voice for CloneVoice {
    fn info(&self) -> VoiceInfo {
        let base = self.base.info();
        VoiceInfo {
            id: "clone",
            name: "Cloned voice",
            language: base.language,
            gender: base.gender,
            engine: "clone",
            description: "A cloned voice: Kokoro speech re-voiced with OpenVoice V2",
            license: "Kokoro-82M Apache-2.0; OpenVoice V2 MIT",
            installed: true,
        }
    }

    fn synthesize(&self, text: &str, params: &Params) -> Result<Audio, TtsError> {
        let a = self.base.synthesize(text, params)?;
        let x = filmcraft_audio_dsp::resample::resample_sinc(&a.samples, a.sample_rate, OV_RATE);
        let y = self.converter.convert(&x, &self.profile.source_colour, &self.profile.target_colour, DEFAULT_TAU, 1)?;
        let mut out = filmcraft_audio_dsp::resample::resample_sinc(&y, OV_RATE, KOKORO_RATE);
        let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak > 0.98 {
            let g = 0.98 / peak;
            out.iter_mut().for_each(|v| *v *= g);
        }
        Ok(Audio { samples: out, sample_rate: KOKORO_RATE })
    }
}

/// The Malayalam (IndicF5) reference of a natural clone: a 10–15 s clip of the speaker and its
/// words, which IndicF5 needs. Kept next to the profile (`<name>.indicf5.json` and the clip,
/// `<name>.indicf5.wav`, mono 24 kHz) and made on first use ([`prepare_cross_lingual`]: Whisper
/// hears the words) or with words the user gives. The clip comes from the best source there is:
/// a recording given for it, else the voice sample's full-band copy (kept since clones store
/// one), else the voice sample itself (16 kHz for VoxCPM2: IndicF5 then speaks band-limited and
/// measurably less like the speaker, see the crate README). It belongs to one voice sample: a
/// voice made again needs a new one. The clip's language does not matter (an English recording
/// speaks Malayalam: cross-lingual prompting).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossReference {
    pub version: u32,
    /// SHA-256 of the voice sample of the profile it belongs to.
    pub sample: String,
    /// Where the clip came from (for the user's records).
    #[serde(default)]
    pub source: String,
    /// The clip's file next to the profile, and its SHA-256.
    pub clip: String,
    pub clip_sha: String,
    /// Where in the source the clip was taken (seconds).
    pub from: f64,
    pub to: f64,
    pub transcript: String,
    /// The user gave the words (else the speech model heard them).
    #[serde(default)]
    pub given: bool,
}

/// Version of [`CrossReference`].
pub const CROSS_REFERENCE_VERSION: u32 = 1;
/// The reference clip's length: the reference implementation clips to 15 s, and a longer clip
/// carries the voice better (measured: speaker similarity 0.48 with 7.4 s, 0.62 with 15.7 s)
/// but is generated again with each utterance (it sets the cost).
pub const CROSS_REFERENCE_SECONDS: (f64, f64) = (10.0, 15.0);
/// The rate the reference clip is kept at (IndicF5's).
pub const CROSS_REFERENCE_RATE: u32 = 24_000;

fn cross_path(voices: &Path, p: &Profile) -> Result<PathBuf, TtsError> {
    Ok(profile_path(voices, &p.id)?.with_extension("indicf5.json"))
}

/// A natural clone's full-band copy of its voice sample (`<name>.full.wav`, at the rate the
/// recording was read at), kept for IndicF5.
pub fn full_sample_path(voices: &Path, p: &Profile) -> Result<PathBuf, TtsError> {
    Ok(profile_path(voices, &p.id)?.with_extension("full.wav"))
}

fn sha_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The bytes of a file next to a profile, capped.
fn read_capped(path: &Path) -> Result<Vec<u8>, TtsError> {
    let meta = std::fs::metadata(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    if meta.len() > crate::wav::MAX_BYTES {
        return Err(TtsError::Invalid(format!("{}: too large for a voice sample", path.display())));
    }
    std::fs::read(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))
}

/// SHA-256 (hex) of a natural clone's voice sample file.
fn sample_hash(voices: &Path, p: &Profile) -> Result<String, TtsError> {
    let path = sample_path(voices, p).ok_or_else(|| TtsError::Invalid(format!("{} has no voice sample (a natural clone has one)", p.id)))?;
    Ok(sha_hex(&read_capped(&path)?))
}

/// The best recording of a natural clone kept with it: the voice sample's full-band copy, else
/// the sample. Mono, its rate, and what it is.
fn cross_source(voices: &Path, p: &Profile) -> Result<(Vec<f32>, u32, String), TtsError> {
    let n = p
        .natural
        .as_ref()
        .ok_or_else(|| TtsError::Invalid(format!("{} is a timbre-only clone; Malayalam needs a natural clone (its voice sample is the reference)", p.id)))?;
    let full = full_sample_path(voices, p)?;
    if full.is_file() {
        let (x, rate) = crate::wav::read(&full)?;
        return Ok((x, rate, format!("the voice sample, full band ({rate} Hz)")));
    }
    let (x, rate) = crate::wav::read(&voices.join(&n.clip))?;
    Ok((x, rate, format!("the voice sample ({rate} Hz)")))
}

/// The Malayalam reference of `id`, when one was made for its current voice sample and its clip
/// is intact.
pub fn cross_reference(models_dir: &Path, id: &str) -> Result<Option<CrossReference>, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let path = cross_path(&voices, &p)?;
    fresh_reference(&voices, &p, &path, &format!("{}.indicf5.wav", slug(id.trim_start_matches(CLONE_PREFIX))))
}

/// A Malayalam reference read from `path` when it was made for `p`'s current voice sample and its
/// clip is intact: the clip is `clip` (a file name of this profile's) with the bytes it had.
fn fresh_reference(voices: &Path, p: &Profile, path: &Path, clip: &str) -> Result<Option<CrossReference>, TtsError> {
    let Some(r) = read_reference(path) else { return Ok(None) };
    let clip_ok = r.clip == clip && read_capped(&voices.join(&r.clip)).is_ok_and(|b| sha_hex(&b) == r.clip_sha);
    let fresh = r.version == CROSS_REFERENCE_VERSION
        && clip_ok
        && r.sample == sample_hash(voices, p)?
        && r.transcript.chars().any(char::is_alphanumeric)
        && r.transcript.chars().count() <= MAX_TRANSCRIPT_CHARS
        && r.from.is_finite()
        && r.to.is_finite();
    Ok(fresh.then_some(r))
}

/// The reference file at `path` as it is on disk, fresh or not (`None`: missing, too large or
/// not one).
fn read_reference(path: &Path) -> Option<CrossReference> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_PROFILE_BYTES {
        return None;
    }
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Correct the words of `id`'s Malayalam reference (the clip stays).
pub fn set_cross_transcript(models_dir: &Path, id: &str, transcript: &str) -> Result<CrossReference, TtsError> {
    let mut r = cross_reference(models_dir, id)?.ok_or_else(|| TtsError::Invalid(format!("{id} has no Malayalam reference yet")))?;
    let text: String = transcript.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TRANSCRIPT_CHARS).collect();
    if !text.chars().any(char::is_alphanumeric) {
        return Err(TtsError::Invalid("give the reference clip's words".into()));
    }
    r.transcript = text;
    r.given = true;
    let voices = voices_dir(models_dir);
    let path = cross_path(&voices, &load_profile(&voices, id)?)?;
    let bytes = serde_json::to_vec_pretty(&r).map_err(|e| TtsError::Model(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| TtsError::Model(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    Ok(r)
}

/// The shortest Malayalam reference a cut at a sentence end leaves ([`trim_cross_reference`]):
/// a clip that ends where a sentence ends carries the voice from 8 s (by ear: the author's 14.6 s
/// clip ended mid-sentence and garbled the first words of every line; the 9.3 s cut of it that
/// ended at a full stop was preferred).
pub const SENTENCE_REFERENCE_MIN_SECONDS: f64 = 8.0;

/// How a Malayalam reference is cut ([`trim_cross_reference`]). Times are seconds into the
/// current clip.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trim {
    /// Keep from here.
    pub start: Option<f64>,
    /// Keep up to here (with `to_sentence`: the cut is at or before it).
    pub end: Option<f64>,
    /// End at the last sentence end the transcript implies that leaves at least
    /// [`SENTENCE_REFERENCE_MIN_SECONDS`]: at the quietest point near where the transcript's
    /// words put it.
    pub to_sentence: bool,
    /// The words of what is kept; by default the transcript up to the sentence end chosen
    /// (needed when `start` or a plain `end` cut words away).
    pub transcript: Option<String>,
}

/// Energy frames of the sentence search, seconds.
const TRIM_FRAME: f64 = 0.02;
/// How far from where the words put a sentence end its pause is looked for: this share of the
/// speech, at least [`TRIM_MIN_WINDOW`] seconds.
const TRIM_WINDOW: f64 = 0.08;
const TRIM_MIN_WINDOW: f64 = 0.8;
/// Fades at a cut, seconds (no click).
const TRIM_FADE: f64 = 0.012;

/// Per-frame level of `x` in dB (frames of [`TRIM_FRAME`]), smoothed over three frames.
fn frame_levels(x: &[f32], rate: u32) -> Vec<f64> {
    let n = ((f64::from(rate) * TRIM_FRAME) as usize).max(1);
    let raw: Vec<f64> = x
        .chunks(n)
        .map(|c| {
            let e = c.iter().map(|v| f64::from(*v) * f64::from(*v)).filter(|v| v.is_finite()).sum::<f64>() / c.len().max(1) as f64;
            10.0 * (e + 1e-12).log10()
        })
        .collect();
    (0..raw.len())
        .map(|i| {
            let w = raw.get(i.saturating_sub(1)..(i + 2).min(raw.len())).unwrap_or(&[]);
            w.iter().sum::<f64>() / w.len().max(1) as f64
        })
        .collect()
}

/// The sentence ends of `text`: the character index just past each `.`, `?`, `!` or `।` that
/// ends a sentence (followed by a space or the end; "..." counts once).
fn sentence_ends(text: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        let next = chars.get(i + 1);
        if matches!(c, '.' | '?' | '!' | '।') && next.is_none_or(|n| n.is_whitespace()) {
            out.push(i + 1);
        }
    }
    out
}

/// Where to cut `x` (the current clip) at a sentence end of `transcript` (its words): the
/// sample, and how many characters of the transcript are kept. The cut is at the quietest
/// point near where the words put the latest sentence end in `(lo + min, hi]` seconds (`only`:
/// that sentence end, a character index).
fn sentence_cut(x: &[f32], rate: u32, transcript: &str, lo: f64, hi: f64, min: f64, only: Option<usize>) -> Result<(usize, usize), TtsError> {
    let levels = frame_levels(x, rate);
    let peak = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() || peak < -80.0 {
        return Err(TtsError::Invalid("the reference clip is silent".into()));
    }
    // the speech: from the first frame within 35 dB of the loudest to the last
    let speech = |l: &f64| *l > peak - 35.0;
    let first = levels.iter().position(speech).unwrap_or(0);
    let last = levels.iter().rposition(speech).unwrap_or(levels.len().saturating_sub(1));
    let (t_a, t_b) = (first as f64 * TRIM_FRAME, (last + 1) as f64 * TRIM_FRAME);
    let span = (t_b - t_a).max(TRIM_FRAME);
    let total = crate::malayalam::syllables(transcript).max(1) as f64;
    let window = (span * TRIM_WINDOW).max(TRIM_MIN_WINDOW);
    let chars: Vec<char> = transcript.chars().collect();
    let ends = sentence_ends(transcript);
    // the latest sentence end whose pause leaves enough before it
    for &k in ends.iter().rev().filter(|k| only.is_none_or(|o| o == **k)) {
        let said: String = chars.get(..k).unwrap_or(&[]).iter().collect();
        let at = t_a + crate::malayalam::syllables(&said) as f64 / total * span;
        // a sentence the kept clip does not reach is not its end
        if at > hi + 0.25 {
            continue;
        }
        let (from, to) = ((at - window).max(lo + min), (at + window).min(hi));
        if from >= to {
            continue;
        }
        // the quietest frame there, nearer ones preferred among equals (0.5 dB per 100 ms)
        let (f0, f1) = ((from / TRIM_FRAME).ceil() as usize, ((to / TRIM_FRAME).floor() as usize).min(levels.len().saturating_sub(1)));
        let best =
            (f0..=f1).filter_map(|f| levels.get(f).map(|l| (f, l + 5.0 * ((f as f64 + 0.5) * TRIM_FRAME - at).abs()))).min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((f, _)) = best {
            let cut = (((f as f64 + 0.5) * TRIM_FRAME) * f64::from(rate)).round() as usize;
            return Ok((cut.min(x.len()), k));
        }
    }
    Err(TtsError::Invalid(format!(
        "no sentence end of the transcript leaves at least {min:.0} s of the clip: give `end` and the words up to it (`transcript`)"
    )))
}

/// Fade `clip` out over its last [`TRIM_FADE`] (and in over its first, `fade_in`): no click at a
/// cut.
fn fade_edges(clip: &mut [f32], rate: u32, fade_in: bool) {
    let fade = ((TRIM_FADE * f64::from(rate)) as usize).min(clip.len() / 4);
    for i in 0..fade {
        let g = i as f32 / fade.max(1) as f32;
        if fade_in && let Some(v) = clip.get_mut(i) {
            *v *= g;
        }
        let n = clip.len();
        if let Some(v) = clip.get_mut(n.saturating_sub(1 + i)) {
            *v *= g;
        }
    }
}

/// The clip before its first trim, kept beside it (`<name>.indicf5.untrimmed.wav` and `.json`).
fn untrimmed_paths(voices: &Path, p: &Profile) -> Result<(PathBuf, PathBuf), TtsError> {
    let base = profile_path(voices, &p.id)?;
    Ok((base.with_extension("indicf5.untrimmed.wav"), base.with_extension("indicf5.untrimmed.json")))
}

/// Cut `id`'s Malayalam reference ([`Trim`]): from `start`, to `end`, or to the last sentence end
/// its transcript implies (`to_sentence`) — IndicF5 continues the clip, so a clip that stops
/// mid-sentence garbles the first words of every line. The clip is written again (faded in and
/// out over a few milliseconds), its words become `transcript` (or the transcript up to the
/// sentence end chosen), and where it lies in its source is updated. The clip before the first
/// trim is kept beside it (`<name>.indicf5.untrimmed.wav`). The kept clip must be at least
/// [`SENTENCE_REFERENCE_MIN_SECONDS`] long.
pub fn trim_cross_reference(models_dir: &Path, id: &str, trim: &Trim) -> Result<CrossReference, TtsError> {
    let mut r =
        cross_reference(models_dir, id)?.ok_or_else(|| TtsError::Invalid(format!("{id} has no Malayalam reference yet: make it with tts.voiceReference")))?;
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let (x, rate) = crate::wav::read(&voices.join(&r.clip))?;
    if rate == 0 || x.is_empty() {
        return Err(TtsError::Invalid("the reference clip is empty".into()));
    }
    let dur = x.len() as f64 / f64::from(rate);
    let check = |v: Option<f64>, what: &str| -> Result<Option<f64>, TtsError> {
        match v {
            Some(t) if !t.is_finite() || t < 0.0 || t > dur + 0.05 => Err(TtsError::Invalid(format!("`{what}` must be a time within the {dur:.2} s clip"))),
            other => Ok(other.map(|t| t.min(dur))),
        }
    };
    let start = check(trim.start, "start")?.unwrap_or(0.0);
    let end = check(trim.end, "end")?.unwrap_or(dur);
    if start >= end {
        return Err(TtsError::Invalid("`start` must come before `end`".into()));
    }
    let given = trim.transcript.as_deref().map(|t| t.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|t| !t.is_empty());
    if trim.transcript.is_some() && given.is_none() {
        return Err(TtsError::Invalid("give the kept clip's words".into()));
    }
    let s0 = ((start * f64::from(rate)).round() as usize).min(x.len());
    let (s1, words) = if trim.to_sentence {
        // words given that begin the transcript and end a sentence: cut at that sentence's end
        let only = given.as_ref().filter(|g| r.transcript.starts_with(g.as_str()) && g.ends_with(['.', '?', '!', '।'])).map(|g| g.chars().count());
        let (cut, k) = sentence_cut(&x, rate, &r.transcript, start, end, SENTENCE_REFERENCE_MIN_SECONDS, only)?;
        let derived: String = r.transcript.chars().take(k).collect::<String>().trim().to_string();
        match (&given, start > 0.0) {
            (Some(g), _) => (cut, g.clone()),
            // words cut away at the start too: they must be given
            (None, true) => return Err(TtsError::Invalid("`start` cuts words away: give the kept clip's words (`transcript`)".into())),
            (None, false) => (cut, derived),
        }
    } else {
        if trim.start.is_none() && trim.end.is_none() {
            return Err(TtsError::Invalid("nothing to trim: give `start`, `end` or `trimToSentence`".into()));
        }
        let words = given.clone().ok_or_else(|| TtsError::Invalid("the cut takes words away: give the kept clip's words (`transcript`)".into()))?;
        (((end * f64::from(rate)).round() as usize).min(x.len()), words)
    };
    let kept = s1.saturating_sub(s0) as f64 / f64::from(rate);
    if kept + 1e-6 < SENTENCE_REFERENCE_MIN_SECONDS {
        return Err(TtsError::Invalid(format!("that keeps {kept:.1} s; a Malayalam reference needs at least {SENTENCE_REFERENCE_MIN_SECONDS:.0} s")));
    }
    if !words.chars().any(char::is_alphanumeric) || words.chars().count() > MAX_TRANSCRIPT_CHARS {
        return Err(TtsError::Invalid("give the kept clip's words".into()));
    }
    let mut clip: Vec<f32> = x.get(s0..s1).unwrap_or(&[]).to_vec();
    fade_edges(&mut clip, rate, s0 > 0);
    // the clip before the first trim stays beside it
    let (keep_wav, keep_json) = untrimmed_paths(&voices, &p)?;
    if !keep_wav.exists() {
        std::fs::copy(voices.join(&r.clip), &keep_wav).map_err(|e| TtsError::Model(format!("{}: {e}", keep_wav.display())))?;
        let bytes = serde_json::to_vec_pretty(&r).map_err(|e| TtsError::Model(e.to_string()))?;
        std::fs::write(&keep_json, bytes).map_err(|e| TtsError::Model(format!("{}: {e}", keep_json.display())))?;
    }
    let path = voices.join(&r.clip);
    let tmp = path.with_extension("wav.part");
    crate::wav::write(&tmp, &clip, rate)?;
    std::fs::rename(&tmp, &path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    r.clip_sha = sha_hex(&read_capped(&path)?);
    let from = r.from;
    r.from = from + start;
    r.to = from + s1 as f64 / f64::from(rate);
    r.given = r.given || given.is_some();
    r.transcript = words;
    let json_path = cross_path(&voices, &p)?;
    let bytes = serde_json::to_vec_pretty(&r).map_err(|e| TtsError::Model(e.to_string()))?;
    let tmp = json_path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| TtsError::Model(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &json_path).map_err(|e| TtsError::Model(format!("{}: {e}", json_path.display())))?;
    Ok(r)
}

/// The reference clip of `r` (mono, [`CROSS_REFERENCE_RATE`]).
pub fn cross_audio(models_dir: &Path, r: &CrossReference) -> Result<Vec<f32>, TtsError> {
    let (x, rate) = crate::wav::read(&voices_dir(models_dir).join(&r.clip))?;
    Ok(filmcraft_audio_dsp::resample::resample_sinc(&x, rate, CROSS_REFERENCE_RATE))
}

/// Where the Malayalam reference of a voice comes from.
pub struct CrossSource<'a> {
    /// Mono speech of the speaker.
    pub audio: &'a [f32],
    pub rate: u32,
    /// What it is (file names, clip ids), for the user's records.
    pub source: &'a str,
}

/// Make the Malayalam reference of `id`: the best 10–15 s between pauses of `source` (else of
/// the recording kept with the voice, [`cross_source`]), kept at 24 kHz, and its words —
/// `transcript` when given, else what `transcribe` (mono 16 kHz → text) hears.
pub fn prepare_cross_lingual(
    models_dir: &Path,
    id: &str,
    source: Option<CrossSource>,
    transcript: Option<&str>,
    transcribe: &mut dyn FnMut(&[f32]) -> Result<String, String>,
) -> Result<CrossReference, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let sample = sample_hash(&voices, &p)?;
    let (x, rate, what) = match source {
        Some(s) => (s.audio.to_vec(), s.rate, s.source.chars().take(500).collect()),
        None => cross_source(&voices, &p)?,
    };
    let s = pick_sample_within(&x, rate, CROSS_REFERENCE_SECONDS)?;
    let (text, given) = match transcript.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => (t.to_string(), true),
        None => {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(&s.audio, rate, 16_000);
            let t = transcribe(&x16).map_err(|e| TtsError::Invalid(format!("the reference clip could not be transcribed ({e}); give its words")))?;
            (t, false)
        }
    };
    let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TRANSCRIPT_CHARS).collect();
    if !text.chars().any(char::is_alphanumeric) {
        return Err(TtsError::Invalid("no words were heard in the reference clip; give its words".into()));
    }
    let clip = format!("{}.indicf5.wav", slug(id.trim_start_matches(CLONE_PREFIX)));
    let audio = filmcraft_audio_dsp::resample::resample_sinc(&s.audio, rate, CROSS_REFERENCE_RATE);
    crate::wav::write(&voices.join(&clip), &audio, CROSS_REFERENCE_RATE)?;
    let clip_sha = sha_hex(&read_capped(&voices.join(&clip))?);
    let r = CrossReference { version: CROSS_REFERENCE_VERSION, sample, source: what, clip, clip_sha, from: s.from, to: s.to, transcript: text, given };
    let path = cross_path(&voices, &p)?;
    let bytes = serde_json::to_vec_pretty(&r).map_err(|e| TtsError::Model(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| TtsError::Model(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    Ok(r)
}

/// The cloned voice `id` speaking Malayalam (IndicF5, feature `indicf5`); its Malayalam
/// reference must have been made ([`prepare_cross_lingual`]).
pub fn load_malayalam(id: &str, models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
    let profile = load_profile(&voices_dir(models_dir), id)?;
    malayalam_voice::load(profile, models_dir)
}

/// A natural voice speaking English from `clip` (mono speech at `rate`) as its isolated
/// reference, without a saved profile: how the voice lab auditions a candidate reference
/// (VoxCPM2's reference mode; give it the sample a promotion would make, [`sample_from`]).
pub fn audition_voice(models_dir: &Path, model: &str, clip: &[f32], rate: u32, cfg: f32, steps: usize) -> Result<Box<dyn Voice>, TtsError> {
    if catalog::voxcpm(model).is_none() || model == "voxcpm-1.5" {
        return Err(TtsError::Invalid(format!("`{model}`: references are auditioned with VoxCPM2 (voxcpm-2)")));
    }
    if rate == 0 || clip.iter().any(|v| !v.is_finite()) || (clip.len() as f64) < MIN_SPEECH_SECONDS * f64::from(rate) {
        return Err(TtsError::Invalid("the reference to audition is empty or damaged".into()));
    }
    if !(cfg.is_finite() && (1.0..=5.0).contains(&cfg)) || !(2..=50).contains(&steps) {
        return Err(TtsError::Invalid("guidance must be 1–5 and steps 2–50".into()));
    }
    natural::from_clip(models_dir, model, clip, rate, cfg, steps)
}

/// The cloned voice speaking Malayalam (IndicF5) from `clip` (mono at `rate`) and its exact
/// words, without a saved reference: how the voice lab auditions a Malayalam reference.
pub fn audition_malayalam(models_dir: &Path, clip: &[f32], rate: u32, transcript: &str, gender: Gender) -> Result<Box<dyn Voice>, TtsError> {
    if rate == 0 || clip.iter().any(|v| !v.is_finite()) || (clip.len() as f64) < 2.0 * f64::from(rate) {
        return Err(TtsError::Invalid("the reference to audition is empty or damaged".into()));
    }
    let words: String = transcript.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TRANSCRIPT_CHARS).collect();
    if !words.chars().any(char::is_alphanumeric) {
        return Err(TtsError::Invalid("a Malayalam reference needs the clip's words".into()));
    }
    malayalam_voice::from_clip(models_dir, clip, rate, &words, gender)
}

/// How a cloned voice reads Malayalam ([`load_malayalam`] reads with [`Reading::default`], the
/// measured best: the crate README's "Sentence starts"). Each utterance is generated after the
/// reference clip, so how the clip ends and how each utterance begins decide whether its first
/// syllables come out clear.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// Silence after the reference clip, ms (the reference implementation adds 50 ms).
    pub tail_ms: u32,
    /// Fill that silence with the clip's own room tone (its quietest stretch) instead of
    /// digital silence.
    pub room_tone: bool,
    /// Each sentence is an utterance of its own (else sentences are packed into as few
    /// utterances as one generation holds).
    pub per_sentence: bool,
    /// Euler steps, classifier-free guidance and sway of the sampler ([`crate::f5::Settings`]).
    pub steps: usize,
    pub cfg: f32,
    pub sway: f32,
    /// Words read in front of each utterance and cut away again (a neutral lead-in), if any.
    pub lead_in: Option<String>,
    /// The reference's words through the English front end before IndicF5 reads them (how the
    /// voice read before 2026-10-20; kept to measure against).
    pub english_transcript: bool,
    /// The pause between utterances follows how the first ends and the style
    /// ([`crate::styles::gap_ms`]); else 250 ms after every utterance, a pause marker too (how the
    /// voice read before 2026-10-28).
    pub natural_pauses: bool,
}

/// The neutral lead-in measured as `Reading::lead_in` ("ശരി." — "right."): IndicF5 runs it into
/// the sentence without a pause, so the cut leaves it heard ("ശരി ഇതാണ്…": 11 of 11 first
/// starts garbled, crate README). Not used by default or in retakes.
pub const LEAD_IN: &str = "ശരി.";

impl Default for Reading {
    fn default() -> Self {
        Reading {
            tail_ms: 50,
            room_tone: false,
            per_sentence: false,
            steps: 32,
            cfg: 2.0,
            sway: -1.0,
            lead_in: None,
            english_transcript: false,
            natural_pauses: true,
        }
    }
}

impl Reading {
    /// How the voice read before the sentence-start fix (2026-10-20): before/after comparisons.
    pub fn before_fix() -> Reading {
        Reading {
            tail_ms: 50,
            room_tone: false,
            per_sentence: false,
            steps: 32,
            cfg: 2.0,
            sway: -1.0,
            lead_in: None,
            english_transcript: true,
            natural_pauses: false,
        }
    }

    /// How the strict check's reading `attempt` of an utterance reads (each attempt also has its
    /// own seed): the first two as set, the later ones with twice the sampling steps (another
    /// path through the sampler; measured as good as 32 on average, at twice the time) — the
    /// sampler settings only, so every attempt speaks after the same reference.
    pub fn for_attempt(&self, attempt: u32) -> Reading {
        let mut r = self.clone();
        if attempt >= 2 {
            r.steps = (self.steps * 2).min(64);
        }
        r
    }

    /// What of the reading decides the reference IndicF5 is given (keys the built references).
    #[cfg(feature = "indicf5")]
    fn reference_key(&self) -> String {
        format!("tail {} room {} english {}", self.tail_ms, self.room_tone, self.english_transcript)
    }

    pub fn check(&self) -> Result<(), TtsError> {
        // the sampler's ranges are IndicF5's: without it no reading is ever voiced
        #[cfg(feature = "indicf5")]
        crate::f5::Settings { steps: self.steps, cfg: self.cfg, sway: self.sway, ..Default::default() }.check()?;
        if self.tail_ms > 2000 {
            return Err(TtsError::Invalid("the pause after the reference is at most 2 s".into()));
        }
        if self.lead_in.as_ref().is_some_and(|l| l.chars().count() > 80) {
            return Err(TtsError::Invalid("a lead-in is a few words".into()));
        }
        Ok(())
    }
}

/// The cloned voice `id` speaking Malayalam with `reading` (feature `indicf5`): the lab's way to
/// read single utterances ([`MalayalamVoice::read`]).
#[cfg(feature = "indicf5")]
pub fn malayalam_voice(id: &str, models_dir: &Path, reading: Reading) -> Result<MalayalamVoice, TtsError> {
    let profile = load_profile(&voices_dir(models_dir), id)?;
    malayalam_voice::load_with(profile, models_dir, reading)
}

#[cfg(feature = "indicf5")]
pub use malayalam_voice::MalayalamVoice;

/// What, beyond the voice's Malayalam identity ([`crate::cache::voice_identity_for`]), decides
/// how the cloned voice `id` reads `script` with `params` (the narration cache keys a reading on
/// it): the style references it reads after (their clips' SHA-256 and words), the styles' presets
/// version when a style is asked for, and natural pauses when it has more than one utterance or
/// a breath. Empty for a plain reading of one utterance, whose key is the one it always had.
pub fn malayalam_reading_identity(models_dir: &Path, id: &str, script: &str, params: &Params) -> Result<String, TtsError> {
    #[cfg(feature = "indicf5")]
    {
        malayalam_voice::reading_identity(models_dir, id, script, params)
    }
    #[cfg(not(feature = "indicf5"))]
    {
        let _ = (models_dir, id, script, params);
        Ok(String::new())
    }
}

/// Cloned voices speaking Malayalam through IndicF5 (feature `indicf5`).
#[cfg(feature = "indicf5")]
mod malayalam_voice {
    use super::*;
    use crate::f5::{self, F5, Reference};
    use crate::heard;
    use crate::malayalam::Part;
    use crate::styles::{self, Asked, Boundary, Sampler};

    /// Seeds between takes.
    const TAKE_SEEDS: u64 = 7919;
    /// Seeds between the strict check's readings of one utterance (apart from the takes' and the
    /// utterances' own).
    const ATTEMPT_SEEDS: u64 = 104_729;
    /// Where the pauses' variation is seeded from (apart from the readings' seeds).
    const GAP_SEEDS: u64 = 0x5eed_9a95;
    /// References kept built: a voice's main one and its styles, for a few readings.
    const REFERENCES: usize = 24;
    /// The room tone kept for `[breath]`, seconds of the clip's quietest stretch.
    const ROOM_SECONDS: f64 = 0.2;

    /// The speaking rate a Malayalam utterance is held within, syllables a second
    /// ([`crate::malayalam::syllables`]): Malayalam read aloud runs about 5–6. The reference's
    /// duration comes from the transcript's characters per second; across languages, and with a
    /// transcript the speech model got wrong, that can be far off (measured on the test lines:
    /// 4.0–4.2 after a 7.6 s clip, 7.3–7.5 when Whisper added a phrase that was never said).
    pub(crate) const SYLLABLE_RATE: (f64, f64) = (4.6, 6.0);

    /// The speed that brings `text`, at `seconds` by the reference's formula, within
    /// [`SYLLABLE_RATE`] (1 when it is).
    pub(crate) fn rate_correction(text: &str, seconds: f64) -> f64 {
        let syl = crate::malayalam::syllables(text) as f64;
        if syl < 4.0 || !(seconds.is_finite() && seconds > 0.05) {
            return 1.0;
        }
        let rate = syl / seconds;
        (rate.clamp(SYLLABLE_RATE.0, SYLLABLE_RATE.1) / rate).clamp(0.5, 2.0)
    }

    /// The release in `models_dir`, loaded once and shared.
    fn model(models_dir: &Path) -> Result<Arc<F5>, TtsError> {
        static M: OnceLock<Mutex<Option<(PathBuf, Arc<F5>)>>> = OnceLock::new();
        if !catalog::voxcpm_installed(models_dir, &catalog::INDICF5) {
            return Err(TtsError::NotInstalled);
        }
        let dir = catalog::voxcpm_dir(models_dir, &catalog::INDICF5);
        let mut g = M.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((d, m)) = g.as_ref()
            && *d == dir
        {
            return Ok(m.clone());
        }
        *g = None;
        let m = Arc::new(F5::load(&dir)?);
        *g = Some((dir, m.clone()));
        Ok(m)
    }

    /// The transcript as the English front end normalises it (numbers and symbols spelled), with
    /// dashes as commas (IndicF5's vocabulary has no dash).
    fn english(t: &str) -> Result<String, TtsError> {
        Ok(crate::voxcpm::text::sentences(t)?.join(" ").replace(" —", ",").replace(['—', '–'], ","))
    }

    /// The reference's words as IndicF5 reads them: written in Malayalam, through the Malayalam
    /// front end (canonical spelling, numbers spelled); else through the English one (numbers and
    /// symbols spelled). `english`: always the English front end, as the voice read before
    /// 2026-10-20 — which drops every virama from Malayalam ("ഫൈവ് പോയിന്റ് ത്രീ" became "ഫൈവ
    /// പോയിന റ ത രീ"), so the words no longer matched the clip.
    pub(crate) fn reference_words(transcript: &str, english_always: bool) -> Result<String, TtsError> {
        if english_always || !crate::malayalam::detect(transcript) {
            return english(transcript);
        }
        let n = crate::malayalam::normalize(transcript);
        Ok(n.text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// The quietest `secs` of `x` (20 ms frames): the clip's room tone.
    pub(crate) fn room_tone(x: &[f32], rate: u32, secs: f64) -> Vec<f32> {
        let n = ((f64::from(rate) * secs) as usize).max(1);
        if x.len() <= n {
            return vec![0.0; n];
        }
        let hop = ((rate / 50) as usize).max(1);
        let energy = |s: usize| x.get(s..s + n).map_or(f64::INFINITY, |w| w.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>());
        let best = (0..=(x.len() - n) / hop).map(|k| k * hop).min_by(|a, b| energy(*a).total_cmp(&energy(*b))).unwrap_or(0);
        x.get(best..best + n).map(<[f32]>::to_vec).unwrap_or_else(|| vec![0.0; n])
    }

    /// The pause after the clip: `ms` of room tone (faded in from the clip's end and out to
    /// silence) or of silence.
    pub(crate) fn tail(clip: &[f32], rate: u32, ms: u32, room: bool) -> Vec<f32> {
        let n = (u64::from(ms) * u64::from(rate) / 1000) as usize;
        if !room || n == 0 {
            return vec![0.0; n];
        }
        tile(&room_tone(clip, rate, ROOM_SECONDS), n, rate)
    }

    /// `n` samples of `tone` repeated, faded in and out over 10 ms (silence without a tone).
    pub(crate) fn tile(tone: &[f32], n: usize, rate: u32) -> Vec<f32> {
        let fade = ((rate / 100) as usize).max(1).min(n / 2).max(1);
        (0..n)
            .map(|i| {
                let v = tone.get(i % tone.len().max(1)).copied().unwrap_or(0.0);
                let g = if i < fade {
                    i as f32 / fade as f32
                } else if n - i <= fade {
                    (n - i) as f32 / fade as f32
                } else {
                    1.0
                };
                v * g
            })
            .collect()
    }

    /// The main reference's clip at IndicF5's rate.
    fn clip_audio(models_dir: &Path, r: &CrossReference) -> Result<Vec<f32>, TtsError> {
        Ok(filmcraft_audio_dsp::resample::resample_sinc(&cross_audio(models_dir, r)?, CROSS_REFERENCE_RATE, f5::RATE))
    }

    /// The reference IndicF5 speaks after, built once per profile, reference and reading.
    pub(crate) fn reference(p: &Profile, r: &CrossReference, m: &F5, models_dir: &Path, reading: &Reading) -> Result<Arc<Reference>, TtsError> {
        type Refs = Vec<(String, Arc<Reference>)>;
        static C: OnceLock<Mutex<Refs>> = OnceLock::new();
        let key = format!("{}|{}|{}|{}|{}|{}", p.id, p.created, r.sample, r.clip_sha, r.transcript, reading.reference_key());
        let cache = C.get_or_init(Default::default);
        if let Some(found) = cache.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()) {
            return Ok(found);
        }
        let mut x = clip_audio(models_dir, r)?;
        let pause = tail(&x, f5::RATE, reading.tail_ms, reading.room_tone);
        x.extend(pause);
        let words = reference_words(&r.transcript, reading.english_transcript)?;
        let built = Arc::new(m.reference(&x, &words)?);
        let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
        c.retain(|(k, _)| *k != key);
        c.push((key, built.clone()));
        let extra = c.len().saturating_sub(REFERENCES);
        c.drain(..extra);
        Ok(built)
    }

    pub(super) fn load(profile: Profile, models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
        Ok(Box::new(load_with(profile, models_dir, Reading::default())?))
    }

    /// A Malayalam voice from a clip and its words (the voice lab's audition), read as
    /// [`Reading::default`].
    pub(super) fn from_clip(models_dir: &Path, clip: &[f32], rate: u32, words: &str, gender: Gender) -> Result<Box<dyn Voice>, TtsError> {
        let reading = Reading::default();
        let m = model(models_dir)?;
        let mut x = filmcraft_audio_dsp::resample::resample_sinc(clip, rate, f5::RATE);
        let pause = tail(&x, f5::RATE, reading.tail_ms, reading.room_tone);
        x.extend(pause);
        let room = room_tone(&x, f5::RATE, ROOM_SECONDS);
        let reference = Arc::new(m.reference(&x, &reference_words(words, reading.english_transcript)?)?);
        Ok(Box::new(MalayalamVoice { gender, model: m, reference, reading, styles: Vec::new(), room }))
    }

    pub(super) fn load_with(profile: Profile, models_dir: &Path, reading: Reading) -> Result<MalayalamVoice, TtsError> {
        reading.check()?;
        let r = cross_reference(models_dir, &profile.id)?
            .ok_or_else(|| TtsError::Invalid(format!("{} has no Malayalam reference yet: make it with tts.voiceReference", profile.id)))?;
        let m = model(models_dir)?;
        let main = reference(&profile, &r, &m, models_dir, &reading)?;
        // its voice styles, each after its own clip
        let mut styles = Vec::new();
        for (style, sr) in style_references(models_dir, &profile.id)? {
            styles.push((style, reference(&profile, &sr, &m, models_dir, &reading)?));
        }
        let room = room_tone(&clip_audio(models_dir, &r)?, f5::RATE, ROOM_SECONDS);
        Ok(MalayalamVoice { gender: gender(&profile), model: m, reference: main, reading, styles, room })
    }

    /// The references a reading of `script` in `params` reads after, and how its utterances are
    /// spaced, as the narration cache keys them ([`super::malayalam_reading_identity`]): empty for
    /// a plain reading of one utterance (the key it always had). Planned as [`MalayalamVoice`]
    /// plans it, without the model.
    pub(super) fn reading_identity(models_dir: &Path, id: &str, script: &str, params: &Params) -> Result<String, TtsError> {
        params.check()?;
        let reading = Reading::default();
        let Some(main) = cross_reference(models_dir, id)? else { return Ok(String::new()) };
        let styles = style_references(models_dir, id)?;
        // what one generation after each reference holds (as `Reference::max_chars`)
        let capacity = |r: &CrossReference| -> Result<usize, TtsError> {
            let samples = clip_audio(models_dir, r)?.len() + (u64::from(reading.tail_ms) * u64::from(f5::RATE) / 1000) as usize;
            let text = f5::model::reference_text(&reference_words(&r.transcript, reading.english_transcript)?);
            Ok(f5::model::max_chars_for(f5::model::reference_seconds(samples), &text))
        };
        let main_max = capacity(&main)?;
        let mut maxes = Vec::new();
        for (s, r) in &styles {
            maxes.push((*s, capacity(r)?));
        }
        let has = |s: &str| styles.iter().any(|(k, _)| *k == s);
        let line = params.delivery.and_then(styles::find);
        let max = |a: Asked| styles::resolve(a.style(line), &has).and_then(|s| maxes.iter().find(|(k, _)| *k == s)).map_or(main_max, |(_, m)| *m);
        let (parts, _) = crate::malayalam::parts(script, &max, crate::malayalam::Numbers::default(), !reading.per_sentence)?;
        let says = parts.iter().filter(|p| matches!(p, Part::Say { .. })).count();
        let mut used: Vec<&'static str> = Vec::new();
        let mut styled = false;
        for p in &parts {
            if let Part::Say { asked, .. } = p {
                let style = asked.style(line);
                styled |= style.is_some();
                if let Some(s) = styles::resolve(style, &has)
                    && !used.contains(&s)
                {
                    used.push(s);
                }
            }
        }
        let mut id = Vec::new();
        if styled {
            id.push(format!("styles v{}", styles::VERSION));
        }
        for s in used {
            if let Some((_, r)) = styles.iter().find(|(k, _)| *k == s) {
                use sha2::Digest;
                let words: String = sha2::Sha256::digest(r.transcript.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect();
                id.push(format!("{s} {} {words}", r.clip_sha));
            }
        }
        // more than one utterance (or a breath): spaced by natural pauses
        if says > 1 || parts.iter().any(|p| matches!(p, Part::Breath)) {
            id.push(format!("pauses v{}", styles::VERSION));
        }
        Ok(id.join("; "))
    }

    /// A cloned voice speaking Malayalam.
    pub struct MalayalamVoice {
        gender: Gender,
        model: Arc<F5>,
        /// The main (plain) reference.
        reference: Arc<Reference>,
        reading: Reading,
        /// The voice's style references ([`crate::styles`]), by style.
        styles: Vec<(&'static str, Arc<Reference>)>,
        /// The main clip's room tone, for `[breath]` (empty: silence).
        room: Vec<f32>,
    }

    /// Frames of the lead-in cut, seconds.
    const CUT_FRAME: f64 = 0.02;

    /// Where the lead-in ends in `a` (a reading of `lead` then `text`): the end of the longest
    /// pause near where the syllables put the boundary.
    pub(crate) fn lead_in_cut(a: &[f32], rate: u32, lead: &str, text: &str) -> usize {
        let (sl, st) = (crate::malayalam::syllables(lead) as f64, crate::malayalam::syllables(text) as f64);
        let n = ((f64::from(rate) * CUT_FRAME) as usize).max(1);
        let levels: Vec<f64> =
            a.chunks(n).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / c.len().max(1) as f64 + 1e-12).log10()).collect();
        let at = (levels.len() as f64 * sl / (sl + st).max(1.0)) as usize;
        let mut sorted = levels.clone();
        sorted.sort_by(f64::total_cmp);
        let loud = sorted.get(sorted.len() * 95 / 100).copied().unwrap_or(-120.0);
        let quiet = |l: f64| l < loud - 30.0;
        let (lo, hi) = (at / 2, (at * 2).min(levels.len()));
        // the longest quiet run starting in [lo, hi)
        let mut best: Option<(usize, usize)> = None;
        let mut f = lo;
        while f < hi {
            if levels.get(f).is_some_and(|l| quiet(*l)) {
                let start = f;
                while levels.get(f).is_some_and(|l| quiet(*l)) {
                    f += 1;
                }
                if best.is_none_or(|(_, len)| f - start > len) {
                    best = Some((start, f - start));
                }
            } else {
                f += 1;
            }
        }
        let frame = match best {
            // the end of the pause, a frame before the speech
            Some((start, len)) => start + len.saturating_sub(1),
            None => at,
        };
        (frame * n).min(a.len())
    }

    impl MalayalamVoice {
        pub fn reading(&self) -> &Reading {
            &self.reading
        }

        /// The reference's words as IndicF5 is given them.
        pub fn reference_text(&self) -> &str {
            self.reference.text()
        }

        /// The reference's length with its pause, seconds.
        pub fn reference_seconds(&self) -> f64 {
            self.reference.seconds()
        }

        /// The longest utterance one generation takes, characters.
        pub fn max_chars(&self) -> usize {
            self.reference.max_chars()
        }

        /// The seed of reading `attempt` of utterance `index` of take `take`: every utterance and
        /// attempt has its own, so one can be read again alone and reproduced.
        pub fn seed(take: u32, index: usize, attempt: u32) -> u64 {
            1u64.wrapping_add(u64::from(take).wrapping_mul(TAKE_SEEDS)).wrapping_add(index as u64).wrapping_add(u64::from(attempt).wrapping_mul(ATTEMPT_SEEDS))
        }

        /// The utterances and pauses `script` is read as.
        pub fn pieces(&self, script: &str) -> Result<Vec<crate::malayalam::Piece>, TtsError> {
            let max = self.reference.max_chars();
            let (pieces, _) = if self.reading.per_sentence { crate::malayalam::sentence_pieces(script, max)? } else { crate::malayalam::pieces(script, max)? };
            Ok(pieces)
        }

        /// One utterance read with `seed` at `pace`, its silence trimmed.
        pub fn read(&self, text: &str, pace: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
            self.read_as(&self.reading, text, pace, seed)
        }

        /// [`MalayalamVoice::read`] with the sampler settings and lead-in of `reading` (its
        /// reference settings are the voice's own).
        pub fn read_as(&self, reading: &Reading, text: &str, pace: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
            self.read_take(reading, &self.reference, None, text, pace, seed)
        }

        /// The styles this voice has a reference for.
        pub fn styles(&self) -> Vec<&'static str> {
            self.styles.iter().map(|(s, _)| *s).collect()
        }

        /// The reference `style` reads after ([`styles::resolve`]): its own, its nearest
        /// fallback's, else the main one (`None`).
        fn reference_for(&self, style: Option<&str>) -> (Arc<Reference>, Option<&'static str>) {
            let has = |s: &str| self.styles.iter().any(|(k, _)| *k == s);
            match styles::resolve(style, &has).and_then(|s| self.styles.iter().find(|(k, _)| *k == s)) {
                Some((k, r)) => (r.clone(), Some(*k)),
                None => (self.reference.clone(), None),
            }
        }

        /// How `script` is read in the line's style `line`: utterances (each with the style its
        /// tags put in force, packed to what its reference holds), pauses, breaths, line breaks.
        pub fn parts(&self, script: &str, line: Option<&'static str>) -> Result<Vec<Part>, TtsError> {
            let max = |a: Asked| self.reference_for(a.style(line)).0.max_chars();
            Ok(crate::malayalam::parts(script, &max, crate::malayalam::Numbers::default(), !self.reading.per_sentence)?.0)
        }

        /// One utterance read after `reference` with `reading`'s steps, sway and lead-in, and the
        /// sampler `sampler` asks for (`None`: `reading`'s guidance and the model's own noise;
        /// exactly as before styles), with `seed` at `pace`, its silence trimmed.
        fn read_take(
            &self,
            reading: &Reading,
            reference: &Reference,
            sampler: Option<Sampler>,
            text: &str,
            pace: f64,
            seed: u64,
        ) -> Result<Vec<f32>, TtsError> {
            let rate = f5::RATE;
            let lead = reading.lead_in.as_deref().map(str::trim).filter(|l| !l.is_empty());
            let said = match lead {
                Some(l) => format!("{l} {text}"),
                None => text.to_string(),
            };
            // pace (and the style's speed) is IndicF5's own speed (the duration it is asked
            // for), held to a natural rate
            let pace = match sampler {
                Some(s) => pace * f64::from(s.speed),
                None => pace,
            };
            let speed = (pace * rate_correction(&said, self.model.natural_seconds(reference, &said))).clamp(0.25, 4.0);
            let (cfg, temperature) = sampler.map_or((reading.cfg, 1.0), |s| (s.emphasis, s.temperature));
            let s = f5::Settings { steps: reading.steps, cfg, sway: reading.sway, speed: speed as f32, seed, temperature };
            let a = self.model.speak(reference, &said, &s, &mut |_, _| true)?;
            let a = match lead {
                Some(l) => a.get(lead_in_cut(&a, rate, l, text)..).unwrap_or(&[]).to_vec(),
                None => a,
            };
            Ok(super::natural::trim_silence(&a, rate).to_vec())
        }

        /// The seed of the pause before part `index` of take `take`.
        fn gap_seed(take: u32, index: usize) -> u64 {
            Self::seed(take, index, 0) ^ GAP_SEEDS
        }

        /// `n` samples of the clip's room tone, faded in and out.
        fn breath(&self, n: usize) -> Vec<f32> {
            tile(&self.room, n, f5::RATE)
        }

        fn speak(&self, script: &str, params: &Params, strict: Option<&heard::Strict>) -> Result<(Audio, Vec<heard::Checked>), TtsError> {
            crate::check_text(script)?;
            params.check()?;
            let line = params.delivery.and_then(styles::find);
            let parts = self.parts(script, line)?;
            let rate = f5::RATE;
            let max = (crate::MAX_SECONDS * f64::from(rate)) as usize;
            let samples = |ms: u32| (u64::from(ms) * u64::from(rate) / 1000) as usize;
            let _gpu = super::natural::gpu("IndicF5 narration")?;
            let mut out: Vec<f32> = Vec::new();
            let mut checks = Vec::new();
            // how the last utterance ended and its style; a pause or breath since; a line break
            let mut last: Option<(Boundary, Option<&'static str>)> = None;
            let mut marked = false;
            let mut line_break: Option<Boundary> = None;
            for (i, part) in parts.into_iter().enumerate() {
                match part {
                    Part::Pause(ms) => {
                        out.resize(out.len().saturating_add(samples(ms)), 0.0);
                        marked = true;
                    }
                    Part::Breath => {
                        let style = last.map_or(line, |(_, s)| s);
                        out.extend(self.breath(samples(styles::breath_ms(style, Self::gap_seed(params.take, i)))));
                        marked = true;
                    }
                    Part::Break(b) => line_break = Some(b),
                    Part::Say { text, asked } => {
                        let style = asked.style(line);
                        if let Some((ended, before)) = last {
                            // the pause after the last utterance: as it ended (and the line breaks
                            // since), in its style; none after a pause marker or a breath
                            let gap = if !self.reading.natural_pauses {
                                Some(styles::FIXED_GAP_MS)
                            } else if marked {
                                None
                            } else {
                                Some(styles::gap_ms(line_break.unwrap_or(ended), before, Self::gap_seed(params.take, i)))
                            };
                            if let Some(ms) = gap {
                                out.resize(out.len().saturating_add(samples(ms)), 0.0);
                            }
                        }
                        let (reference, used) = self.reference_for(style);
                        let sampler = params.sampler(style);
                        let a = match strict {
                            None => self.read_take(&self.reading, &reference, sampler, &text, params.pace, Self::seed(params.take, i, 0))?,
                            Some(st) => {
                                // every retake keeps the line's style, reference and sampler
                                let mut read = |attempt: u32| {
                                    let seed = Self::seed(params.take, i, attempt);
                                    Ok(heard::Read {
                                        samples: self.read_take(&self.reading.for_attempt(attempt), &reference, sampler, &text, params.pace, seed)?,
                                        seed,
                                    })
                                };
                                let (a, mut c) = st.read_until_clear(&text, rate, &mut read)?;
                                c.style = style.map(|s| match used {
                                    Some(u) if u == s => s.to_string(),
                                    Some(u) => format!("{s} (after the {u} reference)"),
                                    None => format!("{s} (after the main reference)"),
                                });
                                checks.push(c);
                                a
                            }
                        };
                        out.extend_from_slice(&a);
                        last = Some((styles::boundary_after(&text), style));
                        marked = false;
                        line_break = None;
                    }
                }
                if out.len() > max {
                    return Err(TtsError::AudioTooLong);
                }
            }
            if last.is_none() {
                return Err(TtsError::Empty);
            }
            let mut out = if params.semitones.abs() > 1e-6 { neural::pitch_shift(out, params.semitones, rate) } else { out };
            let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
            if peak > 0.98 {
                let g = 0.98 / peak;
                out.iter_mut().for_each(|v| *v *= g);
            }
            Ok((Audio { samples: out, sample_rate: rate }, checks))
        }
    }

    #[cfg(test)]
    impl MalayalamVoice {
        pub(crate) fn new(gender: Gender, model: Arc<F5>, reference: Arc<Reference>, reading: Reading) -> MalayalamVoice {
            MalayalamVoice { gender, model, reference, reading, styles: Vec::new(), room: Vec::new() }
        }

        /// The voice with these style references and room tone.
        pub(crate) fn with_styles(self, styles: Vec<(&'static str, Arc<Reference>)>, room: Vec<f32>) -> MalayalamVoice {
            MalayalamVoice { styles, room, ..self }
        }
    }

    impl Voice for MalayalamVoice {
        fn info(&self) -> VoiceInfo {
            VoiceInfo {
                id: "clone",
                name: "Cloned voice (Malayalam)",
                language: crate::malayalam::TAG,
                gender: self.gender,
                engine: "clone",
                description: "A cloned voice speaking Malayalam: IndicF5 continuing the speaker's voice sample",
                license: "IndicF5 MIT (AI4Bharat); Vocos MIT (charactr)",
                installed: true,
            }
        }

        fn synthesize(&self, script: &str, params: &Params) -> Result<Audio, TtsError> {
            Ok(self.speak(script, params, None)?.0)
        }

        fn synthesize_checked(&self, script: &str, params: &Params, strict: &heard::Strict) -> Result<(Audio, Vec<heard::Checked>), TtsError> {
            self.speak(script, params, Some(strict))
        }
    }
}

#[cfg(not(feature = "indicf5"))]
mod malayalam_voice {
    use super::*;

    pub(super) fn load(_profile: Profile, _models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
        Err(TtsError::Invalid("Malayalam voices (IndicF5) are not available in this build (built without the `indicf5` feature)".into()))
    }

    pub(super) fn from_clip(_models_dir: &Path, _clip: &[f32], _rate: u32, _words: &str, _gender: Gender) -> Result<Box<dyn Voice>, TtsError> {
        Err(TtsError::Invalid("Malayalam voices (IndicF5) are not available in this build (built without the `indicf5` feature)".into()))
    }
}

/// The gender of a profile: the natural clone's estimate, else its base voice's.
pub fn gender(p: &Profile) -> Gender {
    if let Some(g) = p.natural.as_ref().and_then(|n| n.gender) {
        return g;
    }
    catalog::find(&p.base_voice).map(|v| v.info.gender).unwrap_or(Gender::Male)
}

/// The natural engine's voices (feature `voxcpm`).
#[cfg(feature = "voxcpm")]
mod natural {
    use super::*;
    use crate::heard;
    use crate::voxcpm::{Prompt, Settings, Variant, VoxCpm, text};

    /// Silence between utterances, ms.
    const UTTERANCE_GAP_MS: u64 = 250;
    /// Seeds between takes (more than any script's utterances).
    const TAKE_SEEDS: u64 = 7919;
    /// Prompts kept built (encoding a sample takes seconds).
    const PROMPTS: usize = 4;
    /// Longest wait for the GPU while another process speaks on it (it is held one script at
    /// a time, so a wait this long means someone reserved it).
    const GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

    /// The GPU, ours while the guard lives; waits for another process's script to finish.
    /// Without a place to keep locks it speaks unlocked rather than refuse.
    pub(super) fn gpu(what: &str) -> Result<Option<filmcraft_locks::Guard>, TtsError> {
        use filmcraft_locks::{LockError, Resource};
        // tests speak with tiny fake models: never wait on (or hold up) a real narration
        if cfg!(test) {
            return Ok(None);
        }
        match filmcraft_locks::acquire(Resource::Gpu, what, GPU_WAIT, &mut |_| {}) {
            Ok(g) => Ok(Some(g)),
            Err(e @ LockError::Busy(..)) => Err(TtsError::Model(format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)"))),
            Err(LockError::Io(_)) => Ok(None),
        }
    }

    /// The release in `models_dir`, loaded once and shared (one at a time: they are large).
    fn model(models_dir: &Path, pkg: &catalog::VoxcpmPackage) -> Result<Arc<VoxCpm>, TtsError> {
        static M: OnceLock<Mutex<Option<(PathBuf, Arc<VoxCpm>)>>> = OnceLock::new();
        if !catalog::voxcpm_installed(models_dir, pkg) {
            return Err(TtsError::NotInstalled);
        }
        let dir = catalog::voxcpm_dir(models_dir, pkg);
        let mut g = M.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((d, m)) = g.as_ref()
            && *d == dir
        {
            return Ok(m.clone());
        }
        // drop the other release before loading this one
        *g = None;
        let m = Arc::new(VoxCpm::load(&dir)?);
        *g = Some((dir, m.clone()));
        Ok(m)
    }

    /// The profile's prompt, built once per profile identity.
    fn prompt(p: &Profile, n: &Natural, m: &VoxCpm, voices: &Path) -> Result<Arc<Prompt>, TtsError> {
        type Prompts = Vec<(String, Arc<Prompt>)>;
        static P: OnceLock<Mutex<Prompts>> = OnceLock::new();
        let key = format!("{}|{}|r{}|{}|{:?}|{}", p.id, p.created, p.revision, n.model, n.mode, n.denoise_db);
        let cache = P.get_or_init(Default::default);
        if let Some(found) = cache.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()) {
            return Ok(found);
        }
        let (x, rate) = crate::wav::read(&voices.join(&n.clip))?;
        let mut x = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, m.encode_rate());
        if n.denoise_db > 0.0 {
            x = denoise(x, m.encode_rate(), n.denoise_db);
        }
        let transcript = match n.mode {
            PromptMode::Continuation => Some(text::sentences(&n.transcript)?.join(" ")),
            PromptMode::Reference => None,
        };
        let built = Arc::new(m.prompt(&x, transcript.as_deref())?);
        let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
        c.retain(|(k, _)| *k != key);
        c.push((key, built.clone()));
        let extra = c.len().saturating_sub(PROMPTS);
        c.drain(..extra);
        Ok(built)
    }

    /// Spectral-gating noise reduction (latency compensated, same length).
    fn denoise(mut x: Vec<f32>, rate: u32, db: f32) -> Vec<f32> {
        use filmcraft_audio_dsp::AudioEffect;
        let mut d = filmcraft_audio_dsp::effects::restoration::DeNoise::new(rate as f32, 1);
        d.set_param("reduction", db);
        let lat = d.latency();
        let n = x.len();
        x.resize(n + lat, 0.0);
        for b in x.chunks_mut(4096) {
            d.process(&mut [b]);
        }
        x.drain(..lat.min(x.len()));
        x.truncate(n);
        x
    }

    /// A natural voice speaking from `clip` as an isolated reference (VoxCPM2), no profile.
    pub(super) fn from_clip(models_dir: &Path, model_id: &str, clip: &[f32], rate: u32, cfg: f32, steps: usize) -> Result<Box<dyn Voice>, TtsError> {
        let pkg = catalog::voxcpm(model_id).ok_or_else(|| TtsError::Invalid(format!("unknown cloning model `{model_id}`")))?;
        let m = model(models_dir, pkg)?;
        let x = filmcraft_audio_dsp::resample::resample_sinc(clip, rate, m.encode_rate());
        let prompt = Arc::new(m.prompt(&x, None)?);
        let gender = estimate_gender(clip, rate).unwrap_or(Gender::Male);
        let natural = Natural {
            model: pkg.id.to_string(),
            clip: String::new(),
            transcript: String::new(),
            mode: PromptMode::Reference,
            cfg,
            timesteps: steps,
            denoise_db: 0.0,
            from: 0.0,
            to: clip.len() as f64 / f64::from(rate.max(1)),
            gender: Some(gender),
        };
        Ok(Box::new(NaturalVoice { gender, natural, model: m, prompt }))
    }

    pub(super) fn load(profile: Profile, models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
        let n = profile.natural.clone().ok_or_else(|| TtsError::Invalid("the voice profile has no voice sample".into()))?;
        let pkg = catalog::voxcpm(&n.model).ok_or_else(|| TtsError::Invalid(format!("unknown cloning model `{}`", n.model)))?;
        let m = model(models_dir, pkg)?;
        let prompt = prompt(&profile, &n, &m, &voices_dir(models_dir))?;
        Ok(Box::new(NaturalVoice { gender: gender(&profile), natural: n, model: m, prompt }))
    }

    /// A natural cloned voice ready to speak.
    pub struct NaturalVoice {
        gender: Gender,
        natural: Natural,
        model: Arc<VoxCpm>,
        prompt: Arc<Prompt>,
    }

    impl Voice for NaturalVoice {
        fn info(&self) -> VoiceInfo {
            VoiceInfo {
                id: "clone",
                name: "Cloned voice",
                language: "en-US",
                gender: self.gender,
                engine: "clone",
                description: "A cloned voice: VoxCPM speaking in the speaker's own voice",
                license: "VoxCPM Apache-2.0 (OpenBMB)",
                installed: true,
            }
        }

        fn synthesize(&self, script: &str, params: &Params) -> Result<Audio, TtsError> {
            Ok(self.speak(script, params, None)?.0)
        }

        /// The strict check of English narration: every utterance heard back (numbers compared as
        /// words, [`heard::english_words`]) and read again with the next seed until it is clear.
        fn synthesize_checked(&self, script: &str, params: &Params, strict: &heard::Strict) -> Result<(Audio, Vec<heard::Checked>), TtsError> {
            self.speak(script, params, Some(strict))
        }
    }

    impl NaturalVoice {
        /// Seeds between attempts of one utterance (more than any script's utterances times takes).
        const ATTEMPT_SEEDS: u64 = 104_729;

        fn speak(&self, script: &str, params: &Params, strict: Option<&heard::Strict>) -> Result<(Audio, Vec<heard::Checked>), TtsError> {
            crate::check_text(script)?;
            params.check()?;
            let rate = self.model.output_rate();
            let max = (crate::MAX_SECONDS * f64::from(rate)) as usize;
            let s = Settings { cfg: self.natural.cfg, timesteps: self.natural.timesteps, ..Settings::default() };
            let gap = (UTTERANCE_GAP_MS * u64::from(rate) / 1000) as usize;
            // a spoken style goes in front of every utterance; VoxCPM2 follows it only after a
            // reference-only prompt (a continuation prompt sets the delivery itself)
            let style = params.style.filter(|_| self.model.variant() == Variant::V2 && !self.prompt.continues()).map(text::style_prefix).unwrap_or_default();
            let _gpu = gpu("VoxCPM narration")?;
            let mut out: Vec<f32> = Vec::new();
            let mut checks = Vec::new();
            let mut spoke = false;
            for (i, piece) in text::pieces(script)?.into_iter().enumerate() {
                match piece {
                    text::Piece::Pause(ms) => out.resize(out.len().saturating_add((u64::from(ms) * u64::from(rate) / 1000) as usize), 0.0),
                    text::Piece::Say(u) => {
                        if spoke {
                            out.resize(out.len().saturating_add(gap), 0.0);
                        }
                        let said = format!("{style}{u}");
                        // each take reads afresh: its own run of seeds (and each retake of an
                        // utterance its own after that)
                        let seed = |attempt: u32| {
                            s.seed
                                .wrapping_add(u64::from(params.take).wrapping_mul(TAKE_SEEDS))
                                .wrapping_add(i as u64)
                                .wrapping_add(u64::from(attempt).wrapping_mul(Self::ATTEMPT_SEEDS))
                        };
                        let mut read = |attempt: u32| -> Result<heard::Read, TtsError> {
                            let a = self.model.speak(&said, Some(&self.prompt), &Settings { seed: seed(attempt), ..s }, &mut |_, _| true)?;
                            Ok(heard::Read { samples: trim_silence(&a, rate).to_vec(), seed: seed(attempt) })
                        };
                        let a = match strict {
                            None => read(0)?.samples,
                            Some(st) => {
                                // what was heard goes through the same normaliser as the script (a
                                // recogniser writes "2023" and "80%"; the voice read "twenty
                                // twenty-three" and "eighty percent"), then any digit left is spelled
                                let norm = |t: &str| heard::english_words(&text::sentences(t).map(|s| s.join(" ")).unwrap_or_else(|_| t.to_string()));
                                let words = heard::english_words(&u);
                                let mut hear = |x: &[f32]| {
                                    st.hearer
                                        .hear(x, rate)
                                        .map(|h| norm(&h))
                                        .map_err(|e| TtsError::Model(format!("the strict check could not hear the reading: {e}")))
                                };
                                let mut second = st.second.map(|h| move |x: &[f32]| h.hear(x, rate).map(|t| norm(&t)).map_err(TtsError::Model));
                                let mut voice = st.voice.map(|v| move |x: &[f32]| v.similarity(x, rate).map_err(TtsError::Model));
                                let mut judges = heard::Judges {
                                    second: second.as_mut().map(|f| f as &mut dyn FnMut(&[f32]) -> Result<String, TtsError>),
                                    agreement: st.agreement,
                                    voice: voice.as_mut().map(|f| f as &mut dyn FnMut(&[f32]) -> Result<f64, TtsError>),
                                    min_similarity: st.min_similarity,
                                };
                                let (a, c) = heard::read_until_clear_judged(&words, &st.policy, rate, &mut read, &mut hear, &mut judges)?;
                                checks.push(c);
                                a
                            }
                        };
                        out.extend_from_slice(&a);
                        spoke = true;
                    }
                }
                if out.len() > max {
                    return Err(TtsError::AudioTooLong);
                }
            }
            if !spoke {
                return Err(TtsError::Empty);
            }
            let mut out = pace_and_pitch(out, rate, params.pace, params.semitones);
            let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
            if peak > 0.98 {
                let g = 0.98 / peak;
                out.iter_mut().for_each(|v| *v *= g);
            }
            Ok((Audio { samples: out, sample_rate: rate }, checks))
        }
    }

    /// `x` without leading and trailing silence (below −50 dB from its peak), 40 ms kept.
    pub(super) fn trim_silence(x: &[f32], rate: u32) -> &[f32] {
        let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let th = peak * 0.003;
        let margin = (rate / 25) as usize;
        let first = x.iter().position(|v| v.abs() > th).unwrap_or(0);
        let last = x.iter().rposition(|v| v.abs() > th).map_or(x.len(), |p| p + 1);
        x.get(first.saturating_sub(margin)..(last + margin).min(x.len())).unwrap_or(x)
    }

    /// Pace by resampling (faster and higher), then the pitch shifter puts the pitch back (and
    /// moves it by `semitones`): the length changes, the pitch only by `semitones`.
    pub(super) fn pace_and_pitch(x: Vec<f32>, rate: u32, pace: f64, semitones: f64) -> Vec<f32> {
        let mut x = x;
        let mut shift = semitones;
        if (pace - 1.0).abs() > 1e-6 && pace.is_finite() && pace > 0.0 {
            let to = (f64::from(rate) / pace).round().clamp(1000.0, 1_000_000.0) as u32;
            x = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, to);
            shift -= 12.0 * pace.log2();
        }
        if shift.abs() > 1e-6 { neural::pitch_shift(x, shift, rate) } else { x }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn trimming_keeps_the_sound() {
            let rate = 16_000;
            let mut x = vec![0f32; 8000];
            x.extend((0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.5));
            x.extend(vec![0f32; 8000]);
            let y = trim_silence(&x, rate);
            assert!(y.len() >= 16_000 && y.len() <= 16_000 + 2 * 640 + 2, "{}", y.len());
            assert!(trim_silence(&[], rate).is_empty());
            assert_eq!(trim_silence(&[0.0; 10], rate).len(), 10, "all silence: unchanged");
        }

        #[test]
        fn pace_changes_the_length_not_the_pitch() {
            let rate = 16_000;
            let x: Vec<f32> = (0..rate * 2).map(|i| (std::f32::consts::TAU * 200.0 * i as f32 / rate as f32).sin() * 0.5).collect();
            let y = pace_and_pitch(x.clone(), rate, 2.0, 0.0);
            assert!((y.len() as f64 - x.len() as f64 / 2.0).abs() < 100.0, "{}", y.len());
            // zero crossings per second stay near 400 (200 Hz)
            let mid = y.get(y.len() / 4..3 * y.len() / 4).unwrap();
            let zc = mid.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count() as f64 / (mid.len() as f64 / f64::from(rate));
            assert!((zc - 400.0).abs() < 60.0, "{zc}");
            assert_eq!(pace_and_pitch(x.clone(), rate, 1.0, 0.0), x, "identity");
        }

        /// A natural voice on a tiny random VoxCPM: scripts with pauses, pace, determinism.
        #[test]
        fn a_natural_voice_speaks_scripts() {
            for v2 in [false, true] {
                let d = std::env::temp_dir().join(format!("filmcraft-clone-natural-voice-{}-{v2}", std::process::id()));
                let _ = std::fs::remove_dir_all(&d);
                crate::voxcpm::model::tests::fake_release(&d, v2);
                let m = Arc::new(crate::voxcpm::model::tests::load_fake(&d));
                let rate = m.encode_rate() as usize;
                let audio: Vec<f32> = (0..rate * 2).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
                let prompt = Arc::new(m.prompt(&audio, if v2 { None } else { Some("hello") }).unwrap());
                let mut natural = super::super::tests::natural_profile().natural.unwrap();
                natural.timesteps = 3;
                let v = NaturalVoice { gender: Gender::Male, natural, model: m.clone(), prompt };
                let a = v.synthesize("hello world. [pause 500ms] hello", &Params::default()).unwrap();
                assert_eq!(a.sample_rate, m.output_rate());
                assert!(a.samples.iter().all(|x| x.is_finite() && x.abs() <= 0.98));
                // at least the pause and the gap between the utterances
                assert!(a.seconds() > 0.5 + 0.25, "{}", a.seconds());
                assert_eq!(v.synthesize("hello world. [pause 500ms] hello", &Params::default()).unwrap(), a, "deterministic");
                let fast = v.synthesize("hello world. [pause 500ms] hello", &Params { pace: 2.0, ..Default::default() }).unwrap();
                assert!(fast.seconds() < a.seconds() * 0.7, "{} vs {}", fast.seconds(), a.seconds());
                assert_eq!(v.synthesize("", &Params::default()), Err(TtsError::Empty));
                assert_eq!(v.synthesize("[pause 1s]", &Params::default()), Err(TtsError::Empty));
                assert!(v.synthesize("hi", &Params { pace: 9.0, ..Default::default() }).is_err());
                assert_eq!(v.info().engine, "clone");
                let _ = std::fs::remove_dir_all(&d);
            }
        }

        /// The English strict check on a tiny random VoxCPM: a recogniser that mishears the first
        /// reading of every utterance gets each read twice, numbers compared as words.
        #[test]
        fn english_narration_is_heard_back_and_retaken() {
            let d = std::env::temp_dir().join(format!("filmcraft-clone-natural-strict-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            crate::voxcpm::model::tests::fake_release(&d, true);
            let m = Arc::new(crate::voxcpm::model::tests::load_fake(&d));
            let rate = m.encode_rate() as usize;
            let audio: Vec<f32> = (0..rate * 2).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
            let prompt = Arc::new(m.prompt(&audio, None).unwrap());
            let mut natural = super::super::tests::natural_profile().natural.unwrap();
            natural.timesteps = 3;
            let v = NaturalVoice { gender: Gender::Male, natural, model: m.clone(), prompt };
            struct Ear(std::sync::atomic::AtomicU32, Vec<&'static str>);
            impl heard::Hearer for Ear {
                fn hear(&self, _: &[f32], _: u32) -> Result<String, String> {
                    let n = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize;
                    // what Whisper writes: digits
                    Ok(if n.is_multiple_of(2) { "mumble".into() } else { self.1[(n / 2) % self.1.len()].to_string() })
                }
                fn id(&self) -> String {
                    "ear".into()
                }
            }
            let ear = Ear(Default::default(), vec!["In 2023 it charged from 10 to 80%.", "Done"]);
            let strict = heard::Strict::new(&ear, heard::Policy { max_takes: 2, ..heard::Policy::english() });
            let (a, report) = v.synthesize_checked("In 2023 it charged from ten to eighty percent. [pause 300ms] Done.", &Params::default(), &strict).unwrap();
            assert_eq!(report.len(), 2);
            for c in &report {
                // the tiny model's readings are too short for the signal scan, so every try is
                // made; what matters: the misheard first reading was read again, and the
                // second, heard with its numbers in digits, compared equal as words
                assert!(c.tries.len() >= 2, "{c:?}");
                assert!(c.tries[0].flagged.contains(&"first".to_string()), "{c:?}");
                assert_eq!(c.tries[1].score.cer, 0.0, "{c:?}");
                assert_eq!(c.kept, 1);
                assert_ne!(c.tries[0].seed, c.tries[1].seed, "a retake reads with its own seed");
            }
            assert!(a.samples.iter().all(|x| x.is_finite()));
            // unchecked synthesis still reads every utterance once, as before
            assert_eq!(v.synthesize("Done.", &Params::default()).unwrap(), v.synthesize("Done.", &Params::default()).unwrap());
            let _ = std::fs::remove_dir_all(&d);
        }
    }
}

#[cfg(not(feature = "voxcpm"))]
mod natural {
    use super::*;

    pub(super) fn load(_profile: Profile, _models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
        Err(TtsError::Invalid("natural cloned voices (VoxCPM) are not available in this build (built without the `voxcpm` feature)".into()))
    }

    pub(super) fn from_clip(_models_dir: &Path, _model: &str, _clip: &[f32], _rate: u32, _cfg: f32, _steps: usize) -> Result<Box<dyn Voice>, TtsError> {
        Err(TtsError::Invalid("natural cloned voices (VoxCPM) are not available in this build (built without the `voxcpm` feature)".into()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn profile() -> Profile {
        Profile {
            version: PROFILE_VERSION,
            id: "clone:my-voice".into(),
            name: "My Voice".into(),
            engine: Engine::Openvoice,
            base_voice: "kokoro-michael".into(),
            source_colour: vec![0.1; COLOUR],
            target_colour: vec![0.2; COLOUR],
            base_similarity: 0.8,
            natural: None,
            reference_seconds: 30.0,
            source: "IMG_3337.MOV".into(),
            created: 1,
            consent: true,
            revision: 0,
        }
    }

    pub(crate) fn natural_profile() -> Profile {
        Profile {
            version: NATURAL_PROFILE_VERSION,
            id: "clone:my-voice".into(),
            name: "My Voice".into(),
            engine: Engine::Voxcpm,
            base_voice: String::new(),
            source_colour: Vec::new(),
            target_colour: Vec::new(),
            base_similarity: 0.0,
            natural: Some(Natural {
                model: "voxcpm-2".into(),
                clip: "my-voice.wav".into(),
                transcript: "Hello there.".into(),
                mode: PromptMode::Reference,
                cfg: 2.0,
                timesteps: 10,
                denoise_db: 0.0,
                from: 1.0,
                to: 15.0,
                gender: Some(Gender::Male),
            }),
            reference_seconds: 14.0,
            source: "IMG_3337.MOV".into(),
            created: 1,
            consent: true,
            revision: 0,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-clone-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn slugs_are_file_safe() {
        assert_eq!(slug("Mohesh's Voice!"), "mohesh-s-voice");
        assert_eq!(slug("  ../../etc/passwd "), "etc-passwd");
        assert_eq!(slug("日本語"), "");
        assert!(slug(&"a".repeat(500)).len() <= 40);
    }

    #[test]
    fn profiles_round_trip_and_are_checked() {
        let d = tmp("round");
        let p = profile();
        save(&d, &p).unwrap();
        assert_eq!(load_profile(&d, "clone:my-voice").unwrap(), p);
        assert_eq!(list(&d), vec![p.clone()]);
        // damaged or consent-less profiles are refused
        for bad in [
            Profile { consent: false, ..p.clone() },
            Profile { target_colour: vec![f32::NAN; COLOUR], ..p.clone() },
            Profile { source_colour: vec![0.0; 3], ..p.clone() },
            Profile { base_voice: "nope".into(), ..p.clone() },
            Profile { version: 99, ..p.clone() },
            Profile { engine: Engine::Voxcpm, ..p.clone() },
        ] {
            std::fs::write(d.join("my-voice.json"), serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(load_profile(&d, "clone:my-voice").is_err(), "{bad:?}");
        }
        assert!(list(&d).is_empty(), "a bad file is skipped, not fatal");
        assert!(load_profile(&d, "clone:missing").is_err());
        assert!(load_profile(&d, "clone:../../x").is_err());
        // a profile under another voice's file name is refused
        std::fs::write(d.join("other.json"), serde_json::to_vec(&p).unwrap()).unwrap();
        assert!(load_profile(&d, "clone:other").is_err());
        std::fs::write(d.join("my-voice.json"), serde_json::to_vec(&p).unwrap()).unwrap();
        delete(&d, "clone:my-voice").unwrap();
        assert!(delete(&d, "clone:my-voice").is_err());
    }

    /// A profile file written before natural clones existed (no `engine`, no `natural`) loads
    /// unchanged as an OpenVoice clone.
    #[test]
    fn version_one_files_still_load() {
        let d = tmp("v1");
        std::fs::create_dir_all(&d).unwrap();
        let colour = |v: f32| serde_json::to_string(&vec![v; COLOUR]).unwrap();
        let json = format!(
            r#"{{"version":1,"id":"clone:mohesh","name":"Mohesh","baseVoice":"kokoro-michael","sourceColour":{},"targetColour":{},"baseSimilarity":0.5,"referenceSeconds":104.0,"source":"IMG_3337.MOV [0.0–120.0 s]","created":1760597880,"consent":true}}"#,
            colour(0.1),
            colour(0.2)
        );
        std::fs::write(d.join("mohesh.json"), json).unwrap();
        let p = load_profile(&d, "clone:mohesh").unwrap();
        assert_eq!(p.engine, Engine::Openvoice);
        assert_eq!(p.natural, None);
        assert_eq!(p.base_voice, "kokoro-michael");
        // and saving it again keeps it a valid version 1 profile
        save(&d, &p).unwrap();
        assert_eq!(load_profile(&d, "clone:mohesh").unwrap(), p);
    }

    #[test]
    fn natural_profiles_round_trip_and_are_checked() {
        let d = tmp("natural");
        let p = natural_profile();
        save(&d, &p).unwrap();
        assert_eq!(load_profile(&d, "clone:my-voice").unwrap(), p);
        let n = p.natural.clone().unwrap();
        for bad in [
            Profile { natural: None, ..p.clone() },
            Profile { version: PROFILE_VERSION, ..p.clone() },
            Profile { natural: Some(Natural { model: "voxcpm-9".into(), ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { clip: "../../etc/passwd".into(), ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { clip: "other.wav".into(), ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { cfg: f32::NAN, ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { cfg: 50.0, ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { timesteps: 0, ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { denoise_db: -3.0, ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { mode: PromptMode::Continuation, transcript: " ".into(), ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { model: "voxcpm-1.5".into(), mode: PromptMode::Reference, ..n.clone() }), ..p.clone() },
            Profile { natural: Some(Natural { transcript: "x".repeat(MAX_TRANSCRIPT_CHARS + 1), ..n.clone() }), ..p.clone() },
            Profile { consent: false, ..p.clone() },
        ] {
            std::fs::write(d.join("my-voice.json"), serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(load_profile(&d, "clone:my-voice").is_err(), "{bad:?}");
        }
        // deleting removes the sample too
        save(&d, &p).unwrap();
        std::fs::write(d.join("my-voice.wav"), crate::wav::encode(&[0.0; 10], 16_000)).unwrap();
        delete(&d, "clone:my-voice").unwrap();
        assert!(!d.join("my-voice.wav").exists());
        assert_eq!(gender(&p), Gender::Male);
    }

    #[test]
    fn the_cache_identity_follows_the_profile() {
        let root = tmp("identity");
        let models = root.join("models");
        let voices = voices_dir(&models);
        let p = profile();
        save(&voices, &p).unwrap();
        let id = |m: &Path| crate::cache::voice_identity("clone:my-voice", Some(m));
        let a = id(&models).unwrap();
        assert!(a.starts_with("clone:my-voice|clone v1|created 1|base kokoro-michael|colour "), "{a}");
        // the voice made again (another time, colour or base voice): another identity, so speech
        // cached for the old one is never served for it
        for changed in [
            Profile { created: 2, ..p.clone() },
            Profile { target_colour: vec![0.3; COLOUR], ..p.clone() },
            Profile { base_voice: "kokoro-puck".into(), ..p.clone() },
        ] {
            save(&voices, &changed).unwrap();
            assert_ne!(id(&models).unwrap(), a, "{changed:?}");
        }
        save(&voices, &p).unwrap();
        assert_eq!(id(&models).unwrap(), a, "the same profile, the same identity");
        // a profile that does not load is not cached
        std::fs::write(voices.join("my-voice.json"), b"{").unwrap();
        assert_eq!(id(&models), None);
        assert_eq!(crate::cache::voice_identity("clone:missing", Some(&models)), None);
        // a natural clone: its sample, transcript and settings are part of the identity
        let n = natural_profile();
        save(&voices, &n).unwrap();
        std::fs::write(voices.join("my-voice.wav"), crate::wav::encode(&[0.1; 100], 16_000)).unwrap();
        let b = id(&models).unwrap();
        assert!(b.starts_with("clone:my-voice|clone v2|created 1|voxcpm-2|"), "{b}");
        std::fs::write(voices.join("my-voice.wav"), crate::wav::encode(&[0.2; 100], 16_000)).unwrap();
        assert_ne!(id(&models).unwrap(), b, "another sample");
        let mut m = n.clone();
        if let Some(x) = m.natural.as_mut() {
            x.timesteps = 6;
        }
        save(&voices, &m).unwrap();
        let c = id(&models).unwrap();
        save(&voices, &n).unwrap();
        assert_ne!(c, id(&models).unwrap(), "other settings");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn speech_only_drops_silence() {
        let rate = 16_000u32;
        // 1 s silence, 1 s tone, 1 s silence, 1 s tone
        let mut x = vec![0f32; rate as usize];
        x.extend((0..rate).map(|i| (i as f32 * 0.1).sin() * 0.3));
        x.extend(vec![0f32; rate as usize]);
        x.extend((0..rate).map(|i| (i as f32 * 0.1).sin() * 0.3));
        let y = speech_only(&x, rate);
        let secs = y.len() as f64 / f64::from(rate);
        assert!((2.0..2.6).contains(&secs), "{secs} s kept");
        assert!(speech_only(&[], rate).is_empty());
    }

    /// Speech-like bursts: `on` seconds of a buzzy tone, `off` of near-silence, repeated.
    fn bursts(rate: u32, pattern: &[(f64, f64)]) -> Vec<f32> {
        let mut x = Vec::new();
        let mut rng = crate::kokoro::dsp::Rng::new(3);
        for (on, off) in pattern {
            let n = (on * f64::from(rate)) as usize;
            x.extend((0..n).map(|i| ((i as f32 * 0.07).sin() + 0.5 * (i as f32 * 0.31).sin()) * 0.2));
            let m = (off * f64::from(rate)) as usize;
            x.extend((0..m).map(|_| rng.normal() as f32 * 0.0005));
        }
        x
    }

    #[test]
    fn the_sample_is_the_densest_stretch_between_pauses() {
        let rate = 16_000;
        // 10 s of sparse speech (long pauses), then 18 s of dense speech, then sparse again
        let mut pattern = vec![(0.5, 2.0); 4];
        pattern.extend(vec![(2.7, 0.3); 6]);
        pattern.extend(vec![(0.5, 2.0); 4]);
        let x = bursts(rate, &pattern);
        let s = pick_sample(&x, rate).unwrap();
        let dense_from = 10.0;
        assert!(s.from >= dense_from - 1.0 && s.to <= dense_from + 18.0 + 2.5, "{:.1}–{:.1}", s.from, s.to);
        assert!(s.speech_seconds >= 14.0, "{}", s.speech_seconds);
        let secs = s.audio.len() as f64 / f64::from(rate);
        assert!((10.0..=25.0).contains(&secs), "{secs}");
        // loudness-normalised, peaks in range
        let l = filmcraft_audio_dsp::loudness::measure(&[&s.audio], f64::from(rate)).integrated_lufs;
        assert!((l - SAMPLE_LUFS).abs() < 1.5, "{l} LUFS");
        assert!(s.audio.iter().all(|v| v.abs() <= 0.95));
    }

    #[test]
    fn short_silent_and_odd_references() {
        let rate = 16_000;
        // 5 s of speech: all of it
        let x = bursts(rate, &[(1.0, 0.3); 5]);
        let s = pick_sample(&x, rate).unwrap();
        assert!(s.speech_seconds >= 4.0, "{}", s.speech_seconds);
        // too little speech, silence, nothing, no rate
        assert!(pick_sample(&bursts(rate, &[(1.0, 3.0)]), rate).is_err());
        assert!(pick_sample(&vec![0.0; rate as usize * 30], rate).is_err());
        assert!(pick_sample(&[], rate).is_err());
        assert!(pick_sample(&x, 0).is_err());
        // pauses longer than half a second are shortened
        let x = bursts(rate, &[(3.0, 2.0), (3.0, 2.0), (3.0, 2.0), (3.0, 0.1)]);
        let s = pick_sample(&x, rate).unwrap();
        let secs = s.audio.len() as f64 / f64::from(rate);
        assert!(secs < 15.0, "{secs} (18.1 s before shortening)");
        // speech without a pause as long as the sample: cut on a half-second grid
        let x = bursts(rate, &[(30.0, 0.0)]);
        let s = pick_sample_within(&x, rate, CONTINUATION_SECONDS).unwrap();
        let secs = s.audio.len() as f64 / f64::from(rate);
        assert!((5.0..=10.5).contains(&secs), "{secs}");
        assert!(pick_sample_within(&x, rate, (f64::NAN, 3.0)).is_err());
        assert!(pick_sample_within(&x, rate, (10.0, 5.0)).is_err());
    }

    #[test]
    fn gender_from_pitch() {
        let rate = 16_000;
        let voice = |f0: f32| -> Vec<f32> {
            (0..rate * 3)
                .map(|i| (1..8).map(|h| (std::f32::consts::TAU * f0 * h as f32 * i as f32 / rate as f32).sin() / h as f32).sum::<f32>() * 0.2)
                .collect()
        };
        assert_eq!(estimate_gender(&voice(110.0), rate as u32), Some(Gender::Male));
        assert_eq!(estimate_gender(&voice(220.0), rate as u32), Some(Gender::Female));
        assert_eq!(estimate_gender(&vec![0.0; 16_000], rate as u32), None);
        assert_eq!(estimate_gender(&[], rate as u32), None);
    }

    #[test]
    fn natural_samples_are_saved_with_their_transcript() {
        let rate = 22_050;
        let mut pattern = vec![(2.0, 0.3); 8];
        pattern.push((0.5, 2.0));
        let x = bursts(rate, &pattern);
        let d = tmp("save-natural");
        let req = NaturalRequest::default();
        // reference mode (VoxCPM2): no transcription
        let asked = std::cell::Cell::new(0);
        let mut tr = |_: &[f32]| {
            asked.set(asked.get() + 1);
            Ok("hello there".to_string())
        };
        let p = save_natural("me", "Me", &x, rate, "voxcpm-2", PromptMode::Reference, &req, "test", &d, &mut tr, &mut |_| {}).unwrap();
        assert_eq!(asked.get(), 0);
        let n = p.natural.clone().unwrap();
        assert_eq!((n.model.as_str(), n.mode, n.transcript.as_str(), n.denoise_db), ("voxcpm-2", PromptMode::Reference, "", 0.0));
        assert_eq!(load_profile(&d, "clone:me").unwrap(), p);
        let (clip, clip_rate) = crate::wav::read(&d.join("me.wav")).unwrap();
        assert_eq!(clip_rate, 16_000, "stored at VoxCPM2's encoder rate");
        // and a full-band copy (the recording's rate) for IndicF5
        assert_eq!(crate::wav::read(&d.join("me.full.wav")).unwrap().1, rate);
        assert!((clip.len() as f64 / 16_000.0 - p.reference_seconds).abs() < 0.01);
        assert!(p.reference_seconds >= 10.0 && p.reference_seconds <= 25.0, "{}", p.reference_seconds);
        // continuation (VoxCPM1.5): transcribed once, denoised when spoken
        let p = save_natural("me", "Me", &x, rate, "voxcpm-1.5", PromptMode::Continuation, &req, "test", &d, &mut tr, &mut |_| {}).unwrap();
        assert!(p.reference_seconds >= 4.0 && p.reference_seconds <= CONTINUATION_SECONDS.1 + 0.5, "a short sample to continue: {}", p.reference_seconds);
        let n = p.natural.unwrap();
        assert_eq!((n.transcript.as_str(), n.denoise_db), ("hello there", CONTINUATION_DENOISE_DB));
        assert_eq!(crate::wav::read(&d.join("me.wav")).unwrap().1, 44_100);
        assert!(!d.join("me.full.wav").exists(), "the sample is the recording's own rate: no copy");
        // a given transcript is used as is; a failed or empty transcription is an error
        let given = NaturalRequest { transcript: Some(" exact words ".into()), ..req.clone() };
        let mut fail = |_: &[f32]| Err("no model".to_string());
        let p = save_natural("me", "Me", &x, rate, "voxcpm-1.5", PromptMode::Continuation, &given, "", &d, &mut fail, &mut |_| {}).unwrap();
        assert_eq!(p.natural.unwrap().transcript, "exact words");
        assert!(save_natural("me", "Me", &x, rate, "voxcpm-1.5", PromptMode::Continuation, &req, "", &d, &mut fail, &mut |_| {}).is_err());
        let mut nothing = |_: &[f32]| Ok("  ...  ".to_string());
        assert!(save_natural("me", "Me", &x, rate, "voxcpm-1.5", PromptMode::Continuation, &req, "", &d, &mut nothing, &mut |_| {}).is_err());
        // silence or a second of speech makes no voice
        assert!(save_natural("me", "Me", &vec![0.0; rate as usize * 20], rate, "voxcpm-2", PromptMode::Reference, &req, "", &d, &mut tr, &mut |_| {}).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A natural clone `clone:my-voice` with a 20 s voice sample of bursts in `<root>/voices`.
    pub(crate) fn natural_clone(root: &Path) -> PathBuf {
        let models = root.join("models");
        let voices = voices_dir(&models);
        save(&voices, &natural_profile()).unwrap();
        let x = bursts(16_000, &[(2.0, 0.3); 9]);
        crate::wav::write(&voices.join("my-voice.wav"), &x, 16_000).unwrap();
        models
    }

    #[test]
    fn the_malayalam_reference_is_made_once_per_sample() {
        let root = tmp("cross");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        let heard = std::cell::Cell::new(0usize);
        let mut tr = |x: &[f32]| {
            heard.set(x.len());
            Ok(" hello   there ".to_string())
        };
        let r = prepare_cross_lingual(&models, "clone:my-voice", None, None, &mut tr).unwrap();
        assert_eq!((r.transcript.as_str(), r.given), ("hello there", false));
        assert!(r.source.contains("16000 Hz"), "{}", r.source);
        // the clip Whisper heard (16 kHz) is the 10–15 s stretch, kept at 24 kHz
        let secs = heard.get() as f64 / 16_000.0;
        assert!((9.0..=15.5).contains(&secs), "{secs}");
        let clip = cross_audio(&models, &r).unwrap();
        assert!((clip.len() as f64 / f64::from(CROSS_REFERENCE_RATE) - secs).abs() < 0.05);
        assert_eq!(crate::wav::read(&voices.join("my-voice.indicf5.wav")).unwrap().1, CROSS_REFERENCE_RATE);
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), Some(r.clone()));
        assert!(voices.join("my-voice.indicf5.json").exists());
        // the identity of the Malayalam voice: present now, follows the transcript; English
        // keeps its own (unchanged) identity
        let en = crate::cache::voice_identity("clone:my-voice", Some(&models)).unwrap();
        assert_eq!(crate::cache::voice_identity_for("clone:my-voice", "en-US", Some(&models)), Some(en.clone()));
        let ml = crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)).unwrap();
        assert_ne!(ml, en);
        assert!(ml.contains("indicf5") && ml.contains("ml-IN"), "{ml}");
        let given = prepare_cross_lingual(&models, "clone:my-voice", None, Some(" exact words "), &mut |_| Err("unused".into())).unwrap();
        assert_eq!((given.transcript.as_str(), given.given), ("exact words", true));
        assert_ne!(crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)).unwrap(), ml);
        // correcting the words keeps the clip
        let fixed = set_cross_transcript(&models, "clone:my-voice", " the  right words ").unwrap();
        assert_eq!((fixed.transcript.as_str(), fixed.clip_sha.as_str()), ("the right words", given.clip_sha.as_str()));
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), Some(fixed));
        assert!(set_cross_transcript(&models, "clone:my-voice", " ... ").is_err());
        // a recording given for it: its best stretch, full band
        let rec = bursts(44_100, &[(2.4, 0.3); 8]);
        let src = CrossSource { audio: &rec, rate: 44_100, source: "IMG_3337.MOV [56–80 s]" };
        let fb = prepare_cross_lingual(&models, "clone:my-voice", Some(src), Some("words"), &mut |_| Err("unused".into())).unwrap();
        assert_eq!(fb.source, "IMG_3337.MOV [56–80 s]");
        assert_ne!(fb.clip_sha, given.clip_sha);
        // the voice sample's full-band copy is preferred to the sample
        crate::wav::write(&full_sample_path(&voices, &load_profile(&voices, "clone:my-voice").unwrap()).unwrap(), &rec, 44_100).unwrap();
        let full = prepare_cross_lingual(&models, "clone:my-voice", None, Some("words"), &mut |_| Err("unused".into())).unwrap();
        assert!(full.source.contains("full band"), "{}", full.source);
        // a damaged clip: no reference
        std::fs::write(voices.join("my-voice.indicf5.wav"), b"RIFF").unwrap();
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        prepare_cross_lingual(&models, "clone:my-voice", None, Some("words"), &mut |_| Err("unused".into())).unwrap();
        // a voice sample made again: the reference is stale
        crate::wav::write(&voices.join("my-voice.wav"), &bursts(16_000, &[(2.5, 0.3); 8]), 16_000).unwrap();
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        assert_eq!(crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)), None);
        // nothing heard, a failed transcription, a damaged file, too little speech
        assert!(prepare_cross_lingual(&models, "clone:my-voice", None, None, &mut |_| Ok(" ... ".into())).is_err());
        assert!(prepare_cross_lingual(&models, "clone:my-voice", None, None, &mut |_| Err("no model".into())).is_err());
        let silent = vec![0f32; 44_100 * 20];
        let src = CrossSource { audio: &silent, rate: 44_100, source: "silence" };
        assert!(prepare_cross_lingual(&models, "clone:my-voice", Some(src), Some("hi"), &mut |_| Err(String::new())).is_err());
        std::fs::write(voices.join("my-voice.indicf5.json"), b"{not json").unwrap();
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        // deleting the voice deletes its side files
        delete(&voices, "clone:my-voice").unwrap();
        assert!(!voices.join("my-voice.indicf5.wav").exists() && !voices.join("my-voice.full.wav").exists());
        // a timbre-only clone has no voice sample
        save(&voices, &Profile { id: "clone:other".into(), ..profile() }).unwrap();
        assert!(prepare_cross_lingual(&models, "clone:other", None, Some("hi"), &mut |_| Err(String::new())).is_err());
        assert!(cross_reference(&models, "clone:missing").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Speech-like audio at `rate`: sentences of `n` syllables each (180 ms of a buzz at 0.3,
    /// dips of 40 ms to −20 dB between syllables), 150 ms pauses at −30 dB between sentences,
    /// and the centres of those pauses (seconds).
    fn sentences_audio(rate: u32, sentences: &[usize]) -> (Vec<f32>, Vec<f64>) {
        let mut x: Vec<f32> = Vec::new();
        let mut pauses = Vec::new();
        let push = |x: &mut Vec<f32>, secs: f64, amp: f32| {
            let n = (secs * f64::from(rate)).round() as usize;
            for _ in 0..n {
                let t = x.len() as f32 / rate as f32;
                let w = std::f32::consts::TAU * 160.0 * t;
                x.push(amp * (0.6 * w.sin() + 0.3 * (2.0 * w).sin() + 0.1 * (3.0 * w).sin()));
            }
        };
        for (k, &n) in sentences.iter().enumerate() {
            for s in 0..n {
                push(&mut x, 0.18, 0.3);
                if s + 1 < n {
                    push(&mut x, 0.04, 0.03);
                }
            }
            if k + 1 < sentences.len() {
                pauses.push(x.len() as f64 / f64::from(rate) + 0.075);
                push(&mut x, 0.15, 0.0095);
            }
        }
        (x, pauses)
    }

    /// "ka ka … ka." per sentence (one syllable a word); the last without its full stop.
    fn sentences_text(sentences: &[usize]) -> Vec<String> {
        sentences.iter().map(|&n| vec!["ka"; n].join(" ")).collect()
    }

    /// `clone:my-voice` with `x` as its Malayalam reference clip and `words` as its transcript.
    fn with_reference(models: &Path, x: &[f32], rate: u32, words: &str) -> CrossReference {
        let mut r = prepare_cross_lingual(models, "clone:my-voice", None, Some("placeholder"), &mut |_| Err("unused".into())).unwrap();
        let voices = voices_dir(models);
        crate::wav::write(&voices.join(&r.clip), x, rate).unwrap();
        r.clip_sha = sha_hex(&read_capped(&voices.join(&r.clip)).unwrap());
        r.transcript = words.into();
        r.from = 56.1;
        r.to = 56.1 + x.len() as f64 / f64::from(rate);
        std::fs::write(voices.join("my-voice.indicf5.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
        assert_eq!(cross_reference(models, "clone:my-voice").unwrap(), Some(r.clone()));
        r
    }

    #[test]
    fn a_reference_is_trimmed_to_its_last_sentence_end() {
        let root = tmp("trim");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let rate = CROSS_REFERENCE_RATE;
        // two whole sentences, and a third cut off mid-way (as a clip picked by length is)
        let (x, pauses) = sentences_audio(rate, &[20, 22, 18]);
        let s = sentences_text(&[20, 22, 18]);
        let words = format!("{}. {}. {}", s[0], s[1], s[2]);
        let before = with_reference(&models, &x, rate, &words);
        let original = std::fs::read(voices.join(&before.clip)).unwrap();
        let r = trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, ..Trim::default() }).unwrap();
        let clip = crate::wav::read(&voices.join(&r.clip)).unwrap().0;
        let cut = clip.len() as f64 / f64::from(rate);
        // at the second sentence's pause (the first would leave less than 8 s)
        assert!((cut - pauses[1]).abs() < 0.08, "cut at {cut:.3} s, pause at {:.3} s", pauses[1]);
        assert_eq!(r.transcript, format!("{}. {}.", s[0], s[1]));
        assert!((r.from - 56.1).abs() < 1e-9 && (r.to - (56.1 + cut)).abs() < 1e-3, "{} {}", r.from, r.to);
        assert!(r.given);
        assert!(clip.last().is_some_and(|v| v.abs() < 1e-3), "faded out");
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), Some(r.clone()), "a fresh reference");
        // the clip before is kept beside it, once
        let p = load_profile(&voices, "clone:my-voice").unwrap();
        let (keep_wav, keep_json) = untrimmed_paths(&voices, &p).unwrap();
        assert_eq!(std::fs::read(&keep_wav).unwrap(), original);
        assert!(std::fs::read_to_string(&keep_json).unwrap().contains(&before.clip_sha));
        // trimmed again it ends at its own last sentence end, and the first copy stays
        let again = trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, ..Trim::default() }).unwrap();
        assert_eq!(again.transcript, r.transcript);
        assert_eq!(std::fs::read(&keep_wav).unwrap(), original);
        // deleting the voice deletes the copy
        delete(&voices, "clone:my-voice").unwrap();
        assert!(!keep_wav.exists() && !keep_json.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_trim_takes_the_words_it_is_given() {
        let root = tmp("trim-words");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let rate = CROSS_REFERENCE_RATE;
        let (x, pauses) = sentences_audio(rate, &[20, 22, 18]);
        let s = sentences_text(&[20, 22, 18]);
        let words = format!("{}. {}. {}", s[0], s[1], s[2]);
        // the words up to a sentence end: cut there
        with_reference(&models, &x, rate, &words);
        let given = format!("{}.   {}.", s[0], s[1]);
        let r = trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, transcript: Some(given), ..Trim::default() }).unwrap();
        let cut = crate::wav::read(&voices.join(&r.clip)).unwrap().0.len() as f64 / f64::from(rate);
        assert!((cut - pauses[1]).abs() < 0.08, "{cut}");
        assert_eq!(r.transcript, format!("{}. {}.", s[0], s[1]), "spaces made single");
        // a plain cut takes the words given
        with_reference(&models, &x, rate, &words);
        let r = trim_cross_reference(
            &models,
            "clone:my-voice",
            &Trim { start: Some(1.0), end: Some(10.0), transcript: Some("the words".into()), ..Trim::default() },
        )
        .unwrap();
        let clip = crate::wav::read(&voices.join(&r.clip)).unwrap().0;
        assert!((clip.len() as f64 / f64::from(rate) - 9.0).abs() < 0.01);
        assert!(clip.first().is_some_and(|v| v.abs() < 1e-3), "faded in");
        assert_eq!((r.transcript.as_str(), r.from, r.to), ("the words", 57.1, 66.1));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hostile_trims_are_refused_and_change_nothing() {
        let root = tmp("trim-hostile");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let rate = CROSS_REFERENCE_RATE;
        assert!(trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, ..Trim::default() }).is_err(), "no reference yet");
        let (x, _) = sentences_audio(rate, &[20, 22, 18]);
        let s = sentences_text(&[20, 22, 18]);
        let r = with_reference(&models, &x, rate, &format!("{}. {}. {}", s[0], s[1], s[2]));
        let dur = x.len() as f64 / f64::from(rate);
        for t in [
            Trim::default(),
            Trim { start: Some(f64::NAN), to_sentence: true, ..Trim::default() },
            Trim { end: Some(f64::INFINITY), transcript: Some("w".into()), ..Trim::default() },
            Trim { start: Some(-1.0), transcript: Some("w".into()), ..Trim::default() },
            Trim { end: Some(dur + 5.0), transcript: Some("w".into()), ..Trim::default() },
            Trim { start: Some(5.0), end: Some(4.0), transcript: Some("w".into()), ..Trim::default() },
            // a plain cut without the words it keeps
            Trim { end: Some(10.0), ..Trim::default() },
            // the start cuts words away
            Trim { start: Some(0.5), to_sentence: true, ..Trim::default() },
            // too short
            Trim { end: Some(5.0), transcript: Some("w".into()), ..Trim::default() },
            Trim { end: Some(9.0), to_sentence: true, ..Trim::default() },
            // empty words
            Trim { end: Some(10.0), transcript: Some("   ".into()), ..Trim::default() },
            Trim { end: Some(10.0), transcript: Some("x".repeat(MAX_TRANSCRIPT_CHARS + 1)), ..Trim::default() },
            // given words that end a sentence too early
            Trim { to_sentence: true, transcript: Some(format!("{}.", s[0])), ..Trim::default() },
        ] {
            assert!(trim_cross_reference(&models, "clone:my-voice", &t).is_err(), "{t:?}");
            assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), Some(r.clone()), "{t:?}");
        }
        // a transcript with no sentence end, or a silent clip: no sentence to cut at
        with_reference(&models, &x, rate, "ka ka ka ka");
        assert!(trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, ..Trim::default() }).is_err());
        with_reference(&models, &vec![0.0; rate as usize * 12], rate, "ka. ka.");
        assert!(trim_cross_reference(&models, "clone:my-voice", &Trim { to_sentence: true, ..Trim::default() }).is_err());
        assert!(!voices.join("my-voice.indicf5.untrimmed.wav").exists(), "nothing trimmed, nothing kept");
        assert!(trim_cross_reference(&models, "clone:nobody", &Trim { to_sentence: true, ..Trim::default() }).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sentence_ends_and_levels() {
        assert_eq!(sentence_ends("One. Two? Three! Four"), vec![4, 9, 16]);
        assert_eq!(sentence_ends("ഇത്. അതും। ശരി..."), vec![4, 10, 17]);
        assert_eq!(sentence_ends("3.5 kilos. Done"), vec![10]);
        assert!(sentence_ends("").is_empty());
        let lv = frame_levels(&[0.0; 4800], 24_000);
        assert_eq!(lv.len(), 10);
        assert!(lv.iter().all(|l| l.is_finite() && *l < -100.0));
        assert!(frame_levels(&[], 24_000).is_empty());
        assert!(frame_levels(&[f32::NAN; 960], 24_000).iter().all(|l| l.is_finite()));
    }

    /// The speaking rate is held within 4.6–6 syllables a second, whatever the transcript did.
    #[cfg(feature = "indicf5")]
    #[test]
    fn the_malayalam_speaking_rate_is_held() {
        use malayalam_voice::{SYLLABLE_RATE, rate_correction};
        let line = "ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി."; // 26 syllables
        assert_eq!(rate_correction(line, 4.63), 1.0, "5.6 a second is natural");
        assert!((rate_correction(line, 3.48) - SYLLABLE_RATE.1 / (26.0 / 3.48)).abs() < 1e-9, "too fast: slowed");
        assert!((rate_correction(line, 6.54) - SYLLABLE_RATE.0 / (26.0 / 6.54)).abs() < 1e-9, "too slow: quickened");
        assert_eq!(rate_correction("ശരി", 0.2), 1.0, "too short to judge");
        assert_eq!(rate_correction(line, f64::NAN), 1.0);
        assert_eq!(rate_correction(line, 0.0), 1.0);
        assert_eq!(rate_correction(line, 1e-3), 1.0);
        assert_eq!(rate_correction(line, 1e9), 2.0, "bounded");
    }

    /// The Malayalam voice on a tiny random IndicF5: scripts with pauses, takes, pace, pitch.
    #[cfg(feature = "indicf5")]
    #[test]
    fn a_malayalam_voice_speaks_scripts() {
        let root = tmp("ml-voice");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let r = prepare_cross_lingual(&models, "clone:my-voice", None, Some("hello there, this is my voice"), &mut |_| Err(String::new())).unwrap();
        let dir = root.join("f5");
        crate::f5::model::tests::fake_release(&dir);
        let m = Arc::new(crate::f5::F5::load_on(&dir, &candle_core::Device::Cpu).unwrap());
        let p = load_profile(&voices, "clone:my-voice").unwrap();
        let reading = Reading { steps: 2, ..Default::default() };
        let reference = malayalam_voice::reference(&p, &r, &m, &models, &reading).unwrap();

        let v = malayalam_voice::MalayalamVoice::new(Gender::Male, m.clone(), reference, reading.clone());
        assert_eq!(v.info().language, "ml-IN");
        let script = "ഇത് 2013 ആണ്. [pause 500ms] ശരി.";
        let a = v.synthesize(script, &Params::default()).unwrap();
        assert_eq!(a.sample_rate, crate::f5::RATE);
        assert!(a.samples.iter().all(|x| x.is_finite() && x.abs() <= 0.98));
        // the pause marker is exact: no other pause after it
        assert!(a.seconds() > 0.5 + 0.08, "{}", a.seconds());
        assert_eq!(v.synthesize(script, &Params::default()).unwrap(), a, "deterministic");
        assert_ne!(v.synthesize(script, &Params { take: 1, ..Default::default() }).unwrap(), a, "another take reads afresh");
        let fast = v.synthesize(script, &Params { pace: 2.0, ..Default::default() }).unwrap();
        assert!(fast.seconds() < a.seconds(), "{} vs {}", fast.seconds(), a.seconds());
        let high = v.synthesize(script, &Params { semitones: 3.0, ..Default::default() }).unwrap();
        assert_ne!(high, a);
        assert_eq!(v.synthesize("", &Params::default()), Err(TtsError::Empty));
        assert_eq!(v.synthesize("[pause 1s]", &Params::default()), Err(TtsError::Empty));
        assert!(v.synthesize("ഇത്", &Params { pace: 9.0, ..Default::default() }).is_err());

        // the strict check: a recogniser that hears the first reading of every utterance garbled
        struct Fake(std::sync::atomic::AtomicU32, Vec<String>);
        impl crate::heard::Hearer for Fake {
            fn hear(&self, _: &[f32], _: u32) -> Result<String, String> {
                let n = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize;
                Ok(if n.is_multiple_of(2) { "ഹ്മ്".into() } else { self.1[(n / 2) % self.1.len()].clone() })
            }
            fn id(&self) -> String {
                "fake".into()
            }
        }
        let says: Vec<String> =
            v.pieces(script).unwrap().into_iter().filter_map(|p| if let crate::malayalam::Piece::Say(s) = p { Some(s) } else { None }).collect();
        let hearer = Fake(Default::default(), says);
        let strict = crate::heard::Strict::new(&hearer, crate::heard::Policy { max_takes: 3, max_cer: 0.2, max_start_cer: 0.2 });
        let (checked, report) = v.synthesize_checked(script, &Params::default(), &strict).unwrap();
        assert_eq!(report.len(), 2, "one report per utterance");
        for (c, i) in report.iter().zip([0usize, 2]) {
            assert_eq!((c.takes, c.kept, c.passed), (2, 1, true), "{c:?}");
            // the kept reading's seed is its own: the second attempt of utterance i
            assert_eq!(c.seed, malayalam_voice::MalayalamVoice::seed(0, i, 1));
            assert_eq!(c.tries[0].seed, malayalam_voice::MalayalamVoice::seed(0, i, 0));
        }
        assert_ne!(checked, a, "retaken utterances read afresh");
        // seeds never collide between utterances, attempts and takes
        let mut seeds: Vec<u64> = (0..4).flat_map(|t| (0..40).flat_map(move |i| (0..8).map(move |k| malayalam_voice::MalayalamVoice::seed(t, i, k)))).collect();
        let n = seeds.len();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), n);

        // a reading with each sentence its own utterance, a pause of room tone after the clip and
        // a lead-in cut away
        let reading = Reading {
            steps: 2,
            per_sentence: true,
            tail_ms: 300,
            room_tone: true,
            lead_in: Some("ശരി.".into()),
            english_transcript: false,
            ..Default::default()
        };
        let reference = malayalam_voice::reference(&p, &r, &m, &models, &reading).unwrap();
        let w = malayalam_voice::MalayalamVoice::new(Gender::Male, m, reference, reading);
        let two = "ഇത് ശരിയാണ്. അത് തെറ്റാണ്.";
        assert_eq!(w.pieces(two).unwrap().len(), 2);
        assert_eq!(v.pieces(two).unwrap().len(), 1, "packed");
        let b = w.synthesize(two, &Params::default()).unwrap();
        assert!(!b.samples.is_empty() && b.samples.iter().all(|x| x.is_finite()));
        assert!(Reading { tail_ms: 5000, ..Default::default() }.check().is_err());
        assert!(Reading { steps: 0, ..Default::default() }.check().is_err());
        assert!(Reading { lead_in: Some("ക".repeat(200)), ..Default::default() }.check().is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Voice styles on a tiny random IndicF5: a delivery reads after its style's reference (or
    /// the nearest one the voice has) with its preset; tags switch styles within a script and
    /// are never read; plain readings are unchanged by styles; pauses follow the punctuation and
    /// the style; `[breath]` is room tone; the strict check hears each part against its own words.
    #[cfg(feature = "indicf5")]
    #[test]
    fn a_malayalam_voice_reads_in_styles() {
        let root = tmp("ml-styles");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let r = prepare_cross_lingual(&models, "clone:my-voice", None, Some("hello there, this is my voice"), &mut |_| Err(String::new())).unwrap();
        // an excited reference: another stretch of the speaker
        let x = bursts(24_000, &[(1.5, 0.2); 5]);
        let n = NewStyle {
            style: "excited",
            audio: &x,
            rate: 24_000,
            source: "IMG_9.MOV [3–11.5 s]",
            from: 3.0,
            to: 11.5,
            transcript: "ഇത് കൊള്ളാം, ശരിക്കും കൊള്ളാം",
            given: true,
            to_sentence: false,
        };
        set_style(&models, "clone:my-voice", &n).unwrap();
        let dir = root.join("f5");
        crate::f5::model::tests::fake_release(&dir);
        let m = Arc::new(crate::f5::F5::load_on(&dir, &candle_core::Device::Cpu).unwrap());
        let p = load_profile(&voices, "clone:my-voice").unwrap();
        let reading = Reading { steps: 2, ..Default::default() };
        let main = malayalam_voice::reference(&p, &r, &m, &models, &reading).unwrap();
        let sr = style_reference(&models, "clone:my-voice", "excited").unwrap().unwrap();
        let excited = malayalam_voice::reference(&p, &sr, &m, &models, &reading).unwrap();
        let plain = malayalam_voice::MalayalamVoice::new(Gender::Male, m.clone(), main.clone(), reading.clone());
        let room = vec![0.01f32; 4800];
        let styled = malayalam_voice::MalayalamVoice::new(Gender::Male, m.clone(), main.clone(), reading.clone()).with_styles(vec![("excited", excited)], room);
        assert_eq!(styled.styles(), ["excited"]);
        let one = "ഇത് ശരിയാണ്.";
        let say = |v: &malayalam_voice::MalayalamVoice, text: &str, q: Params| v.synthesize(text, &q).unwrap();
        // plain reads exactly as before, styles or not
        let base = say(&plain, one, Params::default());
        assert_eq!(say(&styled, one, Params::default()), base);
        // excited: after its own reference, with its preset
        let ex = say(&styled, one, Params { delivery: Some("excited"), ..Default::default() });
        let ex_main = say(&plain, one, Params { delivery: Some("excited"), ..Default::default() });
        assert!(ex != base && ex != ex_main && ex_main != base, "the reference and the preset both count");
        // amazed has no reference of its own: it reads after the excited one (its preset made equal)
        let like_excited = Params { delivery: Some("amazed"), temperature: Some(1.1), emphasis: Some(2.4), speed: Some(1.06), ..Default::default() };
        assert_eq!(say(&styled, one, like_excited), ex);
        // a tag does what a delivery does, and is not read
        assert_eq!(say(&styled, "[excited] ഇത് ശരിയാണ്.", Params::default()), ex);
        assert_eq!(say(&styled, "[Plain]ഇത് ശരിയാണ്.", Params { delivery: Some("excited"), ..Default::default() }), base);
        // an explicit setting beats the preset; the reading is deterministic
        let cooler = say(&styled, one, Params { delivery: Some("excited"), temperature: Some(0.7), ..Default::default() });
        assert_ne!(cooler, ex);
        assert_eq!(say(&styled, one, Params { delivery: Some("excited"), temperature: Some(0.7), ..Default::default() }), cooler);
        assert!(styled.synthesize(one, &Params { temperature: Some(9.0), ..Default::default() }).is_err());
        assert!(styled.synthesize(one, &Params { delivery: Some("angry"), ..Default::default() }).is_err());
        // natural pauses: longer at a line break than the fixed 250 ms, shorter excited than calm
        let two = "ഇത് ശരിയാണ്.\nഅത് തെറ്റാണ്.";
        let fixed = malayalam_voice::MalayalamVoice::new(Gender::Male, m.clone(), main.clone(), Reading { natural_pauses: false, ..reading.clone() });
        let (nat, old) = (say(&plain, two, Params::default()), say(&fixed, two, Params::default()));
        let extra = (nat.samples.len() as f64 - old.samples.len() as f64) / 24_000.0;
        assert!((0.17..=0.29).contains(&extra), "a line break pauses 422–538 ms, not 250: {extra}");
        let same = |d: &'static str| Params { delivery: Some(d), temperature: Some(1.0), emphasis: Some(2.0), speed: Some(1.0), ..Default::default() };
        let (quick, slow) = (say(&plain, two, same("excited")), say(&plain, two, same("calm")));
        assert!(quick.samples.len() < slow.samples.len());
        // a breath is the clip's room tone; without one, silence
        let breathe = say(&styled, "ഇത് ശരിയാണ്. [breath] അത് തെറ്റാണ്.", Params::default());
        assert!(breathe.samples.iter().any(|v| (v.abs() - 0.01).abs() < 1e-6), "room tone");
        assert!(plain.synthesize("ഇത് ശരിയാണ്. [breath] അത് തെറ്റാണ്.", &Params::default()).is_ok());
        // the strict check hears each part against its own words, and says how it was read
        struct Deaf;
        impl crate::heard::Hearer for Deaf {
            fn hear(&self, _: &[f32], _: u32) -> Result<String, String> {
                Ok(String::new())
            }
            fn id(&self) -> String {
                "deaf".into()
            }
        }
        let strict = crate::heard::Strict::new(&Deaf, crate::heard::Policy { max_takes: 2, max_cer: 0.2, max_start_cer: 0.2 });
        let q = Params { delivery: Some("excited"), ..Default::default() };
        let (_, report) = styled.synthesize_checked("ഇത് ശരിയാണ്. [calm] അത് തെറ്റാണ്!", &q, &strict).unwrap();
        assert_eq!(report.len(), 2);
        assert_eq!((report[0].text.as_str(), report[1].text.as_str()), ("ഇത് ശരിയാണ്.", "അത് തെറ്റാണ്!"));
        assert_eq!(report[0].style.as_deref(), Some("excited"));
        assert_eq!(report[1].style.as_deref(), Some("calm (after the main reference)"));
        assert!(report.iter().all(|c| c.takes == 2), "retakes keep the part's settings and read again");
        let (_, plain_report) = plain.synthesize_checked(one, &Params::default(), &strict).unwrap();
        assert_eq!(plain_report[0].style, None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// What the narration cache keys a Malayalam reading on beyond the voice: nothing for a plain
    /// reading of one utterance; the style references a reading uses (and nothing it does not);
    /// natural pauses when it has more than one utterance.
    #[cfg(feature = "indicf5")]
    #[test]
    fn the_reading_identity_follows_the_styles_used() {
        let root = tmp("ml-identity");
        let models = natural_clone(&root);
        prepare_cross_lingual(&models, "clone:my-voice", None, Some("hello there, this is my voice"), &mut |_| Err(String::new())).unwrap();
        let id = |text: &str, q: Params| malayalam_reading_identity(&models, "clone:my-voice", text, &q).unwrap();
        let one = "ഇത് ശരിയാണ്.";
        assert_eq!(id(one, Params::default()), "", "a plain reading of one utterance keeps its key");
        assert_eq!(id("ഇത് ശരിയാണ്. അത് തെറ്റാണ്.", Params::default()), "", "two sentences in one utterance");
        assert_eq!(id("ഇത് ശരിയാണ്.\nഅത് തെറ്റാണ്.", Params::default()), format!("pauses v{}", crate::styles::VERSION));
        assert_eq!(id("ഇത് ശരിയാണ്. [breath]", Params::default()), format!("pauses v{}", crate::styles::VERSION));
        let excited = Params { delivery: Some("excited"), ..Default::default() };
        assert_eq!(id(one, excited), format!("styles v{}", crate::styles::VERSION), "a preset, the main reference");
        // an excited reference: excited (and amazed) readings key on it; calm and plain do not
        let x = bursts(24_000, &[(1.5, 0.2); 5]);
        let n =
            NewStyle {
                style: "excited", audio: &x, rate: 24_000, source: "x", from: 0.0, to: 8.5, transcript: "ഇത് കൊള്ളാം", given: true, to_sentence: false
            };
        let set = set_style(&models, "clone:my-voice", &n).unwrap();
        let with = id(one, excited);
        assert!(with.contains(&set.reference.clip_sha), "{with}");
        assert!(id(one, Params { delivery: Some("amazed"), ..Default::default() }).contains(&set.reference.clip_sha));
        assert!(id("[excited] ഇത് ശരിയാണ്.", Params::default()).contains(&set.reference.clip_sha));
        assert_eq!(id(one, Params { delivery: Some("calm"), ..Default::default() }), format!("styles v{}", crate::styles::VERSION));
        assert_eq!(id(one, Params::default()), "", "plain still keeps its key");
        // new words for the style: another identity
        let n2 = NewStyle { transcript: "ഇത് ശരിക്കും കൊള്ളാം", ..n.clone() };
        set_style(&models, "clone:my-voice", &n2).unwrap();
        assert_ne!(id(one, excited), with);
        // the cache's view: English unchanged, Malayalam extended only when there is something
        let en = crate::cache::voice_identity("clone:my-voice", Some(&models)).unwrap();
        assert_eq!(crate::cache::reading_identity_for("clone:my-voice", "en-US", Some(&models), "Hello.", &excited), Some(en));
        let ml = crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)).unwrap();
        assert_eq!(crate::cache::reading_identity_for("clone:my-voice", "ml-IN", Some(&models), one, &Params::default()), Some(ml.clone()));
        assert_ne!(crate::cache::reading_identity_for("clone:my-voice", "ml-IN", Some(&models), one, &excited), Some(ml));
        // hostile params are refused, never keyed
        assert!(malayalam_reading_identity(&models, "clone:my-voice", one, &Params { speed: Some(f32::NAN), ..Default::default() }).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The strict check's retakes: new seeds first, then more sampling steps; never the lead-in
    /// (measured harmful).
    #[cfg(feature = "indicf5")]
    #[test]
    fn retakes_read_more_carefully() {
        let r = Reading::default();
        assert!(!r.english_transcript, "the reference's words reach IndicF5 as written");
        assert_eq!((r.for_attempt(0), r.for_attempt(1)), (r.clone(), r.clone()));
        assert_eq!((r.for_attempt(2).steps, r.for_attempt(2).lead_in.clone()), (64, None));
        assert!((0..10).all(|a| r.for_attempt(a).lead_in.is_none()));
        assert_eq!(LEAD_IN, "ശരി.");
        assert_eq!(Reading { steps: 48, ..r.clone() }.for_attempt(2).steps, 64, "at most 64");
        for a in 0..10 {
            assert!(r.for_attempt(a).check().is_ok());
            assert_eq!(r.for_attempt(a).reference_key(), r.reference_key(), "every attempt speaks after the same reference");
        }
        assert!(Reading::before_fix().english_transcript);
    }

    /// Regression: the reference's Malayalam words went through the English front end, which
    /// dropped every virama ("ഫൈവ് പോയിന്റ് ത്രീ" became "ഫൈവ പോയിന റ ത രീ"): the words no longer
    /// matched the clip, and IndicF5 said the reference's last word again at the start of every
    /// utterance (heard as "ഇല്ല ഇവിടെ നിന്നാണ്…", the clip ending "…ഇല്ല.").
    #[cfg(feature = "indicf5")]
    #[test]
    fn the_reference_words_reach_indicf5_intact() {
        let t = "ഏഹ്, ഫൈവ് പോയിന്റ് ത്രീ ഫ്ലാഷ്. ഈ ഫൈവ് പോയിന്റ് ത്രീക്ക് വിഷൻ കേപ്പബിലിറ്റീസ് ഇല്ല.";
        let w = malayalam_voice::reference_words(t, false).unwrap();
        assert_eq!(w, t);
        // the English front end mangles it (the old reading, kept to measure against)
        let old = malayalam_voice::reference_words(t, true).unwrap();
        assert!(!old.contains('\u{0D4D}') && old != t, "{old}");
        // English words still go through the English front end; Malayalam numbers are spelled
        assert!(malayalam_voice::reference_words("It costs $5.", false).unwrap().contains("five dollars"));
        assert!(malayalam_voice::reference_words("ഇത് 2021 ആണ്.", false).unwrap().contains("ട്വന്റി ട്വന്റി വൺ"));
        assert_eq!(malayalam_voice::reference_words("  ഇത്   ശരി.  ", false).unwrap(), "ഇത് ശരി.");
    }

    /// The pause after the reference: room tone from the clip's quietest stretch, faded; or
    /// silence.
    #[cfg(feature = "indicf5")]
    #[test]
    fn the_reference_pause_is_room_tone_or_silence() {
        let rate = 24_000;
        // loud speech, then a quiet stretch of hiss, then speech
        let mut clip: Vec<f32> = (0..rate).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        clip.extend((0..rate / 2).map(|i| 0.002 * ((i * 7919 % 101) as f32 / 50.0 - 1.0)));
        clip.extend((0..rate).map(|i| 0.3 * (i as f32 * 0.05).sin()));
        let room = malayalam_voice::tail(&clip, rate, 300, true);
        assert_eq!(room.len(), 7200);
        let peak = room.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.0 && peak <= 0.002 + 1e-6, "{peak}");
        assert_eq!((room[0], *room.last().unwrap()), (0.0, room[room.len() - 1]));
        assert!(room.last().unwrap().abs() < 1e-3);
        assert_eq!(malayalam_voice::tail(&clip, rate, 300, false), vec![0.0; 7200]);
        assert!(malayalam_voice::tail(&clip, rate, 0, true).is_empty());
        // a clip shorter than the room tone wanted
        assert_eq!(malayalam_voice::room_tone(&[0.1; 10], rate, 0.2).len(), 4800);
    }

    /// A lead-in is cut at the pause after it, wherever the syllables put the boundary.
    #[cfg(feature = "indicf5")]
    #[test]
    fn a_lead_in_is_cut_at_its_pause() {
        let rate = 24_000u32;
        let tone = |secs: f64| -> Vec<f32> { (0..(secs * f64::from(rate)) as usize).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect() };
        // 0.4 s of lead-in, a 0.3 s pause, then 2 s of speech
        let mut a = tone(0.4);
        a.extend(std::iter::repeat_n(0.0, (0.3 * f64::from(rate)) as usize));
        a.extend(tone(2.0));
        let cut = malayalam_voice::lead_in_cut(&a, rate, "ശരി.", "ഇത് ഒരു നീണ്ട വാക്യമാണ്, ശരിക്കും.");
        let t = cut as f64 / f64::from(rate);
        assert!((0.66..=0.70).contains(&t), "{t}");
        // no pause: cut where the syllables say, never past the end
        let flat = tone(2.0);
        assert!(malayalam_voice::lead_in_cut(&flat, rate, "ശരി.", "ഇത്.") < flat.len());
        assert_eq!(malayalam_voice::lead_in_cut(&[], rate, "ശരി.", "ഇത്."), 0);
    }

    #[test]
    fn malayalam_needs_a_cloned_voice() {
        let root = tmp("ml-route");
        let models = natural_clone(&root);
        // built-in and Kokoro voices speak English only
        let e = crate::voice_for("basic-female", "ml-IN", Some(&models)).err().unwrap();
        assert!(e.to_string().contains("cloned voices"), "{e}");
        // a clone without its Malayalam reference says how to make it
        if let Err(e) = crate::voice_for("clone:my-voice", "ml", Some(&models)) {
            assert!(e.to_string().contains("tts.voiceReference") || e == TtsError::NotInstalled || e.to_string().contains("indicf5"), "{e}");
        }
        // English routing is unchanged
        assert!(crate::voice_for("basic-female", "en-US", Some(&models)).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn create_refuses_without_consent_or_name() {
        let d = tmp("refuse");
        let e = create("Me", &[0.0; 10], 16_000, None, "", false, &d, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("consent"), "{e}");
        assert!(create("!!!", &[0.0; 10], 16_000, None, "", true, &d, &mut |_| {}).is_err());
        assert!(create("Me", &[0.0; 10], 16_000, Some("nope"), "", true, &d, &mut |_| {}).is_err());
        let req = NaturalRequest::default();
        let mut tr = |_: &[f32]| Ok("hello".to_string());
        let e = create_natural("Me", &[0.0; 10], 16_000, &req, "", false, &d, &mut tr, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("consent"), "{e}");
        assert!(create_natural("!!!", &[0.0; 10], 16_000, &req, "", true, &d, &mut tr, &mut |_| {}).is_err());
        let bad = NaturalRequest { model: "voxcpm-9".into(), ..req.clone() };
        assert!(create_natural("Me", &[0.0; 10], 16_000, &bad, "", true, &d, &mut tr, &mut |_| {}).is_err());
        let bad = NaturalRequest { cfg: f32::NAN, ..req.clone() };
        assert!(create_natural("Me", &[0.0; 10], 16_000, &bad, "", true, &d, &mut tr, &mut |_| {}).is_err());
        // not installed
        assert_eq!(create_natural("Me", &[0.0; 10], 16_000, &req, "", true, &d, &mut tr, &mut |_| {}), Err(TtsError::NotInstalled));
    }

    #[test]
    fn cosine_is_bounded() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
