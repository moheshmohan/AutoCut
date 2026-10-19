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
    let _ = std::fs::remove_file(path.with_extension("wav"));
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
    if f0s.len() < 20 {
        return None;
    }
    f0s.sort_by(f64::total_cmp);
    f0s.get(f0s.len() / 2).copied()
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

/// The Malayalam (IndicF5) reference of a natural clone: the stretch of its voice sample IndicF5
/// continues, and that stretch's transcript, which IndicF5 needs. Kept next to the profile
/// (`<name>.indicf5.json`) and made on first use ([`prepare_cross_lingual`]: Whisper hears the
/// transcript) or with a transcript the user gives. It belongs to one voice sample: a voice made
/// again needs a new one. The sample's language does not matter (an English sample speaks
/// Malayalam: cross-lingual prompting).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossReference {
    pub version: u32,
    /// SHA-256 of the voice sample's file it was made from.
    pub sample: String,
    /// The stretch of the sample (seconds).
    pub from: f64,
    pub to: f64,
    pub transcript: String,
    /// The user gave the transcript (else the speech model heard it).
    #[serde(default)]
    pub given: bool,
}

/// Version of [`CrossReference`].
pub const CROSS_REFERENCE_VERSION: u32 = 1;
/// The reference stretch's length: IndicF5 is trained on short prompts, and every second of it
/// is generated again with each utterance (it sets the cost).
pub const CROSS_REFERENCE_SECONDS: (f64, f64) = (8.0, 12.0);

fn cross_path(voices: &Path, p: &Profile) -> Result<PathBuf, TtsError> {
    Ok(profile_path(voices, &p.id)?.with_extension("indicf5.json"))
}

/// SHA-256 (hex) of a natural clone's voice sample file.
fn sample_hash(voices: &Path, p: &Profile) -> Result<String, TtsError> {
    use sha2::Digest;
    let path = sample_path(voices, p).ok_or_else(|| TtsError::Invalid(format!("{} has no voice sample (a natural clone has one)", p.id)))?;
    let meta = std::fs::metadata(&path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    if meta.len() > crate::wav::MAX_BYTES {
        return Err(TtsError::Invalid(format!("{}: too large for a voice sample", path.display())));
    }
    let bytes = std::fs::read(&path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    Ok(sha2::Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect())
}

/// The stretch of a natural clone's voice sample IndicF5 speaks from ([`CROSS_REFERENCE_SECONDS`]
/// between pauses, loudness-normalised): mono at the sample's rate, the rate, and where it is.
pub fn cross_clip(voices: &Path, p: &Profile) -> Result<(Vec<f32>, u32, f64, f64), TtsError> {
    let n = p
        .natural
        .as_ref()
        .ok_or_else(|| TtsError::Invalid(format!("{} is a timbre-only clone; Malayalam needs a natural clone (its voice sample is the reference)", p.id)))?;
    let (x, rate) = crate::wav::read(&voices.join(&n.clip))?;
    let s = pick_sample_within(&x, rate, CROSS_REFERENCE_SECONDS)?;
    Ok((s.audio, rate, s.from, s.to))
}

/// The Malayalam reference of `id`, when one was made from its current voice sample.
pub fn cross_reference(models_dir: &Path, id: &str) -> Result<Option<CrossReference>, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let path = cross_path(&voices, &p)?;
    let Ok(meta) = std::fs::metadata(&path) else { return Ok(None) };
    if meta.len() > MAX_PROFILE_BYTES {
        return Ok(None);
    }
    let Ok(r) = std::fs::read(&path).map_err(|_| ()).and_then(|b| serde_json::from_slice::<CrossReference>(&b).map_err(|_| ())) else {
        return Ok(None);
    };
    let fresh = r.version == CROSS_REFERENCE_VERSION
        && r.sample == sample_hash(&voices, &p)?
        && r.transcript.chars().any(char::is_alphanumeric)
        && r.transcript.chars().count() <= MAX_TRANSCRIPT_CHARS
        && r.from.is_finite()
        && r.to.is_finite();
    Ok(fresh.then_some(r))
}

/// Make the Malayalam reference of `id`: its clip ([`cross_clip`]) and the clip's transcript —
/// `transcript` when given, else what `transcribe` (mono 16 kHz → text) hears.
pub fn prepare_cross_lingual(
    models_dir: &Path,
    id: &str,
    transcript: Option<&str>,
    transcribe: &mut dyn FnMut(&[f32]) -> Result<String, String>,
) -> Result<CrossReference, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let (clip, rate, from, to) = cross_clip(&voices, &p)?;
    let (text, given) = match transcript.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => (t.to_string(), true),
        None => {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(&clip, rate, 16_000);
            let t = transcribe(&x16).map_err(|e| TtsError::Invalid(format!("the voice sample could not be transcribed ({e}); give its transcript")))?;
            (t, false)
        }
    };
    let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TRANSCRIPT_CHARS).collect();
    if !text.chars().any(char::is_alphanumeric) {
        return Err(TtsError::Invalid("no words were heard in the voice sample; give its transcript".into()));
    }
    let r = CrossReference { version: CROSS_REFERENCE_VERSION, sample: sample_hash(&voices, &p)?, from, to, transcript: text, given };
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

/// Cloned voices speaking Malayalam through IndicF5 (feature `indicf5`).
#[cfg(feature = "indicf5")]
mod malayalam_voice {
    use super::*;
    use crate::f5::{self, F5, Reference};

    /// Silence between utterances, ms.
    const UTTERANCE_GAP_MS: u64 = 250;
    /// Seeds between takes.
    const TAKE_SEEDS: u64 = 7919;
    /// References kept built.
    const REFERENCES: usize = 4;

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

    /// The reference IndicF5 speaks after, built once per profile and reference.
    pub(super) fn reference(p: &Profile, r: &CrossReference, m: &F5, voices: &Path) -> Result<Arc<Reference>, TtsError> {
        type Refs = Vec<(String, Arc<Reference>)>;
        static C: OnceLock<Mutex<Refs>> = OnceLock::new();
        let key = format!("{}|{}|{}|{}|{}|{}", p.id, p.created, r.sample, r.from, r.to, r.transcript);
        let cache = C.get_or_init(Default::default);
        if let Some(found) = cache.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()) {
            return Ok(found);
        }
        let (clip, rate, _, _) = cross_clip(voices, p)?;
        let mut x = filmcraft_audio_dsp::resample::resample_sinc(&clip, rate, f5::RATE);
        // 50 ms of silence after the clip, as the reference prepares it
        x.extend(std::iter::repeat_n(0.0, (f5::RATE / 20) as usize));
        let built = Arc::new(m.reference(&x, &english(&r.transcript)?)?);
        let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
        c.retain(|(k, _)| *k != key);
        c.push((key, built.clone()));
        let extra = c.len().saturating_sub(REFERENCES);
        c.drain(..extra);
        Ok(built)
    }

    pub(super) fn load(profile: Profile, models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
        let voices = voices_dir(models_dir);
        let r = cross_reference(models_dir, &profile.id)?
            .ok_or_else(|| TtsError::Invalid(format!("{} has no Malayalam reference yet: make it with tts.voiceReference", profile.id)))?;
        let m = model(models_dir)?;
        let reference = reference(&profile, &r, &m, &voices)?;
        Ok(Box::new(MalayalamVoice { gender: gender(&profile), model: m, reference, base: f5::Settings::default() }))
    }

    /// A cloned voice speaking Malayalam.
    pub struct MalayalamVoice {
        gender: Gender,
        model: Arc<F5>,
        reference: Arc<Reference>,
        /// Steps and guidance (the reference's defaults); speed and seed come per reading.
        base: f5::Settings,
    }

    #[cfg(test)]
    impl MalayalamVoice {
        pub(super) fn new(gender: Gender, model: Arc<F5>, reference: Arc<Reference>, base: f5::Settings) -> MalayalamVoice {
            MalayalamVoice { gender, model, reference, base }
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
            crate::check_text(script)?;
            params.check()?;
            let (pieces, _) = crate::malayalam::pieces(script, self.reference.max_chars())?;
            let rate = f5::RATE;
            let max = (crate::MAX_SECONDS * f64::from(rate)) as usize;
            let gap = (UTTERANCE_GAP_MS * u64::from(rate) / 1000) as usize;
            let _gpu = super::natural::gpu("IndicF5 narration")?;
            let mut out: Vec<f32> = Vec::new();
            let mut spoke = false;
            for (i, piece) in pieces.into_iter().enumerate() {
                match piece {
                    crate::malayalam::Piece::Pause(ms) => out.resize(out.len().saturating_add((u64::from(ms) * u64::from(rate) / 1000) as usize), 0.0),
                    crate::malayalam::Piece::Say(u) => {
                        if spoke {
                            out.resize(out.len().saturating_add(gap), 0.0);
                        }
                        // pace is IndicF5's own speed (the duration it is asked for); each take
                        // reads afresh with its own run of seeds
                        let seed = 1u64.wrapping_add(u64::from(params.take).wrapping_mul(TAKE_SEEDS)).wrapping_add(i as u64);
                        let s = f5::Settings { speed: params.pace as f32, seed, ..self.base };
                        let a = self.model.speak(&self.reference, &u, &s, &mut |_, _| true)?;
                        out.extend_from_slice(super::natural::trim_silence(&a, rate));
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
            let mut out = if params.semitones.abs() > 1e-6 { neural::pitch_shift(out, params.semitones, rate) } else { out };
            let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
            if peak > 0.98 {
                let g = 0.98 / peak;
                out.iter_mut().for_each(|v| *v *= g);
            }
            Ok(Audio { samples: out, sample_rate: rate })
        }
    }
}

#[cfg(not(feature = "indicf5"))]
mod malayalam_voice {
    use super::*;

    pub(super) fn load(_profile: Profile, _models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
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
        let key = format!("{}|{}|{}|{:?}|{}", p.id, p.created, n.model, n.mode, n.denoise_db);
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
            let mut spoke = false;
            for (i, piece) in text::pieces(script)?.into_iter().enumerate() {
                match piece {
                    text::Piece::Pause(ms) => out.resize(out.len().saturating_add((u64::from(ms) * u64::from(rate) / 1000) as usize), 0.0),
                    text::Piece::Say(u) => {
                        if spoke {
                            out.resize(out.len().saturating_add(gap), 0.0);
                        }
                        let said = format!("{style}{u}");
                        // each take reads afresh: its own run of seeds
                        let seed = s.seed.wrapping_add(u64::from(params.take).wrapping_mul(TAKE_SEEDS)).wrapping_add(i as u64);
                        let a = self.model.speak(&said, Some(&self.prompt), &Settings { seed, ..s }, &mut |_, _| true)?;
                        out.extend_from_slice(trim_silence(&a, rate));
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
            Ok(Audio { samples: out, sample_rate: rate })
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
    }
}

#[cfg(not(feature = "voxcpm"))]
mod natural {
    use super::*;

    pub(super) fn load(_profile: Profile, _models_dir: &Path) -> Result<Box<dyn Voice>, TtsError> {
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
        assert!((clip.len() as f64 / 16_000.0 - p.reference_seconds).abs() < 0.01);
        assert!(p.reference_seconds >= 10.0 && p.reference_seconds <= 25.0, "{}", p.reference_seconds);
        // continuation (VoxCPM1.5): transcribed once, denoised when spoken
        let p = save_natural("me", "Me", &x, rate, "voxcpm-1.5", PromptMode::Continuation, &req, "test", &d, &mut tr, &mut |_| {}).unwrap();
        assert!(p.reference_seconds >= 4.0 && p.reference_seconds <= CONTINUATION_SECONDS.1 + 0.5, "a short sample to continue: {}", p.reference_seconds);
        let n = p.natural.unwrap();
        assert_eq!((n.transcript.as_str(), n.denoise_db), ("hello there", CONTINUATION_DENOISE_DB));
        assert_eq!(crate::wav::read(&d.join("me.wav")).unwrap().1, 44_100);
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
        let r = prepare_cross_lingual(&models, "clone:my-voice", None, &mut tr).unwrap();
        assert_eq!((r.transcript.as_str(), r.given), ("hello there", false));
        // the clip Whisper heard (16 kHz) is the 8–12 s stretch
        let secs = heard.get() as f64 / 16_000.0;
        assert!((7.0..=12.5).contains(&secs), "{secs}");
        assert!(r.to - r.from <= CROSS_REFERENCE_SECONDS.1 + 0.5);
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), Some(r.clone()));
        assert!(voices.join("my-voice.indicf5.json").exists());
        // the identity of the Malayalam voice: present now, follows the transcript; English
        // keeps its own (unchanged) identity
        let en = crate::cache::voice_identity("clone:my-voice", Some(&models)).unwrap();
        assert_eq!(crate::cache::voice_identity_for("clone:my-voice", "en-US", Some(&models)), Some(en.clone()));
        let ml = crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)).unwrap();
        assert_ne!(ml, en);
        assert!(ml.contains("indicf5") && ml.contains("ml-IN"), "{ml}");
        let given = prepare_cross_lingual(&models, "clone:my-voice", Some(" exact words "), &mut |_| Err("unused".into())).unwrap();
        assert_eq!((given.transcript.as_str(), given.given), ("exact words", true));
        assert_ne!(crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)).unwrap(), ml);
        // a voice sample made again: the reference is stale
        crate::wav::write(&voices.join("my-voice.wav"), &bursts(16_000, &[(2.5, 0.3); 8]), 16_000).unwrap();
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        assert_eq!(crate::cache::voice_identity_for("clone:my-voice", "ml-IN", Some(&models)), None);
        // nothing heard, a failed transcription, a damaged file
        assert!(prepare_cross_lingual(&models, "clone:my-voice", None, &mut |_| Ok(" ... ".into())).is_err());
        assert!(prepare_cross_lingual(&models, "clone:my-voice", None, &mut |_| Err("no model".into())).is_err());
        std::fs::write(voices.join("my-voice.indicf5.json"), b"{not json").unwrap();
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        // a timbre-only clone has no voice sample
        save(&voices, &Profile { id: "clone:other".into(), ..profile() }).unwrap();
        assert!(prepare_cross_lingual(&models, "clone:other", Some("hi"), &mut |_| Err(String::new())).is_err());
        assert!(cross_reference(&models, "clone:missing").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The Malayalam voice on a tiny random IndicF5: scripts with pauses, takes, pace, pitch.
    #[cfg(feature = "indicf5")]
    #[test]
    fn a_malayalam_voice_speaks_scripts() {
        let root = tmp("ml-voice");
        let models = natural_clone(&root);
        let voices = voices_dir(&models);
        let r = prepare_cross_lingual(&models, "clone:my-voice", Some("hello there, this is my voice"), &mut |_| Err(String::new())).unwrap();
        let dir = root.join("f5");
        crate::f5::model::tests::fake_release(&dir);
        let m = Arc::new(crate::f5::F5::load_on(&dir, &candle_core::Device::Cpu).unwrap());
        let p = load_profile(&voices, "clone:my-voice").unwrap();
        let reference = malayalam_voice::reference(&p, &r, &m, &voices).unwrap();
        let v = malayalam_voice::MalayalamVoice::new(Gender::Male, m, reference, crate::f5::Settings { steps: 2, ..Default::default() });
        assert_eq!(v.info().language, "ml-IN");
        let script = "ഇത് 2013 ആണ്. [pause 500ms] ശരി.";
        let a = v.synthesize(script, &Params::default()).unwrap();
        assert_eq!(a.sample_rate, crate::f5::RATE);
        assert!(a.samples.iter().all(|x| x.is_finite() && x.abs() <= 0.98));
        assert!(a.seconds() > 0.5 + 0.25, "{}", a.seconds());
        assert_eq!(v.synthesize(script, &Params::default()).unwrap(), a, "deterministic");
        assert_ne!(v.synthesize(script, &Params { take: 1, ..Default::default() }).unwrap(), a, "another take reads afresh");
        let fast = v.synthesize(script, &Params { pace: 2.0, ..Default::default() }).unwrap();
        assert!(fast.seconds() < a.seconds(), "{} vs {}", fast.seconds(), a.seconds());
        let high = v.synthesize(script, &Params { semitones: 3.0, ..Default::default() }).unwrap();
        assert_ne!(high, a);
        assert_eq!(v.synthesize("", &Params::default()), Err(TtsError::Empty));
        assert_eq!(v.synthesize("[pause 1s]", &Params::default()), Err(TtsError::Empty));
        assert!(v.synthesize("ഇത്", &Params { pace: 9.0, ..Default::default() }).is_err());
        let _ = std::fs::remove_dir_all(&root);
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
