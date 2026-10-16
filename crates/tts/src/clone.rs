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
    let min_pause = (0.1 * fps).ceil() as usize;
    let mut cuts: Vec<(usize, f64)> = vec![(0, 0.0)];
    let mut excess = 0.0;
    for (s, len) in runs.iter().take(20_000) {
        if *len >= min_pause {
            cuts.push((s + len / 2, excess));
            excess += (*len as f64 / fps - 0.7).max(0.0);
        }
    }
    cuts.push((n, excess));
    let mut speech_before = Vec::with_capacity(n + 1);
    let mut count = 0usize;
    speech_before.push(0);
    for m in &mask {
        count += usize::from(*m);
        speech_before.push(count);
    }
    let (lo, hi) = SAMPLE_SECONDS;
    // a short recording: everything between its first and last speech
    let lo = lo.min(speech_total * 0.9);
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

/// Speaker gender from the median pitch of the voiced 40 ms frames (autocorrelation at 8 kHz;
/// 165 Hz splits the usual ranges). `None` with too little voiced speech.
pub fn estimate_gender(x: &[f32], rate: u32) -> Option<Gender> {
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
    let median = f0s.get(f0s.len() / 2).copied()?;
    Some(if median > 165.0 { Gender::Female } else { Gender::Male })
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
    let encode_rate = natural_encode_rate(pkg.id);
    progress("finding the best stretch of speech");
    let sample = pick_sample(reference, rate)?;
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
    let voices = voices_dir(models_dir);
    std::fs::create_dir_all(&voices).map_err(|e| TtsError::Model(format!("{}: {e}", voices.display())))?;
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
            model: pkg.id.to_string(),
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
    save(&voices, &profile)?;
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
    use crate::voxcpm::{Prompt, Settings, VoxCpm, text};

    /// Silence between utterances, ms.
    const UTTERANCE_GAP_MS: u64 = 250;
    /// Prompts kept built (encoding a sample takes seconds).
    const PROMPTS: usize = 4;

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
            let mut out: Vec<f32> = Vec::new();
            let mut spoke = false;
            for (i, piece) in text::pieces(script)?.into_iter().enumerate() {
                match piece {
                    text::Piece::Pause(ms) => out.resize(out.len().saturating_add((u64::from(ms) * u64::from(rate) / 1000) as usize), 0.0),
                    text::Piece::Say(u) => {
                        if spoke {
                            out.resize(out.len().saturating_add(gap), 0.0);
                        }
                        let a = self.model.speak(&u, Some(&self.prompt), &Settings { seed: s.seed.wrapping_add(i as u64), ..s }, &mut |_, _| true)?;
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
