//! Voice cloning, part 2: cloned narration voices (feature `kokoro`).
//!
//! A cloned voice is a small [`Profile`] saved as JSON in the voices directory: the speaker's
//! tone colour (from their own recorded speech), the Kokoro voice it is built on, and that
//! voice's tone colour. To speak, the base voice reads the script and the OpenVoice converter
//! ([`crate::openvoice`]) re-voices it from the base colour to the speaker's — words, timing and
//! intonation from Kokoro, timbre from the speaker.
//!
//! **Consent.** A profile records that whoever made it confirmed they may clone that voice
//! (their own, or with the speaker's permission); [`create`] refuses without it and [`load`]
//! refuses a profile without it.
//!
//! **Base voice.** Unless one is named, every installed Kokoro voice reads a calibration
//! passage and the one whose tone colour is closest to the speaker's (cosine) is used, so the
//! converter has the least distance to cover.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use serde::{Deserialize, Serialize};

use crate::kokoro::KOKORO_RATE;
use crate::openvoice::{COLOUR, Converter, DEFAULT_TAU, RATE as OV_RATE};
use crate::{Audio, CLONE_PREFIX, Gender, Params, TtsError, Voice, VoiceInfo, catalog, neural};

/// Profile file format version.
pub const PROFILE_VERSION: u32 = 1;
/// Least speech (after trimming silence) a clone is made from.
pub const MIN_SPEECH_SECONDS: f64 = 3.0;
/// Most speech used (the colour is an average; more adds time, not quality).
pub const MAX_SPEECH_SECONDS: f64 = 120.0;
/// Most profiles listed (each file at most 64 KB).
const MAX_PROFILES: usize = 200;
const MAX_PROFILE_BYTES: u64 = 64 * 1024;

/// What the base voices read to measure their tone colour: varied vowels and consonants, about
/// fifteen seconds.
const CALIBRATION: &str = "The quick brown fox jumps over the lazy dog. Every robot we met had its own personality: some were tiny and playful, others were enormous industrial machines. Would you believe how far they have come in only a few years?";

/// A cloned voice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub version: u32,
    /// `clone:<slug>`.
    pub id: String,
    /// What the user called it.
    pub name: String,
    /// The Kokoro voice it speaks through.
    pub base_voice: String,
    /// The base voice's tone colour (256).
    pub source_colour: Vec<f32>,
    /// The speaker's tone colour (256).
    pub target_colour: Vec<f32>,
    /// Cosine similarity of the two colours before conversion (how close the base voice was).
    pub base_similarity: f32,
    /// Seconds of speech the colour was taken from.
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
        if self.version != PROFILE_VERSION {
            return Err(TtsError::Invalid(format!("voice profile version {} is not supported", self.version)));
        }
        if !self.id.starts_with(CLONE_PREFIX) || slug(self.id.trim_start_matches(CLONE_PREFIX)) != self.id.trim_start_matches(CLONE_PREFIX) {
            return Err(TtsError::Invalid(format!("bad cloned voice id `{}`", self.id)));
        }
        if !ok(&self.source_colour) || !ok(&self.target_colour) {
            return Err(TtsError::Invalid("the voice profile's tone colours are damaged".into()));
        }
        if !self.consent {
            return Err(TtsError::Invalid("this cloned voice was saved without consent".into()));
        }
        if catalog::find(&self.base_voice).is_none() {
            return Err(TtsError::UnknownVoice(self.base_voice.clone()));
        }
        Ok(())
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

/// Remove the profile `id`.
pub fn delete(voices_dir: &Path, id: &str) -> Result<(), TtsError> {
    let path = profile_path(voices_dir, id)?;
    std::fs::remove_file(&path).map_err(|_| TtsError::UnknownVoice(id.to_string()))
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

/// Keep the speech of `x` (mono at `rate`): 20 ms frames above an adaptive level, with 200 ms
/// of hang-over so words are not clipped; everything else (silence, room tone) is dropped.
pub fn speech_only(x: &[f32], rate: u32) -> Vec<f32> {
    let frame = (rate as usize / 50).max(1);
    let levels: Vec<f32> = x.chunks(frame).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len().max(1) as f32).sqrt()).collect();
    if levels.is_empty() {
        return Vec::new();
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
    x.chunks(frame).zip(&keep).filter(|(_, k)| **k).flat_map(|(c, _)| c.iter().copied()).collect()
}

/// Make and save a cloned voice from `reference` (mono speech at `rate`). `base`: a Kokoro
/// voice id, or `None` to pick the closest. `progress` gets one line per step.
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
    if !consent {
        return Err(TtsError::Invalid("cloning a voice needs consent: confirm it is your own voice or that the speaker agreed (consent: true)".into()));
    }
    let id_part = slug(name);
    if id_part.is_empty() {
        return Err(TtsError::Invalid("give the cloned voice a name with letters or digits".into()));
    }
    if let Some(b) = base
        && catalog::find(b).is_none()
    {
        return Err(TtsError::UnknownVoice(b.to_string()));
    }
    if rate == 0 {
        return Err(TtsError::Invalid("the reference audio has no sample rate".into()));
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
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let profile = Profile {
        version: PROFILE_VERSION,
        id: format!("{CLONE_PREFIX}{id_part}"),
        name: name.trim().chars().take(80).collect(),
        base_voice: base_voice.to_string(),
        source_colour,
        target_colour: target,
        base_similarity: sim,
        reference_seconds: take as f64 / f64::from(rate),
        source: source.chars().take(500).collect(),
        created,
        consent: true,
    };
    save(&voices_dir(models_dir), &profile)?;
    Ok(profile)
}

/// The tone colour of a Kokoro voice reading the calibration passage.
fn base_colour(id: &str, models_dir: &Path, conv: &Converter) -> Result<Vec<f32>, TtsError> {
    let v = neural::load(id, models_dir)?;
    let a = v.synthesize(CALIBRATION, &Params::default())?;
    let x = filmcraft_audio_dsp::resample::resample_sinc(&a.samples, a.sample_rate, OV_RATE);
    conv.tone_colour(&x)
}

/// A cloned voice ready to speak.
pub struct CloneVoice {
    profile: Profile,
    base: neural::NeuralVoice,
    converter: Arc<Converter>,
}

/// Load the cloned voice `id` (`clone:<slug>`): its profile from the voices directory next to
/// `models_dir`, its base voice and the converter.
pub fn load(id: &str, models_dir: &Path) -> Result<CloneVoice, TtsError> {
    let profile = load_profile(&voices_dir(models_dir), id)?;
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

/// The gender of a profile's base voice.
pub fn gender(p: &Profile) -> Gender {
    catalog::find(&p.base_voice).map(|v| v.info.gender).unwrap_or(Gender::Male)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Profile {
        Profile {
            version: PROFILE_VERSION,
            id: "clone:my-voice".into(),
            name: "My Voice".into(),
            base_voice: "kokoro-michael".into(),
            source_colour: vec![0.1; COLOUR],
            target_colour: vec![0.2; COLOUR],
            base_similarity: 0.8,
            reference_seconds: 30.0,
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
        ] {
            std::fs::write(d.join("my-voice.json"), serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(load_profile(&d, "clone:my-voice").is_err(), "{bad:?}");
        }
        assert!(list(&d).is_empty(), "a bad file is skipped, not fatal");
        assert!(load_profile(&d, "clone:missing").is_err());
        assert!(load_profile(&d, "clone:../../x").is_err());
        delete(&d, "clone:my-voice").unwrap();
        assert!(delete(&d, "clone:my-voice").is_err());
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

    #[test]
    fn create_refuses_without_consent_or_name() {
        let d = tmp("refuse");
        let e = create("Me", &[0.0; 10], 16_000, None, "", false, &d, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("consent"), "{e}");
        assert!(create("!!!", &[0.0; 10], 16_000, None, "", true, &d, &mut |_| {}).is_err());
        assert!(create("Me", &[0.0; 10], 16_000, Some("nope"), "", true, &d, &mut |_| {}).is_err());
    }

    #[test]
    fn cosine_is_bounded() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
