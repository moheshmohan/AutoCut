//! Text to speech for FilmCraft's Text to Speech panel (narrations).
//!
//! - [`Voice`]: the trait every voice implements: a script in, mono samples out.
//! - [`voices`] / [`voice`]: the voice catalogue (id, name, language, gender, engine, licence).
//! - [`script`]: splits a script into speech and pause markers (`[pause 1s]`, `[pause 500ms]`).
//! - [`formant`]: the built-in voices, an original source–filter formant synthesizer. They need
//!   no download and work everywhere (also in the web build); they sound robotic. The neural voices
//!   (Kokoro-82M, downloaded on first use) are added behind a feature in a later milestone.
//! - `voxcpm` (feature `voxcpm`): VoxCPM zero-shot voice cloning on candle — speech generated
//!   directly in a speaker's voice from a few seconds of it.
//! - [`wav`]: WAV reading and writing for voice samples.
//! - [`heard`]: the strict heard-back check of Malayalam narration (character error rates of
//!   what a recogniser heard, sentence starts scored on their own) and its retake policy.
//! - `voice_lab` (feature `kokoro`): better references for a cloned voice, found in the
//!   speaker's own clips and auditioned (signal quality, whole sentences, speaker similarity).
//! - [`cache`]: synthesized narrations kept on disk (keyed by script, voice identity, pitch and
//!   pace), so a script is read by a voice once across sessions.
//! - [`styles`]: voice styles of Malayalam narration — which reference a delivery reads after
//!   (and the fallbacks), each style's sampler preset (temperature, emphasis, speed), inline
//!   `[excited]` / `[calm]` / `[breath]` tags and natural pauses between utterances.
//!
//! Everything is deterministic: the same script and settings give the same samples.

pub mod cache;
pub mod catalog;
#[cfg(feature = "kokoro")]
pub mod clone;
#[cfg(feature = "indicf5")]
pub mod f5;
pub mod formant;
pub mod heard;
#[cfg(feature = "kokoro")]
pub mod kokoro;
pub mod malayalam;
pub mod ml_facts;
pub mod ml_script;
#[cfg(feature = "kokoro")]
pub mod neural;
#[cfg(feature = "kokoro")]
pub mod openvoice;
pub mod script;
pub mod styles;
#[cfg(feature = "kokoro")]
pub mod voice_lab;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;
pub mod wav;

use serde::Serialize;

/// Voice ids of cloned voices start with this (`clone:<name>`); see `clone` (feature `kokoro`).
pub const CLONE_PREFIX: &str = "clone:";

/// Sample rate of the built-in and Kokoro voices (and timbre-only clones); natural clones speak at
/// their model's rate (44.1 or 48 kHz): every `Audio` carries its own.
pub const SAMPLE_RATE: u32 = 24_000;
/// Longest narration a voice produces, in seconds.
pub const MAX_SECONDS: f64 = 30.0 * 60.0;
/// Longest script accepted, in bytes (same as `filmcraft_project::narration::MAX_TEXT_BYTES`).
pub const MAX_TEXT_BYTES: usize = 64 * 1024;
/// Pace limits.
pub const MIN_PACE: f64 = 0.5;
pub const MAX_PACE: f64 = 2.0;
/// Pitch shift limit in semitones (either way).
pub const MAX_SEMITONES: f64 = 12.0;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TtsError {
    #[error("unknown voice `{0}`")]
    UnknownVoice(String),
    #[error("the script has nothing to say")]
    Empty,
    #[error("the script is longer than {} KB", MAX_TEXT_BYTES / 1024)]
    TextTooLong,
    #[error("the narration would be longer than {} minutes", MAX_SECONDS / 60.0)]
    AudioTooLong,
    #[error("{0}")]
    Invalid(String),
    #[error("voice model: {0}")]
    Model(String),
    #[error("the natural voices are not downloaded yet")]
    NotInstalled,
    #[error("natural voices are not available in this build (built without the `kokoro` feature)")]
    Unavailable,
}

/// How a voice speaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Pitch shift in semitones, ±[`MAX_SEMITONES`].
    pub semitones: f64,
    /// Speaking rate, [`MIN_PACE`]–[`MAX_PACE`] (1 = normal).
    pub pace: f64,
    /// How to speak, in plain words ("warm, friendly, smiling"), for voices that take a spoken
    /// style (the natural cloned voices); the others ignore it.
    pub style: Option<&'static str>,
    /// Which reading (0 the first): voices that vary between readings (the natural cloned
    /// voices) read the same script afresh for each take; the others ignore it.
    pub take: u32,
    /// The voice style of the line ([`styles::STYLES`]; `None`: plain), for voices that keep a
    /// reference per style (Malayalam through IndicF5): it picks the reference read after and the
    /// sampler preset. The others ignore it (they take `style`).
    pub delivery: Option<&'static str>,
    /// IndicF5's sampler, over the style's preset ([`styles::preset`]): the initial-noise scale
    /// ([`styles::TEMPERATURE`]), classifier-free guidance ([`styles::EMPHASIS`]) and speed on top
    /// of `pace` ([`styles::SPEED`]). `None`: the preset's (or, plain, the model's own).
    pub temperature: Option<f32>,
    pub emphasis: Option<f32>,
    pub speed: Option<f32>,
    /// The pitch range of the reading after synthesis ([`styles::EXPRESSIVENESS`]; 1 leaves the
    /// audio exactly as synthesized), over the style's preset: the intonation reshaped
    /// ([`styles::reshaping`]; Malayalam through IndicF5).
    pub expressiveness: Option<f32>,
}

impl Default for Params {
    fn default() -> Self {
        Params { semitones: 0.0, pace: 1.0, style: None, take: 0, delivery: None, temperature: None, emphasis: None, speed: None, expressiveness: None }
    }
}

impl Params {
    pub fn check(&self) -> Result<(), TtsError> {
        if !(MIN_PACE..=MAX_PACE).contains(&self.pace) {
            return Err(TtsError::Invalid(format!("pace must be between {MIN_PACE} and {MAX_PACE}")));
        }
        if !(-MAX_SEMITONES..=MAX_SEMITONES).contains(&self.semitones) {
            return Err(TtsError::Invalid(format!("pitch must be within ±{MAX_SEMITONES} semitones")));
        }
        if let Some(d) = self.delivery
            && styles::find(d).is_none()
        {
            return Err(TtsError::Invalid(format!("unknown voice style `{d}` ({})", styles::STYLES.join(", "))));
        }
        for (what, v, range) in [
            ("temperature", self.temperature, styles::TEMPERATURE),
            ("emphasis", self.emphasis, styles::EMPHASIS),
            ("speed", self.speed, styles::SPEED),
            ("expressiveness", self.expressiveness, styles::EXPRESSIVENESS),
        ] {
            if let Some(v) = v {
                styles::check(what, v, range).map_err(TtsError::Invalid)?;
            }
        }
        Ok(())
    }

    /// The sampler `style` is read with: its preset (none for plain), then what these params ask
    /// for explicitly. `None`: nothing differs from the reading's own settings.
    pub fn sampler(&self, style: Option<&str>) -> Option<styles::Sampler> {
        let preset = styles::preset(style);
        if preset.is_none() && self.temperature.is_none() && self.emphasis.is_none() && self.speed.is_none() && self.expressiveness.is_none() {
            return None;
        }
        let base = preset.unwrap_or(styles::Sampler::PLAIN);
        Some(styles::Sampler {
            temperature: self.temperature.unwrap_or(base.temperature),
            emphasis: self.emphasis.unwrap_or(base.emphasis),
            speed: self.speed.unwrap_or(base.speed),
            expressiveness: self.expressiveness.unwrap_or(base.expressiveness),
        })
    }
}

/// Mono audio at [`SAMPLE_RATE`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate.max(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Gender {
    Female,
    Male,
}

/// A voice in the catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInfo {
    pub id: &'static str,
    pub name: &'static str,
    /// BCP 47 tag.
    pub language: &'static str,
    pub gender: Gender,
    /// `basic` (built in) or `neural` (downloaded).
    pub engine: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    /// Usable now (built in, or downloaded).
    pub installed: bool,
}

/// A voice that turns a script into audio.
pub trait Voice: Send + Sync {
    fn info(&self) -> VoiceInfo;
    /// Speak `text` (pause markers allowed). Fails on an empty or over-long script, or bad params.
    fn synthesize(&self, text: &str, params: &Params) -> Result<Audio, TtsError>;
    /// [`Voice::synthesize`] under the strict heard-back check ([`heard`]): a voice that reads
    /// utterance by utterance (the Malayalam voice) hears each and reads it again until it is
    /// clear, and reports what was heard; the others read once and report nothing.
    fn synthesize_checked(&self, text: &str, params: &Params, strict: &heard::Strict) -> Result<(Audio, Vec<heard::Checked>), TtsError> {
        let _ = strict;
        Ok((self.synthesize(text, params)?, Vec::new()))
    }
}

/// Languages offered (English only for now).
pub const LANGUAGES: &[(&str, &str)] = &[("en-US", "English (United States)")];

/// This build can speak with the natural (neural) voices once their package is downloaded.
pub const NATURAL_VOICES: bool = cfg!(feature = "kokoro");

/// Every voice FilmCraft knows: the built-in voices, then the neural voices (installed when their
/// package is downloaded into `models_dir` and this build has them).
pub fn voices_in(models_dir: Option<&std::path::Path>) -> Vec<VoiceInfo> {
    let installed = models_dir.is_some_and(catalog::installed) && NATURAL_VOICES;
    formant::VOICES.iter().map(|v| v.info).chain(catalog::VOICES.iter().map(|v| VoiceInfo { installed, ..v.info })).collect()
}

/// The built-in voices (always available).
pub fn voices() -> Vec<VoiceInfo> {
    formant::VOICES.iter().map(|v| v.info).collect()
}

/// The voice with this id.
pub fn voice(id: &str) -> Result<Box<dyn Voice>, TtsError> {
    formant::VOICES
        .iter()
        .find(|v| v.info.id == id)
        .map(|v| Box::new(formant::FormantVoice::new(*v)) as Box<dyn Voice>)
        .ok_or_else(|| TtsError::UnknownVoice(id.to_string()))
}

/// The voice `id`: built in, or (feature `kokoro`) a neural voice from the package in `models_dir`.
pub fn voice_in(id: &str, models_dir: Option<&std::path::Path>) -> Result<Box<dyn Voice>, TtsError> {
    if id.starts_with(CLONE_PREFIX) {
        #[cfg(feature = "kokoro")]
        {
            let dir = models_dir.ok_or(TtsError::NotInstalled)?;
            return clone::load(id, dir);
        }
        #[cfg(not(feature = "kokoro"))]
        {
            return Err(TtsError::Unavailable);
        }
    }
    if catalog::find(id).is_some() {
        #[cfg(feature = "kokoro")]
        {
            let dir = models_dir.ok_or(TtsError::NotInstalled)?;
            return neural::load(id, dir).map(|v| Box::new(v) as Box<dyn Voice>);
        }
        #[cfg(not(feature = "kokoro"))]
        {
            let _ = models_dir;
            return Err(TtsError::Unavailable);
        }
    }
    voice(id)
}

/// The languages this build speaks: English, and Malayalam (cloned voices through IndicF5,
/// feature `indicf5`).
pub fn languages() -> Vec<(&'static str, &'static str)> {
    let mut l = LANGUAGES.to_vec();
    if cfg!(feature = "indicf5") {
        l.push((malayalam::TAG, "Malayalam (India)"));
    }
    l
}

/// The voice `id` speaking `language` (a BCP 47 tag): English as [`voice_in`]; Malayalam
/// ([`malayalam::is_tag`]) through a cloned voice's IndicF5 reading (feature `indicf5`).
pub fn voice_for(id: &str, language: &str, models_dir: Option<&std::path::Path>) -> Result<Box<dyn Voice>, TtsError> {
    if !malayalam::is_tag(language) {
        return voice_in(id, models_dir);
    }
    if !id.starts_with(CLONE_PREFIX) {
        return Err(TtsError::Invalid(format!(
            "`{id}` speaks English; Malayalam is spoken by cloned voices (`clone:<name>`, made from a recording of the speaker)"
        )));
    }
    #[cfg(feature = "kokoro")]
    {
        let dir = models_dir.ok_or(TtsError::NotInstalled)?;
        clone::load_malayalam(id, dir)
    }
    #[cfg(not(feature = "kokoro"))]
    {
        let _ = models_dir;
        Err(TtsError::Unavailable)
    }
}

/// The voice used when none is chosen.
pub fn default_voice_id() -> &'static str {
    formant::VOICES.first().map(|v| v.info.id).unwrap_or("basic-female")
}

/// The sentence "Hear this voice" plays.
pub const SAMPLE_SENTENCE: &str = "Hello! This is how I sound when I read your script.";

/// Validate a script before synthesis: length, and that it says something.
pub fn check_text(text: &str) -> Result<(), TtsError> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(TtsError::TextTooLong);
    }
    let says_something = script::parse(text).iter().any(|s| matches!(s, script::Segment::Speech(t) if t.chars().any(char::is_alphanumeric)));
    if !says_something {
        return Err(TtsError::Empty);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_ids_are_unique_and_resolvable() {
        let v = voices();
        assert!(v.len() >= 2);
        for (i, a) in v.iter().enumerate() {
            assert!(v.iter().skip(i + 1).all(|b| b.id != a.id), "duplicate {}", a.id);
            assert_eq!(voice(a.id).unwrap().info().id, a.id);
        }
        assert_eq!(voice("nope").err(), Some(TtsError::UnknownVoice("nope".into())));
        assert!(voice(default_voice_id()).is_ok());
    }

    #[test]
    fn params_are_checked() {
        assert!(Params::default().check().is_ok());
        for p in [
            Params { pace: f64::NAN, ..Default::default() },
            Params { pace: 0.1, ..Default::default() },
            Params { semitones: 40.0, ..Default::default() },
            Params { semitones: f64::INFINITY, ..Default::default() },
            Params { delivery: Some("angry"), ..Default::default() },
            Params { temperature: Some(0.1), ..Default::default() },
            Params { temperature: Some(f32::NAN), ..Default::default() },
            Params { emphasis: Some(4.5), ..Default::default() },
            Params { speed: Some(0.0), ..Default::default() },
            Params { speed: Some(f32::INFINITY), ..Default::default() },
            Params { expressiveness: Some(0.4), ..Default::default() },
            Params { expressiveness: Some(2.5), ..Default::default() },
            Params { expressiveness: Some(f32::NAN), ..Default::default() },
        ] {
            assert!(p.check().is_err(), "{p:?}");
        }
        let ok = Params { delivery: Some("excited"), temperature: Some(0.3), emphasis: Some(4.0), speed: Some(1.4), ..Default::default() };
        assert!(ok.check().is_ok());
    }

    #[test]
    fn the_sampler_is_the_preset_under_what_is_asked() {
        let p = Params::default();
        assert_eq!(p.sampler(None), None, "plain: the reading's own settings");
        assert_eq!(p.sampler(Some("excited")), styles::preset(Some("excited")));
        let q = Params { temperature: Some(0.7), ..Default::default() };
        assert_eq!(q.sampler(None), Some(styles::Sampler { temperature: 0.7, ..styles::Sampler::PLAIN }));
        let calm = styles::preset(Some("calm")).unwrap();
        assert_eq!(q.sampler(Some("calm")), Some(styles::Sampler { temperature: 0.7, ..calm }), "explicit beats the preset");
        let e = Params { expressiveness: Some(1.6), ..Default::default() };
        assert_eq!(e.sampler(None), Some(styles::Sampler { expressiveness: 1.6, ..styles::Sampler::PLAIN }));
        assert_eq!(e.sampler(Some("excited")).map(|s| (s.expressiveness, s.emphasis)), Some((1.6, 2.4)));
    }

    #[test]
    fn empty_scripts_are_refused() {
        assert_eq!(check_text(""), Err(TtsError::Empty));
        assert_eq!(check_text("  [pause 1s] ... "), Err(TtsError::Empty));
        assert_eq!(check_text(&"a".repeat(MAX_TEXT_BYTES + 1)), Err(TtsError::TextTooLong));
        assert!(check_text("Hi").is_ok());
    }
}
