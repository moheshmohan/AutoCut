//! Narrations: audio generated from text (the Text to Speech panel).
//!
//! A [`Narration`] belongs to the project item of the WAV file that was generated for it
//! (`Project::narrations`, keyed by the item id), so every clip of that item can be clicked to
//! reopen the script and voice settings. Editing a narration writes a new WAV (a new item) and
//! swaps the clip over to it; the old item and its narration stay, so undo restores them exactly.

use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};

/// Longest script accepted, in bytes.
pub const MAX_TEXT_BYTES: usize = 64 * 1024;
/// Pace limits (speaking rate multiplier).
pub const MIN_PACE: f64 = 0.5;
pub const MAX_PACE: f64 = 2.0;
/// The sampler settings a Malayalam (IndicF5) narration may ask for, inclusive (the same as
/// `filmcraft_tts::styles`): temperature (the initial noise's scale), emphasis (classifier-free
/// guidance) and speed (on top of the pace).
pub const TEMPERATURE: (f64, f64) = (0.3, 1.5);
pub const EMPHASIS: (f64, f64) = (1.0, 4.0);
pub const SPEED: (f64, f64) = (0.7, 1.4);

/// Text to Speech ▸ Advanced ▸ Vocal pitch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VocalPitch {
    ExtraLow,
    Low,
    #[default]
    Default,
    High,
    ExtraHigh,
}

impl VocalPitch {
    pub const ALL: [VocalPitch; 5] = [VocalPitch::ExtraLow, VocalPitch::Low, VocalPitch::Default, VocalPitch::High, VocalPitch::ExtraHigh];

    pub fn label(self) -> &'static str {
        match self {
            VocalPitch::ExtraLow => "Extra low",
            VocalPitch::Low => "Low",
            VocalPitch::Default => "Default",
            VocalPitch::High => "High",
            VocalPitch::ExtraHigh => "Extra high",
        }
    }

    /// The command / serde id (`extraLow`, `low`, `default`, `high`, `extraHigh`).
    pub fn id(self) -> &'static str {
        match self {
            VocalPitch::ExtraLow => "extraLow",
            VocalPitch::Low => "low",
            VocalPitch::Default => "default",
            VocalPitch::High => "high",
            VocalPitch::ExtraHigh => "extraHigh",
        }
    }

    pub fn from_id(s: &str) -> Option<VocalPitch> {
        let n = s.to_ascii_lowercase().replace([' ', '_', '-'], "");
        Self::ALL.into_iter().find(|p| p.id().to_ascii_lowercase() == n)
    }

    /// Pitch shift in semitones.
    pub fn semitones(self) -> f64 {
        match self {
            VocalPitch::ExtraLow => -6.0,
            VocalPitch::Low => -3.0,
            VocalPitch::Default => 0.0,
            VocalPitch::High => 3.0,
            VocalPitch::ExtraHigh => 6.0,
        }
    }
}

/// Text to Speech ▸ Advanced ▸ Delivery: the feeling a voice speaks with. Voices that take a
/// spoken style (the natural cloned voices) are given [`Delivery::style`]; the others read the
/// script as they always do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Delivery {
    /// The voice's own reading.
    #[default]
    Plain,
    Conversational,
    Warm,
    Excited,
    Amazed,
    Curious,
    Calm,
    Dramatic,
}

impl Delivery {
    pub const ALL: [Delivery; 8] =
        [Delivery::Plain, Delivery::Conversational, Delivery::Warm, Delivery::Excited, Delivery::Amazed, Delivery::Curious, Delivery::Calm, Delivery::Dramatic];

    pub fn label(self) -> &'static str {
        match self {
            Delivery::Plain => "Plain",
            Delivery::Conversational => "Conversational",
            Delivery::Warm => "Warm",
            Delivery::Excited => "Excited",
            Delivery::Amazed => "Amazed",
            Delivery::Curious => "Curious",
            Delivery::Calm => "Calm",
            Delivery::Dramatic => "Dramatic",
        }
    }

    /// The command / serde id (`plain`, `conversational`, `warm`, …).
    pub fn id(self) -> &'static str {
        match self {
            Delivery::Plain => "plain",
            Delivery::Conversational => "conversational",
            Delivery::Warm => "warm",
            Delivery::Excited => "excited",
            Delivery::Amazed => "amazed",
            Delivery::Curious => "curious",
            Delivery::Calm => "calm",
            Delivery::Dramatic => "dramatic",
        }
    }

    pub fn from_id(s: &str) -> Option<Delivery> {
        let n = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|d| d.id() == n)
    }

    /// How the voice is asked to speak (a short style description in plain words); `None`
    /// for [`Delivery::Plain`].
    pub fn style(self) -> Option<&'static str> {
        match self {
            Delivery::Plain => None,
            Delivery::Conversational => Some("natural, conversational, relaxed, like talking to a friend"),
            Delivery::Warm => Some("warm, friendly, smiling, natural storytelling"),
            // not "excited, energetic": that overacted (a higher, strained voice)
            Delivery::Excited => Some("enthusiastic, cheerful, natural storytelling"),
            Delivery::Amazed => Some("amazed, impressed, lively, natural storytelling"),
            Delivery::Curious => Some("curious, playful, light, natural storytelling"),
            Delivery::Calm => Some("calm, gentle, reflective, unhurried"),
            Delivery::Dramatic => Some("dramatic, deep, suspenseful, cinematic trailer voice"),
        }
    }
}

/// How a Malayalam narration says numbers written in digits (`filmcraft_tts::malayalam::Numbers`):
/// years, money and numbers from 100 up in English, in Malayalam script, as Malayali speakers
/// mix them (the default), or every number in Malayalam words. English narrations ignore it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NumberStyle {
    #[default]
    English,
    Malayalam,
}

impl NumberStyle {
    pub const ALL: [NumberStyle; 2] = [NumberStyle::English, NumberStyle::Malayalam];

    pub fn label(self) -> &'static str {
        match self {
            NumberStyle::English => "English words",
            NumberStyle::Malayalam => "Malayalam words",
        }
    }

    /// The command / serde id (`english`, `malayalam`).
    pub fn id(self) -> &'static str {
        match self {
            NumberStyle::English => "english",
            NumberStyle::Malayalam => "malayalam",
        }
    }

    pub fn from_id(s: &str) -> Option<NumberStyle> {
        let n = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|d| d.id() == n)
    }
}

/// The script and voice settings a narration was generated from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Narration {
    /// The script, with pause markers (`[pause 1s]`) as typed.
    pub text: String,
    /// BCP 47 language tag (`en-US`).
    pub language: String,
    /// Voice id (`tts.voices`).
    pub voice: String,
    #[serde(default)]
    pub pitch: VocalPitch,
    /// Speaking rate, [`MIN_PACE`]–[`MAX_PACE`].
    pub pace: f64,
    /// The feeling it was spoken with (projects from before have none: plain).
    #[serde(default, skip_serializing_if = "is_plain")]
    pub delivery: Delivery,
    /// Which reading of the same script (0 the first): voices that vary between readings
    /// (the natural cloned voices) read it afresh for another take; at most [`MAX_TAKE`].
    #[serde(default, skip_serializing_if = "is_zero")]
    pub take: u32,
    /// How a Malayalam narration says numbers (projects from before have none: English).
    #[serde(default, skip_serializing_if = "is_english")]
    pub numbers: NumberStyle,
    /// A Malayalam narration's sampler settings over its delivery's preset ([`TEMPERATURE`],
    /// [`EMPHASIS`], [`SPEED`]); `None`: the preset's (projects from before have none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emphasis: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Length of the speech in the generated file. The file can be longer: when an edit made the
    /// speech shorter than its clip, the file is padded with silence to the clip's length.
    pub speech_duration: Tick,
}

fn is_plain(d: &Delivery) -> bool {
    *d == Delivery::Plain
}

fn is_english(n: &NumberStyle) -> bool {
    *n == NumberStyle::English
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// Highest take number.
pub const MAX_TAKE: u32 = 99;

impl Narration {
    /// Check values read from a project file or a command.
    pub fn check(&self) -> Result<(), String> {
        if self.text.len() > MAX_TEXT_BYTES {
            return Err(format!("the script is longer than {} KB", MAX_TEXT_BYTES / 1024));
        }
        if !(MIN_PACE..=MAX_PACE).contains(&self.pace) {
            return Err(format!("pace must be between {MIN_PACE} and {MAX_PACE}"));
        }
        if self.take > MAX_TAKE {
            return Err(format!("take must be at most {MAX_TAKE}"));
        }
        for (what, v, (lo, hi)) in [("temperature", self.temperature, TEMPERATURE), ("emphasis", self.emphasis, EMPHASIS), ("speed", self.speed, SPEED)] {
            if let Some(v) = v
                && !(v.is_finite() && (lo..=hi).contains(&v))
            {
                return Err(format!("{what} must be a number from {lo} to {hi}"));
            }
        }
        if self.speech_duration < Tick::ZERO {
            return Err("negative speech duration".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Narration {
        Narration {
            text: "Hello there. [pause 1s] Welcome.".into(),
            language: "en-US".into(),
            voice: "basic-female".into(),
            pitch: VocalPitch::High,
            pace: 1.25,
            delivery: Delivery::Warm,
            take: 2,
            numbers: NumberStyle::Malayalam,
            temperature: Some(0.8),
            emphasis: None,
            speed: Some(1.05),
            speech_duration: Tick::from_seconds_f64(2.5),
        }
    }

    #[test]
    fn sampler_settings_round_trip_and_old_projects_read_none() {
        let s = serde_json::to_string(&sample()).unwrap();
        assert!(s.contains("\"temperature\":0.8") && s.contains("\"speed\":1.05") && !s.contains("emphasis"), "{s}");
        let mut old: serde_json::Value = serde_json::from_str(&s).unwrap();
        for k in ["temperature", "speed"] {
            old.as_object_mut().unwrap().remove(k);
        }
        let n: Narration = serde_json::from_value(old).unwrap();
        assert_eq!((n.temperature, n.emphasis, n.speed), (None, None, None));
        for bad in [
            Narration { temperature: Some(0.2), ..sample() },
            Narration { temperature: Some(f64::NAN), ..sample() },
            Narration { emphasis: Some(4.5), ..sample() },
            Narration { speed: Some(0.0), ..sample() },
            Narration { speed: Some(f64::INFINITY), ..sample() },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
        assert!(Narration { temperature: Some(1.5), emphasis: Some(1.0), speed: Some(0.7), ..sample() }.check().is_ok());
    }

    #[test]
    fn round_trips_through_json() {
        let n = sample();
        let s = serde_json::to_string(&n).unwrap();
        assert!(s.contains("\"pitch\":\"high\""), "{s}");
        assert_eq!(serde_json::from_str::<Narration>(&s).unwrap(), n);
    }

    #[test]
    fn deliveries_round_trip_and_old_projects_read_plain() {
        for d in Delivery::ALL {
            assert_eq!(Delivery::from_id(d.id()), Some(d));
            assert_eq!(d.style().is_none(), d == Delivery::Plain);
        }
        assert_eq!(Delivery::from_id(" Excited "), Some(Delivery::Excited));
        assert_eq!(Delivery::from_id("angry"), None);
        let s = serde_json::to_string(&sample()).unwrap();
        assert!(s.contains("\"delivery\":\"warm\""), "{s}");
        // a narration from before deliveries: plain, and none written back
        let mut old: serde_json::Value = serde_json::from_str(&s).unwrap();
        old.as_object_mut().unwrap().remove("delivery");
        let n: Narration = serde_json::from_value(old).unwrap();
        assert_eq!(n.delivery, Delivery::Plain);
        assert!(!serde_json::to_string(&n).unwrap().contains("delivery"));
    }

    #[test]
    fn number_styles_round_trip_and_old_projects_read_english() {
        for n in NumberStyle::ALL {
            assert_eq!(NumberStyle::from_id(n.id()), Some(n));
        }
        assert_eq!(NumberStyle::from_id(" Malayalam "), Some(NumberStyle::Malayalam));
        assert_eq!(NumberStyle::from_id("roman"), None);
        let s = serde_json::to_string(&sample()).unwrap();
        assert!(s.contains("\"numbers\":\"malayalam\""), "{s}");
        let mut old: serde_json::Value = serde_json::from_str(&s).unwrap();
        old.as_object_mut().unwrap().remove("numbers");
        let n: Narration = serde_json::from_value(old).unwrap();
        assert_eq!(n.numbers, NumberStyle::English);
        assert!(!serde_json::to_string(&n).unwrap().contains("numbers"));
    }

    #[test]
    fn pitch_ids_round_trip() {
        for p in VocalPitch::ALL {
            assert_eq!(VocalPitch::from_id(p.id()), Some(p));
        }
        assert_eq!(VocalPitch::from_id("Extra High"), Some(VocalPitch::ExtraHigh));
        assert_eq!(VocalPitch::from_id("loud"), None);
    }

    #[test]
    fn check_rejects_hostile_values() {
        let mut n = sample();
        n.pace = f64::NAN;
        assert!(n.check().is_err());
        n.pace = 3.0;
        assert!(n.check().is_err());
        let mut n = sample();
        n.text = "a".repeat(MAX_TEXT_BYTES + 1);
        assert!(n.check().is_err());
        let mut n = sample();
        n.speech_duration = Tick(-1);
        assert!(n.check().is_err());
        assert!(sample().check().is_ok());
    }
}
