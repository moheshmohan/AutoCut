//! Voice styles of a natural cloned voice speaking Malayalam: one IndicF5 reference per style
//! ([`crate::styles::STYLES`]) beside the voice's main (plain) Malayalam reference. IndicF5 reads in
//! the emotion and prosody of the clip it continues, so an excited stretch of the speaker's own
//! speech reads excited lines, a calm one calm lines (F5-TTS's multi-style generation).
//!
//! - **Where**: `<voices>/<name>.indicf5.<style>.wav` (mono, 24 kHz) and `.json` (a
//!   [`CrossReference`]), checked as the main reference is: the clip's file name and SHA-256, and
//!   the voice sample it belongs to (a voice made again needs its styles made again; a promoted
//!   sample keeps them, [`super::promote`]).
//! - **Made** by [`set_style`] from a stretch of the speaker's recording and its words: cut at the
//!   last sentence end the words imply (IndicF5 garbles the start of every line after a clip that
//!   stops mid-sentence), faded, [`STYLE_SECONDS`] long.
//! - **Versions**: setting or removing a style keeps the voice's references as a version first
//!   ([`super::versions`]; [`super::rollback`] goes back) and moves the profile's revision on.
//! - **The narration cache** keys a styled reading on the style references it read after, so
//!   plain readings keep their entries and a styled one never reuses a plain one.

use std::path::{Path, PathBuf};

use super::versioning::{archive, next_revision, put, put_wav};
use super::*;

/// How long a style reference is, seconds: long enough to carry the voice (the main reference's
/// measurements: 8 s and more is best), short enough to leave room for a line in one generation.
pub const STYLE_SECONDS: (f64, f64) = (5.0, 16.0);

/// A stretch of the speaker's recording to make a style reference from.
#[derive(Clone, Debug, PartialEq)]
pub struct NewStyle<'a> {
    /// The style ([`crate::styles::STYLES`]).
    pub style: &'a str,
    /// The stretch, mono, at `rate`.
    pub audio: &'a [f32],
    pub rate: u32,
    /// Where it is from (`IMG_1234.MOV [12.0–24.5 s]`), and its times in that recording.
    pub source: &'a str,
    pub from: f64,
    pub to: f64,
    /// Its words. IndicF5 aligns its reading to them: they must be exactly what is said.
    pub transcript: &'a str,
    /// The user gave the words (else speech models heard them).
    pub given: bool,
    /// End the clip at the last sentence end its words imply (when they have one).
    pub to_sentence: bool,
}

/// What [`set_style`] made.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleSet {
    pub style: &'static str,
    pub reference: CrossReference,
    /// The voice's references before it, kept.
    pub version: Version,
    /// Seconds cut off the end of the stretch to end at a sentence end.
    pub trimmed: f64,
    /// Something to look at (the words had no sentence end to cut at, …).
    pub warning: Option<String>,
}

/// The style id of `style`, or why it is not one.
fn style_id(style: &str) -> Result<&'static str, TtsError> {
    crate::styles::find(style).ok_or_else(|| {
        let plain = matches!(style.trim().to_ascii_lowercase().as_str(), "plain" | "neutral");
        TtsError::Invalid(if plain {
            "the plain style is the voice's main Malayalam reference: set it with tts.voiceReference".into()
        } else {
            format!("unknown voice style `{}` ({})", style.chars().take(40).collect::<String>(), crate::styles::STYLES.join(", "))
        })
    })
}

/// The reference file and clip name of `style` for `p`.
fn style_paths(voices: &Path, p: &Profile, style: &str) -> Result<(PathBuf, String), TtsError> {
    let base = profile_path(voices, &p.id)?;
    let stem = base.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    Ok((base.with_extension(format!("indicf5.{style}.json")), format!("{stem}.indicf5.{style}.wav")))
}

/// The reference of `id`'s style `style`, when one was made for its current voice sample and its
/// clip is intact.
pub fn style_reference(models_dir: &Path, id: &str, style: &str) -> Result<Option<CrossReference>, TtsError> {
    let style = style_id(style)?;
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let (path, clip) = style_paths(&voices, &p, style)?;
    fresh_reference(&voices, &p, &path, &clip)
}

/// The styles whose reference files are next to `p` (not checked: a quick look for listings;
/// [`style_references`] checks them).
pub fn styles_on_disk(voices: &Path, p: &Profile) -> Vec<&'static str> {
    crate::styles::STYLES.into_iter().filter(|s| style_paths(voices, p, s).is_ok_and(|(json, clip)| json.is_file() && voices.join(clip).is_file())).collect()
}

/// Every style `id` has a reference for, in [`crate::styles::STYLES`] order.
pub fn style_references(models_dir: &Path, id: &str) -> Result<Vec<(&'static str, CrossReference)>, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    if p.natural.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for style in crate::styles::STYLES {
        let (path, clip) = style_paths(&voices, &p, style)?;
        if let Some(r) = fresh_reference(&voices, &p, &path, &clip)? {
            out.push((style, r));
        }
    }
    Ok(out)
}

/// Make `n.style`'s reference of the natural cloned voice `id` from a stretch of the speaker's
/// recording (see the module docs). The voice's references are kept as a version first.
pub fn set_style(models_dir: &Path, id: &str, n: &NewStyle) -> Result<StyleSet, TtsError> {
    let style = style_id(n.style)?;
    let voices = voices_dir(models_dir);
    let mut p = load_profile(&voices, id)?;
    if p.natural.is_none() {
        return Err(TtsError::Invalid(format!("{id} is a timbre-only clone; Malayalam needs a natural clone")));
    }
    let sample = sample_hash(&voices, &p)?;
    if n.rate == 0 || n.audio.is_empty() || n.audio.iter().any(|v| !v.is_finite()) {
        return Err(TtsError::Invalid("the style's recording is empty or damaged".into()));
    }
    if !(n.from.is_finite() && n.to.is_finite() && n.from >= 0.0 && n.to > n.from) {
        return Err(TtsError::Invalid("the style's times make no sense".into()));
    }
    let words: String = n.transcript.split_whitespace().collect::<Vec<_>>().join(" ");
    if !words.chars().any(char::is_alphanumeric) || words.chars().count() > MAX_TRANSCRIPT_CHARS {
        return Err(TtsError::Invalid("a style reference needs the clip's exact words (at most 4000 characters)".into()));
    }
    let longest = (STYLE_SECONDS.1 + 30.0) * f64::from(n.rate);
    if n.audio.len() as f64 > longest {
        return Err(TtsError::Invalid(format!("the stretch is longer than {:.0} s: give the part with the style", STYLE_SECONDS.1 + 30.0)));
    }
    let rate = CROSS_REFERENCE_RATE;
    let x = filmcraft_audio_dsp::resample::resample_sinc(n.audio, n.rate, rate);
    let dur = x.len() as f64 / f64::from(rate);
    // end at the last sentence end the words imply, when they have one
    let (end, words, warning) = if n.to_sentence {
        match sentence_cut(&x, rate, &words, 0.0, dur, STYLE_SECONDS.0, None) {
            Ok((cut, k)) => (cut, words.chars().take(k).collect::<String>().trim().to_string(), None),
            Err(_) => (
                x.len(),
                words,
                Some(format!(
                    "the words have no sentence end that leaves {:.0} s to cut at: the clip is kept whole; if it stops mid-sentence, IndicF5 may garble the start of lines read in this style",
                    STYLE_SECONDS.0
                )),
            ),
        }
    } else {
        (x.len(), words, None)
    };
    let kept = end as f64 / f64::from(rate);
    if !(STYLE_SECONDS.0..=STYLE_SECONDS.1).contains(&kept) {
        return Err(TtsError::Invalid(format!("a style reference is {:.0}–{:.0} s of speech; this one is {kept:.1} s", STYLE_SECONDS.0, STYLE_SECONDS.1)));
    }
    let mut clip: Vec<f32> = x.get(..end).unwrap_or(&[]).to_vec();
    let peak = clip.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak < 1e-3 {
        return Err(TtsError::Invalid("the style's recording is silent".into()));
    }
    fade_edges(&mut clip, rate, true);
    let version = archive(&voices, &p, &format!("style {style} set from {}", n.source.chars().take(200).collect::<String>()))?;
    let (path, name) = style_paths(&voices, &p, style)?;
    put_wav(&voices.join(&name), &clip, rate)?;
    let reference = CrossReference {
        version: CROSS_REFERENCE_VERSION,
        sample,
        source: n.source.chars().take(500).collect(),
        clip_sha: sha_hex(&read_capped(&voices.join(&name))?),
        clip: name,
        from: n.from,
        to: n.from + kept,
        transcript: words,
        given: n.given,
    };
    put(&path, &serde_json::to_vec_pretty(&reference).map_err(|e| TtsError::Model(e.to_string()))?)?;
    p.revision = next_revision(models_dir, id, &p)?;
    save(&voices, &p)?;
    Ok(StyleSet { style, reference, version, trimmed: (dur - kept).max(0.0), warning })
}

/// Remove `id`'s reference of `style` (its lines then read after the nearest style it has, else
/// the main reference). The voice's references are kept as a version first.
pub fn remove_style(models_dir: &Path, id: &str, style: &str) -> Result<Version, TtsError> {
    let style = style_id(style)?;
    let voices = voices_dir(models_dir);
    let mut p = load_profile(&voices, id)?;
    let (path, clip) = style_paths(&voices, &p, style)?;
    if !path.exists() && !voices.join(&clip).exists() {
        return Err(TtsError::Invalid(format!("{id} has no {style} reference")));
    }
    let version = archive(&voices, &p, &format!("style {style} removed"))?;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(voices.join(&clip));
    p.revision = next_revision(models_dir, id, &p)?;
    save(&voices, &p)?;
    Ok(version)
}

/// Tie every style reference of `p` to its current voice sample (a promoted sample keeps them).
pub(super) fn retie(voices: &Path, p: &Profile) -> Result<(), TtsError> {
    let sample = sample_hash(voices, p)?;
    for style in crate::styles::STYLES {
        let (path, _) = style_paths(voices, p, style)?;
        if let Some(mut r) = read_reference(&path)
            && r.sample != sample
        {
            r.sample = sample.clone();
            put(&path, &serde_json::to_vec_pretty(&r).map_err(|e| TtsError::Model(e.to_string()))?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::natural_profile;
    use super::*;

    fn data(name: &str) -> (PathBuf, PathBuf) {
        let d = std::env::temp_dir().join(format!("filmcraft-styles-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (models, voices) = (d.join("models"), d.join("voices"));
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&voices).unwrap();
        (models, voices)
    }

    /// Sentences of `n` syllables (180 ms buzzes with dips between), 300 ms pauses between them.
    fn sentences(rate: u32, sentences: &[usize]) -> Vec<f32> {
        let mut x: Vec<f32> = Vec::new();
        let push = |x: &mut Vec<f32>, secs: f64, amp: f32| {
            for _ in 0..(secs * f64::from(rate)).round() as usize {
                let t = x.len() as f32 / rate as f32;
                x.push(amp * (std::f32::consts::TAU * 150.0 * t).sin());
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
                push(&mut x, 0.3, 0.003);
            }
        }
        x
    }

    fn voice(voices: &Path) -> Profile {
        let p = natural_profile();
        save(voices, &p).unwrap();
        crate::wav::write(&voices.join("my-voice.wav"), &sentences(16_000, &[30, 30]), 16_000).unwrap();
        p
    }

    fn words(n: &[usize]) -> String {
        n.iter().map(|k| format!("{}.", vec!["ക"; *k].join(" "))).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn a_style_is_set_listed_versioned_and_removed() {
        let (models, voices) = data("set");
        voice(&voices);
        assert!(style_references(&models, "clone:my-voice").unwrap().is_empty());
        // two whole sentences and a third cut off: ends at the second sentence's end
        let x = sentences(44_100, &[20, 22, 10]);
        let w = format!("{} {}", words(&[20, 22]), ["ക"; 10].join(" "));
        let n = NewStyle {
            style: "Excited",
            audio: &x,
            rate: 44_100,
            source: "IMG_1.MOV [10.0–24.0 s]",
            from: 10.0,
            to: 24.0,
            transcript: &w,
            given: true,
            to_sentence: true,
        };
        let set = set_style(&models, "clone:my-voice", &n).unwrap();
        assert_eq!(set.style, "excited");
        assert_eq!(set.reference.transcript, words(&[20, 22]));
        assert!(set.trimmed > 1.5 && set.warning.is_none(), "{set:?}");
        assert_eq!(set.reference.clip, "my-voice.indicf5.excited.wav");
        let (clip, rate) = crate::wav::read(&voices.join("my-voice.indicf5.excited.wav")).unwrap();
        assert_eq!(rate, CROSS_REFERENCE_RATE);
        assert!((set.reference.to - set.reference.from - clip.len() as f64 / 24_000.0).abs() < 1e-3);
        assert!(clip.last().is_some_and(|v| v.abs() < 1e-3) && clip.first().is_some_and(|v| v.abs() < 1e-3), "faded");
        assert_eq!(style_reference(&models, "clone:my-voice", "excited").unwrap(), Some(set.reference.clone()));
        let all = style_references(&models, "clone:my-voice").unwrap();
        assert_eq!(all.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ["excited"]);
        // the voice before was kept; the profile moved on
        assert_eq!(set.version.dir, "r0");
        let p = load_profile(&voices, "clone:my-voice").unwrap();
        assert_eq!(p.revision, 1);
        // the main reference and the listing are untouched
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap(), None);
        assert_eq!(list(&voices).len(), 1, "style files are not profiles");
        // a second style: the words have no sentence end, so it is kept whole (and said so)
        let y = sentences(24_000, &[25]);
        let n2 = NewStyle { style: "calm", audio: &y, rate: 24_000, transcript: "ക ക ക", to_sentence: true, ..n.clone() };
        let calm = set_style(&models, "clone:my-voice", &n2).unwrap();
        assert!(calm.warning.is_some() && calm.trimmed == 0.0, "{calm:?}");
        assert_eq!(style_references(&models, "clone:my-voice").unwrap().iter().map(|(s, _)| *s).collect::<Vec<_>>(), ["excited", "calm"]);
        // a damaged clip is not a reference
        let good = std::fs::read(voices.join("my-voice.indicf5.calm.wav")).unwrap();
        std::fs::write(voices.join("my-voice.indicf5.calm.wav"), b"RIFF").unwrap();
        assert_eq!(style_reference(&models, "clone:my-voice", "calm").unwrap(), None);
        std::fs::write(voices.join("my-voice.indicf5.calm.wav"), &good).unwrap();
        // back to before the calm style: it is gone again; forward again: back
        rollback(&models, "clone:my-voice", "r1").unwrap();
        assert_eq!(style_references(&models, "clone:my-voice").unwrap().len(), 1);
        let latest = versions(&models, "clone:my-voice").unwrap().last().unwrap().dir.clone();
        rollback(&models, "clone:my-voice", &latest).unwrap();
        assert_eq!(style_references(&models, "clone:my-voice").unwrap().len(), 2);
        // removing one
        remove_style(&models, "clone:my-voice", "calm").unwrap();
        assert_eq!(style_reference(&models, "clone:my-voice", "calm").unwrap(), None);
        assert!(remove_style(&models, "clone:my-voice", "calm").is_err(), "nothing to remove");
        // a new voice sample (promoted) keeps the styles, tied to it
        let new = sentences(44_100, &[30, 30]);
        let r = NewReference { part: Part::Sample, audio: &new, rate: 44_100, source: "IMG_2.MOV", from: 0.0, to: 20.0, transcript: None, note: "" };
        promote(&models, "clone:my-voice", &r).unwrap();
        assert!(style_reference(&models, "clone:my-voice", "excited").unwrap().is_some());
        // deleting the voice takes them
        delete(&voices, "clone:my-voice").unwrap();
        assert!(!voices.join("my-voice.indicf5.excited.wav").exists() && !voices.join("my-voice.indicf5.excited.json").exists());
    }

    #[test]
    fn hostile_styles_are_refused_and_change_nothing() {
        let (models, voices) = data("hostile");
        voice(&voices);
        let ok = sentences(24_000, &[20, 20]);
        let w = words(&[20, 20]);
        let n = NewStyle { style: "warm", audio: &ok, rate: 24_000, source: "x", from: 0.0, to: 9.0, transcript: &w, given: true, to_sentence: true };
        let nan = vec![f32::NAN; 24_000 * 9];
        let short = sentences(24_000, &[6]);
        let long = sentences(24_000, &[60, 60]);
        let silent = vec![0.0f32; 24_000 * 9];
        let huge = vec![0.1f32; 24_000 * 60];
        for bad in [
            NewStyle { style: "plain", ..n.clone() },
            NewStyle { style: "angry", ..n.clone() },
            NewStyle { style: "../../x", ..n.clone() },
            NewStyle { audio: &nan, ..n.clone() },
            NewStyle { audio: &[], ..n.clone() },
            NewStyle { rate: 0, ..n.clone() },
            NewStyle { from: f64::NAN, ..n.clone() },
            NewStyle { from: 5.0, to: 1.0, ..n.clone() },
            NewStyle { from: -1.0, ..n.clone() },
            NewStyle { transcript: "  ", ..n.clone() },
            NewStyle { audio: &short, transcript: "ക.", ..n.clone() },
            NewStyle { audio: &long, to_sentence: false, ..n.clone() },
            NewStyle { audio: &silent, to_sentence: false, ..n.clone() },
            NewStyle { audio: &huge, to_sentence: false, ..n.clone() },
        ] {
            assert!(set_style(&models, "clone:my-voice", &bad).is_err(), "{} {} {}", bad.style, bad.audio.len(), bad.transcript);
        }
        let t = "x".repeat(MAX_TRANSCRIPT_CHARS + 1);
        assert!(set_style(&models, "clone:my-voice", &NewStyle { transcript: &t, ..n.clone() }).is_err());
        assert!(versions(&models, "clone:my-voice").unwrap().is_empty(), "nothing archived for a refused style");
        assert!(style_references(&models, "clone:my-voice").unwrap().is_empty());
        assert!(set_style(&models, "clone:nobody", &n).is_err());
        assert!(style_reference(&models, "clone:my-voice", "nope").is_err());
        assert!(remove_style(&models, "clone:my-voice", "warm").is_err());
        // a timbre-only clone has none
        save(&voices, &Profile { id: "clone:other".into(), ..super::super::tests::profile() }).unwrap();
        assert!(set_style(&models, "clone:other", &n).is_err());
        assert!(style_references(&models, "clone:other").unwrap().is_empty());
        // the one that works, once
        assert!(set_style(&models, "clone:my-voice", &n).is_ok());
    }

    #[test]
    fn every_style_has_its_reference_files() {
        for s in crate::styles::STYLES {
            assert!(REFERENCE_FILES.contains(&format!("indicf5.{s}.wav").as_str()), "{s}");
            assert!(REFERENCE_FILES.contains(&format!("indicf5.{s}.json").as_str()), "{s}");
        }
    }
}
