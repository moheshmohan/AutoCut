//! Voice activity: where speech is, and energy-based tightening of word bounds.
//!
//! **Speech segments** ([`speech_segments`]): Silero VAD v5 ([`crate::silero`], the model
//! [`crate::models::SILERO_VAD`]) when it is downloaded, else the energy detector
//! ([`energy_segments`]: 10 ms frames above the threshold below, the same hysteresis). Silero
//! tells speech from music, noise and breathing; the energy detector only from quiet.
//!
//! **Word bounds.** Attention alignment places every word boundary on the next word's start, so
//! silence between words is absorbed into the word before it (and leading silence into the first
//! word). Pause detection and text-based editing need the silences, so each word's start and end
//! are moved inward past 10 ms frames whose level is below a threshold set between the clip's
//! noise floor (10th percentile of the levels of frames above −100 dBFS) and its speech level
//! (95th percentile): 30 % of the way up, and at least 6 dB above the floor. A word keeps at least
//! 60 ms.

use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_project::Word;

use crate::silero::{SegmentParams, Silero};
use crate::{SAMPLE_RATE, TICKS_PER_SAMPLE};

/// How speech segments were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum VadMethod {
    Silero,
    Energy,
}

/// Silero VAD from `models_dir`, loaded once per process (`None`: not downloaded or unreadable).
pub fn load_silero(models_dir: &Path) -> Option<Arc<Silero>> {
    static CACHE: OnceLock<Mutex<Option<(std::path::PathBuf, Arc<Silero>)>>> = OnceLock::new();
    let path = crate::models::support_file(models_dir, crate::models::SILERO_VAD)?;
    let mut g = CACHE.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((p, m)) = g.as_ref()
        && *p == path
    {
        return Some(m.clone());
    }
    let m = Arc::new(Silero::load(&path).ok()?);
    *g = Some((path, m.clone()));
    Some(m)
}

/// Speech segments of mono 16 kHz `audio` (sample ranges in order): Silero's when `silero` is
/// given, else the energy detector's.
pub fn speech_segments(audio: &[f32], silero: Option<&Silero>, p: &SegmentParams) -> (Vec<Range<usize>>, VadMethod) {
    match silero {
        Some(m) => {
            let probs = m.probabilities(audio);
            (crate::silero::segments(&probs, audio.len(), p), VadMethod::Silero)
        }
        None => (energy_segments(audio, p), VadMethod::Energy),
    }
}

/// The energy detector: 10 ms frames at or above [`threshold`] are speech, then the hysteresis of
/// [`crate::silero::segments`] (a quiet clip of nothing but noise has no speech: its loud frames
/// must stand 12 dB over the floor).
pub fn energy_segments(audio: &[f32], p: &SegmentParams) -> Vec<Range<usize>> {
    let db = frame_db(audio);
    if db.is_empty() {
        return Vec::new();
    }
    let th = threshold(&db);
    let mut s: Vec<f32> = db.iter().copied().filter(|v| *v > -100.0).collect();
    s.sort_by(f32::total_cmp);
    let q = |f: f32| s.get(((s.len().saturating_sub(1)) as f32 * f) as usize).copied().unwrap_or(-120.0);
    // no dynamics at all: steady noise or silence, no speech
    if s.is_empty() || q(0.95) - q(0.10) < 12.0 {
        return Vec::new();
    }
    let probs: Vec<f32> = db.iter().map(|v| if *v >= th { 1.0 } else { 0.0 }).collect();
    crate::silero::segments_hop(&probs, FRAME, audio.len(), p)
}

const FRAME: usize = (SAMPLE_RATE / 100) as usize;
const MIN_WORD_FRAMES: usize = 6;

/// Frame levels in dBFS (10 ms RMS).
pub fn frame_db(audio: &[f32]) -> Vec<f32> {
    audio
        .chunks(FRAME)
        .map(|c| {
            let e = c.iter().map(|v| v * v).sum::<f32>() / c.len().max(1) as f32;
            10.0 * (e + 1e-12).log10()
        })
        .collect()
}

/// The speech/silence threshold in dBFS.
pub fn threshold(db: &[f32]) -> f32 {
    if db.is_empty() {
        return -60.0;
    }
    // digital silence (< −100 dBFS) would drag the floor far below the recording's own noise
    let mut s: Vec<f32> = db.iter().copied().filter(|v| *v > -100.0).collect();
    if s.is_empty() {
        return -60.0;
    }
    s.sort_by(f32::total_cmp);
    let q = |p: f32| s[((s.len() - 1) as f32 * p) as usize];
    let (floor, speech) = (q(0.10), q(0.95));
    (floor + 0.3 * (speech - floor)).max(floor + 6.0)
}

/// Move word bounds inward past silent frames (see the module docs).
pub fn tighten_words(audio: &[f32], words: &mut [Word]) {
    let db = frame_db(audio);
    if db.is_empty() {
        return;
    }
    let th = threshold(&db);
    let frame_ticks = TICKS_PER_SAMPLE * FRAME as i64;
    for w in words {
        let a = (w.start.0 / frame_ticks).max(0) as usize;
        let b = (((w.end.0 + frame_ticks - 1) / frame_ticks).max(0) as usize).min(db.len());
        if b <= a + MIN_WORD_FRAMES {
            continue;
        }
        let mut s = a;
        while s + MIN_WORD_FRAMES < b && db[s] < th {
            s += 1;
        }
        let mut e = b;
        while e > s + MIN_WORD_FRAMES && db[e - 1] < th {
            e -= 1;
        }
        // nothing voiced at all: leave the word alone
        if (s..e).all(|f| db[f] < th) {
            continue;
        }
        if s > a {
            w.start = filmcraft_time::Tick(s as i64 * frame_ticks);
        }
        if e < b {
            w.end = filmcraft_time::Tick(e as i64 * frame_ticks).max(w.start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seconds_tick;

    #[test]
    fn silence_is_trimmed_from_words() {
        // 0.0–0.5 silence, 0.5–1.0 tone, 1.0–2.0 silence, 2.0–2.5 tone
        let mut audio: Vec<f32> = (0..40_000).map(|i| if i % 2 == 0 { 0.0001 } else { 0.0 }).collect();
        for i in (8_000..16_000).chain(32_000..40_000) {
            audio[i] = 0.3 * (i as f32 * 0.2).sin();
        }
        let mut w = vec![Word::new("a", seconds_tick(0.0), seconds_tick(2.0)), Word::new("b", seconds_tick(2.0), seconds_tick(2.5))];
        tighten_words(&audio, &mut w);
        assert_eq!(w[0].start, seconds_tick(0.5));
        assert_eq!(w[0].end, seconds_tick(1.0));
        assert_eq!((w[1].start, w[1].end), (seconds_tick(2.0), seconds_tick(2.5)));
    }

    #[test]
    fn energy_detector_finds_tones_not_silence() {
        let p = SegmentParams::default();
        // silence and steady noise: no speech
        assert!(speech_segments(&vec![0.0; 32_000], None, &p).0.is_empty());
        let noise: Vec<f32> = (0..32_000u32).map(|i| ((i.wrapping_mul(2_654_435_761) >> 16) as f32 / 65_536.0 - 0.5) * 0.01).collect();
        assert!(energy_segments(&noise, &p).is_empty());
        // 0.5–1.0 s and 2.0–2.5 s of tone over quiet noise
        let mut audio = noise.clone();
        audio.resize(48_000, 0.0);
        for i in (8_000..16_000).chain(32_000..40_000) {
            audio[i] += 0.3 * (i as f32 * 0.2).sin();
        }
        let (s, how) = speech_segments(&audio, None, &p);
        assert_eq!(how, VadMethod::Energy);
        assert_eq!(s.len(), 2, "{s:?}");
        let near = |a: usize, b: usize| a.abs_diff(b) <= 800;
        assert!(near(s[0].start, 8_000) && near(s[0].end, 16_000), "{s:?}");
        assert!(near(s[1].start, 32_000) && near(s[1].end, 40_000), "{s:?}");
        // hostile audio: empty, NaN
        assert!(energy_segments(&[], &p).is_empty());
        let _ = energy_segments(&[f32::NAN; 4000], &p);
    }

    /// Silero on real speech, when the model and the clip are present
    /// (`target/fixtures/silero-test.wav`: Silero's own `tests/data/test.wav`, MIT): it finds
    /// speech in the clip and none in silence.
    #[test]
    fn silero_finds_speech_when_present() {
        let Some(dir) = std::env::var_os("FILMCRAFT_MODELS_DIR").map(std::path::PathBuf::from) else { return };
        let Some(m) = load_silero(&dir) else { return };
        assert!(speech_segments(&vec![0.0; 48_000], Some(&m), &SegmentParams::default()).0.is_empty());
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fixtures/silero-test.wav");
        let Ok(bytes) = std::fs::read(&wav) else { return };
        let audio: Vec<f32> = bytes.get(44..).unwrap_or(&[]).as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c) as f32 / 32768.0).collect();
        let (s, how) = speech_segments(&audio, Some(&m), &SegmentParams::default());
        assert_eq!(how, VadMethod::Silero);
        let speech: usize = s.iter().map(|r| r.len()).sum();
        assert!(s.len() >= 3 && speech > audio.len() / 3, "{} segments, {speech} of {} samples", s.len(), audio.len());
    }
}
