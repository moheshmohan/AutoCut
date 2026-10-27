//! Prosody of speech: where its pitch goes, reshaping that pitch, and how expressive it is.
//!
//! * [`pitch`] — an F0 tracker for speech: the cumulative-mean-normalised difference function
//!   of YIN (de Cheveigné & Kawahara, JASA 2002) computed through the FFT, parabolic
//!   interpolation of each dip, and a dynamic-programming path over every frame's candidates
//!   (the octave-cost / octave-jump-cost / voicing-cost formulation of Boersma, IFA Proceedings
//!   1993) so the track neither jumps octaves nor flickers between voiced and unvoiced. 10 ms
//!   frames, 60–500 Hz by default.
//! * [`psola`] — the pitch contour reshaped by pitch-synchronous overlap-add (TD-PSOLA, Moulines
//!   & Charpentier, Speech Communication 1990): the same duration, unvoiced stretches passed
//!   through untouched, cross-faded at voicing boundaries, the local energy kept. On top of it
//!   [`Shape`]: the pitch range scaled about the utterance's median, a shift, a question's rise
//!   at the end and a gentle lift of stressed syllables.
//! * [`features`] — [`Expressiveness`]: F0 median and range, F0 movement, energy variation over
//!   syllable peaks, syllable rate, voiced fraction and the final contour of a stretch of speech.
//!
//! Mono `f32` in, at any rate from [`MIN_RATE`] to [`MAX_RATE`]; at most [`MAX_SECONDS`].
//! Nothing here panics on hostile input: bad rates and settings are errors, non-finite samples
//! count as silence, empty input gives an empty result.

pub mod features;
pub mod pitch;
pub mod psola;

pub use features::{Expressiveness, Syllable, expressiveness, expressiveness_of, syllables};
pub use pitch::{PitchSettings, PitchTrack, track};
pub use psola::{Shape, psola, reshape, shape_ratios};

/// Seconds between analysis frames (frame `i` is centred on `i · HOP_S`).
pub const HOP_S: f64 = 0.01;
/// Lowest sample rate accepted (the tracker needs four samples per period of its highest pitch).
pub const MIN_RATE: u32 = 4_000;
/// Highest sample rate accepted.
pub const MAX_RATE: u32 = 768_000;
/// Longest audio analysed at once, seconds (two hours: the path search keeps every frame).
pub const MAX_SECONDS: f64 = 7200.0;

/// What went wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum ProsodyError {
    /// The sample rate is outside [`MIN_RATE`]…[`MAX_RATE`].
    Rate(u32),
    /// A setting or shape value out of range (what and why).
    Settings(String),
    /// The audio is longer than [`MAX_SECONDS`].
    TooLong(f64),
    /// A per-frame curve does not match the track it belongs to.
    Mismatch { frames: usize, values: usize },
    /// Memory for the analysis could not be had.
    Memory,
}

impl std::fmt::Display for ProsodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProsodyError::Rate(r) => write!(f, "sample rate {r} Hz is outside {MIN_RATE}–{MAX_RATE} Hz"),
            ProsodyError::Settings(s) => write!(f, "{s}"),
            ProsodyError::TooLong(s) => write!(f, "{s:.0} s of audio is longer than the {MAX_SECONDS:.0} s analysed at once"),
            ProsodyError::Mismatch { frames, values } => write!(f, "{values} values for a track of {frames} frames"),
            ProsodyError::Memory => write!(f, "not enough memory for the analysis"),
        }
    }
}

impl std::error::Error for ProsodyError {}

/// Check a rate and a length (samples) against the limits.
pub(crate) fn check_input(len: usize, rate: u32) -> Result<(), ProsodyError> {
    if !(MIN_RATE..=MAX_RATE).contains(&rate) {
        return Err(ProsodyError::Rate(rate));
    }
    let secs = len as f64 / f64::from(rate);
    if secs > MAX_SECONDS {
        return Err(ProsodyError::TooLong(secs));
    }
    Ok(())
}

/// A zeroed vector of `n`, or [`ProsodyError::Memory`].
pub(crate) fn zeros<T: Clone + Default>(n: usize) -> Result<Vec<T>, ProsodyError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| ProsodyError::Memory)?;
    v.resize(n, T::default());
    Ok(v)
}

/// Semitones from `a` Hz up to `b` Hz.
pub fn semitones(a: f64, b: f64) -> f64 {
    if a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() { 12.0 * (b / a).log2() } else { 0.0 }
}

/// The `p` quantile (0…1) of `v` (sorted here; NaNs dropped). `None` when empty.
pub(crate) fn quantile(v: &[f64], p: f64) -> Option<f64> {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(f64::total_cmp);
    let pos = (s.len() - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let t = pos - lo as f64;
    Some(s.get(lo).copied().unwrap_or(0.0) * (1.0 - t) + s.get(hi).copied().unwrap_or(0.0) * t)
}

#[cfg(test)]
mod tests;
