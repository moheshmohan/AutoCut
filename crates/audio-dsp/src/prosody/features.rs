//! How expressive a stretch of speech is.
//!
//! [`Expressiveness`] of mono speech, from its [`PitchTrack`] and a speech-band level contour
//! (300 Hz–3 kHz, 30 ms around each 10 ms frame):
//!
//! | feature | what it is |
//! |---|---|
//! | `f0_median_hz` | median F0 of the voiced frames |
//! | `f0_range_st` | the 10th to 90th percentile of F0, semitones |
//! | `f0_movement_st_s` | mean absolute F0 slope, semitones per second: between voiced frames 50 ms apart with voicing all the way between them, on the contour median-smoothed over 50 ms (so frame-to-frame tracking jitter hardly counts) |
//! | `final_contour_st` | F0 over the last 100 ms of voicing against the 400 ms before, semitones: positive for a rising end (a question), negative for a falling one |
//! | `energy_variation_db` | standard deviation of the syllable peaks' levels, dB |
//! | `syllable_rate` | syllable peaks per second of speech (active frames: speech and its short gaps, not pauses) |
//! | `voiced_fraction` | voiced frames over the speech frames |
//!
//! **Syllables** ([`syllables`]): peaks of the level contour, after the published intensity-peak
//! syllable counter of de Jong & Wempe (Behavior Research Methods, 2009): a peak counts when it
//! is in speech, voiced (or within 20 ms of voicing), no more than 25 dB under the loud frames,
//! and stands at least 2 dB above the dips on either side of it towards its neighbours, at
//! least 80 ms from the previous one; of two peaks without such a dip between them the higher
//! is kept.

use super::{HOP_S, ProsodyError, check_input, quantile, semitones, zeros};
use super::{PitchSettings, PitchTrack, track};
use crate::biquad::{Biquad, Coeffs, FilterType};

/// The speech band the level contour is measured in, Hz.
const BAND: (f64, f64) = (300.0, 3000.0);
/// Smallest dip between two syllable peaks, dB.
const MIN_DIP_DB: f64 = 2.0;
/// Peaks this far under the loud frames are not syllables, dB.
const PEAK_FLOOR_DB: f64 = 25.0;
/// Closest two syllable peaks, frames.
const MIN_SEPARATION: usize = 8;
/// Furthest a syllable reaches either side of its peak, frames.
const MAX_REACH: usize = 25;
/// Speech activity: hang-over after the level drops and lead-in before it rises, frames.
const HANG: usize = 8;
const LEAD: usize = 3;
/// The final contour: the last stretch of voicing, and the one before it compared, seconds.
const FINAL_TAIL_S: f64 = 0.1;
const FINAL_HEAD_S: f64 = 0.4;
/// F0 movement is measured between frames this far apart, seconds.
const MOVEMENT_S: f64 = 0.05;

/// A syllable's nucleus and extent (frame indices of the track it was found on).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Syllable {
    /// The frame of its level peak.
    pub peak: usize,
    /// The dip before and after it (inclusive), at most 250 ms from the peak.
    pub start: usize,
    pub end: usize,
    /// Speech-band level at the peak, dBFS.
    pub level_db: f64,
    /// Smoothed F0 at the peak (or the nearest voiced frame within 20 ms), Hz.
    pub f0_hz: Option<f64>,
}

/// The expressiveness of a stretch of speech (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Expressiveness {
    /// Length of the audio, seconds.
    pub seconds: f64,
    /// Speech in it (active frames), seconds.
    pub speech_seconds: f64,
    /// Voiced frames, seconds.
    pub voiced_seconds: f64,
    /// Voiced over speech, 0…1.
    pub voiced_fraction: f64,
    pub f0_median_hz: Option<f64>,
    /// 10th–90th percentile of F0, semitones.
    pub f0_range_st: f64,
    /// Mean absolute F0 slope, semitones per second.
    pub f0_movement_st_s: f64,
    /// Rise (positive) or fall (negative) of the end of the last voiced stretch, semitones.
    pub final_contour_st: f64,
    /// Standard deviation of the syllable peaks' levels, dB.
    pub energy_variation_db: f64,
    pub syllables: usize,
    /// Syllable peaks per second of speech.
    pub syllable_rate: f64,
}

/// The expressiveness of `x` (mono at `rate`), tracking its pitch with `s`.
pub fn expressiveness(x: &[f32], rate: u32, s: &PitchSettings) -> Result<Expressiveness, ProsodyError> {
    let t = track(x, rate, s)?;
    expressiveness_of(x, rate, &t)
}

/// The expressiveness of `x` (mono at `rate`) whose pitch track is `t`.
pub fn expressiveness_of(x: &[f32], rate: u32, t: &PitchTrack) -> Result<Expressiveness, ProsodyError> {
    check_input(x.len(), rate)?;
    let hop = if t.hop.is_finite() && t.hop > 0.0 { t.hop } else { HOP_S };
    let mut e = Expressiveness { seconds: x.len() as f64 / f64::from(rate), ..Default::default() };
    if t.is_empty() {
        return Ok(e);
    }
    let levels = band_levels(x, rate, t.len(), hop)?;
    let active = activity(&levels, t);
    let speech = active.iter().filter(|a| **a).count();
    e.speech_seconds = speech as f64 * hop;
    e.voiced_seconds = t.voiced() as f64 * hop;
    e.voiced_fraction = if speech > 0 { (t.voiced() as f64 / speech as f64).min(1.0) } else { 0.0 };
    let lf = smoothed_log2(t);
    let voiced: Vec<f64> = lf.iter().flatten().copied().collect();
    if let Some(m) = quantile(&voiced, 0.5) {
        e.f0_median_hz = Some(m.exp2());
        let (p10, p90) = (quantile(&voiced, 0.1).unwrap_or(m), quantile(&voiced, 0.9).unwrap_or(m));
        e.f0_range_st = 12.0 * (p90 - p10);
    }
    e.f0_movement_st_s = movement(&lf, hop);
    e.final_contour_st = final_contour(&lf, hop);
    let syl = syllables_from(&levels, &active, t, &lf);
    e.syllables = syl.len();
    e.syllable_rate = if e.speech_seconds > 0.0 { syl.len() as f64 / e.speech_seconds } else { 0.0 };
    if syl.len() >= 2 {
        let lv: Vec<f64> = syl.iter().map(|s| s.level_db).collect();
        let mean = lv.iter().sum::<f64>() / lv.len() as f64;
        e.energy_variation_db = (lv.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / lv.len() as f64).sqrt();
    }
    Ok(e)
}

/// The syllables of `x` (mono at `rate`) whose pitch track is `t` (see the module docs).
pub fn syllables(x: &[f32], rate: u32, t: &PitchTrack) -> Result<Vec<Syllable>, ProsodyError> {
    check_input(x.len(), rate)?;
    let hop = if t.hop.is_finite() && t.hop > 0.0 { t.hop } else { HOP_S };
    if t.is_empty() {
        return Ok(Vec::new());
    }
    let levels = band_levels(x, rate, t.len(), hop)?;
    let active = activity(&levels, t);
    Ok(syllables_from(&levels, &active, t, &smoothed_log2(t)))
}

/// The track's F0 as log₂ Hz, median-smoothed over five frames within each voiced run.
pub fn smoothed_log2(t: &PitchTrack) -> Vec<Option<f64>> {
    let raw: Vec<Option<f64>> = t.f0.iter().map(|f| f.map(f64::from).filter(|f| *f > 0.0 && f.is_finite()).map(f64::log2)).collect();
    let mut out = raw.clone();
    for (a, b) in t.runs() {
        for i in a..=b {
            let lo = i.saturating_sub(2).max(a);
            let hi = (i + 2).min(b);
            let mut w: Vec<f64> = raw.get(lo..=hi).unwrap_or(&[]).iter().flatten().copied().collect();
            w.sort_by(f64::total_cmp);
            if let (Some(o), Some(m)) = (out.get_mut(i), w.get(w.len() / 2)) {
                *o = Some(*m);
            }
        }
    }
    out
}

/// Speech-band level (dBFS) of 30 ms around each of `frames` frames `hop` apart.
pub(crate) fn band_levels(x: &[f32], rate: u32, frames: usize, hop: f64) -> Result<Vec<f64>, ProsodyError> {
    let sr = f64::from(rate);
    let hp = Coeffs::design(FilterType::HighPass, sr, BAND.0, std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let lp = Coeffs::design(FilterType::LowPass, sr, BAND.1.min(0.45 * sr), std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let (mut b1, mut b2) = (Biquad::new(), Biquad::new());
    let hop_n = hop * sr;
    // energy per bin: bin i covers [(i − ½)·hop, (i + ½)·hop)
    let mut bins: Vec<f64> = zeros(frames)?;
    let mut counts: Vec<u32> = zeros(frames)?;
    for (n, v) in x.iter().enumerate() {
        let v = if v.is_finite() { f64::from(*v) } else { 0.0 };
        let y = b2.tick(&lp, b1.tick(&hp, v));
        let bin = ((n as f64 / hop_n) + 0.5).floor() as usize;
        if let (Some(b), Some(c)) = (bins.get_mut(bin), counts.get_mut(bin)) {
            *b += y * y;
            *c += 1;
        }
    }
    let mut out: Vec<f64> = zeros(frames)?;
    for (i, o) in out.iter_mut().enumerate() {
        let (mut e, mut c) = (0.0, 0u32);
        for j in i.saturating_sub(1)..=(i + 1) {
            if let (Some(b), Some(k)) = (bins.get(j), counts.get(j)) {
                e += b;
                c += k;
            }
        }
        *o = if c > 0 { (10.0 * (e / f64::from(c) + 1e-12).log10()).max(-120.0) } else { -120.0 };
    }
    Ok(out)
}

/// Which frames are speech: over a threshold 30 % of the way from the floor (10th percentile)
/// to the loud frames (95th), at least 6 dB over the floor and −60 dBFS, with hang-over and
/// lead-in; voiced frames always.
fn activity(levels: &[f64], t: &PitchTrack) -> Vec<bool> {
    let finite: Vec<f64> = levels.iter().copied().filter(|l| *l > -100.0).collect();
    let (Some(floor), Some(loud)) = (quantile(&finite, 0.1), quantile(&finite, 0.95)) else {
        return t.f0.iter().map(Option::is_some).collect();
    };
    let th = (floor + 0.3 * (loud - floor)).max(floor + 6.0).max(-60.0);
    let raw: Vec<bool> = levels.iter().map(|l| *l >= th).collect();
    let mut out = vec![false; raw.len()];
    let mut since = usize::MAX;
    for (i, r) in raw.iter().enumerate() {
        since = if *r { 0 } else { since.saturating_add(1) };
        if since <= HANG
            && let Some(o) = out.get_mut(i)
        {
            *o = true;
        }
        if *r {
            out.iter_mut().skip(i.saturating_sub(LEAD)).take(LEAD.min(i)).for_each(|o| *o = true);
        }
    }
    for (o, f) in out.iter_mut().zip(&t.f0) {
        *o |= f.is_some();
    }
    out
}

fn syllables_from(levels: &[f64], active: &[bool], t: &PitchTrack, lf: &[Option<f64>]) -> Vec<Syllable> {
    let n = levels.len();
    let finite: Vec<f64> = levels.iter().copied().filter(|l| *l > -100.0).collect();
    let Some(loud) = quantile(&finite, 0.95) else { return Vec::new() };
    let lv = |i: usize| levels.get(i).copied().unwrap_or(-120.0);
    let near_voiced = |i: usize| (i.saturating_sub(2)..=(i + 2).min(n.saturating_sub(1))).any(|j| t.f0.get(j).is_some_and(Option::is_some));
    let mut kept: Vec<usize> = Vec::new();
    for i in 1..n.saturating_sub(1) {
        let (a, b, c) = (lv(i - 1), lv(i), lv(i + 1));
        if !(b >= a && b > c) || !active.get(i).copied().unwrap_or(false) || !near_voiced(i) || b < loud - PEAK_FLOOR_DB {
            continue;
        }
        match kept.last().copied() {
            Some(q) => {
                let dip = (q..i).map(lv).fold(f64::INFINITY, f64::min);
                if i - q >= MIN_SEPARATION && lv(q) - dip >= MIN_DIP_DB && b - dip >= MIN_DIP_DB {
                    kept.push(i);
                } else if b > lv(q)
                    && let Some(last) = kept.last_mut()
                {
                    *last = i;
                }
            }
            None => kept.push(i),
        }
    }
    let argmin = |lo: usize, hi: usize| (lo..=hi).min_by(|x, y| lv(*x).total_cmp(&lv(*y))).unwrap_or(lo);
    let mut out = Vec::with_capacity(kept.len());
    for (k, &p) in kept.iter().enumerate() {
        let prev = k.checked_sub(1).and_then(|j| kept.get(j)).copied();
        let next = kept.get(k + 1).copied();
        let lo = prev.map_or(p.saturating_sub(MAX_REACH), |q| q.max(p.saturating_sub(MAX_REACH)));
        let hi = next.map_or((p + MAX_REACH).min(n.saturating_sub(1)), |q| q.min(p + MAX_REACH));
        let f0 = (0..=2usize).flat_map(|d| [p.checked_sub(d), Some(p + d)]).flatten().find_map(|j| lf.get(j).copied().flatten()).map(f64::exp2);
        out.push(Syllable { peak: p, start: argmin(lo, p), end: argmin(p, hi), level_db: lv(p), f0_hz: f0 });
    }
    out
}

/// Mean absolute F0 slope (semitones per second) between voiced frames [`MOVEMENT_S`] apart
/// with voicing all the way between them.
fn movement(lf: &[Option<f64>], hop: f64) -> f64 {
    let lag = ((MOVEMENT_S / hop).round() as usize).max(1);
    let (mut sum, mut n) = (0.0, 0usize);
    let mut run_start: Option<usize> = None;
    for (i, v) in lf.iter().enumerate() {
        let Some(b) = v else {
            run_start = None;
            continue;
        };
        let start = *run_start.get_or_insert(i);
        if i >= start + lag
            && let Some(Some(a)) = lf.get(i - lag)
        {
            sum += 12.0 * (b - a).abs() / (lag as f64 * hop);
            n += 1;
        }
    }
    if n > 0 { sum / n as f64 } else { 0.0 }
}

/// F0 over the last 100 ms of voicing against the 400 ms before it, semitones.
fn final_contour(lf: &[Option<f64>], hop: f64) -> f64 {
    let Some(e) = lf.iter().rposition(Option::is_some) else { return 0.0 };
    let tail_n = (FINAL_TAIL_S / hop).round() as usize;
    let head_n = (FINAL_HEAD_S / hop).round() as usize;
    let tail_start = e.saturating_sub(tail_n.saturating_sub(1));
    let tail: Vec<f64> = lf.get(tail_start..=e).unwrap_or(&[]).iter().flatten().copied().collect();
    let head: Vec<f64> = lf.get(tail_start.saturating_sub(head_n)..tail_start).unwrap_or(&[]).iter().flatten().copied().collect();
    match (quantile(&tail, 0.5), quantile(&head, 0.5)) {
        (Some(a), Some(b)) => semitones(b.exp2(), a.exp2()),
        _ => 0.0,
    }
}
