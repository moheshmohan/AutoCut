//! Exact musical time: every beat and sixteenth of a generated track falls on an integer
//! [`Tick`], and every sample position comes from those ticks, so the beat map an editor cuts on
//! and the audio it hears agree to the sample.

use filmcraft_time::{TICKS_PER_SECOND, Tick};

/// Slowest tempo, beats per minute.
pub const MIN_BPM: f64 = 40.0;
/// Fastest tempo, beats per minute.
pub const MAX_BPM: f64 = 200.0;
/// Tempo used when none (or a non-number) is given.
pub const DEFAULT_BPM: f64 = 96.0;
/// Every generated track is in 4/4.
pub const BEATS_PER_BAR: u64 = 4;
/// Sixteenths in a beat.
pub const STEPS_PER_BEAT: u64 = 4;
/// Sixteenths in a bar.
pub const STEPS_PER_BAR: u64 = BEATS_PER_BAR * STEPS_PER_BEAT;
/// Positions are counted in thousandths of a sixteenth (swing moves notes by part of a step).
pub const MILLI: u64 = 1000;

/// A tempo held exactly, in thousandths of a beat per minute: beat `k` falls on tick
/// `round(k · 60 000 · TPS / milli_bpm)`, with no drift however long the track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tempo {
    mbpm: i64,
}

impl Tempo {
    /// `bpm` clamped to [`MIN_BPM`]..[`MAX_BPM`] (a non-number is [`DEFAULT_BPM`]), to a
    /// thousandth of a beat per minute.
    pub fn from_bpm(bpm: f64) -> Self {
        let bpm = if bpm.is_finite() { bpm.clamp(MIN_BPM, MAX_BPM) } else { DEFAULT_BPM };
        Tempo { mbpm: (bpm * 1000.0).round() as i64 }
    }

    /// Beats per minute.
    pub fn bpm(self) -> f64 {
        self.mbpm as f64 / 1000.0
    }

    /// Thousandths of a beat per minute (the exact tempo).
    pub fn milli_bpm(self) -> i64 {
        self.mbpm
    }

    /// The tick `msteps` thousandths of a sixteenth from the start.
    pub fn tick_at(self, msteps: u64) -> Tick {
        // a sixteenth lasts 15 s / bpm = 15 000 · TPS / mbpm ticks, a thousandth of one 15 · TPS / mbpm
        let num = i128::from(msteps) * 15 * i128::from(TICKS_PER_SECOND);
        let den = i128::from(self.mbpm.max(1));
        let t = (2 * num + den) / (2 * den);
        Tick(i64::try_from(t).unwrap_or(Tick::MAX.0).min(Tick::MAX.0))
    }

    /// The tick of sixteenth `s`.
    pub fn step(self, s: u64) -> Tick {
        self.tick_at(s.saturating_mul(MILLI))
    }

    /// The tick of beat `k`.
    pub fn beat(self, k: u64) -> Tick {
        self.step(k.saturating_mul(STEPS_PER_BEAT))
    }

    /// The tick of the downbeat of bar `b`.
    pub fn bar(self, b: u64) -> Tick {
        self.beat(b.saturating_mul(BEATS_PER_BAR))
    }

    /// Seconds a beat lasts.
    pub fn beat_seconds(self) -> f64 {
        60.0 / self.bpm()
    }

    /// Seconds a bar lasts.
    pub fn bar_seconds(self) -> f64 {
        self.beat_seconds() * BEATS_PER_BAR as f64
    }
}

/// The sample at `sample_rate` nearest tick `t` (0 for times before the start).
pub fn sample_of(t: Tick, sample_rate: u32) -> u64 {
    if t.0 <= 0 {
        return 0;
    }
    let num = i128::from(t.0) * i128::from(sample_rate);
    let den = i128::from(TICKS_PER_SECOND);
    u64::try_from((2 * num + den) / (2 * den)).unwrap_or(u64::MAX)
}

/// The tick of sample `n` at `sample_rate` (exact when the rate divides the tick rate, as 48 kHz does).
pub fn tick_of_sample(n: u64, sample_rate: u32) -> Tick {
    let rate = i64::from(sample_rate.max(1));
    Tick::from_units(i64::try_from(n).unwrap_or(i64::MAX), rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beats_fall_on_exact_ticks_without_drift() {
        let t = Tempo::from_bpm(96.0);
        // 0.625 s a beat
        assert_eq!(t.beat(1), Tick(TICKS_PER_SECOND * 5 / 8));
        assert_eq!(t.beat(10_000), Tick(TICKS_PER_SECOND * 6250));
        assert_eq!(t.bar(3), t.beat(12));
        assert_eq!(t.step(8), t.beat(2));
        // an awkward tempo: beat k is the nearest tick to k · 60 / bpm seconds, never further
        let t = Tempo::from_bpm(97.0);
        for k in [1u64, 7, 97, 1000, 2999] {
            let exact = k as f64 * 60.0 / 97.0 * TICKS_PER_SECOND as f64;
            assert!((t.beat(k).0 as f64 - exact).abs() <= 1.0, "{k}");
        }
        assert!((Tempo::from_bpm(97.123_456).bpm() - 97.123).abs() < 1e-9);
    }

    #[test]
    fn hostile_tempos_are_clamped_and_samples_round() {
        assert_eq!(Tempo::from_bpm(f64::NAN).bpm(), DEFAULT_BPM);
        assert_eq!(Tempo::from_bpm(-5.0).bpm(), MIN_BPM);
        assert_eq!(Tempo::from_bpm(1e300).bpm(), MAX_BPM);
        assert_eq!(Tempo::from_bpm(f64::INFINITY).bpm(), DEFAULT_BPM);
        // huge positions saturate instead of overflowing
        assert!(Tempo::from_bpm(200.0).tick_at(u64::MAX) <= Tick::MAX);
        assert_eq!(sample_of(Tick(-5), 48_000), 0);
        assert_eq!(sample_of(Tick::from_seconds_f64(1.5), 48_000), 72_000);
        assert_eq!(sample_of(Tick(TICKS_PER_SECOND / 96_000), 48_000), 1, "half a sample rounds up");
        assert_eq!(tick_of_sample(48_000, 48_000), Tick(TICKS_PER_SECOND));
        assert_eq!(sample_of(tick_of_sample(12_345, 44_100), 44_100), 12_345);
    }
}
