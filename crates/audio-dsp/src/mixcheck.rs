//! What a mix will sound like on small speakers (a TV, a laptop, a phone), measured: where its
//! energy sits ([`BandMeter`]), how far speech stands above everything else while it plays
//! ([`speech_ratio`]), how hard a limiter works on it ([`GainReductionMeter`]) and whether a
//! signal clips ([`ClipMeter`]).
//!
//! Small speakers reproduce little below about 120 Hz: energy down there makes no sound a viewer
//! hears, but it fills the headroom, so the limiter (and a TV's own dynamics processing) pushes
//! everything else down and distorts on the bass peaks. Speech intelligibility at low volume
//! lives between about 1 and 4 kHz. These meters are pure and streaming (blocks of any length,
//! planar `f32`), so the engine can run them over a sequence's stems as it renders them.

use crate::fft::Fft;

/// Upper edges (Hz) of the analysis bands; the last band runs to Nyquist.
pub const BAND_EDGES: [f64; 5] = [120.0, 500.0, 1000.0, 4000.0, 6000.0];
/// Names of the bands, in order: below 120 Hz, 120–500 Hz, 500 Hz–1 kHz, 1–4 kHz, 4–6 kHz, above
/// 6 kHz.
pub const BAND_NAMES: [&str; 6] = ["sub", "lowMid", "mid", "presence", "upper", "air"];
/// Index of the "below 120 Hz" band in [`Bands::shares`].
pub const SUB: usize = 0;
/// Index of the 1–4 kHz band (speech intelligibility).
pub const PRESENCE: usize = 3;
/// Index of the "above 6 kHz" band.
pub const AIR: usize = 5;

/// Analysis frame (samples) of the band meter: 4096 at 48 kHz resolves 11.7 Hz, fine enough to
/// split at 120 Hz.
const FRAME: usize = 4096;
const HOP: usize = FRAME / 2;
/// Centre frequencies (Hz) of the octave bands of [`Bands::octaves_db`] (nominal, IEC 61260).
pub const OCTAVES: [f64; 10] = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
/// No octave (a bin below the first or above the last).
const NO_OCTAVE: u8 = u8::MAX;

/// Energy share of each band (0…1, summing to 1 when there was any signal), and the level of each
/// octave band relative to the whole.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bands {
    pub shares: [f64; 6],
    /// Each octave of [`OCTAVES`]: its energy relative to the whole signal's, dB (−∞ for none).
    pub octaves_db: [f64; 10],
    /// Seconds analysed.
    pub seconds: f64,
}

impl Bands {
    /// The share of band `i` in dB relative to the whole (−∞ for none).
    pub fn share_db(&self, i: usize) -> f64 {
        self.shares.get(i).map_or(f64::NEG_INFINITY, |s| 10.0 * s.log10())
    }
}

/// Long-term spectrum balance of a (multi-channel) signal: the power in each of the bands of
/// [`BAND_EDGES`], from Hann-windowed FFT frames with half overlap, summed over channels.
#[derive(Clone, Debug)]
pub struct BandMeter {
    fft: Fft,
    window: Vec<f32>,
    /// Per channel: the samples of the frame being filled.
    frames: Vec<Vec<f32>>,
    fill: usize,
    re: Vec<f32>,
    im: Vec<f32>,
    /// FFT bin → band, and → octave ([`NO_OCTAVE`] for none).
    band_of: Vec<(u8, u8)>,
    energy: [f64; 6],
    octave_energy: [f64; 10],
    samples: u64,
    sample_rate: f64,
}

impl BandMeter {
    /// A meter for `channels` channels at `sample_rate` Hz.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let window = (0..FRAME).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos()) as f32).collect();
        let band_of = (0..=FRAME / 2)
            .map(|k| {
                let f = k as f64 * sr / FRAME as f64;
                let band = BAND_EDGES.iter().position(|e| f < *e).unwrap_or(BAND_EDGES.len()) as u8;
                let octave = OCTAVES.iter().position(|c| f >= c / std::f64::consts::SQRT_2 && f < c * std::f64::consts::SQRT_2).map_or(NO_OCTAVE, |o| o as u8);
                (band, octave)
            })
            .collect();
        Self {
            fft: Fft::new(FRAME),
            window,
            frames: vec![vec![0.0; FRAME]; channels.clamp(1, 16)],
            fill: 0,
            re: vec![0.0; FRAME],
            im: vec![0.0; FRAME],
            band_of,
            energy: [0.0; 6],
            octave_energy: [0.0; 10],
            samples: 0,
            sample_rate: sr,
        }
    }

    /// Feed a block of planar audio (missing channels are silence, extra ones ignored).
    pub fn process(&mut self, channels: &[&[f32]]) {
        let n = channels.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut i = 0;
        while i < n {
            let take = (FRAME - self.fill).min(n - i);
            for (k, f) in self.frames.iter_mut().enumerate() {
                let dst = &mut f[self.fill..self.fill + take];
                match channels.get(k).and_then(|c| c.get(i..i + take)) {
                    Some(src) => dst.copy_from_slice(src),
                    None => {
                        // a channel shorter than the block (or missing): what it has, then silence
                        let have = channels.get(k).map_or(&[][..], |c| c.get(i..).unwrap_or(&[]));
                        for (j, d) in dst.iter_mut().enumerate() {
                            *d = have.get(j).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
                        }
                    }
                }
            }
            self.fill += take;
            self.samples += take as u64;
            i += take;
            if self.fill == FRAME {
                self.analyse();
                for f in &mut self.frames {
                    f.copy_within(HOP.., 0);
                }
                self.fill = FRAME - HOP;
            }
        }
    }

    fn analyse(&mut self) {
        for f in &self.frames {
            for ((r, m), (x, w)) in self.re.iter_mut().zip(self.im.iter_mut()).zip(f.iter().zip(&self.window)) {
                *r = if x.is_finite() { x * w } else { 0.0 };
                *m = 0.0;
            }
            self.fft.forward(&mut self.re, &mut self.im);
            for (k, (band, octave)) in self.band_of.iter().enumerate() {
                let (Some(r), Some(m)) = (self.re.get(k), self.im.get(k)) else { break };
                let p = f64::from(*r) * f64::from(*r) + f64::from(*m) * f64::from(*m);
                // the bins between DC and Nyquist stand for their negative twins too
                let w = if k == 0 || k == FRAME / 2 { 1.0 } else { 2.0 };
                if let Some(e) = self.energy.get_mut(*band as usize) {
                    *e += w * p;
                }
                if let Some(e) = self.octave_energy.get_mut(*octave as usize) {
                    *e += w * p;
                }
            }
        }
    }

    /// The shares so far (a partly filled last frame is not counted).
    pub fn bands(&self) -> Bands {
        let total: f64 = self.energy.iter().sum();
        let mut shares = [0.0; 6];
        let mut octaves_db = [f64::NEG_INFINITY; 10];
        if total > 0.0 && total.is_finite() {
            for (s, e) in shares.iter_mut().zip(&self.energy) {
                *s = e / total;
            }
            for (o, e) in octaves_db.iter_mut().zip(&self.octave_energy) {
                *o = 10.0 * (e / total).log10();
            }
        }
        Bands { shares, octaves_db, seconds: self.samples as f64 / self.sample_rate }
    }
}

/// How far speech stood above another signal while it played: over the points where the speech
/// was active, the median and the 10th percentile (its quietest tenth) of the difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ratio {
    pub median_lu: f64,
    /// Nine tenths of the speech stood at least this far above the other signal.
    pub p10_lu: f64,
    /// Points (of `hop_s` each) where speech was active.
    pub points: usize,
    pub active_seconds: f64,
}

/// Loudness used for a signal that is silent (or below it) in [`speech_ratio`], LUFS: a mix whose
/// music stops counts as speech this far above it, not infinitely.
pub const FLOOR_LUFS: f64 = -70.0;

/// The gate above which speech counts as active: 15 LU under its own integrated loudness (a
/// sentence's quiet word still counts, a breath or the room between lines does not), and never
/// under −50 LUFS.
pub fn speech_gate(speech_integrated_lufs: f64) -> f64 {
    if speech_integrated_lufs.is_finite() { (speech_integrated_lufs - 15.0).max(-50.0) } else { f64::INFINITY }
}

/// Speech against `other` (both loudness series of the same points, LUFS, e.g. momentary loudness
/// every 100 ms): where `speech` is at or above `gate`, how far above `other` it stood. `None`
/// when speech was never active.
pub fn speech_ratio(speech: &[f64], other: &[f64], gate: f64, hop_s: f64) -> Option<Ratio> {
    let mut d: Vec<f64> = speech
        .iter()
        .zip(other)
        .filter(|(s, _)| s.is_finite() && **s >= gate)
        .map(|(s, o)| s - if o.is_finite() { o.max(FLOOR_LUFS) } else { FLOOR_LUFS })
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    let at = |q: f64| d.get(((d.len() - 1) as f64 * q).round() as usize).copied();
    let hop = if hop_s.is_finite() && hop_s > 0.0 { hop_s } else { 0.0 };
    Some(Ratio { median_lu: at(0.5)?, p10_lu: at(0.1)?, points: d.len(), active_seconds: d.len() as f64 * hop })
}

/// How hard a limiter worked: the signal before it and after it, compared in short blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GainReduction {
    /// The largest reduction in any block, dB (0 when it never acted).
    pub max_db: f64,
    /// 99th percentile of the reduction over the blocks with signal, dB.
    pub p99_db: f64,
    /// Time with more than 1 dB and more than 2 dB of reduction, seconds.
    pub seconds_over_1db: f64,
    pub seconds_over_2db: f64,
}

/// Blocks quieter than this before the limiter (dBFS peak) are not compared: a limiter never acts
/// there, and rounding would read as reduction.
const GR_FLOOR_DB: f32 = -40.0;
/// Most blocks kept for the percentile (about 3 h of 10 ms blocks); later ones still count for the
/// maximum and the times.
const GR_MAX_BLOCKS: usize = 1_100_000;

/// Streaming gain-reduction meter: peaks of the signal before and after the limiter in blocks of
/// 10 ms (the two must be time-aligned, as the mixer delivers them with its latency compensated).
#[derive(Clone, Debug)]
pub struct GainReductionMeter {
    block: usize,
    pos: usize,
    pre: f32,
    post: f32,
    blocks: Vec<f32>,
    over1: u64,
    over2: u64,
    max: f32,
    sample_rate: f64,
}

impl GainReductionMeter {
    pub fn new(sample_rate: f64) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        Self { block: ((sr * 0.01).round() as usize).max(1), pos: 0, pre: 0.0, post: 0.0, blocks: Vec::new(), over1: 0, over2: 0, max: 0.0, sample_rate: sr }
    }

    /// Feed the same span of the signal before (`pre`) and after (`post`) the limiter.
    pub fn process(&mut self, pre: &[&[f32]], post: &[&[f32]]) {
        let n = pre.iter().chain(post).map(|c| c.len()).max().unwrap_or(0);
        let peak = |chs: &[&[f32]], i: usize| chs.iter().filter_map(|c| c.get(i)).filter(|v| v.is_finite()).fold(0.0f32, |a, v| a.max(v.abs()));
        for i in 0..n {
            self.pre = self.pre.max(peak(pre, i));
            self.post = self.post.max(peak(post, i));
            self.pos += 1;
            if self.pos == self.block {
                self.close();
            }
        }
    }

    fn close(&mut self) {
        let (pre, post) = (std::mem::take(&mut self.pre), std::mem::take(&mut self.post));
        self.pos = 0;
        if pre <= 0.0 || 20.0 * pre.log10() < GR_FLOOR_DB {
            return;
        }
        let gr = if post > 0.0 { (20.0 * (pre / post).log10()).max(0.0) } else { 0.0 };
        let gr = if gr.is_finite() { gr } else { 0.0 };
        self.max = self.max.max(gr);
        if gr > 1.0 {
            self.over1 += 1;
        }
        if gr > 2.0 {
            self.over2 += 1;
        }
        if self.blocks.len() < GR_MAX_BLOCKS {
            self.blocks.push(gr);
        }
    }

    pub fn summary(&self) -> GainReduction {
        let mut b = self.blocks.clone();
        b.sort_by(f32::total_cmp);
        let p99 = if b.is_empty() { 0.0 } else { b.get(((b.len() - 1) as f64 * 0.99).round() as usize).copied().unwrap_or(0.0) };
        let secs = self.block as f64 / self.sample_rate;
        GainReduction {
            max_db: f64::from(self.max),
            p99_db: f64::from(p99),
            seconds_over_1db: self.over1 as f64 * secs,
            seconds_over_2db: self.over2 as f64 * secs,
        }
    }
}

/// Clipping in a signal: samples at or over full scale, and flat tops (runs of equal samples near
/// the top of a signal's range, what a clipped source looks like whatever gain it plays at).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clipping {
    pub peak: f32,
    /// Samples at or above full scale (|x| ≥ 1).
    pub overs: u64,
    /// Runs of at least [`FLAT_RUN`] equal samples at or above [`FLAT_LEVEL`].
    pub flat_runs: u64,
}

/// A run this long of equal samples is a flat top (a sine at 1 kHz changes every sample; 3 equal
/// samples in a row at a high level do not happen in sound that was not clipped).
pub const FLAT_RUN: usize = 3;
/// Level (linear) from which equal samples count as a flat top.
pub const FLAT_LEVEL: f32 = 0.25;

/// Streaming clip detector.
#[derive(Clone, Debug, Default)]
pub struct ClipMeter {
    out: Clipping,
    /// Per channel: the last sample and how many times in a row it repeated.
    run: Vec<(f32, usize)>,
}

impl ClipMeter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process(&mut self, channels: &[&[f32]]) {
        if self.run.len() < channels.len() {
            self.run.resize(channels.len(), (0.0, 0));
        }
        for (c, (last, len)) in channels.iter().zip(self.run.iter_mut()) {
            for &x in c.iter() {
                if !x.is_finite() {
                    continue;
                }
                let a = x.abs();
                self.out.peak = self.out.peak.max(a);
                if a >= 1.0 {
                    self.out.overs += 1;
                }
                if a >= FLAT_LEVEL && x == *last {
                    *len += 1;
                    if *len + 1 == FLAT_RUN {
                        self.out.flat_runs += 1;
                    }
                } else {
                    *len = 0;
                }
                *last = x;
            }
        }
    }

    pub fn clipping(&self) -> Clipping {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;

    const SR: f64 = 48_000.0;

    fn bands_of(x: &[f32]) -> Bands {
        let mut m = BandMeter::new(SR, 1);
        // odd block sizes: the result does not depend on how the stream is cut
        for c in x.chunks(1234) {
            m.process(&[c]);
        }
        m.bands()
    }

    #[test]
    fn a_tone_lands_in_its_band() {
        for (f, band) in [(60.0, SUB), (300.0, 1), (700.0, 2), (2500.0, PRESENCE), (5000.0, 4), (9000.0, AIR)] {
            let b = bands_of(&sine(f, 0.5, SR, 48_000, 0.0));
            assert!(b.shares[band] > 0.99, "{f} Hz: {:?}", b.shares);
            assert!((b.shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // white noise spreads by bandwidth: above 6 kHz holds most of it, below 120 Hz almost none
        let mut r = Rng::new(3);
        let noise: Vec<f32> = (0..96_000).map(|_| r.uniform() * 0.3).collect();
        let b = bands_of(&noise);
        assert!((b.shares[AIR] - (24_000.0 - 6000.0) / 24_000.0).abs() < 0.02, "{:?}", b.shares);
        assert!(b.shares[SUB] < 0.01);
        assert!((b.seconds - 2.0).abs() < 1e-9);
        // octaves: white noise doubles its energy every octave up (+3 dB)
        let step = b.octaves_db[7] - b.octaves_db[6];
        assert!((step - 3.0).abs() < 0.3, "{:?}", b.octaves_db);
        let t = bands_of(&sine(250.0, 0.5, SR, 48_000, 0.0));
        assert!(t.octaves_db[3].abs() < 0.05 && t.octaves_db[5] < -40.0, "{:?}", t.octaves_db);
        // silence and nothing: no shares, no NaN
        assert_eq!(bands_of(&[0.0; 10_000]).shares, [0.0; 6]);
        assert_eq!(BandMeter::new(SR, 2).bands().shares, [0.0; 6]);
    }

    #[test]
    fn channels_add_and_hostile_input_is_harmless() {
        let lo = sine(80.0, 0.5, SR, 24_000, 0.0);
        let hi = sine(3000.0, 0.5, SR, 24_000, 0.0);
        let mut m = BandMeter::new(SR, 2);
        m.process(&[&lo, &hi]);
        let b = m.bands();
        assert!((b.shares[SUB] - 0.5).abs() < 0.01 && (b.shares[PRESENCE] - 0.5).abs() < 0.01, "{:?}", b.shares);
        // ragged, missing and non-finite channels; a zero or NaN rate
        let mut m = BandMeter::new(f64::NAN, 0);
        m.process(&[&[f32::NAN, f32::INFINITY, 1.0][..], &[][..], &[0.5; 9000][..]]);
        m.process(&[]);
        assert!(m.bands().shares.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn speech_ratio_counts_only_while_speech_plays() {
        // speech at −16 on half the points, music at −30 under it and −20 between
        let speech: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { -16.0 } else { f64::NEG_INFINITY }).collect();
        let music: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { -30.0 } else { -20.0 }).collect();
        let r = speech_ratio(&speech, &music, speech_gate(-16.0), 0.1).unwrap();
        assert_eq!(r.points, 50);
        assert!((r.median_lu - 14.0).abs() < 1e-9 && (r.p10_lu - 14.0).abs() < 1e-9);
        assert!((r.active_seconds - 5.0).abs() < 1e-9);
        // silent music counts at the floor, not as infinity
        let r = speech_ratio(&[-20.0; 4], &[f64::NEG_INFINITY; 4], -40.0, 0.1).unwrap();
        assert_eq!(r.median_lu, 50.0);
        // the quietest tenth
        let s: Vec<f64> = vec![-16.0; 10];
        let m: Vec<f64> = (0..10).map(|i| -36.0 + i as f64 * 2.0).collect();
        let r = speech_ratio(&s, &m, -40.0, 0.1).unwrap();
        assert!((r.p10_lu - 4.0).abs() < 1e-9, "{r:?}");
        // never active, or no gate
        assert!(speech_ratio(&[f64::NEG_INFINITY; 5], &[-30.0; 5], -40.0, 0.1).is_none());
        assert!(speech_ratio(&[-16.0; 5], &[-30.0; 5], speech_gate(f64::NEG_INFINITY), 0.1).is_none());
        assert_eq!(speech_gate(-70.0), -50.0);
    }

    #[test]
    fn gain_reduction_is_read_from_the_signal_before_and_after() {
        let n = 48_000;
        let pre = sine(1000.0, 1.0, SR, n, 0.0);
        // a limiter holding −3 dB on the second half
        let g = 10f32.powf(-3.0 / 20.0);
        let post: Vec<f32> = pre.iter().enumerate().map(|(i, v)| if i >= n / 2 { v * g } else { *v }).collect();
        let mut m = GainReductionMeter::new(SR);
        for (a, b) in pre.chunks(777).zip(post.chunks(777)) {
            m.process(&[a], &[b]);
        }
        let s = m.summary();
        assert!((s.max_db - 3.0).abs() < 0.01, "{s:?}");
        assert!((s.seconds_over_2db - 0.5).abs() < 0.011 && (s.seconds_over_1db - 0.5).abs() < 0.011, "{s:?}");
        assert!((s.p99_db - 3.0).abs() < 0.01);
        // untouched: none; quiet blocks are not compared; nothing fed: zeros
        let mut m = GainReductionMeter::new(SR);
        m.process(&[&pre], &[&pre]);
        assert_eq!(m.summary().max_db, 0.0);
        let quiet = sine(1000.0, 0.001, SR, n, 0.0);
        let mut m = GainReductionMeter::new(SR);
        m.process(&[&quiet], &[&[0.0; 10][..]]);
        assert_eq!(m.summary(), GainReduction::default());
        let mut m = GainReductionMeter::new(0.0);
        m.process(&[&[f32::NAN, 2.0][..]], &[]);
        assert!(m.summary().max_db.is_finite());
    }

    #[test]
    fn clipping_overs_and_flat_tops() {
        // a sine driven into a hard clip at 0.5: flat tops on every half cycle, no overs
        let clipped: Vec<f32> = sine(200.0, 1.0, SR, 4800, 0.0).iter().map(|v| v.clamp(-0.5, 0.5)).collect();
        let mut m = ClipMeter::new();
        for c in clipped.chunks(100) {
            m.process(&[c]);
        }
        let c = m.clipping();
        assert_eq!(c.overs, 0);
        assert!((c.flat_runs as i64 - 40).abs() <= 1, "{c:?}"); // 20 cycles, two tops each
        assert!((c.peak - 0.5).abs() < 1e-6);
        // a clean sine: none; one over full scale counts
        let mut m = ClipMeter::new();
        let mut s = sine(1000.0, 0.9, SR, 4800, 0.3);
        s[100] = 1.2;
        m.process(&[&s, &[f32::NAN; 3]]);
        assert_eq!((m.clipping().overs, m.clipping().flat_runs), (1, 0));
        // quiet silence and quiet repeats are not flat tops
        let mut m = ClipMeter::new();
        m.process(&[&[0.0; 100], &[0.1; 100]]);
        assert_eq!(m.clipping().flat_runs, 0);
    }
}
