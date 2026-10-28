//! The pitch contour of speech reshaped: TD-PSOLA and the [`Shape`] built on it.
//!
//! **[`psola`]** resynthesises each voiced run (at least 40 ms) whose ratio is not 1:
//!
//! 1. *Analysis marks*, one per period: the first at the run's strongest peak within its first
//!    period, each next one at the strongest peak within ±20 % of a period from where the
//!    tracked period puts it (peaks of a 1 ms moving average, of the polarity whose peaks are
//!    larger over the run).
//! 2. *Synthesis marks* from the first analysis mark, each the local analysis period over the
//!    ratio after the last (`t += P(t) / ratio(t)`), so the duration is unchanged.
//! 3. Each synthesis mark takes the grain of the nearest analysis mark: the signal under a
//!    raised-cosine window reaching back to the previous mark and on to the next one (their
//!    halves add up to exactly one between two marks), added at the synthesis mark.
//! 4. The local energy (30 ms) is matched to the original's — more grains a second would
//!    otherwise make a raised voice louder — and the result cross-faded with the original over
//!    10 ms at each end of the run. Unvoiced audio and runs left at ratio 1 are the input's.
//!
//! **[`Shape`]** ([`reshape`]): `range` scales each voiced frame's F0 about the utterance's
//! median in the log domain (from a 50 ms median-smoothed contour, so tracking jitter is not
//! scaled), `shift` moves it all, `question_rise` adds a rise over the last half second of
//! voicing (gaps of up to 150 ms bridged) before `question_end`, and `emphasis_db` lifts
//! stressed syllables — F0 above the two syllables either side of it (log mean; full lift at 3
//! semitones), or louder than the median syllable (full lift at 6 dB) — with a raised-cosine
//! gain from the dip before each to the dip after, never past −0.2 dBFS.
//! Ratios are kept within [`MIN_RATIO`]…[`MAX_RATIO`] (an octave either way). A neutral shape
//! returns the input unchanged.

use super::features::{smoothed_log2, syllables};
use super::{PitchSettings, PitchTrack, ProsodyError, check_input, quantile, semitones, track, zeros};

/// The lowest and highest pitch ratio applied.
pub const MIN_RATIO: f64 = 0.5;
pub const MAX_RATIO: f64 = 2.0;
/// Voiced runs shorter than this (frames) are left as they are.
const MIN_RUN: usize = 4;
/// The cross-fade with the original at each end of a run, seconds.
const RAMP_S: f64 = 0.01;
/// Energy matching: gains every 5 ms over 30 ms, at most 12 dB either way.
const GAIN_STEP_S: f64 = 0.005;
const GAIN_WINDOW_S: f64 = 0.03;
const GAIN_LIMIT: f64 = 4.0;
/// A question's rise: over the last half second of voicing, bridging gaps up to 150 ms.
const QUESTION_SPAN_S: f64 = 0.5;
const QUESTION_GAP_S: f64 = 0.15;
/// Emphasis: F0 this far over the median (semitones), or a syllable this much louder than the
/// median one (dB), gets the full lift.
const STRESS_ST: f64 = 3.0;
const STRESS_DB: f64 = 6.0;
/// Syllables either side a syllable's F0 is compared with.
const NEIGHBOURS: usize = 2;
/// Peaks are kept under this.
const PEAK_CEILING: f64 = 0.977;

/// How to reshape a voice's intonation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Pitch range: deviations from the median F0 scaled (log domain), 0.5…2.0; 1 keeps them.
    pub range: f64,
    /// Shift of the whole contour, semitones (−12…12).
    pub shift: f64,
    /// Semitones added over the last voiced stretch before `question_end` (−12…12; negative: a
    /// falling end).
    pub question_rise: f64,
    /// Where the question ends, seconds from the start (`None`: the end of the audio).
    pub question_end: Option<f64>,
    /// Lift of stressed syllables, dB (0…6).
    pub emphasis_db: f64,
}

impl Default for Shape {
    fn default() -> Self {
        Shape { range: 1.0, shift: 0.0, question_rise: 0.0, question_end: None, emphasis_db: 0.0 }
    }
}

impl Shape {
    /// Whether every value is finite and in range.
    pub fn check(&self) -> Result<(), ProsodyError> {
        let within = |v: f64, lo: f64, hi: f64| v.is_finite() && v >= lo && v <= hi;
        let bad = |s: String| Err(ProsodyError::Settings(s));
        if !within(self.range, 0.5, 2.0) {
            return bad(format!("range must be 0.5–2.0, not {}", self.range));
        }
        if !within(self.shift, -12.0, 12.0) {
            return bad(format!("shift must be −12…12 semitones, not {}", self.shift));
        }
        if !within(self.question_rise, -12.0, 12.0) {
            return bad(format!("the question rise must be −12…12 semitones, not {}", self.question_rise));
        }
        if let Some(e) = self.question_end
            && !(e.is_finite() && e >= 0.0)
        {
            return bad(format!("the question's end must be seconds from the start, not {e}"));
        }
        if !within(self.emphasis_db, 0.0, 6.0) {
            return bad(format!("emphasis must be 0–6 dB, not {}", self.emphasis_db));
        }
        Ok(())
    }

    /// A shape that changes nothing.
    pub fn is_neutral(&self) -> bool {
        self.range == 1.0 && self.shift == 0.0 && self.question_rise == 0.0 && self.emphasis_db == 0.0
    }
}

/// `x` (mono at `rate`) with its intonation reshaped by `shape` (its pitch tracked with `s`).
/// The output has exactly the input's length; a neutral shape returns the input unchanged.
pub fn reshape(x: &[f32], rate: u32, shape: &Shape, s: &PitchSettings) -> Result<Vec<f32>, ProsodyError> {
    check_input(x.len(), rate)?;
    shape.check()?;
    s.check()?;
    if shape.is_neutral() || x.is_empty() {
        return Ok(x.to_vec());
    }
    let t = track(x, rate, s)?;
    let ratios = shape_ratios(&t, shape);
    let mut y = psola(x, rate, &t, &ratios)?;
    if shape.emphasis_db > 0.0 {
        emphasise(&mut y, rate, &t, shape.emphasis_db)?;
    }
    Ok(y)
}

/// The pitch ratio of every frame of `t` under `shape` (1 where unvoiced), within
/// [`MIN_RATIO`]…[`MAX_RATIO`].
pub fn shape_ratios(t: &PitchTrack, shape: &Shape) -> Vec<f64> {
    let n = t.len();
    let mut out = vec![1.0; n];
    let lf = smoothed_log2(t);
    let voiced: Vec<f64> = lf.iter().flatten().copied().collect();
    let Some(median) = quantile(&voiced, 0.5) else { return out };
    let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
    let range = if shape.range.is_finite() { shape.range } else { 1.0 };
    let rise = question(&lf, t.hop, shape);
    for (i, o) in out.iter_mut().enumerate() {
        if let Some(Some(l)) = lf.get(i) {
            let st = 12.0 * (range - 1.0) * (l - median) + fin(shape.shift) + rise.get(i).copied().unwrap_or(0.0);
            let r = (st / 12.0).exp2();
            *o = if r.is_finite() { r.clamp(MIN_RATIO, MAX_RATIO) } else { 1.0 };
        }
    }
    out
}

/// Semitones of a question's rise per frame (see the module docs).
fn question(lf: &[Option<f64>], hop: f64, shape: &Shape) -> Vec<f64> {
    let n = lf.len();
    let mut out = vec![0.0; n];
    let q = shape.question_rise;
    if q == 0.0 || !q.is_finite() || n == 0 || !(hop.is_finite() && hop > 0.0) {
        return out;
    }
    let end = shape.question_end.map_or(n - 1, |e| ((e / hop).floor().max(0.0) as usize).min(n - 1));
    let Some(e) = lf.get(..=end).and_then(|v| v.iter().rposition(Option::is_some)) else { return out };
    let span = (QUESTION_SPAN_S / hop).round() as usize;
    let max_gap = (QUESTION_GAP_S / hop).round() as usize;
    let (mut start, mut gap) = (e, 0usize);
    let mut j = e;
    while j > 0 && e - (j - 1) <= span {
        j -= 1;
        if lf.get(j).is_some_and(Option::is_some) {
            start = j;
            gap = 0;
        } else {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    let len = (e - start).max(1) as f64;
    for i in start..=e {
        if lf.get(i).is_some_and(Option::is_some) {
            let u = if e == start { 1.0 } else { (i - start) as f64 / len };
            if let Some(o) = out.get_mut(i) {
                *o = q * u * u * (3.0 - 2.0 * u);
            }
        }
    }
    // the rest of a run the end falls in holds the rise
    for i in e + 1..n {
        if !lf.get(i).is_some_and(Option::is_some) {
            break;
        }
        if let Some(o) = out.get_mut(i) {
            *o = q;
        }
    }
    out
}

/// `x` (mono at `rate`) with the pitch of each frame of `t` multiplied by `ratios` (one per
/// frame; non-finite values count as 1, the rest are kept within [`MIN_RATIO`]…[`MAX_RATIO`]).
/// The output has the input's length; unvoiced audio is passed through, and with every ratio
/// 1 the output is the input. Non-finite samples come out as silence.
pub fn psola(x: &[f32], rate: u32, t: &PitchTrack, ratios: &[f64]) -> Result<Vec<f32>, ProsodyError> {
    check_input(x.len(), rate)?;
    if ratios.len() != t.len() {
        return Err(ProsodyError::Mismatch { frames: t.len(), values: ratios.len() });
    }
    if !(t.hop.is_finite() && t.hop > 0.0) {
        return Err(ProsodyError::Settings(format!("the track's hop must be positive seconds, not {}", t.hop)));
    }
    let mut orig: Vec<f32> = zeros(0)?;
    orig.try_reserve_exact(x.len()).map_err(|_| ProsodyError::Memory)?;
    orig.extend(x.iter().map(|v| if v.is_finite() { *v } else { 0.0 }));
    let ratio = |i: usize| ratios.get(i).copied().filter(|r| r.is_finite()).map_or(1.0, |r| r.clamp(MIN_RATIO, MAX_RATIO));
    if (0..ratios.len()).all(|i| (ratio(i) - 1.0).abs() < 1e-9) {
        return Ok(orig);
    }
    let mut y = orig.clone();
    let run_ctx = Run { x: &orig, rate: f64::from(rate), t, ratio: &ratio };
    for (a, b) in t.runs() {
        if b + 1 - a < MIN_RUN || (a..=b).all(|i| (ratio(i) - 1.0).abs() < 1e-9) {
            continue;
        }
        run_ctx.render(a, b, &mut y)?;
    }
    Ok(y)
}

/// What resynthesising one voiced run needs.
struct Run<'a> {
    x: &'a [f32],
    rate: f64,
    t: &'a PitchTrack,
    ratio: &'a dyn Fn(usize) -> f64,
}

impl Run<'_> {
    /// Frame position (fractional) of sample `n`.
    fn pos(&self, n: f64) -> f64 {
        n / (self.t.hop * self.rate)
    }

    /// A per-frame value at sample `n`, linearly interpolated within frames `a..=b`.
    fn interp(&self, n: f64, a: usize, b: usize, v: &dyn Fn(usize) -> f64) -> f64 {
        let p = self.pos(n).clamp(a as f64, b as f64);
        let i = (p.floor() as usize).clamp(a, b);
        let j = (i + 1).min(b);
        let u = p - i as f64;
        v(i) * (1.0 - u) + v(j) * u
    }

    fn f0(&self, i: usize) -> f64 {
        self.t.f0.get(i).copied().flatten().map_or(100.0, f64::from).max(1.0)
    }

    /// The period (samples) at sample `n` of run `a..=b`.
    fn period(&self, n: f64, a: usize, b: usize) -> f64 {
        self.rate / self.interp(n, a, b, &|i| self.f0(i)).max(1.0)
    }

    /// Resynthesise frames `a..=b` into `y` (see the module docs).
    fn render(&self, a: usize, b: usize, y: &mut [f32]) -> Result<(), ProsodyError> {
        let len = self.x.len();
        let hop_n = self.t.hop * self.rate;
        let s0 = (((a as f64 - 0.5) * hop_n).round().max(0.0) as usize).min(len);
        let s1 = (((b as f64 + 0.5) * hop_n).round().max(0.0) as usize).min(len);
        if s1 <= s0 + 4 {
            return Ok(());
        }
        let longest = (a..=b).map(|i| self.rate / self.f0(i)).fold(0.0, f64::max).ceil() as usize + 2;
        // the 1 ms moving average the marks are placed on, over the run and a period after it
        let (lo, hi) = (s0, (s1 + longest).min(len));
        let half = ((self.rate / 2000.0) as usize).max(1);
        let mut prefix: Vec<f64> = zeros(hi - lo + 2 * half + 2)?;
        let at = |n: isize| if n >= 0 { self.x.get(n as usize).copied().map_or(0.0, f64::from) } else { 0.0 };
        let mut acc = 0.0;
        for (k, p) in prefix.iter_mut().enumerate().skip(1) {
            acc += at(lo as isize - half as isize + k as isize - 1);
            *p = acc;
        }
        let smooth = |n: usize| -> f64 {
            // mean of x[n − half ..= n + half]
            let k0 = n.saturating_sub(lo);
            let (p0, p1) = (prefix.get(k0).copied().unwrap_or(0.0), prefix.get(k0 + 2 * half + 1).copied().unwrap_or(0.0));
            (p1 - p0) / (2 * half + 1) as f64
        };
        let (mut pos_peak, mut neg_peak) = (0.0f64, 0.0f64);
        for n in s0..s1 {
            let v = smooth(n);
            pos_peak = pos_peak.max(v);
            neg_peak = neg_peak.max(-v);
        }
        let pol = if pos_peak >= neg_peak { 1.0 } else { -1.0 };
        let argmax = |from: usize, to: usize| (from..to.min(hi)).max_by(|p, q| (pol * smooth(*p)).total_cmp(&(pol * smooth(*q))));
        // analysis marks
        let mut marks: Vec<usize> = Vec::new();
        let first_period = self.period(s0 as f64, a, b).round() as usize;
        let Some(mut m) = argmax(s0, s0 + first_period.max(2)) else { return Ok(()) };
        let cap = (s1 - s0) / 2 + 8;
        loop {
            marks.push(m);
            if marks.len() > cap {
                break;
            }
            let p = self.period(m as f64, a, b);
            let pred = m as f64 + p;
            let from = (pred - 0.2 * p).round().max(0.0) as usize;
            let to = (pred + 0.2 * p).round().max(0.0) as usize + 1;
            if from >= s1 || from <= m {
                break;
            }
            match argmax(from, to) {
                Some(next) if next > m => m = next,
                _ => break,
            }
        }
        if marks.len() < 3 {
            return Ok(());
        }
        let mark = |k: usize| marks.get(k).copied().unwrap_or(0);
        let last = marks.len() - 1;
        let left = |k: usize| if k == 0 { mark(1) - mark(0) } else { mark(k) - mark(k - 1) };
        let right = |k: usize| if k >= last { mark(last) - mark(last - 1) } else { mark(k + 1) - mark(k) };
        // synthesis marks and the grain each takes
        let mut syn: Vec<(f64, usize)> = Vec::new();
        let mut s = mark(0) as f64;
        let mut k = 0usize;
        while syn.len() <= 4 * cap {
            while k < last && mark(k + 1) as f64 <= s {
                k += 1;
            }
            let j = if k < last {
                let (mk, mk1) = (mark(k) as f64, mark(k + 1) as f64);
                if s - mk <= mk1 - s { k } else { k + 1 }
            } else if s <= mark(last) as f64 + 0.5 {
                last
            } else {
                break;
            };
            // the analysis period at the grain taken: at ratio 1 the next synthesis mark is the
            // next analysis mark
            let ta = right(j) as f64;
            syn.push((s, j));
            let r = self.interp(s, a, b, &|i| (self.ratio)(i));
            s += ta / r.clamp(MIN_RATIO, MAX_RATIO);
        }
        let (Some(&(first_s, _)), Some(&(last_s, _))) = (syn.first(), syn.last()) else { return Ok(()) };
        // overlap-add into a buffer over the grains' reach
        let reach = longest + 2;
        let r0 = (first_s.round() as usize).saturating_sub(reach);
        let r1 = ((last_s.round() as usize) + reach).min(len);
        if r1 <= r0 {
            return Ok(());
        }
        let mut out: Vec<f64> = zeros(r1 - r0)?;
        for &(sp, j) in &syn {
            let (mj, l, r) = (mark(j) as isize, left(j) as isize, right(j) as isize);
            let place = sp.round() as isize;
            for o in -l..=r {
                let w = if o < 0 {
                    0.5 * (1.0 + (std::f64::consts::PI * o as f64 / l.max(1) as f64).cos())
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * o as f64 / r.max(1) as f64).cos())
                };
                let src = at(mj + o);
                let dst = place + o - r0 as isize;
                if dst >= 0
                    && let Some(d) = out.get_mut(dst as usize)
                {
                    *d += src * w;
                }
            }
        }
        // the local energy matched to the original's, and the cross-fade
        let (f0s, f1s) = (first_s.round() as usize, (last_s.round() as usize).min(len));
        if f1s <= f0s + 2 {
            return Ok(());
        }
        let span = f1s - f0s;
        let mut ex: Vec<f64> = zeros(span + 1)?;
        let mut ey: Vec<f64> = zeros(span + 1)?;
        let (mut sx, mut sy) = (0.0, 0.0);
        for i in 0..span {
            let n = f0s + i;
            let xv = at(n as isize);
            let yv = out.get(n - r0).copied().unwrap_or(0.0);
            sx += xv * xv;
            sy += yv * yv;
            if let (Some(a1), Some(b1)) = (ex.get_mut(i + 1), ey.get_mut(i + 1)) {
                *a1 = sx;
                *b1 = sy;
            }
        }
        let step = ((GAIN_STEP_S * self.rate) as usize).max(1);
        let win = ((GAIN_WINDOW_S * self.rate / 2.0) as usize).max(1);
        let gains: Vec<f64> = (0..=span.div_ceil(step))
            .map(|g| {
                let c = (g * step).min(span);
                let (u0, u1) = (c.saturating_sub(win), (c + win).min(span));
                let e_x = ex.get(u1).copied().unwrap_or(0.0) - ex.get(u0).copied().unwrap_or(0.0);
                let e_y = ey.get(u1).copied().unwrap_or(0.0) - ey.get(u0).copied().unwrap_or(0.0);
                if e_y > 1e-12 && e_x >= 0.0 { (e_x / e_y).sqrt().clamp(1.0 / GAIN_LIMIT, GAIN_LIMIT) } else { 1.0 }
            })
            .collect();
        let ramp = ((RAMP_S * self.rate) as usize).min(span / 3).max(1);
        for i in 0..=span {
            let n = f0s + i;
            let g = {
                let p = i as f64 / step as f64;
                let gi = p.floor() as usize;
                let u = p - gi as f64;
                let (ga, gb) = (gains.get(gi).copied().unwrap_or(1.0), gains.get(gi + 1).or(gains.get(gi)).copied().unwrap_or(1.0));
                ga * (1.0 - u) + gb * u
            };
            let w = if i < ramp {
                0.5 - 0.5 * (std::f64::consts::PI * i as f64 / ramp as f64).cos()
            } else if span - i < ramp {
                0.5 - 0.5 * (std::f64::consts::PI * (span - i) as f64 / ramp as f64).cos()
            } else {
                1.0
            };
            let (Some(dst), Some(v)) = (y.get_mut(n), out.get(n - r0)) else { continue };
            let xv = at(n as isize);
            *dst = (w * g * v + (1.0 - w) * xv) as f32;
        }
        Ok(())
    }
}

/// Lift the stressed syllables of `y` by up to `db` (see the module docs).
fn emphasise(y: &mut [f32], rate: u32, t: &PitchTrack, db: f64) -> Result<(), ProsodyError> {
    let syl = syllables(y, rate, t)?;
    if syl.is_empty() || !(db.is_finite() && db > 0.0) {
        return Ok(());
    }
    let median_f0 = t.median_hz();
    let levels: Vec<f64> = syl.iter().map(|s| s.level_db).collect();
    let median_level = quantile(&levels, 0.5).unwrap_or(-120.0);
    let hop_n = t.hop * f64::from(rate);
    let frames = t.len();
    let mut lift_db: Vec<f64> = zeros(frames)?;
    for (k, s) in syl.iter().enumerate() {
        // prominence against the two syllables either side (the utterance's median without them)
        let around: Vec<f64> =
            (k.saturating_sub(NEIGHBOURS)..=k + NEIGHBOURS).filter(|j| *j != k).filter_map(|j| syl.get(j).and_then(|n| n.f0_hz)).map(f64::log2).collect();
        let reference = if around.is_empty() { median_f0 } else { Some((around.iter().sum::<f64>() / around.len() as f64).exp2()) };
        let by_f0 = match (reference, s.f0_hz) {
            (Some(m), Some(f)) => (semitones(m, f) / STRESS_ST).clamp(0.0, 1.0),
            _ => 0.0,
        };
        let by_level = ((s.level_db - median_level) / STRESS_DB).clamp(0.0, 1.0);
        let mut lift = db * by_f0.max(by_level);
        if lift <= 0.0 {
            continue;
        }
        // never past the ceiling
        let (n0, n1) = (((s.start as f64 - 0.5) * hop_n).max(0.0) as usize, (((s.end as f64 + 0.5) * hop_n) as usize).min(y.len()));
        let peak = y.get(n0..n1).unwrap_or(&[]).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > 0.0 {
            lift = lift.min((20.0 * (PEAK_CEILING / f64::from(peak)).log10()).max(0.0));
        }
        let (lo, hi) = (s.start.min(s.peak), s.end.max(s.peak));
        for f in lo..=hi {
            // a raised cosine rising from the dip before to the peak and falling to the dip after
            let u = if f <= s.peak {
                if s.peak == lo { 1.0 } else { (f - lo) as f64 / (s.peak - lo) as f64 }
            } else if hi == s.peak {
                1.0
            } else {
                (hi - f) as f64 / (hi - s.peak) as f64
            };
            let bump = 0.5 - 0.5 * (std::f64::consts::PI * u).cos();
            if let Some(l) = lift_db.get_mut(f) {
                *l = l.max(lift * bump);
            }
        }
    }
    let gain = |f: usize| 10f64.powf(lift_db.get(f).copied().unwrap_or(0.0) / 20.0);
    for (n, v) in y.iter_mut().enumerate() {
        let p = n as f64 / hop_n;
        let i = p.floor() as usize;
        let u = p - i as f64;
        let g = gain(i) * (1.0 - u) + gain((i + 1).min(frames.saturating_sub(1))) * u;
        *v = (f64::from(*v) * g) as f32;
    }
    Ok(())
}
