//! F0 tracking of speech.
//!
//! 1. **Analysis signal.** Rates above 32 kHz are low-passed (a windowed-sinc FIR at 0.45 of
//!    the new rate) and decimated by a whole factor to 16–32 kHz; speech pitch needs no more.
//! 2. **Difference function** (YIN): for each 10 ms frame, `d(τ) = Σ (x_j − x_{j+τ})²` over a
//!    window `W` (25 ms, at least the longest period) for every lag up to the longest period,
//!    computed as `e₀ + e_τ − 2·r(τ)` with the cross-correlation `r` from one complex FFT (the
//!    window and the segment packed as its real and imaginary parts) and the energies from
//!    prefix sums. Then the cumulative-mean-normalised difference `d′(τ) = d(τ)·τ / Σ_{k≤τ} d(k)`
//!    (`d′(0) = 1`): 0 for a perfectly periodic frame, about 1 for noise.
//! 3. **Candidates**: every local minimum of `d′` between the shortest and longest period with
//!    `d′ < 0.85`, the six deepest, each refined by a parabola through `d` at the minimum and its
//!    neighbours (sub-sample lag).
//! 4. **Path** (dynamic programming over all frames): each frame is unvoiced or one of its
//!    candidates. A candidate scores its periodicity `1 − d′`, less [`OCTAVE_COST`] per octave
//!    under the frame's highest candidate (a period's multiples are as periodic as the period:
//!    the shortest wins); unvoiced scores `1 − voicing`, plus up to 1 when the frame is near
//!    silence (30–40 dB under the loud frames, or under −70 dBFS). Moving between voiced frames
//!    costs [`JUMP_COST`] per octave, between voiced and unvoiced [`VOICING_COST`]. The best
//!    path is the track: octave errors cost more than they gain, and voicing does not flicker.
//!    Voiced runs shorter than 30 ms are dropped.

use super::{HOP_S, ProsodyError, check_input, quantile, zeros};
use crate::fft::Fft;

/// Rates above twice this are decimated to between this and twice it.
const TARGET_RATE: u32 = 16_000;
/// The analysis window, seconds (at least the longest period).
const WINDOW_S: f64 = 0.025;
/// Candidates kept per frame.
const K: usize = 6;
/// The largest `d′` a candidate may have.
const CANDIDATE_MAX: f64 = 0.85;
/// Score per octave a candidate lies under the frame's highest one.
pub const OCTAVE_COST: f64 = 0.04;
/// Path cost per octave between consecutive voiced frames.
pub const JUMP_COST: f64 = 0.6;
/// Path cost of a change between voiced and unvoiced.
pub const VOICING_COST: f64 = 0.2;
/// Frames under −70 dBFS are silence.
const SILENCE_DBFS: f64 = -70.0;
/// Frames this many dB under the loud ones (95th percentile) start to count as silence, and
/// count fully [`SILENCE_SPAN_DB`] lower.
const QUIET_DB: f64 = 30.0;
const SILENCE_SPAN_DB: f64 = 10.0;
/// Voiced runs shorter than this (frames) are dropped.
const MIN_RUN: usize = 3;

/// How the tracker works.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchSettings {
    /// Lowest pitch looked for, Hz (30…1000; default 60).
    pub min_hz: f64,
    /// Highest pitch looked for, Hz (at least 1.25 × `min_hz`, at most 2000 and a quarter of the
    /// sample rate; default 500).
    pub max_hz: f64,
    /// Voicing threshold (0.05…0.95; default 0.4): a frame is voiced when its best candidate's
    /// `d′` is under it (subject to the path's costs). Higher hears more as voiced.
    pub voicing: f64,
}

impl Default for PitchSettings {
    fn default() -> Self {
        PitchSettings { min_hz: 60.0, max_hz: 500.0, voicing: 0.4 }
    }
}

impl PitchSettings {
    /// Whether the settings make sense (finite, in range, `min_hz` under `max_hz`).
    pub fn check(&self) -> Result<(), ProsodyError> {
        let bad = |s: String| Err(ProsodyError::Settings(s));
        if !(self.min_hz.is_finite() && (30.0..=1000.0).contains(&self.min_hz)) {
            return bad(format!("the lowest pitch must be 30–1000 Hz, not {}", self.min_hz));
        }
        if !(self.max_hz.is_finite() && self.max_hz >= 1.25 * self.min_hz && self.max_hz <= 2000.0) {
            return bad(format!("the highest pitch must be 1.25 × the lowest ({} Hz) to 2000 Hz, not {}", self.min_hz, self.max_hz));
        }
        if !(self.voicing.is_finite() && (0.05..=0.95).contains(&self.voicing)) {
            return bad(format!("the voicing threshold must be 0.05–0.95, not {}", self.voicing));
        }
        Ok(())
    }
}

/// The pitch of a stretch of audio, frame by frame (frame `i` centred on `i · hop` seconds).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PitchTrack {
    /// Seconds between frames ([`HOP_S`]).
    pub hop: f64,
    /// F0 in Hz, `None` where unvoiced.
    pub f0: Vec<Option<f32>>,
    /// Periodicity of each frame, 0…1: `1 − d′` at its best candidate (0 without one). High
    /// where the speech is clearly voiced, whether or not the path took it as voiced.
    pub confidence: Vec<f32>,
    /// Level of each frame (its analysis window), dBFS (−120 for digital silence).
    pub level_db: Vec<f32>,
}

impl PitchTrack {
    pub fn len(&self) -> usize {
        self.f0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.f0.is_empty()
    }

    /// Seconds at the centre of frame `i`.
    pub fn time(&self, i: usize) -> f64 {
        i as f64 * self.hop
    }

    /// Voiced frames.
    pub fn voiced(&self) -> usize {
        self.f0.iter().filter(|f| f.is_some()).count()
    }

    /// The median F0 of the voiced frames, Hz.
    pub fn median_hz(&self) -> Option<f64> {
        let v: Vec<f64> = self.f0.iter().flatten().map(|f| f64::from(*f)).collect();
        quantile(&v, 0.5)
    }

    /// Runs of voiced frames: `(first, last)`, inclusive.
    pub fn runs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, f) in self.f0.iter().enumerate() {
            match (f.is_some(), start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    out.push((s, i - 1));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push((s, self.f0.len().saturating_sub(1)));
        }
        out
    }
}

/// Track the pitch of `x` (mono at `rate`). Empty audio gives an empty track; non-finite
/// samples count as silence.
pub fn track(x: &[f32], rate: u32, s: &PitchSettings) -> Result<PitchTrack, ProsodyError> {
    check_input(x.len(), rate)?;
    s.check()?;
    if x.is_empty() {
        return Ok(PitchTrack { hop: HOP_S, ..Default::default() });
    }
    let (y, ar) = analysis_signal(x, rate)?;
    if s.max_hz > ar / 4.0 {
        return Err(ProsodyError::Settings(format!("the highest pitch {} Hz needs a sample rate of at least {} Hz", s.max_hz, 4.0 * s.max_hz)));
    }
    let frames = (x.len() as f64 / f64::from(rate) / HOP_S).floor() as usize + 1;
    let mut yin = Yin::new(ar, s)?;
    let mut cands: Vec<Cands> = zeros(frames)?;
    let mut level: Vec<f32> = zeros(frames)?;
    let mut confidence: Vec<f32> = zeros(frames)?;
    for i in 0..frames {
        let centre = (i as f64 * HOP_S * ar).round() as isize;
        let start = centre - (yin.w / 2 + yin.max_lag / 4) as isize;
        let (l, c) = yin.frame(&y, start);
        if let Some(slot) = level.get_mut(i) {
            *slot = l;
        }
        if let Some(slot) = confidence.get_mut(i) {
            *slot = c.best_dn().map_or(0.0, |d| (1.0 - d).clamp(0.0, 1.0) as f32);
        }
        if let Some(slot) = cands.get_mut(i) {
            *slot = c;
        }
    }
    let f0 = path(&cands, &level, s)?;
    Ok(PitchTrack { hop: HOP_S, f0, confidence, level_db: level })
}

/// The signal analysed and its rate: decimated by a whole factor (low-passed first) when the
/// rate is over twice [`TARGET_RATE`]; non-finite samples zeroed.
fn analysis_signal(x: &[f32], rate: u32) -> Result<(Vec<f32>, f64), ProsodyError> {
    let k = (rate / TARGET_RATE).max(1) as usize;
    let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
    if k == 1 {
        let mut y: Vec<f32> = zeros(0)?;
        y.try_reserve_exact(x.len()).map_err(|_| ProsodyError::Memory)?;
        y.extend(x.iter().map(|v| clean(*v)));
        return Ok((y, f64::from(rate)));
    }
    let ar = f64::from(rate) / k as f64;
    // a windowed-sinc low-pass at 0.45 of the new rate (cycles per input sample: 0.45 / k)
    let half = 8 * k;
    let fc = 0.45 / k as f64;
    let mut h: Vec<f64> = (0..=2 * half)
        .map(|j| {
            let t = j as f64 - half as f64;
            let arg = 2.0 * fc * t;
            let sinc = if t == 0.0 { 1.0 } else { (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg) };
            let u = j as f64 / (2 * half) as f64;
            let blackman = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * u).cos() + 0.08 * (4.0 * std::f64::consts::PI * u).cos();
            sinc * blackman
        })
        .collect();
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-12 {
        h.iter_mut().for_each(|v| *v /= sum);
    }
    let n = x.len().div_ceil(k);
    let mut y: Vec<f32> = zeros(0)?;
    y.try_reserve_exact(n).map_err(|_| ProsodyError::Memory)?;
    for m in 0..n {
        let c = m * k;
        let acc: f64 = if c >= half && c + half < x.len() {
            x.get(c - half..=c + half).unwrap_or(&[]).iter().zip(&h).map(|(v, w)| f64::from(clean(*v)) * w).sum()
        } else {
            h.iter()
                .enumerate()
                .map(|(j, w)| {
                    let at = (c + j).checked_sub(half);
                    at.and_then(|a| x.get(a)).map_or(0.0, |v| f64::from(clean(*v)) * w)
                })
                .sum()
        };
        y.push(acc as f32);
    }
    Ok((y, ar))
}

/// One frame's candidates: frequencies (Hz) and `d′` at each, deepest first.
#[derive(Clone, Copy, Debug, Default)]
struct Cands {
    n: u8,
    f: [f32; K],
    dn: [f32; K],
}

impl Cands {
    fn iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.f.iter().zip(&self.dn).take(usize::from(self.n)).map(|(f, d)| (f64::from(*f), f64::from(*d)))
    }

    fn best_dn(&self) -> Option<f64> {
        self.iter().map(|(_, d)| d).min_by(f64::total_cmp)
    }
}

/// The difference-function machinery, sized for one rate and pitch range.
struct Yin {
    w: usize,
    min_lag: usize,
    max_lag: usize,
    seg: usize,
    ar: f64,
    min_hz: f64,
    max_hz: f64,
    fft: Fft,
    re: Vec<f32>,
    im: Vec<f32>,
    rr: Vec<f32>,
    ri: Vec<f32>,
    buf: Vec<f32>,
    prefix: Vec<f64>,
    d: Vec<f64>,
    dn: Vec<f64>,
}

impl Yin {
    fn new(ar: f64, s: &PitchSettings) -> Result<Yin, ProsodyError> {
        let min_lag = ((ar / s.max_hz).floor() as usize).max(2);
        let max_lag = ((ar / s.min_hz).ceil() as usize).max(min_lag + 2);
        let w = ((WINDOW_S * ar).round() as usize).max(max_lag);
        let seg = w + max_lag + 2;
        let n = seg.next_power_of_two().max(2);
        // at 32 kHz and 30 Hz the segment is about 2.2 k samples; anything far beyond is a bug
        if n > 1 << 16 {
            return Err(ProsodyError::Settings(format!("a {n}-point analysis is too large")));
        }
        Ok(Yin {
            w,
            min_lag,
            max_lag,
            seg,
            ar,
            min_hz: s.min_hz,
            max_hz: s.max_hz,
            fft: Fft::new(n),
            re: zeros(n)?,
            im: zeros(n)?,
            rr: zeros(n)?,
            ri: zeros(n)?,
            buf: zeros(seg)?,
            prefix: zeros(seg + 1)?,
            d: zeros(max_lag + 2)?,
            dn: zeros(max_lag + 2)?,
        })
    }

    /// The frame whose segment starts at sample `start` of `y`: its level (dBFS) and candidates.
    fn frame(&mut self, y: &[f32], start: isize) -> (f32, Cands) {
        let n = self.fft.len();
        for (j, b) in self.buf.iter_mut().enumerate() {
            let at = start + j as isize;
            *b = if at >= 0 { y.get(at as usize).copied().unwrap_or(0.0) } else { 0.0 };
        }
        // the window (first W samples) as the real part, the whole segment as the imaginary
        for j in 0..n {
            let v = self.buf.get(j).copied().unwrap_or(0.0);
            if let (Some(r), Some(i)) = (self.re.get_mut(j), self.im.get_mut(j)) {
                *r = if j < self.w { v } else { 0.0 };
                *i = v;
            }
        }
        self.fft.forward(&mut self.re, &mut self.im);
        // A = (Z_k + conj Z_{n−k}) / 2, B = (Z_k − conj Z_{n−k}) / 2i, R = conj(A)·B
        for k in 0..n {
            let k2 = (n - k) % n;
            let (zr, zi) = (self.re.get(k).copied().unwrap_or(0.0), self.im.get(k).copied().unwrap_or(0.0));
            let (cr, ci) = (self.re.get(k2).copied().unwrap_or(0.0), -self.im.get(k2).copied().unwrap_or(0.0));
            let (ar, ai) = (0.5 * (zr + cr), 0.5 * (zi + ci));
            let (dr, di) = (zr - cr, zi - ci);
            let (br, bi) = (0.5 * di, -0.5 * dr);
            if let (Some(r), Some(i)) = (self.rr.get_mut(k), self.ri.get_mut(k)) {
                *r = ar * br + ai * bi;
                *i = ar * bi - ai * br;
            }
        }
        self.fft.inverse(&mut self.rr, &mut self.ri);
        let mut acc = 0.0f64;
        if let Some(p0) = self.prefix.first_mut() {
            *p0 = 0.0;
        }
        for j in 0..self.seg {
            let v = f64::from(self.buf.get(j).copied().unwrap_or(0.0));
            acc += v * v;
            if let Some(p) = self.prefix.get_mut(j + 1) {
                *p = acc;
            }
        }
        let p = |i: usize| self.prefix.get(i).copied().unwrap_or(0.0);
        let e0 = p(self.w);
        let level = (10.0 * (e0 / self.w as f64 + 1e-12).log10()).max(-120.0) as f32;
        let mut c = Cands::default();
        if e0 < 1e-10 * self.w as f64 {
            return (level, c);
        }
        let inv_n = 1.0 / n as f64;
        for tau in 0..=self.max_lag + 1 {
            let et = p(tau + self.w) - p(tau);
            let r = f64::from(self.rr.get(tau).copied().unwrap_or(0.0)) * inv_n;
            if let Some(d) = self.d.get_mut(tau) {
                *d = (e0 + et - 2.0 * r).max(0.0);
            }
        }
        // cumulative-mean normalisation
        let mut sum = 0.0;
        if let Some(d0) = self.dn.first_mut() {
            *d0 = 1.0;
        }
        for tau in 1..=self.max_lag + 1 {
            let d = self.d.get(tau).copied().unwrap_or(0.0);
            sum += d;
            if let Some(dn) = self.dn.get_mut(tau) {
                *dn = if sum > 1e-12 * e0 { d * tau as f64 / sum } else { 1.0 };
            }
        }
        let mut found: Vec<(f64, f64)> = Vec::with_capacity(16);
        for tau in self.min_lag.max(1)..=self.max_lag {
            let (Some(a), Some(b), Some(cn)) = (self.dn.get(tau - 1), self.dn.get(tau), self.dn.get(tau + 1)) else { continue };
            let (a, b, cn) = (*a, *b, *cn);
            if !(b < a && b <= cn && b < CANDIDATE_MAX) {
                continue;
            }
            // the sub-sample minimum: a parabola through d at τ−1, τ, τ+1
            let (da, db, dc) =
                (self.d.get(tau - 1).copied().unwrap_or(0.0), self.d.get(tau).copied().unwrap_or(0.0), self.d.get(tau + 1).copied().unwrap_or(0.0));
            let den = da - 2.0 * db + dc;
            let delta = if den > 1e-12 { (0.5 * (da - dc) / den).clamp(-0.5, 0.5) } else { 0.0 };
            let lag = tau as f64 + delta;
            let f = self.ar / lag;
            if !(f.is_finite() && f >= 0.97 * self.min_hz && f <= 1.03 * self.max_hz) {
                continue;
            }
            // d′ at the parabola's minimum (never above the sampled one)
            let value = (b + 0.5 * (cn - a) * delta + 0.5 * (a - 2.0 * b + cn) * delta * delta).clamp(0.0, b);
            found.push((f, value));
        }
        found.sort_by(|x, y| x.1.total_cmp(&y.1));
        found.truncate(K);
        for (slot, (f, d)) in found.iter().enumerate() {
            if let (Some(fs), Some(ds)) = (c.f.get_mut(slot), c.dn.get_mut(slot)) {
                *fs = *f as f32;
                *ds = *d as f32;
            }
        }
        c.n = found.len() as u8;
        (level, c)
    }
}

/// The best path through the frames' candidates (see the module docs): F0 per frame.
fn path(cands: &[Cands], level: &[f32], s: &PitchSettings) -> Result<Vec<Option<f32>>, ProsodyError> {
    let frames = cands.len();
    let mut out: Vec<Option<f32>> = zeros(frames)?;
    if frames == 0 {
        return Ok(out);
    }
    let loud: Vec<f64> = level.iter().map(|l| f64::from(*l)).filter(|l| *l > -100.0).collect();
    let loud = quantile(&loud, 0.95).unwrap_or(-120.0);
    let unvoiced = |i: usize| {
        let l = level.get(i).map_or(-120.0, |l| f64::from(*l));
        let quiet = if l < SILENCE_DBFS { 1.0 } else { ((loud - l - QUIET_DB) / SILENCE_SPAN_DB).clamp(0.0, 1.0) };
        (1.0 - s.voicing) + quiet
    };
    let voiced = |c: &Cands| -> [f64; K] {
        let hi = c.iter().map(|(f, _)| f).fold(0.0, f64::max);
        let mut v = [f64::NEG_INFINITY; K];
        for (slot, (f, d)) in c.iter().enumerate() {
            if let Some(x) = v.get_mut(slot) {
                *x = (1.0 - d) - OCTAVE_COST * (hi / f).log2().max(0.0);
            }
        }
        v
    };
    let trans = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (None, None) => 0.0,
        (Some(x), Some(y)) => JUMP_COST * (x / y).log2().abs(),
        _ => VOICING_COST,
    };
    // states: 0 unvoiced, 1 + k candidate k
    let state_f = |c: &Cands, st: usize| -> Option<f64> { if st == 0 { None } else { c.f.get(st - 1).map(|f| f64::from(*f)) } };
    let mut back: Vec<[u8; K + 1]> = zeros(frames)?;
    let first = cands.first().copied().unwrap_or_default();
    let mut prev = [f64::NEG_INFINITY; K + 1];
    prev[0] = unvoiced(0);
    for (k, v) in voiced(&first).iter().enumerate().take(usize::from(first.n)) {
        prev[k + 1] = *v;
    }
    let mut prev_c = first;
    for i in 1..frames {
        let c = cands.get(i).copied().unwrap_or_default();
        let local_v = voiced(&c);
        let mut cur = [f64::NEG_INFINITY; K + 1];
        let mut bp = [0u8; K + 1];
        for st in 0..=usize::from(c.n) {
            let local = if st == 0 { unvoiced(i) } else { local_v.get(st - 1).copied().unwrap_or(f64::NEG_INFINITY) };
            let fb = state_f(&c, st);
            let (mut best, mut arg) = (f64::NEG_INFINITY, 0u8);
            for ps in 0..=usize::from(prev_c.n) {
                let p = prev.get(ps).copied().unwrap_or(f64::NEG_INFINITY);
                if p == f64::NEG_INFINITY {
                    continue;
                }
                let v = p - trans(state_f(&prev_c, ps), fb);
                if v > best {
                    best = v;
                    arg = ps as u8;
                }
            }
            if let (Some(cs), Some(b)) = (cur.get_mut(st), bp.get_mut(st)) {
                *cs = best + local;
                *b = arg;
            }
        }
        if let Some(b) = back.get_mut(i) {
            *b = bp;
        }
        prev = cur;
        prev_c = c;
    }
    // backtrack
    let mut st = prev.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(k, _)| k);
    for i in (0..frames).rev() {
        let c = cands.get(i).copied().unwrap_or_default();
        if let Some(o) = out.get_mut(i) {
            *o = state_f(&c, st).map(|f| f as f32);
        }
        st = back.get(i).and_then(|b| b.get(st)).map_or(0, |p| usize::from(*p));
    }
    // short voiced runs are dropped
    let mut i = 0;
    while i < frames {
        if out.get(i).is_some_and(Option::is_some) {
            let start = i;
            while i < frames && out.get(i).is_some_and(Option::is_some) {
                i += 1;
            }
            if i - start < MIN_RUN {
                out.iter_mut().skip(start).take(i - start).for_each(|o| *o = None);
            }
        } else {
            i += 1;
        }
    }
    Ok(out)
}
