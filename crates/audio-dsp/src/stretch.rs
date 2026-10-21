//! Time stretch that keeps pitch (Clip ▸ Speed/Duration ▸ Maintain Audio Pitch): WSOLA,
//! waveform-similarity overlap-add (W. Verhelst and M. Roelands, "An overlap-add technique based
//! on waveform similarity (WSOLA) for high quality time-scale modification of speech", ICASSP
//! 1993), made renderable from any position.
//!
//! ```text
//! output sample o plays input position o · speed (the clip's speed; output 0 = input 0)
//! frames:   periodic Hann windows of N = 30 ms, synthesis hop Hs = N/2 (the windows sum to one)
//! frame k:  output [k·Hs, k·Hs + N); nominal input start p_k = round((k·Hs + N/2) · speed) − N/2,
//!           so the frame's centre lands where the speed puts it
//! WSOLA:    input start q_k = p_k + δ_k, |δ_k| ≤ 12.5 ms: the candidate most like the natural
//!           continuation of the previous frame (input [q_{k−1} + Hs, + N)), by normalised
//!           cross-correlation summed over every channel, so all channels splice at the same
//!           points (the stereo image and inter-channel phase stay put). Searched at ~12 kHz
//!           (box-decimated), refined at the full rate around the two best coarse lags; a 1 %
//!           preference for the nominal position breaks ties (a periodic signal has many).
//! segments: the chain of choices restarts every SEGMENT = 128 frames (1.92 s), so any output
//!           range renders from at most three chains, not from the start of the clip. Chain j
//!           starts on the nominal position WARMUP frames before its junction window and runs to
//!           the end of the next junction window. At junction j (JUNCTION frames on each side of
//!           frame j·SEGMENT) the output moves from chain j−1 to chain j: at the first frame
//!           where the two chose the same input (they lock onto the same waveform, usually within
//!           a few frames of speech), otherwise along a bridge whose choices are pulled from
//!           chain j−1's continuation towards chain j's frames over the window — a sustained tone
//!           then shifts by a fraction of a period per frame instead of cancelling for a frame.
//! ```
//!
//! The output is a pure function of (input, speed, sample rate, channel count, output position):
//! rendering a range in one call or in pieces, in any order, with a fresh or a reused
//! [`TimeStretch`], gives the same samples. Memory is bounded: a few memoised chains of frame
//! positions and one input window of at most [`CHUNK`] samples plus a frame's search region per
//! channel; nothing is sized by the clip's length. Hostile parameters (a speed outside
//! [`MIN_SPEED`]…[`MAX_SPEED`] or not finite, a sample rate outside [`MIN_RATE`]…[`MAX_RATE`], no
//! or more than [`MAX_CHANNELS`] channels, positions beyond ±[`MAX_POSITION`]) are errors, and
//! non-finite input samples read as silence.

use std::fmt;

/// Slowest supported speed (1/16 = 6.25 %).
pub const MIN_SPEED: f64 = 1.0 / 16.0;
/// Fastest supported speed (16 = 1600 %).
pub const MAX_SPEED: f64 = 16.0;
/// Sample-rate range.
pub const MIN_RATE: u32 = 1_000;
pub const MAX_RATE: u32 = 768_000;
/// Most channels one stretcher handles.
pub const MAX_CHANNELS: usize = 64;
/// Output positions (samples from output 0) are limited to ±2⁴⁶ (≈ 46 years at 48 kHz), so
/// frame and input positions stay exact in `f64` and `i64`.
pub const MAX_POSITION: i64 = 1 << 46;
/// Contiguous input read size (samples) when consecutive frames' regions overlap.
pub const CHUNK: usize = 1 << 16;
/// Longest output of [`stretch`] (samples per channel).
pub const MAX_OFFLINE: usize = 1 << 28;

/// Analysis/synthesis frame length (seconds).
const FRAME_S: f64 = 0.030;
/// Search tolerance on each side of the nominal input position (seconds).
const TOLERANCE_S: f64 = 0.0125;
/// Rate of the coarse search.
const COARSE_RATE: f64 = 12_000.0;
/// Frames per segment (one chain each).
const SEGMENT: i64 = 128;
/// Frames on each side of a junction.
const JUNCTION: i64 = 6;
/// Frames a chain runs before its junction window.
const WARMUP: i64 = 8;
/// Memoised chains and bridges (each a few hundred bytes).
const MEMO: usize = 24;
/// Preference for the nominal position: a candidate at the edge of the tolerance loses this much
/// normalised correlation.
const BIAS: f64 = 0.01;
/// A bridge frame's shift towards the next chain takes the smallest alignment: candidates a period
/// apart differ by more than an integer position's phase error costs (≤ 0.01 below ~1 kHz).
const SHIFT_BIAS: f64 = 0.2;

/// Why a stretch cannot be made or rendered.
#[derive(Clone, Debug, PartialEq)]
pub enum StretchError {
    /// The speed is not finite or outside [`MIN_SPEED`]…[`MAX_SPEED`].
    Speed(f64),
    /// The sample rate is outside [`MIN_RATE`]…[`MAX_RATE`].
    SampleRate(u32),
    /// No channels, or more than [`MAX_CHANNELS`].
    Channels(usize),
    /// An output position beyond ±[`MAX_POSITION`].
    Position(i64),
    /// An offline stretch longer than [`MAX_OFFLINE`] samples.
    Length(usize),
}

impl fmt::Display for StretchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StretchError::Speed(s) => write!(f, "time stretch: speed {s} is outside {MIN_SPEED}…{MAX_SPEED}"),
            StretchError::SampleRate(r) => write!(f, "time stretch: sample rate {r} Hz is outside {MIN_RATE}…{MAX_RATE} Hz"),
            StretchError::Channels(c) => write!(f, "time stretch: {c} channels (1…{MAX_CHANNELS} supported)"),
            StretchError::Position(p) => write!(f, "time stretch: position {p} is out of range"),
            StretchError::Length(n) => write!(f, "time stretch: {n} output samples is too long (at most {MAX_OFFLINE})"),
        }
    }
}

impl std::error::Error for StretchError {}

/// Whether [`TimeStretch`] supports `speed` (finite, [`MIN_SPEED`]…[`MAX_SPEED`]).
pub fn speed_supported(speed: f64) -> bool {
    speed.is_finite() && (MIN_SPEED..=MAX_SPEED).contains(&speed)
}

/// Reads planar input samples `[start, start + len)`: one `Vec` per channel. Missing channels or
/// samples are silence, extra ones are ignored, non-finite samples read as zero. Positions before
/// the input's start (negative) or past its end are silence for the caller to supply.
pub type Reader<'a> = dyn FnMut(i64, usize) -> Vec<Vec<f32>> + 'a;

/// A pitch-keeping time stretch of one signal at one speed. See the module docs.
pub struct TimeStretch {
    speed: f64,
    channels: usize,
    /// Synthesis hop.
    hs: usize,
    /// Frame length (2·hs).
    n: usize,
    /// Search tolerance (samples each side).
    tol: usize,
    /// Coarse-search decimation.
    dec: usize,
    window: Vec<f32>,
    /// Segment j → q for frames [first(j), last(j)].
    chains: Vec<(i64, Vec<i64>)>,
    /// Junction j → q for frames [j·SEGMENT − JUNCTION, j·SEGMENT + JUNCTION].
    bridges: Vec<(i64, Vec<i64>)>,
    scratch: Scratch,
}

/// Search scratch, reused between frames.
#[derive(Default)]
struct Scratch {
    /// Decimated templates: target × channel × nd.
    tdec: Vec<f32>,
    /// Decimated region: channel × rd.
    rdec: Vec<f32>,
    /// Coarse scores per lag.
    scores: Vec<f64>,
}

fn first_frame(j: i64) -> i64 {
    j * SEGMENT - JUNCTION - WARMUP
}

fn last_frame(j: i64) -> i64 {
    (j + 1) * SEGMENT + JUNCTION
}

fn memo_find(list: &[(i64, Vec<i64>)], j: i64) -> Option<usize> {
    list.iter().position(|(k, _)| *k == j)
}

fn memo_put(list: &mut Vec<(i64, Vec<i64>)>, j: i64, v: Vec<i64>) -> usize {
    if list.len() >= MEMO {
        list.remove(0);
    }
    list.push((j, v));
    list.len() - 1
}

/// `Σ a·b` over the common length, eight partial sums added in a fixed order (deterministic and
/// vectorisable).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = 0f64;
    for v in acc {
        s += v as f64;
    }
    for (x, y) in ra.iter().zip(rb) {
        s += (*x as f64) * (*y as f64);
    }
    s
}

/// The input window: one buffer of planar samples covering `[start, start + len)`.
struct Input<'a, 'r> {
    read: &'a mut Reader<'r>,
    channels: usize,
    chunk: usize,
    start: i64,
    len: usize,
    buf: Vec<Vec<f32>>,
}

impl<'a, 'r> Input<'a, 'r> {
    fn new(read: &'a mut Reader<'r>, channels: usize, chunk: usize) -> Self {
        Input { read, channels, chunk, start: 0, len: 0, buf: vec![Vec::new(); channels] }
    }

    /// Read `[at, at + len)` into `dst[c][off..off + len]`.
    fn fill(read: &mut Reader<'r>, dst: &mut [Vec<f32>], off: usize, at: i64, len: usize) {
        if len == 0 {
            return;
        }
        let got = read(at, len);
        for (c, d) in dst.iter_mut().enumerate() {
            let Some(span) = d.get_mut(off..off.saturating_add(len)) else { continue };
            match got.get(c) {
                Some(src) => {
                    for (i, v) in span.iter_mut().enumerate() {
                        *v = src.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
                    }
                }
                None => span.fill(0.0),
            }
        }
    }

    /// Make `[lo, hi)` available, keeping what the current window already holds.
    fn ensure(&mut self, lo: i64, hi: i64) {
        if hi <= lo {
            return;
        }
        let end = self.start + self.len as i64;
        if self.len > 0 && lo >= self.start && hi <= end {
            return;
        }
        // callers keep `hi − lo` to a frame's search region (well under a second)
        let want = (hi - lo) as usize;
        let len = want.max(self.chunk);
        let new_end = lo + len as i64;
        let mut fresh: Vec<Vec<f32>> = vec![vec![0.0; len]; self.channels];
        let (keep0, keep1) = (lo.max(self.start), new_end.min(end));
        if self.len > 0 && keep1 > keep0 {
            let (src_off, dst_off, k) = ((keep0 - self.start) as usize, (keep0 - lo) as usize, (keep1 - keep0) as usize);
            for (d, s) in fresh.iter_mut().zip(&self.buf) {
                if let (Some(d), Some(s)) = (d.get_mut(dst_off..dst_off + k), s.get(src_off..src_off + k)) {
                    d.copy_from_slice(s);
                }
            }
            Self::fill(self.read, &mut fresh, 0, lo, (keep0 - lo) as usize);
            Self::fill(self.read, &mut fresh, (keep1 - lo) as usize, keep1, (new_end - keep1) as usize);
        } else {
            Self::fill(self.read, &mut fresh, 0, lo, len);
        }
        self.buf = fresh;
        self.start = lo;
        self.len = len;
    }

    /// Channel `c`, samples `[at, at + len)` (ensured by the caller; empty otherwise).
    fn get(&self, c: usize, at: i64, len: usize) -> &[f32] {
        let Some(ch) = self.buf.get(c) else { return &[] };
        if at < self.start {
            return &[];
        }
        let off = (at - self.start) as usize;
        ch.get(off..off.saturating_add(len)).unwrap_or(&[])
    }
}

impl TimeStretch {
    /// A stretch of `channels` channels at `sample_rate` playing at `speed` (input samples per
    /// output sample: 2 = twice as fast, half as long).
    pub fn new(sample_rate: u32, channels: usize, speed: f64) -> Result<TimeStretch, StretchError> {
        if !speed_supported(speed) {
            return Err(StretchError::Speed(speed));
        }
        if !(MIN_RATE..=MAX_RATE).contains(&sample_rate) {
            return Err(StretchError::SampleRate(sample_rate));
        }
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(StretchError::Channels(channels));
        }
        let sr = sample_rate as f64;
        let hs = ((sr * FRAME_S / 2.0).round() as usize).max(8);
        let n = 2 * hs;
        let dec = ((sr / COARSE_RATE).round() as usize).max(1);
        // a whole number of coarse steps
        let tol = (((sr * TOLERANCE_S).round() as usize) / dec).max(1) * dec;
        // periodic Hann: copies hs apart sum to exactly one
        let window = (0..n).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32).collect();
        Ok(TimeStretch { speed, channels, hs, n, tol, dec, window, chains: Vec::new(), bridges: Vec::new(), scratch: Scratch::default() })
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// Frame length (samples).
    pub fn frame_len(&self) -> usize {
        self.n
    }
    /// Synthesis hop (samples).
    pub fn hop(&self) -> usize {
        self.hs
    }
    /// Search tolerance on each side of the nominal input position (samples).
    pub fn tolerance(&self) -> usize {
        self.tol
    }

    /// Nominal input start of frame `k`.
    fn nominal(&self, k: i64) -> i64 {
        let half = (self.n / 2) as i64;
        let centre = (k as f64 * self.hs as f64 + half as f64) * self.speed;
        centre.round() as i64 - half
    }

    /// Render output samples `[o0, o0 + len)` into `out` (one slice per channel, all `len` long;
    /// extra channels beyond [`TimeStretch::channels`] are silent). `read` supplies the input.
    pub fn render(&mut self, read: &mut Reader<'_>, o0: i64, out: &mut [Vec<f32>]) -> Result<(), StretchError> {
        let len = out.iter().map(Vec::len).max().unwrap_or(0);
        for ch in out.iter_mut() {
            ch.clear();
            ch.resize(len, 0.0);
        }
        if !(-MAX_POSITION..=MAX_POSITION).contains(&o0) {
            return Err(StretchError::Position(o0));
        }
        let o1 = o0.saturating_add(len as i64);
        if o1 > MAX_POSITION {
            return Err(StretchError::Position(o1));
        }
        if len == 0 {
            return Ok(());
        }
        let chunk = if (self.hs as f64 * self.speed) <= (self.n + 2 * self.tol) as f64 { CHUNK } else { 0 };
        let mut input = Input::new(read, self.channels, chunk);
        let (hs, n) = (self.hs as i64, self.n as i64);
        let k0 = (o0 - n).div_euclid(hs) + 1;
        let k1 = (o1 - 1).div_euclid(hs);
        for k in k0..=k1 {
            let q = self.position(&mut input, k);
            let f0 = k * hs;
            // the part of the frame inside the request
            let i0 = (o0 - f0).max(0);
            let i1 = (o1 - f0).min(n);
            if i1 <= i0 {
                continue;
            }
            input.ensure(q + i0, q + i1);
            let w = self.window.get(i0 as usize..i1 as usize).unwrap_or(&[]);
            let at = (f0 + i0 - o0) as usize;
            for (c, dst) in out.iter_mut().enumerate().take(self.channels) {
                let src = input.get(c, q + i0, (i1 - i0) as usize);
                let Some(dst) = dst.get_mut(at..at + (i1 - i0) as usize) else { continue };
                for ((d, s), g) in dst.iter_mut().zip(src).zip(w) {
                    *d += s * g;
                }
            }
        }
        Ok(())
    }

    /// Input start of frame `k` in the output.
    fn position(&mut self, input: &mut Input, k: i64) -> i64 {
        let j = (k + JUNCTION).div_euclid(SEGMENT);
        if k <= j * SEGMENT + JUNCTION {
            let b = self.bridge(input, j);
            self.bridges.get(b).and_then(|(_, v)| v.get((k - (j * SEGMENT - JUNCTION)) as usize)).copied()
        } else {
            let c = self.chain(input, j);
            self.chains.get(c).and_then(|(_, v)| v.get((k - first_frame(j)) as usize)).copied()
        }
        .unwrap_or_else(|| self.nominal(k))
    }

    /// Index of chain `j` in the memo, computed if need be.
    fn chain(&mut self, input: &mut Input, j: i64) -> usize {
        if let Some(i) = memo_find(&self.chains, j) {
            return i;
        }
        let (first, last) = (first_frame(j), last_frame(j));
        let mut q = Vec::with_capacity((last - first + 1) as usize);
        let mut prev = self.nominal(first);
        q.push(prev);
        for k in first + 1..=last {
            let p = self.nominal(k);
            prev = self.search(input, p, &[(prev + self.hs as i64, 1.0)], BIAS);
            q.push(prev);
        }
        memo_put(&mut self.chains, j, q)
    }

    /// Index of junction `j`'s bridge in the memo, computed if need be.
    fn bridge(&mut self, input: &mut Input, j: i64) -> usize {
        if let Some(i) = memo_find(&self.bridges, j) {
            return i;
        }
        let k0 = j * SEGMENT - JUNCTION;
        let len = (2 * JUNCTION + 1) as usize;
        let ca = self.chain(input, j - 1);
        let a: Vec<i64> = self.chains.get(ca).map(|(_, v)| v.iter().skip((k0 - first_frame(j - 1)) as usize).take(len).copied().collect()).unwrap_or_default();
        let cb = self.chain(input, j);
        let b: Vec<i64> = self.chains.get(cb).map(|(_, v)| v.iter().skip((k0 - first_frame(j)) as usize).take(len).copied().collect()).unwrap_or_default();
        let bridged = if a.len() != len || b.len() != len {
            (0..len as i64).map(|i| self.nominal(k0 + i)).collect()
        } else if let Some(m) = (0..len).find(|&i| a[i] == b[i]) {
            // the chains meet: chain j−1 up to there, chain j from there (seamless)
            (0..len).map(|i| if i < m { a[i] } else { b[i] }).collect()
        } else {
            // From chain j−1 to chain j in equal steps: each frame continues the previous one
            // (WSOLA), then moves a share of the way to chain j's frame: the shift aligning the two
            // (the smallest that does), over the steps left. A sustained tone's phase moves by an
            // equal fraction of at most half a period per frame.
            let mut v = Vec::with_capacity(len);
            v.push(a[0]);
            for i in 1..len - 1 {
                let p = self.nominal(k0 + i as i64);
                let prev = v[i - 1];
                let m = self.search(input, p, &[(prev + self.hs as i64, 1.0)], BIAS);
                let shift = self.search(input, m, &[(b[i], 1.0)], SHIFT_BIAS) - m;
                let left = (len - i) as i64;
                v.push(m + (shift as f64 / left as f64).round() as i64);
            }
            v.push(b[len - 1]);
            v
        };
        memo_put(&mut self.bridges, j, bridged)
    }

    /// The input start in `[p − tol, p + tol]` whose frame best matches the weighted `targets`
    /// (template starts and weights), by normalised cross-correlation over all channels, less
    /// `bias` at the edges of the tolerance (linear in the distance from `p`).
    fn search(&mut self, input: &mut Input, p: i64, targets: &[(i64, f64)], bias: f64) -> i64 {
        let (n, tol, dec, chans) = (self.n, self.tol, self.dec, self.channels);
        let tol_i = tol as i64;
        let r0 = p - tol_i;
        let rlen = n + 2 * tol;
        // one window holding the region and every template
        let lo = targets.iter().map(|t| t.0).fold(r0, i64::min);
        let hi = targets.iter().map(|t| t.0 + n as i64).fold(r0 + rlen as i64, i64::max);
        input.ensure(lo, hi);
        // template energies; silent templates carry no weight
        let mut weights: Vec<f64> = Vec::with_capacity(targets.len());
        for &(t, w) in targets {
            let e: f64 = (0..chans)
                .map(|c| {
                    let s = input.get(c, t, n);
                    dot(s, s)
                })
                .sum();
            weights.push(if e > 1e-12 * n as f64 && w > 0.0 { w / e.sqrt() } else { 0.0 });
        }
        if weights.iter().all(|w| *w == 0.0) {
            return p;
        }
        let penalty = |delta: i64| bias * (delta.unsigned_abs() as f64) / tol as f64;
        let eps = 1e-12 * n as f64;
        // exact score of candidate start c
        let exact = |input: &Input, c: i64| -> f64 {
            let mut num = 0.0;
            let mut energy = 0.0;
            for ch in 0..chans {
                let r = input.get(ch, c, n);
                energy += dot(r, r);
                for (&(t, _), &w) in targets.iter().zip(&weights) {
                    if w != 0.0 {
                        num += w * dot(input.get(ch, t, n), r);
                    }
                }
            }
            if energy <= eps { f64::NEG_INFINITY } else { num / energy.sqrt() - penalty(c - p) }
        };
        let mut best = (f64::NEG_INFINITY, p);
        let consider = |score: f64, c: i64, best: &mut (f64, i64)| {
            if score > best.0 {
                *best = (score, c);
            }
        };
        if dec <= 1 {
            for c in r0..=p + tol_i {
                let s = exact(input, c);
                consider(s, c, &mut best);
            }
            return if best.0.is_finite() { best.1 } else { p };
        }
        // ---- coarse: box-decimated region and templates
        let nd = n / dec;
        let lags = 2 * tol / dec + 1;
        let rd = nd + lags - 1;
        let sc = &mut self.scratch;
        sc.rdec.clear();
        sc.rdec.resize(chans * rd, 0.0);
        sc.tdec.clear();
        sc.tdec.resize(targets.len() * chans * nd, 0.0);
        for ch in 0..chans {
            let r = input.get(ch, r0, rd * dec);
            for (m, d) in sc.rdec[ch * rd..(ch + 1) * rd].iter_mut().enumerate() {
                *d = r.get(m * dec..(m + 1) * dec).map_or(0.0, |s| s.iter().sum());
            }
            for (ti, &(t, _)) in targets.iter().enumerate() {
                let s = input.get(ch, t, nd * dec);
                let off = (ti * chans + ch) * nd;
                for (m, d) in sc.tdec[off..off + nd].iter_mut().enumerate() {
                    *d = s.get(m * dec..(m + 1) * dec).map_or(0.0, |s| s.iter().sum());
                }
            }
        }
        // decimated template energies (for the same normalisation at the coarse rate)
        let mut wd: Vec<f64> = Vec::with_capacity(targets.len());
        for (ti, &w) in weights.iter().enumerate() {
            let e: f64 = (0..chans)
                .map(|ch| {
                    let off = (ti * chans + ch) * nd;
                    let s = &sc.tdec[off..off + nd];
                    dot(s, s)
                })
                .sum();
            wd.push(if w != 0.0 && e > 0.0 { targets[ti].1 / e.sqrt() } else { 0.0 });
        }
        // running region energy over the lags
        let mut energy: f64 = (0..chans)
            .map(|ch| {
                let s = &sc.rdec[ch * rd..ch * rd + nd];
                dot(s, s)
            })
            .sum();
        sc.scores.clear();
        for l in 0..lags {
            if l > 0 {
                for ch in 0..chans {
                    let base = ch * rd;
                    let (out_v, in_v) = (sc.rdec[base + l - 1] as f64, sc.rdec[base + l + nd - 1] as f64);
                    energy += in_v * in_v - out_v * out_v;
                }
            }
            let mut num = 0.0;
            for (ti, &w) in wd.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                for ch in 0..chans {
                    let off = (ti * chans + ch) * nd;
                    num += w * dot(&sc.tdec[off..off + nd], &sc.rdec[ch * rd + l..ch * rd + l + nd]);
                }
            }
            let delta = (l * dec) as i64 - tol_i;
            let s = if energy > eps * dec as f64 { num / energy.max(0.0).sqrt() - penalty(delta) } else { f64::NEG_INFINITY };
            sc.scores.push(s);
        }
        // the two best coarse lags (not neighbours)
        let mut picks: [Option<usize>; 2] = [None, None];
        for (l, &s) in sc.scores.iter().enumerate() {
            if !s.is_finite() {
                continue;
            }
            if picks[0].is_none_or(|b| s > sc.scores[b]) {
                picks[0] = Some(l);
            }
        }
        for (l, &s) in sc.scores.iter().enumerate() {
            if !s.is_finite() || picks[0].is_some_and(|b| l.abs_diff(b) < 2) {
                continue;
            }
            if picks[1].is_none_or(|b| s > sc.scores[b]) {
                picks[1] = Some(l);
            }
        }
        // ---- fine: the full rate around each pick
        for l in picks.into_iter().flatten() {
            let centre = r0 + (l * dec) as i64;
            let from = (centre - dec as i64 + 1).max(r0);
            let to = (centre + dec as i64 - 1).min(p + tol_i);
            for c in from..=to {
                let s = exact(input, c);
                consider(s, c, &mut best);
            }
        }
        if best.0.is_finite() { best.1 } else { p }
    }
}

/// Stretch whole planar buffers (all channels as long as the first) at `speed`: the output has
/// `round(len / speed)` samples per channel. For tests and offline tools; long clips render in
/// pieces with [`TimeStretch::render`].
pub fn stretch(channels: &[&[f32]], sample_rate: u32, speed: f64) -> Result<Vec<Vec<f32>>, StretchError> {
    let mut ts = TimeStretch::new(sample_rate, channels.len().max(1), speed)?;
    let len = channels.first().map_or(0, |c| c.len());
    let out_len = (len as f64 / speed).round();
    if out_len >= MAX_OFFLINE as f64 {
        return Err(StretchError::Length(out_len as usize));
    }
    let mut out = vec![vec![0f32; out_len as usize]; channels.len()];
    let mut read = |at: i64, n: usize| -> Vec<Vec<f32>> {
        channels.iter().map(|c| (0..n as i64).map(|i| usize::try_from(at + i).ok().and_then(|j| c.get(j)).copied().unwrap_or(0.0)).collect()).collect()
    };
    ts.render(&mut read, 0, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biquad::{Biquad, Coeffs, FilterType};
    use crate::testutil::*;

    const SR: u32 = 48_000;

    /// A reader over in-memory planar buffers (silence outside them).
    fn reader(chs: &[Vec<f32>]) -> impl FnMut(i64, usize) -> Vec<Vec<f32>> + '_ {
        move |at: i64, n: usize| {
            chs.iter().map(|c| (0..n as i64).map(|i| usize::try_from(at + i).ok().and_then(|j| c.get(j)).copied().unwrap_or(0.0)).collect()).collect()
        }
    }

    fn render(ts: &mut TimeStretch, chs: &[Vec<f32>], o0: i64, n: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0f32; n]; chs.len()];
        let mut r = reader(chs);
        ts.render(&mut r, o0, &mut out).unwrap();
        out
    }

    /// Frequency of the strongest component of `x` (Hann window, 2¹⁶-point FFT, parabolic
    /// interpolation of the log magnitude), and the share of the energy within ±10 Hz of it.
    fn peak_hz(x: &[f32], sr: f64) -> (f64, f64) {
        let n = 1 << 16;
        let m = x.len().min(n);
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        for i in 0..m {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).cos();
            re[i] = x[i] * w as f32;
        }
        crate::fft::Fft::new(n).forward(&mut re, &mut im);
        let pow: Vec<f64> = (0..n / 2).map(|k| (re[k] as f64).powi(2) + (im[k] as f64).powi(2)).collect();
        let mag: Vec<f64> = pow.iter().map(|p| p.sqrt().max(1e-20).ln()).collect();
        let k = (2..n / 2 - 2).max_by(|a, b| mag[*a].total_cmp(&mag[*b])).unwrap();
        let (a, b, c) = (mag[k - 1], mag[k], mag[k + 1]);
        let d = 0.5 * (a - c) / (a - 2.0 * b + c);
        let f = (k as f64 + d) * sr / n as f64;
        let near = (10.0 * n as f64 / sr).ceil() as usize;
        let share = pow[k.saturating_sub(near)..(k + near + 1).min(n / 2)].iter().sum::<f64>() / pow.iter().sum::<f64>();
        (f, share)
    }

    /// Speech-like test signal: a glottal pulse train (100–180 Hz contours) through three moving
    /// formant resonators in 280 ms syllables, with fricative noise and quiet room noise between.
    pub(crate) fn speechlike(secs: f64, seed: u64) -> Vec<f32> {
        let sr = SR as f64;
        let n = (secs * sr) as usize;
        let mut rng = Rng::new(seed);
        let mut out = vec![0f32; n];
        let formants = [[700.0, 1200.0, 2600.0], [300.0, 2300.0, 3000.0], [500.0, 900.0, 2400.0], [400.0, 1900.0, 2600.0]];
        let mut bq = [Biquad::default(), Biquad::default(), Biquad::default()];
        let mut fric = Biquad::default();
        let hp = Coeffs::design(FilterType::HighPass, sr, 3500.0, 0.7, 0.0);
        let (syl, gap) = ((0.28 * sr) as usize, (0.09 * sr) as usize);
        let mut phase = 0.0f64;
        for (i, o) in out.iter_mut().enumerate() {
            let s = i / (syl + gap);
            let t = i % (syl + gap);
            let noise = rng.uniform() as f64;
            if t < syl {
                let u = t as f64 / syl as f64;
                let f0 = 110.0 + 60.0 * (std::f64::consts::PI * u).sin() + 10.0 * (s % 3) as f64;
                phase += f0 / sr;
                let pulse = if phase >= 1.0 {
                    phase -= 1.0;
                    1.0
                } else {
                    0.0
                };
                let env = (std::f64::consts::PI * u).sin().powf(0.6);
                let f = formants[s % formants.len()];
                let mut y = 0.0;
                for (b, fq) in bq.iter_mut().zip(f) {
                    let c = Coeffs::design(FilterType::BandPass, sr, fq * (1.0 + 0.1 * u), 8.0, 0.0);
                    y += b.tick(&c, pulse);
                }
                *o = (15.0 * env * y + 0.002 * noise) as f32;
            } else if s % 2 == 1 {
                let u = (t - syl) as f64 / gap as f64;
                *o = (0.15 * (std::f64::consts::PI * u).sin() * fric.tick(&hp, noise)) as f32;
            } else {
                *o = (0.002 * noise) as f32;
            }
        }
        out
    }

    fn rms_db(x: &[f32]) -> f64 {
        db(rms(x).max(1e-12))
    }

    #[test]
    fn a_tone_keeps_its_pitch_level_and_place() {
        let amp = 0.5f64;
        for speed in [0.25, 0.5, 1.5, 2.0, 4.0] {
            // 3 s of 440 Hz, then a second of silence
            let mut x = sine(440.0, amp, SR as f64, 3 * SR as usize, 0.0);
            x.resize(4 * SR as usize, 0.0);
            let out = stretch(&[&x, &x], SR, speed).unwrap();
            let (l, r) = (&out[0], &out[1]);
            assert_eq!(l.len(), (4.0 * SR as f64 / speed).round() as usize, "{speed}: exact length");
            assert_eq!(l, r, "{speed}: identical channels stay identical");
            let tone_end = 3.0 * SR as f64 / speed;
            // the steady middle: its pitch, its level, nothing but the tone
            let mid = &l[(0.2 * tone_end) as usize..(0.8 * tone_end) as usize];
            let (f, share) = peak_hz(mid, SR as f64);
            assert!((f - 440.0).abs() < 1.0, "{speed}: {f} Hz");
            assert!(share > 0.995, "{speed}: {share} of the energy within 10 Hz of the tone");
            let level = rms(mid) * std::f64::consts::SQRT_2;
            assert!(db(level / amp).abs() < 0.3, "{speed}: level {} dB", db(level / amp));
            // the tone ends where the speed puts its end: within the search tolerance and half a
            // frame of deviation, mapped to the output
            let ts = TimeStretch::new(SR, 1, speed).unwrap();
            let slack = (ts.tolerance() as f64 + ts.frame_len() as f64 / 2.0 * (speed - 1.0).abs()) / speed + ts.hop() as f64;
            let last = l.iter().rposition(|v| v.abs() > 0.05 * amp as f32).unwrap() as f64;
            assert!((last - tone_end).abs() <= slack, "{speed}: ends at {last}, expected {tone_end} ± {slack}");
        }
    }

    #[test]
    fn speed_one_is_the_identity() {
        let x = speechlike(3.0, 7);
        let out = stretch(&[&x], SR, 1.0).unwrap();
        for (i, (a, b)) in x.iter().zip(&out[0]).enumerate() {
            assert!((a - b).abs() < 1e-6, "{i}: {a} vs {b}");
        }
    }

    #[test]
    fn output_does_not_depend_on_how_it_is_cut() {
        let x = speechlike(9.0, 3);
        let y: Vec<f32> = x.iter().enumerate().map(|(i, v)| v * 0.5 + 0.1 * ((i as f32) * 0.03).sin()).collect();
        let chs = vec![x, y];
        let mut rng = Rng::new(11);
        for speed in [0.6, 1.5, 3.0] {
            // a span crossing several junctions (every 1.92 s), starting before the input
            let (o0, len) = (-3000i64, (7.0 * SR as f64 / speed) as usize);
            let whole = render(&mut TimeStretch::new(SR, 2, speed).unwrap(), &chs, o0, len);
            // in random pieces, in order, one stretcher
            let mut ts = TimeStretch::new(SR, 2, speed).unwrap();
            let mut pieces = vec![Vec::new(), Vec::new()];
            let mut at = 0usize;
            while at < len {
                let k = match rng.next_u64() % 4 {
                    0 => 1,
                    1 => 37,
                    2 => 1000 + (rng.next_u64() % 5000) as usize,
                    _ => 30_000,
                }
                .min(len - at);
                let p = render(&mut ts, &chs, o0 + at as i64, k);
                pieces[0].extend_from_slice(&p[0]);
                pieces[1].extend_from_slice(&p[1]);
                at += k;
            }
            assert!(pieces == whole, "{speed}: pieces differ from one render");
            // backwards, a fresh stretcher per piece
            for start in (0..len).step_by(41_111).rev() {
                let k = 9_000.min(len - start);
                let p = render(&mut TimeStretch::new(SR, 2, speed).unwrap(), &chs, o0 + start as i64, k);
                assert!(p[0] == whole[0][start..start + k] && p[1] == whole[1][start..start + k], "{speed}: fresh render at {start}");
            }
        }
    }

    #[test]
    fn no_clicks_or_dips_across_frames_and_junctions() {
        let amp = 0.5f32;
        let step = amp * (2.0 * std::f32::consts::PI * 440.0 / SR as f32);
        for speed in [0.5, 0.8, 1.5, 2.0] {
            let x = sine(440.0, amp as f64, SR as f64, 12 * SR as usize, 0.0);
            let out = stretch(&[&x], SR, speed).unwrap();
            // the steady part: well inside the tone, several junctions
            let len = (10.0 * SR as f64 / speed) as usize;
            let y = &out[0][SR as usize / 4..len];
            let jump = y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0f32, f32::max);
            assert!(jump < 1.1 * step, "{speed}: largest step {jump} vs a clean tone's {step}");
            // 10 ms level steady to within a fraction of a dB
            let blk = SR as usize / 100;
            let levels: Vec<f64> = y.chunks_exact(blk).map(rms_db).collect();
            let (lo, hi) = levels.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            let nominal = db(amp as f64 / std::f64::consts::SQRT_2);
            assert!(nominal - lo < 0.6 && hi - nominal < 0.3, "{speed}: 10 ms level {lo:.2}…{hi:.2} dB vs {nominal:.2}");
        }
    }

    #[test]
    fn channels_splice_together() {
        let x = speechlike(6.0, 5);
        // right = left delayed by 24 samples (0.5 ms) at −3 dB; and an exact inverse
        let d = 24usize;
        let r: Vec<f32> = (0..x.len()).map(|i| if i >= d { 0.7 * x[i - d] } else { 0.0 }).collect();
        let inv: Vec<f32> = x.iter().map(|v| -v).collect();
        for speed in [0.6, 1.5] {
            let out = stretch(&[&x, &r, &inv], SR, speed).unwrap();
            assert!(out[0].iter().zip(&out[2]).all(|(a, b)| *b == -*a), "{speed}: inverse channel spliced elsewhere");
            // the inter-channel delay and level hold over every 100 ms of speech
            let (l, rr) = (&out[0], &out[1]);
            let blk = SR as usize / 10;
            let mut checked = 0;
            let loud = rms(l);
            for b in (blk..l.len() - 2 * blk).step_by(blk) {
                let a = &l[b..b + blk];
                if rms(a) < 0.5 * loud {
                    continue;
                }
                let (mut best, mut lag) = (f64::MIN, 0i64);
                for k in -64i64..=64 {
                    let s = (b as i64 + k) as usize;
                    let c = dot(a, &rr[s..s + blk]) / (dot(a, a) * dot(&rr[s..s + blk], &rr[s..s + blk])).sqrt();
                    if c > best {
                        (best, lag) = (c, k);
                    }
                }
                assert!(lag == d as i64 && best > 0.98, "{speed} at {b}: delay {lag} (corr {best:.3})");
                checked += 1;
            }
            assert!(checked > 10, "{checked} of {}", l.len() / blk);
        }
    }

    #[test]
    fn hostile_parameters_are_errors() {
        for s in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e9, 1e-9, MAX_SPEED * 1.0001, MIN_SPEED * 0.999] {
            assert!(matches!(TimeStretch::new(SR, 2, s).err(), Some(StretchError::Speed(_))), "{s}");
            assert!(stretch(&[&[0.0; 10]], SR, s).is_err(), "{s}");
        }
        for r in [0, 1, 999, 768_001, u32::MAX] {
            assert_eq!(TimeStretch::new(r, 2, 1.5).err(), Some(StretchError::SampleRate(r)));
        }
        for c in [0, MAX_CHANNELS + 1, usize::MAX] {
            assert_eq!(TimeStretch::new(SR, c, 1.5).err(), Some(StretchError::Channels(c)));
        }
        let mut ts = TimeStretch::new(SR, 2, 1.5).unwrap();
        let mut silent = |_: i64, _: usize| Vec::new();
        for o0 in [i64::MIN, i64::MAX, MAX_POSITION + 1, -MAX_POSITION - 1, MAX_POSITION - 5] {
            let mut out = vec![vec![1.0f32; 10], vec![1.0f32; 10]];
            assert!(ts.render(&mut silent, o0, &mut out).is_err(), "{o0}");
        }
        // the extremes that are allowed render
        let mut out = vec![vec![1.0f32; 10]; 2];
        ts.render(&mut silent, MAX_POSITION - 10, &mut out).unwrap();
        ts.render(&mut silent, -MAX_POSITION, &mut out).unwrap();
        assert!(out.iter().flatten().all(|v| *v == 0.0));
        // nothing to render, no channels, ragged channels
        ts.render(&mut silent, 0, &mut []).unwrap();
        let mut ragged = vec![vec![0f32; 5], vec![0f32; 50]];
        ts.render(&mut silent, 100, &mut ragged).unwrap();
        assert_eq!((ragged[0].len(), ragged[1].len()), (50, 50));
        // empty input, NaN and infinite samples, too few or too many channels, short and long reads
        let readers: Vec<Box<dyn FnMut(i64, usize) -> Vec<Vec<f32>>>> = vec![
            Box::new(|_, _| Vec::new()),
            Box::new(|_, n| vec![vec![f32::NAN; n]; 2]),
            Box::new(|_, n| vec![vec![f32::INFINITY; n], vec![f32::NEG_INFINITY; n]]),
            Box::new(|_, n| vec![vec![0.5; n]]),
            Box::new(|_, n| vec![vec![0.1; n]; 9]),
            Box::new(|_, n| vec![vec![0.3; n / 2]; 2]),
            Box::new(|_, _| vec![vec![0.3; 1 << 20]; 2]),
        ];
        for (i, mut rd) in readers.into_iter().enumerate() {
            let mut ts = TimeStretch::new(SR, 2, 0.7).unwrap();
            let mut out = vec![vec![0f32; 200_000]; 2];
            ts.render(&mut *rd, -50_000, &mut out).unwrap();
            assert!(out.iter().flatten().all(|v| v.is_finite() && v.abs() <= 1.0), "reader {i}");
        }
        assert_eq!(stretch(&[], SR, 2.0).unwrap(), Vec::<Vec<f32>>::new());
        assert_eq!(stretch(&[&[]], SR, 2.0).unwrap(), vec![Vec::<f32>::new()]);
    }

    /// Speech measured against the phase-vocoder route (the phase-locked [`crate::effects::PitchShifter`]
    /// shifting by 1/speed, then a windowed-sinc rate change by the speed): the envelope distance
    /// (1/6-octave band levels 100 Hz–8 kHz of 43 ms frames against the input's at the mapped
    /// time, dB) and the harmonicity of voiced frames (normalised autocorrelation peak at 2–12 ms;
    /// the input's for reference). Lower distance and harmonicity closer to the input's are
    /// better. `cargo test --release -p filmcraft-audio-dsp stretch::tests::against_the_phase_vocoder -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn against_the_phase_vocoder() {
        use crate::AudioEffect;
        let sr = SR as f64;
        let x = speechlike(20.0, 9);
        let frame = 2048usize;
        let fft = crate::fft::Fft::new(frame);
        let bands: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let mut f = 100.0f64;
            while f < 8000.0 {
                let g = f * 2f64.powf(1.0 / 6.0);
                let (a, b) = ((f * frame as f64 / sr) as usize, ((g * frame as f64 / sr) as usize).max((f * frame as f64 / sr) as usize + 1));
                v.push((a, b));
                f = g;
            }
            v
        };
        let band_db = |s: &[f32], centre: i64| -> Option<Vec<f64>> {
            let start = centre - frame as i64 / 2;
            if start < 0 || start as usize + frame > s.len() {
                return None;
            }
            let mut re: Vec<f32> =
                (0..frame).map(|i| s[start as usize + i] * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame as f32).cos())).collect();
            let mut im = vec![0f32; frame];
            fft.forward(&mut re, &mut im);
            Some(bands.iter().map(|&(a, b)| 10.0 * ((a..b).map(|k| (re[k] as f64).powi(2) + (im[k] as f64).powi(2)).sum::<f64>() + 1e-9).log10()).collect())
        };
        let energy = |s: &[f32], centre: i64| -> f64 {
            let start = (centre - frame as i64 / 2).max(0) as usize;
            rms(&s[start.min(s.len())..(start + frame).min(s.len())])
        };
        let harmonicity = |s: &[f32], centre: i64| -> f64 {
            let start = (centre - frame as i64 / 2).max(0) as usize;
            let seg = &s[start..(start + frame).min(s.len())];
            let (lo, hi) = ((0.002 * sr) as usize, (0.012 * sr) as usize);
            let w = seg.len() - hi;
            (lo..hi)
                .map(|t| dot(&seg[..w], &seg[t..t + w]) / (dot(&seg[..w], &seg[..w]) * dot(&seg[t..t + w], &seg[t..t + w])).sqrt().max(1e-12))
                .fold(f64::MIN, f64::max)
        };
        let loud = (0..x.len() as i64).step_by(512).map(|c| energy(&x, c)).fold(0.0, f64::max);
        let measure = |y: &[f32], speed: f64| -> (f64, f64, f64) {
            let (mut dist, mut n, mut h, mut hn, mut hin) = (0.0, 0usize, 0.0, 0usize, 0.0);
            let mut c = frame as i64;
            while (c as usize) + frame < y.len() {
                let cin = (c as f64 * speed).round() as i64;
                if energy(&x, cin) > loud * 0.1
                    && let (Some(a), Some(b)) = (band_db(y, c), band_db(&x, cin))
                {
                    let d: f64 = a.iter().zip(&b).map(|(p, q)| (p - q).powi(2)).sum::<f64>() / a.len() as f64;
                    dist += d.sqrt();
                    n += 1;
                    if energy(&x, cin) > loud * 0.3 {
                        h += harmonicity(y, c);
                        hin += harmonicity(&x, cin);
                        hn += 1;
                    }
                }
                c += 512;
            }
            (dist / n.max(1) as f64, h / hn.max(1) as f64, hin / hn.max(1) as f64)
        };
        for speed in [0.5, 0.75, 1.5, 2.0] {
            let w = stretch(&[&x], SR, speed).unwrap().remove(0);
            // the phase vocoder: shift by 1/speed (latency removed), then change the rate by speed
            let mut ps = crate::effects::PitchShifter::new(sr as f32, 1);
            ps.set_param("semitones", (-12.0 * speed.log2()) as f32);
            ps.reset();
            let lat = ps.latency();
            let mut buf = x.clone();
            buf.resize(x.len() + lat, 0.0);
            ps.process(&mut [&mut buf]);
            let shifted = &buf[lat..];
            let pv = crate::resample::resample_sinc(shifted, (sr * speed).round() as u32, SR);
            let (dw, hw, hin) = measure(&w, speed);
            let (dp, hp, _) = measure(&pv, speed);
            println!(
                "speed {speed}: envelope distance WSOLA {dw:.2} dB, phase vocoder {dp:.2} dB; harmonicity WSOLA {hw:.3}, phase vocoder {hp:.3}, input {hin:.3}"
            );
        }
    }

    /// Real-time factor on 60 s of 48 kHz stereo speech-like audio (release:
    /// `cargo test --release -p filmcraft-audio-dsp stretch::tests::speed -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn speed() {
        let x = speechlike(60.0, 1);
        let y: Vec<f32> = x.iter().rev().copied().collect();
        for speed in [0.5, 0.75, 1.5, 2.0, 4.0] {
            let t = std::time::Instant::now();
            let out = stretch(&[&x, &y], SR, speed).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let audio = out[0].len() as f64 / SR as f64;
            println!("speed {speed}: {audio:.1} s of output in {:.3} s — {:.0}× real time", secs, audio / secs);
            // a fresh stretcher asked for 1024 samples in the middle (a seek): its chains first
            let chs = vec![x.clone(), y.clone()];
            let t = std::time::Instant::now();
            let mid = (30.0 * SR as f64 / speed) as i64;
            render(&mut TimeStretch::new(SR, 2, speed).unwrap(), &chs, mid, 1024);
            let at_junction = (SEGMENT * TimeStretch::new(SR, 2, speed).unwrap().hop() as i64) * 9;
            let t1 = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            render(&mut TimeStretch::new(SR, 2, speed).unwrap(), &chs, at_junction, 1024);
            println!("  a seek: {:.1} ms mid-segment, {:.1} ms on a junction", t1 * 1e3, t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
