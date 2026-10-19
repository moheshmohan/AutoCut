//! Tempo and beat tracking of music: onset strength → tempo by autocorrelation → beats by
//! dynamic programming, after D. P. W. Ellis, "Beat Tracking by Dynamic Programming", Journal of
//! New Music Research 36(1), 2007, implemented from the paper, with a local tempo that follows
//! drift.
//!
//! ```text
//! mono ─► box-filtered decimation to ≈ 8 kHz
//!      ─► STFT (≈ 32 ms Hann window, 4 ms hop) ─► 40 mel bands, dB, floor at max − 80 dB
//!      ─► first difference in time, half-wave rectified, summed over bands = onset strength
//!      ─► high-pass (≈ 0.4 Hz), Gaussian smoothing (σ = 20 ms), unit standard deviation
//! tempo: autocorrelation of the onset strength weighted by a log-Gaussian prior around 120 BPM
//!        (Ellis' tempo period strength, σ = 0.7 octave), parabolic refinement; a pulse must rise
//!        above the dip between its repetitions (noise and drones decay smoothly instead); a pick
//!        under 80 BPM whose double has a clear pulse too is doubled (half-time feel)
//! local tempo: the autocorrelation peak within ±20 % of the global period (the nearest preferred)
//!        in 8 s windows every second, median-smoothed and interpolated per frame (tempo drift, rubato)
//! beats: C(t) = O(t) + max over p in [t − 2τ(t), t − τ(t)/2] of C(p) − α·ln²((t − p)/τ(t)),
//!        α = 100, backtracked from the best end in the last period; weak beats at the edges
//!        (silence before and after the music) are dropped
//! attack: each beat moves to the sharpest rise of the signal's first-difference energy (1 ms
//!        blocks, > 3 dB over the previous 10 ms) within ±20 ms, so beats sit on the attack to
//!        the millisecond; an edge beat off the grid (its interval more than 10 % from the next
//!        one's) is dropped
//! ```
//!
//! Positions are samples of the analysed signal. Everything is deterministic. Input samples are
//! hostile: non-finite values count as silence, and the work is bounded by the input length.

use crate::fft::Fft;

/// Tempo range the tracker reports (BPM).
pub const MIN_BPM: f64 = 40.0;
pub const MAX_BPM: f64 = 240.0;
/// Centre and width (octaves) of the tempo prior.
const PRIOR_BPM: f64 = 120.0;
const PRIOR_OCTAVES: f64 = 0.7;
/// A pick slower than this whose double tempo has [`DUPLE_PULSE`] of its pulse is doubled.
const DUPLE_BELOW_BPM: f64 = 80.0;
const DUPLE_PULSE: f64 = 0.4;
/// DP tightness (Ellis' α).
const TIGHTNESS: f64 = 100.0;
/// Below this normalised autocorrelation at the beat period there is no usable pulse.
const MIN_PULSE: f64 = 0.1;
/// Local tempo windows: length and hop (seconds) and how far from the global period they look.
const LOCAL_WINDOW_S: f64 = 8.0;
const LOCAL_HOP_S: f64 = 1.0;
const LOCAL_RANGE: f64 = 1.2;
/// Width (octaves) of the preference for local periods near the global one.
const LOCAL_OCTAVES: f64 = 0.15;
/// Attack refinement: search radius, block and look-back (seconds), and the rise it needs (dB).
const ATTACK_RADIUS_S: f64 = 0.020;
const ATTACK_BLOCK_S: f64 = 0.001;
const ATTACK_BACK_S: f64 = 0.010;
const ATTACK_RISE_DB: f64 = 3.0;

/// Why no beat was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeatError {
    /// Silence, a steady tone, noise or speech: no periodic pulse.
    NoBeat,
}

impl std::fmt::Display for BeatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no beat found in the audio")
    }
}

impl std::error::Error for BeatError {}

/// What the tracker found.
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    pub sample_rate: u32,
    /// The attack of every beat (samples), increasing.
    pub beats: Vec<i64>,
    /// The mean beat period (samples): the least-squares slope of the beat positions.
    pub period: f64,
    /// Tempo of [`Beats::period`].
    pub bpm: f64,
    /// The local tempo at each beat after the first (sample position, BPM; median of five
    /// beat intervals).
    pub tempo: Vec<(i64, f64)>,
}

/// The onset strength envelope of a mono signal.
#[derive(Clone, Debug, PartialEq)]
pub struct Onsets {
    /// One value per frame, unit standard deviation.
    pub env: Vec<f32>,
    /// Frames per second.
    pub frame_rate: f64,
    /// Samples (of the input) per frame.
    pub hop: usize,
    /// Input samples from a frame's start to where an attack it reports usually is.
    pub lag: usize,
}

fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32).collect()
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// Mean of the channels, non-finite samples as silence.
fn mono(channels: &[&[f32]]) -> Vec<f32> {
    let len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let k = 1.0 / channels.len().max(1) as f32;
    (0..len).map(|i| channels.iter().map(|c| c.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0).clamp(-16.0, 16.0)).sum::<f32>() * k).collect()
}

/// Onset strength of `x` at `sample_rate` (see the module docs).
pub fn onset_strength(x: &[f32], sample_rate: u32) -> Onsets {
    let sr = sample_rate.max(1) as f64;
    // decimate to about 8 kHz with a box filter (enough for onsets)
    let d = ((sr / 8000.0).round() as usize).max(1);
    let r = sr / d as f64;
    let y: Vec<f32> = x.chunks(d).map(|c| c.iter().sum::<f32>() / d as f32).collect();
    let n = (1usize << ((0.032 * r).max(64.0).log2().round() as u32)).clamp(64, 4096);
    let hop = ((0.004 * r).round() as usize).max(1);
    let frames = if y.len() >= n { (y.len() - n) / hop + 1 } else { 0 };
    let fft = Fft::new(n);
    let w = hann(n);
    // 40 mel bands, 30 Hz to Nyquist (triangular, on the magnitude-squared spectrum)
    const BANDS: usize = 40;
    let bins = n / 2 + 1;
    let (m0, m1) = (hz_to_mel(30.0), hz_to_mel(r / 2.0));
    let edges: Vec<f64> = (0..BANDS + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (BANDS + 1) as f64)).collect();
    let mut weights: Vec<Vec<(usize, f32)>> = Vec::with_capacity(BANDS);
    for b in 0..BANDS {
        let (l, c, h) = (edges[b], edges[b + 1], edges[b + 2]);
        let mut v = Vec::new();
        for k in 0..bins {
            let f = k as f64 * r / n as f64;
            let wt = if f > l && f < c {
                (f - l) / (c - l)
            } else if f >= c && f < h {
                (h - f) / (h - c)
            } else {
                0.0
            };
            if wt > 0.0 {
                v.push((k, wt as f32));
            }
        }
        weights.push(v);
    }
    let mut db: Vec<[f32; BANDS]> = Vec::with_capacity(frames);
    let (mut re, mut im) = (vec![0f32; n], vec![0f32; n]);
    let mut top = f32::MIN;
    for t in 0..frames {
        let s = t * hop;
        for i in 0..n {
            re[i] = y.get(s + i).copied().unwrap_or(0.0) * w[i];
            im[i] = 0.0;
        }
        fft.forward(&mut re, &mut im);
        let mut row = [0f32; BANDS];
        for (b, ws) in weights.iter().enumerate() {
            let e: f32 = ws.iter().map(|&(k, wt)| wt * (re[k] * re[k] + im[k] * im[k])).sum();
            row[b] = 10.0 * e.max(1e-10).log10();
            top = top.max(row[b]);
        }
        db.push(row);
    }
    let floor = top - 80.0;
    let mut env = vec![0f32; frames];
    for t in 1..frames {
        env[t] = db[t].iter().zip(&db[t - 1]).map(|(a, b)| (a.max(floor) - b.max(floor)).max(0.0)).sum();
    }
    // high-pass at about 0.4 Hz (locally zero mean), then Gaussian smoothing (σ = 20 ms)
    let fr = r / hop as f64;
    let pole = (1.0 - 2.0 * std::f64::consts::PI * 0.4 / fr).clamp(0.0, 0.9999) as f32;
    let (mut px, mut py) = (0f32, 0f32);
    for v in env.iter_mut() {
        let x = *v;
        let yv = x - px + pole * py;
        px = x;
        py = yv;
        *v = yv;
    }
    let sigma = 0.020 * fr;
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f32> = (0..=2 * radius).map(|i| (-0.5 * ((i as f64 - radius as f64) / sigma.max(1e-9)).powi(2)).exp() as f32).collect();
    let ksum: f32 = kernel.iter().sum::<f32>().max(1e-9);
    let smooth: Vec<f32> = (0..frames)
        .map(|t| {
            let mut acc = 0f32;
            for (i, k) in kernel.iter().enumerate() {
                if let Some(v) = (t + i).checked_sub(radius).and_then(|j| env.get(j)) {
                    acc += k * v;
                }
            }
            acc / ksum
        })
        .collect();
    let len = smooth.len().max(1) as f32;
    let mean = smooth.iter().sum::<f32>() / len;
    let sd = (smooth.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / len).sqrt();
    let env = if sd > 1e-12 { smooth.iter().map(|v| v / sd).collect() } else { vec![0.0; smooth.len()] };
    Onsets { env, frame_rate: sr / (hop * d) as f64, hop: hop * d, lag: 3 * n * d / 4 }
}

/// Autocorrelation of `e` (mean removed) at lags `0..=max_lag`, per frame of overlap.
fn autocorrelation(e: &[f32], max_lag: usize) -> Vec<f64> {
    let len = e.len();
    let mean = e.iter().map(|v| *v as f64).sum::<f64>() / len.max(1) as f64;
    let z: Vec<f64> = e.iter().map(|v| *v as f64 - mean).collect();
    (0..=max_lag.min(len.saturating_sub(1)))
        .map(|lag| {
            let n = len - lag;
            z.iter().zip(&z[lag..]).map(|(a, b)| a * b).sum::<f64>() / n.max(1) as f64
        })
        .collect()
}

/// Parabolic interpolation of a peak at `k` of `v`.
fn refine(v: &[f64], k: usize) -> f64 {
    match (k.checked_sub(1).and_then(|i| v.get(i)), v.get(k), v.get(k + 1)) {
        (Some(&a), Some(&b), Some(&c)) => {
            let den = a - 2.0 * b + c;
            k as f64 + if den < 0.0 { (0.5 * (a - c) / den).clamp(-0.5, 0.5) } else { 0.0 }
        }
        _ => k as f64,
    }
}

/// Global beat period in frames (see the module docs), or `None` without a pulse.
fn global_period(env: &[f32], fr: f64) -> Option<f64> {
    let lo = (fr * 60.0 / MAX_BPM).floor().max(2.0) as usize;
    let hi = (fr * 60.0 / MIN_BPM).ceil() as usize;
    let ac = autocorrelation(env, 3 * hi + 2);
    let hi = hi.min(ac.len().saturating_sub(2) / 3);
    if hi <= lo + 2 || ac.first().is_none_or(|a| *a <= 0.0) {
        return None;
    }
    let tau0 = fr * 60.0 / PRIOR_BPM;
    let tps = |lag: usize| -> f64 {
        let w = (-0.5 * ((lag.max(1) as f64 / tau0).log2() / PRIOR_OCTAVES).powi(2)).exp();
        w * ac.get(lag).copied().unwrap_or(0.0)
    };
    let mut best: Option<(f64, usize)> = None;
    for lag in lo..=hi {
        let s = tps(lag);
        if best.is_none_or(|(b, _)| s > b) {
            best = Some((s, lag));
        }
    }
    let (score, mut lag) = best?;
    if score <= 0.0 || pulse(&ac, lag) <= MIN_PULSE * ac[0] {
        return None;
    }
    // a slow pick (under 80 BPM) whose double tempo also has a clear pulse is the half-time feel
    // of music with weak off-beats: the beat is the faster one
    while fr * 60.0 / (lag as f64) < DUPLE_BELOW_BPM && lag / 2 >= lo {
        let h = (lag / 2).saturating_sub(2)..=(lag / 2 + 2).min(ac.len().saturating_sub(1));
        let Some(k) = h.max_by(|&a, &b| ac[a].total_cmp(&ac[b])) else { break };
        if pulse(&ac, k) < DUPLE_PULSE * pulse(&ac, lag) {
            break;
        }
        lag = k;
    }
    Some(refine(&ac, lag))
}

/// How far the autocorrelation at `lag` rises above its lowest point between `lag / 2` and
/// `lag`: a periodic pulse dips between its repetitions, the slowly decaying autocorrelation of
/// noise or a drone does not.
fn pulse(ac: &[f64], lag: usize) -> f64 {
    let at = ac.get(lag).copied().unwrap_or(0.0);
    let dip = ac.get(lag / 2..lag).map_or(at, |w| w.iter().copied().fold(at, f64::min));
    at - dip
}

/// Local period (frames) at every frame: autocorrelation peaks within ±20 % of `period` in
/// sliding windows, median-smoothed, linearly interpolated; `period` where a window has no pulse.
fn local_periods(env: &[f32], fr: f64, period: f64) -> Vec<f64> {
    let n = env.len();
    let win = ((LOCAL_WINDOW_S * fr) as usize).max((4.0 * period) as usize).max(8);
    let hop = ((LOCAL_HOP_S * fr) as usize).max(1);
    if n < win + hop {
        return vec![period; n];
    }
    let lo = ((period / LOCAL_RANGE).floor() as usize).max(2);
    let hi = (period * LOCAL_RANGE).ceil() as usize;
    let mut centres: Vec<(f64, f64)> = Vec::new();
    let mut start = 0usize;
    while start + win <= n {
        let w = &env[start..start + win];
        let ac = autocorrelation(w, hi + 1);
        // the peak nearest the global period wins over a stronger one further away (a dotted or
        // syncopated passage must not pull the beat off)
        let near = |k: usize| ac[k] * (-0.5 * ((k as f64 / period).log2() / LOCAL_OCTAVES).powi(2)).exp();
        let found = (lo..=hi.min(ac.len().saturating_sub(2))).max_by(|&a, &b| near(a).total_cmp(&near(b)));
        let p = match found {
            Some(k) if ac[0] > 0.0 && pulse(&ac, k) > 0.5 * MIN_PULSE * ac[0] => refine(&ac, k),
            _ => period,
        };
        centres.push((start as f64 + win as f64 / 2.0, p));
        start += hop;
    }
    // median of three against single-window outliers
    let raw: Vec<f64> = centres.iter().map(|c| c.1).collect();
    for (i, c) in centres.iter_mut().enumerate() {
        let mut v = [raw[i.saturating_sub(1)], raw[i], raw[(i + 1).min(raw.len() - 1)]];
        v.sort_by(f64::total_cmp);
        c.1 = v[1];
    }
    let mut out = Vec::with_capacity(n);
    let mut k = 0usize;
    for t in 0..n {
        let x = t as f64;
        while k + 1 < centres.len() && centres[k + 1].0 <= x {
            k += 1;
        }
        let p = match (centres.get(k), centres.get(k + 1)) {
            (Some(a), Some(b)) if x >= a.0 => a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0).max(1e-9)).clamp(0.0, 1.0),
            (Some(a), _) => a.1,
            _ => period,
        };
        out.push(p.max(2.0));
    }
    out
}

/// Dynamic-programming beat tracker on a frame-wise period: frame indices of beats.
fn dp_beats(env: &[f32], periods: &[f64]) -> Vec<usize> {
    let n = env.len().min(periods.len());
    if n == 0 {
        return Vec::new();
    }
    let mut score = vec![0f64; n];
    let mut back = vec![usize::MAX; n];
    for t in 0..n {
        let p = periods[t];
        let lo = (p * 0.5).round().max(1.0) as usize;
        let hi = (p * 2.0).round().max(lo as f64) as usize;
        let mut best: Option<(f64, usize)> = None;
        if t >= lo {
            for prev in t.saturating_sub(hi)..=t - lo {
                let r = ((t - prev) as f64 / p).ln();
                let v = score[prev] - TIGHTNESS * r * r;
                if best.is_none_or(|(b, _)| v > b) {
                    best = Some((v, prev));
                }
            }
        }
        // a chain that only loses is not continued: the beat starts a new one
        match best.filter(|(b, _)| *b > 0.0) {
            Some((b, prev)) => {
                score[t] = env[t] as f64 + b;
                back[t] = prev;
            }
            None => score[t] = env[t] as f64,
        }
    }
    let last_p = periods[n - 1].ceil() as usize;
    let tail = n.saturating_sub(last_p + 1);
    let mut t = (tail..n).max_by(|&a, &b| score[a].total_cmp(&score[b]).then(b.cmp(&a))).unwrap_or(0);
    let mut beats = vec![t];
    // a chain is at most `n` long: bound the walk anyway
    while back[t] != usize::MAX && beats.len() <= n {
        t = back[t];
        beats.push(t);
    }
    beats.reverse();
    beats
}

fn local_max(env: &[f32], t: usize, r: usize) -> f32 {
    env.get(t.saturating_sub(r)..(t + r + 1).min(env.len())).map_or(0.0, |s| s.iter().copied().fold(0.0, f32::max))
}

/// The sharpest energy rise within ±[`ATTACK_RADIUS_S`] of `b` (samples), or `b` itself when no
/// rise reaches [`ATTACK_RISE_DB`].
fn attack_near(x: &[f32], sr: f64, b: i64) -> i64 {
    let block = ((ATTACK_BLOCK_S * sr).round() as usize).max(1);
    let back = ((ATTACK_BACK_S / ATTACK_BLOCK_S).round() as usize).max(1);
    let radius = (ATTACK_RADIUS_S * sr).round() as i64;
    // energy of the first difference: transients stand out, bass and pads (whose 1 ms energy
    // swings with their phase) hardly count
    let energy = |start: i64| -> f64 {
        if start < 1 {
            return 0.0;
        }
        let s = start as usize;
        let e = (s + block).min(x.len());
        if e <= s {
            return 0.0;
        }
        (s..e).map(|i| f64::from(x[i] - x[i - 1]).powi(2)).sum::<f64>() / block as f64
    };
    let mut best: Option<(f64, i64)> = None;
    let mut at = b - radius;
    while at <= b + radius {
        let now = energy(at);
        let before = (1..=back).map(|k| energy(at - (k * block) as i64)).sum::<f64>() / back as f64;
        let rise = 10.0 * ((now + 1e-12) / (before + 1e-12)).log10();
        if rise >= ATTACK_RISE_DB && best.is_none_or(|(r, _)| rise > r) {
            best = Some((rise, at));
        }
        at += block as i64;
    }
    match best {
        // the attack starts within the block: its first sample above a tenth of the block's peak
        Some((_, at)) => {
            let s = at.max(0) as usize;
            let seg = x.get(s..(s + block).min(x.len())).unwrap_or(&[]);
            let peak = seg.iter().fold(0f32, |m, v| m.max(v.abs()));
            let first = seg.iter().position(|v| v.abs() >= 0.1 * peak).unwrap_or(0);
            at + first as i64
        }
        None => b,
    }
}

/// Track the beats of planar audio (channels are mixed to mono).
pub fn track(channels: &[&[f32]], sample_rate: u32) -> Result<Beats, BeatError> {
    let x = mono(channels);
    let sr = sample_rate.max(1);
    let o = onset_strength(&x, sr);
    let fr = o.frame_rate;
    let period = global_period(&o.env, fr).ok_or(BeatError::NoBeat)?;
    let periods = local_periods(&o.env, fr, period);
    let mut frames = dp_beats(&o.env, &periods);
    // weak beats at the edges (silence before or after the music) are not beats
    let mut strengths: Vec<f32> = frames.iter().map(|&t| local_max(&o.env, t, 2)).collect();
    strengths.sort_by(f32::total_cmp);
    let median = strengths.get(strengths.len() / 2).copied().unwrap_or(0.0);
    while frames.first().is_some_and(|&t| local_max(&o.env, t, 2) < 0.3 * median) {
        frames.remove(0);
    }
    while frames.last().is_some_and(|&t| local_max(&o.env, t, 2) < 0.3 * median) {
        frames.pop();
    }
    let len = x.len() as i64;
    let mut beats: Vec<i64> = frames.iter().map(|&t| attack_near(&x, sr as f64, (t * o.hop + o.lag) as i64)).filter(|&b| b >= 0 && b < len).collect();
    beats.dedup();
    // edge beats off the grid (the tracker's end search can settle past the last real beat)
    // (`edge`: the outermost interval, `inner`: the one next to it)
    let off_grid = |edge: i64, inner: i64| inner <= 0 || (edge as f64 / inner as f64 - 1.0).abs() > 0.1;
    while let [.., c, b, a] = beats[..]
        && beats.len() > 3
        && off_grid(a - b, b - c)
    {
        beats.pop();
    }
    while let [a, b, c, ..] = beats[..]
        && beats.len() > 3
        && off_grid(b - a, c - b)
    {
        beats.remove(0);
    }
    if beats.len() < 4 {
        return Err(BeatError::NoBeat);
    }
    // beat numbers: an interval about twice its neighbours' skips a beat (a gap in the music)
    let intervals: Vec<f64> = beats.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let local = |i: usize| -> f64 {
        let mut v: Vec<f64> = intervals.get(i.saturating_sub(2)..(i + 3).min(intervals.len())).unwrap_or(&[]).to_vec();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(1.0).max(1.0)
    };
    let steps: Vec<f64> = intervals.iter().enumerate().map(|(i, d)| (d / local(i)).round().clamp(1.0, 8.0)).collect();
    let mut k = vec![0.0f64];
    for st in &steps {
        let last = k.last().copied().unwrap_or(0.0);
        k.push(last + st);
    }
    // the mean period: least-squares slope of position against beat number
    let m = beats.len() as f64;
    let mx = k.iter().sum::<f64>() / m;
    let my = beats.iter().map(|b| *b as f64).sum::<f64>() / m;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (kx, b) in k.iter().zip(&beats) {
        let dx = kx - mx;
        sxy += dx * (*b as f64 - my);
        sxx += dx * dx;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { period * o.hop as f64 };
    if !(slope.is_finite() && slope > 0.0) {
        return Err(BeatError::NoBeat);
    }
    let tempo = (0..intervals.len()).map(|i| (beats[i + 1], 60.0 * sr as f64 / local(i))).collect();
    Ok(Beats { sample_rate: sr, period: slope, bpm: 60.0 * sr as f64 / slope, beats, tempo })
}

/// Generated drums (kick on 1 and 3, snare on 2 and 4, hats on eighths), a bass note per beat
/// and a sustained chord pad, with the tempo moving linearly from `bpm0` to `bpm1` over
/// `seconds`. Returns stereo planar audio and the true beat positions (samples). Used by the
/// tests here and in the engine.
pub fn drift_music(sr: u32, bpm0: f64, bpm1: f64, seconds: f64, seed: u32) -> (Vec<Vec<f32>>, Vec<i64>) {
    let sr_f = sr.max(1) as f64;
    let n = (seconds.clamp(0.0, 3600.0) * sr_f) as usize;
    let mut l = vec![0f32; n];
    let mut rng = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut noise = move || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    // beat times: the phase integrates the tempo, which changes linearly with time
    let bpm_at = |t: f64| bpm0 + (bpm1 - bpm0) * (t / seconds.max(1e-9)).clamp(0.0, 1.0);
    let mut times = Vec::new();
    let mut t = 0.05;
    while t < seconds - 0.05 {
        times.push(t);
        t += 60.0 / bpm_at(t).max(1.0);
    }
    let tau = std::f64::consts::TAU;
    // pad: a C major chord fading in and out every 2 s (tonal content without onsets on beats)
    for (i, s) in l.iter_mut().enumerate() {
        let tt = i as f64 / sr_f;
        let swell = 0.5 - 0.5 * (tau * tt / 2.0).cos();
        let v: f64 = [261.63, 329.63, 392.0].iter().map(|f| (tau * f * tt).sin()).sum();
        *s += (0.03 * swell * v) as f32;
    }
    let mut beats = Vec::with_capacity(times.len());
    for (q, &bt) in times.iter().enumerate() {
        let start = (bt * sr_f).round() as usize;
        beats.push(start as i64);
        let next = times.get(q + 1).copied().unwrap_or(bt + 60.0 / bpm_at(bt).max(1.0));
        let beat = (next - bt) * sr_f;
        let dur = (beat * 0.9) as usize;
        for k in 0..dur.min(n.saturating_sub(start)) {
            let t = k as f64 / sr_f;
            let i = start + k;
            if q % 2 == 0 {
                let f = 50.0 + 90.0 * (-t * 25.0).exp();
                l[i] += (0.8 * (-t * 9.0).exp() * (tau * f * t).sin()) as f32;
            } else if t < 0.2 {
                l[i] += 0.35 * (-t as f32 * 22.0).exp() * noise();
            }
            l[i] += (0.12 * (-t * 3.0).exp() * (tau * 55.0 * 2f64.powf((q % 4) as f64 / 12.0) * t).sin()) as f32;
        }
        for h in 0..2 {
            let hs = start + (h as f64 * beat / 2.0) as usize;
            for k in 0..(sr as usize / 25).min(n.saturating_sub(hs)) {
                let t = k as f32 / sr as f32;
                l[hs + k] += 0.08 * (-t * 120.0).exp() * noise();
            }
        }
    }
    let r = l.clone();
    (vec![l, r], beats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Detected beats matched to the truth within `tol` samples: (precision, recall, worst error
    /// of the matched beats).
    fn score(found: &[i64], truth: &[i64], tol: i64) -> (f64, f64, i64) {
        let mut matched = 0usize;
        let mut worst = 0i64;
        let mut used = vec![false; truth.len()];
        for b in found {
            if let Some((i, d)) = truth.iter().enumerate().filter(|(i, _)| !used[*i]).map(|(i, t)| (i, (t - b).abs())).min_by_key(|x| x.1)
                && d <= tol
            {
                used[i] = true;
                matched += 1;
                worst = worst.max(d);
            }
        }
        (matched as f64 / found.len().max(1) as f64, matched as f64 / truth.len().max(1) as f64, worst)
    }

    fn run(m: &[Vec<f32>], sr: u32) -> Beats {
        let refs: Vec<&[f32]> = m.iter().map(Vec::as_slice).collect();
        track(&refs, sr).expect("beats")
    }

    #[test]
    fn steady_tempo_on_the_synthetic_tracks() {
        for (bpm, sr) in [(90.0, 44_100), (100.0, 22_050), (120.0, 48_000), (128.0, 22_050), (140.0, 44_100), (160.0, 22_050)] {
            let (m, truth) = crate::remix::test_music(sr, bpm, 16, 3);
            let b = run(&m, sr);
            assert!((b.bpm - bpm).abs() < 0.5, "{bpm} BPM at {sr} Hz: found {:.2}", b.bpm);
            let (p, r, worst) = score(&b.beats, &truth, (0.005 * sr as f64) as i64);
            assert!(p >= 0.98 && r >= 0.95, "{bpm} BPM: precision {p:.3} recall {r:.3}");
            eprintln!("{bpm} BPM @ {sr}: {:.2} BPM, {} beats, P {p:.3} R {r:.3}, worst {:.1} ms", b.bpm, b.beats.len(), worst as f64 * 1000.0 / sr as f64);
        }
    }

    /// Tempo that drifts linearly (up and down, ±15 %) is followed beat for beat: the local
    /// tempo, not the mean, decides each step.
    #[test]
    fn follows_tempo_drift() {
        let sr = 22_050;
        for (a, z) in [(110.0, 130.0), (132.0, 108.0), (96.0, 120.0)] {
            let (m, truth) = drift_music(sr, a, z, 40.0, 11);
            let b = run(&m, sr);
            let (p, r, worst) = score(&b.beats, &truth, (0.010 * sr as f64) as i64);
            assert!(p >= 0.95 && r >= 0.95, "{a}→{z}: precision {p:.3} recall {r:.3} ({} beats of {})", b.beats.len(), truth.len());
            let mean = (a + z) / 2.0;
            assert!((b.bpm - mean).abs() < 0.05 * mean, "{a}→{z}: mean {:.1}", b.bpm);
            // the local tempo moves the way the music does
            let (first, last) = (b.tempo[2].1, b.tempo[b.tempo.len() - 3].1);
            assert!((first - a).abs() < 4.0 && (last - z).abs() < 4.0, "{a}→{z}: local tempo {first:.1} → {last:.1}");
            eprintln!("{a}→{z} BPM: P {p:.3} R {r:.3}, worst {:.1} ms, local {first:.1} → {last:.1}", worst as f64 * 1000.0 / sr as f64);
        }
    }

    #[test]
    fn deterministic() {
        let (m, _) = drift_music(22_050, 100.0, 110.0, 12.0, 5);
        assert_eq!(run(&m, 22_050), run(&m, 22_050));
    }

    /// Silence, a steady tone, white noise, NaN and infinite samples, tiny and empty inputs and
    /// absurd sample rates: an error or beats, never a panic.
    #[test]
    fn hostile_input_never_panics() {
        let sr = 22_050u32;
        let z = vec![0f32; sr as usize * 10];
        assert_eq!(track(&[&z], sr), Err(BeatError::NoBeat));
        let tone: Vec<f32> = (0..sr as usize * 20).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        assert_eq!(track(&[&tone], sr), Err(BeatError::NoBeat));
        let mut s = 12345u32;
        let noise: Vec<f32> = (0..sr as usize * 20)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        assert_eq!(track(&[&noise], sr), Err(BeatError::NoBeat));
        let (mut m, _) = crate::remix::test_music(sr, 120.0, 8, 1);
        for (i, v) in m[0].iter_mut().enumerate() {
            if i % 997 == 0 {
                *v = f32::NAN;
            } else if i % 1499 == 0 {
                *v = f32::INFINITY;
            }
        }
        let refs: Vec<&[f32]> = m.iter().map(Vec::as_slice).collect();
        let b = track(&refs, sr).expect("NaN and inf are silence");
        assert!((b.bpm - 120.0).abs() < 1.0);
        for rate in [0u32, 1, 7, 8_000, 384_000, u32::MAX] {
            let _ = track(&[&z[..1000]], rate);
            let _ = track(&refs, rate);
        }
        assert!(track(&[], sr).is_err());
        assert!(track(&[&[]], sr).is_err());
        assert!(track(&[&[1.0, -1.0, 1.0]], sr).is_err());
    }
}
