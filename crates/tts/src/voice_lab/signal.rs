//! The signal side of the reference finder: speech activity, phrases between pauses, and the
//! quality of a stretch of speech (signal to noise, clipping, reverberation, clarity, music under
//! it, bandwidth). Pure functions over mono samples.

use serde::Serialize;

/// Analysis frames, seconds.
pub const FRAME_S: f64 = 0.01;
/// The shortest silence between two phrases, seconds: a cut is made only in one this long.
pub const MIN_PAUSE_S: f64 = 0.25;
/// The shortest phrase kept, seconds (a click is not speech).
const MIN_PHRASE_S: f64 = 0.2;
/// Frames of hang-over after speech drops below the threshold (stops and soft endings), and of
/// lead-in before it rises.
const HANG_FRAMES: usize = 8;
const LEAD_FRAMES: usize = 3;
/// Samples at or above this are at full scale (decoded AAC overshoots a little).
const FULL_SCALE: f32 = 0.985;
/// A flat top this long (samples) is clipping.
const CLIP_RUN: usize = 3;

/// A stretch of a clip, seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

impl Span {
    pub fn new(start: f64, end: f64) -> Span {
        Span { start, end }
    }

    pub fn seconds(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    pub fn overlaps(&self, o: &Span) -> bool {
        self.start < o.end && o.start < self.end
    }

    /// This span widened by `by` seconds each way (never before 0).
    pub fn widened(&self, by: f64) -> Span {
        Span { start: (self.start - by).max(0.0), end: self.end + by }
    }
}

/// Frame levels of a clip: 10 ms RMS in dBFS.
#[derive(Clone, Debug, PartialEq)]
pub struct Frames {
    pub db: Vec<f32>,
    /// Samples per frame.
    pub len: usize,
    pub rate: u32,
}

impl Frames {
    /// Seconds at the start of frame `f`.
    pub fn time(&self, f: usize) -> f64 {
        f as f64 * self.len as f64 / f64::from(self.rate.max(1))
    }

    /// The frame holding time `t` (clamped to the frames there are).
    pub fn frame(&self, t: f64) -> usize {
        if !t.is_finite() || t <= 0.0 {
            return 0;
        }
        let f = (t * f64::from(self.rate) / self.len.max(1) as f64) as usize;
        f.min(self.db.len())
    }
}

/// The 10 ms levels of `x` (mono at `rate`); invalid samples count as silence.
pub fn frames(x: &[f32], rate: u32) -> Frames {
    let len = ((f64::from(rate) * FRAME_S) as usize).max(1);
    let db = x
        .chunks(len)
        .map(|c| {
            let e = c.iter().map(|v| if v.is_finite() { f64::from(*v) * f64::from(*v) } else { 0.0 }).sum::<f64>() / c.len().max(1) as f64;
            (10.0 * (e + 1e-12).log10()) as f32
        })
        .collect();
    Frames { db, len, rate }
}

/// Which frames hold speech, and the levels the decision was made from.
#[derive(Clone, Debug, PartialEq)]
pub struct Activity {
    pub active: Vec<bool>,
    /// The clip's noise floor (10th percentile of its frames), dBFS.
    pub floor_db: f32,
    /// Its loud speech (95th percentile), dBFS.
    pub speech_db: f32,
    pub threshold_db: f32,
}

/// A percentile (0…1) of sorted values.
fn pct(sorted: &[f32], p: f64) -> Option<f32> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    sorted.get(((n - 1) as f64 * p.clamp(0.0, 1.0)) as usize).copied()
}

/// Speech activity: frames above a threshold 30 % of the way from the clip's floor to its loud
/// speech (at least 6 dB over the floor and −60 dBFS), with hang-over and a short lead-in.
pub fn activity(f: &Frames) -> Activity {
    let mut s: Vec<f32> = f.db.iter().copied().filter(|v| v.is_finite() && *v > -100.0).collect();
    s.sort_by(f32::total_cmp);
    let (Some(floor), Some(speech)) = (pct(&s, 0.10), pct(&s, 0.95)) else {
        return Activity { active: vec![false; f.db.len()], floor_db: -100.0, speech_db: -100.0, threshold_db: -60.0 };
    };
    let th = (floor + 0.3 * (speech - floor)).max(floor + 6.0).max(-60.0);
    let raw: Vec<bool> = f.db.iter().map(|v| *v >= th).collect();
    let mut active = vec![false; raw.len()];
    let mut since = usize::MAX;
    for (i, r) in raw.iter().enumerate() {
        since = if *r { 0 } else { since.saturating_add(1) };
        if since <= HANG_FRAMES
            && let Some(a) = active.get_mut(i)
        {
            *a = true;
        }
        if since == 0 {
            for j in i.saturating_sub(LEAD_FRAMES)..i {
                if let Some(a) = active.get_mut(j) {
                    *a = true;
                }
            }
        }
    }
    Activity { active, floor_db: floor, speech_db: speech, threshold_db: th }
}

/// The phrases of a clip: runs of speech, joined across silences shorter than [`MIN_PAUSE_S`],
/// phrases shorter than 0.2 s dropped.
pub fn phrases(f: &Frames, a: &Activity) -> Vec<Span> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    let n = a.active.len();
    while i < n {
        if a.active.get(i) == Some(&true) {
            let s = i;
            while a.active.get(i) == Some(&true) {
                i += 1;
            }
            runs.push((s, i));
        } else {
            i += 1;
        }
    }
    let min_gap = (MIN_PAUSE_S / FRAME_S).round() as usize;
    let mut joined: Vec<(usize, usize)> = Vec::new();
    for (s, e) in runs {
        match joined.last_mut() {
            Some(last) if s.saturating_sub(last.1) < min_gap => last.1 = e,
            _ => joined.push((s, e)),
        }
    }
    let min_len = (MIN_PHRASE_S / FRAME_S).round() as usize;
    joined.into_iter().filter(|(s, e)| e - s >= min_len).map(|(s, e)| Span::new(f.time(s), f.time(e))).collect()
}

/// Group phrases into chunks for the speaker check: consecutive phrases with gaps of at most
/// `max_gap` seconds, closed once they span `len.0` seconds and never longer than `len.1`; chunks
/// with less than `min_speech` seconds of speech are dropped.
pub fn chunks(phrases: &[Span], len: (f64, f64), max_gap: f64, min_speech: f64) -> Vec<Span> {
    let mut out = Vec::new();
    let mut cur: Option<(Span, f64)> = None;
    for p in phrases {
        cur = match cur {
            Some((c, speech)) if p.start - c.end <= max_gap && p.end - c.start <= len.1 => {
                let next = (Span::new(c.start, p.end), speech + p.seconds());
                if next.0.seconds() >= len.0 {
                    if next.1 >= min_speech {
                        out.push(next.0);
                    }
                    None
                } else {
                    Some(next)
                }
            }
            Some((c, speech)) => {
                if speech >= min_speech {
                    out.push(c);
                }
                Some((*p, p.seconds()))
            }
            None => Some((*p, p.seconds())),
        };
        if let Some((c, speech)) = cur
            && c.seconds() >= len.0
        {
            if speech >= min_speech {
                out.push(c);
            }
            cur = None;
        }
    }
    if let Some((c, speech)) = cur
        && speech >= min_speech
    {
        out.push(c);
    }
    out
}

/// The samples of `span` of `x` (mono at `rate`).
pub fn cut(x: &[f32], rate: u32, span: Span) -> &[f32] {
    let at = |t: f64| ((t.max(0.0) * f64::from(rate)) as usize).min(x.len());
    let (a, b) = (at(span.start), at(span.end));
    x.get(a..b.max(a)).unwrap_or(&[])
}

/// The speech frames of `span` only (pauses dropped): what the speaker check listens to.
pub fn speech_of(x: &[f32], f: &Frames, a: &Activity, span: Span) -> Vec<f32> {
    let (f0, f1) = (f.frame(span.start), f.frame(span.end));
    let mut out = Vec::new();
    for k in f0..f1 {
        if a.active.get(k) == Some(&true) {
            out.extend_from_slice(x.get(k * f.len..((k + 1) * f.len).min(x.len())).unwrap_or(&[]));
        }
    }
    out
}

/// How good a stretch of speech is as a voice reference.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Quality {
    /// Mean level of the speech, dBFS.
    pub speech_db: f64,
    /// The floor in the pauses around it, dBFS.
    pub noise_db: f64,
    /// Speech over the floor, dB.
    pub snr_db: f64,
    /// Share of samples in flat tops at full scale.
    pub clipped: f64,
    /// Decay time after word ends, extrapolated to −60 dB (seconds; a reverberation estimate:
    /// a close phone microphone in a soft room is about 0.2–0.4, a hall above 0.8). `None` when
    /// no word ended clearly into a pause.
    pub reverb_s: Option<f64>,
    /// Contrast of the speech's level between syllables and their dips (0…1; noise and
    /// reverberation fill the dips).
    pub clarity: f64,
    /// Share of the pauses that are tonal and loud: music (or a hum) under the speech, 0…1.
    pub music: f64,
    /// Highest frequency the speech reaches (within 45 dB of its 300–3000 Hz level), Hz.
    pub bandwidth_hz: f64,
    /// All of it as one number, 0…1 ([`Quality::rate`]).
    pub score: f64,
}

impl Quality {
    /// The score: signal to noise weighs most, then reverberation, music, clarity, bandwidth and
    /// level; clipping halves it at worst.
    pub fn rate(&mut self) {
        let s_snr = ((self.snr_db - 12.0) / 23.0).clamp(0.0, 1.0);
        let s_clip = (1.0 - self.clipped / 0.002).clamp(0.0, 1.0);
        let s_rev = self.reverb_s.map_or(0.5, |r| ((0.9 - r) / 0.6).clamp(0.0, 1.0));
        let s_music = (1.0 - self.music).clamp(0.0, 1.0);
        let s_bw = ((self.bandwidth_hz - 4000.0) / 6000.0).clamp(0.0, 1.0);
        let s_level = ((self.speech_db + 45.0) / 15.0).clamp(0.0, 1.0);
        let s = 0.35 * s_snr + 0.15 * s_rev + 0.15 * s_music + 0.10 * s_bw + 0.15 * self.clarity.clamp(0.0, 1.0) + 0.10 * s_level;
        self.score = if s.is_finite() { s * (0.5 + 0.5 * s_clip) } else { 0.0 };
    }
}

fn mean_db(levels: impl Iterator<Item = f32>) -> Option<f64> {
    let (mut e, mut n) = (0f64, 0usize);
    for l in levels {
        e += 10f64.powf(f64::from(l) / 10.0);
        n += 1;
    }
    (n > 0).then(|| 10.0 * (e / n as f64 + 1e-12).log10())
}

/// Share of samples of `x` in flat tops at full scale (runs of [`CLIP_RUN`] or more).
pub fn clipped_share(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let (mut run, mut total) = (0usize, 0usize);
    for v in x {
        if v.is_finite() && v.abs() >= FULL_SCALE {
            run += 1;
        } else {
            if run >= CLIP_RUN {
                total += run;
            }
            run = 0;
        }
    }
    if run >= CLIP_RUN {
        total += run;
    }
    total as f64 / x.len() as f64
}

/// Least-squares slope of `ys` over equally spaced points `dt` apart.
fn slope(ys: &[f64], dt: f64) -> Option<f64> {
    let n = ys.len();
    if n < 4 || dt <= 0.0 {
        return None;
    }
    let mx = (n - 1) as f64 * dt / 2.0;
    let my = ys.iter().sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, y) in ys.iter().enumerate() {
        let x = i as f64 * dt - mx;
        num += x * (y - my);
        den += x * x;
    }
    (den > 0.0).then(|| num / den)
}

/// Spectra of 1024-sample Hann frames of `x` at `rate` centred on the frames `pick` says yes to
/// (by their 10 ms frame index), as power per bin; at most `max` frames.
fn spectra(x: &[f32], f: &Frames, frames: impl Iterator<Item = usize>, max: usize) -> (Vec<Vec<f64>>, usize) {
    const N: usize = 1024;
    let fft = filmcraft_audio_dsp::fft::Fft::new(N);
    let window: Vec<f32> = (0..N).map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos()) as f32).collect();
    let mut out = Vec::new();
    let (mut re, mut im) = (vec![0f32; N], vec![0f32; N]);
    for k in frames {
        if out.len() >= max {
            break;
        }
        let centre = k * f.len + f.len / 2;
        let Some(s) = centre.checked_sub(N / 2) else { continue };
        let Some(w) = x.get(s..s + N) else { continue };
        for ((r, i), (v, h)) in re.iter_mut().zip(im.iter_mut()).zip(w.iter().zip(&window)) {
            *r = if v.is_finite() { v * h } else { 0.0 };
            *i = 0.0;
        }
        fft.forward(&mut re, &mut im);
        out.push((0..=N / 2).map(|b| f64::from(re[b]) * f64::from(re[b]) + f64::from(im[b]) * f64::from(im[b])).collect());
    }
    (out, N)
}

/// A pause frame whose spectrum has this share of its energy in narrow peaks is music (or a
/// tone): noise has almost none there (a peak 10 dB over its neighbourhood is rare in noise),
/// notes and their harmonics most of it.
const TONAL_SHARE: f64 = 0.25;

/// The share of a power spectrum's energy in narrow peaks: local maxima at least 10 dB over the
/// mean level (dB) of the 20 bins around them, with their neighbours (a windowed tone is three
/// bins wide).
fn tonal_share(band: &[f64]) -> f64 {
    let n = band.len();
    if n < 24 {
        return 0.0;
    }
    let db: Vec<f64> = band.iter().map(|v| 10.0 * (v + 1e-20).log10()).collect();
    let total: f64 = band.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return 0.0;
    }
    let mut tonal = 0.0;
    let mut k = 1;
    while k + 1 < n {
        let (Some(here), Some(l), Some(r)) = (db.get(k), db.get(k - 1), db.get(k + 1)) else { break };
        if here >= l && here >= r {
            let w = db.get(k.saturating_sub(10)..(k + 11).min(n)).unwrap_or(&[]);
            let around = w.iter().sum::<f64>() / w.len().max(1) as f64;
            if here - around >= 10.0 {
                tonal += band.get(k - 1..k + 2).map_or(0.0, |b| b.iter().sum::<f64>());
                k += 2;
                continue;
            }
        }
        k += 1;
    }
    (tonal / total).clamp(0.0, 1.0)
}

/// The quality of `span` of the clip `x` (mono at `rate`; `f`, `a` its frames and activity),
/// with the pauses within `context` (a few seconds around it) for the floor and the music.
pub fn quality(x: &[f32], f: &Frames, a: &Activity, span: Span, context: Span) -> Quality {
    let (f0, f1) = (f.frame(span.start), f.frame(span.end));
    let (c0, c1) = (f.frame(context.start.min(span.start)), f.frame(context.end.max(span.end)));
    let at = |k: usize| f.db.get(k).copied().unwrap_or(-120.0);
    let is_active = |k: usize| a.active.get(k) == Some(&true);
    let speech_db = mean_db((f0..f1).filter(|k| is_active(*k)).map(at)).unwrap_or(-120.0);
    // the floor: the quieter half of the pauses around, else the quietest frames there
    let mut pauses: Vec<f32> = (c0..c1).filter(|k| !is_active(*k)).map(at).filter(|v| *v > -100.0).collect();
    pauses.sort_by(f32::total_cmp);
    let noise_db = if pauses.len() >= 10 {
        mean_db(pauses.iter().take(pauses.len() / 2).copied()).unwrap_or(-100.0)
    } else {
        let mut all: Vec<f32> = (c0..c1).map(at).filter(|v| *v > -100.0).collect();
        all.sort_by(f32::total_cmp);
        pct(&all, 0.05).map_or(-100.0, f64::from)
    };
    let snr_db = (speech_db - noise_db).clamp(0.0, 90.0);
    let clipped = clipped_share(cut(x, f.rate, span));
    // reverberation: after each phrase end inside the span, from its last loud frame (within
    // 12 dB of the speech level), the slope of the level as it falls 3 dB under that frame down
    // to 4 dB over the floor (at least 10 dB of fall, four frames)
    let mut decays = Vec::new();
    let floor_end = noise_db + 4.0;
    let mut k = f0;
    while k + 1 < f1 {
        if is_active(k) && !is_active(k + 1) {
            let loud = (k.saturating_sub(40)..=k).rev().find(|j| f64::from(at(*j)) >= speech_db - 12.0);
            if let Some(pk) = loud {
                let top = f64::from(at(pk));
                let mut ys = Vec::new();
                for j in pk + 1..pk + 60 {
                    let l = f64::from(at(j));
                    if l <= floor_end || j >= f.db.len() {
                        break;
                    }
                    if l <= top - 3.0 {
                        ys.push(l);
                    }
                }
                let fall = ys.first().zip(ys.last()).map_or(0.0, |(a, b)| a - b);
                if fall >= 10.0
                    && let Some(sl) = slope(&ys, FRAME_S)
                    && sl < -1.0
                {
                    decays.push((60.0 / -sl).clamp(0.02, 3.0));
                }
            }
        }
        k += 1;
    }
    decays.sort_by(f64::total_cmp);
    let reverb_s = decays.get(decays.len() / 2).copied();
    // clarity: syllables against their dips
    let mut lv: Vec<f32> = (f0..f1).map(at).filter(|v| *v > -100.0).collect();
    lv.sort_by(f32::total_cmp);
    let contrast = match (pct(&lv, 0.95), pct(&lv, 0.20)) {
        (Some(hi), Some(lo)) => f64::from(hi - lo),
        _ => 0.0,
    };
    let clarity = ((contrast - 10.0) / 25.0).clamp(0.0, 1.0);
    // music: pause frames that are tonal (low spectral flatness over 100 Hz–5 kHz) and loud
    let rate = f64::from(f.rate.max(1));
    let pause_frames: Vec<usize> = (c0..c1).filter(|k| !is_active(*k) && f64::from(at(*k)) > speech_db - 35.0).collect();
    let all_pauses = (c0..c1).filter(|k| !is_active(*k)).count();
    let music = if all_pauses >= 10 {
        let (specs, n) = spectra(x, f, pause_frames.iter().copied().step_by(2), 80);
        let lo = ((200.0 * n as f64 / rate) as usize).max(1);
        let hi = ((5000.0 * n as f64 / rate) as usize).min(n / 2);
        let tonal = specs.iter().filter(|p| tonal_share(p.get(lo..hi).unwrap_or(&[])) >= TONAL_SHARE).count();
        // the tonal share of the loud pause frames looked at, times the loud ones' share
        let looked = specs.len().max(1) as f64;
        (tonal as f64 / looked * pause_frames.len() as f64 / all_pauses as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // bandwidth: the long-term spectrum of the speech
    let (specs, n) = spectra(x, f, (f0..f1).filter(|k| is_active(*k)).step_by(3), 150);
    let bandwidth_hz = if specs.is_empty() {
        0.0
    } else {
        let bins = n / 2 + 1;
        let mut lt = vec![0f64; bins];
        for p in &specs {
            for (l, v) in lt.iter_mut().zip(p) {
                *l += v;
            }
        }
        let db: Vec<f64> = lt.iter().map(|v| 10.0 * (v / specs.len() as f64 + 1e-20).log10()).collect();
        let bin_hz = rate / n as f64;
        let (r0, r1) = (((300.0 / bin_hz) as usize).max(1), ((3000.0 / bin_hz) as usize).min(bins.saturating_sub(1)));
        let reference = db.get(r0..r1.max(r0 + 1)).map_or(-200.0, |s| s.iter().sum::<f64>() / s.len().max(1) as f64);
        // smoothed over ±3 bins, the highest bin within 45 dB of the reference
        let smooth = |b: usize| {
            let w = db.get(b.saturating_sub(3)..(b + 4).min(bins)).unwrap_or(&[]);
            w.iter().sum::<f64>() / w.len().max(1) as f64
        };
        (r0..bins).rev().find(|b| smooth(*b) >= reference - 45.0).map_or(0.0, |b| b as f64 * bin_hz)
    };
    let mut q = Quality { speech_db, noise_db, snr_db, clipped, reverb_s, clarity, music, bandwidth_hz, score: 0.0 };
    q.rate();
    q
}

/// Loudness-normalise speech to `lufs` (peaks kept under −1 dBFS).
pub fn normalise(x: &mut [f32], rate: u32, lufs: f64) {
    x.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = 0.0);
    let l = filmcraft_audio_dsp::loudness::measure(&[x], f64::from(rate)).integrated_lufs;
    let peak = x.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
    if !l.is_finite() || peak <= 0.0 {
        return;
    }
    let mut g = 10f64.powf(((lufs - l) / 20.0).min(2.0)) as f32;
    if peak * g > 0.89 {
        g = 0.89 / peak;
    }
    x.iter_mut().for_each(|v| *v *= g);
}

/// The long-term balance of speech: the share of its energy below 120 Hz and between 1 and
/// 4 kHz, and its tilt (the mean level of the 1, 2 and 4 kHz octaves minus that of the 125 and
/// 250 Hz ones, dB) — how the mix measures a voice for small speakers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Balance {
    pub sub: f64,
    pub presence: f64,
    pub tilt_db: f64,
}

/// [`Balance`] of `x` (mono at `rate`).
pub fn balance(x: &[f32], rate: u32) -> Balance {
    use filmcraft_audio_dsp::mixcheck::{BandMeter, OCTAVES, PRESENCE, SUB};
    let mut m = BandMeter::new(f64::from(rate), 1);
    m.process(&[x]);
    let b = m.bands();
    let oct = |hz: f64| OCTAVES.iter().position(|c| (*c - hz).abs() < 1.0).and_then(|i| b.octaves_db.get(i).copied()).unwrap_or(f64::NEG_INFINITY);
    let hi = (oct(1000.0) + oct(2000.0) + oct(4000.0)) / 3.0;
    let lo = (oct(125.0) + oct(250.0)) / 2.0;
    let tilt = hi - lo;
    Balance {
        sub: b.shares.get(SUB).copied().unwrap_or(0.0),
        presence: b.shares.get(PRESENCE).copied().unwrap_or(0.0),
        tilt_db: if tilt.is_finite() { tilt } else { 0.0 },
    }
}

/// A light clean-up of a reference: a 70 Hz high-pass (rumble and handling), optionally a gentle
/// noise reduction (`denoise_db`, 0 = none), the edges trimmed to 0.15 s of silence and faded.
pub fn clean(x: &[f32], rate: u32, denoise_db: f32) -> Vec<f32> {
    use filmcraft_audio_dsp::AudioEffect;
    let mut y: Vec<f32> = x.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
    if rate == 0 || y.is_empty() {
        return y;
    }
    // high-pass: a 2nd-order Butterworth twice (24 dB an octave below 70 Hz)
    use filmcraft_audio_dsp::biquad::{Biquad, Coeffs, FilterType};
    let hp = Coeffs::design(FilterType::HighPass, f64::from(rate), 70.0, std::f64::consts::FRAC_1_SQRT_2, 0.0);
    for _ in 0..2 {
        let mut st = Biquad::new();
        for v in y.iter_mut() {
            *v = st.tick(&hp, f64::from(*v)) as f32;
        }
    }
    if denoise_db > 0.0 {
        let mut d = filmcraft_audio_dsp::effects::restoration::DeNoise::new(rate as f32, 1);
        d.set_param("reduction", denoise_db);
        let lat = d.latency();
        let n = y.len();
        y.resize(n + lat, 0.0);
        for b in y.chunks_mut(4096) {
            d.process(&mut [b]);
        }
        y.drain(..lat.min(y.len()));
        y.truncate(n);
    }
    // trim to the speech, 0.15 s either side
    let f = frames(&y, rate);
    let a = activity(&f);
    let first = a.active.iter().position(|v| *v).unwrap_or(0);
    let last = a.active.iter().rposition(|v| *v).map_or(a.active.len(), |p| p + 1);
    let pad = (0.15 / FRAME_S) as usize;
    let (s, e) = (first.saturating_sub(pad) * f.len, ((last + pad) * f.len).min(y.len()));
    let mut out = y.get(s..e.max(s)).unwrap_or(&[]).to_vec();
    fade(&mut out, rate);
    out
}

/// 12 ms fades at both ends (no click at a cut).
pub fn fade(x: &mut [f32], rate: u32) {
    let n = ((f64::from(rate) * 0.012) as usize).min(x.len() / 4);
    let len = x.len();
    for i in 0..n {
        let g = i as f32 / n.max(1) as f32;
        if let Some(v) = x.get_mut(i) {
            *v *= g;
        }
        if let Some(v) = x.get_mut(len - 1 - i) {
            *v *= g;
        }
    }
}

/// Clips joined with `gap` seconds of silence between them (a reference of several sentences).
pub fn join(clips: &[&[f32]], rate: u32, gap: f64) -> Vec<f32> {
    let g = ((gap.max(0.0) * f64::from(rate)) as usize).min(rate as usize * 2);
    let mut out = Vec::new();
    for (i, c) in clips.iter().enumerate() {
        if i > 0 {
            out.resize(out.len() + g, 0.0);
        }
        let mut c = c.to_vec();
        fade(&mut c, rate);
        out.extend_from_slice(&c);
    }
    out
}
