//! Procedural audio for edits: sound effects (whoosh, impact, riser, swish, chime, drone) and music beds,
//! synthesized from oscillators, noise, envelopes and filters. No samples, no models: the same
//! parameters and seed always give the same audio, so a generated cue is reproducible from the
//! project alone.
//!
//! [`generate`] returns planar stereo at the requested rate. Every length and parameter is
//! clamped ([`MAX_SFX_SECONDS`], [`MAX_MUSIC_SECONDS`], tempo, intensity), so hostile input can
//! cost at most a bounded buffer. Music comes from the soundtrack engine ([`crate::music`]:
//! arranged to the exact length, ending on a bar), loudness-normalised to [`MUSIC_LUFS`] (a bed
//! that sits under narration) and peak-limited; effects are normalised to a peak instead.

use std::f64::consts::{PI, TAU};

use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::loudness::LoudnessMeter;

pub use crate::music::Mood;

/// Longest sound effect, seconds.
pub const MAX_SFX_SECONDS: f64 = 30.0;
/// Longest music bed, seconds (15 minutes: 346 MB of stereo f32 at 48 kHz).
pub const MAX_MUSIC_SECONDS: f64 = 900.0;
/// Integrated loudness of a music bed (EBU R128 LUFS): under a −16 LUFS voice.
pub const MUSIC_LUFS: f64 = -20.0;
/// Peak of a normalised sound effect (dBFS ≈ −1).
const SFX_PEAK: f32 = 0.89;

/// What to synthesize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Air rushing past: band-passed noise sweeping up and down, panned across.
    Whoosh,
    /// A cinematic hit: a sub boom with a noise crack and a long reverb tail.
    Impact,
    /// Tension building to a cut: rising noise and pitch, ending abruptly.
    Riser,
    /// A soft, short brush of air: high noise only, no sweep or rumble — a gentler transition
    /// sound than the whoosh.
    Swish,
    /// A soft bell: a few glassy partials on the key's tonic and fifth, ringing out — a gentle
    /// mark for a new chapter.
    Chime,
    /// A sustained, slowly moving low chord.
    Drone,
    /// A cinematic bed: chord progression, pads, bass, drums, pulse and arpeggio in sections.
    Music,
}

impl Kind {
    pub const ALL: [Kind; 7] = [Kind::Whoosh, Kind::Impact, Kind::Riser, Kind::Swish, Kind::Chime, Kind::Drone, Kind::Music];

    pub fn id(self) -> &'static str {
        match self {
            Kind::Whoosh => "whoosh",
            Kind::Impact => "impact",
            Kind::Riser => "riser",
            Kind::Swish => "swish",
            Kind::Chime => "chime",
            Kind::Drone => "drone",
            Kind::Music => "music",
        }
    }

    pub fn from_id(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s.trim()))
    }

    /// Default length in seconds.
    pub fn default_seconds(self) -> f64 {
        match self {
            Kind::Whoosh => 1.2,
            Kind::Impact => 3.0,
            Kind::Riser => 3.0,
            Kind::Swish => 0.5,
            Kind::Chime => 1.8,
            Kind::Drone => 20.0,
            Kind::Music => 60.0,
        }
    }

    /// Longest cue of this kind, seconds.
    pub fn max_seconds(self) -> f64 {
        if matches!(self, Kind::Music | Kind::Drone) { MAX_MUSIC_SECONDS } else { MAX_SFX_SECONDS }
    }
}

/// Generation parameters. Out-of-range values are clamped by [`generate`].
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub kind: Kind,
    pub seconds: f64,
    pub seed: u64,
    /// 0..1: brightness, density and level of the result.
    pub intensity: f64,
    /// Music tempo (40..200 BPM).
    pub bpm: f64,
    /// Music tonic as semitones above A (0 = A, 3 = C, …).
    pub key: i32,
    pub mood: Mood,
}

impl Params {
    pub fn new(kind: Kind) -> Self {
        Self { kind, seconds: kind.default_seconds(), seed: 1, intensity: 0.7, bpm: 96.0, key: 0, mood: Mood::Epic }
    }
}

/// Planar stereo audio.
#[derive(Clone, Debug, PartialEq)]
pub struct Stereo {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

impl Stereo {
    pub fn seconds(&self) -> f64 {
        self.left.len() as f64 / f64::from(self.sample_rate.max(1))
    }
}

/// Why generation was refused.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthError(pub String);

impl std::fmt::Display for SynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SynthError {}

/// A key name ("A", "C#", "Eb", "F# minor") → semitones above A.
pub fn key_from_name(s: &str) -> Option<i32> {
    let t = s.trim();
    let mut chars = t.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let base: i32 = match letter {
        'A' => 0,
        'B' => 2,
        'C' => 3,
        'D' => 5,
        'E' => 7,
        'F' => 8,
        'G' => 10,
        _ => return None,
    };
    let acc = match chars.next() {
        Some('#') | Some('♯') => 1,
        Some('b') | Some('♭') => -1,
        _ => 0,
    };
    Some((base + acc).rem_euclid(12))
}

/// Synthesize `p` at `sample_rate` (8 kHz .. 192 kHz).
pub fn generate(p: &Params, sample_rate: u32) -> Result<Stereo, SynthError> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(SynthError(format!("sample rate {sample_rate} Hz is outside 8–192 kHz")));
    }
    let seconds = if p.seconds.is_finite() { p.seconds } else { p.kind.default_seconds() };
    if seconds <= 0.0 {
        return Err(SynthError("the length must be more than zero seconds".into()));
    }
    let seconds = seconds.clamp(0.05, p.kind.max_seconds());
    if p.kind == Kind::Music {
        let mut req = crate::music::Request::new(p.mood, filmcraft_time::Tick::from_seconds_f64(seconds));
        req.seed = p.seed;
        req.bpm = Some(p.bpm);
        req.key = Some(p.key);
        req.energy = Some(crate::music::Energy::Level(if p.intensity.is_finite() { p.intensity.clamp(0.0, 1.0) } else { 0.7 }));
        return crate::music::generate(&req, sample_rate).map(|t| t.mix);
    }
    let sr = f64::from(sample_rate);
    let n = (seconds * sr).round() as usize;
    let mut left = Vec::new();
    let mut right = Vec::new();
    left.try_reserve_exact(n).map_err(|e| SynthError(format!("cannot allocate {seconds:.1}s of audio: {e}")))?;
    right.try_reserve_exact(n).map_err(|e| SynthError(format!("cannot allocate {seconds:.1}s of audio: {e}")))?;
    left.resize(n, 0.0);
    right.resize(n, 0.0);
    let q = Clamped {
        sr,
        seed: p.seed,
        intensity: if p.intensity.is_finite() { p.intensity.clamp(0.0, 1.0) } else { 0.7 },
        key: p.key.rem_euclid(12),
        mood: p.mood,
    };
    match p.kind {
        Kind::Whoosh => whoosh(&q, &mut left, &mut right),
        Kind::Impact => impact(&q, &mut left, &mut right),
        Kind::Riser => riser(&q, &mut left, &mut right),
        Kind::Swish => swish(&q, &mut left, &mut right),
        Kind::Chime => chime(&q, &mut left, &mut right),
        Kind::Drone | Kind::Music => drone(&q, &mut left, &mut right),
    }
    match p.kind {
        Kind::Music | Kind::Drone => normalize_loudness(&mut left, &mut right, sr, MUSIC_LUFS),
        _ => normalize_peak(&mut left, &mut right, SFX_PEAK * (0.6 + 0.4 * q.intensity as f32)),
    }
    // nothing non-finite ever leaves this module
    for x in left.iter_mut().chain(right.iter_mut()) {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    Ok(Stereo { left, right, sample_rate })
}

struct Clamped {
    sr: f64,
    seed: u64,
    intensity: f64,
    key: i32,
    mood: Mood,
}

// ---------------------------------------------------------------------------------------------
// building blocks

/// xorshift64* — small, fast, deterministic.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn white(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Pink-ish noise (Paul Kellet's economy filter over white noise).
struct Pink {
    rng: Rng,
    b: [f64; 3],
}

impl Pink {
    fn new(seed: u64) -> Self {
        Pink { rng: Rng::new(seed), b: [0.0; 3] }
    }
    fn next(&mut self) -> f64 {
        let w = self.rng.white();
        self.b[0] = 0.99765 * self.b[0] + w * 0.099_046;
        self.b[1] = 0.963 * self.b[1] + w * 0.296_516_4;
        self.b[2] = 0.57 * self.b[2] + w * 1.052_691_3;
        (self.b[0] + self.b[1] + self.b[2] + w * 0.1848) * 0.25
    }
}

/// A biquad whose design is refreshed every [`SWEEP_EVERY`] samples (cheap filter sweeps).
struct Sweep {
    kind: FilterType,
    q: f64,
    c: Coeffs,
    f: Biquad,
    k: usize,
}

const SWEEP_EVERY: usize = 16;

impl Sweep {
    fn new(kind: FilterType, q: f64) -> Self {
        Sweep { kind, q, c: Coeffs::IDENTITY, f: Biquad::default(), k: 0 }
    }
    fn tick(&mut self, x: f64, freq: f64, sr: f64) -> f64 {
        if self.k == 0 {
            self.c = Coeffs::design(self.kind, sr, freq, self.q, 0.0);
        }
        self.k = (self.k + 1) % SWEEP_EVERY;
        self.f.tick(&self.c, x)
    }
}

/// Band-limited sawtooth (PolyBLEP).
fn saw(phase: f64, dt: f64) -> f64 {
    let mut v = 2.0 * phase - 1.0;
    v -= poly_blep(phase, dt);
    v
}

fn poly_blep(t: f64, dt: f64) -> f64 {
    if dt <= 0.0 {
        0.0
    } else if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

fn wrap(phase: f64) -> f64 {
    phase - phase.floor()
}

/// Equal-power pan gains for `pan` in -1 (left) .. 1 (right).
fn pan_gains(pan: f64) -> (f64, f64) {
    let a = (pan.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    (a.cos(), a.sin())
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// MIDI-ish note number → Hz (A4 = 69 = 440 Hz).
fn hz(note: f64) -> f64 {
    440.0 * 2f64.powf((note - 69.0) / 12.0)
}

/// Add a reverb tail over a planar stereo pair with the engine's FDN reverb.
fn reverb(left: &mut [f32], right: &mut [f32], sr: f64, decay: f32, size: f32, mix: f32) {
    let Some(mut fx) = crate::effects::create_effect("reverb", sr as f32, 2) else { return };
    fx.set_param("decay", decay);
    fx.set_param("size", size);
    fx.set_param("damping", 55.0);
    fx.set_param("predelay", 15.0);
    fx.set_param("mix", mix);
    fx.reset();
    const BLOCK: usize = 4096;
    let mut l = left.chunks_mut(BLOCK);
    let mut r = right.chunks_mut(BLOCK);
    while let (Some(a), Some(b)) = (l.next(), r.next()) {
        fx.process(&mut [a, b]);
    }
}

fn normalize_peak(left: &mut [f32], right: &mut [f32], target: f32) {
    let peak = left.iter().chain(right.iter()).fold(0f32, |m, x| if x.is_finite() { m.max(x.abs()) } else { m });
    if peak > 1e-9 {
        let g = target / peak;
        for x in left.iter_mut().chain(right.iter_mut()) {
            *x *= g;
        }
    }
}

fn normalize_loudness(left: &mut [f32], right: &mut [f32], sr: f64, lufs: f64) {
    let mut m = LoudnessMeter::new(sr, 2);
    m.process(&[left, right]);
    let measured = m.integrated();
    let gain_db = crate::loudness::normalize_gain_db(measured, lufs).clamp(-40.0, 40.0);
    let g = 10f64.powf(gain_db / 20.0) as f32;
    for x in left.iter_mut().chain(right.iter_mut()) {
        // gain, then a soft limiter that leaves anything under −3 dBFS untouched
        let y = *x * g;
        *x = soft_limit(y);
    }
}

/// Transparent below 0.7, smoothly saturating to ±0.98 above.
pub(crate) fn soft_limit(x: f32) -> f32 {
    const KNEE: f32 = 0.7;
    const CEIL: f32 = 0.98;
    let a = x.abs();
    if a <= KNEE {
        x
    } else {
        let over = (a - KNEE) / (CEIL - KNEE);
        x.signum() * (KNEE + (CEIL - KNEE) * over.tanh())
    }
}

// ---------------------------------------------------------------------------------------------
// sound effects

fn whoosh(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let n = left.len().max(1);
    let mut noise = Pink::new(q.seed);
    let mut bp = Sweep::new(FilterType::BandPass, 0.9);
    let mut air = Sweep::new(FilterType::LowPass, 0.7);
    let top = 2500.0 + 3500.0 * q.intensity;
    let peak_at = 0.55;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let x = i as f64 / n as f64;
        // a bell: smooth rise to `peak_at`, quicker fall
        let env = if x < peak_at { smoothstep(x / peak_at).powf(1.5) } else { (1.0 - smoothstep((x - peak_at) / (1.0 - peak_at))).powf(2.0) };
        let freq = 250.0 * (top / 250.0).powf(env);
        let s = noise.next();
        let body = bp.tick(s, freq, q.sr);
        let low = air.tick(s, 180.0 + 400.0 * env, q.sr) * 0.6;
        let v = (body * 1.6 + low) * env;
        let (gl, gr) = pan_gains(-0.8 + 1.6 * x);
        *l = (v * gl) as f32;
        *r = (v * gr) as f32;
    }
    reverb(left, right, q.sr, 1.2, 40.0, 18.0);
}

fn impact(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let mut rng = Rng::new(q.seed);
    let mut crack = Sweep::new(FilterType::LowPass, 0.8);
    let mut hiss = Sweep::new(FilterType::HighPass, 0.7);
    let mut phase = 0.0f64;
    let drive = 1.5 + 2.0 * q.intensity;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / q.sr;
        // sub boom: pitch drops from ~110 Hz to ~38 Hz
        let f = 38.0 + 72.0 * (-t / 0.06).exp();
        phase = wrap(phase + f / q.sr);
        let sub = (phase * TAU).sin() * (-t / (0.5 + 0.6 * q.intensity)).exp();
        let sub = (sub * drive).tanh() / drive.tanh();
        let w = rng.white();
        let body = crack.tick(w, 900.0 + 1500.0 * q.intensity, q.sr) * (-t / 0.09).exp() * 1.4;
        let click = hiss.tick(w, 4000.0, q.sr) * (-t / 0.006).exp() * 0.8;
        let mono = sub * 0.9;
        *l = (mono + body + click) as f32;
        *r = (mono + body * 0.9 - click * 0.2) as f32;
    }
    reverb(left, right, q.sr, 3.5, 90.0, 30.0);
}

fn riser(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let n = left.len().max(1);
    let mut noise = Pink::new(q.seed);
    let mut bp = Sweep::new(FilterType::BandPass, 1.4);
    let mut phases = [0.0f64; 3];
    let mut trem = 0.0f64;
    let end_fade = (q.sr * 0.006) as usize;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let x = i as f64 / n as f64;
        let env = x * x;
        let freq = 200.0 * (30.0f64).powf(x);
        let air = bp.tick(noise.next(), freq, q.sr) * 1.8;
        // a rising detuned saw stack, tremolo speeding up
        let pitch = 110.0 * (8.0f64).powf(x);
        let mut tone = 0.0;
        for (k, ph) in phases.iter_mut().enumerate() {
            let det = 1.0 + (k as f64 - 1.0) * 0.006;
            let dt = pitch * det / q.sr;
            *ph = wrap(*ph + dt);
            tone += saw(*ph, dt);
        }
        trem = wrap(trem + (4.0 + 14.0 * x) / q.sr);
        let tremolo = 0.75 + 0.25 * (trem * TAU).sin();
        let mut v = (air + tone * 0.18 * (0.5 + q.intensity)) * env * tremolo;
        let left_to_end = n.saturating_sub(i);
        if left_to_end < end_fade && end_fade > 0 {
            v *= left_to_end as f64 / end_fade as f64;
        }
        let (gl, gr) = pan_gains((x * 9.0).sin() * 0.3);
        *l = (v * gl) as f32;
        *r = (v * gr) as f32;
    }
    reverb(left, right, q.sr, 1.5, 60.0, 15.0);
}

fn swish(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let n = left.len().max(1);
    let mut rng = Rng::new(q.seed);
    let mut bp = Sweep::new(FilterType::BandPass, 0.7);
    let mut hp = Sweep::new(FilterType::HighPass, 0.7);
    // where the band sits: the seed nudges it, the intensity brightens it
    let centre = 3200.0 + 1200.0 * rng.unit() + 1600.0 * q.intensity;
    let peak_at = 0.3;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let x = i as f64 / n as f64;
        // quick in, longer out: a brush, not a swell
        let env = if x < peak_at { smoothstep(x / peak_at) } else { (1.0 - smoothstep((x - peak_at) / (1.0 - peak_at))).powf(2.5) };
        // the band drifts a little (up, then back), never the whoosh's octaves-long sweep
        let freq = centre * (1.0 + 0.25 * (x * PI).sin());
        let s = rng.white();
        let v = hp.tick(bp.tick(s, freq, q.sr), 1500.0, q.sr) * env;
        let (gl, gr) = pan_gains(-0.4 + 0.8 * x);
        *l = (v * gl) as f32;
        *r = (v * gr) as f32;
    }
    reverb(left, right, q.sr, 0.6, 25.0, 10.0);
}

fn chime(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    // the key's tonic in the fifth octave (A5 = 880 Hz for key 0) and its fifth, quieter
    let root = 81.0 + f64::from(q.key);
    let mut rng = Rng::new(q.seed);
    // glassy partials: (ratio, level, decay seconds)
    const PARTIALS: [(f64, f64, f64); 4] = [(1.0, 1.0, 0.55), (2.0, 0.4, 0.35), (3.01, 0.18, 0.2), (4.2, 0.07, 0.12)];
    let ring = 0.8 + 0.5 * q.intensity;
    let notes = [(root, 1.0, -0.2), (root + 7.0, 0.55, 0.2)];
    // the seed detunes each note by a few cents and offsets its phases
    let tunes: Vec<(f64, f64, f64, f64)> = notes.iter().map(|&(note, level, pan)| (hz(note) * (1.0 + 0.002 * rng.white()), level, pan, rng.unit())).collect();
    let mut mallet = Sweep::new(FilterType::BandPass, 1.2);
    let attack = q.sr * 0.003;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / q.sr;
        let soft = (i as f64 / attack).min(1.0);
        let (mut sl, mut sr) = (0.0, 0.0);
        for &(f, level, pan, phase) in &tunes {
            let mut v = 0.0;
            for (ratio, amp, decay) in PARTIALS {
                v += amp * ((f * ratio * t + phase) * TAU).sin() * (-t / (decay * ring)).exp();
            }
            let (gl, gr) = pan_gains(pan);
            sl += v * level * gl;
            sr += v * level * gr;
        }
        // a soft mallet: a few milliseconds of filtered noise
        let tick = mallet.tick(rng.white(), 4000.0, q.sr) * (-t / 0.004).exp() * 0.3;
        *l = ((sl + tick) * soft) as f32;
        *r = ((sr + tick) * soft) as f32;
    }
    reverb(left, right, q.sr, 2.2, 60.0, 25.0);
    // however short it is cut, it (and its reverb) fades out over its last tenth, never stops
    let n = left.len();
    let fade_from = n - n / 10;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate().skip(fade_from) {
        let g = 1.0 - smoothstep((i - fade_from) as f64 / (n - fade_from).max(1) as f64);
        *l *= g as f32;
        *r *= g as f32;
    }
}

fn drone(q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let root = 33.0 + f64::from(q.key); // A1 = 33
    let notes = [root, root + 7.0, root + 12.0, root + if q.mood.minor() { 15.0 } else { 16.0 }];
    let mut ph = [[0.0f64; 2]; 4];
    let mut lp = [Sweep::new(FilterType::LowPass, 0.9), Sweep::new(FilterType::LowPass, 0.9)];
    let mut sub = 0.0f64;
    let fade = (3.0 * q.sr).min(left.len() as f64 / 3.0).max(1.0);
    let n = left.len();
    // the seed picks where the filter's slow sweep starts and how wide the detune is
    let mut rng = Rng::new(q.seed);
    let lfo_start = rng.unit();
    let spread = 0.002 + 0.003 * rng.unit();
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / q.sr;
        let lfo = ((t * 0.07 + lfo_start) * TAU).sin();
        let mut s = [0.0f64; 2];
        for (k, note) in notes.iter().enumerate() {
            for (side, p) in ph.get_mut(k).into_iter().flat_map(|a| a.iter_mut()).enumerate() {
                let det = if side == 0 { 1.0 - spread } else { 1.0 + spread };
                let dt = hz(*note) * det / q.sr;
                *p = wrap(*p + dt);
                if let Some(o) = s.get_mut(side) {
                    *o += saw(*p, dt) * 0.25;
                }
            }
        }
        let cut = 350.0 + 250.0 * (lfo + 1.0) * (0.5 + q.intensity);
        let a = lp[0].tick(s[0], cut, q.sr);
        let b = lp[1].tick(s[1], cut * 1.03, q.sr);
        sub = wrap(sub + hz(root) / q.sr);
        let sb = (sub * TAU).sin() * 0.5;
        let env = smoothstep(i as f64 / fade) * smoothstep(n.saturating_sub(i) as f64 / fade);
        *l = ((a + sb) * env) as f32;
        *r = ((b + sb) * env) as f32;
    }
    reverb(left, right, q.sr, 5.0, 95.0, 35.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64).sqrt()
    }

    fn sane(s: &Stereo, seconds: f64) {
        let n = (seconds * f64::from(SR)).round() as usize;
        assert_eq!(s.left.len(), n);
        assert_eq!(s.right.len(), n);
        assert!(s.left.iter().chain(&s.right).all(|x| x.is_finite() && x.abs() <= 1.0), "finite and in range");
        assert!(rms(&s.left) > 1e-3 && rms(&s.right) > 1e-3, "not silent");
    }

    #[test]
    fn every_kind_is_sane_and_deterministic() {
        for kind in Kind::ALL {
            let mut p = Params::new(kind);
            p.seconds = if kind == Kind::Music { 12.0 } else { kind.default_seconds().min(4.0) };
            let a = generate(&p, SR).unwrap();
            sane(&a, p.seconds);
            assert_eq!(a, generate(&p, SR).unwrap(), "{kind:?} is deterministic");
            p.seed = 99;
            assert_ne!(a.left, generate(&p, SR).unwrap().left, "{kind:?} depends on the seed");
        }
    }

    #[test]
    fn whoosh_swells_in_the_middle() {
        let s = generate(&Params::new(Kind::Whoosh), SR).unwrap();
        let third = s.left.len() / 3;
        let mid = rms(&s.left[third..2 * third]) + rms(&s.right[third..2 * third]);
        let start = rms(&s.left[..third / 3]) + rms(&s.right[..third / 3]);
        assert!(mid > 3.0 * start, "mid {mid} start {start}");
        // it travels left → right
        let early_lr = rms(&s.left[..third]) / rms(&s.right[..third]).max(1e-9);
        let late_lr = rms(&s.left[2 * third..]) / rms(&s.right[2 * third..]).max(1e-9);
        assert!(early_lr > late_lr, "pans across: {early_lr} → {late_lr}");
    }

    #[test]
    fn impact_hits_then_decays_and_is_low() {
        let s = generate(&Params::new(Kind::Impact), SR).unwrap();
        let head = rms(&s.left[..SR as usize / 4]);
        let tail = rms(&s.left[s.left.len() - SR as usize / 2..]);
        assert!(head > 5.0 * tail, "head {head} tail {tail}");
        // mostly low frequencies: few zero crossings in the first quarter second
        let crossings = s.left[..SR as usize / 4].windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        assert!(crossings < 600, "{crossings} crossings");
    }

    #[test]
    fn riser_rises_and_ends_on_a_cut() {
        let s = generate(&Params::new(Kind::Riser), SR).unwrap();
        let q = s.left.len() / 4;
        assert!(rms(&s.left[3 * q..]) > 4.0 * rms(&s.left[..q]));
        assert!(s.left.last().is_some_and(|x| x.abs() < 0.05), "ends quietly, no click");
    }

    #[test]
    fn swish_is_short_bright_and_has_no_rumble() {
        let p = Params::new(Kind::Swish);
        let s = generate(&p, SR).unwrap();
        sane(&s, 0.5);
        // over in its first half: a brush, not a swell
        let half = s.left.len() / 2;
        assert!(rms(&s.left[..half]) > 2.0 * rms(&s.left[half..]), "front-loaded");
        // high noise only: many zero crossings (the whoosh's low body and rumble are not there)
        let crossings = |x: &[f32]| x.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f64 / x.len() as f64;
        let w = generate(&Params::new(Kind::Whoosh), SR).unwrap();
        assert!(crossings(&s.left) > 2.0 * crossings(&w.left), "swish {} whoosh {}", crossings(&s.left), crossings(&w.left));
    }

    #[test]
    fn chime_rings_on_the_key_and_dies_away() {
        let mut p = Params::new(Kind::Chime);
        let s = generate(&p, SR).unwrap();
        sane(&s, 1.8);
        let q = s.left.len() / 6;
        assert!(rms(&s.left[..q]) > 4.0 * rms(&s.left[5 * q..]), "it decays");
        assert!(s.left.first().is_some_and(|x| x.abs() < 0.05), "starts without a click");
        assert!(s.left.last().is_some_and(|x| x.abs() < 1e-3), "and ends without one");
        // tonal: the strongest of a few candidate pitches is the tonic (A5, 880 Hz, for key 0)…
        let power = |x: &[f32], f: f64| {
            let (mut c, mut s) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let a = TAU * f * i as f64 / f64::from(SR);
                c += f64::from(*v) * a.cos();
                s += f64::from(*v) * a.sin();
            }
            c * c + s * s
        };
        let head = &s.left[..SR as usize / 4];
        let best = [660.0, 784.0, 880.0, 988.0, 1046.5].into_iter().max_by(|a, b| power(head, *a).total_cmp(&power(head, *b)));
        assert_eq!(best, Some(880.0));
        // …and moves with the key (C: three semitones up)
        p.key = 3;
        let c = generate(&p, SR).unwrap();
        let head = &c.left[..SR as usize / 4];
        assert!(power(head, 1046.5) > 10.0 * power(head, 880.0));
    }

    #[test]
    fn music_sits_at_the_bed_loudness_with_quiet_intro() {
        let mut p = Params::new(Kind::Music);
        p.seconds = 40.0;
        let s = generate(&p, SR).unwrap();
        sane(&s, p.seconds);
        let mut m = LoudnessMeter::new(f64::from(SR), 2);
        m.process(&[&s.left, &s.right]);
        assert!((m.integrated() - MUSIC_LUFS).abs() < 1.5, "integrated {}", m.integrated());
        // the first second fades in; the middle is busier
        let first = rms(&s.left[..SR as usize]);
        let mid = rms(&s.left[20 * SR as usize..22 * SR as usize]);
        assert!(mid > first * 1.5, "mid {mid} first {first}");
        // every mood renders
        for mood in Mood::ALL {
            let mut q = p;
            q.mood = mood;
            q.seconds = 8.0;
            sane(&generate(&q, SR).unwrap(), 8.0);
        }
    }

    #[test]
    fn hostile_parameters_are_clamped_or_refused() {
        let mut p = Params::new(Kind::Whoosh);
        p.seconds = f64::NAN;
        assert!(generate(&p, SR).is_ok(), "NaN length falls back to the default");
        p.seconds = -1.0;
        assert!(generate(&p, SR).is_err());
        p.seconds = 1e12;
        assert_eq!(generate(&p, SR).unwrap().seconds().round(), MAX_SFX_SECONDS);
        p.seconds = 1.0;
        assert!(generate(&p, 0).is_err());
        assert!(generate(&p, 1_000_000).is_err());
        let mut m = Params::new(Kind::Music);
        m.seconds = 3.0;
        m.bpm = f64::INFINITY;
        m.intensity = f64::NAN;
        m.key = i32::MIN;
        sane(&generate(&m, SR).unwrap(), 3.0);
    }

    #[test]
    fn names_parse() {
        assert_eq!(Kind::from_id(" Whoosh "), Some(Kind::Whoosh));
        assert_eq!(Kind::from_id("boom"), None);
        assert_eq!(Mood::from_id("EPIC"), Some(Mood::Epic));
        assert_eq!(key_from_name("A"), Some(0));
        assert_eq!(key_from_name("C"), Some(3));
        assert_eq!(key_from_name("F# minor"), Some(9));
        assert_eq!(key_from_name("Bb"), Some(1));
        assert_eq!(key_from_name("H"), None);
        assert_eq!(key_from_name(""), None);
    }
}
