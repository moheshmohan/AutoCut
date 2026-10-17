//! Short procedural sound effects for the Elements library (`filmcraft_engine::elements`): a UI
//! click, a pop, swooshes up and down, a resonant filter sweep and a deep boom. Like
//! [`crate::synth`] they are made from oscillators, noise, envelopes and filters — no samples, no
//! models — and the same kind, length and seed always give the same audio.
//!
//! [`generate`] returns planar stereo normalised to a peak of about −1 dBFS. The length is
//! clamped to [`MAX_SECONDS`] and the rate must be 8–192 kHz, so hostile input costs at most a
//! bounded buffer.

use std::f64::consts::{PI, TAU};

use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::synth::{Stereo, SynthError};

/// Longest effect, seconds.
pub const MAX_SECONDS: f64 = 10.0;
/// Shortest effect, seconds.
const MIN_SECONDS: f64 = 0.03;
/// Peak of a normalised effect (≈ −1 dBFS).
const PEAK: f32 = 0.89;

/// What to synthesize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sfx {
    /// A dry interface click: a short high tick over a small body.
    Click,
    /// A bubble pop: a sine whose pitch flicks upward as it dies away.
    Pop,
    /// An airy pass whose band rises from low to high, travelling left to right.
    SwooshUp,
    /// The same pass falling from high to low, travelling right to left.
    SwooshDown,
    /// A resonant low-pass sweeping a detuned saw chord and noise up the spectrum and back.
    Sweep,
    /// A deep sub drop with a rumble and a dark tail (no crack, unlike the synth's impact).
    Boom,
}

impl Sfx {
    pub const ALL: [Sfx; 6] = [Sfx::Click, Sfx::Pop, Sfx::SwooshUp, Sfx::SwooshDown, Sfx::Sweep, Sfx::Boom];

    pub fn id(self) -> &'static str {
        match self {
            Sfx::Click => "click",
            Sfx::Pop => "pop",
            Sfx::SwooshUp => "swoosh-up",
            Sfx::SwooshDown => "swoosh-down",
            Sfx::Sweep => "sweep",
            Sfx::Boom => "boom",
        }
    }

    pub fn from_id(s: &str) -> Option<Sfx> {
        Sfx::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s.trim()))
    }

    /// Default length in seconds.
    pub fn default_seconds(self) -> f64 {
        match self {
            Sfx::Click => 0.12,
            Sfx::Pop => 0.3,
            Sfx::SwooshUp | Sfx::SwooshDown => 0.7,
            Sfx::Sweep => 1.6,
            Sfx::Boom => 3.0,
        }
    }
}

/// Synthesize `kind`, `seconds` long (NaN = the default length), at `sample_rate` (8–192 kHz).
pub fn generate(kind: Sfx, seconds: f64, seed: u64, sample_rate: u32) -> Result<Stereo, SynthError> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(SynthError(format!("sample rate {sample_rate} Hz is outside 8–192 kHz")));
    }
    let seconds = if seconds.is_finite() { seconds } else { kind.default_seconds() };
    if seconds <= 0.0 {
        return Err(SynthError("the length must be more than zero seconds".into()));
    }
    let seconds = seconds.clamp(MIN_SECONDS, MAX_SECONDS);
    let sr = f64::from(sample_rate);
    let n = (seconds * sr).round() as usize;
    let mut left = Vec::new();
    let mut right = Vec::new();
    left.try_reserve_exact(n).map_err(|e| SynthError(format!("cannot allocate {seconds:.2}s of audio: {e}")))?;
    right.try_reserve_exact(n).map_err(|e| SynthError(format!("cannot allocate {seconds:.2}s of audio: {e}")))?;
    left.resize(n, 0.0);
    right.resize(n, 0.0);
    match kind {
        Sfx::Click => click(sr, seed, &mut left, &mut right),
        Sfx::Pop => pop(sr, seed, &mut left, &mut right),
        Sfx::SwooshUp => swoosh(sr, seed, true, &mut left, &mut right),
        Sfx::SwooshDown => swoosh(sr, seed, false, &mut left, &mut right),
        Sfx::Sweep => sweep(sr, seed, &mut left, &mut right),
        Sfx::Boom => boom(sr, seed, &mut left, &mut right),
    }
    fade_out(&mut left, &mut right, (sr * 0.005) as usize);
    // nothing non-finite ever leaves this module
    for x in left.iter_mut().chain(right.iter_mut()) {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    normalize_peak(&mut left, &mut right, PEAK);
    Ok(Stereo { left, right, sample_rate })
}

// ---------------------------------------------------------------------------------------------
// building blocks (kept local so this module does not reach into the synth's internals)

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

/// Pink-ish noise (a three-pole filter over white noise).
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

/// A biquad redesigned every 16 samples (cheap filter sweeps).
struct Moving {
    kind: FilterType,
    q: f64,
    c: Coeffs,
    f: Biquad,
    k: usize,
}

impl Moving {
    fn new(kind: FilterType, q: f64) -> Self {
        Moving { kind, q, c: Coeffs::IDENTITY, f: Biquad::new(), k: 0 }
    }
    fn tick(&mut self, x: f64, freq: f64, sr: f64) -> f64 {
        if self.k == 0 {
            self.c = Coeffs::design(self.kind, sr, freq, self.q, 0.0);
        }
        self.k = (self.k + 1) % 16;
        self.f.tick(&self.c, x)
    }
}

/// A fixed biquad.
struct Fixed {
    c: Coeffs,
    f: Biquad,
}

impl Fixed {
    fn new(kind: FilterType, sr: f64, freq: f64, q: f64) -> Self {
        Fixed { c: Coeffs::design(kind, sr, freq, q, 0.0), f: Biquad::new() }
    }
    fn tick(&mut self, x: f64) -> f64 {
        self.f.tick(&self.c, x)
    }
}

/// Band-limited sawtooth (PolyBLEP) at `phase` (0..1) advancing `dt` per sample.
fn saw(phase: f64, dt: f64) -> f64 {
    let blep = if dt <= 0.0 {
        0.0
    } else if phase < dt {
        let x = phase / dt;
        x + x - x * x - 1.0
    } else if phase > 1.0 - dt {
        let x = (phase - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    };
    2.0 * phase - 1.0 - blep
}

fn wrap(phase: f64) -> f64 {
    phase - phase.floor()
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Equal-power pan gains for `pan` in -1 (left) .. 1 (right).
fn pan_gains(pan: f64) -> (f64, f64) {
    let a = (pan.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    (a.cos(), a.sin())
}

/// Add a reverb tail with the engine's FDN reverb.
fn reverb(left: &mut [f32], right: &mut [f32], sr: f64, decay: f32, size: f32, damping: f32, mix: f32) {
    let Some(mut fx) = crate::effects::create_effect("reverb", sr as f32, 2) else { return };
    fx.set_param("decay", decay);
    fx.set_param("size", size);
    fx.set_param("damping", damping);
    fx.set_param("predelay", 10.0);
    fx.set_param("mix", mix);
    fx.reset();
    const BLOCK: usize = 4096;
    let mut l = left.chunks_mut(BLOCK);
    let mut r = right.chunks_mut(BLOCK);
    while let (Some(a), Some(b)) = (l.next(), r.next()) {
        fx.process(&mut [a, b]);
    }
}

/// Linear fade over the last `len` samples, so a cut-off tail never clicks.
fn fade_out(left: &mut [f32], right: &mut [f32], len: usize) {
    let n = left.len().min(right.len());
    let len = len.min(n);
    if len == 0 {
        return;
    }
    for k in 0..len {
        let i = n - len + k;
        let g = (len - k - 1) as f32 / len as f32;
        if let (Some(l), Some(r)) = (left.get_mut(i), right.get_mut(i)) {
            *l *= g;
            *r *= g;
        }
    }
}

fn normalize_peak(left: &mut [f32], right: &mut [f32], target: f32) {
    let peak = left.iter().chain(right.iter()).fold(0f32, |m, x| m.max(x.abs()));
    if peak > 1e-9 {
        let g = target / peak;
        for x in left.iter_mut().chain(right.iter_mut()) {
            *x *= g;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the effects

fn click(sr: f64, seed: u64, left: &mut [f32], right: &mut [f32]) {
    let mut rng = Rng::new(seed);
    // the seed nudges the tick's pitch, so two clicks are not identical
    let tick_hz = 2000.0 + 500.0 * rng.unit();
    let body_hz = 560.0 + 120.0 * rng.unit();
    let mut hp = Fixed::new(FilterType::HighPass, sr, 3000.0, 0.7);
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / sr;
        let tick = (TAU * tick_hz * t).sin() * (-t / 0.004).exp();
        let body = (TAU * body_hz * t).sin() * (-t / 0.012).exp() * 0.55;
        let snap = hp.tick(rng.white()) * (-t / 0.0015).exp() * 0.8;
        let v = tick + body + snap;
        *l = v as f32;
        *r = (tick + body - snap * 0.3) as f32;
    }
}

fn pop(sr: f64, seed: u64, left: &mut [f32], right: &mut [f32]) {
    let mut rng = Rng::new(seed);
    let base = 240.0 + 60.0 * rng.unit();
    let lift = 850.0 + 200.0 * rng.unit();
    let mut lp = Fixed::new(FilterType::LowPass, sr, 2500.0, 0.7);
    let mut phase = 0.0f64;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / sr;
        // the pitch flicks up over ~20 ms
        let f = base + lift * (1.0 - (-t / 0.018).exp());
        phase = wrap(phase + f / sr);
        let attack = (t / 0.0008).min(1.0);
        let tone = (phase * TAU).sin() * attack * (-t / 0.045).exp();
        let puff = lp.tick(rng.white()) * (-t / 0.002).exp() * 0.35;
        let v = tone + puff;
        *l = v as f32;
        *r = (tone * 0.97 + puff * 0.8) as f32;
    }
    reverb(left, right, sr, 0.6, 25.0, 50.0, 10.0);
}

fn swoosh(sr: f64, seed: u64, up: bool, left: &mut [f32], right: &mut [f32]) {
    let n = left.len().max(1);
    let mut noise = Pink::new(seed);
    let mut bp = Moving::new(FilterType::BandPass, 1.2);
    let mut air = Moving::new(FilterType::LowPass, 0.7);
    let mut whistle = 0.0f64;
    let (lo, hi) = (350.0f64, 7500.0f64);
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let x = i as f64 / n as f64;
        // where the band is: low → high for up, high → low for down
        let s = if up { x } else { 1.0 - x };
        let freq = lo * (hi / lo).powf(s.powf(1.2));
        let env = if up {
            if x < 0.75 { smoothstep(x / 0.75).powf(1.3) } else { 1.0 - smoothstep((x - 0.75) / 0.25) }
        } else if x < 0.15 {
            smoothstep(x / 0.15)
        } else {
            (1.0 - smoothstep((x - 0.15) / 0.85)).powf(1.5)
        };
        let w = noise.next();
        let body = bp.tick(w, freq, sr) * 1.7 + air.tick(w, 200.0 + freq * 0.15, sr) * 0.35;
        // a faint tonal whistle following the band gives the pass its direction
        let pitch = if up { 300.0 * 6f64.powf(x) } else { 1800.0 / 6f64.powf(x) };
        whistle = wrap(whistle + pitch / sr);
        let v = (body + (whistle * TAU).sin() * 0.06) * env;
        let pan = if up { -0.7 + 1.4 * x } else { 0.7 - 1.4 * x };
        let (gl, gr) = pan_gains(pan);
        *l = (v * gl) as f32;
        *r = (v * gr) as f32;
    }
    reverb(left, right, sr, 1.0, 35.0, 55.0, 15.0);
}

fn sweep(sr: f64, seed: u64, left: &mut [f32], right: &mut [f32]) {
    let n = left.len().max(1);
    let mut rng = Rng::new(seed);
    let mut noise = Pink::new(seed.wrapping_add(7));
    let spread = 0.003 + 0.003 * rng.unit();
    // a root, fifth and octave above A2
    let notes = [110.0f64, 165.0, 220.0];
    let mut ph = [[0.0f64; 2]; 3];
    let mut lp = [Moving::new(FilterType::LowPass, 5.0), Moving::new(FilterType::LowPass, 5.0)];
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let x = i as f64 / n as f64;
        // up over 70 %, settling back down over the rest
        let cut = if x < 0.7 { 150.0 * (8000.0f64 / 150.0).powf(smoothstep(x / 0.7)) } else { 8000.0 * (600.0f64 / 8000.0).powf(smoothstep((x - 0.7) / 0.3)) };
        let env = smoothstep(x / 0.1) * (1.0 - smoothstep((x - 0.85) / 0.15));
        let nz = noise.next() * 0.5;
        let mut side = [nz, nz];
        for (note, pair) in notes.iter().zip(ph.iter_mut()) {
            for (k, p) in pair.iter_mut().enumerate() {
                let det = if k == 0 { 1.0 - spread } else { 1.0 + spread };
                let dt = note * det / sr;
                *p = wrap(*p + dt);
                if let Some(o) = side.get_mut(k) {
                    *o += saw(*p, dt) * 0.25;
                }
            }
        }
        let a = lp[0].tick(side[0], cut, sr);
        let b = lp[1].tick(side[1], cut * 1.02, sr);
        *l = (a * env) as f32;
        *r = (b * env) as f32;
    }
    reverb(left, right, sr, 2.0, 60.0, 50.0, 22.0);
}

fn boom(sr: f64, seed: u64, left: &mut [f32], right: &mut [f32]) {
    let mut nl = Rng::new(seed);
    let mut nr = Rng::new(seed.wrapping_add(1));
    let mut rumble = [Fixed::new(FilterType::LowPass, sr, 120.0, 0.7), Fixed::new(FilterType::LowPass, sr, 120.0, 0.7)];
    let mut phase = 0.0f64;
    let drive = 2.2f64;
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 / sr;
        // the pitch drops from ~70 Hz to 28 Hz
        let f = 28.0 + 42.0 * (-t / 0.35).exp();
        phase = wrap(phase + f / sr);
        let attack = (t / 0.004).min(1.0);
        let sub = (phase * TAU).sin() * attack * (-t / 0.9).exp();
        let sub = (sub * drive).tanh() / drive.tanh();
        let decay = (-t / 0.6).exp() * 1.5;
        let a = rumble[0].tick(nl.white()) * decay;
        let b = rumble[1].tick(nr.white()) * decay;
        *l = (sub + a) as f32;
        *r = (sub + b) as f32;
    }
    reverb(left, right, sr, 4.0, 95.0, 85.0, 25.0);
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

    /// Zero crossings per sample: a cheap brightness measure.
    fn zcr(x: &[f32]) -> f64 {
        x.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f64 / x.len().max(1) as f64
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
    }

    #[test]
    fn every_kind_is_sane_deterministic_and_seeded() {
        for kind in Sfx::ALL {
            let a = generate(kind, kind.default_seconds(), 1, SR).unwrap();
            let n = (kind.default_seconds() * f64::from(SR)).round() as usize;
            assert_eq!((a.left.len(), a.right.len()), (n, n), "{kind:?}");
            assert!(a.left.iter().chain(&a.right).all(|x| x.is_finite() && x.abs() <= 0.9), "{kind:?} finite and under the peak");
            assert!(rms(&a.left) > 1e-3 && rms(&a.right) > 1e-3, "{kind:?} is not silent");
            assert!(a.left.last().is_some_and(|x| x.abs() < 1e-3), "{kind:?} ends without a click");
            assert_eq!(a, generate(kind, kind.default_seconds(), 1, SR).unwrap(), "{kind:?} is deterministic");
            assert_ne!(a.left, generate(kind, kind.default_seconds(), 2, SR).unwrap().left, "{kind:?} depends on the seed");
            // every supported rate works
            for rate in [8_000, 44_100, 192_000] {
                let s = generate(kind, 0.2, 1, rate).unwrap();
                assert!(s.left.iter().all(|x| x.is_finite()), "{kind:?} at {rate}");
            }
        }
    }

    #[test]
    fn clicks_and_pops_are_short() {
        for (kind, within) in [(Sfx::Click, 0.03), (Sfx::Pop, 0.12)] {
            let s = generate(kind, kind.default_seconds(), 1, SR).unwrap();
            let k = (within * f64::from(SR)) as usize;
            let head = energy(&s.left[..k]);
            assert!(head > 0.9 * energy(&s.left), "{kind:?}: the energy is in the first {within} s");
        }
    }

    #[test]
    fn swooshes_go_their_way() {
        let up = generate(Sfx::SwooshUp, 0.7, 1, SR).unwrap();
        let down = generate(Sfx::SwooshDown, 0.7, 1, SR).unwrap();
        let q = up.left.len() / 4;
        // brightness rises for up, falls for down (compare the louder middle halves)
        let (u0, u1) = (zcr(&up.left[q / 2..q * 2]), zcr(&up.left[q * 2..q * 7 / 2]));
        let (d0, d1) = (zcr(&down.left[q / 2..q * 2]), zcr(&down.left[q * 2..q * 7 / 2]));
        assert!(u1 > u0, "up gets brighter: {u0} → {u1}");
        assert!(d1 < d0, "down gets darker: {d0} → {d1}");
        // up travels left → right, down right → left
        let lr = |s: &Stereo, a: usize, b: usize| rms(&s.left[a..b]) / rms(&s.right[a..b]).max(1e-9);
        assert!(lr(&up, 0, q) > lr(&up, 3 * q, 4 * q));
        assert!(lr(&down, 0, q) < lr(&down, 3 * q, 4 * q));
    }

    #[test]
    fn sweep_is_brightest_near_its_top_and_boom_is_low() {
        let s = generate(Sfx::Sweep, 1.6, 1, SR).unwrap();
        let n = s.left.len();
        let start = zcr(&s.left[n / 10..n / 5]);
        let top = zcr(&s.left[n * 6 / 10..n * 7 / 10]);
        assert!(top > 2.0 * start, "sweep start {start} top {top}");
        let b = generate(Sfx::Boom, 3.0, 1, SR).unwrap();
        let first = &b.left[..SR as usize / 2];
        assert!(zcr(first) < 0.01, "boom is low: {}", zcr(first));
        let tail = &b.left[b.left.len() - SR as usize / 2..];
        assert!(rms(first) > 4.0 * rms(tail), "boom decays");
    }

    #[test]
    fn hostile_parameters_are_clamped_or_refused() {
        assert!(generate(Sfx::Click, f64::NAN, 1, SR).is_ok(), "NaN length falls back to the default");
        assert!(generate(Sfx::Click, f64::INFINITY, 1, SR).is_ok(), "an infinite length falls back to the default");
        assert!(generate(Sfx::Click, -1.0, 1, SR).is_err());
        assert!(generate(Sfx::Click, 0.0, 1, SR).is_err());
        assert_eq!(generate(Sfx::Boom, 1e12, 1, SR).unwrap().seconds(), MAX_SECONDS);
        assert_eq!(generate(Sfx::Pop, 1e-9, 1, SR).unwrap().left.len(), (MIN_SECONDS * f64::from(SR)).round() as usize);
        assert!(generate(Sfx::Pop, 1.0, 1, 0).is_err());
        assert!(generate(Sfx::Pop, 1.0, 1, 7_999).is_err());
        assert!(generate(Sfx::Pop, 1.0, 1, 1_000_000).is_err());
        assert!(generate(Sfx::Sweep, 0.5, u64::MAX, SR).is_ok());
    }

    #[test]
    fn names_parse() {
        for k in Sfx::ALL {
            assert_eq!(Sfx::from_id(k.id()), Some(k));
        }
        assert_eq!(Sfx::from_id(" Swoosh-Up "), Some(Sfx::SwooshUp));
        assert_eq!(Sfx::from_id("whoosh"), None);
    }
}
