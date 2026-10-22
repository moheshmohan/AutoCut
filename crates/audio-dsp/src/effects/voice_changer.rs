//! Voice Changer: character voices built from the existing DSP (the pitch shifter, RBJ biquads,
//! delay lines, ring modulation), for a robot channel's fun bits.
//!
//! | mode | sound | `character` (0–100 %) |
//! |---|---|---|
//! | Robot | ring modulation (30–90 Hz carrier) of the voice through a comb resonator tuned to a pitch quantised to the nearest semitone (a metallic, monotone formant), band-limited to 200 Hz–5 kHz | the comb's pitch (80–200 Hz, snapped to semitones) and the carrier |
//! | Deep | −5 semitones, a low shelf at 150 Hz, 7 kHz low-pass | how far the low shelf lifts (0–8 dB) |
//! | High | +7 semitones, 250 Hz high-pass, a high shelf at 5 kHz | the high shelf (0–6 dB) |
//! | Echo | a feedback delay (120–520 ms, 45 % feedback) mixed in at 40 % | the delay time |
//! | Radio | walkie-talkie: 400 Hz–2.8 kHz band (two high-pass sections and a low-pass), +4 dB at 1.5 kHz, tanh drive | the drive (1–9×) |
//! | Alien | +4 semitones, a 300–1500 Hz ring modulator mixed in at 40 %, 5.5 Hz vibrato (1.5 ms) | the ring frequency |
//!
//! `pitch` adds semitones to the mode's own shift (the sum is clamped to ±12); `mix` blends with
//! the dry voice, delayed to match. Every mode runs through the pitch shifter, so the latency
//! (its frame) is the same whatever the mode and can be compensated. Feedback stays below 1 and
//! the output is clamped to ±4 (non-finite samples become silence): no mode runs away.

use super::pitch::PitchShifter;
use super::util::DelayLine;
use super::voice::{MAX_RATE, MIN_RATE};
use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::{AudioEffect, ParamSpec, ParamValues, Smoothed, Unit, block_len, param_plumbing};

/// The modes, in parameter order.
pub const MODES: &[&str] = &["Robot", "Deep", "High", "Echo", "Radio", "Alien"];
/// Each mode's own pitch shift (semitones).
const MODE_PITCH: [f32; 6] = [0.0, -5.0, 7.0, 0.0, 0.0, 4.0];
const STAGES: usize = 4;
/// Longest echo (s).
const MAX_ECHO_S: f32 = 0.6;
/// Longest comb period (s): the lowest robot pitch is 80 Hz.
const MAX_COMB_S: f32 = 1.0 / 70.0;
/// Vibrato depth and centre (s).
const VIB_S: f32 = 0.0015;
/// Samples processed at once (bounds the dry line).
const CHUNK: usize = 1024;
const MAX_CHANNELS: usize = 32;

/// The voice changer (see the module docs).
pub struct VoiceChanger {
    pv: ParamValues,
    active: bool,
    sr: f32,
    pitch: PitchShifter,
    coeffs: [Coeffs; STAGES],
    filters: Vec<[Biquad; STAGES]>,
    comb: Vec<DelayLine>,
    echo: Vec<DelayLine>,
    vib: Vec<DelayLine>,
    dry: Vec<DelayLine>,
    mode: usize,
    /// Comb period (samples), echo time (samples), ring carrier (Hz), drive.
    comb_d: f32,
    echo_d: f32,
    ring_hz: f32,
    drive: f32,
    /// Carrier and vibrato phases (cycles).
    carrier: f64,
    lfo: f64,
    mix: Smoothed,
}

/// The nearest equal-tempered pitch (A = 440 Hz) to `hz`.
fn quantise(hz: f32) -> f32 {
    let st = (12.0 * (hz.max(1.0) / 440.0).log2()).round();
    440.0 * 2f32.powf(st / 12.0)
}

impl VoiceChanger {
    pub const PARAMS: &'static [ParamSpec] = &[
        ParamSpec::choice("mode", "Mode", MODES, 0),
        ParamSpec::new("mix", "Mix", 0.0, 100.0, 100.0, Unit::Percent),
        ParamSpec::new("pitch", "Pitch", -12.0, 12.0, 0.0, Unit::Semitones),
        ParamSpec::new("character", "Character", 0.0, 100.0, 50.0, Unit::Percent),
    ];

    pub fn new(sample_rate: f32, channels: usize) -> Self {
        let ok = sample_rate.is_finite() && (MIN_RATE..=MAX_RATE).contains(&sample_rate);
        let sr = if ok { sample_rate } else { 48_000.0 };
        let ch = channels.clamp(1, MAX_CHANNELS);
        let pitch = PitchShifter::new(sr, ch);
        let lat = pitch.latency();
        let line = |s: f32| DelayLine::new((s * sr).ceil() as usize + 4);
        let mut v = VoiceChanger {
            pv: ParamValues::new(Self::PARAMS),
            active: ok,
            sr,
            pitch,
            coeffs: [Coeffs::IDENTITY; STAGES],
            filters: vec![[Biquad::new(); STAGES]; ch],
            comb: (0..ch).map(|_| line(MAX_COMB_S)).collect(),
            echo: (0..ch).map(|_| line(MAX_ECHO_S)).collect(),
            vib: (0..ch).map(|_| line(3.0 * VIB_S)).collect(),
            dry: (0..ch).map(|_| DelayLine::new(lat + CHUNK + 4)).collect(),
            mode: 0,
            comb_d: 1.0,
            echo_d: 1.0,
            ring_hz: 30.0,
            drive: 1.0,
            carrier: 0.0,
            lfo: 0.0,
            mix: Smoothed::with_ms(1.0, sr, 20.0),
        };
        v.apply_params(true);
        v
    }

    /// Whether it processes at all (false: the rate is outside 8–384 kHz and audio passes).
    pub fn active(&self) -> bool {
        self.active
    }

    fn apply_params(&mut self, snap: bool) {
        let mode = self.pv.idx("mode").min(MODES.len() - 1);
        let ch = self.pv.v("character") / 100.0;
        let sr = f64::from(self.sr);
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let d = |k, f, q, g| Coeffs::design(k, sr, f, q, g);
        use FilterType::{BandPass, HighPass, HighShelf, LowPass, LowShelf, Peaking};
        let id = Coeffs::IDENTITY;
        self.coeffs = match mode {
            0 => [d(HighPass, 200.0, q, 0.0), d(LowPass, 5000.0, q, 0.0), id, id],
            1 => [d(LowShelf, 150.0, q, f64::from(8.0 * ch)), d(LowPass, 7000.0, q, 0.0), id, id],
            2 => [d(HighPass, 250.0, q, 0.0), d(HighShelf, 5000.0, q, f64::from(6.0 * ch)), id, id],
            3 => [id, id, id, id],
            4 => [d(HighPass, 400.0, q, 0.0), d(HighPass, 400.0, q, 0.0), d(LowPass, 2800.0, q, 0.0), d(Peaking, 1500.0, 1.2, 4.0)],
            _ => [d(BandPass, 1200.0, 0.3, 0.0), id, id, id],
        };
        self.mode = mode;
        let robot = quantise(80.0 + 120.0 * ch);
        self.comb_d = (self.sr / robot).max(1.0);
        self.echo_d = ((0.12 + 0.4 * ch) * self.sr).max(1.0);
        self.ring_hz = if mode == 5 { 300.0 + 1200.0 * ch } else { 30.0 + 60.0 * ch };
        self.drive = 1.0 + 8.0 * ch;
        let semis = (MODE_PITCH.get(mode).copied().unwrap_or(0.0) + self.pv.v("pitch")).clamp(-12.0, 12.0);
        self.pitch.set_param("semitones", semis);
        self.mix.set(self.pv.v("mix") / 100.0);
        if snap {
            self.mix.snap();
            self.pitch.reset();
        }
    }

    /// The mode's processing of one (already pitch-shifted) sample of channel `c`.
    #[inline]
    fn voice(&mut self, c: usize, x: f32, carrier: f32, lfo: f32) -> f32 {
        let mut y = f64::from(x);
        for (st, co) in self.filters[c].iter_mut().zip(self.coeffs.iter()) {
            y = st.tick(co, y);
        }
        let y = y as f32;
        match self.mode {
            0 => {
                // comb resonator on the quantised pitch, then the ring modulator
                let fb = 0.6 * self.comb[c].read(self.comb_d - 1.0);
                let r = y + fb;
                self.comb[c].push(r);
                0.6 * r * (0.25 + 0.75 * carrier)
            }
            3 => {
                let e = self.echo[c].read(self.echo_d - 1.0);
                self.echo[c].push(y + 0.45 * e);
                y + 0.4 * e
            }
            4 => {
                let g = self.drive;
                (g * y).tanh() / g.tanh().max(1e-6) * 0.8
            }
            5 => {
                self.vib[c].push(x);
                let wobble = self.vib[c].read(self.sr * VIB_S * (1.0 + lfo));
                wobble + 0.4 * y * carrier
            }
            _ => y,
        }
    }
}

impl AudioEffect for VoiceChanger {
    param_plumbing!("voice_changer");
    fn latency(&self) -> usize {
        if self.active { self.pitch.latency() } else { 0 }
    }
    fn reset(&mut self) {
        self.pitch.reset();
        self.filters.iter_mut().for_each(|f| f.iter_mut().for_each(Biquad::reset));
        self.comb.iter_mut().chain(self.echo.iter_mut()).chain(self.vib.iter_mut()).chain(self.dry.iter_mut()).for_each(DelayLine::reset);
        self.carrier = 0.0;
        self.lfo = 0.0;
        self.mix.snap();
    }
    fn process(&mut self, channels: &mut [&mut [f32]]) {
        if !self.active {
            return;
        }
        let n = block_len(channels);
        let nch = channels.len().min(self.filters.len());
        let lat = self.pitch.latency();
        let mut at = 0;
        while at < n {
            let len = CHUNK.min(n - at);
            for c in 0..nch {
                for i in 0..len {
                    let x = channels[c][at + i];
                    self.dry[c].push(if x.is_finite() { x } else { 0.0 });
                }
            }
            {
                // the chunk of every processed channel, without allocating
                let mut part: [&mut [f32]; MAX_CHANNELS] = Default::default();
                let mut k = 0;
                for (slot, chan) in part.iter_mut().zip(channels.iter_mut()).take(nch) {
                    if let Some(s) = chan.get_mut(at..at + len) {
                        *slot = s;
                        k += 1;
                    }
                }
                self.pitch.process(&mut part[..k]);
            }
            for i in 0..len {
                let carrier = (2.0 * std::f64::consts::PI * self.carrier).sin() as f32;
                let lfo = (2.0 * std::f64::consts::PI * self.lfo).sin() as f32;
                self.carrier = (self.carrier + f64::from(self.ring_hz) / f64::from(self.sr)).fract();
                self.lfo = (self.lfo + 5.5 / f64::from(self.sr)).fract();
                let m = self.mix.tick();
                for c in 0..nch {
                    let shifted = channels[c][at + i];
                    let wet = self.voice(c, if shifted.is_finite() { shifted } else { 0.0 }, carrier, lfo);
                    let dry = self.dry[c].tap(len - 1 - i + lat);
                    let y = dry + m * (wet - dry);
                    channels[c][at + i] = if y.is_finite() { crate::flush32(y.clamp(-4.0, 4.0)) } else { 0.0 };
                }
            }
            at += len;
        }
    }
}
