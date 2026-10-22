//! Enhance Voice: the one-knob voice boost. One `amount` (0–100 %) drives a chain made for
//! speech that has to be understood on small speakers:
//!
//! 1. [`VoiceIsolation`] at `amount` intensity (the neural model when installed);
//! 2. a rumble high-pass, 4th-order Butterworth, from 20 Hz at 0 % to 80 Hz at 100 %;
//! 3. a gentle version of AutoCut's speech-first narration EQ (`filmcraft-autocut`'s `mix`):
//!    −3 dB low shelf at 200 Hz (mud), +2 dB at 1.2 kHz and +3 dB at 3 kHz (presence: what words
//!    are understood by at low volume), +2.5 dB high shelf from 6.5 kHz (consonants), all scaled by
//!    `amount`;
//! 4. a de-esser at 6 kHz (up to 6 dB of reduction) for the lift it just gave the sibilants;
//! 5. a compressor, 3:1 at 100 % (1:1 at 0 %) from −20 dB, RMS detector, 3 ms / 120 ms, with
//!    make-up gain;
//! 6. `output` gain, then a true-peak limiter at −1 dBFS.
//!
//! The loudness target (−16 LUFS per clip, or the mix's dialogue target) is not part of the
//! effect: it needs the whole clip measured, so `audio.enhanceVoice` sets it as the clip's gain
//! after measuring the processed clip. At 0 % the chain is transparent but for the limiter
//! (which only touches peaks above −1 dBFS) and a 20 Hz high-pass.

use super::dynamics::{Compressor, Limiter};
use super::essential::DeEsser;
use super::voice::{MAX_RATE, MIN_RATE, VoiceIsolation, VoiceModel};
use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::{AudioEffect, ParamSpec, ParamValues, Smoothed, Unit, block_len, param_plumbing};
use std::sync::Arc;

/// Biquad sections: two high-pass (4th order), low shelf, two peaks, high shelf.
const STAGES: usize = 6;
/// Samples between EQ coefficient updates while `amount` glides.
const COEFF_EVERY: u32 = 64;
/// Channels the EQ runs on (more pass the EQ untouched).
const MAX_EQ_CHANNELS: usize = 32;

/// The EQ at `a` = amount / 100.
fn eq(sr: f64, a: f64) -> [Coeffs; STAGES] {
    let hp = 20.0 + 60.0 * a;
    [
        Coeffs::design(FilterType::HighPass, sr, hp, 0.541_196, 0.0),
        Coeffs::design(FilterType::HighPass, sr, hp, 1.306_563, 0.0),
        Coeffs::design(FilterType::LowShelf, sr, 200.0, std::f64::consts::FRAC_1_SQRT_2, -3.0 * a),
        Coeffs::design(FilterType::Peaking, sr, 1200.0, 0.9, 2.0 * a),
        Coeffs::design(FilterType::Peaking, sr, 3000.0, 1.0, 3.0 * a),
        Coeffs::design(FilterType::HighShelf, sr, 6500.0, std::f64::consts::FRAC_1_SQRT_2, 2.5 * a),
    ]
}

/// Enhance Voice (see the module docs).
pub struct EnhanceVoice {
    pv: ParamValues,
    active: bool,
    sr: f64,
    iso: VoiceIsolation,
    coeffs: [Coeffs; STAGES],
    filters: Vec<[Biquad; STAGES]>,
    deess: DeEsser,
    comp: Compressor,
    limiter: Limiter,
    amount: Smoothed,
    /// The amount the EQ coefficients were designed for.
    designed: f32,
    counter: u32,
    out_gain: Smoothed,
}

impl EnhanceVoice {
    pub const PARAMS: &'static [ParamSpec] =
        &[ParamSpec::new("amount", "Amount", 0.0, 100.0, 70.0, Unit::Percent), ParamSpec::new("output", "Output Gain", -24.0, 24.0, 0.0, Unit::Decibels)];

    /// With the installed voice model, else the DSP fallback.
    pub fn new(sample_rate: f32, channels: usize) -> Self {
        Self::with_model(sample_rate, channels, super::voice::voice_model())
    }

    /// With `model` for the isolation stage (`None`: the DSP fallback).
    pub fn with_model(sample_rate: f32, channels: usize, model: Option<Arc<dyn VoiceModel>>) -> Self {
        let ok = sample_rate.is_finite() && (MIN_RATE..=MAX_RATE).contains(&sample_rate);
        // the stages are built at a safe rate when this one is unusable (and then never run)
        let sr = if ok { sample_rate } else { 48_000.0 };
        let ch = channels.clamp(1, MAX_EQ_CHANNELS);
        let iso = VoiceIsolation::with_model(sample_rate, channels, model);
        let mut deess = DeEsser::new(sr, ch);
        deess.set_param("frequency", 6000.0);
        deess.set_param("threshold", -10.0);
        let mut comp = Compressor::new(sr, ch);
        comp.set_param("threshold", -20.0);
        comp.set_param("knee", 6.0);
        comp.set_param("attack", 3.0);
        comp.set_param("release", 120.0);
        comp.set_param("detector", 1.0);
        let mut limiter = Limiter::new(sr, ch);
        limiter.set_param("ceiling", -1.0);
        limiter.set_param("release", 60.0);
        let mut s = EnhanceVoice {
            pv: ParamValues::new(Self::PARAMS),
            active: ok && iso.active(),
            sr: f64::from(sr),
            iso,
            coeffs: [Coeffs::IDENTITY; STAGES],
            filters: vec![[Biquad::new(); STAGES]; ch],
            deess,
            comp,
            limiter,
            amount: Smoothed::with_ms(0.0, sr, 50.0),
            designed: -1.0,
            counter: 0,
            out_gain: Smoothed::with_ms(1.0, sr, 30.0),
        };
        s.apply_params(true);
        s.reset();
        s
    }

    /// Which model the isolation stage runs.
    pub fn model_name(&self) -> &str {
        self.iso.model_name()
    }

    /// Whether it processes at all (false: the rate is outside 8–384 kHz and audio passes).
    pub fn active(&self) -> bool {
        self.active
    }

    fn apply_params(&mut self, snap: bool) {
        let amount = self.pv.v("amount");
        let a = amount / 100.0;
        self.iso.set_param("intensity", amount);
        self.deess.set_param("reduction", 6.0 * a);
        self.comp.set_param("ratio", 1.0 + 2.0 * a);
        self.comp.set_param("makeup", 4.0 * a);
        self.amount.set(a);
        self.out_gain.set(crate::db_to_gain(self.pv.v("output")));
        if snap {
            self.amount.snap();
            self.out_gain.snap();
            self.redesign(a);
        }
    }

    fn redesign(&mut self, a: f32) {
        self.coeffs = eq(self.sr, f64::from(a));
        self.designed = a;
    }
}

impl AudioEffect for EnhanceVoice {
    param_plumbing!("enhance_voice");
    fn latency(&self) -> usize {
        if self.active { self.iso.latency() + self.deess.latency() + self.comp.latency() + self.limiter.latency() } else { 0 }
    }
    fn reset(&mut self) {
        self.iso.reset();
        self.filters.iter_mut().for_each(|f| f.iter_mut().for_each(Biquad::reset));
        self.deess.reset();
        self.comp.reset();
        self.limiter.reset();
        self.amount.snap();
        self.out_gain.snap();
        let a = self.amount.tick();
        self.redesign(a);
        self.counter = 0;
    }
    fn process(&mut self, channels: &mut [&mut [f32]]) {
        if !self.active {
            return;
        }
        let n = block_len(channels);
        self.iso.process(channels);
        let nch = channels.len().min(self.filters.len());
        for i in 0..n {
            let a = self.amount.tick();
            if self.counter == 0 && (a - self.designed).abs() > 1e-4 {
                self.redesign(a);
            }
            self.counter = (self.counter + 1) % COEFF_EVERY;
            for (c, chan) in channels.iter_mut().enumerate().take(nch) {
                let mut x = f64::from(chan[i]);
                for (st, co) in self.filters[c].iter_mut().zip(self.coeffs.iter()) {
                    x = st.tick(co, x);
                }
                chan[i] = crate::flush32(x as f32);
            }
        }
        self.deess.process(channels);
        self.comp.process(channels);
        for i in 0..n {
            let g = self.out_gain.tick();
            for chan in channels.iter_mut() {
                if let Some(v) = chan.get_mut(i) {
                    *v *= g;
                }
            }
        }
        self.limiter.process(channels);
        for chan in channels.iter_mut() {
            for v in chan.iter_mut().take(n) {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
        }
    }
}
