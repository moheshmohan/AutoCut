//! Voice Isolation's plumbing (hops, latency, resampling, mixing, the registry) with a model that
//! only delays, the DSP fallback's noise reduction, Enhance Voice and the Voice Changer.

use super::*;
use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::effects::enhance_voice::EnhanceVoice;
use crate::effects::voice_changer::{MODES, VoiceChanger};
use crate::testutil::Rng;

const SR: f32 = 48_000.0;

/// A "model" that gives every hop back [`MODEL_DELAY_HOPS`] hops late, unchanged.
struct Delay;

struct DelayStream {
    ring: Vec<Vec<[f32; HOP]>>,
    at: usize,
}

impl VoiceModel for Delay {
    fn name(&self) -> &str {
        "delay"
    }
    fn neural(&self) -> bool {
        true
    }
    fn new_stream(&self, channels: usize) -> Box<dyn VoiceStream> {
        Box::new(DelayStream { ring: vec![vec![[0.0; HOP]; channels]; MODEL_DELAY_HOPS], at: 0 })
    }
}

impl VoiceStream for DelayStream {
    fn process_hop(&mut self, hops: &mut [[f32; HOP]]) {
        for (c, h) in hops.iter_mut().enumerate() {
            std::mem::swap(&mut self.ring[self.at][c], h);
        }
        self.at = (self.at + 1) % MODEL_DELAY_HOPS;
    }
    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|f| f.iter_mut().for_each(|h| h.fill(0.0)));
        self.at = 0;
    }
}

fn delay_model() -> Option<Arc<dyn VoiceModel>> {
    Some(Arc::new(Delay))
}

fn noise(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.uniform() * amp).collect()
}

/// A band-limited test signal (a few partials below 3 kHz).
fn tones(sr: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1130.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 2710.0 * t + 0.4).sin()
        })
        .collect()
}

/// Something like speech: a glottal pulse train (f0 gliding 100–170 Hz) through three formant
/// resonators that move between five vowels, in syllables (about four a second, with gaps), and
/// short fricative bursts of high-passed noise between some of them.
pub(crate) fn speech_like(sr: f32, secs: f32, seed: u64) -> Vec<f32> {
    let n = (sr * secs) as usize;
    let vowels: [[f64; 3]; 5] = [[700.0, 1220.0, 2600.0], [390.0, 2300.0, 3000.0], [300.0, 870.0, 2240.0], [640.0, 1190.0, 2390.0], [500.0, 1500.0, 2500.0]];
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; n];
    let syl = (0.24 * sr) as usize;
    let mut phase = 0.0f64;
    let mut filt = [Biquad::new(); 3];
    let mut fric = [Biquad::new(); 2];
    let hp = Coeffs::design(FilterType::HighPass, f64::from(sr), 4000.0, 0.7, 0.0);
    let lp = Coeffs::design(FilterType::LowPass, f64::from(sr), 9000.0, 0.7, 0.0);
    for (s, start) in (0..n).step_by(syl).enumerate() {
        let v = vowels[(rng.next_u64() % 5) as usize];
        let coeffs: Vec<Coeffs> = v.iter().map(|&f| Coeffs::design(FilterType::BandPass, f64::from(sr), f, f / 90.0, 0.0)).collect();
        let voiced = (syl as f32 * 0.7) as usize;
        let f0a = 100.0 + 70.0 * f64::from(rng.uniform().abs());
        let f0b = 100.0 + 70.0 * f64::from(rng.uniform().abs());
        let fricative = s % 3 == 1;
        for i in 0..syl {
            let Some(o) = out.get_mut(start + i) else { break };
            if i < voiced {
                let u = i as f64 / voiced as f64;
                let f0 = f0a + (f0b - f0a) * u;
                phase = (phase + f0 / f64::from(sr)).fract();
                // a sawtooth-like glottal pulse
                let src = 1.0 - 2.0 * phase;
                let mut y = 0.0;
                for (k, (st, c)) in filt.iter_mut().zip(&coeffs).enumerate() {
                    y += st.tick(c, src) * [1.0, 0.6, 0.35][k];
                }
                let env = (std::f64::consts::PI * u).sin().powf(0.6);
                *o = (0.5 * y * env) as f32;
            } else if fricative {
                let u = (i - voiced) as f64 / (syl - voiced) as f64;
                let x = f64::from(rng.uniform());
                let h = fric[0].tick(&hp, x);
                let y = fric[1].tick(&lp, h);
                *o = (0.15 * y * (std::f64::consts::PI * u).sin()) as f32;
            }
        }
    }
    out
}

/// Scale-invariant SDR of `est` against `reference` (dB), from `skip` on.
pub(crate) fn si_sdr(est: &[f32], reference: &[f32], skip: usize) -> f64 {
    let (e, s) = (&est[skip..], &reference[skip..]);
    let dot: f64 = e.iter().zip(s).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let ss: f64 = s.iter().map(|b| f64::from(*b) * f64::from(*b)).sum();
    let alpha = dot / ss.max(1e-20);
    let (mut target, mut err) = (0.0f64, 0.0f64);
    for (a, b) in e.iter().zip(s) {
        let t = alpha * f64::from(*b);
        target += t * t;
        err += (f64::from(*a) - t).powi(2);
    }
    10.0 * (target / err.max(1e-20)).log10()
}

fn run(fx: &mut dyn AudioEffect, input: &[Vec<f32>], sizes: &mut dyn FnMut() -> usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = input.to_vec();
    let n = input[0].len();
    let mut pos = 0;
    while pos < n {
        let len = sizes().min(n - pos);
        let mut refs: Vec<&mut [f32]> = out.iter_mut().map(|c| &mut c[pos..pos + len]).collect();
        fx.process(&mut refs);
        pos += len;
    }
    out
}

#[test]
fn the_stft_gives_the_input_back_one_hop_late() {
    let mut st = HopStft::new().unwrap();
    let x = noise(1, HOP * 20, 0.5);
    let (mut re, mut im) = (vec![0.0; BINS], vec![0.0; BINS]);
    let mut y = Vec::new();
    for hop in x.as_chunks::<HOP>().0 {
        st.analyse(hop, &mut re, &mut im);
        let mut o = [0.0; HOP];
        st.synthesise(&re, &im, &mut o);
        y.extend_from_slice(&o);
    }
    for i in HOP..x.len() {
        assert!((y[i] - x[i - HOP]).abs() < 1e-5, "sample {i}: {} vs {}", y[i], x[i - HOP]);
    }
}

#[test]
fn at_48k_the_model_output_is_exactly_the_latency_late() {
    let x = noise(2, 20_000, 0.5);
    for intensity in [0.0, 37.0, 100.0] {
        let mut fx = VoiceIsolation::with_model(SR, 1, delay_model());
        assert_eq!(fx.latency(), LATENCY_48K);
        assert!(fx.neural() && fx.active());
        fx.set_param("intensity", intensity);
        fx.reset();
        let y = run(&mut fx, std::slice::from_ref(&x), &mut || 333);
        for i in LATENCY_48K..x.len() {
            assert!((y[0][i] - x[i - LATENCY_48K]).abs() < 1e-6, "intensity {intensity} sample {i}");
        }
        assert!(y[0][..LATENCY_48K].iter().all(|v| *v == 0.0));
    }
}

#[test]
fn other_rates_are_resampled_and_stay_in_sync() {
    for sr in [44_100.0f32, 32_000.0, 96_000.0, 22_050.0, 16_000.0] {
        let n = (sr * 1.5) as usize;
        let x = tones(sr, n);
        let mut fx = VoiceIsolation::with_model(sr, 2, delay_model());
        fx.set_param("intensity", 100.0);
        fx.reset();
        let lat = fx.latency();
        // the wet path alone: the round trip is a pure delay of `lat` samples
        let y = run(&mut fx, &[x.clone(), x.clone()], &mut || 1000);
        let (mut err, mut sig) = (0.0f64, 0.0f64);
        for i in lat + 2000..n {
            err += f64::from(y[0][i] - x[i - lat]).powi(2);
            sig += f64::from(x[i - lat]).powi(2);
        }
        let snr = 10.0 * (sig / err.max(1e-30)).log10();
        assert!(snr > 50.0, "{sr} Hz: the resampled round trip is {snr:.1} dB from the delayed input (latency {lat})");
        assert_eq!(y[0], y[1]);
    }
}

#[test]
fn resampling_is_independent_of_block_sizes() {
    let x = noise(3, 30_000, 0.4);
    let mk = || {
        let mut fx = VoiceIsolation::with_model(44_100.0, 1, Some(Arc::new(SpectralGate)));
        fx.set_param("intensity", 90.0);
        fx.reset();
        fx
    };
    let big = run(&mut mk(), std::slice::from_ref(&x), &mut || usize::MAX);
    let mut rng = Rng::new(9);
    let small = run(&mut mk(), std::slice::from_ref(&x), &mut || (rng.next_u64() % 700) as usize);
    assert_eq!(big, small);
}

#[test]
fn hostile_rates_pass_audio_through_without_latency() {
    for sr in [0.0f32, -48_000.0, f32::NAN, f32::INFINITY, 1e12, 7_999.0, 384_001.0] {
        for (name, mut fx) in [
            ("isolation", Box::new(VoiceIsolation::with_model(sr, 2, delay_model())) as Box<dyn AudioEffect>),
            ("enhance", Box::new(EnhanceVoice::with_model(sr, 2, None))),
            ("changer", Box::new(VoiceChanger::new(sr, 2))),
        ] {
            assert_eq!(fx.latency(), 0, "{name} at {sr}");
            let x = noise(4, 3000, 0.5);
            let y = run(fx.as_mut(), &[x.clone(), x.clone()], &mut || 512);
            assert_eq!(y[0], x, "{name} at {sr}");
        }
    }
    assert!(!VoiceIsolation::with_model(0.0, 2, None).active());
    // zero and absurd channel counts
    let mut fx = VoiceIsolation::with_model(SR, 0, None);
    let mut a = [0.5f32; 64];
    fx.process(&mut [&mut a]);
    let mut fx = VoiceIsolation::with_model(SR, 10_000, None);
    fx.process(&mut [&mut a]);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn hostile_samples_and_params_stay_finite() {
    let mut fx = VoiceIsolation::with_model(SR, 2, None);
    for v in [f32::NAN, f32::INFINITY, -1e30, 1e30] {
        assert!(fx.set_param("intensity", v));
        let p = fx.param("intensity").unwrap();
        assert!((0.0..=100.0).contains(&p), "{v} → {p}");
    }
    let mut l: Vec<f32> = noise(5, 10_000, 0.5);
    l[100] = f32::NAN;
    l[200] = f32::INFINITY;
    l[300] = 1e30;
    let mut r = l.clone();
    fx.process(&mut [&mut l, &mut r]);
    assert!(l.iter().chain(&r).all(|v| v.is_finite()));
    assert!(!fx.set_param("nope", 1.0));
    assert_eq!(intensity_mix(f32::NAN), 0.0);
    assert_eq!(intensity_mix(0.0), 0.0);
    assert_eq!(intensity_mix(100.0), 1.0);
    assert!((intensity_mix(50.0) - 0.969).abs() < 0.01);
}

#[test]
fn the_registry_loads_once_and_installs() {
    let reg = ModelRegistry::new();
    assert!(reg.get().is_none());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c = calls.clone();
    reg.set_loader(move || {
        c.fetch_add(1, Ordering::SeqCst);
        None
    });
    assert!(reg.get().is_none());
    assert!(reg.get().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1, "a failed load is not retried on every instance");
    reg.set_loader(delay_model);
    assert_eq!(reg.get().map(|m| m.name().to_string()).as_deref(), Some("delay"));
    reg.set_loader(|| panic!("a broken loader"));
    assert!(reg.get().is_some(), "the installed model stays");
    reg.install(None);
    assert!(reg.get().is_none(), "a loader that panics counts as no model");
    reg.install(Some(Arc::new(SpectralGate)));
    assert!(!reg.get().unwrap().neural());
}

#[test]
fn offline_processing_lines_up_with_the_input() {
    let x = noise(6, 9000, 0.3);
    let mut fx = VoiceIsolation::with_model(SR, 1, delay_model());
    fx.set_param("intensity", 100.0);
    fx.reset();
    let mut ch = vec![x.clone()];
    process_offline(&mut fx, &mut ch);
    assert_eq!(ch[0].len(), x.len());
    for i in 0..x.len() {
        assert!((ch[0][i] - x[i]).abs() < 1e-6, "{i}");
    }
}

/// The fallback (no neural model): speech-like sound in steady noise 5 dB down comes out cleaner
/// (measured 2026-10-22: SI-SDR 5.0 → 12.9 dB).
#[test]
fn the_dsp_fallback_reduces_steady_noise() {
    let secs = 6.0;
    let s = speech_like(SR, secs, 11);
    let pow = s.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / s.len() as f64;
    // white-ish noise 5 dB below the speech
    let amp = (pow / 10f64.powf(0.5) * 3.0).sqrt() as f32;
    let nz = noise(12, s.len(), amp);
    let mix: Vec<f32> = s.iter().zip(&nz).map(|(a, b)| a + b).collect();
    let mut fx = VoiceIsolation::with_model(SR, 1, None);
    fx.set_param("intensity", 100.0);
    fx.reset();
    let mut ch = vec![mix.clone()];
    process_offline(&mut fx, &mut ch);
    let skip = SR as usize * 2;
    let (before, after) = (si_sdr(&mix, &s, skip), si_sdr(&ch[0], &s, skip));
    eprintln!("DSP fallback: SI-SDR {before:.1} dB → {after:.1} dB");
    assert!(after - before >= 5.0, "the fallback improved SI-SDR by only {:.1} dB ({before:.1} → {after:.1})", after - before);
}

#[test]
fn enhance_voice_lifts_the_speech_band_and_lowers_the_noise_floor() {
    let secs = 6.0;
    let s = speech_like(SR, secs, 21);
    // rumble and hiss under the voice
    let mut rng = Rng::new(22);
    let mut lp = Biquad::new();
    let rumble_c = Coeffs::design(FilterType::LowPass, 48_000.0, 90.0, 0.7, 0.0);
    let nz: Vec<f32> = (0..s.len()).map(|_| 0.02 * rng.uniform() + 0.5 * lp.tick(&rumble_c, f64::from(rng.uniform())) as f32).collect();
    let mix: Vec<f32> = s.iter().zip(&nz).map(|(a, b)| a + b).collect();
    let mut fx = EnhanceVoice::with_model(SR, 1, None);
    fx.set_param("amount", 100.0);
    fx.reset();
    let mut ch = vec![mix.clone()];
    process_offline(&mut fx, &mut ch);
    let out = &ch[0];
    let skip = SR as usize * 2;
    let share = |x: &[f32]| band_share(&x[skip..], SR, 1000.0, 4000.0);
    let floor = |x: &[f32]| noise_floor_db(&x[skip..], SR);
    let (s0, s1) = (share(&mix), share(out));
    let (f0, f1) = (floor(&mix), floor(out));
    eprintln!("Enhance Voice: 1–4 kHz share {:.1} % → {:.1} %, noise floor {f0:.1} → {f1:.1} dBFS", 100.0 * s0, 100.0 * s1);
    assert!(s1 > s0 * 1.5, "1–4 kHz share {s0:.3} → {s1:.3}");
    assert!(f1 < f0 - 6.0, "noise floor {f0:.1} → {f1:.1} dB");
    assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}

/// Share of the energy between `lo` and `hi` Hz (a DFT of 4096-sample frames).
pub(crate) fn band_share(x: &[f32], sr: f32, lo: f32, hi: f32) -> f64 {
    let n = 4096;
    let fft = crate::fft::Fft::new(n);
    let (mut band, mut total) = (0.0f64, 0.0f64);
    for f in x.chunks_exact(n) {
        let mut re: Vec<f32> = f.iter().enumerate().map(|(i, v)| v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())).collect();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);
        for k in 1..n / 2 {
            let p = f64::from(re[k] * re[k] + im[k] * im[k]);
            let hz = k as f32 * sr / n as f32;
            total += p;
            if (lo..hi).contains(&hz) {
                band += p;
            }
        }
    }
    band / total.max(1e-30)
}

/// The noise floor: the 10th percentile of 20 ms RMS levels (dBFS).
pub(crate) fn noise_floor_db(x: &[f32], sr: f32) -> f64 {
    let w = (sr * 0.02) as usize;
    let mut lv: Vec<f64> =
        x.chunks_exact(w).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / w as f64 + 1e-20).log10()).collect();
    lv.sort_by(|a, b| a.total_cmp(b));
    lv.get(lv.len() / 10).copied().unwrap_or(-200.0)
}

#[test]
fn every_voice_changer_mode_is_bounded() {
    let x: Vec<f32> = speech_like(SR, 3.0, 31).iter().zip(noise(32, (SR * 3.0) as usize, 0.05)).map(|(a, b)| 2.0 * a + b).collect();
    for (m, name) in MODES.iter().enumerate() {
        for (pitch, character, mix) in [(0.0, 50.0, 100.0), (12.0, 100.0, 100.0), (-12.0, 0.0, 60.0)] {
            let mut fx = VoiceChanger::new(SR, 2);
            fx.set_param("mode", m as f32);
            fx.set_param("pitch", pitch);
            fx.set_param("character", character);
            fx.set_param("mix", mix);
            fx.reset();
            let y = run(&mut fx, &[x.clone(), x.clone()], &mut || 777);
            let peak = y[0].iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let rms_in = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
            let rms_out = (y[0].iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
            assert!(y[0].iter().all(|v| v.is_finite()), "{name}");
            assert!(peak <= 4.0, "{name}: peak {peak}");
            assert!(rms_out < 4.0 * rms_in, "{name} (pitch {pitch}, character {character}): {rms_in} → {rms_out}: runaway gain");
            assert!(rms_out > 0.05 * rms_in, "{name}: silenced the voice ({rms_in} → {rms_out})");
        }
    }
}

#[test]
fn the_voice_changer_latency_is_the_same_in_every_mode() {
    let base = VoiceChanger::new(SR, 2).latency();
    for m in 0..MODES.len() {
        let mut fx = VoiceChanger::new(SR, 2);
        fx.set_param("mode", m as f32);
        assert_eq!(fx.latency(), base);
    }
    // mix 0: the dry voice, delayed by exactly the latency
    let x = noise(33, 12_000, 0.3);
    let mut fx = VoiceChanger::new(SR, 1);
    fx.set_param("mix", 0.0);
    fx.reset();
    let y = run(&mut fx, std::slice::from_ref(&x), &mut || 500);
    for i in base..x.len() {
        assert!((y[0][i] - x[i - base]).abs() < 1e-6, "{i}");
    }
}
