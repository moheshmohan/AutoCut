//! DeepFilterNet 3's plumbing with random weights (shapes, state per stream, block sizes,
//! hostile input), and, with the real model (`FILMCRAFT_VOICE_MODELS=<models dir>`), how much
//! it cleans speech and how fast; with `FILMCRAFT_VOICE_ORACLE=<dir>` (libDF's own output, see
//! the README), that it matches DeepFilterNet's reference implementation.

use std::sync::Arc;

use filmcraft_audio_dsp::AudioEffect;
use filmcraft_audio_dsp::biquad::{Biquad, Coeffs, FilterType};
use filmcraft_audio_dsp::effects::voice::{HOP, LATENCY_48K, VoiceIsolation, VoiceModel, VoiceStream, process_offline};

use crate::df3::erb_widths;
use crate::{Df3, weights};

const SR: f32 = 48_000.0;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_model() -> Arc<dyn VoiceModel> {
    Arc::new(Df3::new(weights::random(7)).unwrap())
}

fn noise(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.uniform() * amp).collect()
}

/// Something like speech (see `filmcraft-audio-dsp`'s voice tests): a glottal pulse train through
/// moving formants, in syllables with gaps, and fricative bursts.
fn speech_like(sr: f32, secs: f32, seed: u64) -> Vec<f32> {
    let n = (sr * secs) as usize;
    let vowels: [[f64; 3]; 5] = [[700.0, 1220.0, 2600.0], [390.0, 2300.0, 3000.0], [300.0, 870.0, 2240.0], [640.0, 1190.0, 2390.0], [500.0, 1500.0, 2500.0]];
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; n];
    let syl = (0.24 * sr) as usize;
    let mut phase = 0.0f64;
    let mut filt = [Biquad::new(); 3];
    let (mut f0s, mut f1s) = (Biquad::new(), Biquad::new());
    let hp = Coeffs::design(FilterType::HighPass, f64::from(sr), 4000.0, 0.7, 0.0);
    let lp = Coeffs::design(FilterType::LowPass, f64::from(sr), 9000.0, 0.7, 0.0);
    for (s, start) in (0..n).step_by(syl).enumerate() {
        let v = vowels[(rng.next() % 5) as usize];
        let coeffs: Vec<Coeffs> = v.iter().map(|&f| Coeffs::design(FilterType::BandPass, f64::from(sr), f, f / 90.0, 0.0)).collect();
        let voiced = (syl as f32 * 0.7) as usize;
        let f0a = 100.0 + 70.0 * f64::from(rng.uniform().abs());
        let f0b = 100.0 + 70.0 * f64::from(rng.uniform().abs());
        for i in 0..syl {
            let Some(o) = out.get_mut(start + i) else { break };
            if i < voiced {
                let u = i as f64 / voiced as f64;
                phase = (phase + (f0a + (f0b - f0a) * u) / f64::from(sr)).fract();
                let src = 1.0 - 2.0 * phase;
                let mut y = 0.0;
                for (k, (st, c)) in filt.iter_mut().zip(&coeffs).enumerate() {
                    y += st.tick(c, src) * [1.0, 0.6, 0.35][k];
                }
                *o = (0.5 * y * (std::f64::consts::PI * u).sin().powf(0.6)) as f32;
            } else if s % 3 == 1 {
                let u = (i - voiced) as f64 / (syl - voiced) as f64;
                let h = f0s.tick(&hp, f64::from(rng.uniform()));
                let y = f1s.tick(&lp, h);
                *o = (0.15 * y * (std::f64::consts::PI * u).sin()) as f32;
            }
        }
    }
    out
}

/// Scale-invariant SDR (dB) of `est` against `reference`, from `skip` on.
fn si_sdr(est: &[f32], reference: &[f32], skip: usize) -> f64 {
    let (e, s) = (&est[skip..], &reference[skip..]);
    let dot: f64 = e.iter().zip(s).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let ss: f64 = s.iter().map(|b| f64::from(*b) * f64::from(*b)).sum();
    let alpha = dot / ss.max(1e-20);
    let (mut t, mut err) = (0.0f64, 0.0f64);
    for (a, b) in e.iter().zip(s) {
        let x = alpha * f64::from(*b);
        t += x * x;
        err += (f64::from(*a) - x).powi(2);
    }
    10.0 * (t / err.max(1e-20)).log10()
}

fn run(fx: &mut dyn AudioEffect, input: &[Vec<f32>], sizes: &mut dyn FnMut() -> usize) -> Vec<Vec<f32>> {
    let mut out = input.to_vec();
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
fn erb_bands_cover_the_spectrum() {
    let w = erb_widths().unwrap();
    assert_eq!(w.len(), 32);
    assert_eq!(w.iter().sum::<usize>(), 481);
    assert!(w.iter().all(|b| *b >= 2));
    assert!(w.windows(2).all(|p| p[0] <= p[1]), "{w:?}");
}

#[test]
fn random_weights_run_finite_and_independent_of_block_sizes() {
    let x: Vec<f32> = speech_like(SR, 2.0, 1).iter().zip(noise(2, 96_000, 0.1)).map(|(a, b)| a + b).collect();
    let mk = || {
        let mut fx = VoiceIsolation::with_model(SR, 2, Some(random_model()));
        fx.set_param("intensity", 100.0);
        fx.reset();
        fx
    };
    assert_eq!(mk().latency(), LATENCY_48K);
    assert_eq!(mk().model_name(), "DeepFilterNet 3");
    let big = run(&mut mk(), &[x.clone(), x.clone()], &mut || usize::MAX);
    let mut rng = Rng::new(3);
    let small = run(&mut mk(), &[x.clone(), x.clone()], &mut || (rng.next() % 2000) as usize);
    assert_eq!(big, small, "output depends on how the stream is cut");
    assert!(big[0].iter().all(|v| v.is_finite()));
    assert_eq!(big[0], big[1], "identical channels stay identical");
}

#[test]
fn every_stream_keeps_its_own_state() {
    let model = random_model();
    let a: Vec<f32> = speech_like(SR, 1.5, 4);
    let b: Vec<f32> = noise(5, a.len(), 0.3);
    let solo = |x: &[f32]| {
        let mut fx = VoiceIsolation::with_model(SR, 1, Some(model.clone()));
        run(&mut fx, &[x.to_vec()], &mut || 4800)
    };
    let (want_a, want_b) = (solo(&a), solo(&b));
    // the same model, two instances, interleaved chunk by chunk
    let mut fa = VoiceIsolation::with_model(SR, 1, Some(model.clone()));
    let mut fb = VoiceIsolation::with_model(SR, 1, Some(model.clone()));
    let (mut oa, mut ob) = (a.clone(), b.clone());
    for k in 0..a.len().div_ceil(4800) {
        let r = k * 4800..((k + 1) * 4800).min(a.len());
        if k % 2 == 0 {
            fa.process(&mut [&mut oa[r.clone()]]);
            fb.process(&mut [&mut ob[r]]);
        } else {
            fb.process(&mut [&mut ob[r.clone()]]);
            fa.process(&mut [&mut oa[r]]);
        }
    }
    assert_eq!(oa, want_a[0]);
    assert_eq!(ob, want_b[0]);
}

#[test]
fn hostile_samples_do_not_poison_the_stream() {
    let m = Df3::new(weights::random(9)).unwrap();
    let mut s = m.stream(1);
    let mut hop = [[0.0f32; HOP]];
    for v in [f32::NAN, f32::INFINITY, 1e30, -1e30] {
        hop[0].fill(v);
        s.process_hop(&mut hop);
        assert!(hop[0].iter().all(|x| x.is_finite()), "{v}");
    }
    let x = speech_like(SR, 0.5, 6);
    let mut fresh = m.stream(1);
    let mut ok = true;
    for h in x.as_chunks::<HOP>().0.iter().take(30) {
        let mut a = [*h];
        let mut b = a;
        s.process_hop(&mut a);
        fresh.process_hop(&mut b);
        ok &= a[0].iter().all(|v| v.is_finite());
    }
    assert!(ok);
    // channel counts are clamped, empty hop lists are fine
    let mut none: [[f32; HOP]; 0] = [];
    m.stream(0).process_hop(&mut none);
    let _ = m.stream(100_000);
}

#[test]
fn the_model_loads_only_from_the_pinned_file() {
    let dir = std::env::temp_dir().join(format!("filmcraft-voice-load-{}", std::process::id()));
    assert_eq!(crate::load(&dir).err(), Some(crate::VoiceError::NotInstalled));
    let _ = std::fs::create_dir_all(crate::catalog::package_dir(&dir));
    // a file of the pinned size but not a model
    let p = crate::catalog::archive_path(&dir);
    std::fs::write(&p, vec![0u8; crate::catalog::ARCHIVE.size as usize]).unwrap();
    assert!(matches!(crate::load(&dir), Err(crate::VoiceError::Model(_))));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The real model's directory, or `None` to skip.
fn models() -> Option<std::path::PathBuf> {
    std::env::var_os("FILMCRAFT_VOICE_MODELS").map(std::path::PathBuf::from)
}

/// SI-SDR of the real model on speech-like sound, clean and in two kinds of noise. Measured
/// 2026-10-22 (M3 Max): clean 15.4 dB out (the gaps between syllables are gated, which caps every
/// result below), white noise at 0 dB −0.3 → 12.7 dB (+13.0), pink noise and hum at 5 dB 3.5 →
/// 9.3 dB (+5.8). The thresholds leave a margin under those.
#[test]
fn the_real_model_cleans_speech() {
    let Some(dir) = models() else { return };
    let model: Arc<dyn VoiceModel> = Arc::new(crate::load(&dir).unwrap());
    let secs = 8.0;
    let s = speech_like(SR, secs, 41);
    let pow = s.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / s.len() as f64;
    // white noise at 0 dB SNR
    let white: Vec<f32> = noise(42, s.len(), (3.0 * pow).sqrt() as f32);
    // pink-ish noise (a crowd's roar has about the same slope) and mains hum, 5 dB down
    let mut lp = Biquad::new();
    let pink_c = Coeffs::design(FilterType::LowPass, f64::from(SR), 800.0, 0.5, 0.0);
    let raw: Vec<f32> = noise(43, s.len(), 1.0).iter().map(|v| (lp.tick(&pink_c, f64::from(*v)) as f32) + 0.15 * v).collect();
    let rpow = raw.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / s.len() as f64;
    let g = (pow / rpow / 10f64.powf(0.5)).sqrt() as f32;
    let crowd: Vec<f32> = raw.iter().enumerate().map(|(i, v)| g * v + 0.02 * (2.0 * std::f32::consts::PI * 60.0 * i as f32 / SR).sin()).collect();
    let clean = vec![0.0f32; s.len()];
    for (name, nz, min_gain) in [("clean", &clean, 0.0), ("white noise, 0 dB", &white, 10.0), ("pink noise and hum, 5 dB", &crowd, 4.0)] {
        let mix: Vec<f32> = s.iter().zip(nz.iter()).map(|(a, b)| a + b).collect();
        let mut fx = VoiceIsolation::with_model(SR, 1, Some(model.clone()));
        fx.set_param("intensity", 100.0);
        fx.reset();
        let mut ch = vec![mix.clone()];
        let t0 = std::time::Instant::now();
        process_offline(&mut fx, &mut ch);
        let rtf = t0.elapsed().as_secs_f64() / f64::from(secs);
        let skip = 2 * SR as usize;
        let (before, after) = (si_sdr(&mix, &s, skip), si_sdr(&ch[0], &s, skip));
        eprintln!("DeepFilterNet 3, {name}: SI-SDR {before:.1} → {after:.1} dB ({:+.1} dB), real-time factor {rtf:.3}", after - before);
        if name == "clean" {
            // the voice comes through (the gaps between syllables are gated, so not exactly)
            assert!(after >= 12.0, "clean speech came out at {after:.1} dB");
        } else {
            assert!(after - before >= min_gain, "{name}: {before:.1} → {after:.1} dB");
        }
    }
}

/// libDF's output for the same input, frame by frame (`noisy.f32`, `enhanced.f32`: mono 48 kHz
/// little-endian f32, made with DeepFilterNet's `DfTract` at its defaults, no attenuation limit).
#[test]
fn matches_the_reference_implementation() {
    let (Some(dir), Some(oracle)) = (models(), std::env::var_os("FILMCRAFT_VOICE_ORACLE").map(std::path::PathBuf::from)) else { return };
    let read = |n: &str| -> Vec<f32> { std::fs::read(oracle.join(n)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() };
    let (noisy, want) = (read("noisy.f32"), read("enhanced.f32"));
    let m = crate::load(&dir).unwrap();
    let mut s = m.stream(1);
    let mut got = Vec::new();
    for h in noisy.as_chunks::<HOP>().0 {
        let mut a = [*h];
        s.process_hop(&mut a);
        got.extend_from_slice(&a[0]);
    }
    let n = got.len().min(want.len());
    let (mut err, mut sig, mut worst) = (0.0f64, 0.0f64, 0.0f32);
    for i in 0..n {
        err += f64::from(got[i] - want[i]).powi(2);
        sig += f64::from(want[i]).powi(2);
        worst = worst.max((got[i] - want[i]).abs());
    }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    eprintln!("against libDF: {n} samples, difference {snr:.1} dB below the signal, worst sample {worst:.2e}");
    assert!(snr > 40.0, "{snr:.1} dB");
}
