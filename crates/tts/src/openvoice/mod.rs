//! Voice cloning, part 1: the OpenVoice V2 tone-colour converter (feature `kokoro`), pure-Rust
//! inference on candle — Metal on macOS ([`device`]), the CPU elsewhere and as the fallback.
//!
//! A speaker's *tone colour* is a 256-d embedding of their timbre, read from a few seconds of
//! their speech by [`Converter::tone_colour`]. [`Converter::convert`] re-voices any speech from
//! one tone colour to another while keeping its words, timing and intonation. Cloned narration
//! ([`crate::clone`]) is Kokoro speech converted from the Kokoro voice's colour to the user's.
//!
//! Implemented from the OpenVoice paper (arXiv 2312.01479), its MIT-licensed reference code
//! (VITS modules), the released `converter/config.json` and the checkpoint's tensor names and
//! shapes. Weights (MIT, MyShell) are downloaded on first use and never bundled.

mod model;

use std::path::Path;

use candle_core::Tensor;

use crate::TtsError;
use crate::kokoro::weights::{Weights, err};

/// The converter's sample rate.
pub const RATE: u32 = 22_050;
const N_FFT: usize = 1024;
const HOP: usize = model::HOP;
const BINS: usize = model::SPEC;
/// Size of a tone colour.
pub const COLOUR: usize = model::GIN;
/// Shortest reference that yields a tone colour (the encoder halves time six times).
pub const MIN_REFERENCE_SECONDS: f64 = 1.0;
/// Longest stretch converted in one pass; longer audio is converted in overlapping windows.
const WINDOW_SECONDS: f64 = 20.0;
const OVERLAP_SECONDS: f64 = 0.5;
/// Reference audio is embedded in pieces of about this length and averaged (as OpenVoice does).
const PIECE_SECONDS: f64 = 10.0;
/// Noise scale of the posterior sample (OpenVoice's default for conversion).
pub const DEFAULT_TAU: f64 = 0.3;

/// Values of `config.json` this implementation is written for.
const EXPECT: &[(&str, &str)] = &[
    ("data.sampling_rate", "22050"),
    ("data.filter_length", "1024"),
    ("data.hop_length", "256"),
    ("data.win_length", "1024"),
    ("model.zero_g", "true"),
    ("model.inter_channels", "192"),
    ("model.hidden_channels", "192"),
    ("model.resblock", "\"1\""),
    ("model.resblock_kernel_sizes", "[3,7,11]"),
    ("model.upsample_rates", "[8,8,2,2]"),
    ("model.upsample_initial_channel", "512"),
    ("model.upsample_kernel_sizes", "[16,16,4,4]"),
    ("model.gin_channels", "256"),
];

/// A loaded converter.
pub struct Converter {
    /// On [`device`] (the GPU where there is one).
    model: model::Model,
    dir: std::path::PathBuf,
    /// The GPU failed once (an error, a panic, samples that are not finite): the CPU from then on.
    gpu_failed: std::sync::atomic::AtomicBool,
    /// The model on the CPU, loaded when the GPU failed.
    cpu: std::sync::Mutex<Option<std::sync::Arc<model::Model>>>,
}

/// The device the converter runs on: as Kokoro's ([`crate::kokoro::device`]: Metal on macOS,
/// else the CPU); `FILMCRAFT_OPENVOICE_CPU` keeps the CPU.
pub fn device() -> candle_core::Device {
    if std::env::var_os("FILMCRAFT_OPENVOICE_CPU").is_some() {
        return candle_core::Device::Cpu;
    }
    crate::kokoro::device()
}

impl Converter {
    /// Load `config.json` and `checkpoint.pth` from `dir`, on [`device`] (the GPU where there is
    /// one; weights it cannot hold leave the converter on the CPU).
    pub fn load(dir: &Path) -> Result<Converter, TtsError> {
        Self::load_on(dir, &device())
    }

    /// [`Converter::load`] on `dev`.
    pub fn load_on(dir: &Path, dev: &candle_core::Device) -> Result<Converter, TtsError> {
        check_config(&dir.join("config.json"))?;
        let w = Weights::load_key(&dir.join("checkpoint.pth"), "model")?;
        let model = match w.to_device(dev).and_then(|g| model::Model::load(&g)) {
            Ok(m) => m,
            Err(_) => model::Model::load(&w)?,
        };
        Ok(Converter { model, dir: dir.to_path_buf(), gpu_failed: std::sync::atomic::AtomicBool::new(false), cpu: std::sync::Mutex::new(None) })
    }

    /// Whether conversion runs on the GPU now.
    pub fn on_gpu(&self) -> bool {
        !self.model.device().is_cpu() && !self.gpu_failed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `f` with the GPU model, again with the CPU's (loaded then, kept from then on) when that
    /// fails or panics; a refused input is refused on either.
    fn with_model<T>(&self, f: impl Fn(&model::Model) -> Result<T, TtsError>) -> Result<T, TtsError> {
        if self.model.device().is_cpu() {
            return f(&self.model);
        }
        if self.on_gpu() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&self.model))) {
                Ok(Ok(v)) => return Ok(v),
                Ok(Err(TtsError::Invalid(e))) => return Err(TtsError::Invalid(e)),
                _ => self.gpu_failed.store(true, std::sync::atomic::Ordering::Relaxed),
            }
        }
        let cpu = {
            let mut g = self.cpu.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match g.as_ref() {
                Some(m) => m.clone(),
                None => {
                    let m = std::sync::Arc::new(model::Model::load(&Weights::load_key(&self.dir.join("checkpoint.pth"), "model")?)?);
                    *g = Some(m.clone());
                    m
                }
            }
        };
        f(&cpu)
    }

    /// The tone colour of mono speech at [`RATE`]: the mean over ~10 s pieces.
    pub fn tone_colour(&self, speech: &[f32]) -> Result<Vec<f32>, TtsError> {
        let min = (MIN_REFERENCE_SECONDS * f64::from(RATE)) as usize;
        if speech.len() < min {
            return Err(TtsError::Invalid(format!("at least {MIN_REFERENCE_SECONDS} s of speech is needed")));
        }
        self.with_model(|m| Self::tone_colour_with(m, speech))
    }

    fn tone_colour_with(m: &model::Model, speech: &[f32]) -> Result<Vec<f32>, TtsError> {
        let min = (MIN_REFERENCE_SECONDS * f64::from(RATE)) as usize;
        let piece = (PIECE_SECONDS * f64::from(RATE)) as usize;
        let mut sum = vec![0f64; COLOUR];
        let mut count = 0usize;
        for chunk in speech.chunks(piece) {
            if chunk.len() < min {
                continue;
            }
            let (spec, frames) = spectrogram(chunk);
            let t = Tensor::from_vec(spec, (1, BINS, frames), m.device()).map_err(err)?;
            let c = m.tone_colour(&t)?;
            for (s, v) in sum.iter_mut().zip(&c) {
                *s += f64::from(*v);
            }
            count += 1;
        }
        if count == 0 {
            return Err(TtsError::Invalid("the reference audio is too short".into()));
        }
        let out: Vec<f32> = sum.iter().map(|s| (*s / count as f64) as f32).collect();
        if out.iter().any(|x| !x.is_finite()) {
            return Err(TtsError::Model("the tone colour is not finite".into()));
        }
        Ok(out)
    }

    /// Re-voice mono speech at [`RATE`] from tone colour `src` to `tgt`. Same length as the input.
    pub fn convert(&self, speech: &[f32], src: &[f32], tgt: &[f32], tau: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
        if src.len() != COLOUR || tgt.len() != COLOUR {
            return Err(TtsError::Invalid(format!("tone colours have {COLOUR} values")));
        }
        let tau = if tau.is_finite() { tau.clamp(0.0, 1.0) } else { DEFAULT_TAU };
        if speech.len() < N_FFT {
            return Ok(speech.to_vec());
        }
        self.with_model(|m| Self::convert_with(m, speech, src, tgt, tau, seed))
    }

    fn convert_with(m: &model::Model, speech: &[f32], src: &[f32], tgt: &[f32], tau: f64, seed: u64) -> Result<Vec<f32>, TtsError> {
        let n = speech.len();
        let window = (WINDOW_SECONDS * f64::from(RATE)) as usize;
        let overlap = (OVERLAP_SECONDS * f64::from(RATE)) as usize;
        let mut out = vec![0f32; n];
        let mut rng = crate::kokoro::dsp::Rng::new(seed);
        let mut start = 0usize;
        while start < n {
            let end = (start + window).min(n);
            let piece = speech.get(start..end).unwrap_or(&[]);
            let y = Self::convert_piece(m, piece, src, tgt, tau, &mut rng)?;
            // crossfade into what the previous window wrote
            for (i, v) in y.iter().enumerate() {
                let k = start + i;
                let Some(o) = out.get_mut(k) else { break };
                if start > 0 && i < overlap {
                    let a = i as f32 / overlap as f32;
                    *o = *o * (1.0 - a) + v * a;
                } else {
                    *o = *v;
                }
            }
            if end == n {
                break;
            }
            start = end.saturating_sub(overlap);
        }
        Ok(out)
    }

    fn convert_piece(m: &model::Model, piece: &[f32], src: &[f32], tgt: &[f32], tau: f64, rng: &mut crate::kokoro::dsp::Rng) -> Result<Vec<f32>, TtsError> {
        if piece.len() < N_FFT {
            return Ok(piece.to_vec());
        }
        let (spec, frames) = spectrogram(piece);
        let dev = m.device();
        let t = Tensor::from_vec(spec, (1, BINS, frames), dev).map_err(err)?;
        let noise: Vec<f32> = (0..192 * frames).map(|_| rng.normal() as f32).collect();
        let noise = Tensor::from_vec(noise, (1, 192, frames), dev).map_err(err)?;
        let mut y = m.convert(&t, src, tgt, tau, &noise)?;
        y.resize(piece.len(), 0.0);
        if y.iter().any(|x| !x.is_finite()) {
            return Err(TtsError::Model("the converted audio is not finite".into()));
        }
        Ok(y)
    }
}

fn check_config(path: &Path) -> Result<(), TtsError> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    if bytes.len() > 1 << 20 {
        return Err(err("config.json is too large"));
    }
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(err)?;
    for (key, want) in EXPECT {
        let mut cur = &v;
        for part in key.split('.') {
            cur = cur.get(part).ok_or_else(|| err(format!("config.json: missing `{key}`")))?;
        }
        let got = serde_json::to_string(cur).map_err(err)?;
        if got != *want {
            return Err(err(format!("config.json: `{key}` is {got}, this build supports {want}")));
        }
    }
    Ok(())
}

fn hann() -> Vec<f64> {
    (0..N_FFT).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos()).collect()
}

fn reflect(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut j = i.rem_euclid(period);
    if j >= n as isize {
        j = period - j;
    }
    j as usize
}

/// Magnitude spectrogram as VITS computes it: reflect-pad (n_fft − hop)/2 each side, Hann
/// window, no centring; `sqrt(re² + im² + 1e-6)`. Returns `[bins × frames]` row-major by bin.
pub(crate) fn spectrogram(x: &[f32]) -> (Vec<f32>, usize) {
    let pad = (N_FFT - HOP) / 2;
    let padded = x.len() + 2 * pad;
    let frames = if padded >= N_FFT { (padded - N_FFT) / HOP + 1 } else { 0 };
    let mut spec = vec![0f32; BINS * frames];
    let w = hann();
    let fft = filmcraft_audio_dsp::fft::Fft::new(N_FFT);
    let mut re = vec![0f32; N_FFT];
    let mut im = vec![0f32; N_FFT];
    for f in 0..frames {
        let start = (f * HOP) as isize - pad as isize;
        for (k, (r, iv)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
            let idx = reflect(start + k as isize, x.len());
            *r = (f64::from(x.get(idx).copied().unwrap_or(0.0)) * w.get(k).copied().unwrap_or(0.0)) as f32;
            *iv = 0.0;
        }
        fft.forward(&mut re, &mut im);
        for b in 0..BINS {
            let (a, c) = (re.get(b).copied().unwrap_or(0.0), im.get(b).copied().unwrap_or(0.0));
            if let Some(s) = spec.get_mut(b * frames + f) {
                *s = (a * a + c * c + 1e-6).sqrt();
            }
        }
    }
    (spec, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrogram_frames_and_peak_bin() {
        // 1 s of 1 kHz at 22.05 kHz: frames = (n + 768 − 1024) / 256 + 1
        let n = RATE as usize;
        let x: Vec<f32> = (0..n).map(|i| (std::f64::consts::TAU * 1000.0 * i as f64 / f64::from(RATE)).sin() as f32).collect();
        let (spec, frames) = spectrogram(&x);
        assert_eq!(frames, (n + 768 - 1024) / 256 + 1);
        assert_eq!(spec.len(), BINS * frames);
        // the loudest bin of a middle frame is at 1000 / (22050 / 1024) ≈ 46
        let f = frames / 2;
        let best = (0..BINS).max_by(|a, b| spec[a * frames + f].total_cmp(&spec[b * frames + f])).unwrap();
        assert!((45..=47).contains(&best), "peak bin {best}");
        assert!(spec.iter().all(|v| v.is_finite() && *v >= 0.0));
        // shorter than one window even padded: no frames
        assert_eq!(spectrogram(&[0.0; 10]).1, 0);
        assert_eq!(spectrogram(&[]).1, 0);
    }

    #[test]
    fn config_is_checked() {
        let d = std::env::temp_dir().join(format!("filmcraft-ov-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        let p = d.join("config.json");
        std::fs::write(&p, br#"{"data":{"sampling_rate":44100}}"#).unwrap();
        assert!(check_config(&p).is_err());
        assert!(check_config(&d.join("missing.json")).is_err());
    }

    /// A real round trip with the downloaded weights, gated:
    /// FILMCRAFT_OPENVOICE_DIR=<dir with config.json + checkpoint.pth> cargo test -p filmcraft-tts
    /// --features kokoro live_openvoice -- --nocapture
    #[test]
    fn live_openvoice_round_trip() {
        let Some(dir) = std::env::var_os("FILMCRAFT_OPENVOICE_DIR") else {
            eprintln!("skipped (FILMCRAFT_OPENVOICE_DIR unset)");
            return;
        };
        let t0 = std::time::Instant::now();
        let c = Converter::load(Path::new(&dir)).unwrap();
        eprintln!("loaded in {:?}", t0.elapsed());
        // a buzzy vowel-ish source: harmonics of 120 Hz with a slow vibrato
        let n = 3 * RATE as usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                let f0 = 120.0 + 5.0 * (t * 6.0).sin();
                (1..20).map(|h| (std::f64::consts::TAU * f0 * h as f64 * t).sin() / h as f64).sum::<f64>() as f32 * 0.2
            })
            .collect();
        let t1 = std::time::Instant::now();
        let colour = c.tone_colour(&x).unwrap();
        eprintln!("tone colour in {:?}: |c| = {:.3}", t1.elapsed(), colour.iter().map(|v| v * v).sum::<f32>().sqrt());
        assert_eq!(colour.len(), COLOUR);
        let t2 = std::time::Instant::now();
        let y = c.convert(&x, &colour, &colour, DEFAULT_TAU, 1).unwrap();
        eprintln!("converted 3 s in {:?}", t2.elapsed());
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let rms = (y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / y.len() as f64).sqrt();
        assert!(rms > 1e-3, "not silent: {rms}");
    }

    /// The converter on the GPU gives the CPU's tone colour and conversion within float noise
    /// (SNR printed with both times). Gated like [`live_openvoice_round_trip`]; skipped without
    /// a GPU.
    #[test]
    fn live_openvoice_gpu_matches_the_cpu() {
        let Some(dir) = std::env::var_os("FILMCRAFT_OPENVOICE_DIR") else {
            eprintln!("skipped (FILMCRAFT_OPENVOICE_DIR unset)");
            return;
        };
        let gpu_dev = device();
        if gpu_dev.is_cpu() {
            eprintln!("skipped (no GPU)");
            return;
        }
        let cpu = Converter::load_on(Path::new(&dir), &candle_core::Device::Cpu).unwrap();
        let gpu = Converter::load_on(Path::new(&dir), &gpu_dev).unwrap();
        assert!(gpu.on_gpu() && !cpu.on_gpu());
        let n = 12 * RATE as usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                let f0 = 140.0 + 25.0 * (t * 1.3).sin();
                (1..24).map(|h| (std::f64::consts::TAU * f0 * h as f64 * t).sin() / h as f64).sum::<f64>() as f32 * 0.15
            })
            .collect();
        let _ = gpu.convert(&x[..RATE as usize * 2], &[0.1; COLOUR], &[0.1; COLOUR], DEFAULT_TAU, 1).unwrap();
        let (ca, cb) = (cpu.tone_colour(&x).unwrap(), gpu.tone_colour(&x).unwrap());
        let worst = ca.iter().zip(&cb).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-3, "tone colours differ by {worst}");
        let t0 = std::time::Instant::now();
        let a = cpu.convert(&x, &ca, &[0.05; COLOUR], DEFAULT_TAU, 7).unwrap();
        let t1 = std::time::Instant::now();
        let b = gpu.convert(&x, &ca, &[0.05; COLOUR], DEFAULT_TAU, 7).unwrap();
        let t2 = std::time::Instant::now();
        assert!(gpu.on_gpu(), "the GPU did not fail");
        let signal: f64 = a.iter().map(|v| f64::from(*v).powi(2)).sum();
        let noise: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        eprintln!(
            "OpenVoice 12 s: CPU {:.2}s, GPU {:.2}s; GPU vs CPU SNR {snr:.1} dB, tone colour within {worst:.1e}",
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64()
        );
        assert!(snr > 30.0, "SNR {snr}");
    }
}
