//! The log-mel filterbank the ECAPA-TDNN speaker model was trained on (SpeechBrain's `Fbank`,
//! Apache-2.0, as its `hyperparams.yaml` configures it: 80 bands), computed here from that
//! description:
//!
//! - 16 kHz audio, 400-sample periodic Hamming window, hop 160, centred frames (200 zeros of
//!   padding at both ends): `n` samples give `n / 160 + 1` frames;
//! - power spectrum (201 bins);
//! - 80 triangular filters on the HTK mel scale (`2595 · log10(1 + f / 700)`) between 0 and
//!   8 kHz: filter `m` peaks at the `m + 1`-th of 82 evenly spaced mel points and falls to zero
//!   one band (the distance from the point before it) either side;
//! - `10 · log10(max(x, 1e-10))`, floored 80 dB under the clip's loudest value;
//! - the mean of each band over the clip subtracted ("sentence" normalisation, no variance).

/// Mel bands.
pub const N_MELS: usize = 80;
const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
/// Longest clip computed at once, samples (10 minutes): callers window longer audio.
pub const MAX_SAMPLES: usize = 16_000 * 600;

fn to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// The filterbank, `BINS × N_MELS` row-major (spectrum bin by band).
pub fn filters() -> Vec<f32> {
    let (lo, hi) = (to_mel(0.0), to_mel(8000.0));
    let hz: Vec<f64> = (0..N_MELS + 2).map(|i| to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
    let mut w = vec![0f32; BINS * N_MELS];
    for m in 0..N_MELS {
        let (centre, band) = (hz[m + 1], hz[m + 1] - hz[m]);
        for k in 0..BINS {
            let f = 8000.0 * k as f64 / (BINS - 1) as f64;
            let slope = (f - centre) / band;
            w[k * N_MELS + m] = (slope + 1.0).min(1.0 - slope).max(0.0) as f32;
        }
    }
    w
}

/// Normalised log-mel features of mono 16 kHz `audio`: `frames × N_MELS` row-major. Audio beyond
/// [`MAX_SAMPLES`] is ignored; non-finite samples count as silence.
pub fn fbank(audio: &[f32]) -> (usize, Vec<f32>) {
    use rayon::prelude::*;
    let audio = audio.get(..audio.len().min(MAX_SAMPLES)).unwrap_or(&[]);
    let n = audio.len();
    let frames = n / HOP + 1;
    let pad = N_FFT / 2;
    let window: Vec<f32> = (0..N_FFT).map(|i| (0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32).collect();
    let mut cos = vec![0f32; BINS * N_FFT];
    let mut sin = vec![0f32; BINS * N_FFT];
    for k in 0..BINS {
        for i in 0..N_FFT {
            let a = 2.0 * std::f64::consts::PI * ((k * i) % N_FFT) as f64 / N_FFT as f64;
            cos[k * N_FFT + i] = a.cos() as f32;
            sin[k * N_FFT + i] = a.sin() as f32;
        }
    }
    let fb = filters();
    let mut out = vec![0f32; frames * N_MELS];
    out.par_chunks_mut(N_MELS).enumerate().for_each(|(f, row)| {
        let mut x = [0f32; N_FFT];
        let base = (f * HOP) as isize - pad as isize;
        for (i, v) in x.iter_mut().enumerate() {
            let j = base + i as isize;
            let s = if j >= 0 { audio.get(j as usize).copied().filter(|v| v.is_finite()).unwrap_or(0.0) } else { 0.0 };
            *v = s * window[i];
        }
        let mut power = [0f32; BINS];
        for (k, p) in power.iter_mut().enumerate() {
            let (c, s) = (&cos[k * N_FFT..(k + 1) * N_FFT], &sin[k * N_FFT..(k + 1) * N_FFT]);
            let (mut re, mut im) = (0f32, 0f32);
            for i in 0..N_FFT {
                re += x[i] * c[i];
                im += x[i] * s[i];
            }
            *p = re * re + im * im;
        }
        for (m, r) in row.iter_mut().enumerate() {
            let e: f32 = (0..BINS).map(|k| power[k] * fb[k * N_MELS + m]).sum();
            *r = 10.0 * e.max(1e-10).log10();
        }
    });
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_finite() {
        out.iter_mut().for_each(|v| *v = v.max(max - 80.0));
    }
    for m in 0..N_MELS {
        let mean = (0..frames).map(|f| out[f * N_MELS + m]).sum::<f32>() / frames.max(1) as f32;
        for f in 0..frames {
            out[f * N_MELS + m] -= mean;
        }
    }
    (frames, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_normalisation() {
        let audio: Vec<f32> = (0..16_000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5).collect();
        let (frames, f) = fbank(&audio);
        assert_eq!(frames, 101);
        assert_eq!(f.len(), frames * N_MELS);
        // every band has mean zero
        for m in 0..N_MELS {
            let mean: f32 = (0..frames).map(|t| f[t * N_MELS + m]).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-3, "{m}: {mean}");
        }
        assert!(f.iter().all(|v| v.is_finite()));
        // empty and hostile audio
        assert_eq!(fbank(&[]).0, 1);
        assert!(fbank(&[f32::NAN; 1000]).1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn filters_cover_the_spectrum() {
        let w = filters();
        for m in 0..N_MELS {
            let s: f32 = (0..BINS).map(|k| w[k * N_MELS + m]).sum();
            assert!(s > 0.0, "band {m} is empty");
        }
        assert!((to_hz(to_mel(1234.0)) - 1234.0).abs() < 1e-6);
    }
}
