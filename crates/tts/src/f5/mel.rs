//! The mel front end and the vocoder's inverse STFT, as the reference computes them.
//!
//! - [`mel`]: torchaudio's `MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024,
//!   hop_length=256, n_mels=100, power=1, center=True, normalized=False, norm=None)` — a
//!   periodic Hann window, the signal reflection-padded by half a window on each side, the
//!   magnitude of each frame's spectrum through an HTK-scale triangular filterbank (0–12 kHz,
//!   unnormalised) — then `log(max(mel, 1e-5))`. `len / 256 + 1` frames.
//! - [`istft`]: `torch.istft(center=True)` (Vocos' "center" head): each frame's inverse real
//!   FFT windowed and overlap-added, divided by the summed squared window, half a window cut
//!   from each end: `256 · (frames − 1)` samples.
//!
//! Both run on the CPU in double precision (a radix-2 FFT): the mel's quietest bands sit near
//! the 1e-5 floor, where single-precision rounding is a visible share of the value.

use super::{HOP, MELS, N_FFT, RATE};

/// An in-place radix-2 complex FFT of `N_FFT` points in f64.
struct Fft {
    tw: Vec<(f64, f64)>,
    rev: Vec<usize>,
}

impl Fft {
    fn new() -> Fft {
        let bits = N_FFT.trailing_zeros();
        let rev = (0..N_FFT).map(|i| (i.reverse_bits() >> (usize::BITS - bits)) & (N_FFT - 1)).collect();
        let tw = (0..N_FFT / 2)
            .map(|k| {
                let a = -std::f64::consts::TAU * k as f64 / N_FFT as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Fft { tw, rev }
    }

    /// Forward (`e^{-iωt}`) or, with `inverse`, unscaled inverse transform of `re`/`im`.
    fn run(&self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = N_FFT.min(re.len()).min(im.len());
        for i in 0..n {
            let j = self.rev.get(i).copied().unwrap_or(i);
            if j > i && j < n {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = N_FFT / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = self.tw.get(k * stride).copied().unwrap_or((1.0, 0.0));
                    let wi = sign * wi;
                    let (a, b) = (start + k, start + k + half);
                    let (Some(&br), Some(&bi), Some(&ar), Some(&ai)) = (re.get(b), im.get(b), re.get(a), im.get(a)) else { continue };
                    let xr = br * wr - bi * wi;
                    let xi = br * wi + bi * wr;
                    if let (Some(r), Some(i)) = (re.get_mut(b), im.get_mut(b)) {
                        *r = ar - xr;
                        *i = ai - xi;
                    }
                    if let (Some(r), Some(i)) = (re.get_mut(a), im.get_mut(a)) {
                        *r = ar + xr;
                        *i = ai + xi;
                    }
                }
            }
            len <<= 1;
        }
    }
}

/// One-sided bins of an `N_FFT` spectrum.
pub const BINS: usize = N_FFT / 2 + 1;
/// The log floor.
const FLOOR: f32 = 1e-5;

/// Periodic Hann window (`torch.hann_window(1024)`).
pub fn hann() -> Vec<f32> {
    (0..N_FFT).map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos()) as f32).collect()
}

/// torchaudio's `melscale_fbanks(513, 0, 12000, 100, 24000, norm=None, mel_scale="htk")`,
/// `[BINS][MELS]` row-major.
pub fn filterbank() -> Vec<f32> {
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let nyquist = f64::from(RATE / 2);
    let freqs: Vec<f64> = (0..BINS).map(|i| nyquist * i as f64 / (BINS - 1) as f64).collect();
    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(nyquist));
    let pts: Vec<f64> = (0..MELS + 2).map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (MELS + 1) as f64)).collect();
    let mut fb = vec![0f32; BINS * MELS];
    for (k, f) in freqs.iter().enumerate() {
        for m in 0..MELS {
            let (Some(lo), Some(c), Some(hi)) = (pts.get(m), pts.get(m + 1), pts.get(m + 2)) else { continue };
            let down = (f - lo) / (c - lo);
            let up = (hi - f) / (hi - c);
            if let Some(v) = fb.get_mut(k * MELS + m) {
                *v = down.min(up).max(0.0) as f32;
            }
        }
    }
    fb
}

/// Reflect `i` into `0..n` (numpy/torch "reflect": the edge sample is not repeated).
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

/// Log-mel frames of `x` (mono at 24 kHz), `[frames][MELS]` row-major, with the filterbank
/// `fb` ([`filterbank`]). Returns the frame count too.
pub fn mel(x: &[f32], fb: &[f32]) -> (Vec<f32>, usize) {
    let n = x.len();
    let frames = n / HOP + 1;
    let w = hann();
    let fft = Fft::new();
    let half = (N_FFT / 2) as isize;
    let mut out = vec![0f32; frames * MELS];
    let mut re = vec![0f64; N_FFT];
    let mut im = vec![0f64; N_FFT];
    let mut mag = vec![0f64; BINS];
    for f in 0..frames {
        for (k, (r, i)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
            let at = (f * HOP) as isize + k as isize - half;
            *r = f64::from(x.get(reflect(at, n)).copied().unwrap_or(0.0)) * f64::from(w.get(k).copied().unwrap_or(0.0));
            *i = 0.0;
        }
        fft.run(&mut re, &mut im, false);
        for (b, m) in mag.iter_mut().enumerate() {
            *m = re.get(b).copied().unwrap_or(0.0).hypot(im.get(b).copied().unwrap_or(0.0));
        }
        for m in 0..MELS {
            let mut s = 0f64;
            for (b, v) in mag.iter().enumerate() {
                s += f64::from(fb.get(b * MELS + m).copied().unwrap_or(0.0)) * v;
            }
            if let Some(o) = out.get_mut(f * MELS + m) {
                *o = (s as f32).max(FLOOR).ln();
            }
        }
    }
    (out, frames)
}

/// Inverse STFT of `frames` spectra given as magnitude and phase, `[frames][BINS]` row-major
/// each: `HOP · (frames − 1)` samples.
pub fn istft(mag: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    if frames < 2 {
        return Vec::new();
    }
    let w = hann();
    let fft = Fft::new();
    let total = (frames - 1) * HOP + N_FFT;
    let mut y = vec![0f64; total];
    let mut env = vec![0f64; total];
    let mut re = vec![0f64; N_FFT];
    let mut im = vec![0f64; N_FFT];
    for f in 0..frames {
        // the full Hermitian spectrum; irfft ignores the imaginary parts of DC and Nyquist
        for b in 0..BINS {
            let m = f64::from(mag.get(f * BINS + b).copied().unwrap_or(0.0));
            let p = f64::from(phase.get(f * BINS + b).copied().unwrap_or(0.0));
            let (s, c) = p.sin_cos();
            let (r, i) = if b == 0 || b == BINS - 1 { (m * c, 0.0) } else { (m * c, m * s) };
            if let Some(x) = re.get_mut(b) {
                *x = r;
            }
            if let Some(x) = im.get_mut(b) {
                *x = i;
            }
            if b > 0 && b < BINS - 1 {
                if let Some(x) = re.get_mut(N_FFT - b) {
                    *x = r;
                }
                if let Some(x) = im.get_mut(N_FFT - b) {
                    *x = -i;
                }
            }
        }
        fft.run(&mut re, &mut im, true);
        for k in 0..N_FFT {
            let wk = f64::from(w.get(k).copied().unwrap_or(0.0));
            let v = re.get(k).copied().unwrap_or(0.0) / N_FFT as f64;
            if let Some(o) = y.get_mut(f * HOP + k) {
                *o += v * wk;
            }
            if let Some(e) = env.get_mut(f * HOP + k) {
                *e += wk * wk;
            }
        }
    }
    let half = N_FFT / 2;
    (half..total - half).map(|i| (y.get(i).copied().unwrap_or(0.0) / env.get(i).copied().unwrap_or(1.0).max(1e-11)) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_a_dft() {
        let fft = Fft::new();
        let x: Vec<f64> = (0..N_FFT).map(|i| ((i * 7 % 13) as f64 - 6.0) / 6.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0f64; N_FFT]);
        fft.run(&mut re, &mut im, false);
        for k in [0usize, 1, 5, 100, 511, 512, 700] {
            let (mut r, mut i) = (0f64, 0f64);
            for (t, v) in x.iter().enumerate() {
                let a = -std::f64::consts::TAU * ((k * t) % N_FFT) as f64 / N_FFT as f64;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9, "{k}");
        }
        fft.run(&mut re, &mut im, true);
        assert!(re.iter().zip(&x).all(|(a, b)| (a / N_FFT as f64 - b).abs() < 1e-12));
    }

    #[test]
    fn filterbank_is_triangular_and_covers_the_band() {
        let fb = filterbank();
        assert_eq!(fb.len(), BINS * MELS);
        // every filter has some weight, peaks at most 1, and DC is in none
        for m in 0..MELS {
            let col: Vec<f32> = (0..BINS).map(|k| fb[k * MELS + m]).collect();
            let peak = col.iter().copied().fold(0f32, f32::max);
            assert!(peak > 0.0 && peak <= 1.0 + 1e-6, "filter {m}: {peak}");
        }
        assert!((0..MELS).all(|m| fb[m] == 0.0));
    }

    #[test]
    fn mel_frames_and_floor() {
        let fb = filterbank();
        let (m, frames) = mel(&vec![0.0; 24_000], &fb);
        assert_eq!(frames, 24_000 / HOP + 1);
        assert!(m.iter().all(|v| (*v - FLOOR.ln()).abs() < 1e-6), "silence sits on the floor");
        // a 1 kHz tone lights the band around 1 kHz most
        let x: Vec<f32> = (0..24_000).map(|i| (std::f32::consts::TAU * 1000.0 * i as f32 / 24_000.0).sin() * 0.5).collect();
        let (m, _) = mel(&x, &fb);
        let row = &m[50 * MELS..51 * MELS];
        let best = (0..MELS).max_by(|a, b| row[*a].total_cmp(&row[*b])).unwrap();
        let centre = |i: usize| 700.0 * (10f64.powf((2595.0 * (1.0f64 + 12000.0 / 700.0).log10()) * i as f64 / 101.0 / 2595.0) - 1.0);
        assert!((centre(best + 1) - 1000.0).abs() < 150.0, "{best}: {}", centre(best + 1));
        assert!(mel(&[], &fb).1 == 1);
    }

    /// The inverse STFT of a signal's own STFT gives the signal back (away from the edges).
    #[test]
    fn istft_inverts_the_stft() {
        let n = 24_000;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.4 + (i as f32 * 0.01).sin() * 0.3).collect();
        let w = hann();
        let fft = Fft::new();
        let frames = n / HOP + 1;
        let (mut mag, mut ph) = (vec![0f32; frames * BINS], vec![0f32; frames * BINS]);
        for f in 0..frames {
            let mut re: Vec<f64> = (0..N_FFT).map(|k| f64::from(x[reflect((f * HOP + k) as isize - 512, n)] * w[k])).collect();
            let mut im = vec![0f64; N_FFT];
            fft.run(&mut re, &mut im, false);
            for b in 0..BINS {
                mag[f * BINS + b] = re[b].hypot(im[b]) as f32;
                ph[f * BINS + b] = im[b].atan2(re[b]) as f32;
            }
        }
        let y = istft(&mag, &ph, frames);
        assert_eq!(y.len(), (frames - 1) * HOP);
        let d = (0..y.len().min(n)).map(|i| (y[i] - x[i]).abs()).fold(0f32, f32::max);
        assert!(d < 1e-4, "{d}");
        assert!(istft(&[], &[], 0).is_empty());
        assert!(istft(&[1.0; BINS], &[0.0; BINS], 1).is_empty());
    }
}
