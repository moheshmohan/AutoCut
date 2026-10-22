//! Minimal in-place radix-2 complex FFT (power-of-two sizes), allocation-free after planning.
//!
//! Kept in-crate (rather than depending on an FFT library) so the crate stays dependency-free;
//! the STFT effects only need a few fixed power-of-two sizes.

use std::f64::consts::PI;

/// A planned FFT of fixed power-of-two size.
#[derive(Clone, Debug)]
pub struct Fft {
    n: usize,
    /// e^{-2πik/n} for k in 0..n/2.
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
    rev: Vec<u32>,
}

impl Fft {
    /// Plan an FFT of size `n` (must be a power of two ≥ 2).
    pub fn new(n: usize) -> Fft {
        assert!(n >= 2 && n.is_power_of_two(), "FFT size must be a power of two");
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let (tw_re, tw_im) = (0..n / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .unzip();
        Fft { n, tw_re, tw_im, rev }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Forward transform (e^{-iωt}), unscaled.
    pub fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        self.run(re, im, false);
    }

    /// Inverse transform (e^{+iωt}), unscaled (divide by `n` yourself).
    pub fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        self.run(re, im, true);
    }

    fn run(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        assert!(re.len() >= n && im.len() >= n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            let mut start = 0;
            while start < n {
                for k in 0..half {
                    let wr = self.tw_re[k * stride];
                    let wi = sign * self.tw_im[k * stride];
                    let a = start + k;
                    let b = a + half;
                    let xr = re[b] * wr - im[b] * wi;
                    let xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
                start += len;
            }
            len <<= 1;
        }
    }
}

/// Largest size [`MixedFft`] plans (its twiddle table is `n` long).
pub const MAX_MIXED_FFT: usize = 1 << 16;

/// A planned complex FFT of any size whose prime factors are 2, 3 and 5 (960, the 20 ms frame of
/// 48 kHz speech models, is 2⁶·3·5): recursive decimation in time, in place on the caller's
/// buffers with one scratch buffer, allocation-free after planning.
#[derive(Clone, Debug)]
pub struct MixedFft {
    n: usize,
    /// Radices, outermost first.
    factors: Vec<usize>,
    /// e^{-2πik/n} for k in 0..n.
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
}

impl MixedFft {
    /// Plan a transform of size `n`; `None` when `n` is 0, above [`MAX_MIXED_FFT`], or has a
    /// prime factor other than 2, 3 and 5.
    pub fn new(n: usize) -> Option<MixedFft> {
        if n == 0 || n > MAX_MIXED_FFT {
            return None;
        }
        let mut factors = Vec::new();
        let mut m = n;
        for p in [4usize, 2, 3, 5] {
            while m.is_multiple_of(p) {
                factors.push(p);
                m /= p;
            }
        }
        if m != 1 {
            return None;
        }
        let (tw_re, tw_im) = (0..n)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .unzip();
        Some(MixedFft { n, factors, tw_re, tw_im })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Forward transform (e^{-iωt}), unscaled: `re`/`im` (at least `n` long) are replaced by
    /// their spectrum. `scratch_re`/`scratch_im` (at least `n` long) are overwritten. Shorter
    /// buffers leave everything untouched.
    pub fn forward(&self, re: &mut [f32], im: &mut [f32], scratch_re: &mut [f32], scratch_im: &mut [f32]) {
        self.run(re, im, scratch_re, scratch_im, false);
    }

    /// Inverse transform (e^{+iωt}), unscaled (divide by `n` yourself).
    pub fn inverse(&self, re: &mut [f32], im: &mut [f32], scratch_re: &mut [f32], scratch_im: &mut [f32]) {
        self.run(re, im, scratch_re, scratch_im, true);
    }

    fn run(&self, re: &mut [f32], im: &mut [f32], sre: &mut [f32], sim: &mut [f32], inverse: bool) {
        let n = self.n;
        if re.len() < n || im.len() < n || sre.len() < n || sim.len() < n {
            return;
        }
        sre[..n].copy_from_slice(&re[..n]);
        sim[..n].copy_from_slice(&im[..n]);
        let sign = if inverse { -1.0 } else { 1.0 };
        self.rec(&sre[..n], &sim[..n], 0, 1, n, &mut re[..n], &mut im[..n], 0, sign);
    }

    /// Transform the `n` inputs at `base + i·stride` into `out` (contiguous, `n` long).
    #[allow(clippy::too_many_arguments)]
    fn rec(&self, ire: &[f32], iim: &[f32], base: usize, stride: usize, n: usize, ore: &mut [f32], oim: &mut [f32], level: usize, sign: f32) {
        if n == 1 {
            if let (Some(r), Some(i), Some(o), Some(p)) = (ire.get(base), iim.get(base), ore.first_mut(), oim.first_mut()) {
                *o = *r;
                *p = *i;
            }
            return;
        }
        let Some(&p) = self.factors.get(level) else { return };
        let m = n / p;
        for q in 0..p {
            let (r0, r1) = (q * m, (q + 1) * m);
            if let (Some(or), Some(oi)) = (ore.get_mut(r0..r1), oim.get_mut(r0..r1)) {
                self.rec(ire, iim, base + q * stride, stride * p, m, or, oi, level + 1, sign);
            }
        }
        // twiddle step in the full table: W_n^e = W_N^{e·N/n}
        let step = self.n / n;
        let (mut xr, mut xi) = ([0.0f32; 5], [0.0f32; 5]);
        for k in 0..m {
            for q in 0..p {
                let e = (q * k * step) % self.n;
                let (wr, wi) = (self.tw_re[e], sign * self.tw_im[e]);
                let (a, b) = (ore[q * m + k], oim[q * m + k]);
                xr[q] = a * wr - b * wi;
                xi[q] = a * wi + b * wr;
            }
            for r in 0..p {
                let (mut sr, mut si) = (0.0f32, 0.0f32);
                for q in 0..p {
                    let e = (q * r * m * step) % self.n;
                    let (wr, wi) = (self.tw_re[e], sign * self.tw_im[e]);
                    sr += xr[q] * wr - xi[q] * wi;
                    si += xr[q] * wi + xi[q] * wr;
                }
                ore[k + r * m] = sr;
                oim[k + r * m] = si;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_radix_matches_naive_dft_and_round_trips() {
        for n in [1usize, 2, 3, 5, 12, 60, 96, 480, 960] {
            let fft = MixedFft::new(n).unwrap();
            let x: Vec<(f32, f32)> = (0..n).map(|i| (((i * 7 % 13) as f32 - 6.0) / 6.0, ((i * 5 % 11) as f32 - 5.0) / 7.0)).collect();
            let (mut re, mut im): (Vec<f32>, Vec<f32>) = x.iter().copied().unzip();
            let (mut sr, mut si) = (vec![0.0; n], vec![0.0; n]);
            fft.forward(&mut re, &mut im, &mut sr, &mut si);
            for k in 0..n {
                let (mut r, mut i) = (0.0f64, 0.0f64);
                for (t, &(a, b)) in x.iter().enumerate() {
                    let w = -2.0 * PI * ((k * t) % n) as f64 / n as f64;
                    r += a as f64 * w.cos() - b as f64 * w.sin();
                    i += a as f64 * w.sin() + b as f64 * w.cos();
                }
                assert!((re[k] as f64 - r).abs() < 1e-3 * (n as f64).sqrt(), "n {n} re {k}: {} vs {r}", re[k]);
                assert!((im[k] as f64 - i).abs() < 1e-3 * (n as f64).sqrt(), "n {n} im {k}");
            }
            fft.inverse(&mut re, &mut im, &mut sr, &mut si);
            for (t, &(a, b)) in x.iter().enumerate() {
                assert!((re[t] / n as f32 - a).abs() < 1e-4, "n {n} t {t}");
                assert!((im[t] / n as f32 - b).abs() < 1e-4);
            }
        }
        assert!(MixedFft::new(0).is_none());
        assert!(MixedFft::new(7 * 64).is_none());
        assert!(MixedFft::new(MAX_MIXED_FFT * 2).is_none());
        // short buffers are refused, not indexed out of bounds
        let f = MixedFft::new(960).unwrap();
        let (mut a, mut b, mut c, mut d) = (vec![1.0; 10], vec![0.0; 10], vec![0.0; 960], vec![0.0; 960]);
        f.forward(&mut a, &mut b, &mut c, &mut d);
        assert_eq!(a, vec![1.0; 10]);
    }

    #[test]
    fn matches_naive_dft_and_round_trips() {
        let n = 64;
        let fft = Fft::new(n);
        let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();
        let mut re = x.clone();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut i) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                r += v as f64 * a.cos();
                i += v as f64 * a.sin();
            }
            assert!((re[k] as f64 - r).abs() < 1e-4, "re {k}");
            assert!((im[k] as f64 - i).abs() < 1e-4, "im {k}");
        }
        fft.inverse(&mut re, &mut im);
        for t in 0..n {
            assert!((re[t] / n as f32 - x[t]).abs() < 1e-5);
            assert!((im[t] / n as f32).abs() < 1e-5);
        }
    }
}
