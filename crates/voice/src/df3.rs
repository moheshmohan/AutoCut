//! DeepFilterNet 3 as a [`VoiceModel`]: the feature extraction, the network, the staging by
//! local SNR and the deep filter, frame by frame, as DeepFilterNet's own streaming
//! implementation (libDF, `tract.rs`, MIT/Apache-2.0) runs them, so the output matches it.
//!
//! Per hop: every channel's spectrum (960-point Vorbis STFT, [`HopStft`]); the network sees the
//! channels' mean (mono) spectrum — 32 ERB-band log powers, mean-normalised, and the lowest 96
//! bins, unit-normalised, both with exponential state (τ = 1 s) — and says, for the frame two hops
//! back (its look-ahead), a gain per ERB band and 5-tap complex filters for the lowest 96 bins
//! (over the frames two back to two ahead), and how much speech there is (local SNR):
//!
//! | local SNR | done |
//! |---|---|
//! | below −10 dB | nothing but noise: the frame is silenced |
//! | above 30 dB | clean speech: left alone |
//! | 20–30 dB | the band gains only |
//! | else | the band gains above bin 96, the deep filter below |
//!
//! The decoders run only when their output is used (their state stands still otherwise, as in
//! libDF). After five hops of digital silence the stream outputs silence without running.
//! Every channel gets the same gains and filters, on its own spectrum, so the stereo image stays.

use std::sync::Arc;

use filmcraft_audio_dsp::effects::voice::{BINS, HOP, HopStft, MAX_CHANNELS, VoiceModel, VoiceStream};

use crate::VoiceError;
use crate::net::{DF_ORDER, NB_DF, NB_ERB, NetState, Weights};

/// Exponential normalisation (τ = 1 s at 100 frames a second, rounded as libDF rounds it).
const NORM_ALPHA: f32 = 0.99;
/// Local-SNR thresholds (libDF's defaults).
const MIN_DB: f32 = -10.0;
const MAX_DB_ERB: f32 = 30.0;
const MAX_DB_DF: f32 = 20.0;
/// Hops of (near) digital silence after which the model stops running.
const SILENT_HOPS: usize = 5;
const SILENCE_MS: f32 = 1e-7;
/// Samples beyond this are clipped before analysis (+36 dBFS: nothing real).
const MAX_SAMPLE: f32 = 64.0;
/// Frames kept for the deep filter: two back, the output frame, two ahead.
const RING: usize = DF_ORDER;

/// The ERB band widths (bins per band) for 48 kHz, a 960-point FFT and 32 bands, at least two
/// bins each (libDF's `erb_fb`).
pub fn erb_widths() -> Result<Vec<usize>, VoiceError> {
    let (sr, fft, nb, min) = (48_000.0f32, 960usize, NB_ERB, 2i64);
    let freq2erb = |f: f32| 9.265 * (f / (24.7 * 9.265)).ln_1p();
    let erb2freq = |e: f32| 24.7 * 9.265 * ((e / 9.265).exp() - 1.0);
    let width = sr / fft as f32;
    let (lo, hi) = (freq2erb(0.0), freq2erb(sr / 2.0));
    let step = (hi - lo) / nb as f32;
    let mut out = vec![0i64; nb];
    let (mut prev, mut over) = (0i64, 0i64);
    for (i, o) in out.iter_mut().enumerate() {
        let f = erb2freq(lo + (i + 1) as f32 * step);
        let fb = (f / width).round() as i64;
        let mut n = fb - prev - over;
        if n < min {
            over = min - n;
            n = min;
        } else {
            over = 0;
        }
        *o = n;
        prev = fb;
    }
    if let Some(last) = out.last_mut() {
        *last += 1;
    }
    let bins = (fft / 2 + 1) as i64;
    let extra = out.iter().sum::<i64>() - bins;
    if extra > 0
        && let Some(last) = out.last_mut()
    {
        *last -= extra;
    }
    if out.iter().sum::<i64>() != bins || out.iter().any(|w| *w < 1) {
        return Err(VoiceError::Model("ERB bands do not cover the spectrum".into()));
    }
    Ok(out.into_iter().map(|w| w as usize).collect())
}

/// DeepFilterNet 3, ready to run (the weights are shared by every stream).
pub struct Df3 {
    w: Arc<Weights>,
    erb: Arc<Vec<usize>>,
}

impl Df3 {
    pub fn new(w: Weights) -> Result<Self, VoiceError> {
        Ok(Df3 { w: Arc::new(w), erb: Arc::new(erb_widths()?) })
    }

    /// From the model archive's bytes.
    pub fn from_archive(bytes: &[u8]) -> Result<Self, VoiceError> {
        Self::new(crate::weights::from_archive(bytes)?)
    }

    /// A stream with its own state (for tests and offline use; effects call `new_stream`).
    pub fn stream(&self, channels: usize) -> Df3Stream {
        Df3Stream::new(self.w.clone(), self.erb.clone(), channels.clamp(1, MAX_CHANNELS))
    }
}

impl VoiceModel for Df3 {
    fn name(&self) -> &str {
        "DeepFilterNet 3"
    }
    fn neural(&self) -> bool {
        true
    }
    fn new_stream(&self, channels: usize) -> Box<dyn VoiceStream> {
        Box::new(self.stream(channels))
    }
}

/// One stream's state.
pub struct Df3Stream {
    w: Arc<Weights>,
    erb: Arc<Vec<usize>>,
    net: NetState,
    stft: Vec<HopStft>,
    /// `[frame][channel][bin]` noisy spectra, frame `cur` the newest.
    x_re: Vec<Vec<Vec<f32>>>,
    x_im: Vec<Vec<Vec<f32>>>,
    cur: usize,
    mono_re: Vec<f32>,
    mono_im: Vec<f32>,
    feat_erb: Vec<f32>,
    feat_spec: Vec<f32>,
    mean_norm: Vec<f32>,
    unit_norm: Vec<f32>,
    out_re: Vec<f32>,
    out_im: Vec<f32>,
    skip: usize,
    ok: bool,
    /// The last frame's local SNR (dB).
    pub lsnr: f32,
}

/// The normalisations' initial states (libDF's: −60 … −90 dB over the bands, 0.001 … 0.0001
/// over the bins).
fn init_norms(mean: &mut [f32], unit: &mut [f32]) {
    let n = mean.len().max(2);
    for (i, s) in mean.iter_mut().enumerate() {
        *s = -60.0 + i as f32 * (-90.0 - -60.0) / (n - 1) as f32;
    }
    let n = unit.len().max(2);
    for (i, s) in unit.iter_mut().enumerate() {
        *s = 0.001 + i as f32 * (0.0001 - 0.001) / (n - 1) as f32;
    }
}

impl Df3Stream {
    fn new(w: Arc<Weights>, erb: Arc<Vec<usize>>, ch: usize) -> Self {
        let stft: Vec<HopStft> = (0..ch).filter_map(|_| HopStft::new()).collect();
        let ok = stft.len() == ch;
        let (mut mean_norm, mut unit_norm) = (vec![0.0; NB_ERB], vec![0.0; NB_DF]);
        init_norms(&mut mean_norm, &mut unit_norm);
        Df3Stream {
            w,
            erb,
            net: NetState::new(),
            stft,
            x_re: vec![vec![vec![0.0; BINS]; ch]; RING],
            x_im: vec![vec![vec![0.0; BINS]; ch]; RING],
            cur: 0,
            mono_re: vec![0.0; BINS],
            mono_im: vec![0.0; BINS],
            feat_erb: vec![0.0; NB_ERB],
            feat_spec: vec![0.0; 2 * NB_DF],
            mean_norm,
            unit_norm,
            out_re: vec![0.0; BINS],
            out_im: vec![0.0; BINS],
            skip: 0,
            ok,
            lsnr: 0.0,
        }
    }

    fn features(&mut self) {
        // ERB band powers in dB, mean-normalised
        let mut b0 = 0usize;
        for (band, &w) in self.erb.iter().enumerate() {
            let mut p = 0.0f32;
            for k in b0..(b0 + w).min(BINS) {
                p += self.mono_re[k] * self.mono_re[k] + self.mono_im[k] * self.mono_im[k];
            }
            b0 += w;
            let db = 10.0 * (p / w.max(1) as f32 + 1e-10).log10();
            let s = &mut self.mean_norm[band];
            *s = db * (1.0 - NORM_ALPHA) + *s * NORM_ALPHA;
            self.feat_erb[band] = (db - *s) / 40.0;
        }
        // the lowest bins, unit-normalised
        for k in 0..NB_DF {
            let (re, im) = (self.mono_re[k], self.mono_im[k]);
            let s = &mut self.unit_norm[k];
            *s = (re * re + im * im).sqrt() * (1.0 - NORM_ALPHA) + *s * NORM_ALPHA;
            let d = s.max(1e-20).sqrt();
            self.feat_spec[k] = re / d;
            self.feat_spec[NB_DF + k] = im / d;
        }
    }
}

impl VoiceStream for Df3Stream {
    fn process_hop(&mut self, hops: &mut [[f32; HOP]]) {
        let ch = self.stft.len().min(hops.len());
        if !self.ok || ch == 0 {
            return;
        }
        let mut e = 0.0f32;
        for h in hops.iter_mut().take(ch) {
            for x in h.iter_mut() {
                *x = if x.is_finite() { x.clamp(-MAX_SAMPLE, MAX_SAMPLE) } else { 0.0 };
                e += *x * *x;
            }
        }
        if e / (ch * HOP) as f32 >= SILENCE_MS {
            self.skip = 0;
        } else {
            self.skip += 1;
        }
        if self.skip > SILENT_HOPS {
            hops.iter_mut().for_each(|h| h.fill(0.0));
            return;
        }
        let cur = self.cur;
        for c in 0..ch {
            self.stft[c].analyse(&hops[c], &mut self.x_re[cur][c], &mut self.x_im[cur][c]);
        }
        let inv = 1.0 / ch as f32;
        for k in 0..BINS {
            let (mut r, mut i) = (0.0f32, 0.0f32);
            for c in 0..ch {
                r += self.x_re[cur][c][k];
                i += self.x_im[cur][c][k];
            }
            self.mono_re[k] = r * inv;
            self.mono_im[k] = i * inv;
        }
        self.features();
        let w = self.w.clone();
        let lsnr = self.net.encode(&w, &self.feat_erb, &self.feat_spec);
        if !lsnr.is_finite() {
            // never let a bad frame poison the state for the rest of the stream
            self.reset();
            hops.iter_mut().for_each(|h| h.fill(0.0));
            return;
        }
        self.lsnr = lsnr;
        let (gains, zeros, df) = if lsnr < MIN_DB {
            (false, true, false)
        } else if lsnr > MAX_DB_ERB {
            (false, false, false)
        } else if lsnr > MAX_DB_DF {
            (true, false, false)
        } else {
            (true, false, true)
        };
        if gains {
            self.net.decode_erb(&w);
        }
        if df {
            self.net.decode_df(&w);
        }
        if gains || zeros {
            self.skip = 0;
        } else {
            self.skip += 1;
        }
        let t2 = (cur + RING - 2) % RING;
        for c in 0..ch {
            self.out_re.copy_from_slice(&self.x_re[t2][c]);
            self.out_im.copy_from_slice(&self.x_im[t2][c]);
            if zeros {
                self.out_re.fill(0.0);
                self.out_im.fill(0.0);
            } else if gains {
                let mut b0 = 0usize;
                for (&wd, &g) in self.erb.iter().zip(self.net.mask.iter()) {
                    for k in b0..(b0 + wd).min(BINS) {
                        self.out_re[k] *= g;
                        self.out_im[k] *= g;
                    }
                    b0 += wd;
                }
            }
            if df {
                for f in 0..NB_DF {
                    let (mut r, mut i) = (0.0f32, 0.0f32);
                    for o in 0..DF_ORDER {
                        let slot = (cur + 1 + o) % RING;
                        let (sr, si) = (self.x_re[slot][c][f], self.x_im[slot][c][f]);
                        let (cr, ci) = (self.net.coefs[f * 2 * DF_ORDER + 2 * o], self.net.coefs[f * 2 * DF_ORDER + 2 * o + 1]);
                        r += sr * cr - si * ci;
                        i += sr * ci + si * cr;
                    }
                    self.out_re[f] = r;
                    self.out_im[f] = i;
                }
            }
            let (re, im) = (&self.out_re, &self.out_im);
            self.stft[c].synthesise(re, im, &mut hops[c]);
        }
        self.cur = (cur + 1) % RING;
    }

    fn reset(&mut self) {
        self.net.reset();
        self.stft.iter_mut().for_each(HopStft::reset);
        for f in self.x_re.iter_mut().chain(self.x_im.iter_mut()) {
            f.iter_mut().for_each(|c| c.fill(0.0));
        }
        self.cur = 0;
        init_norms(&mut self.mean_norm, &mut self.unit_norm);
        self.skip = 0;
        self.lsnr = 0.0;
    }
}
