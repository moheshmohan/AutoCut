//! Voice Isolation: a speech-enhancement model in the clip effect chain.
//!
//! The model works at 48 kHz in hops of [`HOP`] samples (10 ms) on a 960-point Vorbis-windowed
//! STFT (20 ms) and gives back each hop [`MODEL_DELAY_HOPS`] hops late (the window's overlap and
//! two frames of look-ahead). [`VoiceIsolation`] adds one hop of buffering (so it can take any block
//! size, one sample at a time), resamples from and to the clip's rate when that is not 48 kHz
//! (windowed sinc, the round trip delayed by a whole number of samples), and mixes the dry signal
//! back, delayed to match, by `intensity`: 0 % is the input, 100 % the model's output, and in
//! between the noise is held down by at most `0.6 × intensity` dB. Its latency is reported for
//! delay compensation: [`LATENCY_48K`] samples (40 ms) at 48 kHz.
//!
//! **Which model.** This crate has no machine learning in it: the neural model (DeepFilterNet 3,
//! `filmcraft-voice`, layer L2) is installed by the host, process-wide ([`install_voice_model`], or
//! lazily from a loader the host registers with [`set_voice_model_loader`]). Until one is, the
//! effect runs [`SpectralGate`], a DSP noise gate on the same frames with the same latency, so a
//! project sounds the same shape everywhere (the web build, a machine without the download) and
//! only the quality changes. [`VoiceIsolation::model_name`] says which one an instance runs.
//!
//! Every instance has its own model state (GRU memories, STFT buffers, noise estimates); the
//! weights are shared read-only. Rates outside 8–384 kHz make the effect pass audio through
//! untouched with no latency (and [`VoiceIsolation::active`] false) rather than fail.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use crate::fft::MixedFft;
use crate::{AudioEffect, ParamSpec, ParamValues, Smoothed, Unit, block_len, param_plumbing};

/// The model's sample rate.
pub const MODEL_RATE: u32 = 48_000;
/// Samples per hop (10 ms at 48 kHz).
pub const HOP: usize = 480;
/// STFT window (20 ms at 48 kHz).
pub const WINDOW: usize = 2 * HOP;
/// Spectrum bins of one frame.
pub const BINS: usize = WINDOW / 2 + 1;
/// Hops a model's output lags its input: the window's overlap and two frames of look-ahead.
pub const MODEL_DELAY_HOPS: usize = 3;
/// The effect's latency at 48 kHz: the model's delay and one hop of buffering (40 ms).
pub const LATENCY_48K: usize = (MODEL_DELAY_HOPS + 1) * HOP;
/// Sample rates the effect runs at (others pass through).
pub const MIN_RATE: f32 = 8_000.0;
pub const MAX_RATE: f32 = 384_000.0;
/// Most channels one instance processes (more pass through untouched).
pub const MAX_CHANNELS: usize = 32;

/// A speech-isolation model: 48 kHz, hop by hop.
pub trait VoiceModel: Send + Sync {
    /// Shown to the user ("DeepFilterNet 3").
    fn name(&self) -> &str;
    /// A neural model (not the DSP fallback).
    fn neural(&self) -> bool;
    /// Fresh per-stream state for `channels` channels (1..=[`MAX_CHANNELS`]).
    fn new_stream(&self, channels: usize) -> Box<dyn VoiceStream>;
}

/// One stream's model state.
pub trait VoiceStream: Send {
    /// `hops[c]` holds channel `c`'s newest [`HOP`] samples at 48 kHz; on return it holds the
    /// enhanced hop from [`MODEL_DELAY_HOPS`] hops earlier. Never allocates.
    fn process_hop(&mut self, hops: &mut [[f32; HOP]]);
    /// Back to the freshly created state.
    fn reset(&mut self);
}

// ------------------------------------------------------------------------------- the registry

type Loader = Box<dyn Fn() -> Option<Arc<dyn VoiceModel>> + Send + Sync>;

/// Where new instances find their model: one installed model, or a loader run on first use.
pub struct ModelRegistry {
    model: RwLock<Option<Arc<dyn VoiceModel>>>,
    loader: Mutex<Option<Loader>>,
    /// The loader ran and found nothing (not retried until a model or loader is installed).
    tried: AtomicBool,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub const fn new() -> Self {
        ModelRegistry { model: RwLock::new(None), loader: Mutex::new(None), tried: AtomicBool::new(false) }
    }

    /// Install (or with `None`, remove) the model.
    pub fn install(&self, model: Option<Arc<dyn VoiceModel>>) {
        *self.model.write().unwrap_or_else(PoisonError::into_inner) = model;
        self.tried.store(false, Ordering::SeqCst);
    }

    /// Register how to load the model on first use. The loader runs at most once until a model
    /// or another loader is installed; a panic in it is caught and counts as no model.
    pub fn set_loader(&self, loader: impl Fn() -> Option<Arc<dyn VoiceModel>> + Send + Sync + 'static) {
        *self.loader.lock().unwrap_or_else(PoisonError::into_inner) = Some(Box::new(loader));
        self.tried.store(false, Ordering::SeqCst);
    }

    /// The installed model, loading it with the loader the first time.
    pub fn get(&self) -> Option<Arc<dyn VoiceModel>> {
        if let Some(m) = self.model.read().unwrap_or_else(PoisonError::into_inner).clone() {
            return Some(m);
        }
        if self.tried.load(Ordering::SeqCst) {
            return None;
        }
        let guard = self.loader.lock().unwrap_or_else(PoisonError::into_inner);
        // another thread may have loaded it while this one waited
        if let Some(m) = self.model.read().unwrap_or_else(PoisonError::into_inner).clone() {
            return Some(m);
        }
        let loaded = guard.as_ref().and_then(|f| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok().flatten());
        match &loaded {
            Some(m) => *self.model.write().unwrap_or_else(PoisonError::into_inner) = Some(m.clone()),
            None => self.tried.store(true, Ordering::SeqCst),
        }
        loaded
    }
}

/// The process-wide registry the effect registry's constructors read.
static REGISTRY: ModelRegistry = ModelRegistry::new();

/// Install (or with `None`, remove) the process-wide voice model new instances use.
pub fn install_voice_model(model: Option<Arc<dyn VoiceModel>>) {
    REGISTRY.install(model);
}

/// Register how to load the process-wide model on first use (the host knows where it is
/// downloaded); see [`ModelRegistry::set_loader`].
pub fn set_voice_model_loader(loader: impl Fn() -> Option<Arc<dyn VoiceModel>> + Send + Sync + 'static) {
    REGISTRY.set_loader(loader);
}

/// The process-wide voice model, loading it with the registered loader the first time.
pub fn voice_model() -> Option<Arc<dyn VoiceModel>> {
    REGISTRY.get()
}

// ------------------------------------------------------------------------------- the STFT

/// One channel's 960-point Vorbis-window STFT at hop 480 (Princen–Bradley, so analysis then
/// synthesis gives the input back one hop late), scaled like DeepFilterNet's: the forward
/// transform by `1/960`, the inverse unscaled.
#[derive(Clone, Debug)]
pub struct HopStft {
    fft: MixedFft,
    window: Vec<f32>,
    analysis_mem: Vec<f32>,
    synthesis_mem: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    sre: Vec<f32>,
    sim: Vec<f32>,
}

impl HopStft {
    /// `None` only if the FFT cannot be planned (never for 960).
    pub fn new() -> Option<Self> {
        let fft = MixedFft::new(WINDOW)?;
        let half = HOP as f64;
        let window = (0..WINDOW)
            .map(|i| {
                let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / half).sin();
                (0.5 * std::f64::consts::PI * s * s).sin() as f32
            })
            .collect();
        Some(HopStft {
            fft,
            window,
            analysis_mem: vec![0.0; HOP],
            synthesis_mem: vec![0.0; HOP],
            re: vec![0.0; WINDOW],
            im: vec![0.0; WINDOW],
            sre: vec![0.0; WINDOW],
            sim: vec![0.0; WINDOW],
        })
    }

    pub fn reset(&mut self) {
        self.analysis_mem.fill(0.0);
        self.synthesis_mem.fill(0.0);
    }

    /// The spectrum (`BINS` bins into `re`/`im`) of the window ending with `hop`.
    pub fn analyse(&mut self, hop: &[f32; HOP], re: &mut [f32], im: &mut [f32]) {
        for i in 0..HOP {
            self.re[i] = self.analysis_mem[i] * self.window[i];
            self.re[HOP + i] = hop[i] * self.window[HOP + i];
        }
        self.im.fill(0.0);
        self.analysis_mem.copy_from_slice(hop);
        self.fft.forward(&mut self.re, &mut self.im, &mut self.sre, &mut self.sim);
        let norm = 1.0 / WINDOW as f32;
        for (k, (r, i)) in re.iter_mut().zip(im.iter_mut()).enumerate().take(BINS) {
            *r = self.re[k] * norm;
            *i = self.im[k] * norm;
        }
    }

    /// Overlap-add the frame with spectrum `re`/`im` (`BINS` bins): `out` gets the next hop.
    pub fn synthesise(&mut self, re: &[f32], im: &[f32], out: &mut [f32; HOP]) {
        if re.len() < BINS || im.len() < BINS {
            out.fill(0.0);
            return;
        }
        // Hermitian spectrum; DC and Nyquist are real
        self.re[0] = re[0];
        self.im[0] = 0.0;
        self.re[WINDOW / 2] = re[WINDOW / 2];
        self.im[WINDOW / 2] = 0.0;
        for k in 1..WINDOW / 2 {
            self.re[k] = re[k];
            self.im[k] = im[k];
            self.re[WINDOW - k] = re[k];
            self.im[WINDOW - k] = -im[k];
        }
        self.fft.inverse(&mut self.re, &mut self.im, &mut self.sre, &mut self.sim);
        for i in 0..HOP {
            let x = self.re[i] * self.window[i];
            let y = if x.is_finite() { x + self.synthesis_mem[i] } else { self.synthesis_mem[i] };
            out[i] = crate::flush32(y);
            let tail = self.re[HOP + i] * self.window[HOP + i];
            self.synthesis_mem[i] = if tail.is_finite() { tail } else { 0.0 };
        }
    }
}

// ------------------------------------------------------------------------------- DSP fallback

/// The DSP fallback: a Wiener filter on the model's frames — the noise floor per bin by minimum
/// statistics (the smallest smoothed power over the last 1.6 s, bias-compensated; digital
/// silence ignored, so it holds through pauses and a clip's pre-roll), the gain from a
/// decision-directed a priori SNR (Ephraim–Malah, which keeps musical noise down) with a −20 dB
/// floor, smoothed over neighbouring bins, and the look-ahead used to open it before onsets. It
/// has the neural model's latency and hop.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralGate;

impl VoiceModel for SpectralGate {
    fn name(&self) -> &str {
        "DSP noise gate (voice model not installed)"
    }
    fn neural(&self) -> bool {
        false
    }
    fn new_stream(&self, channels: usize) -> Box<dyn VoiceStream> {
        Box::new(GateStream::new(channels.clamp(1, MAX_CHANNELS)))
    }
}

/// Frames kept: the newest, and the two the look-ahead waits for.
const RING: usize = 3;
const GATE_FLOOR: f32 = 0.1;
/// Decision-directed smoothing of the a priori SNR.
const DD_ALPHA: f32 = 0.98;
/// Smoothing of the power the noise floor is tracked on.
const PS_ALPHA: f32 = 0.85;
/// Minimum statistics: the floor is the smallest smoothed power over `SUBS` windows of `SUB`
/// frames (1.6 s), times the bias of taking a minimum.
const SUB: u32 = 20;
const SUBS: usize = 8;
const MIN_BIAS: f32 = 2.0;
/// Frames quieter than this (digital silence, a clip's pre-roll) leave the floor alone.
const SILENT_POWER: f32 = 1e-12;

struct GateStream {
    stft: Vec<HopStft>,
    /// `[frame][channel]` spectra.
    re: Vec<Vec<Vec<f32>>>,
    im: Vec<Vec<Vec<f32>>>,
    gains: Vec<Vec<f32>>,
    noise: Vec<f32>,
    /// Smoothed power, the minimum in the running window, and the last windows' minima.
    ps: Vec<f32>,
    cur_min: Vec<f32>,
    mins: Vec<Vec<f32>>,
    sub_count: u32,
    slot: usize,
    /// Last frame's power and gain (the decision-directed estimate).
    last_power: Vec<f32>,
    last_gain: Vec<f32>,
    power: Vec<f32>,
    cur: usize,
    started: bool,
    ok: bool,
}

impl GateStream {
    fn new(ch: usize) -> Self {
        let stft: Vec<HopStft> = (0..ch).filter_map(|_| HopStft::new()).collect();
        let ok = stft.len() == ch;
        GateStream {
            stft,
            re: vec![vec![vec![0.0; BINS]; ch]; RING],
            im: vec![vec![vec![0.0; BINS]; ch]; RING],
            gains: vec![vec![1.0; BINS]; RING],
            noise: vec![0.0; BINS],
            ps: vec![0.0; BINS],
            cur_min: vec![f32::MAX; BINS],
            mins: vec![vec![f32::MAX; BINS]; SUBS],
            sub_count: 0,
            slot: 0,
            last_power: vec![0.0; BINS],
            last_gain: vec![1.0; BINS],
            power: vec![0.0; BINS],
            cur: 0,
            started: false,
            ok,
        }
    }

    /// Follow the noise floor (minimum statistics) with this frame's power.
    fn track(&mut self, total: f32) {
        if total < SILENT_POWER {
            return;
        }
        let a = if self.started { PS_ALPHA } else { 0.0 };
        self.started = true;
        for k in 0..BINS {
            self.ps[k] = crate::flush32(a * self.ps[k] + (1.0 - a) * self.power[k]);
            self.cur_min[k] = self.cur_min[k].min(self.ps[k]);
        }
        self.sub_count += 1;
        if self.sub_count >= SUB {
            self.sub_count = 0;
            let slot = self.slot;
            self.mins[slot].copy_from_slice(&self.cur_min);
            self.cur_min.copy_from_slice(&self.ps);
            self.slot = (slot + 1) % SUBS;
        }
        for k in 0..BINS {
            let m = self.mins.iter().fold(self.cur_min[k], |m, w| m.min(w[k]));
            self.noise[k] = (MIN_BIAS * m).max(1e-14);
        }
    }
}

impl VoiceStream for GateStream {
    fn process_hop(&mut self, hops: &mut [[f32; HOP]]) {
        if !self.ok {
            return;
        }
        let ch = self.stft.len().min(hops.len());
        let cur = self.cur;
        for c in 0..ch {
            self.stft[c].analyse(&hops[c], &mut self.re[cur][c], &mut self.im[cur][c]);
        }
        // mono power
        let inv = 1.0 / ch.max(1) as f32;
        let mut total = 0.0f32;
        for k in 0..BINS {
            let (mut r, mut i) = (0.0f32, 0.0f32);
            for c in 0..ch {
                r += self.re[cur][c][k];
                i += self.im[cur][c][k];
            }
            let p = (r * r + i * i) * inv * inv;
            self.power[k] = if p.is_finite() { p } else { 0.0 };
            total += self.power[k];
        }
        self.track(total);
        let g = &mut self.gains[cur];
        for k in 0..BINS {
            let p = self.power[k];
            let n = self.noise[k].max(1e-14);
            let post = p / n;
            let prior = DD_ALPHA * self.last_gain[k] * self.last_gain[k] * self.last_power[k] / n + (1.0 - DD_ALPHA) * (post - 1.0).max(0.0);
            let gk = (prior / (1.0 + prior)).max(GATE_FLOOR);
            g[k] = if gk.is_finite() { gk } else { 1.0 };
            self.last_gain[k] = g[k];
            self.last_power[k] = crate::flush32(p);
        }
        // across neighbouring bins
        let mut prev = g[0];
        for k in 0..BINS {
            let next = if k + 1 < BINS { g[k + 1] } else { g[k] };
            let v = 0.25 * prev + 0.5 * g[k] + 0.25 * next;
            prev = g[k];
            g[k] = v;
        }
        // output the frame two hops back, opened by what the look-ahead sees
        let out = (cur + 1) % RING;
        let mid = (cur + 2) % RING;
        for c in 0..ch {
            for k in 0..BINS {
                let gk = self.gains[out][k].max(self.gains[mid][k]).max(self.gains[cur][k]);
                self.re[out][c][k] *= gk;
                self.im[out][c][k] *= gk;
            }
            let (re, im) = (&self.re[out][c], &self.im[out][c]);
            self.stft[c].synthesise(re, im, &mut hops[c]);
        }
        self.cur = out;
    }

    fn reset(&mut self) {
        self.stft.iter_mut().for_each(HopStft::reset);
        for f in self.re.iter_mut().chain(self.im.iter_mut()) {
            f.iter_mut().for_each(|c| c.fill(0.0));
        }
        self.gains.iter_mut().for_each(|g| g.fill(1.0));
        self.noise.fill(0.0);
        self.ps.fill(0.0);
        self.cur_min.fill(f32::MAX);
        self.mins.iter_mut().for_each(|m| m.fill(f32::MAX));
        self.sub_count = 0;
        self.slot = 0;
        self.last_power.fill(0.0);
        self.last_gain.fill(1.0);
        self.cur = 0;
        self.started = false;
    }
}

// ------------------------------------------------------------------------------- resampling

/// Zero crossings of the interpolation kernel on each side (at the lower of the two rates).
const ZEROS: f64 = 16.0;
/// Kernel table resolution (points per input sample).
const TABLE_RES: usize = 128;

/// A windowed-sinc (Blackman) interpolator: the value of a sampled signal at any position,
/// band-limited to `cutoff` of the input's Nyquist.
#[derive(Clone, Debug)]
struct Interp {
    /// Half-width in input samples.
    k: usize,
    table: Vec<f32>,
}

impl Interp {
    fn new(cutoff: f64) -> Self {
        let fc = cutoff.clamp(0.05, 1.0) * 0.94;
        let k = (ZEROS / fc).ceil().max(1.0) as usize;
        let n = k * TABLE_RES + 2;
        let table = (0..n)
            .map(|i| {
                let d = i as f64 / TABLE_RES as f64;
                if d >= k as f64 {
                    return 0.0;
                }
                let x = std::f64::consts::PI * fc * d;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
                let u = d / k as f64;
                let w = 0.42 + 0.5 * (std::f64::consts::PI * u).cos() + 0.08 * (2.0 * std::f64::consts::PI * u).cos();
                (fc * sinc * w) as f32
            })
            .collect();
        Interp { k, table }
    }

    #[inline]
    fn kernel(&self, d: f32) -> f32 {
        let x = d.abs() * TABLE_RES as f32;
        let i = x as usize;
        let f = x - i as f32;
        match (self.table.get(i), self.table.get(i + 1)) {
            (Some(a), Some(b)) => a + (b - a) * f,
            _ => 0.0,
        }
    }

    /// The signal in `ring` at position `idx + frac` (`0 ≤ frac < 1`).
    fn eval(&self, ring: &Ring, idx: i64, frac: f32) -> f32 {
        let k = self.k as i64;
        let mut acc = 0.0f32;
        for t in idx - k + 1..=idx + k {
            acc += ring.get(t) * self.kernel((idx - t) as f32 + frac);
        }
        acc
    }
}

/// A history of a signal by absolute sample index (power-of-two size; older or negative
/// indices read as silence).
#[derive(Clone, Debug)]
struct Ring {
    buf: Vec<f32>,
    /// One past the newest index written.
    end: i64,
}

impl Ring {
    fn new(min: usize) -> Self {
        Ring { buf: vec![0.0; min.max(2).next_power_of_two()], end: 0 }
    }
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.end = 0;
    }
    fn push(&mut self, v: f32) {
        let mask = self.buf.len() as i64 - 1;
        if let Some(s) = self.buf.get_mut((self.end & mask) as usize) {
            *s = v;
        }
        self.end += 1;
    }
    #[inline]
    fn get(&self, i: i64) -> f32 {
        if i < 0 || i >= self.end || i < self.end - self.buf.len() as i64 {
            return 0.0;
        }
        let mask = self.buf.len() as i64 - 1;
        self.buf.get((i & mask) as usize).copied().unwrap_or(0.0)
    }
}

/// Rate conversion around the model when the clip is not at 48 kHz.
struct Resampling {
    sr: i64,
    /// Clip rate → 48 kHz, and back.
    up: Interp,
    down: Interp,
    x: Vec<Ring>,
    z: Vec<Ring>,
    /// Next 48 kHz sample to compute.
    m_next: i64,
    /// Clip-rate samples pushed.
    n_in: i64,
}

// ------------------------------------------------------------------------------- the effect

/// The running part of a [`VoiceIsolation`] (absent when the rate is unusable).
struct Runner {
    stream: Box<dyn VoiceStream>,
    ch: usize,
    hop_in: Vec<[f32; HOP]>,
    hop_out: Vec<[f32; HOP]>,
    work: Vec<[f32; HOP]>,
    pos: usize,
    rs: Option<Resampling>,
    /// Dry input, delayed by the latency.
    dry: Vec<Ring>,
    latency: usize,
    frame_in: Vec<f32>,
    frame_out: Vec<f32>,
}

impl Runner {
    fn new(model: &dyn VoiceModel, sample_rate: f32, channels: usize) -> Option<Self> {
        if !sample_rate.is_finite() || !(MIN_RATE..=MAX_RATE).contains(&sample_rate) {
            return None;
        }
        let ch = channels.clamp(1, MAX_CHANNELS);
        let sr = sample_rate.round() as i64;
        let (rs, latency) = if sr == i64::from(MODEL_RATE) {
            (None, LATENCY_48K)
        } else {
            let r = sr as f64 / f64::from(MODEL_RATE);
            let up = Interp::new(1.0 / r); // band-limit to 24 kHz when the clip is faster
            let down = Interp::new(r); // and to the clip's Nyquist when it is slower
            let latency = (r * (LATENCY_48K as f64 + 2.0 + down.k as f64)).ceil() as usize + up.k;
            let rs = Resampling { sr, x: vec![Ring::new(2 * up.k + 8); ch], z: vec![Ring::new(2 * down.k + 64); ch], up, down, m_next: 0, n_in: 0 };
            (Some(rs), latency)
        };
        Some(Runner {
            stream: model.new_stream(ch),
            ch,
            hop_in: vec![[0.0; HOP]; ch],
            hop_out: vec![[0.0; HOP]; ch],
            work: vec![[0.0; HOP]; ch],
            pos: 0,
            rs,
            dry: vec![Ring::new(latency + 2); ch],
            latency,
            frame_in: vec![0.0; ch],
            frame_out: vec![0.0; ch],
        })
    }

    fn reset(&mut self) {
        self.stream.reset();
        for h in self.hop_in.iter_mut().chain(self.hop_out.iter_mut()).chain(self.work.iter_mut()) {
            h.fill(0.0);
        }
        self.pos = 0;
        if let Some(rs) = &mut self.rs {
            rs.x.iter_mut().chain(rs.z.iter_mut()).for_each(Ring::reset);
            rs.m_next = 0;
            rs.n_in = 0;
        }
        self.dry.iter_mut().for_each(Ring::reset);
    }

    /// One 48 kHz frame (`frame_in`, all channels) through the hop buffering: `frame_out` gets the
    /// enhanced frame [`LATENCY_48K`] samples older.
    fn step48(&mut self) {
        for c in 0..self.ch {
            self.hop_in[c][self.pos] = self.frame_in[c];
            self.frame_out[c] = self.hop_out[c][self.pos];
        }
        self.pos += 1;
        if self.pos == HOP {
            self.pos = 0;
            self.work.copy_from_slice(&self.hop_in);
            self.stream.process_hop(&mut self.work);
            for (o, w) in self.hop_out.iter_mut().zip(&self.work) {
                for (a, b) in o.iter_mut().zip(w) {
                    *a = if b.is_finite() { *b } else { 0.0 };
                }
            }
        }
    }

    /// One input frame (`frame_in`, all channels, at the clip's rate): `frame_out` gets the wet
    /// frame `latency` samples older.
    fn step(&mut self) {
        if self.rs.is_none() {
            self.step48();
            return;
        }
        let Some(rs) = self.rs.as_mut() else { return };
        let j = rs.n_in;
        for c in 0..self.ch {
            rs.x[c].push(self.frame_in[c]);
        }
        rs.n_in += 1;
        let sr = rs.sr;
        let rate = i64::from(MODEL_RATE);
        let ka = rs.up.k as i64;
        // every 48 kHz sample the input now covers
        loop {
            let Some(rs) = self.rs.as_mut() else { return };
            let p = rs.m_next.saturating_mul(sr);
            let (idx, frac) = (p.div_euclid(rate), p.rem_euclid(rate) as f32 / rate as f32);
            if idx + ka > j {
                break;
            }
            for c in 0..self.ch {
                self.frame_in[c] = rs.up.eval(&rs.x[c], idx, frac);
            }
            rs.m_next += 1;
            self.step48();
            let Some(rs) = self.rs.as_mut() else { return };
            for c in 0..self.ch {
                rs.z[c].push(self.frame_out[c]);
            }
        }
        let Some(rs) = self.rs.as_mut() else { return };
        // output: the model's stream at τ = (j − L)·48000/sr + LATENCY_48K
        let num = (j - self.latency as i64).saturating_mul(rate) + LATENCY_48K as i64 * sr;
        let (idx, frac) = (num.div_euclid(sr), num.rem_euclid(sr) as f32 / sr as f32);
        for c in 0..self.ch {
            self.frame_out[c] = rs.down.eval(&rs.z[c], idx, frac);
        }
    }
}

/// Voice Isolation: keeps speech, removes the rest (noise, crowds, music, hum, room), with a
/// neural model when one is installed (see the module docs).
pub struct VoiceIsolation {
    pv: ParamValues,
    run: Option<Runner>,
    model_name: String,
    neural: bool,
    mix: Smoothed,
}

impl VoiceIsolation {
    pub const PARAMS: &'static [ParamSpec] = &[ParamSpec::new("intensity", "Intensity", 0.0, 100.0, 80.0, Unit::Percent)];

    /// With the installed model ([`voice_model`]), else the DSP fallback.
    pub fn new(sample_rate: f32, channels: usize) -> Self {
        Self::with_model(sample_rate, channels, voice_model())
    }

    /// With `model` (`None`: the DSP fallback).
    pub fn with_model(sample_rate: f32, channels: usize, model: Option<Arc<dyn VoiceModel>>) -> Self {
        let model: Arc<dyn VoiceModel> = model.unwrap_or_else(|| Arc::new(SpectralGate));
        let run = Runner::new(model.as_ref(), sample_rate, channels);
        let rate = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let mut s = VoiceIsolation {
            pv: ParamValues::new(Self::PARAMS),
            run,
            model_name: model.name().to_string(),
            neural: model.neural(),
            mix: Smoothed::with_ms(0.0, rate.min(MAX_RATE), 30.0),
        };
        s.apply_params(true);
        s
    }

    /// Which model this instance runs.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Whether it runs the neural model.
    pub fn neural(&self) -> bool {
        self.neural
    }

    /// Whether it processes at all (false: the rate is outside 8–384 kHz and audio passes).
    pub fn active(&self) -> bool {
        self.run.is_some()
    }

    fn apply_params(&mut self, snap: bool) {
        self.mix.set(intensity_mix(self.pv.v("intensity")));
        if snap {
            self.mix.snap();
        }
    }
}

/// Wet share for an intensity (0–100 %): the dry signal left is at most `−0.6 × intensity` dB, so
/// 0 % is the input and 100 % the model's output alone.
pub fn intensity_mix(intensity: f32) -> f32 {
    let i = if intensity.is_finite() { intensity.clamp(0.0, 100.0) } else { 0.0 };
    let floor = 10f32.powf(-3.0);
    ((1.0 - 10f32.powf(-0.03 * i)) / (1.0 - floor)).clamp(0.0, 1.0)
}

impl AudioEffect for VoiceIsolation {
    param_plumbing!("voice_isolation");
    fn latency(&self) -> usize {
        self.run.as_ref().map_or(0, |r| r.latency)
    }
    fn reset(&mut self) {
        if let Some(r) = &mut self.run {
            r.reset();
        }
        self.mix.snap();
    }
    fn process(&mut self, channels: &mut [&mut [f32]]) {
        let n = block_len(channels);
        let Some(r) = self.run.as_mut() else { return };
        let nch = channels.len().min(r.ch);
        let latency = r.latency as i64;
        for i in 0..n {
            for c in 0..r.ch {
                let x = if c < nch { channels[c][i] } else { 0.0 };
                let x = if x.is_finite() { x } else { 0.0 };
                r.frame_in[c] = x;
                if c < nch {
                    r.dry[c].push(x);
                }
            }
            r.step();
            let m = self.mix.tick();
            for c in 0..nch {
                let ring = &r.dry[c];
                let dry = ring.get(ring.end - 1 - latency);
                let wet = r.frame_out[c];
                channels[c][i] = crate::flush32(dry + m * (wet - dry));
            }
        }
    }
}

/// Run `fx` over whole planar `channels` (each the same length), compensating its latency: the
/// output lines up with the input. For offline processing of a file or a clip.
pub fn process_offline(fx: &mut dyn AudioEffect, channels: &mut [Vec<f32>]) {
    let n = channels.iter().map(Vec::len).min().unwrap_or(0);
    let lat = fx.latency();
    const BLOCK: usize = 4096;
    let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(n + lat); channels.len()];
    let mut buf: Vec<Vec<f32>> = vec![vec![0.0; BLOCK]; channels.len()];
    let total = n + lat;
    let mut at = 0;
    while at < total {
        let len = BLOCK.min(total - at);
        for (b, c) in buf.iter_mut().zip(channels.iter()) {
            for (i, v) in b.iter_mut().take(len).enumerate() {
                *v = c.get(at + i).copied().unwrap_or(0.0);
            }
        }
        let mut refs: Vec<&mut [f32]> = buf.iter_mut().map(|b| &mut b[..len]).collect();
        fx.process(&mut refs);
        for (o, b) in out.iter_mut().zip(&buf) {
            o.extend_from_slice(&b[..len]);
        }
        at += len;
    }
    for (c, o) in channels.iter_mut().zip(out) {
        let start = lat.min(o.len());
        let end = (start + n).min(o.len());
        c.clear();
        c.extend_from_slice(&o[start..end]);
        c.resize(n, 0.0);
    }
}

#[cfg(test)]
#[path = "voice_tests.rs"]
mod tests;
