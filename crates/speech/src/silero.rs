//! Silero VAD v5 (16 kHz) in pure Rust: the speech probability of every 32 ms of audio, and the
//! speech segments found from it.
//!
//! The network is the one Silero Team publishes as `silero_vad_16k.safetensors` (MIT; see
//! [`crate::models::SILERO_VAD`]); its layout follows the authors' own reference definition
//! (`tinygrad_model.py` in the same repository, MIT):
//!
//! 1. each chunk is 512 new samples after the previous chunk's last 64 (zeros before the first),
//!    reflect-padded by 64 on the right (640 samples);
//! 2. a fixed "STFT" convolution (258 filters of 256 taps, hop 128: 4 frames) whose two halves are
//!    combined into 129 magnitudes per frame;
//! 3. four 1-D convolutions with ReLU (129→128, 128→64 stride 2, 64→64 stride 2, 64→128; kernel
//!    3, padding 1), leaving one 128-vector;
//! 4. one LSTM cell (128 → 128; its state carries over from chunk to chunk);
//! 5. ReLU, a 1×1 convolution to one value, sigmoid: the probability that the chunk is speech.
//!
//! The front end (steps 1–3) of every chunk is independent and runs in parallel; the LSTM runs in
//! order. About 0.7 M multiply-adds a chunk: an hour of audio is a few tens of seconds on one core.
//!
//! [`segments`] turns the probabilities into speech segments with the usual hysteresis: speech
//! starts at a chunk at or above `threshold` and ends after `min_silence` below `threshold −
//! 0.15`; segments shorter than `min_speech` are dropped and the rest padded by `pad` (never into
//! the next one).

use std::path::Path;

use crate::SpeechError;

/// New samples per chunk (32 ms).
pub const CHUNK: usize = 512;
/// Samples of the previous chunk each chunk sees.
const CONTEXT: usize = 64;
/// Right reflect padding of the 576-sample input.
const PAD: usize = 64;
const N_FFT: usize = 256;
const HOP: usize = 128;
const BINS: usize = N_FFT / 2 + 1;
const HIDDEN: usize = 128;
/// Largest safetensors header we read (the real one is about 1.5 kB).
const MAX_HEADER: usize = 1 << 20;

/// A 1-D convolution's weights (`out × in × k`, PyTorch layout) and bias.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl Conv {
    /// `x`: `cin × t` row-major → `cout × t'`.
    fn forward(&self, x: &[f32], t: usize, relu: bool) -> (Vec<f32>, usize) {
        let t_out = (t + 2 * self.pad).saturating_sub(self.k) / self.stride + 1;
        let mut y = vec![0f32; self.cout * t_out];
        for o in 0..self.cout {
            let bias = self.b.get(o).copied().unwrap_or(0.0);
            for j in 0..t_out {
                let mut acc = bias;
                for i in 0..self.cin {
                    let wrow = &self.w[(o * self.cin + i) * self.k..(o * self.cin + i + 1) * self.k];
                    let xrow = &x[i * t..(i + 1) * t];
                    for (kk, wv) in wrow.iter().enumerate() {
                        let pos = (j * self.stride + kk) as isize - self.pad as isize;
                        if pos >= 0
                            && let Some(xv) = xrow.get(pos as usize)
                        {
                            acc += wv * xv;
                        }
                    }
                }
                y[o * t_out + j] = if relu { acc.max(0.0) } else { acc };
            }
        }
        (y, t_out)
    }
}

/// The Silero VAD v5 network (16 kHz).
pub struct Silero {
    /// `258 × 256`: the real then the imaginary parts of 129 frequency bins.
    stft: Vec<f32>,
    convs: [Conv; 4],
    /// `512 × 128` each, gates in the order input, forget, cell, output.
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    /// `b_ih + b_hh`.
    b: Vec<f32>,
    final_w: Vec<f32>,
    final_b: f32,
}

/// The tensors of a safetensors file (F32 only): name → (shape, values).
pub(crate) fn read_safetensors(bytes: &[u8]) -> Result<std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>, SpeechError> {
    let bad = |m: &str| SpeechError::Model(format!("safetensors: {m}"));
    let n = bytes.get(..8).and_then(|b| <[u8; 8]>::try_from(b).ok()).map(u64::from_le_bytes).ok_or_else(|| bad("truncated"))?;
    let n = usize::try_from(n).ok().filter(|n| *n <= MAX_HEADER).ok_or_else(|| bad("header too large"))?;
    let header = bytes.get(8..8 + n).ok_or_else(|| bad("truncated header"))?;
    let data = bytes.get(8 + n..).unwrap_or(&[]);
    let v: serde_json::Value = serde_json::from_slice(header).map_err(|e| bad(&e.to_string()))?;
    let obj = v.as_object().ok_or_else(|| bad("header is not an object"))?;
    let mut out = std::collections::BTreeMap::new();
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        if t["dtype"].as_str() != Some("F32") {
            return Err(bad(&format!("{name}: only F32 tensors are read")));
        }
        let shape: Vec<usize> =
            t["shape"].as_array().ok_or_else(|| bad("no shape"))?.iter().filter_map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok())).collect();
        let offs: Vec<usize> =
            t["data_offsets"].as_array().ok_or_else(|| bad("no offsets"))?.iter().filter_map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok())).collect();
        let (Some(&a), Some(&b)) = (offs.first(), offs.get(1)) else { return Err(bad("bad offsets")) };
        let raw = data.get(a..b).ok_or_else(|| bad(&format!("{name}: data out of range")))?;
        let count = shape.iter().try_fold(1usize, |acc, d| acc.checked_mul(*d)).ok_or_else(|| bad("shape overflows"))?;
        if raw.len() != count.saturating_mul(4) {
            return Err(bad(&format!("{name}: {} bytes for {count} values", raw.len())));
        }
        let vals = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        out.insert(name.clone(), (shape, vals));
    }
    Ok(out)
}

impl Silero {
    /// Load `silero_vad_16k.safetensors`.
    pub fn load(path: &Path) -> Result<Self, SpeechError> {
        let bytes = std::fs::read(path).map_err(|e| SpeechError::Model(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpeechError> {
        let mut t = read_safetensors(bytes)?;
        let mut take = |name: &str, shape: &[usize]| -> Result<Vec<f32>, SpeechError> {
            let (s, v) = t.remove(name).ok_or_else(|| SpeechError::Model(format!("silero: no tensor `{name}`")))?;
            if s != shape {
                return Err(SpeechError::Model(format!("silero: `{name}` is {s:?}, expected {shape:?}")));
            }
            Ok(v)
        };
        let conv = |w: Vec<f32>, b: Vec<f32>, cin, cout, stride| Conv { w, b, cin, cout, k: 3, stride, pad: 1 };
        let stft = take("stft_conv.weight", &[2 * BINS, 1, N_FFT])?;
        let c1 = conv(take("conv1.weight", &[128, BINS, 3])?, take("conv1.bias", &[128])?, BINS, 128, 1);
        let c2 = conv(take("conv2.weight", &[64, 128, 3])?, take("conv2.bias", &[64])?, 128, 64, 2);
        let c3 = conv(take("conv3.weight", &[64, 64, 3])?, take("conv3.bias", &[64])?, 64, 64, 2);
        let c4 = conv(take("conv4.weight", &[128, 64, 3])?, take("conv4.bias", &[128])?, 64, 128, 1);
        let w_ih = take("lstm_cell.weight_ih", &[4 * HIDDEN, HIDDEN])?;
        let w_hh = take("lstm_cell.weight_hh", &[4 * HIDDEN, HIDDEN])?;
        let b_ih = take("lstm_cell.bias_ih", &[4 * HIDDEN])?;
        let b_hh = take("lstm_cell.bias_hh", &[4 * HIDDEN])?;
        let final_w = take("final_conv.weight", &[1, HIDDEN, 1])?;
        let final_b = take("final_conv.bias", &[1])?.first().copied().unwrap_or(0.0);
        let b = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
        Ok(Self { stft, convs: [c1, c2, c3, c4], w_ih, w_hh, b, final_w, final_b })
    }

    /// The 128 features of one chunk: `x` holds the 64 context samples then the 512 new ones.
    fn features(&self, x: &[f32]) -> Vec<f32> {
        // reflect padding on the right (the edge sample is not repeated)
        let n = CONTEXT + CHUNK;
        let mut s = vec![0f32; n + PAD];
        for (i, v) in s.iter_mut().enumerate() {
            let j = if i < n { i } else { 2 * (n - 1) - i };
            *v = x.get(j).copied().unwrap_or(0.0);
        }
        let frames = (s.len() - N_FFT) / HOP + 1;
        let mut mag = vec![0f32; BINS * frames];
        for f in 0..frames {
            let win = &s[f * HOP..f * HOP + N_FFT];
            for k in 0..BINS {
                let re: f32 = self.stft[k * N_FFT..(k + 1) * N_FFT].iter().zip(win).map(|(a, b)| a * b).sum();
                let im: f32 = self.stft[(k + BINS) * N_FFT..(k + BINS + 1) * N_FFT].iter().zip(win).map(|(a, b)| a * b).sum();
                mag[k * frames + f] = (re * re + im * im).sqrt();
            }
        }
        let (mut y, mut t) = (mag, frames);
        for c in &self.convs {
            (y, t) = c.forward(&y, t, true);
        }
        // one frame is left: 128 values (any extra frame would be averaged by the reference)
        (0..HIDDEN).map(|c| y.get(c * t..(c + 1) * t).map_or(0.0, |r| r.iter().sum::<f32>() / t.max(1) as f32)).collect()
    }

    /// The speech probability of every 512-sample chunk of mono 16 kHz `audio` (the last chunk
    /// zero-padded). Non-finite samples count as silence.
    pub fn probabilities(&self, audio: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let chunks = audio.len().div_ceil(CHUNK);
        let at = |i: isize| -> f32 { if i < 0 { 0.0 } else { audio.get(i as usize).copied().filter(|v| v.is_finite()).unwrap_or(0.0) } };
        let feats: Vec<Vec<f32>> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let start = (c * CHUNK) as isize - CONTEXT as isize;
                let x: Vec<f32> = (0..CONTEXT + CHUNK).map(|i| at(start + i as isize)).collect();
                self.features(&x)
            })
            .collect();
        let (mut h, mut cs) = (vec![0f32; HIDDEN], vec![0f32; HIDDEN]);
        let mut gates = vec![0f32; 4 * HIDDEN];
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut out = Vec::with_capacity(chunks);
        for x in feats {
            for (g, gv) in gates.iter_mut().enumerate() {
                let wi = &self.w_ih[g * HIDDEN..(g + 1) * HIDDEN];
                let wh = &self.w_hh[g * HIDDEN..(g + 1) * HIDDEN];
                *gv = self.b[g] + wi.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + wh.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
            }
            for k in 0..HIDDEN {
                let i = sigmoid(gates[k]);
                let f = sigmoid(gates[HIDDEN + k]);
                let g = gates[2 * HIDDEN + k].tanh();
                let o = sigmoid(gates[3 * HIDDEN + k]);
                cs[k] = f * cs[k] + i * g;
                h[k] = o * cs[k].tanh();
            }
            let logit = self.final_b + self.final_w.iter().zip(&h).map(|(w, v)| w * v.max(0.0)).sum::<f32>();
            let p = sigmoid(logit);
            out.push(if p.is_finite() { p } else { 0.0 });
        }
        out
    }
}

/// How probabilities become segments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentParams {
    pub threshold: f32,
    /// Shortest speech kept, seconds.
    pub min_speech: f64,
    /// Silence that ends speech, seconds.
    pub min_silence: f64,
    /// Padding on both sides of a segment, seconds.
    pub pad: f64,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self { threshold: 0.5, min_speech: 0.25, min_silence: 0.1, pad: 0.03 }
    }
}

/// Speech segments (sample ranges, in order, non-overlapping) of `probs` (one per [`CHUNK`]) in
/// audio of `len` samples.
pub fn segments(probs: &[f32], len: usize, p: &SegmentParams) -> Vec<std::ops::Range<usize>> {
    segments_hop(probs, CHUNK, len, p)
}

/// [`segments`] for probabilities `hop` samples apart.
pub fn segments_hop(probs: &[f32], hop: usize, len: usize, p: &SegmentParams) -> Vec<std::ops::Range<usize>> {
    let hop = hop.max(1);
    let sr = crate::SAMPLE_RATE as f64;
    let secs = |s: f64| if s.is_finite() && s > 0.0 { (s.min(3600.0) * sr) as usize } else { 0 };
    let (min_speech, min_silence, pad) = (secs(p.min_speech), secs(p.min_silence), secs(p.pad));
    let th = if p.threshold.is_finite() { p.threshold.clamp(0.01, 0.99) } else { 0.5 };
    let neg = (th - 0.15).max(0.01);
    let mut raw: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start: Option<usize> = None;
    let mut temp_end: Option<usize> = None;
    for (i, &pr) in probs.iter().enumerate() {
        let cur = i.saturating_mul(hop);
        if pr >= th {
            temp_end = None;
            if start.is_none() {
                start = Some(cur);
            }
            continue;
        }
        let Some(s) = start else { continue };
        if pr < neg {
            let te = *temp_end.get_or_insert(cur);
            if (cur + hop).saturating_sub(te) >= min_silence {
                if te - s >= min_speech {
                    raw.push(s..te);
                }
                start = None;
                temp_end = None;
            }
        }
    }
    if let Some(s) = start {
        let e = temp_end.unwrap_or(len).min(len);
        if e > s && e - s >= min_speech {
            raw.push(s..e);
        }
    }
    pad_segments(raw, len, pad)
}

/// Pad every segment by `pad` samples on both sides, splitting a shorter gap between two.
pub fn pad_segments(raw: Vec<std::ops::Range<usize>>, len: usize, pad: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = raw.clone();
    for (i, r) in raw.iter().enumerate() {
        let gap_before = if i > 0 { r.start.saturating_sub(raw[i - 1].end) / 2 } else { r.start };
        // the gap after is split at the same point as the next segment's gap before
        let gap_after = match raw.get(i + 1) {
            Some(nx) => nx.start.saturating_sub(r.end) - nx.start.saturating_sub(r.end) / 2,
            None => len.saturating_sub(r.end),
        };
        out[i] = r.start - pad.min(gap_before).min(r.start)..(r.end.saturating_add(pad.min(gap_after))).min(len);
    }
    out.retain(|r| r.end > r.start);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Vec<u8> {
        // a model whose output is a fixed function of the chunk's level: weights chosen so the
        // safetensors reader and the shapes are exercised (not a trained network)
        let mut tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut stft = vec![0f32; 2 * BINS * N_FFT];
        for k in 0..BINS {
            for i in 0..N_FFT {
                let a = 2.0 * std::f32::consts::PI * (k * i) as f32 / N_FFT as f32;
                stft[k * N_FFT + i] = a.cos();
                stft[(k + BINS) * N_FFT + i] = -a.sin();
            }
        }
        tensors.push(("stft_conv.weight", vec![2 * BINS, 1, N_FFT], stft));
        let ones = |n: usize, v: f32| vec![v; n];
        tensors.push(("conv1.weight", vec![128, BINS, 3], ones(128 * BINS * 3, 0.001)));
        tensors.push(("conv1.bias", vec![128], ones(128, 0.0)));
        tensors.push(("conv2.weight", vec![64, 128, 3], ones(64 * 128 * 3, 0.01)));
        tensors.push(("conv2.bias", vec![64], ones(64, 0.0)));
        tensors.push(("conv3.weight", vec![64, 64, 3], ones(64 * 64 * 3, 0.01)));
        tensors.push(("conv3.bias", vec![64], ones(64, 0.0)));
        tensors.push(("conv4.weight", vec![128, 64, 3], ones(128 * 64 * 3, 0.01)));
        tensors.push(("conv4.bias", vec![128], ones(128, 0.0)));
        tensors.push(("lstm_cell.weight_ih", vec![512, 128], ones(512 * 128, 0.05)));
        tensors.push(("lstm_cell.weight_hh", vec![512, 128], ones(512 * 128, 0.0)));
        tensors.push(("lstm_cell.bias_ih", vec![512], ones(512, 0.0)));
        tensors.push(("lstm_cell.bias_hh", vec![512], ones(512, 0.0)));
        tensors.push(("final_conv.weight", vec![1, 128, 1], ones(128, 1.0)));
        tensors.push(("final_conv.bias", vec![1], vec![-3.0]));
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, vals) in &tensors {
            let a = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(name.to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [a, data.len()]}));
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        out
    }

    #[test]
    fn loud_chunks_score_higher_than_silence() {
        let m = Silero::from_bytes(&tiny_model()).unwrap();
        let mut audio = vec![0f32; 16_000];
        audio.extend((0..16_000).map(|i| 0.5 * (i as f32 * 0.3).sin()));
        let p = m.probabilities(&audio);
        assert_eq!(p.len(), 32_000usize.div_ceil(CHUNK));
        assert!(p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(p[40] > p[5], "{} vs {}", p[40], p[5]);
        assert!(m.probabilities(&[]).is_empty());
        // hostile samples are silence, never a NaN out
        assert!(m.probabilities(&[f32::NAN; 2000]).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bad_files_are_errors() {
        assert!(Silero::from_bytes(&[]).is_err());
        assert!(Silero::from_bytes(&[255; 16]).is_err());
        let mut b = tiny_model();
        b.truncate(b.len() - 100);
        assert!(Silero::from_bytes(&b).is_err());
        // a header claiming to be huge
        let mut h = u64::MAX.to_le_bytes().to_vec();
        h.extend([0u8; 32]);
        assert!(Silero::from_bytes(&h).is_err());
    }

    #[test]
    fn hysteresis_makes_segments() {
        let sr = 16_000usize;
        // 1 s silence, 1 s speech with a 64 ms dip, 1 s silence, 0.1 s blip, silence
        let mut probs = vec![0.0f32; 31];
        probs.extend(vec![0.9; 31]);
        probs[45] = 0.2;
        probs[46] = 0.2;
        probs.extend(vec![0.0; 31]);
        probs.extend(vec![0.9; 3]);
        probs.extend(vec![0.0; 31]);
        let len = probs.len() * CHUNK;
        let p = SegmentParams { threshold: 0.5, min_speech: 0.25, min_silence: 0.1, pad: 0.03 };
        let s = segments(&probs, len, &p);
        assert_eq!(s.len(), 1, "{s:?}");
        let pad = (0.03 * sr as f64) as usize;
        assert_eq!(s[0].start, 31 * CHUNK - pad);
        assert_eq!(s[0].end, 62 * CHUNK + pad);
        // speech to the very end closes at the end
        let s = segments(&[0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 10 * CHUNK, &p);
        assert_eq!(s, vec![CHUNK - pad..10 * CHUNK]);
        assert!(segments(&[], 0, &p).is_empty());
        // NaN parameters fall back to sane values
        let weird = SegmentParams { threshold: f32::NAN, min_speech: f64::NAN, min_silence: -1.0, pad: f64::INFINITY };
        assert!(!segments(&probs, len, &weird).is_empty());
    }

    #[test]
    fn padding_never_overlaps() {
        let s = pad_segments(vec![100..200, 210..300], 310, 50);
        assert_eq!(s, vec![50..205, 205..310]);
    }
}
