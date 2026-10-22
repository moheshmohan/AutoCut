//! The DeepFilterNet 3 network written out by hand, one 10 ms frame at a time (the streaming
//! form: the convolutions keep the frames their time kernels need, the GRUs keep their state).
//!
//! Architecture (DeepFilterNet 3, Schröter et al., Interspeech 2023; the layers as exported to
//! ONNX, batch norms folded into the convolutions):
//!
//! - **Encoder**: the ERB features (32 bands) through a 3×3 convolution (1 → 64 channels) and
//!   three separable 1×3 convolutions (frequency strides 2, 2, 1: 32 → 16 → 8 → 8 bands); the
//!   complex features of the lowest 96 bins through a grouped 3×3 convolution (2 → 64) and a
//!   separable 1×3 (stride 2: 96 → 48); the second path is projected (grouped linear, 3072 → 512)
//!   and added to the first; a squeezed GRU (grouped linear 512 → 256, GRU 256, grouped linear
//!   256 → 512) gives the embedding and the local SNR estimate.
//! - **ERB decoder**: a two-layer squeezed GRU on the embedding, then transposed convolutions back
//!   up the encoder's scales with 1×1 skip pathways, ending in a sigmoid gain per ERB band.
//! - **DF decoder**: a two-layer GRU on the embedding (plus a grouped-linear skip), a grouped
//!   linear to 96 × 5 complex deep-filter coefficients (tanh), plus a 5×1 pathway convolution of
//!   the encoder's complex features.
//!
//! Tensors are channel-major per frame (`[channel][frequency]`), like ONNX's NCHW with T = 1.

/// ERB bands.
pub const NB_ERB: usize = 32;
/// Bins the deep filter works on.
pub const NB_DF: usize = 96;
/// Deep-filter taps (frames).
pub const DF_ORDER: usize = 5;
/// Convolution channels.
pub const CH: usize = 64;
/// GRU width.
pub const HIDDEN: usize = 256;
/// Embedding width (`CH × NB_ERB / 4`).
pub const EMB: usize = CH * NB_ERB / 4;

/// A (grouped) 2-D convolution over `[time, frequency]` with a causal time kernel.
#[derive(Clone, Debug)]
pub struct Conv {
    pub out_ch: usize,
    pub in_ch: usize,
    pub groups: usize,
    /// Time taps (frames, oldest first) and frequency taps.
    pub kt: usize,
    pub kf: usize,
    pub stride: usize,
    pub pad: usize,
    /// `[out][in / groups][kt][kf]`.
    pub w: Vec<f32>,
    /// One per output channel (zeros when the layer has none).
    pub b: Vec<f32>,
}

impl Conv {
    pub fn f_out(&self, f_in: usize) -> usize {
        (f_in + 2 * self.pad).saturating_sub(self.kf) / self.stride.max(1) + 1
    }

    /// `frames[t]` (`kt` of them, oldest first) are `[in_ch][f_in]`; `out` is `[out_ch][f_out]`.
    pub fn forward(&self, frames: &[&[f32]], f_in: usize, out: &mut [f32]) {
        let f_out = self.f_out(f_in);
        let cin_g = self.in_ch / self.groups.max(1);
        let cout_g = (self.out_ch / self.groups.max(1)).max(1);
        for o in 0..self.out_ch {
            let g = o / cout_g;
            let Some(row) = out.get_mut(o * f_out..(o + 1) * f_out) else { return };
            row.fill(self.b.get(o).copied().unwrap_or(0.0));
            for ci in 0..cin_g {
                let ich = g * cin_g + ci;
                for (t, frame) in frames.iter().enumerate().take(self.kt) {
                    let Some(inp) = frame.get(ich * f_in..(ich + 1) * f_in) else { continue };
                    for k in 0..self.kf {
                        let wv = self.w.get(((o * cin_g + ci) * self.kt + t) * self.kf + k).copied().unwrap_or(0.0);
                        if wv == 0.0 {
                            continue;
                        }
                        if self.stride == 1 {
                            // in index = f + k − pad
                            let lo = self.pad.saturating_sub(k);
                            let hi = (f_in + self.pad).saturating_sub(k).min(f_out);
                            for f in lo..hi {
                                row[f] += wv * inp[f + k - self.pad];
                            }
                        } else {
                            for (f, r) in row.iter_mut().enumerate() {
                                let idx = (f * self.stride + k) as isize - self.pad as isize;
                                if idx >= 0
                                    && let Some(x) = inp.get(idx as usize)
                                {
                                    *r += wv * x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A depthwise transposed convolution along frequency (kernel 3, stride 2, padding 1, output
/// padding 1: `f` bands become `2f`).
#[derive(Clone, Debug)]
pub struct ConvT {
    pub ch: usize,
    /// `[ch][3]`.
    pub w: Vec<f32>,
}

impl ConvT {
    pub fn forward(&self, inp: &[f32], f_in: usize, out: &mut [f32]) {
        let f_out = 2 * f_in;
        for c in 0..self.ch {
            let (Some(row), Some(src)) = (out.get_mut(c * f_out..(c + 1) * f_out), inp.get(c * f_in..(c + 1) * f_in)) else { return };
            row.fill(0.0);
            for (i, x) in src.iter().enumerate() {
                for k in 0..3 {
                    // out position = 2i + k − 1
                    let o = 2 * i + k;
                    if o >= 1
                        && let (Some(r), Some(w)) = (row.get_mut(o - 1), self.w.get(c * 3 + k))
                    {
                        *r += w * x;
                    }
                }
            }
        }
    }
}

/// A 1×1 convolution (`out = W·in + b` at every frequency).
#[derive(Clone, Debug)]
pub struct Pointwise {
    pub out_ch: usize,
    pub in_ch: usize,
    /// `[out][in]`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Pointwise {
    pub fn forward(&self, inp: &[f32], f: usize, out: &mut [f32]) {
        for o in 0..self.out_ch {
            let Some(row) = out.get_mut(o * f..(o + 1) * f) else { return };
            row.fill(self.b.get(o).copied().unwrap_or(0.0));
            for i in 0..self.in_ch {
                let wv = self.w.get(o * self.in_ch + i).copied().unwrap_or(0.0);
                if let Some(src) = inp.get(i * f..(i + 1) * f) {
                    axpy(wv, src, row);
                }
            }
        }
    }
}

/// A grouped linear layer (`einsum("btgi,gih->btgh")`).
#[derive(Clone, Debug)]
pub struct GLinear {
    pub groups: usize,
    pub inp: usize,
    pub out: usize,
    /// `[groups][inp][out]`.
    pub w: Vec<f32>,
}

impl GLinear {
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        for g in 0..self.groups {
            let (Some(xs), Some(ys)) = (x.get(g * self.inp..(g + 1) * self.inp), y.get_mut(g * self.out..(g + 1) * self.out)) else { return };
            ys.fill(0.0);
            for (i, xv) in xs.iter().enumerate() {
                if let Some(wr) = self.w.get((g * self.inp + i) * self.out..(g * self.inp + i + 1) * self.out) {
                    axpy(*xv, wr, ys);
                }
            }
        }
    }
}

/// A GRU layer as ONNX defines it (gates z, r, h; `linear_before_reset`, as PyTorch computes it).
#[derive(Clone, Debug)]
pub struct Gru {
    pub input: usize,
    pub hidden: usize,
    /// `[3·hidden][input]` and `[3·hidden][hidden]`.
    pub w: Vec<f32>,
    pub r: Vec<f32>,
    /// `[3·hidden]` each.
    pub bw: Vec<f32>,
    pub br: Vec<f32>,
}

/// Scratch for a GRU step.
#[derive(Clone, Debug)]
pub struct GruScratch {
    gx: Vec<f32>,
    gh: Vec<f32>,
}

impl GruScratch {
    pub fn new(hidden: usize) -> Self {
        GruScratch { gx: vec![0.0; 3 * hidden], gh: vec![0.0; 3 * hidden] }
    }
}

impl Gru {
    /// One step: `h` (the state) becomes the output.
    pub fn step(&self, x: &[f32], h: &mut [f32], s: &mut GruScratch) {
        let n = self.hidden;
        matvec(&self.w, self.input, x, &self.bw, &mut s.gx);
        matvec(&self.r, n, h, &self.br, &mut s.gh);
        for j in 0..n.min(h.len()) {
            let z = sigmoid(s.gx[j] + s.gh[j]);
            let r = sigmoid(s.gx[n + j] + s.gh[n + j]);
            let c = (s.gx[2 * n + j] + r * s.gh[2 * n + j]).tanh();
            h[j] = (1.0 - z) * c + z * h[j];
        }
    }
}

/// `y += a·x`.
#[inline]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}

/// Dot product with eight independent accumulators (vectorises without reassociating).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `y = W·x + b` for row-major `W` with `cols` columns.
fn matvec(w: &[f32], cols: usize, x: &[f32], b: &[f32], y: &mut [f32]) {
    for (i, yv) in y.iter_mut().enumerate() {
        let row = w.get(i * cols..(i + 1) * cols).unwrap_or(&[]);
        *yv = dot(row, x) + b.get(i).copied().unwrap_or(0.0);
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

/// Every weight of the three networks.
#[derive(Clone, Debug)]
pub struct Weights {
    // encoder
    pub erb_conv0: Conv,
    pub erb_conv1: (Conv, Pointwise),
    pub erb_conv2: (Conv, Pointwise),
    pub erb_conv3: (Conv, Pointwise),
    pub df_conv0: (Conv, Pointwise),
    pub df_conv1: (Conv, Pointwise),
    pub df_fc_emb: GLinear,
    pub enc_in: GLinear,
    pub enc_gru: Gru,
    pub enc_out: GLinear,
    /// Local SNR: `sigmoid(w·emb + b)·scale + offset`.
    pub lsnr_w: Vec<f32>,
    pub lsnr_b: f32,
    pub lsnr_scale: f32,
    pub lsnr_offset: f32,
    // ERB decoder
    pub erb_in: GLinear,
    pub erb_gru: [Gru; 2],
    pub erb_out: GLinear,
    pub conv3p: Conv,
    pub convt3: (Conv, Pointwise),
    pub conv2p: Conv,
    pub convt2: (ConvT, Pointwise),
    pub conv1p: Conv,
    pub convt1: (ConvT, Pointwise),
    pub conv0p: Conv,
    pub conv0_out: Conv,
    // DF decoder
    pub df_in: GLinear,
    pub df_gru: [Gru; 2],
    pub df_skip: GLinear,
    pub df_convp: (Conv, Pointwise),
    pub df_out: GLinear,
}

/// One stream's network state and scratch.
#[derive(Clone, Debug)]
pub struct NetState {
    /// The last three frames of features (oldest first, rotated).
    erb_hist: [Vec<f32>; 3],
    spec_hist: [Vec<f32>; 3],
    /// The last five frames of `c0`.
    c0_hist: [Vec<f32>; DF_ORDER],
    enc_h: Vec<f32>,
    erb_h: [Vec<f32>; 2],
    df_h: [Vec<f32>; 2],
    gs: GruScratch,
    // activations of the current frame
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    e3: Vec<f32>,
    tmp: Vec<f32>,
    tmp2: Vec<f32>,
    cemb: Vec<f32>,
    emb_a: Vec<f32>,
    pub emb: Vec<f32>,
    h256: Vec<f32>,
    h256b: Vec<f32>,
    d: Vec<f32>,
    dsum: Vec<f32>,
    /// ERB gains of the frame two back.
    pub mask: Vec<f32>,
    /// Deep-filter coefficients `[NB_DF][DF_ORDER][re, im]`.
    pub coefs: Vec<f32>,
    convp: Vec<f32>,
    df_o: Vec<f32>,
}

impl Default for NetState {
    fn default() -> Self {
        Self::new()
    }
}

impl NetState {
    pub fn new() -> Self {
        let z = |n: usize| vec![0.0f32; n];
        NetState {
            erb_hist: [z(NB_ERB), z(NB_ERB), z(NB_ERB)],
            spec_hist: [z(2 * NB_DF), z(2 * NB_DF), z(2 * NB_DF)],
            c0_hist: [z(CH * NB_DF), z(CH * NB_DF), z(CH * NB_DF), z(CH * NB_DF), z(CH * NB_DF)],
            enc_h: z(HIDDEN),
            erb_h: [z(HIDDEN), z(HIDDEN)],
            df_h: [z(HIDDEN), z(HIDDEN)],
            gs: GruScratch::new(HIDDEN),
            e0: z(CH * NB_ERB),
            e1: z(CH * NB_ERB / 2),
            e2: z(CH * NB_ERB / 4),
            e3: z(CH * NB_ERB / 4),
            tmp: z(CH * NB_DF),
            tmp2: z(CH * NB_DF),
            cemb: z(CH * NB_DF / 2),
            emb_a: z(EMB),
            emb: z(EMB),
            h256: z(HIDDEN),
            h256b: z(HIDDEN),
            d: z(CH * NB_ERB),
            dsum: z(CH * NB_ERB),
            mask: z(NB_ERB),
            coefs: z(NB_DF * DF_ORDER * 2),
            convp: z(10 * NB_DF),
            df_o: z(NB_DF * DF_ORDER * 2),
        }
    }

    /// Back to the initial (zero) state, without allocating.
    pub fn reset(&mut self) {
        let NetState {
            erb_hist,
            spec_hist,
            c0_hist,
            enc_h,
            erb_h,
            df_h,
            gs,
            e0,
            e1,
            e2,
            e3,
            tmp,
            tmp2,
            cemb,
            emb_a,
            emb,
            h256,
            h256b,
            d,
            dsum,
            mask,
            coefs,
            convp,
            df_o,
        } = self;
        for v in erb_hist.iter_mut().chain(spec_hist.iter_mut()).chain(c0_hist.iter_mut()).chain(erb_h.iter_mut()).chain(df_h.iter_mut()) {
            v.fill(0.0);
        }
        for v in [enc_h, &mut gs.gx, &mut gs.gh, e0, e1, e2, e3, tmp, tmp2, cemb, emb_a, emb, h256, h256b, d, dsum, mask, coefs, convp, df_o] {
            v.fill(0.0);
        }
    }

    /// The encoder on this frame's features (`feat_erb`: 32 values; `feat_spec`: 96 real parts
    /// then 96 imaginary parts). Returns the local SNR estimate (dB).
    pub fn encode(&mut self, w: &Weights, feat_erb: &[f32], feat_spec: &[f32]) -> f32 {
        self.erb_hist.rotate_left(1);
        self.spec_hist.rotate_left(1);
        copy(feat_erb, &mut self.erb_hist[2]);
        copy(feat_spec, &mut self.spec_hist[2]);
        // ERB path
        let h: [&[f32]; 3] = [&self.erb_hist[0], &self.erb_hist[1], &self.erb_hist[2]];
        w.erb_conv0.forward(&h, NB_ERB, &mut self.e0);
        relu(&mut self.e0);
        separable(&w.erb_conv1, &self.e0, NB_ERB, &mut self.tmp, &mut self.e1);
        separable(&w.erb_conv2, &self.e1, NB_ERB / 2, &mut self.tmp, &mut self.e2);
        separable(&w.erb_conv3, &self.e2, NB_ERB / 4, &mut self.tmp, &mut self.e3);
        // complex path
        self.c0_hist.rotate_left(1);
        let h: [&[f32]; 3] = [&self.spec_hist[0], &self.spec_hist[1], &self.spec_hist[2]];
        w.df_conv0.0.forward(&h, NB_DF, &mut self.tmp);
        w.df_conv0.1.forward(&self.tmp, NB_DF, &mut self.c0_hist[DF_ORDER - 1]);
        relu(&mut self.c0_hist[DF_ORDER - 1]);
        separable(&w.df_conv1, &self.c0_hist[DF_ORDER - 1], NB_DF, &mut self.tmp, &mut self.tmp2);
        // [C][F] → [F][C], projected and added to the ERB path's [F][C]
        transpose(&self.tmp2, CH, NB_DF / 2, &mut self.cemb);
        w.df_fc_emb.forward(&self.cemb, &mut self.emb_a);
        relu(&mut self.emb_a);
        transpose(&self.e3, CH, NB_ERB / 4, &mut self.emb);
        for (e, c) in self.emb.iter_mut().zip(&self.emb_a) {
            *e += c;
        }
        // squeezed GRU
        w.enc_in.forward(&self.emb, &mut self.h256);
        relu(&mut self.h256);
        w.enc_gru.step(&self.h256, &mut self.enc_h, &mut self.gs);
        w.enc_out.forward(&self.enc_h, &mut self.emb);
        relu(&mut self.emb);
        let l = dot(&w.lsnr_w, &self.emb) + w.lsnr_b;
        sigmoid(l) * w.lsnr_scale + w.lsnr_offset
    }

    /// The ERB decoder: [`NetState::mask`] gets the band gains.
    pub fn decode_erb(&mut self, w: &Weights) {
        w.erb_in.forward(&self.emb, &mut self.h256);
        relu(&mut self.h256);
        let [h0, h1] = &mut self.erb_h;
        w.erb_gru[0].step(&self.h256, h0, &mut self.gs);
        w.erb_gru[1].step(h0, h1, &mut self.gs);
        w.erb_out.forward(h1, &mut self.emb_a);
        relu(&mut self.emb_a);
        // [F/4][C] → [C][F/4]
        let f8 = NB_ERB / 4;
        transpose(&self.emb_a, f8, CH, &mut self.d);
        // scale 3: 8 bands
        pathway(&w.conv3p, &self.e3, f8, &mut self.dsum);
        add(&mut self.dsum[..CH * f8], &self.d[..CH * f8]);
        separable(&w.convt3, &self.dsum, f8, &mut self.tmp, &mut self.d);
        // scale 2: 8 → 16 bands
        pathway(&w.conv2p, &self.e2, f8, &mut self.dsum);
        add(&mut self.dsum[..CH * f8], &self.d[..CH * f8]);
        upsample(&w.convt2, &self.dsum, f8, &mut self.tmp, &mut self.d);
        // scale 1: 16 → 32 bands
        let f16 = NB_ERB / 2;
        pathway(&w.conv1p, &self.e1, f16, &mut self.dsum);
        add(&mut self.dsum[..CH * f16], &self.d[..CH * f16]);
        upsample(&w.convt1, &self.dsum, f16, &mut self.tmp, &mut self.d);
        // output: 32 bands
        pathway(&w.conv0p, &self.e0, NB_ERB, &mut self.dsum);
        add(&mut self.dsum[..CH * NB_ERB], &self.d[..CH * NB_ERB]);
        let h: [&[f32]; 1] = [&self.dsum];
        w.conv0_out.forward(&h, NB_ERB, &mut self.mask);
        for m in self.mask.iter_mut() {
            *m = sigmoid(*m);
        }
    }

    /// The DF decoder: [`NetState::coefs`] gets the deep-filter coefficients.
    pub fn decode_df(&mut self, w: &Weights) {
        w.df_in.forward(&self.emb, &mut self.h256);
        relu(&mut self.h256);
        let [h0, h1] = &mut self.df_h;
        w.df_gru[0].step(&self.h256, h0, &mut self.gs);
        w.df_gru[1].step(h0, h1, &mut self.gs);
        w.df_skip.forward(&self.emb, &mut self.h256b);
        for (a, b) in self.h256b.iter_mut().zip(&self.df_h[1]) {
            *a += b;
        }
        w.df_out.forward(&self.h256b, &mut self.df_o);
        // the pathway on the last five frames of c0: [10][96] → [96][10]
        let h: [&[f32]; DF_ORDER] = [&self.c0_hist[0], &self.c0_hist[1], &self.c0_hist[2], &self.c0_hist[3], &self.c0_hist[4]];
        w.df_convp.0.forward(&h, NB_DF, &mut self.tmp);
        w.df_convp.1.forward(&self.tmp, NB_DF, &mut self.convp);
        relu(&mut self.convp);
        let k = 2 * DF_ORDER;
        for f in 0..NB_DF {
            for j in 0..k {
                let o = self.df_o.get(f * k + j).copied().unwrap_or(0.0).tanh();
                let p = self.convp.get(j * NB_DF + f).copied().unwrap_or(0.0);
                if let Some(c) = self.coefs.get_mut(f * k + j) {
                    *c = o + p;
                }
            }
        }
    }
}

fn copy(src: &[f32], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = *s;
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// `[rows][cols]` → `[cols][rows]`.
fn transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    for r in 0..rows {
        for c in 0..cols {
            if let (Some(s), Some(d)) = (src.get(r * cols + c), dst.get_mut(c * rows + r)) {
                *d = *s;
            }
        }
    }
}

/// A separable block: depthwise 1×3 (one frame), 1×1 with bias, ReLU.
fn separable(l: &(Conv, Pointwise), inp: &[f32], f_in: usize, tmp: &mut [f32], out: &mut [f32]) {
    let h: [&[f32]; 1] = [inp];
    l.0.forward(&h, f_in, tmp);
    let f_out = l.0.f_out(f_in);
    l.1.forward(tmp, f_out, out);
    let n = (l.1.out_ch * f_out).min(out.len());
    relu(&mut out[..n]);
}

/// A 1×1 depthwise pathway (scale and bias per channel), ReLU.
fn pathway(l: &Conv, inp: &[f32], f: usize, out: &mut [f32]) {
    let h: [&[f32]; 1] = [inp];
    l.forward(&h, f, out);
    let n = (l.out_ch * f).min(out.len());
    relu(&mut out[..n]);
}

/// A transposed block: depthwise transposed 1×3 (stride 2), 1×1 with bias, ReLU.
fn upsample(l: &(ConvT, Pointwise), inp: &[f32], f_in: usize, tmp: &mut [f32], out: &mut [f32]) {
    l.0.forward(inp, f_in, tmp);
    l.1.forward(tmp, 2 * f_in, out);
    let n = (l.1.out_ch * 2 * f_in).min(out.len());
    relu(&mut out[..n]);
}
